from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.parser import ParsedReport, normalize_compact, normalize_space


@dataclass(slots=True)
class ChunkRecord:
    id: str
    chunk_key: str
    chunk_index_global: int
    chunk_index_in_section: int
    major_section: str | None
    sub_section: str | None
    section_type: str
    note_no: int | None
    note_title: str | None
    subtopic: str | None
    topic: str | None
    content: str
    char_count: int
    section_id: str | None = None
    subsection_path: str | None = None
    as_of_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    evidence_type: str | None = None
    table_meta: dict[str, Any] | None = None
    cell_unit: str | None = None
    risk_domain: str | None = None


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def _build_section_id(
    major_section: str | None,
    sub_section: str | None,
    section_type: str,
    note_no: int | None,
    note_title: str | None,
) -> str:
    base = f"{_slugify(major_section or 'root')}::{_slugify(sub_section or section_type)}"

    # Note chunks: distinguish different notes under the same "주석" sub_section.
    if note_no is not None:
        return f"{base}::note_{note_no}"
    if note_title:
        return f"{base}::note_{_slugify(note_title)[:80]}"
    return base


AS_OF_RE = re.compile(r"(20\d{2})년\s*([01]?\d)월\s*([0-3]?\d)일\s*(?:현재|기준)")
PERIOD_RE = re.compile(
    r"(20\d{2})년\s*([01]?\d)월\s*([0-3]?\d)일\s*부터.*?"
    r"(20\d{2})년\s*([01]?\d)월\s*([0-3]?\d)일\s*까지",
    re.S,
)
DATE_RE = re.compile(r"(20\d{2})년\s*([01]?\d)월\s*([0-3]?\d)일")
FROM_MARKER_RE = re.compile(r"(부터|로\s*개시(?:하는)?|이후\s*시작(?:하는)?)")
TO_MARKER_RE = re.compile(r"(까지|로\s*종료(?:되는)?|종료되는\s*보고기간)")
REPORT_END_RE = re.compile(r"(보고기간종료일\s*현재|당기말\s*현재|당기말)")
PRIOR_END_RE = re.compile(r"(전기말\s*현재|전기말)")


def _fmt_date(y: str, m: str, d: str) -> str:
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def _extract_time_axis(
    content: str,
    sub_section: str | None = None,
    note_title: str | None = None,
    report_year: int | None = None,
) -> tuple[str | None, str | None, str | None]:
    # 주석/표 제목 + 본문을 함께 보고 날짜 표현 탐지
    text = normalize_space("\n".join([x for x in [sub_section, note_title, content] if x]))
    as_of_date = None
    period_start = None
    period_end = None

    m_asof = AS_OF_RE.search(text)
    if m_asof:
        as_of_date = _fmt_date(m_asof.group(1), m_asof.group(2), m_asof.group(3))

    m_period = PERIOD_RE.search(text)
    if m_period:
        period_start = _fmt_date(m_period.group(1), m_period.group(2), m_period.group(3))
        period_end = _fmt_date(m_period.group(4), m_period.group(5), m_period.group(6))

    # 단일 날짜 + 시작/종료 표식 대응 (예: "2020년 1월 1일로 개시하는")
    if period_start is None or period_end is None:
        dates = [
            (_fmt_date(y, m, d), mobj.start(), mobj.end())
            for mobj in DATE_RE.finditer(text)
            for y, m, d in [mobj.groups()]
        ]

        if dates and period_start is None:
            for d, s, e in dates:
                tail = text[e:e + 24]
                if FROM_MARKER_RE.search(tail):
                    period_start = d
                    break

        if dates and period_end is None:
            for d, s, e in dates:
                tail = text[e:e + 24]
                if TO_MARKER_RE.search(tail):
                    period_end = d

        # "동일로 종료되는 보고기간"처럼 종료일이 명시되지 않는 문구 fallback
        if period_end is None and report_year is not None and "종료되는 보고기간" in text:
            period_end = f"{report_year:04d}-12-31"

    # 당기말/전기말/보고기간종료일 현재 fallback
    if as_of_date is None and report_year is not None:
        if REPORT_END_RE.search(text):
            as_of_date = f"{report_year:04d}-12-31"
        elif PRIOR_END_RE.search(text):
            as_of_date = f"{report_year - 1:04d}-12-31"

    return as_of_date, period_start, period_end


def _extract_cell_unit(content: str) -> str | None:
    compact = normalize_compact(content)
    for unit in ("백만원", "천원", "원", "억원", "조원", "USD", "KRW"):
        if normalize_compact(unit) in compact:
            return unit
    return None


def _build_table_meta(content: str, section_type: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    table_like_lines = [line for line in lines if "|" in line]
    if section_type != "financial_statement" and not table_like_lines:
        return None

    target = table_like_lines if table_like_lines else lines
    col_counts = [len([c for c in line.split("|") if c.strip()]) for line in target]
    return {
        "is_table_like": bool(table_like_lines) or section_type == "financial_statement",
        "n_rows": len(target),
        "n_cols": max(col_counts) if col_counts else 0,
    }


def _classify_evidence_type(section_type: str, content: str, note_title: str | None) -> str:
    compact = normalize_compact(f"{content} {note_title or ''}")
    if section_type == "financial_statement":
        return "quant_table"
    if any(k in compact for k in ("소송", "우발", "약정", "보증", "담보", "충당")):
        return "risk_disclosure"
    if re.search(r"[-+]?\d[\d,]*(?:\.\d+)?\s*(원|천원|만원|백만원|억원|조원|USD|KRW)", content):
        return "quant_text"
    if section_type == "audit_subsection":
        return "audit_statement"
    return "narrative"


def _infer_risk_domain_from_text(note_title: str | None, subtopic: str | None, content: str | None) -> str | None:
    nt = normalize_compact(note_title or "")
    st = normalize_compact(subtopic or "")
    blob = normalize_compact(f"{note_title or ''} {subtopic or ''} {content or ''}")
    if not blob:
        return None

    if "우발부채와약정사항" in nt:
        if "소송" in st or "소송" in blob:
            return "litigation"
        if "약정" in st or "우발부채" in st:
            return "contingent_commitment"
        if any(k in st for k in ("지급보증", "담보및지급보증")) or any(
            k in blob for k in ("지급보증", "채무보증", "담보", "보증한도", "관련차입금")
        ):
            return "collateral_guarantee"
        if not st:
            return "contingent_commitment"

    if "재무위험관리" in nt:
        return None

    if any(k in blob for k in ("소송", "분쟁", "피소", "규제기관의조사")):
        return "litigation"
    if any(k in blob for k in ("담보", "지급보증", "채무보증", "연대보증", "제공한담보", "제공받은담보")):
        return "collateral_guarantee"
    if any(k in blob for k in (
        "약정사항",
        "계약상의무",
        "확약",
        "커미트먼트",
        "통합한도액",
        "연대채무",
        "무역금융",
        "상업어음할인",
        "외상매출채권담보대출",
    )):
        return "contingent_commitment"
    return None


def _safe_ascii_token(text: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", (text or "")).strip("_").lower()
    return token or "unknown"


def _build_section_id(section_type: str, note_no: int | None, section_ordinal: int) -> str:
    st = _safe_ascii_token(section_type)
    nn = note_no if note_no is not None else 0
    return f"{st}__s{section_ordinal:04d}__n{nn:04d}"


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "), window.rfind(" "))
            if split_at > max(50, chunk_size // 3):
                end = start + split_at + (2 if window[split_at:split_at + 2] == ". " else 1)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _table_rows(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _group_rows(rows: list[str], group_size: int) -> list[list[str]]:
    return [rows[i:i + group_size] for i in range(0, len(rows), group_size)]


def _append_chunk(
    chunks: list[ChunkRecord],
    stem: str,
    section_ordinal: int,
    report_year: int | None,
    major_section: str | None,
    sub_section: str | None,
    section_type: str,
    note_no: int | None,
    note_title: str | None,
    subtopic: str | None,
    topic: str | None,
    content: str,
    chunk_index_in_section: int,
) -> None:
    content = normalize_space(content)
    if len(content) < 30:
        return

    section_id = _build_section_id(section_type, note_no, section_ordinal)
    subsection_path = " > ".join([x for x in [major_section, sub_section, note_title, subtopic] if x]) or None
    as_of_date, period_start, period_end = _extract_time_axis(
        content,
        sub_section=sub_section,
        note_title=note_title,
        report_year=report_year,
    )
    evidence_type = _classify_evidence_type(section_type, content, note_title)
    table_meta = _build_table_meta(content, section_type)
    cell_unit = _extract_cell_unit(content)
    risk_domain = _infer_risk_domain_from_text(note_title, subtopic, content)

    chunks.append(
        ChunkRecord(
            id=str(uuid.uuid4()),
            chunk_key=f"{stem}__{len(chunks):04d}",
            chunk_index_global=len(chunks),
            chunk_index_in_section=chunk_index_in_section,
            major_section=major_section,
            sub_section=sub_section,
            section_type=section_type,
            note_no=note_no,
            note_title=note_title,
            subtopic=subtopic,
            topic=topic,
            content=content,
            char_count=len(content),
            section_id=section_id,
            subsection_path=subsection_path,
            as_of_date=as_of_date,
            period_start=period_start,
            period_end=period_end,
            evidence_type=evidence_type,
            table_meta=table_meta,
            cell_unit=cell_unit,
            risk_domain=risk_domain,
        )
    )


def _chunk_general_text(prefix: str, text: str) -> list[str]:
    return [f"{prefix}\n\n{chunk}".strip() for chunk in _split_text(text, settings.text_chunk_size, settings.text_chunk_overlap) if chunk.strip()]


def _chunk_note_text(prefix: str, text: str) -> list[str]:
    return [f"{prefix}\n\n{chunk}".strip() for chunk in _split_text(text, settings.note_chunk_size, settings.note_chunk_overlap) if chunk.strip()]


SUBTOPIC_MARKER_RE = re.compile(r"^(?:[가-힣]\.|\(\d+\)|\d+\))\s*")


def _normalize_subtopic_title(title: str) -> str:
    t = normalize_space(title)
    t = SUBTOPIC_MARKER_RE.sub("", t)
    t = re.sub(r"\s*[:：\-]\s*$", "", t)
    return t.strip()


def _is_subtopic_candidate(title: str) -> bool:
    t = normalize_space(title)
    if not t:
        return False
    if SUBTOPIC_MARKER_RE.match(t):
        return True

    # 짧은 명사형 소제목(문장형 제외)
    if len(t) <= 24 and not re.search(r"[.!?]$", t) and re.search(r"[가-힣A-Za-z]", t):
        if not any(x in t for x in ("입니다", "합니다", "하였다", "있습니다", "다.")):
            return True
    return False


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        t = normalize_space(line)
        if t:
            return t
    return ""


def _derive_subtopic_from_text(text: str, note_title: str | None) -> str | None:
    t = normalize_space(text)
    if not t:
        return None

    # 일반 라벨(짧은 명사형) 우선
    first = _normalize_subtopic_title(_first_nonempty_line(t))
    if _is_subtopic_candidate(first):
        return first

    # 문장형 subsection에서 핵심 하위항목 추출
    if "지급보증한 내역" in t:
        return "지급보증한 내역"
    if "기타 약정사항" in t:
        return "기타 약정사항"
    if "소송" in t and len(t) < 200:
        return "소송"

    # note_title이 충당부채 계열일 때 하위항목 승격
    nt = normalize_space(note_title or "")
    if "충당부채" in nt:
        if any(k in t for k in ("판매보증", "품질보증", "하자보수", "사후서비스", "출고한 제품")):
            return "판매보증충당부채"
        if "기술사용료" in t:
            return "기술사용료충당부채"
        if "장기성과" in t:
            return "장기성과급충당부채"
        if "기타충당" in t:
            return "기타충당부채"

    return None


def _extract_note_subtopic_segments(note: dict[str, Any]) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    seen: set[str] = set()
    merged_text: str = ""

    def _append(subtopic: str | None, text: str) -> None:
        if not subtopic:
            return
        key = normalize_compact(subtopic)
        body = normalize_space(text)
        if not key or key in seen or len(body) < 40:
            return
        seen.add(key)
        segments.append((subtopic, body))

    # intro_blocks도 하위항목 단서가 있을 수 있어 함께 검사
    for block in note.get("intro_blocks", []):
        btext = normalize_space(block.get("text", ""))
        if not btext:
            continue
        subtopic = _derive_subtopic_from_text(btext, note.get("note_title"))
        _append(subtopic, btext)

    for sec in note.get("subsections", []):
        sec_title = normalize_space(sec.get("title", ""))
        sec_text = _collect_section_text(sec)
        if not sec_text:
            continue

        subtopic: str | None = None
        if _is_subtopic_candidate(sec_title):
            subtopic = _normalize_subtopic_title(sec_title)
        else:
            # title이 문장형일 때는 text/line 기반으로 하위항목 추출
            subtopic = _derive_subtopic_from_text(sec_title, note.get("note_title"))
            if not subtopic:
                subtopic = _derive_subtopic_from_text(sec_text, note.get("note_title"))

        if subtopic:
            _append(subtopic, sec_text)

        # 소송 섹션 내부에서 질의 빈도가 높은 의미를 미세 분할
        if subtopic and "소송" in normalize_compact(subtopic):
            for sent in re.split(r"(?<=다\.)\s+|(?<=[.!?])\s+", normalize_space(sec_text)):
                s = normalize_space(sent)
                if len(s) < 20:
                    continue
                if "경영진" in s and "판단" in s:
                    _append("소송_경영진판단", s)
                if "유출" in s and any(k in s for k in ("불확실", "시기", "금액")):
                    _append("소송_유출관련", s)

    # 하위제목이 약한 연도(특히 우발부채/약정사항)에서 본문 마커 기반 분할
    note_title = normalize_space(note.get("note_title", ""))
    if "우발부채와 약정사항" in note_title:
        merged_parts: list[str] = []
        for block in note.get("intro_blocks", []):
            merged_parts.append(block.get("text", ""))
        for sec in note.get("subsections", []):
            merged_parts.append(_collect_section_text(sec, include_tables=False))
        merged_text = normalize_space("\n\n".join(p for p in merged_parts if p))

        if merged_text:
            marker_patterns = [
                ("우발부채", r"우발부채"),
                ("소송", r"소송(?:\s*등)?"),
                ("지급보증한 내역", r"지급보증한\s*내역"),
                ("담보 및 지급보증", r"담보\s*및\s*지급보증"),
                ("기타 약정사항", r"기타\s*약정사항"),
            ]
            found: list[tuple[int, str]] = []
            for name, pat in marker_patterns:
                m = re.search(pat, merged_text)
                if m:
                    found.append((m.start(), name))
            found.sort(key=lambda x: x[0])

            for idx, (start, name) in enumerate(found):
                end = found[idx + 1][0] if idx + 1 < len(found) else len(merged_text)
                seg = normalize_space(merged_text[start:end])
                _append(name, seg)

    # 마커/문장부호가 약한 연도 fallback: 핵심 질의 타깃 문장만 별도 subtopic으로 승격
    if merged_text:
        compact_seen = {normalize_compact(s) for s, _ in segments}

        def _pick_focus_snippet(text: str, must_have: tuple[str, ...], prefer_any: tuple[str, ...] = ()) -> str | None:
            candidates = [normalize_space(x) for x in re.split(r"(?<=다\.)\s+|(?<=[.!?])\s+|\n+", text) if normalize_space(x)]
            for sent in candidates:
                if len(sent) < 20:
                    continue
                if all(k in sent for k in must_have) and (not prefer_any or any(k in sent for k in prefer_any)):
                    return sent

            # 문장 분리가 실패한 경우 키워드 주변 window 추출
            for k in must_have:
                idx = text.find(k)
                if idx >= 0:
                    lo = max(0, idx - 110)
                    hi = min(len(text), idx + 170)
                    win = normalize_space(text[lo:hi])
                    if len(win) >= 30 and all(m in win for m in must_have) and (not prefer_any or any(p in win for p in prefer_any)):
                        return win
            return None

        if "소송_경영진판단" not in compact_seen:
            s = _pick_focus_snippet(merged_text, ("경영진", "판단"), ("소송", "유출"))
            if s:
                _append("소송_경영진판단", s)

        if "소송_유출관련" not in compact_seen:
            s = _pick_focus_snippet(merged_text, ("유출",), ("불확실", "시기", "금액"))
            if s:
                _append("소송_유출관련", s)

        if "기타약정사항" not in compact_seen:
            s = _pick_focus_snippet(
                merged_text,
                ("약정",),
                ("통합한도", "무역금융", "상업어음할인", "외상매출채권담보대출", "은행"),
            )
            if s:
                _append("기타 약정사항", s)
    return segments


def _guess_topic(text: str) -> str | None:
    compact = normalize_compact(text)
    keywords = [
        "현금및현금성자산",
        "단기금융상품",
        "매출채권",
        "재고자산",
        "유형자산",
        "무형자산",
        "차입금",
        "충당부채",
        "법인세비용",
        "현금흐름표",
        "영업활동현금흐름",
        "투자활동현금흐름",
        "재무활동현금흐름",
        "재무위험관리",
    ]
    for keyword in keywords:
        if keyword in compact:
            return keyword
    return None


def _financial_statement_chunks(
    stem: str,
    major_title: str,
    section: dict[str, Any],
    chunks: list[ChunkRecord],
    section_ordinal: int,
    report_year: int | None,
) -> None:
    section_title = section["title"]
    block_idx = 0
    title_context = ""
    for block in section.get("content_blocks", []):
        if block["block_type"] != "table":
            continue
        rows = _table_rows(block["text"])
        if not rows:
            continue
        if len(rows) <= 6 or _guess_topic(rows[0]) in {section_title, None} and section_title in normalize_compact(rows[0]):
            title_context = "\n".join(rows)
            continue

        grouped = _group_rows(rows, settings.table_row_group_size)
        for group in grouped:
            topic = _guess_topic("\n".join(group)) or _guess_topic(title_context)
            content_parts = [f"[{section_title}]"]
            if title_context:
                content_parts.append(title_context)
            content_parts.append("\n".join(group))
            content = "\n".join(part for part in content_parts if part)
            _append_chunk(
                chunks,
                stem,
                section_ordinal,
                report_year,
                major_title,
                section_title,
                "financial_statement",
                None,
                None,
                None,
                topic,
                content,
                block_idx,
            )
            block_idx += 1


def _collect_section_text(sec: dict[str, Any], include_tables: bool = True) -> str:
    parts: list[str] = []
    title = sec.get("title", "")
    if title:
        parts.append(title)
    for block in sec.get("content_blocks", []):
        if not include_tables and block.get("block_type") == "table":
            continue
        parts.append(block["text"])
    for child in sec.get("children", []):
        child_text = _collect_section_text(child, include_tables=include_tables)
        if child_text:
            parts.append(child_text)
    return "\n\n".join(p for p in parts if p and p.strip())


def _notes_chunks(
    stem: str,
    major_title: str,
    section: dict[str, Any],
    chunks: list[ChunkRecord],
    section_ordinal: int,
    report_year: int | None,
) -> None:
    for note in section.get("notes", []):
        note_no = note["note_no"]
        note_title = note.get("note_title") or f"주석 {note_no}"

        merged_parts: list[str] = []
        for block in note.get("intro_blocks", []):
            if block.get("block_type") == "table":
                continue
            merged_parts.append(block["text"])
        for sec in note.get("subsections", []):
            sec_text = _collect_section_text(sec, include_tables=False)
            if sec_text:
                merged_parts.append(sec_text)

        merged_text = "\n\n".join(part for part in merged_parts if part.strip())
        if not merged_text:
            continue

        prefix = f"[주석 {note_no}. {note_title}]"
        parts = _chunk_note_text(prefix, merged_text)
        for idx, part in enumerate(parts):
            _append_chunk(
                chunks,
                stem,
                section_ordinal,
                report_year,
                major_title,
                section["title"],
                "note",
                note_no,
                note_title,
                None,
                _guess_topic(merged_text) or note_title,
                part,
                idx,
            )

        # B-lite: 하위항목이 명확할 때만 subtopic 단위 chunk 추가 (기존 note chunk는 유지)
        subtopic_segments = _extract_note_subtopic_segments(note)
        if subtopic_segments:
            offset = len(parts)
            for sidx, (subtopic, sub_text) in enumerate(subtopic_segments):
                sprefix = f"[주석 {note_no}. {note_title} > {subtopic}]"
                sub_parts = _chunk_note_text(sprefix, sub_text)
                for pidx, sub_part in enumerate(sub_parts):
                    _append_chunk(
                        chunks,
                        stem,
                        section_ordinal,
                        report_year,
                        major_title,
                        section["title"],
                        "note",
                        note_no,
                        note_title,
                        subtopic,
                        _guess_topic(sub_text) or subtopic,
                        sub_part,
                        offset + sidx * 100 + pidx,
                    )


def _major_text_chunks(
    stem: str,
    major: dict[str, Any],
    chunks: list[ChunkRecord],
    section_ordinal: int,
    report_year: int | None,
) -> None:
    title = major["title"]
    merged = "\n\n".join(block["text"] for block in major.get("content_blocks", []))
    if not merged:
        return
    for idx, part in enumerate(_chunk_general_text(f"[{title}]", merged)):
        _append_chunk(
            chunks,
            stem,
            section_ordinal,
            report_year,
            title,
            None,
            major.get("major_type", "other"),
            None,
            None,
            None,
            _guess_topic(merged),
            part,
            idx,
        )


def build_chunks(report: ParsedReport) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    stem = _slugify(report.file_path.stem)
    section_ordinal = 0

    for major in report.structured.get("major_sections", []):
        major_title = major["title"]
        section_ordinal += 1
        _major_text_chunks(stem, major, chunks, section_ordinal, report.report_year)

        for section in major.get("sections", []):
            section_type = section.get("section_type")
            section_ordinal += 1
            if section_type == "financial_statement":
                _financial_statement_chunks(stem, major_title, section, chunks, section_ordinal, report.report_year)
            elif section_type == "notes_section":
                _notes_chunks(stem, major_title, section, chunks, section_ordinal, report.report_year)
            else:
                merged = "\n\n".join(block["text"] for block in section.get("content_blocks", []))
                if not merged:
                    continue
                for idx, part in enumerate(_chunk_general_text(f"[{section['title']}]", merged)):
                    _append_chunk(
                        chunks,
                        stem,
                        section_ordinal,
                        report.report_year,
                        major_title,
                        section["title"],
                        section_type or "section",
                        None,
                        None,
                        None,
                        _guess_topic(merged),
                        part,
                        idx,
                    )

    return chunks


def _parse_table_rows(table_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in (table_text or "").splitlines():
        line = normalize_space(line)
        if not line or "|" not in line:
            continue
        cells = [normalize_space(c) for c in line.split("|")]
        # markdown/table 파이프 경계에서 생기는 양끝 빈셀만 제거하고,
        # 내부 빈셀(헤더 정렬용)은 유지하여 열 정렬을 보존한다.
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if not cells:
            continue
        if all(re.fullmatch(r"[-:]+", c or "") for c in cells if c is not None):
            continue
        if len(cells) >= 2:
            rows.append(cells)
    return rows


def _is_amount_like(value: str) -> bool:
    v = normalize_space(value)
    return bool(re.fullmatch(r"\(?[-+]?\d[\d,]*(?:\.\d+)?\)?", v))


def _coerce_amount_from_annotated_text(value: str) -> float | None:
    raw = normalize_space(value)
    if not raw:
        return None

    m = re.match(r"^\(?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\)?", raw)
    if not m:
        return None

    token = normalize_space(m.group(0))
    if not token:
        return None

    core = token.strip("()").replace(",", "")
    if re.fullmatch(r"(?:19|20)\d{2}", core):
        return None

    sign = -1.0 if token.startswith("(") and token.endswith(")") else 1.0
    try:
        return sign * float(core)
    except Exception:
        return None


def _value_type_and_numeric(value: str) -> tuple[str, float | None]:
    raw = normalize_space(value)
    if not raw:
        return "text", None
    if re.fullmatch(r"(?:19|20)\d{2}", raw):
        return "date", None
    if re.fullmatch(r"(?:19|20)\d{2}[.\-]\d{1,2}(?:[.\-]\d{1,2})?", raw):
        return "date", None
    if "%" in raw or re.fullmatch(r"\d+(?:\.\d+)?\s*~\s*\d+(?:\.\d+)?", raw):
        try:
            num = float(raw.replace("%", "").split("~")[0].strip())
            return "rate", num
        except Exception:
            return "rate", None
    if _is_amount_like(raw):
        normalized = raw.replace(",", "")
        sign = -1.0 if normalized.startswith("(") and normalized.endswith(")") else 1.0
        normalized = normalized.strip("()")
        try:
            return "amount", sign * float(normalized)
        except Exception:
            return "amount", None

    coerced = _coerce_amount_from_annotated_text(raw)
    if coerced is not None:
        return "amount", coerced

    return "text", None


def _override_value_type_by_col_label(
    value_raw: str,
    col_label: str,
    value_type: str,
    value_numeric: float | None,
) -> tuple[str, float | None]:
    c = normalize_compact(col_label)
    if any(k in c for k in ("연이자율", "이자율", "rate")):
        raw = normalize_space(value_raw).replace(",", "")
        try:
            return "rate", float(raw)
        except Exception:
            return "rate", None

    if any(k in c for k in ("종료일", "만기", "일자")):
        return "date", None

    return value_type, value_numeric


def _extract_unit_from_table_text(table_text: str) -> str | None:
    compact = normalize_compact(table_text)
    for unit in ("백만원", "천원", "원", "억원", "USD", "KRW"):
        if normalize_compact(unit) in compact:
            return unit
    return None


def _split_row_group_label(row_label: str) -> tuple[str | None, str]:
    label = normalize_space(row_label)
    label = re.sub(r"\(\s*주\s*\d+\s*\)", "", label)
    label = re.sub(r"\(\s*\*+\s*\)", "", label)
    label = normalize_space(label)
    if ":" in label:
        left, right = label.split(":", 1)
        return normalize_space(left) or None, normalize_space(right)
    if "：" in label:
        left, right = label.split("：", 1)
        return normalize_space(left) or None, normalize_space(right)

    compact = normalize_compact(label)
    for prefix in ("유동성장기차입금", "장기차입금", "실차입금기준", "한도기준"):
        p = normalize_compact(prefix)
        if compact.startswith(p) and compact != p:
            right = normalize_space(label[len(prefix):]).lstrip("-:： ")
            if right:
                return prefix, right
    return None, label


def _header_row_count(rows: list[list[str]]) -> int:
    count = 0
    for row in rows[:3]:
        if len(row) < 2:
            continue
        numeric_cells = sum(1 for c in row[1:] if _is_amount_like(c))
        ratio = numeric_cells / max(1, len(row) - 1)
        if ratio < 0.4:
            count += 1
        else:
            break
    return max(1, min(count, 2)) if rows else 0


def _build_col_labels(rows: list[list[str]], header_rows: int) -> list[str]:
    if not rows:
        return []
    h1 = rows[0][1:] if len(rows[0]) > 1 else []
    if header_rows >= 2 and len(rows) >= 2:
        h2 = rows[1][1:] if len(rows[1]) > 1 else []
        width = max(len(h1), len(h2))
        labels: list[str] = []
        carry_parent = ""
        for i in range(width):
            a = h1[i] if i < len(h1) else ""
            b = h2[i] if i < len(h2) else ""
            a_norm = normalize_space(a)
            b_norm = normalize_space(b)
            if a_norm:
                carry_parent = a_norm
            parent = carry_parent
            # 다층 헤더에서 2행이 기간(당기말/전기말)만 줄 때 상위 헤더를 결합
            if b_norm and parent and parent != b_norm:
                label = normalize_space(f"{parent} {b_norm}")
            else:
                label = normalize_space(" ".join([x for x in [a_norm, b_norm] if x]))
            labels.append(label or f"col_{i+1}")
        return labels
    return [normalize_space(c) or f"col_{i+1}" for i, c in enumerate(h1, start=1)]


def _norm_label(text: str | None) -> str:
    return normalize_compact((text or "").lower())


def _infer_table_family(
    note_title: str | None,
    subtopic: str | None,
    table_title: str | None,
    table_text: str | None = None,
) -> str:
    blob = normalize_compact(f"{note_title or ''}{subtopic or ''}{table_title or ''}{table_text or ''}")
    if "현금및현금성자산" in blob:
        return "cash"
    if "충당부채" in blob:
        return "provision"
    if "지급보증" in blob or ("보증" in blob and "차입금" in blob):
        return "guarantee"
    if "차입금" in blob or "리스부채" in blob or "유동성장기차입금" in blob:
        return "loan_lease"
    return "other"


def _infer_row_role(row_label: str | None, row_group: str | None) -> str | None:
    blob = normalize_compact(f"{row_group or ''}{row_label or ''}")
    if "기초" in blob:
        return "opening"
    if "기말" in blob:
        return "ending"
    if "순전입" in blob or "환입" in blob:
        return "inflow"
    if "사용액" in blob:
        return "outflow"
    if "기타" in blob:
        return "other"
    if "계" in blob or "합계" in blob or "총액" in blob:
        return "aggregate"
    return None


def _infer_period_role(col_label: str | None, period_type: str | None) -> str | None:
    blob = normalize_compact(f"{col_label or ''}{period_type or ''}")
    if "당기말" in blob or "당기말현재" in blob or "보고기간종료일현재" in blob:
        return "current_end"
    if "전기말" in blob or "전기말현재" in blob:
        return "prior_end"
    if "당기" in blob:
        return "current"
    if "전기" in blob:
        return "prior"
    return None


def _infer_period_type_from_col_label(col_label: str) -> str | None:
    ccompact = normalize_compact(col_label)
    if any(k in ccompact for k in ("당기말", "당기말현재", "보고기간종료일현재")):
        return "당기말"
    if "전기말" in ccompact or "전기말현재" in ccompact:
        return "전기말"
    if "당기" in ccompact:
        return "당기"
    if "전기" in ccompact:
        return "전기"
    return None


def _extract_row_year(row_label: str | None, row_group: str | None) -> int | None:
    text = normalize_space(f"{row_group or ''} {row_label or ''}")
    m = re.search(r"((?:19|20)\d{2})\s*년?", text)
    return int(m.group(1)) if m else None


def _is_aggregate_row(row_label: str | None, row_group: str | None) -> bool:
    label = normalize_space(f"{row_group or ''} {row_label or ''}")
    compact = normalize_compact(label)
    if compact in {"계", "합계", "총액", "총계"}:
        return True
    return bool(re.search(r"(^|\s)(계|합계|총액|총계)(\s|$)", label))


def _infer_entity_label(table_family: str, row_group: str | None, row_label: str | None, is_aggregate: bool) -> str | None:
    if table_family == "guarantee":
        if is_aggregate:
            return "계"
        label = normalize_space(row_label or row_group or "")
        if not label or label == "-":
            return None
        lc = label.lower()
        if "setk" in lc:
            return "SETK"
        if "samcol" in lc:
            return "SAMCOL"
        if "기타" in normalize_compact(label):
            return "기타"
        return label
    return None


def _infer_structured_risk_domain(
    note_title: str | None,
    subtopic: str | None,
    table_title: str | None,
    row_group: str | None = None,
    row_label: str | None = None,
    col_label: str | None = None,
) -> str | None:
    nt = normalize_compact(note_title or "")
    st = normalize_compact(subtopic or "")
    tt = normalize_compact(table_title or "")
    blob = normalize_compact(
        f"{note_title or ''} {subtopic or ''} {table_title or ''} {row_group or ''} {row_label or ''} {col_label or ''}"
    )
    if not blob:
        return None

    if "우발부채와약정사항" in nt:
        if "소송" in st or "소송" in tt or "소송" in blob:
            return "litigation"
        if "약정" in st or "약정" in tt or "우발부채" in st or "우발부채" in tt:
            return "contingent_commitment"
        if any(k in st or k in tt for k in ("지급보증", "담보및지급보증")) or any(
            k in blob for k in ("지급보증", "채무보증", "담보", "보증한도", "관련차입금")
        ):
            return "collateral_guarantee"
        if not st:
            return "contingent_commitment"

    if "재무위험관리" in nt:
        return None
    if any(k in blob for k in ("소송", "분쟁", "피소", "규제기관의조사")):
        return "litigation"
    if any(k in blob for k in ("담보", "지급보증", "채무보증", "연대보증", "보증한도", "관련차입금")):
        return "collateral_guarantee"
    if any(k in blob for k in (
        "약정사항",
        "계약상의무",
        "확약",
        "커미트먼트",
        "통합한도액",
        "연대채무",
        "무역금융",
        "상업어음할인",
        "외상매출채권담보대출",
    )):
        return "contingent_commitment"
    return None


def _stabilize_row_semantics(
    table_family: str,
    row_group: str | None,
    row_label: str | None,
) -> tuple[str | None, str, str | None, bool, int | None]:
    rg = normalize_space(row_group or "") or None
    rl = normalize_space(row_label or "")

    if table_family == "provision":
        c = normalize_compact(rl)
        if "순전입" in c or "환입" in c:
            rl = "순전입액(환입액)"
        elif c.startswith("기초"):
            rl = "기초"
        elif c.startswith("사용액"):
            rl = "사용액"
        elif c.startswith("기타"):
            rl = "기타"
        elif c.startswith("기말"):
            rl = "기말"

    if table_family == "loan_lease" and not rg:
        c = normalize_compact(rl)
        for prefix in ("유동성장기차입금", "장기차입금"):
            p = normalize_compact(prefix)
            if c.startswith(p) and c != p:
                tail = normalize_space(rl[len(prefix):]).lstrip("-:： ")
                if tail:
                    rg = prefix
                    rl = tail
                    break

    if table_family == "guarantee":
        c = normalize_compact(f"{rg or ''}{rl}")
        if "실차입금기준" in c:
            rg = "실차입금기준"
            rl = "계"
        elif "한도기준" in c:
            rg = "한도기준"
            rl = "계"

    row_role = _infer_row_role(rl, rg)
    is_aggregate = _is_aggregate_row(rl, rg)
    if table_family == "guarantee" and normalize_compact(f"{rg or ''}{rl}") in {"실차입금기준계", "한도기준계"}:
        row_role = "aggregate"
        is_aggregate = True
    if is_aggregate and row_role is None:
        row_role = "aggregate"
    row_year = _extract_row_year(rl, rg)
    return rg, rl, row_role, is_aggregate, row_year


def _stabilize_col_label(table_family: str, col_label: str, col_index: int) -> str:
    label = normalize_space(col_label)
    compact = normalize_compact(label)

    if table_family == "provision":
        if "판매보증" in compact:
            return "판매보증"
        if "기술사용료" in compact:
            return "기술사용료"
        if "장기성과급" in compact:
            return "장기성과급"
        if "기타충당" in compact:
            return "기타충당부채"
        if compact in {"계", "합계", "총액", "총계"}:
            return "계"

    if table_family == "guarantee":
        if "관련차입금" in compact:
            return "관련차입금 당기말" if "당기말" in compact else "관련차입금"
        if "채무보증한도" in compact or "보증한도" in compact:
            return "채무보증한도 당기말" if "당기말" in compact else "채무보증한도"
        # 초기 연도 표는 헤더가 당기말/전기말 또는 col_N 으로만 남는 경우가 있어 위치 기반으로 안정화
        if (
            compact.startswith("col_")
            or compact in {"당기말", "전기말", "당기", "전기"}
            or re.fullmatch(r"(?:19|20)\d{2}(?:년)?", label or "")
        ):
            if col_index == 0:
                return "관련차입금"
            if col_index >= 1:
                return "채무보증한도"

    return label or f"col_{col_index + 1}"


def _iter_note_subsections(note: dict[str, Any]) -> list[tuple[str | None, str, list[dict[str, Any]]]]:
    out: list[tuple[str | None, str, list[dict[str, Any]]]] = []
    for sec in note.get("subsections", []):
        title = normalize_space(sec.get("title", ""))
        out.append((None, title, sec.get("content_blocks", [])))
        stack = list(sec.get("children", []))
        while stack:
            cur = stack.pop(0)
            out.append((None, normalize_space(cur.get("title", "")), cur.get("content_blocks", [])))
            stack.extend(cur.get("children", []))
    return out


def _loan_lease_row_group_overrides(data_rows: list[list[str]]) -> dict[int, str]:
    overrides: dict[int, str] = {}
    lease_rows: list[int] = []
    for ridx, row in enumerate(data_rows):
        if not row:
            continue
        rl = normalize_compact(row[0])
        if "리스부채" in rl:
            lease_rows.append(ridx)

    if len(lease_rows) >= 2:
        # 일반적으로 첫 리스부채 행은 유동성, 다음은 장기로 기재됨
        overrides[lease_rows[0]] = "유동성장기차입금"
        overrides[lease_rows[1]] = "장기차입금"
    return overrides


def _guarantee_amount_col_overrides(row_values: list[str]) -> dict[int, str]:
    parsed = [_value_type_and_numeric(v)[0] if v else "text" for v in row_values]
    amount_positions = [idx for idx, t in enumerate(parsed, start=1) if t == "amount"]
    if len(amount_positions) < 2:
        return {}

    related_pos = amount_positions[-2]
    limit_pos = amount_positions[-1]
    return {
        related_pos: "관련차입금 당기말",
        limit_pos: "채무보증한도 당기말",
    }


def build_structured_tables(report: ParsedReport) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    stem = _slugify(report.file_path.stem)
    seq = 0

    def add_table(
        section_type: str,
        table_text: str,
        note_no: int | None,
        note_title: str | None,
        subtopic: str | None,
        table_title: str | None,
    ) -> None:
        nonlocal seq
        parsed_rows = _parse_table_rows(table_text)
        if len(parsed_rows) < 2:
            return

        seq += 1
        table_id = f"{stem}__y{report.report_year or 0}__n{note_no or 0}__t{seq:04d}"
        unit = _extract_unit_from_table_text(table_text)
        table_family = _infer_table_family(note_title, subtopic, table_title, table_text)
        meta = {
            "table_id": table_id,
            "report_year": report.report_year,
            "note_no": note_no,
            "note_title": note_title,
            "subtopic": subtopic,
            "table_title": table_title,
            "section_type": section_type,
            "unit": unit,
            "table_family": table_family,
            "risk_domain": _infer_structured_risk_domain(note_title, subtopic, table_title),
            "source_chunk_key": f"{table_id}__src",
        }
        table_rows.append(meta)

        header_rows = _header_row_count(parsed_rows)
        col_labels = _build_col_labels(parsed_rows, header_rows)
        data_rows = parsed_rows[header_rows:]
        loan_lease_overrides = _loan_lease_row_group_overrides(data_rows) if table_family == "loan_lease" else {}

        for ridx, row in enumerate(data_rows):
            if not row:
                continue
            raw_row_label = row[0] if row else ""
            row_values = [normalize_space(v) for v in row[1:]]
            row_group, row_label = _split_row_group_label(raw_row_label)
            if table_family == "loan_lease" and not row_group and ridx in loan_lease_overrides:
                row_group = loan_lease_overrides[ridx]
            row_group, row_label, row_role, is_aggregate, row_year = _stabilize_row_semantics(
                table_family,
                row_group,
                row_label,
            )
            guarantee_col_overrides = _guarantee_amount_col_overrides(row_values) if table_family == "guarantee" else {}

            for cidx, value_raw in enumerate(row_values, start=1):
                if not value_raw:
                    continue
                col_label = col_labels[cidx - 1] if cidx - 1 < len(col_labels) else f"col_{cidx}"
                col_label = _stabilize_col_label(table_family, col_label, cidx - 1)
                value_type, value_numeric = _value_type_and_numeric(value_raw)
                if cidx in guarantee_col_overrides and value_type == "amount":
                    col_label = guarantee_col_overrides[cidx]
                value_type, value_numeric = _override_value_type_by_col_label(
                    value_raw,
                    col_label,
                    value_type,
                    value_numeric,
                )
                currency = "USD" if "US$" in value_raw or "USD" in value_raw else ("KRW" if "KRW" in value_raw else None)
                period_type = _infer_period_type_from_col_label(col_label)

                period_role = _infer_period_role(col_label, period_type)
                entity_label = _infer_entity_label(table_family, row_group, row_label, is_aggregate)
                risk_domain = _infer_structured_risk_domain(
                    note_title,
                    subtopic,
                    table_title,
                    row_group=row_group,
                    row_label=row_label,
                    col_label=col_label,
                )
                parse_confidence = 0.70
                if value_type == "amount":
                    parse_confidence = 0.85
                    if row_label and col_label and not str(col_label).startswith("col_"):
                        parse_confidence = 0.95
                    if is_aggregate:
                        parse_confidence = max(parse_confidence, 0.92)
                    if table_family == "guarantee" and entity_label in {"SETK", "SAMCOL", "기타", "계"}:
                        parse_confidence = max(parse_confidence, 0.95)
                    if row_year is not None:
                        parse_confidence = max(parse_confidence, 0.93)

                cell_rows.append(
                    {
                        "table_id": table_id,
                        "report_year": report.report_year,
                        "note_no": note_no,
                        "note_title": note_title,
                        "subtopic": subtopic,
                        "table_title": table_title,
                        "section_type": section_type,
                        "unit": unit,
                        "source_chunk_key": f"{table_id}__r{ridx:03d}",
                        "row_index": ridx,
                        "col_index": cidx - 1,
                        "row_group": row_group,
                        "row_label": row_label,
                        "col_label": col_label,
                        "value_raw": value_raw,
                        "value_numeric": value_numeric,
                        "value_type": value_type,
                        "currency": currency,
                        "as_of_date": None,
                        "period_type": period_type,
                        "table_family": table_family,
                        "table_title_norm": _norm_label(table_title),
                        "note_title_norm": _norm_label(note_title),
                        "subtopic_norm": _norm_label(subtopic),
                        "row_label_norm": _norm_label(row_label),
                        "col_label_norm": _norm_label(col_label),
                        "row_group_norm": _norm_label(row_group),
                        "row_role": row_role,
                        "period_role": period_role,
                        "is_aggregate": is_aggregate,
                        "row_year": row_year,
                        "entity_label": entity_label,
                        "risk_domain": risk_domain,
                        "parse_confidence": parse_confidence,
                    }
                )

    for major in report.structured.get("major_sections", []):
        for section in major.get("sections", []):
            section_type = section.get("section_type") or "section"
            if section_type == "financial_statement":
                for block in section.get("content_blocks", []):
                    if block.get("block_type") == "table":
                        add_table(section_type, block.get("text", ""), None, None, None, section.get("title"))
            elif section_type == "notes_section":
                for note in section.get("notes", []):
                    note_no = note.get("note_no")
                    note_title = note.get("note_title")
                    for block in note.get("intro_blocks", []):
                        if block.get("block_type") == "table":
                            add_table("note", block.get("text", ""), note_no, note_title, None, note_title)

                    for subtopic, sec_title, blocks in _iter_note_subsections(note):
                        for block in blocks:
                            if block.get("block_type") == "table":
                                add_table("note", block.get("text", ""), note_no, note_title, subtopic, sec_title or note_title)

    return table_rows, cell_rows

    return chunks