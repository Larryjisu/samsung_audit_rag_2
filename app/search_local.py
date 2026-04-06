from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sys
from typing import Any

from app.config import settings
from app.db import get_conn
from app.embedder import Embedder
from sentence_transformers import CrossEncoder


YEAR_TOKEN_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:\s*년(?:도)?|\s*회계연도)?")
WHITESPACE_RE = re.compile(r"\s+")
TRAILING_PARTICLE_RE = re.compile(r"[은는이가을를의에와과도만]+$")

AUDIT_SUBSECTION_HINTS = (
    "감사의견",
    "감사의견근거",
    "핵심감사사항",
    "기타사항",
    "감사인의책임",
    "재무제표감사에대한감사인의책임",
    "경영진과지배기구의책임",
)

SUB_SECTION_HINTS = {
    "감사의견근거": "감사의견근거",
    "핵심감사사항": "핵심감사사항",
    "기타사항": "기타사항",
    "감사인의책임": "감사인의 책임",
    "재무제표감사에대한감사인의책임": "재무제표감사에 대한 감사인의 책임",
    "경영진과지배기구의책임": "재무제표에 대한 경영진과 지배기구의 책임",
}

EXACT_MATCH_BOOST = 0.35
QUERY_NOISE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"[?？!]+",
        r"무엇이야|무엇인가요|무엇입니까|뭐야|뭐지",
        r"어떻게설명돼|어떻게설명되나|어떻게설명돼\??|어떻게설명되어있어",
        r"설명해줘|설명해|보여줘|찾아줘|요약해줘|알려줘",
        r"뭐가적혀있어|무엇이적혀있어|어떤내용이야",
        r"관련내용|관련내역|관련사항",
        r"인가요|입니까|이야|야$",
    )
]

AS_OF_HINTS = ("당기말", "기말", "현재")
PERIOD_HINTS = ("부터", "까지", "보고기간", "기간", "개시", "종료")
QUANT_HINTS = ("금액", "수치", "몇원", "얼마", "백만원", "천원", "단위")
RISK_HINTS = ("소송", "우발", "약정", "보증", "충당", "충당부채", "담보")
UNIT_HINTS = ("백만원", "천원", "만원", "원")
RISK_REGEX = "(소송|우발|약정|보증|충당|담보)"
RISK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "소송": ("소송", "계류", "법원", "피소", "원고", "피고", "법적"),
    "담보/보증": ("담보", "보증", "지급보증", "연대보증", "제공한 담보", "제공받은 담보", "담보제공", "담보제공자산"),
    "우발부채/약정": ("우발부채", "약정", "확약", "커미트먼트", "commitment", "계약상 의무", "충당"),
}

RISK_NOTE_ANCHORS: dict[str, tuple[str, ...]] = {
    "소송": ("우발부채와약정사항", "소송"),
    "담보/보증": ("우발부채와약정사항", "지급보증", "담보", "제공한담보", "제공받은담보"),
    "우발부채/약정": ("우발부채와약정사항", "우발부채", "약정"),
}

NOTE_QUERY_STOPWORDS = {
    "무엇", "무엇이", "무엇인가", "뭐", "얼마", "얼마인가", "찾아줘", "알려줘", "요약", "설명",
    "있는가", "있나", "질문", "관련", "내용", "금액", "수치", "기말", "당기말", "잔액",
}


@dataclass(slots=True)
class SearchResult:
    original_query: str
    semantic_query: str
    report_year: int | None
    auto_year_applied: bool
    auto_section_type: str | None
    rerank_applied: bool
    rows: list[dict[str, Any]]


def infer_report_year(query: str) -> tuple[int | None, str]:
    """Extract a 4-digit year token and return a cleaned query string."""
    match = YEAR_TOKEN_RE.search(query)
    if not match:
        return None, query

    year = int(match.group(1))
    cleaned = YEAR_TOKEN_RE.sub(" ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return year, cleaned or query


def compact_text(text: str) -> str:
    return WHITESPACE_RE.sub("", text or "")


def normalize_search_query(text: str) -> str:
    compact = compact_text(text)
    for pattern in QUERY_NOISE_PATTERNS:
        compact = pattern.sub("", compact)
    compact = TRAILING_PARTICLE_RE.sub("", compact)
    return compact or compact_text(text)


def infer_section_type_hint(query: str) -> str | None:
    compact_query = compact_text(query)
    for hint in AUDIT_SUBSECTION_HINTS:
        if hint in compact_query:
            return "audit_subsection"
    return None


def infer_sub_section_hint(query: str) -> str | None:
    compact_query = compact_text(query)
    for key, value in SUB_SECTION_HINTS.items():
        if key in compact_query:
            return key
    return None


def has_any_hint(compact_query: str, hints: tuple[str, ...]) -> bool:
    return any(hint in compact_query for hint in hints)


def infer_query_signals(query: str) -> dict[str, Any]:
    compact_query = compact_text(query)
    unit_hint = ""
    for unit in UNIT_HINTS:
        if unit in compact_query:
            unit_hint = unit
            break

    structure_hint = normalize_search_query(query)
    if len(structure_hint) < 2 or len(structure_hint) > 80:
        structure_hint = ""

    return {
        "wants_as_of": has_any_hint(compact_query, AS_OF_HINTS),
        "wants_period": has_any_hint(compact_query, PERIOD_HINTS),
        "wants_quant": has_any_hint(compact_query, QUANT_HINTS),
        "wants_risk": has_any_hint(compact_query, RISK_HINTS),
        "unit_hint": unit_hint,
        "structure_hint": structure_hint,
    }


def detect_risk_types(query: str) -> list[str]:
    compact_query = compact_text(query).lower()
    found: list[str] = []
    for risk_type, keywords in RISK_KEYWORDS.items():
        if any(compact_text(keyword).lower() in compact_query for keyword in keywords):
            found.append(risk_type)
    return found


def _risk_type_to_domains(risk_type: str | None, query: str) -> tuple[str, ...]:
    if not risk_type:
        return tuple()
    if risk_type == "소송":
        return ("litigation",)
    if risk_type == "담보/보증":
        return ("collateral_guarantee",)
    if risk_type == "우발부채/약정":
        q = compact_text(query)
        if "소송" in q:
            return ("litigation",)
        if "담보" in q or "보증" in q:
            return ("collateral_guarantee",)
        return ("contingent_commitment",)
    return tuple()


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+", cleaned) if s.strip()]
    return sents if sents else [cleaned]


def _keyword_overlap_score(text: str, keywords: tuple[str, ...]) -> int:
    low = (text or "").lower()
    return sum(1 for keyword in keywords if keyword.lower() in low)


def _pick_evidence_sentence(text: str, keywords: tuple[str, ...]) -> str:
    def _informative(sent: str) -> bool:
        stripped = sent.strip()
        if len(stripped) < 12:
            return False
        if re.fullmatch(r"[\[\]{}()0-9\-_.\s|:;,/]+", stripped):
            return False
        return True

    best_sent = ""
    best_score = -1
    fallback_sent = ""
    for sent in _split_sentences(text):
        if not fallback_sent and _informative(sent):
            fallback_sent = sent
        score = _keyword_overlap_score(sent, keywords)
        if not _informative(sent):
            continue
        if score > best_score:
            best_score = score
            best_sent = sent
    if best_score <= 0:
        return fallback_sent[:320]
    return best_sent[:320]


def _extract_amount(text: str) -> str:
    cleaned = text or ""

    def _is_date_like(num: str, unit: str, around: str) -> bool:
        if unit:
            return False
        if re.fullmatch(r"(?:19|20)\d{2}[.\-]\d{1,2}(?:[.\-]\d{1,2})?", num):
            return True
        if re.fullmatch(r"(?:19|20)\d{2}", num) and re.search(r"년|월|일", around):
            return True
        if "." in num:
            left = num.split(".", 1)[0]
            if re.fullmatch(r"(?:19|20)\d{2}", left):
                return True
        return False

    best = ""
    best_score = -10**9
    for m in re.finditer(r"(?<!\d)([\-]?\d[\d,]*(?:\.\d+)?)(?:\s*(백만원|천원|원|억원|만원|천))?(?!\d)", cleaned):
        num = m.group(1)
        unit = m.group(2) or ""
        s, e = m.span(1)
        around = cleaned[max(0, s - 12): min(len(cleaned), e + 12)]
        if _is_date_like(num, unit, around):
            continue

        int_part = num.replace(",", "").split(".", 1)[0].lstrip("-")
        if not unit and "." in num:
            continue
        if not unit and "," not in num:
            if re.fullmatch(r"\d{1,3}", int_part):
                continue
            if re.fullmatch(r"(?:19|20)\d{2}", int_part):
                continue

        score = 0
        if unit:
            score += 5
        if "," in num:
            score += 4
        if len(int_part) >= 4:
            score += 2
        if re.search(r"기말|잔액|금액|계|한도|차입금|보증", around):
            score += 3

        if score > best_score:
            best_score = score
            best = f"{num}{unit}"

    return best


def _is_amount_question(query: str) -> bool:
    if _is_rate_question(query):
        return False
    compact_query = compact_text(query)
    if has_any_hint(compact_query, QUANT_HINTS):
        return True
    return any(token in compact_query for token in ("잔액", "총액", "규모", "증감", "기말", "예정액", "상환액"))


def _is_rate_question(query: str) -> bool:
    compact_query = compact_text(query)
    return any(token in compact_query for token in ("연이자율", "이자율", "금리", "%"))


def _derive_subtopic_column_hints(query: str, subtopic: str) -> list[str]:
    hints: list[str] = []
    for token in [subtopic, *expand_note_keywords(extract_note_keywords(query))]:
        tok = compact_text(token)
        if not tok:
            continue
        if tok.endswith("충당부채") and tok != "충당부채":
            tok = tok[: -len("충당부채")]
        if len(tok) >= 2:
            hints.append(tok)
    return list(dict.fromkeys(hints))


def _clean_table_label(label: str) -> str:
    text = re.sub(r"\([^)]*\)", "", label or "")
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip(" |:-")
    return text


def _extract_table_query_hints(query: str, row_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    year, _ = infer_report_year(query)
    if _is_rate_question(query):
        ask_type = "rate"
    elif _is_amount_question(query):
        ask_type = "amount"
    else:
        ask_type = "description"
    tokens = expand_note_keywords(extract_note_keywords(query))
    compact_query = compact_text(query)

    row_keys: list[str] = []
    col_keys: list[str] = []
    note_keys: list[str] = []
    topic_keywords: list[str] = []
    aggregate_intent = any(key in compact_query for key in ("총액", "합계", "전체", "총계", "소계"))
    row_role_hint: str | None = None
    period_role_hint: str | None = None
    table_family_hint: str | None = None
    entity_keys: list[str] = []

    explicit_row_intent = False
    saw_provision_keyword = False

    for token in tokens:
        tok = compact_text(token)
        if not tok:
            continue

        if tok == "기타" and "기타충당부채" in compact_query:
            pass
        elif any(key in tok for key in ("기초", "기말", "순전입", "환입", "사용액")) or tok in {"기타"}:
            row_keys.append(tok)
            explicit_row_intent = True
        if any(key in tok for key in ("리스부채", "예금", "현금및현금성자산", "지급보증")):
            row_keys.append(tok)
            topic_keywords.append(tok)
        if any(
            key in tok
            for key in (
                "판매보증",
                "기술사용료",
                "장기성과급",
                "기타충당부채",
                "유동성장기차입금",
                "장기차입금",
                "관련차입금",
                "채무보증한도",
                "연이자율",
                "이자율",
                "금리",
            )
        ):
            col_keys.append(tok)
            topic_keywords.append(tok)
        if tok.endswith("충당부채") and tok != "충당부채":
            col_keys.append(tok[: -len("충당부채")])
            saw_provision_keyword = True

        if any(key in tok for key in ("현금및현금성자산", "충당부채", "차입금", "우발부채", "약정", "지급보증")):
            note_keys.append(tok)

    if "기말" in compact_query and "기말" not in row_keys:
        row_keys.append("기말")
        row_role_hint = "ending"
    if "예금" in compact_query:
        row_keys.extend(["예금등", "예금"])
        explicit_row_intent = True
    if "현금" in compact_query:
        row_keys.append("현금")
        explicit_row_intent = True
    if "기초" in compact_query:
        row_keys.append("기초")
        explicit_row_intent = True
        row_role_hint = "opening"
    if "순전입액" in compact_query or "순전입" in compact_query:
        row_keys.extend(["순전입액", "순전입액환입액", "순전입"])
        explicit_row_intent = True
        row_role_hint = "inflow"
    if "사용액" in compact_query:
        row_keys.append("사용액")
        explicit_row_intent = True
        row_role_hint = "outflow"
    if "기타충당부채" in compact_query:
        col_keys.append("기타충당부채")
    if "관련차입금" in compact_query:
        col_keys.append("관련차입금")
    if "채무보증한도" in compact_query:
        col_keys.append("채무보증한도")
    if "실차입금기준" in compact_query:
        row_keys.append("실차입금기준")
        topic_keywords.append("실차입금기준")
    if "한도기준" in compact_query:
        row_keys.append("한도기준")
        topic_keywords.append("한도기준")
    if "리스부채" in compact_query and "리스부채" not in row_keys:
        row_keys.append("리스부채")
    if "유동성장기차입금" in compact_query and "유동성장기차입금" not in col_keys:
        col_keys.append("유동성장기차입금")
    if "유동성장기차입금" in compact_query and "유동성장기차입금" not in row_keys:
        row_keys.append("유동성장기차입금")
    if "장기차입금" in compact_query and "장기차입금" not in col_keys:
        col_keys.append("장기차입금")
    if "장기차입금" in compact_query and "장기차입금" not in row_keys:
        row_keys.append("장기차입금")
    if saw_provision_keyword and not explicit_row_intent:
        row_keys.append("기말")
    if aggregate_intent:
        row_keys.append("계")
        row_role_hint = "aggregate"

    if "당기말" in compact_query:
        period_role_hint = "current_end"
    elif "전기말" in compact_query:
        period_role_hint = "prior_end"
    elif "당기" in compact_query:
        period_role_hint = "current"
    elif "전기" in compact_query:
        period_role_hint = "prior"

    # 기말 질의의 기본 period 해석(당기말 우선)
    if not period_role_hint and "기말" in compact_query and "전기말" not in compact_query:
        period_role_hint = "current_end"

    if "현금및현금성자산" in compact_query or "예금" in compact_query:
        table_family_hint = "cash"
    elif "충당부채" in compact_query:
        table_family_hint = "provision"
    elif "지급보증" in compact_query or "관련차입금" in compact_query or "채무보증한도" in compact_query:
        table_family_hint = "guarantee"
    elif "차입금" in compact_query or "리스부채" in compact_query:
        table_family_hint = "loan_lease"

    query_lower = compact_query.lower()
    if "setk" in query_lower:
        entity_keys.append("setk")
    if "samcol" in query_lower:
        entity_keys.append("samcol")
    if "기타" in compact_query:
        entity_keys.append("기타")

    target_row_year = year
    # "2024년 ... 연도별 상환계획 ... 2025년"처럼 보고연도와 행연도가 함께 있을 때
    # 마지막 연도를 행연도로 해석한다.
    all_years = [int(y) for y in YEAR_TOKEN_RE.findall(query or "")]
    if len(all_years) >= 2 and any(k in compact_query for k in ("행연도", "연도별", "상환계획", "예정액")):
        target_row_year = all_years[-1]

    if "현금및현금성자산" in compact_query:
        note_keys.append("현금및현금성자산")
    if "현금흐름표" in compact_query:
        note_keys.append("현금흐름표")
    if "지급보증" in compact_query:
        note_keys.append("지급보증")

    topic = ""
    if row_meta:
        topic = str(row_meta.get("subtopic") or row_meta.get("note_title") or "")
    if not topic and tokens:
        topic = tokens[0]

    return {
        "year": year,
        "topic": topic,
        "row_keys": list(dict.fromkeys([k for k in row_keys if len(k) >= 2])),
        "col_keys": list(dict.fromkeys([k for k in col_keys if len(k) >= 2])),
        "note_keys": list(dict.fromkeys([k for k in note_keys if len(k) >= 2])),
        "topic_keywords": list(dict.fromkeys([k for k in topic_keywords if len(k) >= 2])),
        "ask_type": ask_type,
        "aggregate_intent": aggregate_intent,
        "row_role_hint": row_role_hint,
        "period_role_hint": period_role_hint,
        "table_family_hint": table_family_hint,
        "target_row_year": target_row_year,
        "entity_keys": list(dict.fromkeys([k for k in entity_keys if k])),
        "query_compact": compact_query,
    }


def _is_date_like_value(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    if re.fullmatch(r"(?:19|20)\d{2}[.\-]\d{1,2}(?:[.\-]\d{1,2})?", v):
        return True
    if re.fullmatch(r"(?:19|20)\d{2}", v):
        return True
    if re.fullmatch(r"(?:19|20)\d{2}[.]\d{1,2}", v):
        return True
    return False


def _parse_pipe_table_cell_candidates(text: str, row_meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if "|" not in (text or ""):
        return []

    cells = [c.strip() for c in (text or "").split("|") if c.strip()]
    if len(cells) < 4:
        return []

    compact_cells = [compact_text(c) for c in cells]
    unit_match = re.search(r"단위\s*[:：]?\s*\(?\s*([A-Za-z가-힣$]+)\s*\)?", text or "")
    unit = unit_match.group(1) if unit_match else ""

    header_anchor = next((i for i, c in enumerate(compact_cells) if "구분" in c), -1)
    if header_anchor < 0:
        header_anchor = next(
            (
                i
                for i, c in enumerate(compact_cells)
                if any(key in c for key in ("판매보증", "기술사용료", "장기성과급", "유동성장기차입금", "장기차입금", "관련차입금", "채무보증한도"))
            ),
            -1,
        )
    if header_anchor < 0:
        return []

    stop_row_tokens = ("기초", "순전입", "환입", "사용액", "기말", "리스부채", "예금", "계")
    col_labels: list[str] = []
    for raw, comp in zip(cells[header_anchor + 1:], compact_cells[header_anchor + 1:], strict=False):
        if not comp:
            continue
        if any(token in comp for token in stop_row_tokens):
            break
        if re.search(r"(?<!\d)[\-]?\d[\d,]*(?:\.\d+)?", comp):
            break
        col_labels.append(raw)
        if len(col_labels) >= 8:
            break

    if not col_labels:
        col_labels = ["금액"]

    start_idx = header_anchor + 1 + len(col_labels)
    candidates: list[dict[str, Any]] = []
    row_idx = 0
    i = start_idx
    while i < len(cells) - 1 and row_idx < 60:
        row_label_raw = _clean_table_label(cells[i])
        row_label_comp = compact_text(row_label_raw)
        if not row_label_comp:
            i += 1
            continue
        if re.search(r"(?<!\d)[\-]?\d[\d,]*(?:\.\d+)?", row_label_comp):
            i += 1
            continue

        j = i + 1
        values: list[tuple[int, str]] = []
        col_idx = 0
        guard = 0
        while j < len(cells) and col_idx < len(col_labels) and guard < len(col_labels) + 8:
            guard += 1
            token = cells[j].strip()
            token_comp = compact_text(token)
            if col_idx > 0 and not re.search(r"\d", token_comp) and any(t in token_comp for t in stop_row_tokens):
                break
            amt = _extract_amount(token)
            if amt and not _is_date_like_value(amt):
                values.append((col_idx, amt))
                col_idx += 1
            elif token in {"-", "—"}:
                values.append((col_idx, token))
                col_idx += 1
            j += 1

        if values:
            for cidx, value in values:
                col_label_raw = col_labels[cidx] if cidx < len(col_labels) else f"col{cidx + 1}"
                candidates.append(
                    {
                        "table_title": str((row_meta or {}).get("subtopic") or (row_meta or {}).get("note_title") or ""),
                        "note_title": str((row_meta or {}).get("note_title") or ""),
                        "subtopic": str((row_meta or {}).get("subtopic") or ""),
                        "header_rows": "|".join(_clean_table_label(c) for c in col_labels),
                        "row_label": row_label_raw,
                        "col_label": _clean_table_label(col_label_raw),
                        "value": value,
                        "unit": unit,
                        "row_index": row_idx,
                        "col_index": cidx,
                    }
                )
            row_idx += 1
            i = j
        else:
            i += 1

    return candidates


def _score_table_cell_candidate(
    candidate: dict[str, Any],
    query_hints: dict[str, Any],
    column_hints: tuple[str, ...],
    row_meta: dict[str, Any] | None,
) -> float:
    score = 0.0
    row_label = compact_text(str(candidate.get("row_label") or ""))
    col_label = compact_text(str(candidate.get("col_label") or ""))
    value = str(candidate.get("value") or "")
    unit = str(candidate.get("unit") or "")

    row_match = False
    col_match = False

    for key in query_hints.get("row_keys", []):
        ck = compact_text(str(key))
        if not ck:
            continue
        if ck == row_label:
            score += 8.0
            row_match = True
        elif ck in row_label:
            score += 3.8
            row_match = True
    for key in query_hints.get("col_keys", []):
        ck = compact_text(str(key))
        if not ck:
            continue
        if ck == col_label:
            score += 8.5
            col_match = True
        elif ck in col_label:
            score += 4.2
            col_match = True
    for key in column_hints:
        ck = compact_text(str(key))
        if ck and ck in col_label:
            score += 5.0
            col_match = True

    if query_hints.get("ask_type") == "amount":
        score += 0.8
    if "기말" in compact_text(query_hints.get("row_keys", [""])[0] if query_hints.get("row_keys") else "") and "기말" in row_label:
        score += 1.5

    if query_hints.get("row_keys") and not row_match:
        score -= 5.0
    if query_hints.get("col_keys") and not col_match:
        score -= 5.0

    if bool(query_hints.get("aggregate_intent")):
        if ("계" in row_label) or ("합계" in row_label) or ("총액" in row_label):
            score += 6.0
        else:
            score -= 4.0

    topic = compact_text(str(query_hints.get("topic") or ""))
    note_blob = compact_text(
        str((row_meta or {}).get("note_title") or "") + " " + str((row_meta or {}).get("subtopic") or "")
    )
    if topic and topic in note_blob:
        score += 1.0

    for key in query_hints.get("note_keys", []):
        ck = compact_text(str(key))
        if ck and ck in note_blob:
            score += 2.0

    if _is_date_like_value(value):
        score -= 20.0
    if "," in value:
        score += 1.2
    if unit:
        score += 0.8

    return score


def _build_compact_row_summary(cells: list[dict[str, Any]], row_label: str) -> str:
    if not cells:
        return ""
    sorted_cells = sorted(cells, key=lambda x: int(x.get("col_index") or 0))
    parts = []
    for cell in sorted_cells[:8]:
        col = _clean_table_label(str(cell.get("col_label") or ""))
        val = str(cell.get("value") or "")
        if not col or not val:
            continue
        parts.append(f"{col} {val}")
    if not parts:
        return ""
    return f"{_clean_table_label(row_label)} | " + " | ".join(parts)


def _extract_loan_lease_amount_by_pattern(text: str, query_hints: dict[str, Any]) -> tuple[str, str]:
    compact_query = str(query_hints.get("query_compact") or "")
    wants_liquidity = "유동성장기차입금" in compact_query
    wants_longterm = ("장기차입금" in compact_query) and not wants_liquidity
    wants_lease = "리스부채" in compact_query
    if not wants_lease and not (wants_liquidity or wants_longterm):
        return "", ""

    row_patterns: list[tuple[str, str]] = []
    if wants_liquidity:
        row_patterns.append(("유동성장기차입금", r"유동성\s*장?기차입금\s*:\s*리스부채[^|]*\|\s*[^|]*\|\s*[^|]*\|\s*([0-9][0-9,]*)\s*\|\s*([0-9][0-9,]*)"))
    if wants_longterm:
        row_patterns.append(("장기차입금", r"(?:^|[\s|])장기차입금\s*:\s*리스부채[^|]*\|\s*[^|]*\|\s*[^|]*\|\s*([0-9][0-9,]*)\s*\|\s*([0-9][0-9,]*)"))
    if not row_patterns:
        row_patterns = [
            ("유동성장기차입금", r"유동성\s*장?기차입금\s*:\s*리스부채[^|]*\|\s*[^|]*\|\s*[^|]*\|\s*([0-9][0-9,]*)\s*\|\s*([0-9][0-9,]*)"),
            ("장기차입금", r"(?:^|[\s|])장기차입금\s*:\s*리스부채[^|]*\|\s*[^|]*\|\s*[^|]*\|\s*([0-9][0-9,]*)\s*\|\s*([0-9][0-9,]*)"),
        ]

    for row_label, pattern in row_patterns:
        m = re.search(pattern, text or "", flags=re.IGNORECASE)
        if not m:
            continue
        current_amount = m.group(1)
        prev_amount = m.group(2)
        evidence = f"{row_label}: 리스부채 | 당기말 {current_amount} | 전기말 {prev_amount}"
        return evidence[:320], current_amount

    return "", ""


def _extract_cash_amount_by_pattern(text: str, query_hints: dict[str, Any]) -> tuple[str, str]:
    compact_query = str(query_hints.get("query_compact") or "")
    if "현금및현금성자산" not in compact_query and "예금" not in compact_query:
        return "", ""

    wants_cash = "현금" in compact_query
    wants_deposit = ("예금" in compact_query) or ("예금등" in compact_query)

    if wants_cash:
        row_candidates = ["현금"]
    elif wants_deposit:
        row_candidates = ["예금등", "예금", "보통예금", "정기예금"]
    else:
        row_candidates = ["현금", "예금등", "예금", "보통예금", "정기예금"]

    for row_key in row_candidates:
        m = re.search(rf"{row_key}[^|]*\|\s*([0-9][0-9,]*)", text or "")
        if m:
            amount = m.group(1)
            return f"{row_key} | 금액 {amount}", amount
    return "", ""


def _extract_guarantee_amount_by_pattern(text: str, query_hints: dict[str, Any]) -> tuple[str, str]:
    compact_query = str(query_hints.get("query_compact") or "")
    if not any(key in compact_query for key in ("지급보증", "관련차입금", "채무보증한도", "실차입금기준", "한도기준")):
        return "", ""

    wants_related = "관련차입금" in compact_query
    wants_limit = "채무보증한도" in compact_query
    wants_aggregate = bool(query_hints.get("aggregate_intent")) or any(key in compact_query for key in ("총액", "합계", "계"))

    if "실차입금기준" in compact_query:
        m = re.search(r"실차입금\s*기준[^0-9]{0,30}([0-9][0-9,]*)", text or "")
        if m:
            amount = m.group(1)
            return f"실차입금 기준 | 금액 {amount}", amount

    if "한도기준" in compact_query:
        m = re.search(r"한도\s*기준[^0-9]{0,30}([0-9][0-9,]*)", text or "")
        if m:
            amount = m.group(1)
            return f"한도 기준 | 금액 {amount}", amount

    if wants_aggregate:
        m = re.search(r"계\s*\|\s*([0-9][0-9,]*)[^|]*\|\s*([0-9][0-9,]*)", text or "")
        if m:
            related = m.group(1)
            limit = m.group(2)
            if wants_related and not wants_limit:
                return f"계 | 관련차입금 {related} | 채무보증한도 {limit}", related
            if wants_limit and not wants_related:
                return f"계 | 관련차입금 {related} | 채무보증한도 {limit}", limit
            return f"계 | 관련차입금 {related} | 채무보증한도 {limit}", limit if wants_limit else related

        # 초기 연도 일부 표는 합계 행 없이 당기말|전기말 2열만 제공됨.
        # 이 경우 질의 의도에 따라 첫 데이터행의 당기말/전기말 값을 보정 추출한다.
        m = re.search(
            r"(?:당기말|당기)\s*\|\s*(?:전기말|전기)\s*(?:\n|\s)+[^|\n]+\|\s*([0-9][0-9,]*)\s*\|\s*([0-9][0-9,]*)",
            text or "",
        )
        if m:
            current_amount = m.group(1)
            prior_amount = m.group(2)
            if wants_limit and not wants_related:
                return f"당기말 {current_amount} | 전기말 {prior_amount}", prior_amount
            if wants_related and not wants_limit:
                return f"당기말 {current_amount} | 전기말 {prior_amount}", current_amount
            return f"당기말 {current_amount} | 전기말 {prior_amount}", current_amount

    return "", ""


def _extract_provision_amount_by_pattern(
    text: str,
    query_hints: dict[str, Any],
    column_hints: tuple[str, ...],
) -> tuple[str, str]:
    compact = compact_text(text)
    if not all(key in compact for key in ("판매보증", "기술사용료", "장기성과급")):
        return "", ""

    row_keys = [compact_text(k) for k in query_hints.get("row_keys", []) if k]
    col_keys = [compact_text(k) for k in [*query_hints.get("col_keys", []), *column_hints] if k]

    target_row = "기말"
    for k in row_keys:
        if "순전입" in k or "환입" in k:
            target_row = "순전입액"
            break
        if "사용액" in k:
            target_row = "사용액"
            break
        if "기초" in k:
            target_row = "기초"
            break
        if k in {"기타", "기타액", "기타항목"}:
            target_row = "기타"
            break
        if "기말" in k:
            target_row = "기말"
            break

    target_col = "판매보증"
    for k in col_keys:
        if "기타충당부채" in k:
            target_col = "기타충당부채"
            break
        if "기술사용료" in k:
            target_col = "기술사용료"
            break
        if "장기성과급" in k:
            target_col = "장기성과급"
            break
        if "판매보증" in k:
            target_col = "판매보증"
            break
        if k in {"계", "합계", "총액"}:
            target_col = "계"
            break

    row_patterns = {
        "기초": r"기초\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|\n]+)",
        "순전입액": r"순\s*전입액(?:\(환입액\))?\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|\n]+)",
        "사용액": r"사용액\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|\n]+)",
        "기타": r"기타\(\*\)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|\n]+)",
        "기말": r"기말\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|\n]+)",
    }
    m = re.search(row_patterns[target_row], text or "")
    if not m:
        return "", ""

    labels = ["판매보증", "기술사용료", "장기성과급", "기타충당부채", "계"]
    values_raw = [m.group(i) for i in range(1, 6)]
    values = [_extract_amount(v) or "-" for v in values_raw]
    col_idx = labels.index(target_col) if target_col in labels else 0
    amount = values[col_idx] if col_idx < len(values) else ""
    if not amount or amount == "-":
        return "", ""

    evidence = f"{target_row} | " + " | ".join(f"{labels[i]} {values[i]}" for i in range(5))
    return evidence[:320], amount


def _score_structured_cell_candidate(
    cell: dict[str, Any],
    query_hints: dict[str, Any],
    note_pref_blobs: tuple[str, ...],
) -> float:
    score = 0.0
    row_label = compact_text(str(cell.get("row_label") or ""))
    row_group = compact_text(str(cell.get("row_group") or ""))
    col_label = compact_text(str(cell.get("col_label") or ""))
    row_label_norm = compact_text(str(cell.get("row_label_norm") or row_label))
    row_group_norm = compact_text(str(cell.get("row_group_norm") or row_group))
    col_label_norm = compact_text(str(cell.get("col_label_norm") or col_label))
    table_title_norm = compact_text(str(cell.get("table_title_norm") or ""))
    note_blob = compact_text(str(cell.get("note_title_norm") or "") + " " + str(cell.get("subtopic_norm") or ""))
    query_compact = compact_text(str(query_hints.get("query_compact") or ""))
    ask_type = str(query_hints.get("ask_type") or "")
    year_row_intent = any(k in query_compact for k in ("행연도", "연도별", "상환계획", "예정액"))

    row_hit = False
    col_hit = False
    for key in query_hints.get("row_keys", []):
        ck = compact_text(str(key))
        if not ck:
            continue
        if ck == row_label_norm or ck == row_group_norm:
            score += 8.0
            row_hit = True
        elif ck in row_label_norm or ck in row_group_norm:
            score += 4.0
            row_hit = True
    for key in query_hints.get("col_keys", []):
        ck = compact_text(str(key))
        if not ck:
            continue
        if ck == col_label_norm:
            score += 9.0
            col_hit = True
        elif ck in col_label_norm:
            score += 4.5
            col_hit = True

    target_row_year = query_hints.get("target_row_year")
    if query_hints.get("row_keys") and not row_hit:
        if year_row_intent and target_row_year and cell.get("row_year") and int(cell.get("row_year")) == int(target_row_year):
            score -= 0.5
        else:
            score -= 5.0
    if query_hints.get("col_keys") and not col_hit:
        score -= 5.0

    if bool(query_hints.get("aggregate_intent")):
        if bool(cell.get("is_aggregate")):
            score += 6.0
        else:
            score -= 4.0
    elif bool(cell.get("is_aggregate")) and "guarantee" in compact_text(str(cell.get("table_family") or "")):
        # 지급보증의 비-합계 질의에서는 합계행 과선택 방지
        score -= 2.5

    row_role_hint = query_hints.get("row_role_hint")
    period_role_hint = query_hints.get("period_role_hint")
    if row_role_hint:
        if row_role_hint == cell.get("row_role"):
            score += 5.0
        elif cell.get("row_role"):
            score -= 2.0
    if period_role_hint:
        if period_role_hint == cell.get("period_role"):
            score += 4.0
        elif cell.get("period_role"):
            score -= 1.5

    if target_row_year:
        if cell.get("row_year"):
            if int(cell.get("row_year")) == int(target_row_year):
                score += 8.0
            else:
                score -= 3.0
        elif year_row_intent:
            score -= 6.0

    entity_keys = [compact_text(str(k)) for k in query_hints.get("entity_keys", [])]
    entity_label = compact_text(str(cell.get("entity_label") or ""))
    if entity_keys:
        if any(k and k in entity_label for k in entity_keys):
            score += 5.0
        elif entity_label:
            score -= 2.5

    for key in query_hints.get("note_keys", []):
        ck = compact_text(str(key))
        if ck and ck in note_blob:
            score += 2.5

    for pref in note_pref_blobs:
        if pref and pref in note_blob:
            score += 2.0

    table_family = str(cell.get("table_family") or "")
    table_family_hint = str(query_hints.get("table_family_hint") or "")
    if table_family_hint:
        if table_family == table_family_hint:
            score += 6.0
        else:
            score -= 4.0

    if "현금및현금성자산" in query_compact and table_family == "cash":
        score += 5.5
    if "지급보증" in query_compact and table_family == "guarantee":
        score += 4.0
    if "충당부채" in query_compact and table_family == "provision":
        score += 4.0
    if ("차입금" in query_compact or "리스부채" in query_compact) and table_family == "loan_lease":
        score += 4.0

    # 동일 family 내에서 표 제목 의미(리스부채/장기차입금/사채)를 추가 반영
    if "리스부채" in query_compact:
        if "리스부채" in table_title_norm:
            score += 6.5
        elif "장기차입금" in table_title_norm:
            score -= 3.0
    if "장기차입금" in query_compact:
        if "장기차입금" in table_title_norm:
            score += 5.5
        elif "리스부채" in table_title_norm:
            score -= 2.0
    if "사채" in query_compact:
        if "사채" in table_title_norm or "사채" in note_blob:
            score += 5.0
        elif table_family in {"loan_lease", "guarantee"}:
            score -= 2.5

    # amount 질의에서 rate/date/text 혼입 방지 (리스부채 케이스 1.8 오답 방지 포함)
    value_raw = str(cell.get("value_raw") or "").strip()
    value_type = str(cell.get("value_type") or "")
    col_blob = f"{col_label_norm} {compact_text(str(cell.get('col_label') or ''))}"
    row_blob = f"{row_group_norm} {row_label_norm}"
    if ask_type == "amount":
        if value_type in {"rate", "date", "text"}:
            score -= 18.0
        if any(k in col_blob for k in ("연이자율", "이자율", "rate")):
            score -= 18.0
        try:
            vn = float(cell.get("value_numeric")) if cell.get("value_numeric") is not None else None
            if vn is not None and abs(vn) < 10 and "." in value_raw and "," not in value_raw:
                score -= 14.0
        except Exception:
            pass
        if "," in value_raw:
            score += 2.0
        if any(k in col_blob for k in ("금액", "당기말", "전기말", "관련차입금", "채무보증한도")):
            score += 2.5

        # 지급보증 컬럼 의도 정렬
        if "지급보증" in query_compact:
            wants_related = "관련차입금" in query_compact
            wants_limit = "채무보증한도" in query_compact
            if wants_related and "관련차입금" in col_blob:
                score += 5.5
            elif wants_related and "채무보증한도" in col_blob:
                score -= 4.0
            if wants_limit and "채무보증한도" in col_blob:
                score += 5.5
            elif wants_limit and "관련차입금" in col_blob:
                score -= 4.0

        # 리스부채(유동성/장기) 행 그룹 정렬
        if ("차입금" in query_compact or "리스부채" in query_compact) and table_family == "loan_lease":
            wants_liquidity = "유동성장기차입금" in query_compact
            wants_longterm = ("장기차입금" in query_compact) and not wants_liquidity
            if wants_liquidity:
                if "유동성장기차입금" in row_blob:
                    score += 7.5
                elif "장기차입금" in row_blob:
                    score -= 5.0
            elif wants_longterm:
                if "장기차입금" in row_blob and "유동성장기차입금" not in row_blob:
                    score += 7.0
                elif "유동성장기차입금" in row_blob:
                    score -= 5.0
    elif ask_type == "rate":
        if value_type == "rate":
            score += 10.0
        elif value_type == "amount":
            score -= 8.0
        else:
            score -= 10.0

        if any(k in col_blob for k in ("연이자율", "이자율", "rate", "%", "금리")):
            score += 7.0
        else:
            score -= 3.0

        if "%" in value_raw:
            score += 1.5

    try:
        score += float(cell.get("parse_confidence") or 0.0) * 2.0
    except Exception:
        pass

    if "현금및현금성자산" in query_compact:
        if "현금및현금성자산" in note_blob:
            score += 7.0
        if "현금흐름표" in note_blob:
            score -= 8.0

    if "리스부채" in query_compact and ("리스부채" in row_label or "리스부채" in row_group):
        score += 3.0

    if ask_type == "amount" and period_role_hint:
        if period_role_hint == cell.get("period_role"):
            score += 1.5
        elif cell.get("period_role"):
            score -= 2.5

    return score


def _normalize_amount_output(amount: str) -> str:
    a = re.sub(r"\s+", " ", amount or "").strip()
    if not a:
        return ""
    # 괄호 음수는 일관되게 음수부호로 정규화: (590,625) -> -590,625
    if re.fullmatch(r"\(\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\)", a):
        inner = a.strip()[1:-1].strip()
        inner = inner.lstrip("+")
        if not inner.startswith("-"):
            inner = f"-{inner}"
        return inner
    return a


def _lookup_structured_cell_answer(
    query_hints: dict[str, Any],
    rows: list[dict[str, Any]],
    report_year: int | None,
) -> dict[str, Any] | None:
    if not report_year:
        return None

    note_nos = [int(r.get("note_no")) for r in rows if r.get("note_no") is not None]
    note_pref_blobs = tuple(
        compact_text(str(r.get("note_title") or "") + " " + str(r.get("subtopic") or ""))
        for r in rows[:5]
    )
    target_value_type = "rate" if str(query_hints.get("ask_type") or "") == "rate" else "amount"

    def _fetch_cells(use_note_filter: bool) -> list[dict[str, Any]]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if use_note_filter and note_nos:
                    cur.execute(
                        """
                        SELECT
                            table_id, report_year, note_no, note_title, subtopic, table_title, section_type, unit,
                            row_index, col_index, row_group, row_label, col_label,
                            value_raw, value_numeric, value_type, currency, period_type,
                            table_family, table_title_norm, note_title_norm, subtopic_norm,
                            row_label_norm, col_label_norm, row_group_norm,
                            row_role, period_role, is_aggregate, row_year, entity_label, parse_confidence
                        FROM structured_table_cells
                        WHERE report_year = %s
                                                    AND value_type = %s
                          AND note_no = ANY(%s)
                        LIMIT 1200
                        """,
                                                (report_year, target_value_type, note_nos),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            table_id, report_year, note_no, note_title, subtopic, table_title, section_type, unit,
                            row_index, col_index, row_group, row_label, col_label,
                                                        value_raw, value_numeric, value_type, currency, period_type,
                                                        table_family, table_title_norm, note_title_norm, subtopic_norm,
                                                        row_label_norm, col_label_norm, row_group_norm,
                                                        row_role, period_role, is_aggregate, row_year, entity_label, parse_confidence
                        FROM structured_table_cells
                        WHERE report_year = %s
                                                    AND value_type = %s
                        LIMIT 2000
                        """,
                                                (report_year, target_value_type),
                    )
                return list(cur.fetchall())

    try:
        cells = _fetch_cells(use_note_filter=True)
    except Exception:
        return None
    if not cells:
        try:
            cells = _fetch_cells(use_note_filter=False)
        except Exception:
            return None
    if not cells:
        return None

    table_family_hint = str(query_hints.get("table_family_hint") or "")
    if table_family_hint:
        fam_cells = [c for c in cells if str(c.get("table_family") or "") == table_family_hint]
        if fam_cells:
            cells = fam_cells

    scored = [(_score_structured_cell_candidate(c, query_hints, note_pref_blobs), c) for c in cells]
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return None

    best_score, best = scored[0]
    best_table = best.get("table_id")
    best_row_idx = best.get("row_index")

    qcompact = compact_text(str(query_hints.get("query_compact") or ""))

    if best.get("table_family") == "guarantee":
        if bool(query_hints.get("aggregate_intent")) and not bool(best.get("is_aggregate")):
            # 지급보증 총액 질의에서 집계행이 아닌 경우는 구조화 스키마가 불충분한 케이스로 간주
            return None
        if ("관련차입금" in qcompact or "채무보증한도" in qcompact):
            best_col = compact_text(str(best.get("col_label") or ""))
            if best_col.startswith("col") or best_col in {"당기말", "전기말"}:
                # 컬럼명이 일반/기간 라벨만 존재하면 관련차입금 vs 채무보증한도 구분 불가
                return None

    if (
        best.get("table_family") == "loan_lease"
        and "리스부채" in qcompact
        and ("유동성장기차입금" in qcompact or "장기차입금" in qcompact)
        and not compact_text(str(best.get("row_group") or ""))
    ):
        row_blob = compact_text(str(best.get("row_label") or ""))
        wants_liquidity = "유동성장기차입금" in qcompact
        wants_longterm = ("장기차입금" in qcompact) and not wants_liquidity
        if (wants_liquidity and "유동성장기차입금" not in row_blob) or (
            wants_longterm and ("장기차입금" not in row_blob or "유동성장기차입금" in row_blob)
        ):
            # 유동성/장기 구분 키가 전혀 없을 때만 fallback 이관
            return None

    row_cells = [
        c for _, c in scored
        if c.get("table_id") == best_table and c.get("row_index") == best_row_idx
    ]
    row_cells.sort(key=lambda c: int(c.get("col_index") or 0))

    row_label = _clean_table_label(str(best.get("row_label") or ""))
    if best.get("row_group"):
        rg = _clean_table_label(str(best.get("row_group") or ""))
        if rg:
            row_label = f"{rg}: {row_label}" if row_label else rg

    parts = []
    for c in row_cells[:8]:
        col = _clean_table_label(str(c.get("col_label") or ""))
        val = str(c.get("value_raw") or "")
        if col and val:
            parts.append(f"{col} {val}")
    evidence = (f"{row_label} | " + " | ".join(parts)).strip(" |") if parts else row_label

    return {
        "year": best.get("report_year"),
        "note_no": best.get("note_no"),
        "note_title": best.get("note_title") or "",
        "subtopic": best.get("subtopic") or "",
        "amount": _normalize_amount_output(str(best.get("value_raw") or "")),
        "evidence_sentence": evidence[:320],
        "table_id": best_table,
        "col_label": str(best.get("col_label") or ""),
        "score": best_score,
    }


def _extract_table_amount_by_column(
    text: str,
    column_hints: tuple[str, ...],
    row_keys: tuple[str, ...] = (),
    aggregate_intent: bool = False,
) -> tuple[str, str]:
    if "|" not in (text or ""):
        return "", ""

    cells = [c.strip() for c in (text or "").split("|") if c.strip()]
    if not cells:
        return "", ""

    compact_cells = [compact_text(c) for c in cells]
    header_anchor = next((i for i, c in enumerate(compact_cells) if c in {"구분", "구분"} or "구분" in c), -1)
    if header_anchor < 0:
        header_anchor = next((i for i, c in enumerate(compact_cells) if "판매보증" in c or "기술사용료" in c or "장기성과급" in c), -1)
    if header_anchor < 0:
        return "", ""

    header_labels_compact: list[str] = []
    header_labels_raw: list[str] = []
    for raw_c, c in zip(cells[header_anchor + 1:], compact_cells[header_anchor + 1:], strict=False):
        if not c:
            continue
        if "기초" in c or "순전입" in c or "환입" in c or "사용액" in c or "기말" in c:
            break
        if re.search(r"(?<!\d)[\-]?\d[\d,]*(?:\.\d+)?", c):
            break
        header_labels_compact.append(c)
        header_labels_raw.append(raw_c)
    if not header_labels_compact:
        return "", ""

    target_col = -1
    best_col_score = -1.0
    for idx, label in enumerate(header_labels_compact):
        score = 0.0
        for hint in column_hints:
            h = compact_text(hint)
            if not h:
                continue
            if h == label:
                score += 8.0
            elif h in label:
                score += 4.0
        if score > best_col_score:
            best_col_score = score
            target_col = idx
    if target_col < 0:
        return "", ""

    desired_rows = [compact_text(k) for k in row_keys if k]
    if aggregate_intent:
        desired_rows = [*desired_rows, "계", "합계", "총액"]
    if not desired_rows:
        desired_rows = ["기말"]

    row_idx = -1
    for key in desired_rows:
        row_idx = next((i for i, c in enumerate(compact_cells) if key and (c == key or key in c)), -1)
        if row_idx >= 0:
            break
    if row_idx < 0:
        return "", ""

    row_amounts: list[str] = []
    for c in cells[row_idx + 1: row_idx + 1 + max(len(header_labels_compact) + 2, 8)]:
        amt = _extract_amount(c)
        if amt:
            row_amounts.append(amt)
    if target_col >= len(row_amounts):
        return "", ""

    summary_pairs: list[str] = []
    pair_count = min(len(header_labels_raw), len(row_amounts))
    for i in range(pair_count):
        label = _clean_table_label(header_labels_raw[i]) or _clean_table_label(header_labels_compact[i])
        if not label:
            continue
        summary_pairs.append(f"{label} {row_amounts[i]}")

    row_label_display = _clean_table_label(cells[row_idx]) or "기말"
    if summary_pairs:
        evidence = f"{row_label_display} | " + " | ".join(summary_pairs[:6])
    else:
        evidence = ""

    if not evidence:
        for sent in _split_sentences(text):
            if "기말" in sent:
                evidence = sent[:320]
                break
    if not evidence:
        evidence = (text or "")[:320]

    return evidence, row_amounts[target_col]


def _pick_amount_evidence(
    text: str,
    keywords: tuple[str, ...],
    column_hints: tuple[str, ...] = (),
    query_hints: dict[str, Any] | None = None,
    row_meta: dict[str, Any] | None = None,
) -> tuple[str, str]:
    hints = query_hints or {"row_keys": [], "col_keys": [], "topic": "", "ask_type": "amount", "query_compact": ""}

    lease_sent, lease_amount = _extract_loan_lease_amount_by_pattern(text, hints)
    if lease_amount:
        return lease_sent, lease_amount

    cash_sent, cash_amount = _extract_cash_amount_by_pattern(text, hints)
    if cash_amount:
        return cash_sent, cash_amount

    guarantee_sent, guarantee_amount = _extract_guarantee_amount_by_pattern(text, hints)
    if guarantee_amount:
        return guarantee_sent, guarantee_amount

    provision_sent, provision_amount = _extract_provision_amount_by_pattern(text, hints, column_hints)
    if provision_amount:
        return provision_sent, provision_amount

    preferred_col_hints = tuple(dict.fromkeys([*hints.get("col_keys", []), *column_hints]))
    if preferred_col_hints:
        table_sent, table_amount = _extract_table_amount_by_column(
            text,
            preferred_col_hints,
            row_keys=tuple(hints.get("row_keys", [])),
            aggregate_intent=bool(hints.get("aggregate_intent")),
        )
        if table_amount:
            return table_sent, table_amount

    parsed_cells = _parse_pipe_table_cell_candidates(text, row_meta=row_meta)
    if parsed_cells:
        scored = sorted(
            ((
                _score_table_cell_candidate(cell, hints, column_hints, row_meta),
                cell,
            ) for cell in parsed_cells),
            key=lambda x: x[0],
            reverse=True,
        )
        best_score, best_cell = scored[0]
        if best_score > 0:
            row_label = str(best_cell.get("row_label") or "")
            row_group = [
                cell
                for _, cell in scored
                if compact_text(str(cell.get("row_label") or "")) == compact_text(row_label)
            ]
            compact_summary = _build_compact_row_summary(row_group, row_label)
            if compact_summary:
                return compact_summary[:320], str(best_cell.get("value") or "")
            col_label = _clean_table_label(str(best_cell.get("col_label") or ""))
            value = str(best_cell.get("value") or "")
            evidence = f"{_clean_table_label(row_label)} | {col_label} {value}".strip()
            return evidence[:320], value

    best_sent = ""
    best_amount = ""
    best_score = -1
    for sent in _split_sentences(text):
        amount = _extract_amount(sent)
        if not amount:
            continue
        score = 1 + _keyword_overlap_score(sent, keywords)
        if score > best_score:
            best_score = score
            best_sent = sent
            best_amount = amount
    return best_sent[:320], best_amount


def extract_note_keywords(query: str) -> list[str]:
    cleaned = YEAR_TOKEN_RE.sub(" ", query or "")
    candidates = re.findall(r"[가-힣a-zA-Z0-9_]+", cleaned)
    out: list[str] = []
    for token in candidates:
        tok = TRAILING_PARTICLE_RE.sub("", token)
        if len(tok) < 2:
            continue
        if tok in NOTE_QUERY_STOPWORDS:
            continue
        out.append(tok)
    return list(dict.fromkeys(out))


def expand_note_keywords(keywords: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in keywords:
        tok = token.strip()
        if not tok:
            continue
        expanded.append(tok)

        if tok.endswith("충당부채") and tok != "충당부채":
            prefix = tok[: -len("충당부채")].strip()
            if len(prefix) >= 2:
                expanded.append(prefix)
            expanded.append("충당부채")

        if tok.endswith("보증") and tok != "보증":
            expanded.append("보증")

        if tok.endswith("약정") and tok != "약정":
            expanded.append("약정")

    return list(dict.fromkeys(expanded))


def _dedup_rows_by_chunk_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("chunk_key") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
    return deduped


def is_fine_grained_row(row: dict[str, Any]) -> bool:
    section_type = str(row.get("section_type") or "")
    if section_type in {"note_table_cell", "financial_statement_row"}:
        return True
    return row.get("table_meta") is not None


def build_parent_lookup_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_year": row.get("report_year"),
        "major_section": row.get("major_section"),
        "note_no": row.get("note_no"),
        "note_title": row.get("note_title"),
        "sub_section": row.get("sub_section"),
    }


def _compute_rollup_limits(query: str, table_query_intent: bool) -> tuple[int, int]:
    compact_query = compact_text(query)
    is_numeric = _is_amount_question(query) or _is_rate_question(query)
    has_compare = any(token in compact_query for token in ("비교", "차이", "각각", "연도별", "추이", "증감"))
    has_explain = any(token in compact_query for token in ("설명", "요약", "의미", "내용", "어떻게", "근거"))

    if table_query_intent or has_compare:
        return 2, 2
    # 단일 수치: fine 1 + parent 1
    if is_numeric and not has_compare and not has_explain:
        return 1, 1
    # 설명형: roll-up 약하게(생략)
    if (not is_numeric) and has_explain:
        return 0, 0
    # 혼합형: fine 2 + parent 2
    return 2, 2


def rollup_parent_context(
    *,
    report_year: int | None,
    major_section: str | None,
    note_no: int | None,
    note_title: str | None,
    sub_section: str | None,
    limit: int = 3,
    query: str = "",
    exclude_chunk_keys: tuple[str, ...] = tuple(),
) -> list[dict[str, Any]]:
    subsection_compact = compact_text(sub_section or "")
    query_compact = compact_text(query)
    clauses = [
        "c.section_type IN ('note', 'note_subsection', 'financial_statement', 'audit_subsection', 'major_section')",
    ]
    where_params: list[Any] = []

    if report_year is not None:
        clauses.append("c.report_year = %s")
        where_params.append(report_year)
    if major_section:
        clauses.append("c.major_section = %s")
        where_params.append(major_section)

    if note_no is not None:
        clauses.append("c.note_no = %s")
        where_params.append(note_no)
    elif note_title:
        clauses.append("COALESCE(c.note_title, '') = %s")
        where_params.append(note_title)

    excluded = [k for k in exclude_chunk_keys if k]
    if excluded:
        placeholders = ", ".join(["%s"] * len(excluded))
        clauses.append(f"c.chunk_key NOT IN ({placeholders})")
        where_params.extend(excluded)

    sql = f"""
        SELECT
            d.file_name,
            c.report_year,
            c.major_section,
            c.sub_section,
            c.section_type,
            c.note_no,
            c.note_title,
            c.subtopic,
            c.section_id,
            c.subsection_path,
            c.as_of_date,
            c.period_start,
            c.period_end,
            c.evidence_type,
            c.risk_domain,
            c.table_meta,
            c.cell_unit,
            c.chunk_key,
            c.content,
            0.0 AS semantic_score,
            ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s)) AS keyword_score,
            CASE
                WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 1.0
                ELSE 0.0
            END AS exact_match_score,
            0.0 AS meta_boost_score,
            (
                CASE
                    WHEN %s <> '' AND regexp_replace(COALESCE(c.sub_section, ''), '\\s+', '', 'g') = %s THEN 1.0
                    ELSE 0.0
                END
                + CASE
                    WHEN c.section_type = 'note' THEN 0.8
                    WHEN c.section_type = 'note_subsection' THEN 0.7
                    WHEN c.section_type = 'financial_statement' THEN 0.5
                    WHEN c.section_type = 'audit_subsection' THEN 0.4
                    WHEN c.section_type = 'major_section' THEN 0.3
                    ELSE 0.0
                END
                + 0.35 * ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s))
            ) AS hybrid_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {' AND '.join(clauses)}
        ORDER BY hybrid_score DESC
        LIMIT %s
    """

    final_params = [
        query,
        query_compact,
        subsection_compact,
        subsection_compact,
        query,
        *where_params,
        limit,
    ]
    return _execute_search_sql(sql, final_params)


def _append_rollup_parent_rows(
    rows: list[dict[str, Any]],
    *,
    query: str,
    table_query_intent: bool,
) -> list[dict[str, Any]]:
    fine_limit, parent_limit = _compute_rollup_limits(query, table_query_intent)
    if fine_limit <= 0 or parent_limit <= 0:
        return rows[: settings.top_k]

    final_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    fine_used = 0

    for row in rows[: settings.top_k]:
        key = str(row.get("chunk_key") or "")
        if key and key not in seen_keys:
            final_rows.append(row)
            seen_keys.add(key)

        if fine_used >= fine_limit or not is_fine_grained_row(row):
            continue

        lookup = build_parent_lookup_keys(row)
        parent_rows = rollup_parent_context(
            report_year=lookup.get("report_year"),
            major_section=lookup.get("major_section"),
            note_no=lookup.get("note_no"),
            note_title=lookup.get("note_title"),
            sub_section=lookup.get("sub_section"),
            limit=parent_limit,
            query=query,
            exclude_chunk_keys=tuple(seen_keys),
        )
        for parent in parent_rows:
            parent_key = str(parent.get("chunk_key") or "")
            if not parent_key or parent_key in seen_keys:
                continue
            parent["rolled_up_from"] = row.get("chunk_key")
            parent["is_parent_context"] = True
            final_rows.append(parent)
            seen_keys.add(parent_key)

        fine_used += 1

    return final_rows


def _infer_table_query_intent(query: str) -> bool:
    compact_query = compact_text(query)
    table_markers = ("표", "행", "열", "당기말", "전기말", "잔액", "한도", "지급액", "총액", "비율")
    if any(marker in compact_query for marker in table_markers):
        return True
    return _is_amount_question(query) or _is_rate_question(query)


def classify_retrieval_mode(query: str, table_query_intent: bool) -> str:
    compact_query = compact_text(query)
    has_explain = any(token in compact_query for token in ("설명", "요약", "의미", "책임", "무엇"))
    has_compare = any(token in compact_query for token in ("비교", "차이", "각각", "연도별", "추이", "증감"))
    wants_amount = _is_amount_question(query) or _is_rate_question(query)
    wants_risk = has_any_hint(compact_query, RISK_HINTS)

    if has_explain and not wants_amount and not table_query_intent:
        return "coarse_first"
    if wants_amount and (table_query_intent or has_compare):
        return "mixed"
    if wants_amount:
        return "fine_first"
    if wants_risk and table_query_intent:
        return "mixed"
    return "coarse_first"


def _compute_confidence(query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"enough": False, "score": 0.0, "reason": "no_rows"}

    top = rows[0]
    top_hybrid = float(top.get("hybrid_score", 0.0) or 0.0)
    top_keyword = float(top.get("keyword_score", 0.0) or 0.0)
    top_exact = float(top.get("exact_match_score", 0.0) or 0.0)
    fine_exists = any(is_fine_grained_row(r) for r in rows)

    query_tokens = [tok for tok in re.split(r"[^0-9A-Za-z가-힣]+", query) if len(tok) >= 2]
    query_compact = compact_text(" ".join(query_tokens))
    top_blob = compact_text(
        str(top.get("content") or "")
        + " "
        + str(top.get("note_title") or "")
        + " "
        + str(top.get("sub_section") or "")
        + " "
        + str(top.get("topic") or "")
    )
    token_hits = 0
    for tok in query_tokens[:8]:
        ctok = compact_text(tok)
        if ctok and ctok in top_blob:
            token_hits += 1

    score = (
        top_hybrid
        + 0.25 * top_keyword
        + 0.30 * top_exact
        + (0.20 if fine_exists else 0.0)
        + (0.08 * token_hits)
        + (0.25 if query_compact and query_compact in top_blob else 0.0)
    )
    enough = score >= 1.15
    reason = "strong_top_match" if enough else "low_top_match_or_context"
    return {"enough": enough, "score": round(float(score), 4), "reason": reason}


def _query_row_focus_bias(row: dict[str, Any], query: str) -> float:
    q = compact_text(query)
    if not q:
        return 0.0

    sub = compact_text(str(row.get("sub_section") or ""))
    note = compact_text(str(row.get("note_title") or ""))
    content = compact_text(str(row.get("content") or ""))
    blob = f"{sub} {note} {content}"

    score = 0.0
    if "재무상태표" in q:
        score += 0.9 if "재무상태표" in sub else -0.45 if any(x in sub for x in ("현금흐름표", "자본변동표")) else 0.0
    if "손익계산서" in q:
        score += 0.9 if "손익계산서" in sub else -0.45 if any(x in sub for x in ("현금흐름표", "자본변동표")) else 0.0
    if "포괄손익계산서" in q:
        score += 0.9 if "포괄손익계산서" in sub else 0.0

    account_hints = ("현금및현금성자산", "매출채권", "재고자산", "영업이익", "매출액")
    for hint in account_hints:
        if hint in q:
            score += 0.55 if hint in blob else 0.0

    return score


def merge_multilevel_rows(rows: list[dict[str, Any]], mode: str, query: str = "") -> list[dict[str, Any]]:
    merged = _dedup_rows_by_chunk_key(rows)

    def _priority(row: dict[str, Any]) -> tuple[int, float, float]:
        section_type = str(row.get("section_type") or "")
        is_parent = bool(row.get("is_parent_context"))
        is_fine = is_fine_grained_row(row)
        if mode == "coarse_first":
            rank = 3 if section_type in {"note", "note_subsection", "audit_subsection", "major_section"} else 1
        elif mode == "fine_first":
            rank = 3 if is_fine else 2 if is_parent else 1
        else:
            rank = 3 if is_fine else 2 if is_parent else 2 if section_type in {"note", "note_subsection"} else 1
        bias = _query_row_focus_bias(row, query)
        return rank, bias, float(row.get("hybrid_score", 0.0) or 0.0)

    merged.sort(key=lambda row: _priority(row), reverse=True)
    return merged


def retrieve_coarse_first(query: str, report_year: int | None = None, sub_section: str | None = None) -> SearchResult:
    coarse = _retrieve_broad(query, report_year=report_year, sub_section=sub_section)
    confidence = _compute_confidence(query, coarse.rows)
    if confidence["enough"]:
        coarse.rows = merge_multilevel_rows(coarse.rows, "coarse_first", query=query)[: settings.top_k]
        return coarse

    fine = retrieve_note_first(query, report_year=coarse.report_year or report_year)
    table_query_intent = _infer_table_query_intent(query)
    fine_rows = _append_rollup_parent_rows(fine.rows, query=query, table_query_intent=table_query_intent)
    coarse.rows = merge_multilevel_rows([*coarse.rows, *fine_rows], "coarse_first", query=query)[: settings.top_k]
    return coarse


def retrieve_fine_first(query: str, report_year: int | None = None, sub_section: str | None = None) -> SearchResult:
    fine = retrieve_note_first(query, report_year=report_year)
    table_query_intent = _infer_table_query_intent(query)
    fine.rows = _append_rollup_parent_rows(fine.rows, query=query, table_query_intent=table_query_intent)

    confidence = _compute_confidence(query, fine.rows)
    has_parent = any(bool(r.get("is_parent_context")) for r in fine.rows)
    if confidence["enough"] and has_parent:
        fine.rows = merge_multilevel_rows(fine.rows, "fine_first", query=query)[: settings.top_k]
        return fine

    coarse = _retrieve_broad(query, report_year=fine.report_year or report_year, sub_section=sub_section)
    fine.rows = merge_multilevel_rows([*fine.rows, *coarse.rows], "fine_first", query=query)[: settings.top_k]
    return fine


def retrieve_mixed(query: str, report_year: int | None = None, sub_section: str | None = None) -> SearchResult:
    fine = retrieve_note_first(query, report_year=report_year)
    table_query_intent = _infer_table_query_intent(query)
    fine_rows = _append_rollup_parent_rows(fine.rows, query=query, table_query_intent=table_query_intent)

    coarse = _retrieve_broad(query, report_year=fine.report_year or report_year, sub_section=sub_section)
    merged_rows = merge_multilevel_rows([*fine_rows, *coarse.rows], "mixed", query=query)[: settings.top_k]

    coarse.rows = merged_rows
    coarse.report_year = coarse.report_year or fine.report_year
    coarse.auto_year_applied = coarse.auto_year_applied or fine.auto_year_applied
    return coarse


def _apply_risk_note_bias(rows: list[dict[str, Any]], risk_type: str | None, query: str) -> list[dict[str, Any]]:
    if not rows or not risk_type:
        return rows

    anchors = RISK_NOTE_ANCHORS.get(risk_type, tuple())
    if not anchors:
        return rows

    q = compact_text(query)

    def _score(row: dict[str, Any]) -> float:
        score = float(row.get("hybrid_score", 0.0))
        note_blob = compact_text(
            str(row.get("note_title") or "")
            + " "
            + str(row.get("subtopic") or "")
            + " "
            + str(row.get("subsection_path") or "")
        )

        for a in anchors:
            ak = compact_text(a)
            if not ak:
                continue
            if ak in note_blob:
                score += 2.6

        if any(tok in q for tok in ("주석", "관련내용", "관련설명")):
            if "우발부채와약정사항" in note_blob:
                score += 3.8

        # 질의 핵심 토큰 재정렬(짧은 질의의 노이즈 완화)
        for tk in expand_note_keywords(extract_note_keywords(query)):
            ck = compact_text(tk)
            if ck and ck in note_blob:
                score += 0.6

        # 담보/보증 설명 질의에서 재무위험관리 과선택 완화
        if risk_type == "담보/보증" and "설명" in q and "재무위험관리" in note_blob and "우발부채와약정사항" not in note_blob:
            score -= 1.6

        # 리스크 질의에서 비관련 노트 과선택 완화
        if risk_type in {"소송", "담보/보증", "우발부채/약정"}:
            if any(bad in note_blob for bad in ("현금흐름표", "계약부채")):
                score -= 4.0

        return score

    return sorted(rows, key=_score, reverse=True)


def _debug_print_candidates(tag: str, rows: list[dict[str, Any]], score_key: str) -> None:
    if os.getenv("SEARCH_DEBUG_CANDIDATES", "").lower() not in {"1", "true", "yes", "y"}:
        return
    print(f"\n[DEBUG] {tag} top candidates")
    for idx, row in enumerate(rows[:10], start=1):
        print(
            f"[{idx}] note_no={row.get('note_no')} | note_title={row.get('note_title')} | "
            f"subtopic={row.get('subtopic')} | {score_key}={row.get(score_key, 0.0):.4f} | "
            f"hybrid={row.get('hybrid_score', 0.0):.4f}"
        )


class Reranker:
    def __init__(self) -> None:
        self.enabled = settings.use_reranker
        self.top_n = settings.rerank_top_n
        self.model_name = settings.reranker_model
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        if not self.enabled or not rows:
            return rows, False

        rerank_count = min(len(rows), max(1, self.top_n))
        rerank_rows = rows[:rerank_count]
        pairs = [(query, row["content"]) for row in rerank_rows]
        scores = self._get_model().predict(pairs)

        for row, score in zip(rerank_rows, scores, strict=True):
            row["rerank_score"] = float(score)

        rerank_rows.sort(key=lambda row: (row["rerank_score"], row["hybrid_score"]), reverse=True)
        return rerank_rows + rows[rerank_count:], True


def _execute_search_sql(sql: str, params: list[Any]) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return list(rows)


def _select_primary_note_keyword(note_keywords: list[str], fallback_query: str) -> str:
    if not note_keywords:
        return compact_text(fallback_query)

    stop_tokens = (
        "무엇",
        "무엇인",
        "설명",
        "내용",
        "얼마",
        "관련",
        "있는",
        "주석",
        "인가",
        "인가요",
    )
    preferred = next((k for k in note_keywords if "사용제한금융상품" in compact_text(k)), None)
    if preferred:
        return compact_text(preferred)

    informative = [k for k in note_keywords if k and not any(t in compact_text(k) for t in stop_tokens)]
    if informative:
        informative.sort(key=lambda x: (len(compact_text(x)), x), reverse=True)
        return compact_text(informative[0])

    return compact_text(note_keywords[0])


def _fetch_note_title_candidates(
    query: str,
    vector: list[float] | Any,
    report_year: int | None,
    keywords: tuple[str, ...],
    limit: int,
    risk_domains: tuple[str, ...] = tuple(),
) -> list[dict[str, Any]]:
    clauses = ["c.section_type = 'note'"]
    note_keywords = expand_note_keywords(extract_note_keywords(query))
    note_query_text = " ".join(note_keywords) if note_keywords else query
    primary_keyword = _select_primary_note_keyword(note_keywords, query)
    compact_query = primary_keyword
    # placeholder order in SQL:
    # 1 vector, 2 query, 3 compact_query, 4 vector, 5 query, 6 compact_query, 7 [report_year], 8 limit
    params: list[Any] = [vector, note_query_text, compact_query, vector, note_query_text, compact_query]
    if report_year is not None:
        clauses.append("c.report_year = %s")
    if risk_domains:
        clauses.append("c.risk_domain = ANY(%s)")

    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT
            d.file_name,
            c.report_year,
            c.major_section,
            c.sub_section,
            c.section_type,
            c.note_no,
            c.note_title,
            c.subtopic,
            c.section_id,
            c.subsection_path,
            c.as_of_date,
            c.period_start,
            c.period_end,
            c.evidence_type,
            c.risk_domain,
            c.table_meta,
            c.cell_unit,
            c.chunk_key,
            c.content,
            1 - (c.embedding <=> %s::vector) AS semantic_score,
            ts_rank_cd(
                to_tsvector('simple', COALESCE(c.note_title, '') || ' ' || COALESCE(c.subtopic, '') || ' ' || COALESCE(c.subsection_path, '')),
                websearch_to_tsquery('simple', %s)
            ) AS keyword_score,
            CASE
                WHEN POSITION(
                    %s IN regexp_replace(COALESCE(c.note_title, '') || COALESCE(c.subtopic, '') || COALESCE(c.subsection_path, ''), '\\s+', '', 'g')
                ) > 0
                THEN 1.0
                ELSE 0.0
            END AS exact_match_score,
            CASE
                WHEN COALESCE(c.evidence_type, '') IN ('quant_text', 'quant_table') THEN 0.05
                ELSE 0.0
            END AS meta_boost_score,
            (0.15 * (1 - (c.embedding <=> %s::vector))
                + 0.55 * ts_rank_cd(
                    to_tsvector('simple', COALESCE(c.note_title, '') || ' ' || COALESCE(c.subtopic, '') || ' ' || COALESCE(c.subsection_path, '')),
                    websearch_to_tsquery('simple', %s)
                )
                + 0.30 * CASE
                    WHEN POSITION(
                        %s IN regexp_replace(COALESCE(c.note_title, '') || COALESCE(c.subtopic, '') || COALESCE(c.subsection_path, ''), '\\s+', '', 'g')
                    ) > 0
                    THEN 1.0
                    ELSE 0.0
                  END
                + CASE WHEN COALESCE(c.evidence_type, '') IN ('quant_text', 'quant_table') THEN 0.05 ELSE 0.0 END
            ) AS hybrid_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {where_sql}
        ORDER BY hybrid_score DESC
        LIMIT %s
    """
    if report_year is not None:
        params.append(report_year)
    if risk_domains:
        params.append(list(risk_domains))
    params.append(limit)
    rows = _execute_search_sql(sql, params)

    if rows:
        title_keywords = tuple(note_keywords) if note_keywords else keywords
        for row in rows:
            title_blob = (row.get("note_title") or "") + " " + (row.get("subtopic") or "") + " " + (row.get("subsection_path") or "")
            subtopic_blob = (row.get("subtopic") or "")
            title_overlap = _keyword_overlap_score(title_blob, title_keywords)
            title_exact = 1.0 if (compact_query and compact_query in compact_text(title_blob)) else 0.0
            subtopic_overlap = _keyword_overlap_score(subtopic_blob, title_keywords)
            subtopic_exact = 1.0 if (compact_query and compact_query in compact_text(subtopic_blob)) else 0.0
            penalty = -0.45 if title_overlap <= 0 and title_exact <= 0 else 0.0
            row["title_match_score"] = float(title_overlap + title_exact + (1.8 * subtopic_exact) + (0.9 * subtopic_overlap))
            row["hybrid_score"] = float(
                row.get("hybrid_score", 0.0)
                + (1.2 * title_exact)
                + (0.35 * title_overlap)
                + (1.8 * subtopic_exact)
                + (0.9 * subtopic_overlap)
                + penalty
            )

        rows.sort(
            key=lambda row: (row.get("title_match_score", 0.0), row.get("hybrid_score", 0.0)),
            reverse=True,
        )
        rows = [row for row in rows if float(row.get("title_match_score", 0.0)) > 0.0]
        _debug_print_candidates("_fetch_note_title_candidates", rows, "title_match_score")
    return rows


def _fetch_note_body_candidates(
    query: str,
    vector: list[float] | Any,
    report_year: int | None,
    limit: int,
    risk_domains: tuple[str, ...] = tuple(),
) -> list[dict[str, Any]]:
    clauses = ["c.section_type = 'note'"]
    note_keywords = expand_note_keywords(extract_note_keywords(query))
    note_query_text = " ".join(note_keywords) if note_keywords else query
    compact_query = _select_primary_note_keyword(note_keywords, query)
    # placeholder order in SQL:
    # 1 vector, 2 query, 3 compact_query, 4 vector, 5 query, 6 compact_query, 7 [report_year], 8 limit
    params: list[Any] = [vector, note_query_text, compact_query, vector, note_query_text, compact_query]
    if report_year is not None:
        clauses.append("c.report_year = %s")
    if risk_domains:
        clauses.append("c.risk_domain = ANY(%s)")

    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT
            d.file_name,
            c.report_year,
            c.major_section,
            c.sub_section,
            c.section_type,
            c.note_no,
            c.note_title,
            c.subtopic,
            c.section_id,
            c.subsection_path,
            c.as_of_date,
            c.period_start,
            c.period_end,
            c.evidence_type,
            c.risk_domain,
            c.table_meta,
            c.cell_unit,
            c.chunk_key,
            c.content,
            1 - (c.embedding <=> %s::vector) AS semantic_score,
            ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s)) AS keyword_score,
            CASE
                WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0
                THEN 1.0
                ELSE 0.0
            END AS exact_match_score,
            CASE
                WHEN COALESCE(c.evidence_type, '') IN ('quant_text', 'quant_table') THEN 0.05
                ELSE 0.0
            END AS meta_boost_score,
            (
                0.45 * (1 - (c.embedding <=> %s::vector))
                + 0.35 * ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s))
                + 0.20 * CASE
                    WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0
                    THEN 1.0
                    ELSE 0.0
                  END
                + CASE WHEN COALESCE(c.evidence_type, '') IN ('quant_text', 'quant_table') THEN 0.05 ELSE 0.0 END
            ) AS hybrid_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {where_sql}
        ORDER BY hybrid_score DESC
        LIMIT %s
    """
    if report_year is not None:
        params.append(report_year)
    if risk_domains:
        params.append(list(risk_domains))
    params.append(limit)
    rows = _execute_search_sql(sql, params)

    if rows:
        body_keywords = tuple(note_keywords) if note_keywords else tuple()
        for row in rows:
            body_text = row.get("content") or ""
            title_text = (row.get("note_title") or "") + " " + (row.get("subtopic") or "") + " " + (row.get("subsection_path") or "")
            subtopic_text = (row.get("subtopic") or "")
            body_overlap = _keyword_overlap_score(body_text, body_keywords) if body_keywords else 0
            title_overlap = _keyword_overlap_score(title_text, body_keywords) if body_keywords else 0
            subtopic_overlap = _keyword_overlap_score(subtopic_text, body_keywords) if body_keywords else 0
            subtopic_exact = 1.0 if (compact_query and compact_query in compact_text(subtopic_text)) else 0.0
            row["body_match_score"] = float(body_overlap + (0.6 * title_overlap) + (1.1 * subtopic_overlap) + (1.8 * subtopic_exact))
            row["hybrid_score"] = float(
                row.get("hybrid_score", 0.0)
                + (0.25 * body_overlap)
                + (0.35 * title_overlap)
                + (1.1 * subtopic_overlap)
                + (1.8 * subtopic_exact)
            )

        rows.sort(
            key=lambda row: (row.get("body_match_score", 0.0), row.get("hybrid_score", 0.0)),
            reverse=True,
        )
        _debug_print_candidates("_fetch_note_body_candidates", rows, "body_match_score")

    return rows


def _fetch_same_note_table_candidates(
    note_row: dict[str, Any],
    query: str,
    report_year: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    note_no = note_row.get("note_no")
    if note_no is None:
        return []

    params: list[Any] = [query, compact_text(query), query, compact_text(query), note_no]
    clauses = ["c.note_no = %s", "c.chunk_key <> %s"]
    params.append(note_row.get("chunk_key"))

    if report_year is not None:
        clauses.append("c.report_year = %s")
        params.append(report_year)

    sql = f"""
        SELECT
            d.file_name,
            c.report_year,
            c.major_section,
            c.sub_section,
            c.section_type,
            c.note_no,
            c.note_title,
            c.subtopic,
            c.section_id,
            c.subsection_path,
            c.as_of_date,
            c.period_start,
            c.period_end,
            c.evidence_type,
            c.table_meta,
            c.cell_unit,
            c.chunk_key,
            c.content,
            0.0 AS semantic_score,
            ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s)) AS keyword_score,
            CASE
                WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 1.0
                ELSE 0.0
            END AS exact_match_score,
            CASE
                WHEN c.table_meta IS NOT NULL OR COALESCE(c.evidence_type, '') = 'quant_table' THEN 0.12
                ELSE 0.0
            END AS meta_boost_score,
            (
                CASE
                    WHEN c.table_meta IS NOT NULL OR COALESCE(c.evidence_type, '') = 'quant_table' THEN 0.4
                    ELSE 0.0
                END
                +
                ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s))
                + CASE WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 0.3 ELSE 0.0 END
                + CASE WHEN c.table_meta IS NOT NULL OR COALESCE(c.evidence_type, '') = 'quant_table' THEN 0.12 ELSE 0.0 END
            ) AS hybrid_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {' AND '.join(clauses)}
        ORDER BY hybrid_score DESC
        LIMIT %s
    """
    params.append(limit)
    return _execute_search_sql(sql, params)


def _fetch_same_section_candidates(
    section_id: str | None,
    query: str,
    report_year: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    if not section_id:
        return []
    params: list[Any] = [query, compact_text(query), query, compact_text(query), section_id]
    clauses = ["c.section_id = %s"]
    if report_year is not None:
        clauses.append("c.report_year = %s")
        params.append(report_year)

    sql = f"""
        SELECT
            d.file_name,
            c.report_year,
            c.major_section,
            c.sub_section,
            c.section_type,
            c.note_no,
            c.note_title,
            c.subtopic,
            c.section_id,
            c.subsection_path,
            c.as_of_date,
            c.period_start,
            c.period_end,
            c.evidence_type,
            c.table_meta,
            c.cell_unit,
            c.chunk_key,
            c.content,
            0.0 AS semantic_score,
            ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s)) AS keyword_score,
            CASE
                WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 1.0
                ELSE 0.0
            END AS exact_match_score,
            0.03 AS meta_boost_score,
            (
                ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s))
                + CASE WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 0.3 ELSE 0.0 END
                + 0.03
            ) AS hybrid_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {' AND '.join(clauses)}
        ORDER BY hybrid_score DESC
        LIMIT %s
    """
    params.append(limit)
    return _execute_search_sql(sql, params)


def _retrieve_broad(query: str, report_year: int | None = None, sub_section: str | None = None) -> SearchResult:
    auto_year_applied = False
    auto_section_type: str | None = None
    auto_sub_section_compact: str | None = None
    semantic_query = query
    if report_year is None:
        inferred_year, semantic_query = infer_report_year(query)
        if inferred_year is not None:
            report_year = inferred_year
            auto_year_applied = True

    if not semantic_query:
        semantic_query = query

    compact_query = normalize_search_query(semantic_query)
    signals = infer_query_signals(semantic_query)
    if sub_section is None:
        auto_section_type = infer_section_type_hint(compact_query)
        auto_sub_section_compact = infer_sub_section_hint(compact_query)

    embedder = Embedder()
    vector = embedder.encode_texts([semantic_query])[0]

    clauses = []
    params = []
    if report_year is not None:
        clauses.append("c.report_year = %s")
        params.append(report_year)
    if sub_section is not None:
        clauses.append("c.sub_section = %s")
        params.append(sub_section)
    if auto_section_type is not None:
        clauses.append("c.section_type = %s")
        params.append(auto_section_type)
    if auto_sub_section_compact is not None:
        clauses.append("regexp_replace(COALESCE(c.sub_section, ''), '\\s+', '', 'g') = %s")
        params.append(auto_sub_section_compact)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
        WITH scored AS (
            SELECT
                d.file_name,
                c.report_year,
                c.major_section,
                c.sub_section,
                c.section_type,
                c.note_no,
                c.note_title,
                c.section_id,
                c.subsection_path,
                c.as_of_date,
                c.period_start,
                c.period_end,
                c.evidence_type,
                c.table_meta,
                c.cell_unit,
                c.chunk_key,
                c.content,
                1 - (c.embedding <=> %s::vector) AS semantic_score,
                ts_rank_cd(
                    to_tsvector('simple', COALESCE(c.content, '')),
                    websearch_to_tsquery('simple', %s)
                ) AS keyword_score,
                CASE
                    WHEN POSITION(
                        %s IN regexp_replace(
                            COALESCE(c.sub_section, '') || COALESCE(c.note_title, '') || COALESCE(c.content, ''),
                            '\\s+',
                            '',
                            'g'
                        )
                    ) > 0
                    THEN 1.0
                    ELSE 0.0
                END AS exact_match_score,
                (
                    CASE
                        WHEN %s::boolean AND c.as_of_date IS NOT NULL THEN 0.08
                        ELSE 0.0
                    END
                    + CASE
                        WHEN %s::boolean AND (c.period_start IS NOT NULL OR c.period_end IS NOT NULL) THEN 0.08
                        ELSE 0.0
                    END
                    + CASE
                        WHEN %s::boolean AND COALESCE(c.evidence_type, '') IN ('quant_text', 'quant_table') THEN 0.12
                        ELSE 0.0
                    END
                    + CASE
                        WHEN %s::boolean
                             AND (
                                 regexp_replace(COALESCE(c.evidence_type, ''), '\\s+', '', 'g') ~ %s
                                 OR regexp_replace(COALESCE(c.subsection_path, ''), '\\s+', '', 'g') ~ %s
                                 OR regexp_replace(COALESCE(c.note_title, ''), '\\s+', '', 'g') ~ %s
                             )
                        THEN 0.10
                        ELSE 0.0
                    END
                    + CASE
                        WHEN %s::boolean AND c.table_meta IS NOT NULL THEN 0.04
                        ELSE 0.0
                    END
                    + CASE
                        WHEN %s <> ''
                             AND regexp_replace(COALESCE(c.cell_unit, ''), '\\s+', '', 'g') = %s
                        THEN 0.06
                        ELSE 0.0
                    END
                    + CASE
                        WHEN %s <> ''
                             AND POSITION(
                                 %s IN regexp_replace(
                                     COALESCE(c.section_id, '') || COALESCE(c.subsection_path, '') || COALESCE(c.note_title, ''),
                                     '\\s+',
                                     '',
                                     'g'
                                 )
                             ) > 0
                        THEN 0.08
                        ELSE 0.0
                    END
                ) AS meta_boost_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {where_sql}
        ),
        ranked AS (
            SELECT
                *,
                (%s * semantic_score + %s * keyword_score + %s * exact_match_score + meta_boost_score) AS hybrid_score
            FROM scored
            ORDER BY hybrid_score DESC
            LIMIT %s
        )
        SELECT * FROM ranked
        ORDER BY hybrid_score DESC
    """
    final_params = [
        vector,
        semantic_query,
        compact_query,
        signals["wants_as_of"],
        signals["wants_period"],
        signals["wants_quant"],
        signals["wants_risk"],
        RISK_REGEX,
        RISK_REGEX,
        RISK_REGEX,
        signals["wants_quant"],
        compact_text(signals["unit_hint"]),
        compact_text(signals["unit_hint"]),
        signals["structure_hint"],
        signals["structure_hint"],
        *params,
        settings.semantic_weight,
        settings.keyword_weight,
        EXACT_MATCH_BOOST,
        settings.hybrid_candidate_k,
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, final_params)
            rows = cur.fetchall()

    rows = list(rows)
    rerank_applied = False
    if rows:
        reranker = Reranker()
        try:
            rows, rerank_applied = reranker.rerank(semantic_query, rows)
        except Exception as exc:
            print(f"[WARN] reranker disabled due to runtime error: {exc}")

    rows = rows[: settings.top_k]

    return SearchResult(
        original_query=query,
        semantic_query=semantic_query,
        report_year=report_year,
        auto_year_applied=auto_year_applied,
        auto_section_type=auto_section_type,
        rerank_applied=rerank_applied,
        rows=rows,
    )


def retrieve(query: str, report_year: int | None = None, sub_section: str | None = None) -> SearchResult:
    table_query_intent = _infer_table_query_intent(query)
    mode = classify_retrieval_mode(query, table_query_intent)

    if mode == "coarse_first":
        result = retrieve_coarse_first(query, report_year=report_year, sub_section=sub_section)
    elif mode == "fine_first":
        result = retrieve_fine_first(query, report_year=report_year, sub_section=sub_section)
    else:
        result = retrieve_mixed(query, report_year=report_year, sub_section=sub_section)

    if sub_section is not None and result.rows:
        filtered = [row for row in result.rows if row.get("sub_section") == sub_section]
        result.rows = filtered

    result.rows = _dedup_rows_by_chunk_key(result.rows)[: settings.top_k]
    return result


def retrieve_note_first(
    query: str,
    report_year: int | None = None,
    risk_type: str | None = None,
) -> SearchResult:
    auto_year_applied = False
    semantic_query = query
    if report_year is None:
        inferred_year, semantic_query = infer_report_year(query)
        if inferred_year is not None:
            report_year = inferred_year
            auto_year_applied = True

    if not semantic_query:
        semantic_query = query

    inferred_risks = detect_risk_types(semantic_query)
    effective_risk_type = risk_type or (inferred_risks[0] if inferred_risks else None)
    risk_domains = _risk_type_to_domains(effective_risk_type, semantic_query)

    embedder = Embedder()
    vector = embedder.encode_texts([semantic_query])[0]
    keywords = RISK_KEYWORDS.get(effective_risk_type or "", tuple())

    rows: list[dict[str, Any]] = []
    confidence_rule = ""

    # 1) same-year note_title
    title_rows = _fetch_note_title_candidates(
        semantic_query,
        vector,
        report_year,
        keywords,
        risk_domains=risk_domains,
        limit=8,
    )
    if not title_rows and risk_domains:
        title_rows = _fetch_note_title_candidates(semantic_query, vector, report_year, keywords, risk_domains=tuple(), limit=8)
    title_rows = _apply_risk_note_bias(title_rows, effective_risk_type, semantic_query)
    if title_rows:
        selected_note = title_rows[0]
        selected_note["retrieval_stage"] = "note_title"
        confidence_rule = "note_title_match"
    else:
        # 2) same-year note body fallback
        body_rows = _fetch_note_body_candidates(semantic_query, vector, report_year, risk_domains=risk_domains, limit=8)
        if not body_rows and risk_domains:
            body_rows = _fetch_note_body_candidates(semantic_query, vector, report_year, risk_domains=tuple(), limit=8)
        body_rows = _apply_risk_note_bias(body_rows, effective_risk_type, semantic_query)
        if body_rows:
            selected_note = body_rows[0]
            selected_note["retrieval_stage"] = "note_body"
            confidence_rule = "note_body_match"
        else:
            selected_note = None

    if selected_note is None:
        return SearchResult(
            original_query=query,
            semantic_query=semantic_query,
            report_year=report_year,
            auto_year_applied=auto_year_applied,
            auto_section_type=None,
            rerank_applied=False,
            rows=[],
        )

    rows.append(selected_note)

    same_year = report_year if report_year is not None else selected_note.get("report_year")

    # 3) same note table-like chunk support
    note_table_rows = _fetch_same_note_table_candidates(selected_note, semantic_query, same_year, limit=3)
    if note_table_rows:
        note_table_rows[0]["retrieval_stage"] = "same_note_table"
        rows.append(note_table_rows[0])
        confidence_rule = confidence_rule + "+table_support"

    # 4) same section fallback
    if selected_note.get("section_id"):
        section_rows = _fetch_same_section_candidates(selected_note.get("section_id"), semantic_query, same_year, limit=3)
        if section_rows:
            section_rows[0]["retrieval_stage"] = "same_section"
            rows.append(section_rows[0])
            if not confidence_rule.endswith("section_fallback"):
                confidence_rule = confidence_rule + "+section_fallback"

    rows = _dedup_rows_by_chunk_key(rows)

    for row in rows:
        row["confidence_rule"] = confidence_rule or "note_only"

    return SearchResult(
        original_query=query,
        semantic_query=semantic_query,
        report_year=report_year,
        auto_year_applied=auto_year_applied,
        auto_section_type=None,
        rerank_applied=False,
        rows=rows[: settings.top_k],
    )


def _fetch_parent_rollup_candidates(
    note_row: dict[str, Any],
    fine_row: dict[str, Any],
    query: str,
    report_year: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    note_no = note_row.get("note_no")
    if note_no is None:
        return []

    fine_path = compact_text(str(fine_row.get("subsection_path") or ""))
    clauses = [
        "c.note_no = %s",
        "c.chunk_key <> %s",
        "c.chunk_key <> %s",
        "c.section_type IN ('note', 'note_subsection', 'note_intro_block', 'note_block')",
    ]
    where_params: list[Any] = [note_no, note_row.get("chunk_key"), fine_row.get("chunk_key")]
    if report_year is not None:
        clauses.append("c.report_year = %s")
        where_params.append(report_year)

    sql = f"""
        SELECT
            d.file_name,
            c.report_year,
            c.major_section,
            c.sub_section,
            c.section_type,
            c.note_no,
            c.note_title,
            c.subtopic,
            c.section_id,
            c.subsection_path,
            c.as_of_date,
            c.period_start,
            c.period_end,
            c.evidence_type,
            c.table_meta,
            c.cell_unit,
            c.chunk_key,
            c.content,
            0.0 AS semantic_score,
            ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s)) AS keyword_score,
            CASE
                WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 1.0
                ELSE 0.0
            END AS exact_match_score,
            0.05 AS meta_boost_score,
            (
                ts_rank_cd(to_tsvector('simple', COALESCE(c.content, '')), websearch_to_tsquery('simple', %s))
                + CASE WHEN POSITION(%s IN regexp_replace(COALESCE(c.content, ''), '\\s+', '', 'g')) > 0 THEN 0.3 ELSE 0.0 END
                + CASE
                    WHEN c.section_type = 'note' THEN 0.30
                    WHEN c.section_type = 'note_subsection' THEN 0.22
                    WHEN c.section_type IN ('note_intro_block', 'note_block') THEN 0.12
                    ELSE 0.0
                  END
            ) AS hybrid_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {' AND '.join(clauses)}
        ORDER BY hybrid_score DESC
        LIMIT %s
    """
    params = [
        query,
        compact_text(query),
        query,
        compact_text(query),
        *where_params,
    ]
    params.append(limit)
    rows = _execute_search_sql(sql, params)

    if fine_path:
        for row in rows:
            path = compact_text(str(row.get("subsection_path") or ""))
            if path and (fine_path in path or path in fine_path):
                row["hybrid_score"] = float(row.get("hybrid_score", 0.0)) + 0.18

        rows.sort(key=lambda row: float(row.get("hybrid_score", 0.0)), reverse=True)

    return rows


def retrieve_risk_structured(query: str, report_year: int | None = None) -> list[dict[str, Any]]:
    inferred_year = report_year
    cleaned_query = query
    if inferred_year is None:
        inferred_year, cleaned_query = infer_report_year(query)

    risk_types = detect_risk_types(cleaned_query)
    if not risk_types:
        risk_types = list(RISK_KEYWORDS.keys())

    records: list[dict[str, Any]] = []
    for risk_type in risk_types:
        result = retrieve_note_first(query, report_year=inferred_year, risk_type=risk_type)
        records.append(build_structured_extraction(query, result, risk_type=risk_type, max_rows=2))

    return records


def build_structured_extraction(
    query: str,
    result: SearchResult,
    risk_type: str = "",
    max_rows: int = 2,
) -> dict[str, Any]:
    rows = (result.rows or [])[: max(1, max_rows)]
    if not rows:
        return {
            "year": result.report_year,
            "found": False,
            "note_title": "",
            "subtopic": "",
            "evidence_sentence": "",
            "amount": "",
            "confidence_rule": "not_found_in_note",
        }

    amount_question = _is_amount_question(query)
    rate_question = _is_rate_question(query)
    metric_question = amount_question or rate_question
    query_table_hints = _extract_table_query_hints(query, row_meta=rows[0])

    if metric_question:
        note_keys = tuple(query_table_hints.get("note_keys", []))
        if note_keys:
            has_note_match = False
            for row in rows:
                blob = compact_text(str(row.get("note_title") or "") + " " + str(row.get("subtopic") or ""))
                if any(compact_text(k) and compact_text(k) in blob for k in note_keys):
                    has_note_match = True
                    break

            if not has_note_match:
                try:
                    embedder = Embedder()
                    vec = embedder.encode_texts([result.semantic_query or query])[0]
                    extra_title = _fetch_note_title_candidates(
                        result.semantic_query or query,
                        vec,
                        result.report_year,
                        tuple(),
                        limit=max(6, max_rows * 3),
                    )
                    extra_body = _fetch_note_body_candidates(
                        result.semantic_query or query,
                        vec,
                        result.report_year,
                        limit=max(4, max_rows * 2),
                    )
                    rows = _dedup_rows_by_chunk_key(rows + extra_title + extra_body)
                except Exception:
                    pass

    if metric_question and rows:
        def _note_selection_score(row: dict[str, Any]) -> float:
            score = float(row.get("hybrid_score", 0.0))
            note_title = compact_text(str(row.get("note_title") or ""))
            subtopic = compact_text(str(row.get("subtopic") or ""))
            note_blob = f"{note_title} {subtopic}"
            compact_query = str(query_table_hints.get("query_compact") or "")

            for key in query_table_hints.get("note_keys", []):
                ck = compact_text(str(key))
                if ck and ck in note_blob:
                    score += 3.0
            for key in query_table_hints.get("topic_keywords", []):
                ck = compact_text(str(key))
                if ck and (ck in note_blob):
                    score += 1.2

            if "현금및현금성자산" in compact_query:
                if "현금및현금성자산" in note_title:
                    score += 6.0
                if "현금흐름" in note_title:
                    score -= 5.0

            if "지급보증" in compact_query and "지급보증" in note_blob:
                score += 2.5
            if "차입금" in compact_query and "차입금" in note_title:
                score += 2.0

            return score

        rows = sorted(rows, key=_note_selection_score, reverse=True)

    top = rows[0]
    column_hints = tuple(_derive_subtopic_column_hints(query, str(top.get("subtopic") or "")))
    query_table_hints = _extract_table_query_hints(query, row_meta=top)

    query_keywords = expand_note_keywords(extract_note_keywords(query))
    if risk_type:
        query_keywords.extend(RISK_KEYWORDS.get(risk_type, tuple()))
    keywords = tuple(dict.fromkeys([kw for kw in query_keywords if kw]))

    evidence_sentence = ""
    amount = ""
    source_idx = 0
    selected_row = top
    structured_used = False
    fallback_pattern_used = False
    structured_col_label = ""

    eval_rows = rows
    compact_query = str(query_table_hints.get("query_compact") or "")
    if metric_question and "현금및현금성자산" in compact_query:
        try:
            embedder = Embedder()
            vec = embedder.encode_texts([result.semantic_query or query])[0]
            extra_cash_rows = _fetch_note_body_candidates(
                result.semantic_query or query,
                vec,
                result.report_year,
                limit=max(8, max_rows * 4),
            )
            extra_cash_rows = [
                row
                for row in extra_cash_rows
                if "현금및현금성자산" in compact_text(
                    str(row.get("note_title") or "")
                    + " "
                    + str(row.get("subtopic") or "")
                    + " "
                    + str(row.get("content") or "")[:240]
                )
            ]
            if extra_cash_rows:
                eval_rows = _dedup_rows_by_chunk_key(extra_cash_rows + eval_rows)
        except Exception:
            pass

        cash_rows = [
            row
            for row in eval_rows
            if "현금및현금성자산" in compact_text(str(row.get("note_title") or "") + " " + str(row.get("subtopic") or ""))
        ]
        if cash_rows:
            remaining = [row for row in eval_rows if row not in cash_rows]
            eval_rows = cash_rows + remaining

    if metric_question:
        structured = _lookup_structured_cell_answer(
            query_table_hints,
            eval_rows,
            report_year=(result.report_year or top.get("report_year")),
        )
        if structured:
            amount = structured.get("amount") or ""
            evidence_sentence = structured.get("evidence_sentence") or ""
            structured_col_label = str(structured.get("col_label") or "")
            structured_used = bool(amount)
            for idx, row in enumerate(eval_rows):
                if (
                    structured.get("note_no") is not None
                    and row.get("note_no") == structured.get("note_no")
                    and compact_text(str(row.get("note_title") or "")) == compact_text(str(structured.get("note_title") or ""))
                ):
                    selected_row = row
                    source_idx = idx
                    break
            if not selected_row:
                selected_row = top

    for idx, row in enumerate(eval_rows):
        if metric_question and amount:
            break
        content = str(row.get("content") or "")
        if metric_question:
            if rate_question:
                m = re.search(r"(?:연)?이자율[^0-9\-+]*([0-9]+(?:\.\d+)?)", content)
                if not m:
                    m = re.search(r"([0-9]+(?:\.\d+)?)\s*%", content)
                if m:
                    amount = m.group(1)
                    evidence_sentence = _pick_evidence_sentence(content, tuple([*keywords, "연이자율", "이자율", "금리"]))
                    source_idx = idx
                    selected_row = row
                    if not structured_used:
                        fallback_pattern_used = True
                    break

            sent, amt = _pick_amount_evidence(
                content,
                keywords,
                column_hints=column_hints,
                query_hints=query_table_hints,
                row_meta=row,
            )
            if amt and not rate_question:
                evidence_sentence = sent
                amount = _normalize_amount_output(amt)
                source_idx = idx
                selected_row = row
                if not structured_used:
                    fallback_pattern_used = True
                break
        else:
            sent = _pick_evidence_sentence(content, keywords)
            if sent:
                evidence_sentence = sent
                source_idx = idx
                selected_row = row
                break

    if amount_question and not amount:
        for idx, row in enumerate(rows):
            amt = _extract_amount(str(row.get("content") or ""))
            if amt:
                amount = _normalize_amount_output(amt)
                if not evidence_sentence:
                    evidence_sentence = _pick_evidence_sentence(str(row.get("content") or ""), keywords)
                source_idx = idx
                selected_row = row
                if not structured_used:
                    fallback_pattern_used = True
                break

    if not evidence_sentence:
        evidence_sentence = str(top.get("content") or "")[:320]

    confidence_rule = str(selected_row.get("confidence_rule") or selected_row.get("retrieval_stage") or "note_only")
    if metric_question:
        confidence_rule += "+amount" if amount else "+amount_missing"
    else:
        confidence_rule += "+content_sentence" if evidence_sentence else "+content_missing"
    if metric_question:
        if structured_used:
            confidence_rule += "+structured_lookup"
        elif fallback_pattern_used:
            confidence_rule += "+fallback_pattern"
    if source_idx > 0:
        confidence_rule += f"+row{source_idx + 1}"

    selected_subtopic = selected_row.get("subtopic") or ""
    if structured_used and structured_col_label:
        if "충당부채" in compact_text(str(selected_row.get("note_title") or "")):
            col = _clean_table_label(re.sub(r"\([^)]*\)", "", structured_col_label)).strip()
            if col and col not in {"계", "합계", "총액"}:
                selected_subtopic = col if col.endswith("충당부채") else f"{col}충당부채"

    return {
        "year": selected_row.get("report_year") or result.report_year,
        "found": bool(selected_row.get("note_title") or selected_row.get("subtopic") or selected_row.get("content")),
        "note_title": selected_row.get("note_title") or "",
        "subtopic": selected_subtopic,
        "evidence_sentence": evidence_sentence,
        "amount": amount,
        "confidence_rule": confidence_rule,
    }

def print_search_result(result: SearchResult) -> None:
    print(f"\n[Query] {result.original_query}")
    if result.auto_year_applied:
        print(f"[AutoFilter] report_year={result.report_year}")
    if result.auto_section_type is not None:
        print(f"[AutoFilter] section_type={result.auto_section_type}")
    print(
        f"[Retrieval] hybrid(semantic={settings.semantic_weight:.2f}, "
        f"keyword={settings.keyword_weight:.2f}), candidates={settings.hybrid_candidate_k}"
    )
    if result.rerank_applied:
        print(f"[Reranker] applied model={settings.reranker_model}, top_n={settings.rerank_top_n}")
    else:
        print("[Reranker] not applied")
    print()
    for idx, row in enumerate(result.rows, start=1):
        print(f"[{idx}] {row['file_name']} ({row['report_year']})")
        print(f"- chunk_key : {row['chunk_key']}")
        print(f"- major     : {row['major_section']}")
        print(f"- sub       : {row['sub_section']}")
        print(f"- note      : {row['note_no']} / {row['note_title']}")
        print(f"- semantic  : {row['semantic_score']:.4f}")
        print(f"- keyword   : {row['keyword_score']:.4f}")
        print(f"- exact     : {row['exact_match_score']:.4f}")
        print(f"- meta      : {row.get('meta_boost_score', 0.0):.4f}")
        print(f"- hybrid    : {row['hybrid_score']:.4f}")
        if 'rerank_score' in row:
            print(f"- rerank    : {row['rerank_score']:.4f}")
        print(f"- snippet   : {row['content'][:300]}\n")


def search(query: str, report_year: int | None = None, sub_section: str | None = None) -> None:
    result = retrieve(query, report_year=report_year, sub_section=sub_section)
    print_search_result(result)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        raise SystemExit("사용법: poetry run python -m app.search '현금및현금성자산'")
    search(query)
