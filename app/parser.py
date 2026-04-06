from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from bs4 import BeautifulSoup, FeatureNotFound, NavigableString, Tag


STATEMENT_TITLES = {
    "재무상태표",
    "손익계산서",
    "포괄손익계산서",
    "자본변동표",
    "현금흐름표",
}

SKIP_COMPACT_TEXTS = {
    "별첨주석은본재무제표의일부입니다.",
    "계속;",
    "계속",
}

YEAR_RE = re.compile(r"(20\d{2})년")
FILENAME_YEAR_RE = re.compile(r"(20\d{2})")

# note parser regex
NOTE_START_RE = re.compile(r"^\s*(\d{1,3})\s*[.\)]\s*(.+?)\s*$")
DECIMAL_SECTION_RE = re.compile(r"^\s*(\d{1,3}\.\d{1,3})\s+(.+?)\s*$")
KOREAN_SECTION_RE = re.compile(r"^\s*([가-하])\.\s*(.+?)\s*$")
PAREN_SECTION_RE = re.compile(r"^\s*\((\d{1,3})\)\s*(.+?)\s*$")
CONTINUATION_RE = re.compile(r"^\s*(.+?)\s*,?\s*계\s*속\s*:?\s*$")
ONLY_CONTINUATION_RE = re.compile(r"^\s*계\s*속\s*:?\s*$")


@dataclass(slots=True)
class ParsedReport:
    file_path: Path
    file_name: str
    title: str
    report_year: int | None
    raw_html: str
    parser_used: str
    structured: dict[str, Any]


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def normalize_compact(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def read_html_text(file_path: str | Path) -> str:
    path = Path(file_path)
    raw = path.read_bytes()
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("euc-kr", errors="ignore")


def build_soup_with_fallback(html_text: str) -> tuple[BeautifulSoup, str]:
    errors: list[str] = []
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            return BeautifulSoup(html_text, parser), parser
        except FeatureNotFound as exc:
            errors.append(f"{parser}: {exc}")
        except Exception as exc:
            errors.append(f"{parser}: {exc}")
    raise RuntimeError("No HTML parser available: " + " | ".join(errors))


def tag_text(tag: Tag) -> str:
    return normalize_space(tag.get_text(" ", strip=True))


def get_section_level(tag: Tag) -> Optional[int]:
    for cls in (tag.get("class") or []):
        m = re.match(r"(?i)^section-(\d+)$", str(cls).strip())
        if m:
            return int(m.group(1))
    return None


def is_section_header(tag: Tag) -> bool:
    return isinstance(tag, Tag) and tag.name in ("h2", "h3") and get_section_level(tag) in (1, 2)


def get_section_name(header_tag: Tag) -> str:
    text = normalize_space(header_tag.get_text(" ", strip=True))
    if text:
        return text

    level = get_section_level(header_tag)
    for sib in header_tag.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if not isinstance(sib, Tag):
            break
        if is_section_header(sib):
            break
        sib_level = get_section_level(sib)
        if sib_level == level and sib.name in ("p", "div", "span"):
            text = normalize_space(sib.get_text(" ", strip=True))
            if text:
                return text
        if sib.name not in ("p", "div", "span"):
            break
    return "(섹션명 없음)"


def iter_section_nodes(header_tag: Tag) -> Iterable[Tag]:
    level = get_section_level(header_tag) or 1
    for sib in header_tag.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if not isinstance(sib, Tag):
            continue
        if is_section_header(sib):
            sib_level = get_section_level(sib) or 99
            if sib_level <= level:
                break
        yield sib


def table_rows_to_text(table_tag: Tag) -> str:
    lines: list[str] = []
    for tr in table_tag.find_all("tr"):
        row: list[str] = []
        for cell in tr.find_all(["th", "td"]):
            txt = tag_text(cell)
            if txt:
                row.append(txt)
        if row:
            lines.append(" | ".join(row))
    return "\n".join(lines).strip()


def detect_statement_title_from_table(table_tag: Tag) -> str | None:
    first_tr = table_tag.find("tr")
    if first_tr is None:
        return None
    cells = first_tr.find_all(["td", "th"])
    if not cells:
        return None
    for cell in cells[:3]:
        compact = normalize_compact(tag_text(cell))
        if compact in STATEMENT_TITLES:
            return compact
    table_compact = normalize_compact(table_rows_to_text(table_tag))
    for title in STATEMENT_TITLES:
        if title in table_compact[:80]:
            return title
    return None


def infer_company_name(raw_html: str, soup: BeautifulSoup) -> str:
    title = normalize_space(soup.title.get_text()) if soup.title else ""
    if "삼성전자" in title:
        return "삼성전자주식회사"
    text = normalize_space((soup.body or soup).get_text(" ", strip=True))
    if "삼성전자주식회사" in text[:5000]:
        return "삼성전자주식회사"
    return "삼성전자주식회사"


def infer_report_year(path: Path, raw_html: str) -> int | None:
    filename_match = FILENAME_YEAR_RE.search(path.stem)
    if filename_match:
        return int(filename_match.group(1))

    early_text = raw_html[:6000]
    for pattern in [
        re.compile(r"제\s*\d+\s*기.*?(20\d{2})년\s*01월\s*01일", re.S),
        re.compile(r"제\s*\d+\s*기.*?(20\d{2})년\s*12월\s*31일", re.S),
    ]:
        m = pattern.search(early_text)
        if m:
            return int(m.group(1))

    years = [int(y) for y in YEAR_RE.findall(early_text)]
    if years:
        counts: dict[int, int] = {}
        for y in years:
            counts[y] = counts.get(y, 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return None


def classify_major_type(title: str) -> str:
    compact = normalize_compact(title)
    if compact in {"(섹션명없음)", ""}:
        return "other"
    if "독립된감사인의감사보고서" in compact:
        return "audit_report"
    if "첨부" in compact and "재무제표" in compact:
        return "financial_statements_bundle"
    if "내부회계관리제도" in compact and ("검토보고서" in compact or "검토의견" in compact):
        return "internal_control_review"
    if "내부회계관리제도운영실태평가보고서" in compact:
        return "internal_control_management_report"
    if "외부감사실시내용" in compact:
        return "external_audit_details"
    return "other"


def _append_block(target: list[dict[str, Any]], block_type: str, text: str, **meta: Any) -> None:
    text = normalize_space(text)
    if not text:
        return
    if normalize_compact(text) in SKIP_COMPACT_TEXTS:
        return
    block = {"block_type": block_type, "text": text}
    if meta:
        block.update(meta)
    target.append(block)


# -----------------------------
# hierarchical note parsing
# -----------------------------

def clean_heading_text(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\b계\s*속\s*:?;?\s*$", "", text).strip(" ,:-")
    return text.strip(" ,")


def normalize_heading_text(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\s*계\s*속\s*:?\\s*$", "", text).strip(" :")
    return text.strip()


def clean_title_tail(title: str) -> str:
    if not title:
        return ""
    title = re.sub(r"\s*:\s*$", "", title)
    title = re.sub(r"\s*계\s*속\s*$", "", title)
    return title.strip()


def make_block(block_type: str, text: str) -> dict[str, str]:
    return {"block_type": block_type, "text": normalize_space(text)}


def make_section_node(code: str, title: str, level: int) -> dict[str, Any]:
    return {
        "code": code,
        "title": clean_title_tail(title),
        "level": level,
        "content_blocks": [],
        "children": [],
    }


def make_note_node(note_no: int, title: str) -> dict[str, Any]:
    return {
        "note_no": note_no,
        "note_title": clean_title_tail(title),
        "intro_blocks": [],
        "subsections": [],
        "raw_blocks": [],
    }


def append_to_current(container: Optional[dict[str, Any]], block: dict[str, str]) -> None:
    if container is not None:
        container["content_blocks"].append(block)


def append_to_note_intro(note: Optional[dict[str, Any]], block: dict[str, str]) -> None:
    if note is not None:
        note["intro_blocks"].append(block)


def classify_heading(line: str) -> dict[str, Any] | None:
    line = normalize_heading_text(line)
    if not line:
        return None

    m = DECIMAL_SECTION_RE.match(line)
    if m:
        code, title = m.groups()
        note_no = int(code.split(".")[0])
        return {
            "kind": "decimal",
            "code": code,
            "title": title,
            "level": 1,
            "note_no": note_no,
        }

    m = KOREAN_SECTION_RE.match(line)
    if m:
        code, title = m.groups()
        return {
            "kind": "korean",
            "code": f"{code}.",
            "title": title,
            "level": 2,
        }

    m = PAREN_SECTION_RE.match(line)
    if m:
        code, title = m.groups()
        return {
            "kind": "paren",
            "code": f"({code})",
            "title": title,
            "level": 3,
        }

    m = NOTE_START_RE.match(line)
    if m:
        no, title = m.groups()
        return {
            "kind": "note",
            "note_no": int(no),
            "title": title,
            "level": 0,
        }

    return None


def parse_embedded_note_heading(text: str) -> dict[str, Any] | None:
    text = normalize_space(text)

    m = re.match(r"^\s*(\d{1,3})\s*[.\)]\s*(.+?)\s*:\s*(.+)$", text)
    if m:
        note_no, title, rest = m.groups()
        return {
            "type": "note_with_rest",
            "note_no": int(note_no),
            "title": clean_title_tail(title),
            "rest": normalize_space(rest),
        }

    m = re.match(r"^\s*(\d{1,3})\s*[.\)]\s*(.+?)\s*,?\s*계\s*속\s*:\s*(.+)$", text)
    if m:
        note_no, title, rest = m.groups()
        return {
            "type": "note_continue_with_rest",
            "note_no": int(note_no),
            "title": clean_title_tail(title),
            "rest": normalize_space(rest),
        }

    return None


def split_inline_child_heading(text: str) -> dict[str, str] | None:
    text = normalize_space(text)

    m = KOREAN_SECTION_RE.match(text)
    if m:
        code, title = m.groups()
        return {"kind": "korean", "code": f"{code}.", "title": clean_title_tail(title)}

    m = PAREN_SECTION_RE.match(text)
    if m:
        code, title = m.groups()
        return {"kind": "paren", "code": f"({code})", "title": clean_title_tail(title)}

    return None


def flatten_lines_from_block(block: dict[str, Any]) -> list[dict[str, str]]:
    block_type = block.get("block_type", "paragraph")
    text = normalize_space(block.get("text", ""))
    if not text:
        return []
    if block_type == "table":
        return [make_block("table", text)]
    lines = [normalize_space(x) for x in text.split("\n")]
    lines = [x for x in lines if x]
    return [make_block("paragraph", x) for x in lines]


def postprocess_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def clean_section(node: dict[str, Any]) -> dict[str, Any] | None:
        node["title"] = clean_title_tail(node.get("title", ""))
        cleaned_children = []
        for child in node.get("children", []):
            cleaned = clean_section(child)
            if cleaned is not None:
                cleaned_children.append(cleaned)
        node["children"] = cleaned_children

        has_content = bool(node.get("content_blocks"))
        has_children = bool(node.get("children"))
        has_title = bool(node.get("title"))
        is_root = node.get("code") == "ROOT"

        if is_root and not has_content and not has_children:
            return None
        if not is_root and not has_title and not has_content and not has_children:
            return None
        return node

    result = []
    for note in notes:
        note["note_title"] = clean_title_tail(note.get("note_title", ""))
        cleaned_subs = []
        for sec in note.get("subsections", []):
            cleaned = clean_section(sec)
            if cleaned is None:
                continue
            if cleaned.get("code") == "ROOT":
                if cleaned.get("content_blocks"):
                    note["intro_blocks"].extend(cleaned["content_blocks"])
                cleaned_subs.extend(cleaned.get("children", []))
            else:
                cleaned_subs.append(cleaned)
        note["subsections"] = cleaned_subs
        result.append(note)
    return result


def parse_notes_hierarchical(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    current_note: Optional[dict[str, Any]] = None
    current_decimal: Optional[dict[str, Any]] = None
    current_korean: Optional[dict[str, Any]] = None
    current_paren: Optional[dict[str, Any]] = None

    def reset_lower(level: int) -> None:
        nonlocal current_decimal, current_korean, current_paren
        if level <= 1:
            current_decimal = None
        if level <= 2:
            current_korean = None
        if level <= 3:
            current_paren = None

    def start_new_note(note_no: int, title: str) -> None:
        nonlocal current_note, current_decimal, current_korean, current_paren
        current_note = make_note_node(note_no, title)
        notes.append(current_note)
        current_decimal = None
        current_korean = None
        current_paren = None

    def ensure_note(note_no: int, title: str) -> None:
        nonlocal current_note
        if current_note is None or current_note["note_no"] != note_no:
            start_new_note(note_no, title)
        elif title and not current_note["note_title"]:
            current_note["note_title"] = clean_title_tail(title)

    def start_decimal_section(code: str, title: str) -> None:
        nonlocal current_decimal, current_korean, current_paren, current_note
        if current_note is None:
            return
        node = make_section_node(code, title, level=1)
        current_note["subsections"].append(node)
        current_decimal = node
        current_korean = None
        current_paren = None

    def start_korean_section(code: str, title: str) -> None:
        nonlocal current_korean, current_paren, current_decimal, current_note
        if current_note is None:
            return
        parent = current_decimal
        if parent is None:
            parent = make_section_node("ROOT", "", level=1)
            current_note["subsections"].append(parent)
            current_decimal = parent
        node = make_section_node(code, title, level=2)
        parent["children"].append(node)
        current_korean = node
        current_paren = None

    def start_paren_section(code: str, title: str) -> None:
        nonlocal current_paren, current_korean, current_decimal, current_note
        if current_note is None:
            return
        parent = current_korean or current_decimal
        if parent is None:
            parent = make_section_node("ROOT", "", level=1)
            current_note["subsections"].append(parent)
            current_decimal = parent
        node = make_section_node(code, title, level=3)
        parent["children"].append(node)
        current_paren = node

    def get_current_target() -> Optional[dict[str, Any]]:
        if current_paren is not None:
            return current_paren
        if current_korean is not None:
            return current_korean
        if current_decimal is not None and current_decimal.get("code") != "ROOT":
            return current_decimal
        return None

    for raw_block in content_blocks:
        for block in flatten_lines_from_block(raw_block):
            text = block["text"]
            block_type = block["block_type"]

            if current_note is not None:
                current_note["raw_blocks"].append(block)

            if block_type == "table":
                target = get_current_target()
                if target is not None:
                    append_to_current(target, block)
                else:
                    append_to_note_intro(current_note, block)
                continue

            if ONLY_CONTINUATION_RE.match(text):
                continue

            embedded = parse_embedded_note_heading(text)
            if embedded:
                ensure_note(embedded["note_no"], embedded["title"])
                reset_lower(1)
                rest = embedded["rest"]
                inline_child = split_inline_child_heading(rest)
                if inline_child:
                    if inline_child["kind"] == "korean":
                        start_korean_section(inline_child["code"], inline_child["title"])
                    else:
                        start_paren_section(inline_child["code"], inline_child["title"])
                else:
                    append_to_note_intro(current_note, make_block("paragraph", rest))
                continue

            heading = classify_heading(text)
            if heading:
                if heading["kind"] == "note":
                    start_new_note(heading["note_no"], heading["title"])
                    continue
                if heading["kind"] == "decimal":
                    ensure_note(heading["note_no"], "")
                    start_decimal_section(heading["code"], heading["title"])
                    continue
                if heading["kind"] == "korean":
                    start_korean_section(heading["code"], heading["title"])
                    continue
                if heading["kind"] == "paren":
                    start_paren_section(heading["code"], heading["title"])
                    continue

            if CONTINUATION_RE.match(text):
                continue

            target = get_current_target()
            if target is not None:
                append_to_current(target, block)
            else:
                append_to_note_intro(current_note, block)

    return postprocess_notes(notes)


def collect_notes_nodes(major_nodes: list[Tag]) -> list[Tag]:
    notes_header = None
    for node in major_nodes:
        if is_section_header(node) and get_section_level(node) == 2 and "주석" in get_section_name(node):
            notes_header = node
            break
    if notes_header is None:
        return []

    nodes: list[Tag] = []
    for sib in notes_header.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if not isinstance(sib, Tag):
            continue
        if is_section_header(sib) and (get_section_level(sib) or 99) <= 1:
            break
        nodes.append(sib)
    return nodes


def extract_financial_sections(major_nodes: list[Tag]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sections: list[dict[str, Any]] = []
    major_blocks: list[dict[str, Any]] = []
    used_tables: set[int] = set()

    tables = [node for node in major_nodes if node.name == "table"]
    idx = 0
    while idx < len(tables):
        table = tables[idx]
        statement_title = detect_statement_title_from_table(table)
        if statement_title and idx + 1 < len(tables):
            data_table = tables[idx + 1]
            data_text = table_rows_to_text(data_table)
            title_text = table_rows_to_text(table)
            if data_text:
                sections.append({
                    "title": statement_title,
                    "section_type": "financial_statement",
                    "content_blocks": [
                        {"block_type": "table", "text": title_text},
                        {"block_type": "table", "text": data_text},
                    ],
                    "notes": [],
                })
                used_tables.add(id(table))
                used_tables.add(id(data_table))
                idx += 2
                continue
        idx += 1

    notes_nodes = collect_notes_nodes(major_nodes)
    if notes_nodes:
        note_blocks: list[dict[str, Any]] = []
        for node in notes_nodes:
            if node.name == "p" and "PGBRK" in (node.get("class") or []):
                continue
            if node.name == "table":
                text = table_rows_to_text(node)
                if text:
                    note_blocks.append({"block_type": "table", "text": text})
            else:
                text = tag_text(node)
                if text:
                    note_blocks.append({"block_type": "paragraph", "text": text})

        notes = parse_notes_hierarchical(note_blocks)
        sections.append({
            "title": "주석",
            "section_type": "notes_section",
            "content_blocks": [],
            "notes": notes,
        })

    for node in major_nodes:
        if node.name == "table" and id(node) in used_tables:
            continue
        if is_section_header(node):
            continue
        if node.name == "table":
            text = table_rows_to_text(node)
            if text:
                _append_block(major_blocks, "table", text)
        else:
            text = tag_text(node)
            if text:
                _append_block(major_blocks, "paragraph", text)

    return sections, major_blocks


def extract_generic_major_content(major_nodes: list[Tag]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for node in major_nodes:
        if is_section_header(node):
            continue
        if node.name == "table":
            text = table_rows_to_text(node)
            if text:
                _append_block(blocks, "table", text)
        else:
            text = tag_text(node)
            if text:
                _append_block(blocks, "paragraph", text)
    return blocks


def has_bold_style(tag: Tag) -> bool:
    style = (tag.get("style") or "").lower().replace(" ", "")
    if "font-weight:bold" in style:
        return True
    if "font-weight:700" in style:
        return True
    return False


def is_bold_p_tag(tag: Tag) -> bool:
    return tag.name == "p" and has_bold_style(tag)


def iter_text_segments_with_bold(node: Tag) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    for text_node in node.find_all(string=True):
        text = normalize_space(str(text_node))
        if not text:
            continue

        is_bold = False
        parent = text_node.parent
        while isinstance(parent, Tag):
            if has_bold_style(parent):
                is_bold = True
                break
            if parent is node:
                break
            parent = parent.parent

        if segments and segments[-1][1] == is_bold:
            merged_text = normalize_space(f"{segments[-1][0]} {text}")
            segments[-1] = (merged_text, is_bold)
        else:
            segments.append((text, is_bold))
    return segments


def extract_audit_report_sections(major_nodes: list[Tag]) -> list[dict[str, Any]]:
    """Parse '독립된 감사인의 감사보고서' by bold subtitle boundaries.

    Rules:
    - Ignore everything before the first bold p subtitle.
    - Each bold p starts a new subsection.
    - Collect all following tag texts until the next bold p.
    - Stop parsing entirely when a table appears.
    """
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_parts: list[str] = []
    has_seen_first_subtitle = False

    def flush_current() -> None:
        nonlocal current_title, current_parts
        if not current_title:
            return
        content = "\n\n".join(part for part in current_parts if part.strip())
        if not content:
            return
        sections.append(
            {
                "title": current_title,
                "section_type": "audit_subsection",
                "content_blocks": [{"block_type": "paragraph", "text": content}],
                "notes": [],
            }
        )

    for node in major_nodes:
        if is_section_header(node):
            continue
        if node.name == "table" and has_seen_first_subtitle:
            break

        if node.name == "p":
            segments = iter_text_segments_with_bold(node)
            if not segments:
                continue

            for seg_text, seg_is_bold in segments:
                if seg_is_bold:
                    subtitle = normalize_space(seg_text)
                    if not subtitle:
                        continue
                    if has_seen_first_subtitle:
                        flush_current()
                    has_seen_first_subtitle = True
                    current_title = subtitle
                    current_parts = []
                else:
                    if has_seen_first_subtitle:
                        current_parts.append(seg_text)
            continue

        if not has_seen_first_subtitle:
            continue

        text = tag_text(node)
        if text:
            current_parts.append(text)

    flush_current()
    return sections


def parse_html_file(file_path: str | Path) -> ParsedReport:
    path = Path(file_path)
    raw_html = read_html_text(path)
    soup, parser_name = build_soup_with_fallback(raw_html)

    title = normalize_space(soup.title.get_text()) if soup.title else path.stem
    report_year = infer_report_year(path, raw_html)
    company = infer_company_name(raw_html, soup)

    structured: dict[str, Any] = {
        "document_meta": {
            "source_file": path.name,
            "company": company,
            "report_year_guess": report_year,
            "parser_used": parser_name,
        },
        "major_sections": [],
    }

    headers = [tag for tag in soup.find_all(["h2", "h3"]) if is_section_header(tag)]
    major_headers = [h for h in headers if get_section_level(h) == 1]

    if not major_headers:
        body = soup.body or soup
        major = {
            "title": "UNKNOWN",
            "major_type": "other",
            "content_blocks": [],
            "sections": [],
        }
        for child in body.find_all(recursive=False):
            if child.name == "table":
                text = table_rows_to_text(child)
                if text:
                    _append_block(major["content_blocks"], "table", text)
            else:
                text = tag_text(child)
                if text:
                    _append_block(major["content_blocks"], "paragraph", text)
        structured["major_sections"].append(major)
    else:
        for header in major_headers:
            title_text = get_section_name(header)
            major_nodes = list(iter_section_nodes(header))
            major_type = classify_major_type(title_text)

            major_section = {
                "title": title_text,
                "major_type": major_type,
                "content_blocks": [],
                "sections": [],
            }

            if major_type == "financial_statements_bundle":
                sections, major_blocks = extract_financial_sections(major_nodes)
                major_section["sections"] = sections
                major_section["content_blocks"] = major_blocks
            elif major_type == "audit_report":
                major_section["sections"] = extract_audit_report_sections(major_nodes)
                major_section["content_blocks"] = []
            else:
                major_section["content_blocks"] = extract_generic_major_content(major_nodes)

            structured["major_sections"].append(major_section)

    return ParsedReport(
        file_path=path,
        file_name=path.name,
        title=title,
        report_year=report_year,
        raw_html=raw_html,
        parser_used=parser_name,
        structured=structured,
    )


def save_parsed_json(report: ParsedReport, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / f"{report.file_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.structured, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path