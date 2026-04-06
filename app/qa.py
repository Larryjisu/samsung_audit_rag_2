from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import re
import sys

from app.generator import RagAnswerGenerator
from app.search_local import (
    SearchResult,
    build_structured_extraction,
    detect_risk_types,
    retrieve,
    retrieve_note_first,
)


SYSTEM_PROMPT = """당신은 삼성전자 2014-2024년 감사보고서를 분석하는 금융 RAG 어시스턴트다.

답변 규칙:
1. 반드시 제공된 검색 근거만 사용한다.
2. 검색 근거에 없는 사실은 추측하지 않는다.
3. 연도별 정보가 다르면 연도별로 구분해서 답한다.
4. 먼저 질문에 대한 직접 답변을 2-4문장으로 작성한다.
5. 답변은 '근거 1, 근거 2' 같은 라벨 나열 형식으로 쓰지 않는다.
6. 마지막에만 근거 출처를 1-2줄로 요약한다.
7. 숫자나 문구가 불확실하면 '근거 부족'이라고 명시한다.
"""

TEST_QUESTIONS = [
    "2020년도 감사의견근거는 무엇이야?",
    "2020년도 핵심감사사항은 무엇이야?",
    "2014년도 감사인의 책임은 어떻게 설명돼?",
    "2024년도 감사 의견은 무엇이야?",
    "2021년도 현금및현금성자산은 어떻게 설명돼?",
    "2022년도 재무활동 현금흐름 관련 내용 찾아줘",
    "2019년도 중요한 회계처리방침에서 현금및현금성자산 설명을 보여줘",
    "2023년도 독립된 감사인의 감사보고서에서 기타사항은 뭐야?",
    "2020년도 재무제표감사에 대한 감사인의 책임을 요약해줘",
    "2018년도 감사의견은 적정의견이야?",
]

GENERIC_MARKERS = [
    "비교", "변화", "추이", "종합", "전체", "대표", "관련", "정리", "요약",
    "어떤", "무엇", "포함", "연도별", "최근", "여러", "각각"
]

RISK_TOPIC_EXPANSION = {
    "소송": ["소송", "계류중인 소송", "충당부채"],
    "담보/보증": ["담보", "보증", "제공한 담보", "지급보증"],
    "우발부채/약정": ["우발부채", "약정사항", "기타 약정사항", "지급보증", "소송"],
}

AMOUNT_PATTERNS = [
    r"\d[\d,]*\s*백만원",
    r"\d[\d,]*\s*천원",
    r"\d[\d,]*\s*원",
    r"\d[\d,]*",
]


def format_context(rows: list[dict[str, object]], max_sources: int = 3) -> str:
    blocks: list[str] = []
    for idx, row in enumerate(rows[:max_sources], start=1):
        blocks.append(
            "\n".join(
                [
                    f"[출처 {idx}]",
                    f"- file_name: {row['file_name']}",
                    f"- report_year: {row['report_year']}",
                    f"- major_section: {row['major_section']}",
                    f"- sub_section: {row['sub_section']}",
                    f"- chunk_key: {row['chunk_key']}",
                    f"- content: {row['content']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_rag_user_prompt(result: SearchResult) -> str:
    context = format_context(result.rows, max_sources=1)
    return "\n\n".join(
        [
            f"[User Question]\n{result.original_query}",
            f"[Retrieved Context]\n{context}",
            "[Assistant Task]\n"
            "- 질문에 대한 최종 답변을 먼저 작성하라 (2-4문장).\n"
            "- '근거 1', '근거 2' 같은 라벨 나열/복붙을 금지한다.\n"
            "- 마지막에만 '출처: 연도 | 대제목 | 소제목' 형식으로 1-2줄 작성하라.\n"
            "- 근거가 부족하면 답변 본문에서 명확히 '근거 부족'이라고 적어라.",
        ]
    )


def summarize_top_content(raw: str, max_sentences: int = 3) -> str:
    text = raw.strip()
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if not parts:
        return text[:360]
    return " ".join(parts[:max_sentences]).strip()


def extract_years(question: str) -> list[str]:
    years = re.findall(r"(20\d{2})", question)
    return sorted(set(years))


def detect_query_mode(question: str) -> str:
    years = extract_years(question)

    if len(years) >= 2:
        return "generic"

    if any(marker in question for marker in GENERIC_MARKERS):
        return "generic"

    return "local"


def detect_generic_risk_bucket(question: str, risk_types: list[str]) -> str | None:
    if risk_types:
        primary = risk_types[0]
        if primary in {"소송", "담보/보증", "우발부채/약정"}:
            return primary

    if "우발부채" in question or "약정" in question:
        return "우발부채/약정"
    if "담보" in question or "보증" in question:
        return "담보/보증"
    if "소송" in question:
        return "소송"

    return None


def expand_generic_queries(question: str, risk_types: list[str]) -> list[str]:
    years = extract_years(question)
    bucket = detect_generic_risk_bucket(question, risk_types)

    if not years:
        years = [""]

    queries: list[str] = []

    if bucket and bucket in RISK_TOPIC_EXPANSION:
        for year in years:
            for topic in RISK_TOPIC_EXPANSION[bucket]:
                q = f"{year}년 {topic}".strip()
                queries.append(q)
    else:
        queries.append(question)
        for year in years:
            queries.append(f"{year}년 {question}")

    deduped: list[str] = []
    seen = set()
    for q in queries:
        q_norm = re.sub(r"\s+", " ", q).strip()
        if q_norm and q_norm not in seen:
            seen.add(q_norm)
            deduped.append(q_norm)

    return deduped


def extract_amount_candidates(text: str, max_items: int = 3) -> list[str]:
    found: list[str] = []
    clean = re.sub(r"\s+", " ", text)

    for pattern in AMOUNT_PATTERNS:
        matches = re.findall(pattern, clean)
        for m in matches:
            m = m.strip()
            if m and m not in found:
                found.append(m)
            if len(found) >= max_items:
                return found

    return found


def retrieve_many(queries: list[str], top_k_per_query: int = 2) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen_keys: set[tuple[object, object, object, object]] = set()

    for q in queries:
        result = retrieve(q)

        for row in result.rows[:top_k_per_query]:
            key = (
                row.get("report_year"),
                row.get("major_section"),
                row.get("sub_section"),
                row.get("chunk_key"),
            )
            if key in seen_keys:
                continue

            seen_keys.add(key)
            new_row = dict(row)
            new_row["matched_query"] = q
            merged.append(new_row)

    merged.sort(
        key=lambda r: (
            str(r.get("report_year", "")),
            float(r.get("hybrid_score", 0.0)),
        ),
        reverse=True,
    )
    return merged


def infer_risk_type_from_row(row: dict[str, object]) -> str:
    text = " ".join(
        [
            str(row.get("major_section", "")),
            str(row.get("sub_section", "")),
            str(row.get("content", "")),
        ]
    )

    if "소송" in text:
        return "소송"
    if "담보" in text or "보증" in text:
        return "담보/보증"
    if "우발부채" in text or "약정" in text:
        return "우발부채/약정"
    return "기타"


def build_generic_aggregation(
    question: str,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []

    for row in rows:
        content = str(row.get("content", "")).strip()
        if not content:
            continue

        evidence = summarize_top_content(content, max_sentences=2)
        amounts = extract_amount_candidates(content, max_items=3)

        item = {
            "year": row.get("report_year"),
            "risk_type": infer_risk_type_from_row(row),
            "major_section": row.get("major_section"),
            "sub_section": row.get("sub_section"),
            "note_title": row.get("sub_section") or row.get("major_section"),
            "evidence": evidence,
            "amounts": amounts,
            "chunk_key": row.get("chunk_key"),
            "matched_query": row.get("matched_query"),
            "hybrid_score": row.get("hybrid_score", 0.0),
        }
        items.append(item)

    return items


def render_generic_answer(question: str, items: list[dict[str, object]]) -> str:
    if not items:
        return "근거 부족: 여러 근거를 결합할 검색 결과를 찾지 못했습니다."

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        year = str(item.get("year", "연도미상"))
        grouped[year].append(item)

    lines: list[str] = []
    for year in sorted(grouped.keys()):
        lines.append(f"{year}년:")

        used = set()
        year_items = sorted(
            grouped[year],
            key=lambda x: float(x.get("hybrid_score", 0.0)),
            reverse=True,
        )

        for item in year_items:
            key = (
                item.get("note_title"),
                item.get("evidence"),
            )
            if key in used:
                continue
            used.add(key)

            sub = str(item.get("note_title", ""))
            evidence = str(item.get("evidence", ""))
            amounts = item.get("amounts", [])

            if amounts:
                amt_text = ", ".join(amounts[:2])
                lines.append(f"- {sub}: {evidence} (금액 후보: {amt_text})")
            else:
                lines.append(f"- {sub}: {evidence}")

        lines.append("")

    lines.append("출처 요약:")
    source_seen = set()
    for item in items[:6]:
        src = f"{item.get('year')} | {item.get('major_section')} | {item.get('sub_section')}"
        if src not in source_seen:
            source_seen.add(src)
            lines.append(f"- {src}")

    return "\n".join(lines).strip()


def print_generic_debug(question: str, subqueries: list[str], merged_rows: list[dict[str, object]]) -> None:
    print("\n=== Generic Query Mode ===")
    print(f"question: {question}")
    print(f"subqueries ({len(subqueries)}):")
    for q in subqueries:
        print(f"- {q}")

    print("\n=== Merged Retrieved Rows ===")
    for idx, row in enumerate(merged_rows, start=1):
        print(
            f"[{idx}] year={row.get('report_year')} | "
            f"major={row.get('major_section')} | sub={row.get('sub_section')} | "
            f"matched_query={row.get('matched_query')} | "
            f"hybrid={float(row.get('hybrid_score', 0.0)):.4f}"
        )
        print(f"    {str(row.get('content', ''))[:220]}\n")


def run_generic(question: str, debug: bool = False) -> None:
    risk_types = detect_risk_types(question)
    subqueries = expand_generic_queries(question, risk_types)
    merged_rows = retrieve_many(subqueries, top_k_per_query=2)
    aggregated = build_generic_aggregation(question, merged_rows)

    if debug:
        print_generic_debug(question, subqueries, merged_rows)

    print("=== Aggregated Answer ===")
    print(render_generic_answer(question, aggregated))


@lru_cache(maxsize=1)
def get_generator() -> RagAnswerGenerator:
    return RagAnswerGenerator()


def generate_answer(result: SearchResult, thinking: bool = False) -> tuple[str, str, str]:
    generator = get_generator()
    generation = generator.generate(SYSTEM_PROMPT.strip(), build_rag_user_prompt(result), thinking=thinking)
    cleaned = clean_generated_answer(generation.answer)
    if not cleaned:
        cleaned = build_fallback_answer(result)
    return cleaned, generation.model_name, generation.device


def generate_grounded_answer(result: SearchResult) -> str:
    """Return a deterministic answer synthesized only from retrieved context."""
    if not result.rows:
        return "근거 부족: 검색 결과가 없어 답변할 수 없습니다."

    summary_row, summary_text = synthesize_grounded_summary(result.semantic_query, result.rows)
    if summary_row is None:
        return build_fallback_answer(result)

    year = summary_row.get("report_year")
    answer_text = f"{year}년도 보고서 기준 요약: {summary_text}"

    source = (
        f"출처: {summary_row.get('report_year')} | {summary_row.get('major_section')} | {summary_row.get('sub_section')}"
    )
    return f"{answer_text}\n\n{source}"


def clean_generated_answer(answer: str) -> str:
    text = answer.strip()
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def build_fallback_answer(result: SearchResult) -> str:
    if not result.rows:
        return "근거 부족: 검색 결과가 없어 답변을 생성할 수 없습니다."

    top = result.rows[0]
    raw = str(top.get("content", ""))
    _, summary = synthesize_grounded_summary(result.semantic_query, [top])
    if not summary:
        summary = summarize_top_content(raw, max_sentences=2)

    source = (
        f"출처: {top.get('report_year')} | {top.get('major_section')} | {top.get('sub_section')}"
    )
    return f"{summary}\n\n{source}"


def synthesize_grounded_summary(
    query: str,
    rows: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str]:
    query_tokens = [tok for tok in re.split(r"\s+", query.strip()) if tok]
    compact_query = re.sub(r"\s+", "", query)

    best_row: dict[str, object] | None = None
    best_sentences: list[tuple[int, str]] = []
    best_score = -1

    for row in rows[:3]:
        raw = str(row.get("content", ""))
        clean = re.sub(r"^\[[^\]]+\]\s*", "", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            continue

        sentence_candidates = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if len(s.strip()) >= 18
        ]
        if not sentence_candidates:
            continue

        scored: list[tuple[int, str]] = []
        total = 0
        for sentence in sentence_candidates:
            compact_sentence = re.sub(r"\s+", "", sentence)
            score = 0
            if compact_query and compact_query in compact_sentence:
                score += 3
            for tok in query_tokens:
                if len(tok) >= 2 and tok in sentence:
                    score += 1
            if sentence.endswith("."):
                score += 1
            scored.append((score, sentence))
            total += score

        if total > best_score:
            best_score = total
            best_row = row
            best_sentences = sorted(scored, key=lambda x: x[0], reverse=True)[:3]

    if best_row is None:
        return None, ""

    summary = " ".join(sentence for _, sentence in best_sentences).strip()
    summary = re.sub(r"\s+", " ", summary)
    return best_row, summary


def print_result(result: SearchResult) -> None:
    print(f"\n=== Question ===\n{result.original_query}")
    print(f"\n=== System Prompt ===\n{SYSTEM_PROMPT.strip()}")
    print(f"\n=== User Prompt ===\n{build_rag_user_prompt(result)}")
    print("\n=== Retrieved Top-K ===")
    for idx, row in enumerate(result.rows, start=1):
        print(f"[{idx}] {row['report_year']} | {row['major_section']} | {row['sub_section']}")
        print(
            f"    hybrid={row['hybrid_score']:.4f} semantic={row['semantic_score']:.4f} "
            f"keyword={row['keyword_score']:.4f}"
        )
        print(f"    {row['content'][:220]}\n")


def run_single(question: str, use_llm: bool = False, thinking: bool = False, debug: bool = False) -> None:
    query_mode = detect_query_mode(question)

    if query_mode == "generic":
        run_generic(question, debug=debug)
        return

    risk_types = detect_risk_types(question)
    if risk_types:
        selected_risk_type = risk_types[0]
        note_first_result = retrieve_note_first(question, risk_type=selected_risk_type)
        structured = build_structured_extraction(
            question,
            note_first_result,
            risk_type=selected_risk_type,
            max_rows=2,
        )
        print(json.dumps(structured, ensure_ascii=False, indent=2))
        return

    result = retrieve(question)
    if debug:
        print_result(result)

    if use_llm:
        answer, model_name, device = generate_answer(result, thinking=thinking)
    else:
        answer, model_name, device = generate_grounded_answer(result), "extractive", "n/a"

    print("=== Generated Answer ===")
    print(f"[model] {model_name} | [device] {device}")
    print(answer)


def run_test_suite(generate_answers: bool = False, use_llm: bool = False, thinking: bool = False) -> None:
    print("=== RAG QA Test Set (10 questions) ===")
    for idx, question in enumerate(TEST_QUESTIONS, start=1):
        result = retrieve(question)
        top = result.rows[0] if result.rows else None
        print(f"\n[{idx}] {question}")
        if top is None:
            print("- no result")
            continue
        print(f"- top_year   : {top['report_year']}")
        print(f"- top_major  : {top['major_section']}")
        print(f"- top_sub    : {top['sub_section']}")
        print(f"- top_score  : {top['hybrid_score']:.4f}")
        print(f"- top_snippet: {str(top['content'])[:180]}")
        if generate_answers:
            if use_llm:
                answer, model_name, device = generate_answer(result, thinking=thinking)
            else:
                answer, model_name, device = generate_grounded_answer(result), "extractive", "n/a"
            print(f"- llm_model  : {model_name} ({device})")
            print(f"- llm_answer : {answer[:260]}")


def main(argv: list[str] | None = None) -> None:
    raw_args = argv if argv is not None else sys.argv[1:]
    dash_variants = str.maketrans({"–": "-", "—": "-", "−": "-"})
    args = [a.translate(dash_variants).strip() for a in raw_args]
    if not args:
        raise SystemExit(
            "사용법: poetry run python -m app.qa '질문'\n"
            "   또는: poetry run python -m app.qa --test\n"
            "   또는: poetry run python -m app.qa --test --generate\n"
            "옵션: --llm (LLM 답변), --thinking (LLM 모드), --debug (디버그 출력)"
        )

    use_llm = "--llm" in args
    use_thinking = "--thinking" in args
    use_debug = "--debug" in args
    args = [a for a in args if a not in {"--llm", "--thinking", "--debug"}]

    if args == ["--test"]:
        run_test_suite(generate_answers=False, use_llm=use_llm, thinking=use_thinking)
        return

    if args == ["--test", "--generate"]:
        run_test_suite(generate_answers=True, use_llm=use_llm, thinking=use_thinking)
        return

    question = " ".join(args).strip()
    # 실수로 질문 텍스트에 옵션 토큰이 섞인 경우 제거
    question_tokens = [tok for tok in question.split() if tok not in {"--llm", "--thinking", "--debug"}]
    question = " ".join(question_tokens).strip()
    if not question:
        raise SystemExit("질문이 비어 있습니다.")
    run_single(question, use_llm=use_llm, thinking=use_thinking, debug=use_debug)


if __name__ == "__main__":
    main()