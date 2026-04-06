from __future__ import annotations

import sys
from typing import Any

from app.qa import (
    TEST_QUESTIONS,
    generate_answer,
    generate_grounded_answer,
)
from app.search_hybrid import retrieve_hybrid


def _year_span_text(rows: list[dict[str, Any]]) -> str:
    years = sorted({int(y) for y in [row.get("report_year") for row in rows] if isinstance(y, int)})
    if not years:
        return "-"
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}~{years[-1]}"


def _evidence_bundle(rows: list[dict[str, Any]], max_evidence: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        note = str(row.get("note_title") or "-").strip()
        subtopic = str(row.get("subtopic") or "-").strip()
        year = str(row.get("report_year") or "-").strip()
        key = f"{year}|{note}|{subtopic}"
        if key in seen:
            continue
        seen.add(key)
        out.append(f"- {year} | {note} | {subtopic}")
        if len(out) >= max_evidence:
            break
    return out


def format_hybrid_answer(
    query: str,
    *,
    summary: str,
    rows: list[dict[str, Any]],
    mode: str,
    max_evidence: int = 3,
) -> str:
    year_span = _year_span_text(rows)
    evidence_lines = _evidence_bundle(rows, max_evidence=max_evidence)
    if not evidence_lines:
        evidence_lines = ["- 근거 부족"]

    return "\n".join(
        [
            f"질문: {query}",
            f"요약: {summary}",
            f"연도범위: {year_span}",
            f"검색모드: {mode}",
            "대표근거:",
            *evidence_lines,
        ]
    )


def answer_hybrid(query: str, *, mode: str = "mixed", max_evidence: int = 3) -> str:
    """Presentation-friendly hybrid answer string for routing layer use."""
    try:
        result = retrieve_hybrid(query, mode=mode)
        if not result.rows:
            return "\n".join(
                [
                    f"질문: {query}",
                    "요약: 근거 부족 (검색 결과 없음)",
                    "연도범위: -",
                    f"검색모드: {mode}",
                    "대표근거:",
                    "- 근거 부족",
                ]
            )

        summary = generate_grounded_answer(result)
        return format_hybrid_answer(
            query,
            summary=summary,
            rows=result.rows,
            mode=mode,
            max_evidence=max_evidence,
        )
    except Exception as exc:
        return "\n".join(
            [
                f"질문: {query}",
                "요약: 답변 생성 실패",
                "연도범위: -",
                f"검색모드: {mode}",
                "대표근거:",
                "- 근거 부족",
                f"오류: {str(exc)[:160]}",
            ]
        )


def run_single(question: str, use_llm: bool = False, thinking: bool = False) -> None:
    result = retrieve_hybrid(question, mode="mixed")

    if not result.rows:
        print(answer_hybrid(question, mode="mixed"))
        return

    if use_llm:
        summary, model_name, device = generate_answer(result, thinking=thinking)
    else:
        summary, model_name, device = generate_grounded_answer(result), "extractive", "n/a"

    formatted = format_hybrid_answer(
        question,
        summary=summary,
        rows=result.rows,
        mode="mixed",
        max_evidence=3,
    )

    print("=== Generated Answer (Hybrid) ===")
    print(f"[model] {model_name} | [device] {device}")
    print(formatted)


def run_test_suite(generate_answers: bool = False, use_llm: bool = False, thinking: bool = False) -> None:
    print("=== RAG QA Hybrid Test Set (10 questions) ===")
    for idx, question in enumerate(TEST_QUESTIONS, start=1):
        result = retrieve_hybrid(question, mode="mixed")
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
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit(
            "사용법: poetry run python -m app.qa_hybrid '질문'\n"
            "   또는: poetry run python -m app.qa_hybrid --test\n"
            "   또는: poetry run python -m app.qa_hybrid --test --generate\n"
            "옵션: --llm (기본은 데이터 기반 추출 답변), --thinking (LLM 모드에서만 유효)"
        )

    use_llm = "--llm" in args
    use_thinking = "--thinking" in args
    args = [a for a in args if a not in {"--llm", "--thinking"}]

    if args == ["--test"]:
        run_test_suite(generate_answers=False, use_llm=use_llm, thinking=use_thinking)
        return

    if args == ["--test", "--generate"]:
        run_test_suite(generate_answers=True, use_llm=use_llm, thinking=use_thinking)
        return

    question = " ".join(args).strip()
    if not question:
        raise SystemExit("질문이 비어 있습니다.")
    run_single(question, use_llm=use_llm, thinking=use_thinking)


if __name__ == "__main__":
    main()
