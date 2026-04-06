from __future__ import annotations

import re
import sys

from app.qa_hybrid import answer_hybrid
from app.qa_local import answer_local, format_local_answer


FLOW_HINTS = ("흐름", "추이", "추세", "변화", "연도별")
RISK_HINTS = ("리스크", "소송", "담보", "보증", "우발", "약정", "충당")
NUMERIC_HINTS = ("얼마", "금액", "총액", "잔액", "몇", "내역", "현금", "리스부채", "관련차입금", "채무보증한도")
EXPLAIN_HINTS = ("설명", "비교", "왜", "어떻게", "집중", "근거")
EVAL_HINTS = ("평가", "지표", "recall", "precision", "ndcg", "accuracy", "faithfulness", "groundedness")


def get_retriever_metrics_guide() -> str:
    return "\n".join(
        [
            "[Retriever 평가 지표]",
            "- Recall@k: top-k 안에 정답 근거가 포함되는 비율 (놓치지 않는 능력)",
            "- Precision@k: top-k 중 실제 관련 근거 비율 (노이즈 억제 능력)",
            "- nDCG@k: 관련 근거를 상위에 배치하는 랭킹 품질",
        ]
    )


def get_rag_metrics_guide() -> str:
    return "\n".join(
        [
            "[RAG 전체 평가 지표]",
            "- Correctness/Accuracy: 정답성과 사실 일치",
            "- Relevance: 질문 의도 적합성",
            "- Faithfulness/Groundedness: 검색 근거 기반 답변 여부(환각 억제)",
        ]
    )


def get_retriever_eval_requirements() -> str:
    return "\n".join(
        [
            "[Retriever 평가 준비물]",
            "- 평가용 질문셋",
            "- 질문별 정답 근거(문서ID/섹션/표/문단/청크)",
            "- 모델의 top-k 검색 로그(점수/순위 포함 권장)",
            "- 핵심 조합: 질문 + 정답 근거 + 검색 결과",
        ]
    )


def get_rag_eval_requirements() -> str:
    return "\n".join(
        [
            "[RAG 평가 준비물]",
            "- 평가용 질문셋",
            "- 질문별 정답 답변",
            "- 모델 생성 답변",
            "- 실제 사용된 retrieved context",
            "- 핵심 조합: 질문 + 정답 답변 + 생성 답변 + 사용 근거",
        ]
    )


def is_eval_metrics_question(query: str) -> bool:
    compact = _compact(query).lower()
    return any(h in compact for h in EVAL_HINTS)


def answer_eval_metrics(query: str) -> str:
    compact = _compact(query).lower()

    wants_retriever = any(k in compact for k in ("retriever", "리트리버", "검색기", "recall", "precision", "ndcg"))
    wants_rag = any(k in compact for k in ("rag", "정답성", "accuracy", "correctness", "faithfulness", "groundedness"))
    wants_prepare = any(k in compact for k in ("준비물", "필요", "데이터셋", "정답근거", "로그", "context"))

    blocks: list[str] = ["[AUTO ROUTE: EVAL-METRICS]"]

    if wants_retriever or (not wants_retriever and not wants_rag):
        blocks.append(get_retriever_metrics_guide())
    if wants_rag or (not wants_retriever and not wants_rag):
        blocks.append(get_rag_metrics_guide())

    if wants_prepare:
        blocks.append(get_retriever_eval_requirements())
        blocks.append(get_rag_eval_requirements())

    return "\n\n".join(blocks)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _has_year_range_expression(query: str) -> bool:
    text = query or ""
    patterns = (
        r"((?:19|20)\d{2})\s*[~\-]\s*((?:19|20)\d{2})",
        r"((?:19|20)\d{2})\s*년\s*부터\s*((?:19|20)\d{2})\s*년\s*까지",
    )
    for pat in patterns:
        m = re.search(pat, text)
        if m and m.group(1) != m.group(2):
            return True
    return False


def _is_strong_numeric_query(compact_query: str) -> bool:
    """Strong numeric cue should override risk keywords and route to LOCAL."""
    return any(h in compact_query for h in NUMERIC_HINTS)


def classify_query_route(query: str) -> str:
    """Return local or hybrid by retrieval-range need.

    Priority:
    1) period_flow -> hybrid
    2) strong_numeric -> local
    3) risk_probe -> hybrid
    4) explanatory_compare -> hybrid
    5) fallback -> hybrid
    """
    compact = _compact(query)

    # 1) period_flow
    if _has_year_range_expression(query) or any(h in compact for h in FLOW_HINTS):
        return "hybrid"

    # 2) strong_numeric (risk보다 우선)
    if _is_strong_numeric_query(compact):
        return "local"

    # 3) risk_probe
    if any(h in compact for h in RISK_HINTS):
        return "hybrid"

    # 4) explanatory_compare
    if any(h in compact for h in EXPLAIN_HINTS):
        return "hybrid"

    # 5) fallback
    return "hybrid"


def answer_auto(query: str) -> str:
    if is_eval_metrics_question(query):
        return answer_eval_metrics(query)

    route = classify_query_route(query)
    if route == "local":
        payload = answer_local(query)
        body = format_local_answer(payload)
        return f"[AUTO ROUTE: LOCAL]\n{body}"

    body = answer_hybrid(query)
    return f"[AUTO ROUTE: HYBRID]\n질문: {query}\n{body}"


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit("사용법: poetry run python -m app.qa_auto '질문'")

    query = " ".join(args).strip()
    if not query:
        raise SystemExit("질문이 비어 있습니다.")

    print(answer_auto(query))


if __name__ == "__main__":
    main()
