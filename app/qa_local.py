from __future__ import annotations

import sys
from typing import Any

from app.search_local import build_structured_extraction, retrieve


def answer_local(query: str, *, max_rows: int = 3) -> dict[str, Any]:
	try:
		result = retrieve(query)
		structured = build_structured_extraction(query, result, max_rows=max_rows)

		rows = result.rows or []
		top = rows[0] if rows else {}

		amount = str(structured.get("amount") or "").strip()
		evidence_sentence = str(structured.get("evidence_sentence") or "").strip()
		final_answer = amount or evidence_sentence or "근거 부족"

		note_title = str(structured.get("note_title") or top.get("note_title") or "")
		subtopic = str(structured.get("subtopic") or top.get("subtopic") or "")
		confidence_rule = str(structured.get("confidence_rule") or top.get("confidence_rule") or "")

		return {
			"ok": True,
			"query": query,
			"answer": final_answer,
			"amount": amount,
			"note_title": note_title,
			"subtopic": subtopic,
			"evidence_sentence": evidence_sentence,
			"confidence_rule": confidence_rule,
			"found": bool(structured.get("found", False)),
			"row_count": len(rows),
		}
	except Exception as exc:
		return {
			"ok": False,
			"query": query,
			"answer": "답변 생성 실패",
			"error": str(exc),
			"note_title": "",
			"subtopic": "",
			"evidence_sentence": "",
			"confidence_rule": "",
			"found": False,
			"row_count": 0,
		}


def format_local_answer(payload: dict[str, Any]) -> str:
	query = str(payload.get("query") or "")
	answer = str(payload.get("answer") or "근거 부족")
	note_title = str(payload.get("note_title") or "-")
	subtopic = str(payload.get("subtopic") or "-")
	evidence = str(payload.get("evidence_sentence") or "-")
	confidence = str(payload.get("confidence_rule") or "-")

	if not payload.get("ok", False):
		err = str(payload.get("error") or "원인 미상")
		return "\n".join(
			[
				f"질문: {query}",
				"답변: 답변 생성 실패",
				f"메타: row_count={payload.get('row_count', 0)}, found={payload.get('found', False)}",
				f"오류: {err[:160]}",
			]
		)

	return "\n".join(
		[
			f"질문: {query}",
			f"답변: {answer}",
			f"주석: {note_title}",
			f"세부항목: {subtopic}",
			f"근거: {evidence}",
			f"판단근거: {confidence}",
		]
	)


def main(argv: list[str] | None = None) -> None:
	args = argv if argv is not None else sys.argv[1:]
	if not args:
		raise SystemExit("사용법: poetry run python -m app.qa_local '질문'")

	query = " ".join(args).strip()
	if not query:
		raise SystemExit("질문이 비어 있습니다.")

	payload = answer_local(query)
	print(format_local_answer(payload))


if __name__ == "__main__":
	main()
