from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from app.chunker import build_chunks, build_structured_tables
from app.config import settings
from app.db import get_conn
from app.embedder import Embedder
from app.parser import parse_html_file, save_parsed_json


def upsert_document(cur, report) -> int:
    cur.execute(
        """
        INSERT INTO documents (file_name, title, report_year, source_path, raw_html, parsed_json)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (file_name) DO UPDATE SET
            title = EXCLUDED.title,
            report_year = EXCLUDED.report_year,
            source_path = EXCLUDED.source_path,
            raw_html = EXCLUDED.raw_html,
            parsed_json = EXCLUDED.parsed_json
        RETURNING id
        """,
        (
            report.file_name,
            report.title,
            report.report_year,
            str(report.file_path),
            report.raw_html,
            __import__('json').dumps(report.structured, ensure_ascii=False),
        ),
    )
    return cur.fetchone()["id"]


def replace_chunks(cur, document_id: int, report, chunks, embeddings, embedding_model: str) -> None:
    cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        cur.execute(
            """
            INSERT INTO chunks (
                id, document_id, chunk_key, chunk_index_global, chunk_index_in_section,
                company, report_year, major_section, sub_section, section_type,
                note_no, note_title, subtopic, section_id, subsection_path,
                as_of_date, period_start, period_end, evidence_type, risk_domain, table_meta, cell_unit,
                topic, content, char_count,
                embedding_model, embedding
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s::jsonb, %s,
                %s, %s, %s,
                %s, %s::vector
            )
            """,
            (
                chunk.id,
                document_id,
                chunk.chunk_key,
                chunk.chunk_index_global,
                chunk.chunk_index_in_section,
                report.structured["document_meta"]["company"],
                report.report_year,
                chunk.major_section,
                chunk.sub_section,
                chunk.section_type,
                chunk.note_no,
                chunk.note_title,
                getattr(chunk, "subtopic", None),
                getattr(chunk, "section_id", None),
                getattr(chunk, "subsection_path", None),
                getattr(chunk, "as_of_date", None),
                getattr(chunk, "period_start", None),
                getattr(chunk, "period_end", None),
                getattr(chunk, "evidence_type", None),
                getattr(chunk, "risk_domain", None),
                json.dumps(getattr(chunk, "table_meta", None), ensure_ascii=False)
                if getattr(chunk, "table_meta", None) is not None
                else None,
                getattr(chunk, "cell_unit", None),
                chunk.topic,
                chunk.content,
                chunk.char_count,
                embedding_model,
                embedding,
            ),
        )


def replace_structured_tables(cur, document_id: int, report, table_rows, cell_rows) -> None:
    cur.execute("DELETE FROM structured_table_cells WHERE document_id = %s", (document_id,))
    cur.execute("DELETE FROM structured_tables WHERE document_id = %s", (document_id,))

    for t in table_rows:
        cur.execute(
            """
            INSERT INTO structured_tables (
                table_id, document_id, report_year, note_no, note_title, subtopic,
                table_title, section_type, unit, risk_domain, source_chunk_key
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                t.get("table_id"),
                document_id,
                t.get("report_year"),
                t.get("note_no"),
                t.get("note_title"),
                t.get("subtopic"),
                t.get("table_title"),
                t.get("section_type"),
                t.get("unit"),
                t.get("risk_domain"),
                t.get("source_chunk_key"),
            ),
        )

    for c in cell_rows:
        cur.execute(
            """
            INSERT INTO structured_table_cells (
                table_id, report_year, document_id, note_no, note_title, subtopic,
                table_title, section_type, unit, source_chunk_key,
                row_index, col_index, row_group, row_label, col_label,
                value_raw, value_numeric, value_type, currency, as_of_date, period_type, risk_domain
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                c.get("table_id"),
                c.get("report_year"),
                document_id,
                c.get("note_no"),
                c.get("note_title"),
                c.get("subtopic"),
                c.get("table_title"),
                c.get("section_type"),
                c.get("unit"),
                c.get("source_chunk_key"),
                c.get("row_index"),
                c.get("col_index"),
                c.get("row_group"),
                c.get("row_label"),
                c.get("col_label"),
                c.get("value_raw"),
                c.get("value_numeric"),
                c.get("value_type"),
                c.get("currency"),
                c.get("as_of_date"),
                c.get("period_type"),
                c.get("risk_domain"),
            ),
        )


def ingest_one(html_path: Path, embedder: Embedder) -> None:
    configured_model = (settings.embedding_model or "").strip()
    actual_model = (embedder.model_name or "").strip()
    if configured_model and actual_model and configured_model != actual_model:
        raise ValueError(
            "EMBEDDING_MODEL mismatch: "
            f"settings={configured_model!r}, embedder={actual_model!r}. "
            "환경 설정과 실제 임베딩 모델을 일치시켜 주세요."
        )

    report = parse_html_file(html_path)
    save_parsed_json(report, settings.parsed_dir)
    chunks = build_chunks(report)
    table_rows, cell_rows = build_structured_tables(report)
    if not chunks:
        print(f"[SKIP] {html_path.name}: usable chunk가 없습니다.")
        return

    embeddings = embedder.encode_texts([chunk.content for chunk in chunks])

    with get_conn() as conn:
        with conn.cursor() as cur:
            document_id = upsert_document(cur, report)
            replace_chunks(cur, document_id, report, chunks, embeddings, embedder.model_name)
            replace_structured_tables(cur, document_id, report, table_rows, cell_rows)

    print(
        f"[DONE] {html_path.name} | parser={report.parser_used} | year={report.report_year} | "
        f"chunks={len(chunks)} | tables={len(table_rows)} | cells={len(cell_rows)}"
    )


def main() -> None:
    html_dir = Path(settings.html_dir)
    html_files = sorted(list(html_dir.glob("*.htm")) + list(html_dir.glob("*.html")))
    if not html_files:
        raise FileNotFoundError(f"HTML 파일이 없습니다: {html_dir}")

    embedder = Embedder()
    for html_path in tqdm(html_files, desc="ingest"):
        ingest_one(html_path, embedder)


if __name__ == "__main__":
    main()
