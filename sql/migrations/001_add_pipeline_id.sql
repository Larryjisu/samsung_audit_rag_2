-- 001_add_pipeline_id.sql
-- 목적: 단일 DB 내 local/hybrid 파이프라인 논리 격리

ALTER TABLE IF EXISTS documents
    ADD COLUMN IF NOT EXISTS pipeline_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE IF EXISTS chunks
    ADD COLUMN IF NOT EXISTS pipeline_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE IF EXISTS structured_tables
    ADD COLUMN IF NOT EXISTS pipeline_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE IF EXISTS structured_table_cells
    ADD COLUMN IF NOT EXISTS pipeline_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_documents_pipeline_id ON documents(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_chunks_pipeline_id ON chunks(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_structured_tables_pipeline_id ON structured_tables(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_structured_table_cells_pipeline_id ON structured_table_cells(pipeline_id);

-- 기존 유니크 제약을 유지하면서 파이프라인 단위 유니크를 추가로 보장
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_file_name_pipeline_id
    ON documents(file_name, pipeline_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_chunk_key_pipeline_id
    ON chunks(chunk_key, pipeline_id);
