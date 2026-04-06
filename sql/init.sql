CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    title TEXT,
    report_year INT,
    source_path TEXT NOT NULL,
    raw_html TEXT,
    parsed_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_key TEXT NOT NULL UNIQUE,
    chunk_index_global INT NOT NULL,
    chunk_index_in_section INT NOT NULL,
    company TEXT NOT NULL,
    report_year INT,
    major_section TEXT,
    sub_section TEXT,
    section_type TEXT NOT NULL,
    note_no INT,
    note_title TEXT,
    subtopic TEXT,
    topic TEXT,
    section_id TEXT,
    subsection_path TEXT,
    as_of_date DATE,
    period_start DATE,
    period_end DATE,
    evidence_type TEXT,
    risk_domain TEXT,
    table_meta JSONB,
    cell_unit TEXT,
    content TEXT NOT NULL,
    char_count INT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS subsection_path TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS subtopic TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS as_of_date DATE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS period_start DATE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS period_end DATE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS evidence_type TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS risk_domain TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_meta JSONB;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS cell_unit TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_report_year ON documents(report_year);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_report_year ON chunks(report_year);
CREATE INDEX IF NOT EXISTS idx_chunks_sub_section ON chunks(sub_section);
CREATE INDEX IF NOT EXISTS idx_chunks_note_no ON chunks(note_no);
CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic);
CREATE INDEX IF NOT EXISTS idx_chunks_subtopic ON chunks(subtopic);
CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_subsection_path ON chunks(subsection_path);
CREATE INDEX IF NOT EXISTS idx_chunks_evidence_type ON chunks(evidence_type);
CREATE INDEX IF NOT EXISTS idx_chunks_risk_domain ON chunks(risk_domain);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS structured_tables (
    id BIGSERIAL PRIMARY KEY,
    table_id TEXT NOT NULL UNIQUE,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    report_year INT,
    note_no INT,
    note_title TEXT,
    subtopic TEXT,
    table_title TEXT,
    section_type TEXT,
    unit TEXT,
    risk_domain TEXT,
    source_chunk_key TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS structured_table_cells (
    id BIGSERIAL PRIMARY KEY,
    table_id TEXT NOT NULL REFERENCES structured_tables(table_id) ON DELETE CASCADE,
    report_year INT,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    note_no INT,
    note_title TEXT,
    subtopic TEXT,
    table_title TEXT,
    section_type TEXT,
    unit TEXT,
    source_chunk_key TEXT,
    row_index INT NOT NULL,
    col_index INT NOT NULL,
    row_group TEXT,
    row_label TEXT,
    col_label TEXT,
    value_raw TEXT,
    value_numeric NUMERIC,
    value_type TEXT,
    currency TEXT,
    as_of_date DATE,
    period_type TEXT,
    risk_domain TEXT,
    table_family TEXT GENERATED ALWAYS AS (
        CASE
            WHEN POSITION('현금및현금성자산' IN lower(replace(replace(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''), ' ', ''), E'\n', ''))) > 0 THEN 'cash'
            WHEN POSITION('충당부채' IN lower(replace(replace(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''), ' ', ''), E'\n', ''))) > 0 THEN 'provision'
            WHEN POSITION('지급보증' IN lower(replace(replace(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''), ' ', ''), E'\n', ''))) > 0 THEN 'guarantee'
            WHEN POSITION('차입금' IN lower(replace(replace(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''), ' ', ''), E'\n', ''))) > 0
                 OR POSITION('리스부채' IN lower(replace(replace(COALESCE(row_group, '') || COALESCE(row_label, ''), ' ', ''), E'\n', ''))) > 0 THEN 'loan_lease'
            ELSE 'other'
        END
    ) STORED,
    table_title_norm TEXT GENERATED ALWAYS AS (
        lower(replace(replace(replace(replace(replace(COALESCE(table_title, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
    ) STORED,
    note_title_norm TEXT GENERATED ALWAYS AS (
        lower(replace(replace(replace(replace(replace(COALESCE(note_title, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
    ) STORED,
    subtopic_norm TEXT GENERATED ALWAYS AS (
        lower(replace(replace(replace(replace(replace(COALESCE(subtopic, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
    ) STORED,
    row_label_norm TEXT GENERATED ALWAYS AS (
        lower(replace(replace(replace(replace(replace(COALESCE(row_label, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
    ) STORED,
    col_label_norm TEXT GENERATED ALWAYS AS (
        lower(replace(replace(replace(replace(replace(COALESCE(col_label, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
    ) STORED,
    row_group_norm TEXT GENERATED ALWAYS AS (
        lower(replace(replace(replace(replace(replace(COALESCE(row_group, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
    ) STORED,
    row_role TEXT GENERATED ALWAYS AS (
        CASE
            WHEN POSITION('기초' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'opening'
            WHEN POSITION('기말' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'ending'
            WHEN POSITION('순전입' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'inflow'
            WHEN POSITION('환입' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'inflow'
            WHEN POSITION('사용액' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'outflow'
            WHEN POSITION('기타' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'other'
            WHEN POSITION('계' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
                 OR POSITION('합계' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
                 OR POSITION('총액' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'aggregate'
            ELSE NULL
        END
    ) STORED,
    period_role TEXT GENERATED ALWAYS AS (
        CASE
            WHEN POSITION('당기말' IN COALESCE(col_label, '') || COALESCE(period_type, '')) > 0 THEN 'current_end'
            WHEN POSITION('전기말' IN COALESCE(col_label, '') || COALESCE(period_type, '')) > 0 THEN 'prior_end'
            WHEN POSITION('당기' IN COALESCE(col_label, '') || COALESCE(period_type, '')) > 0 THEN 'current'
            WHEN POSITION('전기' IN COALESCE(col_label, '') || COALESCE(period_type, '')) > 0 THEN 'prior'
            ELSE NULL
        END
    ) STORED,
    is_aggregate BOOLEAN GENERATED ALWAYS AS (
        POSITION('계' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
        OR POSITION('합계' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
        OR POSITION('총액' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
    ) STORED,
    row_year INT GENERATED ALWAYS AS (
        CASE
            WHEN substring(COALESCE(row_label, '') FROM '((?:19|20)[0-9]{2})') IS NOT NULL THEN substring(COALESCE(row_label, '') FROM '((?:19|20)[0-9]{2})')::INT
            WHEN substring(COALESCE(row_group, '') FROM '((?:19|20)[0-9]{2})') IS NOT NULL THEN substring(COALESCE(row_group, '') FROM '((?:19|20)[0-9]{2})')::INT
            ELSE NULL
        END
    ) STORED,
    entity_label TEXT GENERATED ALWAYS AS (
        CASE
            WHEN (POSITION('지급보증' IN lower(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''))) > 0)
                 AND NOT (
                    POSITION('계' IN COALESCE(row_label, '')) > 0
                    OR POSITION('합계' IN COALESCE(row_label, '')) > 0
                    OR POSITION('총액' IN COALESCE(row_label, '')) > 0
                 )
            THEN COALESCE(row_label, '')
            ELSE NULL
        END
    ) STORED,
    parse_confidence DOUBLE PRECISION GENERATED ALWAYS AS (
        CASE
            WHEN value_type = 'amount' AND COALESCE(row_label, '') <> '' AND COALESCE(col_label, '') <> '' THEN 0.95
            WHEN value_type = 'amount' THEN 0.85
            ELSE 0.70
        END
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS table_family TEXT GENERATED ALWAYS AS (
    CASE
        WHEN POSITION('현금및현금성자산' IN lower(replace(replace(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''), ' ', ''), E'\n', ''))) > 0 THEN 'cash'
        WHEN POSITION('충당부채' IN lower(replace(replace(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''), ' ', ''), E'\n', ''))) > 0 THEN 'provision'
        WHEN POSITION('지급보증' IN lower(replace(replace(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''), ' ', ''), E'\n', ''))) > 0 THEN 'guarantee'
        WHEN POSITION('차입금' IN lower(replace(replace(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''), ' ', ''), E'\n', ''))) > 0
             OR POSITION('리스부채' IN lower(replace(replace(COALESCE(row_group, '') || COALESCE(row_label, ''), ' ', ''), E'\n', ''))) > 0 THEN 'loan_lease'
        ELSE 'other'
    END
) STORED;
ALTER TABLE structured_tables ADD COLUMN IF NOT EXISTS risk_domain TEXT;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS risk_domain TEXT;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS table_title_norm TEXT GENERATED ALWAYS AS (
    lower(replace(replace(replace(replace(replace(COALESCE(table_title, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS note_title_norm TEXT GENERATED ALWAYS AS (
    lower(replace(replace(replace(replace(replace(COALESCE(note_title, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS subtopic_norm TEXT GENERATED ALWAYS AS (
    lower(replace(replace(replace(replace(replace(COALESCE(subtopic, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS row_label_norm TEXT GENERATED ALWAYS AS (
    lower(replace(replace(replace(replace(replace(COALESCE(row_label, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS col_label_norm TEXT GENERATED ALWAYS AS (
    lower(replace(replace(replace(replace(replace(COALESCE(col_label, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS row_group_norm TEXT GENERATED ALWAYS AS (
    lower(replace(replace(replace(replace(replace(COALESCE(row_group, ''), ' ', ''), E'\n', ''), E'\t', ''), '(', ''), ')', ''))
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS row_role TEXT GENERATED ALWAYS AS (
    CASE
        WHEN POSITION('기초' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'opening'
        WHEN POSITION('기말' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'ending'
        WHEN POSITION('순전입' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'inflow'
        WHEN POSITION('환입' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'inflow'
        WHEN POSITION('사용액' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'outflow'
        WHEN POSITION('기타' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'other'
        WHEN POSITION('계' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
             OR POSITION('합계' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
             OR POSITION('총액' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0 THEN 'aggregate'
        ELSE NULL
    END
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS period_role TEXT GENERATED ALWAYS AS (
    CASE
        WHEN POSITION('당기말' IN COALESCE(col_label, '') || COALESCE(period_type, '')) > 0 THEN 'current_end'
        WHEN POSITION('전기말' IN COALESCE(col_label, '') || COALESCE(period_type, '')) > 0 THEN 'prior_end'
        WHEN POSITION('당기' IN COALESCE(col_label, '') || COALESCE(period_type, '')) > 0 THEN 'current'
        WHEN POSITION('전기' IN COALESCE(col_label, '') || COALESCE(period_type, '')) > 0 THEN 'prior'
        ELSE NULL
    END
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS is_aggregate BOOLEAN GENERATED ALWAYS AS (
    POSITION('계' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
    OR POSITION('합계' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
    OR POSITION('총액' IN COALESCE(row_label, '') || COALESCE(row_group, '')) > 0
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS row_year INT GENERATED ALWAYS AS (
    CASE
        WHEN substring(COALESCE(row_label, '') FROM '((?:19|20)[0-9]{2})') IS NOT NULL THEN substring(COALESCE(row_label, '') FROM '((?:19|20)[0-9]{2})')::INT
        WHEN substring(COALESCE(row_group, '') FROM '((?:19|20)[0-9]{2})') IS NOT NULL THEN substring(COALESCE(row_group, '') FROM '((?:19|20)[0-9]{2})')::INT
        ELSE NULL
    END
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS entity_label TEXT GENERATED ALWAYS AS (
    CASE
        WHEN (POSITION('지급보증' IN lower(COALESCE(note_title, '') || COALESCE(subtopic, '') || COALESCE(table_title, ''))) > 0)
             AND NOT (
                POSITION('계' IN COALESCE(row_label, '')) > 0
                OR POSITION('합계' IN COALESCE(row_label, '')) > 0
                OR POSITION('총액' IN COALESCE(row_label, '')) > 0
             )
        THEN COALESCE(row_label, '')
        ELSE NULL
    END
) STORED;
ALTER TABLE structured_table_cells ADD COLUMN IF NOT EXISTS parse_confidence DOUBLE PRECISION GENERATED ALWAYS AS (
    CASE
        WHEN value_type = 'amount' AND COALESCE(row_label, '') <> '' AND COALESCE(col_label, '') <> '' THEN 0.95
        WHEN value_type = 'amount' THEN 0.85
        ELSE 0.70
    END
) STORED;

CREATE INDEX IF NOT EXISTS idx_structured_tables_doc ON structured_tables(document_id);
CREATE INDEX IF NOT EXISTS idx_structured_tables_year_note ON structured_tables(report_year, note_no);
CREATE INDEX IF NOT EXISTS idx_structured_tables_note_title ON structured_tables(note_title);
CREATE INDEX IF NOT EXISTS idx_structured_tables_risk_domain ON structured_tables(risk_domain);

CREATE INDEX IF NOT EXISTS idx_structured_cells_table_id ON structured_table_cells(table_id);
CREATE INDEX IF NOT EXISTS idx_structured_cells_lookup ON structured_table_cells(report_year, note_no, row_label, col_label);
CREATE INDEX IF NOT EXISTS idx_structured_cells_value_type ON structured_table_cells(value_type);
CREATE INDEX IF NOT EXISTS idx_structured_cells_risk_domain ON structured_table_cells(risk_domain);
CREATE INDEX IF NOT EXISTS idx_structured_cells_table_family ON structured_table_cells(table_family);
CREATE INDEX IF NOT EXISTS idx_structured_cells_norm_labels ON structured_table_cells(report_year, row_label_norm, col_label_norm);
CREATE INDEX IF NOT EXISTS idx_structured_cells_roles ON structured_table_cells(row_role, period_role, is_aggregate);
