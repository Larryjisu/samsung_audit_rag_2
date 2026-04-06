from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from app.config import settings


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(settings.db_url, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
