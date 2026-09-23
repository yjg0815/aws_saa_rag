# psycopg 기반 book_chunks 데이터 접근 계층 (pgvector 사용).
#
# 커넥션 생성 자체(POSTGRES_* env 읽기 + pgvector 타입 등록)는 rag_common.db에
# 이미 있어서 그대로 재사용한다 — 여기서는 book_chunks 스키마에 맞는 쿼리
# 함수만 정의한다. get_connection을 다시 export해서 다른 모듈이 notion_rag_test와
# 똑같이 "from storage.db import get_connection"으로 쓸 수 있게 한다.

import re
from pathlib import Path

import psycopg
from pgvector.psycopg import Vector
from rag_common.db import get_connection  # noqa: F401 (re-export)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_schema(conn: psycopg.Connection) -> None:
    """storage/schema.sql 전체를 실행한다 (CREATE EXTENSION / CREATE TABLE IF NOT EXISTS / INDEX 등).

    문장들이 전부 IF NOT EXISTS / ADD COLUMN IF NOT EXISTS 형태라 멱등적이다 —
    이미 만들어진 DB에 다시 실행해도 안전하다.
    """
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()


def insert_chunks(conn: psycopg.Connection, chunks: list[dict]) -> None:
    """청크들을 book_chunks에 삽입한다.

    Args:
        chunks: {"page_number", "chapter"(nullable), "chunk_text", "embedding"} dict의 리스트.

    정적 문서(PDF 한 번 수집)라 notion_rag_test의 discover_pages 같은 증분
    동기화/캐싱이 필요 없다 — scripts/ingest_book.py가 책 전체를 한 번만
    처리한다고 전제한다. 재수집하고 싶으면 호출하는 쪽에서 TRUNCATE
    book_chunks를 직접 실행하면 된다(이 함수는 delete-then-insert를 하지 않는다).
    """
    if not chunks:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO book_chunks (page_number, chapter, content, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            [
                (c["page_number"], c.get("chapter"), c["chunk_text"], Vector(c["embedding"]))
                for c in chunks
            ],
        )
    conn.commit()


def search_similar(conn: psycopg.Connection, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """query_embedding과 코사인 거리가 가장 가까운 top_k개의 청크를 반환한다.

    Returns:
        {"chunk_id", "page_number", "chapter", "chunk_text", "distance"} dict의 리스트.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, page_number, chapter, content, embedding <=> %s AS distance
            FROM book_chunks
            ORDER BY distance ASC
            LIMIT %s
            """,
            (Vector(query_embedding), top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "chunk_id": chunk_id,
            "page_number": page_number,
            "chapter": chapter,
            "chunk_text": content,
            "distance": distance,
        }
        for chunk_id, page_number, chapter, content, distance in rows
    ]


def search_bm25(conn: psycopg.Connection, query_text: str, top_k: int = 5) -> list[dict]:
    """query_text와 book_chunks.content_tsv를 Postgres 내장 full-text search로 매칭해 top_k개를 반환한다.

    OR 기반 토큰 매칭이다 — notion_rag_test/storage/db.py의 search_bm25에서
    실측으로 검증한 패턴을 그대로 쓴다: 쿼리를 공백 기준으로 토큰화해서 각
    토큰의 plainto_tsquery를 OR(||)로 묶는다. plainto_tsquery에 문장을
    통째로 넣는(AND) 방식은 자연어 질문 전체가 원문에 그대로 없으면 0건
    매칭되는 문제가 실제로 있었다. 토큰마다 regexp_replace(token, '[./]', ' ')도
    적용한다 — schema.sql의 content_tsv 주석과 동일한 이유(점/슬래시로 이어진
    문자열이 하나의 토큰으로 묶이는 Postgres 기본 파서 동작 때문).

    Returns:
        {"chunk_id", "page_number", "chapter", "chunk_text", "bm25_score"} dict의 리스트.
        매칭되는 토큰이 하나도 없으면 빈 리스트를 반환한다.
    """
    tokens = [t for t in re.sub(r"[./]", " ", query_text).split() if t]
    if not tokens:
        return []

    or_tsquery_sql = " || ".join(["plainto_tsquery('simple', %s)"] * len(tokens))

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, page_number, chapter, content,
                   ts_rank(content_tsv, {or_tsquery_sql}) AS bm25_score
            FROM book_chunks
            WHERE content_tsv @@ ({or_tsquery_sql})
            ORDER BY bm25_score DESC
            LIMIT %s
            """,
            (*tokens, *tokens, top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "chunk_id": chunk_id,
            "page_number": page_number,
            "chapter": chapter,
            "chunk_text": content,
            "bm25_score": bm25_score,
        }
        for chunk_id, page_number, chapter, content, bm25_score in rows
    ]
