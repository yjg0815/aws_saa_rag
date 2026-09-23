-- Schema for AWS SAA RAG (book_chunks만 있는 단일 테이블 구조).
-- 정적 문서(PDF 한 번 수집) 프로젝트라 notion_rag_test의 pages/child_ids
-- 캐싱 구조 같은 증분 동기화 지원은 두지 않는다.

CREATE EXTENSION IF NOT EXISTS vector;

-- NOTE: vector(1024)는 voyage-4-lite 기준이다.
CREATE TABLE IF NOT EXISTS book_chunks (
    id            BIGSERIAL PRIMARY KEY,
    page_number   INT NOT NULL,
    chapter       TEXT,
    content       TEXT NOT NULL,
    embedding     vector(1024) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cosine-distance ANN index for top-k retrieval.
CREATE INDEX IF NOT EXISTS book_chunks_embedding_idx
    ON book_chunks USING hnsw (embedding vector_cosine_ops);

-- BM25(키워드) 검색용 tsvector. notion_rag_test에 hybrid retrieval을 붙일 때
-- 검증한 것과 동일한 패턴을 쓴다: regexp_replace로 '.'과 '/'를 공백으로
-- 바꾼 뒤 토큰화한다. (참고: 원래 계획했던 스니펫은 regexp_replace 없이
-- to_tsvector('simple', content)만 쓰는 형태였는데, notion_rag_test 작업
-- 중 Postgres 기본 파서가 점/슬래시로 이어진 문자열을 "host"/"file" 토큰
-- 타입으로 인식해서 하나의 lexeme으로 묶어버리는 문제를 실제로 발견해서
-- 고쳤다 — 치환 없이는 "10.0.0.0"만 검색해도 "10.0.0.0/16" 같은 CIDR
-- 표기가 든 문서와 매칭되지 않는다. AWS 챕터(특히 VPC)는 CIDR 표기,
-- 서비스 엔드포인트 도메인, ARN 등 점/슬래시가 섞인 정확한 용어가 아주
-- 많아서 이 프로젝트는 처음부터 검증된(고쳐진) 패턴으로 시작한다.)
ALTER TABLE book_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', regexp_replace(content, '[./]', ' ', 'g'))) STORED;

CREATE INDEX IF NOT EXISTS book_chunks_content_tsv_idx
    ON book_chunks USING GIN (content_tsv);
