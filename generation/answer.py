# CLI 진입점 + 복붙용 프롬프트 조립 함수.
#
# build_context/build_prompt는 notion_rag_test/generation/answer.py의 패턴을
# 참고했지만, book_chunks 필드(page_title/page_url/heading_path 대신
# page_number/chapter)에 맞게 새로 작성했다 — Notion 쪽처럼 page_url이 없어서
# 출처 표기를 "{chapter} p.{page_number}"로 대체했다. Anthropic API 호출은
# 여기 없다 — 검색+조립까지만 하고, 조립된 텍스트를 사람이 직접 Claude
# 채팅에 붙여넣어서 답변을 받는 흐름이다(streamlit_app.py의 st.code() 블록이
# 이 함수의 결과를 그대로 보여준다).
#
# 사용법:
#     python -m generation.answer "질문" [--top-k 5]

import argparse
import os
import sys

from dotenv import load_dotenv
from rag_common.voyage import get_voyage_client

from retrieval.search import search
from storage.db import get_connection


def build_context(chunks: list[dict]) -> str:
    """검색 결과 청크들을 번호와 출처가 붙은 컨텍스트 블록으로 만든다.

    각 청크는 "[n] 출처: {chapter} p.{page_number}\\n{chunk_text}" 형태가 되고,
    청크 사이는 빈 줄로 구분한다. 번호를 매기는 이유는 build_prompt가 만드는
    최종 프롬프트에서 "답변 끝에 참고한 출처 번호를 표시해"라고 요청하기
    때문이다 — notion_rag_test와 같은 이유.
    """
    parts = [
        f"[{i}] 출처: {c['chapter'] or '(챕터 미상)'} p.{c['page_number']}\n{c['chunk_text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: list[dict]) -> str:
    """지시문 + 컨텍스트 + 질문을 합쳐 복사-붙여넣기용 최종 프롬프트 문자열을 만든다."""
    context = build_context(chunks)
    return (
        "다음 컨텍스트만 근거로 답변하고, 컨텍스트에 없으면 모른다고 답해. "
        "답변 끝에 참고한 출처 번호를 표시해.\n\n"
        f"[컨텍스트]\n{context}\n\n"
        f"[질문]\n{query}"
    )


def _print_results(query: str, results: list[dict]) -> None:
    print(f'\n질문: "{query}"')
    if not results:
        print("검색 결과가 없습니다.")
        return

    print(f"상위 {len(results)}개 결과:\n")
    for i, r in enumerate(results, start=1):
        chapter = r["chapter"] or "(챕터 미상)"
        score = r.get("relevance_score")
        score_line = f"relevance_score={score:.4f}" if score is not None else ""
        print(f"[{i}] p.{r['page_number']} | {chapter} {score_line}")
        print(f"{r['chunk_text']}\n")
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="AWS SAA 책 검색 CLI (hybrid+rerank)")
    parser.add_argument("query", nargs="+", help="검색할 질문")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    query = " ".join(args.query).strip()

    load_dotenv()

    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    model = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")
    rerank_model = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")

    conn = get_connection()
    try:
        results = search(
            conn, voyage_client, query, model=model, top_k=args.top_k, rerank_model=rerank_model
        )
    finally:
        conn.close()

    _print_results(query, results)


if __name__ == "__main__":
    main()
