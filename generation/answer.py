# CLI 진입점: 질문을 넣으면 retrieval/search.py의 hybrid+rerank 검색 결과를
# 출처(페이지 번호/챕터)와 함께 콘솔에 바로 출력한다.
#
# notion_rag_test/generation/answer.py와 달리 이 프로젝트는 복사-붙여넣기용
# 프롬프트 조립이 필요 없다(사람이 바로 검색 결과를 읽고 싶을 뿐이라서) —
# 그래서 build_prompt/build_context 같은 문자열 조립 함수 없이, 검색
# 결과만 사람이 읽기 좋은 형태로 print한다.
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
