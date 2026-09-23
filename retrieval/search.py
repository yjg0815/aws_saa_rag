# 검색 시점 로직: 질문을 임베딩해서 관련도 높은 청크를 가져오고, 리랭커로 한 번 더 추린다.
#
# notion_rag_test/retrieval/search.py의 hybrid+rerank 흐름(dense 검색 +
# BM25 검색 → RRF 병합 → 리랭커로 최종 top_k)과 같은 패턴이다. rag_common의
# voyage/fusion을 가져다 쓰고, DB 쿼리는 book_chunks 스키마에 맞게
# storage.db에 새로 작성했다.

from rag_common.fusion import reciprocal_rank_fusion
from rag_common.voyage import embed_query, rerank_documents

from storage.db import search_bm25, search_similar

_DEFAULT_MODEL = "voyage-4-lite"
_DEFAULT_RERANK_MODEL = "rerank-2.5-lite"


def search(
    conn,
    voyage_client,
    query: str,
    model: str = _DEFAULT_MODEL,
    top_k: int = 5,
    rerank: bool = True,
    rerank_model: str = _DEFAULT_RERANK_MODEL,
    fetch_k: int | None = None,
    hybrid: bool = True,
    bm25_fetch_k: int | None = None,
) -> list[dict]:
    """query를 dense 임베딩(+옵션으로 BM25 키워드) 검색으로 1차 후보를 모으고, rerank=True면 리랭커로 top_k개만 추린다.

    embed_query는 input_type="query"로 임베딩한다 — 저장할 때 쓴
    input_type="document" 임베딩과는 벡터 공간이 다르므로, 검색 쪽에서도
    반드시 query용 임베딩을 써야 storage.db.search_similar의 코사인 거리
    비교가 의미를 가진다.

    fetch_k(dense 1차 후보 개수)는 top_k보다 넉넉하게 잡아야 리랭커/RRF가
    고를 거리가 생긴다 — 지정하지 않으면 top_k*4, 최소 20으로 자동 설정한다.

    hybrid=True(기본값)면 dense 후보와 storage.db.search_bm25의 BM25(키워드)
    후보를 rag_common.fusion.reciprocal_rank_fusion으로 병합해서 그 순서를
    쓴다 — CIDR 표기, 서비스명, 약어 같은 정확한 기술 용어는 dense
    임베딩만으로는 top-k에 못 들어올 수 있는데, BM25가 이런 케이스를
    보완한다. hybrid=False면 dense 결과만 쓴다. bm25_fetch_k를 지정하지
    않으면 fetch_k와 같은 값을 쓴다.

    hybrid와 rerank는 서로 완전히 독립적이다.

    Returns:
        rerank=False일 때는 병합만 된 상태로 top_k개(hybrid=True면 rrf_score
        내림차순, hybrid=False면 storage.db.search_similar 그대로 distance
        오름차순). rerank=True일 때는 각 dict에 relevance_score가 추가되고,
        relevance_score 내림차순으로 정렬된 top_k개만 반환된다.
    """
    query_embedding = embed_query(voyage_client, query, model=model)

    if fetch_k is None:
        fetch_k = max(top_k * 4, 20)

    dense_candidates = search_similar(conn, query_embedding, top_k=fetch_k)

    if hybrid:
        bm25_candidates = search_bm25(conn, query, top_k=bm25_fetch_k or fetch_k)
        candidates = reciprocal_rank_fusion([dense_candidates, bm25_candidates])
    else:
        candidates = dense_candidates

    if not rerank:
        return candidates[:top_k]

    if not candidates:
        return []

    documents = [c["chunk_text"] for c in candidates]
    ranked = rerank_documents(voyage_client, query, documents, model=rerank_model, top_k=top_k)

    results = []
    for r in ranked:
        candidate = dict(candidates[r["index"]])
        candidate["relevance_score"] = r["relevance_score"]
        results.append(candidate)
    return results
