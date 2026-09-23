# PDF 경로를 인자로 받아 book_chunks 테이블을 한 번에 채우는 일회성 수집 스크립트.
#
# 정적 문서(PDF)라 notion_rag_test의 discover_pages 같은 증분 동기화/캐싱
# 구조는 필요 없다 — 한 번 실행해서 끝나는 전체 재수집 방식이다.
#
# 사용법:
#     python -m scripts.ingest_book "<PDF 경로>"

import bisect
import os
import sys

from dotenv import load_dotenv
from rag_common.chunking import chunk_text
from rag_common.voyage import embed_chunks, get_voyage_client

from ingestion.pdf_extractor import extract_pdf_pages
from storage.db import get_connection, init_schema, insert_chunks

_MAX_CHARS = 1500
_OVERLAP_CHARS = 200

# AWS Certified Solutions Architect STUDY GUIDE(4/e, 총 593페이지)의 챕터
# 경계. Desktop/업무/EIC/EIC문서/AWS_CSA_STUDY_GUIDE_분할_문서/의 분할 PDF
# 파일명에 박혀 있는 페이지 범위를 실제 원본 PDF에서 페이지별 텍스트를
# 직접 뽑아 대조해서 검증했다 — 예: 70페이지가 정확히 "컴퓨트 서비스"
# 챕터 제목으로 시작함을 확인. 부록 A/B 경계(524, 564페이지)는 파일명에
# 범위가 없어서 "부록"/"기타 서비스 핵심 정리" 텍스트가 실제로 나타나는
# 페이지를 직접 검색해서 찾았다. 이 표는 이 특정 PDF에만 유효하다 — 다른
# 판/다른 책이면 재검증해야 한다.
_CHAPTER_RANGES = [
    (1, 69, "1장 클라우드 컴퓨팅과 AWS 개요"),
    (70, 123, "2장 컴퓨트 서비스"),
    (124, 155, "3장 AWS 스토리지"),
    (156, 217, "4장 Amazon VPC"),
    (218, 261, "5장 데이터베이스 서비스"),
    (262, 283, "6장 자격 인증과 권한 부여"),
    (284, 321, "7장 모니터링"),
    (322, 349, "8장 DNS와 CDN"),
    (350, 365, "9장 데이터 유입, 변환, 그리고 분석"),
    (366, 409, "10장 복원성 아키텍처"),
    (410, 453, "11장 고성능 아키텍처"),
    (454, 497, "12장 보안성 아키텍처"),
    (498, 523, "13장 비용최적화 아키텍처"),
    (524, 563, "부록 A. 평가 문제 정답 및 해설"),
    (564, 593, "부록 B. 기타 서비스 핵심 정리"),
]


def _chapter_for_page(page_number: int) -> str | None:
    """page_number가 속한 챕터 이름을 반환한다. 어느 범위에도 안 걸리면(목차 감지가 안 되는 경우) None."""
    for start, end, chapter in _CHAPTER_RANGES:
        if start <= page_number <= end:
            return chapter
    return None


def _build_full_text_with_offsets(pages: list[dict]) -> tuple[str, list[int], list[int]]:
    """페이지들을 이어붙여 하나의 텍스트로 만들고, 각 페이지가 시작하는 offset을 같이 기록한다.

    rag_common.chunking.chunk_text는 원본 동작을 바꾸지 않기로 했으므로(이미
    검증된 함수) 각 조각의 원본 offset을 반환하지 않는다 — 그래서 여기서는
    반환받은 조각 문자열을 전체 텍스트 안에서 다시 찾아 페이지를 역산한다
    (_locate_chunk_page 참고).

    Returns:
        (전체 텍스트, 페이지별 시작 offset 리스트, 그 offset에 대응하는 page_number 리스트).
        뒤 두 리스트는 bisect로 이분 탐색할 수 있도록 offset 오름차순으로 정렬돼 있다.
    """
    full_text = ""
    offsets = []
    page_numbers = []
    for page in pages:
        offsets.append(len(full_text))
        page_numbers.append(page["page_number"])
        full_text += page["text"] + "\n"
    return full_text, offsets, page_numbers


def _locate_chunk_page(full_text: str, piece: str, search_from: int, offsets: list[int], page_numbers: list[int]) -> tuple[int, int]:
    """piece가 full_text의 어느 위치에서 시작하는지 찾아 그 지점의 page_number를 역산한다.

    chunk_text()가 만드는 조각들은 항상 이전 조각보다 뒤에서(또는 같은
    지점에서) 시작한다(슬라이딩 윈도우가 앞으로만 진행하므로) — 그래서
    search_from을 매번 "직전에 찾은 위치"로 갱신해 두면 find()가 항상 올바른
    occurrence를 찾는다(문서 안에 같은 문장이 반복돼도 엉뚱한 이전 위치로
    되돌아가지 않는다).

    Returns:
        (이 piece가 시작하는 full_text 안의 offset, 그 offset이 속한 page_number).
        못 찾으면(이론상 나올 수 없지만 방어적으로) offset=search_from, 가장
        가까운 페이지를 반환한다.
    """
    idx = full_text.find(piece, search_from)
    if idx == -1:
        idx = search_from  # 방어적 fallback — 정상 흐름에서는 도달하지 않아야 함

    page_idx = bisect.bisect_right(offsets, idx) - 1
    page_idx = max(0, page_idx)
    return idx, page_numbers[page_idx]


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.ingest_book "<PDF 경로>"', file=sys.stderr)
        sys.exit(1)
    pdf_path = sys.argv[1]

    load_dotenv()

    print(f"PDF 텍스트 추출 중: {pdf_path}")
    pages = extract_pdf_pages(pdf_path)
    print(f"  {len(pages)}페이지 추출 완료.")

    full_text, offsets, page_numbers = _build_full_text_with_offsets(pages)
    print(f"전체 텍스트 길이: {len(full_text)}자")

    print(f"청킹 중 (max_chars={_MAX_CHARS}, overlap_chars={_OVERLAP_CHARS})...")
    pieces = chunk_text(full_text, max_chars=_MAX_CHARS, overlap_chars=_OVERLAP_CHARS)
    print(f"  {len(pieces)}개 조각 생성됨.")

    chunks = []
    search_from = 0
    for piece in pieces:
        idx, page_number = _locate_chunk_page(full_text, piece, search_from, offsets, page_numbers)
        search_from = idx + 1
        chunks.append(
            {
                "page_number": page_number,
                "chapter": _chapter_for_page(page_number),
                "chunk_text": piece,
            }
        )

    voyage_client = get_voyage_client(os.environ["VOYAGE_API_KEY"])
    model = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")
    print(f"{len(chunks)}개 청크를 {model}로 임베딩 중 (input_type=document)...")
    embedded_chunks = embed_chunks(voyage_client, chunks, model=model)

    conn = get_connection()
    try:
        init_schema(conn)
        insert_chunks(conn, embedded_chunks)
    finally:
        conn.close()

    total_chars = sum(len(c["chunk_text"]) for c in chunks)
    # 정확한 토큰 수가 아니라 대략치다 — 별도 토크나이저 라이브러리를 쓰지
    # 않기로 했다. 한국어/영어가 섞인 텍스트라 글자당 토큰 비율이 일정하지
    # 않으므로, 대략 글자수/2 ~ 글자수/3 사이로 어림잡는 것이 보수적인 추정이다.
    print(f"\n=== 수집 완료 ===")
    print(f"총 청크 수: {len(chunks)}개")
    print(f"총 글자 수: {total_chars}자")
    print(f"대략적인 토큰 수(추정, 정확한 값 아님): {total_chars // 3} ~ {total_chars // 2}")


if __name__ == "__main__":
    main()
