# PDF에서 페이지별 텍스트를 추출한다.
#
# PyMuPDF(fitz)를 골랐다 — 실제 AWS SAA Study Guide PDF(593페이지)의 표본
# 페이지(0, 1, 50, 150, 300)를 pdfplumber와 직접 비교한 뒤 결정했다:
# 둘 다 본문 텍스트 품질은 비슷했지만(둘 다 표지 페이지에서 동일한
# 이미지/폰트 인코딩 깨짐이 있었는데, 이건 표지 디자인 자체의 문제라 본문과
# 무관하다), 표가 섞인 페이지(예: 50페이지의 서비스 카테고리 표)에서
# PyMuPDF의 get_text()가 셀 내용을 더 자연스러운 문장 흐름으로 이어붙였고
# (pdfplumber는 셀 경계마다 줄바꿈이 끼어들어 문장이 더 자주 끊겼다),
# pdfplumber의 extract_tables()도 이 표 레이아웃을 구조화된 표로 인식하지
# 못했다(빈 결과) — 즉 표 추출 기능을 쓸 실익이 없었다. 이 프로젝트는 책
# 전체를 텍스트 청크로 만들어 임베딩하는 용도라 표를 별도 구조로 보존할
# 필요가 없으므로, 더 단순하고 빠른 PyMuPDF만 쓴다.

# "import fitz"는 옛 별칭이고 PyMuPDF가 향후 제거 예정이라고 경고하므로
# 정식 모듈명(pymupdf)을 fitz로 alias해서 쓴다 — 나머지 코드는 그대로 유지.
import pymupdf as fitz


def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """PDF의 각 페이지에서 텍스트를 추출해 페이지 번호와 함께 반환한다.

    페이지 번호는 1부터 시작한다(사람이 책을 펼쳐서 찾는 번호와 맞추기 위해
    — PyMuPDF 내부 인덱스는 0부터지만 book_chunks.page_number는 1부터로
    저장한다).

    Returns:
        {"page_number": int, "text": str} dict의 리스트, 페이지 순서대로.
        텍스트가 비어 있는 페이지(예: 장 구분용 백지)도 그대로 포함한다 —
        빈 페이지를 걸러내는 건 이 함수의 책임이 아니라 호출하는 쪽
        (scripts/ingest_book.py)의 책임이다.
    """
    doc = fitz.open(pdf_path)
    try:
        return [
            {"page_number": i + 1, "text": doc[i].get_text()}
            for i in range(doc.page_count)
        ]
    finally:
        doc.close()
