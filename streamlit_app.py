# AWS SAA Study Guide 검색 UI (Streamlit Community Cloud 배포용).
#
# retrieval.search.search()를 FastAPI/HTTP 레이어 없이 in-process로 직접
# 호출한다 — generation/answer.py CLI가 이미 이 패턴이라 그대로 따랐다.
#
# 로컬 실행: streamlit run streamlit_app.py (.streamlit/secrets.toml 필요,
#   .streamlit/secrets.toml.example을 복사해서 만들 것)
# 배포: share.streamlit.io에서 이 레포를 연결하고 App settings > Secrets에
#   .streamlit/secrets.toml.example과 같은 키로 실제 값을 입력한다.

import os

import streamlit as st

st.set_page_config(page_title="AWS SAA Study Guide 검색", page_icon="📚")

_ENV_KEYS_FROM_SECRETS = [
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_SSLMODE",
    "VOYAGE_API_KEY",
    "VOYAGE_MODEL",
    "VOYAGE_RERANK_MODEL",
]


def _load_secrets_into_env() -> None:
    """st.secrets의 값들을 os.environ에 반영한다.

    storage.db.get_connection()(rag_common.db 경유)과 rag_common.voyage.get_voyage_client는
    전부 os.environ 기반으로 이미 짜여 있고 검증까지 끝났다 — 이 함수들을 손대지
    않고 그대로 재사용하기 위해, Streamlit Cloud가 주는 st.secrets 값을 여기서
    os.environ으로 한 번만 옮겨준다(로컬/CLI 스크립트는 이 다리가 필요 없이
    .env를 직접 읽지만, Streamlit Cloud는 비밀값을 st.secrets로만 주기 때문에
    이 변환이 꼭 필요하다).
    """
    for key in _ENV_KEYS_FROM_SECRETS:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])


_load_secrets_into_env()

from rag_common.voyage import get_voyage_client  # noqa: E402 (secrets를 env로 옮긴 뒤 import)

from generation.answer import build_prompt  # noqa: E402
from retrieval.search import search  # noqa: E402
from storage.db import get_connection  # noqa: E402


@st.cache_resource
def _voyage_client():
    return get_voyage_client(os.environ["VOYAGE_API_KEY"])


def _check_password() -> bool:
    """st.secrets['APP_PASSWORD']와 입력값이 일치해야 검색 UI를 보여준다.

    무료 티어 Streamlit 앱은 public이라 링크를 아는 사람 누구나 접속할 수
    있다 — 정식 인증은 아니고, 최소한의 접근 게이트 정도로 충분하다는
    전제로 만들었다.
    """
    if st.session_state.get("authenticated"):
        return True

    password = st.text_input("비밀번호", type="password")
    if not password:
        return False

    if password == st.secrets.get("APP_PASSWORD"):
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.error("비밀번호가 틀렸습니다.")
    return False


st.title("📚 AWS SAA Study Guide 검색")

if not _check_password():
    st.stop()

query = st.text_input("질문을 입력하세요", placeholder="예: S3 스토리지 클래스는 어떤 종류가 있어?")
top_k = st.slider("검색 결과 개수", min_value=1, max_value=10, value=5)

if st.button("검색", type="primary") and query.strip():
    model = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")
    rerank_model = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")

    with st.spinner("검색 중..."):
        conn = get_connection()
        try:
            results = search(
                conn, _voyage_client(), query, model=model, top_k=top_k, rerank_model=rerank_model
            )
        finally:
            conn.close()

    if not results:
        st.info("검색 결과가 없습니다.")

    for i, r in enumerate(results, start=1):
        chapter = r["chapter"] or "(챕터 미상)"
        score = r.get("relevance_score")
        score_label = f" · relevance_score={score:.4f}" if score is not None else ""
        with st.container(border=True):
            st.markdown(f"**[{i}] p.{r['page_number']} · {chapter}**{score_label}")
            st.write(r["chunk_text"])

    if results:
        st.subheader("복붙용 프롬프트")
        st.caption("아래 블록을 그대로 복사해서 Claude 채팅에 붙여넣으면 됩니다.")
        st.code(build_prompt(query, results), language=None)
