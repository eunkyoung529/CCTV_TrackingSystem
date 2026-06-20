# app.py
import os
os.environ["YOLO_VERBOSE"] = "False"

import streamlit as st

st.set_page_config(
    page_title="실시간 뺑소니 차량 탐색 시스템",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with open("assets/style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

from core.state import init_session_state
init_session_state()

st.title("🚨 실시간 뺑소니 차량 탐색 시스템")
st.markdown("---")

st.markdown(
    """
    ### 프로젝트 소개
    뺑소니 사고가 발생했을 때, **목격된 차량 정보**와 **CCTV 영상**을 활용해
    용의 차량을 실시간으로 추적·식별하는 데모입니다.

    #### 사용 흐름
    1. **차량 정보 입력** — 색상 / 제조사 / 세부 모델 입력
    2. **영상 업로드 & 실시간 탐색** — CCTV 영상을 올리면 화면에서 바로 차량 인식 + 일치 차량은 빨간 박스로 강조
    3. **결과 확인** — 일치율 높은 후보 차량 정리 출력

    👉 왼쪽 사이드바에서 **1️⃣ 차량정보입력** 부터 시작하세요.
    """
)

st.markdown("---")
st.subheader("📋 현재 진행 상황")
c1, c2, c3 = st.columns(3)
with c1:
    done = st.session_state.get("vehicle_info") is not None
    st.metric("1. 차량정보", "✅ 완료" if done else "⏳ 대기")
with c2:
    done = st.session_state.get("search_done", False)
    st.metric("2. 영상 탐색", "✅ 완료" if done else "⏳ 대기")
with c3:
    done = st.session_state.get("results") is not None
    st.metric("3. 결과", "✅ 확인가능" if done else "⏳ 대기")

# 사이드바 — 기상 환경 선택 (모델 conf/iou에 영향)
st.sidebar.markdown("### ⚙️ 환경 설정")
st.session_state.weather = st.sidebar.selectbox(
    "기상 환경",
    ["맑음", "박무,연무", "박무,비", "눈"],
    index=["맑음", "박무,연무", "박무,비", "눈"].index(st.session_state.get("weather", "맑음")),
    help="기상에 따라 탐지 민감도가 자동 조정됩니다.",
)