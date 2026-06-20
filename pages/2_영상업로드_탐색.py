# pages/2_영상업로드_탐색.py
import os
import tempfile
import cv2
import streamlit as st

from core.state import init_session_state, reset_search
from core.engine import VehicleTracker, load_encoders

init_session_state()

st.title("2️⃣ 영상 업로드 & 실시간 탐색")

if not st.session_state.vehicle_info:
    st.warning("먼저 **1️⃣ 차량정보입력**을 완료하세요.")
    st.stop()

v = st.session_state.vehicle_info
st.info(
    f"🔍 찾는 차량 — 색상: **{v['color']}** · 제조사: **{v['brand']}** · 모델: **{v['model']}**"
)

# ---------- 영상 업로드 ----------
uploaded = st.file_uploader(
    "CCTV 영상 업로드 (mp4, avi, mov)",
    type=["mp4", "avi", "mov", "mkv"],
)

# ---------- 옵션 ----------
col1, col2, col3 = st.columns(3)
with col1:
    threshold = st.slider("일치율 임계값 (%)", 30, 99, int(st.session_state.threshold), 1)
    st.session_state.threshold = threshold
with col2:
    max_seconds = st.slider("분석할 최대 길이 (초)", 5, 60, 10, 1,
                            help="긴 영상은 앞부분만 분석합니다.")
with col3:
    frame_skip = st.slider("프레임 스킵", 1, 5, 2, 1,
                           help="값이 클수록 빠르지만 덜 촘촘하게 분석합니다.")

st.caption("⚙️ 기상 환경은 왼쪽 사이드바(메인 페이지)에서 변경할 수 있습니다. "
           f"현재: **{st.session_state.weather}**")

# ---------- 탐색 시작 ----------
start = st.button("▶️ 실시간 탐색 시작", type="primary", use_container_width=True,
                  disabled=(uploaded is None))
stop = st.sidebar.button("⏹️ 탐색 중지")

# 실시간 표시 영역
video_area = st.empty()
stat_area = st.empty()
prog_area = st.empty()

if start and uploaded is not None:
    reset_search()

    # 업로드 영상을 임시 파일로 저장
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.close()
    st.session_state.uploaded_video_path = tfile.name

    encoders = load_encoders()
    tracker = VehicleTracker(
        target=v, encoders=encoders, weather=st.session_state.weather
    )

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("영상을 열 수 없습니다.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * max_seconds)

    frame_idx = 0
    processed = 0
    with st.spinner("🤖 모델 로딩 및 분석 중..."):
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx > max_frames:
                break
            if frame_skip > 1 and frame_idx % frame_skip != 0:
                continue

            # 사이드바 중지 버튼 (rerun 기반이라 한계 있음 → 아래 설명 참고)
            if stop:
                st.warning("사용자가 탐색을 중지했습니다.")
                break

            frame_rgb, stats = tracker.process_frame(frame)
            processed += 1

            # 실시간 표시
            video_area.image(frame_rgb, channels="RGB", use_container_width=True)
            stat_area.markdown(
                f"**추적 차량:** `{stats['total_tracks']}대`  |  "
                f"**분류 완료:** `{stats['classified']}대`  |  "
                f"**🎯 일치 차량:** `{stats['matches']}대`  |  "
                f"**엔진:** `{tracker.backend}`"
            )
            prog_area.progress(
                min(frame_idx / max_frames, 1.0),
                text=f"분석 진행률 {int(min(frame_idx/max_frames,1.0)*100)}%",
            )

    cap.release()

    # 결과 저장
    st.session_state.results = tracker.get_all_results()
    st.session_state.search_done = True
    prog_area.empty()

    st.success(
        f"✅ 탐색 완료! 총 {stats['total_tracks']}대 추적, "
        f"{stats['classified']}대 분류, "
        f"임계값 {threshold}% 이상 일치 {stats['matches']}대. "
        f"사이드바에서 **3️⃣ 결과출력** 페이지로 이동하세요."
    )

elif st.session_state.search_done:
    st.info("이미 탐색이 완료되었습니다. **3️⃣ 결과출력** 페이지에서 확인하세요. "
            "다시 분석하려면 영상을 업로드하고 탐색 시작을 누르세요.")