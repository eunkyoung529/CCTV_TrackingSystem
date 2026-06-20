# pages/3_결과출력.py
import streamlit as st

from core.state import init_session_state

init_session_state()

st.title("3️⃣ 탐색 결과")

if not st.session_state.search_done or st.session_state.results is None:
    st.warning("아직 탐색이 완료되지 않았습니다. **2️⃣ 영상업로드_탐색**을 먼저 실행하세요.")
    st.stop()

results = st.session_state.results

# 임계값 조정
st.markdown("#### 🎚️ 일치율 임계값 조정")
threshold_pct = st.slider("임계값 (%)", 30, 99,
                          int(st.session_state.threshold), 1,
                          label_visibility="collapsed")
st.session_state.threshold = threshold_pct
threshold = threshold_pct / 100.0

st.markdown("---")

# 후보 추출 + 정렬
candidates = [r for r in results if r["match"]["overall"] >= threshold]
candidates.sort(key=lambda r: r["match"]["overall"], reverse=True)

# 요약
m1, m2, m3 = st.columns(3)
m1.metric("분류된 차량 (전체)", f"{len(results)}대")
m2.metric("임계값 이상 후보", f"{len(candidates)}대")
m3.metric("적용 임계값", f"{threshold_pct}%")

st.markdown("---")

if not candidates:
    st.error(f"⚠️ 임계값 {threshold_pct}% 이상 일치 차량이 없습니다. 슬라이더를 낮춰보세요.")

    # 분류된 차량 일치율 순 미리보기
    preview = sorted(results, key=lambda r: r["match"]["overall"], reverse=True)
    if preview:
        st.markdown("### 🔎 분류된 차량 (일치율 순)")
        for i, r in enumerate(preview[:15], 1):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(
                    f"**#{i}** Track {r['track_id']} — "
                    f"`{r['pred']['model']}` · `{r['pred']['color']}` · `{r['pred']['brand']}`"
                )
            with c2:
                st.metric("일치율", f"{r['match']['overall']*100:.1f}%",
                          r["match"]["grade"], label_visibility="collapsed")
    st.stop()

st.success(f"🚨 일치 차량을 **{len(candidates)}대** 발견했습니다.")

v = st.session_state.vehicle_info
for rank, r in enumerate(candidates, 1):
    match = r["match"]
    pred = r["pred"]
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"

    with st.container(border=True):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown(f"### {medal} 후보 #{rank}")
            st.markdown(f"🎯 **Track ID:** `{r['track_id']}`")
            st.markdown(f"📊 **최종 등급:** `{match['grade']}`")
            st.progress(match["overall"],
                        text=f"종합 일치율 {match['overall']*100:.1f}%")

            with st.expander("⬇️ 요소별 세부 일치율"):
                items = [
                    ("색상",   v["color"], pred["color"], match["per_field"]["color"]),
                    ("제조사", v["brand"], pred["brand"], match["per_field"]["brand"]),
                    ("모델",   v["model"], pred["model"], match["per_field"]["model"]),
                ]
                for label, tv, pv, prob in items:
                    cc1, cc2, cc3 = st.columns([1, 2, 1])
                    cc1.markdown(f"**{label}**")
                    cc2.markdown(f"입력 `{tv}` · 예측 `{pv}`")
                    cc3.metric("확률", f"{prob*100:.0f}%", label_visibility="collapsed")
                if "resolved" in match:
                    st.caption(f"🔎 매칭된 실제 라벨: `{match['resolved']}`")

        with c2:
            st.markdown("**탐지 화면:**")
            st.image(r["frame"], use_container_width=True,
                     caption=f"Track ID: {r['track_id']}")

st.markdown("---")
if st.button("🔄 새로운 사건 시작하기", use_container_width=True):
    for k in ["vehicle_info", "search_done", "results", "uploaded_video_path"]:
        st.session_state[k] = None
    st.rerun()