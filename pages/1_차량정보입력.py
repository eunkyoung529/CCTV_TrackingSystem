# pages/1_차량정보입력.py
import streamlit as st
import json

from core.state import init_session_state, reset_search

init_session_state()

st.title("1️⃣ 범인 차량 정보 입력")
st.caption("목격된 차량의 색상·제조사·세부 모델을 입력하세요.")

with open("label_encoders.json", "r", encoding="utf-8") as f:
    encoders = json.load(f)

brands = [b for b in encoders["brand"].keys() if b != "unknown"]
colors = [c for c in encoders["color"].keys() if c not in ("unknown", "기타")]
all_models = [m for m in encoders["model"].keys() if m != "unknown"]

# 차종(prefix) 추출: "세단_쏘나타" → "세단"
model_types = sorted({m.split("_")[0] for m in all_models if "_" in m})

mode = st.radio("입력 방식", ["토글로 선택", "텍스트로 입력"], horizontal=True)

with st.container(border=True):
    if mode == "토글로 선택":
        c1, c2 = st.columns(2)
        with c1:
            color = st.selectbox(
                "색상", colors,
                index=colors.index("검은색") if "검은색" in colors else 0,
            )
            model_type = st.selectbox(
                "차종", model_types,
                index=model_types.index("세단") if "세단" in model_types else 0,
            )
        with c2:
            brand = st.selectbox(
                "제조사", brands,
                index=brands.index("현대자동차") if "현대자동차" in brands else 0,
            )
            # 차종으로 세부모델 필터링
            candidates = [m for m in all_models if m.startswith(model_type + "_")]
            if candidates:
                model_label = st.selectbox("세부 모델", candidates)
            else:
                st.warning(f"'{model_type}' 세부 모델 없음")
                model_label = "unknown"
    else:
        c1, c2 = st.columns(2)
        with c1:
            color = st.text_input("색상", value="검은색")
            brand = st.text_input("제조사", value="현대자동차")
        with c2:
            model_label = st.text_input("세부 모델 (예: 세단_쏘나타)", value="세단_쏘나타")
        if model_label not in encoders["model"]:
            st.info(f"💡 '{model_label}'은 라벨에 없습니다. 가까운 라벨로 자동 매칭됩니다.")

st.markdown(" ")
register = st.button("✅ 등록", type="primary", use_container_width=True)

if register:
    st.session_state.vehicle_info = {
        "color": color,
        "brand": brand,
        "model": model_label,
    }
    reset_search()
    st.success("등록 완료! 사이드바에서 **2️⃣ 영상업로드_탐색** 으로 진행하세요.")
    st.json(st.session_state.vehicle_info)

if st.session_state.vehicle_info:
    st.markdown("---")
    st.caption("현재 등록된 정보:")
    st.json(st.session_state.vehicle_info)