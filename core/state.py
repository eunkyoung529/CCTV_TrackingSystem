# core/state.py
import streamlit as st


def init_session_state():
    defaults = {
        "vehicle_info": None,      # {"color":..., "brand":..., "model":...}
        "weather": "맑음",
        "results": None,           # 탐색 후 분류된 모든 차량
        "search_done": False,
        "threshold": 60,           # 일치율 임계값(%)
        "uploaded_video_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_search():
    st.session_state.results = None
    st.session_state.search_done = False