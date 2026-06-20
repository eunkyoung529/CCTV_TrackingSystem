# core/matching.py
"""
ConvNeXt 멀티헤드 출력(brand/color/model) vs 사용자 입력 차량정보 → 일치율 계산.
year가 사라졌으므로 3속성 가중 평균.
"""
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _resolve_label(user_value, label_dict: dict):
    """사용자 입력값을 encoders 실제 키로 매핑 (완전일치 → 부분일치 → None)"""
    if user_value is None:
        return None
    if user_value in label_dict:
        return user_value
    for key in label_dict.keys():
        if key == "unknown":
            continue
        if user_value in key or key in user_value:
            return key
    return None


def compute_match(probs_dict: dict, target_dict: dict, encoders: dict) -> dict:
    """
    Args:
        probs_dict: {"brand": np.array, "color": np.array, "model": np.array} (softmax 확률)
        target_dict: {"brand":..., "color":..., "model":...}
        encoders: label_encoders.json
    Returns:
        {"per_field": {...}, "overall": float, "grade": str, "resolved": {...}}
    """
    per_field = {}
    resolved = {}

    for field in ["brand", "color", "model"]:
        user_val = target_dict.get(field)
        actual = _resolve_label(user_val, encoders[field])
        resolved[field] = actual
        if actual is None:
            per_field[field] = 0.0
            continue
        idx = encoders[field][actual]
        per_field[field] = float(probs_dict[field][idx])

    # 가중치: 세부모델(model)이 가장 식별력 높음
    weights = {"brand": 0.3, "color": 0.3, "model": 0.4}
    overall = sum(per_field[k] * weights[k] for k in per_field)

    if overall >= 0.85:
        grade = "A (확실함)"
    elif overall >= 0.70:
        grade = "B (높음)"
    elif overall >= 0.55:
        grade = "C (보통)"
    else:
        grade = "D (낮음)"

    return {"per_field": per_field, "overall": overall, "grade": grade, "resolved": resolved}