# core/engine.py
"""
finalModel.zip의 app_onlyCPU.py 추론 로직을 모듈화.
- ONNX 세션 (YOLO 1024 + ConvNeXt 224) 캐싱 로드
- 커스텀 NMS + IoU 추적
- 프레임 1장을 처리하는 process_frame() 제너레이터 친화 함수
"""
import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from PIL import ImageFont, ImageDraw, Image
import streamlit as st

from core.matching import softmax, compute_match


# 경로
MODELS_DIR = "models"
YOLO_PATH = os.path.join(MODELS_DIR, "yolov8n.onnx")
CONVNEXT_PATH = os.path.join(MODELS_DIR, "convnext_cbam_multihead_phase4.onnx")
ENCODER_PATH = "label_encoders.json"

# 기상별 하이퍼파라미터 (원본 그대로)
WEATHER_ADAPTATION_MAP = {
    "맑음":      {"conf": 0.10, "iou": 0.45},
    "박무,연무":  {"conf": 0.08, "iou": 0.45},
    "박무,비":    {"conf": 0.06, "iou": 0.50},
    "눈":        {"conf": 0.05, "iou": 0.50},
}

QUALITY_THRESHOLD = 2500
MAX_LOST_FRAMES = 15
VEHICLE_COCO = [2, 3, 5, 7]


# ---------- 캐싱 로더 ----------
@st.cache_resource(show_spinner=False)
def load_encoders():
    with open(ENCODER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_sessions():
    if not os.path.exists(CONVNEXT_PATH):
        os.makedirs(MODELS_DIR, exist_ok=True)
        try:
            import gdown
            gdrive_id = "1rO0DG_1ss2ufYbWaJhQPcmR0wn-hYV4L"
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            with st.spinner("⬇️ ConvNeXt 가중치를 다운로드하는 중입니다... (최초 1회)"):
                gdown.download(url, CONVNEXT_PATH, quiet=False)
        except Exception as e:
            st.error(
                f"모델 자동 다운로드 실패: {e}\n\n"
                "터미널에서 `python download_model.py`를 직접 실행하거나, "
                "구글 드라이브에서 파일을 받아 models/ 폴더에 넣어주세요."
            )
            st.stop()

    providers = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    y = ort.InferenceSession(YOLO_PATH, sess_options=opts, providers=providers)
    c = ort.InferenceSession(CONVNEXT_PATH, sess_options=opts, providers=providers)
    backend = y.get_providers()[0]
    return y, c, y.get_inputs()[0].name, c.get_inputs()[0].name, backend


@st.cache_resource(show_spinner=False)
def load_font():
    paths = [
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for fp in paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, 15)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------- 수학 모듈 (원본 그대로) ----------
def calculate_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def apply_custom_nms(preds, iou_thr):
    if not preds:
        return []
    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)
    keep = []
    while preds:
        best = preds.pop(0)
        keep.append(best)
        preds = [p for p in preds if calculate_iou(best["box"], p["box"]) < iou_thr]
    return keep


def preprocess_yolo(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def preprocess_convnext(roi_bgr):
    img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


# ---------- 탐색기 클래스 ----------
class VehicleTracker:
    """프레임 단위로 호출하면 추적 상태를 누적 관리하는 객체"""

    def __init__(self, target, encoders, weather="맑음"):
        self.target = target
        self.encoders = encoders
        self.id_to = {
            "brand": {v: k for k, v in encoders["brand"].items()},
            "color": {v: k for k, v in encoders["color"].items()},
            "model": {v: k for k, v in encoders["model"].items()},
        }
        self.params = WEATHER_ADAPTATION_MAP.get(weather, WEATHER_ADAPTATION_MAP["맑음"])

        self.track_registry = {}
        self.last_frame_tracks = {}
        self.next_track_id = 1

        self.yolo, self.cls, self.yn, self.cn, self.backend = load_sessions()
        self.font = load_font()

        # 매칭 결과 누적 (track_id -> best match info)
        self.match_results = {}

    def _classify(self, roi):
        t = preprocess_convnext(roi)
        out = self.cls.run(None, {self.cn: t})
        probs = {
            "brand": softmax(out[0][0]),
            "color": softmax(out[1][0]),
            "model": softmax(out[2][0]),
        }
        pred = {
            "brand": self.id_to["brand"].get(int(np.argmax(out[0]))),
            "color": self.id_to["color"].get(int(np.argmax(out[1]))),
            "model": self.id_to["model"].get(int(np.argmax(out[2]))),
        }
        return probs, pred

    def process_frame(self, frame):
        """한 프레임 처리 → BBox/라벨 그려진 RGB 프레임 + 통계 반환"""
        oh, ow = frame.shape[:2]

        # STEP 1: YOLO
        yt = preprocess_yolo(frame)
        out = np.squeeze(self.yolo.run(None, {self.yn: yt})[0], axis=0)

        raw = []
        for col in range(out.shape[1]):
            x, y, w, h = out[0:4, col]
            scores = out[4:, col]
            bi = np.argmax(scores)
            conf = scores[bi]
            if conf >= self.params["conf"] and bi in VEHICLE_COCO:
                xs, ys = ow / 1024, oh / 1024
                xmin = max(0, int((x - w / 2) * xs))
                ymin = max(0, int((y - h / 2) * ys))
                xmax = min(ow, int((x + w / 2) * xs))
                ymax = min(oh, int((y + h / 2) * ys))
                raw.append({"box": [xmin, ymin, xmax, ymax], "conf": float(conf)})

        refined = apply_custom_nms(raw, self.params["iou"])

        # STEP 2: 추적 + 분류
        current_tracks = {}
        render_jobs = []

        for pred in refined:
            x1, y1, x2, y2 = pred["box"]
            w, h = x2 - x1, y2 - y1
            if w < 40 or h < 40:
                continue

            # IoU 매칭
            best_id, best_iou = -1, 0.1
            for lid, lbox in self.last_frame_tracks.items():
                iou = calculate_iou(pred["box"], lbox)
                if iou > best_iou:
                    best_iou, best_id = iou, lid

            track_id = best_id if best_id != -1 else self.next_track_id
            if best_id == -1:
                self.next_track_id += 1

            current_tracks[track_id] = pred["box"]
            score = (w * h) * pred["conf"]

            if track_id not in self.track_registry:
                self.track_registry[track_id] = {
                    "best_score": 0, "is_classified": False, "lost_counter": 0,
                    "attrs": {"brand": "분석중...", "color": "", "model": ""},
                }
            else:
                self.track_registry[track_id]["lost_counter"] = 0

            reg = self.track_registry[track_id]

            # 분류 (1회)
            if not reg["is_classified"] and score > reg["best_score"]:
                reg["best_score"] = score
                if score > QUALITY_THRESHOLD:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        probs, pred_attrs = self._classify(roi)
                        reg["attrs"] = pred_attrs
                        reg["is_classified"] = True

                        # 일치율 계산
                        match = compute_match(probs, self.target, self.encoders)
                        self.match_results[track_id] = {
                            "track_id": track_id,
                            "pred": pred_attrs,
                            "match": match,
                            "frame": cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB),
                            "box": (x1, y1, x2, y2),
                        }

            # 일치 여부 판단 (빨간 박스 강조)
            is_match = False
            if track_id in self.match_results:
                if self.match_results[track_id]["match"]["overall"] >= (st.session_state.get("threshold", 60) / 100.0):
                    is_match = True

            # 박스 색상
            box_color = (255, 0, 0) if is_match else (0, 255, 100)  # 빨강 vs 초록 (RGB 기준은 아래 PIL에서)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 0, 255) if is_match else (0, 255, 100),  # BGR
                          3 if is_match else 2)

            attrs = reg["attrs"]
            if reg["is_classified"]:
                label = f"[{track_id}] {attrs['brand']} {attrs['color']} {attrs['model']}"
                if is_match:
                    label = "🎯 " + label
                txt_col = (255, 0, 0) if is_match else (255, 255, 0)
                bg_col = (200, 0, 0) if is_match else (0, 100, 100)
            else:
                label = f"[{track_id}] 분석 대기중..."
                txt_col, bg_col = (255, 150, 0), (0, 50, 100)

            bw = len(label) * 12 + 15
            cv2.rectangle(frame, (x1, y1 - 32),
                          (x1 + bw, y1),
                          (0, 0, 200) if is_match else (100, 100, 0), -1)  # BGR
            render_jobs.append((label, (x1 + 5, y1 - 28), txt_col))

        # 유실 트랙 정리
        for oid in list(self.track_registry.keys()):
            if oid not in current_tracks:
                self.track_registry[oid]["lost_counter"] += 1
                if self.track_registry[oid]["lost_counter"] > MAX_LOST_FRAMES:
                    del self.track_registry[oid]

        self.last_frame_tracks = current_tracks

        # 한글 텍스트 PIL 렌더링
        if render_jobs:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for txt, pos, col in render_jobs:
                draw.text(pos, txt, font=self.font, fill=col)
            frame_rgb = np.array(pil)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        stats = {
            "total_tracks": self.next_track_id - 1,
            "classified": sum(1 for r in self.track_registry.values() if r["is_classified"]),
            "matches": sum(
                1 for m in self.match_results.values()
                if m["match"]["overall"] >= (st.session_state.get("threshold", 60) / 100.0)
            ),
        }
        return frame_rgb, stats

    def get_all_results(self):
        """탐색 후 분류된 모든 차량 결과 반환"""
        return list(self.match_results.values())