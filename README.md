# CCTV_TrackingSystem
고급컴퓨터비전_프로젝트

# 🚨 실시간 뺑소니 차량 탐색 시스템

CCTV 영상에서 용의 차량을 실시간으로 추적·식별하는 AI 기반 차량 관제 시스템입니다.
목격된 차량의 속성(색상·제조사·세부 모델)을 입력하면, 업로드한 CCTV 영상에서
일치하는 차량을 실시간 바운딩 박스로 강조하여 찾아냅니다.

---

## 📌 프로젝트 소개

뺑소니 사고가 발생했을 때, 목격자 진술로 확보한 차량 정보와 주변 CCTV 영상을 결합해
용의 차량을 자동으로 탐지·추적·식별하는 데모 시스템입니다.

**핵심 파이프라인:** 객체 탐지(YOLOv8n) → 다중 객체 추적(IoU 기반) → 속성 분류(ConvNeXt-Tiny + CBAM) → 일치율 매칭

### 주요 기능
- 찾을 차량의 색상 / 제조사 / 세부 모델 입력
- CCTV 영상 업로드 후 **실시간 바운딩 박스 스트리밍** (분석 과정을 화면에서 직접 확인)
- 입력 차량과 일치하는 차량은 🎯 빨간 박스로 강조 표시
- 기상 환경(맑음·박무·비·눈)에 따른 탐지 임계값 동적 제어
- 일치율 기준 후보 차량 정렬 출력

### 실행 화면
> [실행 캡처 이미지를 여기에 삽입하세요 — 예: `docs/screenshot_main.png`, `docs/screenshot_detect.png`]

---

## 🛠 개발 환경 및 의존성

### 개발 환경
- **운영체제**: Windows 10/11, macOS, Linux (크로스 플랫폼 지원)
- **Python**: 3.9 이상 권장
- **실행 방식**: Streamlit 기반 웹 애플리케이션 (로컬 `localhost:8501`)
- **추론 엔진**: ONNX Runtime (CUDA / CoreML / CPU 자동 감지·할당)
- **하드웨어**: GPU 없이 CPU만으로도 구동 가능 (저사양 노트북 호환)

### 핵심 의존성 (`requirements.txt`)
| 패키지 | 용도 |
| --- | --- |
| `streamlit` | 웹 UI 및 실시간 영상 스트리밍 인터페이스 |
| `onnxruntime` | YOLOv8n / ConvNeXt ONNX 모델 추론 |
| `opencv-python-headless` | 영상 프레임 처리 및 전처리 |
| `numpy` | 행렬 연산 (NMS, IoU, softmax 등) |
| `Pillow` | 한글 라벨 텍스트 렌더링 |
| `gdown` | 대용량 가중치 파일 자동 다운로드 |

```txt
streamlit>=1.32.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
onnxruntime>=1.16.0
Pillow>=10.0.0
gdown>=5.0.0
```

> 본 프로젝트는 객체 탐지에 순수 ONNX 추론을 사용하므로 `ultralytics`, `torch`, `lapx` 등
> 무거운 학습용 프레임워크 의존성이 없습니다. 따라서 설치가 가볍고 빠릅니다.

---

## ⚙️ 설치 및 실행 방법

### 1. 저장소 복제
```bash
git clone https://github.com/eunkyoung529/CCTV_TrackingSystem.git
cd CCTV_TrackingSystem
```

### 2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 대용량 가중치 파일 다운로드
`convnext_cbam_multihead_phase4.onnx`(약 107MB)는 GitHub 용량 제한(100MB)으로
구글 드라이브에 보관되어 있습니다. 아래 중 한 가지 방법으로 받으세요.

**방법 A — 자동 다운로드**
```bash
python download_model.py
```

**방법 B — 수동 다운로드**
아래 링크에서 파일을 받아 `models/` 폴더에 넣으세요.
https://drive.google.com/file/d/1rO0DG_1ss2ufYbWaJhQPcmR0wn-hYV4L/view?usp=sharing

> ※ 앱 첫 실행 시 모델 파일이 없으면 자동으로 다운로드되는 안전장치가 적용되어 있어,
> 이 단계를 건너뛰고 바로 5번을 실행해도 최초 1회 자동으로 받아집니다.

### 5. 웹 앱 실행
```bash
streamlit run app.py
```
실행 후 브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

### 📁 폴더 구조
```
[레포이름]/
├── app.py                                    # 메인 엔트리 (인트로 + 진행 상황)
├── download_model.py                         # 대용량 가중치 자동 다운로드 스크립트
├── requirements.txt                          # 의존성 목록
├── README.md
├── .gitignore
├── label_encoders.json                       # 클래스 라벨 인코더
├── models/
│   ├── yolov8n.onnx                          # 객체 탐지 모델 (GitHub 포함, 12MB)
│   └── convnext_cbam_multihead_phase4.onnx   # 속성 분류 모델 (다운로드 후 생성, 107MB)
├── pages/
│   ├── 1_차량정보입력.py                      # 찾을 차량 속성 입력
│   ├── 2_영상업로드_탐색.py                   # 영상 업로드 + 실시간 탐색
│   └── 3_결과출력.py                          # 일치 차량 후보 출력
├── core/
│   ├── __init__.py
│   ├── engine.py                             # ONNX 추론 엔진 (탐지+추적+분류)
│   ├── matching.py                           # 일치율 계산 로직
│   └── state.py                              # 세션 상태 관리
└── assets/
    └── style.css                             # 커스텀 스타일
```

---

## 🔄 데이터 파이프라인

본 시스템은 **객체 탐지 → 다중 객체 추적 → 속성 분류 → 일치율 매칭**의 단계로 구성됩니다.

### 학습 데이터셋 개요
- **JSON 라벨 파일**: 406,554개
- **이미지–라벨 데이터 쌍**: 376,731개
- **클래스 수**: 차종(brand) 29개, 색상(color) 10개, 모델(model) 223개
- **데이터 분할**: Train 80% / Validation 10% / Test 10%
  - 계층적 분할(Stratified Sampling)로 롱테일 클래스 불균형 보정
  - 난수 시드 고정(`seed=42`)으로 Data Leakage 원천 차단

### 추론 파이프라인 (실시간 동작 흐름)
```
[사용자 입력]  찾을 차량 속성 (색상 / 제조사 / 세부 모델)
      │
      ▼
[영상 업로드]  CCTV 영상 1개 업로드 → 임시 파일로 저장
      │
      ▼
[STEP 1] 객체 탐지 — YOLOv8n (ONNX, 1024×1024 입력)
   · 프레임을 1024 스케일로 업샘플링하여 소형 차량 미탐 방지
   · COCO 차량 클래스(2,3,5,7)만 필터링
      │
      ▼
[STEP 2] 후처리 — 커스텀 NMS + 기상별 동적 임계값
   · 기상 도메인(맑음/박무/비/눈)에 따라 conf·IoU 임계값 동적 스위칭
   · IoU 기반 NMS로 중복 박스(유령 박스) 제거
      │
      ▼
[STEP 3] 다중 객체 추적 — IoU 기반 프레임 간 ID 매칭
   · 직전 프레임 박스와 IoU 매칭으로 동일 차량에 Track ID 유지
   · 유실 트랙은 일정 프레임 후 메모리에서 제거
      │
      ▼
[STEP 4] 속성 분류 — ConvNeXt-Tiny + CBAM (ONNX, 224×224 입력)
   · 트랙별 품질 점수가 가장 높은 시점의 ROI를 1회 분류
   · 멀티헤드 출력: brand / color / model (softmax 확률)
      │
      ▼
[STEP 5] 일치율 매칭
   · 사용자 입력 속성과 모델 예측 확률을 가중 평균(model 0.4 / color 0.3 / brand 0.3)
   · 임계값 이상 차량은 실시간 화면에서 🎯 빨간 박스로 강조
      │
      ▼
[결과 출력]  일치율 순으로 후보 차량 카드 정리 출력
```

### 채택 모델
| 모듈 | 채택 모델 | 성능 |
| --- | --- | --- |
| 객체 탐지 + 추적 | 1024 스케일 YOLOv8n + 기상 도메인 적응 제어 (Phase 4) | mAP50 46.12% (악천후 강인성 확보), ID Switch 86.3% 감소 |
| 속성 분류 | ConvNeXt-Tiny + CBAM + Focal Loss(γ=2.0) | brand 99.48% / color 94.21% / model 95.23% Acc, model Macro F1 64.19% |

### 모델 고도화 과정 (Ablation Study)
각 단계별로 기법을 누적 적용하며 성능 향상을 검증했습니다.

**속성 분류 (ConvNeXt 계열) — model Macro F1 기준**
| 단계 | 적용 기법 | model F1 |
| --- | --- | --- |
| Baseline | 기본 구조 + Random Split | 58.71% |
| Augmentation | 계층적 분할 + Random Erasing + Gaussian Blur | 60.93% |
| Architecture | 백본 교체(ConvNeXt) + CBAM 어텐션 | 60.01% |
| Optimization | Cosine Annealing + 가중 손실함수(γ=2.0) | **64.19%** |

**객체 탐지 (YOLOv8n) — mAP50 기준**
| 단계 | 적용 기법 | mAP50 |
| --- | --- | --- |
| Phase 1 (Baseline) | 기본 YOLOv8n | 7.48% |
| Phase 2 (Scale) | 기하학적 변형 + 1024 해상도 확장 | 10.66% |
| Phase 3 (Optimization) | NMS + Confidence Grid Search | 57.16% |
| Phase 4 (Adaptation) | 기상 도메인별 동적 임계값 제어 | **46.12%** (악천후 강인성) |

---

## 👥 팀원별 역할 분담

| 이름 | 담당 역할 |
| --- | --- |
| 김은경 | 데이터 전처리, Streamlit UI 구현 |
| 김민정 | YOLOv8n 탐지 모델 학습, IoU 기반 다중 객체 추적 구현 |

---

## 📎 참고 사항
- 시연용 영상은 `pages/2_영상업로드_탐색.py`에서 직접 업로드하여 사용합니다.
- 긴 영상은 앞부분 일정 길이(기본 10초)만 분석하도록 설정되어 있습니다.
- 기상 환경은 메인 페이지 좌측 사이드바에서 선택할 수 있습니다.