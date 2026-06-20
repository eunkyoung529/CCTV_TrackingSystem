# download_model.py
"""
GitHub 용량 제한(100MB)을 초과하는 ConvNeXt 가중치를
구글 드라이브에서 models/ 폴더로 자동 다운로드.
"""
import os
import gdown

MODELS_DIR = "models"
CONVNEXT_GDRIVE_ID = "1rO0DG_1ss2ufYbWaJhQPcmR0wn-hYV4L"
CONVNEXT_FILENAME = "convnext_cbam_multihead_phase4.onnx"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, CONVNEXT_FILENAME)

    if os.path.exists(dest):
        print(f"✅ 이미 존재합니다: {dest}")
        return

    print(f"⬇️  ConvNeXt 가중치 다운로드 중... → {dest}")
    url = f"https://drive.google.com/uc?id={CONVNEXT_GDRIVE_ID}"
    gdown.download(url, dest, quiet=False)
    print("✅ 다운로드 완료!")


if __name__ == "__main__":
    main()