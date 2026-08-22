"""
data_prep/download_assets.py

Downloads the MediaPipe HandLandmarker `.task` model file used by the
modern MediaPipe Tasks API (mp.tasks.vision.HandLandmarker), which replaces
the older/deprecated `mp.solutions.hands` API.

Run this once before extract_landmarks.py or infer_webcam.py:
    python data_prep/download_assets.py

If your machine has no direct internet access, download the file manually
from the URL printed below and place it at the path printed below.
"""
import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    os.makedirs(os.path.dirname(config.MEDIAPIPE_MODEL_PATH), exist_ok=True)
    if os.path.exists(config.MEDIAPIPE_MODEL_PATH):
        print(f"already present: {config.MEDIAPIPE_MODEL_PATH}")
        return
    print(f"downloading {config.MEDIAPIPE_MODEL_URL}")
    print(f"        -> {config.MEDIAPIPE_MODEL_PATH}")
    try:
        urllib.request.urlretrieve(config.MEDIAPIPE_MODEL_URL, config.MEDIAPIPE_MODEL_PATH)
        print("done.")
    except Exception as e:
        print(f"[error] automatic download failed: {e}")
        print("Manually download the file from the URL above and save it to the path above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
