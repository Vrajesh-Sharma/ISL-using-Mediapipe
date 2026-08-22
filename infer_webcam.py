"""
infer_webcam.py

Live webcam demo: MediaPipe Hands (video-tracking mode, faster than
static_image_mode) detects the hand every frame -> crop with the same
margin used at train time -> same landmark feature engineering -> fused
model prediction, smoothed over a short rolling window to reduce flicker.

Usage:
    python infer_webcam.py --camera 0
Press 'q' to quit.
"""
import argparse
import collections
import time
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import config
from models.fusion_model import FusionISLModel
from utils.landmark_features import build_full_landmark_feature
from utils.augmentations import build_eval_transform


def load_model(device):
    ckpt = torch.load(f"{config.CHECKPOINT_DIR}/best_model.pt", map_location=device)
    model = FusionISLModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def hand_bbox_from_landmarks(landmarks_px, img_w, img_h, margin_ratio):
    xs = [p[0] for p in landmarks_px]; ys = [p[1] for p in landmarks_px]
    x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_ratio, h * margin_ratio
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(img_w, x2 + mx), min(img_h, y2 + my)
    return int(x1), int(y1), int(x2), int(y2)


def union_bbox(b1, b2):
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))


def main(camera_idx, min_conf, smooth_window):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    transform = build_eval_transform(config.IMAGE_SIZE)

    if not os.path.exists(config.MEDIAPIPE_MODEL_PATH):
        raise FileNotFoundError(
            f"{config.MEDIAPIPE_MODEL_PATH} not found — run `python data_prep/download_assets.py` first."
        )
    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.MEDIAPIPE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,   # tracking-assisted, faster frame-to-frame than IMAGE mode
        num_hands=config.MP_MAX_NUM_HANDS,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=config.MP_MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_idx}")

    pred_history = collections.deque(maxlen=smooth_window)
    fps_t0, fps_count, fps = time.time(), 0, 0.0
    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - t_start) * 1000)
        frame_idx += 1
        result = detector.detect_for_video(mp_image, timestamp_ms)

        label_text = "no hand"
        if result.hand_landmarks:
            hands_detected, bbox = [], None
            for hand_lms, handed in zip(result.hand_landmarks,
                                         result.handedness or [None] * len(result.hand_landmarks)):
                pts_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                pts_px = [(lm.x * w, lm.y * h) for lm in hand_lms]
                score = handed[0].score if handed is not None else 1.0
                hands_detected.append({"landmarks": pts_norm, "score": score})
                bb = hand_bbox_from_landmarks(pts_px, w, h, config.CROP_MARGIN_RATIO)
                bbox = bb if bbox is None else union_bbox(bbox, bb)
                for conn in HAND_CONNECTIONS:
                    p1, p2 = hand_lms[conn.start], hand_lms[conn.end]
                    cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 200, 0), 2)

            x1, y1, x2, y2 = bbox
            crop = rgb[y1:y2, x1:x2]
            if crop.size > 0:
                lm_feat = build_full_landmark_feature(hands_detected, config.LANDMARK_FEATURE_DIM_PER_HAND)
                img_t = transform(image=crop)["image"].unsqueeze(0).to(device)
                lm_t = torch.from_numpy(lm_feat.astype(np.float32)).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = model.predict_proba(img_t, lm_t)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
                pred_history.append(pred_idx)
                # majority vote over the smoothing window -> less flicker on borderline frames
                smoothed = max(set(pred_history), key=pred_history.count)
                label_text = f"{config.IDX_TO_CLASS[smoothed]}  ({probs[pred_idx]*100:.0f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps = fps_count / (time.time() - fps_t0)
            fps_count, fps_t0 = 0, time.time()

        cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"{fps:.1f} FPS", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("ISL Recognition (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min_conf", type=float, default=0.6)
    ap.add_argument("--smooth_window", type=int, default=7)
    args = ap.parse_args()
    main(args.camera, args.min_conf, args.smooth_window)
