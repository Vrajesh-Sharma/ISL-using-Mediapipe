"""
data_prep/extract_landmarks.py

Runs MediaPipe Hands ONCE over an entire image folder-tree and caches:
  - manifest.csv   : filepath, label, crop bbox (pixels), num_hands_detected
  - landmarks.npy  : (N, LANDMARK_FEATURE_DIM) float32, row-aligned with manifest

This is a one-time offline pass so training doesn't re-run MediaPipe every
epoch (MediaPipe on CPU is the bottleneck otherwise). Re-run this script any
time the source images change.

Usage:
    python data_prep/extract_landmarks.py --data_dir /path/to/isl_train --split train
    python data_prep/extract_landmarks.py --data_dir /path/to/isl_test_unseen --split test --unlabeled
"""
import argparse
import os
import sys
import csv
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.landmark_features import build_full_landmark_feature

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(data_dir, unlabeled=False):
    """
    Labeled layout:   data_dir/<class_name>/*.jpg
    Unlabeled layout: data_dir/*.jpg  (flat)
    Returns list of (filepath, label_or_None)
    """
    items = []
    if unlabeled:
        for fn in sorted(os.listdir(data_dir)):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                items.append((os.path.join(data_dir, fn), None))
        return items

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in config.CLASS_TO_IDX:
            print(f"[warn] folder '{class_name}' not in CLASSES, skipping")
            continue
        for fn in sorted(os.listdir(class_dir)):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                items.append((os.path.join(class_dir, fn), class_name))
    return items


def hand_bbox_from_landmarks(landmarks_px, img_w, img_h, margin_ratio):
    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_ratio, h * margin_ratio
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(img_w, x2 + mx), min(img_h, y2 + my)
    return int(x1), int(y1), int(x2), int(y2)


def union_bbox(b1, b2):
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))


def process(data_dir, split, unlabeled):
    items = list_images(data_dir, unlabeled)
    print(f"[{split}] found {len(items)} images under {data_dir}")

    if not os.path.exists(config.MEDIAPIPE_MODEL_PATH):
        raise FileNotFoundError(
            f"{config.MEDIAPIPE_MODEL_PATH} not found — run `python data_prep/download_assets.py` first."
        )
    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.MEDIAPIPE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=config.MP_MAX_NUM_HANDS,
        min_hand_detection_confidence=config.MP_MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MP_MIN_HAND_PRESENCE_CONFIDENCE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    manifest_rows = []
    landmark_feats = []
    n_no_detect = 0

    for idx, (fp, label) in enumerate(tqdm(items, desc=f"MediaPipe [{split}]")):
        img_bgr = cv2.imread(fp)
        if img_bgr is None:
            print(f"[warn] unreadable image, skipping: {fp}")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)

        hands_detected = []
        bbox = None
        if result.hand_landmarks:
            for hand_lms, handed in zip(result.hand_landmarks,
                                         result.handedness or [None] * len(result.hand_landmarks)):
                pts_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                pts_px = [(lm.x * w, lm.y * h) for lm in hand_lms]
                score = handed[0].score if handed is not None else 1.0
                hands_detected.append({"landmarks": pts_norm, "score": score})
                bb = hand_bbox_from_landmarks(pts_px, w, h, config.CROP_MARGIN_RATIO)
                bbox = bb if bbox is None else union_bbox(bbox, bb)
        else:
            n_no_detect += 1
            # Fallback: no hand found -> use full image as crop, zero landmark features.
            bbox = (0, 0, w, h)

        feat = build_full_landmark_feature(hands_detected, config.LANDMARK_FEATURE_DIM_PER_HAND)
        landmark_feats.append(feat)
        manifest_rows.append({
            "idx": idx, "filepath": fp, "label": label if label is not None else "",
            "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
            "num_hands": len(hands_detected),
            "img_w": w, "img_h": h,
        })

    detector.close()

    out_prefix = os.path.join(config.CACHE_DIR, split)
    with open(out_prefix + "_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    np.save(out_prefix + "_landmarks.npy", np.stack(landmark_feats))

    print(f"[{split}] done. {n_no_detect}/{len(items)} images had NO hand detected "
          f"({100 * n_no_detect / max(1, len(items)):.1f}%). "
          f"Consider inspecting these separately (see data_prep/audit_no_detect.py).")
    print(f"[{split}] cached -> {out_prefix}_manifest.csv , {out_prefix}_landmarks.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test", "custom"])
    ap.add_argument("--unlabeled", action="store_true", help="flat folder, no class subfolders / no labels")
    args = ap.parse_args()
    process(args.data_dir, args.split, args.unlabeled)
