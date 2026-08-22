"""
data_prep/dataset.py

Reads the manifest/landmarks cache produced by extract_landmarks.py and
serves (image_crop_tensor, landmark_feature_tensor, label) triples.

The hand crop (with margin) is computed once by MediaPipe and cached as a
bounding box, not as a saved image — this avoids duplicating ~90k images on
disk while still cropping out most irrelevant background before the CNN
ever sees the picture.
"""
import os
import sys
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.augmentations import build_train_transform, build_eval_transform, jitter_landmark_vector


def load_manifest(split):
    prefix = os.path.join(config.CACHE_DIR, split)
    rows = []
    with open(prefix + "_manifest.csv", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    landmarks = np.load(prefix + "_landmarks.npy")
    assert len(rows) == landmarks.shape[0], "manifest/landmarks row mismatch — re-run extract_landmarks.py"
    return rows, landmarks


class ISLDataset(Dataset):
    def __init__(self, rows, landmarks, indices, train, image_size=None):
        self.rows = rows
        self.landmarks = landmarks
        self.indices = indices
        self.train = train
        image_size = image_size or config.IMAGE_SIZE
        self.transform = build_train_transform(image_size) if train else build_eval_transform(image_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row_idx = self.indices[i]
        row = self.rows[row_idx]
        img_bgr = cv2.imread(row["filepath"])
        if img_bgr is None:
            # extremely defensive fallback — shouldn't happen if extraction ran cleanly
            img_bgr = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h
        crop = img_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        aug = self.transform(image=crop_rgb)
        img_t = aug["image"]

        lm = self.landmarks[row_idx].copy()
        if self.train:
            lm = jitter_landmark_vector(lm, sigma=0.008, p=0.5)
        lm_t = torch.from_numpy(lm.astype(np.float32))

        label_str = row["label"]
        label = config.CLASS_TO_IDX[label_str] if label_str in config.CLASS_TO_IDX else -1
        return img_t, lm_t, label


def stratified_train_val_split(rows, val_split, seed):
    """Simple per-class stratified split so rare classes still get val coverage."""
    rng = np.random.RandomState(seed)
    by_class = {}
    for idx, r in enumerate(rows):
        by_class.setdefault(r["label"], []).append(idx)
    train_idx, val_idx = [], []
    for cls, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_split))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def class_counts(rows, indices):
    counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    for idx in indices:
        lbl = rows[idx]["label"]
        if lbl in config.CLASS_TO_IDX:
            counts[config.CLASS_TO_IDX[lbl]] += 1
    return counts
