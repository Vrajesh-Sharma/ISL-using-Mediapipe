# ISL Hand-Sign Recognition — Complete Guide

## Project Overview
This repository builds a **36-class Indian Sign Language recognizer** (digits 0‑9, letters A‑Z) that **fuses two modalities**:

1. **Image branch** – a lightweight pretrained CNN (MobileNetV3‑Small/Large or EfficientNet‑B0) on a hand‑cropped RGB image.
2. **Landmark branch** – a small MLP on MediaPipe hand‑landmark geometric features (179‑dim vector: 21×3 coords + fingertip‑to‑wrist distances + fingertip‑to‑palm distances + 15 joint angles + handedness flag).

The two embeddings are combined via a **learned gate** (per‑branch scalar) and fed to a classifier. The gate lets the model down‑weight the image branch when MediaPipe fails (landmark vector is all‑zeros) and vice‑versa, making it robust to the large background/lighting generalization gap between the clean training set and the unseen test set.

## Directory Structure
```
ISL-using-Mediapipe/
├─ assets/                # hand_landmarker.task model (downloaded)
├─ cache/                 # extracted landmark caches (npz)
├─ checkpoints/           # trained model .pt files
├─ data/
│  ├─ train/              # 54k images, folder-per-class (0‑9, A‑Z)
│  └─ test_unseen/        # 36k disturbed‑background images
├─ data_prep/             # scripts to extract/ audit landmarks
├─ models/
│  └─ fusion_model.py     # two‑branch fusion model
├─ outputs/               # logs, classification reports, confusion matrices
├─ utils/
│  ├─ augmentations.py    # train/val augmentations
│  └─ landmark_features.py# feature engineering
├─ config.py              # paths, hyper‑parameters, label map
├─ train.py               # training loop
├─ evaluate.py            # evaluation + reports
├─ infer_webcam.py        # real‑time webcam demo
├─ requirements.txt
├─ README.md              # concise overview
└─ PROJECT_GUIDE.md       # this file
```

## Quick Start — How to Run the Whole Pipeline

### 1. Environment
```bash
# Clone / cd into the repo (already done)
cd E:\sem 7\CV\ISL-using-Mediapipe

# Create a fresh venv (recommended)
python -m venv .venv
.\.venv\Scripts\activate.exe

# Install dependencies
pip install -r requirements.txt --break-system-packages   # drop flag if not needed on Linux/macOS
```

### 2. Download MediaPipe hand‑landmarker model
```bash
python data_prep/download_assets.py
# If no internet, manually download from the URL in config.py and place at ./assets/hand_landmarker.task
```

### 3. Prepare the data (if you have your own ISL image folders)
- **Expected layout**: `data/train/0/xxx.jpg`, `data/train/1/xxx.jpg`, … `data/train/A/xxx.jpg`, … `data/test_unseen/` same layout.
- **Extract landmarks** (one‑time, offline — this is the expensive step):
  ```bash
  python data_prep/extract_landmarks.py --data_dir data/train --split train
  python data_prep/extract_landmarks.py --data_dir data/test_unseen --split test
  ```
- **Audit detection failure rate** (helps decide if you need to lower confidence thresholds):
  ```bash
  python data_prep/audit_no_detect.py --split train
  python data_prep/audit_no_detect.py --split test --copy_to ./no_detect_test   # optional
  ```

### 4. Train the model
```bash
python train.py
```
- First `FREEZE_BACKBONE_EPOCHS` (default **2**) epochs train the landmark branch + fusion head only (CNN backbone frozen).
- After epoch 2 the backbone unfreezes; training proceeds with cosine LR schedule, AMP, label smoothing, class‑balanced sampling.
- Checkpoints are saved as `checkpoints/best_model.pt` based on **validation macro‑F1**.
- Training history: `outputs/train_history.json`.
- On CPU this takes a while; a GPU is strongly recommended. Adjust `BATCH_SIZE`, `NUM_WORKERS`, `IMAGE_BACKBONE` in `config.py` for your hardware.

### 5. Evaluate on val / test splits
```bash
python evaluate.py --split val      # sanity check (should look good)
python evaluate.py --split test     # the number that actually matters
```
- Produces in `outputs/`:
  - `{split}_classification_report.txt`
  - `{split}_confusion_matrix.csv` + `.png`
  - Focused confusion report for `config.CONFUSABLE_GROUPS`.

### 6. Real‑time webcam demo
```bash
python infer_webcam.py --camera 0
```
- Press **q** to quit.
- `--min_conf` raises/lower MediaPipe’s detection threshold; `--smooth_window` (default 7) majority‑votes recent predictions to reduce flicker.

### 7. (Optional) Webcam inference with a custom checkpoint
If you want to use a different checkpoint than `best_model.pt`, edit `infer_webcam.py` line 31 or pass a different path.

## Architecture Details (for reference)
- **Image backbone**: `build_backbone()` in `models/fusion_model.py` strips the classifier head, outputs a 1280‑dim (small) / 1280‑dim (large) / 1280‑dim (efficientnet‑b0) feature vector.
- **Landmark feature dim**: 179 (see `config.py` comment). Engineered in `utils/landmark_features.py`.
- **Fusion**: concatenation → MLP with a **learned gate** (`gate = softmax(Linear([img_emb, lm_emb]))`). The gate produces two scalars (one per branch) that multiply their respective embeddings before the final classifier.
- **Training tricks**:
  - Landmark dropout (`LANDMARK_DROPOUT_PROB` ≈ 0.35) randomly zeroes landmark vectors each batch, teaching the model to rely on the image branch when MediaPipe misses.
  - Class‑balanced `WeightedRandomSampler` so every class contributes roughly equally per batch.
  - Early stopping on **validation macro‑F1**, not accuracy.

## Tuning Tips
| What to change | Why |
|---|---|
| `config.IMAGE_BACKBONE` = `mobilenet_v3_large` / `efficientnet_b0` | Trade speed vs accuracy if you have GPU headroom. |
| `config.CROP_MARGIN_RATIO` | Tighten/loosen the hand crop around the landmarks. |
| `config.MP_MIN_HAND_DETECTION_CONFIDENCE` | Lower if many test images have faint hands. |
| `utils/augmentations.py` | Ramp up brightness/contrast/jitter if the val‑test macro‑F1 gap is large. |
| `config.LM_EMBED_DIM` | Increase if the landmark branch seems dominated by the image branch. |

## License
This project is intended for educational / research purposes. See individual third‑party licences (MediaPipe, torch, etc.).