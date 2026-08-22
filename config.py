"""
config.py — single source of truth for paths, hyperparameters, and label maps.
Edit the paths in Section 1 to match your machine before running anything.
"""
import os

# ------------------------------------------------------------------
# 1. PATHS — EDIT THESE
# ------------------------------------------------------------------
# Training data: expects a folder-per-class layout, e.g.
#   TRAIN_DIR/0/xxx.jpg, TRAIN_DIR/1/xxx.jpg, ..., TRAIN_DIR/A/xxx.jpg, ... TRAIN_DIR/Z/xxx.jpg
TRAIN_DIR = "data/train"          # ~54,000 clean images, 36 class folders
TEST_DIR = "data/test_unseen"     # ~36,000 disturbed-background images, same layout
                                           # (if test is unlabeled, see infer_folder.py)

CACHE_DIR = "/tmp/cache"                     # where extracted landmarks/crops are cached
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./outputs"

for d in (CACHE_DIR, CHECKPOINT_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------
# 2. LABEL MAP — 36 classes: 0-9 then A-Z (edit if your folder names differ)
# ------------------------------------------------------------------
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)  # 36

# Visually-confusable class groups — used only for a focused confusion report,
# NOT used anywhere in training. Edit based on what you actually observe.
CONFUSABLE_GROUPS = [
    ["M", "N", "S", "T"],       # closed-fist finger-count signs
    ["U", "V", "R", "K"],       # two-finger variants
    ["1", "D"], ["2", "V"], ["3", "W"], ["6", "W"],
]

# ------------------------------------------------------------------
# 3. MEDIAPIPE (new Tasks API — mp.tasks.vision.HandLandmarker) / CROP SETTINGS
# ------------------------------------------------------------------
# The modern MediaPipe Tasks API needs a downloaded model file (a few MB).
# Run `python data_prep/download_assets.py` once, or download manually from
# MEDIAPIPE_MODEL_URL and place it at MEDIAPIPE_MODEL_PATH.
MEDIAPIPE_MODEL_PATH = "./assets/hand_landmarker.task"
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

MP_MIN_HAND_DETECTION_CONFIDENCE = 0.5
MP_MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5   # only used for video/webcam mode
MP_MAX_NUM_HANDS = 2                # ISL sometimes uses two hands; single-hand images just leave slot 2 zeroed

CROP_MARGIN_RATIO = 0.30            # expand tight landmark bbox by this fraction on each side before cropping
IMAGE_SIZE = 160                    # input resolution to the CNN branch (160 keeps webcam inference fast)

# Landmark feature vector size (see utils/landmark_features.py):
# per hand: 21*3 normalized coords (63) + 5 fingertip-to-wrist dist + 5 fingertip-to-palm dist
# + 15 joint angles + 1 handedness flag = 89 features per hand, x2 hands + 1 "both-hands-present" flag = 179
LANDMARK_FEATURE_DIM_PER_HAND = 89
LANDMARK_FEATURE_DIM = LANDMARK_FEATURE_DIM_PER_HAND * 2 + 1

# ------------------------------------------------------------------
# 4. TRAINING HYPERPARAMETERS
# ------------------------------------------------------------------
SEED = 42
VAL_SPLIT = 0.12                    # carved out of TRAIN_DIR only; TEST_DIR is never touched during training
BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 40
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 3
EARLY_STOP_PATIENCE = 8
IMAGE_BACKBONE = "mobilenet_v3_small"   # or "mobilenet_v3_large" / "efficientnet_b0"
FREEZE_BACKBONE_EPOCHS = 2              # train fusion head first, then unfreeze CNN backbone
MIXED_PRECISION = True
USE_CLASS_BALANCED_SAMPLER = True

# Fusion head dims
IMG_EMBED_DIM = 256
LM_EMBED_DIM = 128
FUSION_HIDDEN_DIM = 256
DROPOUT = 0.35
