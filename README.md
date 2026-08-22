# ISL Hand-Sign Recognition — Landmark + Image Fusion Pipeline

36-class recognizer (digits 0–9, letters A–Z) for Indian Sign Language,
built to generalize from a clean ~54k-image training set to a genuinely
unseen ~36k-image test set with disturbed/varied backgrounds, and to run
in real time on a webcam.

## Why this architecture (short version)

A plain image classifier trained on a clean dataset tends to quietly learn
"background texture / lighting → class" shortcuts instead of "finger
configuration → class". Two things in this pipeline specifically fight that:

1. **MediaPipe hand landmarks as an explicit, separate input.** The
   landmark branch never sees pixels — only wrist-centered, scale-normalized
   (x,y,z) coordinates plus fingertip distances and joint angles. That
   representation is mathematically incapable of encoding background or
   lighting, so it forces at least part of the model's decision to be about
   hand *shape*.
2. **Tight MediaPipe-derived crop + heavy background/lighting augmentation
   on the image branch.** The CNN still sees pixels (useful for skin-fold /
   depth cues the 21 sparse landmarks miss), but its input is cropped down
   to mostly-hand, and training deliberately varies brightness, contrast,
   hue, noise, blur, JPEG artifacts, shadows, and patchy occlusion so the
   CNN can't rely on any single background/lighting cue either.

The two branches are fused with a small learned gate (not just concatenation),
so if MediaPipe fails to find a hand on a hard test image (all-zero landmark
vector), the model can lean more on the image branch instead of being
confused by a meaningless landmark input.

## 1. Install

```bash
cd isl_recognition
pip install -r requirements.txt --break-system-packages   # drop the flag if not needed on your system
python data_prep/download_assets.py                        # downloads hand_landmarker.task (~10MB)
```

If `download_assets.py` fails because your machine has no internet access to
`storage.googleapis.com`, download it manually from the URL it prints and
place it at `./assets/hand_landmarker.task`.

## 2. Point the code at your data

Edit the top of `config.py`:

```python
TRAIN_DIR = "/path/to/isl_train"        # ~54,000 images, one subfolder per class: 0/ 1/ ... 9/ A/ ... Z/
TEST_DIR  = "/path/to/isl_test_unseen"  # ~36,000 images, same layout (or flat + --unlabeled, see below)
```

Class folder names must match `config.CLASSES` exactly (`"0"`...`"9"`,
`"A"`...`"Z"`). If your folders are named differently, edit `CLASSES` or
rename the folders.

## 3. Extract MediaPipe landmarks (one-time, offline)

This is the expensive step — it runs MediaPipe once over every image and
caches the results, so training doesn't re-run hand detection every epoch.

```bash
python data_prep/extract_landmarks.py --data_dir /path/to/isl_train --split train
python data_prep/extract_landmarks.py --data_dir /path/to/isl_test_unseen --split test
```

If your test set has no class-subfolder structure (flat folder, no labels),
add `--unlabeled` — you'll still get predictions in `evaluate.py`/inference,
just no accuracy numbers.

Check how often MediaPipe failed to find a hand (motion blur, extreme
lighting, hand out of frame, etc.):

```bash
python data_prep/audit_no_detect.py --split train
python data_prep/audit_no_detect.py --split test --copy_to ./no_detect_test   # optional, copies the files out for a look
```

A handful of percent is normal and handled gracefully (fallback: full image
as crop, zero landmark vector — the image branch alone still carries these).
If it's high (>10-15%), it's usually worth lowering
`MP_MIN_HAND_DETECTION_CONFIDENCE` in `config.py` and re-running extraction.

## 4. Train

```bash
python train.py
```

- First `FREEZE_BACKBONE_EPOCHS` epochs train the landmark branch + fusion
  head only (CNN backbone frozen) — this warms up the head fast without
  wrecking the pretrained CNN weights with a randomly-initialized head's
  large early gradients.
- Backbone unfreezes after that; cosine LR schedule with warmup, AMP on GPU,
  label smoothing, class-balanced sampling (all 36 classes contribute
  roughly equally per batch even if your raw counts aren't perfectly even).
- Early stopping and checkpointing are both on **validation macro-F1**, not
  accuracy — this matters for a 36-way problem where a few classes being
  systematically confused (e.g. visually similar hand shapes) would hide
  behind a fine overall accuracy number.
- Val split is carved out of `TRAIN_DIR` only (`VAL_SPLIT`, default 12%,
  stratified per class). `TEST_DIR` is never touched by `train.py`.
- Best checkpoint: `checkpoints/best_model.pt`. Training curve:
  `outputs/train_history.json`.

Expect this to take a while on CPU — a GPU is strongly recommended for 54k
images. Adjust `BATCH_SIZE`, `NUM_WORKERS`, `IMAGE_BACKBONE` in `config.py`
to fit your hardware (`mobilenet_v3_small` is the default for speed;
`efficientnet_b0` trades some speed for accuracy if you have GPU headroom).

## 5. Evaluate — including the real target: the unseen test set

```bash
python evaluate.py --split val     # sanity check, should look good
python evaluate.py --split test    # the number that actually matters
```

Produces, in `outputs/`:
- `{split}_classification_report.txt` — per-class precision/recall/F1 + accuracy
- `{split}_confusion_matrix.csv` and `.png`
- a console-printed focused report on `config.CONFUSABLE_GROUPS` (edit this
  list once you've seen your actual confusion matrix — the defaults are
  just common ISL/ASL-style lookalikes to get you started)

**A meaningfully lower macro-F1 on `test` than `val` is the expected,
informative signal here** — it tells you how much the background/lighting
generalization gap actually is, which is the whole point of this exercise.
If the gap is large, the two biggest levers are (a) more aggressive image
augmentation in `utils/augmentations.py`, and (b) checking whether the
landmark branch's `gate` is being dominated by the image branch — if so,
the fusion head may need more training or `LM_EMBED_DIM` may need to go up
so the landmark branch has more capacity to compete.

## 6. Real-time webcam demo

```bash
python infer_webcam.py --camera 0
```

Press `q` to quit. `--min_conf` raises/lowers MediaPipe's detection
threshold; `--smooth_window` controls how many recent frames' predictions
get majority-voted to reduce flicker on borderline frames.

## Files you'll likely want to tune first

- `config.py` → `IMAGE_BACKBONE`, `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS` for
  your hardware; `CONFUSABLE_GROUPS` once you've seen real confusions.
- `utils/augmentations.py` → dial augmentation strength up/down depending on
  how big the val/test generalization gap turns out to be.
- `utils/landmark_features.py` → `normalize_landmarks()` intentionally does
  **not** rotate-normalize (hand orientation can be meaningful in ISL). If
  your data turns out to need orientation invariance instead, that's the
  function to change — the docstring explains the tradeoff.

## Design notes / things I deliberately did NOT do

- **No horizontal flip augmentation** — flipping a hand sign can silently
  turn it into a different (or invalid) sign, so it's off by default
  (`utils/augmentations.py`, `HorizontalFlip(p=0.0)`).
- **No background-image-pool compositing wired in by default** — the tight
  MediaPipe crop already removes most background, so this is coded as an
  opt-in hook (`paste_on_random_background` in `utils/augmentations.py`)
  rather than a default, since it needs you to supply a folder of arbitrary
  background images to be useful.
- **Landmarks are cached as a bounding box + feature vector, not as cropped
  image files** — avoids duplicating ~90k images on disk; the crop happens
  at data-loading time instead.
