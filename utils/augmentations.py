"""
utils/augmentations.py

Augmentation pipelines for the IMAGE branch. The landmark branch already gets
its own (much lighter) jitter directly inside dataset.py, since it operates
on numeric features rather than pixels.

Written against albumentations>=2.0 (see requirements.txt for the pinned
version this was validated against — parameter names changed noticeably
between albumentations 1.x and 2.x, e.g. GaussNoise's var_limit -> std_range,
CoarseDropout's max_holes/max_height/max_width -> num_holes_range/
hole_height_range/hole_width_range, ImageCompression's quality_lower/
quality_upper -> quality_range, ShiftScaleRotate -> Affine).

Design intent, mapped to the brief's requirements:
  - background robustness  -> CoarseDropout (patchy occlusion), RandomShadow,
                               Affine w/ border replicate+random fill,
                               background-swap hook (paste hand crop onto a
                               random background image, if a background pool
                               is supplied)
  - illumination robustness -> RandomBrightnessContrast, HueSaturationValue,
                               CLAHE, RandomGamma, ToGray (sometimes)
  - scale/rotation/translation -> Affine (scale + translate_percent + rotate)
  - camera/sensor noise      -> GaussNoise, ISONoise, ImageCompression, MotionBlur
"""
import random
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size):
    return A.Compose([
        A.LongestMaxSize(max_size=int(image_size * 1.15)),
        A.PadIfNeeded(min_height=int(image_size * 1.15), min_width=int(image_size * 1.15),
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.RandomCrop(height=image_size, width=image_size, p=0.6, pad_if_needed=True),
        A.Resize(image_size, image_size),
        A.Affine(translate_percent=(-0.08, 0.08), scale=(0.85, 1.15), rotate=(-20, 20),
                  border_mode=cv2.BORDER_REPLICATE, p=0.8),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
            A.RandomGamma(gamma_limit=(60, 160), p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.85),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ToGray(p=0.08),
        A.OneOf([
            A.GaussNoise(std_range=(0.05, 0.20), p=1.0),
            A.ISONoise(p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.ImageCompression(quality_range=(35, 90), p=1.0),
        ], p=0.4),
        A.CoarseDropout(num_holes_range=(1, 6),
                         hole_height_range=(0.03, 0.15), hole_width_range=(0.03, 0.15),
                         fill=0, p=0.4),
        A.RandomShadow(p=0.15),
        A.HorizontalFlip(p=0.0),  # OFF by default: flipping can turn a valid sign into an invalid/different one
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(image_size):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def paste_on_random_background(hand_rgb, hand_mask, background_pool):
    """
    Optional stronger background-robustness trick: if you have a folder of
    arbitrary background images (indoor/outdoor scenes, textures, etc.),
    pass their file list as `background_pool` and this will alpha-composite
    the (roughly-masked) hand crop onto a random background before the rest
    of the pipeline runs. Wire this into dataset.py's __getitem__ if you have
    such a background pool; skipped entirely if background_pool is empty,
    since a tight MediaPipe-based crop already suppresses most background.
    """
    if not background_pool:
        return hand_rgb
    bg_path = random.choice(background_pool)
    bg = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
    bg = cv2.resize(bg, (hand_rgb.shape[1], hand_rgb.shape[0]))
    mask = hand_mask.astype(np.float32)[..., None]
    return (hand_rgb.astype(np.float32) * mask + bg.astype(np.float32) * (1 - mask)).astype(np.uint8)


def jitter_landmark_vector(vec, sigma=0.01, p=0.5):
    """Small Gaussian jitter on the numeric landmark features (translation-free
    since features are already wrist-centered/scale-normalized) — mimics
    MediaPipe's own frame-to-frame jitter so the classifier isn't brittle to it."""
    if random.random() > p:
        return vec
    noise = np.random.normal(0, sigma, size=vec.shape).astype(np.float32)
    return vec + noise
