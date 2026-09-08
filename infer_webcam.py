"""
infer_webcam.py

Live webcam ISL recognition with:
- MediaPipe Hand Landmarker
- Image + landmark fusion model
- Rolling prediction smoothing
- Stable-sign detection
- Direct Windows SAPI5 text-to-speech

Usage:
    python infer_webcam.py --camera 0

Press Q to quit.
"""

import argparse
import collections
import os
import queue
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import config
from models.fusion_model import FusionISLModel
from utils.augmentations import build_eval_transform
from utils.landmark_features import build_full_landmark_feature


# ============================================================
# WINDOWS SAPI5 TEXT-TO-SPEECH
# ============================================================

def tts_worker(tts_queue):
    """
    Dedicated Windows SAPI5 speech worker.

    Uses pywin32 instead of pyttsx3.
    COM is initialized inside this worker because SAPI5 is
    a Windows COM component.
    """

    import pythoncom
    import win32com.client

    print("[TTS] Initializing Windows SAPI5...")

    # Initialize COM for this thread
    pythoncom.CoInitialize()

    speaker = None

    try:
        # Direct Windows SAPI5
        speaker = win32com.client.Dispatch("SAPI.SpVoice")

        # Speech settings
        speaker.Rate = 0
        speaker.Volume = 100

        # --------------------------------------------------------
        # List available voices
        # --------------------------------------------------------

        voices = speaker.GetVoices()

        print(f"[TTS] Available voices: {voices.Count}")

        for i in range(voices.Count):
            try:
                voice = voices.Item(i)
                description = voice.GetDescription()
            except Exception:
                description = "Unknown"

            print(f"[TTS] Voice {i}: {description}")

        # --------------------------------------------------------
        # Select an English voice
        # --------------------------------------------------------

        selected_voice = None

        for i in range(voices.Count):
            try:
                voice = voices.Item(i)
                description = voice.GetDescription()

                if (
                    "English" in description
                    or "David" in description
                    or "Zira" in description
                ):
                    selected_voice = voice
                    break

            except Exception:
                pass

        if selected_voice is not None:
            speaker.Voice = selected_voice
            print("[TTS] English voice selected.")

        print("[TTS] Speech engine ready.")

    except Exception as e:

        print(
            f"[TTS ERROR] Could not initialize Windows SAPI5: {e}"
        )

        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

        return

    # ========================================================
    # SPEECH LOOP
    # ========================================================

    while True:

        text = tts_queue.get()

        # ----------------------------------------------------
        # Shutdown signal
        # ----------------------------------------------------

        if text is None:

            tts_queue.task_done()
            break

        text = str(text).strip()

        if not text:

            tts_queue.task_done()
            continue

        try:

            print(f"[TTS] Speaking: {text}")

            # 0 = synchronous speech.
            #
            # The worker waits until the current speech
            # finishes before processing another item.
            speaker.Speak(text, 0)

            print(f"[TTS] Finished: {text}")

        except Exception as e:

            print(
                f"[TTS ERROR] "
                f"Could not speak '{text}': {e}"
            )

        finally:

            tts_queue.task_done()

    # ========================================================
    # CLEANUP
    # ========================================================

    try:

        # Stop any remaining speech
        speaker.Speak("", 2)

    except Exception:
        pass

    try:

        pythoncom.CoUninitialize()

    except Exception:
        pass

    print("[TTS] Worker stopped.")


# ============================================================
# MODEL
# ============================================================

def load_model(device):
    """
    Load the trained FusionISLModel.
    """

    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        "best_model.pt"
    )

    if not os.path.isfile(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print(
        f"[INFO] Loading checkpoint: "
        f"{checkpoint_path}"
    )

    # PyTorch 2.6+ changed the default of weights_only
    # from False to True.
    #
    # This checkpoint is trusted and contains objects
    # that require normal pickle loading.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = FusionISLModel(
        num_classes=config.NUM_CLASSES
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)
    model.eval()

    print("[INFO] Model loaded successfully.")

    return model


# ============================================================
# BOUNDING BOX
# ============================================================

def get_hand_bbox(
    hand_landmarks,
    frame_width,
    frame_height
):
    """
    Get bounding box around one detected hand.
    """

    xs = [
        landmark.x
        for landmark in hand_landmarks
    ]

    ys = [
        landmark.y
        for landmark in hand_landmarks
    ]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    # --------------------------------------------------------
    # Convert normalized coordinates to pixels
    # --------------------------------------------------------

    x1 = int(x_min * frame_width)
    y1 = int(y_min * frame_height)

    x2 = int(x_max * frame_width)
    y2 = int(y_max * frame_height)

    # --------------------------------------------------------
    # Add crop margin
    # --------------------------------------------------------

    margin_x = int(
        (x2 - x1) * config.CROP_MARGIN_RATIO
    )

    margin_y = int(
        (y2 - y1) * config.CROP_MARGIN_RATIO
    )

    x1 -= margin_x
    y1 -= margin_y

    x2 += margin_x
    y2 += margin_y

    # --------------------------------------------------------
    # Clamp to image boundaries
    # --------------------------------------------------------

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(frame_width - 1, x2)
    y2 = min(frame_height - 1, y2)

    return x1, y1, x2, y2


def get_union_bbox(
    hand_landmarks_list,
    frame_width,
    frame_height
):
    """
    Get one bounding box containing all detected hands.
    """

    boxes = []

    for hand_landmarks in hand_landmarks_list:

        box = get_hand_bbox(
            hand_landmarks,
            frame_width,
            frame_height
        )

        boxes.append(box)

    if not boxes:
        return None

    x1 = min(
        box[0]
        for box in boxes
    )

    y1 = min(
        box[1]
        for box in boxes
    )

    x2 = max(
        box[2]
        for box in boxes
    )

    y2 = max(
        box[3]
        for box in boxes
    )

    return x1, y1, x2, y2


# ============================================================
# CONVERT MEDIAPIPE LANDMARKS
# ============================================================

def convert_mediapipe_landmarks(
    result
):
    """
    Convert MediaPipe Tasks API output into the format
    expected by build_full_landmark_feature().

    Expected project format:

        [
            {
                "landmarks": [(x, y, z), ...],
                "score": float
            },
            ...
        ]
    """

    hands_for_features = []

    hands_detected = result.hand_landmarks

    for hand_idx, hand_landmarks in enumerate(
        hands_detected
    ):

        # ----------------------------------------------------
        # Convert NormalizedLandmark objects to tuples
        # ----------------------------------------------------

        landmarks_xyz = [
            (
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
            )
            for landmark in hand_landmarks
        ]

        # ----------------------------------------------------
        # Get handedness score
        # ----------------------------------------------------

        score = 1.0

        if (
            result.handedness
            and hand_idx < len(result.handedness)
            and result.handedness[hand_idx]
        ):

            try:

                score = float(
                    result.handedness[hand_idx][0].score
                )

            except Exception:

                score = 1.0

        # ----------------------------------------------------
        # Build project-compatible dictionary
        # ----------------------------------------------------

        hands_for_features.append(
            {
                "landmarks": landmarks_xyz,
                "score": score
            }
        )

    return hands_for_features


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index"
    )

    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.6,
        help="Minimum hand detection confidence"
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Prediction smoothing window"
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[INFO] Using device: {device}"
    )

    # ========================================================
    # LOAD FUSION MODEL
    # ========================================================

    model = load_model(device)

    # ========================================================
    # IMAGE EVALUATION TRANSFORM
    # ========================================================

    eval_transform = build_eval_transform(
        config.IMAGE_SIZE
    )

    # ========================================================
    # MEDIAPIPE MODEL
    # ========================================================

    mp_model_path = (
        config.MEDIAPIPE_MODEL_PATH
    )

    if not os.path.isfile(mp_model_path):

        raise FileNotFoundError(
            f"MediaPipe model not found:\n"
            f"{mp_model_path}"
        )

    print(
        f"[INFO] Loading MediaPipe model: "
        f"{mp_model_path}"
    )

    base_options = BaseOptions(
        model_asset_path=mp_model_path
    )

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,

        running_mode=(
            mp_vision.RunningMode.VIDEO
        ),

        num_hands=(
            config.MP_MAX_NUM_HANDS
        ),

        min_hand_detection_confidence=(
            args.min_conf
        ),

        min_hand_presence_confidence=(
            args.min_conf
        ),

        min_tracking_confidence=(
            args.min_conf
        )
    )

    detector = (
        mp_vision.HandLandmarker
        .create_from_options(options)
    )

    print(
        "[INFO] MediaPipe hand detector ready."
    )

    # ========================================================
    # CAMERA
    # ========================================================

    print(
        f"[INFO] Opening camera {args.camera}..."
    )

    cap = cv2.VideoCapture(
        args.camera
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"{args.camera}"
        )

    print(
        "[INFO] Camera opened successfully."
    )

    # ========================================================
    # TTS
    # ========================================================

    tts_queue = queue.Queue()

    tts_thread = threading.Thread(
        target=tts_worker,
        args=(tts_queue,),
        daemon=True
    )

    tts_thread.start()

    # ========================================================
    # PREDICTION SMOOTHING
    # ========================================================

    pred_history = collections.deque(
        maxlen=args.smooth_window
    )

    # ========================================================
    # STABLE SIGN SPEECH SETTINGS
    # ========================================================

    # Number of consecutive frames required
    # before a sign is considered stable.
    STABLE_FRAMES = 8

    # Minimum time between spoken signs.
    SPEECH_COOLDOWN = 1.0

    # Last sign that was actually spoken.
    last_spoken_class = ""

    # Current candidate sign.
    candidate_class = ""

    # Number of consecutive frames for candidate.
    candidate_count = 0

    # Time when last sign was spoken.
    last_spoken_time = 0.0

    # ========================================================
    # FPS
    # ========================================================

    fps_counter = 0
    fps = 0.0

    fps_start_time = time.time()

    # ========================================================
    # MEDIAPIPE TIMESTAMP
    # ========================================================

    frame_timestamp_ms = 0

    # ========================================================
    # HAND CONNECTIONS
    # ========================================================

    # Standard 21-point MediaPipe hand connections.
    #
    # We define them manually because the installed
    # MediaPipe version uses the Tasks API and does not
    # expose mp.solutions.hands.

    hand_connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index finger
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle finger
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring finger
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (5, 9),
        (9, 13),
        (13, 17)
    ]

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Read webcam frame
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                print(
                    "[WARNING] Could not read frame."
                )

                break

            # ------------------------------------------------
            # Mirror webcam
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            # ------------------------------------------------
            # Convert BGR -> RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # Create MediaPipe image
            # ------------------------------------------------

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb_frame
            )

            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            frame_timestamp_ms += 33

            # ------------------------------------------------
            # Detect hands
            # ------------------------------------------------

            result = detector.detect_for_video(
                mp_image,
                frame_timestamp_ms
            )

            hands_detected = (
                result.hand_landmarks
            )

            current_class = ""
            confidence = 0.0

            # =================================================
            # HAND DETECTED
            # =================================================

            if hands_detected:

                # ------------------------------------------------
                # Draw landmarks
                # ------------------------------------------------

                for hand_landmarks in (
                    hands_detected
                ):

                    # Draw connections
                    for connection in (
                        hand_connections
                    ):

                        start_idx = (
                            connection[0]
                        )

                        end_idx = (
                            connection[1]
                        )

                        start = (
                            hand_landmarks[
                                start_idx
                            ]
                        )

                        end = (
                            hand_landmarks[
                                end_idx
                            ]
                        )

                        sx = int(
                            start.x
                            * frame_width
                        )

                        sy = int(
                            start.y
                            * frame_height
                        )

                        ex = int(
                            end.x
                            * frame_width
                        )

                        ey = int(
                            end.y
                            * frame_height
                        )

                        cv2.line(
                            frame,
                            (sx, sy),
                            (ex, ey),
                            (0, 255, 0),
                            2
                        )

                    # ------------------------------------------------
                    # Draw landmark points
                    # ------------------------------------------------

                    for landmark in (
                        hand_landmarks
                    ):

                        x = int(
                            landmark.x
                            * frame_width
                        )

                        y = int(
                            landmark.y
                            * frame_height
                        )

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 0, 255),
                            -1
                        )

                # ------------------------------------------------
                # Get combined hand bounding box
                # ------------------------------------------------

                bbox = get_union_bbox(
                    hands_detected,
                    frame_width,
                    frame_height
                )

                if bbox is not None:

                    x1, y1, x2, y2 = bbox

                    # ------------------------------------------------
                    # Draw bounding box
                    # ------------------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                    # ------------------------------------------------
                    # Crop hand region
                    # ------------------------------------------------

                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    if crop.size != 0:

                        # =================================================
                        # LANDMARK FEATURE
                        # =================================================

                        # Convert MediaPipe Tasks output into
                        # the dictionary format expected by
                        # build_full_landmark_feature().
                        hands_for_features = (
                            convert_mediapipe_landmarks(
                                result
                            )
                        )

                        landmark_feature = (
                            build_full_landmark_feature(
                                hands_for_features,
                                config.LANDMARK_FEATURE_DIM_PER_HAND
                            )
                        )

                        # =================================================
                        # IMAGE PREPROCESSING
                        # =================================================

                        crop_rgb = cv2.cvtColor(
                            crop,
                            cv2.COLOR_BGR2RGB
                        )

                        transformed = (
                            eval_transform(
                                image=crop_rgb
                            )
                        )

                        # Albumentations Compose returns
                        # {"image": tensor, ...}
                        img_t = transformed[
                            "image"
                        ]

                        # =================================================
                        # IMAGE TENSOR
                        # =================================================

                        if not torch.is_tensor(
                            img_t
                        ):

                            img_t = torch.from_numpy(
                                np.asarray(img_t)
                            )

                        # ToTensorV2 already returns
                        # CHW format.

                        img_t = img_t.float()

                        img_t = (
                            img_t
                            .unsqueeze(0)
                            .to(device)
                        )

                        # =================================================
                        # LANDMARK TENSOR
                        # =================================================

                        lm_t = torch.tensor(
                            landmark_feature,
                            dtype=torch.float32,
                            device=device
                        ).unsqueeze(0)

                        # =================================================
                        # MODEL PREDICTION
                        # =================================================

                        with torch.no_grad():

                            probs = (
                                model.predict_proba(
                                    img_t,
                                    lm_t
                                )
                            )

                        # ------------------------------------------------
                        # Convert probabilities to NumPy
                        # ------------------------------------------------

                        if torch.is_tensor(
                            probs
                        ):

                            probs = (
                                probs
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        probs = np.asarray(
                            probs
                        )

                        if probs.ndim > 1:

                            probs_1d = probs[0]

                        else:

                            probs_1d = probs

                        # =================================================
                        # RAW PREDICTION
                        # =================================================

                        pred_idx = int(
                            np.argmax(
                                probs_1d
                            )
                        )

                        confidence = float(
                            probs_1d[
                                pred_idx
                            ]
                        )

                        # =================================================
                        # SMOOTHING
                        # =================================================

                        pred_history.append(
                            pred_idx
                        )

                        if pred_history:

                            smoothed_idx = (
                                collections.Counter(
                                    pred_history
                                )
                                .most_common(1)[0][0]
                            )

                            current_class = str(
                                config.IDX_TO_CLASS[
                                    smoothed_idx
                                ]
                            )

                        # ------------------------------------------------
                        # Recognition log
                        # ------------------------------------------------

                        print(
                            f"[RECOGNITION] "
                            f"Sign: {current_class} | "
                            f"Confidence: "
                            f"{confidence * 100:.1f}%"
                        )

            # =================================================
            # NO HAND
            # =================================================

            else:

                # Clear previous predictions.
                pred_history.clear()

                current_class = ""
                confidence = 0.0

            # =================================================
            # STABLE SIGN DETECTION
            # =================================================

            if current_class:

                # ------------------------------------------------
                # Same sign continues
                # ------------------------------------------------

                if (
                    current_class
                    == candidate_class
                ):

                    candidate_count += 1

                # ------------------------------------------------
                # New sign candidate
                # ------------------------------------------------

                else:

                    candidate_class = (
                        current_class
                    )

                    candidate_count = 1

                # ------------------------------------------------
                # Check stability
                # ------------------------------------------------

                current_time = time.time()

                if (
                    candidate_count
                    >= STABLE_FRAMES
                    and candidate_class
                    != last_spoken_class
                    and (
                        current_time
                        - last_spoken_time
                        >= SPEECH_COOLDOWN
                    )
                ):

                    print(
                        f"[TTS QUEUED] "
                        f"{candidate_class}"
                    )

                    tts_queue.put(
                        candidate_class
                    )

                    last_spoken_class = (
                        candidate_class
                    )

                    last_spoken_time = (
                        current_time
                    )

            # =================================================
            # NO CURRENT SIGN
            # =================================================

            else:

                candidate_class = ""
                candidate_count = 0

                # Allow the same sign to be spoken
                # again after the hand disappears.
                last_spoken_class = ""

            # =================================================
            # FPS
            # =================================================

            fps_counter += 1

            elapsed = (
                time.time()
                - fps_start_time
            )

            if elapsed >= 1.0:

                fps = (
                    fps_counter
                    / elapsed
                )

                fps_counter = 0

                fps_start_time = (
                    time.time()
                )

            # =================================================
            # UI
            # =================================================

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            if current_class:

                cv2.putText(
                    frame,
                    f"Sign: {current_class}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: "
                    f"{confidence * 100:.1f}%",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Stable: "
                    f"{min(candidate_count, STABLE_FRAMES)}"
                    f"/{STABLE_FRAMES}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # ------------------------------------------------
            # Quit instruction
            # ------------------------------------------------

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # =================================================
            # SHOW FRAME
            # =================================================

            cv2.imshow(
                "ISL Recognition",
                frame
            )

            # =================================================
            # QUIT
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print(
            "\n[INFO] Interrupted by user."
        )

    finally:

        print(
            "[INFO] Shutting down..."
        )

        # ====================================================
        # STOP TTS WORKER
        # ====================================================

        try:

            tts_queue.put(None)

            tts_thread.join(
                timeout=5.0
            )

        except Exception as e:

            print(
                f"[TTS WARNING] "
                f"Could not stop TTS worker: {e}"
            )

        # ====================================================
        # CLOSE MEDIAPIPE
        # ====================================================

        try:
            detector.close()
        except Exception:
            pass

        # ====================================================
        # CLOSE CAMERA
        # ====================================================

        try:
            cap.release()
        except Exception:
            pass

        # ====================================================
        # CLOSE OPENCV WINDOWS
        # ====================================================

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print(
            "[INFO] Program stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()