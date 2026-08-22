"""
utils/landmark_features.py

Turns raw MediaPipe Hands landmarks (21 points x,y,z, image-normalized [0,1])
into a pose-invariant feature vector: translation- and scale-invariant
coordinates, fingertip distances, and joint angles. This is what makes the
landmark branch describe "the hand shape" rather than "where the hand was
in the picture".
"""
import numpy as np

# MediaPipe Hands landmark indices
WRIST = 0
THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]
FINGERS = [THUMB, INDEX, MIDDLE, RING, PINKY]
FINGERTIPS = [4, 8, 12, 16, 20]
MCP_JOINTS = [1, 5, 9, 13, 17]  # base of each finger, used for palm-size reference


def _to_array(landmarks):
    """landmarks: list of 21 (x,y,z) tuples -> (21,3) float array."""
    return np.asarray(landmarks, dtype=np.float32).reshape(21, 3)


def normalize_landmarks(landmarks):
    """
    Translation + scale invariant normalization.
    - Center on the wrist (landmark 0).
    - Scale by the wrist-to-middle_finger_mcp distance (a stable proxy for
      palm size that doesn't change much across articulations).
    Rotation is deliberately left un-normalized: hand *orientation* in ISL
    can carry meaning (e.g. palm facing vs back of hand), so we don't want
    to throw it away. Camera-plane in-image rotation robustness is instead
    handled by rotation augmentation on the image branch + a small random
    rotation jitter applied to landmarks during training (see dataset.py).
    """
    pts = _to_array(landmarks)
    wrist = pts[WRIST].copy()
    centered = pts - wrist
    scale = np.linalg.norm(centered[MCP_JOINTS[2]])  # middle finger MCP
    scale = max(scale, 1e-6)
    normalized = centered / scale
    return normalized  # (21,3)


def _angle(a, b, c):
    """Angle at point b formed by rays b->a and b->c, in radians."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1) + 1e-8
    n2 = np.linalg.norm(v2) + 1e-8
    cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_ang))


def joint_angles(norm_pts):
    """15 flexion angles: 3 joints per finger (MCP, PIP, DIP) x 5 fingers."""
    angles = []
    wrist = norm_pts[WRIST]
    for finger in FINGERS:
        chain = [wrist] + [norm_pts[i] for i in finger]
        for i in range(1, len(chain) - 1):
            angles.append(_angle(chain[i - 1], chain[i], chain[i + 1]))
    return angles  # len 15


def fingertip_distances(norm_pts):
    """5 fingertip-to-wrist distances + 5 fingertip-to-palm-center distances."""
    wrist = norm_pts[WRIST]
    palm_center = norm_pts[MCP_JOINTS].mean(axis=0)
    tip_to_wrist = [float(np.linalg.norm(norm_pts[t] - wrist)) for t in FINGERTIPS]
    tip_to_palm = [float(np.linalg.norm(norm_pts[t] - palm_center)) for t in FINGERTIPS]
    return tip_to_wrist + tip_to_palm  # len 10


def hand_feature_vector(landmarks, handedness_score=1.0):
    """
    Full 89-d feature vector for ONE detected hand:
      63  normalized (x,y,z) coords, flattened
      10  fingertip distances (to wrist + to palm center)
      15  joint flexion angles
       1  handedness confidence/flag (placeholder; extend with left/right one-hot if needed)
    """
    norm_pts = normalize_landmarks(landmarks)
    coords = norm_pts.flatten().tolist()          # 63
    dists = fingertip_distances(norm_pts)          # 10
    angles = joint_angles(norm_pts)                 # 15
    feat = coords + dists + angles + [float(handedness_score)]  # 89
    return np.asarray(feat, dtype=np.float32)


def empty_hand_vector(dim):
    return np.zeros(dim, dtype=np.float32)


def build_full_landmark_feature(hands_detected, dim_per_hand):
    """
    hands_detected: list of dicts, each {"landmarks": [(x,y,z)x21], "score": float},
                     length 0, 1, or 2 (sorted left-to-right by x for determinism).
    Returns a fixed-length vector: [hand1_feat | hand2_feat | both_hands_present_flag]
    Missing hands are zero-filled so the model can learn "one hand vs two hands"
    from the flag rather than from arbitrary zeros looking like a real pose.
    """
    hands_detected = sorted(hands_detected, key=lambda h: float(np.mean([p[0] for p in h["landmarks"]])))
    slots = []
    for i in range(2):
        if i < len(hands_detected):
            slots.append(hand_feature_vector(hands_detected[i]["landmarks"], hands_detected[i].get("score", 1.0)))
        else:
            slots.append(empty_hand_vector(dim_per_hand))
    both_present_flag = np.asarray([1.0 if len(hands_detected) == 2 else 0.0], dtype=np.float32)
    return np.concatenate(slots + [both_present_flag])
