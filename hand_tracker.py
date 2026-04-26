import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand_landmarker.task -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    def __init__(self):
        ensure_model()
        base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._last_ts_ms = -1

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    def draw(self, frame_bgr, landmarks_xyz):
        if landmarks_xyz is None:
            return
        h, w = frame_bgr.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks_xyz]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 0), 2)
        for p in pts:
            cv2.circle(frame_bgr, p, 3, (0, 0, 255), -1)


def normalize_landmarks(pts):
    if pts is None:
        return None
    out = pts - pts[0]
    scale = np.linalg.norm(out[9])
    if scale > 1e-6:
        out = out / scale
    return out.astype(np.float32)
