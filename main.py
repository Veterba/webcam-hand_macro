import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from actions import ACTIONS
from config import (
    ACTION_MAP,
    CONFIDENCE_THRESHOLD,
    COOLDOWN_SECONDS,
    GESTURES,
    NUM_FEATURES,
    SEQ_LEN,
)
from gesture_model import GestureNet
from hand_tracker import HandTracker


def run(model_path):
    if not Path(model_path).exists():
        raise SystemExit(f"model not found at {model_path} - run train.py first")

    model = GestureNet(num_classes=len(GESTURES), seq_len=SEQ_LEN, num_features=NUM_FEATURES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQ_LEN)
    last_trigger = {g: 0.0 for g in GESTURES}

    pred_label = "..."
    pred_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        landmarks = tracker.process(frame)
        tracker.draw(frame, landmarks)

        flat = (
            landmarks.flatten()
            if landmarks is not None
            else np.zeros(NUM_FEATURES, dtype=np.float32)
        )
        buffer.append(flat)

        if len(buffer) == SEQ_LEN:
            x = torch.from_numpy(np.stack(buffer)).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]
                idx = int(probs.argmax())
                pred_label = GESTURES[idx]
                pred_conf = float(probs[idx])

            if (
                pred_label != "idle"
                and pred_conf >= CONFIDENCE_THRESHOLD
                and time.time() - last_trigger[pred_label] >= COOLDOWN_SECONDS
            ):
                action_name = ACTION_MAP.get(pred_label)
                fn = ACTIONS.get(action_name) if action_name else None
                if fn is not None:
                    fn()
                    last_trigger[pred_label] = time.time()
                    print(f"triggered: {pred_label} ({pred_conf:.2f})")

        color = (0, 255, 0) if pred_conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(
            frame,
            f"{pred_label}: {pred_conf:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        cv2.imshow("hand macro", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/gesture_model.pt")
    args = p.parse_args()
    run(args.model)
