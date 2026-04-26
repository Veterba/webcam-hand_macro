import argparse
from pathlib import Path

import cv2
import numpy as np

from config import GESTURES, NUM_FEATURES, SEQ_LEN
from hand_tracker import HandTracker, normalize_landmarks


def _next_free_index(out_dir):
    used = set()
    for f in out_dir.glob("*.npy"):
        try:
            used.add(int(f.stem))
        except ValueError:
            continue
    i = 0
    while i in used:
        i += 1
    return i


def capture(label, num_samples, data_dir):
    out_dir = Path(data_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")
    print(f"Capturing '{label}'. SPACE = record one {SEQ_LEN}-frame sample. Q = quit.")

    saved = 0
    recording = False
    buffer = []

    while saved < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("warning: failed to read frame; retrying")
            cv2.waitKey(50)
            continue
        frame = cv2.flip(frame, 1)
        landmarks = tracker.process(frame)
        tracker.draw(frame, landmarks)
        norm = normalize_landmarks(landmarks)

        flat = (
            norm.flatten()
            if norm is not None
            else np.zeros(NUM_FEATURES, dtype=np.float32)
        )

        if recording:
            buffer.append(flat)
            cv2.putText(
                frame,
                f"REC {len(buffer)}/{SEQ_LEN}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            if len(buffer) >= SEQ_LEN:
                arr = np.stack(buffer)
                idx = _next_free_index(out_dir)
                np.save(out_dir / f"{idx:04d}.npy", arr)
                saved += 1
                print(f"saved sample {saved}/{num_samples} -> {idx:04d}.npy")
                buffer = []
                recording = False
        else:
            cv2.putText(
                frame,
                f"{label}: {saved}/{num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "SPACE = record   Q = quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and not recording:
            recording = True
            buffer = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, choices=GESTURES)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    capture(args.label, args.samples, args.data_dir)
