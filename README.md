# hand_macro

Trigger macOS shortcuts with hand gestures from your webcam. You record a few samples
of each gesture, train a small neural net on the landmarks, and then a background loop
classifies live frames and runs an action when it sees a gesture it knows.

This is a personal project — the bundled actions open my Obsidian daily note and take
a screenshot — but the gesture/action layer is decoupled, so you can wire it to
whatever shell command you like.

## How it works

The pipeline has four stages:

1. **Hand landmarks (MediaPipe).** Each webcam frame goes through Google's
   `hand_landmarker` model, which returns 21 (x, y, z) keypoints for one hand. The
   first time you run anything the `.task` file is downloaded into `models/`.
2. **Normalization.** Raw landmarks are in image coordinates, so the same gesture
   looks completely different at different positions and distances. Before anything
   else we subtract the wrist (landmark 0) and divide by the wrist-to-middle-MCP
   distance (landmark 9). Result: translation- and scale-invariant pose.
3. **Temporal window.** A sliding deque of the last `SEQ_LEN` (=30) frames is kept.
   Each frame contributes a 63-dim vector (21 landmarks × 3 coords), so the input to
   the classifier is a `(30, 63)` tensor — about a second of motion.
4. **Classifier (`GestureNet`).** A small 1D CNN over the time axis:
   ```
   Conv1d(63 -> 64, k=3) -> ReLU -> MaxPool
   Conv1d(64 -> 128, k=3) -> ReLU -> MaxPool
   Flatten -> Linear(64) -> Dropout(0.3) -> Linear(num_classes)
   ```
   Trained with Adam + cross-entropy. The convs slide over time, so the model picks
   up motion patterns (e.g. a swipe), not just static poses.

At inference time the loop computes a softmax every frame once the buffer is full.
A gesture only fires its action if:

- the predicted label is not `idle`,
- the softmax probability is above `CONFIDENCE_THRESHOLD` (0.85),
- the same gesture hasn't fired within `COOLDOWN_SECONDS` (2.0), and
- the predicted label has *changed* since the last fire (so holding a pose doesn't
  re-trigger forever).

`idle` is a real class in the dataset — you have to record it, otherwise the model
has no concept of "do nothing" and will pick its best guess on every frame.

## Project layout

```
config.py         # gesture list, hyperparameters, action map, OBS_VAULT env
hand_tracker.py   # MediaPipe wrapper + normalize_landmarks
capture.py        # record .npy clips into data/<label>/
gesture_model.py  # GestureNet definition
train.py          # load data/, train, save best checkpoint
main.py           # live loop: predict + dispatch action
actions.py        # the actual Python functions that get called
data/             # gitignored; one folder per gesture, .npy clips inside
models/           # gitignored; downloaded landmarker + trained checkpoint
```

## Setup

Requires Python 3.10+ and a working webcam. macOS only for the bundled actions
(they call `open` and `screencapture`); the recognition side is cross-platform.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OBS_VAULT to your Obsidian vault name
```

On macOS you'll need to grant camera access to your terminal the first time you run
`capture.py` or `main.py`. If you want the screenshot/Obsidian actions to actually
fire, the same terminal also needs Accessibility permission.

## Recording data

For each gesture you want, capture ~30 short clips:

```sh
python capture.py --label idle --samples 30
python capture.py --label open_obsidian --samples 30
python capture.py --label screenshot --samples 30
```

A window opens showing the landmarker overlay. Press **SPACE** to record one
30-frame clip; the counter shows progress. Press **Q** to quit early. Each clip is
saved as `data/<label>/NNNN.npy`.

Tips:
- Vary your hand position, distance, and angle between samples. The augmentation
  helps but isn't a substitute for actual variety.
- For `idle`, record yourself doing nothing — sitting, typing, scratching your nose.
  This is what teaches the model to *not* fire.
- Re-running `capture.py` adds new clips alongside existing ones (it picks the next
  free index, so deleting bad clips and re-capturing is safe).

## Training

```sh
python train.py
```

That's it — it loads everything in `data/`, does a stratified 80/20 split, trains
for 40 epochs, and saves the best-by-val-accuracy checkpoint to
`models/gesture_model.pt`. You'll see a per-epoch line like:

```
epoch  12/40  train_loss=0.1834  val_acc=0.967
```

If `val_acc` plateaus low, the usual culprits are:
- not enough variety in the captures (especially `idle`),
- two gestures that are too similar over a 1-second window,
- too few samples per class (try 50+).

## Running

```sh
python main.py
```

Live window with the prediction label and confidence in the corner. Trigger one of
your gestures and the mapped action fires. `Q` quits.

## Adding a new gesture

Three places to edit, then capture + retrain:

1. **`actions.py`** — write the function and register it:
   ```python
   def lock_screen():
       _run(["pmset", "displaysleepnow"])

   ACTIONS = {
       ...,
       "lock_screen": lock_screen,
   }
   ```
2. **`config.py`** — add the label and map gesture → action:
   ```python
   GESTURES = ["idle", "open_obsidian", "screenshot", "fist"]
   ACTION_MAP = {
       ...,
       "fist": "lock_screen",
   }
   ```
3. Capture and retrain:
   ```sh
   python capture.py --label fist --samples 30
   python train.py
   ```

## Configuration

`config.py` holds the knobs:

- `SEQ_LEN` — number of frames per sample. Longer = catches slower gestures, but
  needs more data and adds latency at inference.
- `NUM_FEATURES` — 63 (21 landmarks × 3). Don't change this unless you also change
  what the tracker returns.
- `CONFIDENCE_THRESHOLD` — minimum softmax probability before an action fires. Raise
  it if you get false positives.
- `COOLDOWN_SECONDS` — minimum time between consecutive fires of the same gesture.

## Limitations

- Single hand only (`num_hands=1` in the tracker config).
- The model has no notion of *where* the hand is on screen — that's intentional, but
  it means you can't bind "swipe left near the top of the frame" vs. "swipe left at
  the bottom" to different actions.
- 30-frame window means \~1 s latency between starting a gesture and firing.
- The bundled actions assume macOS. The recognition layer doesn't.

## License

MIT.
