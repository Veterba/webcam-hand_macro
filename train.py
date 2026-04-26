import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import GESTURES, NUM_FEATURES, SEQ_LEN
from gesture_model import GestureNet


def load_data(data_dir):
    X, y = [], []
    for label_idx, label in enumerate(GESTURES):
        folder = Path(data_dir) / label
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.npy")):
            arr = np.load(f)
            if arr.shape == (SEQ_LEN, NUM_FEATURES):
                X.append(arr)
                y.append(label_idx)
    if not X:
        raise RuntimeError("no training data found — run capture.py first")
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def augment(x):
    x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
    scale = np.random.uniform(0.9, 1.1)
    x = x * scale
    shift = np.random.uniform(-0.05, 0.05, size=(1, x.shape[1])).astype(np.float32)
    x = x + shift
    if np.random.rand() < 0.3:
        seq_len = x.shape[0]
        new_idx = np.linspace(0, seq_len - 1, seq_len) + np.random.uniform(-1, 1, seq_len)
        new_idx = np.clip(new_idx, 0, seq_len - 1).astype(np.int32)
        x = x[new_idx]
    return x.astype(np.float32)


def stratified_split(y, train_ratio, rng):
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * train_ratio))
        train_idx.extend(idx[:cut].tolist())
        val_idx.extend(idx[cut:].tolist())
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


class GestureDataset(Dataset):
    def __init__(self, X, y, train):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.train:
            x = augment(x)
        return torch.from_numpy(x), int(self.y[i])


def train(data_dir, model_path, epochs, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y = load_data(data_dir)
    counts = {GESTURES[c]: int((y == c).sum()) for c in range(len(GESTURES))}
    print(f"loaded {len(X)} samples — {counts}")

    rng = np.random.default_rng(seed)
    tr, va = stratified_split(y, train_ratio=0.8, rng=rng)
    if len(va) == 0:
        va = tr[:1]

    train_dl = DataLoader(GestureDataset(X[tr], y[tr], True), batch_size=16, shuffle=True)
    val_dl = DataLoader(GestureDataset(X[va], y[va], False), batch_size=16)

    model = GestureNet(num_classes=len(GESTURES), seq_len=SEQ_LEN, num_features=NUM_FEATURES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_loss = running / max(len(tr), 1)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
        acc = correct / max(len(va), 1)
        print(f"epoch {ep + 1:>3}/{epochs}  train_loss={train_loss:.4f}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)

    print(f"best val_acc={best_acc:.3f} — saved to {model_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", default="models/gesture_model.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args.data_dir, args.model, args.epochs, args.seed)
