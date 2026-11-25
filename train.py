import argparse
import json
import re
from pathlib import Path

import torch
from torch import nn


DATA_DIR = Path(__file__).resolve().parent / "data"
EPOCHS = 400
CAPTURE_NAME = re.compile(r"capture_(\d{4}-\d{2}-\d{2}T\d{6}\.\d{6}Z)(?:_(\d+))?\.jsonl")


def latest_capture():
    captures = []
    for path in DATA_DIR.glob("capture_*.jsonl"):
        match = CAPTURE_NAME.fullmatch(path.name)
        if match and path.is_file():
            timestamp, suffix = match.groups()
            captures.append((timestamp, int(suffix or 0), path))
    if not captures:
        raise FileNotFoundError(f"No capture files found in {DATA_DIR}.")
    return max(captures)[2]


def load_capture(path):
    inputs, targets = [], []
    with path.open(encoding="utf-8") as capture:
        for line_number, line in enumerate(capture, 1):
            try:
                row = json.loads(line)
                landmarks = torch.tensor(row["x"], dtype=torch.float32)
                target = torch.tensor(row["y"], dtype=torch.float32)
                if landmarks.shape != (21, 3) or target.shape != (3,):
                    raise ValueError("expected x with shape (21, 3) and y with shape (3,)")
                if not (landmarks.isfinite().all() and target.isfinite().all()):
                    raise ValueError("coordinates must be finite numbers")
            except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
                raise ValueError(f"{path}, line {line_number}: {error}") from error
            inputs.append(landmarks.flatten())
            targets.append(target)
    if not inputs:
        raise ValueError(f"No samples found in {path}.")
    return torch.stack(inputs), torch.stack(targets)


def build_model():
    return nn.Sequential(nn.Linear(63, 32), nn.ReLU(), nn.Linear(32, 3))


def main():
    parser = argparse.ArgumentParser(description="Train a fingertip position model.")
    parser.add_argument("capture", nargs="?", type=Path, help="JSONL file (default: newest capture)")
    args = parser.parse_args()
    try:
        path = args.capture.expanduser() if args.capture else latest_capture()
        if path.suffix.lower() != ".jsonl":
            raise ValueError("The capture file must have a .jsonl extension.")
        inputs, targets = load_capture(path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training stopped: {error}\n")

    print(f"Training on {len(inputs)} samples from {path}")
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS}  training MSE {loss.item():.6f}")

    checkpoint = path.with_suffix(".pt")
    try:
        torch.save(model.state_dict(), checkpoint)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Could not save weights to {checkpoint}: {error}\n")
    print(f"Saved weights to {checkpoint}")


if __name__ == "__main__":
    main()
