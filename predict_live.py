import argparse
import sys
from math import isfinite
from pathlib import Path
from pickle import UnpicklingError

import glfw
import torch

from collect_data import draw_live_views, landmark_coordinates, open_live_views
from train import DATA_DIR, build_model
from webcam_landmarks import annotate_frame


def latest_checkpoint():
    checkpoints = (path for path in DATA_DIR.glob("*.pt") if path.is_file())
    checkpoint = max(
        checkpoints, key=lambda path: (path.stat().st_mtime_ns, path.name), default=None
    )
    if checkpoint is None:
        raise FileNotFoundError(f"No .pt checkpoints found in {DATA_DIR}.")
    return checkpoint


def load_model(checkpoint):
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


@torch.inference_mode()
def predict_position(model, hands):
    landmarks = landmark_coordinates(hands)
    if landmarks is None:
        return None
    inputs = torch.tensor(landmarks, dtype=torch.float32).reshape(1, 63)
    position = model(inputs)[0].tolist()
    return position if all(isfinite(value) for value in position) else None


def run_live(model):
    with open_live_views("Live Fingertip Prediction") as (box, sphere, preview, landmarker, camera):
        def on_key(window, key, scancode, action, mods):
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(window, True)

        glfw.set_key_callback(box, on_key)
        glfw.set_key_callback(preview, on_key)
        while not (glfw.window_should_close(box) or glfw.window_should_close(preview)):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            frame, hands = annotate_frame(frame, landmarker)
            draw_live_views(box, sphere, preview, frame, predict_position(model, hands))
            glfw.poll_events()


def main(run=run_live, description="Predict fingertip position from the live webcam."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("checkpoint", nargs="?", type=Path, help="Weights file (default: latest .pt)")
    args = parser.parse_args()
    try:
        checkpoint = args.checkpoint.expanduser() if args.checkpoint else latest_checkpoint()
        model = load_model(checkpoint)
        print(f"Loaded {checkpoint}. Press Esc in either window to quit.", flush=True)
        run(model)
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, UnpicklingError) as error:
        print(f"Live prediction stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
