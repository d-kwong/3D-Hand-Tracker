import json
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

import cv2
import glfw
from OpenGL import GL

from target_box import DEFAULT_ORBIT, draw_scene, open_target_box, random_target
from webcam_landmarks import annotate_frame, create_landmarker


DATA_DIR = Path(__file__).resolve().parent / "data"


def open_session_file():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")
    stem = f"capture_{timestamp}"
    path = DATA_DIR / f"{stem}.jsonl"
    suffix = 1
    while True:
        try:
            return path.open("x", encoding="utf-8")
        except FileExistsError:
            path = DATA_DIR / f"{stem}_{suffix}.jsonl"
            suffix += 1


def draw_preview(width, height, frame):
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    left = round((width - frame_width * scale) / 2)
    top = round((height + frame_height * scale) / 2)
    GL.glViewport(0, 0, width, height)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    # OpenCV starts at the top row, while OpenGL's window origin is at the bottom.
    GL.glWindowPos2i(left, top)
    GL.glPixelZoom(scale, -scale)
    GL.glDrawPixels(frame_width, frame_height, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, frame)


@contextmanager
def open_live_views(title="Target Box"):
    with ExitStack() as resources:
        box, sphere = resources.enter_context(open_target_box(title))
        preview = glfw.create_window(640, 480, "Webcam Landmarks", None, None)
        if not preview:
            raise RuntimeError("Could not create the webcam preview window.")
        resources.callback(glfw.destroy_window, preview)
        glfw.make_context_current(preview)
        glfw.swap_interval(0)

        landmarker = resources.enter_context(create_landmarker())
        camera = cv2.VideoCapture(0)
        resources.callback(camera.release)
        if not camera.isOpened():
            raise RuntimeError("Could not open the default webcam. Check camera permissions.")
        yield box, sphere, preview, landmarker, camera


def draw_live_views(box, sphere, preview, frame, target, *, pan=(0, 0),
                    orbit=DEFAULT_ORBIT, draw_extra=None):
    box_width, box_height = glfw.get_framebuffer_size(box)
    preview_width, preview_height = glfw.get_framebuffer_size(preview)
    if box_width and box_height:
        glfw.make_context_current(box)
        draw_scene(box_width, box_height, target, sphere, pan=pan, orbit=orbit,
                   draw_extra=draw_extra)
        glfw.swap_buffers(box)
    if preview_width and preview_height:
        glfw.make_context_current(preview)
        draw_preview(preview_width, preview_height, frame)
        glfw.swap_buffers(preview)
    return all((box_width, box_height, preview_width, preview_height))


def landmark_coordinates(hands):
    landmarks = [[point.x, point.y, point.z] for point in hands[0]] if hands else []
    if len(landmarks) == 21 and all(isfinite(value) for point in landmarks for value in point):
        return landmarks
    return None


def collect():
    with ExitStack() as resources:
        box, sphere, preview, landmarker, camera = resources.enter_context(open_live_views())
        capture_requested = False

        def on_key(window, key, scancode, action, mods):
            nonlocal capture_requested
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_SPACE:
                capture_requested = True
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

        glfw.set_key_callback(box, on_key)
        glfw.set_key_callback(preview, on_key)
        print("Place your index fingertip at the target. Space captures, Esc quits.")

        target = random_target()
        output = None
        count = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            frame, hands = annotate_frame(frame, landmarker)
            both_visible = draw_live_views(box, sphere, preview, frame, target)

            glfw.poll_events()
            if glfw.window_should_close(box) or glfw.window_should_close(preview):
                break
            if not capture_requested:
                continue
            capture_requested = False
            if not both_visible:
                print("Capture skipped. Both windows must be visible.", flush=True)
                continue

            landmarks = landmark_coordinates(hands)
            if landmarks is None:
                print("Capture skipped. No valid hand landmarks in this frame.", flush=True)
                continue

            if output is None:
                output = resources.enter_context(open_session_file())
                print(f"Writing samples to {output.name}", flush=True)
            output.write(json.dumps({"x": landmarks, "y": target}, allow_nan=False) + "\n")
            output.flush()
            count += 1
            print(f"Saved sample {count} at {tuple(round(value, 3) for value in target)}", flush=True)
            target = random_target()


def main():
    try:
        collect()
    except (OSError, RuntimeError) as error:
        print(f"Collection stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
