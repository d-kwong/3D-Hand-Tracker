from pathlib import Path
from time import monotonic_ns
from urllib.request import urlopen

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def create_landmarker():
    if not MODEL_PATH.exists():
        print("Downloading the MediaPipe Hand Landmarker model...")
        with urlopen(MODEL_URL, timeout=30) as response:
            MODEL_PATH.write_bytes(response.read())

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def annotate_frame(frame, landmarker):
    vision = mp.tasks.vision
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, monotonic_ns() // 1_000_000)

    for landmarks in result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )
    return frame, result.hand_landmarks


def main():
    camera = cv2.VideoCapture(0)
    try:
        if not camera.isOpened():
            raise RuntimeError("Could not open the default webcam. Check camera permissions.")

        with create_landmarker() as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")

                frame, _ = annotate_frame(frame, landmarker)
                cv2.imshow("MediaPipe Hand Landmarks", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
