"""AirCmd milestone 1: show hand landmarks from a local webcam."""

import argparse
import shutil
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp

from landmarks import HAND_CONNECTIONS, landmark_to_pixel


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model() -> None:
    """Download the official model once; webcam frames always stay local."""
    if MODEL_PATH.is_file():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = MODEL_PATH.with_suffix(".task.download")
    print("First run: downloading Google's hand-tracking model...")
    try:
        with urlopen(MODEL_URL, timeout=30) as response:
            with temporary_file.open("wb") as output:
                shutil.copyfileobj(response, output)
        if temporary_file.stat().st_size < 100_000:
            raise OSError("The downloaded model appears incomplete.")
        temporary_file.replace(MODEL_PATH)
    except (OSError, TimeoutError) as error:
        temporary_file.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the hand-tracking model. "
            "Check your connection and try again."
        ) from error


def draw_hand(frame, hand_landmarks) -> None:
    """Draw MediaPipe's 21 points and their connections on the camera frame."""
    height, width = frame.shape[:2]
    points = [landmark_to_pixel(point, width, height) for point in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 210, 80), 2)

    for point in points:
        cv2.circle(frame, point, 4, (60, 240, 110), -1)


def run(camera_index: int) -> None:
    ensure_model()

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Close other camera apps, "
            "check Windows camera permissions, or try --camera 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    previous_timestamp = -1
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Camera stopped providing frames.")
                    break

                # Mirroring makes the preview feel like looking in a mirror.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # MediaPipe VIDEO mode requires monotonically increasing times.
                timestamp = max(time.monotonic_ns() // 1_000_000, previous_timestamp + 1)
                previous_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)

                for hand in result.hand_landmarks:
                    draw_hand(frame, hand)

                hand_found = bool(result.hand_landmarks)
                status = "Hand detected" if hand_found else "Show your hand to the camera"
                status_color = (60, 240, 110) if hand_found else (220, 220, 220)

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (25, 30, 38), -1)
                cv2.putText(
                    frame, "AirCmd - Hand Tracking", (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                cv2.putText(
                    frame, status, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2,
                )
                cv2.putText(
                    frame, "Press Q or ESC to quit", (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

                cv2.imshow("AirCmd", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirCmd hand-tracking prototype")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    try:
        run(args.camera)
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"AirCmd error: {error}\n")
