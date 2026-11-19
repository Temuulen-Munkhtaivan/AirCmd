import cv2
import mediapipe as mp
from pathlib import Path
import time

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open the webcam.")
        return
    
    model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

    if not model_path.is_file():
        print("Hand tracking model not found.")
        camera.release()
        return
    
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1
    )
    
    detector = None

    try:
        
        detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        last_timestamp_ms = -1
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            # Convert OpenCV's BGR image into RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame into a MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate an increasing timestamp
            timestamp_ms = max(
                time.monotonic_ns() // 1_000_000,
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            # Detect hands in the current frame
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Count detected hands
            hand_count = len(result.hand_landmarks)
            # Get the camera frame dimensions
            height, width = frame.shape[:2]

            # Loop through detected hands
            for hand in result.hand_landmarks:
                # Store the pixel positions of all 21 landmarks
                points = []

                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                # Draw lines between connected landmarks
                for start, end in HAND_CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 255, 0),
                        2
                    )

                # Draw green circles over the lines
                for point in points:
                    cv2.circle(frame, point, 4, (0, 255, 0), -1)
            cv2.imshow("AirCmd", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if detector is not None:
            detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
