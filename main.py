import cv2
import mediapipe as mp
from pathlib import Path
import time
import math

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def distance(point1, point2):
    return math.hypot(
        point1.x - point2.x,
        point1.y - point2.y
    )

def joint_angle(a, b, c):
    ab = distance(a, b)
    bc = distance(b, c)
    ac = distance(a, c)

    if ab == 0 or bc == 0:
        return 0

    cosine = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))

def is_finger_extended(hand, mcp, pip, tip):
    tip_distance = distance(hand[tip], hand[mcp])
    joint_distance = distance(hand[pip], hand[mcp])

    angle = joint_angle(hand[mcp], hand[pip], hand[tip])

    return tip_distance > joint_distance * 1.6 and angle > 155

def is_thumb_extended(hand):
    tip_to_pinky = distance(hand[4], hand[17])
    base_to_pinky = distance(hand[2], hand[17])

    return tip_to_pinky > base_to_pinky * 1.25

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
            
            gesture = "NONE"

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
                    
                # Landmark indices for four fingers
                finger_ids = [
                    (5, 6, 8),      # Index
                    (9, 10, 12),    # Middle
                    (13, 14, 16),   # Ring
                    (17, 18, 20)    # Pinky
                ]

                extended_count = 0
                
                thumb_extended = is_thumb_extended(hand)

                thumb_status = "EXTENDED" if thumb_extended else "FOLDED"

                cv2.putText(
                    frame,
                    f"Thumb: {thumb_status}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # Check each finger
                for mcp, pip, tip in finger_ids:
                    if is_finger_extended(hand, mcp, pip, tip):
                        extended_count += 1
                        
                # Check the complete gesture AFTER counting
                if extended_count == 4 and thumb_extended:
                    gesture = "OPEN PALM"

                # Display how many fingers are extended
                cv2.putText(
                    frame,
                    f"Extended fingers: {extended_count}/4",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
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
