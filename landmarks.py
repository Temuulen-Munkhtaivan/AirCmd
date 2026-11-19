"""Small geometry helpers for drawing MediaPipe's 21 hand landmarks."""

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20), # Little finger
)


def landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized MediaPipe landmark to an on-screen pixel."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be positive.")

    x = max(0, min(width - 1, int(landmark.x * width)))
    y = max(0, min(height - 1, int(landmark.y * height)))
    return x, y
