import unittest
from types import SimpleNamespace

from landmarks import HAND_CONNECTIONS, landmark_to_pixel


class LandmarkGeometryTests(unittest.TestCase):
    def test_normalized_point_is_converted_to_pixels(self):
        point = SimpleNamespace(x=0.5, y=0.25)
        self.assertEqual(landmark_to_pixel(point, 640, 480), (320, 120))

    def test_outside_points_are_clamped_to_frame(self):
        point = SimpleNamespace(x=1.2, y=-0.1)
        self.assertEqual(landmark_to_pixel(point, 640, 480), (639, 0))

    def test_zero_size_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            landmark_to_pixel(SimpleNamespace(x=0, y=0), 0, 480)

    def test_connections_reference_valid_landmark_indices(self):
        self.assertTrue(HAND_CONNECTIONS)
        self.assertEqual({index for pair in HAND_CONNECTIONS for index in pair}, set(range(21)))


if __name__ == "__main__":
    unittest.main()
