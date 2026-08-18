import unittest
from pathlib import Path

import numpy as np

from panorama import stitch_images


class PanoramaTests(unittest.TestCase):
    def test_stitch_two_images(self):
        img1 = np.full((200, 300, 3), 255, dtype=np.uint8)
        img2 = np.full((200, 300, 3), 200, dtype=np.uint8)

        for i in range(100):
            img1[:, i : i + 2] = [0, 255, 0]
            img2[:, i : i + 2] = [255, 0, 0]

        left = Path("test_left.jpg")
        right = Path("test_right.jpg")

        try:
            import cv2

            cv2.imwrite(str(left), img1)
            cv2.imwrite(str(right), img2)

            result = stitch_images([str(left), str(right)])

            self.assertIsNotNone(result)
            self.assertGreater(result.shape[0], 0)
            self.assertGreater(result.shape[1], 300)
        finally:
            for path in (left, right):
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    unittest.main()
