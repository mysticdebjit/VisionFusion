# tests/test_segmentation.py
import os
import unittest
import cv2
import numpy as np
from models.segmentation import analyze_image
from ultralytics import YOLO

class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.model = YOLO('/content/drive/MyDrive/brain.pt')  # Assuming you're using YOLOv8
        self.test_image_path = '/content/image1-30.jpeg'

    def test_analyze_image(self):
        result = analyze_image(self.test_image_path, self.model)
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        self.assertTrue(os.path.isdir(result))

    def test_empty_image(self):
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite('empty.jpg', empty_image)
        with self.assertRaises(ValueError):
            analyze_image('empty.jpg', self.model)

    def test_invalid_image_path(self):
        with self.assertRaises(ValueError):
            analyze_image('nonexistent.jpg', self.model)

if __name__ == '__main__':
    unittest.main()