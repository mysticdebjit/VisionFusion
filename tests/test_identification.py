# tests/test_identification.py

import unittest
from models.objectclassification import preprocess_image, enhance_image, sharpen_image

class TestIdentification(unittest.TestCase):
    def setUp(self):
        self.test_image_path = '/content/image1-30.jpeg'

    def test_preprocess_image(self):
        result = preprocess_image(self.test_image_path)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[:2], (640, 640))

    def test_enhance_image(self):
        image = preprocess_image(self.test_image_path)
        enhanced = enhance_image(image)
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, image.shape)

    def test_sharpen_image(self):
        image = preprocess_image(self.test_image_path)
        sharpened = sharpen_image(image)
        self.assertIsNotNone(sharpened)
        self.assertEqual(sharpened.shape, image.shape)

if __name__ == '__main__':
    unittest.main()