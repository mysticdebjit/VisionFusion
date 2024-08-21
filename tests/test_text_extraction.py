# tests/test_text_extraction.py

import unittest
from models.textextraction import process_image

class TestTextExtraction(unittest.TestCase):
    def setUp(self):
        self.test_image_path = '/content/text.jpg'

    def test_process_image(self):
        extracted_text = process_image(self.test_image_path)
        self.assertIsNotNone(extracted_text)
        self.assertIsInstance(extracted_text, str)
        self.assertTrue(len(extracted_text) > 0)

    def test_empty_image(self):
        empty_image_path = '/content/white.jpg'
        extracted_text = process_image(empty_image_path)
        self.assertEqual(extracted_text, '')

    def test_invalid_image_path(self):
        with self.assertRaises(Exception):  # You might want to be more specific about the exception
            process_image('nonexistent.jpg')

if __name__ == '__main__':
    unittest.main()