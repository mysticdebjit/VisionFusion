# tests/test_summarization.py

import unittest
from models.summarization import summarize_analysis, generate_analysis_string

class TestSummarization(unittest.TestCase):
    def setUp(self):
        self.sample_results = {
            "Object Detection Model": [
                {"name": "Car", "confidence": 0.95, "attributes": {"color": "red", "model": "sedan"}},
                {"name": "Person", "confidence": 0.88, "attributes": {"pose": "standing"}}
            ],
            "Segmentation Model": [
                {"name": "Road", "confidence": 0.97, "attributes": {"condition": "good"}},
                {"name": "Sky", "confidence": 0.99, "attributes": {"weather": "clear"}}
            ]
        }

    def test_generate_analysis_string(self):
        analysis_string = generate_analysis_string(self.sample_results)
        self.assertIsInstance(analysis_string, str)
        self.assertIn("Object Detection Model", analysis_string)
        self.assertIn("Segmentation Model", analysis_string)

    def test_summarize_analysis(self):
        analysis_string = generate_analysis_string(self.sample_results)
        summary = summarize_analysis(analysis_string)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

    def test_empty_analysis(self):
        empty_results = {}
        analysis_string = generate_analysis_string(empty_results)
        summary = summarize_analysis(analysis_string)
        self.assertIn("error", summary.lower())

if __name__ == '__main__':
    unittest.main()