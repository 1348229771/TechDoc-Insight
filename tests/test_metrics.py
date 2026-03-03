import unittest
import os
# Set environment variable to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.metrics import SummarizationMetrics

class TestSummarizationMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SummarizationMetrics()
        self.predictions = [
            "The quick brown fox jumps over the lazy dog.",
            "This is a test summary."
        ]
        self.references = [
            "The quick brown fox jumped over the lazy dog.",
            "This is a summary for testing."
        ]

    def test_compute_rouge(self):
        """Test ROUGE score computation."""
        scores = self.metrics.compute_rouge(self.predictions, self.references)
        
        # Check if all ROUGE types are present
        self.assertIn('rouge1', scores)
        self.assertIn('rouge2', scores)
        self.assertIn('rougeL', scores)
        
        # Check if scores are floats and within range [0, 1]
        for key, value in scores.items():
            self.assertIsInstance(value, float)
            self.assertTrue(0.0 <= value <= 1.0)

    def test_compute_bleu(self):
        """Test BLEU score computation."""
        score = self.metrics.compute_bleu(self.predictions, self.references)
        
        # Check if score is a float and within range [0, 100] (sacrebleu returns 0-100)
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 100.0)

    def test_compute_all_metrics(self):
        """Test computing all metrics together."""
        all_metrics = self.metrics.compute_all_metrics(self.predictions, self.references)
        
        # Check for ROUGE scores
        self.assertIn('rouge1', all_metrics)
        self.assertIn('rouge2', all_metrics)
        self.assertIn('rougeL', all_metrics)
        
        # Check for BLEU score
        self.assertIn('bleu', all_metrics)

if __name__ == '__main__':
    unittest.main()
