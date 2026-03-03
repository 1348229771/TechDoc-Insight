import unittest
import os
# Set environment variable to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from unittest.mock import MagicMock, patch
from utils.metrics import evaluate_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Mock model
        self.model = MagicMock()
        self.model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        # Mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.batch_decode.return_value = ["This is a summary."]
        self.tokenizer.pad_token_id = 0
        
        # Mock data loader
        self.data_loader = [
            {
                'input_ids': torch.tensor([[4, 5, 6]]),
                'attention_mask': torch.tensor([[1, 1, 1]]),
                'labels': torch.tensor([[1, 2, 3]])
            }
        ]
        
        self.device = torch.device("cpu")

    @patch('utils.metrics.SummarizationMetrics')
    def test_evaluate_model(self, mock_metrics_class):
        # Mock metrics calculation
        mock_metrics_instance = mock_metrics_class.return_value
        mock_metrics_instance.compute_all_metrics.return_value = {
            'rouge1': 0.5, 'rouge2': 0.3, 'rougeL': 0.4, 'bleu': 20.0
        }
        
        results = evaluate_model(
            self.model, self.tokenizer, self.data_loader, self.device,
            max_length=50, num_beams=2
        )
        
        # Verify results structure
        self.assertIn('metrics', results)
        self.assertIn('predictions', results)
        self.assertIn('references', results)
        
        # Verify metrics
        self.assertEqual(results['metrics']['rouge1'], 0.5)
        
        # Verify model call
        self.model.eval.assert_called_once()
        self.model.generate.assert_called()

if __name__ == '__main__':
    unittest.main()
