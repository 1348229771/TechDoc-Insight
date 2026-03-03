import unittest
import os
# Set environment variable to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import os
import shutil
import tempfile
import torch
from unittest.mock import MagicMock
from utils.data_loader import TechnicalDocumentDataset, get_data_statistics

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create a dummy data file
        self.data_path = os.path.join(self.test_dir, 'test_data.json')
        self.test_data = [
            {'input': 'This is input 1', 'target': 'Summary 1'},
            {'input': 'This is input 2', 'target': 'Summary 2'}
        ]
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f)
            
        # Mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        
        # Mock tokenizer call return value
        def tokenizer_side_effect(text, **kwargs):
            if 'text_target' in kwargs: # processing target
                return {'input_ids': torch.tensor([[1, 2, 3]])}
            else: # processing input
                return {'input_ids': torch.tensor([[4, 5, 6]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
                
        self.tokenizer.side_effect = tokenizer_side_effect
        # Also need to make tokenizer callable directly
        self.tokenizer.__call__ = MagicMock(side_effect=tokenizer_side_effect)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_dataset_len(self):
        dataset = TechnicalDocumentDataset(self.data_path, self.tokenizer)
        self.assertEqual(len(dataset), 2)

    def test_get_data_statistics(self):
        stats = get_data_statistics(self.data_path)
        self.assertEqual(stats['num_samples'], 2)
        self.assertEqual(stats['input_length']['min'], 4) # "This is input 1" -> 4 words
        self.assertEqual(stats['target_length']['max'], 2) # "Summary 1" -> 2 words

if __name__ == '__main__':
    unittest.main()
