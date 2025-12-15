"""
Tests for model modules
"""
import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_model import RainfallLSTM, create_model


class TestRainfallLSTM(unittest.TestCase):
    """Test cases for RainfallLSTM model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_size = 40
        self.sequence_length = 7
        self.batch_size = 16
        self.model = RainfallLSTM(input_size=self.input_size)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, 64)
        self.assertEqual(self.model.num_layers, 2)
        self.assertEqual(self.model.dropout, 0.2)
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        # Create dummy input
        x = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        expected_shape = (self.batch_size, 1)
        self.assertEqual(output.shape, expected_shape)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        param_count = self.model.count_parameters()
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)
    
    def test_create_model_function(self):
        """Test model creation function"""
        model = create_model(input_size=self.input_size, device='cpu')
        self.assertIsInstance(model, RainfallLSTM)
        self.assertEqual(model.input_size, self.input_size)
    
    def test_model_with_different_parameters(self):
        """Test model with different hyperparameters"""
        model = RainfallLSTM(
            input_size=20,
            hidden_size=32,
            num_layers=1,
            dropout=0.1
        )
        
        self.assertEqual(model.input_size, 20)
        self.assertEqual(model.hidden_size, 32)
        self.assertEqual(model.num_layers, 1)
        self.assertEqual(model.dropout, 0.1)
        
        # Test forward pass
        x = torch.randn(8, 5, 20)
        output = model(x)
        self.assertEqual(output.shape, (8, 1))


if __name__ == '__main__':
    unittest.main()