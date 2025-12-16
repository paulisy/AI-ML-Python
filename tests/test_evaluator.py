"""
Tests for model evaluation modules
"""
import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.evaluator import ModelEvaluator
from models.lstm_model import RainfallLSTM


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ModelEvaluator(device='cpu')
        self.batch_size = 8
        self.sequence_length = 7
        self.n_features = 40
        self.n_samples = 100
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertEqual(self.evaluator.device, 'cpu')
        self.assertIsInstance(self.evaluator.metrics, dict)
    
    def test_make_predictions(self):
        """Test prediction generation"""
        # Create dummy model and data
        model = RainfallLSTM(input_size=self.n_features)
        model.eval()
        
        X_test = np.random.randn(self.n_samples, self.sequence_length, self.n_features)
        
        # Make predictions
        predictions = self.evaluator.make_predictions(model, X_test, batch_size=16)
        
        # Check output shape
        expected_shape = (self.n_samples, 1)
        self.assertEqual(predictions.shape, expected_shape)
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Create dummy data
        y_true = np.random.randn(self.n_samples, 1)
        y_pred = y_true + np.random.randn(self.n_samples, 1) * 0.1  # Add small noise
        
        # Create dummy scaler
        scaler = MagicMock()
        scaler.inverse_transform.side_effect = lambda x: x * 10  # Simple scaling
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(y_true, y_pred, scaler)
        
        # Check that all expected metrics are present
        expected_keys = [
            'mse', 'rmse', 'mae', 'r2', 'baseline_rmse', 
            'improvement_over_baseline', 'accuracy', 'precision', 'recall',
            'y_true_mm', 'y_pred_mm', 'y_true_binary', 'y_pred_binary'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['r2'], 0)  # R² should be positive for good predictions
        self.assertGreater(metrics['rmse'], 0)     # RMSE should be positive
        self.assertGreater(metrics['mae'], 0)      # MAE should be positive
        self.assertGreaterEqual(metrics['accuracy'], 0)  # Accuracy should be between 0 and 1
        self.assertLessEqual(metrics['accuracy'], 1)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_predictions_vs_actual(self, mock_close, mock_savefig):
        """Test plotting functionality"""
        # Create dummy metrics
        n_samples = 50
        metrics = {
            'y_true_mm': np.random.randn(n_samples, 1),
            'y_pred_mm': np.random.randn(n_samples, 1),
            'y_true_binary': np.random.randint(0, 2, (n_samples, 1)),
            'y_pred_binary': np.random.randint(0, 2, (n_samples, 1)),
            'r2': 0.75,
            'rmse': 5.2,
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.90
        }
        
        # Test plotting (should not raise errors)
        try:
            self.evaluator.plot_predictions_vs_actual('test_output', metrics)
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
        except Exception as e:
            self.fail(f"Plotting failed with error: {e}")
    
    def test_print_metrics(self):
        """Test metrics printing"""
        # Create dummy metrics
        metrics = {
            'rmse': 8.5,
            'mae': 6.2,
            'mse': 72.25,
            'r2': 0.65,
            'baseline_rmse': 12.0,
            'improvement_over_baseline': 29.2,
            'accuracy': 0.82,
            'precision': 0.78,
            'recall': 0.85
        }
        
        # Test that printing doesn't raise errors
        try:
            self.evaluator.print_metrics(metrics)
        except Exception as e:
            self.fail(f"Metrics printing failed with error: {e}")


if __name__ == '__main__':
    unittest.main()