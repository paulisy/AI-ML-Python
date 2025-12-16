#!/usr/bin/env python3
"""
AgroWeather AI - Model Evaluation
Evaluate trained LSTM on test set
"""
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.evaluator import ModelEvaluator
from utils.helpers import setup_logging, get_device


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate trained LSTM model on test set')
    parser.add_argument('--model-path', type=str, default='models/saved/rainfall_lstm_best.pth',
                       help='Path to trained model file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed test data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save evaluation outputs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Get device
        device = get_device()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(device=device)
        
        # Run evaluation
        metrics = evaluator.evaluate_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        # Print file locations
        print("\nFiles generated:")
        print(f"  📊 {args.output_dir}/plots/model_evaluation.png")
        print(f"  📊 {args.output_dir}/plots/error_analysis.png")
        print(f"  📄 {args.output_dir}/reports/evaluation_report.txt")
        print(f"  💾 {args.output_dir}/reports/evaluation_report.pkl")
        
        logger.info("Model evaluation completed successfully!")
        
        # Return success code based on model performance
        if metrics['r2'] > 0.5 and metrics['rmse'] < 15:
            logger.info("Model performance is acceptable!")
            return 0
        else:
            logger.warning("Model performance may need improvement")
            return 0  # Still success, just a warning
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())