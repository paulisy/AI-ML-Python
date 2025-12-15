#!/usr/bin/env python3
"""
Standalone script for training LSTM model
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_model import create_model, model_summary
from models.trainer import ModelTrainer
from data.loaders import load_processed_data
from utils.helpers import setup_logging, get_device
import argparse
import pickle


def main():
    """Main model training script"""
    parser = argparse.ArgumentParser(description='Train LSTM model for rainfall prediction')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--model-dir', type=str, default='models/saved',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        print("🚀 AgroWeather AI - LSTM Model Training")
        print("=" * 60)
        
        # Get device
        device = get_device()
        
        # Load processed data
        print("📂 Loading processed data...")
        data, metadata = load_processed_data(args.data_dir)
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {len(data['X_train'])}")
        print(f"   Validation samples: {len(data['X_val'])}")
        print(f"   Test samples: {len(data['X_test'])}")
        print(f"   Features: {metadata['n_features']}")
        print(f"   Sequence length: {metadata['sequence_length']}")
        
        # Create model
        print("\n🏗️  Creating LSTM model...")
        model = create_model(
            input_size=metadata['n_features'],
            device=device
        )
        
        # Print model summary
        model_summary(model, input_shape=(1, metadata['sequence_length'], metadata['n_features']))
        
        # Initialize trainer
        trainer = ModelTrainer(model, device)
        
        # Train model
        print("\n🎯 Starting training...")
        training_results = trainer.train(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            save_dir=args.model_dir
        )
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_metrics = trainer.evaluate(data['X_test'], data['y_test'])
        
        # Save training metadata
        training_metadata = {
            'model_config': {
                'input_size': metadata['n_features'],
                'sequence_length': metadata['sequence_length'],
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2
            },
            'training_config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'patience': args.patience
            },
            'results': {
                'best_val_loss': training_results['best_val_loss'],
                'final_epoch': training_results['final_epoch'],
                'test_metrics': test_metrics
            },
            'data_info': metadata
        }
        
        os.makedirs(args.model_dir, exist_ok=True)
        with open(f'{args.model_dir}/training_metadata.pkl', 'wb') as f:
            pickle.dump(training_metadata, f)
        
        # Plot training history
        trainer.plot_training_history(f'{args.model_dir}/training_history.png')
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Best validation loss: {training_results['best_val_loss']:.4f}")
        print(f"   Test R²: {test_metrics['r2']:.4f}")
        print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"   Models saved to: {args.model_dir}")
        
        logger.info("Model training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())