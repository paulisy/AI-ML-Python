#!/usr/bin/env python3
"""
Complete pipeline script for AgroWeather AI
Runs the entire data collection, processing, and training pipeline
"""
import sys
import os
import argparse
import subprocess
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.helpers import setup_logging, create_directory_structure


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main pipeline script"""
    parser = argparse.ArgumentParser(description='Run complete AgroWeather AI pipeline')
    parser.add_argument('--skip-collection', action='store_true', 
                       help='Skip data collection (use existing raw data)')
    parser.add_argument('--skip-cleaning', action='store_true',
                       help='Skip data cleaning (use existing cleaned data)')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature engineering (use existing features)')
    parser.add_argument('--skip-ml-prep', action='store_true',
                       help='Skip ML data preparation (use existing processed data)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip model evaluation')
    parser.add_argument('--use-chunks', action='store_true',
                       help='Use date chunks for data collection')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_file='logs/pipeline.log')
    
    print("🌦️  AgroWeather AI - Complete Pipeline")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create directory structure
    print("📁 Creating directory structure...")
    create_directory_structure()
    
    success = True
    
    # Step 1: Data Collection
    if not args.skip_collection:
        cmd = "python scripts/collect_data.py"
        if args.use_chunks:
            cmd += " --use-chunks"
        success &= run_command(cmd, "Data Collection")
    else:
        print("\n⏭️  Skipping data collection")
    
    # Step 2: Data Cleaning
    if success and not args.skip_cleaning:
        cmd = "python scripts/clean_data.py"
        success &= run_command(cmd, "Data Cleaning")
    else:
        print("\n⏭️  Skipping data cleaning")
    
    # Step 3: Feature Engineering
    if success and not args.skip_features:
        cmd = "python scripts/engineer_features.py"
        success &= run_command(cmd, "Feature Engineering")
    else:
        print("\n⏭️  Skipping feature engineering")
    
    # Step 4: ML Data Preparation
    if success and not args.skip_ml_prep:
        cmd = "python scripts/prepare_ml_data.py"
        success &= run_command(cmd, "ML Data Preparation")
    else:
        print("\n⏭️  Skipping ML data preparation")
    
    # Step 5: Model Training
    if success and not args.skip_training:
        cmd = f"python scripts/train_model.py --epochs {args.epochs}"
        success &= run_command(cmd, "Model Training")
    else:
        print("\n⏭️  Skipping model training")
    
    # Step 6: Model Evaluation
    if success and not args.skip_evaluation:
        cmd = "python scripts/evaluate_model.py"
        success &= run_command(cmd, "Model Evaluation")
    else:
        print("\n⏭️  Skipping model evaluation")
    
    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ All steps executed without errors")
        
        # Show what was created
        print("\n📊 Generated Artifacts:")
        artifacts = [
            ("Raw Data", "data/raw/nigerian_weather_raw.csv"),
            ("Cleaned Data", "data/cleaned/weather_data_cleaned.csv"),
            ("Engineered Features", "data/processed/weather_features.csv"),
            ("ML Training Data", "data/processed/X_train.npy"),
            ("Trained Model", "models/saved/rainfall_lstm_best.pth"),
            ("Training History", "models/saved/training_history.pkl"),
            ("Evaluation Plots", "outputs/plots/model_evaluation.png"),
            ("Evaluation Report", "outputs/reports/evaluation_report.txt")
        ]
        
        for name, path in artifacts:
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"   ✅ {name}: {path} ({size:,} bytes)")
            else:
                print(f"   ❌ {name}: {path} (not found)")
        
        logger.info("Pipeline completed successfully!")
    else:
        print("❌ PIPELINE FAILED!")
        print("⚠️  Some steps encountered errors - check logs for details")
        logger.error("Pipeline failed!")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())