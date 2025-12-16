#!/usr/bin/env python3
"""
Display project information and structure
"""
import os
import sys
from datetime import datetime


def get_file_size(filepath):
    """Get human readable file size"""
    try:
        size = os.path.getsize(filepath)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except OSError:
        return "N/A"


def check_file_exists(filepath):
    """Check if file exists and return status"""
    return "✅" if os.path.exists(filepath) else "❌"


def main():
    """Display project information"""
    print("🌦️  AgroWeather AI - Project Information")
    print("=" * 70)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Project structure
    print("📁 Project Structure:")
    print("-" * 40)
    structure = [
        ("Source Code", "src/"),
        ("  ├── Data Modules", "src/data/"),
        ("  ├── Model Modules", "src/models/"),
        ("  └── Utilities", "src/utils/"),
        ("Configuration", "config/"),
        ("Scripts", "scripts/"),
        ("Tests", "tests/"),
        ("Data Storage", "data/"),
        ("  ├── Raw Data", "data/raw/"),
        ("  ├── Cleaned Data", "data/cleaned/"),
        ("  └── Processed Data", "data/processed/"),
        ("Model Storage", "models/"),
        ("Notebooks", "notebooks/"),
        ("Outputs", "outputs/")
    ]
    
    for name, path in structure:
        status = "✅" if os.path.exists(path) else "❌"
        print(f"   {status} {name:<20} {path}")
    
    print()
    
    # Key files
    print("📄 Key Files:")
    print("-" * 40)
    key_files = [
        ("Project README", "README.md"),
        ("Requirements", "requirements.txt"),
        ("Setup Script", "setup.py"),
        ("Configuration", "config/settings.py"),
        ("Main Pipeline", "scripts/run_full_pipeline.py"),
        ("Data Collection", "scripts/collect_data.py"),
        ("Data Cleaning", "scripts/clean_data.py"),
        ("Feature Engineering", "scripts/engineer_features.py"),
        ("ML Preparation", "scripts/prepare_ml_data.py"),
        ("Model Training", "scripts/train_model.py"),
        ("Model Evaluation", "scripts/evaluate_model.py")
    ]
    
    for name, filepath in key_files:
        status = check_file_exists(filepath)
        size = get_file_size(filepath)
        print(f"   {status} {name:<20} {filepath:<30} ({size})")
    
    print()
    
    # Data files
    print("💾 Data Files:")
    print("-" * 40)
    data_files = [
        ("Raw Weather Data", "data/raw/nigerian_weather_raw.csv"),
        ("Cleaned Data", "data/cleaned/weather_data_cleaned.csv"),
        ("Engineered Features", "data/processed/weather_features.csv"),
        ("Training Data (X)", "data/processed/X_train.npy"),
        ("Training Data (y)", "data/processed/y_train.npy"),
        ("Validation Data (X)", "data/processed/X_val.npy"),
        ("Test Data (X)", "data/processed/X_test.npy"),
        ("Feature Scaler", "data/processed/feature_scaler.pkl"),
        ("Target Scaler", "data/processed/target_scaler.pkl"),
        ("Metadata", "data/processed/metadata.pkl")
    ]
    
    for name, filepath in data_files:
        status = check_file_exists(filepath)
        size = get_file_size(filepath)
        print(f"   {status} {name:<20} {filepath:<35} ({size})")
    
    print()
    
    # Model files
    print("🤖 Model Files:")
    print("-" * 40)
    model_files = [
        ("Best Model", "models/saved/rainfall_lstm_best.pth"),
        ("Full Model", "models/saved/rainfall_lstm_full.pth"),
        ("Training History", "models/saved/training_history.pkl"),
        ("Training Metadata", "models/saved/training_metadata.pkl")
    ]
    
    for name, filepath in model_files:
        status = check_file_exists(filepath)
        size = get_file_size(filepath)
        print(f"   {status} {name:<20} {filepath:<35} ({size})")
    
    print()
    
    # Evaluation outputs
    print("📊 Evaluation Outputs:")
    print("-" * 40)
    eval_files = [
        ("Evaluation Plots", "outputs/plots/model_evaluation.png"),
        ("Error Analysis", "outputs/plots/error_analysis.png"),
        ("Evaluation Report", "outputs/reports/evaluation_report.txt"),
        ("Report Data", "outputs/reports/evaluation_report.pkl")
    ]
    
    for name, filepath in eval_files:
        status = check_file_exists(filepath)
        size = get_file_size(filepath)
        print(f"   {status} {name:<20} {filepath:<35} ({size})")
    
    print()
    
    # Usage instructions
    print("🚀 Quick Start:")
    print("-" * 40)
    print("   1. Install dependencies:")
    print("      pip install -r requirements.txt")
    print()
    print("   2. Set up environment:")
    print("      cp .env.example .env  # Add your API key")
    print()
    print("   3. Run complete pipeline:")
    print("      python scripts/run_full_pipeline.py")
    print()
    print("   4. Or run individual steps:")
    print("      python scripts/collect_data.py")
    print("      python scripts/clean_data.py")
    print("      python scripts/engineer_features.py")
    print("      python scripts/prepare_ml_data.py")
    print("      python scripts/train_model.py")
    print("      python scripts/evaluate_model.py")
    print()
    
    # Module imports
    print("📦 Module Usage:")
    print("-" * 40)
    print("   from src.data.collectors import WeatherCollector")
    print("   from src.data.processors import WeatherDataCleaner, FeatureEngineer")
    print("   from src.models import RainfallLSTM, ModelTrainer, ModelEvaluator")
    print("   from src.data.loaders import load_processed_data")
    print()
    
    print("=" * 70)
    print("✅ Project refactoring complete!")
    print("🎯 Ready for development and deployment!")


if __name__ == "__main__":
    main()