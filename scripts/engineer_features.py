#!/usr/bin/env python3
"""
Standalone script for feature engineering
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.processors.feature_engineer import FeatureEngineer
from utils.helpers import setup_logging
import argparse


def main():
    """Main feature engineering script"""
    parser = argparse.ArgumentParser(description='Engineer features from cleaned weather data')
    parser.add_argument('--input', type=str, default='data/cleaned/weather_data_cleaned.csv',
                       help='Input cleaned data file')
    parser.add_argument('--output', type=str, default='data/processed/weather_features.csv',
                       help='Output features file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Engineer features
        features_df = engineer.engineer_features(
            input_file=args.input,
            output_file=args.output
        )
        
        logger.info(f"Feature engineering completed! Dataset has {len(features_df)} records with {len(features_df.columns)} features.")
        return 0
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())