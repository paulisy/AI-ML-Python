#!/usr/bin/env python3
"""
Standalone script for cleaning weather data
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.processors.cleaner import WeatherDataCleaner
from utils.helpers import setup_logging
import argparse


def main():
    """Main data cleaning script"""
    parser = argparse.ArgumentParser(description='Clean raw weather data')
    parser.add_argument('--input', type=str, default='data/raw/nigerian_weather_raw.csv',
                       help='Input raw data file')
    parser.add_argument('--output', type=str, default='data/cleaned/weather_data_cleaned.csv',
                       help='Output cleaned data file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Initialize cleaner
        cleaner = WeatherDataCleaner()
        
        # Clean data
        cleaned_df = cleaner.clean_data(
            input_file=args.input,
            output_file=args.output
        )
        
        logger.info(f"Data cleaning completed! Cleaned dataset has {len(cleaned_df)} records.")
        return 0
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())