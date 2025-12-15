#!/usr/bin/env python3
"""
Standalone script for collecting weather data
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.collectors.weather_collector import WeatherCollector
from utils.helpers import setup_logging
import argparse


def main():
    """Main data collection script"""
    parser = argparse.ArgumentParser(description='Collect weather data from Visual Crossing API')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--use-chunks', action='store_true', help='Use predefined date chunks')
    parser.add_argument('--output', type=str, default='data/raw/nigerian_weather_raw.csv', 
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize collector
    collector = WeatherCollector()
    
    # Collect data
    success = collector.collect_data(
        start_date=args.start_date,
        end_date=args.end_date,
        use_chunks=args.use_chunks,
        output_file=args.output
    )
    
    if success:
        logger.info("Data collection completed successfully!")
        return 0
    else:
        logger.error("Data collection failed!")
        return 1


if __name__ == "__main__":
    exit(main())