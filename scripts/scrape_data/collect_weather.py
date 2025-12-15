"""
Collecting data from Internal
5.1156, 7.3636 - Aba, Aba South, Abia State, Nigeria
"""
import requests
import pandas as pd
import time
from datetime import datetime
from decouple import config
import os

CITIES = {'name': 'Aba', 'lat': 5.1156, 'long': 7.3636}

def get_date_ranges_for_api_limits():
    """
    Helper function to break down date ranges for API limits.
    Modify these ranges as needed for your API calls.
    """
    ranges = [
        ('2014-01-01', '2014-12-31'),
        ('2015-01-01', '2015-12-31'),
        ('2016-01-01', '2016-12-31'),
        ('2017-01-01', '2017-12-31'),
        ('2018-01-01', '2018-12-31'),
        ('2019-01-01', '2019-12-31'),
        ('2020-01-01', '2020-12-31'),
        ('2021-01-01', '2021-12-31'),
        ('2022-01-01', '2022-12-31'),
        ('2023-01-01', '2023-12-31'),
        ('2024-01-01', '2024-12-31'),
        ('2025-01-01', '2016-12-10'),
    ]
    return ranges

def fetch_weather_for_city(cities, start_date, end_date):
    """
    Fetch historical weather data for a city

    Args:
        city_name (str): Name of city
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pandas.DataFrame: Weather data
    """
    print(f"📡 Fetching data for {cities['name']}...")

    # Build API call
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/"
        f"services/timeline/{cities.get('lat', None)},{cities.get('long', None)}/{start_date}/{end_date}"
    )

    # API parameters
    params = {
        'key': config('VISUAL_CROSSING_API_KEY'),
        'unitGroup': 'metric',
        'include': 'days',
        'elements': 'datetime,tempmax,tempmin,temp,humidity,precip,windspeed,pressure,cloudcover,dew,conditions',
    }


    try:
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
        
        data = response.json()

        records = []

        for day in data['days']:
            record = {
                'datetime': day['datetime'],
                'tempmax': day.get('tempmax'),
                'tempmin': day.get('tempmin'),
                'temp_avg': day.get('temp'),
                'humidity': day.get('humidity'),
                'rainfall': day.get('precip', 0),  # Default to 0 if missing
                'wind_speed': day.get('windspeed'),
                'pressure': day.get('pressure'),
                'cloudcover': day.get('cloudcover'),
                'dew': day.get('dew'),
                'conditions': day.get('conditions'),
            }
            records.append(record)
        
        # Convert to dataframe
        df = pd.DataFrame(records)

        print(f"✅ Got {len(df)} days of data for {cities['name']}")

        return df
    except Exception as e:
        print(f"❌ Error fetching data for {cities['name']}: {e}")
        return None


def main(start_date=None, end_date=None, use_chunks=False):
    """
    Main function to collect data for all cities.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format  
        use_chunks (bool): If True, process all predefined chunks
    """
    print("🌦️  AgroWeather AI - Data Collection")
    print("=" * 50)
    
    # Determine date ranges to process
    if use_chunks:
        date_ranges = get_date_ranges_for_api_limits()
        print(f"📅 Processing {len(date_ranges)} date chunks to respect API limits")
    else:
        # Use provided dates or defaults
        if not start_date:
            start_date = '2024-01-01'
        if not end_date:
            end_date = '2025-12-01'
        date_ranges = [(start_date, end_date)]
    
    print(f"Cities: {CITIES['name']}")
    print()
    
    # Check if file exists and load existing data
    output_file = 'data/raw/nigerian_weather_raw.csv'
    existing_dates = set()
    
    try:
        existing_df = pd.read_csv(output_file)
        existing_dates = set(existing_df['datetime'].tolist())
        print(f"📋 Found existing file with {len(existing_df)} records")
        print(f"� Date rangse in existing data: {existing_df['datetime'].min()} to {existing_df['datetime'].max()}")
    except FileNotFoundError:
        print("📋 No existing file found - will create new one")
        existing_df = None
    
    # Process each date range
    total_new_records = 0
    
    for i, (start_date, end_date) in enumerate(date_ranges, 1):
        print(f"\n🔄 Processing chunk {i}/{len(date_ranges)}: {start_date} to {end_date}")
        
        df = fetch_weather_for_city(CITIES, start_date, end_date)
        
        if df is not None:
            # Filter out dates that already exist
            new_df = df[~df['datetime'].isin(existing_dates)]
            
            if len(new_df) > 0:
                # Append to existing file or create new one
                if existing_df is not None or total_new_records > 0:
                    # Append new data
                    new_df.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    # Create new file (first chunk)
                    new_df.to_csv(output_file, index=False)
                
                # Update existing dates to avoid duplicates in next chunks
                existing_dates.update(new_df['datetime'].tolist())
                total_new_records += len(new_df)
                
                print(f"✅ Added {len(new_df)} new records (filtered out {len(df) - len(new_df)} duplicates)")
            else:
                print("⚠️  All data already exists - no new records to add")
        
        # Be polite - wait between requests
        if i < len(date_ranges):  # Don't wait after the last request
            print("⏳ Waiting 3 seconds before next request...")
            time.sleep(3)
    
    # Final summary
    print()
    print("=" * 50)
    if total_new_records > 0:
        final_count = len(existing_df) + total_new_records if existing_df is not None else total_new_records
        print(f"✅ SUCCESS! Added {total_new_records} new records")
        print(f"📁 Total records in file: {final_count}")
        print(f"📁 Saved to: {output_file}")
    else:
        print("❌ No new data added!")


if __name__ == "__main__":
    # USAGE EXAMPLES:
    # 1. Process all chunks automatically (recommended for API limits):
    # main(use_chunks=True)
    
    # 2. Process specific date range:
    # main('2024-01-01', '2024-03-31')
    
    # 3. Process single chunk (change dates as needed):
    main('2024-01-01', '2024-03-31')  # Change these dates for each run
