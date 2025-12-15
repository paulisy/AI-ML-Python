"""
Weather data collection from Visual Crossing API
"""
import requests
import pandas as pd
import time
from datetime import datetime
from decouple import config
import os
from typing import Optional, List, Tuple


class WeatherCollector:
    """
    Collects historical weather data from Visual Crossing API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather collector
        
        Args:
            api_key: Visual Crossing API key. If None, loads from environment
        """
        self.api_key = api_key or config('VISUAL_CROSSING_API_KEY')
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        
        # Default location: Aba, Abia State, Nigeria
        self.default_location = {
            'name': 'Aba',
            'lat': 5.1156,
            'long': 7.3636
        }
    
    def get_date_ranges_for_api_limits(self) -> List[Tuple[str, str]]:
        """
        Get predefined date ranges to respect API limits
        
        Returns:
            List of (start_date, end_date) tuples
        """
        return [
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
        ]
    
    def fetch_weather_data(self, location: dict, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch weather data for a specific location and date range
        
        Args:
            location: Dict with 'name', 'lat', 'long' keys
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with weather data or None if error
        """
        print(f"📡 Fetching data for {location['name']} ({start_date} to {end_date})...")
        
        # Build API URL
        url = f"{self.base_url}/{location['lat']},{location['long']}/{start_date}/{end_date}"
        
        # API parameters
        params = {
            'key': self.api_key,
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
                    'rainfall': day.get('precip', 0),
                    'wind_speed': day.get('windspeed'),
                    'pressure': day.get('pressure'),
                    'cloudcover': day.get('cloudcover'),
                    'dew': day.get('dew'),
                    'conditions': day.get('conditions'),
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            print(f"✅ Got {len(df)} days of data")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def collect_data(self, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    use_chunks: bool = False,
                    output_file: str = 'data/raw/nigerian_weather_raw.csv',
                    location: Optional[dict] = None) -> bool:
        """
        Main data collection method
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_chunks: If True, use predefined date chunks
            output_file: Path to save collected data
            location: Location dict, uses default if None
            
        Returns:
            True if successful, False otherwise
        """
        print("🌦️  AgroWeather AI - Data Collection")
        print("=" * 50)
        
        # Use default location if none provided
        if location is None:
            location = self.default_location
        
        # Determine date ranges
        if use_chunks:
            date_ranges = self.get_date_ranges_for_api_limits()
            print(f"📅 Processing {len(date_ranges)} date chunks")
        else:
            start_date = start_date or '2024-01-01'
            end_date = end_date or '2024-12-31'
            date_ranges = [(start_date, end_date)]
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Check existing data
        existing_dates = set()
        try:
            existing_df = pd.read_csv(output_file)
            existing_dates = set(existing_df['datetime'].tolist())
            print(f"📋 Found {len(existing_df)} existing records")
        except FileNotFoundError:
            print("📋 Creating new data file")
            existing_df = None
        
        # Collect data for each range
        total_new_records = 0
        
        for i, (start_date, end_date) in enumerate(date_ranges, 1):
            print(f"\n🔄 Chunk {i}/{len(date_ranges)}: {start_date} to {end_date}")
            
            df = self.fetch_weather_data(location, start_date, end_date)
            
            if df is not None:
                # Filter out existing dates
                new_df = df[~df['datetime'].isin(existing_dates)]
                
                if len(new_df) > 0:
                    # Save data
                    if existing_df is not None or total_new_records > 0:
                        new_df.to_csv(output_file, mode='a', header=False, index=False)
                    else:
                        new_df.to_csv(output_file, index=False)
                    
                    existing_dates.update(new_df['datetime'].tolist())
                    total_new_records += len(new_df)
                    print(f"✅ Added {len(new_df)} new records")
                else:
                    print("⚠️  All data already exists")
            
            # Rate limiting
            if i < len(date_ranges):
                print("⏳ Waiting 3 seconds...")
                time.sleep(3)
        
        # Summary
        print("\n" + "=" * 50)
        if total_new_records > 0:
            print(f"✅ SUCCESS! Added {total_new_records} new records")
            print(f"📁 Saved to: {output_file}")
            return True
        else:
            print("❌ No new data collected")
            return False