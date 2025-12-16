#!/usr/bin/env python3
"""
Test Visual Crossing API integration
"""
import os
import sys
sys.path.append('agroweather_backend')

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agroweather.settings')
import django
django.setup()

from weather.data_providers import get_weather_provider
from datetime import datetime, timedelta

def test_visual_crossing():
    print("🧪 Testing Visual Crossing API Integration")
    print("=" * 50)
    
    # Get weather provider
    provider = get_weather_provider()
    print(f"Provider type: {type(provider).__name__}")
    
    # Test API call
    end_date = datetime.now().date() - timedelta(days=2)
    start_date = end_date - timedelta(days=7)
    
    print(f"Fetching data from {start_date} to {end_date}")
    
    try:
        data = provider.get_historical_data(
            latitude=5.1058,
            longitude=7.3536,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        print(f"✅ Success! Got {len(data)} days of data")
        print(f"Columns: {list(data.columns)}")
        print(f"Sample data:")
        print(data.head(2))
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_visual_crossing()