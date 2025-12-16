#!/usr/bin/env python3
"""
Test the full weather prediction pipeline
"""
import os
import sys
sys.path.append('agroweather_backend')

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agroweather.settings')
import django
django.setup()

from weather.services import get_weather_service

def test_full_pipeline():
    print("🧪 Testing Full Weather Prediction Pipeline")
    print("=" * 60)
    
    try:
        # Get weather service
        service = get_weather_service()
        print("✅ Weather service initialized")
        
        # Generate forecast
        print("\n🔮 Generating forecast...")
        forecast = service.generate_forecast(
            latitude=5.1058,
            longitude=7.3536,
            days=2
        )
        
        print("✅ Forecast generated successfully!")
        print(f"Forecast days: {forecast['forecast_days']}")
        
        for i, pred in enumerate(forecast['forecasts']):
            print(f"Day {i+1}: {pred['date']} - {pred['rainfall']:.2f}mm (confidence: {pred['confidence_score']:.0%}) [{pred['model_version']}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_full_pipeline()