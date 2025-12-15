"""
Tests for data collection modules
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.collectors.weather_collector import WeatherCollector


class TestWeatherCollector(unittest.TestCase):
    """Test cases for WeatherCollector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.collector = WeatherCollector(api_key='test_key')
    
    def test_initialization(self):
        """Test collector initialization"""
        self.assertEqual(self.collector.api_key, 'test_key')
        self.assertIn('name', self.collector.default_location)
        self.assertIn('lat', self.collector.default_location)
        self.assertIn('long', self.collector.default_location)
    
    def test_get_date_ranges(self):
        """Test date range generation"""
        ranges = self.collector.get_date_ranges_for_api_limits()
        self.assertIsInstance(ranges, list)
        self.assertTrue(len(ranges) > 0)
        
        # Check format of first range
        start_date, end_date = ranges[0]
        self.assertRegex(start_date, r'\d{4}-\d{2}-\d{2}')
        self.assertRegex(end_date, r'\d{4}-\d{2}-\d{2}')
    
    @patch('requests.get')
    def test_fetch_weather_data_success(self, mock_get):
        """Test successful weather data fetch"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'days': [
                {
                    'datetime': '2024-01-01',
                    'tempmax': 30.0,
                    'tempmin': 20.0,
                    'temp': 25.0,
                    'humidity': 70.0,
                    'precip': 5.0,
                    'windspeed': 10.0,
                    'pressure': 1013.0,
                    'cloudcover': 50.0,
                    'dew': 15.0,
                    'conditions': 'Partly cloudy'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test fetch
        location = {'name': 'Test', 'lat': 0.0, 'long': 0.0}
        df = self.collector.fetch_weather_data(location, '2024-01-01', '2024-01-01')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn('datetime', df.columns)
        self.assertIn('rainfall', df.columns)
    
    @patch('requests.get')
    def test_fetch_weather_data_error(self, mock_get):
        """Test weather data fetch with API error"""
        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_get.return_value = mock_response
        
        # Test fetch
        location = {'name': 'Test', 'lat': 0.0, 'long': 0.0}
        df = self.collector.fetch_weather_data(location, '2024-01-01', '2024-01-01')
        
        self.assertIsNone(df)


if __name__ == '__main__':
    unittest.main()