"""
Weather data providers for production predictions
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from django.conf import settings
import os


class WeatherDataProvider:
    """Base class for weather data providers"""
    
    def get_historical_data(self, latitude: float, longitude: float, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical weather data for a location"""
        raise NotImplementedError
    
    def format_for_model(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Format raw API data to match training features"""
        raise NotImplementedError


class VisualCrossingProvider(WeatherDataProvider):
    """
    Visual Crossing Weather API provider
    Free tier: 1000 requests/day
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    def get_historical_data(self, latitude: float, longitude: float, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather data from Visual Crossing API
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with raw weather data
        """
        # Ensure we don't request future dates
        today = datetime.now().date()
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        # Adjust dates if they're in the future or too recent
        if end_date_obj >= today:
            end_date_obj = today - timedelta(days=2)  # Visual Crossing needs 1-2 day delay
            end_date = end_date_obj.isoformat()
        
        if start_date_obj >= today:
            start_date_obj = today - timedelta(days=32)
            start_date = start_date_obj.isoformat()
        
        print(f"📡 Visual Crossing API: {start_date} to {end_date}")
        
        url = f"{self.base_url}/{latitude},{longitude}/{start_date}/{end_date}"
        
        params = {
            'key': self.api_key,
            'unitGroup': 'metric',
            'include': 'days',
            'elements': 'datetime,tempmax,tempmin,temp,humidity,precip,windspeed,cloudcover,pressure,dew,conditions'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            days_data = data.get('days', [])
            df = pd.DataFrame(days_data)
            
            if df.empty:
                raise ValueError("No weather data returned from API")
            
            # Standardize column names to match training data
            column_mapping = {
                'datetime': 'datetime',
                'tempmax': 'tempmax',
                'tempmin': 'tempmin', 
                'temp': 'temp_avg',
                'humidity': 'humidity',
                'precip': 'rainfall',
                'windspeed': 'wind_speed',
                'cloudcover': 'cloudcover',
                'pressure': 'pressure',
                'dew': 'dew'  # Keep as 'dew' to match training
            }
            
            df = df.rename(columns=column_mapping)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Fill missing values with reasonable defaults
            df['rainfall'] = df['rainfall'].fillna(0.0)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing weather data: {e}")


class OpenMeteoProvider(WeatherDataProvider):
    """
    Open-Meteo API provider (free, no API key required)
    """
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    def get_historical_data(self, latitude: float, longitude: float, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather data from Open-Meteo API
        """
        # Ensure we don't request future dates
        today = datetime.now().date()
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        # Adjust dates if they're in the future
        if end_date_obj >= today:
            end_date_obj = today - timedelta(days=1)
            end_date = end_date_obj.isoformat()
        
        if start_date_obj >= today:
            start_date_obj = today - timedelta(days=30)
            start_date = start_date_obj.isoformat()
        
        print(f"📡 Adjusted date range: {start_date} to {end_date}")
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m,precipitation_sum,wind_speed_10m_max,cloud_cover_mean,surface_pressure,dew_point_2m_mean',
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            daily_data = data.get('daily', {})
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'datetime': daily_data.get('time', []),
                'tempmax': daily_data.get('temperature_2m_max', []),
                'tempmin': daily_data.get('temperature_2m_min', []),
                'temp_avg': daily_data.get('temperature_2m_mean', []),
                'humidity': daily_data.get('relative_humidity_2m', []),
                'rainfall': daily_data.get('precipitation_sum', []),
                'wind_speed': daily_data.get('wind_speed_10m_max', []),
                'cloudcover': daily_data.get('cloud_cover_mean', []),
                'pressure': daily_data.get('surface_pressure', []),
                'dew_point': daily_data.get('dew_point_2m_mean', [])
            })
            
            if df.empty:
                raise ValueError("No weather data returned from API")
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Fill missing values
            df['rainfall'] = df['rainfall'].fillna(0.0)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing weather data: {e}")


class ProductionFeatureEngineer:
    """
    Feature engineering for production inference
    Must match exactly the training pipeline
    """
    
    def __init__(self):
        self.required_base_features = [
            'datetime', 'tempmax', 'tempmin', 'temp_avg', 'humidity', 
            'rainfall', 'wind_speed', 'cloudcover', 'pressure', 'dew'
        ]
    
    def validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate that input data has required columns"""
        missing_cols = set(self.required_base_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features (matches training)"""
        df_feat = df.copy()
        
        # Extract temporal components
        df_feat['year'] = df_feat['datetime'].dt.year
        df_feat['month'] = df_feat['datetime'].dt.month
        df_feat['day'] = df_feat['datetime'].dt.day
        df_feat['day_of_year'] = df_feat['datetime'].dt.dayofyear
        df_feat['week_of_year'] = df_feat['datetime'].dt.isocalendar().week
        df_feat['day_of_week'] = df_feat['datetime'].dt.dayofweek
        
        # Cyclical encoding
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        df_feat['day_of_year_sin'] = np.sin(2 * np.pi * df_feat['day_of_year'] / 365)
        df_feat['day_of_year_cos'] = np.cos(2 * np.pi * df_feat['day_of_year'] / 365)
        
        return df_feat
    
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features (matches training)"""
        df_feat = df.copy()
        
        # Rainfall lags
        lags = [1, 2, 3, 7, 14, 30]
        for lag in lags:
            df_feat[f'rainfall_lag_{lag}'] = df_feat['rainfall'].shift(lag)
            
        # Temperature lags
        temp_lags = [1, 7]
        for lag in temp_lags:
            df_feat[f'temp_avg_lag_{lag}'] = df_feat['temp_avg'].shift(lag)
            
        # Humidity lags
        df_feat['humidity_lag_1'] = df_feat['humidity'].shift(1)
        
        return df_feat
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features (matches training)"""
        df_feat = df.copy()
        
        windows = [3, 7, 14, 30]
        
        for window in windows:
            # Rainfall rolling statistics
            df_feat[f'rainfall_{window}day_avg'] = df_feat['rainfall'].rolling(window=window, min_periods=1).mean()
            df_feat[f'rainfall_{window}day_max'] = df_feat['rainfall'].rolling(window=window, min_periods=1).max()
            df_feat[f'rainfall_{window}day_sum'] = df_feat['rainfall'].rolling(window=window, min_periods=1).sum()
            
            # Temperature rolling averages
            df_feat[f'temp_avg_{window}day'] = df_feat['temp_avg'].rolling(window=window, min_periods=1).mean()
            
            # Humidity rolling averages
            df_feat[f'humidity_{window}day_avg'] = df_feat['humidity'].rolling(window=window, min_periods=1).mean()
        
        # Cumulative features
        df_feat['rainfall_cumsum_30day'] = df_feat['rainfall'].rolling(window=30, min_periods=1).sum()
        df_feat['rainfall_cumsum_90day'] = df_feat['rainfall'].rolling(window=90, min_periods=1).sum()
        
        # Rolling standard deviation
        df_feat['rainfall_7day_std'] = df_feat['rainfall'].rolling(window=7, min_periods=1).std()
        df_feat['temp_7day_std'] = df_feat['temp_avg'].rolling(window=7, min_periods=1).std()
        
        return df_feat
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features (matches training)"""
        df_feat = df.copy()
        
        # Temperature-based features
        df_feat['temp_range'] = df_feat['tempmax'] - df_feat['tempmin']
        df_feat['temp_range_norm'] = df_feat['temp_range'] / df_feat['temp_avg']
        
        # Growing Degree Days
        base_temp = 10
        df_feat['gdd_base10'] = np.maximum(0, df_feat['temp_avg'] - base_temp)
        
        # Comfort indices
        df_feat['heat_index'] = df_feat['temp_avg'] + 0.5 * df_feat['humidity']
        
        # Pressure-based features
        df_feat['pressure_change'] = df_feat['pressure'].diff()
        df_feat['pressure_trend'] = df_feat['pressure'].rolling(window=3).apply(lambda x: x.iloc[-1] - x.iloc[0])
        
        # Wind chill
        df_feat['wind_chill'] = df_feat['temp_avg'] - (df_feat['wind_speed'] * 0.5)
        
        return df_feat
    
    def create_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary threshold features (matches training)"""
        df_feat = df.copy()
        
        # Rainfall thresholds
        df_feat['has_rain'] = (df_feat['rainfall'] > 0).astype(int)
        df_feat['is_rainy_day'] = (df_feat['rainfall'] > 1).astype(int)
        df_feat['is_heavy_rain'] = (df_feat['rainfall'] > 10).astype(int)
        df_feat['is_extreme_rain'] = (df_feat['rainfall'] > 50).astype(int)
        
        # Temperature thresholds
        df_feat['is_hot_day'] = (df_feat['tempmax'] > 35).astype(int)
        df_feat['is_cool_day'] = (df_feat['tempmin'] < 20).astype(int)
        
        # Humidity thresholds
        df_feat['humidity_high'] = (df_feat['humidity'] > 80).astype(int)
        df_feat['humidity_very_high'] = (df_feat['humidity'] > 90).astype(int)
        
        # Cloud cover thresholds
        df_feat['cloud_cover_high'] = (df_feat['cloudcover'] > 70).astype(int)
        df_feat['is_clear_day'] = (df_feat['cloudcover'] < 20).astype(int)
        
        # Combined conditions
        df_feat['humid_and_cloudy'] = ((df_feat['humidity'] > 80) & (df_feat['cloudcover'] > 70)).astype(int)
        df_feat['rain_likely'] = ((df_feat['humidity'] > 85) & (df_feat['cloudcover'] > 80)).astype(int)
        
        return df_feat
    
    def engineer_features_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline for inference
        Must match training exactly
        """
        self.validate_input_data(df)
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Apply all feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_lagged_features(df)
        df = self.create_rolling_features(df)
        df = self.create_derived_features(df)
        df = self.create_binary_features(df)
        
        return df


def get_weather_provider() -> WeatherDataProvider:
    """Factory function to get configured weather provider"""
    
    # Try Visual Crossing first (if API key available)
    visual_crossing_key = getattr(settings, 'VISUAL_CROSSING_API_KEY', None)
    if visual_crossing_key and visual_crossing_key.strip():
        print(f"🔑 Using Visual Crossing API (key: {visual_crossing_key[:8]}...)")
        return VisualCrossingProvider(visual_crossing_key)
    
    # Fallback to Open-Meteo (free)
    print("🌐 Using Open-Meteo API (free)")
    return OpenMeteoProvider()