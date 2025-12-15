"""
Feature engineering for weather data
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import os


class FeatureEngineer:
    """
    Handles feature engineering for weather prediction models
    """
    
    def __init__(self):
        self.feature_stats = {}
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime
        
        Args:
            df: DataFrame with datetime column
            
        Returns:
            DataFrame with temporal features added
        """
        print("📅 Creating temporal features...")
        df_feat = df.copy()
        
        # Extract temporal components
        df_feat['year'] = df_feat['datetime'].dt.year
        df_feat['month'] = df_feat['datetime'].dt.month
        df_feat['day'] = df_feat['datetime'].dt.day
        df_feat['day_of_year'] = df_feat['datetime'].dt.dayofyear
        df_feat['week_of_year'] = df_feat['datetime'].dt.isocalendar().week
        df_feat['day_of_week'] = df_feat['datetime'].dt.dayofweek
        
        # Cyclical encoding for seasonal patterns
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        df_feat['day_of_year_sin'] = np.sin(2 * np.pi * df_feat['day_of_year'] / 365)
        df_feat['day_of_year_cos'] = np.cos(2 * np.pi * df_feat['day_of_year'] / 365)
        
        print(f"   ✅ Added temporal features: month, day_of_year, week_of_year, cyclical encodings")
        
        return df_feat
    
    def create_lagged_features(self, df: pd.DataFrame, target_col: str = 'rainfall') -> pd.DataFrame:
        """
        Create lagged features for time series prediction
        
        Args:
            df: DataFrame sorted by datetime
            target_col: Column to create lags for
            
        Returns:
            DataFrame with lagged features
        """
        print("⏮️  Creating lagged features...")
        df_feat = df.copy()
        
        # Rainfall lags (key predictors from EDA)
        lags = [1, 2, 3, 7, 14, 30]
        for lag in lags:
            col_name = f'{target_col}_lag_{lag}'
            df_feat[col_name] = df_feat[target_col].shift(lag)
            
        # Temperature lags
        temp_lags = [1, 7]
        for lag in temp_lags:
            df_feat[f'temp_avg_lag_{lag}'] = df_feat['temp_avg'].shift(lag)
            
        # Humidity lags
        df_feat['humidity_lag_1'] = df_feat['humidity'].shift(1)
        
        print(f"   ✅ Added lagged features: rainfall lags {lags}, temp/humidity lags")
        
        return df_feat
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: DataFrame sorted by datetime
            
        Returns:
            DataFrame with rolling features
        """
        print("📊 Creating rolling window features...")
        df_feat = df.copy()
        
        # Rolling averages (different windows)
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
        
        # Rolling standard deviation (volatility)
        df_feat['rainfall_7day_std'] = df_feat['rainfall'].rolling(window=7, min_periods=1).std()
        df_feat['temp_7day_std'] = df_feat['temp_avg'].rolling(window=7, min_periods=1).std()
        
        print(f"   ✅ Added rolling features: {len(windows)} window sizes, cumulative sums, volatility")
        
        return df_feat
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived weather features
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with derived features
        """
        print("🔬 Creating derived features...")
        df_feat = df.copy()
        
        # Temperature-based features
        df_feat['temp_range'] = df_feat['tempmax'] - df_feat['tempmin']
        df_feat['temp_range_norm'] = df_feat['temp_range'] / df_feat['temp_avg']
        
        # Growing Degree Days (important for agriculture)
        base_temp = 10  # Base temperature for crops
        df_feat['gdd_base10'] = np.maximum(0, df_feat['temp_avg'] - base_temp)
        
        # Comfort indices
        df_feat['heat_index'] = df_feat['temp_avg'] + 0.5 * df_feat['humidity']  # Simplified heat index
        
        # Pressure-based features
        df_feat['pressure_change'] = df_feat['pressure'].diff()
        df_feat['pressure_trend'] = df_feat['pressure'].rolling(window=3).apply(lambda x: x.iloc[-1] - x.iloc[0])
        
        # Wind chill (simplified)
        df_feat['wind_chill'] = df_feat['temp_avg'] - (df_feat['wind_speed'] * 0.5)
        
        print("   ✅ Added derived features: temp_range, GDD, heat_index, pressure_change, wind_chill")
        
        return df_feat
    
    def create_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary threshold features
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with binary features
        """
        print("🎯 Creating binary threshold features...")
        df_feat = df.copy()
        
        # Rainfall thresholds
        df_feat['has_rain'] = (df_feat['rainfall'] > 0).astype(int)
        df_feat['is_rainy_day'] = (df_feat['rainfall'] > 1).astype(int)  # > 1mm
        df_feat['is_heavy_rain'] = (df_feat['rainfall'] > 10).astype(int)  # > 10mm
        df_feat['is_extreme_rain'] = (df_feat['rainfall'] > 50).astype(int)  # > 50mm
        
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
        
        print("   ✅ Added binary features: rain thresholds, temperature/humidity/cloud thresholds, combined conditions")
        
        return df_feat
    
    def encode_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode weather conditions text into binary features
        
        Args:
            df: DataFrame with 'conditions' column
            
        Returns:
            DataFrame with encoded condition features
        """
        if 'conditions' not in df.columns:
            print("   ⚠️  No 'conditions' column found, skipping condition encoding")
            return df
            
        print("🏷️  Encoding weather conditions...")
        df_feat = df.copy()
        
        # Convert to lowercase for consistent matching
        conditions_lower = df_feat['conditions'].str.lower().fillna('')
        
        # Define condition keywords
        condition_keywords = {
            'has_rain': ['rain', 'shower', 'drizzle', 'precipitation'],
            'has_cloud': ['cloud', 'overcast', 'partly'],
            'has_clear': ['clear', 'sunny'],
            'has_overcast': ['overcast'],
            'has_storm': ['storm', 'thunder'],
            'has_fog': ['fog', 'mist']
        }
        
        # Create binary features for each condition
        for feature_name, keywords in condition_keywords.items():
            df_feat[feature_name] = 0
            for keyword in keywords:
                df_feat[feature_name] |= conditions_lower.str.contains(keyword, na=False).astype(int)
        
        print(f"   ✅ Encoded conditions into {len(condition_keywords)} binary features")
        
        return df_feat
    
    def engineer_features(self, 
                         input_file: str = 'data/cleaned/weather_data_cleaned.csv',
                         output_file: str = 'data/processed/weather_features.csv') -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            input_file: Path to cleaned data
            output_file: Path to save engineered features
            
        Returns:
            DataFrame with all engineered features
        """
        print("⚙️  AgroWeather AI - Feature Engineering Pipeline")
        print("=" * 70)
        
        # Load cleaned data
        print("📂 Loading cleaned data...")
        df = pd.read_csv(input_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"✅ Loaded {len(df)} records")
        
        # Apply feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_lagged_features(df)
        df = self.create_rolling_features(df)
        df = self.create_derived_features(df)
        df = self.create_binary_features(df)
        df = self.encode_conditions(df)
        
        # Remove rows with NaN values (from lagged features)
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        final_rows = len(df)
        
        if initial_rows != final_rows:
            print(f"\n🧹 Removed {initial_rows - final_rows} rows with NaN values (from lagged features)")
        
        # Feature summary
        feature_cols = [col for col in df.columns if col not in ['datetime']]
        print(f"\n📊 Feature Engineering Summary:")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   Final dataset: {len(df)} records")
        print(f"   Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        # Save engineered features
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"💾 Saved engineered features to: {output_file}")
        
        print("\n" + "=" * 70)
        print("✅ FEATURE ENGINEERING COMPLETE!")
        
        return df