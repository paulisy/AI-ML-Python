"""
Weather data cleaning and preprocessing
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os


class WeatherDataCleaner:
    """
    Handles cleaning and preprocessing of raw weather data
    """
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def load_raw_data(self, file_path: str = 'data/raw/nigerian_weather_raw.csv') -> pd.DataFrame:
        """
        Load raw weather data
        
        Args:
            file_path: Path to raw data file
            
        Returns:
            Raw weather DataFrame
        """
        print(f"📂 Loading raw data from {file_path}...")
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} records")
        print(f"   Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        return df
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Analyze data quality and missing values
        
        Args:
            df: Weather DataFrame
            
        Returns:
            Dictionary with quality statistics
        """
        print("\n🔍 Analyzing data quality...")
        
        stats = {
            'total_records': len(df),
            'date_range': (df['datetime'].min(), df['datetime'].max()),
            'missing_values': {},
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            stats['missing_values'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            if missing_count > 0:
                print(f"   ⚠️  {col}: {missing_count} missing ({missing_pct:.1f}%)")
        
        if stats['duplicates'] > 0:
            print(f"   ⚠️  Found {stats['duplicates']} duplicate records")
        
        self.cleaning_stats['quality_check'] = stats
        return stats
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies
        
        Args:
            df: Weather DataFrame with missing values
            
        Returns:
            DataFrame with missing values handled
        """
        print("\n🔧 Handling missing values...")
        df_clean = df.copy()
        
        # Strategy 1: Forward fill for weather conditions (assumes persistence)
        weather_cols = ['tempmax', 'tempmin', 'temp_avg', 'humidity', 'pressure', 'cloudcover', 'dew']
        for col in weather_cols:
            if col in df_clean.columns:
                missing_before = df_clean[col].isnull().sum()
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                missing_after = df_clean[col].isnull().sum()
                if missing_before > 0:
                    print(f"   {col}: {missing_before} → {missing_after} missing")
        
        # Strategy 2: Zero fill for rainfall (missing often means no rain)
        if 'rainfall' in df_clean.columns:
            missing_before = df_clean['rainfall'].isnull().sum()
            df_clean['rainfall'] = df_clean['rainfall'].fillna(0)
            if missing_before > 0:
                print(f"   rainfall: {missing_before} → 0 missing (filled with 0)")
        
        # Strategy 3: Interpolate wind speed
        if 'wind_speed' in df_clean.columns:
            missing_before = df_clean['wind_speed'].isnull().sum()
            df_clean['wind_speed'] = df_clean['wind_speed'].interpolate()
            missing_after = df_clean['wind_speed'].isnull().sum()
            if missing_before > 0:
                print(f"   wind_speed: {missing_before} → {missing_after} missing")
        
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove extreme outliers using IQR method
        
        Args:
            df: Weather DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        print("\n🎯 Removing extreme outliers...")
        df_clean = df.copy()
        
        # Define columns to check for outliers
        numeric_cols = ['tempmax', 'tempmin', 'temp_avg', 'humidity', 'rainfall', 'wind_speed', 'pressure']
        
        total_removed = 0
        
        for col in numeric_cols:
            if col in df_clean.columns:
                # Calculate IQR
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds (more conservative for weather data)
                lower_bound = Q1 - 3 * IQR  # 3 IQR instead of 1.5
                upper_bound = Q3 + 3 * IQR
                
                # Count outliers
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    print(f"   {col}: {outlier_count} outliers (< {lower_bound:.1f} or > {upper_bound:.1f})")
                    # Remove outliers
                    df_clean = df_clean[~outliers]
                    total_removed += outlier_count
        
        if total_removed > 0:
            print(f"   ✅ Removed {total_removed} outlier records")
            print(f"   📊 Dataset: {len(df)} → {len(df_clean)} records")
        
        return df_clean
    
    def validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that data values are within reasonable ranges
        
        Args:
            df: Weather DataFrame
            
        Returns:
            DataFrame with invalid values handled
        """
        print("\n✅ Validating data ranges...")
        df_clean = df.copy()
        
        # Define reasonable ranges for Nigerian weather
        ranges = {
            'tempmax': (15, 50),      # Max temp: 15-50°C
            'tempmin': (10, 40),      # Min temp: 10-40°C  
            'temp_avg': (15, 45),     # Avg temp: 15-45°C
            'humidity': (0, 100),     # Humidity: 0-100%
            'rainfall': (0, 500),     # Daily rainfall: 0-500mm (extreme but possible)
            'wind_speed': (0, 100),   # Wind speed: 0-100 km/h
            'pressure': (900, 1100),  # Pressure: 900-1100 hPa
            'cloudcover': (0, 100),   # Cloud cover: 0-100%
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in df_clean.columns:
                invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    print(f"   ⚠️  {col}: {invalid_count} values outside range [{min_val}, {max_val}]")
                    # Clip to valid range
                    df_clean[col] = df_clean[col].clip(min_val, max_val)
        
        return df_clean
    
    def clean_data(self, 
                   input_file: str = 'data/raw/nigerian_weather_raw.csv',
                   output_file: str = 'data/cleaned/weather_data_cleaned.csv') -> pd.DataFrame:
        """
        Complete data cleaning pipeline
        
        Args:
            input_file: Path to raw data file
            output_file: Path to save cleaned data
            
        Returns:
            Cleaned DataFrame
        """
        print("🧹 AgroWeather AI - Data Cleaning Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        df = self.load_raw_data(input_file)
        
        # Step 2: Quality check
        self.check_data_quality(df)
        
        # Step 3: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 4: Remove duplicates
        if df.duplicated().sum() > 0:
            print(f"\n🔄 Removing {df.duplicated().sum()} duplicate records...")
            df = df.drop_duplicates().reset_index(drop=True)
        
        # Step 5: Remove outliers
        df = self.remove_outliers(df)
        
        # Step 6: Validate ranges
        df = self.validate_data_ranges(df)
        
        # Step 7: Final validation
        print(f"\n📊 Final dataset: {len(df)} records")
        print(f"   Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        # Step 8: Save cleaned data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"💾 Saved cleaned data to: {output_file}")
        
        print("\n" + "=" * 60)
        print("✅ DATA CLEANING COMPLETE!")
        
        return df