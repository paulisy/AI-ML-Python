"""
AgroWeather AI - Data Cleaning Pipeline
Step 1: Inspection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def inspect_data(df):
    """
    Comprehensive data inspection
    """
    print("🔍 DATA INSPECTION REPORT")
    print("=" * 70)
    
    # 1. Basic info
    print("\n📋 BASIC INFORMATION:")
    print(f"   Total records: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # 2. Data types
    print("\n📊 DATA TYPES:")
    print(df.dtypes)
    
    # 3. Missing values
    print("\n❓ MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    
    # 4. Statistical summary
    print("\n📈 STATISTICAL SUMMARY:")
    print(df.describe())
    
    # 5. Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔄 DUPLICATE ROWS: {duplicates}")
    if duplicates > 0:
        print(f"   ⚠️  Found {duplicates} duplicate rows")
    else:
        print("   ✅ No duplicates found!")
    
    # 6. Unique values in categorical columns
    print("\n🏷️  UNIQUE VALUES:")
    print(f"   Conditions: {df['conditions'].nunique()} unique weather types")
    print(f"   Sample conditions: {df['conditions'].value_counts().head()}")
    
    return missing, duplicates


def fix_data_types(df):
    """
    Convert columns to appropriate data types
    """
    print("\n🔧 FIXING DATA TYPES...")
    
    # 1. Convert datetime column
    print("   Converting 'datetime' to datetime format...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 2. Ensure numeric columns are float (not object/string)
    numeric_cols = [
        'tempmax', 'tempmin', 'temp_avg', 'humidity', 
        'rainfall', 'wind_speed', 'pressure', 'cloudcover', 'dew'
    ]
    
    for col in numeric_cols:
        if df[col].dtype == 'object':
            print(f"   ⚠️  Converting {col} from object to float...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Keep 'conditions' as string (categorical)
    df['conditions'] = df['conditions'].astype(str)
    
    print("   ✅ Data types fixed!")
    
    return df

def detect_outliers(df):
    """
    Detect and handle outliers using domain knowledge.
    """
    print("\n🔍 DETECTING OUTLIERS...")

    # Define realistic bounds for Nigeria (Aba region)
    bounds = {
        'tempmax': (15, 45),      # Max temp 15-45°C
        'tempmin': (10, 35),      # Min temp 10-35°C
        'temp_avg': (15, 40),     # Avg temp 15-40°C
        'humidity': (30, 100),    # Humidity 30-100%
        'rainfall': (0, 300),     # Rainfall 0-300mm (extreme but possible)
        'wind_speed': (0, 100),   # Wind 0-100 km/h
        'pressure': (980, 1050),  # Pressure 980-1050 hPa
        'cloudcover': (0, 100),   # Cloud cover 0-100%
        'dew': (5, 35),           # Dew point 5-35°C
    }

    outliers_found = {}

    for col, (min_val, max_val) in bounds.items():
        outliers = df[(df[col] < min_val) | (df[col] > max_val)]
        outliers_count = len(outliers)

        if outliers_count > 0:
            outliers_found[col] = outliers_count
            print(f"   ⚠️  {col}: {outliers_count} outliers found")
            print(f"      Range: {df[col].min():.2f} to {df[col].max():.2f}")
            print(f"      Expected: {min_val} to {max_val}")
    
    if not outliers_found:
        print("   ✅ No outliers detected!")
    
    return outliers_found, bounds

def handle_outliers(df, bounds):
    """
    Handle outliers by capping values to the min/max bounds.
    """
    print("\n🔧 HANDLING OUTLIERS...")

    for col, (min_val, max_val) in bounds.items():
        # Cap values at boundaries
        outliers_before = len(df[(df[col] < min_val) | (df[col] > max_val)])

        if outliers_before > 0:
             print(f"   Capping {col} to range [{min_val}, {max_val}]...")
             
             df[col] = df[col].clip(lower=min_val, upper=max_val)
             
             outliers_after = len(df[(df[col] < min_val) | (df[col] > max_val)])
             
             print(f"      Fixed {outliers_before - outliers_after} outliers")
    
    print("   ✅ Outliers handled!")
    
    return df


def encode_conditions_simple(df):
    """
    Convert 'conditions' text into binary features based on observed values
    """
    print("\n🔤 ENCODING 'CONDITIONS' COLUMN (ADAPTED)...")
    
    # Create binary features
    df['has_rain'] = df['conditions'].str.contains('Rain', case=False, na=False).astype(int)
    df['has_partly_cloudy'] = df['conditions'].str.contains('Partially cloudy', case=False, na=False).astype(int)
    df['has_overcast'] = df['conditions'].str.contains('Overcast', case=False, na=False).astype(int)
    df['has_clear'] = df['conditions'].str.contains('Clear', case=False, na=False).astype(int)
    
    # Drop the original text column
    df = df.drop('conditions', axis=1)
    
    print("   ✅ Created 4 binary features")
    
    return df


def engineer_features(df):
    """
    Create new features for ML training.
    """
    print("\n🔬 FEATURE ENGINEERING...")

    # Ensure that date is sorted
    df = df.sort_values('datetime').reset_index(drop=True)

    # 1. Date components
    print("     Creating date features....")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['day_of_the_year'] = df['datetime'].dt.dayofyear
    df['week_of_the_year'] = df['datetime'].dt.isocalendar().week
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday

    # 2. Season (based off Nigerian context)
    print("     Creating season features...")
    def get_season(month):
        if month in [11, 12, 1, 2, 3]:  # Nov-Mar
            return 'dry_season'
        else:  # Apr-Oct
            return 'rainy_season'
    
    df['Season'] = df['month'].apply(get_season)

    # Rolling averages (7-day window)
    print("     Calculating rolling averages...")
    df['rainfall_7day_avg'] = df['rainfall'].rolling(window=7, min_periods=1).mean()
    df['temp_avg_7day'] = df['temp_avg'].rolling(window=7, min_periods=1).mean()
    df['humidity_7day_avg'] = df['humidity'].rolling(window=7, min_periods=1).mean()

    # 4. Cumulative rainfall (useful for planting decisions)
    print("   Calculating cumulative rainfall...")
    df['rainfall_cumsum_30day'] = df['rainfall'].rolling(window=30, min_periods=1).sum()
    
    # 5. Lagged features (yesterday's values)
    # They allow models to learn:
    # - Momentum (“If it rained yesterday, it might rain today”)
    # - Delayed effects
    # - Seasonality and patterns
    print("   Creating lagged features...")
    df['rainfall_lag_1'] = df['rainfall'].shift(1)  # Yesterday
    df['rainfall_lag_7'] = df['rainfall'].shift(7)  # Last week
    df['temp_avg_lag_1'] = df['temp_avg'].shift(1)

    # 6. Temperature range (daily variation)
    print("   Calculating temperature range...")
    df['temp_range'] = df['tempmax'] - df['tempmin']

    # 7. Growing Degree Days (GDD) - Base 10°C for general crops
    print("   Calculating Growing Degree Days (GDD)...")
    df['gdd_base10'] = ((df['tempmax'] + df['tempmin']) / 2) - 10
    df['gdd_base10'] = df['gdd_base10'].clip(lower=0)  # GDD can't be negative
    
    # Cumulative GDD (running total)
    df['gdd_cumsum'] = df['gdd_base10'].cumsum()

    # 8. Binary features (useful for classification)
    print("   Creating binary features...")
    df['is_rainy_day'] = (df['rainfall'] > 1.0).astype(int)  # 1mm threshold
    df['is_heavy_rain'] = (df['rainfall'] > 10.0).astype(int)  # 10mm threshold
    df['is_hot_day'] = (df['tempmax'] > 33).astype(int)  # Hot threshold
    

    # print(f" Original length before dropping: {len(df)}")
    df = df.dropna().reset_index(drop=True)
    # print(f" Original length after dropping: {len(df)}")

    print(f"   ✅ Created {len(df.columns) - 15} new features!")
    print(f"   Total features now: {len(df.columns)}")

    print("   Creating threshold-based features...")
    # Humidity threshold (discovered from EDA)
    df['humidity_high'] = (df['humidity'] > 70).astype(int)
    df['humidity_very_high'] = (df['humidity'] > 85).astype(int)
    
    # Cloud cover threshold
    df['cloud_cover_high'] = (df['cloudcover'] > 70).astype(int)
    
    # Combined conditions (interaction features)
    df['humid_and_cloudy'] = ((df['humidity'] > 70) & 
                               (df['cloudcover'] > 70)).astype(int)
    
    df['rain_likely'] = ((df['humidity'] > 85) & 
                         (df['cloudcover'] > 70) & 
                         (df['rainfall_7day_avg'] > 5)).astype(int)
    
    # Encode text conditions
    df = encode_conditions_simple(df)
    
    return df


def save_cleaned_data(df):
    """
    Save cleaned data to CSV
    """
    print("\n💾 SAVING CLEANED DATA...")
    
    # Create output directory
    import os
    os.makedirs('data/cleaned', exist_ok=True)
    
    # Save
    output_file = 'data/cleaned/weather_data_cleaned.csv'
    df.to_csv(output_file, index=False)
    
    print(f"   ✅ Saved to: {output_file}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    
    return output_file


def generate_cleaning_report(df, output_file):
    """
    Generate summary report of cleaning process
    """
    print("\n" + "=" * 70)
    print("📊 DATA CLEANING SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\n✅ Cleaned data saved to: {output_file}")
    print(f"\n📈 Final dataset shape: {df.shape}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    
    print(f"\n📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Total days: {(df['datetime'].max() - df['datetime'].min()).days}")
    
    print(f"\n🌧️  Rainfall statistics:")
    print(f"   Total rainfall: {df['rainfall'].sum():.2f} mm")
    print(f"   Average daily: {df['rainfall'].mean():.2f} mm")
    print(f"   Rainy days: {(df['rainfall'] > 1.0).sum()} ({(df['rainfall'] > 1.0).sum() / len(df) * 100:.1f}%)")
    
    print(f"\n🌡️  Temperature statistics:")
    print(f"   Average temp: {df['temp_avg'].mean():.2f}°C")
    print(f"   Min recorded: {df['tempmin'].min():.2f}°C")
    print(f"   Max recorded: {df['tempmax'].max():.2f}°C")
    
    print(f"\n✅ Data quality:")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicates: {df.duplicated().sum()}")
    
    print("\n" + "=" * 70)
    print("✅ DATA CLEANING COMPLETE - READY FOR ML TRAINING!")
    print("=" * 70)




def main():
    """
    Complete data cleaning pipeline
    """
    print("🌦️  AgroWeather AI - Data Cleaning Pipeline")
    print("=" * 70)
    
    # Load raw data
    print("\n📂 Step 1: Loading raw data...")
    df = pd.read_csv('data/raw/nigerian_weather_raw.csv')
    print(f"✅ Loaded {len(df)} records")
    
    # Step 1: Inspect
    print("\n" + "=" * 70)
    missing, duplicates = inspect_data(df)

    sleep(10)
    
    # Step 2: Fix data types
    print("\n" + "=" * 70)
    df = fix_data_types(df)
    
    sleep(10)

    
    # Step 4: Detect and handle outliers
    print("\n" + "=" * 70)
    outliers_found, bounds = detect_outliers(df)
    if outliers_found:
        df = handle_outliers(df, bounds)
    
    sleep(10)

    
    # Step 5: Feature engineering
    print("\n" + "=" * 70)
    df = engineer_features(df)
    sleep(10)

    
    # Step 6: Save cleaned data
    print("\n" + "=" * 70)
    output_file = save_cleaned_data(df)
    sleep(10)

    
    # Generate report
    generate_cleaning_report(df, output_file)


if __name__ == "__main__":
    main()
