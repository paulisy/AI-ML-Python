"""
AgroWeather AI - Prepare Data for ML Training
Split data temporally and create sequences for LSTM
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_cleaned_data():
    """Load cleaned data"""
    print("📂 Loading cleaned data...")
    df = pd.read_csv('data/cleaned/weather_data_cleaned.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"✅ Loaded {len(df)} records")
    print(f"   Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    return df

def check_data_continuity(df):
    """Check for missing dates in time series"""
    print("\n🔍 Checking data continuity...")
    
    date_range = pd.date_range(start=df['datetime'].min(), 
                               end=df['datetime'].max(), 
                               freq='D')
    
    existing_dates = set(df['datetime'])
    expected_dates = set(date_range)
    missing_dates = expected_dates - existing_dates
    
    if missing_dates:
        print(f"   ⚠️  Found {len(missing_dates)} missing dates!")
        print(f"   First few missing: {sorted(list(missing_dates))[:5]}")
        return False
    else:
        print("   ✅ No missing dates - continuous time series!")
        return True


def temporal_split(df, train_end='2021-12-31', val_end='2022-12-31'):
    """
    Split data temporally (respecting time order)
    
    Args:
        df: DataFrame with datetime column
        train_end: Last date for training set
        val_end: Last date for validation set
        
    Returns:
        train_df, val_df, test_df
    """
    print("\n✂️  Performing temporal split...")
    
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    
    # Split
    train_df = df[df['datetime'] <= train_end].copy()
    val_df = df[(df['datetime'] > train_end) & (df['datetime'] <= val_end)].copy()
    test_df = df[df['datetime'] > val_end].copy()
    
    # Report
    print(f"\n   📊 Split Summary:")
    print(f"   TRAIN: {len(train_df)} records ({train_df['datetime'].min().date()} to {train_df['datetime'].max().date()})")
    print(f"          {len(train_df)/len(df)*100:.1f}% of data")
    
    print(f"   VAL:   {len(val_df)} records ({val_df['datetime'].min().date()} to {val_df['datetime'].max().date()})")
    print(f"          {len(val_df)/len(df)*100:.1f}% of data")
    
    print(f"   TEST:  {len(test_df)} records ({test_df['datetime'].min().date()} to {test_df['datetime'].max().date()})")
    print(f"          {len(test_df)/len(df)*100:.1f}% of data")
    
    # Validate split
    assert len(train_df) > 0, "Training set is empty!"
    assert len(val_df) > 0, "Validation set is empty!"
    assert len(test_df) > 0, "Test set is empty!"
    assert train_df['datetime'].max() < val_df['datetime'].min(), "Train/Val overlap!"
    assert val_df['datetime'].max() < test_df['datetime'].min(), "Val/Test overlap!"
    
    print("   ✅ Split validation passed!")
    
    return train_df, val_df, test_df


def select_features(df):
    """
    Select features for ML model based on EDA insights
    """
    print("\n🎯 Selecting features...")
    
    # Target variable
    target = 'rainfall'
    
    # Feature groups (organized by type)
    
    # 1. Core weather features (from EDA: high correlation with rainfall)
    core_features = [
        'tempmax', 'tempmin', 'temp_avg',
        'humidity', 'cloudcover', 'pressure',
        'wind_speed', 'dew'
    ]
    
    # 2. Temporal features (capture seasonality)
    temporal_features = [
        'month', 'day_of_year', 'week_of_year'
    ]
    
    # 3. Lagged features (capture persistence - KEY from EDA!)
    lagged_features = [
        'rainfall_lag_1', 'rainfall_lag_7',
        'temp_avg_lag_1'
    ]
    
    # 4. Rolling features (capture trends - TOP PREDICTOR from EDA!)
    rolling_features = [
        'rainfall_7day_avg', 'temp_avg_7day',
        'humidity_7day_avg', 'rainfall_cumsum_30day'
    ]
    
    # 5. Derived features
    derived_features = [
        'temp_range', 'gdd_base10'
    ]
    
    # 6. Binary features (from text encoding + thresholds)
    binary_features = [
        'is_rainy_day', 'is_heavy_rain', 'is_hot_day',
        'has_rain', 'has_cloud', 'has_clear', 'has_overcast',
        'humidity_high', 'humidity_very_high',
        'cloud_cover_high', 'humid_and_cloudy', 'rain_likely'
    ]
    
    # Combine all features
    all_features = (core_features + temporal_features + lagged_features + 
                   rolling_features + derived_features + binary_features)
    
    # Check which features actually exist in dataframe
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]
    
    if missing_features:
        print(f"   ⚠️  Missing features: {missing_features}")
    
    print(f"   ✅ Selected {len(available_features)} features")
    print(f"      Core weather: {len([f for f in core_features if f in available_features])}")
    print(f"      Temporal: {len([f for f in temporal_features if f in available_features])}")
    print(f"      Lagged: {len([f for f in lagged_features if f in available_features])}")
    print(f"      Rolling: {len([f for f in rolling_features if f in available_features])}")
    print(f"      Binary: {len([f for f in binary_features if f in available_features])}")
    
    return available_features, target

def scale_features(train_df, val_df, test_df, features):
    """
    Scale features using StandardScaler (fit only on training data!)
    """
    print("\n📏 Scaling features...")
    
    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit on TRAINING data only (prevent data leakage!)
    X_train = train_df[features].values
    y_train = train_df['rainfall'].values.reshape(-1, 1)
    
    feature_scaler.fit(X_train)
    target_scaler.fit(y_train)
    
    # Transform all sets
    X_train_scaled = feature_scaler.transform(X_train)
    X_val_scaled = feature_scaler.transform(val_df[features].values)
    X_test_scaled = feature_scaler.transform(test_df[features].values)
    
    y_train_scaled = target_scaler.transform(y_train)
    y_val_scaled = target_scaler.transform(val_df['rainfall'].values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(test_df['rainfall'].values.reshape(-1, 1))
    
    print("   ✅ Features scaled (mean=0, std=1)")
    print(f"      Feature scaler: mean={feature_scaler.mean_[:3]}, std={feature_scaler.scale_[:3]}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            feature_scaler, target_scaler)


def create_sequences(X, y, sequence_length=7):
    """
    Create sequences for LSTM
    
    For each prediction, LSTM looks at previous 'sequence_length' days
    
    Example (sequence_length=3):
        Input: [Day1, Day2, Day3] → Output: Day4
        Input: [Day2, Day3, Day4] → Output: Day5
    """
    print(f"\n🔄 Creating sequences (sequence_length={sequence_length})...")
    
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        # Input: previous 'sequence_length' days
        X_seq.append(X[i:i+sequence_length])
        # Output: next day's rainfall
        y_seq.append(y[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"   ✅ Created {len(X_seq)} sequences")
    print(f"      Input shape: {X_seq.shape}  (samples, time_steps, features)")
    print(f"      Output shape: {y_seq.shape}")
    
    return X_seq, y_seq

def save_processed_data(data_dict, output_dir='data/processed'):
    """Save all processed data"""
    print(f"\n💾 Saving processed data to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(f'{output_dir}/X_train.npy', data_dict['X_train'])
    np.save(f'{output_dir}/X_val.npy', data_dict['X_val'])
    np.save(f'{output_dir}/X_test.npy', data_dict['X_test'])
    np.save(f'{output_dir}/y_train.npy', data_dict['y_train'])
    np.save(f'{output_dir}/y_val.npy', data_dict['y_val'])
    np.save(f'{output_dir}/y_test.npy', data_dict['y_test'])
    
    # Save scalers
    with open(f'{output_dir}/feature_scaler.pkl', 'wb') as f:
        pickle.dump(data_dict['feature_scaler'], f)
    with open(f'{output_dir}/target_scaler.pkl', 'wb') as f:
        pickle.dump(data_dict['target_scaler'], f)
    
    # Save metadata
    metadata = {
        'features': data_dict['features'],
        'target': data_dict['target'],
        'sequence_length': data_dict['sequence_length'],
        'train_size': len(data_dict['X_train']),
        'val_size': len(data_dict['X_val']),
        'test_size': len(data_dict['X_test']),
        'n_features': data_dict['X_train'].shape[2]
    }
    
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("   ✅ All data saved!")
    print(f"      Files: X_train.npy, X_val.npy, X_test.npy, etc.")


def main():
    """
    Main data preparation pipeline
    """
    print("🌦️  AgroWeather AI - ML Data Preparation")
    print("=" * 70)
    
    # Step 1: Load data
    df = load_cleaned_data()
    
    # Step 2: Check continuity
    is_continuous = check_data_continuity(df)
    if not is_continuous:
        print("   ⚠️  Warning: Time series has gaps. LSTM may struggle.")
        print("   Consider filling gaps or using different sequence approach.")
    
    # Step 3: Temporal split
    train_df, val_df, test_df = temporal_split(df)
    
    # Step 4: Select features
    features, target = select_features(df)
    
    # Step 5: Scale features
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train_scaled, y_val_scaled, y_test_scaled,
     feature_scaler, target_scaler) = scale_features(train_df, val_df, test_df, features)
    
    # Step 6: Create sequences for LSTM
    sequence_length = 7  # Use 7 days of history (matches rainfall_7day_avg!)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
    
    # Step 7: Save everything
    data_dict = {
        'X_train': X_train_seq,
        'X_val': X_val_seq,
        'X_test': X_test_seq,
        'y_train': y_train_seq,
        'y_val': y_val_seq,
        'y_test': y_test_seq,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'features': features,
        'target': target,
        'sequence_length': sequence_length
    }
    
    save_processed_data(data_dict)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Dataset Shapes:")
    print(f"   Training:   {X_train_seq.shape} → {y_train_seq.shape}")
    print(f"   Validation: {X_val_seq.shape} → {y_val_seq.shape}")
    print(f"   Testing:    {X_test_seq.shape} → {y_test_seq.shape}")
    print(f"\n🎯 Ready for LSTM training!")


if __name__ == "__main__":
    main()


