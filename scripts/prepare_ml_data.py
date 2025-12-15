#!/usr/bin/env python3
"""
Standalone script for preparing ML-ready data
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from utils.helpers import setup_logging
import argparse


def temporal_split(df, train_end='2021-12-31', val_end='2022-12-31'):
    """Split data temporally"""
    print("✂️  Performing temporal split...")
    
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    
    train_df = df[df['datetime'] <= train_end].copy()
    val_df = df[(df['datetime'] > train_end) & (df['datetime'] <= val_end)].copy()
    test_df = df[df['datetime'] > val_end].copy()
    
    print(f"   TRAIN: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   VAL:   {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   TEST:  {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def select_features(df):
    """Select features for ML model"""
    print("🎯 Selecting features...")
    
    # Core features
    core_features = [
        'tempmax', 'tempmin', 'temp_avg',
        'humidity', 'cloudcover', 'pressure',
        'wind_speed', 'dew'
    ]
    
    # Temporal features
    temporal_features = [
        'month', 'day_of_year', 'week_of_year'
    ]
    
    # Lagged features
    lagged_features = [col for col in df.columns if 'lag_' in col]
    
    # Rolling features
    rolling_features = [col for col in df.columns if ('day_avg' in col or 'day_sum' in col or 'day_max' in col or 'cumsum' in col)]
    
    # Binary features
    binary_features = [col for col in df.columns if col.startswith(('is_', 'has_', 'humidity_', 'cloud_cover_', 'rain_likely'))]
    
    # Derived features
    derived_features = ['temp_range', 'gdd_base10']
    
    # Combine all features
    all_features = (core_features + temporal_features + lagged_features + 
                   rolling_features + binary_features + derived_features)
    
    # Check which features exist
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"   ✅ Selected {len(available_features)} features")
    
    return available_features, 'rainfall'


def scale_features(train_df, val_df, test_df, features):
    """Scale features using StandardScaler"""
    print("📏 Scaling features...")
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit on training data only
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
    
    print("   ✅ Features scaled")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            feature_scaler, target_scaler)


def create_sequences(X, y, sequence_length=7):
    """Create sequences for LSTM"""
    print(f"🔄 Creating sequences (length={sequence_length})...")
    
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"   ✅ Created {len(X_seq)} sequences")
    print(f"      Input shape: {X_seq.shape}")
    print(f"      Output shape: {y_seq.shape}")
    
    return X_seq, y_seq


def save_processed_data(data_dict, output_dir='data/processed'):
    """Save all processed data"""
    print(f"💾 Saving processed data to {output_dir}...")
    
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


def main():
    """Main ML data preparation script"""
    parser = argparse.ArgumentParser(description='Prepare ML-ready data from engineered features')
    parser.add_argument('--input', type=str, default='data/processed/weather_features.csv',
                       help='Input features file')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--sequence-length', type=int, default=7,
                       help='Sequence length for LSTM')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        print("🌦️  AgroWeather AI - ML Data Preparation")
        print("=" * 70)
        
        # Load data
        print("📂 Loading engineered features...")
        df = pd.read_csv(args.input)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"✅ Loaded {len(df)} records")
        
        # Temporal split
        train_df, val_df, test_df = temporal_split(df)
        
        # Select features
        features, target = select_features(df)
        
        # Scale features
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train_scaled, y_val_scaled, y_test_scaled,
         feature_scaler, target_scaler) = scale_features(train_df, val_df, test_df, features)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, args.sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, args.sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, args.sequence_length)
        
        # Save everything
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
            'sequence_length': args.sequence_length
        }
        
        save_processed_data(data_dict, args.output_dir)
        
        print("\n" + "=" * 70)
        print("✅ ML DATA PREPARATION COMPLETE!")
        
        logger.info("ML data preparation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"ML data preparation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())