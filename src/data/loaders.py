"""
Data loading utilities
"""
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, Any, Optional


def load_raw_data(file_path: str = 'data/raw/nigerian_weather_raw.csv') -> pd.DataFrame:
    """
    Load raw weather data
    
    Args:
        file_path: Path to raw data file
        
    Returns:
        Raw weather DataFrame
    """
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_cleaned_data(file_path: str = 'data/cleaned/weather_data_cleaned.csv') -> pd.DataFrame:
    """
    Load cleaned weather data
    
    Args:
        file_path: Path to cleaned data file
        
    Returns:
        Cleaned weather DataFrame
    """
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_processed_data(data_dir: str = 'data/processed') -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load processed ML-ready data
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        Tuple of (data_dict, metadata_dict)
    """
    # Load numpy arrays
    data = {
        'X_train': np.load(f'{data_dir}/X_train.npy'),
        'X_val': np.load(f'{data_dir}/X_val.npy'),
        'X_test': np.load(f'{data_dir}/X_test.npy'),
        'y_train': np.load(f'{data_dir}/y_train.npy'),
        'y_val': np.load(f'{data_dir}/y_val.npy'),
        'y_test': np.load(f'{data_dir}/y_test.npy'),
    }
    
    # Load scalers
    with open(f'{data_dir}/feature_scaler.pkl', 'rb') as f:
        data['feature_scaler'] = pickle.load(f)
    
    with open(f'{data_dir}/target_scaler.pkl', 'rb') as f:
        data['target_scaler'] = pickle.load(f)
    
    # Load metadata
    with open(f'{data_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return data, metadata


def load_model_artifacts(model_dir: str = 'models/saved') -> Dict[str, Any]:
    """
    Load saved model and training artifacts
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Dictionary with model artifacts
    """
    artifacts = {}
    
    # Load training history if available
    try:
        with open(f'{model_dir}/training_history.pkl', 'rb') as f:
            artifacts['training_history'] = pickle.load(f)
    except FileNotFoundError:
        artifacts['training_history'] = None
    
    # Load training metadata if available
    try:
        with open(f'{model_dir}/training_metadata.pkl', 'rb') as f:
            artifacts['training_metadata'] = pickle.load(f)
    except FileNotFoundError:
        artifacts['training_metadata'] = None
    
    return artifacts