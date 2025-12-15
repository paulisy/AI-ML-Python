"""
Configuration settings for AgroWeather AI
"""
import os
from decouple import config

# API Configuration
VISUAL_CROSSING_API_KEY = config('VISUAL_CROSSING_API_KEY', default='')

# Data Configuration
DATA_CONFIG = {
    'raw_data_path': 'data/raw/nigerian_weather_raw.csv',
    'cleaned_data_path': 'data/cleaned/weather_data_cleaned.csv',
    'processed_data_dir': 'data/processed',
    'location': {
        'name': 'Aba',
        'lat': 5.1156,
        'long': 7.3636
    }
}

# Model Configuration
MODEL_CONFIG = {
    'sequence_length': 7,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 10
}

# Training Configuration
TRAINING_CONFIG = {
    'train_split_end': '2021-12-31',
    'val_split_end': '2022-12-31',
    'random_seed': 42,
    'model_save_dir': 'models/saved'
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'lag_periods': [1, 2, 3, 7, 14, 30],
    'rolling_windows': [3, 7, 14, 30],
    'rainfall_thresholds': {
        'light_rain': 1.0,
        'heavy_rain': 10.0,
        'extreme_rain': 50.0
    },
    'temperature_thresholds': {
        'hot_day': 35.0,
        'cool_day': 20.0
    },
    'humidity_thresholds': {
        'high': 80.0,
        'very_high': 90.0
    }
}

# Paths Configuration
PATHS = {
    'data': {
        'raw': 'data/raw',
        'cleaned': 'data/cleaned',
        'processed': 'data/processed'
    },
    'models': 'models/saved',
    'outputs': {
        'plots': 'outputs/plots',
        'reports': 'outputs/reports'
    },
    'logs': 'logs',
    'notebooks': 'notebooks'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/agroweather.log'
}