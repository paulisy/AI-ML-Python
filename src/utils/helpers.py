"""
Helper utilities for AgroWeather AI
"""
import logging
import torch
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('agroweather')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device() -> str:
    """
    Get the best available device for PyTorch
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    
    return device


def save_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        file_path: Path to save config file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Add timestamp
    config['created_at'] = datetime.now().isoformat()
    
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Configuration saved to: {file_path}")


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        file_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    print(f"📂 Configuration loaded from: {file_path}")
    return config


def create_directory_structure(base_path: str = '.') -> None:
    """
    Create the standard project directory structure
    
    Args:
        base_path: Base path for the project
    """
    directories = [
        'data/raw',
        'data/cleaned', 
        'data/processed',
        'models/saved',
        'outputs/plots',
        'outputs/reports',
        'logs',
        'config'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
    
    print("📁 Directory structure created successfully!")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def print_data_info(data_dict: Dict[str, Any]) -> None:
    """
    Print information about loaded data
    
    Args:
        data_dict: Dictionary containing data arrays
    """
    print("\n📊 Dataset Information:")
    print("-" * 40)
    
    for key, value in data_dict.items():
        if hasattr(value, 'shape'):
            print(f"   {key:12s}: {str(value.shape):15s} | {value.dtype}")
        else:
            print(f"   {key:12s}: {type(value).__name__}")
    
    print("-" * 40)