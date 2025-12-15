# AgroWeather AI - Rainfall Prediction System

A machine learning system for predicting rainfall patterns in Nigeria using LSTM neural networks and historical weather data.

## Project Overview

This project uses historical weather data from Visual Crossing API to train an LSTM model for rainfall prediction in Nigerian agricultural regions, specifically focusing on Aba, Abia State.

## Features

- **Data Collection**: Automated weather data collection from Visual Crossing API
- **Data Processing**: Comprehensive data cleaning and feature engineering
- **ML Model**: LSTM neural network for time series rainfall prediction
- **Evaluation**: Model performance metrics and visualization

## Project Structure

```
agroweather-ai/
├── src/                    # Source code
│   ├── data/              # Data handling modules
│   ├── models/            # ML models and training
│   └── utils/             # Utility functions
├── data/                  # Data storage
│   ├── raw/              # Raw collected data
│   ├── processed/        # Processed ML-ready data
│   └── cleaned/          # Cleaned data
├── models/               # Saved model artifacts
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit tests
└── scripts/             # Standalone scripts
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file
4. Run data collection: `python scripts/collect_data.py`
5. Train model: `python scripts/train_model.py`

## Usage

See individual scripts and notebooks for detailed usage instructions.

## Model Architecture

- **Input**: 7-day sequences of 40+ weather features
- **Architecture**: 2-layer LSTM (64 units each) + Dense layers
- **Output**: Next-day rainfall prediction
- **Features**: Temperature, humidity, pressure, wind, derived features

## Data Sources

- **Weather Data**: Visual Crossing API
- **Location**: Aba, Abia State, Nigeria (5.1156°N, 7.3636°E)
- **Time Range**: 2014-2024

## License

MIT License