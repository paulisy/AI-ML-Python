# ⚡ Quick Reference - AgroWeather AI

## 🎯 **I Just Want To...**

### **Run Everything:**
```bash
python scripts/run_full_pipeline.py
```

### **See What I Have:**
```bash
python scripts/project_info.py
```

### **Train a Model:**
```bash
python scripts/train_model.py
```

### **Test My Model:**
```bash
python scripts/evaluate_model.py
```

### **Get New Data:**
```bash
python scripts/collect_data.py
```

## 📁 **Where Is Everything?**

| What You Want | Where It Is |
|---------------|-------------|
| **Ready-to-use scripts** | `scripts/` folder |
| **Your data files** | `data/` folder |
| **Your trained models** | `models/saved/` folder |
| **Results & plots** | `outputs/` folder |
| **Code modules** | `src/` folder |

## 🔄 **Old vs New Locations**

| Old Location | New Location | Easy Script |
|-------------|-------------|-------------|
| `scripts/scrape_data/collect_weather.py` | `src/data/collectors/weather_collector.py` | `scripts/collect_data.py` |
| `scripts/clean_data_and_feature_extraction/` | `src/data/processors/` | `scripts/clean_data.py` |
| `scripts/models/lstm_model.py` | `src/models/lstm_model.py` | `scripts/train_model.py` |
| `scripts/models/prepare_ml_data.py` | `scripts/prepare_ml_data.py` | `scripts/prepare_ml_data.py` |

## 💻 **Using in Python Code**

```python
# Always add this first:
import sys
sys.path.append('src')

# Then import what you need:
from data.collectors import WeatherCollector
from data.processors import WeatherDataCleaner, FeatureEngineer  
from models import RainfallLSTM, ModelTrainer, ModelEvaluator
from data.loaders import load_processed_data
```

## 🎛️ **Command Options**

```bash
# Collect data for specific dates:
python scripts/collect_data.py --start-date 2024-01-01 --end-date 2024-12-31

# Train with more epochs:
python scripts/train_model.py --epochs 200

# Run pipeline with options:
python scripts/run_full_pipeline.py --epochs 100 --skip-collection

# Get help for any script:
python scripts/train_model.py --help
```

## 📊 **Key Files**

| File | What It Does |
|------|-------------|
| `scripts/run_full_pipeline.py` | **Runs everything automatically** |
| `scripts/project_info.py` | **Shows what files you have** |
| `data/raw/nigerian_weather_raw.csv` | **Your raw weather data** |
| `models/saved/rainfall_lstm_best.pth` | **Your best trained model** |
| `outputs/plots/model_evaluation.png` | **Model performance charts** |
| `.env` | **Your API keys** |

## 🆘 **Quick Fixes**

| Problem | Solution |
|---------|----------|
| "Module not found" | Add `sys.path.append('src')` to your script |
| "No data found" | Run `python scripts/collect_data.py` |
| "Model not found" | Run `python scripts/train_model.py` |
| "Don't know what I have" | Run `python scripts/project_info.py` |
| "Want to start over" | Run `python scripts/run_full_pipeline.py` |

## 🎯 **Most Used Commands**

```bash
# The Big Three:
python scripts/run_full_pipeline.py    # Do everything
python scripts/project_info.py         # Check status  
python scripts/evaluate_model.py       # Test model

# Individual steps:
python scripts/collect_data.py         # Get data
python scripts/train_model.py          # Train model
python scripts/evaluate_model.py       # Test model
```

## 📈 **Understanding Results**

When you run evaluation, look for:
- **RMSE < 10 mm**: Good model
- **R² > 0.7**: Strong predictions  
- **Accuracy > 80%**: Good rain detection

## 🚀 **Getting Started**

1. **First time?** Run: `python scripts/run_full_pipeline.py`
2. **Check results:** Run: `python scripts/project_info.py`
3. **Experiment:** Run: `python scripts/train_model.py --epochs 100`
4. **Test changes:** Run: `python scripts/evaluate_model.py`

That's it! Everything else is just details. 🎉