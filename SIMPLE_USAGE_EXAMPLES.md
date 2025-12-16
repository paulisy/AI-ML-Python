# 🎯 Simple Usage Examples - No Confusion!

## 😊 Don't Worry - Here's How to Use Everything Simply

I know the refactoring looks complex, but you can use it just like before! Here are the simplest ways to do common tasks.

## 🚀 **The Easiest Way - One Command Does Everything**

```bash
# This runs your entire pipeline automatically:
python scripts/run_full_pipeline.py

# That's it! It will:
# 1. Collect weather data
# 2. Clean the data
# 3. Create features
# 4. Prepare ML data
# 5. Train the model
# 6. Evaluate the model
```

## 📋 **Step-by-Step (If You Want Control)**

### **Step 1: Get Weather Data**
```bash
python scripts/collect_data.py
# Downloads weather data to data/raw/
```

### **Step 2: Clean the Data**
```bash
python scripts/clean_data.py
# Cleans data and saves to data/cleaned/
```

### **Step 3: Create Features**
```bash
python scripts/engineer_features.py
# Creates ML features and saves to data/processed/
```

### **Step 4: Prepare for Machine Learning**
```bash
python scripts/prepare_ml_data.py
# Creates training/test sets in data/processed/
```

### **Step 5: Train Your Model**
```bash
python scripts/train_model.py
# Trains LSTM model and saves to models/saved/
```

### **Step 6: Test Your Model**
```bash
python scripts/evaluate_model.py
# Creates evaluation plots in outputs/
```

## 🔍 **Check What You Have**

```bash
python scripts/project_info.py
# Shows you all your files and their status
```

## 💻 **Using in Your Own Python Scripts**

### **Simple Example - Collect Data**
```python
# Add this at the top of your script
import sys
sys.path.append('src')

# Now you can use the modules
from data.collectors import WeatherCollector

# Collect data
collector = WeatherCollector()
collector.collect_data(start_date='2024-01-01', end_date='2024-12-31')
print("Data collected!")
```

### **Simple Example - Train Model**
```python
import sys
sys.path.append('src')

from models import RainfallLSTM, ModelTrainer
from data.loaders import load_processed_data

# Load data
data, metadata = load_processed_data()

# Create and train model
model = RainfallLSTM(input_size=metadata['n_features'])
trainer = ModelTrainer(model)

# Train
trainer.train(
    X_train=data['X_train'],
    y_train=data['y_train'],
    X_val=data['X_val'],
    y_val=data['y_val'],
    epochs=50
)
print("Model trained!")
```

### **Simple Example - Evaluate Model**
```python
import sys
sys.path.append('src')

from models import ModelEvaluator

# Evaluate model
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model()

print(f"Model RMSE: {metrics['rmse']:.2f} mm")
print(f"Model R²: {metrics['r2']:.3f}")
```

## 📁 **Where to Find Your Results**

### **Your Data:**
- `data/raw/nigerian_weather_raw.csv` - Raw weather data
- `data/cleaned/weather_data_cleaned.csv` - Cleaned data
- `data/processed/` - ML-ready data files

### **Your Models:**
- `models/saved/rainfall_lstm_best.pth` - Your best trained model
- `models/saved/training_history.pkl` - Training progress

### **Your Results:**
- `outputs/plots/model_evaluation.png` - Model performance plots
- `outputs/reports/evaluation_report.txt` - Performance summary

## 🎛️ **Customizing Commands**

### **Collect Specific Date Range:**
```bash
python scripts/collect_data.py --start-date 2023-01-01 --end-date 2023-12-31
```

### **Train with More Epochs:**
```bash
python scripts/train_model.py --epochs 200
```

### **Run Pipeline with Options:**
```bash
python scripts/run_full_pipeline.py --epochs 100 --skip-collection
```

## 🔧 **Quick Fixes for Common Issues**

### **"Module not found" Error:**
```python
# Add this at the top of your Python scripts:
import sys
sys.path.append('src')
```

### **"No data found" Error:**
```bash
# Check what you have:
python scripts/project_info.py

# If no data, collect it:
python scripts/collect_data.py
```

### **"Model not found" Error:**
```bash
# Train a model first:
python scripts/train_model.py
```

## 📊 **Understanding Your Results**

After running evaluation, you'll see something like:
```
🎯 REGRESSION METRICS:
   RMSE:  14.368 mm/day  ← Lower is better (< 10 is good)
   R² Score: 0.053       ← Higher is better (> 0.7 is good)

🎲 CLASSIFICATION METRICS:
   Accuracy:  56.4%      ← Higher is better (> 80% is good)
```

## 🎯 **Most Common Workflows**

### **"I want to experiment with the model"**
```bash
# Quick training with different settings:
python scripts/train_model.py --epochs 50
python scripts/evaluate_model.py

python scripts/train_model.py --epochs 100  
python scripts/evaluate_model.py

# Compare the results!
```

### **"I want to collect more recent data"**
```bash
python scripts/collect_data.py --start-date 2024-01-01 --end-date 2024-12-31
python scripts/clean_data.py
python scripts/engineer_features.py
python scripts/prepare_ml_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

### **"I want to use this in a Jupyter notebook"**
```python
# In your notebook cell:
import sys
sys.path.append('src')

# Now use any module:
from models import RainfallLSTM
from data.collectors import WeatherCollector
# etc.
```

## 🎉 **Remember**

1. **Use `scripts/run_full_pipeline.py` for everything at once**
2. **Use individual scripts in `scripts/` folder for step-by-step**
3. **Use `python scripts/project_info.py` to see what you have**
4. **Add `sys.path.append('src')` to use modules in your own code**
5. **All your old functionality is still there, just organized better!**

The refactoring made your code professional and maintainable, but you can still use it simply! 🚀