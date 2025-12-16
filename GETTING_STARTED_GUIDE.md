# 🚀 AgroWeather AI - Getting Started Guide

## 😅 Feeling Lost After Refactoring? You're Not Alone!

The refactoring created a professional structure, but I know it can feel overwhelming. This guide will help you understand and navigate the new organization easily.

## 🎯 **Quick Navigation - What You Need Most**

### **Just Want to Run Things? Use These Commands:**

```bash
# Run everything at once (easiest option)
python scripts/run_full_pipeline.py

# Or run step by step:
python scripts/collect_data.py      # Get weather data
python scripts/clean_data.py        # Clean the data  
python scripts/engineer_features.py # Create features
python scripts/prepare_ml_data.py   # Prepare for ML
python scripts/train_model.py       # Train the model
python scripts/evaluate_model.py    # Evaluate results

# Check what you have:
python scripts/project_info.py
```

### **Want to Use the Code in Python? Import Like This:**

```python
# Data collection
from src.data.collectors import WeatherCollector
collector = WeatherCollector()

# Data cleaning  
from src.data.processors import WeatherDataCleaner
cleaner = WeatherDataCleaner()

# Model training
from src.models import RainfallLSTM, ModelTrainer
model = RainfallLSTM(input_size=40)
trainer = ModelTrainer(model)

# Model evaluation
from src.models import ModelEvaluator
evaluator = ModelEvaluator()
```

## 📁 **Simple Structure Explanation**

Think of it like organizing your house:

```
🏠 Your Project House
├── 📚 src/                    # The "library" - organized code you can import
│   ├── 🔧 data/              # Data handling tools
│   ├── 🤖 models/            # AI model tools  
│   └── 🛠️ utils/             # Helper tools
├── 🗃️ data/                  # The "filing cabinet" - all your data
├── 💾 models/                # The "safe" - saved AI models
├── 📋 scripts/               # The "toolbox" - ready-to-use scripts
├── 🧪 tests/                 # The "quality check" - tests
└── 📊 outputs/               # The "results folder" - plots and reports
```

## 🔄 **Before vs After - What Changed**

### **OLD WAY (What You're Used To):**
```
scripts/
├── scrape_data/collect_weather.py          # ❌ Gone
├── clean_data_and_feature_extraction/      # ❌ Gone  
├── models/lstm_model.py                    # ❌ Moved
└── models/prepare_ml_data.py               # ❌ Moved
```

### **NEW WAY (Where Everything Went):**
```
scripts/
├── collect_data.py          # ✅ Same function, cleaner
├── clean_data.py            # ✅ Same function, cleaner
├── engineer_features.py     # ✅ Same function, cleaner
├── prepare_ml_data.py       # ✅ Same function, cleaner
├── train_model.py           # ✅ Same function, cleaner
└── evaluate_model.py        # ✅ NEW! Evaluate your model

src/                         # ✅ NEW! Organized, importable code
├── data/collectors/         # ✅ Weather collection (was in scrape_data/)
├── data/processors/         # ✅ Cleaning & features (was in clean_data_and_feature_extraction/)
└── models/                  # ✅ AI models (was in scripts/models/)
```

## 🎯 **Most Common Tasks - How to Do Them**

### **1. "I want to collect new weather data"**
```bash
# Easy way:
python scripts/collect_data.py

# With options:
python scripts/collect_data.py --start-date 2024-01-01 --end-date 2024-12-31
```

### **2. "I want to train a new model"**
```bash
# Easy way (runs everything):
python scripts/run_full_pipeline.py

# Just training:
python scripts/train_model.py --epochs 100
```

### **3. "I want to see how good my model is"**
```bash
python scripts/evaluate_model.py
# Creates plots in outputs/plots/ and reports in outputs/reports/
```

### **4. "I want to use the code in my own script"**
```python
# Example: Collect data in your own script
import sys
sys.path.append('src')  # Add this line to use the modules

from data.collectors import WeatherCollector

collector = WeatherCollector()
collector.collect_data(start_date='2024-01-01', end_date='2024-12-31')
```

### **5. "I want to modify the LSTM model"**
The model code is now in: `src/models/lstm_model.py`
```python
# To use it:
from src.models import RainfallLSTM

# Create model
model = RainfallLSTM(input_size=40, hidden_size=64, num_layers=2)
```

## 🔍 **Finding Your Old Code**

### **Looking for weather collection?**
- **Old location**: `scripts/scrape_data/collect_weather.py`
- **New location**: `src/data/collectors/weather_collector.py`
- **Easy script**: `scripts/collect_data.py`

### **Looking for data cleaning?**
- **Old location**: `scripts/clean_data_and_feature_extraction/clean_weather_data.py`
- **New location**: `src/data/processors/cleaner.py`
- **Easy script**: `scripts/clean_data.py`

### **Looking for the LSTM model?**
- **Old location**: `scripts/models/lstm_model.py`
- **New location**: `src/models/lstm_model.py`
- **Easy script**: `scripts/train_model.py`

### **Looking for ML data preparation?**
- **Old location**: `scripts/models/prepare_ml_data.py`
- **New location**: `src/data/processors/feature_engineer.py` + `scripts/prepare_ml_data.py`
- **Easy script**: `scripts/prepare_ml_data.py`

## 🛠️ **Development Workflow**

### **For Quick Experiments:**
```bash
# Use the simple scripts in scripts/ folder
python scripts/collect_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

### **For Serious Development:**
```python
# Import the modules from src/ folder
from src.models import RainfallLSTM, ModelTrainer
from src.data.collectors import WeatherCollector

# Now you can use them in your code
```

## 📚 **Key Files You Should Know**

### **Most Important Scripts:**
1. `scripts/run_full_pipeline.py` - **Runs everything automatically**
2. `scripts/project_info.py` - **Shows you what files you have**
3. `scripts/train_model.py` - **Trains your AI model**
4. `scripts/evaluate_model.py` - **Tests your model performance**

### **Configuration:**
1. `.env` - **Your API keys and settings**
2. `config/settings.py` - **Project configuration**
3. `requirements.txt` - **Python packages needed**

### **Your Data:**
1. `data/raw/` - **Raw weather data from API**
2. `data/cleaned/` - **Cleaned, ready-to-use data**
3. `data/processed/` - **ML-ready data (arrays, scalers)**

### **Your Models:**
1. `models/saved/` - **Your trained AI models**
2. `outputs/plots/` - **Graphs and visualizations**
3. `outputs/reports/` - **Performance reports**

## 🆘 **When You're Stuck**

### **"I just want to run the old workflow"**
```bash
python scripts/run_full_pipeline.py
# This does everything the old way did, but better!
```

### **"I want to see what I have"**
```bash
python scripts/project_info.py
# Shows you all your files and their status
```

### **"I want to go back to the old structure"**
Don't! The new structure is much better. But if you're really stuck:
1. All the same functionality exists
2. It's just organized better
3. Use the scripts in `scripts/` folder - they work exactly like before

### **"I want to understand what each file does"**
```bash
python scripts/collect_data.py --help
python scripts/train_model.py --help
python scripts/evaluate_model.py --help
# Each script explains what it does
```

## 🎉 **Benefits of New Structure**

1. **Easier to find things** - Everything has a logical place
2. **Easier to reuse code** - Import modules instead of copy-paste
3. **Easier to test** - Proper testing framework
4. **Easier to share** - Professional structure others can understand
5. **Easier to extend** - Add new features without breaking existing code

## 💡 **Pro Tips**

1. **Start with scripts/**: Use the ready-made scripts first
2. **Check project_info.py**: Always run this to see what you have
3. **Use the pipeline**: `run_full_pipeline.py` does everything
4. **Read the help**: Add `--help` to any script to see options
5. **Don't panic**: All your old functionality is still there, just organized better!

---

## 🤝 **Need Help?**

The refactoring made things more professional, but I understand it can feel overwhelming. Remember:

- **All the same functionality exists**
- **It's just organized better**
- **Start with the scripts/ folder**
- **Use `python scripts/project_info.py` to see what you have**

You've got this! The new structure will make your life easier once you get used to it. 🚀