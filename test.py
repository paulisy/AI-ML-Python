# test.py
import pandas as pd
import requests
# from dotenv import load_dotenv
from decouple import config
import os

# load_dotenv()
api_key = config('VISUAL_CROSSING_API_KEY')
print(f"API Key loaded: {api_key[:10]}...")  # Print first 10 chars
print("✅ Setup complete!")