import pandas as pd
import numpy as np


def clean_weather_data(input_file, output_file):
    """
    Clean and validate weather data.
    """

    print("🧹 Cleaning weather data...")


    # Read the CSV file
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} records")

    

