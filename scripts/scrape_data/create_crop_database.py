import pandas as pd

# Nigerian crop data (researched requirements)
CROPS = [
    {
        'crop_name': 'Maize',
        'min_temp': 18,
        'max_temp': 35,
        'optimal_rainfall': 600,  # mm per season
        'min_rainfall': 500,
        'max_rainfall': 800,
        'growing_days': 90,
        'planting_months': 'April,May,June,July',
        'base_temp_gdd': 10,  # For Growing Degree Days calculation
        'soil_type': 'loamy,sandy-loam',
        'description': 'Staple food crop, drought-tolerant'
    },
    {
        'crop_name': 'Rice',
        'min_temp': 20,
        'max_temp': 37,
        'optimal_rainfall': 1200,
        'min_rainfall': 1000,
        'max_rainfall': 2000,
        'growing_days': 120,
        'planting_months': 'May,June,July',
        'base_temp_gdd': 10,
        'soil_type': 'clay,loamy',
        'description': 'High water requirement, flood-tolerant'
    },
    {
        'crop_name': 'Cassava',
        'min_temp': 20,
        'max_temp': 35,
        'optimal_rainfall': 1000,
        'min_rainfall': 800,
        'max_rainfall': 1500,
        'growing_days': 300,  # 10 months
        'planting_months': 'March,April,May,June',
        'base_temp_gdd': 12,
        'soil_type': 'sandy,loamy,clay',
        'description': 'Very hardy, drought-resistant'
    },
    {
        'crop_name': 'Yam',
        'min_temp': 25,
        'max_temp': 35,
        'optimal_rainfall': 1200,
        'min_rainfall': 1000,
        'max_rainfall': 1500,
        'growing_days': 240,  # 8 months
        'planting_months': 'March,April,May',
        'base_temp_gdd': 15,
        'soil_type': 'loamy,sandy-loam',
        'description': 'Important tuber crop'
    },
    {
        'crop_name': 'Sorghum',
        'min_temp': 15,
        'max_temp': 40,
        'optimal_rainfall': 450,
        'min_rainfall': 350,
        'max_rainfall': 700,
        'growing_days': 100,
        'planting_months': 'May,June,July',
        'base_temp_gdd': 8,
        'soil_type': 'sandy,loamy',
        'description': 'Extremely drought-tolerant'
    },
    {
        'crop_name': 'Millet',
        'min_temp': 15,
        'max_temp': 40,
        'optimal_rainfall': 400,
        'min_rainfall': 300,
        'max_rainfall': 650,
        'growing_days': 90,
        'planting_months': 'May,June,July',
        'base_temp_gdd': 8,
        'soil_type': 'sandy,sandy-loam',
        'description': 'Very drought-tolerant, good for arid regions'
    },
    {
        'crop_name': 'Groundnut',
        'min_temp': 20,
        'max_temp': 35,
        'optimal_rainfall': 500,
        'min_rainfall': 400,
        'max_rainfall': 700,
        'growing_days': 120,
        'planting_months': 'April,May,June',
        'base_temp_gdd': 10,
        'soil_type': 'sandy,sandy-loam',
        'description': 'Legume, nitrogen-fixing'
    },
    {
        'crop_name': 'Cowpea',
        'min_temp': 20,
        'max_temp': 35,
        'optimal_rainfall': 500,
        'min_rainfall': 400,
        'max_rainfall': 700,
        'growing_days': 75,
        'planting_months': 'May,June,July',
        'base_temp_gdd': 10,
        'soil_type': 'sandy,loamy',
        'description': 'Fast-growing legume'
    },
    {
        'crop_name': 'Tomato',
        'min_temp': 18,
        'max_temp': 32,
        'optimal_rainfall': 600,
        'min_rainfall': 500,
        'max_rainfall': 800,
        'growing_days': 80,
        'planting_months': 'March,April,August,September',
        'base_temp_gdd': 10,
        'soil_type': 'loamy,sandy-loam',
        'description': 'High value vegetable crop'
    },
    {
        'crop_name': 'Pepper',
        'min_temp': 20,
        'max_temp': 32,
        'optimal_rainfall': 700,
        'min_rainfall': 600,
        'max_rainfall': 900,
        'growing_days': 90,
        'planting_months': 'March,April,August,September',
        'base_temp_gdd': 12,
        'soil_type': 'loamy,sandy-loam',
        'description': 'High value, continuous harvest'
    },
]

def main():
    """
    Create crop database CSV
    """
    print("🌾 Creating crop database...")
    
    # Convert to DataFrame
    df = pd.DataFrame(CROPS)
    
    # Save
    output_file = 'data/raw/nigerian_crops.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created crop database with {len(df)} crops")
    print(f"📁 Saved to: {output_file}")
    print()
    print("Crops included:")
    for crop in df['crop_name']:
        print(f"  • {crop}")
    print()
    print(df.head())

if __name__ == "__main__":
    main()