import pandas as pd
import numpy as np

def generate_aqi_data(n_samples=1000):
    np.random.seed(42)
    
    # Features: PM2.5, PM10, NO2, CO, SO2, O3
    pm25 = np.random.uniform(10, 200, n_samples)
    pm10 = pm25 * 1.5 + np.random.normal(0, 10, n_samples)
    no2 = np.random.uniform(5, 100, n_samples)
    co = np.random.uniform(0.1, 5, n_samples)
    so2 = np.random.uniform(2, 50, n_samples)
    o3 = np.random.uniform(10, 150, n_samples)
    
    # AQI calculation (simplified linear relationship for regression)
    # AQI is typically the max of individual pollutant indices, but for regression we'll use a continuous score
    aqi = (0.5 * pm25 + 0.3 * pm10 + 0.1 * no2 + 0.1 * co + 0.1 * so2 + 0.1 * o3) + np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame({
        'PM2.5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'CO': co,
        'SO2': so2,
        'O3': o3,
        'AQI': aqi
    })
    
    df.to_csv('aqi_dataset.csv', index=False)
    print("Dataset 'aqi_dataset.csv' generated successfully!")

if __name__ == "__main__":
    generate_aqi_data()
