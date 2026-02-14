import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_aqi_model():
    print("Loading dataset...")
    df = pd.read_csv('aqi_dataset.csv')
    
    X = df.drop('AQI', axis=1)
    y = df['AQI']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    print("Saving model and feature list...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(list(X.columns), 'features.joblib')
    print("Model and features saved successfully!")

if __name__ == "__main__":
    train_aqi_model()
