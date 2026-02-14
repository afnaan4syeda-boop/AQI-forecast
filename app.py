from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and feature list
model = joblib.load('model.joblib')
features = joblib.load('features.joblib')

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        input_data = [float(request.form.get(f)) for f in features]
        final_features = np.array(input_data).reshape(1, -1)
        
        # Prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        # Determine AQI Category
        category = ""
        color = ""
        if output <= 50:
            category = "Good"
            color = "#00e400"
        elif output <= 100:
            category = "Moderate"
            color = "#ffff00"
        elif output <= 150:
            category = "Unhealthy for Sensitive Groups"
            color = "#ff7e00"
        elif output <= 200:
            category = "Unhealthy"
            color = "#ff0000"
        elif output <= 300:
            category = "Very Unhealthy"
            color = "#8f3f97"
        else:
            category = "Hazardous"
            color = "#7e0023"

        return render_template('index.html', 
                             prediction_text=f'Predicted AQI: {output}',
                             aqi_category=category,
                             category_color=color,
                             features=features)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             features=features)

if __name__ == "__main__":
    app.run(debug=True)
