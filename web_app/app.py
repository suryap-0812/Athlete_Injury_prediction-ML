from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# Add the parent directory to Python path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Load the trained model, scaler, and feature metadata
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_athlete_injury_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scaler.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_names.json')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Load feature metadata
    with open(FEATURES_PATH, 'r') as f:
        feature_metadata = json.load(f)
    
    print(" Model, scaler, and feature metadata loaded successfully!")
except Exception as e:
    print(f" Error loading model/metadata: {e}")
    model = None
    scaler = None
    feature_metadata = None

# Sport types available in the dataset (from feature metadata)
SPORT_TYPES = ['basketball', 'football', 'running'] if feature_metadata is None else list(feature_metadata['categorical_mappings']['sport_type'].keys())

def predict_injury_risk(input_data):
    """
    Predict injury probability for a new athlete using feature metadata
    """
    if model is None or scaler is None or feature_metadata is None:
        return None, None, "Model or metadata not loaded properly"
    
    try:
        # Create DataFrame with the same structure as training data
        input_df = pd.DataFrame([input_data])
        
        # Apply categorical mappings using metadata
        if 'gender' in input_df.columns and 'gender' in feature_metadata['categorical_mappings']:
            gender_value = input_df['gender'].iloc[0]
            gender_mapping = feature_metadata['categorical_mappings']['gender'].get(gender_value, {})
            for feature, value in gender_mapping.items():
                input_df[feature] = value
            input_df = input_df.drop(columns=['gender'])
        
        if 'sport_type' in input_df.columns and 'sport_type' in feature_metadata['categorical_mappings']:
            sport_value = input_df['sport_type'].iloc[0]
            sport_mapping = feature_metadata['categorical_mappings']['sport_type'].get(sport_value, {})
            for feature, value in sport_mapping.items():
                input_df[feature] = value
            input_df = input_df.drop(columns=['sport_type'])
        
        # Get expected features from metadata
        expected_features = feature_metadata['feature_names']
        
        # Ensure all expected columns are present
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[expected_features]
        
        # Scale features
        features_scaled = scaler.transform(input_df)
        
        # Make prediction
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = model.predict(features_scaled)[0]
        
        return prediction, probability, None
    
    except Exception as e:
        return None, None, str(e)

@app.route('/')
def index():
    return render_template('index.html', sport_types=SPORT_TYPES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Convert string values to appropriate types
        input_data = {
            'age': int(data['age']),
            'height_cm': float(data['height_cm']),
            'weight_kg': float(data['weight_kg']),
            'training_load': float(data['training_load']),
            'training_intensity': float(data['training_intensity']),
            'recovery_time_hrs': float(data['recovery_time_hrs']),
            'prior_injury_count': int(data['prior_injury_count']),
            'fatigue_level': float(data['fatigue_level']),
            'wellness_score': float(data['wellness_score']),
            'external_load': float(data['external_load']),
            'gender': data['gender'],
            'sport_type': data['sport_type']
        }
        
        # Make prediction
        prediction, probability, error = predict_injury_risk(input_data)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Prepare response
        risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
        risk_color = "danger" if prediction == 1 else "success"
        
        # Generate recommendations based on risk factors
        recommendations = generate_recommendations(input_data, probability)
        
        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability * 100, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_recommendations(data, probability):
    """Generate personalized recommendations based on input data"""
    recommendations = []
    
    # Training load recommendations
    if data['training_load'] > 8:
        recommendations.append("⚠️ Consider reducing training load to prevent overtraining")
    
    # Recovery recommendations
    if data['recovery_time_hrs'] < 6:
        recommendations.append("😴 Increase recovery time - aim for at least 7-8 hours")
    
    # Fatigue recommendations
    if data['fatigue_level'] > 7:
        recommendations.append("💪 High fatigue detected - consider rest day or light training")
    
    # Wellness recommendations
    if data['wellness_score'] < 6:
        recommendations.append("🏥 Focus on improving overall wellness and nutrition")
    
    # Prior injury recommendations
    if data['prior_injury_count'] > 2:
        recommendations.append("🩹 History of injuries - consider preventive physiotherapy")
    
    # Age-specific recommendations
    if data['age'] > 30:
        recommendations.append("👨‍⚕️ Consider age-appropriate training modifications and recovery protocols")
    
    # General recommendations based on probability
    if probability > 0.7:
        recommendations.append("🚨 URGENT: Consult with medical staff before next training session")
    elif probability > 0.5:
        recommendations.append("⚡ Monitor closely and consider modified training intensity")
    else:
        recommendations.append(" Continue current training regimen with regular monitoring")
    
    return recommendations

@app.route('/features')
def get_features():
    """Return available features and their metadata"""
    if feature_metadata is None:
        return jsonify({'error': 'Feature metadata not loaded'}), 500
    
    return jsonify({
        'feature_names': feature_metadata['feature_names'],
        'categorical_mappings': feature_metadata['categorical_mappings'],
        'sport_types': list(feature_metadata['categorical_mappings']['sport_type'].keys()),
        'gender_options': list(feature_metadata['categorical_mappings']['gender'].keys())
    })

@app.route('/health')
def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    metadata_status = "loaded" if feature_metadata is not None else "not loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'scaler_status': scaler_status,
        'metadata_status': metadata_status
    })

if __name__ == '__main__':
    print(" Starting Athlete Injury Prediction Web App...")
    print(f" Model path: {MODEL_PATH}")
    print(f" Scaler path: {SCALER_PATH}")
    print(f" Features path: {FEATURES_PATH}")
    print(" Access the app at: http://localhost:5000")
    print(" API endpoints:")
    print("   - GET  /health    - Health check")
    print("   - GET  /features  - Feature metadata")
    print("   - POST /predict   - Injury prediction")
    app.run(debug=True, host='0.0.0.0', port=5000)