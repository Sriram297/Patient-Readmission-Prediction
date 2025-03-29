from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and preprocessing pipeline
try:
    pipeline_objects = joblib.load('D:/Python/Patient_Readmission_Prediction ML Project/patient_readmission_pipeline.pkl')
    best_model = pipeline_objects['model']
    column_transformer = pipeline_objects['scaler']
    label_encoder = pipeline_objects['encoder']
    trained_columns = joblib.load('D:/Python/Patient_Readmission_Prediction ML Project/trained_columns.pkl')
except Exception as e:
    print(f"Error loading model or columns: {str(e)}")
    exit(1)

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is inside the templates folder

# Function to preprocess input data
def preprocess_input(data):
    # Map age categories to numerical values
    age_mapping = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    if 'age' in data:
        data['age'] = age_mapping.get(data['age'], data['age'])

    # Encode binary columns (Yes/No values → 1/0)
    binary_columns = ['glucose_test', 'A1Ctest', 'change', 'diabetes_med', 'gender', 'follow_up_within_30days']
    for col in binary_columns:
        if col in data:
            data[col] = 1 if data[col].lower() == 'yes' else 0

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Ensure all trained columns exist in the input
    df = df.reindex(columns=trained_columns, fill_value=0)

    # Apply transformations (scaling, encoding, etc.)
    transformed_input = column_transformer.transform(df)

    return transformed_input

# API route for prediction
@app.route('/predict/', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Preprocess the input
        input_features = preprocess_input(data)

        # Make prediction
        prediction = best_model.predict(input_features)
        probability = best_model.predict_proba(input_features)[:, 1]

        return jsonify({"prediction": int(prediction[0]), "probability": float(probability[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
