#Introduction: 
#This code defines a Flask web application that provides an endpoint to make predictions based on input data. 
#It includes functions for preprocessing input data, such as encoding categorical variables and scaling numerical features.
#Then uses a pre-trained model to make predictions.

#Explanation:
#In this code, the following steps are performed:

#Preprocessing:
#The age column is processed to group age ranges.
#Binary columns (e.g., glucose_test, A1Ctest, etc.) are encoded to 1/0 values using encode_binary_column.
#Categorical columns (medical_specialty, diag_1, diag_2, diag_3) are handled via One-Hot Encoding in handle_categorical_columns.

#Prediction Route:
#The /predict/ route processes incoming data, applies necessary transformations (such as encoding and scaling). 
#Then uses the pre-trained model (best_model) to generate predictions and probabilities. 
#It returns these as a JSON response.

#Testing Route:
#A simple GET route is defined at the root (/) to test if the Flask app is running.

#Running the App:
#The app is run on port 8080 with debug mode enabled.



from flask import Flask, jsonify, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pipeline objects (model, scaler, encoder) and trained columns
try:
    pipeline_objects = joblib.load('D:/Python/patient_readmission_pipeline.pkl')
    best_model = pipeline_objects['model']
    column_transformer = pipeline_objects['scaler']
    label_encoder = pipeline_objects['encoder']
    trained_columns = joblib.load('D:/Python/trained_columns.pkl')
except Exception as e:
    print(f"Error loading model or columns: {str(e)}")
    exit(1)

# Function to preprocess 'age' categories into numeric values
def preprocess_age(age):
    age_mapping = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95
    }
    return age_mapping.get(age, age)  # Default to the original value if not found

# Function to encode binary 'Yes'/'No' columns to 1/0
def encode_binary_column(df, column):
    if column in df.columns:
        df[column] = df[column].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
    return df

# Function to handle categorical columns via One-Hot Encoding
def handle_categorical_columns(df):
    categorical_columns = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']
    for col in categorical_columns:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    return df

# Define the prediction route
@app.route('/predict/', methods=['POST'])
def predict():
    try:
        # Parse input JSON data into a DataFrame
        input_data = request.get_json(force=True)
        input_df = pd.DataFrame([input_data])

        # Preprocess the 'age' column
        if 'age' in input_df.columns:
            input_df['age'] = input_df['age'].apply(preprocess_age)

        # Encode binary columns like 'glucose_test', 'A1Ctest', 'change', etc.
        binary_columns = ['glucose_test', 'A1Ctest', 'change', 'diabetes_med', 'gender', 'follow_up_within_30days']
        for col in binary_columns:
            input_df = encode_binary_column(input_df, col)

        # Handle categorical columns by applying One-Hot Encoding
        input_df = handle_categorical_columns(input_df)

        # Reindex the input DataFrame to match the trained columns
        input_df = input_df.reindex(columns=trained_columns, fill_value=0)

        # Apply the same transformations that were used for model training (e.g., scaling)
        input_scaled = column_transformer.transform(input_df)

        # Make a prediction using the pre-trained model
        prediction = best_model.predict(input_scaled)
        prediction_proba = best_model.predict_proba(input_scaled)[:, 1]

        # Return prediction and probability
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Define a simple GET route for testing
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Hello, This is my REST API for the project Patients Readmission Prediction!"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8080)
