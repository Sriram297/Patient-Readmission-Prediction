# Patients Readmission Prediction

## Overview
This repository contains a Jupyter Notebook for predicting patient readmissions using machine learning techniques. The goal is to analyze patient data and build a predictive model to determine the likelihood of readmission within a given timeframe.

## Dataset
The dataset used for this project contains patient records, including features such as:
- Patient demographics
- Medical history
- Hospital visit details
- Previous readmission status

## Methodology

### Data Preprocessing:
- Handling missing values
- Feature engineering
- Encoding categorical variables

### Exploratory Data Analysis (EDA):
- Visualizing key trends
- Understanding correlations

### Model Training & Evaluation:
- Applied multiple machine learning models (e.g., Logistic Regression, Random Forest, XGBoost)
- Evaluated using accuracy, precision, recall, and F1-score

## Flask API Implementation
This project includes a **Flask API** for making predictions using the trained model. The API supports:
- **POST requests** to send patient data and receive predictions
- **GET requests** to check API status

### Running the Flask API
1. Install required dependencies:
   ```bash
   pip install flask pandas numpy scikit-learn xgboost
   ```
2. Run the Flask API script:
   ```bash
   python Flask API.py
   ```
3. Test API endpoints using **Postman**.

## Dependencies
To run the notebook, install the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost flask
```

## Usage
Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
```
Navigate to the directory:
```bash
cd your-repo
```
Launch Jupyter Notebook:
```bash
jupyter notebook "Patients Readmission Prediction.ipynb"
```

## Results
The best-performing model achieved a significant improvement in readmission prediction, helping healthcare providers optimize patient care and resource allocation.

## Contributing
Feel free to fork the repository and submit pull requests for improvements.
