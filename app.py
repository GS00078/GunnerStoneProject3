import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)

CLF_model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@flask_app.route("/")
def index():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():

    float_features = [float(x) for x in request.form.values()]
    
    feature_names = [
        'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
        'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio',
        'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
        'thinness  1-19 years', 'thinness 5-9 years',
        'Income composition of resources', 'Schooling'
    ]
    
    input_data = pd.DataFrame([float_features], columns=feature_names)
    
    # Apply scaling to the input features
    scaled_features = scaler.transform(input_data)
    
    result = CLF_model.predict(scaled_features)
    
    predicted_text = f"{result[0]:.2f} years"
    return render_template("index.html", predicted_text=predicted_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
