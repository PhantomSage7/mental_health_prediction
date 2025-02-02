# app/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
models = {cluster: joblib.load(f'../models/model_cluster_{cluster}.pkl') for cluster in range(5)}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cluster = data['cluster']
    features = pd.DataFrame([data['features']])
    
    model = models[cluster]
    prediction = model.predict(features)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)