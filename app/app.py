# app/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load models
models = {}
for cluster in range(5):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", f"model_cluster_{cluster}.pkl"))
    models[cluster] = joblib.load(model_path)

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