# scripts/4_deploy_model.py
import joblib
import pandas as pd
import os

# Load models
models = {}
for cluster in range(5):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", f"model_cluster_{cluster}.pkl"))
    models[cluster] = joblib.load(model_path)

def predict(cluster, features):
    """
    Predict using the model for the specified cluster.
    """
    model = models[cluster]
    prediction = model.predict([features])
    return prediction[0]

if __name__ == '__main__':
    # Example usage
    cluster = int(input("Enter cluster (0-4): "))
    features = list(map(float, input("Enter features (comma-separated): ").split(',')))
    
    prediction = predict(cluster, features)
    print(f"Prediction: {prediction}")