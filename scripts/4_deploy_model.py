# scripts/4_deploy_model.py
import joblib
import numpy as np
import os

SEED = 42
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
STRESS_THRESHOLD = 4.0  # Clinical cutoff

class MentalHealthPredictor:
    def __init__(self):
        self.models = {
            cluster: joblib.load(os.path.join(MODEL_DIR, f"model_cluster_{cluster}.pkl"))
            for cluster in range(5)
        }
        self.expected_features = 20  # Update based on your model
    
    def predict(self, cluster, features):
        if len(features) != self.expected_features:
            raise ValueError(f"Expected {self.expected_features} features, got {len(features)}")
            
        prediction = self.models[cluster].predict([features])[0]
        
        # Clinical interpretation
        if prediction > STRESS_THRESHOLD:
            return {"prediction": prediction, "alert": "High risk detected"}
        return {"prediction": prediction}

if __name__ == "__main__":
    predictor = MentalHealthPredictor()
    cluster = int(input("Enter cluster (0-4): "))
    features = list(map(float, input("Enter features (comma-separated): ").split(',')))
    
    try:
        result = predictor.predict(cluster, features)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")