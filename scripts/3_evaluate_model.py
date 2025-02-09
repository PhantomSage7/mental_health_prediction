# scripts/3_evaluate_model.py
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os

def evaluate_cross_dataset():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = pd.read_csv(os.path.join(base_path, "data", "combined_data.csv"))
    
    # Cross-dataset evaluation
    for cluster in data['cluster'].unique():
        try:
            model = joblib.load(os.path.join(base_path, "models", f"model_cluster_{cluster}.pkl"))
        except FileNotFoundError:
            continue
            
        # StudentLife -> CrossCheck
        test_data = data[(data['cluster'] == cluster) & (data['dataset_source'] == 'crosscheck')]
        if test_data.empty:
            continue
            
        X_test = test_data.select_dtypes(include=np.number)
        y_test = test_data['target']
        
        y_pred = model.predict(X_test)
        
        print(f"\nCluster {cluster} Cross-Dataset Performance:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'High'], zero_division=0))
        print(f"F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")

if __name__ == "__main__":
    evaluate_cross_dataset()