# scripts/3_evaluate_model.py
'''import pandas as pd
import numpy as np
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
            print(f"No test data found for cluster {cluster}.")
            continue
    
        print(f"Test data shape for cluster {cluster}: {test_data.shape}")
        print(test_data.columns)  # Check the columns in test_data
    
        if 'target' not in test_data.columns:
            print(f"'target' column not found in cluster {cluster} for dataset 'crosscheck'.")
            continue
    
        X_test = test_data.select_dtypes(include=np.number)
        y_test = test_data['target']
    
        y_pred = model.predict(X_test)
    
        print(f"\nCluster {cluster} Cross-Dataset Performance:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'High'], zero_division=0))
        print(f"F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")

if __name__ == "__main__":
    evaluate_cross_dataset()'''

'''import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os

def evaluate_cross_dataset():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = pd.read_csv(os.path.join(base_path, "data", "combined_data.csv"))

    # Check if 'ema_neg_score' is present and create 'target' if necessary
    if 'ema_neg_score' in data.columns:
        data['target'] = pd.cut(
            data['ema_neg_score'],
            bins=[-np.inf, data['ema_neg_score'].median(), np.inf],
            labels=[0, 1]
        )
    else:
        print("Warning: 'ema_neg_score' not found. Cannot create 'target' column.")

    # Cross-dataset evaluation
    for cluster in data['cluster'].unique():
        try:
            model = joblib.load(os.path.join(base_path, "models", f"model_cluster_{cluster}.pkl"))
        except FileNotFoundError:
            continue
        
        # StudentLife -> CrossCheck
        test_data = data[(data['cluster'] == cluster) & (data['dataset_source'] == 'crosscheck')]
    
        if test_data.empty:
            print(f"No test data found for cluster {cluster}.")
            continue
    
        print(f"Test data shape for cluster {cluster}: {test_data.shape}")
        print(test_data.columns)  # Check the columns in test_data
    
        if 'target' not in test_data.columns:
            print(f"'target' column not found in cluster {cluster} for dataset 'crosscheck'.")
            continue
    
        X_test = test_data.select_dtypes(include=np.number)
        y_test = test_data['target']
    
        y_pred = model.predict(X_test)
    
        print(f"\nCluster {cluster} Cross-Dataset Performance:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'High'], zero_division=0))
        print(f"F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")

if __name__ == "__main__":
    evaluate_cross_dataset()'''

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score,
    precision_score,
    recall_score
)
import joblib
import os

def evaluate_cross_dataset():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = pd.read_csv(os.path.join(base_path, "data", "combined_data.csv"))
    
    # Recreate target if missing
    if 'target' not in data.columns:
        data['target'] = pd.cut(
            data['ema_neg_score'],
            bins=[-np.inf, data['ema_neg_score'].median(), np.inf],
            labels=[0, 1]
        )

    for cluster in data['cluster'].unique():
        model_path = os.path.join(base_path, "models", f"model_cluster_{cluster}.pkl")
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        test_data = data[(data['cluster'] == cluster) & (data['dataset_source'] == 'crosscheck')]
        
        if test_data.empty:
            print(f"No test data for cluster {cluster}")
            continue
            
        # Prepare features
        X_test = test_data.drop(columns=['target', 'dataset_source', 'cluster'])
        X_test = X_test.select_dtypes(include=np.number)
        y_test = test_data['target']
        
        # Predict and evaluate
        try:
            y_pred = model.predict(X_test)
        except ValueError as e:
            print(f"Feature mismatch in cluster {cluster}: {str(e)}")
            continue
            
        print(f"\n{'='*40}\nCluster {cluster} Evaluation\n{'='*40}")
        print(classification_report(y_test, y_pred, target_names=['Low', 'High'], zero_division=0))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}\n")

if __name__ == "__main__":
    evaluate_cross_dataset()