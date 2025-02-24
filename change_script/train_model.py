#training Model


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
import joblib
import os
from collections import Counter

SEED = 42
np.random.seed(SEED)

def train_models(data):
    # Create target variable
    data['target'] = pd.cut(
        data['ema_neg_score'],
        bins=[-np.inf, data['ema_neg_score'].median(), np.inf],
        labels=[0, 1]
    )

    # Check for NaN values in the target variable
    print("Initial data size:", data.shape)
    if data['target'].isnull().any():
        print("NaN values found in target variable. Dropping NaN rows.")
        data = data.dropna(subset=['target'])

    print("Data size after dropping NaN in target:", data.shape)

    # Change the model directory to be at the project root level
    models_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Unique clusters found:", data['cluster'].unique())

    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        print(f"Cluster {cluster} - Cluster data size: {cluster_data.shape}")

        # Split datasets
        train_data = cluster_data[cluster_data['dataset_source'] == 'studentlife']
        test_data = cluster_data[cluster_data['dataset_source'] == 'crosscheck']

        print(f"Cluster {cluster} - Train data size: {train_data.shape}, Test data size: {test_data.shape}")

        if train_data.empty or test_data.empty:
            print(f"Skipping cluster {cluster} - insufficient data")
            continue

        # Prepare features
        X_train = train_data.drop(columns=['target', 'dataset_source', 'cluster'])
        X_train = X_train.select_dtypes(include=np.number)
        y_train = train_data['target']

        # Handle class imbalance using SMOTE-Tomek
        if len(Counter(y_train)) > 1:
            smote_tomek = SMOTEENN(random_state=SEED)
            X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train

        # Train and save model
        model = RandomForestClassifier(random_state=SEED)
        model.fit(X_res, y_res)
        joblib.dump(model, os.path.join(models_dir, f"model_cluster_{cluster}.pkl"))
        print(f"Model for cluster {cluster} saved.")

if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = pd.read_csv(os.path.join(base_path, "data", "combined_data.csv"))
    train_models(data)

