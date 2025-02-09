# scripts/2_train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
from collections import Counter

SEED = 42
np.random.seed(SEED)

# Load data
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
combined_data = pd.read_csv(os.path.join(base_path, "combined_data.csv"))

# Convert target to binary classes
combined_data['target'] = pd.cut(
    combined_data['ema_neg_score'],
    bins=[-np.inf, combined_data['ema_neg_score'].median(), np.inf],
    labels=[0, 1]
)

# Train per-cluster
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(models_dir, exist_ok=True)

for cluster in combined_data['cluster'].unique():
    cluster_data = combined_data[combined_data['cluster'] == cluster]
    
    # Cross-dataset split
    X_train = cluster_data[cluster_data['dataset_source'] == 'studentlife']
    X_test = cluster_data[cluster_data['dataset_source'] == 'crosscheck']
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Skipping cluster {cluster} due to missing cross-dataset samples")
        continue
        
    y_train = X_train['target']
    y_test = X_test['target']
    
    # Log class distribution
    print(f"\nCluster {cluster} Class Distribution:")
    print(f"Train: {Counter(y_train)}")
    print(f"Test: {Counter(y_test)}")

    # SMOTE with safety checks
    if len(Counter(y_train)) > 1 and min(Counter(y_train).values()) > 1:
        smote = SMOTE(k_neighbors=min(5, min(Counter(y_train).values()) - 1), random_state=SEED)
        X_res, y_res = smote.fit_resample(X_train.drop(columns=['target', 'dataset_source']), y_train)
    else:
        X_res, y_res = X_train, y_train

    # Train model
    model = RandomForestClassifier(random_state=SEED)
    model.fit(X_res.select_dtypes(include=np.number), y_res)
    
    # Save
    joblib.dump(model, os.path.join(models_dir, f"model_cluster_{cluster}.pkl"))