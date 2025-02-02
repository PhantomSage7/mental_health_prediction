# scripts/3_evaluate_model.py
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load preprocessed data
data = pd.read_csv('../data/combined_data.csv')

# Initialize LOSO-CV
logo = LeaveOneGroupOut()
groups = data['study_id']

# Evaluate models
for cluster in data['cluster'].unique():
    cluster_data = data[data['cluster'] == cluster]
    model = joblib.load(f'../models/model_cluster_{cluster}.pkl')
    
    for train_index, test_index in logo.split(cluster_data, groups=cluster_data['study_id']):
        X_train, X_test = cluster_data.iloc[train_index], cluster_data.iloc[test_index]
        y_train, y_test = X_train['ema_neg_score'], X_test['ema_neg_score']
        
        # Train the model (optional, if not already trained)
        model.fit(X_train.drop(columns=['study_id', 'eureka_id', 'date', 'cluster', 'ema_neg_score', 'ema_pos_score', 'ema_score']), y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test.drop(columns=['study_id', 'eureka_id', 'date', 'cluster', 'ema_neg_score', 'ema_pos_score', 'ema_score']))
        
        print(f"Cluster {cluster} Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
        print("-----------------------------")