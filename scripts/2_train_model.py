# scripts/2_train_model.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load preprocessed data
data = pd.read_csv('../data/combined_data.csv')

# Clustering for personalization
features_for_clustering = data.filter(like='ema_').columns.tolist() + ['total_activity_duration', 'total_convo_duration']
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(data[features_for_clustering])

# Train a model for each cluster
models = {}
for cluster in data['cluster'].unique():
    cluster_data = data[data['cluster'] == cluster]
    
    # Features and target
    X = cluster_data.drop(columns=['study_id', 'eureka_id', 'date', 'cluster', 'ema_neg_score', 'ema_pos_score', 'ema_score'])
    y = cluster_data['ema_neg_score']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Oversampling with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    
    # Save the model
    joblib.dump(model, f'../models/model_cluster_{cluster}.pkl')
    print(f"Model for cluster {cluster} trained and saved.")

print("All models trained and saved to 'models/' directory.")