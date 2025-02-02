import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
from collections import Counter

# Load preprocessed data
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
combined_data = pd.read_csv(os.path.join(base_path, "combined_data.csv"), low_memory=False)

# Drop non-numeric columns
non_numeric_columns = ['study_id', 'eureka_id', 'date']
combined_data = combined_data.drop(columns=non_numeric_columns, errors='ignore')

# Convert all remaining non-numeric columns to numeric
combined_data = combined_data.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with 0
combined_data = combined_data.fillna(0)

# Clustering for personalization
features_for_clustering = combined_data.filter(like='ema_').columns.tolist() + ['total_activity_duration', 'total_convo_duration']
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
combined_data['cluster'] = kmeans.fit_predict(combined_data[features_for_clustering])

# Train a model for each cluster
models = {}
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Ensure the models directory exists
os.makedirs(models_dir, exist_ok=True)

for cluster in combined_data['cluster'].unique():
    cluster_data = combined_data[combined_data['cluster'] == cluster]
    
    # Features and target
    X = cluster_data.drop(columns=['cluster', 'ema_neg_score', 'ema_pos_score', 'ema_score'], errors='ignore')
    y = cluster_data['ema_neg_score']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if SMOTE is applicable
    if len(y_train) > 2:  # SMOTE needs at least 3 samples
        # Count occurrences of each class
        counts = Counter(y_train)
        minority_count = min(counts.values())
        
        # Set k_neighbors based on the minority class count
        k_neighbors = min(5, minority_count - 1)  # Ensure k_neighbors is not more than the available samples
        
        if minority_count > 1:  # Only apply SMOTE if enough samples are available
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        else:
            print(f"Skipping SMOTE for cluster {cluster} as there are too few samples ({minority_count}) compared to neighbors ({k_neighbors}).")
            X_resampled, y_resampled = X_train, y_train  # Use original data if not enough samples for SMOTE
    else:
        print(f"Skipping SMOTE for cluster {cluster} due to insufficient samples ({len(y_train)}).")
        X_resampled, y_resampled = X_train, y_train  # Use original data if too few samples
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    
    # Save the model
    model_path = os.path.join(models_dir, f"model_cluster_{cluster}.pkl")
    joblib.dump(model, model_path)
    print(f"Model for cluster {cluster} trained and saved to '{model_path}'.")

print("All models trained and saved to 'models/' directory.")