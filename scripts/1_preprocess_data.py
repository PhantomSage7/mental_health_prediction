# scripts/1_preprocess_data.py
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import random

# Set global seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def make_unique_columns(columns):
    seen = {}
    return [f"{col}_{seen[col]}" if (seen.update({col: seen.get(col, 0)+1}) or True) else col for col in columns]

# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
crosscheck_path = os.path.join(base_path, "crosscheck_daily_data_cleaned_w_sameday.csv")
studentlife_path = os.path.join(base_path, "studentlife_daily_data_cleaned_w_sameday_08282021.csv")

# Load data with source tracking
crosscheck_data = pd.read_csv(crosscheck_path).assign(dataset_source='crosscheck')
studentlife_data = pd.read_csv(studentlife_path).assign(dataset_source='studentlife')

# Validate feature alignment
stress_corr = pd.concat([
    crosscheck_data['ema_STRESSED'].rename('crosscheck'), 
    studentlife_data['ema_Stress_level'].rename('studentlife')
], axis=1).corr().iloc[0,1]
print(f"Stress feature cross-dataset correlation: {stress_corr:.2f}")

# Rename columns
studentlife_data.rename(columns={
    'day': 'date',
    'ema_Stress_level': 'ema_STRESSED',
    # ... other mappings ...
}, inplace=True)

# Unique columns
crosscheck_data.columns = [f"crosscheck_{col}" if col in studentlife_data.columns else col for col in crosscheck_data.columns]
studentlife_data.columns = make_unique_columns(studentlife_data.columns)

# Combine datasets
combined_data = pd.concat([crosscheck_data, studentlife_data], axis=0, ignore_index=True)
combined_data.fillna(0, inplace=True)

# Feature engineering
combined_data['total_activity_duration'] = combined_data.filter(like='act_').sum(axis=1)
combined_data['total_convo_duration'] = combined_data.filter(like='audio_convo_duration_').sum(axis=1)

# Clustering with cross-dataset analysis
features_for_clustering = combined_data.filter(regex='^(ema_|total_)').columns.tolist()
kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
combined_data['cluster'] = kmeans.fit_predict(combined_data[features_for_clustering])

# Analyze cluster distribution
cluster_dist = combined_data.groupby(['dataset_source', 'cluster']).size().unstack()
print("Cluster distribution:\n", cluster_dist)

# Save
output_path = os.path.join(base_path, "combined_data.csv")
combined_data.to_csv(output_path, index=False)