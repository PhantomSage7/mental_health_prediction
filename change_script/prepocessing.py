#preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def make_unique_columns(columns):
    seen = {}
    unique_cols = []
    for col in columns:
        count = seen.get(col, 0) + 1
        seen[col] = count
        unique_cols.append(f"{col}_{count}" if count > 1 else col)
    return unique_cols

# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
crosscheck_path = os.path.join(base_path, "crosscheck_daily_data_cleaned_w_sameday.csv")
studentlife_path = os.path.join(base_path, "studentlife_daily_data_cleaned_w_sameday_08282021.csv")

# Load data
crosscheck = pd.read_csv(crosscheck_path)
studentlife = pd.read_csv(studentlife_path)

# Align features across datasets
studentlife = studentlife.rename(columns={
    'day': 'date',
    'ema_Stress_level': 'ema_STRESSED',
    'ema_Sleep_hour': 'ema_SLEEP_HOURS',
    'ema_Sleep_rate': 'ema_SLEEP_QUALITY'
})

# Select important common features using Decision Tree
shared_columns = list(set(crosscheck.columns) & set(studentlife.columns))
common_features = set()

if 'label' in crosscheck.columns and len(shared_columns) > 0:  # Ensure we have features to train on
    dt = DecisionTreeClassifier(random_state=SEED)
    dt.fit(crosscheck[shared_columns], crosscheck['label'])
    feature_importances = pd.Series(dt.feature_importances_, index=shared_columns)
    common_features = set(feature_importances.nlargest(5).index)  # Selecting top 5 important features

# Handle column conflicts
crosscheck.columns = [
    f"crosscheck_{col}" if (col in studentlife.columns and (not common_features or col not in common_features))
    else col 
    for col in crosscheck.columns
]

# Make StudentLife columns unique
studentlife.columns = make_unique_columns(studentlife.columns)

# Add source identifiers
crosscheck = crosscheck.assign(dataset_source='crosscheck')
studentlife = studentlife.assign(dataset_source='studentlife')

# Combine data
combined = pd.concat([crosscheck, studentlife], ignore_index=True)
combined.fillna(0, inplace=True)

# Feature engineering
combined['total_activity_duration'] = combined.filter(like='act_').sum(axis=1)
combined['total_convo_duration'] = combined.filter(like='audio_convo_duration_').sum(axis=1)

# Clustering
cluster_features = combined.filter(regex='^(ema_|total_)').columns.tolist()
kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
combined['cluster'] = kmeans.fit_predict(combined[cluster_features])

# Save
output_path = os.path.join(base_path, "combined_data.csv")
combined.to_csv(output_path, index=False)





