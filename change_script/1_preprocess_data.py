'''# scripts/1_preprocess_data.py
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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

# Rename StudentLife columns to match CrossCheck
studentlife = studentlife.rename(columns={
    'day': 'date',
    'ema_Stress_level': 'ema_STRESSED',  # Correct column name from sample
    'ema_Sleep_hour': 'ema_SLEEP_HOURS',
    'ema_Sleep_rate': 'ema_SLEEP_QUALITY'
})

# Handle column conflicts
crosscheck.columns = [
    f"crosscheck_{col}" if col in studentlife.columns and col not in ['date', 'ema_STRESSED'] 
    else col 
    for col in crosscheck.columns
]

# Make StudentLife columns unique
studentlife.columns = make_unique_columns(studentlife.columns)

# Add source identifiers
crosscheck = crosscheck.assign(dataset_source='crosscheck')
studentlife = studentlife.assign(dataset_source='studentlife')

# Validate columns
required_columns = {'date', 'ema_STRESSED', 'dataset_source'}
for df, name in [(crosscheck, 'CrossCheck'), (studentlife, 'StudentLife')]:
    missing = required_columns - set(df.columns)
    if missing:
        print(f"Missing in {name}: {missing}")
        exit()

# Calculate stress correlation
stress_corr = pd.concat([
    crosscheck['ema_STRESSED'].rename('crosscheck'),
    studentlife['ema_STRESSED'].rename('studentlife')
], axis=1).corr().iloc[0,1]
print(f"Stress correlation: {stress_corr:.2f}")

# Combine data
combined = pd.concat([crosscheck, studentlife], ignore_index=True)
combined.fillna(0, inplace=True)

# Feature engineering
activity_cols = [c for c in combined.columns if c.startswith('act_')]
convo_cols = [c for c in combined.columns if 'audio_convo_duration' in c]

combined['total_activity_duration'] = combined[activity_cols].sum(axis=1)
combined['total_convo_duration'] = combined[convo_cols].sum(axis=1)

# Clustering
cluster_features = combined.filter(regex='^(ema_|total_)').columns.tolist()
kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
combined['cluster'] = kmeans.fit_predict(combined[cluster_features])

# Analyze clusters
cluster_dist = combined.groupby(['dataset_source', 'cluster']).size().unstack()
print("Cluster distribution:\n", cluster_dist)

# Save
output_path = os.path.join(base_path, "combined_data.csv")
combined.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")'''


import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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

# Handle column conflicts
common_features = {'date', 'ema_STRESSED', 'ema_SLEEP_HOURS', 'ema_SLEEP_QUALITY'}
crosscheck.columns = [
    f"crosscheck_{col}" if (col in studentlife.columns and col not in common_features)
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