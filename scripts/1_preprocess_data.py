# scripts/1_preprocess_data.py
import os
import pandas as pd
from sklearn.cluster import KMeans

# Function to ensure unique column names
def make_unique_columns(columns):
    """Ensure unique column names by appending a suffix if duplicates exist."""
    seen = {}
    unique_columns = []
    
    for col in columns:
        if col in seen:
            seen[col] += 1
            unique_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_columns.append(col)
    
    return unique_columns

# Define paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
crosscheck_path = os.path.join(base_path, "crosscheck_daily_data_cleaned_w_sameday.csv")
studentlife_path = os.path.join(base_path, "studentlife_daily_data_cleaned_w_sameday_08282021.csv")

# Read datasets
crosscheck_data = pd.read_csv(crosscheck_path)
studentlife_data = pd.read_csv(studentlife_path)

# Reset index to avoid index conflicts
crosscheck_data = crosscheck_data.reset_index(drop=True)
studentlife_data = studentlife_data.reset_index(drop=True)

# Rename columns for consistency
studentlife_data.rename(columns={
    'day': 'date',
    'ema_Mood_happy': 'ema_HOPEFUL',
    'ema_Mood_sad': 'ema_DEPRESSED',
    'ema_Stress_level': 'ema_STRESSED',
    'ema_Sleep_hour': 'sleep_duration',
    'ema_Sleep_rate': 'quality_activity',
    'ema_Behavior_calm': 'ema_CALM',
    'ema_Behavior_anxious': 'ema_THINK',
}, inplace=True)

# Ensure unique column names
crosscheck_data.columns = [f"crosscheck_{col}" if col in studentlife_data.columns else col for col in crosscheck_data.columns]
studentlife_data.columns = make_unique_columns(studentlife_data.columns)  # Apply function to StudentLife dataset

# Combine datasets
combined_data = pd.concat([crosscheck_data, studentlife_data], axis=0, ignore_index=True)

# Handle missing values
combined_data.fillna(0, inplace=True)

# Feature engineering
combined_data['total_activity_duration'] = combined_data.filter(like='act_').sum(axis=1)
combined_data['total_convo_duration'] = combined_data.filter(like='audio_convo_duration_').sum(axis=1)

# Clustering for personalization
features_for_clustering = combined_data.filter(like='ema_').columns.tolist() + ['total_activity_duration', 'total_convo_duration']
kmeans = KMeans(n_clusters=5, random_state=42)
combined_data['cluster'] = kmeans.fit_predict(combined_data[features_for_clustering])

# Save preprocessed data
output_path = os.path.join(base_path, "combined_data.csv")
combined_data.to_csv(output_path, index=False)
print(f"Data preprocessing complete. Combined data saved to '{output_path}'.")