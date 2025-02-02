# scripts/1_preprocess_data.py
import pandas as pd
import os

# Load datasets
crosscheck_data = pd.read_csv('../data/crosscheck_daily_data_cleaned_w_sameday.csv')
studentlife_data = pd.read_csv('../data/studentlife_daily_data_cleaned_w_sameday_08282021.csv')

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

# Combine datasets
combined_data = pd.concat([crosscheck_data, studentlife_data], ignore_index=True)

# Handle missing values
combined_data.fillna(0, inplace=True)

# Feature engineering
combined_data['total_activity_duration'] = combined_data.filter(like='act_').sum(axis=1)
combined_data['total_convo_duration'] = combined_data.filter(like='audio_convo_duration_').sum(axis=1)

# Save preprocessed data
combined_data.to_csv('../data/combined_data.csv', index=False)
print("Data preprocessing complete. Combined data saved to 'data/combined_data.csv'.")