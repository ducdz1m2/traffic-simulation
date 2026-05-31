"""Preprocess Traffic Collisions dataset for binary injury vs non-injury classification."""

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
import os

RAW = 'data/toronto-dataset/Traffic_Collisions_Toronto_data_converted.csv'
OUT = 'data/toronto-dataset/Traffic_processed.csv'

df = pd.read_csv(RAW, low_memory=False)
print(f'Raw: {df.shape}')

# Drop rows with null target
df = df[df['Injury_Collisions'] != '<Null>'].copy()
print(f'After null target drop: {df.shape}')

# Binary target
df['target_bin'] = (df['Injury_Collisions'] == 'YES').astype(int)
print(f'Injury rate: {df.target_bin.mean()*100:.2f}%')

# Parse datetime
df['dp'] = pd.to_datetime(df['OccurrenceDate'], utc=True)
df['Year'] = df['dp'].dt.year.fillna(0).astype(int)
df['Month_num'] = df['dp'].dt.month.fillna(1).astype(int)
df['DayOfWeek_num'] = df['dp'].dt.dayofweek.fillna(0).astype(int)
df['is_weekend'] = (df['DayOfWeek_num'] >= 5).astype(int)
df['Season'] = df['Month_num'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}).fillna(0).astype(int)

# Hour is actually DST indicator (4=EDT, 5=EST)
df['is_dst'] = (df['Hour'] == 4).astype(int)

# Spatial grid 250m
lat_c, lon_c = 0.00225, 0.003125
lat0, lon0 = df['Latitude'].min(), df['Longitude'].min()
df['g_i'] = ((df['Latitude'] - lat0) / lat_c).astype(int)
df['g_j'] = ((df['Longitude'] - lon0) / lon_c).astype(int)

# Encode categoricals
for col in ['Division', 'Month', 'Day_of_Week']:
    if col in df.columns:
        df[f'{col}_e'] = LabelEncoder().fit_transform(df[col].astype(str))

# Frequency encode Neighbourhood (high cardinality)
hood_counts = df['Neighbourhood'].value_counts()
df['Neighbourhood_freq'] = df['Neighbourhood'].map(hood_counts) / len(df)

# FTR and PD as binary
df['FTR_b'] = (df['FTR_Collisions'] == 'YES').astype(int)
df['PD_b'] = (df['PD_Collisions'] == 'YES').astype(int)

# Feature columns
feat_cols = [
    'Year', 'Month_num', 'DayOfWeek_num', 'is_weekend', 'Season', 'is_dst',
    'g_i', 'g_j', 'Latitude', 'Longitude',
    'Fatalities', 'FTR_b', 'PD_b', 'Neighbourhood_freq',
    'Division_e', 'Month_e', 'Day_of_Week_e',
]
feat_cols = [c for c in feat_cols if c in df.columns]

df_out = df[feat_cols + ['target_bin', 'Year']].copy()

# Impute missing
for c in df_out.select_dtypes(include=['int64','float64']).columns:
    df_out[c] = df_out[c].fillna(df_out[c].median())
for c in df_out.select_dtypes(include=['object']).columns:
    df_out[c] = df_out[c].fillna(df_out[c].mode()[0] if len(df_out[c].mode()) > 0 else 'Unknown')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_out.to_csv(OUT, index=False)
print(f'Saved: {OUT}')
print(f'Shape: {df_out.shape}')
print(f'Features: {feat_cols}')
print(f'Target distribution:\n{df_out.target_bin.value_counts().to_dict()}')
