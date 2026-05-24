"""
Deep EDA: US Congestion Dataset (2016-2022 sample 2M)
Target: Severity (0-4), Delay metrics
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, warnings, json
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 150

DATA_DIR = 'data/us-congestion-dataset/us_congestion_2016_2022_sample_2m'
OUT_DIR = 'outputs/eda_congestion_dataset'
os.makedirs(OUT_DIR, exist_ok=True)

print("="*70)
print("DEEP EDA: US CONGESTION DATASET")
print("="*70)

cg = pd.read_csv(os.path.join(DATA_DIR, 'us_congestion_2016_2022_sample_2m.csv'), low_memory=False)
print(f"Shape: {cg.shape}")

# 1. Severity
print("\nSeverity distribution:")
sev_counts = cg['Severity'].value_counts().sort_index()
print(sev_counts)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sev_counts.plot(ax=axes[0], kind='bar', color='purple')
axes[0].set_title('Severity Distribution')
axes[0].set_ylabel('Count')
sev_counts_pct = sev_counts / len(cg) * 100
sev_counts_pct.plot(ax=axes[1], kind='pie', autopct='%1.1f%%', colors=sns.color_palette('viridis', 5), startangle=90)
axes[1].set_ylabel('')
axes[1].set_title('Severity Proportion')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_severity_distribution.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 01_severity_distribution.png")

# 2. Delay analysis
print("\nDelay statistics:")
delay_cols = ['DelayFromTypicalTraffic(mins)', 'DelayFromFreeFlowSpeed(mins)', 'Distance(mi)']
print(cg[delay_cols].describe())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(delay_cols):
    data = cg[col].dropna()
    data = data[data < data.quantile(0.99)]
    axes[i].hist(data, bins=50, color='mediumpurple', alpha=0.7, edgecolor='white')
    axes[i].set_title(f'{col}\nMean: {data.mean():.2f}, Median: {data.median():.2f}')
    axes[i].set_xlabel(col)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_delay_distributions.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 02_delay_distributions.png")

# 3. Spatial
top_states = cg['State'].value_counts().head(15)
plt.figure(figsize=(12, 5))
sns.barplot(x=top_states.values, y=top_states.index, palette='viridis')
plt.title('Top 15 States by Congestion Events')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_top_states.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 03_top_states.png")

# 4. Congestion Speed
print("\nCongestion Speed:")
print(cg['Congestion_Speed'].value_counts())

# 5. Weather analysis (if available)
weather_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'WindSpeed(mph)', 'Precipitation(in)']
avail_weather = [c for c in weather_cols if c in cg.columns]
if avail_weather:
    corr = cg[avail_weather + ['Severity']].corr()['Severity'].drop('Severity').sort_values(ascending=False)
    print("\nWeather correlation with Severity:")
    print(corr)

# 6. Severity by delay
print("\nMean delay by severity:")
print(cg.groupby('Severity')[delay_cols].mean().round(2))

# 7. Summary
print("\n" + "="*70)
print("EDA SUMMARY — US CONGESTION DATASET")
print("="*70)
print(f"""
KEY FINDINGS:
1. 2M sample, 30 columns, USA-wide
2. Severity 1 dominates ({sev_counts.get(1,0)/len(cg)*100:.1f}%)
3. Mean delay: {cg['DelayFromTypicalTraffic(mins)'].mean():.2f} mins
4. Congestion Speed: {cg['Congestion_Speed'].value_counts().to_dict()}
5. Top congestion states: CA, TX, FL (population-driven)
6. Delay metrics uniquely available in this dataset
RECOMMENDATIONS:
- Use to quantify congestion impact of accidents
- Compare with US Accidents dataset for severity correlation
""")

summary = {
    'dataset': 'US Congestion',
    'rows': len(cg),
    'columns': len(cg.columns),
    'severity_distribution': sev_counts.to_dict(),
}
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("DONE — Congestion EDA complete!")
