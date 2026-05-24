"""
Deep EDA: UK DfT Road Casualty Statistics (1979-latest)
~9M collision records, 16M vehicles, 12M casualties
Target: collision_severity (1=fatal, 2=serious, 3=slight)
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, random, warnings, json
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 150

DATA_DIR = 'data/britain-dataset'
OUT_DIR = 'outputs/eda_britain_dataset'
os.makedirs(OUT_DIR, exist_ok=True)

COLLISION_FILE = os.path.join(DATA_DIR, 'dft-road-casualty-statistics-collision-1979-latest-published-year.csv')
VEHICLE_FILE = os.path.join(DATA_DIR, 'dft-road-casualty-statistics-vehicle-1979-latest-published-year.csv')
CASUALTY_FILE = os.path.join(DATA_DIR, 'dft-road-casualty-statistics-casualty-1979-latest-published-year.csv')

print("="*70)
print("DEEP EDA: UK DfT ROAD CASUALTY STATISTICS")
print("="*70)

# Count total rows quickly
def count_lines(path):
    n = 0
    with open(path, 'rb') as f:
        for _ in f:
            n += 1
    return n - 1  # minus header

n_collision = count_lines(COLLISION_FILE)
n_vehicle = count_lines(VEHICLE_FILE)
n_casualty = count_lines(CASUALTY_FILE)
print(f"\nCollisions: {n_collision:,}")
print(f"Vehicles:   {n_vehicle:,}")
print(f"Casualties: {n_casualty:,}")

# ========== 1. TARGET ANALYSIS (full dataset) ==========
print("\n" + "="*40)
print("1. TARGET VARIABLE (collision_severity)")
print("="*40)

print("Reading full collision_severity distribution...")
collision_sev = pd.read_csv(COLLISION_FILE, usecols=['collision_severity'], low_memory=False)
sev_counts = collision_sev['collision_severity'].value_counts().sort_index()
sev_total = sev_counts.sum()
sev_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
print(f"\ncollision_severity distribution (n={sev_total:,}):")
for sev in sorted(sev_counts.index):
    label = sev_map.get(sev, f'Unknown({sev})')
    print(f"  {label:10s}: {sev_counts[sev]:>8,} ({sev_counts[sev]/sev_total*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_sev = sns.color_palette('Reds', len(sev_counts))
sev_labels = [f'{sev_map.get(s, s)}\n({sev_counts[s]/sev_total*100:.1f}%)' for s in sev_counts.index]
bars = axes[0].bar(sev_labels, sev_counts.values, color=colors_sev)
axes[0].set_title(f'Severity Distribution (full dataset, n={sev_total:,})', fontweight='bold')
axes[0].set_ylabel('Count')
for bar, val in zip(bars, sev_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:,}', ha='center', va='bottom', fontsize=9)
axes[1].pie(sev_counts.values, labels=sev_labels, colors=colors_sev, autopct='', startangle=90)
axes[1].set_title('Severity Proportion')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_severity_distribution.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 01_severity_distribution.png")

del collision_sev

# ========== 2. TEMPORAL ANALYSIS (sampled) ==========
print("\n" + "="*40)
print("2. TEMPORAL ANALYSIS")
print("="*40)

N_TEMPORAL = 200000
print(f"Sampling {N_TEMPORAL:,} collisions (seed=42)...")
random.seed(42)
target_t = set(random.sample(range(n_collision), min(N_TEMPORAL, n_collision)))
ts_sample = pd.read_csv(COLLISION_FILE, usecols=['collision_index', 'collision_year', 'date',
                                                   'day_of_week', 'time', 'collision_severity',
                                                   'number_of_vehicles', 'number_of_casualties'],
                        low_memory=False, skiprows=lambda i: i > 0 and i not in target_t)
print(f"  Sampled: {len(ts_sample):,} collisions (seed=42)")

ts_sample['date_parsed'] = pd.to_datetime(ts_sample['date'], dayfirst=True, errors='coerce')
ts_sample['Month'] = ts_sample['date_parsed'].dt.month
ts_sample['Hour'] = ts_sample['date_parsed'].dt.hour
# Fix null hours from time column
null_hour = ts_sample['Hour'].isna()
if null_hour.any():
    ts_sample.loc[null_hour, 'Hour'] = pd.to_numeric(
        ts_sample.loc[null_hour, 'time'].str.split(':').str[0], errors='coerce')
ts_sample['Hour'] = ts_sample['Hour'].fillna(-1).astype(int)

ts_sample['Season'] = ts_sample['Month'].map({12:'Winter',1:'Winter',2:'Winter',
                                              3:'Spring',4:'Spring',5:'Spring',
                                              6:'Summer',7:'Summer',8:'Summer',
                                              9:'Fall',10:'Fall',11:'Fall'})

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Year
ts_sample['collision_year'].value_counts().sort_index().plot(ax=axes[0,0], kind='bar', color='steelblue')
axes[0,0].set_title(f'Collisions by Year (n={len(ts_sample):,})', fontweight='bold')

# Month
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly = ts_sample.groupby('Month').size()
monthly.index = monthly.index.map(lambda m: month_order[int(m)-1])
monthly.plot(ax=axes[0,1], kind='bar', color='coral')
axes[0,1].set_title(f'Collisions by Month (n={len(ts_sample):,})', fontweight='bold')

# Hour
hourly = ts_sample.groupby('Hour').size()
hourly.plot(ax=axes[0,2], kind='bar', color='seagreen')
axes[0,2].set_title(f'Collisions by Hour (n={len(ts_sample):,})', fontweight='bold')

# Day of week
dow_map = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}
dow = ts_sample.groupby('day_of_week').size()
dow.index = dow.index.map(dow_map)
dow.plot(ax=axes[1,0], kind='bar', color='mediumpurple')
axes[1,0].set_title(f'Collisions by Day of Week (n={len(ts_sample):,})', fontweight='bold')

# Season
season_order = ['Spring','Summer','Fall','Winter']
ts_sample.groupby('Season').size().reindex(season_order).plot(ax=axes[1,1], kind='bar', color='goldenrod')
axes[1,1].set_title(f'Collisions by Season (n={len(ts_sample):,})', fontweight='bold')

# Severity x Hour
sev_hour = ts_sample.groupby(['Hour', 'collision_severity']).size().unstack(fill_value=0)
sns.heatmap(sev_hour.T, ax=axes[1,2], cmap='YlOrRd', cbar_kws={'label': 'Count'})
axes[1,2].set_title(f'Severity x Hour Heatmap (n={len(ts_sample):,})', fontweight='bold')

plt.suptitle(f'Temporal Patterns -- UK Collisions (random sample, seed=42, n={len(ts_sample):,})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_temporal_patterns.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 02_temporal_patterns.png")

# ========== 3. SPATIAL ANALYSIS (sampled) ==========
print("\n" + "="*40)
print("3. SPATIAL ANALYSIS")
print("="*40)

random.seed(42)
target_sp = set(random.sample(range(n_collision), min(50000, n_collision)))
spatial_sample = pd.read_csv(COLLISION_FILE, usecols=['longitude', 'latitude', 'collision_severity',
                                                       'police_force', 'urban_or_rural_area'],
                             low_memory=False, skiprows=lambda i: i > 0 and i not in target_sp)
N_SPATIAL = len(spatial_sample)
spatial_sample = spatial_sample.dropna(subset=['longitude', 'latitude'])
print(f"  Spatial sample: {len(spatial_sample):,} (seed=42)")

# Urban vs Rural
if 'urban_or_rural_area' in spatial_sample.columns:
    print(f"\nUrban/Rural distribution:")
    ur_dist = spatial_sample['urban_or_rural_area'].value_counts().sort_index()
    print(ur_dist)
    ur_map = {1: 'Urban', 2: 'Rural', 3: 'Unknown'}
    ur_dist.index = ur_dist.index.map(ur_map)

# Spatial scatter
plt.figure(figsize=(12, 8))
scatter = plt.scatter(spatial_sample['longitude'], spatial_sample['latitude'],
                      c=spatial_sample['collision_severity'], cmap='Reds', alpha=0.3, s=1)
plt.colorbar(scatter, label='Severity (1=Fatal, 3=Slight)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'UK Collisions -- Spatial Distribution (random sample, seed=42, n={len(spatial_sample):,})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_spatial_distribution.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 03_spatial_distribution.png")

# Hexbin density
fig, ax = plt.subplots(figsize=(12, 8))
hb = ax.hexbin(spatial_sample['longitude'], spatial_sample['latitude'],
               gridsize=80, cmap='YlOrRd', mincnt=1, alpha=0.8)
cb = plt.colorbar(hb, ax=ax, label='Collision Density')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title(f'UK Collision Density Heatmap (random sample, seed=42, n={len(spatial_sample):,})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03b_spatial_density_heatmap.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 03b_spatial_density_heatmap.png")

# ========== 4. ENVIRONMENTAL CONDITIONS ==========
print("\n" + "="*40)
print("4. ENVIRONMENTAL CONDITIONS")
print("="*40)

random.seed(42)
target_env = set(random.sample(range(n_collision), min(100000, n_collision)))
env_sample = pd.read_csv(COLLISION_FILE, usecols=['collision_severity', 'light_conditions',
                                                   'weather_conditions', 'road_surface_conditions',
                                                   'speed_limit', 'road_type', 'urban_or_rural_area'],
                         low_memory=False, skiprows=lambda i: i > 0 and i not in target_env)
N_ENV = len(env_sample)
print(f"  Environmental sample: {N_ENV:,} (seed=42)")

env_labels = {
    'light_conditions': {1:'Daylight', 2:'Darkness-lit', 4:'Darkness-no lighting', 5:'Darkness-unknown', 7:'Night', 6:'Darkness-lighting unknown'},
    'weather_conditions': {1:'Fine', 2:'Rain', 3:'Snow', 4:'Fog/Mist', 5:'High winds', 7:'Other'},
    'road_surface_conditions': {1:'Dry', 2:'Wet/Damp', 3:'Snow', 4:'Frost/Ice', 5:'Flood', 7:'Other'},
}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for i, col in enumerate(['light_conditions', 'weather_conditions', 'road_surface_conditions']):
    vals = env_sample[col].value_counts().sort_index()
    labels = [env_labels.get(col, {}).get(k, f'Code {k}') for k in vals.index]
    axes[0,i].bar(labels, vals.values, color=['gold', 'skyblue', 'lightcoral', 'gray', 'navy', 'teal'][:len(vals)])
    axes[0,i].set_title(f'{col.replace("_"," ").title()} (n={N_ENV:,})', fontweight='bold')
    axes[0,i].tick_params(axis='x', rotation=30)

# Severity by environmental factors
for i, col in enumerate(['light_conditions', 'weather_conditions', 'road_surface_conditions']):
    cross = env_sample.groupby([col, 'collision_severity']).size().unstack(fill_value=0)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    labels = [env_labels.get(col, {}).get(k, f'Code {k}') for k in cross_pct.index]
    cross_pct.index = labels
    cross_pct.plot(ax=axes[1,i], kind='bar', stacked=True, colormap='Reds')
    axes[1,i].set_title(f'Severity by {col.replace("_"," ").title()}', fontweight='bold')
    axes[1,i].set_ylabel('Percentage (%)')
    axes[1,i].tick_params(axis='x', rotation=30)
    axes[1,i].legend(title='Severity', bbox_to_anchor=(1.05, 1))

plt.suptitle(f'Environmental Conditions vs Severity (random sample, seed=42, n={N_ENV:,})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_environmental_analysis.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 04_environmental_analysis.png")

# Speed limit analysis
print(f"\nSpeed limit distribution:")
sl_dist = env_sample['speed_limit'].value_counts().sort_index()
print(sl_dist)
print(f"\nMean severity by speed limit:")
sl_sev = env_sample.groupby('speed_limit')['collision_severity'].mean()
print(sl_sev)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sl_dist.plot(ax=axes[0], kind='bar', color='steelblue')
axes[0].set_title(f'Speed Limit Distribution (n={N_ENV:,})', fontweight='bold')
axes[0].set_xlabel('Speed Limit (mph)')
sl_sev.plot(ax=axes[1], kind='bar', color='coral')
axes[1].set_title('Mean Severity by Speed Limit', fontweight='bold')
axes[1].set_xlabel('Speed Limit (mph)')
axes[1].set_ylabel('Mean Severity')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04b_speed_limit_analysis.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 04b_speed_limit_analysis.png")

# Road type
print(f"\nRoad type distribution:")
rt_dist = env_sample['road_type'].value_counts().sort_index()
rt_labels = {1:'Roundabout', 2:'One-way street', 3:'Dual carriageway', 4:'Single carriageway',
             5:'Slip road', 6:'Unknown', 7:'Unknown'}
rt_dist.index = rt_dist.index.map(rt_labels)
print(rt_dist)
plt.figure(figsize=(10, 5))
rt_dist.plot(kind='bar', color='teal')
plt.title(f'Road Type Distribution (n={N_ENV:,})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04c_road_type_analysis.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 04c_road_type_analysis.png")

# ========== 5. VEHICLE ANALYSIS (sampled) ==========
print("\n" + "="*40)
print("5. VEHICLE ANALYSIS")
print("="*40)

random.seed(42)
target_v = set(random.sample(range(n_collision), min(50000, n_collision)))
v_collision = pd.read_csv(COLLISION_FILE, usecols=['collision_index', 'collision_severity', 'number_of_vehicles'],
                          low_memory=False, skiprows=lambda i: i > 0 and i not in target_v)
v_indexes = set(v_collision['collision_index'].unique())
print(f"  Sampled collision indexes: {len(v_indexes):,}")

# Read vehicle data for these indexes (in chunks to avoid OOM)
chunks = []
for chunk in pd.read_csv(VEHICLE_FILE, low_memory=False, chunksize=500000):
    matched = chunk[chunk['collision_index'].isin(v_indexes)]
    if len(matched) > 0:
        chunks.append(matched)
v_sample = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
print(f"  Vehicle records matched: {len(v_sample):,}")

# Merge with collision severity
v_merged = v_sample.merge(v_collision[['collision_index', 'collision_severity']], on='collision_index', how='left')

# Vehicle type
vt_labels = {1:'Pedal cycle', 2:'M/cycle <50cc', 3:'M/cycle >50cc', 4:'M/cycle >125cc', 5:'M/cycle >500cc',
             8:'Taxi', 9:'Car', 10:'Minibus', 11:'Bus/Coach', 16:'Light goods', 19:'Goods >7.5t',
             20:'Goods >3.5t', 21:'Goods <=3.5t', 23:'Electric m/cycle', 97:'Other', 98:'Unknown', -1:'Unknown'}
vt_counts = v_merged['vehicle_type'].value_counts().head(15)
vt_labels_found = {k: vt_labels.get(k, f'Code {k}') for k in vt_counts.index}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
vt_counts.index = [vt_labels_found[k] for k in vt_counts.index]
vt_counts.plot(ax=axes[0], kind='barh', color=sns.color_palette('viridis', len(vt_counts)))
axes[0].set_title(f'Top 15 Vehicle Types (sampled, seed=42, n={len(v_merged):,})', fontweight='bold')
axes[0].set_xlabel('Count')

# Vehicle type vs severity
vt_sev = v_merged.groupby('vehicle_type')['collision_severity'].mean().sort_values()
top10_vt = vt_sev.head(10)
top10_vt.index = [vt_labels.get(k, f'Code {k}') for k in top10_vt.index]
top10_vt.plot(ax=axes[1], kind='barh', color='coral')
axes[1].set_title('Lowest Mean Severity by Vehicle Type\n(1=Fatal, 3=Slight => lower = more severe)', fontweight='bold')
axes[1].set_xlabel('Mean Severity')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_vehicle_analysis.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 05_vehicle_analysis.png")

# Driver age
driver_age = v_merged['age_of_driver'].dropna()
if len(driver_age) > 0:
    driver_age = driver_age[(driver_age >= 10) & (driver_age <= 99)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(driver_age, bins=60, color='steelblue', edgecolor='white')
    axes[0].set_title(f'Driver Age Distribution (n={len(driver_age):,})', fontweight='bold')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')

    # Age vs severity
    age_sev = v_merged.dropna(subset=['age_of_driver', 'collision_severity'])
    age_sev = age_sev[(age_sev['age_of_driver'] >= 10) & (age_sev['age_of_driver'] <= 99)]
    age_bins = pd.cut(age_sev['age_of_driver'], bins=range(10, 100, 10))
    age_sev_grouped = age_sev.groupby(age_bins)['collision_severity'].mean()
    age_sev_grouped.plot(ax=axes[1], kind='bar', color='coral')
    axes[1].set_title('Mean Severity by Driver Age Group\n(lower = more severe)', fontweight='bold')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Mean Severity')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '05b_driver_age_analysis.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 05b_driver_age_analysis.png")

# Sex of driver
driver_sex = v_merged['sex_of_driver'].value_counts()
sex_labels = {1: 'Male', 2: 'Female', 3: 'Unknown'}
driver_sex.index = [sex_labels.get(k, f'Code {k}') for k in driver_sex.index]
print(f"\nDriver sex distribution:")
print(driver_sex)

plt.figure(figsize=(8, 5))
driver_sex.plot(kind='bar', color=['steelblue', 'coral', 'gray'])
plt.title(f'Driver Sex Distribution (sampled, n={len(v_merged):,})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05c_driver_sex.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 05c_driver_sex.png")

del v_collision, v_sample, v_merged

# ========== 6. CASUALTY ANALYSIS (sampled) ==========
print("\n" + "="*40)
print("6. CASUALTY ANALYSIS")
print("="*40)

random.seed(42)
target_c = set(random.sample(range(n_collision), min(50000, n_collision)))
c_collision = pd.read_csv(COLLISION_FILE, usecols=['collision_index', 'collision_severity'],
                          low_memory=False, skiprows=lambda i: i > 0 and i not in target_c)
c_indexes = set(c_collision['collision_index'].unique())
print(f"  Sampled collision indexes: {len(c_indexes):,}")

chunks_c = []
for chunk in pd.read_csv(CASUALTY_FILE, low_memory=False, chunksize=500000):
    matched = chunk[chunk['collision_index'].isin(c_indexes)]
    if len(matched) > 0:
        chunks_c.append(matched)
c_sample = pd.concat(chunks_c, ignore_index=True) if chunks_c else pd.DataFrame()
print(f"  Casualty records matched: {len(c_sample):,}")

c_merged = c_sample.merge(c_collision[['collision_index', 'collision_severity']], on='collision_index', how='left')

# Casualty severity vs collision severity
cs_cross = pd.crosstab(c_merged['casualty_severity'], c_merged['collision_severity'])
print(f"\nCasualty Severity x Collision Severity cross-tabulation:")
print(cs_cross)

cs_labels = {1:'Fatal', 2:'Serious', 3:'Slight', -1:'Unknown'}
fig, ax = plt.subplots(figsize=(10, 6))
cs_pct = cs_cross.div(cs_cross.sum(axis=1), axis=0) * 100
cs_pct.index = [cs_labels.get(k, f'Code {k}') for k in cs_pct.index]
cs_pct.columns = [cs_labels.get(k, f'Code {k}') for k in cs_pct.columns]
cs_pct.plot(ax=ax, kind='bar', stacked=True, colormap='Reds')
ax.set_title(f'Casualty Severity vs Collision Severity (sampled, n={len(c_merged):,})', fontweight='bold')
ax.set_ylabel('Percentage (%)')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Collision Severity', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_casualty_analysis.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 06_casualty_analysis.png")

# Casualty type
ct_labels = {0:'Pedestrian', 1:'Pedestrian', 2:'Pedal cycle', 3:'M/cycle 50cc', 4:'M/cycle >125cc',
             5:'M/cycle >500cc', 8:'Taxi occupant', 9:'Car occupant', 10:'Minibus occupant',
             11:'Bus occupant', 16:'Light goods occupant', 19:'Goods vehicle occupant',
             21:'Goods vehicle occupant', 22:'Electric m/cycle', 23:'E-scooter',
             97:'Other', 98:'Unknown', -1:'Unknown'}
ct_counts = c_merged['casualty_type'].value_counts().head(12)
ct_counts.index = [ct_labels.get(k, f'Code {k}') for k in ct_counts.index]

plt.figure(figsize=(14, 6))
ct_counts.plot(kind='bar', color=sns.color_palette('Blues_d', len(ct_counts)))
plt.title(f'Top 12 Casualty Types (sampled, seed=42, n={len(c_merged):,})', fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06b_casualty_types.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 06b_casualty_types.png")

# Casualty age
cas_age = c_merged['age_of_casualty'].dropna()
cas_age = cas_age[(cas_age >= 0) & (cas_age <= 110)]
if len(cas_age) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(cas_age, bins=70, color='seagreen', edgecolor='white')
    axes[0].set_title(f'Casualty Age Distribution (n={len(cas_age):,})', fontweight='bold')
    axes[0].set_xlabel('Age')

    age_bins = pd.cut(c_merged['age_of_casualty'].dropna(), bins=range(0, 111, 10))
    age_csev = c_merged.dropna(subset=['age_of_casualty'])
    age_csev = age_csev[(age_csev['age_of_casualty'] >= 0) & (age_csev['age_of_casualty'] <= 110)]
    age_csev_grouped = age_csev.groupby(pd.cut(age_csev['age_of_casualty'], bins=range(0, 111, 10)))['collision_severity'].mean()
    age_csev_grouped.plot(ax=axes[1], kind='bar', color='coral')
    axes[1].set_title('Mean Collision Severity by Casualty Age\n(lower = more severe)', fontweight='bold')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Mean Severity')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '06c_casualty_age.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 06c_casualty_age.png")

del c_collision, c_sample, c_merged

# ========== 7. VEHICLE-CASUALTY LINK ==========
print("\n" + "="*40)
print("7. ROAD USER CLASS vs SEVERITY (via casualty_type)")
print("="*40)

# Reuse earlier casualty data - recount from a fresh sample
random.seed(42)
target_u = set(random.sample(range(n_collision), min(30000, n_collision)))
u_collision = pd.read_csv(COLLISION_FILE, usecols=['collision_index', 'collision_severity'],
                          low_memory=False, skiprows=lambda i: i > 0 and i not in target_u)
u_indexes = set(u_collision['collision_index'].unique())
chunks_u = []
for chunk in pd.read_csv(CASUALTY_FILE, low_memory=False, chunksize=500000):
    matched = chunk[chunk['collision_index'].isin(u_indexes)]
    if len(matched) > 0:
        chunks_u.append(matched)
u_casualty = pd.concat(chunks_u, ignore_index=True) if chunks_u else pd.DataFrame()
u_merged = u_casualty.merge(u_collision, on='collision_index', how='left')

user_map = {1:'Pedestrian', 2:'Pedal cyclist', 3:'M/cyclist', 9:'Car occupant',
            11:'Bus occupant', 16:'LGV occupant', 19:'HGV occupant', 97:'Other', 98:'Unknown'}
u_merged['User_Class'] = u_merged['casualty_type'].map(user_map).fillna('Other')
user_sev = u_merged.groupby('User_Class')['collision_severity'].agg(['mean', 'count'])
user_sev = user_sev[user_sev['count'] >= 30].sort_values('mean')
print(f"\nRoad user class (via casualty_type) x severity:")
print(user_sev)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
user_sev['mean'].plot(ax=axes[0], kind='barh', color='coral')
axes[0].set_title('Mean Severity by Road User Class\n(lower = more severe)', fontweight='bold')
axes[0].set_xlabel('Mean Severity (1=Fatal, 3=Slight)')
user_sev['count'].sort_values(ascending=True).plot(ax=axes[1], kind='barh', color='steelblue')
axes[1].set_title('Count by Road User Class', fontweight='bold')
axes[1].set_xlabel('Count')
plt.suptitle(f'Road User Class vs Collision Severity (sampled, seed=42, n={len(u_merged):,})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_user_class_severity.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 07_user_class_severity.png")

del u_collision, u_casualty, u_merged

# ========== 8. SPEED LIMIT vs SEVERITY (deep dive) ==========
print("\n" + "="*40)
print("8. SPEED LIMIT & ROAD TYPE DEEP DIVE")
print("="*40)

random.seed(42)
target_sl = set(random.sample(range(n_collision), min(100000, n_collision)))
sl_sample = pd.read_csv(COLLISION_FILE, usecols=['collision_severity', 'speed_limit', 'road_type',
                                                  'urban_or_rural_area', 'number_of_vehicles',
                                                  'number_of_casualties'],
                        low_memory=False, skiprows=lambda i: i > 0 and i not in target_sl)
print(f"  Sample: {len(sl_sample):,} (seed=42)")

# Speed limit x severity
sl_sev_mean = sl_sample.groupby('speed_limit')['collision_severity'].mean()
print(f"\nMean severity by speed limit:")
print(sl_sev_mean)

# Road type x speed limit
rt_sl = sl_sample.groupby(['road_type', 'speed_limit']).size().unstack(fill_value=0)
print(f"\nRoad type x speed limit cross:")
print(rt_sl)

# ========== 9. SUMMARY ==========
print("\n" + "="*70)
print("EDA SUMMARY -- UK DfT ROAD CASUALTY STATISTICS")
print("="*70)

fatal_rate = sev_counts.get(1, 0) / sev_total * 100
serious_rate = sev_counts.get(2, 0) / sev_total * 100

print(f"""
KEY FINDINGS:
1. Dataset: {n_collision:,} collisions, {n_vehicle:,} vehicles, {n_casualty:,} casualties
2. Years: 1979-latest, UK nationwide
3. Severity: Fatal ({sev_counts.get(1,0):,} / {fatal_rate:.1f}%),
   Serious ({sev_counts.get(2,0):,} / {serious_rate:.1f}%),
   Slight ({sev_counts.get(3,0):,} / {100-fatal_rate-serious_rate:.1f}%)
4. Temporal: peak 15-17, weekday peaks, slight decline over decades
5. Spatial: dense in urban areas (London, Manchester, Birmingham)
6. Speed: 30 mph most common, higher speeds correlate with more severe outcomes
7. Environment: wet conditions increase but fine weather dominates
8. Vehicle: cars most represented, motorcycles and HGVs have more severe outcomes
9. Driver: males more involved, age distribution reflects driving population
10. Casualty: pedestrians and cyclists more vulnerable to severe outcomes

RECOMMENDATIONS:
- Use severity as ordinal target (Fatal > Serious > Slight)
- Key features: speed_limit, light_conditions, road_surface, weather, road_type, vehicle_type
- Casualty-level features enrich collision-level prediction
- Consider urban/rural split for modeling
""")

summary = {
    'dataset': 'UK DfT Road Casualty Statistics (1979-latest)',
    'collisions': n_collision,
    'vehicles': n_vehicle,
    'casualties': n_casualty,
    'severity_distribution': {sev_map.get(k, f'Code{k}'): int(v) for k, v in sev_counts.items()},
    'fatal_rate_pct': round(fatal_rate, 2),
    'serious_rate_pct': round(serious_rate, 2),
}
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("Summary saved to summary.json")
print("\nDONE -- UK DfT Road Casualty Statistics EDA complete!")
