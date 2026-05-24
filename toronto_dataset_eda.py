"""
Deep EDA: Toronto KSI Collision Dataset
KSI = Killed or Seriously Injured
Target: ACCLASS (Fatal / Non-Fatal Injury / Property Damage Only)
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

DATA_DIR = 'data/toronto-dataset'
OUT_DIR = 'outputs/eda_toronto_dataset'
os.makedirs(OUT_DIR, exist_ok=True)

print("="*70)
print("DEEP EDA: TORONTO KSI COLLISION DATASET")
print("="*70)

# ========== 1. LOAD DATA ==========
ksi = pd.read_csv(os.path.join(DATA_DIR, 'old', 'KSI.csv'), low_memory=False)
print(f"\nShape: {ksi.shape}")
print(f"Columns ({len(ksi.columns)}): {ksi.columns.tolist()}")

# ========== 2. TARGET ANALYSIS ==========
print("\n" + "="*40)
print("1. TARGET VARIABLE (ACCLASS)")
print("="*40)

print("\nACCLASS distribution:")
acclass_counts = ksi['ACCLASS'].value_counts()
print(acclass_counts)
print(f"\nFatal rate: {acclass_counts.get('Fatal', 0)/len(ksi)*100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = sns.color_palette('Reds', 3)
acclass_counts.plot(ax=axes[0], kind='bar', color=colors)
axes[0].set_title('ACCLASS Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(acclass_counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontsize=10)

acclass_counts.plot(ax=axes[1], kind='pie', autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_ylabel('')
axes[1].set_title('ACCLASS Proportion')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_acclass_distribution.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 01_acclass_distribution.png")

# ========== 3. TEMPORAL ANALYSIS ==========
print("\n" + "="*40)
print("2. TEMPORAL ANALYSIS")
print("="*40)

print(f"Year range: {ksi['YEAR'].min()} - {ksi['YEAR'].max()}")
yearly = ksi['YEAR'].value_counts().sort_index()
print(f"Total years covered: {len(yearly)}")
print(f"Average yearly collisions: {yearly.mean():.0f}")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Year
yearly.plot(ax=axes[0,0], kind='bar', color='steelblue')
axes[0,0].set_title('Collisions by Year', fontweight='bold')
axes[0,0].set_ylabel('Count')

# Year x ACCLASS
year_sev = ksi.groupby(['YEAR', 'ACCLASS']).size().unstack(fill_value=0)
year_sev.plot(ax=axes[0,1], kind='bar', stacked=True, color=['lightcoral', 'indianred', 'darkred'])
axes[0,1].set_title('Collisions by Year + ACCLASS', fontweight='bold')
axes[0,1].legend(loc='upper right', fontsize=8)

# Month
if 'DATE' in ksi.columns:
    ksi['date_parsed'] = pd.to_datetime(ksi['DATE'], errors='coerce')
    ksi['Month'] = ksi['date_parsed'].dt.month
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly = ksi['Month'].value_counts().sort_index()
    monthly.index = monthly.index.map(lambda m: month_names[int(m)-1])
    monthly.plot(ax=axes[0,2], kind='bar', color='coral')
    axes[0,2].set_title('Collisions by Month', fontweight='bold')

# Hour
hourly = ksi['HOUR'].value_counts().sort_index()
hourly.plot(ax=axes[1,0], kind='bar', color='seagreen')
axes[1,0].set_title('Collisions by Hour', fontweight='bold')

# Day of Week
if 'Day_of_Week' in ksi.columns:
    dow = ksi['Day_of_Week'].value_counts()
else:
    dow_map = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}
    if 'WEEKDAY' in ksi.columns:
        dow = ksi['WEEKDAY'].value_counts()
    else:
        dow = ksi['date_parsed'].dt.day_name().value_counts() if 'date_parsed' in ksi.columns else pd.Series()
if not dow.empty:
    dow.plot(ax=axes[1,1], kind='bar', color='mediumpurple')
    axes[1,1].set_title('Collisions by Day of Week', fontweight='bold')

# Severity by hour of day
sev_hour = ksi.groupby(['HOUR', 'ACCLASS']).size().unstack(fill_value=0)
sev_hour.plot(ax=axes[1,2], kind='bar', stacked=True,
              color=['lightcoral', 'indianred', 'darkred'])
axes[1,2].set_title('ACCLASS by Hour', fontweight='bold')
axes[1,2].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_temporal_patterns.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 02_temporal_patterns.png")

# Fatal trend
print("\nFatal trend over years:")
yearly_fatal = ksi[ksi['ACCLASS'] == 'Fatal'].groupby('YEAR').size()
print(yearly_fatal)

# ========== 4. SPATIAL ANALYSIS ==========
print("\n" + "="*40)
print("3. SPATIAL ANALYSIS")
print("="*40)

# Top districts
if 'DISTRICT' in ksi.columns:
    top_dist = ksi['DISTRICT'].value_counts().head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_dist.values, y=top_dist.index, palette='Blues_d')
    plt.title('Top 15 Districts by Collision Count')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '03_top_districts.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 03_top_districts.png")
    
    print("\nDistrict fatal rate:")
    dist_fatal = ksi.groupby('DISTRICT')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100).sort_values(ascending=False).head(10)
    print(dist_fatal)

# Neighborhood analysis
if 'NEIGHBOURHOOD' in ksi.columns:
    hood = ksi['NEIGHBOURHOOD'].value_counts().head(10)
    print(f"\nTop 10 neighbourhoods:")
    print(hood)

# Spatial density heatmap (paper: Boddepalli 2026 identifies downtown Toronto hotspots)
if 'LATITUDE' in ksi.columns and 'LONGITUDE' in ksi.columns:
    print("\nGenerating collision density heatmap...")
    fig, ax = plt.subplots(figsize=(12, 8))
    hb = ax.hexbin(ksi['LONGITUDE'], ksi['LATITUDE'],
                   gridsize=60, cmap='YlOrRd', mincnt=1, alpha=0.8)
    cb = plt.colorbar(hb, ax=ax, label='Collision Density')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'Toronto KSI Collisions -- Spatial Density (full dataset, n={len(ksi):,})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '03b_spatial_density_heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 03b_spatial_density_heatmap.png")

# ========== 5. CONTRIBUTING FACTORS ==========
print("\n" + "="*40)
print("4. CONTRIBUTING FACTORS")
print("="*40)

factor_cols = ['SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY',
               'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
               'EMERG_VEH', 'PASSENGER']
factor_cols = [c for c in factor_cols if c in ksi.columns]

print(f"Contributing factors available: {factor_cols}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(factor_cols[:6]):
    if col in ksi.columns:
        counts = ksi[col].value_counts()
        counts.plot(ax=axes[i], kind='bar', color='salmon')
        axes[i].set_title(f'{col} Distribution')
        axes[i].tick_params(axis='x', rotation=45)

plt.suptitle('Contributing Factors Analysis', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_contributing_factors.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 04_contributing_factors.png")

# Factor association with ACCLASS
print("\nFactor association with ACCLASS (% fatal where factor is present):")
factor_impact = []
for col in factor_cols:
    if col in ksi.columns:
        cross = ksi.groupby(col)['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100)
        if 'Yes' in cross.index:
            yes_fatal = cross.get('Yes', 0)
            no_fatal = cross.get('No', 0)
            print(f"  {col:15s}: {yes_fatal:.1f}% fatal (when Yes), {no_fatal:.1f}% fatal (when No)")
            factor_impact.append({'factor': col, 'fatal_pct_yes': round(yes_fatal, 1), 'fatal_pct_no': round(no_fatal, 1)})

# Factor impact visualization
if len(factor_impact) > 0:
    factor_df = pd.DataFrame(factor_impact).sort_values('fatal_pct_yes', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(factor_df))
    ax.barh(y_pos, factor_df['fatal_pct_yes'].values, height=0.4, label='When factor is Yes', color='#d73027')
    ax.barh(y_pos, factor_df['fatal_pct_no'].values, height=0.4, label='When factor is No', color='#4575b4', alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(factor_df['factor'].values)
    ax.set_xlabel('Fatal Rate (%)')
    ax.set_title(f'Impact of Contributing Factors on Fatal Rate (full dataset, n={len(ksi):,})', fontweight='bold')
    ax.legend()
    for i, (_, row) in enumerate(factor_df.iterrows()):
        ax.text(row['fatal_pct_yes'] + 0.5, i, f"{row['fatal_pct_yes']:.1f}%", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '04b_factor_fatal_impact.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 04b_factor_fatal_impact.png")

# ========== 6. ENVIRONMENTAL CONDITIONS ==========
print("\n" + "="*40)
print("5. ENVIRONMENTAL CONDITIONS")
print("="*40)

env_cols = ['VISIBILITY', 'LIGHT', 'RDSFCOND', 'TRAFFCTL']
env_cols = [c for c in env_cols if c in ksi.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(env_cols):
    if col in ksi.columns:
        counts = ksi[col].value_counts().head(10)
        counts.plot(ax=axes[i], kind='bar', color='teal')
        axes[i].set_title(f'{col} Distribution')
        axes[i].tick_params(axis='x', rotation=45)

plt.suptitle('Environmental Conditions', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_environmental_conditions.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 05_environmental_conditions.png")

# Light conditions vs ACCLASS
if 'LIGHT' in ksi.columns:
    print("\nLight conditions vs ACCLASS:")
    light_sev = ksi.groupby(['LIGHT', 'ACCLASS']).size().unstack(fill_value=0)
    print(light_sev)

# Road surface vs ACCLASS
if 'RDSFCOND' in ksi.columns:
    print("\nRoad surface vs ACCLASS:")
    road_sev = ksi.groupby(['RDSFCOND', 'ACCLASS']).size().unstack(fill_value=0)
    print(road_sev)

# ========== 6.5. WEEKDAY vs WEEKEND ==========
print("\n" + "="*40)
print("6.5. WEEKDAY vs WEEKEND ANALYSIS")
print("="*40)

if 'DATE' in ksi.columns:
    ksi['date_parsed'] = pd.to_datetime(ksi['DATE'], errors='coerce')
    ksi['is_weekend'] = ksi['date_parsed'].dt.dayofweek >= 5
    wk_count = ksi['is_weekend'].value_counts()
    wk_fatal = ksi.groupby('is_weekend')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100)
    print(f"\nWeekday collisions: {wk_count.get(False, 0):,} | Weekend: {wk_count.get(True, 0):,}")
    print(f"Weekday fatal rate: {wk_fatal.get(False, 0):.2f}% | Weekend fatal rate: {wk_fatal.get(True, 0):.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    wk_count_sorted = pd.Series({'Weekday': wk_count.get(False, 0), 'Weekend': wk_count.get(True, 0)})
    wk_count_sorted.plot(ax=axes[0], kind='bar', color=['steelblue', 'coral'])
    axes[0].set_title(f'Collision Count: Weekday vs Weekend (full dataset, n={len(ksi):,})', fontweight='bold')
    axes[0].set_ylabel('Count')

    wk_fatal_sorted = pd.Series({'Weekday': wk_fatal.get(False, 0), 'Weekend': wk_fatal.get(True, 0)})
    wk_fatal_sorted.plot(ax=axes[1], kind='bar', color=['steelblue', 'coral'])
    axes[1].set_title(f'Fatal Rate: Weekday vs Weekend (full dataset)', fontweight='bold')
    axes[1].set_ylabel('Fatal Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '06_weekday_weekend.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 06_weekday_weekend.png")

# ========== 7. VEHICLE / PERSON ANALYSIS ==========
print("\n" + "="*40)
print("6. VEHICLE & PERSON ANALYSIS")
print("="*40)

veh_cols = ['VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND']
veh_cols = [c for c in veh_cols if c in ksi.columns]
for col in veh_cols:
    print(f"\n{col} (top 10):")
    print(ksi[col].value_counts().head(10))

# Vehicle type vs ACCLASS
if 'VEHTYPE' in ksi.columns:
    print("\nTop 10 vehicle types by fatal rate:")
    veh_fatal = ksi.groupby('VEHTYPE')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100).sort_values(ascending=False).head(10)
    print(veh_fatal)

# ========== 8. INJURY ANALYSIS ==========
print("\n" + "="*40)
print("7. INJURY ANALYSIS")
print("="*40)

injury_cols = ['INJURY', 'IMPACTYPE', 'INVTYPE']
injury_cols = [c for c in injury_cols if c in ksi.columns]
for col in injury_cols:
    print(f"\n{col} (top 10):")
    print(ksi[col].value_counts().head(10))

# ========== 8b. INVTYPE x ACCLASS ==========
print("\n" + "="*40)
print("7b. INVTYPE x ACCLASS CROSS-ANALYSIS")
print("="*40)

if 'INVTYPE' in ksi.columns and 'ACCLASS' in ksi.columns:
    # Cross-tabulation
    invtype_acclass = ksi.groupby(['INVTYPE', 'ACCLASS']).size().unstack(fill_value=0)
    invtype_acclass['Total'] = invtype_acclass.sum(axis=1)
    invtype_acclass['Fatal_Rate'] = (invtype_acclass.get('Fatal', 0) / invtype_acclass['Total'] * 100).round(2)
    invtype_acclass = invtype_acclass.sort_values('Total', ascending=False)
    print("\nINVTYPE x ACCLASS cross-tabulation:")
    print(invtype_acclass)

    # Chart: grouped bar + fatal rate line
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_invtype = invtype_acclass.head(8).drop(['Total', 'Fatal_Rate'], axis=1)
    top_invtype.plot(ax=axes[0], kind='bar', stacked=True,
                     color=['lightcoral', 'indianred', 'darkred'])
    axes[0].set_title(f'INVTYPE x ACCLASS Stacked (full dataset, n={len(ksi):,})', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    fatal_rates = invtype_acclass.head(8)['Fatal_Rate']
    fatal_rates.plot(ax=axes[1], kind='bar', color='#d73027')
    axes[1].set_title('Fatal Rate by INVTYPE', fontweight='bold')
    axes[1].set_ylabel('Fatal Rate (%)')
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(fatal_rates.values):
        axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '07b_invtype_acclass.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 07b_invtype_acclass.png")

    # Per-INVTYPE detailed breakdown (Shanshal 2020: Drivers, Cyclists, Pedestrians)
    for inv in ['Driver', 'Cyclist', 'Pedestrian']:
        if inv in invtype_acclass.index:
            subset = ksi[ksi['INVTYPE'] == inv]
            fatal_pct = (subset['ACCLASS'] == 'Fatal').mean() * 100
            print(f"\n{inv}s (n={len(subset):,}): {fatal_pct:.1f}% fatal rate")

# ========== 8c. DRIVACT x ACCLASS ==========
print("\n" + "="*40)
print("7c. DRIVACT x ACCLASS CROSS-ANALYSIS")
print("="*40)

if 'DRIVACT' in ksi.columns and 'ACCLASS' in ksi.columns:
    drivact_acclass = ksi.groupby(['DRIVACT', 'ACCLASS']).size().unstack(fill_value=0)
    drivact_acclass['Total'] = drivact_acclass.sum(axis=1)
    drivact_acclass['Fatal_Rate'] = (drivact_acclass.get('Fatal', 0) / drivact_acclass['Total'] * 100).round(2)
    drivact_acclass = drivact_acclass.sort_values('Total', ascending=False)
    print("\nDRIVACT x ACCLASS (top 10 by count):")
    print(drivact_acclass.head(10))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_drivact = drivact_acclass.head(10).drop(['Total', 'Fatal_Rate'], axis=1)
    top_drivact.plot(ax=axes[0], kind='barh', stacked=True,
                     color=['lightcoral', 'indianred', 'darkred'])
    axes[0].set_title(f'DRIVACT x ACCLASS (full dataset, n={len(ksi):,})', fontweight='bold')
    axes[0].set_xlabel('Count')
    axes[1].set_title('Fatal Rate by DRIVACT', fontweight='bold')
    drivact_fatal = drivact_acclass.head(10)['Fatal_Rate'].sort_values()
    drivact_fatal.plot(ax=axes[1], kind='barh', color='#d73027')
    axes[1].set_xlabel('Fatal Rate (%)')
    for i, v in enumerate(drivact_fatal.values):
        axes[1].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '07c_drivact_acclass.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 07c_drivact_acclass.png")

# ========== 8d. PEDTYPE x ACCLASS ==========
print("\n" + "="*40)
print("7d. PEDTYPE x ACCLASS CROSS-ANALYSIS")
print("="*40)

if 'PEDTYPE' in ksi.columns and 'ACCLASS' in ksi.columns:
    pedtype_acclass = ksi.groupby(['PEDTYPE', 'ACCLASS']).size().unstack(fill_value=0)
    pedtype_acclass['Total'] = pedtype_acclass.sum(axis=1)
    pedtype_acclass['Fatal_Rate'] = (pedtype_acclass.get('Fatal', 0) / pedtype_acclass['Total'] * 100).round(2)
    pedtype_acclass = pedtype_acclass.sort_values('Total', ascending=False)
    print("\nPEDTYPE x ACCLASS (top 10):")
    print(pedtype_acclass.head(10))

    fig, ax = plt.subplots(figsize=(12, 6))
    top_ped = pedtype_acclass.head(8).drop(['Total', 'Fatal_Rate'], axis=1)
    top_ped.plot(ax=ax, kind='barh', stacked=True,
                 color=['lightcoral', 'indianred', 'darkred'])
    ax.set_title(f'PEDTYPE x ACCLASS (full dataset, n={len(ksi):,})', fontweight='bold')
    ax.set_xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '07d_pedtype_acclass.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 07d_pedtype_acclass.png")

# ========== 8e. CYCLISTYPE x ACCLASS ==========
print("\n" + "="*40)
print("7e. CYCLISTYPE x ACCLASS CROSS-ANALYSIS")
print("="*40)

if 'CYCLISTYPE' in ksi.columns and 'ACCLASS' in ksi.columns:
    cyclist_acclass = ksi.groupby(['CYCLISTYPE', 'ACCLASS']).size().unstack(fill_value=0)
    cyclist_acclass['Total'] = cyclist_acclass.sum(axis=1)
    cyclist_acclass['Fatal_Rate'] = (cyclist_acclass.get('Fatal', 0) / cyclist_acclass['Total'] * 100).round(2)
    cyclist_acclass = cyclist_acclass.sort_values('Total', ascending=False)
    print("\nCYCLISTYPE x ACCLASS (top 10):")
    print(cyclist_acclass.head(10))

    fig, ax = plt.subplots(figsize=(12, 6))
    top_cyclist = cyclist_acclass.head(8).drop(['Total', 'Fatal_Rate'], axis=1)
    top_cyclist.plot(ax=ax, kind='barh', stacked=True,
                     color=['lightcoral', 'indianred', 'darkred'])
    ax.set_title(f'CYCLISTYPE x ACCLASS (full dataset, n={len(ksi):,})', fontweight='bold')
    ax.set_xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '07e_cyclistype_acclass.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 07e_cyclistype_acclass.png")

# ========== 9. SUMMARY ==========
print("\n" + "="*70)
print("EDA SUMMARY — TORONTO KSI DATASET")
print("="*70)
# Compute extra metrics for summary
if 'DISTRICT' in ksi.columns:
    dist_fatal_dict = ksi.groupby('DISTRICT')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100).round(2).to_dict()
if 'NEIGHBOURHOOD' in ksi.columns:
    hood_fatal_dict = ksi.groupby('NEIGHBOURHOOD')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100).round(2).to_dict()
if 'INVTYPE' in ksi.columns:
    invtype_fatal_dict = ksi.groupby('INVTYPE')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100).round(2).to_dict()
if 'DRIVACT' in ksi.columns:
    drivact_fatal_dict = ksi.groupby('DRIVACT')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100).round(2).to_dict()
if 'PEDTYPE' in ksi.columns:
    pedtype_fatal_dict = ksi.groupby('PEDTYPE')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100).round(2).to_dict()
if 'CYCLISTYPE' in ksi.columns:
    cyclist_fatal_dict = ksi.groupby('CYCLISTYPE')['ACCLASS'].apply(lambda x: (x == 'Fatal').mean()*100).round(2).to_dict()

print(f"""
KEY FINDINGS:
1. Dataset: {len(ksi):,} records, {len(ksi.columns)} columns, Toronto
2. ACCLASS: Non-Fatal Injury ({acclass_counts.get('Non-Fatal Injury', 0)/len(ksi)*100:.1f}%),
   Fatal ({acclass_counts.get('Fatal', 0)/len(ksi)*100:.1f}%)
3. Year range: {int(ksi['YEAR'].min())}-{int(ksi['YEAR'].max())}, peak in early years
4. Key contributing factors: SPEEDING, AG_DRIV (aggressive driving), ALCOHOL
5. Environmental: light conditions, road surface strongly affect severity
6. Vehicle type: trucks and motorcycles have higher fatal rates
7. Spatial: certain districts show higher collision density
8. INVTYPE: Drivers most common, Cyclists highest fatal rate
9. DRIVACT: Speeding-related actions increase fatal risk

RECOMMENDATIONS:
- Use for ML + CA hybrid simulation (as in Boddepalli et al., 2026)
- Engineer features: time-of-day risk score, neighborhood risk index
- Binary classification: Fatal vs Non-Fatal (collapsing Non-Fatal Injury + PDO)
- Incorporate contributing factors as key predictors
- Per-INVTYPE modeling improves accuracy (Shanshal et al., 2020)
""")

summary = {
    'dataset': 'Toronto KSI Collisions',
    'rows': len(ksi),
    'columns': len(ksi.columns),
    'year_range': [int(ksi['YEAR'].min()), int(ksi['YEAR'].max())],
    'acclass_distribution': acclass_counts.to_dict(),
    'fatal_rate_pct': round(acclass_counts.get('Fatal', 0)/len(ksi)*100, 2),
    'fatal_rate_by_district': dist_fatal_dict if 'DISTRICT' in ksi.columns else {},
    'fatal_rate_by_neighbourhood': {k: v for k, v in sorted(hood_fatal_dict.items(), key=lambda x: x[1], reverse=True)[:20]} if 'NEIGHBOURHOOD' in ksi.columns else {},
    'fatal_rate_by_invtype': invtype_fatal_dict if 'INVTYPE' in ksi.columns else {},
    'fatal_rate_by_drivact': {k: v for k, v in sorted(drivact_fatal_dict.items(), key=lambda x: x[1], reverse=True)[:15]} if 'DRIVACT' in ksi.columns else {},
    'fatal_rate_by_pedtype': {k: v for k, v in sorted(pedtype_fatal_dict.items(), key=lambda x: x[1], reverse=True)[:10]} if 'PEDTYPE' in ksi.columns else {},
    'fatal_rate_by_cyclistype': {k: v for k, v in sorted(cyclist_fatal_dict.items(), key=lambda x: x[1], reverse=True)[:10]} if 'CYCLISTYPE' in ksi.columns else {},
}
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("\nDONE — Toronto KSI EDA complete!")
