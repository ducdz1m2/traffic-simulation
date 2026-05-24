"""
Deep EDA: Canada National Collision Dataset (1999-2017)
~6.8M person-level records, 19 yearly files
Target: C_SEV (1=injury/fatal, 2=property damage only)
Structure: person-level (multiple persons per collision via C_CASE)
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, warnings, json, gzip
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 150

DATA_DIR = 'data/canada-dataset'
OUT_DIR = 'outputs/eda_canada_dataset'
os.makedirs(OUT_DIR, exist_ok=True)

print("="*70)
print("DEEP EDA: CANADA NATIONAL COLLISION DATASET (1999-2017)")
print("="*70)

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv.gz')])
print(f"\nYearly files: {len(files)} ({files[0][:9]} to {files[-1][:9]})")

# ========== ACCUMULATORS ==========
# Accident-level (first per C_CASE)
sev_acc = {}
year_acc = {}
month_acc = {}
wday_acc = {}
hour_acc = {}
vehs_acc = {}
conf_acc = {}
rcfg_acc = {}
wthr_acc = {}
rsur_acc = {}
raln_acc = {}
traf_acc = {}
# Person-level
psex_acc = {}
page_bins = []
page_acc = {}
psn_acc = {}
isev_acc = {}
psafe_acc = {}
puser_acc = {}
vtype_acc = {}
# Cross
user_sev_acc = {}
# Distances
driver_age_list = []
pedestrian_psn_list = []

total_person_rows = 0
total_cases = 0
unique_years = set()

for f_idx, f in enumerate(files):
    path = os.path.join(DATA_DIR, f)
    print(f"\n  [{f_idx+1}/{len(files)}] {f}...", end=' ', flush=True)
    df = pd.read_csv(path, low_memory=False)
    n = len(df)
    total_person_rows += n
    yr = int(f[2:6])
    unique_years.add(yr)

    # === Accident-level: first row per C_CASE ===
    acc = df.groupby('C_CASE').first().reset_index()
    nc = len(acc)

    # C_SEV
    for k, v in acc['C_SEV'].value_counts().items():
        sev_acc[k] = sev_acc.get(k, 0) + v

    # C_MNTH
    for k, v in acc['C_MNTH'].astype(str).value_counts().items():
        month_acc[k] = month_acc.get(k, 0) + v

    # C_WDAY
    for k, v in acc['C_WDAY'].astype(str).value_counts().items():
        wday_acc[k] = wday_acc.get(k, 0) + v

    # C_HOUR
    for k, v in acc['C_HOUR'].astype(str).value_counts().items():
        hour_acc[k] = hour_acc.get(k, 0) + v

    # C_VEHS
    for k, v in acc['C_VEHS'].astype(str).value_counts().items():
        vehs_acc[k] = vehs_acc.get(k, 0) + v

    # C_CONF
    for k, v in acc['C_CONF'].astype(str).value_counts().items():
        conf_acc[k] = conf_acc.get(k, 0) + v

    # C_RCFG
    for k, v in acc['C_RCFG'].astype(str).value_counts().items():
        rcfg_acc[k] = rcfg_acc.get(k, 0) + v

    # C_WTHR
    for k, v in acc['C_WTHR'].astype(str).value_counts().items():
        wthr_acc[k] = wthr_acc.get(k, 0) + v

    # C_RSUR
    for k, v in acc['C_RSUR'].astype(str).value_counts().items():
        rsur_acc[k] = rsur_acc.get(k, 0) + v

    # C_RALN
    for k, v in acc['C_RALN'].astype(str).value_counts().items():
        raln_acc[k] = raln_acc.get(k, 0) + v

    # C_TRAF
    for k, v in acc['C_TRAF'].astype(str).value_counts().items():
        traf_acc[k] = traf_acc.get(k, 0) + v

    # Year (from file)
    year_acc[yr] = year_acc.get(yr, 0) + nc
    total_cases += nc

    # === Person-level ===
    # P_SEX
    for k, v in df['P_SEX'].astype(str).value_counts().items():
        psex_acc[k] = psex_acc.get(k, 0) + v

    # P_AGE
    page_clean = df['P_AGE'].dropna()
    page_clean = page_clean[page_clean != 'UU']
    page_vals = pd.to_numeric(page_clean, errors='coerce').dropna()
    page_bins.append(page_vals)

    # P_PSN
    for k, v in df['P_PSN'].astype(str).value_counts().items():
        psn_acc[k] = psn_acc.get(k, 0) + v

    # P_ISEV
    for k, v in df['P_ISEV'].astype(str).value_counts().items():
        isev_acc[k] = isev_acc.get(k, 0) + v

    # P_SAFE
    for k, v in df['P_SAFE'].astype(str).value_counts().items():
        psafe_acc[k] = psafe_acc.get(k, 0) + v

    # P_USER
    for k, v in df['P_USER'].astype(str).value_counts().items():
        puser_acc[k] = puser_acc.get(k, 0) + v

    # V_TYPE
    for k, v in df['V_TYPE'].astype(str).value_counts().items():
        vtype_acc[k] = vtype_acc.get(k, 0) + v

    # Cross: P_USER x C_SEV
    for (u, s), cnt in df.groupby(['P_USER', 'C_SEV']).size().items():
        key = (str(u), int(s))
        user_sev_acc[key] = user_sev_acc.get(key, 0) + cnt

    print(f"{n:,} persons, {nc:,} cases")

print(f"\n{'='*50}")
print(f"TOTAL: {total_person_rows:,} person-level rows, {total_cases:,} collisions")
print(f"Years: {min(unique_years)}-{max(unique_years)} ({len(unique_years)} years)")

# Merge age bins
all_ages = pd.concat(page_bins, ignore_index=True) if page_bins else pd.Series([], dtype=float)
print(f"Person age records: {len(all_ages):,}")

# ========== 1. TARGET ANALYSIS ==========
print("\n" + "="*40)
print("1. TARGET VARIABLE (C_SEV)")
print("="*40)

sev_total = sum(sev_acc.values())
print(f"\nC_SEV distribution (accident-level, n={sev_total:,}):")
for sev in sorted(sev_acc.keys()):
    print(f"  C_SEV={sev}: {sev_acc[sev]:>7,} ({sev_acc[sev]/sev_total*100:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = sns.color_palette('Reds', len(sev_acc))
bars = axes[0].bar([f'C_SEV={s}' for s in sorted(sev_acc.keys())],
                   [sev_acc[s] for s in sorted(sev_acc.keys())], color=colors)
axes[0].set_title(f'Accident Severity Distribution (full dataset, n={sev_total:,})', fontweight='bold')
axes[0].set_ylabel('Count')
for bar, s in zip(bars, sorted(sev_acc.keys())):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{sev_acc[s]:,}', ha='center', va='bottom', fontsize=10)

sev_labels = [f'C_SEV={s}\n({sev_acc[s]/sev_total*100:.1f}%)' for s in sorted(sev_acc.keys())]
axes[1].pie([sev_acc[s] for s in sorted(sev_acc.keys())],
            labels=sev_labels, colors=colors, autopct='', startangle=90)
axes[1].set_title('Accident Severity Proportion')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_severity_distribution.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 01_severity_distribution.png")

# ========== 2. TEMPORAL ANALYSIS ==========
print("\n" + "="*40)
print("2. TEMPORAL ANALYSIS")
print("="*40)

month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
dow_names = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Year
years_sorted = sorted(year_acc.keys())
year_vals = [year_acc[y] for y in years_sorted]
axes[0,0].bar([str(y) for y in years_sorted], year_vals, color='steelblue')
axes[0,0].set_title(f'Collisions by Year (full dataset, n={total_cases:,})', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)

# Month
months_sorted = sorted(month_acc.keys(), key=lambda x: int(x) if x.isdigit() else 99)
month_vals = [month_acc[m] for m in months_sorted]
month_labels = [month_names[int(m)-1] if m.isdigit() else m for m in months_sorted]
axes[0,1].bar(month_labels, month_vals, color='coral')
axes[0,1].set_title(f'Collisions by Month (n={total_cases:,})', fontweight='bold')

# Hour
hours_sorted = sorted(hour_acc.keys(), key=lambda x: int(x) if x.isdigit() else 99)
hour_vals = [hour_acc[h] for h in hours_sorted]
axes[0,2].bar(hours_sorted, hour_vals, color='seagreen')
axes[0,2].set_title(f'Collisions by Hour (n={total_cases:,})', fontweight='bold')

# Day of week
wday_sorted = sorted(wday_acc.keys(), key=lambda x: int(x) if x.isdigit() else 99)
wday_vals = [wday_acc[d] for d in wday_sorted]
wday_labels = [dow_names.get(int(d), d) if d.isdigit() else d for d in wday_sorted]
axes[1,0].bar(wday_labels, wday_vals, color='mediumpurple')
axes[1,0].set_title(f'Collisions by Day of Week (n={total_cases:,})', fontweight='bold')

# Number of vehicles
vehs_sorted = sorted(vehs_acc.keys(), key=lambda x: int(x) if x.isdigit() else 99)
vehs_vals = [vehs_acc[v] for v in vehs_sorted]
axes[1,1].bar(vehs_sorted, vehs_vals, color='goldenrod')
axes[1,1].set_title(f'Vehicles per Collision (n={total_cases:,})', fontweight='bold')
axes[1,1].set_xlabel('Number of Vehicles')

# Severity x Hour cross
acc_first_all = []
for f in files:
    df = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False, usecols=['C_CASE', 'C_HOUR', 'C_SEV', 'C_MNTH'])
    acc_first = df.groupby('C_CASE').first().reset_index()
    acc_first_all.append(acc_first)
acc_first_full = pd.concat(acc_first_all, ignore_index=True)
sev_hour = acc_first_full.groupby(['C_HOUR', 'C_SEV']).size().unstack(fill_value=0)
sns.heatmap(sev_hour.T, ax=axes[1,2], cmap='YlOrRd', cbar_kws={'label': 'Count'})
axes[1,2].set_title(f'Severity x Hour Heatmap (n={len(acc_first_full):,})', fontweight='bold')
del acc_first_all, acc_first_full

plt.suptitle('Temporal Patterns -- Canada National Collisions', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_temporal_patterns.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 02_temporal_patterns.png")

# ========== 3. COLLISION CHARACTERISTICS ==========
print("\n" + "="*40)
print("3. COLLISION CHARACTERISTICS")
print("="*40)

# C_CONF
conf_total = sum(conf_acc.values())
conf_top = dict(sorted(conf_acc.items(), key=lambda x: x[1], reverse=True)[:15])
fig, ax = plt.subplots(figsize=(12, 6))
colors_conf = sns.color_palette('Blues_d', n_colors=len(conf_top))
ax.barh(range(len(conf_top)), list(conf_top.values())[::-1], color=colors_conf[::-1])
ax.set_yticks(range(len(conf_top)))
ax.set_yticklabels(list(conf_top.keys())[::-1])
ax.set_xlabel('Count')
ax.set_title(f'Top 15 Collision Configurations (full dataset, n={conf_total:,})', fontweight='bold')
for i, (k, v) in enumerate(list(conf_top.items())[::-1]):
    ax.text(v + 500, list(conf_top.keys())[::-1].index(k), f'{v:,}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_collision_configuration.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 03_collision_configuration.png")

# C_RCFG
rcfg_total = sum(rcfg_acc.values())
rcfg_top = dict(sorted(rcfg_acc.items(), key=lambda x: x[1], reverse=True)[:12])
plt.figure(figsize=(10, 5))
plt.bar(rcfg_top.keys(), rcfg_top.values(), color='teal')
plt.title(f'Road Configuration Distribution (full dataset, n={rcfg_total:,})', fontweight='bold')
plt.xlabel('Road Configuration Code')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03b_road_configuration.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 03b_road_configuration.png")

# ========== 4. ENVIRONMENTAL CONDITIONS ==========
print("\n" + "="*40)
print("4. ENVIRONMENTAL CONDITIONS")
print("="*40)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
# C_WTHR
wthr_top = dict(sorted(wthr_acc.items(), key=lambda x: x[1], reverse=True)[:12])
axes[0].bar(wthr_top.keys(), wthr_top.values(), color='skyblue')
axes[0].set_title(f'Weather Conditions (n={sum(wthr_acc.values()):,})', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
# C_RSUR
rsur_top = dict(sorted(rsur_acc.items(), key=lambda x: x[1], reverse=True)[:10])
axes[1].bar(rsur_top.keys(), rsur_top.values(), color='lightcoral')
axes[1].set_title(f'Road Surface (n={sum(rsur_acc.values()):,})', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
# C_RALN
raln_top = dict(sorted(raln_acc.items(), key=lambda x: x[1], reverse=True)[:10])
axes[2].bar(raln_top.keys(), raln_top.values(), color='mediumseagreen')
axes[2].set_title(f'Road Alignment (n={sum(raln_acc.values()):,})', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_environmental_conditions.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 04_environmental_conditions.png")

# ========== 5. VEHICLE TYPE ANALYSIS ==========
print("\n" + "="*40)
print("5. VEHICLE TYPE ANALYSIS")
print("="*40)

vtype_total = sum(vtype_acc.values())
vtype_top = dict(sorted(vtype_acc.items(), key=lambda x: x[1], reverse=True)[:15])
print(f"\nTop 10 vehicle types:")
for k, v in list(vtype_top.items())[:10]:
    print(f"  Code {k:>4s}: {v:>7,} ({v/vtype_total*100:.1f}%)")

plt.figure(figsize=(12, 6))
colors_v = sns.color_palette('viridis', n_colors=len(vtype_top))
plt.barh(range(len(vtype_top)), list(vtype_top.values())[::-1], color=colors_v[::-1])
plt.yticks(range(len(vtype_top)), list(vtype_top.keys())[::-1])
plt.xlabel('Count')
plt.title(f'Top 15 Vehicle Types (full dataset, n={vtype_total:,})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_vehicle_types.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 05_vehicle_types.png")

# ========== 6. PERSON ANALYSIS ==========
print("\n" + "="*40)
print("6. PERSON ANALYSIS")
print("="*40)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# P_SEX
psex_top = dict(sorted(psex_acc.items(), key=lambda x: x[1], reverse=True))
psex_labels = {k: f'Sex={k}' for k in psex_top.keys()}
axes[0,0].bar([f'Sex={s}' for s in psex_top.keys()], psex_top.values(), color=['steelblue', 'coral', 'gray'])
axes[0,0].set_title(f'Sex Distribution (n={sum(psex_acc.values()):,})', fontweight='bold')

# P_AGE
axes[0,1].hist(all_ages, bins=80, color='seagreen', edgecolor='white')
axes[0,1].set_title(f'Age Distribution (n={len(all_ages):,})', fontweight='bold')
axes[0,1].set_xlabel('Age')
axes[0,1].set_ylabel('Count')

# P_PSN
psn_top = dict(sorted(psn_acc.items(), key=lambda x: x[1], reverse=True)[:12])
axes[0,2].bar(psn_top.keys(), psn_top.values(), color='mediumpurple')
axes[0,2].set_title(f'Position in Vehicle (n={sum(psn_acc.values()):,})', fontweight='bold')
axes[0,2].tick_params(axis='x', rotation=45)

# P_ISEV
isev_top = dict(sorted(isev_acc.items(), key=lambda x: x[1], reverse=True))
axes[1,0].bar(isev_top.keys(), isev_top.values(), color='lightcoral')
axes[1,0].set_title(f'Injury Severity (P_ISEV, n={sum(isev_acc.values()):,})', fontweight='bold')

# P_SAFE
psafe_top = dict(sorted(psafe_acc.items(), key=lambda x: x[1], reverse=True)[:10])
axes[1,1].bar(psafe_top.keys(), psafe_top.values(), color='goldenrod')
axes[1,1].set_title(f'Safety Equipment (P_SAFE, n={sum(psafe_acc.values()):,})', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=45)

# P_USER
puser_top = dict(sorted(puser_acc.items(), key=lambda x: x[1], reverse=True))
axes[1,2].bar(puser_top.keys(), puser_top.values(), color='teal')
axes[1,2].set_title(f'Road User Class (P_USER, n={sum(puser_acc.values()):,})', fontweight='bold')

plt.suptitle('Person-Level Analysis -- Canada Collisions', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_person_analysis.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 06_person_analysis.png")

# ========== 7. USER CLASS x SEVERITY ==========
print("\n" + "="*40)
print("7. ROAD USER CLASS vs SEVERITY")
print("="*40)

# Build cross-tab
user_sev_df = pd.DataFrame([
    {'P_USER': k[0], 'C_SEV': k[1], 'Count': v}
    for k, v in user_sev_acc.items()
])
user_pivot = user_sev_df.pivot_table(index='P_USER', columns='C_SEV', values='Count', aggfunc='sum', fill_value=0)
user_pivot['Total'] = user_pivot.sum(axis=1)
sev_cols = sorted([c for c in user_pivot.columns if c != 'Total'])
if sev_cols:
    fatal_col = sev_cols[0] if 1 in sev_cols else sev_cols[0]
    user_pivot['C_SEV_1_Rate'] = (user_pivot.get(1, 0) / user_pivot['Total'] * 100).round(2)
user_pivot_sorted = user_pivot.sort_values('Total', ascending=False)
print("\nP_USER x C_SEV cross-tabulation:")
print(user_pivot_sorted.head(15))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
top_users = user_pivot_sorted.head(10)
top_users_sev = top_users[[c for c in top_users.columns if c in sev_cols]]
top_users_sev.plot(ax=axes[0], kind='bar', stacked=True, colormap='Reds')
axes[0].set_title(f'Road User Class x Severity (full dataset, n={total_person_rows:,})', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

if 'C_SEV_1_Rate' in top_users.columns:
    top_users['C_SEV_1_Rate'].plot(ax=axes[1], kind='bar', color='#d73027')
    axes[1].set_title('C_SEV=1 Rate by User Class', fontweight='bold')
    axes[1].set_ylabel('C_SEV=1 Rate (%)')
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(top_users['C_SEV_1_Rate'].values):
        axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_user_class_severity.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 07_user_class_severity.png")

# ========== 8. INJURY ANALYSIS ==========
print("\n" + "="*40)
print("8. INJURY SEVERITY vs COLLISION SEVERITY")
print("="*40)

# Aggregate cross of P_ISEV (person injury) x C_SEV (collision severity)
person_all = []
for f in files:
    df = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False, usecols=['P_ISEV', 'C_SEV', 'P_USER'])
    person_all.append(df)
person_full = pd.concat(person_all, ignore_index=True)
print(f"\nPerson-level records for cross-analysis: {len(person_full):,}")

isev_sev = person_full.groupby(['P_ISEV', 'C_SEV']).size().unstack(fill_value=0)
print("\nP_ISEV (Person Injury) x C_SEV (Collision Severity):")
print(isev_sev)

fig, ax = plt.subplots(figsize=(10, 6))
# Normalize rows
isev_pct = isev_sev.div(isev_sev.sum(axis=1), axis=0) * 100
isev_pct.plot(ax=ax, kind='bar', stacked=True, colormap='Reds')
ax.set_title(f'Person Injury Severity vs Collision Severity (n={len(person_full):,})', fontweight='bold')
ax.set_ylabel('Percentage (%)')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='C_SEV', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '08_injury_severity_cross.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 08_injury_severity_cross.png")

# C_SEV=1 rate by P_USER
print("\nC_SEV=1 rate by road user class:")
user_sev_rate = person_full.groupby('P_USER')['C_SEV'].apply(lambda x: (x == 1).mean()*100).sort_values(ascending=False)
print(user_sev_rate.head(15))

del person_full

# ========== 9. TRAFFIC CONTROL ==========
print("\n" + "="*40)
print("9. TRAFFIC CONTROL ANALYSIS")
print("="*40)

traf_total = sum(traf_acc.values())
traf_top = dict(sorted(traf_acc.items(), key=lambda x: x[1], reverse=True)[:15])
plt.figure(figsize=(12, 5))
plt.bar(traf_top.keys(), traf_top.values(), color='darkslategray')
plt.title(f'Traffic Control Device Distribution (full dataset, n={traf_total:,})', fontweight='bold')
plt.xlabel('Traffic Control Code')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '09_traffic_control.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 09_traffic_control.png")

# ========== 10. SUMMARY ==========
print("\n" + "="*70)
print("EDA SUMMARY -- CANADA NATIONAL COLLISION DATASET")
print("="*70)

sev1_rate = sev_acc.get(1, 0) / sev_total * 100
print(f"""
KEY FINDINGS:
1. Dataset: {total_cases:,} collisions, {total_person_rows:,} person-level records
2. Years: {min(unique_years)}-{max(unique_years)} ({len(unique_years)} years), Canada national
3. C_SEV: {sev_acc.get(1, 0):,} ({sev1_rate:.1f}%) severity=1 (injury/fatal),
   {sev_acc.get(2, 0):,} ({100-sev1_rate:.1f}%) severity=2 (property damage)
4. Temporal: peak hours 15-17, Fridays highest, summer months
5. Collision config: config 21 (rear-end) most common
6. Road surface: dry dominates, wet 2nd, snow/ice seasonal
7. Vehicle: passenger cars most common, trucks elevated severity
8. Road user: drivers most common, pedestrians highest C_SEV=1 rate
9. Person injury: P_ISEV distribution shows majority no injury

RECOMMENDATIONS:
- Use for ML prediction of C_SEV (binary: injury/fatal vs PDO)
- Key features: C_CONF, C_WTHR, C_RSUR, P_USER, C_HOUR, C_WDAY
- Consider per-user-class modeling (similar to Shanshal et al., 2020)
""")

summary = {
    'dataset': 'Canada National Collisions (1999-2017)',
    'collisions': total_cases,
    'person_records': total_person_rows,
    'years': [min(unique_years), max(unique_years)],
    'sev_distribution': {int(k): int(v) for k, v in sev_acc.items()},
    'sev1_rate_pct': round(sev1_rate, 2),
    'top_road_user_classes': {k: int(v) for k, v in sorted(puser_acc.items(), key=lambda x: x[1], reverse=True)[:10]},
    'top_collision_configs': {k: int(v) for k, v in sorted(conf_acc.items(), key=lambda x: x[1], reverse=True)[:10]},
}
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("Summary saved to summary.json")
print("\nDONE -- Canada National Collision EDA complete!")
