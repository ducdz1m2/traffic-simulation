import json

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s]})

def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s]})

md("""# Toronto Collision Severity Prediction
## Boddepalli et al. (2026) - Paper Reproduction

**Pipeline per paper:**
1. Binary: Traffic Collisions - Injury vs Non-Injury
2. Multiclass: KSI per-person - INJURY 5 levels
3. Temporal split: 2006-2022 train, 2023-2024 test
4. Model params: RF (n=100, d=10), XGB (lr=0.1, d=6, n=150)
5. Spatial grid 250m, time features
""")

code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import os, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi'] = 150
OB = 'outputs/toronto_ml/binary'; OM = 'outputs/toronto_ml/multiclass'
for d in [OB, OM]: os.makedirs(d, exist_ok=True)

sev_map = {'Unknown': 0, 'Minimal': 1, 'Minor': 2, 'Major': 3, 'Fatal': 4}
rev_sev = {v:k for k,v in sev_map.items()}; tnames = [rev_sev[i] for i in range(5)]
paper_auc = {'Fatal': 0.96, 'Major': 0.88, 'Minimal': 0.96, 'Minor': 0.95, 'Unknown': 0.87}
RF_N=100; RF_D=10; XGB_LR=0.1; XGB_D=6; XGB_N=150
print('Ready')
""")

# === BINARY: Traffic ===
md("## Part A: Binary Classification (Traffic Collisions)")

code("""traf = pd.read_csv(r'data\\\\toronto-dataset\\\\Traffic_Collisions_Toronto_data_converted.csv', low_memory=False)
traf = traf[traf['Injury_Collisions'] != '<Null>'].copy()
traf['target'] = (traf['Injury_Collisions'] == 'YES').astype(int)
print(f'Traffic rows: {len(traf)}, Injury rate: {traf.target.mean()*100:.2f}%')

dp = pd.to_datetime(traf['OccurrenceDate'], utc=True)
traf['Year'] = dp.dt.year.fillna(0).astype(int)
traf['Month'] = dp.dt.month.fillna(1).astype(int)
traf['DayOfWeek'] = dp.dt.dayofweek.fillna(0).astype(int)
traf['is_weekend'] = (traf['DayOfWeek'] >= 5).astype(int)
traf['Season'] = traf['Month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}).fillna(0).astype(int)
traf['is_dst'] = (traf['Hour'] == 4).astype(int)

lat_c, lon_c = 0.00225, 0.003125
traf['g_i'] = ((traf['Latitude'] - traf['Latitude'].min())/lon_c).fillna(0).astype(int)
traf['g_j'] = ((traf['Longitude'] - traf['Longitude'].min())/lat_c).fillna(0).astype(int)

for col in ['Division', 'Month', 'Day_of_Week']:
    traf[col+'_e'] = LabelEncoder().fit_transform(traf[col].astype(str))
hood_cnts = traf['Neighbourhood'].value_counts()
traf['Neighbourhood_freq'] = traf['Neighbourhood'].map(hood_cnts) / len(traf)
traf['Fatalities'] = traf['Fatalities'].fillna(0)

feat_bin = ['Year','Month','DayOfWeek','is_weekend','Season','is_dst',
            'g_i','g_j','Latitude','Longitude','Fatalities','Neighbourhood_freq',
            'Division_e','Month_e','Day_of_Week_e']
feat_bin = [c for c in feat_bin if c in traf.columns]

tr_b = traf[~traf['Year'].isin([2023, 2024])].copy()
te_b = traf[traf['Year'].isin([2023, 2024])].copy()
Xtr_b = tr_b[feat_bin].fillna(0).values; ytr_b = tr_b['target'].values
Xte_b = te_b[feat_bin].fillna(0).values; yte_b = te_b['target'].values
ss_b = StandardScaler(); Xtr_bs = ss_b.fit_transform(Xtr_b); Xte_bs = ss_b.transform(Xte_b)

rf_b = RandomForestClassifier(n_estimators=RF_N, max_depth=RF_D, class_weight='balanced', random_state=42, n_jobs=-1)
rf_b.fit(Xtr_bs, ytr_b); pb_b = rf_b.predict_proba(Xte_bs)[:, 1]
print(f'RF: Acc={accuracy_score(yte_b, pb_b>=0.5):.4f} AUC={roc_auc_score(yte_b, pb_b):.4f}')

xgb_b = XGBClassifier(learning_rate=XGB_LR, max_depth=XGB_D, n_estimators=XGB_N,
                      random_state=42, eval_metric='logloss',
                      scale_pos_weight=(ytr_b==0).sum()/max((ytr_b==1).sum(),1))
xgb_b.fit(Xtr_bs, ytr_b); pb_xb = xgb_b.predict_proba(Xte_bs)[:, 1]
print(f'XGB: Acc={accuracy_score(yte_b, pb_xb>=0.5):.4f} AUC={roc_auc_score(yte_b, pb_xb):.4f}')
""")

# === MULTICLASS: KSI per-person (paper baseline) ===
md("## Part B: Multiclass - INJURY 5 levels (KSI per-person)")

code("""ksi = pd.read_csv(r'data\\\\toronto-dataset\\\\KSI_converted.csv', low_memory=False)

# Drop irrelevant columns per paper
drop_cols = ['OFFSET','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND',
             'ObjectId','INDEX_','STREET1','STREET2','ACCLOC','LOCCOORD',
             'PASSENGER','TRSN_CITY_VEH','EMERG_VEH','ALCOHOL','DISABILITY','FATAL_NO']
drop_cols = [c for c in drop_cols if c in ksi.columns]
ksi.drop(columns=drop_cols, inplace=True)

# Per-person, drop null INJURY (paper approach)
df = ksi.dropna(subset=['INJURY']).copy()
df['target'] = df['INJURY'].map(sev_map).astype(int)
print(f'Multi rows: {len(df)}')
for name in tnames:
    cnt = (df.target == sev_map[name]).sum()
    print(f'  {name}: {cnt} ({cnt/len(df)*100:.1f}%)')

# Feature engineering
dp = pd.to_datetime(df['DATE'], errors='coerce')
df['Year'] = dp.dt.year.fillna(0).astype(int)
df['Month'] = dp.dt.month.fillna(1).astype(int)
df['DayOfWeek'] = dp.dt.day_name()
df['is_weekend'] = (dp.dt.dayofweek >= 5).astype(int)
df['Season'] = df['Month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}).fillna(0).astype(int)

light_map = {'Daylight':1,'Dawn':1.5,'Dusk':1.5,'Dark Artificial':3,'Dark':4}
rdsfc_map = {'Dry':1,'Wet':2,'Loose Snow':3,'Slush':3.5,'Packed Snow':4,'Ice':5}
vis_map = {'Clear':1,'Cloudy':2,'Rain':3,'Fog':3,'Mist':3,'Snow':4,'Freezing Rain':5,'Drifting Snow':5}
for col, m in [('LIGHT',light_map),('RDSFCOND',rdsfc_map),('VISIBILITY',vis_map)]:
    if col in df.columns: df[col+'_o'] = df[col].map(m).fillna(0)

lat_c, lon_c = 0.00225, 0.003125
df['g_i'] = ((df['LATITUDE'] - df['LATITUDE'].min())/lat_c).fillna(0).astype(int)
df['g_j'] = ((df['LONGITUDE'] - df['LONGITUDE'].min())/lon_c).fillna(0).astype(int)
if 'INVAGE' in df.columns:
    df['INVAGE'] = pd.to_numeric(df['INVAGE'], errors='coerce').fillna(df['INVAGE'].median())

drop_feat = ['INJURY','ACCLASS','ACCNUM','DATE','YEAR']
drop_feat = [c for c in drop_feat if c in df.columns]
df.drop(columns=drop_feat, inplace=True)

for c in df.select_dtypes(include=['object']).columns:
    m = df[c].mode(); df[c] = df[c].fillna(m[0] if len(m)>0 else 'Unknown')
for c in df.select_dtypes(include=['int64','float64']).columns:
    df[c] = df[c].fillna(df[c].median())

cat_f = [c for c in ['ROAD_CLASS','DISTRICT','TRAFFCTL','IMPACTYPE','INVTYPE','VEHTYPE',
         'INITDIR','MANOEUVER','DRIVACT','DRIVCOND','DayOfWeek','Season'] if c in df.columns]
bool_f = [c for c in ['PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','SPEEDING','AG_DRIV','REDLIGHT'] if c in df.columns]
for c in cat_f: df[c+'_e'] = LabelEncoder().fit_transform(df[c].astype(str))
for c in bool_f: df[c+'_b'] = (df[c].astype(str).str.upper()=='YES').astype(int)

f_cols = ([c+'_e' for c in cat_f] + [c+'_b' for c in bool_f] +
          [c+'_o' for c in ['LIGHT','RDSFCOND','VISIBILITY'] if c+'_o' in df.columns] +
          ['INVAGE','Month','Year','is_weekend','g_i','g_j','LATITUDE','LONGITUDE','HOUR'])
f_cols = [c for c in f_cols if c in df.columns]
print(f'Features: {len(f_cols)}')

tr = df[~df['Year'].isin([2023, 2024])].copy()
te = df[df['Year'].isin([2023, 2024])].copy()
print(f'Train: {len(tr)}, Test: {len(te)}')

Xtr = tr[f_cols].fillna(0).values; ytr = tr['target'].values
Xte = te[f_cols].fillna(0).values; yte = te['target'].values
ss = StandardScaler(); Xtr_s = ss.fit_transform(Xtr); Xte_s = ss.transform(Xte)
print(f'Train: {np.bincount(ytr)}')
print(f'Test:  {np.bincount(yte)}')
""")

md("### Models (paper params)")

code("""print('RF (n=100, d=10)...')
rf = RandomForestClassifier(n_estimators=RF_N, max_depth=RF_D, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(Xtr_s, ytr); ypb_rf = rf.predict_proba(Xte_s); yp_rf = rf.predict(Xte_s)
print(f'Acc={accuracy_score(yte,yp_rf):.4f} MF1={f1_score(yte,yp_rf,average="macro"):.4f}')

print('XGB (lr=0.1, d=6, n=150)...')
xgb = XGBClassifier(learning_rate=XGB_LR, max_depth=XGB_D, n_estimators=XGB_N,
                    random_state=42, eval_metric='mlogloss', objective='multi:softprob', num_class=5)
xgb.fit(Xtr_s, ytr); ypb_x = xgb.predict_proba(Xte_s); yp_x = xgb.predict(Xte_s)
print(f'Acc={accuracy_score(yte,yp_x):.4f} MF1={f1_score(yte,yp_x,average="macro"):.4f}')
""")

md("### Per-class ROC AUC vs Paper")

code("""fig, axes = plt.subplots(2, 3, figsize=(18, 12)); axes = axes.flatten()
for i, name in enumerate(tnames):
    yb = (yte == i).astype(int)
    if yb.sum() == 0:
        axes[i].text(.5,.5,f'{name}\\n(no data)',ha='center',va='center',transform=axes[i].transAxes)
        axes[i].set_title(name); continue
    for n, pb, c in [('RF',ypb_rf,'steelblue'),('XGB',ypb_x,'coral')]:
        prob = pb[:, i] if pb.shape[1] > i else np.zeros(len(yte))
        auc = roc_auc_score(yb, prob)
        fpr, tpr, _ = roc_curve(yb, prob)
        axes[i].plot(fpr, tpr, label=f'{n} AUC={auc:.3f}', lw=2, color=c)
    axes[i].plot([0,1],[0,1],'k--',alpha=.3)
    axes[i].axhline(paper_auc.get(name,0), color='gray', ls=':', label=f'Paper={paper_auc.get(name,0):.2f}')
    axes[i].legend(); axes[i].set_title(f'ROC - {name}')
for j in range(5,6): axes[j].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OM}/roc_baseline.png',dpi=150); plt.show()

print(f'\\n{"Class":<12} {"Paper":<8} {"RF AUC":<20} {"XGB AUC":<20}')
print('-'*60)
for i, name in enumerate(tnames):
    yb = (yte == i).astype(int)
    rf_a = roc_auc_score(yb, ypb_rf[:, i]) if yb.sum()>0 else 0
    x_a = roc_auc_score(yb, ypb_x[:, i]) if yb.sum()>0 else 0
    pv = paper_auc.get(name, 0)
    print(f'{name:<12} {pv:<8.2f} {rf_a:.4f} (gap={rf_a-pv:+.4f})  {x_a:.4f} (gap={x_a-pv:+.4f})')
""")

# === SUMMARY ===
md("## Summary")

code("""print('Paper Reproduction Complete!')
print(f'Outputs: {OB}/, {OM}/')
""")

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.3"}
    },
    "cells": cells
}

path = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-02\code\toronto_ml_severity.ipynb"
with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f'Written: {path}, Cells: {len(cells)}')
