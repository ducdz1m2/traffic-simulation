import json

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s]})

def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s]})

md("""# Toronto Collision Severity Prediction - IMPROVED
## Boddepalli et al. (2026) - Enhanced with imputation + augmented features

**Improvements over paper:**
1. Impute null INJURY as Unknown → 19,754 training rows
2. Augment per-person with accident-level features (num_persons, age stats, unique types)
3. Temporal split: 2006-2022 train, 2023-2024 test
""")

code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import os, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi'] = 150
OM = 'outputs/toronto_ml/multiclass_improved'; os.makedirs(OM, exist_ok=True)
sev_map = {'Unknown':0,'Minimal':1,'Minor':2,'Major':3,'Fatal':4}
rev_sev = {v:k for k,v in sev_map.items()}; tnames = [rev_sev[i] for i in range(5)]
paper_auc = {'Fatal':0.96,'Major':0.88,'Minimal':0.96,'Minor':0.95,'Unknown':0.87}
print('Ready')
""")

code("""ksi = pd.read_csv(r'data\\\\toronto-dataset\\\\KSI_converted.csv', low_memory=False)

# Build accident-level aggregation
event_cols = ['ROAD_CLASS','DISTRICT','LATITUDE','LONGITUDE','TRAFFCTL','VISIBILITY',
              'LIGHT','RDSFCOND','IMPACTYPE','YEAR','DATE']
acc_list = []
for accnum, grp in ksi.groupby('ACCNUM', sort=False):
    feat = {'ACCNUM': accnum}
    for c in event_cols:
        vals = grp[c].dropna()
        feat[c] = vals.iloc[0] if len(vals) > 0 else (np.nan if c in ['LATITUDE','LONGITUDE'] else '<Null>')
    feat['num_persons'] = len(grp)
    ages = pd.to_numeric(grp['INVAGE'],errors='coerce').dropna()
    feat['min_age_acc'] = ages.min() if len(ages) > 0 else np.nan
    feat['max_age_acc'] = ages.max() if len(ages) > 0 else np.nan
    feat['avg_age_acc'] = ages.mean() if len(ages) > 0 else np.nan
    for col in ['INVTYPE','VEHTYPE','INITDIR','MANOEUVER','DRIVACT','DRIVCOND']:
        feat['n_unique_'+col.lower()] = grp[col].dropna().nunique()
    for col in ['PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','SPEEDING','AG_DRIV','REDLIGHT']:
        feat[col] = 'Yes' if (grp[col]=='Yes').any() else '<Null>'
    acc_list.append(feat)
acc_df = pd.DataFrame(acc_list)

# Per-person, impute null INJURY
person_cols = ['ACCNUM','INJURY','INVAGE','INVTYPE','VEHTYPE','INITDIR','MANOEUVER',
               'DRIVACT','DRIVCOND','HOUR','YEAR']
person_cols = [c for c in person_cols if c in ksi.columns]
persons = ksi[person_cols].copy()
persons['INJURY'] = persons['INJURY'].fillna('Unknown')
persons['target'] = persons['INJURY'].map(sev_map).astype(int)
df = persons.merge(acc_df, on='ACCNUM', how='left')
print(f'Total rows: {len(df)}')
for name in tnames:
    cnt = (df.target == sev_map[name]).sum()
    print(f'  {name}: {cnt} ({cnt/len(df)*100:.1f}%)')
""")

code("""# Feature engineering
dp = pd.to_datetime(df['DATE'], errors='coerce')
df['Year'] = dp.dt.year.fillna(0).astype(int)
df['Month'] = dp.dt.month.fillna(1).astype(int)
df['DayOfWeek'] = dp.dt.day_name()
df['is_weekend'] = (dp.dt.dayofweek >= 5).astype(int)
df['Season'] = df['Month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}).fillna(0).astype(int)
df['is_night'] = df['HOUR'].isin([22,23,0,1,2,3,4,5]).astype(int)

light_map={'Daylight':1,'Dawn':1.5,'Dusk':1.5,'Dark Artificial':3,'Dark':4}
rdsfc_map={'Dry':1,'Wet':2,'Loose Snow':3,'Slush':3.5,'Packed Snow':4,'Ice':5}
vis_map={'Clear':1,'Cloudy':2,'Rain':3,'Fog':3,'Mist':3,'Snow':4,'Freezing Rain':5,'Drifting Snow':5}
for col,m in [('LIGHT',light_map),('RDSFCOND',rdsfc_map),('VISIBILITY',vis_map)]:
    if col in df.columns: df[col+'_o'] = df[col].map(m).fillna(0)

lat_c,lon_c=0.00225,0.003125
df['g_i']=((df['LATITUDE']-df['LATITUDE'].min())/lat_c).fillna(0).astype(int)
df['g_j']=((df['LONGITUDE']-df['LONGITUDE'].min())/lon_c).fillna(0).astype(int)
if 'INVAGE' in df.columns:
    df['INVAGE'] = pd.to_numeric(df['INVAGE'],errors='coerce').fillna(df['INVAGE'].median())

for c in ['INJURY','ACCNUM','DATE']:
    if c in df.columns: df.drop(columns=[c],inplace=True)
for c in df.select_dtypes(include=['object']).columns:
    m=df[c].mode(); df[c]=df[c].fillna(m[0] if len(m)>0 else 'Unknown')
for c in df.select_dtypes(include=['int64','float64']).columns:
    df[c]=df[c].fillna(df[c].median())

cat_f=[c for c in ['ROAD_CLASS','DISTRICT','TRAFFCTL','IMPACTYPE','INVTYPE','VEHTYPE',
         'INITDIR','MANOEUVER','DRIVACT','DRIVCOND','DayOfWeek','Season'] if c in df.columns]
bool_f=[c for c in ['PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','SPEEDING','AG_DRIV','REDLIGHT'] if c in df.columns]
for c in cat_f: df[c+'_e']=LabelEncoder().fit_transform(df[c].astype(str))
for c in bool_f: df[c+'_b']=(df[c].astype(str).str.upper()=='YES').astype(int)

acc_feats=['num_persons','min_age_acc','max_age_acc','avg_age_acc',
           'n_unique_invtype','n_unique_vehtype','n_unique_initdir',
           'n_unique_manoeuver','n_unique_drivact','n_unique_drivcond']
acc_feats=[c for c in acc_feats if c in df.columns]

f_cols=([c+'_e' for c in cat_f]+[c+'_b' for c in bool_f]+
        [c+'_o' for c in ['LIGHT','RDSFCOND','VISIBILITY'] if c+'_o' in df.columns]+
        ['INVAGE','Month','Year','is_weekend','is_night','g_i','g_j','LATITUDE','LONGITUDE','HOUR']+acc_feats)
f_cols=[c for c in f_cols if c in df.columns]
print(f'Features: {len(f_cols)}')

tr=df[~df['Year'].isin([2023,2024])].copy(); te=df[df['Year'].isin([2023,2024])].copy()
print(f'Train: {len(tr)}, Test: {len(te)}')
Xtr=tr[f_cols].fillna(0).values; ytr=tr['target'].values
Xte=te[f_cols].fillna(0).values; yte=te['target'].values
ss=StandardScaler(); Xtr_s=ss.fit_transform(Xtr); Xte_s=ss.transform(Xte)
print(f'Train: {np.bincount(ytr)}'); print(f'Test:  {np.bincount(yte)}')
""")

code("""print('RF (n=100, d=10)...')
rf=RandomForestClassifier(n_estimators=100,max_depth=10,class_weight='balanced',random_state=42,n_jobs=-1)
rf.fit(Xtr_s,ytr); ypb_rf=rf.predict_proba(Xte_s); yp_rf=rf.predict(Xte_s)
print(f'Acc={accuracy_score(yte,yp_rf):.4f} MF1={f1_score(yte,yp_rf,average="macro"):.4f}')

print('XGB (lr=0.1, d=6, n=150)...')
xgb=XGBClassifier(learning_rate=0.1,max_depth=6,n_estimators=150,random_state=42,
                  eval_metric='mlogloss',objective='multi:softprob',num_class=5)
xgb.fit(Xtr_s,ytr); ypb_x=xgb.predict_proba(Xte_s); yp_x=xgb.predict(Xte_s)
print(f'Acc={accuracy_score(yte,yp_x):.4f} MF1={f1_score(yte,yp_x,average="macro"):.4f}')

print('XGB+SMOTE...')
sm=SMOTE(random_state=42)
Xtr_sm,ytr_sm=sm.fit_resample(Xtr_s,ytr)
xgb_sm=XGBClassifier(learning_rate=0.1,max_depth=6,n_estimators=150,random_state=42,
                     eval_metric='mlogloss',objective='multi:softprob',num_class=5)
xgb_sm.fit(Xtr_sm,ytr_sm); ypb_xsm=xgb_sm.predict_proba(Xte_s)
print(f'Acc={accuracy_score(yte,xgb_sm.predict(Xte_s)):.4f} MF1={f1_score(yte,xgb_sm.predict(Xte_s),average="macro"):.4f}')
""")

code("""fig,axes=plt.subplots(2,3,figsize=(18,12)); axes=axes.flatten()
colors=['steelblue','coral','forestgreen']
models_data={'RF':ypb_rf,'XGB':ypb_x,'XGB+SMOTE':ypb_xsm}
for i,name in enumerate(tnames):
    yb=(yte==i).astype(int)
    if yb.sum()==0:
        axes[i].text(.5,.5,f'{name}\\n(no data)',ha='center',va='center',transform=axes[i].transAxes)
        axes[i].set_title(name); continue
    for idx,(mn,pb) in enumerate(models_data.items()):
        prob=pb[:,i] if pb.shape[1]>i else np.zeros(len(yte))
        auc=roc_auc_score(yb,prob)
        fpr,tpr,_=roc_curve(yb,prob)
        axes[i].plot(fpr,tpr,label=f'{mn} AUC={auc:.3f}',lw=2,color=colors[idx])
    axes[i].plot([0,1],[0,1],'k--',alpha=.3)
    axes[i].axhline(paper_auc.get(name,0),color='gray',ls=':',label=f'Paper={paper_auc.get(name,0):.2f}')
    axes[i].legend(); axes[i].set_title(f'ROC - {name}')
for j in range(5,6): axes[j].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OM}/roc.png',dpi=150); plt.show()

print(f'\\n{"Class":<12} {"Paper":<8} {"Best AUC":<20} {"Model":<15}')
print('-'*55)
for name in tnames:
    best_auc,best_model=0,''
    for mn,pb in models_data.items():
        i=sev_map[name]
        yb=(yte==i).astype(int)
        if yb.sum()==0: continue
        prob=pb[:,i] if pb.shape[1]>i else np.zeros(len(yte))
        auc=roc_auc_score(yb,prob)
        if auc>best_auc: best_auc,best_model=auc,mn
    gap=best_auc-paper_auc.get(name,0)
    print(f'{name:<12} {paper_auc.get(name,0):<8.2f} {best_auc:.4f} (gap={gap:+.4f})  {best_model:<15}')
""")

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.3"}
    },
    "cells": cells
}

path = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-02\code\toronto_ml_severity_improved.ipynb"
with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f'Written: {path}, Cells: {len(cells)}')
