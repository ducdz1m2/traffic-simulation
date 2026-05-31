import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import os, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, ConfusionMatrixDisplay, precision_score, recall_score, precision_recall_curve)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

sns.set_theme(style='whitegrid'); plt.rcParams['figure.dpi'] = 150
OB = 'outputs/toronto_ml/binary'
OM = 'outputs/toronto_ml/multiclass'
for d in [OB, OM]: os.makedirs(d, exist_ok=True)

sev_map = {'Unknown': 0, 'Minimal': 1, 'Minor': 2, 'Major': 3, 'Fatal': 4}
rev_sev = {v:k for k,v in sev_map.items()}
tnames = [rev_sev[i] for i in range(5)]
print('Ready')


print('='*60)
print('BINARY: Injury vs Non-Injury')
print('='*60)

df = pd.read_csv(r'data\\toronto-dataset\\Traffic_processed.csv', low_memory=False)
print(f'Shape: {df.shape}')
print(f'Target: {df.target_bin.value_counts().to_dict()}')

# Feature columns
feat = [c for c in df.columns if c not in ['target_bin', 'Year']]

# Temporal split
tr = df[~df['Year'].isin([2023, 2024])].copy()
te = df[df['Year'].isin([2023, 2024])].copy()
print(f'Train: {len(tr)} ({len(tr)/len(df)*100:.0f}%), Test: {len(te)} ({len(te)/len(df)*100:.0f}%)')

Xtr = tr[feat].fillna(0); ytr = tr['target_bin']
Xte = te[feat].fillna(0); yte = te['target_bin']

ss = StandardScaler()
Xtr_s = ss.fit_transform(Xtr); Xte_s = ss.transform(Xte)
print(f'Features: {len(feat)}, Train+rate: {ytr.mean()*100:.2f}%, Test+rate: {yte.mean()*100:.2f}%')


print('\\n========== RF ==========')
rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(Xtr_s, ytr)
pb = rf.predict_proba(Xte_s)[:, 1]
p50 = (pb >= 0.5).astype(int)

prec, rec, thr = precision_recall_curve(yte, pb)
f1s = 2*prec*rec/(prec+rec+1e-12)
best_t = thr[np.argmax(f1s[:-1])]
p_opt = (pb >= best_t).astype(int)

print(f'Default (0.5):  Acc={accuracy_score(yte,p50):.4f} F1={f1_score(yte,p50):.4f} AUC={roc_auc_score(yte,pb):.4f}')
print(f'Optimal ({best_t:.3f}): F1={np.max(f1s):.4f}')
print(classification_report(yte, p_opt, target_names=['Non-Injury','Injury']))

fig,ax=plt.subplots(figsize=(8,6))
fpr, tpr, _ = roc_curve(yte, pb)
auc = roc_auc_score(yte, pb)
ax.plot(fpr, tpr, lw=2, label=f'RF AUC={auc:.3f}')
ax.plot([0,1],[0,1],'k--',alpha=.5)
ax.legend(); ax.set_title('Binary ROC - RF'); plt.tight_layout()
plt.savefig(f'{OB}/rf_roc.png', dpi=150); plt.show()


print('\\n========== XGBoost ==========')
cl_weight = (ytr==0).sum() / max((ytr==1).sum(), 1)
xgb = XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=200, random_state=42, eval_metric='logloss', scale_pos_weight=cl_weight)
xgb.fit(Xtr_s, ytr)
pb_x = xgb.predict_proba(Xte_s)[:, 1]
p50_x = (pb_x >= 0.5).astype(int)

prec, rec, thr = precision_recall_curve(yte, pb_x)
f1s_x = 2*prec*rec/(prec+rec+1e-12)
best_t_x = thr[np.argmax(f1s_x[:-1])]
p_opt_x = (pb_x >= best_t_x).astype(int)

print(f'Default (0.5): Acc={accuracy_score(yte,p50_x):.4f} F1={f1_score(yte,p50_x):.4f} AUC={roc_auc_score(yte,pb_x):.4f}')
print(f'Optimal ({best_t_x:.3f}): F1={np.max(f1s_x):.4f}')
print(classification_report(yte, p_opt_x, target_names=['Non-Injury','Injury']))


print('\\n========== RF + SMOTE + GridSearch ==========')
sm = SMOTE(random_state=42)
Xtr_sm, ytr_sm = sm.fit_resample(Xtr_s, ytr)
print(f'Before SMOTE: {np.bincount(ytr.astype(int))}')
print(f'After SMOTE:  {np.bincount(ytr_sm.astype(int))}')

gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
                  param_grid={'n_estimators':[100,200],'max_depth':[8,12,16]}, cv=3, scoring='f1_macro')
gs.fit(Xtr_sm, ytr_sm)
print(f'Best: {gs.best_params_}, CV F1: {gs.best_score_:.4f}')

pb_gs = gs.predict_proba(Xte_s)[:, 1]
p50_gs = (pb_gs >= 0.5).astype(int)
print(f'Test: Acc={accuracy_score(yte,p50_gs):.4f} F1={f1_score(yte,p50_gs):.4f} AUC={roc_auc_score(yte,pb_gs):.4f}')
print(classification_report(yte, p50_gs, target_names=['Non-Injury','Injury']))


fig, ax = plt.subplots(figsize=(10, 8))
for n, pb, c in [('RF', pb, 'steelblue'), ('XGB', pb_x, 'coral'), ('RF+SMOTE+GS', pb_gs, 'forestgreen')]:
    fpr, tpr, _ = roc_curve(yte, pb)
    auc = roc_auc_score(yte, pb)
    ax.plot(fpr, tpr, label=f'{n} AUC={auc:.3f}', lw=2, color=c)
ax.plot([0,1],[0,1],'k--',alpha=.5)
ax.legend(); ax.set_title('Binary ROC Comparison - Injury vs Non-Injury')
plt.tight_layout(); plt.savefig(f'{OB}/roc_comparison.png', dpi=150); plt.show()

print('\\nBinary Summary:')
print(f'{"Model":<20} {"Acc":<8} {"F1":<8} {"AUC":<8}')
print('-'*44)
for n, pb, p50 in [('RF', pb, p50), ('XGB', pb_x, p50_x), ('RF+SMOTE+GS', pb_gs, p50_gs)]:
    print(f'{n:<20} {accuracy_score(yte,p50):<8.4f} {f1_score(yte,p50):<8.4f} {roc_auc_score(yte,pb):<8.4f}')


print('='*60)
print('MULTICLASS ORDINAL: INJURY 5 levels (per-accident)')
print('='*60)

ksi = pd.read_csv(r'data\\toronto-dataset\\KSI_converted.csv', low_memory=False)
print(f'Per-person: {len(ksi)}, ACCNUM: {ksi.ACCNUM.nunique()}')

# Drop irrelevant cols
drop_irr = ['OFFSET','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND',
            'ObjectId','INDEX_','STREET1','STREET2','X','Y','ACCLOC','LOCCOORD',
            'TIME','HOUR','HOOD_ID','NEIGHBOURHOOD','POLICE_DIVISION','WARDNUM','DIVISION',
            'PASSENGER','TRSN_CITY_VEH','EMERG_VEH','ALCOHOL','DISABILITY']
drop_irr = [c for c in drop_irr if c in ksi.columns]
ksi.drop(columns=drop_irr, inplace=True)
print(f'After drop: {ksi.shape}')

# Per-accident aggregation
event_cols = ['ACCLASS','DATE','ROAD_CLASS','DISTRICT','LATITUDE','LONGITUDE',
              'TRAFFCTL','VISIBILITY','LIGHT','RDSFCOND','IMPACTYPE','YEAR']

aggregated = []
for accnum, grp in ksi.groupby('ACCNUM', sort=False):
    row = {'ACCNUM': accnum}
    for c in event_cols:
        vals = grp[c].dropna()
        row[c] = vals.iloc[0] if len(vals) > 0 else '<Null>'

    row['num_persons'] = len(grp)

    ages = pd.to_numeric(grp['INVAGE'], errors='coerce').dropna()
    row['min_age'] = ages.min() if len(ages) > 0 else '<Null>'
    row['max_age'] = ages.max() if len(ages) > 0 else '<Null>'
    row['num_unique_ages'] = ages.nunique() if len(ages) > 0 else 0

    for col in ['INVTYPE','VEHTYPE','INITDIR','MANOEUVER','DRIVACT','DRIVCOND']:
        vals = grp[col].dropna().unique()
        row[col + '_list'] = ','.join(vals) if len(vals) > 0 else '<Null>'
        row['num_unique_' + col.lower()] = len(vals)

    for col in ['PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','SPEEDING','AG_DRIV','REDLIGHT']:
        row[col] = 'Yes' if (grp[col] == 'Yes').any() else '<Null>'

    injuries = grp['INJURY'].map(sev_map).dropna()
    row['target'] = int(injuries.max()) if len(injuries) > 0 else 0
    aggregated.append(row)

df = pd.DataFrame(aggregated)
print(f'Collisions: {len(df)}')
for i, name in enumerate(tnames):
    cnt = (df.target == i).sum()
    print(f'  {name}: {cnt} ({cnt/len(df)*100:.1f}%)')


# Parse datetime
df['dp'] = pd.to_datetime(df['DATE'], errors='coerce')
df['Year'] = df['dp'].dt.year.fillna(0).astype(int)
df['Month'] = df['dp'].dt.month.fillna(1).astype(int)
df['DayOfWeek'] = df['dp'].dt.day_name()
df['is_weekend'] = (df['dp'].dt.dayofweek >= 5).astype(int)
df['Season'] = df['Month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

# Ordinal encoding for environment
light_map = {'Daylight':1,'Dawn':1.5,'Dusk':1.5,'Dark Artificial':3,'Dark':4}
rdsfc_map = {'Dry':1,'Wet':2,'Loose Snow':3,'Slush':3.5,'Packed Snow':4,'Ice':5}
vis_map = {'Clear':1,'Cloudy':2,'Rain':3,'Fog':3,'Mist':3,'Snow':4,'Freezing Rain':5,'Drifting Snow':5}
for col, m in [('LIGHT',light_map),('RDSFCOND',rdsfc_map),('VISIBILITY',vis_map)]:
    if col in df.columns: df[col+'_o'] = df[col].map(m).fillna(0)

# Spatial grid 250m
lat_c, lon_c = 0.00225, 0.003125
lat0, lon0 = df['LATITUDE'].min(), df['LONGITUDE'].min()
df['g_i'] = ((df['LATITUDE'] - lat0)/lat_c).astype(int)
df['g_j'] = ((df['LONGITUDE'] - lon0)/lon_c).astype(int)

cnt_feats = ['num_persons','num_unique_invtype','num_unique_vehtype',
             'num_unique_initdir','num_unique_manoeuver','num_unique_drivact',
             'num_unique_drivcond','num_unique_ages','min_age','max_age']

# Drop leakage
drop_leak = ['ACCLASS','ACCNUM','DATE','dp']
drop_leak = [c for c in drop_leak if c in df.columns]
df.drop(columns=drop_leak, inplace=True)

# Impute
for c in df.select_dtypes(include=['object']).columns:
    m = df[c].mode(); df[c] = df[c].fillna(m[0] if len(m)>0 else 'Unknown')
for c in df.select_dtypes(include=['int64','float64']).columns:
    df[c] = df[c].fillna(df[c].median())
for c in cnt_feats:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

print(f'Shape: {df.shape}')
for i, name in enumerate(tnames):
    cnt = (df.target == i).sum()
    print(f'  {name}: {cnt} ({cnt/len(df)*100:.1f}%)')


cat_f = ['ROAD_CLASS','DISTRICT','TRAFFCTL','IMPACTYPE',
         'INVTYPE_list','VEHTYPE_list','INITDIR_list','MANOEUVER_list','DRIVACT_list','DRIVCOND_list',
         'DayOfWeek','Season']
bool_f = ['PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','SPEEDING','AG_DRIV','REDLIGHT']

for c in cat_f:
    if c in df.columns: df[c+'_e'] = LabelEncoder().fit_transform(df[c].astype(str))
for c in bool_f:
    if c in df.columns: df[c+'_b'] = df[c].map({'Yes':1,'No':0}).fillna(0).astype(int)

cnt_feats_clean = [c for c in cnt_feats if c in df.columns]
f_cols = ([c+'_e' for c in cat_f if c+'_e' in df.columns] +
          [c+'_b' for c in bool_f if c+'_b' in df.columns] +
          [c+'_o' for c in ['LIGHT','RDSFCOND','VISIBILITY'] if c+'_o' in df.columns] +
          cnt_feats_clean +
          ['Month','Year','is_weekend','g_i','g_j','LATITUDE','LONGITUDE'])
f_cols = [c for c in f_cols if c in df.columns]
print(f'Features: {len(f_cols)}')

# Temporal split
tr = df[~df['Year'].isin([2023, 2024])].copy()
te = df[df['Year'].isin([2023, 2024])].copy()
ttl = len(tr)+len(te)
print(f'Train: {len(tr)} ({len(tr)/ttl*100:.0f}%), Test: {len(te)} ({len(te)/ttl*100:.0f}%)')

Xtr = tr[f_cols].fillna(0); ytr = tr['target']
Xte = te[f_cols].fillna(0); yte = te['target']
ss = StandardScaler(); Xtr_s = ss.fit_transform(Xtr); Xte_s = ss.transform(Xte)
print(f'X_train: {Xtr.shape}, classes: {np.bincount(ytr.astype(int))}')
print(f'X_test:  {Xte.shape}, classes: {np.bincount(yte.astype(int))}')


try:
    from mord import LogisticAT
    print('\\n========== Ordered Logistic Regression ==========')
    olr = LogisticAT(alpha=0.0)
    olr.fit(Xtr_s, ytr)
    yp_olr = olr.predict(Xte_s)
    # For ROC AUC, use decision function
    if hasattr(olr, 'decision_function'):
        ypd_olr = olr.decision_function(Xte_s)
    else:
        ypd_olr = None
    print(f'Acc={accuracy_score(yte, yp_olr):.4f} Macro-F1={f1_score(yte,yp_olr,average="macro"):.4f}')
    print(classification_report(yte, yp_olr, target_names=tnames, zero_division=0))
except ImportError:
    print('mord not installed. Skipping Ordered Logistic Regression.')
    olr = None; yp_olr = None; ypd_olr = None


print('\\n========== RF Multiclass ==========')
rf_m = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf_m.fit(Xtr_s, ytr)
yp_rf = rf_m.predict(Xte_s); ypb_rf = rf_m.predict_proba(Xte_s)
print(f'Acc={accuracy_score(yte, yp_rf):.4f} Macro-F1={f1_score(yte,yp_rf,average="macro"):.4f}')
print(classification_report(yte, yp_rf, target_names=tnames, zero_division=0))

fig,ax=plt.subplots(figsize=(10,8))
ConfusionMatrixDisplay.from_predictions(yte, yp_rf, display_labels=tnames, ax=ax, cmap='Blues', normalize='true')
ax.set_title('RF CM'); plt.tight_layout(); plt.savefig(f'{OM}/rf_cm.png',dpi=150); plt.show()


print('\\n========== XGBoost Multiclass ==========')
xgb_m = XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=200, random_state=42,
                       eval_metric='mlogloss', objective='multi:softprob', num_class=5)
xgb_m.fit(Xtr_s, ytr)
yp_xgb = xgb_m.predict(Xte_s); ypb_xgb = xgb_m.predict_proba(Xte_s)
print(f'Acc={accuracy_score(yte, yp_xgb):.4f} Macro-F1={f1_score(yte,yp_xgb,average="macro"):.4f}')
print(classification_report(yte, yp_xgb, target_names=tnames, zero_division=0))

fig,ax=plt.subplots(figsize=(10,8))
ConfusionMatrixDisplay.from_predictions(yte, yp_xgb, display_labels=tnames, ax=ax, cmap='Oranges', normalize='true')
ax.set_title('XGB CM'); plt.tight_layout(); plt.savefig(f'{OM}/xgb_cm.png',dpi=150); plt.show()


print('\\n========== RF + SMOTE + GridSearch ==========')
sm = SMOTE(random_state=42)
Xtr_sm, ytr_sm = sm.fit_resample(Xtr_s, ytr)
print(f'Before SMOTE: {np.bincount(ytr.astype(int))}')
print(f'After SMOTE:  {np.bincount(ytr_sm.astype(int))}')

gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
                  param_grid={'n_estimators':[100,200],'max_depth':[8,12,16]}, cv=3, scoring='f1_macro')
gs.fit(Xtr_sm, ytr_sm)
print(f'Best: {gs.best_params_}, CV F1: {gs.best_score_:.4f}')

yp_gs = gs.predict(Xte_s); ypb_gs = gs.predict_proba(Xte_s)
print(f'Test: Acc={accuracy_score(yte,yp_gs):.4f} Macro-F1={f1_score(yte,yp_gs,average="macro"):.4f}')
print(classification_report(yte, yp_gs, target_names=tnames, zero_division=0))


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
paper_auc = {'Fatal': 0.96, 'Major': 0.88, 'Minimal': 0.96, 'Minor': 0.95, 'Unknown': 0.87}

results = []
for i, name in enumerate(tnames):
    yb = (yte == i).astype(int)
    if yb.sum() == 0:
        axes[i].text(.5,.5,f'{name}\\n(no data)',ha='center',va='center',transform=axes[i].transAxes)
        axes[i].set_title(f'{name}'); continue

    for n, pb, c in [('RF',ypb_rf,'steelblue'),('XGB',ypb_xgb,'coral'),('RF+SMOTE+GS',ypb_gs,'forestgreen')]:
        prob = pb[:, i] if pb.shape[1] > i else np.zeros(len(yte))
        fpr, tpr, _ = roc_curve(yb, prob)
        auc = roc_auc_score(yb, prob)
        axes[i].plot(fpr, tpr, label=f'{n} AUC={auc:.3f}', lw=2, color=c)
        results.append({'class':name,'model':n,'auc':auc})

    axes[i].plot([0,1],[0,1],'k--',alpha=.3)
    axes[i].axhline(paper_auc.get(name,0), color='gray', ls=':', label=f'Paper={paper_auc.get(name,0):.2f}')
    axes[i].legend(); axes[i].set_title(f'ROC - {name}')

for j in range(5, 6): axes[j].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OM}/roc_multiclass.png', dpi=150); plt.show()

res = pd.DataFrame(results)
print('\\nPer-class AUC:')
piv = res.pivot(index='class', columns='model', values='auc')
print(piv.to_string())
print(f'\\nPaper reference:')
for k,v in paper_auc.items(): print(f'  {k}: {v}')


fi = pd.DataFrame({'f':f_cols, 'i':gs.best_estimator_.feature_importances_}).sort_values('i', ascending=False)
fig,ax=plt.subplots(figsize=(12,10))
fi.head(20).plot(ax=ax, x='f', y='i', kind='barh', color='steelblue', legend=False)
ax.set_title('Top 20 Features (RF+SMOTE+GS)'); plt.tight_layout()
plt.savefig(f'{OM}/fi_top20.png', dpi=150); plt.show()


print('='*60)
print('BINARY (Traffic: Injury vs Non-Injury)')
print('='*60)
print(f'RF:                 AUC={roc_auc_score(yte,pb):.4f}')
print(f'XGB:                AUC={roc_auc_score(yte,pb_x):.4f}')
print(f'RF+SMOTE+GS:        AUC={roc_auc_score(yte,pb_gs):.4f}')
print(f'\\nPaper reference: RF with ~83% accuracy for binary injury vs non-injury')

print(f'\\n{"="*60}')
print('MULTICLASS ORDINAL (KSI: INJURY 5 levels)')
print('='*60)
print(f'\\nPer-class AUC (paper target):')
print(f'{"Class":<12} {"Paper":<8} {"Best Model":<20}')
print('-'*40)
paper_auc = {'Fatal': 0.96, 'Major': 0.88, 'Minimal': 0.96, 'Minor': 0.95, 'Unknown': 0.87}
for name in tnames:
    best = res[res['class']==name].sort_values('auc', ascending=False)
    if len(best) > 0:
        b = best.iloc[0]
        print(f'{name:<12} {paper_auc.get(name,0):<8.2f} {b["model"]:<20} {b["auc"]:.4f}')


