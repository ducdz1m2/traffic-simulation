import os, sys, glob, warnings, time, tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             roc_auc_score, average_precision_score, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\canada-dataset"
SAMPLE_SIZE = 50000
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_CLUSTERS_MAX = 8
FCM_M = 2.0
PCA_N = 6
MICE_ITER = 5

# R environment (bat dong set de tranh loi stats.dll)
os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

# ============================================================
# LOGGING
# ============================================================
LOG_FILE = None

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def section(title):
    log("")
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    section("1. LOAD DATA")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv*")))
    log(f"Found {len(files)} CSV files in {DATA_DIR}")

    df_list = []
    for fpath in files:
        df = pd.read_csv(fpath, dtype=str, low_memory=False)
        df_list.append(df)
        log(f"  {os.path.basename(fpath):30} {len(df):>8,} rows")

    df = pd.concat(df_list, ignore_index=True)
    log(f"Total: {len(df):,} records")
    return df

# ============================================================
# 2. PREPROCESSING
# ============================================================
NA_CODES = {"U", "UU", "UUUU", "X", "XX", "XXXX", "N", "NN", "NNNN", "Q", "QQ"}

def clean_numeric(series):
    return pd.to_numeric(series.replace(NA_CODES, np.nan), errors="coerce")

NUM_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS',
            'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF',
            'V_ID', 'V_TYPE', 'V_YEAR', 'P_ID', 'P_AGE', 'P_PSN',
            'P_SEX', 'P_SAFE', 'P_ISEV', 'P_USER']

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS',
                'V_TYPE', 'V_YEAR', 'P_SEX', 'P_AGE', 'P_PSN', 'P_USER']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE']
CATEGORICAL_COLS = ['V_TYPE', 'P_SEX']
TARGET = 'Fatality'

def preprocess(df):
    section("2. PREPROCESSING")

    df['P_SEX'] = df['P_SEX'].replace({'M': 1, 'F': 0})

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    df['Fatality'] = (df['P_ISEV'] == 3).astype(int)
    log(f"Fatality: 0={(df['Fatality']==0).sum():,}  1={(df['Fatality']==1).sum():,}  rate={df['Fatality'].mean()*100:.3f}%")

    miss = {c: int(df[c].isnull().sum()) for c in NUM_COLS
            if c in df.columns and df[c].isnull().sum() > 0}
    log(f"Columns with missing: {len(miss)}")
    for c, n in sorted(miss.items(), key=lambda x: -x[1])[:10]:
        log(f"  {c}: {n:,} ({n/len(df)*100:.2f}%)")
    return df

def take_sample(df, size):
    section("   Sampling data")
    df = df.dropna(subset=['Fatality']).copy()
    valid = df['Fatality'].value_counts()
    valid = valid[valid >= 2].index
    df = df[df['Fatality'].isin(valid)]
    actual = min(size, len(df))
    df_s, _ = train_test_split(df, train_size=actual,
                               stratify=df['Fatality'], random_state=RANDOM_STATE)
    log(f"Sampled {len(df_s):,} (fatality rate={df_s['Fatality'].mean()*100:.3f}%)")
    return df_s.reset_index(drop=True)

# ============================================================
# 3. IMPUTATION
# ============================================================
def story_imputation(df):
    section("3a. STORY-BASED IMPUTATION")

    # C_WTHR rain/snow -> C_RSUR wet
    rain_codes = [3, 4, 5]
    mask_rain = df['C_WTHR'].isin(rain_codes) & (df['C_RSUR'].isnull() | (df['C_RSUR'] == 9))
    before = df['C_RSUR'].isnull().sum()
    df.loc[mask_rain, 'C_RSUR'] = 2
    n_rain = before - df['C_RSUR'].isnull().sum()

    # C_WTHR clear/cloudy -> C_RSUR dry
    clear_codes = [1, 2]
    mask_clear = df['C_WTHR'].isin(clear_codes) & (df['C_RSUR'].isnull() | (df['C_RSUR'] == 9))
    before = df['C_RSUR'].isnull().sum()
    df.loc[mask_clear, 'C_RSUR'] = 1
    n_clear = before - df['C_RSUR'].isnull().sum()

    # C_RSUR dry -> C_WTHR clear
    before = df['C_WTHR'].isnull().sum()
    df.loc[(df['C_RSUR'] == 1) & (df['C_WTHR'].isnull()), 'C_WTHR'] = 1
    n_dry = before - df['C_WTHR'].isnull().sum()

    # C_RSUR wet -> C_WTHR rain
    before = df['C_WTHR'].isnull().sum()
    df.loc[(df['C_RSUR'] == 2) & (df['C_WTHR'].isnull()), 'C_WTHR'] = 3
    n_wet = before - df['C_WTHR'].isnull().sum()

    # V_TYPE mode -> V_YEAR
    vtype_yr = df.groupby('V_TYPE')['V_YEAR'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    before = df['V_YEAR'].isnull().sum()
    df['V_YEAR'] = df['V_YEAR'].fillna(vtype_yr)
    n_vtype_yr = before - df['V_YEAR'].isnull().sum()

    # P_PSN -> P_USER
    pos_to_user = {
        11: 1, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2,
        21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3,
        31: 4, 32: 4, 33: 4, 34: 4, 35: 4, 36: 4, 37: 4, 38: 4, 39: 4,
    }
    before = df['P_USER'].isnull().sum()
    mask_pos = df['P_USER'].isnull() | (df['P_USER'].isin([9]))
    df.loc[mask_pos, 'P_USER'] = df.loc[mask_pos, 'P_PSN'].map(pos_to_user)
    n_pos = before - df['P_USER'].isnull().sum()

    # C_VEHS from max V_ID per case
    max_vid = df.groupby('C_CASE')['V_ID'].transform('max')
    before = df['C_VEHS'].isnull().sum()
    df['C_VEHS'] = df['C_VEHS'].fillna(max_vid)
    n_vehs = before - df['C_VEHS'].isnull().sum()

    # C_CONF head-on -> C_RCFG undivided
    headon_codes = list(range(1, 11))
    before = df['C_RCFG'].isnull().sum()
    mask_h = df['C_CONF'].isin(headon_codes) & (df['C_RCFG'].isnull() | (df['C_RCFG'] == 9))
    df.loc[mask_h, 'C_RCFG'] = 2
    n_rcfg = before - df['C_RCFG'].isnull().sum()

    log(f"  C_RSUR (rain->wet):   {n_rain}")
    log(f"  C_RSUR (clear->dry):  {n_clear}")
    log(f"  C_WTHR (dry->clear):  {n_dry}")
    log(f"  C_WTHR (wet->rain):   {n_wet}")
    log(f"  V_YEAR (V_TYPE):     {n_vtype_yr}")
    log(f"  P_USER (P_PSN->):     {n_pos}")
    log(f"  C_VEHS (V_ID max):   {n_vehs}")
    log(f"  C_RCFG (C_CONF->):    {n_rcfg}")
    return df

def filter_data(df):
    section("3b. FILTER DATA")
    n0 = len(df)

    rules = [
        ('C_CASE/P_ISEV null', df['C_CASE'].isnull() | df['P_ISEV'].isnull()),
        ('P_PSN=99 & V_YEAR null', (df['P_PSN'] == 99) & df['V_YEAR'].isnull()),
        ('P_AGE < 0', pd.to_numeric(df['P_AGE'], errors='coerce') < 0),
        ('P_AGE > 120', pd.to_numeric(df['P_AGE'], errors='coerce') > 120),
    ]

    cy = pd.to_numeric(df['C_YEAR'], errors='coerce')
    vy = pd.to_numeric(df['V_YEAR'], errors='coerce')
    rules.append(('V_YEAR > C_YEAR+1', cy.notna() & vy.notna() & (vy > cy + 1)))

    for name, mask in rules:
        c = int(mask.sum())
        if c > 0:
            log(f"  {name}: removing {c}")
            df = df[~mask]

    df = df.dropna(subset=['Fatality'])
    log(f"Filter: {n0:,} -> {len(df):,}  (removed {n0-len(df):,})")
    return df

MICE_COLS = ['C_WTHR', 'C_RSUR', 'C_CONF', 'C_RCFG', 'C_RALN',
             'C_TRAF', 'V_TYPE', 'P_SEX', 'P_USER', 'P_PSN', 'P_SAFE']

def mice_imputation(df, n_iter=MICE_ITER):
    section("3c. MICE IMPUTATION")

    try:
        import rpy2.robjects as ro

        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        tmp_path = tmp.name
        tmp.close()

        df[MICE_COLS].to_csv(tmp_path, index=False)
        ro.globalenv['csv_path'] = tmp_path
        ro.globalenv['m'] = n_iter

        ro.r('''
            library(mice)
            r_df <- read.csv(csv_path, na.strings = c("NA",""))
            for (col in names(r_df)) r_df[[col]] <- as.factor(r_df[[col]])
            suppressMessages({
                imp <- mice(r_df, method=rep("polyreg",ncol(r_df)),
                            m=1, maxit=m, seed=42, printFlag=FALSE)
            })
            filled <- complete(imp)
            write.csv(filled, csv_path, row.names=FALSE)
        ''')

        filled = pd.read_csv(tmp_path)
        os.unlink(tmp_path)

        for col in MICE_COLS:
            df[col] = filled[col].values.astype(int)

        log(f"MICE (R polyreg, maxit={n_iter}) done for {len(MICE_COLS)} cols")

    except Exception as e:
        log(f"MICE via R failed: {e}")
        log("  Falling back to IterativeImputer (RandomForest)...")
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            mice_df = df[MICE_COLS].copy()
            imp = IterativeImputer(
                estimator=RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
                max_iter=n_iter, random_state=RANDOM_STATE, initial_strategy='most_frequent'
            )
            imputed = imp.fit_transform(mice_df)
            imputed = pd.DataFrame(imputed, columns=MICE_COLS).round(0).astype(int)
            for col in MICE_COLS:
                df[col] = imputed[col].values
            log("  IterativeImputer fallback done")
        except Exception as e2:
            log(f"  Fallback also failed: {e2}")

    remaining = {c: int(df[c].isnull().sum()) for c in NUM_COLS
                 if c in df.columns and df[c].isnull().sum() > 0}
    if remaining:
        log(f"  Remaining missing: {remaining}")
    else:
        log("  No remaining missing")
    return df

def simple_imputation(df):
    section("3d. SIMPLE IMPUTATION (fill remaining)")

    for col in ['P_AGE', 'V_YEAR', 'C_HOUR']:
        if df[col].isnull().sum() > 0:
            v = df[col].median()
            df[col] = df[col].fillna(v)
            log(f"  {col}: median={v:.0f}")

    for col in ['C_MNTH', 'C_WDAY']:
        if df[col].isnull().sum() > 0:
            v = df[col].mode(dropna=True)
            v = v.iloc[0] if len(v) > 0 else 0
            df[col] = df[col].fillna(v)
            log(f"  {col}: mode={v}")

    all_feat = FEATURE_COLS + NOMINAL_COLS
    for col in all_feat:
        if col in df.columns and df[col].isna().sum() > 0:
            v = df[col].mode(dropna=True)
            v = v.iloc[0] if len(v) > 0 else 0
            df[col] = df[col].fillna(v)
    return df

# ============================================================
# 4. FEATURE PREPARATION
# ============================================================
def prepare_features(df):
    section("4. FEATURE PREPARATION")
    m = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()

    for col in CATEGORICAL_COLS:
        m[col] = m[col].astype(str)
        m[col] = LabelEncoder().fit_transform(m[col])

    m = m.dropna().reset_index(drop=True)

    X_dense = csr_matrix(m[FEATURE_COLS].values)
    ohe = OneHotEncoder(sparse_output=True, min_frequency=0.001,
                        handle_unknown='infrequent_if_exist')
    X_ohe = ohe.fit_transform(m[NOMINAL_COLS])
    X_lasso = hstack([X_dense, X_ohe], format='csr')

    X_raw = m[FEATURE_COLS + NOMINAL_COLS].values
    y = m[TARGET].astype(int).values

    log(f"Lasso feats: {X_lasso.shape[1]} | Raw feats: {X_raw.shape[1]} | n={len(y):,}")
    return X_lasso, X_raw, y

# ============================================================
# 5. SAMPLERS
# ============================================================
def _make_sampler(name, n_min, seed=RANDOM_STATE):
    k = max(1, min(5, max(1, n_min - 1)))
    if name == 'under':
        return RandomUnderSampler(random_state=seed)
    if name == 'smote':
        return SMOTE(random_state=seed, k_neighbors=k)
    if name == 'smotenc':
        return SMOTE(random_state=seed, k_neighbors=k)
    if name == 'borderline':
        return BorderlineSMOTE(random_state=seed, k_neighbors=k)
    if name == 'smote_tomek':
        return SMOTETomek(random_state=seed, smote=SMOTE(k_neighbors=k))
    if name == 'adasyn':
        return ADASYN(random_state=seed, n_neighbors=k)
    if name == 'kmeans_smote':
        return KMeansSMOTE(random_state=seed, k_neighbors=k, cluster_balance_threshold='auto')
    if name == 'nearmiss':
        return NearMiss(version=1, n_neighbors=k)
    return None

def rose_r(X, y, sampling_strategy, random_state):
    n_maj = int(np.sum(y == 0))
    n_min = max(1, int(np.sum(y == 1)))
    if n_min < 2 or n_maj < 2:
        return X, y
    if sampling_strategy < 1:
        n_synth = max(0, int(n_maj * sampling_strategy) - n_min)
    else:
        n_synth = max(0, int(n_min * sampling_strategy) - n_min)
    if n_synth <= 0:
        return X, y

    try:
        import rpy2.robjects as ro
        n_total = n_maj + n_min + n_synth
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        tmp_path = tmp.name
        tmp.close()
        Xd = X.toarray() if hasattr(X, 'toarray') else X
        pd.DataFrame(np.column_stack([Xd, y.astype(int)])).to_csv(tmp_path, index=False, header=False)
        ro.globalenv['csv_path'] = tmp_path
        ro.globalenv['n_total'] = int(n_total)
        ro.globalenv['seed_val'] = int(random_state)
        ro.r('''
            library(ROSE)
            d <- read.csv(csv_path, header=FALSE)
            colnames(d)[ncol(d)] <- "y"
            d$y <- as.factor(d$y)
            r <- ROSE(y~., d, N=n_total, seed=seed_val)
            write.csv(r$data, csv_path, row.names=FALSE)
        ''')
        rd = pd.read_csv(tmp_path)
        os.unlink(tmp_path)
        return rd.drop(columns=['y']).values.astype(float), rd['y'].values.astype(int)
    except Exception:
        k = max(1, min(5, n_min - 1))
        return SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)

def apply_sampler(X, y, name, seed=RANDOM_STATE):
    if name == 'no':
        return X, y
    if name == 'rose':
        return rose_r(X, y, 0.5, seed)
    if name == 'mixed':
        Xm, ym = RandomUnderSampler(random_state=seed, sampling_strategy=0.4).fit_resample(X, y)
        return rose_r(Xm, ym, 0.8, seed)
    n_min = max(1, int(np.sum(y == 1)))
    if n_min < 2 and name != 'under':
        return X, y
    k = min(5, n_min - 1)
    if k < 1 and name in ('nearmiss',):
        return X, y
    sampler = _make_sampler(name, n_min, seed)
    if sampler is not None:
        return sampler.fit_resample(X, y)
    return X, y

SAMPLER_NAMES = ['no', 'under', 'rose', 'mixed', 'smote', 'smotenc',
                 'borderline', 'smote_tomek', 'adasyn', 'kmeans_smote', 'nearmiss']

# ============================================================
# 6. EVALUATION
# ============================================================
def evaluate(name, model, X_test, y_test, prefix=""):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prauc = average_precision_score(y_test, y_prob)
    gm = np.sqrt(sens * spec) if sens * spec >= 0 else 0

    line = (f"  {name:30s} | Acc={acc:.4f} | Sens={sens:.4f} | Prec={prec:.4f} | "
            f"F1={f1:.4f} | MCC={mcc:.4f} | G={gm:.4f} | AUC={auc:.4f}")
    log(f"{prefix}{line}")
    return {'Model': name, 'Accuracy': acc, 'Sensitivity': sens, 'Specificity': spec,
            'Precision': prec, 'F1': f1, 'MCC': mcc, 'G_mean': gm, 'AUC_ROC': auc, 'PR_AUC': prauc}

def get_clf(mname, spw=1):
    if mname == 'Lasso':
        return LogisticRegression(penalty='l1', solver='saga', random_state=RANDOM_STATE,
                                  n_jobs=-1, max_iter=1000)
    if mname == 'XGBoost':
        return XGBClassifier(scale_pos_weight=spw, use_label_encoder=False,
                             eval_metric='logloss', random_state=RANDOM_STATE,
                             n_jobs=-1, verbosity=0)
    if mname == 'RandomForest':
        return RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE,
                                      n_jobs=-1)
    return None

# ============================================================
# 7. FCM CLUSTERING
# ============================================================
def find_best_k(X, max_k):
    section("   Selecting optimal cluster count")
    best_k, best_sil = 2, -1
    Xs = StandardScaler().fit_transform(X)

    try:
        import skfuzzy
        from sklearn.metrics import silhouette_score
        for k in range(2, max_k + 1):
            try:
                cntr, u, _, _, _, _, _ = skfuzzy.cluster.cmeans(
                    Xs.T, k, FCM_M, error=0.005, maxiter=1000, init=None, seed=RANDOM_STATE)
                if np.any(np.isnan(u)):
                    continue
                labels = np.argmax(u, axis=0)
                if len(np.unique(labels)) < 2:
                    continue
                sil = silhouette_score(Xs, labels)
                log(f"  K={k}  silhouette={sil:.4f}")
                if sil > best_sil:
                    best_sil, best_k = sil, k
            except Exception:
                continue

    except ImportError:
        log("  skfuzzy not available, using KMeans silhouette")
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xs)
            labels = km.labels_
            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(Xs, labels)
            log(f"  K={k}  silhouette={sil:.4f}")
            if sil > best_sil:
                best_sil, best_k = sil, k

    log(f"  -> Best: K={best_k}  (silhouette={best_sil:.4f})")
    return best_k

def run_fcm(X, k):
    section("7. FCM CLUSTERING")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_samples = Xs.shape[0]
    log(f"Input: {n_samples} samples x {Xs.shape[1]} features, K={k}")

    try:
        import skfuzzy
        cntr, u, _, _, _, _, fpc = skfuzzy.cluster.cmeans(
            Xs.T, k, FCM_M, error=0.005, maxiter=1000, init=None, seed=RANDOM_STATE)
        hard = np.argmax(u, axis=0)
        mem = u.T
        log(f"FPC (partition coeff) = {fpc:.4f}")
    except Exception as e:
        log(f"FCM error: {e}, using KMeans + soft membership fallback")
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xs)
        hard = km.labels_
        dist = np.maximum(cdist(Xs, km.cluster_centers_), 1e-10)
        inv = 1.0 / dist
        mem = inv / inv.sum(axis=1, keepdims=True)

    for i in range(k):
        mask = hard == i
        log(f"  Cluster {i}: {mask.sum():,}  (mean mem={mem[mask, i].mean():.4f})")

    return hard, mem, scaler

# ============================================================
# 8. PER-CLUSTER SAMPLING
# ============================================================
def pick_best_sampler(X_k, y_k, seed=RANDOM_STATE):
    n_min = int(np.sum(y_k == 1))
    n_maj = int(np.sum(y_k == 0))
    try:
        if n_min < 5:
            test_size = max(0.1, 2.0 / max(n_min, 1))
        else:
            test_size = 0.3
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_k, y_k, test_size=min(test_size, 0.5), random_state=seed, stratify=y_k)
    except Exception:
        X_tr, X_va, y_tr, y_va = X_k, X_k, y_k, y_k

    best_f1, best_name = -1, 'no'
    for sname in SAMPLER_NAMES:
        if sname not in ('no', 'under') and n_min < 2:
            continue
        try:
            X_s, y_s = apply_sampler(X_tr, y_tr, sname, seed)
            if len(np.unique(y_s)) < 2 or len(np.unique(y_va)) < 2:
                continue
            spw = max(1, np.sum(y_s == 0) / max(np.sum(y_s == 1), 1))
            m = XGBClassifier(scale_pos_weight=spw, use_label_encoder=False,
                              eval_metric='logloss', random_state=seed,
                              n_jobs=-1, verbosity=0, n_estimators=100)
            m.fit(X_s, y_s)
            f1 = f1_score(y_va, m.predict(X_va))
            if f1 > best_f1:
                best_f1, best_name = f1, sname
        except Exception as e:
            log(f"    sampler={sname:15s} FAILED: {e}")

    return best_name, best_f1

def per_cluster_pipeline(X, y, hard, mem, seed=RANDOM_STATE):
    section("8. PER-CLUSTER SAMPLING")
    k = len(np.unique(hard))
    X_out, y_out = [], []
    cluster_info = []

    for i in range(k):
        mask = hard == i
        Xi, yi = X[mask], y[mask]

        if len(yi) == 0:
            log(f"  Cluster {i}: empty, skipping")
            continue

        mem_i = mem[mask, i]

        n_maj = int(np.sum(yi == 0))
        n_min = int(np.sum(yi == 1))
        ir = n_maj / max(n_min, 1)
        diff = 1.0 - float(np.mean(mem_i))

        log(f"  Cluster {i}: n={len(yi):,}  IR={ir:.2f}  difficulty={diff:.3f}  "
            f"(maj={n_maj} min={n_min})")

        if n_min < 2 or n_maj < 2:
            log(f"    -> Too few samples, skipping sampling")
            X_out.append(Xi)
            y_out.append(yi)
            cluster_info.append(
                dict(cluster=i, ir=round(ir,2), difficulty=round(diff,3),
                     sampler='no', cv_f1=-1, n_before=len(yi), n_after=len(yi)))
            continue

        if ir <= 1.5:
            log(f"    -> IR ≤ 1.5, no sampling needed")
            X_out.append(Xi)
            y_out.append(yi)
            cluster_info.append(
                dict(cluster=i, ir=round(ir,2), difficulty=round(diff,3),
                     sampler='no', cv_f1=-1, n_before=len(yi), n_after=len(yi)))
            continue

        best_fn, best_f1 = pick_best_sampler(Xi, yi, seed)
        log(f"    -> Best sampler: {best_fn}  (CV F1={best_f1:.4f})")

        Xs, ys = apply_sampler(Xi, yi, best_fn, seed)
        X_out.append(Xs)
        y_out.append(ys)
        cluster_info.append(
            dict(cluster=i, ir=round(ir,2), difficulty=round(diff,3),
                 sampler=best_fn, cv_f1=round(best_f1,4),
                 n_before=len(yi), n_after=len(ys)))

    X_all = np.vstack(X_out)
    y_all = np.concatenate(y_out)
    log(f"  -> Combined: {len(y_all):,}  (was {len(y):,})")
    return X_all, y_all, cluster_info

def apply_sampling_by_cluster(X, y, hard, cluster_info, seed=RANDOM_STATE):
    k = len(np.unique(hard))
    X_parts, y_parts = [], []
    for i in range(k):
        mask = hard == i
        Xi, yi = X[mask], y[mask]

        # Find sampler chosen for this cluster
        sname = 'no'
        for ci in cluster_info:
            if ci['cluster'] == i:
                sname = ci['sampler']
                break

        # Empty cluster guard
        if len(yi) == 0:
            continue

        if len(np.unique(yi)) < 2:
            X_parts.append(Xi.toarray() if hasattr(Xi, 'toarray') else np.asarray(Xi))
            y_parts.append(yi)
            continue

        try:
            Xs, ys = apply_sampler(Xi, yi, sname, seed)
            if hasattr(Xs, 'toarray'):
                Xs_dense = Xs.toarray()
            else:
                Xs_dense = np.asarray(Xs)
            if Xs_dense.ndim == 1:
                Xs_dense = Xs_dense.reshape(-1, 1)
            expected_cols = X.shape[1] if not hasattr(X, 'toarray') else X.shape[1]
            if Xs_dense.shape[1] != expected_cols:
                log(f"    WARNING: sampler {sname} changed feature dim from {expected_cols} to {Xs_dense.shape[1]}, using original")
                X_parts.append(Xi.toarray() if hasattr(Xi, 'toarray') else np.asarray(Xi))
                y_parts.append(yi)
            else:
                X_parts.append(Xs_dense)
                y_parts.append(ys)
        except Exception as e:
            log(f"    WARNING: sampler {sname} on cluster {i} failed: {e}, using original")
            X_parts.append(Xi.toarray() if hasattr(Xi, 'toarray') else np.asarray(Xi))
            y_parts.append(yi)
    if not X_parts:
        return np.empty((0, X.shape[1])), np.array([])
    return np.vstack(X_parts), np.concatenate(y_parts)

# ============================================================
# 9. RUN EXPERIMENT
# ============================================================
def run_experiment(X_tr_raw, X_te_raw, y_tr_raw, y_te,
                   X_tr_lasso, X_te_lasso, y_tr_lasso=None,
                   label="", baseline_samplers=None):
    prefix = f"[{label}] "
    results = []
    if baseline_samplers is None:
        baseline_samplers = ['no', 'under', 'rose', 'mixed']
    if y_tr_lasso is None:
        y_tr_lasso = y_tr_raw

    for mname in ['Lasso', 'XGBoost', 'RandomForest']:
        X_te = X_te_lasso if mname == 'Lasso' else X_te_raw
        y_cur = y_tr_lasso if mname == 'Lasso' else y_tr_raw

        for sname in baseline_samplers:
            X_tr_base = X_tr_lasso if mname == 'Lasso' else X_tr_raw
            try:
                X_tr_s, y_tr_s = apply_sampler(X_tr_base, y_cur, sname, RANDOM_STATE)
            except Exception as e:
                log(f"{prefix}{mname} + {sname} sampler FAILED: {e}, skipping")
                continue

            if len(np.unique(y_tr_s)) < 2:
                log(f"{prefix}{mname} + {sname}: only one class after sampling, skipping")
                continue

            spw = max(1, np.sum(y_tr_s == 0) / max(np.sum(y_tr_s == 1), 1))
            clf = get_clf(mname, spw)
            if clf is None:
                continue
            try:
                clf.fit(X_tr_s, y_tr_s)
                display_name = f"{mname} + {sname}"
                res = evaluate(display_name, clf, X_te, y_te, prefix=prefix)
                res['Pipeline'] = label
                res['Sampling'] = sname
                results.append(res)
            except Exception as e:
                log(f"{prefix}{mname} + {sname} FAILED: {e}")

    return results

# ============================================================
# 9. FCM EVALUATION (with threshold tuning)
# ============================================================
def evaluate_with_threshold(name, model, X_val, y_val, X_te, y_te, prefix=""):
    y_val_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.05, 0.96, 0.05):
        y_val_pred = (y_val_prob >= thresh).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    y_te_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_te_prob >= best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    acc = accuracy_score(y_te, y_pred)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = f1_score(y_te, y_pred)
    mcc = matthews_corrcoef(y_te, y_pred)
    auc = roc_auc_score(y_te, y_te_prob)
    gm = np.sqrt(sens * spec) if sens * spec >= 0 else 0

    line = (f"  {name:30s} | Acc={acc:.4f} | Sens={sens:.4f} | Prec={prec:.4f} | "
            f"F1={f1:.4f} | MCC={mcc:.4f} | G={gm:.4f} | AUC={auc:.4f} | th={best_thresh:.2f}")
    log(f"{prefix}{line}")
    return {'Model': name, 'Accuracy': acc, 'Sensitivity': sens, 'Specificity': spec,
            'Precision': prec, 'F1': f1, 'MCC': mcc, 'G_mean': gm, 'AUC_ROC': auc,
            'Best_Threshold': best_thresh}

def run_fcm_tuned(X_tr_raw, y_tr_raw, X_te_raw, y_te,
                  X_tr_lasso, X_te_lasso, y_tr_lasso=None):
    section("9. FCM PIPELINE CLASSIFIERS (with threshold tuning)")
    results = []
    if y_tr_lasso is None:
        y_tr_lasso = y_tr_raw

    val_frac = 0.2
    for mname in ['Lasso', 'XGBoost', 'RandomForest']:
        X_tr_cur = X_tr_lasso if mname == 'Lasso' else X_tr_raw
        X_te_cur = X_te_lasso if mname == 'Lasso' else X_te_raw
        y_cur = y_tr_lasso if mname == 'Lasso' else y_tr_raw

        try:
            X_tr2, X_val, y_tr2, y_val = train_test_split(
                X_tr_cur, y_cur, test_size=val_frac, random_state=RANDOM_STATE, stratify=y_cur)
        except Exception:
            X_tr2, X_val, y_tr2, y_val = X_tr_cur, X_tr_cur, y_cur, y_cur

        spw = max(1, np.sum(y_tr2 == 0) / max(np.sum(y_tr2 == 1), 1))
        clf = get_clf(mname, spw)
        if clf is None:
            continue
        try:
            clf.fit(X_tr2, y_tr2)
            display_name = f"{mname}"
            res = evaluate_with_threshold(display_name, clf, X_val, y_val, X_te_cur, y_te, prefix="[FCM] ")
            res['Pipeline'] = 'FCM'
            res['Sampling'] = 'tuned'
            results.append(res)
        except Exception as e:
            log(f"[FCM] {mname} FAILED: {e}")

    return results

# ============================================================
# MAIN
# ============================================================
def main():
    global LOG_FILE
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f"pipeline_fcm_{ts}.log"
    t0 = time.time()

    log(f"Pipeline FCM started  |  log={LOG_FILE}")
    log(f"Config: SAMPLE_SIZE={SAMPLE_SIZE}, TEST_SIZE={TEST_SIZE}, K_MAX={N_CLUSTERS_MAX}")

    # 1-2. Load & preprocess
    df = load_data()
    df = preprocess(df)
    df = take_sample(df, SAMPLE_SIZE)

    # 3. Imputation
    df = story_imputation(df)
    df = filter_data(df)
    df = mice_imputation(df)
    df = simple_imputation(df)

    # 4. Features
    X_lasso, X_raw, y = prepare_features(df)

    # Split
    section("   TRAIN/TEST SPLIT")
    indices = np.arange(len(y))
    tr_idx, te_idx, y_tr, y_te = train_test_split(
        indices, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_tr_lasso = X_lasso[tr_idx]
    X_te_lasso = X_lasso[te_idx]
    X_tr_raw   = X_raw[tr_idx]
    X_te_raw   = X_raw[te_idx]
    log(f"Train: {len(y_tr):,}  Test: {len(y_te):,}")
    log(f"  Train fatality rate: {y_tr.mean()*100:.3f}%")
    log(f"  Test  fatality rate: {y_te.mean()*100:.3f}%")

    # ===== BASELINE =====
    section("5-6. BASELINE (no FCM, no per-cluster sampling)")
    bl_results = run_experiment(X_tr_raw, X_te_raw, y_tr, y_te,
                                X_tr_lasso, X_te_lasso,
                                label="BASELINE")

    # ===== FCM CLUSTERING (with PCA on Lasso features) =====
    section("   PCA dimensionality reduction")
    X_tr_lasso_dense = X_tr_lasso.toarray() if hasattr(X_tr_lasso, 'toarray') else np.asarray(X_tr_lasso)
    pca = PCA(n_components=PCA_N)
    X_tr_pca = pca.fit_transform(X_tr_lasso_dense)
    var_ratio = pca.explained_variance_ratio_.sum()
    log(f"PCA on Lasso feats: {X_tr_lasso_dense.shape[1]} -> {PCA_N} components (explained variance={var_ratio:.3f})")

    best_k = find_best_k(X_tr_pca, N_CLUSTERS_MAX)
    hard_labels, membership, scaler = run_fcm(X_tr_pca, best_k)

    # ===== PER-CLUSTER SAMPLING (raw features) =====
    X_tr_fcm, y_tr_fcm, cluster_info = per_cluster_pipeline(
        X_tr_raw, y_tr, hard_labels, membership)

    # ===== APPLY SAME SAMPLERS TO LASSO FEATURES =====
    section("   Applying FCM sampling to Lasso features")
    X_tr_lasso_fcm, y_tr_lasso_fcm = apply_sampling_by_cluster(
        X_tr_lasso, y_tr, hard_labels, cluster_info)

    # ===== FCM PIPELINE (with threshold tuning) =====
    fcm_results = run_fcm_tuned(X_tr_fcm, y_tr_fcm, X_te_raw, y_te,
                                 X_tr_lasso_fcm, X_te_lasso,
                                 y_tr_lasso=y_tr_lasso_fcm)

    # ===== COMPARISON =====
    section("10. COMPARISON (best baseline per model vs FCM)")
    baseline_best = {}
    for br in bl_results:
        model_key = br['Model'].split(' + ')[0]
        if model_key not in baseline_best or br['F1'] > baseline_best[model_key]['F1']:
            baseline_best[model_key] = {'name': br['Model'], 'F1': br['F1'],
                                        'sampling': br.get('Sampling', 'no')}

    log(f"{'Model':30s}  {'Best BL Sampler':>18s}  {'BL F1':>8s}  {'FCM F1':>8s}  {'Delta':>8s}")
    log("-"*75)
    comp_rows = []
    for fr in fcm_results:
        mname = fr['Model'].split(' + ')[0]
        if mname in baseline_best:
            bf1 = baseline_best[mname]['F1']
            ff1 = fr['F1']
            d = ff1 - bf1
            log(f"{mname:30s}  {baseline_best[mname]['sampling']:>18s}  {bf1:>8.4f}  {ff1:>8.4f}  {d:>+8.4f}")
            comp_rows.append({'Model': mname, 'Best_BL_Sampling': baseline_best[mname]['sampling'],
                              'Baseline_F1': bf1, 'FCM_F1': ff1, 'Delta': d})

    # ===== SAVE =====
    out = "outputs_fcm"
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(bl_results + fcm_results).to_csv(
        os.path.join(out, f"results_{ts}.csv"), index=False)
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(out, f"comparison_{ts}.csv"), index=False)
    if cluster_info:
        pd.DataFrame(cluster_info).to_csv(
            os.path.join(out, f"clusters_{ts}.csv"), index=False)
    log(f"\nResults saved to {out}/")
    log(f"Total time: {time.time()-t0:.1f}s")
    log("DONE")

if __name__ == "__main__":
    main()
