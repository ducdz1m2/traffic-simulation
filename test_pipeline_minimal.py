import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# R environment (Windows)
os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

# ============================================================
# MINIMAL CONFIG FOR TESTING
# ============================================================
PREPROCESSED_DIR = r"data\preprocessed"
OUT_DIR = "outputs_test"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_SIZE = 10000  # Use 1k preprocessed dataset
TRAIN_RATIO = 0.7
RANDOM_STATE = 42

# Minimal grid search
UNDER_STRATEGIES = [0.4, 0.5]  # Only 2 values
ROSE_STRATEGIES = [0.6, 0.7]  # Only 2 values

# NeighbourhoodCleaningRule parameter grid
NCR_NEIGHBORS = [3, 5, 7]
NCR_THRESHOLDS = [0.3, 0.5, 0.7]

# Minimal sampling
SAMPLER_NAMES = ['no', 'under', 'rose', 'mixed']  # Base techniques

print("=" * 60)
print("MINIMAL PIPELINE TEST")
print("=" * 60)
print(f"Using preprocessed dataset: {SAMPLE_SIZE} samples")
print(f"Grid search: {len(UNDER_STRATEGIES)} x {len(ROSE_STRATEGIES)} = {len(UNDER_STRATEGIES) * len(ROSE_STRATEGIES)} combinations")
print(f"Sampling techniques: {SAMPLER_NAMES}")

# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

preprocessed_file = os.path.join(PREPROCESSED_DIR, f"preprocessed_{SAMPLE_SIZE}.csv")
df = pd.read_csv(preprocessed_file)
print(f"Loaded: {len(df):,} rows")
print(f"Fatality rate: {df['Fatality'].mean()*100:.3f}%")

# ============================================================
# 2. TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)

df_train, df_test = train_test_split(
    df,
    train_size=TRAIN_RATIO,
    stratify=df['Fatality'],
    random_state=RANDOM_STATE
)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Train: {len(df_train):,} (fatal={df_train['Fatality'].sum():,})")
print(f"Test: {len(df_test):,} (fatal={df_test['Fatality'].sum():,})")

# ============================================================
# 5. FEATURE PREPARATION
# ============================================================
print("\n" + "=" * 60)
print("FEATURE PREPARATION")
print("=" * 60)

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

def prepare_features(df, ohe=None):
    df_model = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    X_dense = csr_matrix(df_model[FEATURE_COLS].values)
    
    if ohe is None:
        ohe = OneHotEncoder(sparse_output=True, min_frequency=0.001, handle_unknown='infrequent_if_exist')
        X_ohe_sparse = ohe.fit_transform(df_model[NOMINAL_COLS])
    else:
        X_ohe_sparse = ohe.transform(df_model[NOMINAL_COLS])
    
    X_features = hstack([X_dense, X_ohe_sparse], format='csr')
    y = df_model[TARGET].astype(int).values
    
    return X_features, y, ohe

X_train, y_train, ohe = prepare_features(df_train)
X_test, y_test, _ = prepare_features(df_test, ohe=ohe)

nominal_indices = list(range(len(FEATURE_COLS), len(FEATURE_COLS) + len(NOMINAL_COLS)))

print(f"Train: {X_train.shape[1]} features, {len(y_train)} samples")
print(f"Test: {X_test.shape[1]} features, {len(y_test)} samples")

# ============================================================
# 6. GRID SEARCH (MINIMAL)
# ============================================================
print("\n" + "=" * 60)
print("GRID SEARCH (MINIMAL)")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler
import tempfile
import rpy2.robjects as ro

def rose_r(X, y, sampling_strategy, random_state, nominal_indices=None):
    n_maj = int(np.sum(y == 0))
    n_min = max(1, int(np.sum(y == 1)))
    
    if sampling_strategy < 1:
        n_synth = max(0, int(n_maj * sampling_strategy) - n_min)
    else:
        n_synth = max(0, int(n_min * sampling_strategy) - n_min)
    
    if n_synth <= 0:
        return X, y
    
    n_total = n_maj + n_min + n_synth
    
    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_path = tmp.name
    tmp.close()
    
    Xd = X.toarray() if hasattr(X, 'toarray') else X
    pd.DataFrame(np.column_stack([Xd, y.astype(int)])).to_csv(tmp_path, index=False, header=False)
    
    ro.globalenv['csv_path'] = tmp_path
    ro.globalenv['n_total'] = int(n_total)
    ro.globalenv['seed_val'] = int(random_state)
    
    if nominal_indices is not None:
        nom_idx_r = ro.IntVector([i + 1 for i in nominal_indices])
        ro.globalenv['nominal_indices'] = nom_idx_r
        ro.r('''
            library(ROSE)
            d <- read.csv(csv_path, header=FALSE)
            colnames(d)[ncol(d)] <- "y"
            d$y <- as.factor(d$y)
            for (i in nominal_indices) {
                d[,i] <- as.factor(d[,i])
            }
            r <- ROSE(y~., d, N=n_total, seed=seed_val)
            write.csv(r$data, csv_path, row.names=FALSE)
        ''')
    else:
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

def rose_ncr_r(X, y, sampling_strategy, random_state, nominal_indices=None, n_neighbors=3, threshold_cleaning=0.5):
    """ROSE oversampling followed by NeighbourhoodCleaningRule undersampling"""
    # Step 1: Apply ROSE oversampling
    X_rose, y_rose = rose_r(X, y, sampling_strategy, random_state, nominal_indices)
    
    # Step 2: Apply NeighbourhoodCleaningRule undersampling
    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule(n_neighbors=n_neighbors, threshold_cleaning=threshold_cleaning)
    X_final, y_final = ncr.fit_resample(X_rose, y_rose)
    
    n_orig = X.shape[0] if hasattr(X, 'shape') else len(X)
    n_rose = X_rose.shape[0] if hasattr(X_rose, 'shape') else len(X_rose)
    n_final = X_final.shape[0] if hasattr(X_final, 'shape') else len(X_final)
    print(f"    ROSE + NCR(k={n_neighbors}, threshold={threshold_cleaning}): {n_orig} -> {n_rose} -> {n_final} samples")
    return X_final, y_final

results = []

for under_strat in UNDER_STRATEGIES:
    for rose_strat in ROSE_STRATEGIES:
        print(f"\n>>> Under={under_strat}, ROSE={rose_strat}")
        
        try:
            Xm, ym = RandomUnderSampler(random_state=42, sampling_strategy=under_strat).fit_resample(X_train, y_train)
            X_sampled, y_sampled = rose_r(Xm, ym, rose_strat, 42, nominal_indices)
            
            print(f"  Sampled: {len(y_sampled):,} (fatal={y_sampled.sum():,})")
            
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model.fit(X_sampled, y_sampled)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            g_mean = np.sqrt(sens * spec) if (sens * spec) >= 0 else 0
            
            print(f"  Acc={acc:.3f} | MCC={mcc:.3f}")
            
            results.append({
                'under_strategy': under_strat,
                'rose_strategy': rose_strat,
                'mcc': mcc,
                'accuracy': acc
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

df_results = pd.DataFrame(results)
print("\n" + "=" * 60)
print("GRID SEARCH RESULTS")
print("=" * 60)
print(df_results)

best_params = df_results.nlargest(1, 'mcc').iloc[0]
BEST_UNDER = best_params['under_strategy']
BEST_ROSE = best_params['rose_strategy']
print(f"\nBest: under={BEST_UNDER}, rose={BEST_ROSE} (MCC={best_params['mcc']:.3f})")

# ============================================================
# 7. SAMPLING (MINIMAL)
# ============================================================
print("\n" + "=" * 60)
print("SAMPLING (MINIMAL)")
print("=" * 60)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def apply_sampler(X, y, name, seed=42, nominal_indices=None, under_strategy=None, rose_strategy=None, ncr_neighbors=3, ncr_threshold=0.5):
    if name == 'no':
        return X, y
    if name == 'rose':
        strategy = rose_strategy if rose_strategy is not None else 0.5
        return rose_r(X, y, strategy, seed, nominal_indices)
    if name == 'rose_ncr':
        strategy = rose_strategy if rose_strategy is not None else 0.5
        return rose_ncr_r(X, y, strategy, seed, nominal_indices, n_neighbors=ncr_neighbors, threshold_cleaning=ncr_threshold)
    if name == 'mixed':
        under_strat = under_strategy if under_strategy is not None else 0.4
        rose_strat = rose_strategy if rose_strategy is not None else 0.8
        Xm, ym = RandomUnderSampler(random_state=seed, sampling_strategy=under_strat).fit_resample(X, y)
        return rose_r(Xm, ym, rose_strat, seed, nominal_indices)
    
    n_min = max(1, int(np.sum(y == 1)))
    n_maj = int(np.sum(y == 0))
    
    if n_min < 2 and name != 'under':
        print(f"  Skipped {name}: minority too small")
        return X, y
    
    k = max(1, min(5, n_min - 1))
    
    if name == 'under':
        return RandomUnderSampler(random_state=seed).fit_resample(X, y)
    if name == 'smote':
        return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    
    return X, y

sampled_datasets = {}

for sampler_name in SAMPLER_NAMES:
    print(f"\n>>> {sampler_name.upper()}")
    
    X_sampled, y_sampled = apply_sampler(
        X_train, y_train, sampler_name, 42, nominal_indices,
        under_strategy=BEST_UNDER, rose_strategy=BEST_ROSE
    )
    
    print(f"  Samples: {len(y_sampled):,} (fatal={y_sampled.sum():,})")
    sampled_datasets[sampler_name] = (X_sampled, y_sampled)

# ============================================================
# 7.5. NEIGHBOURHOODCLEANINGRULE PARAMETER TUNING
# ============================================================
print("\n" + "=" * 60)
print("NEIGHBOURHOODCLEANINGRULE PARAMETER TUNING")
print("=" * 60)
print(f"Neighbors: {NCR_NEIGHBORS}")
print(f"Thresholds: {NCR_THRESHOLDS}")
print(f"Total combinations: {len(NCR_NEIGHBORS) * len(NCR_THRESHOLDS)}")

ncr_results = []

for n_neighbors in NCR_NEIGHBORS:
    for threshold in NCR_THRESHOLDS:
        print(f"\n>>> NCR k={n_neighbors}, threshold={threshold}")
        
        try:
            X_sampled, y_sampled = apply_sampler(
                X_train, y_train, 'rose_ncr', 42, nominal_indices,
                under_strategy=BEST_UNDER, rose_strategy=BEST_ROSE,
                ncr_neighbors=n_neighbors, ncr_threshold=threshold
            )
            
            print(f"  Samples: {len(y_sampled):,} (fatal={y_sampled.sum():,})")
            
            # Quick evaluation
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
            
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model.fit(X_sampled, y_sampled)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            g_mean = np.sqrt(sens * spec) if (sens * spec) >= 0 else 0
            
            # Combined score: prioritize acc, sen, spec (equal weights)
            combined_score = (acc + sens + spec) / 3
            
            print(f"  Acc={acc:.3f} | Sens={sens:.3f} | Spec={spec:.3f} | Combined={combined_score:.3f}")
            
            ncr_results.append({
                'n_neighbors': n_neighbors,
                'threshold': threshold,
                'samples': len(y_sampled),
                'fatal_rate': y_sampled.mean(),
                'accuracy': acc,
                'sensitivity': sens,
                'specificity': spec,
                'f1': f1,
                'mcc': mcc,
                'g_mean': g_mean,
                'combined_score': combined_score
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

# Save NCR results
df_ncr = pd.DataFrame(ncr_results)
ncr_path = os.path.join(OUT_DIR, 'ncr_tuning.csv')
df_ncr.to_csv(ncr_path, index=False)
print(f"\nNCR tuning results saved to: {ncr_path}")

print("\n" + "=" * 60)
print("NEIGHBOURHOODCLEANINGRULE TUNING RESULTS")
print("=" * 60)
print(df_ncr[['n_neighbors', 'threshold', 'combined_score', 'accuracy', 'sensitivity', 'specificity']])

# Select best NCR parameters (by combined score)
best_ncr = df_ncr.nlargest(1, 'combined_score').iloc[0]
BEST_NCR_NEIGHBORS = int(best_ncr['n_neighbors'])
BEST_NCR_THRESHOLD = best_ncr['threshold']
print(f"\nBest NCR: k={BEST_NCR_NEIGHBORS}, threshold={BEST_NCR_THRESHOLD} (Combined={best_ncr['combined_score']:.3f})")

# Add best NCR to sampled_datasets
X_best_ncr, y_best_ncr = apply_sampler(
    X_train, y_train, 'rose_ncr', 42, nominal_indices,
    under_strategy=BEST_UNDER, rose_strategy=BEST_ROSE,
    ncr_neighbors=BEST_NCR_NEIGHBORS, ncr_threshold=BEST_NCR_THRESHOLD
)
sampled_datasets['rose_ncr'] = (X_best_ncr, y_best_ncr)

# ============================================================
# 8. MODEL EVALUATION (MINIMAL)
# ============================================================
print("\n" + "=" * 60)
print("MODEL EVALUATION (MINIMAL)")
print("=" * 60)

from sklearn.linear_model import LogisticRegressionCV

all_results = []

for sampler_name, (X_train_s, y_train_s) in sampled_datasets.items():
    print(f"\n>>> {sampler_name.upper()}")
    
    # Lasso only (skip XGBoost for minimal test)
    lasso = LogisticRegressionCV(
        penalty='l1', solver='saga', Cs=5, cv=3,
        max_iter=500, random_state=42, n_jobs=-1
    )
    lasso.fit(X_train_s, y_train_s)
    y_pred = lasso.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    g_mean = np.sqrt(sens * spec) if (sens * spec) >= 0 else 0
    
    print(f"  Acc={acc:.3f} | Sens={sens:.3f} | Spec={spec:.3f} | F1={f1:.3f} | MCC={mcc:.3f} | G={g_mean:.3f}")
    
    all_results.append({
        'Sampler': sampler_name,
        'Accuracy': acc,
        'Sensitivity': sens,
        'Specificity': spec,
        'F1': f1,
        'MCC': mcc,
        'G_mean': g_mean
    })

df_eval = pd.DataFrame(all_results)
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(df_eval)

# Save results
df_results.to_csv(os.path.join(OUT_DIR, 'grid_search.csv'), index=False)
df_eval.to_csv(os.path.join(OUT_DIR, 'evaluation.csv'), index=False)

print("\n" + "=" * 60)
print("TEST COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Results saved to: {OUT_DIR}/")
