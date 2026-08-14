"""
Grid Search for Mixed Sampling (Undersampling + ROSE) Parameters
Tests different combinations of under_strategy and rose_strategy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add sample_canada_train.py functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_canada_train import FEATURE_COLS, NOMINAL_COLS, RANDOM_STATE, rose_r

# ============================================================
# CONFIG
# ============================================================
TRAIN_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\train.csv"
TEST_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\test.csv"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\outputs\grid_search_mixed"

# Grid search parameters
UNDER_STRATEGIES = [0.3, 0.4, 0.5, 0.6]  # Undersampling majority class
ROSE_STRATEGIES = [0.5, 0.6, 0.7, 0.8, 0.9]  # ROSE oversampling minority class

# For quick testing
MAX_SAMPLES = None  # Use full dataset for actual grid search

TARGET = 'Fatality'

# ============================================================
# FEATURE PREPARATION
# ============================================================
def prepare_features(df, feature_cols, nominal_cols, target, ohe=None):
    """Prepare features using OHE"""
    df_model = df[feature_cols + nominal_cols + [target]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    X_dense = csr_matrix(df_model[feature_cols].values)
    
    if ohe is None:
        ohe = OneHotEncoder(sparse_output=True, min_frequency=0.001, handle_unknown='infrequent_if_exist')
        X_ohe_sparse = ohe.fit_transform(df_model[nominal_cols])
    else:
        X_ohe_sparse = ohe.transform(df_model[nominal_cols])
    
    X_features = hstack([X_dense, X_ohe_sparse], format='csr')
    y = df_model[target].astype(int).values
    
    return X_features, y, ohe

# ============================================================
# SAMPLING FUNCTIONS
# ============================================================
def apply_mixed_sampling(X, y, under_strategy, rose_strategy, nominal_indices, seed=RANDOM_STATE):
    """Apply mixed sampling: undersampling + ROSE"""
    Xm, ym = RandomUnderSampler(random_state=seed, sampling_strategy=under_strategy).fit_resample(X, y)
    Xr, yr = rose_r(Xm, ym, rose_strategy, seed, nominal_indices)
    return Xr, yr

# ============================================================
# MAIN GRID SEARCH
# ============================================================
def main():
    print("=" * 60)
    print("  GRID SEARCH FOR MIXED SAMPLING PARAMETERS")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"\nLoading train data from: {TRAIN_FILE}")
    df_train = pd.read_csv(TRAIN_FILE, low_memory=False)
    if MAX_SAMPLES:
        df_train = df_train.sample(n=min(MAX_SAMPLES, len(df_train)), random_state=RANDOM_STATE)
    print(f"  Train samples: {len(df_train):,}")
    
    print(f"\nLoading test data from: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE, low_memory=False)
    if MAX_SAMPLES:
        df_test = df_test.sample(n=min(MAX_SAMPLES, len(df_test)), random_state=RANDOM_STATE)
    print(f"  Test samples: {len(df_test):,}")
    
    # Prepare features
    print(f"\nPreparing features...")
    df_model = df_train[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    X_train = df_model[FEATURE_COLS + NOMINAL_COLS].values.astype(np.float32)
    y_train = df_model[TARGET].astype(int).values
    
    # Calculate nominal indices
    nominal_indices = list(range(len(FEATURE_COLS), len(FEATURE_COLS) + len(NOMINAL_COLS)))
    print(f"  Nominal indices: {nominal_indices}")
    print(f"  Train samples: {len(y_train):,} (fatal={y_train.sum():,}, rate={y_train.mean()*100:.3f}%)")
    
    # Fit OHE on original data
    _, _, ohe = prepare_features(df_train, FEATURE_COLS, NOMINAL_COLS, TARGET)
    
    # Prepare test features
    X_test, y_test, _ = prepare_features(df_test, FEATURE_COLS, NOMINAL_COLS, TARGET, ohe=ohe)
    print(f"  Test samples: {len(y_test):,} (fatal={y_test.sum():,}, rate={y_test.mean()*100:.3f}%)")
    
    # Grid search
    print(f"\n{'=' * 60}")
    print(f"  GRID SEARCH")
    print(f"{'=' * 60}")
    print(f"  Under strategies: {UNDER_STRATEGIES}")
    print(f"  ROSE strategies: {ROSE_STRATEGIES}")
    print(f"  Total combinations: {len(UNDER_STRATEGIES) * len(ROSE_STRATEGIES)}")
    
    results = []
    
    for under_strat in UNDER_STRATEGIES:
        for rose_strat in ROSE_STRATEGIES:
            print(f"\n>>> Under={under_strat}, ROSE={rose_strat}")
            
            try:
                # Apply mixed sampling
                X_sampled, y_sampled = apply_mixed_sampling(
                    X_train, y_train, under_strat, rose_strat, nominal_indices, RANDOM_STATE
                )
                
                print(f"  Sampled: {len(y_sampled):,} (fatal={y_sampled.sum():,}, rate={y_sampled.mean()*100:.3f}%)")
                
                # Prepare sampled features
                df_sampled = pd.DataFrame(X_sampled, columns=FEATURE_COLS + NOMINAL_COLS)
                df_sampled[TARGET] = y_sampled
                X_train_sampled, y_train_sampled, _ = prepare_features(df_sampled, FEATURE_COLS, NOMINAL_COLS, TARGET, ohe=ohe)
                
                # Quick evaluation with simple metrics
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_train_sampled, y_train_sampled)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                g_mean = np.sqrt(sens * spec) if (sens * spec) >= 0 else 0
                
                print(f"  Acc={acc:.3f} | Sens={sens:.3f} | Spec={spec:.3f} | F1={f1:.3f} | MCC={mcc:.3f} | G={g_mean:.3f}")
                
                results.append({
                    'under_strategy': under_strat,
                    'rose_strategy': rose_strat,
                    'samples': len(y_sampled),
                    'fatal_rate': y_sampled.mean(),
                    'accuracy': acc,
                    'sensitivity': sens,
                    'specificity': spec,
                    'f1': f1,
                    'mcc': mcc,
                    'g_mean': g_mean
                })
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
    
    # Save results
    df_results = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'grid_search_results.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SAVED TO: {output_path}")
    print(f"{'=' * 60}")
    
    # Show best results
    print(f"\nTOP 5 BY G-MEAN:")
    print(df_results.nlargest(5, 'g_mean')[['under_strategy', 'rose_strategy', 'g_mean', 'accuracy', 'sensitivity', 'specificity', 'f1', 'mcc']])
    
    print(f"\nTOP 5 BY MCC:")
    print(df_results.nlargest(5, 'mcc')[['under_strategy', 'rose_strategy', 'mcc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'g_mean']])

if __name__ == "__main__":
    main()
