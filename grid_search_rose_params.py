import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
import itertools

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TRAIN_DATA_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\train.csv"
TEST_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\test.csv"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\outputs\rose_grid_search"
RANDOM_STATE = 42

# Feature columns
FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

# Grid search parameters
# sampling_strategy: target ratio for minority class
# Note: method, h, M parameters cause conflicts in R ROSE function, so we only tune sampling_strategy
GRID_PARAMS = {
    'sampling_strategy': [0.3, 0.5, 0.8, 1.0, 1.5]
}

# R environment
os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

# ============================================================
# ROSE SAMPLING FUNCTION WITH PARAMETERS
# ============================================================
def rose_r_with_params(X, y, sampling_strategy, random_state, nominal_indices=None):
    """ROSE sampling with sampling_strategy only (other params cause R conflicts)"""
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

    import rpy2.robjects as ro
    import tempfile
    n_total = n_maj + n_min + n_synth
    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_path = tmp.name
    tmp.close()
    Xd = X.toarray() if hasattr(X, 'toarray') else X
    pd.DataFrame(np.column_stack([Xd, y.astype(int)])).to_csv(tmp_path, index=False, header=False)
    
    ro.globalenv['csv_path'] = tmp_path
    ro.globalenv['n_total'] = n_total
    ro.globalenv['seed_val'] = random_state
    
    # Build R code - simple version with only N and seed
    r_code = '''
        library(ROSE)
        d <- read.csv(csv_path, header=FALSE)
        colnames(d)[ncol(d)] <- "y"
        d$y <- as.factor(d$y)
    '''
    
    # Convert nominal indices if specified
    if nominal_indices is not None:
        nom_idx_r = ro.IntVector([i + 1 for i in nominal_indices])
        ro.globalenv['nominal_indices'] = nom_idx_r
        r_code += '''
        for (i in nominal_indices) {
            d[,i] <- as.factor(d[,i])
        }
        '''
    
    r_code += '''
        r <- ROSE(y~., d, N=n_total, seed=seed_val)
        write.csv(r$data, csv_path, row.names=FALSE)
    '''
    
    ro.r(r_code)
    
    rd = pd.read_csv(tmp_path)
    os.unlink(tmp_path)
    return rd.drop(columns=['y']).values.astype(float), rd['y'].values.astype(int)

# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_features(df):
    """Prepare features for sampling"""
    m = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
    
    for col in FEATURE_COLS + NOMINAL_COLS:
        m[col] = pd.to_numeric(m[col], errors='coerce')
    
    m = m.dropna().reset_index(drop=True)
    y = m[TARGET].astype(int).values
    X = m[FEATURE_COLS + NOMINAL_COLS].values.astype(np.float32)
    
    return X, y, m

def reconstruct_dataframe(X_sampled, y_sampled):
    """Reconstruct DataFrame from sampled data"""
    sampled_df = pd.DataFrame(X_sampled, columns=FEATURE_COLS + NOMINAL_COLS)
    
    for col in NOMINAL_COLS:
        sampled_df[col] = np.round(sampled_df[col]).astype(int)
        sampled_df[col] = np.clip(sampled_df[col], 0, None)
    
    sampled_df[TARGET] = y_sampled
    return sampled_df

def prepare_features_with_ohe(df, feature_cols, nominal_cols, target):
    """Prepare features using One-Hot Encoding"""
    df_model = df[feature_cols + nominal_cols + [target]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    X_num = df_model[feature_cols].values.astype(float)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat = ohe.fit_transform(df_model[nominal_cols].astype(str))
    X_features = hstack([csr_matrix(X_num), X_cat])
    y = df_model[target].astype(int).values
    
    return X_features, y, ohe

def prepare_features_with_ohe_using_fitted(df, fitted_ohe, feature_cols, nominal_cols, target):
    """Prepare features using already-fitted OHE encoder"""
    df_model = df[feature_cols + nominal_cols + [target]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    X_num = df_model[feature_cols].values.astype(float)
    X_cat = fitted_ohe.transform(df_model[nominal_cols].astype(str))
    X_features = hstack([csr_matrix(X_num), X_cat])
    y = df_model[target].astype(int).values
    
    return X_features, y

# ============================================================
# XGBOOST EVALUATION
# ============================================================
def evaluate_xgboost(X_train, X_test, y_train, y_test, random_state=42):
    """Evaluate XGBoost and return metrics"""
    spw = max(1, sum(y_train == 0) / max(sum(y_train == 1), 1))
    xgb = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.5,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        objective='binary:logistic',
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    prauc = average_precision_score(y_test, y_prob)
    g_mean = np.sqrt(sens * spec) if (sens * spec) >= 0 else 0
    
    return {
        'Accuracy': acc,
        'Sensitivity': sens,
        'Specificity': spec,
        'Precision': prec,
        'F1': f1,
        'MCC': mcc,
        'G_mean': g_mean,
        'AUC_ROC': auc,
        'PR_AUC': prauc
    }

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    print("=" * 60)
    print("  GRID SEARCH ROSE PARAMETERS")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load train data
    print(f"\nLoading train data from: {TRAIN_DATA_FILE}")
    df_train = pd.read_csv(TRAIN_DATA_FILE, low_memory=False)
    print(f"Original train data: {len(df_train):,} records")
    
    # Load test data
    print(f"Loading test data from: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE, low_memory=False)
    print(f"Test data: {len(df_test):,} records")
    
    # Prepare features
    print("\n" + "=" * 60)
    print("  PREPARING FEATURES")
    print("=" * 60)
    X, y, df_prepared = prepare_features(df_train)
    
    nominal_indices = list(range(len(FEATURE_COLS), len(FEATURE_COLS) + len(NOMINAL_COLS)))
    
    print(f"\nOriginal class distribution:")
    print(f"  Non-Fatal: {(y==0).sum():,}")
    print(f"  Fatal: {(y==1).sum():,}")
    print(f"  Imbalance ratio: {(y==0).sum() / max((y==1).sum(), 1):.2f}")
    
    # Prepare test features with OHE
    print("\nPreparing test features with OHE...")
    X_test_ohe, y_test_ohe, ohe = prepare_features_with_ohe(
        df_test, FEATURE_COLS, NOMINAL_COLS, TARGET)
    print(f"Test: {len(y_test_ohe):,} (fatal={y_test_ohe.sum():,})")
    
    # Generate all parameter combinations
    param_names = list(GRID_PARAMS.keys())
    param_values = list(GRID_PARAMS.values())
    all_combinations = list(itertools.product(*param_values))
    
    print(f"\n" + "=" * 60)
    print("  GRID SEARCH")
    print("=" * 60)
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Parameters: {param_names}")
    
    all_results = []
    
    for idx, combination in enumerate(all_combinations):
        params = dict(zip(param_names, combination))
        print(f"\n{'=' * 60}")
        print(f"  COMBINATION {idx + 1}/{len(all_combinations)}")
        print(f"{'=' * 60}")
        print(f"Parameters: {params}")
        
        try:
            # Apply ROSE with parameters
            X_sampled, y_sampled = rose_r_with_params(
                X, y, 
                sampling_strategy=params['sampling_strategy'],
                random_state=RANDOM_STATE,
                nominal_indices=nominal_indices
            )
            
            print(f"  Samples: {len(y_sampled):,}")
            print(f"  Non-Fatal: {(y_sampled==0).sum():,}")
            print(f"  Fatal: {(y_sampled==1).sum():,}")
            print(f"  Imbalance ratio: {(y_sampled==0).sum() / max((y_sampled==1).sum(), 1):.2f}")
            
            # Reconstruct DataFrame
            df_sampled = reconstruct_dataframe(X_sampled, y_sampled)
            
            # Prepare train features with OHE
            X_train_ohe, y_train_ohe, _ = prepare_features_with_ohe(
                df_sampled, FEATURE_COLS, NOMINAL_COLS, TARGET)
            
            # Evaluate XGBoost
            print(f"  Evaluating XGBoost...")
            results = evaluate_xgboost(X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe, RANDOM_STATE)
            
            # Add parameters to results
            results['sampling_strategy'] = params['sampling_strategy']
            
            print(f"  Accuracy: {results['Accuracy']:.4f}")
            print(f"  Sensitivity: {results['Sensitivity']:.4f}")
            print(f"  G-mean: {results['G_mean']:.4f}")
            print(f"  MCC: {results['MCC']:.4f}")
            print(f"  AUC-ROC: {results['AUC_ROC']:.4f}")
            
            all_results.append(results)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    print("\n" + "=" * 60)
    print("  SAVING RESULTS")
    print("=" * 60)
    
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(OUTPUT_DIR, 'grid_search_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    # Find best parameters by different metrics
    print("\n" + "=" * 60)
    print("  BEST PARAMETERS BY METRIC")
    print("=" * 60)
    
    metrics_to_check = ['MCC', 'G_mean', 'AUC_ROC', 'Sensitivity', 'F1']
    
    for metric in metrics_to_check:
        best_idx = df_results[metric].idxmax()
        best_row = df_results.iloc[best_idx]
        print(f"\nBest by {metric}:")
        print(f"  Value: {best_row[metric]:.4f}")
        print(f"  sampling_strategy: {best_row['sampling_strategy']}")
        print(f"  Accuracy: {best_row['Accuracy']:.4f}")
        print(f"  Sensitivity: {best_row['Sensitivity']:.4f}")
        print(f"  MCC: {best_row['MCC']:.4f}")
    
    # Print full results sorted by MCC
    print("\n" + "=" * 60)
    print("  ALL RESULTS (SORTED BY MCC)")
    print("=" * 60)
    
    df_sorted = df_results.sort_values('MCC', ascending=False)
    display_cols = ['sampling_strategy', 'Accuracy', 'Sensitivity', 'G_mean', 'MCC', 'AUC_ROC']
    print(df_sorted[display_cols].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("  COMPLETED")
    print("=" * 60)
    print(f"Total evaluations: {len(all_results)}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
