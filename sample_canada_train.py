import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek

# R environment (bat dong set de tranh loi stats.dll)
os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TRAIN_DATA_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\train.csv"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\sampled"
RANDOM_STATE = 42

# Feature columns (numeric only - true numeric values)
FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
# Nominal columns (categorical - will be OHE for Lasso)
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
# No LabelEncoder needed - all nominal columns will be OHE
CATEGORICAL_COLS = []

SAMPLER_NAMES = ['under', 'rose', 'rose_nearmiss', 'mixed', 'smote', 'borderline', 'smote_tomek', 'adasyn', 'nearmiss']

# ============================================================
# SAMPLING FUNCTIONS
# ============================================================
def _make_sampler(name, n_min, seed=RANDOM_STATE):
    k = max(1, min(5, max(1, n_min - 1)))
    if name == 'under':
        return RandomUnderSampler(random_state=seed)
    if name == 'smote':
        return SMOTE(random_state=seed, k_neighbors=k)
    if name == 'borderline':
        return BorderlineSMOTE(random_state=seed, k_neighbors=k)
    if name == 'smote_tomek':
        return SMOTETomek(random_state=seed, smote=SMOTE(k_neighbors=k))
    if name == 'adasyn':
        return ADASYN(random_state=seed, n_neighbors=k)
    if name == 'nearmiss':
        return NearMiss(version=1, n_neighbors=k)
    return None

def rose_r(X, y, sampling_strategy, random_state, nominal_indices=None):
    """ROSE sampling using R - properly handle categorical variables"""
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
    ro.globalenv['n_total'] = int(n_total)
    ro.globalenv['seed_val'] = int(random_state)
    
    # Convert nominal indices to R vector (1-indexed)
    if nominal_indices is not None:
        nom_idx_r = ro.IntVector([i + 1 for i in nominal_indices])  # R is 1-indexed
        ro.globalenv['nominal_indices'] = nom_idx_r
        ro.r('''
            library(ROSE)
            d <- read.csv(csv_path, header=FALSE)
            colnames(d)[ncol(d)] <- "y"
            d$y <- as.factor(d$y)
            # Convert nominal columns to factor
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

def rose_nearmiss_r(X, y, sampling_strategy, random_state, nominal_indices=None):
    """ROSE oversampling followed by NearMiss undersampling"""
    # Step 1: Apply ROSE oversampling
    X_rose, y_rose = rose_r(X, y, sampling_strategy, random_state, nominal_indices)
    
    # Step 2: Apply NearMiss undersampling
    from imblearn.under_sampling import NearMiss
    nm = NearMiss(version=1)
    X_final, y_final = nm.fit_resample(X_rose, y_rose)
    
    n_orig = X.shape[0] if hasattr(X, 'shape') else len(X)
    n_rose = X_rose.shape[0] if hasattr(X_rose, 'shape') else len(X_rose)
    n_final = X_final.shape[0] if hasattr(X_final, 'shape') else len(X_final)
    print(f"    ROSE + NearMiss: {n_orig} -> {n_rose} -> {n_final} samples")
    return X_final, y_final

def apply_sampler(X, y, name, seed=RANDOM_STATE, nominal_indices=None, under_strategy=None, rose_strategy=None):
    """Apply a sampling technique to the data"""
    if name == 'rose':
        strategy = rose_strategy if rose_strategy is not None else 0.5
        return rose_r(X, y, strategy, seed, nominal_indices)
    if name == 'rose_nearmiss':
        strategy = rose_strategy if rose_strategy is not None else 0.5
        return rose_nearmiss_r(X, y, strategy, seed, nominal_indices)
    if name == 'mixed':
        under_strat = under_strategy if under_strategy is not None else 0.4
        rose_strat = rose_strategy if rose_strategy is not None else 0.8
        Xm, ym = RandomUnderSampler(random_state=seed, sampling_strategy=under_strat).fit_resample(X, y)
        return rose_r(Xm, ym, rose_strat, seed, nominal_indices)
    
    n_min = max(1, int(np.sum(y == 1)))
    n_maj = int(np.sum(y == 0))
    
    # Skip sampling if minority class too small or majority class too small
    if n_min < 2 and name != 'under':
        print(f"  Skipped {name}: minority class too small ({n_min})")
        return X, y
    if n_maj < 2:
        print(f"  Skipped {name}: majority class too small ({n_maj})")
        return X, y
    
    k = min(5, n_min - 1)
    if k < 1 and name in ('nearmiss',):
        print(f"  Skipped {name}: k_neighbors would be {k}")
        return X, y
    
    try:
        sampler = _make_sampler(name, n_min, seed)
        if sampler is not None:
            return sampler.fit_resample(X, y)
    except Exception as e:
        print(f"  Skipped {name}: error - {e}")
        return X, y
    return X, y

# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_features(df):
    """Prepare features for sampling"""
    m = df[FEATURE_COLS + NOMINAL_COLS + ['Fatality']].copy()
    
    # Convert all columns to numeric where possible
    for col in FEATURE_COLS + NOMINAL_COLS:
        m[col] = pd.to_numeric(m[col], errors='coerce')

    m = m.dropna().reset_index(drop=True)
    y = m['Fatality'].astype(int).values
    X = m[FEATURE_COLS + NOMINAL_COLS].values.astype(np.float32)

    print(f"Features: {X.shape[1]}  Samples: {len(y):,}  "
          f"Fatality rate: {y.mean()*100:.3f}%")
    return X, y, m

def reconstruct_dataframe(X_sampled, y_sampled, original_df, sampler_name):
    """Reconstruct DataFrame with feature columns and target from sampled data"""
    # Create a new DataFrame with the sampled feature data
    sampled_df = pd.DataFrame(X_sampled, columns=FEATURE_COLS + NOMINAL_COLS)
    
    # For ROSE and MIXED sampling, round nominal columns to nearest integer
    # to preserve categorical nature, and clip negative values to 0
    if sampler_name in ['rose', 'mixed']:
        for col in NOMINAL_COLS:
            sampled_df[col] = np.round(sampled_df[col]).astype(int)
            sampled_df[col] = np.clip(sampled_df[col], 0, None)
    
    # Add the target column
    sampled_df['Fatality'] = y_sampled
    
    return sampled_df

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    print("=" * 60)
    print("  SAMPLING CANADA TRAIN DATASET")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load train data
    print(f"\nLoading train data from: {TRAIN_DATA_FILE}")
    df = pd.read_csv(TRAIN_DATA_FILE, low_memory=False)
    print(f"Original data: {len(df):,} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare features
    print("\n" + "=" * 60)
    print("  PREPARING FEATURES")
    print("=" * 60)
    X, y, df_prepared = prepare_features(df)
    
    # Calculate nominal indices (indices of nominal columns in X)
    nominal_indices = [FEATURE_COLS.index(col) for col in FEATURE_COLS]  # Start after numeric features
    nominal_indices = list(range(len(FEATURE_COLS), len(FEATURE_COLS) + len(NOMINAL_COLS)))
    
    print(f"\nClass distribution:")
    print(f"  Class 0 (Non-Fatality): {(y==0).sum():,}")
    print(f"  Class 1 (Fatality): {(y==1).sum():,}")
    print(f"Nominal column indices: {nominal_indices}")
    
    # Apply each sampling technique
    print("\n" + "=" * 60)
    print("  APPLYING SAMPLING TECHNIQUES")
    print("=" * 60)
    
    for sampler_name in SAMPLER_NAMES:
        print(f"\n>>> {sampler_name.upper()} SAMPLING ...")
        
        # Apply sampling
        # Best parameters from grid search (MCC priority): under=0.5, rose=0.5
        X_sampled, y_sampled = apply_sampler(X, y, sampler_name, RANDOM_STATE, nominal_indices, 
                                             under_strategy=0.5, rose_strategy=0.5)
        
        print(f"  Samples: {len(y_sampled):,}  (fatal={y_sampled.sum():,}  rate={y_sampled.mean()*100:.3f}%)")
        
        # Reconstruct DataFrame with all original columns
        df_sampled = reconstruct_dataframe(X_sampled, y_sampled, df_prepared, sampler_name)
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, f"train_{sampler_name}.csv")
        df_sampled.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("  COMPLETED")
    print("=" * 60)
    print(f"All sampled datasets saved to: {OUTPUT_DIR}")
    print(f"Sampling techniques applied: {', '.join(SAMPLER_NAMES)}")

if __name__ == "__main__":
    main()
