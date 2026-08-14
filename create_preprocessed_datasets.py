import os
import warnings
import glob
import numpy as np
import pandas as pd

# R environment (Windows)
os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r"data\canada-dataset"
OUTPUT_DIR = r"data\preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZES = [1000, 10000]
RANDOM_STATE = 42

print("=" * 60)
print("CREATE PREPROCESSED DATASETS")
print("=" * 60)
print(f"Sample sizes: {SAMPLE_SIZES}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv*")))
df_list = []
for f in files:
    df = pd.read_csv(f, dtype=str, low_memory=False)
    df_list.append(df)
    print(f"  {os.path.basename(f):30s} {len(df):>8,} rows")

df_full = pd.concat(df_list, ignore_index=True)
print(f"\nTotal: {len(df_full):,} rows")

# ============================================================
# 2. CLEAN AND CREATE TARGET
# ============================================================
print("\n" + "=" * 60)
print("CLEANING DATA")
print("=" * 60)

NA_CODES = {"U", "UU", "UUUU", "X", "XX", "XXXX", "N", "NN", "NNNN", "Q", "QQ"}

def clean_numeric(series):
    return pd.to_numeric(series.replace(NA_CODES, np.nan), errors="coerce")

num_cols_all = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS',
                'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF',
                'V_ID', 'V_TYPE', 'V_YEAR', 'P_ID', 'P_AGE', 'P_PSN',
                'P_SEX', 'P_SAFE', 'P_ISEV', 'P_USER']

df_full['P_SEX'] = df_full['P_SEX'].replace({'M': 1, 'F': 0})
for col in num_cols_all:
    if col in df_full.columns:
        df_full[col] = clean_numeric(df_full[col])

df_full['Fatality'] = (df_full['P_ISEV'] == 3).astype(int)
print(f"Fatality rate: {df_full['Fatality'].mean()*100:.3f}%")

# ============================================================
# 3. PREPROCESSING FUNCTIONS
# ============================================================
def story_imputation(df):
    rain_codes = [3, 4, 5]
    mask_rain = df['C_WTHR'].isin(rain_codes) & (df['C_RSUR'].isnull() | (df['C_RSUR'] == 9))
    df.loc[mask_rain, 'C_RSUR'] = 2
    
    clear_codes = [1, 2]
    mask_clear = df['C_WTHR'].isin(clear_codes) & (df['C_RSUR'].isnull() | (df['C_RSUR'] == 9))
    df.loc[mask_clear, 'C_RSUR'] = 1
    
    mask_road = (df['C_RCFG'] == 1) & (df['C_RALN'].isnull() | (df['C_RALN'] == 9))
    df.loc[mask_road, 'C_RALN'] = 1
    
    mask_not_road = (df['C_RCFG'] != 1) & (df['C_RALN'].isnull() | (df['C_RALN'] == 9))
    df.loc[mask_not_road, 'C_RALN'] = 2
    
    return df

def mice_imputation(df, n_iter=5):
    import rpy2.robjects as ro
    import tempfile
    
    mice_cols = ['C_WTHR', 'C_RSUR', 'C_CONF', 'C_RCFG', 'C_RALN',
                 'C_TRAF', 'V_TYPE', 'P_SEX', 'P_USER', 'P_PSN', 'P_SAFE']
    
    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_path = tmp.name
    tmp.close()
    
    df[mice_cols].to_csv(tmp_path, index=False)
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
    
    for col in mice_cols:
        df[col] = filled[col].values.astype(int)
    
    return df

def simple_imputation(df):
    for col in ['P_AGE', 'C_HOUR']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    if df['V_YEAR'].isnull().sum() > 0:
        v = df['V_YEAR'].mode(dropna=True)
        df['V_YEAR'] = df['V_YEAR'].fillna(v.iloc[0] if len(v) > 0 else df['V_YEAR'].median())
    
    for col in ['C_MNTH', 'C_WDAY']:
        if df[col].isnull().sum() > 0:
            v = df[col].mode(dropna=True)
            df[col] = df[col].fillna(v.iloc[0] if len(v) > 0 else df[col].median())
    
    return df

# ============================================================
# 4. GENERATE PREPROCESSED DATASETS
# ============================================================
for sample_size in SAMPLE_SIZES:
    print("\n" + "=" * 60)
    print(f"PROCESSING {sample_size:,} SAMPLES")
    print("=" * 60)
    
    # Sample data
    df_sampled = df_full.sample(n=min(sample_size, len(df_full)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled: {len(df_sampled):,} rows (fatal={df_sampled['Fatality'].sum():,}, rate={df_sampled['Fatality'].mean()*100:.3f}%)")
    
    # Story imputation
    print("  Story imputation...")
    df_sampled = story_imputation(df_sampled)
    
    # MICE imputation
    print("  MICE imputation...")
    df_sampled = mice_imputation(df_sampled, n_iter=5)
    
    # Simple imputation
    print("  Simple imputation...")
    df_sampled = simple_imputation(df_sampled)
    
    # Check for remaining missing values
    missing = df_sampled.isnull().sum()
    if missing.sum() > 0:
        print(f"  Warning: {missing.sum()} missing values remaining")
        print(missing[missing > 0])
    else:
        print("  No missing values remaining")
    
    # Save preprocessed dataset
    output_path = os.path.join(OUTPUT_DIR, f"preprocessed_{sample_size}.csv")
    df_sampled.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)
print(f"Preprocessed datasets saved to: {OUTPUT_DIR}/")
print("Files:")
for sample_size in SAMPLE_SIZES:
    output_path = os.path.join(OUTPUT_DIR, f"preprocessed_{sample_size}.csv")
    print(f"  - preprocessed_{sample_size}.csv")
