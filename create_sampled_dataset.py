import os
import glob
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import chisquare, ks_2samp, chi2_contingency

# R environment (bat dong set de tranh loi stats.dll)
os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\canada-dataset"
SAMPLE_SIZE = 100000
RANDOM_STATE = 1
OUTPUT_FILE_RAW = "data/canada-dataset-sampled-raw.csv"
OUTPUT_FILE_IMPUTED = "data/canada-dataset-sampled-imputed.csv"
MICE_ITER = 5

NA_CODES = {"U", "UU", "UUUU", "X", "XX", "XXXX", "N", "NN", "NNNN", "Q", "QQ"}

MICE_COLS = ['C_WTHR', 'C_RSUR', 'C_CONF', 'C_RCFG', 'C_RALN',
             'C_TRAF', 'V_TYPE', 'P_SEX', 'P_USER', 'P_PSN', 'P_SAFE']

# Columns to check
cat_cols = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS',
            'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF',
            'V_TYPE', 'P_SEX', 'P_PSN', 'P_SAFE', 'P_USER', 'Fatality']
num_cols = ['P_AGE', 'V_YEAR']

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def compute_psi(expected, actual, bins=10):
    eps = 1e-6
    psi = 0
    all_vals = np.concatenate([expected, actual])
    if all_vals.dtype.kind in 'if':
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bin_edges[-1] += eps
        bin_edges[0] -= eps
        exp_counts, _ = np.histogram(expected, bins=bin_edges)
        act_counts, _ = np.histogram(actual, bins=bin_edges)
        exp_pct = exp_counts / len(expected)
        act_pct = act_counts / len(actual)
    else:
        unique = np.union1d(np.unique(expected), np.unique(actual))
        exp_counts = np.array([np.sum(expected == u) for u in unique])
        act_counts = np.array([np.sum(actual == u) for u in unique])
        exp_pct = exp_counts / len(expected)
        act_pct = act_counts / len(actual)
    for e, a in zip(exp_pct, act_pct):
        e = max(e, eps)
        a = max(a, eps)
        psi += (a - e) * np.log(a / e)
    return psi

def evaluate_sample(pop_series, samp_series, col, col_type='cat'):
    if col_type == 'cat':
        pop_dist = pop_series.value_counts(normalize=True).sort_index()
        samp_dist = samp_series.value_counts(normalize=True).sort_index()
        common = pop_dist.index.intersection(samp_dist.index)
        if len(common) < 2:
            return None
        observed = (samp_dist[common] * len(samp_series)).fillna(0).values
        expected_prop = (pop_dist[common] / pop_dist[common].sum()).values
        expected = expected_prop * len(samp_series)
        _, p_chi = chisquare(observed, expected)
        ct = pd.crosstab(pop_series, samp_series, dropna=False)
        cv = cramers_v(ct.values) if ct.shape == (2, 2) else 0
        psi = compute_psi(pop_series.dropna().values, samp_series.dropna().values)
        return {'column': col, 'type': 'categorical', 'chi2_p': p_chi,
                'cramers_v': cv, 'psi': psi}
    else:
        pop_vals = pop_series.dropna()
        samp_vals = samp_series.dropna()
        if len(pop_vals) < 5 or len(samp_vals) < 5:
            return None
        ks_stat, ks_p = ks_2samp(pop_vals, samp_vals)
        psi = compute_psi(pop_vals.values, samp_vals.values)
        return {'column': col, 'type': 'numeric', 'ks_stat': ks_stat,
                'ks_p': ks_p, 'psi': psi}

def clean_numeric(series):
    return pd.to_numeric(series.replace(NA_CODES, np.nan), errors="coerce")

def preprocess(df):
    df['P_SEX'] = df['P_SEX'].replace({'M': 1, 'F': 0})
    df['P_SEX'] = pd.to_numeric(df['P_SEX'], errors='coerce').fillna(0).astype(int)
    
    num_cols = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS',
                'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF',
                'V_ID', 'V_TYPE', 'V_YEAR', 'P_ID', 'P_AGE', 'P_PSN',
                'P_SEX', 'P_SAFE', 'P_ISEV', 'P_USER']
    
    for col in num_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    
    df['Fatality'] = (df['P_ISEV'] == 3).astype(int)
    print(f"Fatality: 0={(df['Fatality']==0).sum():,}  1={(df['Fatality']==1).sum():,}  rate={df['Fatality'].mean()*100:.3f}%")
    return df

def take_sample(df, size, random_state):
    df = df.copy()
    df = df.dropna(subset=['Fatality'])
    
    # Ensure enough samples per stratum
    valid_strata = df['Fatality'].value_counts()
    valid_strata = valid_strata[valid_strata >= 2].index
    df = df[df['Fatality'].isin(valid_strata)]
    
    actual_size = min(size, len(df))
    df_sample, _ = train_test_split(
        df, train_size=actual_size,
        stratify=df['Fatality'],
        random_state=random_state
    )
    print(f"Sampled {len(df_sample):,} (fatality rate={df_sample['Fatality'].mean()*100:.3f}%)")
    return df_sample.reset_index(drop=True)

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

    print("Story imputation done")
    return df

def mice_imputation(df, n_iter=MICE_ITER):
    print(f"MICE imputation (n_iter={n_iter})...")
    
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

        print("MICE (R polyreg) done")

    except Exception as e:
        print(f"MICE via R failed: {e}")
        print("Falling back to IterativeImputer (RandomForest)...")
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            from sklearn.ensemble import RandomForestClassifier

            mice_df = df[MICE_COLS].copy()
            imp = IterativeImputer(
                estimator=RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
                max_iter=n_iter, random_state=RANDOM_STATE, initial_strategy='most_frequent'
            )
            imputed = imp.fit_transform(mice_df)
            imputed = pd.DataFrame(imputed, columns=MICE_COLS).round(0).astype(int)
            for col in MICE_COLS:
                df[col] = imputed[col].values
            print("IterativeImputer fallback done")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

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
    
    print("Simple imputation done")
    return df

# ============================================================
# MAIN
# ============================================================
print("=" * 60)
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

print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)
df_full = preprocess(df_full)

print("\n" + "=" * 60)
print("SAMPLING")
print("=" * 60)
df_sampled = take_sample(df_full, SAMPLE_SIZE, RANDOM_STATE)

print("\n" + "=" * 60)
print("VALIDATION - Sample vs Population")
print("=" * 60)

# Use reference dataframe (dropna for validation)
df_ref = df_full.dropna(subset=['Fatality', 'P_ISEV']).copy()

all_results = []
failed_cols = []

# Categorical checks
for col in cat_cols:
    if col in df_ref.columns and col in df_sampled.columns:
        result = evaluate_sample(df_ref[col], df_sampled[col], col, 'cat')
        if result:
            all_results.append(result)
            print(f"  {col:12s} | Chi2 p={result['chi2_p']:.4f} | CV={result['cramers_v']:.4f} | PSI={result['psi']:.6f}")
            if result['chi2_p'] < 0.05:
                failed_cols.append(f"{col} (chi2_p={result['chi2_p']:.4f})")

# Numeric checks
for col in num_cols:
    if col in df_ref.columns and col in df_sampled.columns:
        result = evaluate_sample(df_ref[col], df_sampled[col], col, 'num')
        if result:
            all_results.append(result)
            print(f"  {col:12s} | KS D={result['ks_stat']:.4f} | KS p={result['ks_p']:.4f} | PSI={result['psi']:.6f}")
            if result['ks_p'] < 0.05:
                failed_cols.append(f"{col} (ks_p={result['ks_p']:.4f})")

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
if failed_cols:
    print(f"FAILED columns (p < 0.05): {len(failed_cols)}")
    for col in failed_cols:
        print(f"  - {col}")
else:
    print("ALL COLUMNS PASSED (p >= 0.05)")

print(f"\nTotal checks: {len(all_results)}")
print(f"Passed: {len(all_results) - len(failed_cols)}")
print(f"Failed: {len(failed_cols)}")

print("\n" + "=" * 60)
print("SAVING RAW SAMPLE")
print("=" * 60)
os.makedirs(os.path.dirname(OUTPUT_FILE_RAW), exist_ok=True)
df_sampled.to_csv(OUTPUT_FILE_RAW, index=False)
print(f"Saved to: {OUTPUT_FILE_RAW}")
print(f"Shape: {df_sampled.shape}")
print(f"Columns: {list(df_sampled.columns)}")

print("\n" + "=" * 60)
print("IMPUTATION")
print("=" * 60)
df_imputed = df_sampled.copy()
df_imputed = story_imputation(df_imputed)
df_imputed = mice_imputation(df_imputed, MICE_ITER)
df_imputed = simple_imputation(df_imputed)

print("\n" + "=" * 60)
print("SAVING IMPUTED SAMPLE")
print("=" * 60)
os.makedirs(os.path.dirname(OUTPUT_FILE_IMPUTED), exist_ok=True)
df_imputed.to_csv(OUTPUT_FILE_IMPUTED, index=False)
print(f"Saved to: {OUTPUT_FILE_IMPUTED}")
print(f"Shape: {df_imputed.shape}")
print(f"Columns: {list(df_imputed.columns)}")

# Check remaining missing values
missing_counts = df_imputed.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]
if len(missing_cols) > 0:
    print(f"\nRemaining missing values: {len(missing_cols)} columns")
    print(missing_cols)
else:
    print("\nNo missing values remaining")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Raw sample: {OUTPUT_FILE_RAW}")
print(f"Imputed sample: {OUTPUT_FILE_IMPUTED}")
