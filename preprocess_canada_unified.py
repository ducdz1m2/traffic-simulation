import os
import glob
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from evidently import Report
from evidently.metrics import ValueDrift, CorrelationMatrix

os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

DATA_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-08\code\data\canada-dataset"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-08\code\data"

SAMPLE_SIZE = 0.1
RANDOM_STATE = 1
MICE_ITER = 5

OUTPUT_FILE_RAW = os.path.join(OUTPUT_DIR, "canada-dataset-sampled-raw.csv")
OUTPUT_FILE_IMPUTED = os.path.join(OUTPUT_DIR, "canada-dataset-sampled-imputed.csv")

VIZ_DIR = os.path.join(OUTPUT_DIR, "viz_distribution")
os.makedirs(VIZ_DIR, exist_ok=True)

NA_CODES = {"U", "UU", "UUUU", "X", "XX", "XXXX", "N", "NN", "NNNN", "Q", "QQ"}
MICE_COLS = ['C_WTHR', 'C_RSUR', 'C_CONF', 'C_RCFG', 'C_RALN',
             'C_TRAF', 'V_TYPE', 'P_SEX', 'P_USER', 'P_PSN', 'P_SAFE']

cat_cols = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS',
            'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF',
            'V_TYPE', 'P_SEX', 'P_PSN', 'P_SAFE', 'P_USER', 'Fatality']
num_cols = ['P_AGE', 'V_YEAR']


def prepare_for_evidently(df):
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0 and s.dtype.kind in 'if':
                df[col] = s.astype(int).astype(str)
            else:
                df[col] = s.astype(str)
    return df


def evaluate_with_evidently(df_pop, df_samp):
    num_cols_eff = [c for c in num_cols if c in df_pop.columns and c in df_samp.columns]
    cat_cols_eff = [c for c in cat_cols if c in df_pop.columns and c in df_samp.columns]

    metrics = []
    for col in num_cols_eff:
        metrics.append(ValueDrift(column=col, method='ks'))
        metrics.append(ValueDrift(column=col, method='wasserstein'))

    for col in cat_cols_eff:
        metrics.append(ValueDrift(column=col, method='chisquare'))
        metrics.append(ValueDrift(column=col, method='jensenshannon'))

    metrics.append(CorrelationMatrix(kind='pearson'))

    df_pop_str = prepare_for_evidently(df_pop)
    df_samp_str = prepare_for_evidently(df_samp)

    report = Report(metrics=metrics)
    snapshot = report.run(reference_data=df_pop_str, current_data=df_samp_str)

    html_path = os.path.join(VIZ_DIR, "evidently_report.html")
    snapshot.save_html(html_path)
    print(f"  Evidently HTML report saved to: {html_path}")

    results = []
    for metric in report.items():
        fp = metric.get_fingerprint()
        result = snapshot.metric_results.get(fp)
        if result is None:
            continue
        cname = getattr(metric, 'column', None)
        mname = getattr(metric, 'method', None)
        if cname is None or mname is None:
            continue
        val = result.value
        if isinstance(val, pd.DataFrame):
            continue
        mtype = 'numerical' if mname in ('ks', 'wasserstein') else 'categorical'
        results.append({
            'column': cname,
            'method': mname,
            'type': mtype,
            'drift_score': float(val),
        })

    numeric_for_corr = num_cols_eff + [c for c in cat_cols_eff
                                        if c in df_pop.columns
                                        and pd.to_numeric(df_pop[c], errors='coerce').notna().sum() > 0]
    if len(numeric_for_corr) >= 2:
        pop_sub = df_pop[numeric_for_corr].apply(pd.to_numeric, errors='coerce')
        samp_sub = df_samp[numeric_for_corr].apply(pd.to_numeric, errors='coerce')
        corr_pop = pop_sub.corr().values
        corr_samp = samp_sub.corr().values
        diff = corr_pop - corr_samp
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        frob = float(np.linalg.norm(diff, 'fro'))
        results.append({'column': 'correlation_matrix', 'method': 'rmse',
                        'type': 'correlation', 'drift_score': rmse})
        results.append({'column': 'correlation_matrix', 'method': 'frobenius',
                        'type': 'correlation', 'drift_score': frob})

    return results, snapshot


def plot_correlation_heatmap(df_pop, df_samp):
    numeric_for_corr = [c for c in num_cols if c in df_pop.columns and c in df_samp.columns]
    cat_numeric = [c for c in cat_cols if c in df_pop.columns
                   and pd.to_numeric(df_pop[c], errors='coerce').notna().sum() > 0]
    numeric_for_corr = numeric_for_corr + cat_numeric
    if len(numeric_for_corr) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pop_sub = df_pop[numeric_for_corr].apply(pd.to_numeric, errors='coerce')
    samp_sub = df_samp[numeric_for_corr].apply(pd.to_numeric, errors='coerce')
    sns.heatmap(pop_sub.corr(), annot=True, fmt='.2f', ax=axes[0], cmap='coolwarm')
    axes[0].set_title('Population Correlation')
    sns.heatmap(samp_sub.corr(), annot=True, fmt='.2f', ax=axes[1], cmap='coolwarm')
    axes[1].set_title('Sample Correlation')
    plt.tight_layout()
    fp = os.path.join(VIZ_DIR, "correlation_comparison.png")
    plt.savefig(fp, dpi=150)
    plt.close(fig)


def print_drift_report(results):
    print("\n" + "=" * 70)
    print("EVIDENTLY DRIFT REPORT - Sample vs Population")
    print("=" * 70)
    for m in ['ks', 'chisquare', 'wasserstein', 'jensenshannon']:
        subset = [r for r in results if r['method'] == m]
        if not subset:
            continue
        print(f"\n  [{m}]")
        for r in subset:
            print(f"    {r['column']:12s}  drift_score = {r['drift_score']:.6f}")
    corr = [r for r in results if r['type'] == 'correlation']
    if corr:
        print(f"\n  [correlation]")
        for r in corr:
            print(f"    {r['method']:12s}  drift_score = {r['drift_score']:.6f}")
    print(f"\n{'=' * 70}")
    print(f"Total checks: {len(results)}")
    print(f"{'=' * 70}")


def clean_numeric(series):
    return pd.to_numeric(series.replace(NA_CODES, np.nan), errors="coerce")


def preprocess(df):
    df['P_SEX'] = df['P_SEX'].replace({'M': 1, 'F': 0})
    df['P_SEX'] = pd.to_numeric(df['P_SEX'], errors='coerce').fillna(0).astype(int)

    num_cols_all = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS',
                    'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF',
                    'V_ID', 'V_TYPE', 'V_YEAR', 'P_ID', 'P_AGE', 'P_PSN',
                    'P_SEX', 'P_SAFE', 'P_ISEV', 'P_USER']

    for col in num_cols_all:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    df['Fatality'] = (df['P_ISEV'] == 3).astype(int)
    print(f"Fatality: 0={(df['Fatality']==0).sum():,}  1={(df['Fatality']==1).sum():,}  rate={df['Fatality'].mean()*100:.3f}%")
    return df


def take_sample(df, size, random_state):
    df = df.copy()
    df = df.dropna(subset=['Fatality'])

    if size is None or size == 0 or size >= len(df):
        print("Using FULL dataset (no sampling)")
        return df.reset_index(drop=True)
    elif isinstance(size, float) and 0 < size < 1:
        actual_size = int(len(df) * size)
        print(f"Using {size*100:.1f}% of dataset: {actual_size:,} samples")
    else:
        actual_size = min(int(size), len(df))
        print(f"Using {actual_size:,} samples")

    valid_strata = df['Fatality'].value_counts()
    valid_strata = valid_strata[valid_strata >= 2].index
    df = df[df['Fatality'].isin(valid_strata)]

    actual_size = min(actual_size, len(df))
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


def main():
    print("=" * 60)
    print("CANADA DATASET PREPROCESSING - UNIFIED")
    print("=" * 60)
    print(f"Sample size config: {SAMPLE_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv*")))
    df_list = []

    total_original_rows = 0
    if SAMPLE_SIZE is not None and SAMPLE_SIZE > 0:
        if isinstance(SAMPLE_SIZE, float) and 0 < SAMPLE_SIZE < 1:
            total_rows = 0
            for f in files:
                with pd.read_csv(f, dtype=str, low_memory=False, chunksize=10000) as reader:
                    chunk_count = sum(len(chunk) for chunk in reader)
                total_rows += chunk_count
            total_original_rows = total_rows
            target_total = int(total_rows * SAMPLE_SIZE)
            samples_per_file = max(1, target_total // len(files))
            print(f"Original dataset size: {total_rows:,} rows")
            print(f"Target total: {target_total:,} samples ({SAMPLE_SIZE*100:.1f}%)")
            print(f"Samples per file: {samples_per_file:,}")
        else:
            total_rows = 0
            for f in files:
                with pd.read_csv(f, dtype=str, low_memory=False, chunksize=10000) as reader:
                    chunk_count = sum(len(chunk) for chunk in reader)
                total_rows += chunk_count
            total_original_rows = total_rows
            samples_per_file = max(1, int(SAMPLE_SIZE) // len(files))
            print(f"Original dataset size: {total_rows:,} rows")
            print(f"Target total: {SAMPLE_SIZE:,} samples")
            print(f"Samples per file: {samples_per_file:,}")
    else:
        samples_per_file = None
        for f in files:
            with pd.read_csv(f, dtype=str, low_memory=False, chunksize=10000) as reader:
                chunk_count = sum(len(chunk) for chunk in reader)
            total_original_rows += chunk_count

    for f in files:
        df = pd.read_csv(f, dtype=str, low_memory=False)
        print(f"  {os.path.basename(f):30s} {len(df):>8,} rows")

        if samples_per_file is not None and samples_per_file < len(df):
            df = df.sample(n=min(samples_per_file, len(df)), random_state=RANDOM_STATE)
            print(f"    -> Sampled: {len(df):,} rows")

        df_list.append(df)

    df_full = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal after concat: {len(df_full):,} rows")

    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    df_full = preprocess(df_full)

    print("\n" + "=" * 60)
    print("SAMPLING")
    print("=" * 60)
    df_sampled = take_sample(df_full, SAMPLE_SIZE, RANDOM_STATE)

    if SAMPLE_SIZE is None or SAMPLE_SIZE == 0:
        is_sample_mode = False
    elif isinstance(SAMPLE_SIZE, float) and 0 < SAMPLE_SIZE < 1:
        is_sample_mode = True
    else:
        is_sample_mode = SAMPLE_SIZE < total_original_rows

    print(f"\nSample mode: {is_sample_mode} (original={total_original_rows:,}, current={len(df_sampled):,})")

    if is_sample_mode:
        results, snapshot = evaluate_with_evidently(df_full, df_sampled)
        print_drift_report(results)
        plot_correlation_heatmap(df_full, df_sampled)
    else:
        print("\n" + "=" * 60)
        print("SKIPPING VALIDATION (FULL DATASET)")
        print("=" * 60)

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
    if is_sample_mode:
        print(f"Visualizations: {VIZ_DIR}/")

if __name__ == "__main__":
    main()
