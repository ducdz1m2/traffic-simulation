import os, sys, glob, warnings, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# R environment (bat dong set de tranh loi stats.dll)
os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.sparse import csr_matrix, hstack
from scipy.stats import wasserstein_distance, ks_2samp
import umap.umap_ as umap
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================
# CONFIG
# ============================================================
RAW_DATA_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\canada-dataset-sampled-raw.csv"
IMPUTED_DATA_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\canada-dataset-sampled-imputed.csv"
RANDOM_STATE = 1
SEEDS = [1, 2, 3]
SAMPLE_SIZE = 200000  

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS',
                'V_TYPE', 'V_YEAR', 'P_SEX', 'P_AGE', 'P_PSN', 'P_USER']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE']
CATEGORICAL_COLS = ['V_TYPE', 'P_SEX']

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    print(f"Loading raw data from: {RAW_DATA_FILE}")
    df_raw = pd.read_csv(RAW_DATA_FILE, low_memory=False)
    print(f"Raw data: {len(df_raw):,} records")
    print(f"Raw data dtypes:\n{df_raw[FEATURE_COLS + NOMINAL_COLS + ['Fatality']].dtypes}")
    
    print(f"Loading imputed data from: {IMPUTED_DATA_FILE}")
    df_imputed = pd.read_csv(IMPUTED_DATA_FILE, low_memory=False)
    print(f"Imputed data: {len(df_imputed):,} records")
    print(f"Imputed data dtypes:\n{df_imputed[FEATURE_COLS + NOMINAL_COLS + ['Fatality']].dtypes}")
    
    return df_raw, df_imputed



SAMPLER_NAMES = ['under', 'rose', 'mixed', 'smote', 'borderline', 'smote_tomek', 'adasyn', 'nearmiss']

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
    except Exception as e:
        print(f"  ROSE R failed: {e}, falling back to SMOTE")
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

def compute_metrics(X, y, X_2d=None):
    """Compute various metrics to evaluate class separation/overlap."""
    metrics = {}
    n_samples = len(y)
    
    # Silhouette Score (-1 to 1, higher is better)
    if n_samples > 2 and len(np.unique(y)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(X, y)
        except:
            metrics['silhouette'] = np.nan
    else:
        metrics['silhouette'] = np.nan
    
    # Davies-Bouldin Index (>=0, lower is better)
    if n_samples > 2 and len(np.unique(y)) > 1:
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X, y)
        except:
            metrics['davies_bouldin'] = np.nan
    else:
        metrics['davies_bouldin'] = np.nan
    
    # Calinski-Harabasz Index (>=0, higher is better)
    if n_samples > 2 and len(np.unique(y)) > 1:
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, y)
        except:
            metrics['calinski_harabasz'] = np.nan
    else:
        metrics['calinski_harabasz'] = np.nan
    
    # Wasserstein Distance between class 0 and class 1 (>=0, lower is better)
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    if len(X_0) > 0 and len(X_1) > 0:
        try:
            # Compute average Wasserstein distance across all features
            wass_dists = []
            for i in range(X.shape[1]):
                wass = wasserstein_distance(X_0[:, i], X_1[:, i])
                wass_dists.append(wass)
            metrics['wasserstein'] = np.mean(wass_dists)
        except:
            metrics['wasserstein'] = np.nan
    else:
        metrics['wasserstein'] = np.nan
    
    # KS Statistic (0 to 1, lower is better)
    if len(X_0) > 0 and len(X_1) > 0:
        try:
            ks_stats = []
            for i in range(X.shape[1]):
                ks_stat, _ = ks_2samp(X_0[:, i], X_1[:, i])
                ks_stats.append(ks_stat)
            metrics['ks_statistic'] = np.mean(ks_stats)
        except:
            metrics['ks_statistic'] = np.nan
    else:
        metrics['ks_statistic'] = np.nan
    
    # Overlap Coefficient (0 to 1, lower is better)
    # Based on 2D embedding if provided
    if X_2d is not None and len(X_2d) > 0:
        try:
            from scipy.stats import gaussian_kde
            X_0_2d = X_2d[y == 0]
            X_1_2d = X_2d[y == 1]
            
            if len(X_0_2d) > 10 and len(X_1_2d) > 10:
                # Estimate KDE for each class
                kde_0 = gaussian_kde(X_0_2d.T)
                kde_1 = gaussian_kde(X_1_2d.T)
                
                # Sample points to estimate overlap
                min_x = min(X_2d[:, 0].min(), X_2d[:, 1].min())
                max_x = max(X_2d[:, 0].max(), X_2d[:, 1].max())
                grid = np.linspace(min_x, max_x, 100)
                mesh = np.array(np.meshgrid(grid, grid)).T.reshape(-1, 2)
                
                p0 = kde_0(mesh.T)
                p1 = kde_1(mesh.T)
                
                # Overlap coefficient
                overlap = np.sum(np.minimum(p0, p1)) / np.sum(np.maximum(p0, p1) + 1e-10)
                metrics['overlap_coefficient'] = overlap
            else:
                metrics['overlap_coefficient'] = np.nan
        except:
            metrics['overlap_coefficient'] = np.nan
    else:
        metrics['overlap_coefficient'] = np.nan
    
    return metrics


# ============================================================
# 3. FEATURE PREPARATION
# ============================================================
def prepare_features(df):
    m = df[FEATURE_COLS + NOMINAL_COLS + ['Fatality']].copy()
    
    # Convert all columns to numeric where possible
    for col in FEATURE_COLS + NOMINAL_COLS:
        m[col] = pd.to_numeric(m[col], errors='coerce')
    
    # Encode categorical columns
    for col in CATEGORICAL_COLS:
        m[col] = m[col].astype(str)
        m[col] = LabelEncoder().fit_transform(m[col])

    m = m.dropna().reset_index(drop=True)
    y = m['Fatality'].astype(int).values
    X = m[FEATURE_COLS + NOMINAL_COLS].values.astype(np.float32)

    print(f"Features: {X.shape[1]}  Samples: {len(y):,}  "
          f"Fatality rate: {y.mean()*100:.3f}%")
    return X, y

# ============================================================
# 5. VISUALIZATION
# ============================================================
OUT_DIR = "outputs_fcm"
os.makedirs(OUT_DIR, exist_ok=True)
RAW_IMPUTED_DIR = os.path.join(OUT_DIR, "raw_vs_imputed")
SAMPLING_DIR = os.path.join(OUT_DIR, "sampling_techniques")
os.makedirs(RAW_IMPUTED_DIR, exist_ok=True)
os.makedirs(SAMPLING_DIR, exist_ok=True)

def plot_embedding(X_2d, y, title, filename, alpha=0.4, output_dir=OUT_DIR):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {0: '#1f77b4', 1: '#d62728'}
    labels = {0: 'Non-Fatality', 1: 'Fatality'}

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[cls], label=f"{labels[cls]} (n={mask.sum():,})",
                   alpha=alpha, s=3, edgecolors='none')

    ax.set_title(title, fontsize=14)
    ax.legend(markerscale=5, fontsize=10)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close(fig)


def run():
    t0 = time.time()

    # 1. Load data
    print("=" * 60)
    print("  LOAD DATA")
    print("=" * 60)
    df_raw, df_imputed = load_data()

    # Visualization: RAW vs IMPUTED comparison
    print("\n" + "=" * 60)
    print("  VISUALIZATION: RAW vs IMPUTED")
    print("=" * 60)
    
    # Sample from raw data
    df_raw_sample = df_raw.sample(n=min(SAMPLE_SIZE, len(df_raw)), random_state=RANDOM_STATE)
    X_raw_numeric, y_raw = prepare_features(df_raw_sample)
    
    # Sample from imputed data
    df_imp_sample = df_imputed.sample(n=min(SAMPLE_SIZE, len(df_imputed)), random_state=RANDOM_STATE)
    X_imp_numeric, y_imp = prepare_features(df_imp_sample)
    
    # Scale both datasets ONLY for visualization
    scaler_raw = StandardScaler()
    X_raw_scaled = scaler_raw.fit_transform(X_raw_numeric)
    
    scaler_imp = StandardScaler()
    X_imp_scaled = scaler_imp.fit_transform(X_imp_numeric)
    
    # PCA for raw data
    print("\n>>> PCA (Raw Data) ...")
    pca_raw = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_raw = pca_raw.fit_transform(X_raw_scaled)
    var_exp_raw = pca_raw.explained_variance_ratio_
    print(f"  Explained variance: PC1={var_exp_raw[0]:.4f}  PC2={var_exp_raw[1]:.4f}  total={var_exp_raw.sum():.4f}")
    plot_embedding(X_pca_raw, y_raw, "PCA - Raw Data", "pca_raw.png", output_dir=RAW_IMPUTED_DIR)
    
    # PCA for imputed data
    print("\n>>> PCA (Imputed Data) ...")
    pca_imp = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_imp = pca_imp.fit_transform(X_imp_scaled)
    var_exp_imp = pca_imp.explained_variance_ratio_
    print(f"  Explained variance: PC1={var_exp_imp[0]:.4f}  PC2={var_exp_imp[1]:.4f}  total={var_exp_imp.sum():.4f}")
    plot_embedding(X_pca_imp, y_imp, "PCA - Imputed Data", "pca_imputed.png", output_dir=RAW_IMPUTED_DIR)
    
    # t-SNE for raw data
    print("\n>>> t-SNE (Raw Data) ...")
    tsne_raw = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        learning_rate="auto",
        init="random",
        random_state=RANDOM_STATE
    )
    X_tsne_raw = tsne_raw.fit_transform(X_raw_scaled)
    plot_embedding(X_tsne_raw, y_raw, "t-SNE - Raw Data", "tsne_raw.png", output_dir=RAW_IMPUTED_DIR)
    
    # t-SNE for imputed data
    print("\n>>> t-SNE (Imputed Data) ...")
    tsne_imp = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        learning_rate="auto",
        init="random",
        random_state=RANDOM_STATE
    )
    X_tsne_imp = tsne_imp.fit_transform(X_imp_scaled)
    plot_embedding(X_tsne_imp, y_imp, "t-SNE - Imputed Data", "tsne_imputed.png", output_dir=RAW_IMPUTED_DIR)
    
    # UMAP for raw data
    print("\n>>> UMAP (Raw Data) ...")
    reducer_raw = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                              n_neighbors=30, min_dist=0.3)
    X_umap_raw = reducer_raw.fit_transform(X_raw_scaled)
    plot_embedding(X_umap_raw, y_raw, "UMAP - Raw Data", "umap_raw.png", output_dir=RAW_IMPUTED_DIR)
    
    # UMAP for imputed data
    print("\n>>> UMAP (Imputed Data) ...")
    reducer_imp = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                              n_neighbors=30, min_dist=0.3)
    X_umap_imp = reducer_imp.fit_transform(X_imp_scaled)
    plot_embedding(X_umap_imp, y_imp, "UMAP - Imputed Data", "umap_imputed.png", output_dir=RAW_IMPUTED_DIR)
    
    # Compute and compare metrics
    print("\n" + "=" * 60)
    print("  METRICS COMPARISON: RAW vs IMPUTED")
    print("=" * 60)
    
    metrics_raw = compute_metrics(X_raw_scaled, y_raw, X_umap_raw)
    metrics_raw['dataset'] = 'raw'
    metrics_raw['n_samples'] = len(y_raw)
    metrics_raw['n_fatal'] = y_raw.sum()
    metrics_raw['fatality_rate'] = y_raw.mean()
    
    metrics_imp = compute_metrics(X_imp_scaled, y_imp, X_umap_imp)
    metrics_imp['dataset'] = 'imputed'
    metrics_imp['n_samples'] = len(y_imp)
    metrics_imp['n_fatal'] = y_imp.sum()
    metrics_imp['fatality_rate'] = y_imp.mean()
    
    print(f"\nRaw Data Metrics:")
    print(f"  Silhouette={metrics_raw.get('silhouette', np.nan):.4f}  "
          f"DB={metrics_raw.get('davies_bouldin', np.nan):.4f}  "
          f"CH={metrics_raw.get('calinski_harabasz', np.nan):.4f}  "
          f"Wasserstein={metrics_raw.get('wasserstein', np.nan):.4f}  "
          f"KS={metrics_raw.get('ks_statistic', np.nan):.4f}  "
          f"Overlap={metrics_raw.get('overlap_coefficient', np.nan):.4f}")
    
    print(f"\nImputed Data Metrics:")
    print(f"  Silhouette={metrics_imp.get('silhouette', np.nan):.4f}  "
          f"DB={metrics_imp.get('davies_bouldin', np.nan):.4f}  "
          f"CH={metrics_imp.get('calinski_harabasz', np.nan):.4f}  "
          f"Wasserstein={metrics_imp.get('wasserstein', np.nan):.4f}  "
          f"KS={metrics_imp.get('ks_statistic', np.nan):.4f}  "
          f"Overlap={metrics_imp.get('overlap_coefficient', np.nan):.4f}")
    
    # Save comparison metrics
    df_metrics_comp = pd.DataFrame([metrics_raw, metrics_imp])
    metrics_comp_path = os.path.join(OUT_DIR, "raw_vs_imputed_metrics.csv")
    df_metrics_comp.to_csv(metrics_comp_path, index=False)
    print(f"\nComparison metrics saved to: {metrics_comp_path}")

    # Continue with sampling techniques on imputed data
    print("\n" + "=" * 60)
    print("  SAMPLING TECHNIQUES ON IMPUTED DATA")
    print("=" * 60)
    
    # Use NUMERIC data (NOT scaled) for sampling
    X_numeric, y = X_imp_numeric, y_imp
    
    print(f"\nFeature matrix (numeric): {X_numeric.shape}")
    print(f"Fatality: 0={(y==0).sum():,}  1={(y==1).sum():,}")

    # Fit scaler on original numeric data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Fit PCA and UMAP on scaled data (for visualization)
    print("\n" + "=" * 60)
    print("  FIT PCA/UMAP ON SCALED DATA")
    print("=" * 60)
    
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_scaled)
    var_exp = pca.explained_variance_ratio_
    print(f"PCA fitted: PC1={var_exp[0]:.4f}  PC2={var_exp[1]:.4f}  total={var_exp.sum():.4f}")
    
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                        n_neighbors=30, min_dist=0.3)
    reducer.fit(X_scaled)
    print("UMAP fitted on scaled data")

    # ---- Visualization with different sampling techniques ----
    print("\n" + "=" * 60)
    print("  VISUALIZATION WITH DIFFERENT SAMPLING TECHNIQUES")
    print("=" * 60)
    
    metrics_results = []
    
    for sampler_name in SAMPLER_NAMES:
        print(f"\n>>> {sampler_name.upper()} SAMPLING ...")
        # Apply sampling on NUMERIC data
        X_s_numeric, y_s = apply_sampler(X_numeric, y, sampler_name, RANDOM_STATE)
        print(f"  Samples: {len(y_s):,}  (fatal={y_s.sum():,}  rate={y_s.mean()*100:.3f}%)")
        
        # Scale sampled data for visualization
        X_s_scaled = scaler.transform(X_s_numeric)
        
        # Transform using fitted PCA
        X_pca = pca.transform(X_s_scaled)
        plot_embedding(X_pca, y_s, f"PCA - {sampler_name.upper()} Sampling", f"pca_{sampler_name}.png", alpha=0.4, output_dir=SAMPLING_DIR)
        
        # Transform using fitted UMAP
        X_umap = reducer.transform(X_s_scaled)
        plot_embedding(X_umap, y_s, f"UMAP - {sampler_name.upper()} Sampling", f"umap_{sampler_name}.png", alpha=0.4, output_dir=SAMPLING_DIR)
        
        # Compute metrics on scaled data
        metrics = compute_metrics(X_s_scaled, y_s, X_umap)
        metrics['sampler'] = sampler_name
        metrics['n_samples'] = len(y_s)
        metrics['n_fatal'] = y_s.sum()
        metrics['fatality_rate'] = y_s.mean()
        metrics_results.append(metrics)
        
        print(f"  Metrics: Silhouette={metrics.get('silhouette', np.nan):.4f}  "
              f"DB={metrics.get('davies_bouldin', np.nan):.4f}  "
              f"CH={metrics.get('calinski_harabasz', np.nan):.4f}  "
              f"Wasserstein={metrics.get('wasserstein', np.nan):.4f}  "
              f"KS={metrics.get('ks_statistic', np.nan):.4f}  "
              f"Overlap={metrics.get('overlap_coefficient', np.nan):.4f}")
    
    # Save metrics to CSV
    df_metrics = pd.DataFrame(metrics_results)
    metrics_path = os.path.join(OUT_DIR, "sampling_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("  METRICS SUMMARY")
    print("=" * 60)
    metric_cols = ['sampler', 'n_samples', 'n_fatal', 'fatality_rate', 
                   'silhouette', 'davies_bouldin', 'calinski_harabasz', 
                   'wasserstein', 'ks_statistic', 'overlap_coefficient']
    print(df_metrics[metric_cols].to_string(index=False))

    print(f"\nDone! Total time: {time.time()-t0:.1f}s")
    print(f"Plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    run()
