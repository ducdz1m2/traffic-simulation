import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
import pickle

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TRAIN_DATA_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\train.csv"
TEST_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\test.csv"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\outputs\rose_evaluation"
RANDOM_STATE = 42

# Feature columns
FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

# ROSE strategies to test
ROSE_STRATEGIES = [0.3, 0.5, 0.8, 1.0, 1.5]

# R environment
os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

# ============================================================
# ROSE SAMPLING FUNCTION
# ============================================================
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

def reconstruct_dataframe(X_sampled, y_sampled, original_df):
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
# VISUALIZATION
# ============================================================
def plot_class_distribution(y_original, y_sampled_dict, output_path):
    """Plot class distribution before and after sampling"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original distribution
    ax = axes[0]
    unique, counts = np.unique(y_original, return_counts=True)
    ax.bar(['Non-Fatal', 'Fatal'], counts, color=['blue', 'red'], alpha=0.7)
    ax.set_title('Original Data', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    for i, v in enumerate(counts):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)
    
    imbalance_ratio = counts[0] / max(counts[1], 1)
    ax.text(0.5, 0.95, f'Imbalance Ratio: {imbalance_ratio:.2f}', 
            transform=ax.transAxes, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Sampled distributions
    for idx, (strategy, y_sampled) in enumerate(y_sampled_dict.items()):
        if idx >= 5:
            break
        ax = axes[idx + 1]
        unique, counts = np.unique(y_sampled, return_counts=True)
        ax.bar(['Non-Fatal', 'Fatal'], counts, color=['blue', 'red'], alpha=0.7)
        ax.set_title(f'ROSE Strategy: {strategy}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        for i, v in enumerate(counts):
            ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)
        
        imbalance_ratio = counts[0] / max(counts[1], 1)
        ax.text(0.5, 0.95, f'Imbalance Ratio: {imbalance_ratio:.2f}', 
                transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide empty subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to: {output_path}")

def plot_pca_overlap(X_original, y_original, X_sampled_dict, y_sampled_dict, output_path):
    """Plot PCA visualization to show overlap"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original data
    ax = axes[0]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_original)
    
    scatter = ax.scatter(X_pca[y_original == 0, 0], X_pca[y_original == 0, 1], 
                        c='blue', alpha=0.5, label='Non-Fatal', s=10)
    scatter = ax.scatter(X_pca[y_original == 1, 0], X_pca[y_original == 1, 1], 
                        c='red', alpha=0.5, label='Fatal', s=10)
    ax.set_title('Original Data', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax.legend(fontsize=8)
    
    # Sampled data
    for idx, (strategy, X_sampled) in enumerate(X_sampled_dict.items()):
        if idx >= 5:
            break
        ax = axes[idx + 1]
        
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_sampled)
        
        y_sampled = y_sampled_dict[strategy]
        scatter = ax.scatter(X_pca[y_sampled == 0, 0], X_pca[y_sampled == 0, 1], 
                            c='blue', alpha=0.5, label='Non-Fatal', s=10)
        scatter = ax.scatter(X_pca[y_sampled == 1, 0], X_pca[y_sampled == 1, 1], 
                            c='red', alpha=0.5, label='Fatal', s=10)
        ax.set_title(f'ROSE Strategy: {strategy}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.legend(fontsize=8)
    
    # Hide empty subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA overlap plot saved to: {output_path}")

def calculate_overlap_metric(X, y):
    """Calculate a simple overlap metric based on nearest neighbors"""
    from sklearn.neighbors import NearestNeighbors
    
    # Separate classes
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    
    if len(X_0) == 0 or len(X_1) == 0:
        return 0.0
    
    # Sample if too large
    max_samples = 5000
    if len(X_0) > max_samples:
        idx = np.random.choice(len(X_0), max_samples, replace=False)
        X_0 = X_0[idx]
    if len(X_1) > max_samples:
        idx = np.random.choice(len(X_1), max_samples, replace=False)
        X_1 = X_1[idx]
    
    # Find nearest neighbors of class 0 in class 1
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_1)
    distances, _ = nbrs.kneighbors(X_0)
    
    # Average distance (lower = more overlap)
    avg_distance = np.mean(distances)
    
    return avg_distance

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
    print("  EVALUATE ROSE SAMPLING STRATEGIES")
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
    
    # Apply ROSE with different strategies
    print("\n" + "=" * 60)
    print("  APPLYING ROSE SAMPLING")
    print("=" * 60)
    
    X_sampled_dict = {}
    y_sampled_dict = {}
    df_sampled_dict = {}
    class_dist_info = []
    
    for strategy in ROSE_STRATEGIES:
        print(f"\n>>> ROSE Strategy: {strategy}")
        
        try:
            X_sampled, y_sampled = rose_r(X, y, strategy, RANDOM_STATE, nominal_indices)
            
            print(f"  Samples: {len(y_sampled):,}")
            print(f"  Non-Fatal: {(y_sampled==0).sum():,}")
            print(f"  Fatal: {(y_sampled==1).sum():,}")
            print(f"  Imbalance ratio: {(y_sampled==0).sum() / max((y_sampled==1).sum(), 1):.2f}")
            
            X_sampled_dict[strategy] = X_sampled
            y_sampled_dict[strategy] = y_sampled
            
            df_sampled = reconstruct_dataframe(X_sampled, y_sampled, df_prepared)
            df_sampled_dict[strategy] = df_sampled
            
            class_dist_info.append({
                'Strategy': strategy,
                'Total_Samples': len(y_sampled),
                'Non_Fatal': (y_sampled==0).sum(),
                'Fatal': (y_sampled==1).sum(),
                'Imbalance_Ratio': (y_sampled==0).sum() / max((y_sampled==1).sum(), 1),
                'Fatal_Rate': y_sampled.mean()
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save class distribution info
    df_class_dist = pd.DataFrame(class_dist_info)
    class_dist_path = os.path.join(OUTPUT_DIR, 'class_distribution.csv')
    df_class_dist.to_csv(class_dist_path, index=False)
    print(f"\nClass distribution info saved to: {class_dist_path}")
    
    # Visualizations
    print("\n" + "=" * 60)
    print("  CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Class distribution plot
    dist_plot_path = os.path.join(OUTPUT_DIR, 'class_distribution_plot.png')
    plot_class_distribution(y, y_sampled_dict, dist_plot_path)
    
    # PCA overlap plot
    pca_plot_path = os.path.join(OUTPUT_DIR, 'pca_overlap_plot.png')
    plot_pca_overlap(X, y, X_sampled_dict, y_sampled_dict, pca_plot_path)
    
    # Calculate overlap metrics
    print("\n" + "=" * 60)
    print("  CALCULATING OVERLAP METRICS")
    print("=" * 60)
    
    overlap_info = []
    original_overlap = calculate_overlap_metric(X, y)
    print(f"Original overlap metric: {original_overlap:.4f}")
    overlap_info.append({'Strategy': 'Original', 'Overlap_Metric': original_overlap})
    
    for strategy in X_sampled_dict.keys():
        overlap = calculate_overlap_metric(X_sampled_dict[strategy], y_sampled_dict[strategy])
        print(f"ROSE {strategy}: {overlap:.4f}")
        overlap_info.append({'Strategy': f'ROSE_{strategy}', 'Overlap_Metric': overlap})
    
    df_overlap = pd.DataFrame(overlap_info)
    overlap_path = os.path.join(OUTPUT_DIR, 'overlap_metrics.csv')
    df_overlap.to_csv(overlap_path, index=False)
    print(f"Overlap metrics saved to: {overlap_path}")
    
    # XGBoost evaluation
    print("\n" + "=" * 60)
    print("  XGBOOST EVALUATION")
    print("=" * 60)
    
    # Prepare test features with OHE
    print("Preparing test features with OHE...")
    X_test_ohe, y_test_ohe, ohe = prepare_features_with_ohe(
        df_test, FEATURE_COLS, NOMINAL_COLS, TARGET)
    print(f"Test: {len(y_test_ohe):,} (fatal={y_test_ohe.sum():,})")
    
    # Evaluate original data
    print("\n>>> Original Data")
    X_train_ohe_orig, y_train_ohe_orig, _ = prepare_features_with_ohe(
        df_prepared, FEATURE_COLS, NOMINAL_COLS, TARGET)
    results_orig = evaluate_xgboost(X_train_ohe_orig, X_test_ohe, y_train_ohe_orig, y_test_ohe, RANDOM_STATE)
    results_orig['Strategy'] = 'Original'
    print(f"  Accuracy: {results_orig['Accuracy']:.4f}")
    print(f"  Sensitivity: {results_orig['Sensitivity']:.4f}")
    print(f"  Specificity: {results_orig['Specificity']:.4f}")
    print(f"  F1: {results_orig['F1']:.4f}")
    print(f"  MCC: {results_orig['MCC']:.4f}")
    print(f"  G-mean: {results_orig['G_mean']:.4f}")
    print(f"  AUC-ROC: {results_orig['AUC_ROC']:.4f}")
    
    all_results = [results_orig]
    
    # Evaluate each ROSE strategy
    for strategy in df_sampled_dict.keys():
        print(f"\n>>> ROSE Strategy: {strategy}")
        df_sampled = df_sampled_dict[strategy]
        
        X_train_ohe, y_train_ohe, _ = prepare_features_with_ohe(
            df_sampled, FEATURE_COLS, NOMINAL_COLS, TARGET)
        
        results = evaluate_xgboost(X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe, RANDOM_STATE)
        results['Strategy'] = f'ROSE_{strategy}'
        
        print(f"  Accuracy: {results['Accuracy']:.4f}")
        print(f"  Sensitivity: {results['Sensitivity']:.4f}")
        print(f"  Specificity: {results['Specificity']:.4f}")
        print(f"  F1: {results['F1']:.4f}")
        print(f"  MCC: {results['MCC']:.4f}")
        print(f"  G-mean: {results['G_mean']:.4f}")
        print(f"  AUC-ROC: {results['AUC_ROC']:.4f}")
        
        all_results.append(results)
    
    # Save XGBoost results
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(OUTPUT_DIR, 'xgboost_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\nXGBoost results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    print("\nClass Distribution:")
    print(df_class_dist.to_string(index=False))
    
    print("\nOverlap Metrics:")
    print(df_overlap.to_string(index=False))
    
    print("\nXGBoost Results:")
    metric_cols = ['Strategy', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'MCC', 'G_mean', 'AUC_ROC', 'PR_AUC']
    print(df_results[metric_cols].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("  COMPLETED")
    print("=" * 60)
    print(f"All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
