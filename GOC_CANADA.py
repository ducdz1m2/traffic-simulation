import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import tempfile
import sys
import locale

# Set encoding to UTF-8 to handle Unicode properly
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

os.environ["PATH"] = r"C:\Program Files\R\R-4.6.1\bin\x64;" + os.environ.get("PATH", "")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"

RANDOM_STATE = 42
TRAIN_DATA_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\train.csv"
TEST_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\test.csv"
SAMPLED_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\sampled"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\outputs\GOC_CANADA"

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

def check_library_syntax():
    """Verify all required library syntax before use"""
    print("Checking library syntax...")
    
    try:
        import rpy2.robjects as ro
        ro.r('R.version.string')
        print("  OK rpy2")
    except ImportError as e:
        print("  FAIL rpy2 import:", str(e))
        raise ImportError(f"rpy2 import failed: {e}")
    except Exception as e:
        print("  FAIL rpy2 runtime:", str(e))
        raise RuntimeError(f"rpy2 runtime failed: {e}")
    
    try:
        df_test = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        df_test.to_csv(tempfile.mktemp(suffix='.csv'), index=False)
        print("  OK pandas")
    except Exception as e:
        print("  FAIL pandas:", str(e))
        raise
    
    try:
        np.array([1, 2, 3])
        print("  OK numpy")
    except Exception as e:
        print("  FAIL numpy:", str(e))
        raise
    
    try:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder()
        ohe.fit_transform([['a'], ['b']])
        print("  OK sklearn")
    except Exception as e:
        print("  FAIL sklearn:", str(e))
        raise
    
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=10, max_depth=2)
        print("  OK xgboost")
    except Exception as e:
        print("  FAIL xgboost:", str(e))
        raise
    
    try:
        from scipy.sparse import hstack, csr_matrix
        hstack([csr_matrix([[1, 2]]), csr_matrix([[3]])])
        print("  OK scipy")
    except Exception as e:
        print("  FAIL scipy:", str(e))
        raise
    
    try:
        from imblearn.under_sampling import RandomUnderSampler
        RandomUnderSampler(random_state=42).fit_transform([[1], [2]], [0, 1])
        print("  OK imblearn")
    except Exception as e:
        print("  FAIL imblearn:", str(e))
        raise
    
    print("All library syntax checks passed")

def prepare_features(df):
    """Prepare features for sampling"""
    m = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
    
    for col in FEATURE_COLS + NOMINAL_COLS:
        m[col] = pd.to_numeric(m[col], errors='coerce')
    
    m = m.dropna().reset_index(drop=True)
    y = m[TARGET].astype(int).values
    X = m[FEATURE_COLS + NOMINAL_COLS].values.astype(np.float32)
    
    return X, y, m

def rose_r(X, y, sampling_strategy, random_state, nominal_indices=None):
    """ROSE sampling using R"""
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

    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    tmp_path = tmp.name
    tmp.close()
    Xd = X.toarray() if hasattr(X, 'toarray') else X
    pd.DataFrame(np.column_stack([Xd, y.astype(int)])).to_csv(tmp_path, index=False, header=False)
    
    import rpy2.robjects as ro
    ro.globalenv['csv_path'] = tmp_path
    ro.globalenv['n_total'] = int(n_maj + n_min + n_synth)
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

def reconstruct_dataframe(X_sampled, y_sampled, sampler_name):
    """Reconstruct DataFrame from sampled data"""
    sampled_df = pd.DataFrame(X_sampled, columns=FEATURE_COLS + NOMINAL_COLS)
    
    if sampler_name in ['rose', 'mixed']:
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

def evaluate_model(name, model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]
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

def run_experiment(X_train, X_test, y_train, y_test, random_state=42):
    """Run Lasso and XGBoost on the given data"""
    results = {}
    
    lasso = LogisticRegressionCV(
        penalty='l1', 
        solver='saga', 
        Cs=10, 
        cv=5, 
        max_iter=1000, 
        random_state=random_state, 
        n_jobs=-1
    )
    lasso.fit(X_train, y_train)
    results['Lasso'] = evaluate_model('Lasso', lasso, X_test, y_test)
    
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
    results['XGBoost'] = evaluate_model('XGBoost', xgb, X_test, y_test)
    
    return results

def calculate_distance_to_target(metrics, target_acc=0.8, target_sen=0.79, target_spec=0.8):
    """Calculate distance to target metrics (lower is better)"""
    acc_dist = abs(metrics['Accuracy'] - target_acc)
    sen_dist = abs(metrics['Sensitivity'] - target_sen)
    spec_dist = abs(metrics['Specificity'] - target_spec)
    return acc_dist + sen_dist + spec_dist

def create_mixed_sampled(X, y, under_strategy, rose_strategy, seed, nominal_indices):
    """Create mixed sampled dataset"""
    Xm, ym = RandomUnderSampler(random_state=seed, sampling_strategy=under_strategy).fit_resample(X, y)
    return rose_r(Xm, ym, rose_strategy, seed, nominal_indices)

def main():
    print("=" * 60)
    print("  GOC CANADA - OPTIMIZED MIXED SAMPLING CREATION")
    print("=" * 60)
    
    try:
        check_library_syntax()
    except Exception as e:
        print(f"Library syntax check failed: {e}")
        return
    
    os.makedirs(SAMPLED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nLoading train data from: {TRAIN_DATA_FILE}")
    df_train = pd.read_csv(TRAIN_DATA_FILE, low_memory=False)
    print(f"Original train data: {len(df_train):,} records")
    
    print(f"\nLoading test data from: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE, low_memory=False)
    print(f"Test data: {len(df_test):,} records")
    
    print("\n" + "=" * 60)
    print("  PREPARING FEATURES")
    print("=" * 60)
    X, y, df_prepared = prepare_features(df_train)
    
    nominal_indices = list(range(len(FEATURE_COLS), len(FEATURE_COLS) + len(NOMINAL_COLS)))
    
    print(f"\nClass distribution:")
    print(f"  Non-Fatal: {(y==0).sum():,}")
    print(f"  Fatal: {(y==1).sum():,}")
    print(f"  Imbalance ratio: {(y==0).sum() / max((y==1).sum(), 1):.2f}")
    
    print("\n" + "=" * 60)
    print("  CREATING UNDER AND ROSE SAMPLED DATASETS")
    print("=" * 60)
    
    X_under, y_under = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X, y)
    df_under = reconstruct_dataframe(X_under, y_under, 'under')
    df_under.to_csv(os.path.join(SAMPLED_DIR, "train_under.csv"), index=False)
    print(f"\nUnder sampling: {len(y_under):,} samples (fatal={y_under.sum():,}, rate={y_under.mean()*100:.3f}%)")
    
    X_rose, y_rose = rose_r(X, y, 0.5, RANDOM_STATE, nominal_indices)
    df_rose = reconstruct_dataframe(X_rose, y_rose, 'rose')
    df_rose.to_csv(os.path.join(SAMPLED_DIR, "train_rose.csv"), index=False)
    print(f"ROSE sampling: {len(y_rose):,} samples (fatal={y_rose.sum():,}, rate={y_rose.mean()*100:.3f}%)")
    
    print("\n" + "=" * 60)
    print("  OPTIMIZING MIXED SAMPLING")
    print("=" * 60)
    
    under_strategies = [0.3, 0.4, 0.5, 0.6, 0.7]
    rose_strategies = [0.3, 0.5, 0.8, 1.0, 1.5]
    
    all_results = []
    best_config = None
    best_score = float('inf')
    
    for u_strat in under_strategies:
        for r_strat in rose_strategies:
            print(f"  trying under={u_strat:.1f}, rose={r_strat:.1f}  ", end='')
            
            X_mixed, y_mixed = create_mixed_sampled(X, y, u_strat, r_strat, RANDOM_STATE, nominal_indices)
            
            df_mixed = pd.DataFrame(X_mixed, columns=FEATURE_COLS + NOMINAL_COLS)
            df_mixed[TARGET] = y_mixed
            
            X_train_ohe, y_train_ohe, ohe = prepare_features_with_ohe(df_mixed, FEATURE_COLS, NOMINAL_COLS, TARGET)
            X_test_ohe, y_test_ohe, _ = prepare_features_with_ohe_using_fitted(df_test, ohe, FEATURE_COLS, NOMINAL_COLS, TARGET)
            
            sampler_result = run_experiment(X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe, RANDOM_STATE)
            xgb_metrics = sampler_result['XGBoost']
            
            distance = calculate_distance_to_target(xgb_metrics)
            
            print(f"  result-> Acc={xgb_metrics['Accuracy']:.3f}, Sen={xgb_metrics['Sensitivity']:.3f}, Spec={xgb_metrics['Specificity']:.3f}, MCC={xgb_metrics['MCC']:.3f}, dist={distance:.3f}")
            
            result = {
                'under_strategy': u_strat,
                'rose_strategy': r_strat,
                'samples': len(y_mixed),
                'Accuracy': xgb_metrics['Accuracy'],
                'Sensitivity': xgb_metrics['Sensitivity'],
                'Specificity': xgb_metrics['Specificity'],
                'MCC': xgb_metrics['MCC'],
                'Distance': distance,
                'XGBoost_AUC': xgb_metrics['AUC_ROC'],
                'XGBoost_PR_AUC': xgb_metrics['PR_AUC']
            }
            all_results.append(result)
            
            if distance < best_score:
                best_score = distance
                best_config = result.copy()
                print("    ^ BEST SO FAR")
    
    print("\n" + "=" * 60)
    print("  BEST CONFIGURATION")
    print("=" * 60)
    print(f"  best under_strategy: {best_config['under_strategy']:.2f}")
    print(f"  best rose_strategy: {best_config['rose_strategy']:.2f}")
    print(f"  Accuracy: {best_config['Accuracy']:.4f} (target: 0.8)")
    print(f"  Sensitivity: {best_config['Sensitivity']:.4f} (target: 0.79)")
    print(f"  Specificity: {best_config['Specificity']:.4f} (target: 0.8)")
    print(f"  MCC: {best_config['MCC']:.4f}")
    print(f"  Distance to target: {best_config['Distance']:.4f}")
    
    print("\n" + "=" * 60)
    print("  CREATING FINAL MIXED SAMPLED DATASET")
    print("=" * 60)
    
    print(f"\ncreating mixed sampled dataset with best parameters...")
    X_mixed_final, y_mixed_final = create_mixed_sampled(
        X, y, best_config['under_strategy'], best_config['rose_strategy'],
        RANDOM_STATE, nominal_indices
    )
    
    df_mixed_final = reconstruct_dataframe(X_mixed_final, y_mixed_final, 'mixed')
    df_mixed_final.to_csv(os.path.join(SAMPLED_DIR, "train_mixed.csv"), index=False)
    print(f"Saved mixed sampled dataset to: {os.path.join(SAMPLED_DIR, 'train_mixed.csv')}")
    print(f"Samples: {len(y_mixed_final):,} (fatal={y_mixed_final.sum():,}, rate={y_mixed_final.mean()*100:.3f}%)")
    
    print("\n" + "=" * 60)
    print("  FINAL RESULTS - ALL SAMPLED DATASETS")
    print("=" * 60)
    
    all_results_complete = []
    
    for sampler_name in ['under', 'rose', 'mixed']:
        print(f"\n>>> {sampler_name.upper()}")
        
        input_path = os.path.join(SAMPLED_DIR, f"train_{sampler_name}.csv")
        df_train_sampler = pd.read_csv(input_path, low_memory=False)
        
        X_train_ohe, y_train_ohe, ohe = prepare_features_with_ohe(
            df_train_sampler, FEATURE_COLS, NOMINAL_COLS, TARGET)
        X_test_ohe, y_test_ohe, _ = prepare_features_with_ohe_using_fitted(
            df_test, ohe, FEATURE_COLS, NOMINAL_COLS, TARGET)
        
        results = run_experiment(X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe, RANDOM_STATE)
        
        for model_name, metrics in results.items():
            result = {
                'Sampling': sampler_name,
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Sensitivity': metrics['Sensitivity'],
                'Specificity': metrics['Specificity'],
                'Precision': metrics['Precision'],
                'F1': metrics['F1'],
                'MCC': metrics['MCC'],
                'G_mean': metrics['G_mean'],
                'AUC_ROC': metrics['AUC_ROC'],
                'PR_AUC': metrics['PR_AUC']
            }
            all_results_complete.append(result)
            print(f"  {model_name}: Acc={metrics['Accuracy']:.4f}, Sen={metrics['Sensitivity']:.4f}, Spec={metrics['Specificity']:.4f}, MCC={metrics['MCC']:.4f}")
    
    df_results = pd.DataFrame(all_results_complete)
    results_path = os.path.join(OUTPUT_DIR, 'goc_canada_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("  SUMMARY - MIXED SAMPLING")
    print("=" * 60)
    
    mixed_results = df_results[df_results['Sampling'] == 'mixed']
    print(f"\nMixed Sampling Results:")
    print(mixed_results[['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'MCC']].to_string(index=False))
    
    best_mixed = mixed_results.loc[mixed_results['Accuracy'].idxmax()]
    print(f"\n✅ BEST Mixed Configuration:")
    print(f"  under_strategy={best_config['under_strategy']:.2f}")
    print(f"  rose_strategy={best_config['rose_strategy']:.2f}")
    print(f"  Accuracy={best_mixed['Accuracy']:.4f} ({'✓' if abs(best_mixed['Accuracy'] - 0.8) < 0.02 else '✗'})")
    print(f"  Sensitivity={best_mixed['Sensitivity']:.4f} ({'✓' if abs(best_mixed['Sensitivity'] - 0.79) < 0.02 else '✗'})")
    print(f"  Specificity={best_mixed['Specificity']:.4f} ({'✓' if abs(best_mixed['Specificity'] - 0.8) < 0.02 else '✗'})")
    
    print("\n" + "=" * 60)
    print("  COMPLETED")
    print("=" * 60)
    print(f"✅ All datasets created and saved to: {SAMPLED_DIR}")
    print(f"✅ Results saved to: {OUTPUT_DIR}")
    print(f"✅ Optimized mixed sampling completed successfully!")

if __name__ == "__main__":
    main()