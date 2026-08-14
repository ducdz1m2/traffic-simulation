import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SAMPLED_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\sampled"
TRAIN_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\train.csv"
TEST_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA\split\test.csv"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\outputs\sampled_evaluation"
RANDOM_STATE = 42

# For testing: set to None to use full dataset, or set to a number to limit samples
MAX_SAMPLES = None  # Use full dataset for actual evaluation

# Feature columns (numeric only - true numeric values)
FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
# Nominal columns (categorical - will be OHE for Lasso)
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
# No LabelEncoder needed - all nominal columns will be OHE
CATEGORICAL_COLS = []
TARGET = 'Fatality'

# Sampling techniques to evaluate (including 'no' for original train data)
SAMPLER_NAMES = ['under', 'rose', 'mixed', 'smote', 'borderline', 'smote_tomek', 'adasyn', 'nearmiss']

# Deep classifier model path (not used in comparison)
# DEEP_MODEL_PATH = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\models\best_deep_classifier.pt"

# ============================================================
# FEATURE PREPARATION (WITH OHE)
# ============================================================
def prepare_features_with_ohe(df, feature_cols, nominal_cols, target):
    """Prepare features using One-Hot Encoding (original method)"""
    df_model = df[feature_cols + nominal_cols + [target]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    # Numeric features
    X_num = df_model[feature_cols].values.astype(float)
    
    # Categorical features with OHE
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat = ohe.fit_transform(df_model[nominal_cols].astype(str))
    
    # Concatenate
    X_features = hstack([csr_matrix(X_num), X_cat])
    y = df_model[target].astype(int).values
    
    print(f'Features: {X_features.shape[1]} | Samples: {len(y):,}')
    return X_features, y, ohe

def prepare_features_with_ohe_using_fitted(df, fitted_ohe, feature_cols, nominal_cols, target):
    """Prepare features using already-fitted OHE encoder"""
    df_model = df[feature_cols + nominal_cols + [target]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    # Numeric features
    X_num = df_model[feature_cols].values.astype(float)
    
    # Categorical features with fitted OHE
    X_cat = fitted_ohe.transform(df_model[nominal_cols].astype(str))
    
    # Concatenate
    X_features = hstack([csr_matrix(X_num), X_cat])
    y = df_model[target].astype(int).values
    
    print(f'Features: {X_features.shape[1]} | Samples: {len(y):,}')
    return X_features, y, None

# ============================================================
# DEEP CLASSIFIER MODEL
# ============================================================
# ============================================================
# MODEL EVALUATION
# ============================================================
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
    
    print(f'    {name:30s} | Acc={acc:.3f} | Sens={sens:.3f} | Spec={spec:.3f} | Prec={prec:.3f} | F1={f1:.3f} | MCC={mcc:.3f} | G={g_mean:.3f} | AUC={auc:.3f} | PR={prauc:.3f}')
    
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
    """Run Lasso and XGBoost on the given data (both use OHE features)"""
    results = []
    
    # === Lasso Regression (5-fold CV) ===
    print(f'\n  >>> Lasso Regression (5-fold CV) <<<')
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
    res = evaluate_model('Lasso', lasso, X_test, y_test)
    res['Model'] = 'Lasso'
    results.append(res)
    
    # === XGBoost ===
    print(f'\n  >>> XGBoost <<<')
    # Calculate scale_pos_weight for imbalanced data
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
    res = evaluate_model('XGBoost', xgb, X_test, y_test)
    res['Model'] = 'XGBoost'
    results.append(res)
    
    return results

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    print("=" * 60)
    print("  EVALUATE SAMPLED DATASETS (OHE ONLY)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load test data once
    print(f"\nLoading test data from: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE, low_memory=False)
    if MAX_SAMPLES:
        df_test = df_test.sample(n=min(MAX_SAMPLES, len(df_test)), random_state=RANDOM_STATE)
    print(f"  Test samples: {len(df_test):,}")
    
    all_results = []
    
    for sampler_name in SAMPLER_NAMES:
        print(f"\n{'=' * 60}")
        print(f"  SAMPLING: {sampler_name.upper()}")
        print(f"{'=' * 60}")
        
        # Load train data
        if sampler_name == 'no':
            input_path = TRAIN_FILE
            print(f"\nLoading original train data from: {input_path}")
        else:
            input_path = os.path.join(SAMPLED_DIR, f"train_{sampler_name}.csv")
            print(f"\nLoading sampled data from: {input_path}")
        
        if not os.path.exists(input_path):
            print(f"  File not found: {input_path}")
            continue
        
        df_train = pd.read_csv(input_path, low_memory=False)
        if MAX_SAMPLES:
            df_train = df_train.sample(n=min(MAX_SAMPLES, len(df_train)), random_state=RANDOM_STATE)
        print(f"  Train samples: {len(df_train):,}")
        
        # === OHE-based evaluation ===
        print(f"\n{'=' * 60}")
        print(f"  METHOD: ONE-HOT ENCODING (OHE)")
        print(f"{'=' * 60}")
        
        print(f"\nPreparing train features with OHE...")
        X_train_ohe, y_train_ohe, ohe = prepare_features_with_ohe(
            df_train, FEATURE_COLS, NOMINAL_COLS, TARGET)
        print(f"  Train: {len(y_train_ohe):,} (fatal={y_train_ohe.sum():,}, rate={y_train_ohe.mean()*100:.3f}%)")
        
        print(f"\nPreparing test features with OHE (using train OHE)...")
        X_test_ohe, y_test_ohe, _ = prepare_features_with_ohe_using_fitted(
            df_test, ohe, FEATURE_COLS, NOMINAL_COLS, TARGET)
        print(f"  Test: {len(y_test_ohe):,} (fatal={y_test_ohe.sum():,}, rate={y_test_ohe.mean()*100:.3f}%)")
        
        print(f"\nRunning experiments with OHE...")
        results_ohe = run_experiment(X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe, RANDOM_STATE)
        for res in results_ohe:
            res['Sampling'] = sampler_name
            res['Method'] = 'OHE'
        all_results.extend(results_ohe)
    
    # Save results
    print(f"\n{'=' * 60}")
    print(f"  SAVING RESULTS")
    print(f"{'=' * 60}")
    
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(OUTPUT_DIR, 'sampled_evaluation_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    
    metric_cols = ['Sampling', 'Model', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'MCC', 'G_mean', 'AUC_ROC', 'PR_AUC']
    print(df_results[metric_cols].to_string(index=False))
    
    # Aggregate by model
    print(f"\n{'=' * 60}")
    print(f"  AGGREGATED BY MODEL")
    print(f"{'=' * 60}")
    
    agg_results = df_results.groupby(['Model'])[metric_cols[2:]].mean().round(4)
    print(agg_results.to_string())
    
    print(f"\n{'=' * 60}")
    print(f"  COMPLETED")
    print(f"{'=' * 60}")
    print(f"Total evaluations: {len(all_results)}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
