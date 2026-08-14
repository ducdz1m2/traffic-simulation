import os
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"data\CANADA\split"
EMBEDDING_FILE = r"data\embeddings\contrastive_embedding_weights.pkl"
OUTPUT_DIR = r"outputs\full_data_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

print("=" * 60)
print("EVALUATE FULL DATA (NO SAMPLING)")
print("=" * 60)
print(f"Data directory: {DATA_DIR}")
print(f"Embedding file: {EMBEDDING_FILE}")

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

df_train = pd.read_csv(train_path, low_memory=False)
df_test = pd.read_csv(test_path, low_memory=False)

print(f"Train: {len(df_train):,} rows (fatal rate: {df_train[TARGET].mean()*100:.3f}%)")
print(f"Test: {len(df_test):,} rows (fatal rate: {df_test[TARGET].mean()*100:.3f}%)")

# ============================================================
# FEATURE PREPARATION WITH OHE
# ============================================================
print("\n" + "=" * 60)
print("METHOD 1: ONE-HOT ENCODING (OHE)")
print("=" * 60)

def prepare_features_with_ohe(df, feature_cols, nominal_cols, target, fitted_ohe=None):
    df_model = df[feature_cols + nominal_cols + [target]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    X_num = df_model[feature_cols].values.astype(float)
    
    if fitted_ohe is None:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_cat = ohe.fit_transform(df_model[nominal_cols].astype(str))
    else:
        ohe = fitted_ohe
        X_cat = ohe.transform(df_model[nominal_cols].astype(str))
    
    X_features = hstack([csr_matrix(X_num), X_cat])
    y = df_model[target].astype(int).values
    
    print(f'Features: {X_features.shape[1]} | Samples: {len(y):,}')
    return X_features, y, ohe

print("\nPreparing train features with OHE...")
X_train_ohe, y_train, ohe = prepare_features_with_ohe(df_train, FEATURE_COLS, NOMINAL_COLS, TARGET)
print(f"  Train: {len(y_train):,} (fatal={y_train.sum():,}, rate={y_train.mean()*100:.3f}%)")

print("\nPreparing test features with OHE...")
X_test_ohe, y_test, _ = prepare_features_with_ohe(df_test, FEATURE_COLS, NOMINAL_COLS, TARGET, fitted_ohe=ohe)
print(f"  Test: {len(y_test):,} (fatal={y_test.sum():,}, rate={y_test.mean()*100:.3f}%)")

# ============================================================
# FEATURE PREPARATION WITH EMBEDDINGS
# ============================================================
print("\n" + "=" * 60)
print("METHOD 2: ENTITY EMBEDDINGS")
print("=" * 60)

print(f"\nLoading embeddings from: {EMBEDDING_FILE}")
with open(EMBEDDING_FILE, 'rb') as f:
    embedding_data = pickle.load(f)

embedding_weights = embedding_data['embedding_weights']
label_encoders = embedding_data['label_encoders']
embedding_dims = embedding_data['embedding_dims']

print(f"  Loaded embeddings for {len(NOMINAL_COLS)} categorical features")

def prepare_features_with_embeddings(df, embedding_weights, label_encoders, embedding_dims, feature_cols, nominal_cols, target):
    df_model = df[feature_cols + nominal_cols + [target]].copy()
    df_model = df_model.dropna().reset_index(drop=True)
    
    # Numeric features
    X_num = df_model[feature_cols].values.astype(float)
    
    # Categorical embeddings
    cat_embeddings = []
    for col in nominal_cols:
        le = label_encoders[col]
        emb_weights = embedding_weights[col]
        
        # Handle unseen values
        values = df_model[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        ).values
        
        emb = emb_weights[values]
        cat_embeddings.append(emb)
    
    X_cat = np.concatenate(cat_embeddings, axis=1)
    X_features = np.concatenate([X_num, X_cat], axis=1)
    y = df_model[target].astype(int).values
    
    print(f'Features: {X_features.shape[1]} | Samples: {len(y):,}')
    return X_features, y

print("\nPreparing train features with embeddings...")
X_train_emb, y_train_emb = prepare_features_with_embeddings(
    df_train, embedding_weights, label_encoders, embedding_dims, 
    FEATURE_COLS, NOMINAL_COLS, TARGET)
print(f"  Train: {len(y_train_emb):,} (fatal={y_train_emb.sum():,}, rate={y_train_emb.mean()*100:.3f}%)")

print("\nPreparing test features with embeddings...")
X_test_emb, y_test_emb = prepare_features_with_embeddings(
    df_test, embedding_weights, label_encoders, embedding_dims, 
    FEATURE_COLS, NOMINAL_COLS, TARGET)
print(f"  Test: {len(y_test_emb):,} (fatal={y_test_emb.sum():,}, rate={y_test_emb.mean()*100:.3f}%)")

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred)  # Sensitivity = Recall for positive class
    spec = tn / (tn + fp)  # Specificity = TN / (TN + FP)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    g_mean = np.sqrt(sens * spec)
    
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.5
    
    try:
        prauc = average_precision_score(y_test, y_proba)
    except:
        prauc = 0.0
    
    print(f"  {model_name:20} | Acc={acc:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | Prec={prec:.4f} | F1={f1:.4f} | MCC={mcc:.4f} | G={g_mean:.4f} | AUC={auc:.4f} | PR={prauc:.4f}")
    
    return {
        'Method': model_name,
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
# TRAIN AND EVALUATE MODELS
# ============================================================
print("\n" + "=" * 60)
print("TRAINING AND EVALUATING MODELS")
print("=" * 60)

all_results = []

# === OHE Models ===
print("\n>>> OHE + Lasso Regression (5-fold CV) <<<")
lasso = LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=1000, penalty='l1', solver='saga')
lasso.fit(X_train_ohe, y_train)
result = evaluate_model('OHE + Lasso', lasso, X_test_ohe, y_test)
all_results.append(result)

print("\n>>> OHE + XGBoost <<<")
xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
xgb_model.fit(X_train_ohe, y_train)
result = evaluate_model('OHE + XGBoost', xgb_model, X_test_ohe, y_test)
all_results.append(result)

# === Embedding Models ===
print("\n>>> Embeddings + Lasso Regression (5-fold CV) <<<")
lasso_emb = LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=1000, penalty='l1', solver='saga')
lasso_emb.fit(X_train_emb, y_train_emb)
result = evaluate_model('Embeddings + Lasso', lasso_emb, X_test_emb, y_test_emb)
all_results.append(result)

print("\n>>> Embeddings + XGBoost <<<")
xgb_emb = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
xgb_emb.fit(X_train_emb, y_train_emb)
result = evaluate_model('Embeddings + XGBoost', xgb_emb, X_test_emb, y_test_emb)
all_results.append(result)

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

df_results = pd.DataFrame(all_results)
results_path = os.path.join(OUTPUT_DIR, 'full_data_evaluation_results.csv')
df_results.to_csv(results_path, index=False)
print(f"Results saved to: {results_path}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(df_results.to_string(index=False))

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)
