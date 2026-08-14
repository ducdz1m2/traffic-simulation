import os
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import torch

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SAMPLED_DIR = r"data\CANADA\sampled"
DATA_DIR = r"data\CANADA\split"
OUTPUT_DIR = r"data\embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

print("=" * 60)
print("TEST FT-TRANSFORMER ON UNDER SAMPLING")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading train_under.csv...")
train_path = os.path.join(SAMPLED_DIR, "train_under.csv")
df_train = pd.read_csv(train_path, low_memory=False)
print(f"  Loaded: {len(df_train):,} rows")
print(f"  Fatality rate: {df_train['Fatality'].mean()*100:.3f}%")

print("\nLoading test.csv...")
test_path = os.path.join(DATA_DIR, "test.csv")
df_test = pd.read_csv(test_path, low_memory=False)
print(f"  Loaded: {len(df_test):,} rows")
print(f"  Fatality rate: {df_test['Fatality'].mean()*100:.3f}%")

# Prepare data
df_train_model = df_train[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_train_model = df_train_model.dropna().reset_index(drop=True)

df_test_model = df_test[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_test_model = df_test_model.dropna().reset_index(drop=True)

# Convert categorical to string
for col in NOMINAL_COLS:
    df_train_model[col] = df_train_model[col].astype(str)
    df_test_model[col] = df_test_model[col].astype(str)

# Split train into train/val for FT-Transformer
df_ft_train, df_ft_val = train_test_split(df_train_model, test_size=0.2, random_state=RANDOM_STATE, stratify=df_train_model[TARGET])

print(f"\nFT-Transformer Train: {len(df_ft_train):,} rows")
print(f"FT-Transformer Val: {len(df_ft_val):,} rows")
print(f"Test: {len(df_test_model):,} rows")

# ============================================================
# TRAIN FT-TRANSFORMER
# ============================================================
print("\n" + "=" * 60)
print("TRAINING FT-TRANSFORMER")
print("=" * 60)

data_config = DataConfig(
    target=[TARGET],
    continuous_cols=FEATURE_COLS,
    categorical_cols=NOMINAL_COLS,
)

trainer_config = TrainerConfig(
    auto_lr_find=False,
    batch_size=512,
    max_epochs=30,
    early_stopping_patience=5,
    checkpoints="valid_loss",
    load_best=True,
)

optimizer_config = OptimizerConfig()

model_config = FTTransformerConfig(
    task="classification",
    learning_rate=1e-3,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

print("\nTraining FT-Transformer...")
tabular_model.fit(train=df_ft_train, validation=df_ft_val)

# ============================================================
# EVALUATE FT-TRANSFORMER DIRECTLY
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING FT-TRANSFORMER DIRECTLY")
print("=" * 60)

# Use the trained FT-Transformer for prediction
y_pred_ft = tabular_model.predict(df_test_model)

print(f"Prediction output type: {type(y_pred_ft)}")
print(f"Prediction output shape: {y_pred_ft.shape if hasattr(y_pred_ft, 'shape') else 'N/A'}")

# Handle different output types
if isinstance(y_pred_ft, pd.DataFrame):
    print(f"Prediction columns: {y_pred_ft.columns.tolist()}")
    # Use the prediction column
    if 'Fatality_prediction' in y_pred_ft.columns:
        y_pred_ft = y_pred_ft['Fatality_prediction'].values
    else:
        y_pred_ft = y_pred_ft.iloc[:, 0].values
elif isinstance(y_pred_ft, np.ndarray):
    if y_pred_ft.ndim > 1:
        y_pred_ft = y_pred_ft[:, 0]

# Convert to binary if needed
if y_pred_ft.dtype == float:
    y_pred_ft = (y_pred_ft >= 0.5).astype(int)

y_test = df_test_model[TARGET].astype(int).values

print(f"y_pred_ft shape: {y_pred_ft.shape}, dtype: {y_pred_ft.dtype}")
print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ft).ravel()

acc = accuracy_score(y_test, y_pred_ft)
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
prec = precision_score(y_test, y_pred_ft, zero_division=0)
f1 = f1_score(y_test, y_pred_ft)
mcc = matthews_corrcoef(y_test, y_pred_ft)
g_mean = np.sqrt(sens * spec) if (sens * spec) >= 0 else 0

# Note: Skipping AUC/PR-AUC as predict_proba is not easily accessible
auc = 0.0
prauc = 0.0

print(f"\nFT-Transformer Results:")
print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity: {sens:.4f}")
print(f"Specificity: {spec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"G-mean: {g_mean:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"PR-AUC: {prauc:.4f}")

# Save results
results = {
    'Method': 'FT-Transformer',
    'Sampling': 'under',
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

results_path = os.path.join(OUTPUT_DIR, 'ft_transformer_under_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\nResults saved to: {results_path}")

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)
