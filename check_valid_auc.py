"""
Check if pytorch-tabular supports valid_auc metric
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Install pytorch-tabular if not available
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformerConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
except:
    print("Installing pytorch-tabular...")
    os.system('pip install pytorch-tabular')
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformerConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

from sklearn.model_selection import train_test_split

print("=" * 60)
print("CHECKING VALID_AUC SUPPORT")
print("=" * 60)

# ============================================================
# CREATE SYNTHETIC DATA
# ============================================================
print("\nCreating synthetic data...")
np.random.seed(42)
n_samples = 1000

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

data = {
    'C_YEAR': np.random.randint(1999, 2018, n_samples),
    'C_MNTH': np.random.randint(1, 13, n_samples),
    'C_WDAY': np.random.randint(1, 8, n_samples),
    'C_HOUR': np.random.randint(0, 24, n_samples),
    'C_VEHS': np.random.randint(1, 5, n_samples),
    'V_YEAR': np.random.randint(1990, 2018, n_samples),
    'P_AGE': np.random.randint(18, 80, n_samples),
    'C_CONF': np.random.choice(['01', '02', '03', '21', '22', '23', '24', '25'], n_samples),
    'C_RCFG': np.random.choice(['01', '02', '03', '04', '05', '06', '07', '08', '09'], n_samples),
    'C_RALN': np.random.choice(['01', '02', '03'], n_samples),
    'C_TRAF': np.random.choice(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13'], n_samples),
    'C_WTHR': np.random.choice(['1', '2', '3', '4', '5', '6', '7'], n_samples),
    'C_RSUR': np.random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9'], n_samples),
    'P_SAFE': np.random.choice(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], n_samples),
    'P_SEX': np.random.choice(['M', 'F'], n_samples),
    'P_PSN': np.random.choice(['11', '12', '13', '21', '22', '23', '31', '32', '33'], n_samples),
    'P_USER': np.random.choice(['1', '2', '3', '4', '5'], n_samples),
    'V_TYPE': np.random.choice(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], n_samples),
    'Fatality': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)
print(f"Created synthetic dataset: {len(df)} rows")

# Prepare data
df_model = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_model = df_model.dropna().reset_index(drop=True)

for col in NOMINAL_COLS:
    df_model[col] = df_model[col].astype(str)

df_train, df_val = train_test_split(df_model, test_size=0.2, random_state=42, stratify=df_model[TARGET])
print(f"Train: {len(df_train)} rows")
print(f"Val: {len(df_val)} rows")

# ============================================================
# TRAIN 1 EPOCH AND CHECK METRICS
# ============================================================
print("\nTraining for 1 epoch to check available metrics...")

try:
    data_config = DataConfig(
        target=[TARGET],
        continuous_cols=FEATURE_COLS,
        categorical_cols=NOMINAL_COLS,
        continuous_feature_transform="quantile_normal",
        normalize_continuous_features=True,
    )
    
    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=64,
        max_epochs=1,  # Only 1 epoch
        early_stopping_patience=10,
        checkpoints="valid_loss",
        load_best=True,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=1.0,
        progress_bar="none",
    )
    
    optimizer_config = OptimizerConfig(
        optimizer="AdamW",
        lr_scheduler="ReduceLROnPlateau",
        lr_scheduler_params={
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
        }
    )
    
    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        input_embed_dim=32,
        num_heads=4,
        num_attn_blocks=2,
        attn_dropout=0.1,
        ff_dropout=0.1,
        embedding_dropout=0.1,
        batch_norm_continuous_input=True,
        head_config={
            "layers": "64-32",
            "dropout": 0.1,
        },
    )
    
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    
    tabular_model.fit(train=df_train, validation=df_val)
    
    # Check available metrics
    print("\n" + "=" * 60)
    print("AVAILABLE METRICS")
    print("=" * 60)
    print(f"\nCallback metrics keys:")
    print(tabular_model.model.trainer.callback_metrics.keys())
    
    # Check for specific metrics
    metrics_keys = list(tabular_model.model.trainer.callback_metrics.keys())
    print(f"\nChecking for specific metrics:")
    print(f"  valid_loss: {'valid_loss' in metrics_keys}")
    print(f"  valid_accuracy: {'valid_accuracy' in metrics_keys}")
    print(f"  valid_auc: {'valid_auc' in metrics_keys}")
    print(f"  valid_auroc: {'valid_auroc' in metrics_keys}")
    print(f"  valid_f1: {'valid_f1' in metrics_keys}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
