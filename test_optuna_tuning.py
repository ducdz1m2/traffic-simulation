"""
Test Optuna hyperparameter tuning with synthetic data
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# Configure PyTorch for CPU (for testing)
device = torch.device('cpu')
print(f"Using device: {device}")

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

# Install optuna if not available
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
except:
    print("Installing optuna and optuna-integration...")
    os.system('pip install optuna optuna-integration[pytorch_lightning]')
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback

from sklearn.model_selection import train_test_split

print("=" * 60)
print("TEST OPTUNA HYPERPARAMETER TUNING")
print("=" * 60)

# ============================================================
# CREATE SYNTHETIC DATA
# ============================================================
print("\nCreating synthetic data...")
np.random.seed(42)
n_samples = 1000  # Small dataset for testing

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
print(f"Fatality rate: {df['Fatality'].mean()*100:.3f}%")

# Prepare data
df_model = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_model = df_model.dropna().reset_index(drop=True)

for col in NOMINAL_COLS:
    df_model[col] = df_model[col].astype(str)

df_train, df_val = train_test_split(df_model, test_size=0.2, random_state=42, stratify=df_model[TARGET])
print(f"Train: {len(df_train)} rows")
print(f"Val: {len(df_val)} rows")

# ============================================================
# OPTUNA HYPERPARAMETER TUNING (TEST)
# ============================================================
print("\n" + "=" * 60)
print("TESTING OPTUNA HYPERPARAMETER TUNING")
print("=" * 60)

def get_optuna_hyperparameters(df_train, df_val, n_trials=5, timeout=600):
    """Test Optuna hyperparameter tuning with reduced trials"""
    print(f"\nStarting Optuna hyperparameter tuning...")
    print(f"  n_trials: {n_trials}")
    print(f"  timeout: {timeout}s")
    
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        input_embed_dim = trial.suggest_categorical('input_embed_dim', [16, 32, 64])
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        num_attn_blocks = trial.suggest_int('num_attn_blocks', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        
        max_epochs = 5  # Reduced for testing
        early_stopping_patience = trial.suggest_int('early_stopping_patience', 3, 5)
        
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
                batch_size=batch_size,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                checkpoints="valid_loss",
                load_best=True,
                accelerator="cpu",  # Use CPU for testing
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
                    "patience": 2,
                }
            )
            
            model_config = FTTransformerConfig(
                task="classification",
                learning_rate=learning_rate,
                input_embed_dim=input_embed_dim,
                num_heads=num_heads,
                num_attn_blocks=num_attn_blocks,
                attn_dropout=dropout,
                ff_dropout=dropout,
                embedding_dropout=dropout,
                batch_norm_continuous_input=True,
                head_config={
                    "layers": "64-32",
                    "dropout": dropout,
                },
            )
            
            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
            )
            
            tabular_model.fit(train=df_train, validation=df_val)
            
            # Return validation loss (minimize)
            val_loss = tabular_model.model.trainer.callback_metrics.get('valid_loss', float('inf'))
            return val_loss.item() if hasattr(val_loss, 'item') else val_loss
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')  # Return worst score for failed trials
    
    # Create study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_trial.params
    best_value = study.best_trial.value
    
    print(f"\nOptuna optimization completed:")
    print(f"  Best validation loss: {best_value:.4f}")
    print(f"  Best parameters:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    
    best_params['max_epochs'] = 5  # For testing
    
    return best_params

# Run Optuna tuning
try:
    hyperparams = get_optuna_hyperparameters(
        df_train, 
        df_val, 
        n_trials=3,  # Very few trials for testing
        timeout=300  # 5 minutes
    )
    
    print("\n" + "=" * 60)
    print("OPTUNA TUNING TEST PASSED")
    print("=" * 60)
    print("Optuna hyperparameter tuning works correctly!")
    
except Exception as e:
    print(f"\nOptuna tuning test failed: {e}")
    import traceback
    traceback.print_exc()
