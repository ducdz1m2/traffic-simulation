"""
Test script to verify DeepFeatureExtractor works with FT-Transformer
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

warnings.filterwarnings("ignore")

# Configure PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import pytorch-tabular
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformerConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.feature_extractor import DeepFeatureExtractor
    print("✓ pytorch-tabular imported successfully")
except ImportError as e:
    print(f"✗ Error importing pytorch-tabular: {e}")
    exit(1)

# ============================================================
# CREATE MINIMAL SYNTHETIC DATA
# ============================================================
print("\n" + "="*60)
print("CREATING SYNTHETIC DATA")
print("="*60)

np.random.seed(42)
n_samples = 10000  # Increased to test dynamic batch size

# Create synthetic data
data = {
    'C_YEAR': np.random.randint(1999, 2018, n_samples),
    'C_MNTH': np.random.randint(1, 13, n_samples),
    'C_WDAY': np.random.randint(1, 8, n_samples),
    'C_HOUR': np.random.randint(0, 24, n_samples),
    'C_VEHS': np.random.randint(1, 5, n_samples),
    'V_YEAR': np.random.randint(1990, 2018, n_samples),
    'P_AGE': np.random.randint(18, 80, n_samples),
    'C_CONF': np.random.choice(['01', '02', '03', '04'], n_samples),
    'C_RCFG': np.random.choice(['01', '02', '03'], n_samples),
    'C_RALN': np.random.choice(['01', '02', '03'], n_samples),
    'C_TRAF': np.random.choice(['01', '02', '03'], n_samples),
    'C_WTHR': np.random.choice(['01', '02', '03', '04'], n_samples),
    'C_RSUR': np.random.choice(['01', '02', '03'], n_samples),
    'P_SAFE': np.random.choice(['01', '02', '03'], n_samples),
    'P_SEX': np.random.choice(['M', 'F'], n_samples),
    'P_PSN': np.random.choice(['11', '12', '13'], n_samples),
    'P_USER': np.random.choice(['01', '02', '03'], n_samples),
    'V_TYPE': np.random.choice(['01', '02', '03', '04'], n_samples),
    'Fatality': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)
print(f"Created synthetic dataset: {len(df)} rows")
print(f"Fatality rate: {df['Fatality'].mean()*100:.2f}%")

# ============================================================
# PREPARE DATA
# ============================================================
FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

df_model = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_model = df_model.dropna().reset_index(drop=True)

for col in NOMINAL_COLS:
    df_model[col] = df_model[col].astype(str)

df_train, df_val = train_test_split(df_model, test_size=0.2, random_state=42, stratify=df_model[TARGET])

print(f"Train: {len(df_train)} rows")
print(f"Val: {len(df_val)} rows")

# ============================================================
# TRAIN FT-TRANSFORMER (QUICK)
# ============================================================
print("\n" + "="*60)
print("TRAINING FT-TRANSFORMER")
print("="*60)

# Dynamic batch size for testing
n_train_samples = len(df_train)
if n_train_samples > 1000000:
    batch_size = 2048
elif n_train_samples > 500000:
    batch_size = 1024
else:
    batch_size = 512

print(f"Using batch_size: {batch_size} (based on {n_train_samples:,} samples)")

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
        max_epochs=3,  # Quick test
        early_stopping_patience=5,
        checkpoints="valid_loss",
        load_best=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
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
            "patience": 3,
        }
    )
    
    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        input_embed_dim=32,  # Smaller for speed
        num_heads=4,
        num_attn_blocks=2,  # Fewer blocks for speed
        attn_dropout=0.1,
        ff_dropout=0.1,
        embedding_dropout=0.1,
        batch_norm_continuous_input=True,
        head_config={
            "layers": "32",
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
    print("✓ Model trained successfully")
    
except Exception as e:
    print(f"✗ Error training model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# TEST BACKBONE EXTRACTION VIA MODEL.PREDICT
# ============================================================
print("\n" + "="*60)
print("TESTING BACKBONE EXTRACTION")
print("="*60)

try:
    # Use model.predict with ret_model_output=True to get backbone features
    tabular_model.model.eval()
    
    # Prepare inference dataloader
    inference_dataloader = tabular_model.datamodule.prepare_inference_dataloader(df_train)
    
    embeddings_list = []
    with torch.no_grad():
        for batch in inference_dataloader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, list) and (len(v) == 0):
                    continue
                batch[k] = v.to(tabular_model.model.device)
            
            # Get model output with backbone features
            _, ret_value = tabular_model.model.predict(batch, ret_model_output=True)
            
            # Extract backbone features
            if 'backbone_features' in ret_value:
                backbone_emb = ret_value['backbone_features'].detach().cpu()
                embeddings_list.append(backbone_emb)
    
    X_train_emb = torch.cat(embeddings_list, dim=0).numpy()
    y_train = df_train[TARGET].astype(int).values
    
    print(f"✓ Embeddings extracted successfully")
    print(f"  Train embeddings shape: {X_train_emb.shape}")
    print(f"  Expected shape: ({len(df_train)}, <embedding_dim>)")
    
    # Verify shape
    assert X_train_emb.shape[0] == len(df_train), f"Row count mismatch! Got {X_train_emb.shape[0]}, expected {len(df_train)}"
    assert len(X_train_emb.shape) == 2, "Embeddings should be 2D!"
    
    print("\n✓ ALL TESTS PASSED!")
    print("="*60)
    
except Exception as e:
    print(f"✗ Error extracting embeddings: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
