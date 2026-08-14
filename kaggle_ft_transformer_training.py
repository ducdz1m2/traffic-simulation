"""
Kaggle Notebook: FT-Transformer Training on Sampled NCDB Dataset
GPU Options: T4 x2, P100, or TPU v5e-8
"""

import os
import warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch

# Configure PyTorch for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

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


warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
# Kaggle data paths
DATA_DIR = "/kaggle/input/datasets/ngducchilly/sampled-ncdb-1999-2017"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories for outputs
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

# Sampling methods to evaluate
SAMPLING_METHODS = ['rose', 'mixed', 'under', 'borderline', 'smote', 'adasyn', 'nearmiss']

# ============================================================
# HYPERPARAMETER CONFIGURATION
# ============================================================
# Simple fixed hyperparameters based on dataset size

def get_hyperparameters(df_train):
    """
    Return fixed hyperparameters based on dataset size
    """
    n_samples = len(df_train)
    
    print(f"\nHyperparameters for {n_samples:,} samples:")
    
    # Adjust based on dataset size
    if n_samples < 50000:
        # Small dataset
        max_epochs = 50
        batch_size = 256
        input_embed_dim = 64
        num_heads = 4
        num_attn_blocks = 3
        dropout = 0.2
        learning_rate = 1e-3
        early_stopping_patience = 15
    elif n_samples < 200000:
        # Medium dataset
        max_epochs = 50
        batch_size = 512
        input_embed_dim = 64
        num_heads = 8
        num_attn_blocks = 4
        dropout = 0.2
        learning_rate = 1e-3
        early_stopping_patience = 15
    elif n_samples < 1000000:
        # Large dataset
        max_epochs = 40
        batch_size = 1024
        input_embed_dim = 128
        num_heads = 8
        num_attn_blocks = 4
        dropout = 0.2
        learning_rate = 1e-3
        early_stopping_patience = 10
    else:
        # Very large dataset
        max_epochs = 30
        batch_size = 2048
        input_embed_dim = 128
        num_heads = 16
        num_attn_blocks = 4
        dropout = 0.2
        learning_rate = 1e-3
        early_stopping_patience = 10
    
    params = {
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'input_embed_dim': input_embed_dim,
        'num_heads': num_heads,
        'num_attn_blocks': num_attn_blocks,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
    }
    
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return params

print("=" * 60)
print("FT-TRANSFORMER TRAINING ON KAGGLE")
print("=" * 60)

# ============================================================
# CONFIG - TRAIN ON SAMPLED DATA ONLY
# ============================================================
print("\nConfiguration: Train embeddings on sampled datasets only")
print("Evaluate embedding quality on test.csv (no fine-tuning)")
print("Generate visualization for each embedding")
print("Use fixed hyperparameters based on dataset size")

# ============================================================
# LOAD TEST DATA (for evaluation only)
# ============================================================
print("\nLoading test data...")
test_path = os.path.join(DATA_DIR, "test.csv")
if os.path.exists(test_path):
    df_test = pd.read_csv(test_path, low_memory=False)
    print(f"Test samples: {len(df_test):,}")
    print(f"Test fatality rate: {df_test['Fatality'].mean()*100:.3f}%")
else:
    print(f"Warning: Test file not found at {test_path}")
    print("Will use train/test split for evaluation")
    df_test = None

# ============================================================
# RESULTS STORAGE
# ============================================================
all_results = []

# ============================================================
# LOOP THROUGH SAMPLING METHODS
# ============================================================
for sampling in SAMPLING_METHODS:
    print("\n" + "=" * 60)
    print(f"SAMPLING METHOD: {sampling.upper()}")
    print("=" * 60)
    
    # Load train data
    train_path = os.path.join(DATA_DIR, f"train_{sampling}.csv")
    
    print(f"Looking for: {train_path}")
    print(f"File exists: {os.path.exists(train_path)}")
    
    if not os.path.exists(train_path):
        print(f"  File not found: {train_path}, skipping...")
        continue
    
    try:
        df_train = pd.read_csv(train_path, low_memory=False)
        print(f"\nLoaded train_{sampling}.csv: {len(df_train):,} rows")
        print(f"Fatality rate: {df_train['Fatality'].mean()*100:.3f}%")
    except Exception as e:
        print(f"Error loading data: {e}")
        continue
    
    # Prepare data
    try:
        df_train_model = df_train[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
        df_train_model = df_train_model.dropna().reset_index(drop=True)
        
        for col in NOMINAL_COLS:
            df_train_model[col] = df_train_model[col].astype(str)
        
        # Split train into train/val
        df_ft_train, df_ft_val = train_test_split(df_train_model, test_size=0.2, random_state=RANDOM_STATE, stratify=df_train_model[TARGET])
    except Exception as e:
        print(f"Error preparing data: {e}")
        continue
    
    print(f"FT-Transformer Train: {len(df_ft_train):,} rows")
    print(f"FT-Transformer Val: {len(df_ft_val):,} rows")
    
    # ============================================================
    # GET HYPERPARAMETERS
    # ============================================================
    print(f"\nGetting hyperparameters for {sampling}...")
    hyperparams = get_hyperparameters(df_ft_train)
    
    # ============================================================
    # TRAIN FT-TRANSFORMER WITH FIXED HYPERPARAMETERS
    # ============================================================
    print("\nTraining FT-Transformer with fixed hyperparameters...")
    
    try:
        data_config = DataConfig(
            target=[TARGET],
            continuous_cols=FEATURE_COLS,
            categorical_cols=NOMINAL_COLS,
            continuous_feature_transform="quantile_normal",
            normalize_continuous_features=True,
        )
        
        # Configure trainer with fixed hyperparameters
        trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=hyperparams['batch_size'],
            max_epochs=hyperparams['max_epochs'],
            early_stopping_patience=hyperparams['early_stopping_patience'],
            checkpoints="valid_loss",
            load_best=True,
            accelerator="gpu",
            devices=1,
            gradient_clip_val=1.0,
            progress_bar="rich",
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
            learning_rate=hyperparams['learning_rate'],
            input_embed_dim=hyperparams['input_embed_dim'],
            num_heads=hyperparams['num_heads'],
            num_attn_blocks=hyperparams['num_attn_blocks'],
            attn_dropout=hyperparams['dropout'],
            ff_dropout=hyperparams['dropout'],
            embedding_dropout=hyperparams['dropout'],
            batch_norm_continuous_input=True,
            head_config={
                "layers": "128-64",
                "dropout": hyperparams['dropout'],
            },
        )
        
        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        
        tabular_model.fit(train=df_ft_train, validation=df_ft_val)
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # ============================================================
    # EXTRACT BACKBONE EMBEDDINGS
    # ============================================================
    print("\nExtracting backbone embeddings...")
    
    try:
        # Use model.predict with ret_model_output=True to get backbone features
        tabular_model.model.eval()
        
        # Prepare inference dataloader
        inference_dataloader = tabular_model.datamodule.prepare_inference_dataloader(df_ft_train)
        
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
        y_train = df_ft_train[TARGET].astype(int).values
        
        # Save embeddings to individual file (memory-efficient for large datasets)
        embeddings_file = os.path.join(EMBEDDINGS_DIR, f'{sampling}_embeddings.npz')
        np.savez_compressed(
            embeddings_file,
            embeddings=X_train_emb,
            labels=y_train,
            sampling_method=sampling
        )
        print(f"Embeddings saved to: {embeddings_file}")
        print(f"Train embeddings shape: {X_train_emb.shape}")
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    
    # ============================================================
    # EXTRACT TEST EMBEDDINGS (for evaluation only, no fine-tuning)
    # ============================================================
    print("\nExtracting test embeddings for evaluation...")
    
    if df_test is not None:
        try:
            # Prepare test data
            df_test_model = df_test[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
            df_test_model = df_test_model.dropna().reset_index(drop=True)
            
            for col in NOMINAL_COLS:
                df_test_model[col] = df_test_model[col].astype(str)
            
            y_test = df_test_model[TARGET].astype(int).values
            
            # Extract test embeddings using trained model
            tabular_model.model.eval()
            inference_dataloader = tabular_model.datamodule.prepare_inference_dataloader(df_test_model)
            
            embeddings_list = []
            with torch.no_grad():
                for batch in inference_dataloader:
                    for k, v in batch.items():
                        if isinstance(v, list) and (len(v) == 0):
                            continue
                        batch[k] = v.to(tabular_model.model.device)
                    
                    _, ret_value = tabular_model.model.predict(batch, ret_model_output=True)
                    
                    if 'backbone_features' in ret_value:
                        backbone_emb = ret_value['backbone_features'].detach().cpu()
                        embeddings_list.append(backbone_emb)
            
            X_test_emb = torch.cat(embeddings_list, dim=0).numpy()
            print(f"Test embeddings shape: {X_test_emb.shape}")
            print(f"Test fatality rate: {y_test.mean()*100:.3f}%")
            
        except Exception as e:
            print(f"Error extracting test embeddings: {e}")
            print("Falling back to train/test split...")
            X_train_emb, X_test_emb, y_train, y_test = train_test_split(
                X_train_emb, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
            )
    else:
        # No test file, use train/test split
        print("Using train/test split for evaluation")
        X_train_emb, X_test_emb, y_train, y_test = train_test_split(
            X_train_emb, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
    
    # ============================================================
    # CALCULATE EMBEDDING QUALITY METRICS ON TEST DATA
    # ============================================================
    print("\nCalculating embedding quality metrics on test data...")
    
    # Sample subset for faster computation
    n_samples_test = min(5000, len(X_test_emb))
    indices_test = np.random.choice(len(X_test_emb), n_samples_test, replace=False)
    X_test_sample = X_test_emb[indices_test]
    y_test_sample = y_test[indices_test]
    
    # Sample train subset for PCA fitting
    n_samples_train = min(5000, len(X_train_emb))
    indices_train = np.random.choice(len(X_train_emb), n_samples_train, replace=False)
    X_train_sample = X_train_emb[indices_train]
    y_train_sample = y_train[indices_train]
    
    # Silhouette Score on test
    try:
        sil_test = silhouette_score(X_test_sample, y_test_sample)
    except:
        sil_test = None
    
    # Davies-Bouldin Index on test
    try:
        db_test = davies_bouldin_score(X_test_sample, y_test_sample)
    except:
        db_test = None
    
    # Intra-class variance on test
    var_test_0 = np.var(X_test_sample[y_test_sample == 0], axis=0).mean()
    var_test_1 = np.var(X_test_sample[y_test_sample == 1], axis=0).mean()
    
    # Inter-class distance on test
    mean_test_0 = X_test_sample[y_test_sample == 0].mean(axis=0)
    mean_test_1 = X_test_sample[y_test_sample == 1].mean(axis=0)
    dist_test = np.linalg.norm(mean_test_0 - mean_test_1)
    
    print(f"\n{sampling.upper()} Embedding Quality (Test Data):")
    print(f"  Silhouette Score: {sil_test:.4f}" if sil_test else "  Silhouette Score: N/A")
    print(f"  Davies-Bouldin Index: {db_test:.4f}" if db_test else "  Davies-Bouldin Index: N/A")
    print(f"  Inter-Class Distance: {dist_test:.4f}")
    
    # Append results
    all_results.append({
        'sampling': sampling,
        'n_samples': len(df_ft_train),
        'fatality_rate': df_train['Fatality'].mean(),
        'silhouette': sil_test,
        'davies_bouldin': db_test,
        'inter_class_dist': dist_test,
        'embedding_dim': X_train_emb.shape[1]
    })
    
    # ============================================================
    # VISUALIZATION (PCA: fit on train, transform on test)
    # ============================================================
    print("\nGenerating visualization...")
    
    # Fit PCA on train sample
    pca = PCA(n_components=2)
    pca.fit(X_train_sample)
    
    # Transform both train and test samples
    X_train_2d = pca.transform(X_train_sample)
    X_test_2d = pca.transform(X_test_sample)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Train embedding visualization
    scatter_train = axes[0].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_sample, cmap='viridis', alpha=0.6, s=10)
    axes[0].set_title(f'FT-Transformer Backbone Embeddings - {sampling.upper()} (Train)')
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')
    plt.colorbar(scatter_train, ax=axes[0], label='Fatality')
    
    # Test embedding visualization
    scatter_test = axes[1].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_sample, cmap='viridis', alpha=0.6, s=10)
    axes[1].set_title(f'FT-Transformer Backbone Embeddings - {sampling.upper()} (Test)')
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    plt.colorbar(scatter_test, ax=axes[1], label='Fatality')
    
    plt.tight_layout()
    viz_path = os.path.join(OUTPUT_DIR, f'ft_transformer_{sampling}_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {viz_path}")
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'ft_transformer_{sampling}_model')
    tabular_model.save_model(model_path)
    print(f"Model saved to: {model_path}")

# ============================================================
# SAVE ALL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING ALL RESULTS")
print("=" * 60)

if all_results:
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(OUTPUT_DIR, 'ft_transformer_embedding_quality.csv')
    df_results.to_csv(results_path, index=False)
    print(f"Embedding quality results saved to: {results_path}")
    print("\nEmbedding Quality Summary:")
    print(df_results.to_string(index=False))

print(f"\nAll embeddings saved individually to: {EMBEDDINGS_DIR}")
print("Each sampling method has its own .npz file with embeddings and labels")

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)
