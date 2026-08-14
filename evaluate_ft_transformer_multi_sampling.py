import os
import warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SAMPLED_DIR = r"data\CANADA\sampled"
DATA_DIR = r"data\CANADA\split"
OUTPUT_DIR = r"outputs\ft_transformer_multi_sampling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

SAMPLING_METHODS = ['rose', 'mied', 'under']

print("=" * 60)
print("EVALUATE FT-TRANSFORMER EMBEDDINGS ON MULTIPLE SAMPLING")
print("=" * 60)

# Load test data once
print("\nLoading test.csv...")
test_path = os.path.join(DATA_DIR, "test.csv")
df_test = pd.read_csv(test_path, low_memory=False)
df_test_model = df_test[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_test_model = df_test_model.dropna().reset_index(drop=True)

for col in NOMINAL_COLS:
    df_test_model[col] = df_test_model[col].astype(str)

print(f"  Loaded: {len(df_test):,} rows")
print(f"  Fatality rate: {df_test['Fatality'].mean()*100:.3f}%")

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
    train_path = os.path.join(SAMPLED_DIR, f"train_{sampling}.csv")
    df_train = pd.read_csv(train_path, low_memory=False)
    print(f"\nLoaded train_{sampling}.csv: {len(df_train):,} rows")
    print(f"Fatality rate: {df_train['Fatality'].mean()*100:.3f}%")
    
    # Prepare data
    df_train_model = df_train[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
    df_train_model = df_train_model.dropna().reset_index(drop=True)
    
    for col in NOMINAL_COLS:
        df_train_model[col] = df_train_model[col].astype(str)
    
    # Split train into train/val
    df_ft_train, df_ft_val = train_test_split(df_train_model, test_size=0.2, random_state=RANDOM_STATE, stratify=df_train_model[TARGET])
    
    print(f"FT-Transformer Train: {len(df_ft_train):,} rows")
    print(f"FT-Transformer Val: {len(df_ft_val):,} rows")
    
    # ============================================================
    # TRAIN FT-TRANSFORMER
    # ============================================================
    print("\nTraining FT-Transformer...")
    
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
    
    tabular_model.fit(train=df_ft_train, validation=df_ft_val)
    
    # ============================================================
    # EXTRACT EMBEDDINGS (PROBABILITIES)
    # ============================================================
    print("\nExtracting embeddings (probabilities)...")
    
    # Get predictions for train data
    train_pred_df = tabular_model.predict(df_ft_train)
    
    if 'Fatality_1_probability' in train_pred_df.columns:
        X_train_emb = train_pred_df[['Fatality_0_probability', 'Fatality_1_probability']].values
    else:
        X_train_emb = train_pred_df.select_dtypes(include=[np.number]).values
    
    y_train = df_ft_train[TARGET].astype(int).values
    
    # Get predictions for test data
    test_pred_df = tabular_model.predict(df_test_model)
    
    if 'Fatality_1_probability' in test_pred_df.columns:
        X_test_emb = test_pred_df[['Fatality_0_probability', 'Fatality_1_probability']].values
    else:
        X_test_emb = test_pred_df.select_dtypes(include=[np.number]).values
    
    y_test = df_test_model[TARGET].astype(int).values
    
    print(f"Train embeddings shape: {X_train_emb.shape}")
    print(f"Test embeddings shape: {X_test_emb.shape}")
    
    # ============================================================
    # CALCULATE QUALITY METRICS
    # ============================================================
    print("\nCalculating quality metrics...")
    
    # Sample subset for faster computation
    n_samples = min(5000, len(X_train_emb))
    indices = np.random.choice(len(X_train_emb), n_samples, replace=False)
    X_train_sample = X_train_emb[indices]
    y_train_sample = y_train[indices]
    
    # Silhouette Score
    try:
        sil_train = silhouette_score(X_train_sample, y_train_sample)
    except:
        sil_train = None
    
    try:
        sil_test = silhouette_score(X_test_emb[:5000], y_test[:5000])
    except:
        sil_test = None
    
    # Davies-Bouldin Index
    try:
        db_train = davies_bouldin_score(X_train_sample, y_train_sample)
    except:
        db_train = None
    
    try:
        db_test = davies_bouldin_score(X_test_emb[:5000], y_test[:5000])
    except:
        db_test = None
    
    # Intra-class variance
    var_train_0 = np.var(X_train_sample[y_train_sample == 0], axis=0).mean()
    var_train_1 = np.var(X_train_sample[y_train_sample == 1], axis=0).mean()
    var_test_0 = np.var(X_test_emb[:5000][y_test[:5000] == 0], axis=0).mean()
    var_test_1 = np.var(X_test_emb[:5000][y_test[:5000] == 1], axis=0).mean()
    
    # Inter-class distance
    mean_train_0 = X_train_sample[y_train_sample == 0].mean(axis=0)
    mean_train_1 = X_train_sample[y_train_sample == 1].mean(axis=0)
    dist_train = np.linalg.norm(mean_train_0 - mean_train_1)
    
    mean_test_0 = X_test_emb[:5000][y_test[:5000] == 0].mean(axis=0)
    mean_test_1 = X_test_emb[:5000][y_test[:5000] == 1].mean(axis=0)
    dist_test = np.linalg.norm(mean_test_0 - mean_test_1)
    
    # Store results
    results = {
        'Sampling': sampling,
        'Dataset': 'Train',
        'Silhouette_Score': sil_train,
        'Davies_Bouldin_Index': db_train,
        'Intra_Variance_Class_0': var_train_0,
        'Intra_Variance_Class_1': var_train_1,
        'Inter_Class_Distance': dist_train
    }
    all_results.append(results)
    
    results = {
        'Sampling': sampling,
        'Dataset': 'Test',
        'Silhouette_Score': sil_test,
        'Davies_Bouldin_Index': db_test,
        'Intra_Variance_Class_0': var_test_0,
        'Intra_Variance_Class_1': var_test_1,
        'Inter_Class_Distance': dist_test
    }
    all_results.append(results)
    
    print(f"\n{sampling.upper()} Results:")
    print(f"  Train Silhouette: {sil_train:.4f}" if sil_train else "  Train Silhouette: N/A")
    print(f"  Test Silhouette: {sil_test:.4f}" if sil_test else "  Test Silhouette: N/A")
    print(f"  Train Davies-Bouldin: {db_train:.4f}" if db_train else "  Train Davies-Bouldin: N/A")
    print(f"  Test Davies-Bouldin: {db_test:.4f}" if db_test else "  Test Davies-Bouldin: N/A")
    print(f"  Train Inter-Class Distance: {dist_train:.4f}")
    print(f"  Test Inter-Class Distance: {dist_test:.4f}")
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Train
    ax = axes[0]
    scatter = ax.scatter(X_train_sample[:, 0], X_train_sample[:, 1], c=y_train_sample, cmap='viridis', alpha=0.6, s=10)
    ax.set_title(f'FT-Transformer Probabilities - {sampling.upper()} (Train)')
    ax.set_xlabel('Fatality_0_probability')
    ax.set_ylabel('Fatality_1_probability')
    plt.colorbar(scatter, ax=ax, label='Fatality')
    
    # Test
    ax = axes[1]
    scatter = ax.scatter(X_test_emb[:5000, 0], X_test_emb[:5000, 1], c=y_test[:5000], cmap='viridis', alpha=0.6, s=10)
    ax.set_title(f'FT-Transformer Probabilities - {sampling.upper()} (Test)')
    ax.set_xlabel('Fatality_0_probability')
    ax.set_ylabel('Fatality_1_probability')
    plt.colorbar(scatter, ax=ax, label='Fatality')
    
    plt.tight_layout()
    viz_path = os.path.join(OUTPUT_DIR, f'ft_transformer_{sampling}_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {viz_path}")

# ============================================================
# SAVE ALL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING ALL RESULTS")
print("=" * 60)

df_results = pd.DataFrame(all_results)
results_path = os.path.join(OUTPUT_DIR, 'ft_transformer_multi_sampling_results.csv')
df_results.to_csv(results_path, index=False)
print(f"Results saved to: {results_path}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(df_results.to_string(index=False))

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)
