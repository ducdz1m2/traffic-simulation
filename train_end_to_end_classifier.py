import os
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"data\CANADA\sampled"
EMBEDDING_FILE = r"data\embeddings\contrastive_embedding_weights.pkl"
OUTPUT_DIR = r"data\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Feature columns
FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

print("=" * 60)
print("TRAIN END-TO-END CLASSIFIER WITH EMBEDDINGS")
print("=" * 60)
print(f"Data directory: {DATA_DIR}")
print(f"Embedding file: {EMBEDDING_FILE}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================
# LOAD EMBEDDINGS
# ============================================================
print("\n" + "=" * 60)
print("LOADING EMBEDDINGS")
print("=" * 60)

with open(EMBEDDING_FILE, 'rb') as f:
    embedding_data = pickle.load(f)

embedding_weights = embedding_data['embedding_weights']
label_encoders = embedding_data['label_encoders']
embedding_dims = embedding_data['embedding_dims']
CONTRASTIVE_DIM = embedding_data['contrastive_dim']

print(f"Loaded embeddings for {len(NOMINAL_COLS)} categorical features")
print(f"Contrastive dimension: {CONTRASTIVE_DIM}D")

# ============================================================
# LOAD UNDERSAMPLED DATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING UNDERSAMPLED DATA")
print("=" * 60)

train_path = os.path.join(DATA_DIR, "train_under.csv")
df = pd.read_csv(train_path, low_memory=False)
print(f"Loaded: {len(df):,} rows")
print(f"Fatality rate: {df['Fatality'].mean()*100:.3f}%")

# ============================================================
# PREPARE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("PREPARING FEATURES")
print("=" * 60)

df_model = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_model = df_model.dropna().reset_index(drop=True)
print(f"After dropping NA: {len(df_model):,} rows")

# Numeric features
X_num = df_model[FEATURE_COLS].values.astype(float)

# Categorical features with embeddings
X_cat_embeddings = []
for col in NOMINAL_COLS:
    le = label_encoders[col]
    weights = embedding_weights[col]
    
    cat_values = df_model[col].astype(str)
    
    # Handle unseen values
    cat_encoded = []
    for val in cat_values:
        if val in le.classes_:
            cat_encoded.append(le.transform([val])[0])
        else:
            cat_encoded.append(0)
    cat_encoded = np.array(cat_encoded)
    
    cat_embeddings = weights[cat_encoded]
    X_cat_embeddings.append(cat_embeddings)

# Concatenate all features
X_features = np.hstack([X_num] + X_cat_embeddings)
y = df_model[TARGET].astype(int).values

print(f"Features: {X_features.shape[1]} | Samples: {len(y):,}")

# ============================================================
# SPLIT DATA
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_features, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(y_train):,} (fatal={y_train.sum():,})")
print(f"Val: {len(y_val):,} (fatal={y_val.sum():,})")

# ============================================================
# DATASET CLASS
# ============================================================
class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# ENSEMBLE MODEL
# ============================================================
class EnsembleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Multiple heads for ensemble
        self.head1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.head2 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.head3 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x):
        out1 = self.head1(x)
        out2 = self.head2(x)
        out3 = self.head3(x)
        
        # Weighted ensemble
        weights = torch.softmax(self.ensemble_weights, dim=0)
        ensemble = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
        
        return ensemble.squeeze()

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# ============================================================
# TRAIN MODEL
# ============================================================
print("\n" + "=" * 60)
print("TRAINING ENSEMBLE MODEL")
print("=" * 60)

train_dataset = ClassificationDataset(X_train, y_train)
val_dataset = ClassificationDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = EnsembleClassifier(X_features.shape[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Model architecture:")
print(model)
print(f"Device: {device}")

criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

best_val_acc = 0
best_val_sens = 0
best_val_spec = 0
best_val_gmean = 0
patience_counter = 0
max_patience = 20

for epoch in range(200):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        train_correct += (predicted == y_batch).sum().item()
        train_total += y_batch.size(0)
    
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_tp = 0
    val_tn = 0
    val_fp = 0
    val_fn = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)
            
            # Calculate confusion matrix
            y_np = y_batch.cpu().numpy()
            pred_np = predicted.cpu().numpy()
            tp = np.sum((pred_np == 1) & (y_np == 1))
            tn = np.sum((pred_np == 0) & (y_np == 0))
            fp = np.sum((pred_np == 1) & (y_np == 0))
            fn = np.sum((pred_np == 0) & (y_np == 1))
            val_tp += tp
            val_tn += tn
            val_fp += fp
            val_fn += fn
    
    val_acc = val_correct / val_total
    val_sens = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
    val_spec = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0
    val_gmean = np.sqrt(val_sens * val_spec) if (val_sens * val_spec) >= 0 else 0
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/200 | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Sens: {val_sens:.4f} | Val Spec: {val_spec:.4f} | Val G-mean: {val_gmean:.4f}")
    
    # Check if all metrics >= 0.9
    if val_acc >= 0.9 and val_sens >= 0.9 and val_spec >= 0.9:
        print(f"\n*** TARGET REACHED: Acc={val_acc:.4f}, Sens={val_sens:.4f}, Spec={val_spec:.4f} ***")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_ensemble_model.pt'))
        break
    
    # Early stopping based on G-mean
    if val_gmean > best_val_gmean:
        best_val_gmean = val_gmean
        best_val_acc = val_acc
        best_val_sens = val_sens
        best_val_spec = val_spec
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_ensemble_model.pt'))
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nBest validation metrics:")
print(f"  Accuracy: {best_val_acc:.4f}")
print(f"  Sensitivity: {best_val_sens:.4f}")
print(f"  Specificity: {best_val_spec:.4f}")
print(f"  G-mean: {best_val_gmean:.4f}")

# Load best model
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_ensemble_model.pt')))

# ============================================================
# EVALUATE ON TEST DATA
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON TEST DATA")
print("=" * 60)

# Load test data
test_path = r"data\CANADA\split\test.csv"
df_test = pd.read_csv(test_path, low_memory=False)
print(f"Test samples: {len(df_test):,}")

# Prepare test features
df_test_model = df_test[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_test_model = df_test_model.dropna().reset_index(drop=True)

X_test_num = df_test_model[FEATURE_COLS].values.astype(float)

X_test_cat_embeddings = []
for col in NOMINAL_COLS:
    le = label_encoders[col]
    weights = embedding_weights[col]
    
    cat_values = df_test_model[col].astype(str)
    
    cat_encoded = []
    for val in cat_values:
        if val in le.classes_:
            cat_encoded.append(le.transform([val])[0])
        else:
            cat_encoded.append(0)
    cat_encoded = np.array(cat_encoded)
    
    cat_embeddings = weights[cat_encoded]
    X_test_cat_embeddings.append(cat_embeddings)

X_test_features = np.hstack([X_test_num] + X_test_cat_embeddings)
y_test = df_test_model[TARGET].astype(int).values

print(f"Test features: {X_test_features.shape[1]} | Samples: {len(y_test):,}")

# Evaluate
test_dataset = ClassificationDataset(X_test_features, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
test_correct = 0
test_total = 0
test_tp = 0
test_tn = 0
test_fp = 0
test_fn = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        outputs = model(X_batch)
        predicted = (outputs > 0.5).float()
        
        test_correct += (predicted == y_batch).sum().item()
        test_total += y_batch.size(0)
        
        y_np = y_batch.cpu().numpy()
        pred_np = predicted.cpu().numpy()
        tp = np.sum((pred_np == 1) & (y_np == 1))
        tn = np.sum((pred_np == 0) & (y_np == 0))
        fp = np.sum((pred_np == 1) & (y_np == 0))
        fn = np.sum((pred_np == 0) & (y_np == 1))
        test_tp += tp
        test_tn += tn
        test_fp += fp
        test_fn += fn

test_acc = test_correct / test_total
test_sens = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
test_spec = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0
test_gmean = np.sqrt(test_sens * test_spec) if (test_sens * test_spec) >= 0 else 0
test_f1 = 2 * test_tp / (2 * test_tp + test_fp + test_fn) if (2 * test_tp + test_fp + test_fn) > 0 else 0
test_mcc = (test_tp * test_tn - test_fp * test_fn) / np.sqrt((test_tp + test_fp) * (test_tp + test_fn) * (test_tn + test_fp) * (test_tn + test_fn)) if (test_tp + test_fp) * (test_tp + test_fn) * (test_tn + test_fp) * (test_tn + test_fn) > 0 else 0

print(f"\nTest Results:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Sensitivity: {test_sens:.4f}")
print(f"  Specificity: {test_spec:.4f}")
print(f"  F1: {test_f1:.4f}")
print(f"  MCC: {test_mcc:.4f}")
print(f"  G-mean: {test_gmean:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

torch.save({
    'model_state_dict': model.state_dict(),
    'ensemble_weights': model.ensemble_weights.data.cpu().numpy(),
    'input_dim': X_features.shape[1],
    'test_metrics': {
        'accuracy': test_acc,
        'sensitivity': test_sens,
        'specificity': test_spec,
        'f1': test_f1,
        'mcc': test_mcc,
        'g_mean': test_gmean
    }
}, os.path.join(OUTPUT_DIR, 'final_ensemble_model.pt'))

print(f"Model saved to: {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)
