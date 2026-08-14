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
OUTPUT_DIR = r"data\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Feature columns
FEATURE_COLS = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_VEHS', 'V_YEAR', 'P_AGE']
NOMINAL_COLS = ['C_CONF', 'C_RCFG', 'C_RALN', 'C_TRAF', 'C_WTHR', 'C_RSUR', 'P_SAFE',
                'P_SEX', 'P_PSN', 'P_USER', 'V_TYPE']
TARGET = 'Fatality'

# Embedding dimensions
EMBEDDING_DIM_BASE = 50
EMBEDDING_DIM_MAX = 50

print("=" * 60)
print("TRAIN END-TO-END DEEP CLASSIFIER WITH FOCAL LOSS")
print("=" * 60)
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING UNDERSAMPLED DATA")
print("=" * 60)

train_path = os.path.join(DATA_DIR, "train_under.csv")
if not os.path.exists(train_path):
    print(f"Error: {train_path} not found")
    exit(1)

df = pd.read_csv(train_path, low_memory=False)
print(f"Loaded: {len(df):,} rows")
print(f"Fatality rate: {df['Fatality'].mean()*100:.3f}%")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

df_model = df[FEATURE_COLS + NOMINAL_COLS + [TARGET]].copy()
df_model = df_model.dropna().reset_index(drop=True)
print(f"After dropping NA: {len(df_model):,} rows")

# Split data
df_train, df_val = train_test_split(df_model, test_size=0.2, random_state=RANDOM_STATE, stratify=df_model[TARGET])
print(f"Train: {len(df_train):,} rows")
print(f"Val: {len(df_val):,} rows")

# Prepare numeric features
X_train_num = df_train[FEATURE_COLS].values.astype(float)
X_val_num = df_val[FEATURE_COLS].values.astype(float)

# Prepare categorical features (fit on entire dataset)
label_encoders = {}
cat_cardinalities = {}

for col in NOMINAL_COLS:
    le = LabelEncoder()
    le.fit(df_model[col].astype(str))
    label_encoders[col] = le
    cat_cardinalities[col] = len(le.classes_)
    print(f"  {col}: {cat_cardinalities[col]} unique values")

X_train_cat = {}
X_val_cat = {}

for col in NOMINAL_COLS:
    le = label_encoders[col]
    X_train_cat[col] = le.transform(df_train[col].astype(str))
    X_val_cat[col] = le.transform(df_val[col].astype(str))

y_train = df_train[TARGET].astype(int).values
y_val = df_val[TARGET].astype(int).values

# Calculate embedding dimensions
embedding_dims = {}
for col, cardinality in cat_cardinalities.items():
    dim = min(EMBEDDING_DIM_MAX, max(2, int(np.ceil(cardinality / 2))))
    embedding_dims[col] = dim

print(f"\nTotal embedding dimension: {sum(embedding_dims.values())}D")

# ============================================================
# DATASET CLASS
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat_dict, y):
        self.X_num = torch.FloatTensor(X_num)
        self.y = torch.FloatTensor(y)
        self.X_cat_dict = {col: torch.LongTensor(X_cat_dict[col]) for col in X_cat_dict}
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        item = {'numeric': self.X_num[idx], 'target': self.y[idx]}
        for col in self.X_cat_dict:
            item[col] = self.X_cat_dict[col][idx]
        return item

# ============================================================
# DEEP CLASSIFIER MODEL
# ============================================================
class DeepClassifier(nn.Module):
    def __init__(self, num_features, cat_cardinalities, embedding_dims):
        super().__init__()
        self.numeric_layer = nn.Linear(num_features, 64)
        self.numeric_bn = nn.BatchNorm1d(64)
        
        self.embeddings = nn.ModuleDict()
        self.embedding_bn = nn.ModuleDict()
        for col, cardinality in cat_cardinalities.items():
            self.embeddings[col] = nn.Embedding(cardinality + 1, embedding_dims[col])
            self.embedding_bn[col] = nn.BatchNorm1d(embedding_dims[col])
        
        total_embedding_dim = sum(embedding_dims.values())
        
        # Deep network
        self.fc1 = nn.Linear(64 + total_embedding_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        
        self.fc6 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, numeric, cat_dict):
        # Numeric features
        x_num = self.relu(self.numeric_bn(self.numeric_layer(numeric)))
        
        # Categorical embeddings
        cat_embeddings = []
        for col in cat_dict:
            emb = self.embeddings[col](cat_dict[col])
            emb = self.embedding_bn[col](emb)
            cat_embeddings.append(emb)
        
        x_cat = torch.cat(cat_embeddings, dim=1)
        x = torch.cat([x_num, x_cat], dim=1)
        
        # Deep network
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout(self.relu(self.bn4(self.fc4(x))))
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        
        return self.sigmoid(x)

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
print("TRAINING DEEP CLASSIFIER")
print("=" * 60)

train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
val_dataset = TabularDataset(X_val_num, X_val_cat, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

model = DeepClassifier(len(FEATURE_COLS), cat_cardinalities, embedding_dims)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"\nModel architecture:")
print(model)
print(f"Device: {device}")

criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

best_val_gmean = 0
patience_counter = 0
max_patience = 30

for epoch in range(200):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        numeric = batch['numeric'].to(device)
        target = batch['target'].float().to(device)
        cat_dict = {col: batch[col].to(device) for col in X_train_cat}
        
        optimizer.zero_grad()
        outputs = model(numeric, cat_dict).squeeze()
        loss = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        train_correct += (predicted == target).sum().item()
        train_total += target.size(0)
    
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
        for batch in val_loader:
            numeric = batch['numeric'].to(device)
            target = batch['target'].float().to(device)
            cat_dict = {col: batch[col].to(device) for col in X_val_cat}
            
            outputs = model(numeric, cat_dict).squeeze()
            loss = criterion(outputs, target)
            
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == target).sum().item()
            val_total += target.size(0)
            
            y_np = target.cpu().numpy()
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
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_deep_classifier.pt'))
        break
    
    # Early stopping based on G-mean
    if val_gmean > best_val_gmean:
        best_val_gmean = val_gmean
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_deep_classifier.pt'))
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nBest validation G-mean: {best_val_gmean:.4f}")

# Load best model
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_deep_classifier.pt')))

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

X_test_cat = {}
for col in NOMINAL_COLS:
    le = label_encoders[col]
    cat_values = df_test_model[col].astype(str)
    
    cat_encoded = []
    for val in cat_values:
        if val in le.classes_:
            cat_encoded.append(le.transform([val])[0])
        else:
            cat_encoded.append(0)
    cat_encoded = np.array(cat_encoded)
    X_test_cat[col] = cat_encoded

y_test = df_test_model[TARGET].astype(int).values

print(f"Test features: {len(FEATURE_COLS) + sum(embedding_dims.values())}D | Samples: {len(y_test):,}")

# Evaluate
test_dataset = TabularDataset(X_test_num, X_test_cat, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model.eval()
test_correct = 0
test_total = 0
test_tp = 0
test_tn = 0
test_fp = 0
test_fn = 0

with torch.no_grad():
    for batch in test_loader:
        numeric = batch['numeric'].to(device)
        target = batch['target'].float().to(device)
        cat_dict = {col: batch[col].to(device) for col in X_test_cat}
        
        outputs = model(numeric, cat_dict).squeeze()
        predicted = (outputs > 0.5).float()
        
        test_correct += (predicted == target).sum().item()
        test_total += target.size(0)
        
        y_np = target.cpu().numpy()
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
    'label_encoders': label_encoders,
    'cat_cardinalities': cat_cardinalities,
    'embedding_dims': embedding_dims,
    'test_metrics': {
        'accuracy': test_acc,
        'sensitivity': test_sens,
        'specificity': test_spec,
        'f1': test_f1,
        'mcc': test_mcc,
        'g_mean': test_gmean
    }
}, os.path.join(OUTPUT_DIR, 'final_deep_classifier.pt'))

print(f"Model saved to: {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)
