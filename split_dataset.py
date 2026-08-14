import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\canada-dataset-sampled-imputed.csv"
DATA_DIR = r"C:\Users\ADMIN\Desktop\TT\B2205935-LeNgocDuc-TranThanhPhuc-tuan-07\code\data\CANADA"

# Create directory structure
ORIGINAL_DIR = os.path.join(DATA_DIR, "original")
SPLIT_DIR = os.path.join(DATA_DIR, "split")
SAMPLED_DIR = os.path.join(DATA_DIR, "sampled")

TRAIN_RATIO = 0.7
RANDOM_STATE = 42

# ============================================================
# CREATE DIRECTORIES
# ============================================================
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(SAMPLED_DIR, exist_ok=True)

print("=" * 60)
print("  SPLIT DATASET INTO TRAIN/TEST")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print(f"\nLoading data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Total samples: {len(df):,}")
print(f"Fatality distribution:")
print(df['Fatality'].value_counts())
print(f"Fatality rate: {df['Fatality'].mean()*100:.3f}%")

# ============================================================
# COPY TO ORIGINAL DIRECTORY
# ============================================================
original_path = os.path.join(ORIGINAL_DIR, "canada-dataset-sampled-imputed.csv")
if not os.path.exists(original_path):
    df.to_csv(original_path, index=False)
    print(f"\nCopied original data to: {original_path}")
else:
    print(f"\nOriginal data already exists at: {original_path}")

# ============================================================
# SPLIT DATA (STRATIFIED)
# ============================================================
print(f"\nSplitting data: {TRAIN_RATIO*100}% train, {(1-TRAIN_RATIO)*100}% test")
print("Using stratified split to preserve Fatality ratio...")

df_train, df_test = train_test_split(
    df,
    train_size=TRAIN_RATIO,
    stratify=df['Fatality'],
    random_state=RANDOM_STATE
)

# Reset index for clean sequential indices
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"\nTrain set: {len(df_train):,} samples")
print(f"Train Fatality rate: {df_train['Fatality'].mean()*100:.3f}%")
print(f"Train Fatality distribution:")
print(df_train['Fatality'].value_counts())

print(f"\nTest set: {len(df_test):,} samples")
print(f"Test Fatality rate: {df_test['Fatality'].mean()*100:.3f}%")
print(f"Test Fatality distribution:")
print(df_test['Fatality'].value_counts())

# ============================================================
# SAVE TRAIN/TEST
# ============================================================
train_path = os.path.join(SPLIT_DIR, "train.csv")
test_path = os.path.join(SPLIT_DIR, "test.csv")

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print(f"\nSaved train set to: {train_path}")
print(f"Saved test set to: {test_path}")

# ============================================================
# DIRECTORY STRUCTURE INFO
# ============================================================
print("\n" + "=" * 60)
print("  DIRECTORY STRUCTURE")
print("=" * 60)
print(f"""
{DATA_DIR}/
├── original/
│   └── canada-dataset-sampled-imputed.csv  (original full dataset)
├── split/
│   ├── train.csv                           (70% for training & sampling)
│   └── test.csv                            (30% for evaluation)
└── sampled/                                (for sampling outputs)
    ├── smote_train.csv
    ├── rose_train.csv
    ├── borderline_train.csv
    ├── adasyn_train.csv
    ├── nearmiss_train.csv
    └── ... (other sampling techniques)
""")

print("\nUsage:")
print("1. Use train.csv for applying sampling techniques")
print("2. Save sampled datasets to sampled/ directory")
print("3. Use test.csv for final evaluation (no sampling applied)")
print("4. Keep original/ as backup reference")

print("\nDone!")
