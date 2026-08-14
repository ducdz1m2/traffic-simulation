import os
import warnings
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, ks_2samp, chi2_contingency
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.figsize"] = (12, 5)

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "data/canada-dataset"
SAMPLE_SIZE = 100000
N_SEEDS = 3
MAX_ATTEMPTS = 200
OUT_DIR = "outputs/seed_search"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns to check
cat_cols = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS',
            'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF',
            'V_TYPE', 'P_SEX', 'P_PSN', 'P_SAFE', 'P_USER', 'Fatality']
num_cols = ['P_AGE', 'V_YEAR']

NA_CODES = {"U", "UU", "UUUU", "X", "XX", "XXXX", "N", "NN", "NNNN", "Q", "QQ"}

def clean_numeric(series):
    return pd.to_numeric(series.replace(NA_CODES, np.nan), errors="coerce")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv*")))
df_list = []
for f in files:
    df = pd.read_csv(f, dtype=str, low_memory=False)
    df_list.append(df)
    print(f"  {os.path.basename(f):30s} {len(df):>8,} rows")

df_full = pd.concat(df_list, ignore_index=True)
print(f"\n  Total: {len(df_full):,} rows")

# ============================================================
# 2. CREATE TARGET & CLEAN
# ============================================================
print("\n" + "=" * 60)
print("PREPARING TARGET & CLEANING")
print("=" * 60)

num_cols_all = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'C_SEV', 'C_VEHS',
                'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF',
                'V_ID', 'V_TYPE', 'V_YEAR', 'P_ID', 'P_AGE', 'P_PSN',
                'P_SEX', 'P_SAFE', 'P_ISEV', 'P_USER']

df_full['P_SEX'] = df_full['P_SEX'].replace({'M': 1, 'F': 0})
for col in num_cols_all:
    if col in df_full.columns:
        df_full[col] = clean_numeric(df_full[col])

df_full['Fatality'] = (df_full['P_ISEV'] == 3).astype(int)

fatality_rate = df_full['Fatality'].mean()
print(f"  Fatality rate (population): {fatality_rate*100:.4f}%")
print(f"  Class 0 (Non-fatal): {(1-fatality_rate)*100:.2f}%")
print(f"  Class 1 (Fatal):     {fatality_rate*100:.2f}%")

# ============================================================
# 3. STRATIFIED SAMPLE FUNCTION (stratify by Fatality only)
# ============================================================
def take_sample_strat(df, size, random_state):
    df = df.copy()
    # Drop rows where Fatality is NaN
    df = df.dropna(subset=['Fatality'])
    # Ensure enough samples per stratum
    valid_strata = df['Fatality'].value_counts()
    valid_strata = valid_strata[valid_strata >= 2].index
    df = df[df['Fatality'].isin(valid_strata)]
    actual_size = min(size, len(df))
    df_sample, _ = train_test_split(
        df, train_size=actual_size,
        stratify=df['Fatality'],
        random_state=random_state
    )
    return df_sample.reset_index(drop=True)

# ============================================================
# 4. EVALUATION METRICS
# ============================================================
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def compute_psi(expected, actual, bins=10):
    eps = 1e-6
    psi = 0
    # Merge both distributions to get common bins
    all_vals = np.concatenate([expected, actual])
    if all_vals.dtype.kind in 'if':
        # numeric: use quantile bins from expected
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bin_edges[-1] += eps
        bin_edges[0] -= eps
        exp_counts, _ = np.histogram(expected, bins=bin_edges)
        act_counts, _ = np.histogram(actual, bins=bin_edges)
        exp_pct = exp_counts / len(expected)
        act_pct = act_counts / len(actual)
    else:
        # categorical: use categories
        unique = np.union1d(np.unique(expected), np.unique(actual))
        exp_counts = np.array([np.sum(expected == u) for u in unique])
        act_counts = np.array([np.sum(actual == u) for u in unique])
        exp_pct = exp_counts / len(expected)
        act_pct = act_counts / len(actual)
    for e, a in zip(exp_pct, act_pct):
        e = max(e, eps)
        a = max(a, eps)
        psi += (a - e) * np.log(a / e)
    return psi

def evaluate_sample(pop_series, samp_series, col, col_type='cat'):
    if col_type == 'cat':
        pop_dist = pop_series.value_counts(normalize=True).sort_index()
        samp_dist = samp_series.value_counts(normalize=True).sort_index()
        common = pop_dist.index.intersection(samp_dist.index)
        if len(common) < 2:
            return None
        observed = (samp_dist[common] * len(samp_series)).fillna(0).values
        expected_prop = (pop_dist[common] / pop_dist[common].sum()).values
        expected = expected_prop * len(samp_series)
        _, p_chi = chisquare(observed, expected)
        # Cramér's V on contingency table
        ct = pd.crosstab(pop_series, samp_series, dropna=False)
        cv = cramers_v(ct.values) if ct.shape == (2, 2) else 0
        # PSI
        psi = compute_psi(pop_series.dropna().values, samp_series.dropna().values)
        return {'column': col, 'type': 'categorical', 'chi2_p': p_chi,
                'cramers_v': cv, 'psi': psi}
    else:
        # numeric
        pop_vals = pop_series.dropna()
        samp_vals = samp_series.dropna()
        if len(pop_vals) < 5 or len(samp_vals) < 5:
            return None
        # KS test
        ks_stat, ks_p = ks_2samp(pop_vals, samp_vals)
        # PSI
        psi = compute_psi(pop_vals.values, samp_vals.values)
        return {'column': col, 'type': 'numeric', 'ks_stat': ks_stat,
                'ks_p': ks_p, 'psi': psi}

# ============================================================
# 5. SEED SEARCH (stratified by Fatality)
# ============================================================
print("\n" + "=" * 60)
print("SEED SEARCH")
print(f"  Target: {N_SEEDS} seeds | Sample size: {SAMPLE_SIZE:,}")
print("=" * 60)

# Precompute population distributions
df_ref = df_full.dropna(subset=['Fatality', 'P_ISEV']).copy()
pop_dists = {}
for col in cat_cols:
    pop_dists[col] = df_ref[col].value_counts(normalize=True).sort_index()
for col in num_cols:
    pop_dists[col] = df_ref[col].dropna().values

VALID_SEEDS = []
tried = 0

while len(VALID_SEEDS) < N_SEEDS and tried < MAX_ATTEMPTS:
    seed = tried + 1
    tried += 1

    df_s = take_sample_strat(df_ref, SAMPLE_SIZE, seed)

    checks = []
    eval_results = []

    # --- Categorical checks ---
    for col in cat_cols:
        result = evaluate_sample(df_ref[col], df_s[col], col, 'cat')
        eval_results.append(result)

    # --- Numeric checks ---
    for col in num_cols:
        result = evaluate_sample(df_ref[col], df_s[col], col, 'num')
        eval_results.append(result)

    # Determine pass/fail (Benjamini-Hochberg not needed for seed search; use simple threshold)
    all_p = []
    for r in eval_results:
        if r is not None:
            if r['type'] == 'categorical':
                all_p.append(r['chi2_p'])
            else:
                all_p.append(r['ks_p'])

    # Bonferroni correction
    n_tests = len(all_p)
    alpha = 0.05 / max(n_tests, 1)

    passed = all(p > alpha for p in all_p) if all_p else False

    if passed:
        VALID_SEEDS.append(seed)
        status = "PASS"
    else:
        n_fail = sum(1 for p in all_p if p <= alpha)
        status = f"FAIL ({n_fail}/{n_tests} failed)"

    print(f"  Seed {seed:3d}: {status}  [{len(VALID_SEEDS)}/{N_SEEDS}]")

print(f"\n  Found {len(VALID_SEEDS)} valid seeds: {VALID_SEEDS}")

if len(VALID_SEEDS) < N_SEEDS:
    print(f"  WARNING: Only found {len(VALID_SEEDS)} seeds. Using all available.")
SEEDS = VALID_SEEDS[:N_SEEDS]

# ============================================================
# 6. DETAILED EVALUATION TABLE
# ============================================================
print("\n" + "=" * 60)
print("DETAILED EVALUATION PER SEED")
print("=" * 60)

all_rows = []
for seed in SEEDS:
    df_s = take_sample_strat(df_ref, SAMPLE_SIZE, seed)

    print(f"\n--- Seed {seed} ---")

    for col in cat_cols:
        r = evaluate_sample(df_ref[col], df_s[col], col, 'cat')
        if r:
            all_rows.append({**r, 'seed': seed})
            print(f"  {col:12s} | Chi2 p={r['chi2_p']:.4f} | CV={r['cramers_v']:.4f} | PSI={r['psi']:.6f}")

    for col in num_cols:
        r = evaluate_sample(df_ref[col], df_s[col], col, 'num')
        if r:
            all_rows.append({**r, 'seed': seed})
            print(f"  {col:12s} | KS D={r['ks_stat']:.4f} | KS p={r['ks_p']:.4f} | PSI={r['psi']:.6f}")

df_eval = pd.DataFrame(all_rows)
df_eval.to_csv(os.path.join(OUT_DIR, "seed_evaluation.csv"), index=False)
print(f"\nSaved: {OUT_DIR}/seed_evaluation.csv")

# ============================================================
# 7. PLOTS (individual figure per variable)
# ============================================================
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

for seed in SEEDS:
    df_s = take_sample_strat(df_ref, SAMPLE_SIZE, seed)
    seed_dir = os.path.join(OUT_DIR, f"seed_{seed}")
    plots_dir = os.path.join(seed_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Categorical variables - one figure per variable
    for col in cat_cols:
        pop_counts = df_ref[col].value_counts(normalize=True).sort_index()
        samp_counts = df_s[col].value_counts(normalize=True).sort_index()
        all_idx = pop_counts.index.union(samp_counts.index)
        pop_aligned = pop_counts.reindex(all_idx, fill_value=0)
        samp_aligned = samp_counts.reindex(all_idx, fill_value=0)
        x = np.arange(len(all_idx))
        w = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - w/2, pop_aligned.values, w, label='Population', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + w/2, samp_aligned.values, w, label='Sample', alpha=0.8, color='coral')
        ax.set_title(f'{col} — Population vs Sample (Seed {seed})', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(all_idx, rotation=90, fontsize=7)
        ax.set_ylabel('Proportion')
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{col}.png"), bbox_inches='tight')
        plt.close()
        print(f"  Seed {seed}: saved plots/{col}.png")

    # Numeric variables - one histogram per variable
    for col in num_cols:
        pop_vals = df_ref[col].dropna()
        samp_vals = df_s[col].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(pop_vals, bins=80, alpha=0.6, density=True, label='Population', color='steelblue')
        ax.hist(samp_vals, bins=80, alpha=0.6, density=True, label='Sample', color='coral')
        ax.set_title(f'{col} — Population vs Sample (Seed {seed})', fontsize=13)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{col}.png"), bbox_inches='tight')
        plt.close()
        print(f"  Seed {seed}: saved plots/{col}.png")

    # Fatality rate comparison bar (separate figure)
    fig, ax = plt.subplots(figsize=(8, 5))
    pop_rate = df_ref['Fatality'].value_counts(normalize=True).sort_index()
    samp_rate = df_s['Fatality'].value_counts(normalize=True).sort_index()
    x = np.arange(2)
    w = 0.35
    bars1 = ax.bar(x - w/2, pop_rate.values, w, label='Population', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + w/2, samp_rate.values, w, label='Sample', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(['Non-Fatal (0)', 'Fatal (1)'])
    ax.set_ylabel('Proportion')
    ax.set_title(f'Fatality Distribution (Seed {seed})', fontsize=13)
    ax.legend(fontsize=10)
    for i in range(2):
        ax.text(i - w/2, pop_rate.values[i] + 0.01, f'{pop_rate.values[i]*100:.2f}%', ha='center', fontsize=10, color='steelblue', fontweight='bold')
        ax.text(i + w/2, samp_rate.values[i] + 0.01, f'{samp_rate.values[i]*100:.2f}%', ha='center', fontsize=10, color='coral', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "Fatality.png"), bbox_inches='tight')
    plt.close()
    print(f"  Seed {seed}: saved plots/Fatality.png")

# ============================================================
# 8. OVERVIEW FIGURE FOR MAIN PAPER
# ============================================================
print("\n" + "=" * 60)
print("GENERATING OVERVIEW FIGURE (for main paper)")
print("=" * 60)

# Prepare data for all seeds
all_samples = {seed: take_sample_strat(df_ref, SAMPLE_SIZE, seed) for seed in SEEDS}

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25)

# --- Panel A: Fatality rate comparison ---
ax1 = fig.add_subplot(gs[0, 0])
pop_rate = df_ref['Fatality'].value_counts(normalize=True).sort_index()
x = np.arange(2)
w = 0.18
offsets = [-0.27, -0.09, 0.09, 0.27]
colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71']
labels = ['Population'] + [f'Seed {s}' for s in SEEDS]
all_rates = [pop_rate] + [all_samples[s]['Fatality'].value_counts(normalize=True).sort_index() for s in SEEDS]

for i, (rate, offset, color) in enumerate(zip(all_rates, offsets, colors)):
    bars = ax1.bar(x + offset, rate.values, w, label=labels[i], alpha=0.85, color=color, edgecolor='white', linewidth=0.5)
    for j, v in enumerate(rate.values):
        ax1.text(x[j] + offset, v + 0.012, f'{v*100:.2f}%', ha='center', fontsize=7, color=color, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(['Non-Fatal', 'Fatal'], fontsize=10)
ax1.set_ylabel('Proportion', fontsize=10)
ax1.set_title('A) Fatality Rate: Population vs Samples', fontsize=12, fontweight='bold', loc='left')
ax1.legend(fontsize=8, loc='upper right')

# --- Panel B: PSI heatmap for categorical variables ---
ax2 = fig.add_subplot(gs[0, 1:3])
psi_data = {}
for col in cat_cols:
    for seed in SEEDS:
        r = evaluate_sample(df_ref[col], all_samples[seed][col], col, 'cat')
        if r:
            psi_data[(col, seed)] = r['psi']

psi_df = pd.DataFrame(psi_data, index=pd.Index([0])).T.unstack(level=1)
psi_df.columns = [f'Seed {s}' for s in SEEDS]

sns.heatmap(psi_df, annot=True, fmt='.6f', cmap='RdYlGn_r', center=0.00005,
            linewidths=0.5, ax=ax2, cbar_kws={'label': 'PSI', 'shrink': 0.8},
            annot_kws={'fontsize': 7})
ax2.set_title('B) Population Stability Index (PSI) — Categorical Variables', fontsize=12, fontweight='bold', loc='left')
ax2.set_ylabel('')
ax2.tick_params(axis='y', labelsize=8)
ax2.tick_params(axis='x', labelsize=9)

# --- Panel C: KS D statistics for numeric variables ---
ax3 = fig.add_subplot(gs[1, 0])
ks_data = {col: [] for col in num_cols}
for col in num_cols:
    for seed in SEEDS:
        r = evaluate_sample(df_ref[col], all_samples[seed][col], col, 'num')
        if r:
            ks_data[col].append(r['ks_stat'])

x = np.arange(len(SEEDS))
w = 0.3
for i, col in enumerate(num_cols):
    offset = (i - 0.5) * w * 0.8
    bars = ax3.bar(x + offset, ks_data[col], w * 0.8, label=col, alpha=0.8, edgecolor='white')
    for j, v in enumerate(ks_data[col]):
        ax3.text(x[j] + offset, v + 0.0001, f'{v:.4f}', ha='center', fontsize=8)

ax3.set_xticks(x)
ax3.set_xticklabels([f'Seed {s}' for s in SEEDS], fontsize=10)
ax3.set_ylabel('KS D statistic', fontsize=10)
ax3.set_title('C) Kolmogorov–Smirnov D (Numeric Variables)', fontsize=12, fontweight='bold', loc='left')
ax3.legend(fontsize=9)

# --- Panel D: Chi-square p-values heatmap ---
ax4 = fig.add_subplot(gs[1, 1:3])
chi2_data = {}
for col in cat_cols:
    for seed in SEEDS:
        r = evaluate_sample(df_ref[col], all_samples[seed][col], col, 'cat')
        if r:
            chi2_data[(col, seed)] = r['chi2_p']

chi2_df = pd.DataFrame(chi2_data, index=pd.Index([0])).T.unstack(level=1)
chi2_df.columns = [f'Seed {s}' for s in SEEDS]

# Clip p-values for color scale
chi2_plot = chi2_df.copy()
for col in chi2_plot.columns:
    chi2_plot[col] = chi2_plot[col].clip(0, 1)

sns.heatmap(chi2_plot, annot=chi2_df.map(lambda x: f'{x:.4f}'),
            fmt='', cmap='RdYlGn', vmin=0, vmax=1, linewidths=0.5, ax=ax4,
            cbar_kws={'label': 'p-value', 'shrink': 0.8},
            annot_kws={'fontsize': 7})
ax4.set_title('D) Chi-square p-values (Categorical Variables)', fontsize=12, fontweight='bold', loc='left')
ax4.set_ylabel('')
ax4.tick_params(axis='y', labelsize=8)
ax4.tick_params(axis='x', labelsize=9)

plt.savefig(os.path.join(OUT_DIR, "overview_paper.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {OUT_DIR}/overview_paper.png")

# ============================================================
# 9. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

summary_rows = []
for seed in SEEDS:
    df_s = take_sample_strat(df_ref, SAMPLE_SIZE, seed)
    row = {'seed': seed, 'n': len(df_s),
           'pop_fatality_rate': fatality_rate * 100,
           'samp_fatality_rate': df_s['Fatality'].mean() * 100}
    # Aggregate metrics
    for col in cat_cols:
        r = evaluate_sample(df_ref[col], df_s[col], col, 'cat')
        if r:
            row[f'{col}_chi2_p'] = r['chi2_p']
            row[f'{col}_cramers_v'] = r['cramers_v']
            row[f'{col}_psi'] = r['psi']
    for col in num_cols:
        r = evaluate_sample(df_ref[col], df_s[col], col, 'num')
        if r:
            row[f'{col}_ks_stat'] = r['ks_stat']
            row[f'{col}_ks_p'] = r['ks_p']
            row[f'{col}_psi'] = r['psi']
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, "seed_summary.csv"), index=False)
print(df_summary.to_string(index=False))

print(f"\n{'=' * 60}")
print(f"  VALIDATED SEEDS: {SEEDS}")
print(f"  Plots saved to: {OUT_DIR}/")
print(f"{'=' * 60}")
