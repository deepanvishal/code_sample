# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "model3"  # Change to model1, model2, or model3

# Column names - update if different in your CSV
COL_JACCARD = 'common_procedure_jaccard'
COL_COVERAGE = 'coverage_overlap_avg'
COL_CODE_COUNT = 'common_procedure_count'

# =============================================================================
print("Loading data...")
df = pd.read_csv('data58.csv')

print("="*70)
print(f"MODEL: {MODEL_NAME.upper()}")
print(f"PROVIDERS: {df['primary_pin'].nunique():,}")
print(f"PAIRS: {len(df):,}")
print("="*70)

# Check columns exist
missing_cols = []
for col in [COL_JACCARD, COL_COVERAGE, COL_CODE_COUNT]:
    if col not in df.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"\nWARNING: Missing columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    print("\nUpdate COL_JACCARD, COL_COVERAGE, COL_CODE_COUNT at top of script.")

# Get model columns
sim_cols = sorted([c for c in df.columns if c.startswith(f'{MODEL_NAME}_sim_run_')])
n_runs = len(sim_cols)

print(f"\nRuns found: {n_runs}")
if n_runs == 0:
    print("ERROR: No runs found for this model")
    exit()

# =============================================================================
print("\nCreating ranks...")
# =============================================================================

# Rank by Jaccard (higher = better = rank 1)
if COL_JACCARD in df.columns:
    df['rank_jaccard'] = df.groupby('primary_pin')[COL_JACCARD].rank(ascending=False, method='min', na_option='bottom')
else:
    df['rank_jaccard'] = np.nan

# Rank by Coverage Avg (higher = better = rank 1)
if COL_COVERAGE in df.columns:
    df['rank_coverage'] = df.groupby('primary_pin')[COL_COVERAGE].rank(ascending=False, method='min', na_option='bottom')
else:
    df['rank_coverage'] = np.nan

# Rank by Code Count (higher = better = rank 1)
if COL_CODE_COUNT in df.columns:
    df['rank_code_count'] = df.groupby('primary_pin')[COL_CODE_COUNT].rank(ascending=False, method='min', na_option='bottom')
else:
    df['rank_code_count'] = np.nan

# Rank by Model Similarity (higher = better = rank 1) for each run
rank_cols = []
for sim_col in sim_cols:
    rank_col = sim_col.replace('_sim_', '_rank_')
    df[rank_col] = df.groupby('primary_pin')[sim_col].rank(ascending=False, method='min', na_option='bottom')
    rank_cols.append(rank_col)

print("Done.\n")

# =============================================================================
print("="*70)
print("SUMMARY 1: MODEL RANK vs JACCARD RANK")
print("="*70)
print("""
How much does model ranking align with Jaccard ranking?
Diff=0 means exact same rank for that pair.
""")

if COL_JACCARD in df.columns:
    print(f"{'Run':<25} {'Diff=0':>10} {'Diff=1':>10} {'Diff=2':>10} {'Diff=3+':>10} {'Avg Diff':>10} {'Valid':>10}")
    print("-"*85)
    
    for rank_col in rank_cols:
        diff = (df[rank_col] - df['rank_jaccard']).abs()
        valid = diff.dropna()
        n_valid = len(valid)
        
        if n_valid > 0:
            print(f"{rank_col:<25} {(diff == 0).sum():>10,} {(diff == 1).sum():>10,} {(diff == 2).sum():>10,} {(diff >= 3).sum():>10,} {valid.mean():>10.3f} {n_valid:>10,}")
        else:
            print(f"{rank_col:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>10}")
else:
    print("SKIPPED - column not found")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 2: MODEL RANK vs COVERAGE RANK")
print("="*70)
print("""
How much does model ranking align with Coverage ranking?
""")

if COL_COVERAGE in df.columns:
    print(f"{'Run':<25} {'Diff=0':>10} {'Diff=1':>10} {'Diff=2':>10} {'Diff=3+':>10} {'Avg Diff':>10} {'Valid':>10}")
    print("-"*85)
    
    for rank_col in rank_cols:
        diff = (df[rank_col] - df['rank_coverage']).abs()
        valid = diff.dropna()
        n_valid = len(valid)
        
        if n_valid > 0:
            print(f"{rank_col:<25} {(diff == 0).sum():>10,} {(diff == 1).sum():>10,} {(diff == 2).sum():>10,} {(diff >= 3).sum():>10,} {valid.mean():>10.3f} {n_valid:>10,}")
        else:
            print(f"{rank_col:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>10}")
else:
    print("SKIPPED - column not found")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 3: MODEL RANK vs CODE COUNT RANK")
print("="*70)
print("""
How much does model ranking align with Code Count ranking?
""")

if COL_CODE_COUNT in df.columns:
    print(f"{'Run':<25} {'Diff=0':>10} {'Diff=1':>10} {'Diff=2':>10} {'Diff=3+':>10} {'Avg Diff':>10} {'Valid':>10}")
    print("-"*85)
    
    for rank_col in rank_cols:
        diff = (df[rank_col] - df['rank_code_count']).abs()
        valid = diff.dropna()
        n_valid = len(valid)
        
        if n_valid > 0:
            print(f"{rank_col:<25} {(diff == 0).sum():>10,} {(diff == 1).sum():>10,} {(diff == 2).sum():>10,} {(diff >= 3).sum():>10,} {valid.mean():>10.3f} {n_valid:>10,}")
        else:
            print(f"{rank_col:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>10}")
else:
    print("SKIPPED - column not found")

# =============================================================================
# CONSISTENCY ACROSS RUNS
# =============================================================================

if n_runs >= 2:
    print("\n" + "="*70)
    print("SUMMARY 4: RUN-TO-RUN RANK CONSISTENCY")
    print("="*70)
    print("""
How consistent are ranks between runs?
""")
    print(f"{'Comparison':<20} {'Diff=0':>10} {'Diff=1':>10} {'Diff=2':>10} {'Diff=3+':>10} {'Avg Diff':>10} {'Valid':>10}")
    print("-"*80)
    
    for col1, col2 in combinations(rank_cols, 2):
        diff = (df[col1] - df[col2]).abs()
        valid = diff.dropna()
        n_valid = len(valid)
        
        run1 = col1.split('_')[-1]
        run2 = col2.split('_')[-1]
        
        if n_valid > 0:
            print(f"run_{run1} vs run_{run2:<8} {(diff == 0).sum():>10,} {(diff == 1).sum():>10,} {(diff == 2).sum():>10,} {(diff >= 3).sum():>10,} {valid.mean():>10.3f} {n_valid:>10,}")
        else:
            print(f"run_{run1} vs run_{run2:<8} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>10}")

    # =============================================================================
    print("\n" + "="*70)
    print("SUMMARY 5: RANK STABILITY ACROSS ALL RUNS")
    print("="*70)
    print("""
For each pair, max rank - min rank across all runs.
Swing=0 means rank never changed.
""")
    
    rank_data = df[rank_cols]
    
    # Only compute on rows with at least 2 non-NA values
    valid_rows = rank_data.notna().sum(axis=1) >= 2
    rank_data_valid = rank_data[valid_rows]
    
    if len(rank_data_valid) > 0:
        swing = rank_data_valid.max(axis=1, skipna=True) - rank_data_valid.min(axis=1, skipna=True)
        
        s0 = (swing == 0).sum()
        s1 = (swing == 1).sum()
        s2 = (swing == 2).sum()
        s3 = (swing >= 3).sum()
        total = len(swing)
        
        print(f"Swing=0:  {s0:>10,} ({100*s0/total:.1f}%)")
        print(f"Swing=1:  {s1:>10,} ({100*s1/total:.1f}%)")
        print(f"Swing=2:  {s2:>10,} ({100*s2/total:.1f}%)")
        print(f"Swing=3+: {s3:>10,} ({100*s3/total:.1f}%)")
        print(f"\nValid pairs: {total:,}")
        print(f"Avg Swing:   {swing.mean():.3f}")
        print(f"Avg Std:     {rank_data_valid.std(axis=1, skipna=True).mean():.3f}")
    else:
        print("No valid pairs with 2+ runs")

else:
    print("\n" + "="*70)
    print("CONSISTENCY METRICS: SKIPPED (need 2+ runs)")
    print("="*70)

# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

# Average alignment across runs
avg_diff_jaccard = []
avg_diff_coverage = []
avg_diff_code = []

for rank_col in rank_cols:
    if COL_JACCARD in df.columns:
        diff = (df[rank_col] - df['rank_jaccard']).abs().dropna()
        if len(diff) > 0:
            avg_diff_jaccard.append(diff.mean())
    
    if COL_COVERAGE in df.columns:
        diff = (df[rank_col] - df['rank_coverage']).abs().dropna()
        if len(diff) > 0:
            avg_diff_coverage.append(diff.mean())
    
    if COL_CODE_COUNT in df.columns:
        diff = (df[rank_col] - df['rank_code_count']).abs().dropna()
        if len(diff) > 0:
            avg_diff_code.append(diff.mean())

print(f"""
MODEL: {MODEL_NAME.upper()}
RUNS: {n_runs}

ALIGNMENT (lower = better):""")

if avg_diff_jaccard:
    print(f"  vs Jaccard Rank:     {np.mean(avg_diff_jaccard):.3f} avg diff")
else:
    print(f"  vs Jaccard Rank:     N/A")

if avg_diff_coverage:
    print(f"  vs Coverage Rank:    {np.mean(avg_diff_coverage):.3f} avg diff")
else:
    print(f"  vs Coverage Rank:    N/A")

if avg_diff_code:
    print(f"  vs Code Count Rank:  {np.mean(avg_diff_code):.3f} avg diff")
else:
    print(f"  vs Code Count Rank:  N/A")

if n_runs >= 2:
    rank_data = df[rank_cols]
    valid_rows = rank_data.notna().sum(axis=1) >= 2
    rank_data_valid = rank_data[valid_rows]
    
    if len(rank_data_valid) > 0:
        swing = rank_data_valid.max(axis=1, skipna=True) - rank_data_valid.min(axis=1, skipna=True)
        zero_pct = 100 * (swing == 0).sum() / len(swing)
        
        print(f"""
CONSISTENCY:
  Pairs with Zero Swing: {zero_pct:.1f}%
  Avg Rank Std:          {rank_data_valid.std(axis=1, skipna=True).mean():.3f}
""")

print("="*70)
