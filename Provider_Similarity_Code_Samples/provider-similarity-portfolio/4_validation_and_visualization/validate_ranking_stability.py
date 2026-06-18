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

df = pd.read_csv('data58.csv')

# Create overlap-based ranks (handle NaN)
df['overlap_rank'] = df.groupby('primary_pin')['common_procedure_count'].rank(ascending=False, method='min', na_option='bottom')
df['claims_rank'] = df.groupby('primary_pin')['primary_procedure_claims_in_overlap'].rank(ascending=False, method='min', na_option='bottom')

total_pins = df['primary_pin'].nunique()
total_pairs = len(df)

print("="*70)
print(f"TOTAL PROVIDERS: {total_pins:,}")
print(f"TOTAL PAIRS: {total_pairs:,}")
print("="*70)

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 1: MODEL #1 vs OVERLAP CODES #1")
print("="*70)
print("""
What this shows:
- For each provider, model picks a #1 alternative
- Overlap codes also has a #1 (most common procedures)
- How often do they pick the same #1?
""")
print(f"{'Run':<25} {'Match':>10} {'No Match':>10} {'Skipped':>10} {'Match %':>10}")
print("-"*65)

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    for col in rank_cols:
        match = 0
        no_match = 0
        skipped = 0
        
        for pin, g in df.groupby('primary_pin'):
            g_valid = g.dropna(subset=[col, 'overlap_rank'])
            if len(g_valid) == 0:
                skipped += 1
                continue
            
            model_top1 = g_valid.loc[g_valid[col].idxmin(), 'alternative_pin']
            codes_top1 = g_valid.loc[g_valid['overlap_rank'].idxmin(), 'alternative_pin']
            
            if model_top1 == codes_top1:
                match += 1
            else:
                no_match += 1
        
        valid_pins = match + no_match
        pct = 100 * match / valid_pins if valid_pins > 0 else 0
        print(f"{col:<25} {match:>10,} {no_match:>10,} {skipped:>10,} {pct:>9.1f}%")
    print()

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 2: MODEL #1 vs CLAIMS OVERLAP #1")
print("="*70)
print("""
What this shows:
- Same as above, but comparing to claims-based ranking
- Claims overlap = total claims from common procedures
- Does model agree with claims-based best match?
""")
print(f"{'Run':<25} {'Match':>10} {'No Match':>10} {'Skipped':>10} {'Match %':>10}")
print("-"*65)

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    for col in rank_cols:
        match = 0
        no_match = 0
        skipped = 0
        
        for pin, g in df.groupby('primary_pin'):
            g_valid = g.dropna(subset=[col, 'claims_rank'])
            if len(g_valid) == 0:
                skipped += 1
                continue
            
            model_top1 = g_valid.loc[g_valid[col].idxmin(), 'alternative_pin']
            claims_top1 = g_valid.loc[g_valid['claims_rank'].idxmin(), 'alternative_pin']
            
            if model_top1 == claims_top1:
                match += 1
            else:
                no_match += 1
        
        valid_pins = match + no_match
        pct = 100 * match / valid_pins if valid_pins > 0 else 0
        print(f"{col:<25} {match:>10,} {no_match:>10,} {skipped:>10,} {pct:>9.1f}%")
    print()

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 3: MODEL'S #1 LANDS WHERE IN OVERLAP RANKING?")
print("="*70)
print("""
What this shows:
- Model picks #1 alternative
- Where does overlap codes rank that same alternative?
- Ideally: model's #1 = overlap's #1, #2, or #3
""")
print(f"{'Run':<25} {'@Rank1':>8} {'@Rank2':>8} {'@Rank3':>8} {'@Rank4':>8} {'@Rank5+':>8} {'Skip':>6}")
print("-"*75)

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    for col in rank_cols:
        counts = {1: 0, 2: 0, 3: 0, 4: 0, '5+': 0}
        skipped = 0
        
        for pin, g in df.groupby('primary_pin'):
            g_valid = g.dropna(subset=[col, 'overlap_rank'])
            if len(g_valid) == 0:
                skipped += 1
                continue
            
            model_top1_alt = g_valid.loc[g_valid[col].idxmin(), 'alternative_pin']
            overlap_rank_row = g_valid.loc[g_valid['alternative_pin'] == model_top1_alt, 'overlap_rank']
            
            if len(overlap_rank_row) == 0 or pd.isna(overlap_rank_row.values[0]):
                skipped += 1
                continue
            
            overlap_rank_of_model_pick = overlap_rank_row.values[0]
            
            if overlap_rank_of_model_pick <= 4:
                counts[int(overlap_rank_of_model_pick)] += 1
            else:
                counts['5+'] += 1
        
        print(f"{col:<25} {counts[1]:>8,} {counts[2]:>8,} {counts[3]:>8,} {counts[4]:>8,} {counts['5+']:>8,} {skipped:>6,}")
    print()

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 4: OVERLAP'S #1 LANDS WHERE IN MODEL RANKING?")
print("="*70)
print("""
What this shows:
- Overlap codes picks #1 alternative (most common procedures)
- Where does model rank that same alternative?
- Ideally: overlap's #1 should be in model's top 3
""")
print(f"{'Run':<25} {'@Rank1':>8} {'@Rank2':>8} {'@Rank3':>8} {'@Rank4':>8} {'@Rank5+':>8} {'Skip':>6}")
print("-"*75)

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    for col in rank_cols:
        counts = {1: 0, 2: 0, 3: 0, 4: 0, '5+': 0}
        skipped = 0
        
        for pin, g in df.groupby('primary_pin'):
            g_valid = g.dropna(subset=[col, 'overlap_rank'])
            if len(g_valid) == 0:
                skipped += 1
                continue
            
            overlap_top1_alt = g_valid.loc[g_valid['overlap_rank'].idxmin(), 'alternative_pin']
            model_rank_row = g_valid.loc[g_valid['alternative_pin'] == overlap_top1_alt, col]
            
            if len(model_rank_row) == 0 or pd.isna(model_rank_row.values[0]):
                skipped += 1
                continue
            
            model_rank_of_overlap_pick = model_rank_row.values[0]
            
            if model_rank_of_overlap_pick <= 4:
                counts[int(model_rank_of_overlap_pick)] += 1
            else:
                counts['5+'] += 1
        
        print(f"{col:<25} {counts[1]:>8,} {counts[2]:>8,} {counts[3]:>8,} {counts[4]:>8,} {counts['5+']:>8,} {skipped:>6,}")
    print()

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 5: RANK DIFFERENCE (MODEL vs OVERLAP)")
print("="*70)
print("""
What this shows:
- For every pair: |model_rank - overlap_rank|
- Average across all valid pairs
- Lower = model ranking is closer to overlap ranking
""")
print(f"{'Run':<25} {'Avg Diff':>12} {'Valid Pairs':>15}")
print("-"*55)

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    for col in rank_cols:
        diff = (df[col] - df['overlap_rank']).abs()
        valid_diff = diff.dropna()
        avg_diff = valid_diff.mean() if len(valid_diff) > 0 else 0
        print(f"{col:<25} {avg_diff:>12.3f} {len(valid_diff):>15,}")
    print()

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 6: RANK DIFFERENCE DISTRIBUTION")
print("="*70)
print("""
What this shows:
- Count of pairs by how far model rank differs from overlap rank
- Diff=0 means exact match
- More Diff=0 = better alignment with overlap
""")
print(f"{'Run':<25} {'Diff=0':>10} {'Diff=1':>10} {'Diff=2':>10} {'Diff=3+':>10} {'NaN':>8}")
print("-"*75)

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    for col in rank_cols:
        diff = (df[col] - df['overlap_rank']).abs()
        d0 = (diff == 0).sum()
        d1 = (diff == 1).sum()
        d2 = (diff == 2).sum()
        d3 = (diff >= 3).sum()
        nan_count = diff.isna().sum()
        print(f"{col:<25} {d0:>10,} {d1:>10,} {d2:>10,} {d3:>10,} {nan_count:>8,}")
    print()

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 7: RUN-TO-RUN PAIRWISE #1 AGREEMENT")
print("="*70)
print("""
What this shows:
- Between any two runs, how many providers have same #1?
- Higher = more stable across runs
""")

for model in ['model1', 'model2']:
    print(f"\n{model.upper()}:")
    print(f"{'Comparison':<25} {'Same #1':>10} {'Different':>10} {'Skipped':>10} {'Same %':>10}")
    print("-"*65)
    
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    for col1, col2 in combinations(rank_cols, 2):
        match = 0
        diff = 0
        skipped = 0
        
        for pin, g in df.groupby('primary_pin'):
            g_valid = g.dropna(subset=[col1, col2])
            if len(g_valid) == 0:
                skipped += 1
                continue
            
            top1_run1 = g_valid.loc[g_valid[col1].idxmin(), 'alternative_pin']
            top1_run2 = g_valid.loc[g_valid[col2].idxmin(), 'alternative_pin']
            
            if top1_run1 == top1_run2:
                match += 1
            else:
                diff += 1
        
        valid = match + diff
        pct = 100 * match / valid if valid > 0 else 0
        run1 = col1.split('_')[-1]
        run2 = col2.split('_')[-1]
        print(f"run_{run1} vs run_{run2:<14} {match:>10,} {diff:>10,} {skipped:>10,} {pct:>9.1f}%")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 8: RUN-TO-RUN RANK STABILITY (ALL ALTERNATIVES)")
print("="*70)
print("""
What this shows:
- For all pairs: avg |rank_run_i - rank_run_j|
- Lower = ranks are more consistent between runs
""")

for model in ['model1', 'model2']:
    print(f"\n{model.upper()}:")
    print(f"{'Comparison':<25} {'Avg Rank Diff':>15} {'Valid Pairs':>15}")
    print("-"*60)
    
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    for col1, col2 in combinations(rank_cols, 2):
        diff = (df[col1] - df[col2]).abs()
        valid_diff = diff.dropna()
        avg_diff = valid_diff.mean() if len(valid_diff) > 0 else 0
        run1 = col1.split('_')[-1]
        run2 = col2.split('_')[-1]
        print(f"run_{run1} vs run_{run2:<14} {avg_diff:>15.3f} {len(valid_diff):>15,}")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 9: REPEATABILITY - HOW MANY RUNS AGREE ON #1?")
print("="*70)
print("""
What this shows:
- For each provider, count how many runs picked the same #1
- Example: 7/7 means all 7 runs picked identical #1
- Higher agreement = more repeatable model
""")

for model in ['model1', 'model2']:
    print(f"\n{model.upper()}:")
    
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    n_runs = len(rank_cols)
    
    if n_runs == 0:
        print("  No runs found")
        continue
    
    agreement_counts = {i: 0 for i in range(1, n_runs + 1)}
    skipped = 0
    
    for pin, g in df.groupby('primary_pin'):
        top1_per_run = []
        for col in rank_cols:
            g_valid = g.dropna(subset=[col])
            if len(g_valid) > 0:
                top1_per_run.append(g_valid.loc[g_valid[col].idxmin(), 'alternative_pin'])
        
        if len(top1_per_run) == 0:
            skipped += 1
            continue
        
        most_common_count = max(top1_per_run.count(x) for x in set(top1_per_run))
        if most_common_count <= n_runs:
            agreement_counts[most_common_count] += 1
    
    print(f"{'Agreement Level':<20} {'Providers':>12} {'%':>10}")
    print("-"*45)
    valid_pins = total_pins - skipped
    for k in sorted(agreement_counts.keys(), reverse=True):
        pct = 100 * agreement_counts[k] / valid_pins if valid_pins > 0 else 0
        print(f"{k}/{n_runs} runs agree{'':<8} {agreement_counts[k]:>12,} {pct:>9.1f}%")
    if skipped > 0:
        print(f"{'Skipped (no data)':<20} {skipped:>12,}")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 10: REPEATABILITY - HOW MANY RUNS AGREE ON TOP-3?")
print("="*70)
print("""
What this shows:
- For each provider, count how many runs had identical top-3 set
- Stricter than #1 agreement
""")

for model in ['model1', 'model2']:
    print(f"\n{model.upper()}:")
    
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    n_runs = len(rank_cols)
    
    if n_runs == 0:
        print("  No runs found")
        continue
    
    agreement_counts = {i: 0 for i in range(1, n_runs + 1)}
    skipped = 0
    
    for pin, g in df.groupby('primary_pin'):
        top3_per_run = []
        for col in rank_cols:
            g_valid = g.dropna(subset=[col])
            if len(g_valid) >= 3:
                top3_per_run.append(frozenset(g_valid.nsmallest(3, col)['alternative_pin']))
            elif len(g_valid) > 0:
                top3_per_run.append(frozenset(g_valid.nsmallest(len(g_valid), col)['alternative_pin']))
        
        if len(top3_per_run) == 0:
            skipped += 1
            continue
        
        most_common_count = max(top3_per_run.count(x) for x in set(top3_per_run))
        if most_common_count <= n_runs:
            agreement_counts[most_common_count] += 1
    
    print(f"{'Agreement Level':<20} {'Providers':>12} {'%':>10}")
    print("-"*45)
    valid_pins = total_pins - skipped
    for k in sorted(agreement_counts.keys(), reverse=True):
        pct = 100 * agreement_counts[k] / valid_pins if valid_pins > 0 else 0
        print(f"{k}/{n_runs} runs agree{'':<8} {agreement_counts[k]:>12,} {pct:>9.1f}%")
    if skipped > 0:
        print(f"{'Skipped (no data)':<20} {skipped:>12,}")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 11: RANK VARIANCE ACROSS RUNS")
print("="*70)
print("""
What this shows:
- For each pair, how much does rank vary across runs?
- Avg Std: average standard deviation of ranks
- Avg Range: average (max_rank - min_rank) across runs
- Lower = more repeatable
""")

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    if len(rank_cols) < 2:
        continue
    
    rank_std = df[rank_cols].std(axis=1, skipna=True)
    rank_range = df[rank_cols].max(axis=1, skipna=True) - df[rank_cols].min(axis=1, skipna=True)
    
    valid_std = rank_std.dropna()
    valid_range = rank_range.dropna()
    
    avg_std = valid_std.mean() if len(valid_std) > 0 else 0
    avg_range = valid_range.mean() if len(valid_range) > 0 else 0
    
    print(f"{model}: Avg Std = {avg_std:.3f}, Avg Range = {avg_range:.3f}, Valid Pairs = {len(valid_std):,}")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 12: PAIRS WITH ZERO RANK CHANGE ACROSS ALL RUNS")
print("="*70)
print("""
What this shows:
- Count of pairs where rank was identical in every single run
- Higher = more stable/repeatable
""")

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    if len(rank_cols) < 2:
        continue
    
    rank_range = df[rank_cols].max(axis=1, skipna=True) - df[rank_cols].min(axis=1, skipna=True)
    
    zero_change = (rank_range == 0).sum()
    some_change = (rank_range > 0).sum()
    nan_pairs = rank_range.isna().sum()
    valid_pairs = zero_change + some_change
    
    pct_zero = 100 * zero_change / valid_pairs if valid_pairs > 0 else 0
    pct_some = 100 * some_change / valid_pairs if valid_pairs > 0 else 0
    
    print(f"{model}: Zero change = {zero_change:,} ({pct_zero:.1f}%), Some change = {some_change:,} ({pct_some:.1f}%), NaN = {nan_pairs:,}")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 13: MAX RANK SWING DISTRIBUTION")
print("="*70)
print("""
What this shows:
- Across all runs, what's the biggest rank change for each pair?
- Swing=0: rank never changed
- Swing=1: rank moved by 1 position at most
- Lower swing = more stable
""")

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    if len(rank_cols) < 2:
        continue
    
    rank_range = df[rank_cols].max(axis=1, skipna=True) - df[rank_cols].min(axis=1, skipna=True)
    
    s0 = (rank_range == 0).sum()
    s1 = (rank_range == 1).sum()
    s2 = (rank_range == 2).sum()
    s3 = (rank_range >= 3).sum()
    nan_count = rank_range.isna().sum()
    valid = s0 + s1 + s2 + s3
    
    print(f"\n{model}:")
    print(f"  Swing=0: {s0:>10,} ({100*s0/valid:.1f}%)" if valid > 0 else f"  Swing=0: {s0:>10,}")
    print(f"  Swing=1: {s1:>10,} ({100*s1/valid:.1f}%)" if valid > 0 else f"  Swing=1: {s1:>10,}")
    print(f"  Swing=2: {s2:>10,} ({100*s2/valid:.1f}%)" if valid > 0 else f"  Swing=2: {s2:>10,}")
    print(f"  Swing=3+:{s3:>10,} ({100*s3/valid:.1f}%)" if valid > 0 else f"  Swing=3+:{s3:>10,}")
    if nan_count > 0:
        print(f"  NaN:     {nan_count:>10,}")

# =============================================================================
print("\n" + "="*70)
print("SUMMARY 14: MODEL1 vs MODEL2 AGREEMENT")
print("="*70)
print("""
What this shows:
- Using average rank across all runs for each model
- How often do model1 and model2 agree on #1?
""")

m1_cols = sorted([c for c in df.columns if c.startswith('model1_rank_run_')])
m2_cols = sorted([c for c in df.columns if c.startswith('model2_rank_run_')])

if m1_cols and m2_cols:
    df['model1_rank_avg'] = df[m1_cols].mean(axis=1, skipna=True)
    df['model2_rank_avg'] = df[m2_cols].mean(axis=1, skipna=True)
    
    match = 0
    diff = 0
    skipped = 0
    
    for pin, g in df.groupby('primary_pin'):
        g_valid = g.dropna(subset=['model1_rank_avg', 'model2_rank_avg'])
        if len(g_valid) == 0:
            skipped += 1
            continue
        
        m1_top1 = g_valid.loc[g_valid['model1_rank_avg'].idxmin(), 'alternative_pin']
        m2_top1 = g_valid.loc[g_valid['model2_rank_avg'].idxmin(), 'alternative_pin']
        
        if m1_top1 == m2_top1:
            match += 1
        else:
            diff += 1
    
    valid = match + diff
    pct = 100 * match / valid if valid > 0 else 0
    print(f"Same #1: {match:,} ({pct:.1f}%)")
    print(f"Different #1: {diff:,} ({100-pct:.1f}%)")
    print(f"Skipped: {skipped:,}")

# =============================================================================
print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)
print("""
What this shows:
- Side-by-side comparison of model1 vs model2
- Repeatability: internal consistency across runs
- Alignment: agreement with overlap-based rankings
""")

comparison = []

for model in ['model1', 'model2']:
    rank_cols = sorted([c for c in df.columns if c.startswith(f'{model}_rank_run_')])
    
    if len(rank_cols) < 2:
        continue
    
    row = {'Model': model}
    
    # Repeatability metrics
    rank_range = df[rank_cols].max(axis=1, skipna=True) - df[rank_cols].min(axis=1, skipna=True)
    zero_change = (rank_range == 0).sum()
    row['Pairs Zero Change'] = f"{zero_change:,}"
    
    consistent = 0
    valid_pins = 0
    for pin, g in df.groupby('primary_pin'):
        top1s = []
        for col in rank_cols:
            g_valid = g.dropna(subset=[col])
            if len(g_valid) > 0:
                top1s.append(g_valid.loc[g_valid[col].idxmin(), 'alternative_pin'])
        
        if len(top1s) > 0:
            valid_pins += 1
            if len(set(top1s)) == 1:
                consistent += 1
    
    pct_consistent = 100 * consistent / valid_pins if valid_pins > 0 else 0
    row['Providers Same #1 All Runs'] = f"{consistent:,} ({pct_consistent:.1f}%)"
    
    rank_std = df[rank_cols].std(axis=1, skipna=True).dropna()
    row['Avg Rank Std'] = f"{rank_std.mean():.3f}" if len(rank_std) > 0 else "N/A"
    
    # Alignment metrics
    df[f'{model}_rank_avg'] = df[rank_cols].mean(axis=1, skipna=True)
    match_overlap = 0
    valid_align = 0
    for pin, g in df.groupby('primary_pin'):
        g_valid = g.dropna(subset=[f'{model}_rank_avg', 'overlap_rank'])
        if len(g_valid) == 0:
            continue
        valid_align += 1
        model_top1 = g_valid.loc[g_valid[f'{model}_rank_avg'].idxmin(), 'alternative_pin']
        codes_top1 = g_valid.loc[g_valid['overlap_rank'].idxmin(), 'alternative_pin']
        if model_top1 == codes_top1:
            match_overlap += 1
    
    pct_match = 100 * match_overlap / valid_align if valid_align > 0 else 0
    row['Match Overlap #1'] = f"{match_overlap:,} ({pct_match:.1f}%)"
    
    comparison.append(row)

comparison_df = pd.DataFrame(comparison)
print("\n" + comparison_df.to_string(index=False))

# Save
comparison_df.to_csv('data39.csv', index=False)
print("\n✓ Saved: model_comparison_summary.csv")
