# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
import warnings
import gc

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

N_RANDOM_PAIRS = 1_000_000
BATCH_SIZE = 100_000  # Larger batch = fewer loop iterations

print("="*80)
print("NOTEBOOK 2: PRECOMPUTE SIMILAR PAIRS")
print("="*80)
print(f"Random pairs: {N_RANDOM_PAIRS:,}")
print(f"No top-K boost (unbiased)")
print(f"Metrics: code_overlap (Jaccard), weighted_jaccard, coverage_overlap")
print(f"All metrics normalized to 0-1")

start_time = time.time()

# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOAD DATA")
print("="*80)

proc_df = pd.read_parquet('data46.parquet')

# Load embeddings only to get PIN list (for alignment with Notebook 1)
# TODO: If slow, save PIN list separately in Notebook 1
with open('data51.pkl', 'rb') as f:
    provider_code_embeddings = pickle.load(f)

notebook1_pins = set(provider_code_embeddings.keys())
del provider_code_embeddings  # Free memory immediately
gc.collect()

parquet_pins = set(proc_df['PIN'].unique())
common_pins = sorted(notebook1_pins & parquet_pins)

proc_df = proc_df[proc_df['PIN'].isin(common_pins)]

n_providers = len(common_pins)
print(f"Providers: {n_providers:,}")
print(f"Possible pairs: {n_providers * (n_providers - 1) // 2:,}")

# =============================================================================
print("\n" + "="*80)
print("STEP 2: BUILD INDEX MAPPINGS")
print("="*80)

pin_to_idx = {pin: idx for idx, pin in enumerate(common_pins)}
idx_to_pin = {idx: pin for pin, idx in pin_to_idx.items()}

all_codes = proc_df['code'].unique()
code_to_idx = {c: i for i, c in enumerate(all_codes)}
n_codes = len(all_codes)

print(f"Unique codes: {n_codes:,}")

# =============================================================================
print("\n" + "="*80)
print("STEP 3: BUILD SPARSE MATRICES")
print("="*80)

proc_df['pin_idx'] = proc_df['PIN'].map(pin_to_idx)
proc_df['code_idx'] = proc_df['code'].map(code_to_idx)

rows = proc_df['pin_idx'].values
cols = proc_df['code_idx'].values
claims = proc_df['claims'].values.astype(np.float32)

# Binary code matrix (for Jaccard)
code_matrix = csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(n_providers, n_codes)
)

# Claims matrix (for coverage)
claims_matrix = csr_matrix(
    (claims, (rows, cols)),
    shape=(n_providers, n_codes)
)

# Precompute per-provider totals
codes_per_provider = np.array(code_matrix.sum(axis=1)).flatten()
claims_per_provider = np.array(claims_matrix.sum(axis=1)).flatten()

print(f"Code matrix: {code_matrix.shape}, nnz: {code_matrix.nnz:,}")
print(f"Claims matrix: {claims_matrix.shape}, nnz: {claims_matrix.nnz:,}")

# =============================================================================
print("\n" + "="*80)
print("STEP 4: GENERATE RANDOM PAIRS")
print("="*80)

print(f"Generating {N_RANDOM_PAIRS:,} random pairs...")

n_attempts = N_RANDOM_PAIRS * 3
i_random = np.random.randint(0, n_providers, n_attempts, dtype=np.int32)
j_random = np.random.randint(0, n_providers, n_attempts, dtype=np.int32)

# Remove self-pairs
mask = i_random != j_random
i_random, j_random = i_random[mask], j_random[mask]

# Ensure i < j (fixed swap)
swap_mask = i_random > j_random
temp = i_random[swap_mask].copy()
i_random[swap_mask] = j_random[swap_mask]
j_random[swap_mask] = temp

# Remove duplicates
random_pairs = np.unique(np.column_stack([i_random, j_random]), axis=0)
random_pairs = random_pairs[:N_RANDOM_PAIRS]

print(f"Generated: {len(random_pairs):,} unique pairs")

# =============================================================================
print("\n" + "="*80)
print("STEP 5: COMPUTE METRICS (VECTORIZED)")
print("="*80)

n_pairs = len(random_pairs)
code_overlaps = np.zeros(n_pairs, dtype=np.float32)
coverage_overlaps = np.zeros(n_pairs, dtype=np.float32)
weighted_jaccards = np.zeros(n_pairs, dtype=np.float32)

for start in tqdm(range(0, n_pairs, BATCH_SIZE), desc="Computing metrics"):
    end = min(start + BATCH_SIZE, n_pairs)
    batch = random_pairs[start:end]
    
    i_idx = batch[:, 0]
    j_idx = batch[:, 1]
    
    # Slice sparse matrices (CSR row slicing is O(1))
    codes_i = code_matrix[i_idx]
    codes_j = code_matrix[j_idx]
    
    # Intersection mask (reused for both metrics)
    intersection_mask = codes_i.multiply(codes_j)
    
    # Code overlap (Jaccard): |A ∩ B| / |A ∪ B|
    intersection = np.array(intersection_mask.sum(axis=1)).flatten()
    union = codes_per_provider[i_idx] + codes_per_provider[j_idx] - intersection
    
    code_overlaps[start:end] = np.divide(
        intersection, union,
        out=np.zeros_like(intersection),
        where=union > 0
    )
    
    # Coverage: reuse intersection_mask for claims
    claims_i = claims_matrix[i_idx]
    claims_j = claims_matrix[j_idx]
    
    overlap_claims_i = np.array(claims_i.multiply(intersection_mask).sum(axis=1)).flatten()
    overlap_claims_j = np.array(claims_j.multiply(intersection_mask).sum(axis=1)).flatten()
    
    total_i = claims_per_provider[i_idx]
    total_j = claims_per_provider[j_idx]
    
    cov_i = np.divide(overlap_claims_i, total_i, out=np.zeros_like(overlap_claims_i), where=total_i > 0)
    cov_j = np.divide(overlap_claims_j, total_j, out=np.zeros_like(overlap_claims_j), where=total_j > 0)
    
    coverage_overlaps[start:end] = (cov_i + cov_j) * 0.5
    
    # Weighted Jaccard: (A's claims in common + B's claims in common) / (A's total + B's total)
    total_both = total_i + total_j
    weighted_jaccards[start:end] = np.divide(
        overlap_claims_i + overlap_claims_j, total_both,
        out=np.zeros_like(total_both),
        where=total_both > 0
    )

# =============================================================================
print("\n" + "="*80)
print("STEP 6: VERIFY SCALES")
print("="*80)

print(f"Code overlap (Jaccard):  min={code_overlaps.min():.4f}, max={code_overlaps.max():.4f}, mean={code_overlaps.mean():.4f}")
print(f"Weighted Jaccard:        min={weighted_jaccards.min():.4f}, max={weighted_jaccards.max():.4f}, mean={weighted_jaccards.mean():.4f}")
print(f"Coverage overlap:        min={coverage_overlaps.min():.4f}, max={coverage_overlaps.max():.4f}, mean={coverage_overlaps.mean():.4f}")

assert code_overlaps.max() <= 1.0, "Code overlap > 1!"
assert weighted_jaccards.max() <= 1.0, "Weighted Jaccard > 1!"
assert coverage_overlaps.max() <= 1.0, "Coverage overlap > 1!"

print("✓ All metrics in 0-1 range")

# Distribution
print(f"\nDistribution:")
print(f"  Code overlap = 0:       {(code_overlaps == 0).sum():,} ({100*(code_overlaps == 0).mean():.1f}%)")
print(f"  Code overlap > 0.5:     {(code_overlaps > 0.5).sum():,} ({100*(code_overlaps > 0.5).mean():.1f}%)")
print(f"  Weighted Jaccard = 0:   {(weighted_jaccards == 0).sum():,} ({100*(weighted_jaccards == 0).mean():.1f}%)")
print(f"  Weighted Jaccard > 0.5: {(weighted_jaccards > 0.5).sum():,} ({100*(weighted_jaccards > 0.5).mean():.1f}%)")
print(f"  Coverage overlap = 0:   {(coverage_overlaps == 0).sum():,} ({100*(coverage_overlaps == 0).mean():.1f}%)")
print(f"  Coverage overlap > 0.5: {(coverage_overlaps > 0.5).sum():,} ({100*(coverage_overlaps > 0.5).mean():.1f}%)")

# =============================================================================
print("\n" + "="*80)
print("STEP 7: SAVE OUTPUTS")
print("="*80)

# Create DataFrame efficiently with correct dtypes
pair_labels = pd.DataFrame({
    'i': random_pairs[:, 0].astype(np.int32),
    'j': random_pairs[:, 1].astype(np.int32),
    'code_overlap': code_overlaps,
    'weighted_jaccard': weighted_jaccards,
    'coverage_overlap': coverage_overlaps
})

pair_labels.to_parquet('data41.parquet', index=False, compression='snappy')
print("✓ pair_labels.parquet")

with open('data43.pkl', 'wb') as f:
    pickle.dump(pin_to_idx, f)
print("✓ pin_to_idx.pkl")

with open('data30.pkl', 'wb') as f:
    pickle.dump(idx_to_pin, f)
print("✓ idx_to_pin.pkl")

with open('data1.pkl', 'wb') as f:
    pickle.dump(common_pins, f)
print("✓ aligned_pins.pkl")

# =============================================================================
elapsed = time.time() - start_time

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"Time: {elapsed/60:.1f} minutes")
print(f"\nOutputs:")
print(f"  pair_labels.parquet: {len(pair_labels):,} pairs")
print(f"  pin_to_idx.pkl: {n_providers:,} providers")
print(f"  idx_to_pin.pkl: {n_providers:,} providers")
print(f"  aligned_pins.pkl: {n_providers:,} providers")
print(f"\nMetrics (all 0-1 scale):")
print(f"  Code overlap (Jaccard) mean: {code_overlaps.mean():.4f}")
print(f"  Weighted Jaccard mean:       {weighted_jaccards.mean():.4f}")
print(f"  Coverage overlap mean:       {coverage_overlaps.mean():.4f}")
print("="*80)
