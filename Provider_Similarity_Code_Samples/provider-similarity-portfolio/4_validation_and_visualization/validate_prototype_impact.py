# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

"""
PROTOTYPE IMPACT ANALYSIS
==========================

Adds pre and post prototype similarity to the top 10 county CSV.

Output:
  - all_providers_top10_alternatives_with_prototype.csv
  - tower_weights_comparison.csv (Table 1)
  - tower_weights_by_label.csv (Table 2)
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

print("\n" + "="*80)
print("PROTOTYPE IMPACT ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")

embeddings_df = pd.read_parquet('data20.parquet')
all_pins = embeddings_df['PIN'].values
emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
raw_embeddings = embeddings_df[emb_cols].values
print(f"  Raw embeddings: {raw_embeddings.shape}")

weighted_embeddings = np.load('gat_weighted_embeddings.npy')
print(f"  Weighted embeddings: {weighted_embeddings.shape}")

top10_df = pd.read_csv('data4.csv')
print(f"  Top 10 CSV: {top10_df.shape}")

with open('data27.pkl', 'rb') as f:
    metadata = pickle.load(f)

tower_dims = metadata['tower_dims']
idx_to_label = metadata['idx_to_label']
n_labels = metadata['n_specialties']
embedding_dim = metadata['embedding_dim']

print(f"  Labels: {n_labels}")

pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

# ============================================================================
# LOAD TRAINED MODEL FOR TABLE 2
# ============================================================================

print("\nLoading trained model...")

class PrototypeWeightModel(nn.Module):
    def __init__(self, n_labels, embedding_dim, n_towers=6):
        super().__init__()
        self.n_labels = n_labels
        self.embedding_dim = embedding_dim
        self.n_towers = n_towers
        self.prototypes = nn.Parameter(torch.randn(n_labels, embedding_dim) * 0.1)
        self.weight_profiles = nn.Parameter(torch.ones(n_labels, n_towers))
        self.temperature = nn.Parameter(torch.tensor(1.0))

checkpoint = torch.load('gat_trained_prototype_model.pth', map_location='cpu')
model = PrototypeWeightModel(n_labels, embedding_dim, n_towers=6)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✓ Model loaded")

# ============================================================================
# TABLE 1: TOWER WEIGHTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("TABLE 1: TOWER WEIGHTS")
print("="*80)

tower_names = ['Procedures', 'Diagnoses', 'Demographics', 'Place', 'Cost', 'PIN Summary']
tower_keys = ['procedures', 'diagnoses', 'demographics', 'place', 'cost', 'pin']

pre_weights = []
for key in tower_keys:
    start, end = tower_dims[key]
    dims = end - start
    weight = dims / embedding_dim
    pre_weights.append(weight)

with torch.no_grad():
    raw_weight_profiles = model.weight_profiles.numpy()
    post_weight_profiles = F.softmax(torch.tensor(raw_weight_profiles), dim=1).numpy()

post_weights_avg = post_weight_profiles.mean(axis=0)

table1_data = []
for i, tower in enumerate(tower_names):
    start, end = tower_dims[tower_keys[i]]
    dims = end - start
    table1_data.append({
        'Tower': tower,
        'Dimensions': dims,
        'Pre Prototype Weight (%)': round(pre_weights[i] * 100, 2),
        'Post Prototype Weight Avg (%)': round(post_weights_avg[i] * 100, 2),
        'Change (%)': round((post_weights_avg[i] - pre_weights[i]) * 100, 2)
    })

table1_df = pd.DataFrame(table1_data)
table1_df.to_csv('data73.csv', index=False)
print("✓ Saved: tower_weights_comparison.csv")
print(table1_df.to_string(index=False))

# ============================================================================
# TABLE 2: WEIGHTS BY LABEL
# ============================================================================

print("\n" + "="*80)
print("TABLE 2: WEIGHTS BY LABEL")
print("="*80)

table2_data = []
for idx in range(n_labels):
    label = idx_to_label[idx]
    weights = post_weight_profiles[idx]
    row = {'Label': label}
    for i, tower in enumerate(tower_names):
        row[tower + ' (%)'] = round(weights[i] * 100, 2)
    table2_data.append(row)

table2_df = pd.DataFrame(table2_data)
table2_df = table2_df.sort_values('Label').reset_index(drop=True)
table2_df.to_csv('data72.csv', index=False)
print("✓ Saved: tower_weights_by_label.csv")
print(table2_df.to_string(index=False))

# ============================================================================
# ADD PRE/POST SIMILARITY TO CSV
# ============================================================================

print("\n" + "="*80)
print("ADDING PRE/POST PROTOTYPE SIMILARITY")
print("="*80)

raw_norm = raw_embeddings / (np.linalg.norm(raw_embeddings, axis=1, keepdims=True) + 1e-8)
weighted_norm = weighted_embeddings / (np.linalg.norm(weighted_embeddings, axis=1, keepdims=True) + 1e-8)

pre_sims = []
post_sims = []

for _, row in tqdm(top10_df.iterrows(), total=len(top10_df), desc="Computing similarities"):
    primary_pin = row['primary_pin']
    alt_pin = row['alternative_pin']
    
    if primary_pin not in pin_to_idx or alt_pin not in pin_to_idx:
        pre_sims.append(np.nan)
        post_sims.append(np.nan)
        continue
    
    primary_idx = pin_to_idx[primary_pin]
    alt_idx = pin_to_idx[alt_pin]
    
    pre_sim = np.dot(raw_norm[primary_idx], raw_norm[alt_idx])
    post_sim = np.dot(weighted_norm[primary_idx], weighted_norm[alt_idx])
    
    pre_sims.append(round(pre_sim, 6))
    post_sims.append(round(post_sim, 6))

top10_df['pre_prototype_similarity'] = pre_sims
top10_df['post_prototype_similarity'] = post_sims

top10_df.to_csv('data8.csv', index=False)
print(f"\n✓ Saved: all_providers_top10_alternatives_with_prototype.csv")
print(f"  Rows: {len(top10_df):,}")
print(f"  Columns added: pre_prototype_similarity, post_prototype_similarity")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print("\nOutput files:")
print("  1. all_providers_top10_alternatives_with_prototype.csv")
print("  2. tower_weights_comparison.csv")
print("  3. tower_weights_by_label.csv")
