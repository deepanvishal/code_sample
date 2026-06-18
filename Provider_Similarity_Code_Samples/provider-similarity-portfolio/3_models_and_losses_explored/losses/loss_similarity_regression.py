# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import gc
import torch

print("="*80)
print("CLEARING MEMORY")
print("="*80)

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    print(f"GPU memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

gc.collect()
print("Memory cleared\n")

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import time
import random

print("\n" + "="*80)
print("NOTEBOOK 3: GAT TRAINING WITH SIMILARITY REGRESSION")
print("="*80)
print("Key features:")
print("  - Similarity loss: MSE regression (not contrastive margin)")
print("  - Target: CODE_WEIGHT*code_overlap + (1-CODE_WEIGHT)*coverage_overlap")
print("  - Both metrics normalized to 0-1 scale")
print("  - 2x weight boost on similarity loss")
print("  - Loads pair_labels.parquet from Notebook 2")

SEED = 42
SIMILARITY_BOOST = 2.0
CODE_WEIGHT = 0.5  # Balance between code overlap and coverage overlap

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "="*80)
print("STEP 1: LOAD DATA")
print("="*80)

with open('data51.pkl', 'rb') as f:
    provider_code_embeddings = pickle.load(f)

with open('data59.pkl', 'rb') as f:
    provider_specialty_labels = pickle.load(f)

pair_labels_df = pd.read_parquet('data41.parquet')

with open('data43.pkl', 'rb') as f:
    pin_to_idx = pickle.load(f)

with open('data1.pkl', 'rb') as f:
    aligned_pins = pickle.load(f)

print(f"Providers: {len(provider_code_embeddings):,}")
print(f"Pairs for similarity training: {len(pair_labels_df):,}")

# Precompute combined target and build lookup dict
print("\nBuilding pair lookup (one-time cost)...")

# Verify both metrics are 0-1 scale
print(f"Code overlap:     min={pair_labels_df['code_overlap'].min():.4f}, max={pair_labels_df['code_overlap'].max():.4f}")
print(f"Coverage overlap: min={pair_labels_df['coverage_overlap'].min():.4f}, max={pair_labels_df['coverage_overlap'].max():.4f}")

assert pair_labels_df['code_overlap'].max() <= 1.0, "Code overlap not normalized!"
assert pair_labels_df['coverage_overlap'].max() <= 1.0, "Coverage overlap not normalized!"
print("✓ Both metrics in 0-1 range")

pair_labels_df['combined_target'] = CODE_WEIGHT * pair_labels_df['code_overlap'] + (1 - CODE_WEIGHT) * pair_labels_df['coverage_overlap']

# Create lookup using vectorized operations
pair_i = pair_labels_df['i'].values.astype(np.int32)
pair_j = pair_labels_df['j'].values.astype(np.int32)
pair_targets = pair_labels_df['combined_target'].values.astype(np.float32)

# Build dict using zip (faster than loop)
pair_to_target = dict(zip(zip(pair_i, pair_j), pair_targets))

print(f"Pair lookup built: {len(pair_to_target):,} pairs")
print(f"Combined target: mean={pair_targets.mean():.4f}, median={np.median(pair_targets):.4f}, max={pair_targets.max():.4f}")

# Pre-build set of keys for O(1) existence check
pair_keys_set = set(pair_to_target.keys())

# Pre-build numpy arrays for fast batch lookup (used in similarity_loss)
pair_i_arr = pair_i
pair_j_arr = pair_j
pair_targets_arr = pair_targets

# Build reverse lookup: pin -> matrix index (as numpy array for speed)
n_providers = len(aligned_pins)
pin_to_idx_arr = np.full(max(aligned_pins) + 1, -1, dtype=np.int32)
for pin, idx in pin_to_idx.items():
    pin_to_idx_arr[pin] = idx

del pair_labels_df  # Free memory
gc.collect()

specialties = sorted(set(provider_specialty_labels.values()))
specialty_to_idx = {spec: idx for idx, spec in enumerate(specialties)}
num_specialties = len(specialties)

labeled_provider_ids = [pid for pid in aligned_pins if pid in provider_specialty_labels]
unlabeled_provider_ids = [pid for pid in aligned_pins if pid not in provider_specialty_labels]

print(f"Specialties: {num_specialties}, Labeled: {len(labeled_provider_ids):,}, Unlabeled: {len(unlabeled_provider_ids):,}")

print("\n" + "="*80)
print("STEP 2: MODEL")
print("="*80)

class SemiSupervisedGAT(nn.Module):
    def __init__(self, code_embedding_dim, num_heads, provider_embedding_dim, num_specialties, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.provider_embedding_dim = provider_embedding_dim
        
        self.attention_weights = nn.Linear(code_embedding_dim, num_heads, bias=False)
        self.output_transform = nn.Linear(code_embedding_dim * num_heads, provider_embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(provider_embedding_dim, num_specialties)
        self.decoder = nn.Linear(provider_embedding_dim, code_embedding_dim)
        
        # Uncertainty weights for 3 losses
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_rec = nn.Parameter(torch.zeros(1))
        self.log_var_sim = nn.Parameter(torch.zeros(1))
    
    def forward(self, code_embeddings, attention_masks):
        scores = self.attention_weights(code_embeddings)
        scores = scores.masked_fill(attention_masks.unsqueeze(-1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=1)
        attn = attn.masked_fill(attention_masks.unsqueeze(-1) == 0, 0.0)
        
        head_outputs = (code_embeddings.unsqueeze(-1) * attn.unsqueeze(2)).sum(dim=1)
        head_outputs = head_outputs.view(head_outputs.shape[0], -1)
        
        provider_embedding = self.dropout_layer(self.output_transform(head_outputs))
        logits = self.classifier(provider_embedding)
        reconstructed = self.decoder(provider_embedding)
        
        return logits, provider_embedding, reconstructed

print("✓ GAT with 3 uncertainty-weighted losses")

print("\n" + "="*80)
print("STEP 3: DATA PREPARATION")
print("="*80)

class ProviderDataset(Dataset):
    def __init__(self, provider_ids, code_embeddings_dict, labels_dict=None):
        self.provider_ids = provider_ids
        self.code_embeddings_dict = code_embeddings_dict
        self.labels_dict = labels_dict
    
    def __len__(self):
        return len(self.provider_ids)
    
    def __getitem__(self, idx):
        pid = self.provider_ids[idx]
        emb = self.code_embeddings_dict[pid]
        label = self.labels_dict.get(pid, -1) if self.labels_dict else -1
        return pid, emb, label

def collate_fn(batch):
    pids, embs, labels = zip(*batch)
    
    max_len = max(e.shape[0] for e in embs)
    batch_size = len(embs)
    emb_dim = embs[0].shape[1]
    
    # Pre-allocate with correct dtype
    padded = torch.zeros(batch_size, max_len, emb_dim, dtype=torch.float32)
    masks = torch.zeros(batch_size, max_len, dtype=torch.float32)
    
    for i, e in enumerate(embs):
        length = e.shape[0]
        padded[i, :length] = e
        masks[i, :length] = 1.0
    
    return (torch.tensor(pids, dtype=torch.long), padded, masks, 
            torch.tensor(labels, dtype=torch.long))

labels_dict_indexed = {pid: specialty_to_idx[provider_specialty_labels[pid]] 
                       for pid in labeled_provider_ids}

train_idx, temp_idx = train_test_split(
    range(len(labeled_provider_ids)), test_size=0.3, random_state=SEED,
    stratify=[labels_dict_indexed[labeled_provider_ids[i]] for i in range(len(labeled_provider_ids))]
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=SEED,
    stratify=[labels_dict_indexed[labeled_provider_ids[i]] for i in temp_idx]
)

train_providers = [labeled_provider_ids[i] for i in train_idx]
val_providers = [labeled_provider_ids[i] for i in val_idx]
test_providers = [labeled_provider_ids[i] for i in test_idx]
all_train_providers = train_providers + unlabeled_provider_ids

n_labeled, n_unlabeled = len(train_providers), len(unlabeled_provider_ids)
n_total = len(all_train_providers)

sampling_weights = [0.5 * n_total / n_labeled if pid in labels_dict_indexed 
                   else 0.5 * n_total / n_unlabeled for pid in all_train_providers]

sampler = WeightedRandomSampler(sampling_weights, n_total, replacement=True)

train_dataset = ProviderDataset(all_train_providers, provider_code_embeddings, labels_dict_indexed)
val_dataset = ProviderDataset(val_providers, provider_code_embeddings, labels_dict_indexed)
test_dataset = ProviderDataset(test_providers, provider_code_embeddings, labels_dict_indexed)

BATCH_SIZE = 256
NUM_WORKERS = 8

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                         collate_fn=collate_fn, num_workers=NUM_WORKERS,
                         pin_memory=True, persistent_workers=True, generator=g)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_providers):,} labeled + {len(unlabeled_provider_ids):,} unlabeled")
print(f"Batches/epoch: {len(train_loader)}")

print("\n" + "="*80)
print("STEP 4: LOSS FUNCTIONS")
print("="*80)

def similarity_loss_mse(embeddings, provider_ids, pin_to_idx_arr, pair_to_target, pair_keys_set):
    """
    MSE loss between cosine similarity and combined overlap target.
    Optimized: minimal CPU work, vectorized operations.
    """
    # Get provider IDs as numpy (already on CPU from collate_fn)
    pids_np = provider_ids.numpy()
    
    # Fast lookup using pre-built array
    max_pid = pin_to_idx_arr.shape[0] - 1
    valid_pids_mask = pids_np <= max_pid
    
    matrix_indices = np.where(valid_pids_mask, pin_to_idx_arr[np.minimum(pids_np, max_pid)], -1)
    valid_mask = matrix_indices >= 0
    
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=embeddings.device), 0
    
    valid_batch_idx = np.where(valid_mask)[0]
    valid_matrix_idx = matrix_indices[valid_mask]
    n = len(valid_batch_idx)
    
    # Generate upper triangle pairs
    triu_i, triu_j = np.triu_indices(n, k=1)
    
    # Get matrix indices, ensure i < j for lookup
    mat_i = valid_matrix_idx[triu_i]
    mat_j = valid_matrix_idx[triu_j]
    keys_min = np.minimum(mat_i, mat_j)
    keys_max = np.maximum(mat_i, mat_j)
    
    # Vectorized pair existence check using numba-style approach
    # Filter pairs that exist (still need Python but minimized)
    n_potential = len(keys_min)
    
    if n_potential > 5000:
        # Sample for large batches to avoid O(n^2) blowup
        sample_idx = np.random.choice(n_potential, min(5000, n_potential), replace=False)
        keys_min = keys_min[sample_idx]
        keys_max = keys_max[sample_idx]
        triu_i = triu_i[sample_idx]
        triu_j = triu_j[sample_idx]
    
    # Check existence (unavoidable but now on smaller set)
    valid_pairs = []
    valid_targets = []
    valid_i = []
    valid_j = []
    
    for idx in range(len(keys_min)):
        key = (keys_min[idx], keys_max[idx])
        if key in pair_keys_set:
            valid_pairs.append(idx)
            valid_targets.append(pair_to_target[key])
            valid_i.append(triu_i[idx])
            valid_j.append(triu_j[idx])
    
    if len(valid_pairs) == 0:
        return torch.tensor(0.0, device=embeddings.device), 0
    
    valid_i = np.array(valid_i)
    valid_j = np.array(valid_j)
    
    # Compute cosine similarities on GPU
    valid_indices = torch.tensor(valid_batch_idx, device=embeddings.device, dtype=torch.long)
    valid_embeddings = embeddings[valid_indices]
    emb_norm = F.normalize(valid_embeddings, p=2, dim=1)
    
    emb_i = emb_norm[valid_i]
    emb_j = emb_norm[valid_j]
    pred_sims = (emb_i * emb_j).sum(dim=1)
    
    targets = torch.tensor(valid_targets, device=embeddings.device, dtype=torch.float32)
    
    loss = F.mse_loss(pred_sims, targets)
    
    return loss, len(valid_targets)

def reconstruction_loss_fast(code_embeddings, reconstructed, attention_masks):
    diff = code_embeddings - reconstructed.unsqueeze(1)
    masked_diff = diff * attention_masks.unsqueeze(-1)
    return (masked_diff ** 2).sum() / attention_masks.sum()

print("✓ Similarity loss: MSE regression on cosine sim vs combined target")
print("✓ Reconstruction loss: MSE on decoder output")
print(f"✓ Similarity boost: {SIMILARITY_BOOST}x")

print("\n" + "="*80)
print("STEP 5: INITIALIZE")
print("="*80)

CODE_EMBEDDING_DIM = 128
NUM_HEADS = 4
PROVIDER_EMBEDDING_DIM = 512

LEARNING_RATE = 0.001 * (256 / 128) ** 0.5
MAX_EPOCHS = 150
PATIENCE = 20
VALIDATE_EVERY = 5

model = SemiSupervisedGAT(CODE_EMBEDDING_DIM, NUM_HEADS, PROVIDER_EMBEDDING_DIM, 
                          num_specialties, dropout=0.2).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"LR: {LEARNING_RATE:.5f}, Validate every: {VALIDATE_EVERY} epochs")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
criterion_cls = nn.CrossEntropyLoss()

print("\n" + "="*80)
print("STEP 6: TRAINING")
print("="*80)

best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
history = []

print(f"{'Ep':<4} {'Loss':<8} {'VAcc':<7} {'Pairs':<6} {'Time':<6} {'Status':<12}")
print("-" * 50)

start_time = time.time()

for epoch in range(MAX_EPOCHS):
    epoch_start = time.time()
    model.train()
    
    epoch_loss, epoch_pairs = 0, 0
    
    for pids, code_embs, masks, labels in train_loader:
        # pids stays on CPU (used for similarity lookup)
        code_embs = code_embs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        logits, embeddings, reconstructed = model(code_embs, masks)
        
        # Loss 1: Classification (labeled only)
        labeled_mask = labels != -1
        if labeled_mask.any():
            loss_cls = criterion_cls(logits[labeled_mask], labels[labeled_mask])
        else:
            loss_cls = torch.tensor(0.0, device=device)
        
        # Loss 2: Reconstruction
        loss_rec = reconstruction_loss_fast(code_embs, reconstructed, masks)
        
        # Loss 3: Similarity (MSE regression)
        loss_sim, n_pairs = similarity_loss_mse(embeddings, pids, pin_to_idx_arr, pair_to_target, pair_keys_set)
        
        # Uncertainty weighting with 2x boost on similarity
        precision_cls = torch.exp(-model.log_var_cls)
        precision_rec = torch.exp(-model.log_var_rec)
        precision_sim = torch.exp(-model.log_var_sim)
        
        total_loss = (
            precision_cls * loss_cls + model.log_var_cls +
            precision_rec * loss_rec + model.log_var_rec +
            SIMILARITY_BOOST * (precision_sim * loss_sim + model.log_var_sim)
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += total_loss.item()
        epoch_pairs += n_pairs
    
    epoch_loss /= len(train_loader)
    avg_pairs = epoch_pairs / len(train_loader)
    epoch_time = time.time() - epoch_start
    
    if (epoch + 1) % VALIDATE_EVERY == 0 or epoch == 0:
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for pids, code_embs, masks, labels in val_loader:
                code_embs = code_embs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                logits, _, _ = model(code_embs, masks)
                # labels already contains correct indices from dataset
                labels_batch = labels.to(device, non_blocking=True)
                
                val_correct += (logits.argmax(1) == labels_batch).sum().item()
                val_total += len(labels_batch)
        
        val_acc = 100 * val_correct / val_total
        
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            status = "✓ Best"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'val_acc': val_acc}, 'best_model.pt')
        else:
            patience_counter += VALIDATE_EVERY
            status = f"P{patience_counter}/{PATIENCE}"
        
        print(f"{epoch+1:<4} {epoch_loss:<8.4f} {val_acc:<7.2f} {avg_pairs:<6.0f} {epoch_time:<6.2f} {status:<12}")
        
        history.append({'epoch': epoch, 'train_loss': epoch_loss, 'val_acc': val_acc})
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    else:
        print(f"{epoch+1:<4} {epoch_loss:<8.4f} {'--':<7} {avg_pairs:<6.0f} {epoch_time:<6.2f}")

total_time = time.time() - start_time
print(f"\nTraining: {total_time/60:.1f} min, Best: epoch {best_epoch+1}, acc {best_val_acc:.2f}%")

print("\n" + "="*80)
print("STEP 7: TEST")
print("="*80)

checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
test_correct, test_total = 0, 0

with torch.no_grad():
    for pids, code_embs, masks, labels in test_loader:
        code_embs = code_embs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        logits, _, _ = model(code_embs, masks)
        labels_batch = labels.to(device, non_blocking=True)
        
        test_correct += (logits.argmax(1) == labels_batch).sum().item()
        test_total += len(labels_batch)

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

print("\n" + "="*80)
print("STEP 8: EXTRACT EMBEDDINGS")
print("="*80)

print(f"Extracting {len(aligned_pins):,} embeddings...")

all_embeddings = []
model.eval()

extract_dataset = ProviderDataset(aligned_pins, provider_code_embeddings, None)
extract_loader = DataLoader(extract_dataset, batch_size=512, shuffle=False,
                           collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

with torch.no_grad():
    for pids, code_embs, masks, _ in extract_loader:
        code_embs = code_embs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        _, embs, _ = model(code_embs, masks)
        all_embeddings.append(embs.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0)
print(f"✓ Extracted {all_embeddings.shape[0]:,} embeddings")

embeddings_np = all_embeddings.numpy()
all_provider_embeddings = {pid: embeddings_np[i].copy() for i, pid in enumerate(aligned_pins)}

print("\n" + "="*80)
print("STEP 9: CREATE DATAFRAME")
print("="*80)

embeddings_df = pd.DataFrame(
    embeddings_np,
    columns=[f'emb_{i}' for i in range(PROVIDER_EMBEDDING_DIM)]
)
embeddings_df.insert(0, 'provider_id', aligned_pins)
embeddings_df['provider_id'] = embeddings_df['provider_id'].astype(int)

print(f"DataFrame: {embeddings_df.shape}")

print("\n" + "="*80)
print("STEP 10: SAVE")
print("="*80)

with open('data74.pkl', 'wb') as f:
    pickle.dump(history, f)
print("✓ training_history.pkl")

with open('data55.pkl', 'wb') as f:
    pickle.dump(all_provider_embeddings, f)
print("✓ provider_embeddings_final.pkl")

embeddings_df.to_csv('data53.csv', index=False)
print("✓ provider_embeddings_df.csv")

embeddings_df.to_pickle('data54.pkl')
print("✓ provider_embeddings_df.pkl")

results = {
    'best_epoch': best_epoch, 'best_val_acc': best_val_acc, 'test_acc': test_acc,
    'total_time': total_time, 'num_providers': len(embeddings_df),
    'similarity_boost': SIMILARITY_BOOST, 'code_weight': CODE_WEIGHT
}

with open('data75.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✓ training_results.pkl")

print("\n" + "="*80)
print("STEP 11: COMPUTE SIMILARITY RANKINGS FOR PAIRS")
print("="*80)

MODEL_NAME = "model1"
PAIRS_FILE = "data6.csv"
OUTPUT_FILE = "data58.csv"

if os.path.exists(OUTPUT_FILE):
    pairs_df = pd.read_csv(OUTPUT_FILE)
    print(f"Loaded existing results: {pairs_df.shape}")
else:
    pairs_df = pd.read_csv(PAIRS_FILE)
    print(f"Loaded fresh pairs: {pairs_df.shape}")

existing_sim_cols = [c for c in pairs_df.columns if c.startswith(f'{MODEL_NAME}_sim_run_')]
if existing_sim_cols:
    last_run = max(int(c.split('_')[-1]) for c in existing_sim_cols)
    run_id = last_run + 1
else:
    run_id = 1

print(f"This is {MODEL_NAME} run {run_id}")

# Vectorized similarity computation
print("Computing similarities (vectorized)...")

emb_tensor = torch.tensor(embeddings_np, dtype=torch.float32)
emb_norm = F.normalize(emb_tensor, p=2, dim=1)

# Build lookup as numpy array for O(1) index access
pin_set = set(aligned_pins)
pin_to_emb_idx = {pid: i for i, pid in enumerate(aligned_pins)}

primary_pins = pairs_df['primary_pin'].values
alt_pins = pairs_df['alternative_pin'].values

# Vectorized valid check using numpy isin
primary_valid = np.isin(primary_pins, aligned_pins)
alt_valid = np.isin(alt_pins, aligned_pins)
valid_mask = primary_valid & alt_valid

similarities = np.full(len(pairs_df), np.nan, dtype=np.float32)

if valid_mask.any():
    valid_primary = primary_pins[valid_mask]
    valid_alt = alt_pins[valid_mask]
    
    # Vectorized index lookup
    primary_idx = np.array([pin_to_emb_idx[p] for p in valid_primary])
    alt_idx = np.array([pin_to_emb_idx[a] for a in valid_alt])
    
    # Batch cosine similarity
    emb_primary = emb_norm[primary_idx]
    emb_alt = emb_norm[alt_idx]
    
    sims = (emb_primary * emb_alt).sum(dim=1).numpy()
    similarities[valid_mask] = sims

sim_col = f'{MODEL_NAME}_sim_run_{run_id}'
pairs_df[sim_col] = similarities

rank_col = f'{MODEL_NAME}_rank_run_{run_id}'
pairs_df[rank_col] = pairs_df.groupby('primary_pin')[sim_col].rank(ascending=False, method='min')

pairs_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved: {OUTPUT_FILE}")
print(f"  Added columns: {sim_col}, {rank_col}")
print(f"  Shape: {pairs_df.shape}")

valid_sims = pairs_df[sim_col].dropna()
print(f"\nSimilarity stats:")
print(f"  Mean: {valid_sims.mean():.4f}")
print(f"  Std:  {valid_sims.std():.4f}")
print(f"  Min:  {valid_sims.min():.4f}")
print(f"  Max:  {valid_sims.max():.4f}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"Test accuracy: {test_acc:.2f}%")
print(f"Training time: {total_time/60:.1f} min")
print(f"Embeddings: {len(embeddings_df):,} × {PROVIDER_EMBEDDING_DIM}")
print(f"Rankings: {MODEL_NAME} run {run_id} added")
print(f"Similarity boost: {SIMILARITY_BOOST}x")
print(f"Code weight: {CODE_WEIGHT}")
print("="*80)
