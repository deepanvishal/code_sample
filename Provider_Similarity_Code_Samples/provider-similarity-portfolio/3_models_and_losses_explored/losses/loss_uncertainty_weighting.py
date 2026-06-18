# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
import time
import random
import os

print("\n" + "="*80)
print("NOTEBOOK 3: GAT TRAINING WITH UNCERTAINTY WEIGHTING")
print("="*80)
print("Multi-task learning with three losses:")
print("  1. Classification Loss (specialty prediction, labeled only)")
print("  2. Reconstruction Loss (code recovery, all providers)")
print("  3. Contrastive Loss (similarity learning, all providers via pairs)")
print("\nLoss Balancing: Uncertainty Weighting (Kendall et al., 2018)")
print("  - Learn task-specific uncertainty parameters")
print("  - Automatic balancing without manual tuning")
print("  - Battle-tested: 6000+ citations, used in production")

SEED = 42

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "="*80)
print("STEP 1: LOAD DATA FROM NOTEBOOK 1 & 2")
print("="*80)

print("\nLoading Notebook 1 outputs...")
with open('data51.pkl', 'rb') as f:
    provider_code_embeddings = pickle.load(f)

with open('data59.pkl', 'rb') as f:
    provider_specialty_labels = pickle.load(f)

print(f"Providers with embeddings: {len(provider_code_embeddings):,}")
print(f"Labeled providers: {len(provider_specialty_labels):,}")

print("\nLoading Notebook 2 outputs...")
similarity_matrix = load_npz('data64.npz')

with open('data43.pkl', 'rb') as f:
    pin_to_idx = pickle.load(f)

with open('data30.pkl', 'rb') as f:
    idx_to_pin = pickle.load(f)

with open('data1.pkl', 'rb') as f:
    aligned_pins = pickle.load(f)

print(f"Similarity matrix shape: {similarity_matrix.shape}")
print(f"Non-zero pairs: {similarity_matrix.nnz:,}")
print(f"Aligned providers: {len(aligned_pins):,}")

specialties = sorted(set(provider_specialty_labels.values()))
specialty_to_idx = {spec: idx for idx, spec in enumerate(specialties)}
num_specialties = len(specialties)

print(f"\nSpecialties: {num_specialties}")

labeled_provider_ids = [pid for pid in aligned_pins if pid in provider_specialty_labels]
unlabeled_provider_ids = [pid for pid in aligned_pins if pid not in provider_specialty_labels]

print(f"Labeled: {len(labeled_provider_ids):,}")
print(f"Unlabeled: {len(unlabeled_provider_ids):,}")

print("\n" + "="*80)
print("STEP 2: MODEL ARCHITECTURE")
print("="*80)

class SemiSupervisedGAT(nn.Module):
    def __init__(self, code_embedding_dim, num_heads, provider_embedding_dim, num_specialties, dropout=0.2):
        super().__init__()
        self.code_embedding_dim = code_embedding_dim
        self.num_heads = num_heads
        self.provider_embedding_dim = provider_embedding_dim
        self.dropout = dropout
        
        self.attention_weights = nn.ModuleList([
            nn.Linear(code_embedding_dim, 1, bias=False) for _ in range(num_heads)
        ])
        
        self.output_transform = nn.Linear(code_embedding_dim * num_heads, provider_embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(provider_embedding_dim, num_specialties)
        self.decoder = nn.Linear(provider_embedding_dim, code_embedding_dim)
        
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_rec = nn.Parameter(torch.zeros(1))
        self.log_var_con = nn.Parameter(torch.zeros(1))
    
    def forward(self, code_embeddings_batch, attention_masks_batch):
        batch_size = code_embeddings_batch.shape[0]
        all_head_outputs = []
        
        for head_idx in range(self.num_heads):
            attention_layer = self.attention_weights[head_idx]
            scores = attention_layer(code_embeddings_batch).squeeze(-1)
            
            scores = scores.masked_fill(attention_masks_batch == 0, float('-inf'))
            attention_weights = F.softmax(scores, dim=1)
            attention_weights = attention_weights.masked_fill(attention_masks_batch == 0, 0.0)
            
            weighted_codes = code_embeddings_batch * attention_weights.unsqueeze(-1)
            head_output = weighted_codes.sum(dim=1)
            all_head_outputs.append(head_output)
        
        concatenated = torch.cat(all_head_outputs, dim=1)
        provider_embedding = self.output_transform(concatenated)
        provider_embedding = self.dropout_layer(provider_embedding)
        
        logits = self.classifier(provider_embedding)
        reconstructed = self.decoder(provider_embedding)
        
        return logits, provider_embedding, reconstructed

print("Model components:")
print("  - Multi-head GAT encoder")
print("  - Classification head (specialty prediction)")
print("  - Reconstruction decoder (code recovery)")
print("  - Uncertainty parameters: log_var_cls, log_var_rec, log_var_con")

print("\n" + "="*80)
print("STEP 3: DATA PREPARATION")
print("="*80)

class ProviderDataset(Dataset):
    def __init__(self, provider_ids, code_embeddings_dict, labels_dict=None):
        self.provider_ids = provider_ids
        self.code_embeddings_dict = code_embeddings_dict
        self.labels_dict = labels_dict
        self.has_labels = labels_dict is not None
    
    def __len__(self):
        return len(self.provider_ids)
    
    def __getitem__(self, idx):
        provider_id = self.provider_ids[idx]
        code_embeddings = self.code_embeddings_dict[provider_id]
        
        if self.has_labels:
            label = self.labels_dict[provider_id]
            return provider_id, code_embeddings, label
        else:
            return provider_id, code_embeddings

def collate_fn(batch):
    if len(batch[0]) == 3:
        provider_ids, code_embeddings_list, labels = zip(*batch)
        has_labels = True
    else:
        provider_ids, code_embeddings_list = zip(*batch)
        has_labels = False
    
    max_codes = max(emb.shape[0] for emb in code_embeddings_list)
    batch_size = len(code_embeddings_list)
    embedding_dim = code_embeddings_list[0].shape[1]
    
    padded_embeddings = torch.zeros(batch_size, max_codes, embedding_dim)
    attention_masks = torch.zeros(batch_size, max_codes)
    
    for i, emb in enumerate(code_embeddings_list):
        seq_len = emb.shape[0]
        padded_embeddings[i, :seq_len, :] = emb
        attention_masks[i, :seq_len] = 1
    
    if has_labels:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return provider_ids, padded_embeddings, attention_masks, labels_tensor
    else:
        return provider_ids, padded_embeddings, attention_masks

labels_dict_indexed = {pid: specialty_to_idx[provider_specialty_labels[pid]] 
                       for pid in labeled_provider_ids}

train_idx, temp_idx = train_test_split(
    range(len(labeled_provider_ids)), 
    test_size=0.3, 
    random_state=SEED, 
    stratify=[labels_dict_indexed[labeled_provider_ids[i]] for i in range(len(labeled_provider_ids))]
)

val_idx, test_idx = train_test_split(
    temp_idx, 
    test_size=0.5, 
    random_state=SEED,
    stratify=[labels_dict_indexed[labeled_provider_ids[i]] for i in temp_idx]
)

train_providers = [labeled_provider_ids[i] for i in train_idx]
val_providers = [labeled_provider_ids[i] for i in val_idx]
test_providers = [labeled_provider_ids[i] for i in test_idx]

all_train_providers = train_providers + unlabeled_provider_ids

print(f"Train (labeled):   {len(train_providers):,}")
print(f"Train (unlabeled): {len(unlabeled_provider_ids):,}")
print(f"Train (total):     {len(all_train_providers):,}")
print(f"Val:               {len(val_providers):,}")
print(f"Test:              {len(test_providers):,}")

train_dataset = ProviderDataset(all_train_providers, provider_code_embeddings, labels_dict_indexed)
val_dataset = ProviderDataset(val_providers, provider_code_embeddings, labels_dict_indexed)
test_dataset = ProviderDataset(test_providers, provider_code_embeddings, labels_dict_indexed)

BATCH_SIZE = 128
NUM_WORKERS = 4

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=collate_fn, 
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    collate_fn=collate_fn, 
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    collate_fn=collate_fn, 
    num_workers=NUM_WORKERS
)

print(f"\nBatch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

print("\n" + "="*80)
print("STEP 4: LOSS FUNCTIONS")
print("="*80)

def contrastive_loss(embeddings, provider_ids, similarity_matrix, pin_to_idx, margin=0.5):
    batch_size = len(provider_ids)
    total_loss = 0.0
    n_pairs = 0
    
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            pin_a = provider_ids[i]
            pin_b = provider_ids[j]
            
            if pin_a not in pin_to_idx or pin_b not in pin_to_idx:
                continue
            
            idx_a = pin_to_idx[pin_a]
            idx_b = pin_to_idx[pin_b]
            
            sim_label = similarity_matrix[idx_a, idx_b]
            
            if sim_label == 0:
                continue
            
            emb_sim = F.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            )[0]
            
            if sim_label == 1:
                loss = torch.clamp(margin - emb_sim, min=0)
            elif sim_label == -1:
                loss = torch.clamp(emb_sim + margin, min=0)
            else:
                continue
            
            total_loss += loss
            n_pairs += 1
    
    if n_pairs > 0:
        return total_loss / n_pairs, n_pairs
    else:
        return torch.tensor(0.0, device=embeddings.device), 0

def reconstruction_loss(code_embeddings, reconstructed, attention_masks):
    mask_expanded = attention_masks.unsqueeze(-1).expand_as(code_embeddings)
    masked_original = code_embeddings * mask_expanded
    masked_reconstructed = reconstructed.unsqueeze(1).expand_as(code_embeddings) * mask_expanded
    
    num_codes_per_sample = attention_masks.sum(dim=1, keepdim=True)
    loss = ((masked_original - masked_reconstructed) ** 2).sum() / num_codes_per_sample.sum()
    
    return loss

print("Loss 1: Classification (CrossEntropyLoss)")
print("  - Applied to labeled providers only")
print("  - Predicts specialty from provider embedding")

print("\nLoss 2: Reconstruction (MSE)")
print("  - Applied to all providers")
print("  - Recovers code embeddings from provider embedding")

print("\nLoss 3: Contrastive (Margin-based)")
print("  - Applied to pairs from similarity matrix")
print("  - Positive pairs: Pull together (similarity = 1)")
print("  - Negative pairs: Push apart (similarity = -1)")
print("  - Neutral pairs: Ignored (similarity = 0)")

print("\nUncertainty Weighting:")
print("  total_loss = precision_cls * loss_cls + log_var_cls")
print("             + precision_rec * loss_rec + log_var_rec")
print("             + precision_con * loss_con + log_var_con")
print("  where precision = exp(-log_var)")

print("\n" + "="*80)
print("STEP 5: MODEL INITIALIZATION")
print("="*80)

CODE_EMBEDDING_DIM = 128
NUM_HEADS = 4
PROVIDER_EMBEDDING_DIM = 512
DROPOUT = 0.2

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
GRAD_CLIP_NORM = 1.0
MAX_EPOCHS = 100
PATIENCE = 20
CONTRASTIVE_MARGIN = 0.5

model = SemiSupervisedGAT(
    code_embedding_dim=CODE_EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    provider_embedding_dim=PROVIDER_EMBEDDING_DIM,
    num_specialties=num_specialties,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model architecture:")
print(f"  Code embedding dim: {CODE_EMBEDDING_DIM}")
print(f"  Attention heads: {NUM_HEADS}")
print(f"  Provider embedding dim: {PROVIDER_EMBEDDING_DIM}")
print(f"  Specialties: {num_specialties}")
print(f"  Dropout: {DROPOUT}")
print(f"\nParameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
print(f"\nUncertainty parameters:")
print(f"  log_var_cls: {model.log_var_cls.item():.4f}")
print(f"  log_var_rec: {model.log_var_rec.item():.4f}")
print(f"  log_var_con: {model.log_var_con.item():.4f}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion_cls = nn.CrossEntropyLoss()

print(f"\nHyperparameters:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Gradient clip: {GRAD_CLIP_NORM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {MAX_EPOCHS}")
print(f"  Patience: {PATIENCE}")
print(f"  Contrastive margin: {CONTRASTIVE_MARGIN}")

print("\n" + "="*80)
print("STEP 6: TRAINING LOOP")
print("="*80)

best_val_loss = float('inf')
best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
history = []

print("\nStarting training...\n")
print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10} {'Pairs/Batch':<12} {'Time':<8} {'Status':<15}")
print("-" * 85)

start_time = time.time()

for epoch in range(MAX_EPOCHS):
    epoch_start = time.time()
    model.train()
    
    epoch_loss_cls = 0
    epoch_loss_rec = 0
    epoch_loss_con = 0
    epoch_loss_total = 0
    epoch_pairs = 0
    correct = 0
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        provider_ids, code_embeddings, attention_masks, labels = batch_data
        
        code_embeddings = code_embeddings.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits, embeddings, reconstructed = model(code_embeddings, attention_masks)
        
        labeled_mask = torch.tensor([pid in labels_dict_indexed for pid in provider_ids], 
                                     device=device)
        
        if labeled_mask.any():
            labeled_indices = torch.where(labeled_mask)[0]
            logits_labeled = logits[labeled_indices]
            labels_batch = torch.tensor([labels_dict_indexed[provider_ids[i.item()]] 
                                        for i in labeled_indices], device=device)
            loss_cls = criterion_cls(logits_labeled, labels_batch)
            
            preds = logits_labeled.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total_samples += len(labels_batch)
        else:
            loss_cls = torch.tensor(0.0, device=device)
        
        loss_rec = reconstruction_loss(code_embeddings, reconstructed, attention_masks)
        
        loss_con, n_pairs = contrastive_loss(
            embeddings, provider_ids, similarity_matrix, pin_to_idx, CONTRASTIVE_MARGIN
        )
        
        precision_cls = torch.exp(-model.log_var_cls)
        precision_rec = torch.exp(-model.log_var_rec)
        precision_con = torch.exp(-model.log_var_con)
        
        total_loss = (precision_cls * loss_cls + model.log_var_cls +
                     precision_rec * loss_rec + model.log_var_rec +
                     precision_con * loss_con + model.log_var_con)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        
        epoch_loss_cls += loss_cls.item()
        epoch_loss_rec += loss_rec.item()
        epoch_loss_con += loss_con.item()
        epoch_loss_total += total_loss.item()
        epoch_pairs += n_pairs
    
    epoch_loss_cls /= len(train_loader)
    epoch_loss_rec /= len(train_loader)
    epoch_loss_con /= len(train_loader)
    epoch_loss_total /= len(train_loader)
    avg_pairs = epoch_pairs / len(train_loader)
    train_acc = 100 * correct / total_samples if total_samples > 0 else 0
    
    model.eval()
    val_loss_cls = 0
    val_loss_rec = 0
    val_loss_con = 0
    val_loss_total = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            provider_ids, code_embeddings, attention_masks, labels = batch_data
            
            code_embeddings = code_embeddings.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            logits, embeddings, reconstructed = model(code_embeddings, attention_masks)
            
            labels_batch = torch.tensor([labels_dict_indexed[pid] for pid in provider_ids], 
                                       device=device)
            loss_cls = criterion_cls(logits, labels_batch)
            loss_rec = reconstruction_loss(code_embeddings, reconstructed, attention_masks)
            loss_con, _ = contrastive_loss(embeddings, provider_ids, similarity_matrix, 
                                          pin_to_idx, CONTRASTIVE_MARGIN)
            
            precision_cls = torch.exp(-model.log_var_cls)
            precision_rec = torch.exp(-model.log_var_rec)
            precision_con = torch.exp(-model.log_var_con)
            
            total_loss = (precision_cls * loss_cls + model.log_var_cls +
                         precision_rec * loss_rec + model.log_var_rec +
                         precision_con * loss_con + model.log_var_con)
            
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels_batch).sum().item()
            val_total += len(labels_batch)
            
            val_loss_cls += loss_cls.item()
            val_loss_rec += loss_rec.item()
            val_loss_con += loss_con.item()
            val_loss_total += total_loss.item()
    
    val_loss_cls /= len(val_loader)
    val_loss_rec /= len(val_loader)
    val_loss_con /= len(val_loader)
    val_loss_total /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    epoch_time = time.time() - epoch_start
    
    status = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss_total
        best_epoch = epoch
        patience_counter = 0
        status = "✓ Best"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss_total,
            'val_acc': val_acc,
        }, 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            status = "Early stop"
        else:
            status = f"Patience {patience_counter}/{PATIENCE}"
    
    history.append({
        'epoch': epoch,
        'train_loss_total': epoch_loss_total,
        'train_loss_cls': epoch_loss_cls,
        'train_loss_rec': epoch_loss_rec,
        'train_loss_con': epoch_loss_con,
        'train_acc': train_acc,
        'val_loss_total': val_loss_total,
        'val_loss_cls': val_loss_cls,
        'val_loss_rec': val_loss_rec,
        'val_loss_con': val_loss_con,
        'val_acc': val_acc,
        'avg_pairs': avg_pairs,
        'log_var_cls': model.log_var_cls.item(),
        'log_var_rec': model.log_var_rec.item(),
        'log_var_con': model.log_var_con.item(),
    })
    
    print(f"{epoch+1:<6} {epoch_loss_total:<12.4f} {val_loss_total:<12.4f} "
          f"{val_acc:<10.2f} {avg_pairs:<12.1f} {epoch_time:<8.1f} {status:<15}")
    
    if (epoch + 1) % 10 == 0:
        print(f"  └─ Uncertainty: cls={model.log_var_cls.item():.3f}, "
              f"rec={model.log_var_rec.item():.3f}, con={model.log_var_con.item():.3f}")
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

total_time = time.time() - start_time

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Best epoch: {best_epoch+1}")
print(f"Best val accuracy: {best_val_acc:.2f}%")
print(f"Best val loss: {best_val_loss:.4f}")

print("\n" + "="*80)
print("STEP 7: TEST EVALUATION")
print("="*80)

checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
test_correct = 0
test_total = 0
test_loss_total = 0

with torch.no_grad():
    for batch_data in test_loader:
        provider_ids, code_embeddings, attention_masks, labels = batch_data
        
        code_embeddings = code_embeddings.to(device)
        attention_masks = attention_masks.to(device)
        
        logits, embeddings, reconstructed = model(code_embeddings, attention_masks)
        
        labels_batch = torch.tensor([labels_dict_indexed[pid] for pid in provider_ids], 
                                   device=device)
        
        loss_cls = criterion_cls(logits, labels_batch)
        loss_rec = reconstruction_loss(code_embeddings, reconstructed, attention_masks)
        loss_con, _ = contrastive_loss(embeddings, provider_ids, similarity_matrix, 
                                      pin_to_idx, CONTRASTIVE_MARGIN)
        
        precision_cls = torch.exp(-model.log_var_cls)
        precision_rec = torch.exp(-model.log_var_rec)
        precision_con = torch.exp(-model.log_var_con)
        
        total_loss = (precision_cls * loss_cls + model.log_var_cls +
                     precision_rec * loss_rec + model.log_var_rec +
                     precision_con * loss_con + model.log_var_con)
        
        preds = logits.argmax(dim=1)
        test_correct += (preds == labels_batch).sum().item()
        test_total += len(labels_batch)
        test_loss_total += total_loss.item()

test_acc = 100 * test_correct / test_total
test_loss_total /= len(test_loader)

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Loss: {test_loss_total:.4f}")

print("\n" + "="*80)
print("STEP 8: EXTRACT PROVIDER EMBEDDINGS")
print("="*80)

print(f"Extracting embeddings for ALL {len(aligned_pins):,} providers...")
print("  This includes both labeled and unlabeled providers")

all_provider_embeddings = {}

model.eval()
with torch.no_grad():
    for idx, provider_id in enumerate(aligned_pins):
        code_embs = provider_code_embeddings[provider_id].unsqueeze(0).to(device)
        num_codes = code_embs.shape[1]
        attention_mask = torch.ones(1, num_codes).to(device)
        
        _, provider_emb, _ = model(code_embs, attention_mask)
        all_provider_embeddings[provider_id] = provider_emb.squeeze(0).cpu()
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx+1:,} / {len(aligned_pins):,} providers...")

print(f"\nExtracted embeddings for {len(all_provider_embeddings):,} providers")

print("\n" + "="*80)
print("STEP 9: CREATE EMBEDDINGS DATAFRAME")
print("="*80)

all_provider_ids_list = list(all_provider_embeddings.keys())
embeddings_array_list = []

for provider_id in all_provider_ids_list:
    embeddings_array_list.append(all_provider_embeddings[provider_id].numpy())

embeddings_matrix = np.vstack(embeddings_array_list)

embeddings_df = pd.DataFrame(
    embeddings_matrix, 
    columns=[f'emb_{i}' for i in range(PROVIDER_EMBEDDING_DIM)]
)
embeddings_df.insert(0, 'provider_id', all_provider_ids_list)
embeddings_df['provider_id'] = embeddings_df['provider_id'].astype(int)

print(f"Embeddings DataFrame shape: {embeddings_df.shape}")
print(f"  Total providers: {len(embeddings_df):,}")
print(f"  Embedding dimensions: {PROVIDER_EMBEDDING_DIM}")
print(f"  Columns: provider_id, emb_0, emb_1, ..., emb_{PROVIDER_EMBEDDING_DIM-1}")

print("\n" + "="*80)
print("STEP 10: SAVE RESULTS")
print("="*80)

with open('data74.pkl', 'wb') as f:
    pickle.dump(history, f)
print("✓ training_history.pkl")

with open('data55.pkl', 'wb') as f:
    pickle.dump(all_provider_embeddings, f)
print("✓ provider_embeddings_final.pkl (dict format)")

embeddings_df.to_csv('data53.csv', index=False)
print("✓ provider_embeddings_df.csv")

embeddings_df.to_pickle('data54.pkl')
print("✓ provider_embeddings_df.pkl")

results = {
    'best_epoch': best_epoch,
    'best_val_acc': best_val_acc,
    'best_val_loss': best_val_loss,
    'test_acc': test_acc,
    'test_loss': test_loss_total,
    'total_time': total_time,
    'num_providers': len(all_provider_embeddings),
    'num_labeled': len(labeled_provider_ids),
    'num_unlabeled': len(unlabeled_provider_ids),
    'hyperparameters': {
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'contrastive_margin': CONTRASTIVE_MARGIN,
        'provider_embedding_dim': PROVIDER_EMBEDDING_DIM,
    },
    'final_uncertainty': {
        'log_var_cls': model.log_var_cls.item(),
        'log_var_rec': model.log_var_rec.item(),
        'log_var_con': model.log_var_con.item(),
        'precision_cls': torch.exp(-model.log_var_cls).item(),
        'precision_rec': torch.exp(-model.log_var_rec).item(),
        'precision_con': torch.exp(-model.log_var_con).item(),
    }
}

with open('data75.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✓ training_results.pkl")

print("\nFinal uncertainty parameters:")
print(f"  log_var_cls: {model.log_var_cls.item():.4f} (precision: {torch.exp(-model.log_var_cls).item():.4f})")
print(f"  log_var_rec: {model.log_var_rec.item():.4f} (precision: {torch.exp(-model.log_var_rec).item():.4f})")
print(f"  log_var_con: {model.log_var_con.item():.4f} (precision: {torch.exp(-model.log_var_con).item():.4f})")

print("\nEmbeddings summary:")
print(f"  Total providers with embeddings: {len(embeddings_df):,}")
print(f"  Labeled providers: {len(labeled_provider_ids):,}")
print(f"  Unlabeled providers: {len(unlabeled_provider_ids):,}")
print(f"  Embedding dimension: {PROVIDER_EMBEDDING_DIM}")

print("\n" + "="*80)
print("NOTEBOOK 3 COMPLETE!")
print("="*80)
print("\nOutputs:")
print("  1. best_model.pt - Best model checkpoint")
print("  2. training_history.pkl - Loss curves and metrics per epoch")
print("  3. training_results.pkl - Final results and hyperparameters")
print("  4. provider_embeddings_final.pkl - All embeddings (dict format)")
print("  5. provider_embeddings_df.csv - All embeddings (DataFrame CSV)")
print("  6. provider_embeddings_df.pkl - All embeddings (DataFrame pickle)")

print("\nMethod: Uncertainty Weighting (Kendall et al., 2018)")
print("  - Battle-tested: 6000+ citations")
print("  - Used in production by Google and others")
print("  - Automatic task balancing without manual tuning")
print("  - Reproducible results with fixed seed")

print(f"\nSummary:")
print(f"  Training time: {total_time/60:.1f} minutes")
print(f"  Best validation accuracy: {best_val_acc:.2f}%")
print(f"  Test accuracy: {test_acc:.2f}%")
print(f"  Converged at epoch: {best_epoch+1}/{MAX_EPOCHS}")
print(f"  Total providers with embeddings: {len(embeddings_df):,}")
print(f"  Embedding dimension: {PROVIDER_EMBEDDING_DIM}")

print("\nEmbeddings DataFrame preview:")
print(embeddings_df.head())
