# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Assuming you have loaded:
# proc_tensor, diag_tensor, demo_tensor, plc_tensor, cost_ctg_tensor, PIN_smry_tensor
# pin_list, pin_to_label

# Get dimensions
proc_dim = proc_tensor.shape[1]
diag_dim = diag_tensor.shape[1]
demo_dim = demo_tensor.shape[1]
plc_dim = plc_tensor.shape[1]
cost_dim = cost_ctg_tensor.shape[1]
pin_dim = PIN_smry_tensor.shape[1]
n_samples = proc_tensor.shape[0]

print(f"Dataset dimensions:")
print(f"  Procedures: {proc_dim}")
print(f"  Diagnoses: {diag_dim}")
print(f"  Demographics: {demo_dim}")
print(f"  Place: {plc_dim}")
print(f"  Cost: {cost_dim}")
print(f"  PIN: {pin_dim}")
print(f"  Total samples: {n_samples}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
PROC_LATENT_DIM = 128
DIAG_LATENT_DIM = 96
LEARNING_RATE = 0.0005
KL_WEIGHT_START = 0.01
KL_WEIGHT_END = 0.05
BATCH_SIZE = 128
EPOCHS = 30
CLASSIFICATION_WEIGHT = 2.0
PROC_RECON_WEIGHT = 1.0
DIAG_RECON_WEIGHT = 1.0

total_latent_dim = PROC_LATENT_DIM + DIAG_LATENT_DIM + demo_dim + plc_dim + cost_dim + pin_dim

print(f"\nLatent dimensions:")
print(f"  Proc VAE: {PROC_LATENT_DIM}")
print(f"  Diag VAE: {DIAG_LATENT_DIM}")
print(f"  Demographics: {demo_dim}")
print(f"  Place: {plc_dim}")
print(f"  Cost: {cost_dim}")
print(f"  PIN: {pin_dim}")
print(f"  Total: {total_latent_dim}")

# ============================================================================
# ENCODE LABELS
# ============================================================================
print("\nEncoding labels...")
label_encoder = LabelEncoder()
all_labels = list(pin_to_label.values())
label_encoder.fit(all_labels)
n_classes = len(label_encoder.classes_)

pin_to_label_numeric = {pin: label_encoder.transform([label])[0] 
                        for pin, label in pin_to_label.items()}

print(f"Number of classes: {n_classes}")
for i, label_name in enumerate(label_encoder.classes_):
    count = sum(1 for l in pin_to_label.values() if l == label_name)
    print(f"  {i}: {label_name} ({count} samples)")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
labeled_indices = [i for i, pin in enumerate(pin_list) if pin in pin_to_label]
labeled_labels = [pin_to_label[pin_list[i]] for i in labeled_indices]

label_to_indices = defaultdict(list)
for idx, label in zip(labeled_indices, labeled_labels):
    label_to_indices[label].append(idx)

train_indices = []
val_indices = []

for label, indices in label_to_indices.items():
    random.shuffle(indices)
    split_point = int(0.8 * len(indices))
    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

print(f"\nTrain samples: {len(train_indices)}")
print(f"Val samples: {len(val_indices)}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class HierarchicalVAE(nn.Module):
    """Hierarchical VAE with multi-stage compression"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # Encoder: input → 2048 → 512 → 128 → latent
        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.enc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.enc3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
        
        # Decoder: latent → 128 → 512 → 2048 → output
        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.dec2 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.dec3 = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.dec4 = nn.Linear(2048, input_dim)
    
    def encode(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        mu = self.mu_layer(h3)
        logvar = self.logvar_layer(h3)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h1 = self.dec1(z)
        h2 = self.dec2(h1)
        h3 = self.dec3(h2)
        return self.dec4(h3)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class HybridModelWithClassifier(nn.Module):
    """Hybrid model with classification head"""
    def __init__(self, proc_dim, diag_dim, proc_latent, diag_latent, 
                 demo_dim, plc_dim, cost_dim, pin_dim, n_classes):
        super().__init__()
        
        self.proc_tower = HierarchicalVAE(proc_dim, proc_latent)
        self.diag_tower = HierarchicalVAE(diag_dim, diag_latent)
        
        total_dim = proc_latent + diag_latent + demo_dim + plc_dim + cost_dim + pin_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, proc, diag, demo, plc, cost, pin):
        proc_recon, proc_mu, proc_logvar = self.proc_tower(proc)
        diag_recon, diag_mu, diag_logvar = self.diag_tower(diag)
        return {
            'proc_recon': proc_recon,
            'proc_mu': proc_mu,
            'proc_logvar': proc_logvar,
            'diag_recon': diag_recon,
            'diag_mu': diag_mu,
            'diag_logvar': diag_logvar
        }
    
    def get_embeddings(self, proc, diag, demo, plc, cost, pin):
        """Get embeddings (raw for classifier, normalized for similarity)"""
        proc_mu, _ = self.proc_tower.encode(proc)
        diag_mu, _ = self.diag_tower.encode(diag)
        
        # Raw concatenation for classifier
        raw_emb = torch.cat([proc_mu, diag_mu, demo, plc, cost, pin], dim=1)
        
        # Normalized for cosine similarity
        proc_norm = F.normalize(proc_mu, p=2, dim=1)
        diag_norm = F.normalize(diag_mu, p=2, dim=1)
        demo_norm = F.normalize(demo, p=2, dim=1)
        plc_norm = F.normalize(plc, p=2, dim=1)
        cost_norm = F.normalize(cost, p=2, dim=1)
        pin_norm = F.normalize(pin, p=2, dim=1)
        norm_emb = torch.cat([proc_norm, diag_norm, demo_norm, plc_norm, cost_norm, pin_norm], dim=1)
        
        return raw_emb, norm_emb
    
    def classify(self, raw_emb):
        return self.classifier(raw_emb)


# Initialize model
model = HybridModelWithClassifier(
    proc_dim, diag_dim, PROC_LATENT_DIM, DIAG_LATENT_DIM,
    demo_dim, plc_dim, cost_dim, pin_dim, n_classes
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
print(f"Parameters per labeled sample: {total_params / len(labeled_indices):.1f}")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def vae_loss_mse_log(recon, x, mu, logvar, kl_weight):
    """MSE on log-space with masking"""
    mask = (x > 0).float()
    mse = ((recon - x) ** 2) * mask
    recon_loss = mse.sum() / (mask.sum() + 1e-8)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


def compute_loss(outputs, proc, diag, labels, logits, kl_weight):
    """Compute total loss"""
    # VAE losses
    proc_recon_loss, proc_kl = vae_loss_mse_log(
        outputs['proc_recon'], proc, outputs['proc_mu'], outputs['proc_logvar'], kl_weight
    )
    diag_recon_loss, diag_kl = vae_loss_mse_log(
        outputs['diag_recon'], diag, outputs['diag_mu'], outputs['diag_logvar'], kl_weight
    )
    
    # Normalize reconstruction losses
    proc_nnz = (proc > 0).sum()
    diag_nnz = (diag > 0).sum()
    total_nnz = proc_nnz + diag_nnz
    
    proc_recon_norm = proc_recon_loss * (proc_nnz / total_nnz) * PROC_RECON_WEIGHT
    diag_recon_norm = diag_recon_loss * (diag_nnz / total_nnz) * DIAG_RECON_WEIGHT
    
    vae_loss = proc_recon_norm + diag_recon_norm + kl_weight * (proc_kl + diag_kl)
    
    # Classification loss
    if labels is not None and logits is not None:
        class_loss = F.cross_entropy(logits, labels)
        total_loss = vae_loss + CLASSIFICATION_WEIGHT * class_loss
    else:
        class_loss = torch.tensor(0.0)
        total_loss = vae_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'proc_recon': proc_recon_loss.item(),
        'diag_recon': diag_recon_loss.item(),
        'kl': (proc_kl + diag_kl).item(),
        'classification': class_loss.item()
    }


# ============================================================================
# CLASS-BALANCED BATCH SAMPLER
# ============================================================================

def create_balanced_batches(indices, pin_list, pin_to_label_numeric, batch_size):
    """Create batches with equal samples per class"""
    label_to_idx = defaultdict(list)
    for idx in indices:
        pin = pin_list[idx]
        if pin in pin_to_label_numeric:
            label = pin_to_label_numeric[pin]
            label_to_idx[label].append(idx)
    
    labels = list(label_to_idx.keys())
    n_classes = len(labels)
    samples_per_class = batch_size // n_classes
    
    for label in labels:
        random.shuffle(label_to_idx[label])
    
    min_size = min(len(label_to_idx[label]) for label in labels)
    max_batches = min_size // samples_per_class
    
    batches = []
    pointers = {label: 0 for label in labels}
    
    for _ in range(max_batches):
        batch = []
        for label in labels:
            idx_list = label_to_idx[label]
            start = pointers[label]
            end = start + samples_per_class
            
            if end > len(idx_list):
                batch.extend(idx_list[start:])
                needed = end - len(idx_list)
                batch.extend(idx_list[:needed])
                pointers[label] = needed
            else:
                batch.extend(idx_list[start:end])
                pointers[label] = end
        
        batches.append(batch)
    
    random.shuffle(batches)
    return batches


# ============================================================================
# VALIDATION
# ============================================================================

def validate(model, val_indices, kl_weight):
    """Validation"""
    model.eval()
    val_losses = []
    
    val_labeled = [idx for idx in val_indices if pin_list[idx] in pin_to_label_numeric]
    if len(val_labeled) == 0:
        return 0.0
    
    with torch.no_grad():
        for i in range(0, len(val_labeled), BATCH_SIZE):
            batch_idx = val_labeled[i:i+BATCH_SIZE]
            
            proc_batch = proc_tensor[batch_idx].to(device)
            diag_batch = diag_tensor[batch_idx].to(device)
            demo_batch = demo_tensor[batch_idx].to(device)
            plc_batch = plc_tensor[batch_idx].to(device)
            cost_batch = cost_ctg_tensor[batch_idx].to(device)
            pin_batch = PIN_smry_tensor[batch_idx].to(device)
            
            labels = torch.tensor(
                [pin_to_label_numeric[pin_list[idx]] for idx in batch_idx],
                dtype=torch.long, device=device
            )
            
            outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
            raw_emb, _ = model.get_embeddings(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
            logits = model.classify(raw_emb)
            
            loss, _ = compute_loss(outputs, proc_batch, diag_batch, labels, logits, kl_weight)
            val_losses.append(loss.item())
    
    model.train()
    return np.mean(val_losses)


# ============================================================================
# TRAINING
# ============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_loss = float('inf')
history = {'train': [], 'val': []}

print("\n" + "="*80)
print("TRAINING: HIERARCHICAL VAE + CLASSIFICATION HEAD")
print("="*80)

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = {'total': 0, 'proc_recon': 0, 'diag_recon': 0, 'kl': 0, 'classification': 0}
    
    kl_weight = KL_WEIGHT_START + (KL_WEIGHT_END - KL_WEIGHT_START) * (epoch / EPOCHS)
    balanced_batches = create_balanced_batches(train_indices, pin_list, pin_to_label_numeric, BATCH_SIZE)
    
    n_batches = 0
    for batch_idx in balanced_batches:
        proc_batch = proc_tensor[batch_idx].to(device)
        diag_batch = diag_tensor[batch_idx].to(device)
        demo_batch = demo_tensor[batch_idx].to(device)
        plc_batch = plc_tensor[batch_idx].to(device)
        cost_batch = cost_ctg_tensor[batch_idx].to(device)
        pin_batch = PIN_smry_tensor[batch_idx].to(device)
        
        labels = torch.tensor(
            [pin_to_label_numeric[pin_list[idx]] for idx in batch_idx],
            dtype=torch.long, device=device
        )
        
        outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        raw_emb, _ = model.get_embeddings(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        logits = model.classify(raw_emb)
        
        loss, losses_dict = compute_loss(outputs, proc_batch, diag_batch, labels, logits, kl_weight)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        for key in epoch_losses:
            epoch_losses[key] += losses_dict[key]
        n_batches += 1
    
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    val_loss = validate(model, val_indices, kl_weight)
    scheduler.step()
    
    history['train'].append(epoch_losses['total'])
    history['val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_classifier.pth')
        print(f'  New best model saved (val_loss: {val_loss:.4f})')
    
    gap = val_loss - epoch_losses['total']
    
    print(f'Epoch {epoch+1}/{EPOCHS}: '
          f'Train={epoch_losses["total"]:.4f}, Val={val_loss:.4f}, Gap={gap:.4f} | '
          f'Proc={epoch_losses["proc_recon"]:.4f}, Diag={epoch_losses["diag_recon"]:.4f}, '
          f'KL={epoch_losses["kl"]:.4f}, Class={epoch_losses["classification"]:.4f}')
    
    if gap > 1.5:
        print(f'  WARNING: Overfitting (gap > 1.5)')

print("\n" + "="*80)
print(f"Training complete. Best val loss: {best_val_loss:.4f}")
print("="*80)

# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================

print("\nExtracting embeddings...")
model.load_state_dict(torch.load('best_model_classifier.pth'))
model.eval()

all_embeddings = []
batch_size_infer = 256

with torch.no_grad():
    for i in range(0, n_samples, batch_size_infer):
        end_idx = min(i + batch_size_infer, n_samples)
        
        proc_batch = proc_tensor[i:end_idx].to(device)
        diag_batch = diag_tensor[i:end_idx].to(device)
        demo_batch = demo_tensor[i:end_idx].to(device)
        plc_batch = plc_tensor[i:end_idx].to(device)
        cost_batch = cost_ctg_tensor[i:end_idx].to(device)
        pin_batch = PIN_smry_tensor[i:end_idx].to(device)
        
        raw_emb, norm_emb = model.get_embeddings(
            proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch
        )
        all_embeddings.append(norm_emb.cpu().numpy())

final_embeddings = np.vstack(all_embeddings)

# Create dataframe
emb_df = pd.DataFrame({
    'PIN': pin_list
})

for i in range(final_embeddings.shape[1]):
    emb_df[f'emb_{i}'] = final_embeddings[:, i]

emb_df.to_parquet('data52.parquet', index=False)

print(f"\nSaved embeddings: {final_embeddings.shape}")
print("File: provider_embeddings_classifier.parquet")
print("\n" + "="*80)
print("COMPLETE")
print("="*80)
