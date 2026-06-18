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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Assuming you have loaded:
# proc_tensor, diag_tensor, demo_tensor, plc_tensor, cost_ctg_tensor, PIN_smry_tensor
# all_pins, pin_to_label

# Get dimensions
proc_dim = proc_tensor.shape[1]      # 14978
diag_dim = diag_tensor.shape[1]      # 9116
demo_dim = demo_tensor.shape[1]      # 5
plc_dim = plc_tensor.shape[1]        # 4
cost_dim = cost_ctg_tensor.shape[1]  # 11
pin_dim = PIN_smry_tensor.shape[1]   # 2

n_samples = proc_tensor.shape[0]     # 25347

print(f"Dataset dimensions:")
print(f"  Procedures: {proc_dim} cols")
print(f"  Diagnoses: {diag_dim} cols")
print(f"  Demographics: {demo_dim} cols")
print(f"  Place of Service: {plc_dim} cols")
print(f"  Cost Category: {cost_dim} cols")
print(f"  PIN Summary: {pin_dim} cols")
print(f"  Total samples: {n_samples}")

# Latent dimensions per tower
proc_latent_dim = 128
diag_latent_dim = 96
demo_latent_dim = 8
plc_latent_dim = 4
cost_latent_dim = 8
pin_latent_dim = 4

total_latent_dim = proc_latent_dim + diag_latent_dim + demo_latent_dim + plc_latent_dim + cost_latent_dim + pin_latent_dim

print(f"\nLatent dimensions:")
print(f"  Tower 1 (Procedures): {proc_latent_dim}")
print(f"  Tower 2 (Diagnoses): {diag_latent_dim}")
print(f"  Tower 3 (Demographics): {demo_latent_dim}")
print(f"  Tower 4 (Place): {plc_latent_dim}")
print(f"  Tower 5 (Cost): {cost_latent_dim}")
print(f"  Tower 6 (PIN Summary): {pin_latent_dim}")
print(f"  Total: {total_latent_dim}")

# ============================================================================
# MODEL ARCHITECTURE - MULTI-TOWER VAE
# ============================================================================

class TowerVAE(nn.Module):
    """Single VAE tower for one modality"""
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layers
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class MultiTowerVAE(nn.Module):
    """Multi-tower VAE with independent towers"""
    def __init__(self, proc_dim, diag_dim, demo_dim, plc_dim, cost_dim, pin_dim,
                 proc_latent, diag_latent, demo_latent, plc_latent, cost_latent, pin_latent):
        super().__init__()
        
        # Tower 1: Procedures (large, sparse)
        self.proc_tower = TowerVAE(proc_dim, proc_latent, hidden_dims=[1024, 512, 256])
        
        # Tower 2: Diagnoses (large, sparse)
        self.diag_tower = TowerVAE(diag_dim, diag_latent, hidden_dims=[1024, 512, 256])
        
        # Tower 3: Demographics (small, dense)
        self.demo_tower = TowerVAE(demo_dim, demo_latent, hidden_dims=[16, 8])
        
        # Tower 4: Place of Service (small, dense)
        self.plc_tower = TowerVAE(plc_dim, plc_latent, hidden_dims=[8, 4])
        
        # Tower 5: Cost Category (small, dense)
        self.cost_tower = TowerVAE(cost_dim, cost_latent, hidden_dims=[16, 8])
        
        # Tower 6: PIN Summary (tiny, dense)
        self.pin_tower = TowerVAE(pin_dim, pin_latent, hidden_dims=[8, 4])
    
    def forward(self, proc, diag, demo, plc, cost, pin):
        proc_recon, proc_mu, proc_logvar = self.proc_tower(proc)
        diag_recon, diag_mu, diag_logvar = self.diag_tower(diag)
        demo_recon, demo_mu, demo_logvar = self.demo_tower(demo)
        plc_recon, plc_mu, plc_logvar = self.plc_tower(plc)
        cost_recon, cost_mu, cost_logvar = self.cost_tower(cost)
        pin_recon, pin_mu, pin_logvar = self.pin_tower(pin)
        
        return {
            'proc': (proc_recon, proc_mu, proc_logvar),
            'diag': (diag_recon, diag_mu, diag_logvar),
            'demo': (demo_recon, demo_mu, demo_logvar),
            'plc': (plc_recon, plc_mu, plc_logvar),
            'cost': (cost_recon, cost_mu, cost_logvar),
            'pin': (pin_recon, pin_mu, pin_logvar)
        }
    
    def get_embeddings(self, proc, diag, demo, plc, cost, pin):
        """Extract and normalize embeddings from all towers"""
        proc_mu, _ = self.proc_tower.encode(proc)
        diag_mu, _ = self.diag_tower.encode(diag)
        demo_mu, _ = self.demo_tower.encode(demo)
        plc_mu, _ = self.plc_tower.encode(plc)
        cost_mu, _ = self.cost_tower.encode(cost)
        pin_mu, _ = self.pin_tower.encode(pin)
        
        # Normalize each tower independently
        proc_mu_norm = F.normalize(proc_mu, p=2, dim=1)
        diag_mu_norm = F.normalize(diag_mu, p=2, dim=1)
        demo_mu_norm = F.normalize(demo_mu, p=2, dim=1)
        plc_mu_norm = F.normalize(plc_mu, p=2, dim=1)
        cost_mu_norm = F.normalize(cost_mu, p=2, dim=1)
        pin_mu_norm = F.normalize(pin_mu, p=2, dim=1)
        
        # Concatenate all normalized embeddings
        combined = torch.cat([
            proc_mu_norm, diag_mu_norm, demo_mu_norm, 
            plc_mu_norm, cost_mu_norm, pin_mu_norm
        ], dim=1)
        
        return combined, {
            'proc': proc_mu_norm,
            'diag': diag_mu_norm,
            'demo': demo_mu_norm,
            'plc': plc_mu_norm,
            'cost': cost_mu_norm,
            'pin': pin_mu_norm
        }


# Initialize model
model = MultiTowerVAE(
    proc_dim, diag_dim, demo_dim, plc_dim, cost_dim, pin_dim,
    proc_latent_dim, diag_latent_dim, demo_latent_dim, 
    plc_latent_dim, cost_latent_dim, pin_latent_dim
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def vae_loss(recon, x, mu, logvar, kl_weight=0.001):
    """VAE loss = Reconstruction + KL divergence"""
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss

def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet loss for supervised learning"""
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def compute_total_loss(outputs, inputs, use_triplet=False, 
                       triplet_embeddings=None, kl_weight=0.001):
    """Compute combined loss for all towers"""
    total_loss = 0
    losses_dict = {}
    
    # VAE losses for each tower
    for key in ['proc', 'diag', 'demo', 'plc', 'cost', 'pin']:
        recon, mu, logvar = outputs[key]
        x = inputs[key]
        loss = vae_loss(recon, x, mu, logvar, kl_weight)
        total_loss += loss
        losses_dict[f'{key}_vae'] = loss.item()
    
    # Optional triplet loss
    if use_triplet and triplet_embeddings is not None:
        anchor_emb, positive_emb, negative_emb = triplet_embeddings
        trip_loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
        total_loss += 0.1 * trip_loss
        losses_dict['triplet'] = trip_loss.item()
    
    return total_loss, losses_dict

# ============================================================================
# TRAINING SETUP
# ============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

batch_size = 64
epochs = 30
n_batches = (n_samples + batch_size - 1) // batch_size

print(f"\nTraining setup:")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  Batches per epoch: {n_batches}")
print(f"  Labeled samples: {len(pin_to_label)}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\nStarting training...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_losses = {
        'proc_vae': 0, 'diag_vae': 0, 'demo_vae': 0,
        'plc_vae': 0, 'cost_vae': 0, 'pin_vae': 0, 'triplet': 0
    }
    triplet_count = 0
    
    # Shuffle indices
    indices = torch.randperm(n_samples)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Get batch data
        proc_batch = proc_tensor[batch_indices].to(device)
        diag_batch = diag_tensor[batch_indices].to(device)
        demo_batch = demo_tensor[batch_indices].to(device)
        plc_batch = plc_tensor[batch_indices].to(device)
        cost_batch = cost_ctg_tensor[batch_indices].to(device)
        pin_batch = PIN_smry_tensor[batch_indices].to(device)
        batch_pins = [all_pins[i] for i in batch_indices]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        
        inputs = {
            'proc': proc_batch,
            'diag': diag_batch,
            'demo': demo_batch,
            'plc': plc_batch,
            'cost': cost_batch,
            'pin': pin_batch
        }
        
        # Triplet loss setup (if labeled samples in batch)
        use_triplet = False
        triplet_embeddings = None
        
        labeled_in_batch = []
        labels_in_batch = []
        for i, pin in enumerate(batch_pins):
            if pin in pin_to_label:
                labeled_in_batch.append(i)
                labels_in_batch.append(pin_to_label[pin])
        
        if len(labeled_in_batch) >= 2:
            # Extract embeddings for triplet loss
            combined_emb, _ = model.get_embeddings(
                proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch
            )
            
            # Create triplets
            triplets_anchor = []
            triplets_positive = []
            triplets_negative = []
            
            for i in range(len(labeled_in_batch)):
                anchor_idx = labeled_in_batch[i]
                anchor_label = labels_in_batch[i]
                
                positive_candidates = [labeled_in_batch[j] for j in range(len(labeled_in_batch))
                                     if labels_in_batch[j] == anchor_label and j != i]
                
                if positive_candidates:
                    positive_idx = random.choice(positive_candidates)
                    
                    negative_candidates = [labeled_in_batch[j] for j in range(len(labeled_in_batch))
                                         if labels_in_batch[j] != anchor_label]
                    
                    if negative_candidates:
                        negative_idx = random.choice(negative_candidates)
                        
                        triplets_anchor.append(combined_emb[anchor_idx:anchor_idx+1])
                        triplets_positive.append(combined_emb[positive_idx:positive_idx+1])
                        triplets_negative.append(combined_emb[negative_idx:negative_idx+1])
            
            if triplets_anchor:
                use_triplet = True
                triplet_embeddings = (
                    torch.cat(triplets_anchor),
                    torch.cat(triplets_positive),
                    torch.cat(triplets_negative)
                )
                triplet_count += 1
        
        # Compute loss
        loss, losses_dict = compute_total_loss(
            outputs, inputs, use_triplet, triplet_embeddings, kl_weight=0.001
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        epoch_loss += loss.item()
        for key, val in losses_dict.items():
            epoch_losses[key] += val
    
    # Average losses
    avg_loss = epoch_loss / n_batches
    for key in epoch_losses:
        epoch_losses[key] /= max(n_batches if key != 'triplet' else triplet_count, 1)
    
    scheduler.step(avg_loss)
    
    if epoch % 5 == 0:
        print(f'\nEpoch {epoch}/{epochs}:')
        print(f'  Total Loss: {avg_loss:.2f}')
        print(f'  Proc: {epoch_losses["proc_vae"]:.2f}, Diag: {epoch_losses["diag_vae"]:.2f}, '
              f'Demo: {epoch_losses["demo_vae"]:.2f}')
        print(f'  Place: {epoch_losses["plc_vae"]:.2f}, Cost: {epoch_losses["cost_vae"]:.2f}, '
              f'PIN: {epoch_losses["pin_vae"]:.2f}')
        print(f'  Triplet: {epoch_losses["triplet"]:.4f} ({triplet_count} batches)')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

print("\n" + "="*80)
print("Training completed!")
print("="*80)

# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================

print("\nExtracting final embeddings...")

model.eval()
all_combined_embeddings = []
all_tower_embeddings = {
    'proc': [], 'diag': [], 'demo': [], 'plc': [], 'cost': [], 'pin': []
}

batch_size_inference = 128
n_batches_inference = (n_samples + batch_size_inference - 1) // batch_size_inference

print(f"Processing {n_samples} samples in {n_batches_inference} batches...")

with torch.no_grad():
    for batch_idx in range(n_batches_inference):
        start_idx = batch_idx * batch_size_inference
        end_idx = min(start_idx + batch_size_inference, n_samples)
        
        proc_batch = proc_tensor[start_idx:end_idx].to(device)
        diag_batch = diag_tensor[start_idx:end_idx].to(device)
        demo_batch = demo_tensor[start_idx:end_idx].to(device)
        plc_batch = plc_tensor[start_idx:end_idx].to(device)
        cost_batch = cost_ctg_tensor[start_idx:end_idx].to(device)
        pin_batch = PIN_smry_tensor[start_idx:end_idx].to(device)
        
        combined_emb, tower_embs = model.get_embeddings(
            proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch
        )
        
        all_combined_embeddings.append(combined_emb.cpu().numpy())
        for key in tower_embs:
            all_tower_embeddings[key].append(tower_embs[key].cpu().numpy())
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{n_batches_inference} batches")

# Combine all embeddings
final_embeddings = np.vstack(all_combined_embeddings)
tower_embeddings = {key: np.vstack(val) for key, val in all_tower_embeddings.items()}

print(f"\nFinal embeddings shape: {final_embeddings.shape}")
print(f"Expected: ({n_samples}, {total_latent_dim})")

# Create embeddings dataframe
embedding_data = {'PIN': all_pins}

# Add tower embeddings with proper naming
for i in range(proc_latent_dim):
    embedding_data[f'tower1_proc_emb_{i}'] = tower_embeddings['proc'][:, i]

for i in range(diag_latent_dim):
    embedding_data[f'tower2_diag_emb_{i}'] = tower_embeddings['diag'][:, i]

for i in range(demo_latent_dim):
    embedding_data[f'tower3_demo_emb_{i}'] = tower_embeddings['demo'][:, i]

for i in range(plc_latent_dim):
    embedding_data[f'tower4_plc_emb_{i}'] = tower_embeddings['plc'][:, i]

for i in range(cost_latent_dim):
    embedding_data[f'tower5_cost_emb_{i}'] = tower_embeddings['cost'][:, i]

for i in range(pin_latent_dim):
    embedding_data[f'tower6_pin_emb_{i}'] = tower_embeddings['pin'][:, i]

embeddings_df = pd.DataFrame(embedding_data)

print(f"\nEmbeddings dataframe: {embeddings_df.shape}")
print(f"\nColumn breakdown:")
print(f"  Tower 1 (Procedures): tower1_proc_emb_0 to tower1_proc_emb_{proc_latent_dim-1}")
print(f"  Tower 2 (Diagnoses): tower2_diag_emb_0 to tower2_diag_emb_{diag_latent_dim-1}")
print(f"  Tower 3 (Demographics): tower3_demo_emb_0 to tower3_demo_emb_{demo_latent_dim-1}")
print(f"  Tower 4 (Place): tower4_plc_emb_0 to tower4_plc_emb_{plc_latent_dim-1}")
print(f"  Tower 5 (Cost): tower5_cost_emb_0 to tower5_cost_emb_{cost_latent_dim-1}")
print(f"  Tower 6 (PIN Summary): tower6_pin_emb_0 to tower6_pin_emb_{pin_latent_dim-1}")

print("\nSample embeddings:")
print(embeddings_df.head())

# Save embeddings
embeddings_df.to_parquet('data56.parquet', index=False)
torch.save(model.state_dict(), 'multi_tower_vae.pth')

print("\nSaved:")
print("  - provider_embeddings_multi_tower_vae.parquet")
print("  - multi_tower_vae.pth")

print("\n" + "="*80)
print("COMPLETE - All embeddings are normalized and ready for cosine similarity!")
print("="*80)
