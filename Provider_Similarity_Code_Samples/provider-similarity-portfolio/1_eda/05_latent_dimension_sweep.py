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
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

try:
    # Try loading from Phase 2 outputs
    final_embeddings = np.load('hospital_embeddings_480d.npy')
    all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()
    with open('data44.pkl', 'rb') as f:
        pin_to_label = pickle.load(f)
    
    print(f"✓ Loaded existing data:")
    print(f"  Embeddings shape: {final_embeddings.shape}")
    print(f"  Total hospitals: {len(all_pins)}")
    print(f"  Labeled hospitals: {len(pin_to_label)}")
except FileNotFoundError:
    print("\n⚠ Could not find data files!")
    print("Please ensure the following files exist in the current directory:")
    print("  - hospital_embeddings_480d.npy")
    print("  - all_pins.npy")
    print("  - pin_to_label.pkl")
    raise

# Create PIN to index mapping
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}
labeled_indices = [pin_to_idx[pin] for pin in pin_to_label.keys()]

# Convert to tensor
embeddings_tensor = torch.FloatTensor(final_embeddings).to(device)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

INPUT_DIM = 480
LATENT_DIMS = [32, 64, 128, 256]  # Different dimensions to test
HIDDEN_DIM = 256
LEARNING_RATE = 0.0005
EPOCHS = 20
BATCH_SIZE = 120
TEMPERATURE = 0.1

print(f"\nExperiment Configuration:")
print(f"  Input dimension: {INPUT_DIM}")
print(f"  Latent dimensions to test: {LATENT_DIMS}")
print(f"  Hidden dimension: {HIDDEN_DIM}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class CompressionNetwork(nn.Module):
    """Compress 480 dims to specified output dimension"""
    def __init__(self, input_dim=480, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    Supervised Contrastive Loss
    Pull same-class together, push different-class apart
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Create label mask (1 if same class, 0 otherwise)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    similarity_matrix = similarity_matrix / temperature
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Mask out self-similarity
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # Compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    
    # Loss
    loss = -mean_log_prob_pos
    loss = loss.mean()
    
    return loss


# ============================================================================
# LABEL PROCESSING
# ============================================================================

print("\n" + "="*80)
print("PROCESSING LABELS")
print("="*80)

unique_labels = sorted(set(pin_to_label.values()))
label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
n_classes = len(unique_labels)

print(f"Number of classes: {n_classes}")

pin_to_label_numeric = {pin: label_to_numeric[label] for pin, label in pin_to_label.items()}

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================

print("\n" + "="*80)
print("CREATING TRAIN/VAL SPLIT")
print("="*80)

label_to_indices = defaultdict(list)
for pin, label in pin_to_label.items():
    idx = pin_to_idx[pin]
    numeric_label = label_to_numeric[label]
    label_to_indices[numeric_label].append(idx)

train_indices = []
val_indices = []

for label, indices in label_to_indices.items():
    random.shuffle(indices)
    split_point = int(0.8 * len(indices))
    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

print(f"Train samples: {len(train_indices)}")
print(f"Val samples: {len(val_indices)}")

# ============================================================================
# BATCH SAMPLER
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


def validate(model, val_indices, pin_list, pin_to_label_numeric):
    """Compute validation loss"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for i in range(0, len(val_indices), BATCH_SIZE):
            batch_idx = val_indices[i:i+BATCH_SIZE]
            
            emb_batch = embeddings_tensor[batch_idx]
            labels_batch = torch.tensor(
                [pin_to_label_numeric[pin_list[idx]] for idx in batch_idx],
                dtype=torch.long, device=device
            )
            
            compressed = model(emb_batch)
            loss = supervised_contrastive_loss(compressed, labels_batch, TEMPERATURE)
            val_losses.append(loss.item())
    
    model.train()
    return np.mean(val_losses)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(output_dim, epochs=20):
    """Train a compression model with specified output dimension"""
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {INPUT_DIM}D → {output_dim}D")
    print(f"{'='*80}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize model
    model = CompressionNetwork(INPUT_DIM, HIDDEN_DIM, output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    history = {'train': [], 'val': [], 'epoch': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        # Create balanced batches
        balanced_batches = create_balanced_batches(
            train_indices, all_pins, pin_to_label_numeric, BATCH_SIZE
        )
        
        for batch_idx in balanced_batches:
            emb_batch = embeddings_tensor[batch_idx]
            labels_batch = torch.tensor(
                [pin_to_label_numeric[all_pins[idx]] for idx in batch_idx],
                dtype=torch.long, device=device
            )
            
            compressed = model(emb_batch)
            loss = supervised_contrastive_loss(compressed, labels_batch, TEMPERATURE)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Validation
        val_loss = validate(model, val_indices, all_pins, pin_to_label_numeric)
        scheduler.step()
        
        train_loss = np.mean(epoch_losses)
        history['epoch'].append(epoch + 1)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs}: "
                  f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")
    
    training_time = time.time() - start_time
    
    print(f"\n  ✓ Training complete in {training_time:.1f}s")
    print(f"  Final - Train: {history['train'][-1]:.4f}, Val: {history['val'][-1]:.4f}")
    
    return history, model


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

print("\n" + "="*80)
print("RUNNING EXPERIMENTS FOR ALL LATENT DIMENSIONS")
print("="*80)

results = {}

for latent_dim in LATENT_DIMS:
    history, model = train_model(latent_dim, EPOCHS)
    results[latent_dim] = history

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE")
print("="*80)

# ============================================================================
# CREATE RESULTS DATAFRAME
# ============================================================================

print("\n" + "="*80)
print("CREATING RESULTS TABLE")
print("="*80)

# Create comprehensive dataframe with all epochs and dimensions
all_data = []

for latent_dim in LATENT_DIMS:
    history = results[latent_dim]
    for i in range(len(history['epoch'])):
        all_data.append({
            'Latent_Dim': latent_dim,
            'Epoch': history['epoch'][i],
            'Train_Loss': history['train'][i],
            'Val_Loss': history['val'][i]
        })

results_df = pd.DataFrame(all_data)

# Save full results
results_df.to_csv('data32.csv', index=False)
print(f"\n✓ Saved full results: latent_dim_comparison_full.csv")
print(f"  Shape: {results_df.shape}")

# Create summary table (final epoch only)
summary_data = []
for latent_dim in LATENT_DIMS:
    history = results[latent_dim]
    summary_data.append({
        'Latent_Dimension': latent_dim,
        'Final_Train_Loss': history['train'][-1],
        'Final_Val_Loss': history['val'][-1],
        'Best_Train_Loss': min(history['train']),
        'Best_Val_Loss': min(history['val']),
        'Train_Improvement': history['train'][0] - history['train'][-1],
        'Val_Improvement': history['val'][0] - history['val'][-1]
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('data33.csv', index=False)

print(f"\n✓ Saved summary: latent_dim_comparison_summary.csv")
print(f"\n{summary_df.to_string(index=False)}")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Color palette
colors = sns.color_palette("husl", len(LATENT_DIMS))
color_map = {dim: colors[i] for i, dim in enumerate(LATENT_DIMS)}

# Plot 1: Training Loss
for latent_dim in LATENT_DIMS:
    history = results[latent_dim]
    ax1.plot(history['epoch'], history['train'], 
             label=f'{latent_dim}D', 
             color=color_map[latent_dim],
             linewidth=2.5,
             marker='o',
             markersize=4,
             markevery=2)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Loss (Contrastive)', fontsize=12, fontweight='bold')
ax1.set_title('Training Loss by Latent Dimension', fontsize=14, fontweight='bold')
ax1.legend(title='Latent Dim', fontsize=10, title_fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, EPOCHS)

# Plot 2: Validation Loss
for latent_dim in LATENT_DIMS:
    history = results[latent_dim]
    ax2.plot(history['epoch'], history['val'], 
             label=f'{latent_dim}D', 
             color=color_map[latent_dim],
             linewidth=2.5,
             marker='s',
             markersize=4,
             markevery=2)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Validation Loss (Contrastive)', fontsize=12, fontweight='bold')
ax2.set_title('Validation Loss by Latent Dimension', fontsize=14, fontweight='bold')
ax2.legend(title='Latent Dim', fontsize=10, title_fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, EPOCHS)

plt.tight_layout()
plt.savefig('latent_dimension_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: latent_dimension_comparison.png")

# Create combined plot
plt.figure(figsize=(12, 7))

for latent_dim in LATENT_DIMS:
    history = results[latent_dim]
    plt.plot(history['epoch'], history['val'], 
             label=f'{latent_dim}D', 
             color=color_map[latent_dim],
             linewidth=3,
             marker='o',
             markersize=5)

plt.xlabel('Epoch', fontsize=13, fontweight='bold')
plt.ylabel('Validation Loss (Contrastive)', fontsize=13, fontweight='bold')
plt.title('Validation Loss Comparison Across Latent Dimensions', fontsize=15, fontweight='bold')
plt.legend(title='Latent Dimension', fontsize=11, title_fontsize=12, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(1, EPOCHS)

plt.tight_layout()
plt.savefig('latent_dimension_comparison_combined.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved combined visualization: latent_dimension_comparison_combined.png")

plt.close('all')

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL ANALYSIS SUMMARY")
print("="*80)

print("\nPerformance Ranking (by final validation loss):")
ranking = summary_df.sort_values('Final_Val_Loss')
for idx, row in ranking.iterrows():
    print(f"  {row['Latent_Dimension']:3d}D: Val Loss = {row['Final_Val_Loss']:.4f} "
          f"(improved by {row['Val_Improvement']:.4f} from epoch 1)")

best_dim = ranking.iloc[0]['Latent_Dimension']
print(f"\n📊 Based on validation loss, {int(best_dim)}D achieved the best performance.")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)

print("\nGenerated files:")
print("  1. latent_dim_comparison_full.csv - All epochs and metrics")
print("  2. latent_dim_comparison_summary.csv - Final summary statistics")
print("  3. latent_dimension_comparison.png - Training & validation plots")
print("  4. latent_dimension_comparison_combined.png - Combined validation plot")
