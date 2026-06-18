# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

set_seeds(42)
device = torch.device('cpu')

print("="*80)
print("VARIATIONAL GRAPH AUTOENCODER (VGAE) - CPU VERSION")
print("="*80)
print(f"Device: {device}")
print()

# ============================================================================
# SAMPLE DATA GENERATION (Comment this out when using real data)
# ============================================================================
print("Generating sample data for testing...")
np.random.seed(42)

n_sample_providers = 500
n_sample_codes = 200
n_transactions = 5000

sample_pins = [f"PIN_{i:04d}" for i in range(n_sample_providers)]
sample_codes = [f"CODE_{i:03d}" for i in range(n_sample_codes)]

pins = np.random.choice(sample_pins, n_transactions)
codes = np.random.choice(sample_codes, n_transactions)
values = np.random.lognormal(mean=5, sigma=2, size=n_transactions)

amt_smry_df = pd.DataFrame({
    'PIN': pins,
    'Medical_code': codes,
    'dollar_value': values
})

print(f"Sample data: {len(amt_smry_df)} transactions")
print(f"Unique PINs: {amt_smry_df['PIN'].nunique()}")
print(f"Unique Codes: {amt_smry_df['Medical_code'].nunique()}")
print()
# ============================================================================
# END SAMPLE DATA - Use your real amt_smry_df below this line
# ============================================================================

print("Step 1: Processing data...")

amt_smry_df['dollar_value'] = pd.to_numeric(amt_smry_df['dollar_value'], errors='coerce').astype(np.float32)
amt_smry_df = amt_smry_df.dropna(subset=['dollar_value'])

grouped = amt_smry_df.groupby(['PIN', 'Medical_code'])['dollar_value'].sum().reset_index()

feature_dicts = []
pin_list = []

for pin in grouped['PIN'].unique():
    pin_data = grouped[grouped['PIN'] == pin]
    spending_dict = dict(zip(pin_data['Medical_code'], pin_data['dollar_value']))
    feature_dicts.append(spending_dict)
    pin_list.append(pin)

vectorizer = DictVectorizer(dtype=np.float32, sparse=True)
amt_smry_sparse = vectorizer.fit_transform(feature_dicts)

row_sums = np.array(amt_smry_sparse.sum(axis=1)).flatten()

if (row_sums == 0).any():
    non_zero_mask = row_sums > 0
    amt_smry_sparse = amt_smry_sparse[non_zero_mask]
    pin_list = [pin_list[i] for i in range(len(pin_list)) if non_zero_mask[i]]
    row_sums = row_sums[non_zero_mask]

row_sums_inv = 1.0 / row_sums
row_sums_inv = row_sums_inv.reshape(-1, 1)
amt_smry_normalized = amt_smry_sparse.multiply(row_sums_inv)
amt_smry_vectors = amt_smry_normalized.toarray().astype(np.float32)

n_providers = len(pin_list)
n_codes = amt_smry_vectors.shape[1]

print(f"Providers: {n_providers:,}, Codes: {n_codes:,}")

print("\nStep 2: Computing TF-IDF weights...")

code_doc_freq = (amt_smry_vectors > 0).sum(axis=0)
idf = np.log(n_providers / (code_doc_freq + 1))

amt_smry_tfidf = amt_smry_vectors * idf
max_weight = amt_smry_tfidf.max()
amt_smry_tfidf = amt_smry_tfidf / max_weight

print(f"IDF computed for {n_codes} codes")

print("\nStep 3: Building bipartite graph...")

edge_list = []
edge_weights = []

for i, provider_vector in enumerate(amt_smry_tfidf):
    nonzero_indices = np.nonzero(provider_vector)[0]
    for code_idx in nonzero_indices:
        edge_list.append([i, n_providers + code_idx])
        edge_weights.append(provider_vector[code_idx])

edge_index = torch.LongTensor(edge_list).t()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

edge_weight = torch.FloatTensor(edge_weights)
edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

print(f"Graph: {n_providers + n_codes} nodes, {len(edge_index[0])} edges")

del amt_smry_sparse, amt_smry_normalized, amt_smry_vectors

print("\nStep 4: Building VGAE Model...")

class VariationalGCNEncoder(nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels):
        super().__init__()
        # Use embedding layer instead of identity matrix
        self.initial_embed = nn.Embedding(num_nodes, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, edge_index, edge_weight=None):
        # Create node indices for ALL nodes
        num_nodes = self.initial_embed.num_embeddings
        x = self.initial_embed(torch.arange(num_nodes, device=edge_index.device))
        
        # Apply first convolution
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # Output mean and log std
        mu = self.conv_mu(x, edge_index, edge_weight)
        logstd = self.conv_logstd(x, edge_index, edge_weight)
        
        return mu, logstd

# Create encoder
encoder = VariationalGCNEncoder(
    num_nodes=n_providers + n_codes,
    hidden_channels=256,
    out_channels=128
)

# Create VGAE model
model = VGAE(encoder).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nStep 5: Moving graph to CPU and training...")

# Keep graph data on CPU
edge_index_cpu = edge_index
edge_weight_cpu = edge_weight

print("Starting training (10 epochs)...")

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Encode (returns mu and logstd, sampling happens inside)
    z = model.encode(edge_index_cpu, edge_weight_cpu)
    
    # Compute losses
    recon_loss = model.recon_loss(z, edge_index_cpu)
    kl_loss = model.kl_loss() / (n_providers + n_codes)
    
    loss = recon_loss + kl_loss
    
    loss.backward()
    optimizer.step()
    
    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f} (Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f})")

print("\nStep 6: Extracting provider embeddings...")

model.eval()
with torch.no_grad():
    # Get embeddings for all nodes
    z = model.encode(edge_index_cpu, edge_weight_cpu)
    
    # Extract only provider embeddings (first n_providers nodes)
    provider_embeddings = z[:n_providers].cpu().numpy()

# Normalize
provider_embeddings = provider_embeddings / (np.linalg.norm(provider_embeddings, axis=1, keepdims=True) + 1e-8)

embeddings_df = pd.DataFrame(
    provider_embeddings,
    columns=[f'emb_{i}' for i in range(128)]
)
embeddings_df['PIN'] = pin_list

print(f"\nEmbeddings shape: {embeddings_df.shape}")
print(embeddings_df.head())

print("\nStep 7: Quality check...")

emb_cols = [f'emb_{i}' for i in range(128)]
emb_std = embeddings_df[emb_cols].std().mean()
print(f"Embedding std: {emb_std:.6f} (good if > 0.05)")

sims = []
for _ in range(100):
    pin1, pin2 = np.random.choice(embeddings_df['PIN'].values, 2, replace=False)
    emb1 = embeddings_df[embeddings_df['PIN'] == pin1][emb_cols].values
    emb2 = embeddings_df[embeddings_df['PIN'] == pin2][emb_cols].values
    sims.append(np.dot(emb1, emb2.T)[0][0])

print(f"Random pair similarity: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")

print("\nStep 8: Saving...")

embeddings_df.to_csv('data77.csv', index=False)

torch.save({
    'model_state': model.state_dict(),
    'pin_list': pin_list,
    'n_providers': n_providers,
    'n_codes': n_codes,
    'vectorizer': vectorizer
}, 'vgae_model.pt')

print("Saved: vgae_embeddings.csv, vgae_model.pt")
print("\n" + "="*80)
print("✓ VGAE TRAINING COMPLETE")
print(f"✓ Device used: {device}")
print("="*80)
