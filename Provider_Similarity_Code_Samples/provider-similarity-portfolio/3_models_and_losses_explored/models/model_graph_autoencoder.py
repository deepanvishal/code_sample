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
import cudf
import cupy as cp
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("GRAPH AUTOENCODER - STEP 1")
print("="*80)
print(f"Device: {device}\n")

# Assuming amt_smry_df has columns: PIN, Medical_code, claims
print(f"Data shape: {len(amt_smry_df):,} rows")

print("\nStep 1: Building TF-IDF weighted graph...")

df = cudf.DataFrame(amt_smry_df[['PIN', 'Medical_code', 'claims']])

grouped = df.groupby(['PIN', 'Medical_code'], as_index=False)['claims'].sum()
provider_totals = grouped.groupby('PIN')['claims'].transform('sum')
grouped['distribution'] = grouped['claims'] / provider_totals

code_provider_counts = grouped.groupby('Medical_code')['PIN'].nunique()
n_total_providers = grouped['PIN'].nunique()
idf_values = cp.log(n_total_providers / code_provider_counts.values)
idf_series = cudf.Series(idf_values, index=code_provider_counts.index)
grouped['idf'] = grouped['Medical_code'].map(idf_series)

grouped['weight'] = grouped['distribution'] * (1.0 + grouped['idf'])
grouped['weight'] = grouped['weight'] / grouped['weight'].max()

grouped = grouped.to_pandas()

all_providers = grouped['PIN'].unique()
all_codes = grouped['Medical_code'].unique()
provider_to_idx = {p: i for i, p in enumerate(all_providers)}
code_to_idx = {c: i + len(all_providers) for i, c in enumerate(all_codes)}
n_providers = len(all_providers)
n_codes = len(all_codes)

print(f"Providers: {n_providers:,}, Codes: {n_codes:,}, Edges: {len(grouped):,}")

grouped['provider_idx'] = grouped['PIN'].map(provider_to_idx)
grouped['code_idx'] = grouped['Medical_code'].map(code_to_idx)

edge_index = torch.LongTensor(
    np.vstack([
        np.concatenate([grouped['provider_idx'].values, grouped['code_idx'].values]),
        np.concatenate([grouped['code_idx'].values, grouped['provider_idx'].values])
    ])
)

edge_weight = torch.FloatTensor(
    np.concatenate([grouped['weight'].values, grouped['weight'].values])
)

x = torch.eye(n_providers + n_codes, dtype=torch.float32)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight).to(device)

print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")

del grouped
torch.cuda.empty_cache()

print("\nStep 2: Building Graph Autoencoder...")

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        
    def encode(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)
    
    def forward(self, data):
        z = self.encode(data.x, data.edge_index, data.edge_attr)
        return z

model = GraphAutoencoder(
    input_dim=n_providers + n_codes,
    hidden_dim=256,
    embedding_dim=128
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nStep 3: Training Graph Autoencoder (10 epochs)...")

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    z = model(data)
    
    reconstructed = model.decode(z, data.edge_index)
    loss = F.mse_loss(reconstructed, data.edge_attr)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")

print("\nStep 4: Extracting provider embeddings...")

model.eval()
with torch.no_grad():
    z = model(data)
    provider_embeddings = z[:n_providers].cpu().numpy()

provider_embeddings = provider_embeddings / (np.linalg.norm(provider_embeddings, axis=1, keepdims=True) + 1e-8)

embeddings_df = pd.DataFrame(
    provider_embeddings,
    columns=[f'emb_{i}' for i in range(128)]
)
embeddings_df['PIN'] = all_providers

print(f"\nEmbeddings created: {embeddings_df.shape}")
print(embeddings_df.head())

print("\nStep 5: Quality check...")

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
print(f"(Good: mean ~0.3-0.5, std > 0.1)")

print("\nStep 6: Saving...")

embeddings_df.to_csv('data29.csv', index=False)

torch.save({
    'model_state': model.state_dict(),
    'provider_to_idx': provider_to_idx,
    'code_to_idx': code_to_idx,
    'n_providers': n_providers,
    'n_codes': n_codes
}, 'graph_autoencoder_model.pt')

print("Saved: graph_autoencoder_embeddings.csv, graph_autoencoder_model.pt")
print("="*80)
