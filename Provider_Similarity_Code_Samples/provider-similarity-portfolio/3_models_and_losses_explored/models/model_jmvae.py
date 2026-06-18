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
import warnings
warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("JMVAE - TWO MODALITIES: PROCEDURES & DIAGNOSES")
print("="*80)
print(f"Device: {device}\n")

# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================
print("Generating sample data...")
np.random.seed(42)

n_pins = 1000
proc_codes = [f'PROC_{i:03d}' for i in range(300)]
diag_codes = [f'DIAG_{i:03d}' for i in range(400)]
pins = list(range(1, n_pins + 1))

amt_data = []
for pin in pins:
    # Procedures
    for code in np.random.choice(proc_codes, size=np.random.randint(5, 20), replace=False):
        amt_data.append({
            'PIN': pin, 
            'Medical_code': code, 
            'dollar_value': np.random.exponential(1000),
            'code_type': 'Procedure'
        })
    # Diagnoses
    for code in np.random.choice(diag_codes, size=np.random.randint(3, 15), replace=False):
        amt_data.append({
            'PIN': pin, 
            'Medical_code': code, 
            'dollar_value': np.random.exponential(500),
            'code_type': 'Diagnosis'
        })

amt_smry_df = pd.DataFrame(amt_data)

print(f"Amount summary data: {amt_smry_df.shape}")
print(f"Code types: {amt_smry_df['code_type'].value_counts().to_dict()}")
print()

# ============================================================================
# SPLIT INTO PROCEDURES AND DIAGNOSES
# ============================================================================
print("Step 1: Splitting into Procedures and Diagnoses...")

proc_df = amt_smry_df[amt_smry_df['code_type'] == 'Procedure'].copy()
diag_df = amt_smry_df[amt_smry_df['code_type'] == 'Diagnosis'].copy()

print(f"Procedure data: {proc_df.shape}")
print(f"Diagnosis data: {diag_df.shape}")

# ============================================================================
# PROCESS PROCEDURE DATA
# ============================================================================
print("\nStep 2: Processing procedure data...")

proc_df['dollar_value'] = pd.to_numeric(proc_df['dollar_value'], errors='coerce').astype(np.float32)
proc_df = proc_df.dropna(subset=['dollar_value'])

proc_grouped = proc_df.groupby(['PIN', 'Medical_code'])['dollar_value'].sum().reset_index()

proc_feature_dicts = []
proc_pin_list = []

for pin in proc_grouped['PIN'].unique():
    pin_data = proc_grouped[proc_grouped['PIN'] == pin]
    spending_dict = dict(zip(pin_data['Medical_code'], pin_data['dollar_value']))
    proc_feature_dicts.append(spending_dict)
    proc_pin_list.append(pin)

proc_vectorizer = DictVectorizer(dtype=np.float32, sparse=True)
proc_sparse = proc_vectorizer.fit_transform(proc_feature_dicts)

# Normalize procedures
proc_row_sums = np.array(proc_sparse.sum(axis=1)).flatten()
proc_row_sums_inv = 1.0 / (proc_row_sums + 1e-8)
proc_row_sums_inv = proc_row_sums_inv.reshape(-1, 1)
proc_normalized = proc_sparse.multiply(proc_row_sums_inv)
proc_vectors = proc_normalized.toarray().astype(np.float32)

print(f"Procedure vectors: {proc_vectors.shape}")
print(f"Procedure PINs: {len(proc_pin_list)}")

# ============================================================================
# PROCESS DIAGNOSIS DATA
# ============================================================================
print("\nStep 3: Processing diagnosis data...")

diag_df['dollar_value'] = pd.to_numeric(diag_df['dollar_value'], errors='coerce').astype(np.float32)
diag_df = diag_df.dropna(subset=['dollar_value'])

diag_grouped = diag_df.groupby(['PIN', 'Medical_code'])['dollar_value'].sum().reset_index()

diag_feature_dicts = []
diag_pin_list = []

for pin in diag_grouped['PIN'].unique():
    pin_data = diag_grouped[diag_grouped['PIN'] == pin]
    spending_dict = dict(zip(pin_data['Medical_code'], pin_data['dollar_value']))
    diag_feature_dicts.append(spending_dict)
    diag_pin_list.append(pin)

diag_vectorizer = DictVectorizer(dtype=np.float32, sparse=True)
diag_sparse = diag_vectorizer.fit_transform(diag_feature_dicts)

# Normalize diagnoses
diag_row_sums = np.array(diag_sparse.sum(axis=1)).flatten()
diag_row_sums_inv = 1.0 / (diag_row_sums + 1e-8)
diag_row_sums_inv = diag_row_sums_inv.reshape(-1, 1)
diag_normalized = diag_sparse.multiply(diag_row_sums_inv)
diag_vectors = diag_normalized.toarray().astype(np.float32)

print(f"Diagnosis vectors: {diag_vectors.shape}")
print(f"Diagnosis PINs: {len(diag_pin_list)}")

# ============================================================================
# ALIGN PINS (Get common PINs between both modalities)
# ============================================================================
print("\nStep 4: Aligning PINs across modalities...")

common_pins = list(set(proc_pin_list) & set(diag_pin_list))
common_pins.sort()

print(f"Common PINs: {len(common_pins)}")

# Create aligned matrices
proc_aligned = np.zeros((len(common_pins), proc_vectors.shape[1]), dtype=np.float32)
diag_aligned = np.zeros((len(common_pins), diag_vectors.shape[1]), dtype=np.float32)

proc_pin_to_idx = {pin: idx for idx, pin in enumerate(proc_pin_list)}
diag_pin_to_idx = {pin: idx for idx, pin in enumerate(diag_pin_list)}

for i, pin in enumerate(common_pins):
    proc_aligned[i] = proc_vectors[proc_pin_to_idx[pin]]
    diag_aligned[i] = diag_vectors[diag_pin_to_idx[pin]]

print(f"Aligned procedure vectors: {proc_aligned.shape}")
print(f"Aligned diagnosis vectors: {diag_aligned.shape}")

# ============================================================================
# CONVERT TO TENSORS
# ============================================================================
print("\nStep 5: Converting to tensors...")

proc_tensor = torch.FloatTensor(proc_aligned).to(device)
diag_tensor = torch.FloatTensor(diag_aligned).to(device)

print(f"Procedure tensor: {proc_tensor.shape}")
print(f"Diagnosis tensor: {diag_tensor.shape}")

# ============================================================================
# CREATE JMVAE MODEL
# ============================================================================
print("\nStep 6: Creating JMVAE model...")

proc_dim = proc_tensor.shape[1]
diag_dim = diag_tensor.shape[1]
hidden_dim = 128
latent_dim = 64

print(f"Dimensions - Procedure: {proc_dim}, Diagnosis: {diag_dim}")
print(f"Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")

# Encoders
proc_enc1 = nn.Linear(proc_dim, hidden_dim).to(device)
proc_enc2 = nn.Linear(hidden_dim, hidden_dim//2).to(device)

diag_enc1 = nn.Linear(diag_dim, hidden_dim).to(device)
diag_enc2 = nn.Linear(hidden_dim, hidden_dim//2).to(device)

# Joint latent layers
joint_dim = (hidden_dim//2) * 2
mu_layer = nn.Linear(joint_dim, latent_dim).to(device)
logvar_layer = nn.Linear(joint_dim, latent_dim).to(device)

# Decoders
proc_dec1 = nn.Linear(latent_dim, hidden_dim//2).to(device)
proc_dec2 = nn.Linear(hidden_dim//2, proc_dim).to(device)

diag_dec1 = nn.Linear(latent_dim, hidden_dim//2).to(device)
diag_dec2 = nn.Linear(hidden_dim//2, diag_dim).to(device)

# Collect all parameters
all_parameters = []
for layer in [proc_enc1, proc_enc2, diag_enc1, diag_enc2,
              mu_layer, logvar_layer, proc_dec1, proc_dec2,
              diag_dec1, diag_dec2]:
    all_parameters.extend(layer.parameters())

optimizer = torch.optim.Adam(all_parameters, lr=0.001)

print("Model created successfully")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_data(proc_batch, diag_batch):
    # Procedure encoding
    proc_h1 = F.relu(proc_enc1(proc_batch))
    proc_h2 = F.relu(proc_enc2(proc_h1))
    
    # Diagnosis encoding
    diag_h1 = F.relu(diag_enc1(diag_batch))
    diag_h2 = F.relu(diag_enc2(diag_h1))
    
    # Joint representation
    joint_h = torch.cat([proc_h2, diag_h2], dim=1)
    mu = mu_layer(joint_h)
    logvar = logvar_layer(joint_h)
    
    return mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def decode_data(z):
    # Procedure decoding
    proc_h = F.relu(proc_dec1(z))
    proc_recon = F.softmax(proc_dec2(proc_h), dim=1)
    
    # Diagnosis decoding
    diag_h = F.relu(diag_dec1(z))
    diag_recon = F.softmax(diag_dec2(diag_h), dim=1)
    
    return proc_recon, diag_recon

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\nStep 7: Training JMVAE...")

batch_size = 32
epochs = 50
n_samples = len(common_pins)
n_batches = (n_samples + batch_size - 1) // batch_size

print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Batches per epoch: {n_batches}")

# Set to training mode
for layer in [proc_enc1, proc_enc2, diag_enc1, diag_enc2,
              mu_layer, logvar_layer, proc_dec1, proc_dec2,
              diag_dec1, diag_dec2]:
    layer.train()

for epoch in range(epochs):
    total_loss = 0
    
    # Shuffle indices
    indices = torch.randperm(n_samples)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Get batch data
        proc_batch = proc_tensor[batch_indices]
        diag_batch = diag_tensor[batch_indices]
        
        optimizer.zero_grad()
        
        # Forward pass
        mu, logvar = encode_data(proc_batch, diag_batch)
        z = reparameterize(mu, logvar)
        proc_recon, diag_recon = decode_data(z)
        
        # Reconstruction losses (MSE)
        proc_loss = F.mse_loss(proc_recon, proc_batch)
        diag_loss = F.mse_loss(diag_recon, diag_batch)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(proc_batch)
        
        # Total loss
        loss = proc_loss + diag_loss + 0.001 * kl_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / n_batches
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:2d}: Loss={avg_loss:.4f}')

print("Training completed!")

# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================
print("\nStep 8: Extracting embeddings...")

# Set to evaluation mode
for layer in [proc_enc1, proc_enc2, diag_enc1, diag_enc2,
              mu_layer, logvar_layer]:
    layer.eval()

all_embeddings = []

with torch.no_grad():
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        proc_batch = proc_tensor[start_idx:end_idx]
        diag_batch = diag_tensor[start_idx:end_idx]
        
        mu, _ = encode_data(proc_batch, diag_batch)
        all_embeddings.append(mu.cpu())

# Combine all embeddings
final_embeddings = torch.cat(all_embeddings, dim=0).numpy()

# Create embeddings dataframe
embeddings_df = pd.DataFrame(
    final_embeddings,
    columns=[f'emb_{i}' for i in range(latent_dim)]
)
embeddings_df['PIN'] = common_pins

print(f"\nFinal embeddings: {embeddings_df.shape}")
print(embeddings_df.head())

# ============================================================================
# SAVE EMBEDDINGS AND MODEL
# ============================================================================
print("\nStep 9: Saving embeddings and model...")

# Save embeddings
embeddings_df.to_csv('data31.csv', index=False)
print("Saved: jmvae_proc_diag_embeddings.csv")

# Save model and metadata
torch.save({
    'proc_enc1': proc_enc1.state_dict(),
    'proc_enc2': proc_enc2.state_dict(),
    'diag_enc1': diag_enc1.state_dict(),
    'diag_enc2': diag_enc2.state_dict(),
    'mu_layer': mu_layer.state_dict(),
    'logvar_layer': logvar_layer.state_dict(),
    'proc_dec1': proc_dec1.state_dict(),
    'proc_dec2': proc_dec2.state_dict(),
    'diag_dec1': diag_dec1.state_dict(),
    'diag_dec2': diag_dec2.state_dict(),
    'proc_vectorizer': proc_vectorizer,
    'diag_vectorizer': diag_vectorizer,
    'common_pins': common_pins,
    'proc_dim': proc_dim,
    'diag_dim': diag_dim,
    'hidden_dim': hidden_dim,
    'latent_dim': latent_dim
}, 'jmvae_proc_diag_model.pt')
print("Saved: jmvae_proc_diag_model.pt")

print("\n" + "="*80)
print("JMVAE TRAINING COMPLETE")
print("="*80)

# ============================================================================
# INFERENCE EXAMPLE
# ============================================================================
print("\n" + "="*80)
print("INFERENCE EXAMPLE")
print("="*80)

def inference_on_new_data(new_amt_smry_df, model_path='jmvae_proc_diag_model.pt'):
    """
    Run inference on new data
    
    Args:
        new_amt_smry_df: DataFrame with columns ['PIN', 'Medical_code', 'dollar_value', 'code_type']
        model_path: Path to saved model
    
    Returns:
        embeddings_df: DataFrame with embeddings for new PINs
    """
    
    print("Loading saved model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model architecture
    proc_dim = checkpoint['proc_dim']
    diag_dim = checkpoint['diag_dim']
    hidden_dim = checkpoint['hidden_dim']
    latent_dim = checkpoint['latent_dim']
    
    # Recreate encoders
    proc_enc1 = nn.Linear(proc_dim, hidden_dim).to(device)
    proc_enc2 = nn.Linear(hidden_dim, hidden_dim//2).to(device)
    diag_enc1 = nn.Linear(diag_dim, hidden_dim).to(device)
    diag_enc2 = nn.Linear(hidden_dim, hidden_dim//2).to(device)
    joint_dim = (hidden_dim//2) * 2
    mu_layer = nn.Linear(joint_dim, latent_dim).to(device)
    logvar_layer = nn.Linear(joint_dim, latent_dim).to(device)
    
    # Load weights
    proc_enc1.load_state_dict(checkpoint['proc_enc1'])
    proc_enc2.load_state_dict(checkpoint['proc_enc2'])
    diag_enc1.load_state_dict(checkpoint['diag_enc1'])
    diag_enc2.load_state_dict(checkpoint['diag_enc2'])
    mu_layer.load_state_dict(checkpoint['mu_layer'])
    logvar_layer.load_state_dict(checkpoint['logvar_layer'])
    
    # Load vectorizers
    proc_vectorizer = checkpoint['proc_vectorizer']
    diag_vectorizer = checkpoint['diag_vectorizer']
    
    print("Processing new data...")
    
    # Split into procedures and diagnoses
    new_proc_df = new_amt_smry_df[new_amt_smry_df['code_type'] == 'Procedure'].copy()
    new_diag_df = new_amt_smry_df[new_amt_smry_df['code_type'] == 'Diagnosis'].copy()
    
    # Process procedures
    new_proc_df['dollar_value'] = pd.to_numeric(new_proc_df['dollar_value'], errors='coerce').astype(np.float32)
    new_proc_grouped = new_proc_df.groupby(['PIN', 'Medical_code'])['dollar_value'].sum().reset_index()
    
    new_proc_dicts = []
    new_proc_pins = []
    for pin in new_proc_grouped['PIN'].unique():
        pin_data = new_proc_grouped[new_proc_grouped['PIN'] == pin]
        spending_dict = dict(zip(pin_data['Medical_code'], pin_data['dollar_value']))
        new_proc_dicts.append(spending_dict)
        new_proc_pins.append(pin)
    
    new_proc_sparse = proc_vectorizer.transform(new_proc_dicts)
    new_proc_row_sums = np.array(new_proc_sparse.sum(axis=1)).flatten()
    new_proc_row_sums_inv = 1.0 / (new_proc_row_sums + 1e-8)
    new_proc_vectors = new_proc_sparse.multiply(new_proc_row_sums_inv.reshape(-1, 1)).toarray().astype(np.float32)
    
    # Process diagnoses
    new_diag_df['dollar_value'] = pd.to_numeric(new_diag_df['dollar_value'], errors='coerce').astype(np.float32)
    new_diag_grouped = new_diag_df.groupby(['PIN', 'Medical_code'])['dollar_value'].sum().reset_index()
    
    new_diag_dicts = []
    new_diag_pins = []
    for pin in new_diag_grouped['PIN'].unique():
        pin_data = new_diag_grouped[new_diag_grouped['PIN'] == pin]
        spending_dict = dict(zip(pin_data['Medical_code'], pin_data['dollar_value']))
        new_diag_dicts.append(spending_dict)
        new_diag_pins.append(pin)
    
    new_diag_sparse = diag_vectorizer.transform(new_diag_dicts)
    new_diag_row_sums = np.array(new_diag_sparse.sum(axis=1)).flatten()
    new_diag_row_sums_inv = 1.0 / (new_diag_row_sums + 1e-8)
    new_diag_vectors = new_diag_sparse.multiply(new_diag_row_sums_inv.reshape(-1, 1)).toarray().astype(np.float32)
    
    # Align PINs
    new_common_pins = list(set(new_proc_pins) & set(new_diag_pins))
    
    new_proc_aligned = np.zeros((len(new_common_pins), proc_dim), dtype=np.float32)
    new_diag_aligned = np.zeros((len(new_common_pins), diag_dim), dtype=np.float32)
    
    new_proc_pin_to_idx = {pin: idx for idx, pin in enumerate(new_proc_pins)}
    new_diag_pin_to_idx = {pin: idx for idx, pin in enumerate(new_diag_pins)}
    
    for i, pin in enumerate(new_common_pins):
        new_proc_aligned[i] = new_proc_vectors[new_proc_pin_to_idx[pin]]
        new_diag_aligned[i] = new_diag_vectors[new_diag_pin_to_idx[pin]]
    
    # Convert to tensors
    new_proc_tensor = torch.FloatTensor(new_proc_aligned).to(device)
    new_diag_tensor = torch.FloatTensor(new_diag_aligned).to(device)
    
    print(f"Generating embeddings for {len(new_common_pins)} PINs...")
    
    # Set to eval mode
    proc_enc1.eval()
    proc_enc2.eval()
    diag_enc1.eval()
    diag_enc2.eval()
    mu_layer.eval()
    
    # Generate embeddings
    with torch.no_grad():
        proc_h1 = F.relu(proc_enc1(new_proc_tensor))
        proc_h2 = F.relu(proc_enc2(proc_h1))
        
        diag_h1 = F.relu(diag_enc1(new_diag_tensor))
        diag_h2 = F.relu(diag_enc2(diag_h1))
        
        joint_h = torch.cat([proc_h2, diag_h2], dim=1)
        mu = mu_layer(joint_h)
        
        new_embeddings = mu.cpu().numpy()
    
    # Create dataframe
    new_embeddings_df = pd.DataFrame(
        new_embeddings,
        columns=[f'emb_{i}' for i in range(latent_dim)]
    )
    new_embeddings_df['PIN'] = new_common_pins
    
    print(f"Generated embeddings: {new_embeddings_df.shape}")
    
    return new_embeddings_df


# Test inference on a sample of the training data
print("\nTesting inference on sample data...")
sample_pins = np.random.choice(common_pins, size=min(50, len(common_pins)), replace=False)
sample_data = amt_smry_df[amt_smry_df['PIN'].isin(sample_pins)]

new_embeddings = inference_on_new_data(sample_data)
print("\nSample of new embeddings:")
print(new_embeddings.head())

print("\n" + "="*80)
print("ALL DONE!")
print("="*80)
