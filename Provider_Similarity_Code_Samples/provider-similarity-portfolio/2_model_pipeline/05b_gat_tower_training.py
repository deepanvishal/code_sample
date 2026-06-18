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
from scipy.sparse import load_npz
from collections import defaultdict
import warnings
import gc
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CODE_EMBEDDING_DIM = 128
PROVIDER_EMBEDDING_DIM = 128
NUM_HEADS = 4
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32

print("\n" + "="*80)
print("NOTEBOOK 2: GAT TRAINING & PROVIDER EMBEDDINGS")
print("="*80)
print(f"Seed: {SEED}")

print("\n" + "="*80)
print("LOADING ARTIFACTS FROM NOTEBOOK 1")
print("="*80)

with open('data11.pkl', 'rb') as f:
    code_embeddings = pickle.load(f)
print(f"Loaded code embeddings: {len(code_embeddings)} codes")

all_specialty_codes = np.load('all_specialty_codes.npy', allow_pickle=True).tolist()
print(f"Loaded specialty codes: {len(all_specialty_codes)}")

proc_matrix_filtered = load_npz('data47.npz')
print(f"Loaded procedure matrix: {proc_matrix_filtered.shape}")

with open('data40.pkl', 'rb') as f:
    nb1_metadata = pickle.load(f)
print(f"Loaded metadata (test_mode: {nb1_metadata['test_mode']})")

all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('data44.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

if nb1_metadata['test_mode']:
    num_providers = nb1_metadata['num_providers']
    all_pins = all_pins[:num_providers]

print(f"Total providers: {len(all_pins)}")
print(f"Labeled providers: {len([p for p in all_pins if p in pin_to_label])}")

print("\n" + "="*80)
print("INITIALIZING PROVIDER EMBEDDINGS")
print("="*80)

print("Computing weighted average of code embeddings for each provider...")
provider_init_embeddings = np.zeros((len(all_pins), CODE_EMBEDDING_DIM), dtype=np.float32)

for provider_idx in range(len(all_pins)):
    provider_codes = proc_matrix_filtered[provider_idx].nonzero()[1]
    provider_claims = proc_matrix_filtered[provider_idx].data
    
    if len(provider_codes) == 0:
        continue
    
    total_weight = 0
    for code_idx, claim_count in zip(provider_codes, provider_claims):
        code_id = all_specialty_codes[code_idx]
        if code_id in code_embeddings:
            provider_init_embeddings[provider_idx] += code_embeddings[code_id] * claim_count
            total_weight += claim_count
    
    if total_weight > 0:
        provider_init_embeddings[provider_idx] /= total_weight

print(f"Initialized {len(provider_init_embeddings)} provider embeddings")

np.save('provider_init_embeddings.npy', provider_init_embeddings)
print("Saved: provider_init_embeddings.npy")

print("\n" + "="*80)
print("DEFINING GAT MODEL")
print("="*80)

class ProviderGAT(nn.Module):
    def __init__(self, code_dim, provider_dim, num_specialties, num_heads, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.provider_dim = provider_dim
        
        self.W_heads = nn.ModuleList([
            nn.Linear(code_dim, provider_dim, bias=False) for _ in range(num_heads)
        ])
        
        self.a_heads = nn.ParameterList([
            nn.Parameter(torch.randn(2 * provider_dim, 1)) for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.specialty_classifier = nn.Linear(provider_dim * num_heads, num_specialties)
    
    def forward(self, provider_emb, code_embs):
        provider_emb = provider_emb.squeeze(0)
        
        head_outputs = []
        attention_weights_all_heads = []
        
        for head_idx in range(self.num_heads):
            W = self.W_heads[head_idx]
            a = self.a_heads[head_idx]
            
            provider_h = W(provider_emb)
            code_h = W(code_embs)
            
            n_codes = code_embs.shape[0]
            provider_repeated = provider_h.repeat(n_codes, 1)
            
            concat = torch.cat([provider_repeated, code_h], dim=1)
            e_scores = self.leaky_relu(concat @ a).squeeze(-1)
            
            attention_weights = F.softmax(e_scores, dim=0)
            attention_weights_all_heads.append(attention_weights.detach())
            
            attention_weights = self.dropout(attention_weights)
            
            attended_code = (attention_weights.unsqueeze(1) * code_h).sum(dim=0)
            head_outputs.append(attended_code)
        
        combined_emb = torch.cat(head_outputs, dim=0)
        
        specialty_logits = self.specialty_classifier(combined_emb)
        
        return specialty_logits, combined_emb

print(f"Model architecture:")
print(f"  Input: {CODE_EMBEDDING_DIM}-dim code embeddings")
print(f"  GAT heads: {NUM_HEADS}")
print(f"  Output: {PROVIDER_EMBEDDING_DIM * NUM_HEADS}-dim provider embeddings")

print("\n" + "="*80)
print("PREPARING TRAINING DATA")
print("="*80)

specialties = sorted(list(set(pin_to_label.values())))
specialty_to_id = {spec: idx for idx, spec in enumerate(specialties)}
num_specialties = len(specialties)

print(f"Number of specialties: {num_specialties}")

provider_code_data = {}
for provider_idx in range(len(all_pins)):
    pin = all_pins[provider_idx]
    if pin not in pin_to_label:
        continue
    
    provider_codes = proc_matrix_filtered[provider_idx].nonzero()[1]
    
    code_emb_list = []
    for code_idx in provider_codes:
        code_id = all_specialty_codes[code_idx]
        if code_id in code_embeddings:
            code_emb_list.append(code_embeddings[code_id])
    
    if len(code_emb_list) > 0:
        provider_code_data[provider_idx] = {
            'code_embs': np.array(code_emb_list, dtype=np.float32),
            'specialty_id': specialty_to_id[pin_to_label[pin]]
        }

valid_labeled_indices = list(provider_code_data.keys())
print(f"Valid providers for training: {len(valid_labeled_indices)}")

model = ProviderGAT(CODE_EMBEDDING_DIM, PROVIDER_EMBEDDING_DIM, 
                    num_specialties, NUM_HEADS, dropout=0.3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("\n" + "="*80)
print("TRAINING GAT MODEL")
print("="*80)

for epoch in range(EPOCHS):
    model.train()
    
    np.random.seed(SEED + epoch)
    np.random.shuffle(valid_labeled_indices)
    batch_indices = [valid_labeled_indices[i:i+BATCH_SIZE] 
                     for i in range(0, len(valid_labeled_indices), BATCH_SIZE)]
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_idx_list in batch_indices:
        batch_loss = 0
        batch_correct = 0
        batch_total = 0
        
        for provider_idx in batch_idx_list:
            data = provider_code_data[provider_idx]
            
            provider_emb = torch.FloatTensor(provider_init_embeddings[provider_idx]).unsqueeze(0).to(device)
            code_embs = torch.FloatTensor(data['code_embs']).to(device)
            specialty_label = torch.LongTensor([data['specialty_id']]).to(device)
            
            specialty_logits, _ = model(provider_emb, code_embs)
            loss = criterion(specialty_logits.unsqueeze(0), specialty_label)
            
            batch_loss += loss
            pred = specialty_logits.argmax()
            batch_correct += (pred == specialty_label[0]).item()
            batch_total += 1
        
        if batch_total > 0:
            optimizer.zero_grad()
            (batch_loss / batch_total).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            correct += batch_correct
            total += batch_total
    
    if (epoch + 1) % 10 == 0:
        acc = correct / total if total > 0 else 0
        avg_loss = epoch_loss / total if total > 0 else 0
        print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Accuracy={acc:.3f}")

del provider_code_data
gc.collect()

print("\n" + "="*80)
print("GENERATING FINAL PROVIDER EMBEDDINGS")
print("="*80)

model.eval()
final_embeddings = np.zeros((len(all_pins), PROVIDER_EMBEDDING_DIM * NUM_HEADS), dtype=np.float32)

print("Generating embeddings for all providers...")
INFERENCE_BATCH_SIZE = 128

with torch.no_grad():
    for batch_start in range(0, len(all_pins), INFERENCE_BATCH_SIZE):
        batch_end = min(batch_start + INFERENCE_BATCH_SIZE, len(all_pins))
        
        for provider_idx in range(batch_start, batch_end):
            provider_codes = proc_matrix_filtered[provider_idx].nonzero()[1]
            
            if len(provider_codes) == 0:
                final_embeddings[provider_idx] = np.tile(provider_init_embeddings[provider_idx], NUM_HEADS)
                continue
            
            code_emb_list = []
            for code_idx in provider_codes:
                code_id = all_specialty_codes[code_idx]
                if code_id in code_embeddings:
                    code_emb_list.append(code_embeddings[code_id])
            
            if len(code_emb_list) == 0:
                final_embeddings[provider_idx] = np.tile(provider_init_embeddings[provider_idx], NUM_HEADS)
                continue
            
            provider_emb = torch.FloatTensor(provider_init_embeddings[provider_idx]).unsqueeze(0).to(device)
            code_embs = torch.FloatTensor(np.array(code_emb_list, dtype=np.float32)).to(device)
            
            _, updated_emb = model(provider_emb, code_embs)
            final_embeddings[provider_idx] = updated_emb.cpu().numpy()
        
        if (batch_end) % 1000 == 0 or batch_end == len(all_pins):
            print(f"  Processed {batch_end}/{len(all_pins)} providers")

print(f"Generated embeddings shape: {final_embeddings.shape}")

print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

specialty_column = [pin_to_label.get(pin, 'UNLABELED') for pin in all_pins]

embedding_dict = {'PIN': all_pins}
for i in range(final_embeddings.shape[1]):
    embedding_dict[f'emb_{i}'] = final_embeddings[:, i]
embedding_dict['specialty'] = specialty_column

embedding_df = pd.DataFrame(embedding_dict)

embedding_df.to_csv('data36.csv', index=False, float_format='%.6f')
print("Saved: me2vec_provider_embeddings.csv")

np.save('me2vec_provider_embeddings.npy', final_embeddings)
print("Saved: me2vec_provider_embeddings.npy")

torch.save(model.state_dict(), 'me2vec_gat_model.pth')
print("Saved: me2vec_gat_model.pth (for attention extraction)")

metadata = {
    'code_embedding_dim': CODE_EMBEDDING_DIM,
    'provider_embedding_dim': PROVIDER_EMBEDDING_DIM,
    'num_heads': NUM_HEADS,
    'num_providers': len(all_pins),
    'num_labeled': len(pin_to_label),
    'num_specialties': num_specialties,
    'specialty_to_id': specialty_to_id,
    'specialties': specialties,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'seed': SEED
}

with open('data35.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("Saved: me2vec_metadata.pkl")

print("\n" + "="*80)
print("NOTEBOOK 2 COMPLETE!")
print("="*80)
print("\nOutputs:")
print("  1. me2vec_provider_embeddings.csv - Provider embeddings (human-readable)")
print("  2. me2vec_provider_embeddings.npy - Provider embeddings (numpy array)")
print("  3. me2vec_gat_model.pth - Trained GAT model (for attention extraction)")
print("  4. provider_init_embeddings.npy - Pre-GAT embeddings")
print("  5. me2vec_metadata.pkl - Training metadata")
print(f"\nFinal embeddings: {len(all_pins)} providers × {final_embeddings.shape[1]} dimensions")
