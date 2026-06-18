# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import networkx as nx
from scipy.sparse import load_npz, save_npz
from gensim.models import Word2Vec
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import warnings
import time
import os
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

CODE_EMBEDDING_DIM = 128
WALK_LENGTH = 80
NUM_WALKS = 10
P = 4.0
Q = 1.0
PROGRESS_INTERVAL = 10000
SAVE_INTERVAL = 50000

TEST_MODE = False
TEST_NUM_PROVIDERS = 1000
TEST_NUM_CODES = 500

if TEST_MODE:
    WALK_LENGTH = 40
    NUM_WALKS = 5
    SAVE_INTERVAL = 10000

print("\n" + "="*80)
print("NOTEBOOK 1: PROCEDURE CODE EMBEDDINGS (MEMORY-OPTIMIZED)")
print("="*80)
print(f"Mode: {'TEST' if TEST_MODE else 'PRODUCTION'}")
print(f"Seed: {SEED}")
print(f"Walk parameters: length={WALK_LENGTH}, num_walks={NUM_WALKS}, p={P}, q={Q}")
print("\nOptimizations:")
print("  - Global cache per worker (low memory)")
print("  - Incremental walk saving (prevents accumulation)")
print("  - Safe for 32+ workers on large graphs")

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

proc_matrix = load_npz('data49.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('data44.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

with open('data68.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

specialty_code_indices = specialty_mappings['code_indices']

print(f"Total providers: {len(all_pins)}")
print(f"Labeled providers: {len([p for p in all_pins if p in pin_to_label])}")
print(f"Total procedure codes: {proc_matrix.shape[1]}")

if TEST_MODE:
    print(f"\n{'='*80}")
    print("TEST MODE - Sampling Data")
    print(f"{'='*80}")
    
    if TEST_NUM_PROVIDERS:
        labeled_pins = [pin for pin in all_pins if pin in pin_to_label]
        unlabeled_pins = [pin for pin in all_pins if pin not in pin_to_label]
        
        n_labeled_sample = min(TEST_NUM_PROVIDERS // 2, len(labeled_pins))
        n_unlabeled_sample = min(TEST_NUM_PROVIDERS - n_labeled_sample, len(unlabeled_pins))
        
        np.random.seed(SEED)
        sampled_labeled = np.random.choice(labeled_pins, n_labeled_sample, replace=False).tolist()
        sampled_unlabeled = np.random.choice(unlabeled_pins, n_unlabeled_sample, replace=False).tolist()
        sampled_pins = sampled_labeled + sampled_unlabeled
        
        pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}
        sampled_indices = [pin_to_idx[pin] for pin in sampled_pins]
        
        proc_matrix = proc_matrix[sampled_indices, :]
        all_pins = sampled_pins
        
        print(f"Sampled {len(all_pins)} providers ({n_labeled_sample} labeled, {n_unlabeled_sample} unlabeled)")
    
    if TEST_NUM_CODES:
        all_codes = list(range(proc_matrix.shape[1]))
        np.random.seed(SEED)
        sampled_code_indices = sorted(np.random.choice(all_codes, 
                                                       min(TEST_NUM_CODES, len(all_codes)), 
                                                       replace=False))
        proc_matrix = proc_matrix[:, sampled_code_indices]
        print(f"Sampled {len(sampled_code_indices)} procedure codes")

all_specialty_codes = set()
for spec, code_indices in specialty_code_indices.items():
    all_specialty_codes.update(code_indices)
all_specialty_codes = sorted(list(all_specialty_codes))

print(f"Specialty-relevant codes: {len(all_specialty_codes)}")

proc_matrix_filtered = proc_matrix[:, all_specialty_codes]

print("\n" + "="*80)
print("BUILDING CO-OCCURRENCE GRAPH")
print("="*80)

start_time = time.time()
n_providers, n_codes = proc_matrix_filtered.shape

print(f"Computing co-occurrence matrix ({n_providers} providers × {n_codes} codes)...")
proc_matrix_binary = (proc_matrix_filtered > 0).astype(float)
cooccurrence_matrix = proc_matrix_binary.T @ proc_matrix_binary
del proc_matrix_binary

cooccurrence_matrix = cooccurrence_matrix.tocoo()
print(f"Total co-occurrence edges: {cooccurrence_matrix.nnz}")

print("Building NetworkX graph...")
G = nx.Graph()
for i, j, weight in zip(cooccurrence_matrix.row, cooccurrence_matrix.col, cooccurrence_matrix.data):
    if i < j and weight > 0:
        code1 = all_specialty_codes[i]
        code2 = all_specialty_codes[j]
        G.add_edge(code1, code2, weight=weight)

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")
print(f"Time: {time.time() - start_time:.1f}s")

print("\n" + "="*80)
print("PREPARING GLOBAL CACHES FOR WORKERS")
print("="*80)

print("Building neighbor caches...")
_global_neighbor_cache = {node: list(G.neighbors(node)) for node in G.nodes()}
_global_neighbor_set_cache = {node: set(neighbors) for node, neighbors in _global_neighbor_cache.items()}
_global_p = P
_global_q = Q

print(f"Neighbor cache size: {len(_global_neighbor_cache)} nodes")
avg_degree = sum(len(v) for v in _global_neighbor_cache.values()) / len(_global_neighbor_cache)
print(f"Average node degree: {avg_degree:.1f}")

def init_worker():
    """Initialize global variables in each worker process"""
    global neighbor_cache, neighbor_set_cache, p, q
    neighbor_cache = _global_neighbor_cache
    neighbor_set_cache = _global_neighbor_set_cache
    p = _global_p
    q = _global_q

def biased_random_walk(start_node_and_length):
    """Generate a single biased random walk using global caches"""
    start_node, walk_length = start_node_and_length
    walk = [start_node]
    
    neighbors = neighbor_cache[start_node]
    if len(neighbors) == 0:
        return [str(n) for n in walk]
    
    walk.append(np.random.choice(neighbors))
    
    for _ in range(walk_length - 2):
        cur = walk[-1]
        prev = walk[-2]
        neighbors = neighbor_cache[cur]
        
        if len(neighbors) == 0:
            break
        
        prev_neighbors = neighbor_set_cache[prev]
        
        probs = np.array([
            1.0/p if n == prev else (1.0 if n in prev_neighbors else 1.0/q)
            for n in neighbors
        ], dtype=np.float32)
        
        probs = probs / probs.sum()
        next_node = np.random.choice(neighbors, p=probs)
        walk.append(next_node)
    
    return [str(n) for n in walk]

print("\n" + "="*80)
print("GENERATING BIASED RANDOM WALKS")
print("="*80)

num_workers = cpu_count()
print(f"Using {num_workers} CPU workers")
print(f"Each worker will initialize caches once (memory-efficient)")

nodes = list(G.nodes())
walk_tasks = []
np.random.seed(SEED)

for walk_iter in range(NUM_WALKS):
    shuffled_nodes = nodes.copy()
    np.random.shuffle(shuffled_nodes)
    for node in shuffled_nodes:
        walk_tasks.append((node, WALK_LENGTH))

print(f"Generating {len(walk_tasks)} walks...")
print(f"Estimated time: ~{len(walk_tasks) * WALK_LENGTH / (num_workers * 50000):.1f} hours")
print(f"Saving walks incrementally every {SAVE_INTERVAL} walks")

os.makedirs('temp_walks', exist_ok=True)

start_time = time.time()
walks_completed = 0
total_walks = len(walk_tasks)
last_report_time = start_time
chunk_walks = []
chunk_number = 0

print(f"\n{'='*80}")
print(f"{'Completed':<12} | {'Progress':<12} | {'Rate':<15} | {'Elapsed':<12} | {'Remaining':<12}")
print(f"{'-'*12} | {'-'*12} | {'-'*15} | {'-'*12} | {'-'*12}")

with Pool(num_workers, initializer=init_worker) as pool:
    for walk in pool.imap(biased_random_walk, walk_tasks, chunksize=100):
        chunk_walks.append(walk)
        walks_completed += 1
        
        if len(chunk_walks) >= SAVE_INTERVAL:
            with open(f'temp_walks/data70_{chunk_number}.pkl', 'wb') as f:
                pickle.dump(chunk_walks, f)
            chunk_number += 1
            chunk_walks = []
        
        if walks_completed % PROGRESS_INTERVAL == 0 or walks_completed == total_walks:
            current_time = time.time()
            elapsed = current_time - start_time
            
            pct = walks_completed / total_walks * 100
            overall_rate = walks_completed / elapsed if elapsed > 0 else 0
            remaining = (total_walks - walks_completed) / overall_rate if overall_rate > 0 else 0
            
            print(f"{walks_completed:<12,} | {pct:>5.1f}%      | {overall_rate:>6.0f} walks/s | "
                  f"{elapsed/60:>5.1f} min   | {remaining/3600:>5.2f} hrs")
            
            last_report_time = current_time

if len(chunk_walks) > 0:
    with open(f'temp_walks/data70_{chunk_number}.pkl', 'wb') as f:
        pickle.dump(chunk_walks, f)
    chunk_number += 1

print(f"{'='*80}")

elapsed = time.time() - start_time
print(f"\nGenerated {walks_completed} walks in {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
print(f"Saved in {chunk_number} chunks")

print("\nLoading and combining walk chunks...")
all_walks = []
for i in range(chunk_number):
    with open(f'temp_walks/data70_{i}.pkl', 'rb') as f:
        chunk = pickle.load(f)
        all_walks.extend(chunk)
    os.remove(f'temp_walks/data70_{i}.pkl')
os.rmdir('temp_walks')

print(f"Loaded {len(all_walks)} walks")

with open('data60.pkl', 'wb') as f:
    pickle.dump(all_walks, f)
print("Saved: random_walks.pkl")

del _global_neighbor_cache, _global_neighbor_set_cache

print("\n" + "="*80)
print("TRAINING WORD2VEC ON WALKS")
print("="*80)

start_time = time.time()
print(f"Training Word2Vec (vector_size={CODE_EMBEDDING_DIM}, window=10, epochs=10)...")
w2v_model = Word2Vec(
    all_walks, 
    vector_size=CODE_EMBEDDING_DIM, 
    window=10,
    min_count=1, 
    sg=1, 
    workers=num_workers, 
    epochs=10, 
    negative=5, 
    ns_exponent=0.75,
    seed=SEED
)

code_embeddings = {}
for node in G.nodes():
    code_embeddings[node] = w2v_model.wv[str(node)]

print(f"Trained embeddings for {len(code_embeddings)} procedure codes")
print(f"Time: {time.time() - start_time:.1f}s")

with open('data11.pkl', 'wb') as f:
    pickle.dump(code_embeddings, f)
print("Saved: code_embeddings.pkl")

del all_walks, G, cooccurrence_matrix
import gc
gc.collect()

print("\n" + "="*80)
print("SAVING ADDITIONAL ARTIFACTS")
print("="*80)

np.save('all_specialty_codes.npy', np.array(all_specialty_codes))
print("Saved: all_specialty_codes.npy")

save_npz('data47.npz', proc_matrix_filtered)
print(f"Saved: proc_matrix_filtered.npz (shape: {proc_matrix_filtered.shape})")

metadata = {
    'code_embedding_dim': CODE_EMBEDDING_DIM,
    'num_codes': len(code_embeddings),
    'num_providers': len(all_pins),
    'walk_length': WALK_LENGTH,
    'num_walks': NUM_WALKS,
    'p': P,
    'q': Q,
    'seed': SEED,
    'test_mode': TEST_MODE,
    'num_workers': num_workers
}

with open('data40.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("Saved: notebook1_metadata.pkl")

print("\n" + "="*80)
print("NOTEBOOK 1 COMPLETE!")
print("="*80)
print("\nOutputs:")
print("  1. code_embeddings.pkl - Code embeddings from Word2Vec")
print("  2. random_walks.pkl - Frozen random walks (reproducible)")
print("  3. all_specialty_codes.npy - Code index mapping")
print("  4. proc_matrix_filtered.npz - Provider × Code claim counts")
print("  5. notebook1_metadata.pkl - Pipeline metadata")
print(f"\nCode embeddings: {len(code_embeddings)} codes × {CODE_EMBEDDING_DIM} dimensions")
print(f"Procedure matrix: {proc_matrix_filtered.shape[0]} providers × {proc_matrix_filtered.shape[1]} codes")
print(f"\nMemory optimizations applied:")
print(f"  - Global cache per worker (not copied per task)")
print(f"  - Incremental saving every {SAVE_INTERVAL} walks")
print(f"  - Used {num_workers} workers efficiently")
print("\nNext step: Run notebook_2_gat_training.py")
