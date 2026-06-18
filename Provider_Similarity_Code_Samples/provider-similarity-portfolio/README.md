# Provider Similarity & Ranking — Code Samples

Embedding-based provider similarity and ranking. The system turns provider claims behavior into vectors, then produces a directional, density-aware similarity score used to rank substitute providers when one is removed from a network under optimization constraints.

> **Note on sharing.** These are sanitized code samples. Proprietary table/column names and identifiers have been replaced with generic placeholders (see `REDACTIONS.md`). Scripts read local intermediate artifacts (parquet/pkl/npz) produced by earlier stages and are not runnable against any production system. The intent is to show approach, structure, and engineering — not to run end to end.

## Layout

```
1_eda/                          Exploratory analysis that shaped the design
2_model_pipeline/               The pipeline that shipped (in run order)
3_models_and_losses_explored/   Alternatives evaluated and rejected
4_validation_and_visualization/ How results were checked, at several levels
```

## 1 · EDA
- `01_code_frequency_analysis.py` — claim-count distribution per code; low-frequency outliers.
- `02_rare_code_identification.py` — composite rarity score, percentile cutoff.
- `03_label_separation_analysis.py` — centroid separation between specialty labels (pre-model separability check).
- `04_provider_overlap_analysis.py` — provider-to-provider overlap and closest comparable provider per label.
- `05_latent_dimension_sweep.py` — embedding-dimension comparison (the basis for choosing 128).

## 2 · Model pipeline (run order)
1. `01_code_embeddings_node2vec.py` — code co-occurrence graph → biased random walks → Word2Vec → 128-d code embeddings.
2. `02_provider_embedding_me2vec.py` (+ `02b_procedure_encoder.py`) — aggregate a provider's code embeddings into a provider embedding (procedures tower).
3. `03_provider_embedding_diagnosis.py` — the diagnosis tower.
4. `04_precompute_training_pairs.py` — weighted-Jaccard + random pairs → weak labels for triplet training.
5. `05_combine_towers.py` — concatenate the towers with per-tower normalization (~1045-d provider vector).
6. `05b_gat_tower_training.py` — graph-attention tower training (attention-weighted aggregation; the GAT variant referenced in the write-up).
7. `06_cnp_prototype_model.py` — **the model that shipped**: learned query-dependent tower weighting + CNP (Conditional Neighbor Probability) directional similarity + real-provider prototypes (high in-degree hubs).
8. `07_generate_top10_alternatives.py` — final scoring: top-10 directional alternatives per provider.

## 3 · Models & losses explored
**Models** (one representative file per algorithm): JMVAE, hierarchical 6-tower VAE, hybrid VAE, graph autoencoder (GAE/VGAE), GraphSAGE, link-prediction treatment GNN.
**Losses**: uncertainty-weighted multi-task (Kendall 2018), similarity regression (MSE to Jaccard), and a switchable triplet/contrastive file (`loss_options_triplet_contrastive.py`, `LOSS_OPTION` selector) — batch-hard triplet is the variant that won. Pure reconstruction/KL/center losses live inside the model files above.

## 4 · Validation & visualization
- Embedding structure: `viz_tsne.py`, `viz_provider_transformation.py`.
- Model behavior: `validate_attention_weights.py`, `validate_gat_metrics.py`.
- Ranking quality (non-business): `validate_ranking_stability.py` (Kendall τ / Spearman across runs).
- Business alignment: `validate_model_vs_analytics.py`, `validate_county_coverage.py`, `validate_prototype_impact.py`.
- Cluster/threshold checking: `validate_threshold_gmm_silhouette.py` (GMM, silhouette, gap statistic).

See `REDACTIONS.md` for the full list of sanitization changes.
