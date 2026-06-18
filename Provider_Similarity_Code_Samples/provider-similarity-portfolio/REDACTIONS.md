# Redactions Log

Every change applied to sanitize these files for external sharing. Logic and structure are unchanged; only proprietary identifiers were replaced.

## Global
- A sanitization header was prepended to every file.
- All file I/O references generic local artifact names (e.g. `procedure_df.parquet`, `pin_to_label.pkl`, `procedure_vectors.npz`) — no warehouse paths, project IDs, URLs, or credentials were present in the source.

## Per-file replacements
- 2_model_pipeline/03_provider_embedding_diagnosis.py: 1x  dummy PIN -> labeled placeholder
- 2_model_pipeline/03_provider_embedding_diagnosis.py: 1x  dummy PIN -> labeled placeholder
- 2_model_pipeline/03_provider_embedding_diagnosis.py: 1x  dummy PIN -> labeled placeholder
- 2_model_pipeline/07_generate_top10_alternatives.py: 1x  CVS-internal column name -> generic
- 4_validation_and_visualization/validate_attention_weights.py: 1x  dummy PIN -> labeled placeholder
- 4_validation_and_visualization/validate_attention_weights.py: 1x  dummy PIN -> labeled placeholder
- 4_validation_and_visualization/validate_attention_weights.py: 1x  dummy PIN -> labeled placeholder
- 4_validation_and_visualization/validate_county_coverage.py: 1x  CVS-internal column name -> generic
- 4_validation_and_visualization/validate_county_coverage.py: 1x  abbreviated column -> generic
- 4_validation_and_visualization/validate_county_coverage.py: 1x  real specialty code -> placeholder
- 4_validation_and_visualization/validate_threshold_gmm_silhouette.py: 1x  CVS-internal column name -> generic

## Reviewer note
This scrub targeted: internal column names (`srv_spclty_ctg_cd`, `state_postal_cd`), a hardcoded specialty code (`'WHO'`), and example provider IDs. Please do a final read for any business logic, metric values, or comments you consider sensitive before sharing.


---

# Data-file Renaming

Every reference to a data artifact (`.parquet`, `.pkl`, `.npz`, `.csv`, `.json`) was renamed to a generic `dataN` name. The mapping is applied consistently across all files, so the pipeline still reads coherently (whatever one stage writes, the next reads). Runtime variables inside filenames (e.g. `{PIN_A}`, `{PRIMARY_SPECIALTY_FILTER}`) and file extensions are preserved.

Total references rewritten: 151
Unique data files mapped: 78

| original | renamed |
|---|---|
| `aligned_pins.pkl` | `data1.pkl` |
| `all_codes_with_rarity.parquet` | `data2.parquet` |
| `all_pin_names.parquet` | `data3.parquet` |
| `all_providers_top10_alternatives_gat_county.csv` | `data4.csv` |
| `all_providers_top10_alternatives_me2vec.csv` | `data5.csv` |
| `all_providers_top10_alternatives_me2vec_county.csv` | `data6.csv` |
| `all_providers_top10_alternatives_me2vec_county_WITH_GAT_METRICS.csv` | `data7.csv` |
| `all_providers_top10_alternatives_with_prototype.csv` | `data8.csv` |
| `attention_cache.pkl` | `data9.pkl` |
| `code_desc_df.parquet` | `data10.parquet` |
| `code_embeddings.pkl` | `data11.pkl` |
| `concordance_results.csv` | `data12.csv` |
| `cooccurrence_matrix.npz` | `data13.npz` |
| `cost_df.parquet` | `data14.parquet` |
| `county_df.parquet` | `data15.parquet` |
| `county_level_alternatives_{}_threshold_{}.csv` | `data16.csv` |
| `demo_df.parquet` | `data17.parquet` |
| `diagnosis_PIN_{}_{}_{}.csv` | `data18.csv` |
| `diagnosis_df.parquet` | `data19.parquet` |
| `final_gat_all_towers.parquet` | `data20.parquet` |
| `final_gat_all_towers_metadata.pkl` | `data21.pkl` |
| `final_me2vec_all_towers_1046d.parquet` | `data22.parquet` |
| `final_me2vec_all_towers_metadata.pkl` | `data23.pkl` |
| `gat_analysis_summary_{}_{}_{}.csv` | `data24.csv` |
| `gat_attention_weights_{}_{}_{}.json` | `data25.json` |
| `gat_cnp_prototypes.pkl` | `data26.pkl` |
| `gat_hybrid_model_metadata.pkl` | `data27.pkl` |
| `gat_provider_to_prototype.pkl` | `data28.pkl` |
| `graph_autoencoder_embeddings.csv` | `data29.csv` |
| `idx_to_pin.pkl` | `data30.pkl` |
| `jmvae_proc_diag_embeddings.csv` | `data31.csv` |
| `latent_dim_comparison_full.csv` | `data32.csv` |
| `latent_dim_comparison_summary.csv` | `data33.csv` |
| `me2vec_linear_towers_scalers.pkl` | `data34.pkl` |
| `me2vec_metadata.pkl` | `data35.pkl` |
| `me2vec_provider_embeddings.csv` | `data36.csv` |
| `me2vec_provider_embeddings.parquet` | `data37.parquet` |
| `me2vec_provider_embeddings_diagnosis.parquet` | `data38.parquet` |
| `model_comparison_summary.csv` | `data39.csv` |
| `notebook1_metadata.pkl` | `data40.pkl` |
| `pair_labels.parquet` | `data41.parquet` |
| `pin_df.parquet` | `data42.parquet` |
| `pin_to_idx.pkl` | `data43.pkl` |
| `pin_to_label.pkl` | `data44.pkl` |
| `place_df.parquet` | `data45.parquet` |
| `proc_grouped.parquet` | `data46.parquet` |
| `proc_matrix_filtered.npz` | `data47.npz` |
| `procedure_df.parquet` | `data48.parquet` |
| `procedure_vectors.npz` | `data49.npz` |
| `prov_spl.parquet` | `data50.parquet` |
| `provider_code_embeddings.pkl` | `data51.pkl` |
| `provider_embeddings_classifier.parquet` | `data52.parquet` |
| `provider_embeddings_df.csv` | `data53.csv` |
| `provider_embeddings_df.pkl` | `data54.pkl` |
| `provider_embeddings_final.pkl` | `data55.pkl` |
| `provider_embeddings_multi_tower_vae.parquet` | `data56.parquet` |
| `provider_level_alternatives_{}_threshold_{}.csv` | `data57.csv` |
| `provider_pairs_with_rankings.csv` | `data58.csv` |
| `provider_specialty_labels.pkl` | `data59.pkl` |
| `random_walks.pkl` | `data60.pkl` |
| `rare_codes.parquet` | `data61.parquet` |
| `sampled_pairs_with_scores.parquet` | `data62.parquet` |
| `silhouette_results.csv` | `data63.csv` |
| `similarity_matrix.npz` | `data64.npz` |
| `specialty_category_combined_matrix_me2vec.csv` | `data65.csv` |
| `specialty_category_count_matrix_me2vec.csv` | `data66.csv` |
| `specialty_category_percentage_matrix_me2vec.csv` | `data67.csv` |
| `specialty_code_mappings.pkl` | `data68.pkl` |
| `state_level_alternatives_{}_threshold_{}.csv` | `data69.csv` |
| `temp_walks/chunk_{}.pkl` | `data70.pkl` |
| `threshold_recommendation.pkl` | `data71.pkl` |
| `tower_weights_by_label.csv` | `data72.csv` |
| `tower_weights_comparison.csv` | `data73.csv` |
| `training_history.pkl` | `data74.pkl` |
| `training_results.pkl` | `data75.pkl` |
| `treatment_embeddings_step1.csv` | `data76.csv` |
| `vgae_embeddings.csv` | `data77.csv` |
| `{model_dir}/final_pipeline_metadata.pkl` | `data78.pkl` |
