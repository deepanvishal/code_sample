# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

# ============================================================================
# STEP 13: SIMPLE VISUALIZATION
# ============================================================================
print("\nSTEP 13: Creating visualization...")

# Filter to labeled data only for visualization
labeled_embeddings_df = embeddings_df.merge(
    pd.DataFrame({'PIN': list(pin_to_label.keys()), 'true_label': list(pin_to_label.values())}), 
    on='PIN', 
    how='inner'
)

print(f"Visualizing {len(labeled_embeddings_df)} labeled hospitals")

# Extract data for visualization
embedding_columns = [col for col in labeled_embeddings_df.columns if col.startswith('emb_')]
embeddings_matrix = labeled_embeddings_df[embedding_columns].values.astype(np.float32)
pins = labeled_embeddings_df['PIN'].values
labels_array = labeled_embeddings_df['true_label'].values

# Apply t-SNE
print("Applying t-SNE...")
from sklearn.manifold import TSNE
n_samples = len(embeddings_matrix)
perplexity = min(30, max(5, n_samples // 4))
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tsne_embedding = tsne.fit_transform(embeddings_matrix)

# Create visualization
unique_labels = np.unique(labels_array)
n_categories = len(unique_labels)

if n_categories <= 10:
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
elif n_categories <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
else:
    colors = plt.cm.hsv(np.linspace(0, 1, n_categories))

color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
point_colors = [color_map[label] for label in labels_array]

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.suptitle('Hospital Embeddings Visualization - JMVAE', fontsize=16, fontweight='bold')

# t-SNE Plot
ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=point_colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.grid(True, alpha=0.3)

# Add cluster center annotations
print("Adding cluster center annotations...")
for label in unique_labels:
    label_mask = labels_array == label
    if label_mask.sum() > 1:
        tsne_center_x = tsne_embedding[label_mask, 0].mean()
        tsne_center_y = tsne_embedding[label_mask, 1].mean()
        ax.annotate(label, 
                    (tsne_center_x, tsne_center_y),
                    fontsize=10, 
                    fontweight='bold',
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor='white', 
                             alpha=0.8,
                             edgecolor='black'))

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_map[label], label=label) for label in unique_labels]
fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, title='Hospital Types')

plt.tight_layout()
plt.subplots_adjust(right=0.88)
plt.savefig('jmvae_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved: jmvae_visualization.png")
