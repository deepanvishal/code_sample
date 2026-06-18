# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import gc

def create_label_separation_analysis():
    """Analyze separation between all label pairs using centroid differences."""
    
    print("Creating label centroid separation analysis...")
    
    # Step 1: Calculate centroids for each label
    all_labels = sorted(df['label'].unique())
    label_centroids = {}
    
    print(f"Calculating centroids for {len(all_labels)} labels...")
    for label in all_labels:
        mask = df['label'] == label
        provider_count = mask.sum()
        label_centroids[label] = embeddings[mask.values].mean(axis=0)
        print(f"  {label}: {provider_count} providers")
    
    # Step 2: Calculate pairwise differences (absolute differences)
    print("\nCalculating pairwise centroid differences...")
    
    separation_data = []
    difference_matrices = {}
    
    for label_a, label_b in combinations(all_labels, 2):
        centroid_a = label_centroids[label_a]
        centroid_b = label_centroids[label_b]
        
        # Calculate absolute difference across all 64 dimensions
        diff_vector = np.abs(centroid_a - centroid_b)
        
        # Summary statistics
        mean_diff = diff_vector.mean()
        max_diff = diff_vector.max()
        min_diff = diff_vector.min()
        std_diff = diff_vector.std()
        
        separation_data.append({
            'Label_A': label_a,
            'Label_B': label_b,
            'Pair': f"{label_a} vs {label_b}",
            'Mean_Difference': mean_diff,
            'Max_Difference': max_diff,
            'Min_Difference': min_diff,
            'Std_Difference': std_diff,
            'Difference_Vector': diff_vector
        })
        
        # Store for visualization
        difference_matrices[f"{label_a}_vs_{label_b}"] = diff_vector
    
    # Create separation summary dataframe
    separation_df = pd.DataFrame(separation_data)
    separation_df = separation_df.sort_values('Mean_Difference', ascending=False)
    
    print(f"✅ Calculated {len(separation_df)} label pair comparisons")
    
    return separation_df, difference_matrices, label_centroids

def plot_separation_analysis(separation_df, difference_matrices, top_n=10, bottom_n=10):
    """Create visualization of label separation patterns."""
    
    print(f"Creating separation visualizations...")
    
    # Get top and bottom separated pairs
    top_separated = separation_df.head(top_n)
    bottom_separated = separation_df.tail(bottom_n)
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    
    # Plot 1: Overall separation ranking
    ax1 = axes[0]
    y_pos = range(len(separation_df))
    ax1.barh(y_pos, separation_df['Mean_Difference'], alpha=0.7)
    ax1.set_yticks(y_pos[::2])  # Show every other label to avoid crowding
    ax1.set_yticklabels(separation_df['Pair'].iloc[::2], fontsize=8)
    ax1.set_xlabel('Mean Absolute Difference Across 64 Dimensions')
    ax1.set_title('Label Pair Separation Ranking (Higher = Better Separation)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top separated pairs - dimension-wise differences
    ax2 = axes[1]
    x_positions = range(64)
    
    for i, (_, row) in enumerate(top_separated.head(5).iterrows()):
        diff_vector = row['Difference_Vector']
        ax2.plot(x_positions, diff_vector, marker='o', markersize=2, 
                linewidth=1, alpha=0.8, label=f"{row['Pair'][:20]}...")
    
    ax2.set_xlabel('Embedding Dimension (0-63)')
    ax2.set_ylabel('Absolute Difference')
    ax2.set_title('Top 5 Well-Separated Label Pairs - Dimension-wise Differences')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bottom separated pairs - dimension-wise differences  
    ax3 = axes[2]
    
    for i, (_, row) in enumerate(bottom_separated.head(5).iterrows()):
        diff_vector = row['Difference_Vector']
        ax3.plot(x_positions, diff_vector, marker='o', markersize=2,
                linewidth=1, alpha=0.8, label=f"{row['Pair'][:20]}...")
    
    ax3.set_xlabel('Embedding Dimension (0-63)')
    ax3.set_ylabel('Absolute Difference')
    ax3.set_title('Top 5 Poorly-Separated Label Pairs - Dimension-wise Differences')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of separation matrix
    ax4 = axes[3]
    
    # Create separation matrix
    all_labels = sorted(separation_df['Label_A'].unique().tolist() + separation_df['Label_B'].unique().tolist())
    separation_matrix = np.zeros((len(all_labels), len(all_labels)))
    
    for _, row in separation_df.iterrows():
        idx_a = all_labels.index(row['Label_A'])
        idx_b = all_labels.index(row['Label_B'])
        separation_matrix[idx_a, idx_b] = row['Mean_Difference']
        separation_matrix[idx_b, idx_a] = row['Mean_Difference']  # Symmetric
    
    sns.heatmap(separation_matrix, 
                xticklabels=all_labels, 
                yticklabels=all_labels,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                ax=ax4,
                cbar_kws={'label': 'Mean Separation'})
    ax4.set_title('Label Separation Heatmap (Higher = Better Separation)')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax4.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # Memory cleanup
    plt.close()
    gc.collect()

def create_individual_pair_plots(separation_df, difference_matrices, pairs_to_plot=None):
    """Create individual bar plots for specific label pairs."""
    
    if pairs_to_plot is None:
        # Default to top 6 and bottom 6
        top_pairs = separation_df.head(3)['Pair'].tolist()
        bottom_pairs = separation_df.tail(3)['Pair'].tolist()
        pairs_to_plot = top_pairs + bottom_pairs
    
    n_pairs = len(pairs_to_plot)
    cols = 3
    rows = (n_pairs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    x_positions = range(64)
    
    for i, pair_name in enumerate(pairs_to_plot):
        if i >= len(axes):
            break
            
        # Find the difference vector
        pair_key = pair_name.replace(' vs ', '_vs_')
        if pair_key in difference_matrices:
            diff_vector = difference_matrices[pair_key]
        else:
            # Try reverse order
            labels = pair_name.split(' vs ')
            pair_key = f"{labels[1]}_vs_{labels[0]}"
            diff_vector = difference_matrices.get(pair_key, np.zeros(64))
        
        # Get separation score
        separation_score = separation_df[separation_df['Pair'] == pair_name]['Mean_Difference'].iloc[0]
        
        axes[i].bar(x_positions, diff_vector, alpha=0.7, color='steelblue')
        axes[i].set_xlabel('Embedding Dimension')
        axes[i].set_ylabel('Absolute Difference')
        axes[i].set_title(f'{pair_name}\nSeparation Score: {separation_score:.4f}')
        axes[i].grid(True, alpha=0.3)
        
        # Highlight dimensions with highest differences
        top_dims = np.argsort(diff_vector)[-5:]  # Top 5 dimensions
        for dim in top_dims:
            axes[i].bar(dim, diff_vector[dim], color='red', alpha=0.8)
    
    # Hide unused subplots
    for i in range(len(pairs_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    gc.collect()

# Execute the analysis
separation_df, difference_matrices, label_centroids = create_label_separation_analysis()

print(f"\n📊 LABEL SEPARATION ANALYSIS RESULTS:")
print(f"Total label pairs analyzed: {len(separation_df)}")

print(f"\n🏆 TOP 5 BEST SEPARATED LABEL PAIRS:")
print(separation_df[['Pair', 'Mean_Difference']].head().to_string(index=False))

print(f"\n⚠️ TOP 5 POORLY SEPARATED LABEL PAIRS:")
print(separation_df[['Pair', 'Mean_Difference']].tail().to_string(index=False))

print(f"\n📈 Creating visualizations...")
plot_separation_analysis(separation_df, difference_matrices)

print(f"\n📊 Creating individual pair analysis...")
create_individual_pair_plots(separation_df, difference_matrices)

print(f"\n✅ Analysis complete!")
print(f"Access results via:")
print(f"- separation_df: Summary statistics for all label pairs")
print(f"- difference_matrices: 64-dimensional difference vectors")
print(f"- label_centroids: Centroid embeddings for each label")