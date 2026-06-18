# ---------------------------------------------------------------------------
# Portfolio code sample. Sanitized for external sharing:
# proprietary table/column names and identifiers have been replaced with
# generic placeholders. Reads local intermediate artifacts (parquet/pkl/npz);
# not runnable against any production warehouse. Logic is unchanged.
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gc

def create_enhanced_overlaps_analysis():
    """Create enhanced overlaps analysis with comparable providers and all label similarities."""
    
    print("Creating enhanced overlaps analysis...")
    
    # Start with existing overlaps_df
    enhanced_analysis = overlaps_df.copy()
    
    # Get all unique labels
    all_labels = df['label'].unique()
    print(f"Analyzing against {len(all_labels)} labels: {sorted(all_labels)}")
    
    # Step 1: Add comparable Provider B (closest to overlap label centroid)
    print("Step 1: Finding comparable providers...")
    
    # Calculate centroids for all labels
    label_centroids = {}
    for label in all_labels:
        mask = df['label'] == label
        label_centroids[label] = embeddings[mask.values].mean(axis=0)
    
    # Find closest provider to each label's centroid
    closest_providers = {}
    for label in all_labels:
        mask = df['label'] == label
        label_embeddings = embeddings[mask.values]
        label_pins = df[mask]['PIN'].values
        
        # Calculate distances to centroid
        centroid = label_centroids[label]
        similarities = cosine_similarity(label_embeddings, centroid.reshape(1, -1)).flatten()
        
        # Find closest provider (highest similarity)
        closest_idx = similarities.argmax()
        closest_pin = label_pins[closest_idx]
        closest_name = new_scores_with_names[new_scores_with_names['PIN'] == closest_pin]['PIN_name'].iloc[0]
        
        closest_providers[label] = {
            'pin': closest_pin,
            'name': closest_name,
            'similarity': similarities[closest_idx]
        }
    
    # Add comparable provider columns
    enhanced_analysis['Provider_B_PIN'] = enhanced_analysis['overlap_label'].map(
        lambda x: closest_providers[x]['pin']
    )
    enhanced_analysis['Provider_B_Name'] = enhanced_analysis['overlap_label'].map(
        lambda x: closest_providers[x]['name']
    )
    
    # Step 2: Calculate Provider A to Provider B similarity
    print("Step 2: Calculating Provider A to Provider B similarities...")
    
    def calculate_provider_similarity(row):
        # Get embeddings for Provider A and Provider B
        provider_a_idx = df[df['PIN'] == row['outlier_pin']].index[0]
        provider_b_idx = df[df['PIN'] == row['Provider_B_PIN']].index[0]
        
        emb_a = embeddings[provider_a_idx - df.index[0]]
        emb_b = embeddings[provider_b_idx - df.index[0]]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([emb_a], [emb_b])[0][0]
        return similarity
    
    enhanced_analysis['Provider_A_to_B_Similarity'] = enhanced_analysis.apply(
        calculate_provider_similarity, axis=1
    )
    
    # Step 3: Calculate Provider A similarity to ALL label centroids
    print("Step 3: Calculating similarities to all label centroids...")
    
    # For each label, calculate all providers' similarities to that label's centroid
    label_similarity_distributions = {}
    
    for label in all_labels:
        print(f"  Processing {label} similarity distribution...")
        
        # Get all providers (regardless of their assigned label)
        all_provider_similarities = []
        
        for _, provider_row in df.iterrows():
            provider_emb = embeddings[provider_row.name - df.index[0]]
            centroid = label_centroids[label]
            similarity = cosine_similarity([provider_emb], [centroid.reshape(1, -1)])[0][0]
            all_provider_similarities.append(similarity)
        
        label_similarity_distributions[label] = np.array(all_provider_similarities)
    
    # Step 4: Calculate similarities and percentiles for each overlapping provider
    print("Step 4: Calculating Provider A similarities and percentiles for all labels...")
    
    for label in all_labels:
        print(f"  Processing {label} metrics...")
        
        # Calculate similarity column
        similarities = []
        percentiles = []
        
        for _, row in enhanced_analysis.iterrows():
            # Get Provider A embedding
            provider_a_idx = df[df['PIN'] == row['outlier_pin']].index[0]
            provider_a_emb = embeddings[provider_a_idx - df.index[0]]
            
            # Calculate similarity to this label's centroid
            centroid = label_centroids[label]
            similarity = cosine_similarity([provider_a_emb], [centroid.reshape(1, -1)])[0][0]
            similarities.append(similarity)
            
            # Calculate percentile within this label's distribution
            distribution = label_similarity_distributions[label]
            percentile = (similarity > distribution).mean() * 100
            percentiles.append(percentile)
        
        # Add columns for this label
        label_clean = label.replace(' ', '_').replace('-', '_')
        enhanced_analysis[f'{label_clean}_Similarity'] = np.round(similarities, 4)
        enhanced_analysis[f'{label_clean}_Percentile'] = np.round(percentiles, 1)
    
    # Step 5: Add provider names for readability
    enhanced_analysis = enhanced_analysis.merge(
        new_scores_with_names[['PIN', 'PIN_name']], 
        left_on='outlier_pin', right_on='PIN', how='left'
    ).drop('PIN', axis=1)
    
    # Reorder columns for better readability
    base_cols = [
        'outlier_pin', 'PIN_name', 'outlier_label', 'overlap_label',
        'Provider_B_PIN', 'Provider_B_Name', 'Provider_A_to_B_Similarity',
        'own_distance', 'other_distance', 'overlap_score'
    ]
    
    # Add all label similarity and percentile columns
    label_cols = []
    for label in sorted(all_labels):
        label_clean = label.replace(' ', '_').replace('-', '_')
        label_cols.extend([f'{label_clean}_Similarity', f'{label_clean}_Percentile'])
    
    enhanced_analysis = enhanced_analysis[base_cols + label_cols]
    
    print(f"✅ Enhanced overlaps analysis complete!")
    print(f"Shape: {enhanced_analysis.shape}")
    print(f"Columns: {len(enhanced_analysis.columns)}")
    
    # Memory cleanup
    del label_similarity_distributions
    gc.collect()
    
    return enhanced_analysis

# Execute the analysis
enhanced_overlaps_analysis = create_enhanced_overlaps_analysis()

# Display summary
print(f"\n📊 ENHANCED OVERLAPS ANALYSIS SUMMARY:")
print(f"Total overlapping providers: {len(enhanced_overlaps_analysis)}")
print(f"Total columns: {len(enhanced_overlaps_analysis.columns)}")
print(f"\nSample of Provider A to Provider B similarities:")
print(enhanced_overlaps_analysis[['PIN_name', 'outlier_label', 'Provider_B_Name', 'overlap_label', 'Provider_A_to_B_Similarity']].head())

print(f"\n🎯 KEY INSIGHTS:")
print("- Each row represents one overlapping provider")
print("- Provider_A_to_B_Similarity shows how similar the outlier is to the typical provider in the overlap label")
print("- For each of the 15 labels, you get similarity score and percentile ranking")
print("- Percentile shows how the outlier ranks among ALL providers when compared to that label's centroid")

print(f"\n📋 Access the data:")
print("enhanced_overlaps_analysis.head()")
print("enhanced_overlaps_analysis.columns.tolist()")
