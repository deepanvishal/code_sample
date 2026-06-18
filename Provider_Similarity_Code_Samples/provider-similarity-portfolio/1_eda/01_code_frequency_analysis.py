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

def analyze_code_frequencies(df, code_col='code', claims_col='claims'):
    """
    Analyze code frequencies to identify low-frequency outliers
    Shows distribution, thresholds, and code counts at each threshold
    
    Args:
        df: DataFrame with two columns (already aggregated):
            - code_col: Code (procedure/diagnosis)
            - claims_col: Total number of claims for this code
    
    Returns:
        Dictionary with analysis results
    """
    
    print("Analyzing code frequencies...")
    
    # Use data as-is (already aggregated)
    code_stats = df.copy()
    code_stats = code_stats[[code_col, claims_col]].copy()
    code_stats.columns = ['code', 'total_claims']
    
    print(f"\nTotal unique codes: {len(code_stats):,}")
    print(f"Total claims: {code_stats['total_claims'].sum():,}")
    print(f"Mean claims per code: {code_stats['total_claims'].mean():.2f}")
    print(f"Median claims per code: {code_stats['total_claims'].median():.2f}")
    
    # Calculate IQR statistics
    Q1 = code_stats['total_claims'].quantile(0.25)
    Q2 = code_stats['total_claims'].quantile(0.50)
    Q3 = code_stats['total_claims'].quantile(0.75)
    IQR = Q3 - Q1
    lower_outlier = max(1, Q1 - 1.5 * IQR)
    
    thresholds = {
        'min': code_stats['total_claims'].min(),
        'Q1': Q1,
        'Q2': Q2,
        'Q3': Q3,
        'max': code_stats['total_claims'].max(),
        'IQR': IQR,
        'lower_outlier': lower_outlier
    }
    
    print("\n" + "="*80)
    print("IQR ANALYSIS - CODE FREQUENCY THRESHOLDS")
    print("="*80)
    print(f"  Min claims: {thresholds['min']:.0f}")
    print(f"  Q1 (25th percentile): {Q1:.0f}")
    print(f"  Median (50th percentile): {Q2:.0f}")
    print(f"  Q3 (75th percentile): {Q3:.0f}")
    print(f"  Max claims: {thresholds['max']:.0f}")
    print(f"  IQR: {IQR:.0f}")
    print(f"  Lower outlier threshold (Q1-1.5*IQR): {lower_outlier:.0f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Code Frequency Distribution Analysis', fontsize=16, fontweight='bold')
    
    data = code_stats['total_claims']
    
    # Plot 1: Original distribution
    ax1 = axes[0, 0]
    ax1.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(Q1, color='orange', linestyle='--', linewidth=2, label=f'Q1: {Q1:.0f}')
    ax1.axvline(Q2, color='green', linestyle='--', linewidth=2, label=f'Median: {Q2:.0f}')
    ax1.axvline(Q3, color='red', linestyle='--', linewidth=2, label=f'Q3: {Q3:.0f}')
    ax1.axvline(lower_outlier, color='darkred', linestyle=':', linewidth=2, 
               label=f'Lower: {lower_outlier:.0f}')
    ax1.set_xlabel('Total Claims per Code', fontsize=11)
    ax1.set_ylabel('Number of Codes', fontsize=11)
    ax1.set_title('Original Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-transformed distribution
    ax2 = axes[0, 1]
    log_data = np.log10(data + 1)
    log_Q1 = np.log10(Q1 + 1)
    log_Q2 = np.log10(Q2 + 1)
    log_Q3 = np.log10(Q3 + 1)
    log_lower = np.log10(lower_outlier + 1)
    
    ax2.hist(log_data, bins=50, color='seagreen', alpha=0.7, edgecolor='black')
    ax2.axvline(log_Q1, color='orange', linestyle='--', linewidth=2, label=f'Q1: {Q1:.0f}')
    ax2.axvline(log_Q2, color='green', linestyle='--', linewidth=2, label=f'Median: {Q2:.0f}')
    ax2.axvline(log_Q3, color='red', linestyle='--', linewidth=2, label=f'Q3: {Q3:.0f}')
    ax2.axvline(log_lower, color='darkred', linestyle=':', linewidth=2, 
               label=f'Lower: {lower_outlier:.0f}')
    ax2.set_xlabel('Log10(Claims + 1)', fontsize=11)
    ax2.set_ylabel('Number of Codes', fontsize=11)
    ax2.set_title('Log-Transformed Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    ax3 = axes[1, 0]
    box_data = ax3.boxplot([data], vert=True, patch_artist=True, widths=0.6)
    for patch in box_data['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax3.axhline(Q1, color='orange', linestyle='--', linewidth=2, label=f'Q1: {Q1:.0f}')
    ax3.axhline(Q2, color='green', linestyle='--', linewidth=2, label=f'Median: {Q2:.0f}')
    ax3.axhline(Q3, color='red', linestyle='--', linewidth=2, label=f'Q3: {Q3:.0f}')
    ax3.axhline(lower_outlier, color='darkred', linestyle=':', linewidth=2, 
               label=f'Lower: {lower_outlier:.0f}')
    ax3.set_ylabel('Total Claims', fontsize=11)
    ax3.set_title('Box Plot', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticklabels([''])
    
    # Plot 4: Distribution ranges
    ax4 = axes[1, 1]
    below_lower = (code_stats['total_claims'] < lower_outlier).sum()
    lower_to_q1 = ((code_stats['total_claims'] >= lower_outlier) & (code_stats['total_claims'] < Q1)).sum()
    q1_to_q2 = ((code_stats['total_claims'] >= Q1) & (code_stats['total_claims'] < Q2)).sum()
    q2_to_q3 = ((code_stats['total_claims'] >= Q2) & (code_stats['total_claims'] < Q3)).sum()
    above_q3 = (code_stats['total_claims'] >= Q3).sum()
    
    categories = ['<Lower\nOutlier', 'Lower to\nQ1', 'Q1 to\nMedian', 'Median\nto Q3', '≥Q3']
    counts = [below_lower, lower_to_q1, q1_to_q2, q2_to_q3, above_q3]
    colors_bar = ['darkred', 'orange', 'yellow', 'lightgreen', 'green']
    
    bars = ax4.bar(range(len(categories)), counts, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.set_ylabel('Number of Codes', fontsize=11)
    ax4.set_title('Code Distribution by Range', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/len(code_stats)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Specific claim count breakdown
    print("\n" + "="*80)
    print("CLAIM COUNT BREAKDOWN")
    print("="*80)
    
    breakpoints = [1, 5, 10, 20, 50, 100]
    breakdown_data = []
    
    for i, threshold in enumerate(breakpoints):
        if i == 0:
            # Exactly 1
            codes_count = (code_stats['total_claims'] == 1).sum()
            label = "= 1"
        else:
            # Between previous and current
            prev = breakpoints[i-1]
            codes_count = ((code_stats['total_claims'] > prev) & 
                          (code_stats['total_claims'] <= threshold)).sum()
            label = f"> {prev} and ≤ {threshold}"
        
        breakdown_data.append({
            'Threshold': label,
            'Codes': codes_count,
            'Codes %': f"{codes_count/len(code_stats)*100:.1f}%"
        })
    
    # Add > 100
    codes_count = (code_stats['total_claims'] > 100).sum()
    breakdown_data.append({
        'Threshold': "> 100",
        'Codes': codes_count,
        'Codes %': f"{codes_count/len(code_stats)*100:.1f}%"
    })
    
    breakdown_df = pd.DataFrame(breakdown_data)
    print(breakdown_df.to_string(index=False))
    
    # Threshold impact analysis
    print("\n" + "="*80)
    print("THRESHOLD IMPACT ANALYSIS - CODES FILTERED OUT")
    print("="*80)
    
    threshold_options = [
        ('Lower Outlier', lower_outlier),
        ('Q1', Q1),
        ('Median', Q2),
        ('Q3', Q3),
        ('Custom: 5', 5),
        ('Custom: 10', 10),
        ('Custom: 20', 20),
        ('Custom: 50', 50),
        ('Custom: 100', 100)
    ]
    
    impact_data = []
    
    for name, threshold_val in threshold_options:
        # Codes below threshold
        codes_removed = (code_stats['total_claims'] < threshold_val).sum()
        
        impact_data.append({
            'Threshold': f"{name} (< {threshold_val:.0f})",
            'Codes Removed': codes_removed,
            'Codes Removed %': f"{codes_removed/len(code_stats)*100:.1f}%"
        })
    
    impact_df = pd.DataFrame(impact_data)
    print(impact_df.to_string(index=False))
    
    return {
        'code_stats': code_stats,
        'thresholds': thresholds,
        'breakdown': breakdown_df,
        'threshold_impact': impact_df
    }


# Usage Example:
# Input: Already aggregated data with code and total claims
# df = pd.DataFrame({
#     'code': ['99213', '99214', '70450', 'A1234', ...],
#     'claims': [1234, 5678, 2, 890, ...]
# })
# 
# # Filter for specific code type first if needed
# procedure_df = df[df['code_type'] == 'procedure']
# 
# # Run analysis
# results = analyze_code_frequencies(procedure_df, 
#                                   code_col='code', 
#                                   claims_col='claims')
