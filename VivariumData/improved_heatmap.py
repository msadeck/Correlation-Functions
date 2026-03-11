import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmaps_improved(frames, corr_matrices, timepoints=[0, 25, 50], save_path=None):
    """
    Plot correlation matrix heatmaps at specific timepoints with improved formatting.
    
    Parameters:
    -----------
    frames : array-like
        All time points
    corr_matrices : ndarray
        Correlation matrices with shape (n_frames, 4, 4)
    timepoints : list
        Specific timepoints to visualize (will find nearest available)
    save_path : str, optional
        Path to save the figure
    """
    labels = ["T-cell\nPD1-", "T-cell\nPD1+", "Tumor\nPDL1-", "Tumor\nPDL1+"]
    
    n_plots = len(timepoints)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, t in enumerate(timepoints):
        # Find nearest frame
        frame_idx = np.argmin(np.abs(np.array(frames) - t))
        actual_time = frames[frame_idx]
        
        corr = corr_matrices[frame_idx]
        
        # Create heatmap with better formatting
        im = sns.heatmap(
            corr, 
            annot=True,           # Show numbers
            fmt='.2f',            # 2 decimal places
            cmap='RdBu_r',        # Red-Blue reversed colormap
            center=1.0,           # Center colormap at 1.0
            vmin=0,               # Minimum value
            vmax=2.5,             # Maximum value (adjust if needed)
            cbar_kws={
                'label': 'Correlation Coefficient',
                'shrink': 0.8
            },
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx],
            square=True,          # Make cells square
            linewidths=2,         # Thicker grid lines
            linecolor='white',    # White grid lines
            annot_kws={'size': 12, 'weight': 'bold'}  # Bold annotation text
        )
        
        # Improve title
        axes[idx].set_title(f"Time = {actual_time:.1f}", fontsize=16, pad=15, weight='bold')
        
        # Rotate x-axis labels for better readability
        axes[idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        axes[idx].set_yticklabels(labels, rotation=0, fontsize=11)
        
        # Remove axis labels (they're redundant with tick labels)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_single_heatmap_large(frames, corr_matrices, timepoint=0, save_path=None):
    """
    Plot a single large correlation heatmap with excellent formatting.
    
    Parameters:
    -----------
    frames : array-like
        All time points
    corr_matrices : ndarray
        Correlation matrices with shape (n_frames, 4, 4)
    timepoint : float
        Timepoint to visualize
    save_path : str, optional
        Path to save the figure
    """
    labels = ["T-cell PD1-", "T-cell PD1+", "Tumor PDL1-", "Tumor PDL1+"]
    
    # Find nearest frame
    frame_idx = np.argmin(np.abs(np.array(frames) - timepoint))
    actual_time = frames[frame_idx]
    
    corr = corr_matrices[frame_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = sns.heatmap(
        corr,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=1.0,
        vmin=0,
        vmax=2.5,
        cbar_kws={
            'label': 'Correlation Coefficient',
            'shrink': 0.8,
            'pad': 0.02
        },
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
        linewidths=3,
        linecolor='white',
        annot_kws={'size': 16, 'weight': 'bold'}
    )
    
    # Styling
    ax.set_title(f"Spatial Correlations at Time = {actual_time:.1f}", 
                 fontsize=20, pad=20, weight='bold')
    
    # Better tick labels
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(labels, rotation=0, fontsize=14)
    
    # Add colorbar label styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Correlation Coefficient', size=14, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# Example usage:
if __name__ == "__main__":
    print("Improved heatmap functions loaded!")
    print("\nUsage examples:")
    print("\n1. Multiple timepoints:")
    print("   fig, axes = plot_correlation_heatmaps_improved(frames, corr_matrices, timepoints=[0, 26, 52])")
    print("   plt.show()")
    print("\n2. Single large heatmap:")
    print("   fig, ax = plot_single_heatmap_large(frames, corr_matrices, timepoint=26)")
    print("   plt.show()")
