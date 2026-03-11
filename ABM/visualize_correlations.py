import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns

def moving_average(data, window):
    """Apply moving average smoothing"""
    return np.convolve(data, np.ones(window)/window, mode='same')

def plot_self_correlations_vivarium(frames, corr_matrices, radius=50, window=5, save_path=None):
    """
    Plot self-correlations (diagonal elements) over time for vivarium data.

    Parameters:
    -----------
    frames : array-like
        Time points
    corr_matrices : ndarray
        Correlation matrices with shape (n_frames, 4, 4)
    radius : float
        Neighborhood radius used in correlation calculation
    window : int
        Window size for moving average smoothing
    save_path : str, optional
        Path to save the figure
    """
    # Vivarium phenotype colors and labels
    colors = {
        0: "cornflowerblue",  # PD1n (T-cell, PD-1 negative)
        1: "royalblue",       # PD1p (T-cell, PD-1 positive)
        2: "lightcoral",      # PDL1n (Tumor, PD-L1 negative)
        3: "crimson"          # PDL1p (Tumor, PD-L1 positive)
    }
    labels = {
        0: "T-cell PD1-",
        1: "T-cell PD1+",
        2: "Tumor PDL1-",
        3: "Tumor PDL1+"
    }

    N = 4  # Number of phenotypes

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.grid(True, linestyle='--', alpha=0.2)

    for i in range(N):
        smoothed = moving_average(corr_matrices[:, i, i], window=window)
        avg_line = np.mean(corr_matrices[:, i, i])

        # Smoothed line
        ax.plot(frames, smoothed, color=colors[i], linewidth=2.5, label=labels[i])

        # Average line
        ax.plot(frames, [avg_line]*len(frames), color=colors[i], linewidth=2,
                linestyle='--', alpha=0.7)

        # Add text label at the end
        ax.text(frames[-1] + 0.5, avg_line, f"{avg_line:.2f}", color=colors[i],
                fontsize=9, va='center', ha='left')

    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Self-Correlation", fontsize=14)
    ax.set_title(f"Vivarium Self-Correlations (radius={radius})", fontsize=16)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_cross_correlations_vivarium(frames, corr_matrices, radius=50, window=5, save_path=None):
    """
    Plot cross-correlations (off-diagonal elements) over time for vivarium data.

    Parameters:
    -----------
    frames : array-like
        Time points
    corr_matrices : ndarray
        Correlation matrices with shape (n_frames, 4, 4)
    radius : float
        Neighborhood radius used in correlation calculation
    window : int
        Window size for moving average smoothing
    save_path : str, optional
        Path to save the figure
    """
    colors = {
        0: "cornflowerblue",
        1: "royalblue",
        2: "lightcoral",
        3: "crimson"
    }
    labels = {
        0: "T-cell PD1-",
        1: "T-cell PD1+",
        2: "Tumor PDL1-",
        3: "Tumor PDL1+"
    }

    N = 4

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.grid(True, linestyle='--', alpha=0.15)

    # Plot all cross-correlations
    for i in range(N):
        for j in range(i+1, N):
            smoothed = moving_average(corr_matrices[:, i, j], window=window)
            avg_line = np.mean(corr_matrices[:, i, j])

            # Blend colors for cross-correlation
            c1 = np.array(to_rgba(colors[i]))
            c2 = np.array(to_rgba(colors[j]))
            blended_color = (c1 + c2) / 2

            label_pair = f"{labels[i]} × {labels[j]}"

            # Smoothed line
            ax.plot(frames, smoothed, color=blended_color, linewidth=2.5,
                   label=label_pair, alpha=0.8)

            # Average line
            ax.plot(frames, [avg_line]*len(frames), color=blended_color,
                    linewidth=2, linestyle='--', alpha=0.5)

            # Text label
            ax.text(frames[-1] + 0.5, avg_line, f"{avg_line:.2f}",
                   color=blended_color, fontsize=8, va='center', ha='left')

    ax.axhline(1, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Cross-Correlation", fontsize=14)
    ax.set_title(f"Vivarium Cross-Correlations (radius={radius})", fontsize=16)
    ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_correlation_heatmaps(frames, corr_matrices, timepoints=[0, 25, 50], save_path=None):
    """
    Plot correlation matrix heatmaps at specific timepoints.
    
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
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, t in enumerate(timepoints):
        # Find nearest frame
        frame_idx = np.argmin(np.abs(np.array(frames) - t))
        actual_time = frames[frame_idx]
        corr = corr_matrices[frame_idx]
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=1.0,
                    vmin=0, vmax=2, cbar_kws={'label': 'Correlation'},
                    xticklabels=labels, yticklabels=labels, ax=axes[idx],
                    square=True, linewidths=0.5, annot_kws={'fontsize': 8})
        
        axes[idx].set_title(f"Time = {actual_time:.1f}", fontsize=10)
        axes[idx].tick_params(axis='both', labelsize=7)
        
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_zero_counts_vivarium(frames, zero_counts, save_path=None):
    """
    Plot number of cells with zero neighbors over time.

    Parameters:
    -----------
    frames : array-like
        Time points
    zero_counts : ndarray
        Zero counts with shape (n_frames, 4)
    save_path : str, optional
        Path to save the figure
    """
    colors = {
        0: "cornflowerblue",
        1: "royalblue",
        2: "lightcoral",
        3: "crimson"
    }
    labels = {
        0: "T-cell PD1-",
        1: "T-cell PD1+",
        2: "Tumor PDL1-",
        3: "Tumor PDL1+"
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for j in range(4):
        ax.plot(frames, zero_counts[:, j], marker='o', markersize=3,
               label=labels[j], color=colors[j], linewidth=2, alpha=0.8)

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Number of cells with ZERO neighbors", fontsize=14)
    ax.set_title("Zero-neighbor cells per phenotype over time", fontsize=16)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_all_correlations_grid(frames, corr_matrices, radius=50, window=5, save_path=None):
    """
    Plot all correlation time series in a grid layout.

    Parameters:
    -----------
    frames : array-like
        Time points
    corr_matrices : ndarray
        Correlation matrices with shape (n_frames, 4, 4)
    radius : float
        Neighborhood radius used in correlation calculation
    window : int
        Window size for moving average smoothing
    save_path : str, optional
        Path to save the figure
    """
    labels = ["T PD1-", "T PD1+", "Tu PDL1-", "Tu PDL1+"]
    colors = ["cornflowerblue", "royalblue", "lightcoral", "crimson"]

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    fig.suptitle(f"All Correlations Over Time (radius={radius})", fontsize=18, y=0.995)

    for i in range(4):
        for j in range(4):
            ax = axes[i, j]

            # Extract and smooth correlation
            corr_data = corr_matrices[:, i, j]
            smoothed = moving_average(corr_data, window=window)
            avg_line = np.mean(corr_data)

            # Determine color
            if i == j:
                color = colors[i]
            else:
                c1 = np.array(to_rgba(colors[i]))
                c2 = np.array(to_rgba(colors[j]))
                color = (c1 + c2) / 2

            # Plot
            ax.plot(frames, smoothed, color=color, linewidth=2, alpha=0.8)
            ax.axhline(avg_line, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3, linewidth=1)

            # Labels
            if i == 0:
                ax.set_title(labels[j], fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=12, fontweight='bold')
            if i == 3:
                ax.set_xlabel("Time", fontsize=10)

            # Styling
            ax.grid(alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add average value as text
            ax.text(0.95, 0.95, f"μ={avg_line:.2f}", transform=ax.transAxes,
                   fontsize=9, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


if __name__ == "__main__":
    print("Correlation visualization functions loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_self_correlations_vivarium()")
    print("  - plot_cross_correlations_vivarium()")
    print("  - plot_correlation_heatmaps()")
    print("  - plot_zero_counts_vivarium()")
    print("  - plot_all_correlations_grid()")
