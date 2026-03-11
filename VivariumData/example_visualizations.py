"""
Example script for visualizing vivarium correlation results.

Prerequisites:
--------------
You should have already run correlations_by_framez() and have:
- frames: array of timepoints
- corr_matrices: correlation matrices (shape: n_frames x 4 x 4)
- zero_counts_joint: zero neighbor counts (shape: n_frames x 4)

Usage:
------
Assuming you've already computed correlations like this:

    from correlation_package import correlations_by_framez
    import pandas as pd

    viv_df = pd.read_csv('extracted_cell_data.csv')

    frames, corr_matrices, zero_counts_joint = correlations_by_framez(
        viv_df,
        frame_col='timepoint',
        x_col='x_coord',
        y_col='y_coord',
        label_col='phenotype',
        N=4,
        radius=50,
        normalization="p_joint",
        return_zero_counts=True
    )

Then run this script to create all visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from EQL_Corr_vivarium.ABM.visualize_correlations import (
    plot_self_correlations_vivarium,
    plot_cross_correlations_vivarium,
    plot_correlation_heatmaps,
    plot_zero_counts_vivarium,
    plot_all_correlations_grid
)

# ============================================================================
# EDIT THIS SECTION WITH YOUR DATA
# ============================================================================

# If you have your correlation results saved, load them:
# frames = np.load('frames.npy')
# corr_matrices = np.load('corr_matrices.npy')
# zero_counts_joint = np.load('zero_counts_joint.npy')

# OR if they're in variables already, just use them directly:
# They should be passed to this script or loaded from somewhere

# For demonstration purposes, we'll check if they exist in the namespace
try:
    # Check if variables exist (they should be defined when running this in your session)
    print(f"Found frames with shape: {frames.shape}")
    print(f"Found corr_matrices with shape: {corr_matrices.shape}")
    print(f"Found zero_counts_joint with shape: {zero_counts_joint.shape}")
except NameError:
    print("ERROR: Correlation results not found!")
    print("Please ensure you have run correlations_by_framez() first.")
    print("\nExample:")
    print("  from correlation_package import correlations_by_framez")
    print("  import pandas as pd")
    print("  viv_df = pd.read_csv('extracted_cell_data.csv')")
    print("  frames, corr_matrices, zero_counts_joint = correlations_by_framez(")
    print("      viv_df, 'timepoint', 'x_coord', 'y_coord', 'phenotype',")
    print("      N=4, radius=50, normalization='p_joint', return_zero_counts=True)")
    exit(1)

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

# Set the radius value used in your correlation calculation
RADIUS = 50  # Adjust this to match what you used
WINDOW = 5   # Smoothing window

print("\n" + "="*80)
print("Creating Visualizations")
print("="*80)

# 1. Self-correlations plot
print("\n1. Plotting self-correlations...")
fig1, ax1 = plot_self_correlations_vivarium(
    frames, corr_matrices,
    radius=RADIUS,
    window=WINDOW,
    save_path='vivarium_self_correlations.png'
)
plt.show()
print("   ✓ Saved: vivarium_self_correlations.png")

# 2. Cross-correlations plot
print("\n2. Plotting cross-correlations...")
fig2, ax2 = plot_cross_correlations_vivarium(
    frames, corr_matrices,
    radius=RADIUS,
    window=WINDOW,
    save_path='vivarium_cross_correlations.png'
)
plt.show()
print("   ✓ Saved: vivarium_cross_correlations.png")

# 3. Correlation heatmaps at specific timepoints
print("\n3. Plotting correlation heatmaps...")
# Choose timepoints to visualize (adjust as needed)
timepoints = [0, 26, 52]  # beginning, middle, end
fig3, axes3 = plot_correlation_heatmaps(
    frames, corr_matrices,
    timepoints=timepoints,
    save_path='vivarium_correlation_heatmaps.png'
)
plt.show()
print("   ✓ Saved: vivarium_correlation_heatmaps.png")

# 4. Zero neighbor counts
print("\n4. Plotting zero neighbor counts...")
fig4, ax4 = plot_zero_counts_vivarium(
    frames, zero_counts_joint,
    save_path='vivarium_zero_counts.png'
)
plt.show()
print("   ✓ Saved: vivarium_zero_counts.png")

# 5. Full correlation grid (comprehensive view)
print("\n5. Plotting full correlation grid...")
fig5, axes5 = plot_all_correlations_grid(
    frames, corr_matrices,
    radius=RADIUS,
    window=WINDOW,
    save_path='vivarium_all_correlations_grid.png'
)
plt.show()
print("   ✓ Saved: vivarium_all_correlations_grid.png")

print("\n" + "="*80)
print("All visualizations created successfully!")
print("="*80)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

phenotype_labels = ["T-cell PD1-", "T-cell PD1+", "Tumor PDL1-", "Tumor PDL1+"]

print("\nMean Self-Correlations (over all timepoints):")
for i in range(4):
    mean_self = np.mean(corr_matrices[:, i, i])
    std_self = np.std(corr_matrices[:, i, i])
    print(f"  {phenotype_labels[i]:20s}: {mean_self:.3f} ± {std_self:.3f}")

print("\nMean Cross-Correlations (over all timepoints):")
for i in range(4):
    for j in range(i+1, 4):
        mean_cross = np.mean(corr_matrices[:, i, j])
        std_cross = np.std(corr_matrices[:, i, j])
        print(f"  {phenotype_labels[i]:20s} × {phenotype_labels[j]:20s}: {mean_cross:.3f} ± {std_cross:.3f}")

print("\nMean Zero Neighbor Counts (over all timepoints):")
for i in range(4):
    mean_zero = np.mean(zero_counts_joint[:, i])
    std_zero = np.std(zero_counts_joint[:, i])
    print(f"  {phenotype_labels[i]:20s}: {mean_zero:.1f} ± {std_zero:.1f}")

print("\n" + "="*80)
