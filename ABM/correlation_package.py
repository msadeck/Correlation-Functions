import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import integrate
import matplotlib as mpl
from scipy import interpolate
import time
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

def count_occupied_pairs(A):
    count = 0
    n = A.shape[0]
    for x in range(n):
        for y in range(n):
            if A[x, y] == 1:
                if x < n - 1 and A[x + 1, y] == 1:
                    count += 1
                if y < n - 1 and A[x, y + 1] == 1:
                    count += 1
    return count

def compute_F(A):
    X = A.shape[0]
    chi = 2 * X * (X - 1)
    C1 = np.sum(A == 1)
    C2 = count_occupied_pairs(A)
    return (C2 * X**4) / (chi * C1**2) if C1 > 0 else 0

def compute_derivative(t, y):
    """
    Compute dy/dt using finite differences:
    - Forward difference at the first point
    - Centered differences for interior points
    - Backward difference at the last point

    Parameters:
    - t: time array (1D)
    - y: corresponding y-values (1D), e.g., ABM output

    Returns:
    - dydt: array of derivatives (same length as t)
    """
    t = np.asarray(t).flatten()
    y = np.asarray(y).flatten()
    n = len(t)
    dydt = np.zeros(n)

    # Forward difference for the first point
    dydt[0] = (y[1] - y[0]) / (t[1] - t[0])

    # Centered differences for internal points
    for i in range(1, n - 1):
        dydt[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])

    # Backward difference for the last point
    dydt[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])

    return dydt


def counts_matrix(A, N, radius):
    pos = {p: np.argwhere(A == p) for p in range(N)}
    trees = {p: KDTree(pos[p]) if len(pos[p])>0 else None for p in range(N)}
    total = np.zeros((N, N), dtype=float)
    avg   = np.zeros((N, N), dtype=float)

    for i in range(N):
        Xi = pos[i]
        ni = len(Xi)
        if ni == 0:
            total[i,:] = 0
            avg[i,:] = np.nan
            continue
        for j in range(i, N):  # only upper triangle
            if len(pos[j]) == 0:
                total[i,j] = total[j,i] = 0
                avg[i,j] = avg[j,i] = 0.0
                continue

            neigh_idx = trees[j].query_radius(Xi, r=radius)
            counts_per_i = np.array([len(idxs) for idxs in neigh_idx])

            if i == j:
                # subtract self-counts
                counts_per_i = counts_per_i - 1
                counts_per_i[counts_per_i < 0] = 0
                # each (i,i) edge counted twice → divide by 2
                total[i,i] = counts_per_i.sum() / 2
                avg[i,i]   = counts_per_i.mean()
            else:
                # off-diagonal: fill both (i,j) and (j,i)
                total[i,j] = counts_per_i.sum()
                avg[i,j]   = counts_per_i.mean()

                # symmetric counts (from j’s perspective)
                neigh_idx_j = trees[i].query_radius(pos[j], r=radius)
                counts_per_j = np.array([len(idxs) for idxs in neigh_idx_j])
                total[j,i] = counts_per_j.sum()
                avg[j,i]   = counts_per_j.mean()
    return total, avg


def correlation_matrix_for_snapshot_unordered(A, N, radius, global_normalization=False):
    """
    Compute correlation matrix for a single snapshot using unordered neighbor counts.

    Parameters:
    - A: 2D array of phenotypes
    - N: number of phenotypes
    - radius: neighborhood radius
    - global_normalization: if True, normalize P_joint over all possible site pairs (n*(n-1))
                            if False, normalize over total neighbor counts (default)

    Returns:
    - corr: N x N correlation matrix
    """
    # Use the unordered counts function
    total, avg = counts_matrix_for_snapshot_unordered(A, N, radius)
    
    n_sites = A.shape[0]
    P = np.array([np.sum(A == p) / n_sites**2 for p in range(N)])  # marginal probabilities

    # Determine joint probability normalization
    if global_normalization:
        total_pairs = 2*(n_sites * (n_sites - 1))# all possible site pairs
    else:
        total_pairs = total.sum()  # only observed neighbor pairs

    if total_pairs == 0:
        return np.full((N, N), np.nan)  # avoid div0

    P_joint = total / total_pairs

    # Compute correlation matrix
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if P[i] > 0 and P[j] > 0:
                corr[i,j] = P_joint[i,j] / (P[i] * P[j])
            else:
                corr[i,j] = np.nan

    return corr

def correlation_time_series_unordered(A_series, N, radius, global_normalization=False):
    """
    Compute correlation coefficients over time using unordered counts.
    Returns a dictionary mapping (i,j) -> array of length T.
    """
    T = len(A_series)
    corr_dict = {(i,j): np.zeros(T) for i in range(N) for j in range(i, N)}

    for t in range(T):
        corr_matrix = correlation_matrix_for_snapshot_unordered(
            A_series[t], N, radius, global_normalization=global_normalization
        )
        for i in range(N):
            for j in range(i, N):
                corr_dict[(i,j)][t] = corr_matrix[i,j]

    return corr_dict

def correlation_from_dataframe(df, N, radius=5, grid_size=None):
    """
    Compute phenotype correlation matrix from a DataFrame with columns:
        - x, y (pixel coordinates)
        - phenotype (integer labels 0..N-1)

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'x', 'y', and 'phenotype' columns.
    N : int
        Number of phenotype types.
    radius : int
        Pixel radius for neighborhood correlation (default = 5).
    grid_size : int or None
        If None, size is inferred as max(x,y)+1.

    Returns
    -------
    corr : (N x N) numpy array
        Correlation matrix.
    """

    # --- Check required columns ---
    if not {'x','y','phenotype'}.issubset(df.columns):
        raise ValueError("DataFrame must contain columns: x, y, phenotype")

    # --- Infer grid size if needed ---
    if grid_size is None:
        max_x = df['x'].max()
        max_y = df['y'].max()
        grid_size = int(max(max_x, max_y) + 1)

    # --- Build grid ---
    A = -1 * np.ones((grid_size, grid_size), dtype=int)  # empty sites = -1
    for _, row in df.iterrows():
        A[int(row.y), int(row.x)] = int(row.phenotype)

    # --- Compute correlation matrix using your existing machinery ---
    corr = correlation_matrix_for_snapshot_unordered(
        A=A,
        N=N,
        radius=radius,
        global_normalization=False
    )

    return corr

import numpy as np
from sklearn.neighbors import KDTree
# Codependent functions to count neighbors from point coordinates and labels #

import numpy as np
from sklearn.neighbors import KDTree

import numpy as np
from sklearn.neighbors import KDTree

def counts_and_zero_neighbors_normalized(coords, labels, N, radius):
    """
    Counts neighbors and returns zero neighbor counts normalized 
    by the number of cells of each phenotype (relative abundance normalization).

    Returns:
    - total: NxN matrix of total neighbor counts
    - avg: NxN matrix of average neighbors per cell
    - zero_counts_normalized: 1D array, fraction of cells of type i with zero neighbors
    """
    total = np.zeros((N, N), dtype=float)
    avg = np.zeros((N, N), dtype=float)
    zero_counts_normalized = np.zeros(N, dtype=float)

    tree = KDTree(coords)

    for i in range(N):
        idx_i = np.where(labels == i)[0]  # indices of cells of phenotype i
        n_i = len(idx_i)  # number of cells of type i

        if n_i == 0:
            avg[i, :] = np.nan
            zero_counts_normalized[i] = np.nan
            continue

        neigh_idx_i = tree.query_radius(coords[idx_i], r=radius)

        # Neighbor counts per (i,j)
        for j in range(N):
            counts = np.array([np.sum(labels[neigh] == j) for neigh in neigh_idx_i])
            
            if i == j:
                counts -= 1
                counts[counts < 0] = 0
                total[i, i] = np.sum(counts) / 2
                avg[i, i] = np.mean(counts)
            else:
                total[i, j] = np.sum(counts)
                avg[i, j] = np.mean(counts)

        # Normalize zero counts by relative abundance
        zero_raw = np.sum([len(neigh) == 1 for neigh in neigh_idx_i])  # raw zeros
        zero_counts_normalized[i] = zero_raw / n_i  # normalized (fraction)

    return total, avg, zero_counts_normalized


def spatial_correlation_from_pointsz(coords, labels, N, radius, normalization="p_joint", return_zero_counts=False):
    """
    Compute the spatial correlation matrix for points, optionally returning zero neighbor counts.

    Inputs:
    - coords: Nx2 array of x,y positions
    - labels: array of length N with phenotype labels
    - N: number of phenotypes
    - radius: neighborhood radius
    - normalization: normalization method
    - return_zero_counts: if True, also return 1D array of zero neighbor counts

    Returns:
    - corr: NxN correlation matrix
    - zero_counts (optional): 1D array of length N with number of cells of each type with zero neighbors
    """
    total, avg, zero_counts = counts_and_zero_neighbors_normalized(coords, labels, N, radius)
    corr = normalize_correlation(total, labels, N, radius, normalization, coords=coords)
    
    if return_zero_counts:
        return corr, zero_counts
    else:
        return corr


def spatial_correlation_from_dataframez(df, N, radius, normalization="p_joint", return_zero_counts=False):
    """
    Compute correlation matrix from a dataframe, optionally returning zero neighbor counts.

    The dataframe must contain 'x_microns', 'y_microns', 'phenotype' columns.
    """
    required = {'x_microns', 'y_microns', 'phenotype'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    coords = df[['x_microns', 'y_microns']].to_numpy()
    labels = df['phenotype'].to_numpy().astype(int)

    return spatial_correlation_from_pointsz(coords, labels, N, radius, normalization, return_zero_counts=return_zero_counts)

def correlations_by_framez(df, frame_col, x_col, y_col, label_col, N, radius, normalization="p_joint", return_zero_counts=False):
    """
    Compute correlation matrices for each frame, optionally returning zero neighbor counts per frame.
    """
    frames = sorted(df[frame_col].unique())
    all_corr_matrices = []
    all_zero_counts = [] if return_zero_counts else None

    for f in frames:
        df_f = df[df[frame_col] == f]
        coords = df_f[[x_col, y_col]].to_numpy()
        labels = df_f[label_col].to_numpy().astype(int)

        if return_zero_counts:
            corr, zero_counts = spatial_correlation_from_pointsz(coords, labels, N, radius, normalization, return_zero_counts=True)
            all_zero_counts.append(zero_counts)
        else:
            corr = spatial_correlation_from_pointsz(coords, labels, N, radius, normalization)
        
        all_corr_matrices.append(corr)

    if return_zero_counts:
        return frames, np.array(all_corr_matrices), np.array(all_zero_counts)
    else:
        return frames, np.array(all_corr_matrices)



def counts_from_points(coords, labels, N, radius):
    """
    Counts neighbors of each phenotype type within a radius.
    Inputs:
    - coords: array of x,y positions of cells, shape (num_cells, 2)
    - labels: array of phenotype labels for each cell, shape (num_cells,)
    - N: number of phenotypes
    - radius: neighborhood radius

    Returns:
    - total: total counts of neighbors for each (i,j) pair
    - avg: average counts per i cell for each j
    """
    total = np.zeros((N, N), dtype=float)
    avg   = np.zeros((N, N), dtype=float)

    tree = KDTree(coords) #initializes KDTree for fast spatial queries in given radius

    for i in range(N):
        idx_i = np.where(labels == i)[0]
        if len(idx_i) == 0:
            avg[i, :] = np.nan
            continue

        neigh_idx_i = tree.query_radius(coords[idx_i], r=radius) #for each cell of type i, find all neighbors within radius

        for j in range(N):
            counts = [np.sum(labels[neigh] == j) for neigh in neigh_idx_i] #for each i cell, count how many neighbors are of type j

            if i == j: #special case, when counting same-type neighbors
                counts = np.array(counts) - 1 #subtract 1 to remove self-count
                counts[counts < 0] = 0
                total[i, i] = np.sum(counts) / 2 #avoid double counted for self counts (i,j)=(j,i)
                avg[i, i]   = np.mean(counts)
            else:
                counts = np.array(counts)
                total[i, j] = np.sum(counts)
                avg[i, j]   = np.mean(counts)

    return total, avg


def normalize_correlation(total, labels, N, radius, normalization, coords=None):
    """
    Normalize neighbor counts according to the chosen method.
    total: NxN matrix of raw neighborhood counts between phenotypes calculated by counts_from_points
    labels: 1D array of phenotype labels for each cell
    N: number of phenotypes
    radius: neighborhood radius
    normalization: one of "none", "p_joint", "expected_uniform", "density_corrected"
    coords: array of x,y coordinates of cells (required for density_corrected)
    """
    n = len(labels) #total number of cells
    P = np.array([np.sum(labels == p) / n for p in range(N)])  # array of length N, fraction of each phenotype
    total_pairs = total.sum() #sum of all entries in total count matrix (i.e. total neighbor pairs)
    corr = np.full((N, N), np.nan) #initialize correlation matrix with NaNs

    if total_pairs == 0: #if no pairs, return NaN matrix
        return corr

    if normalization == "none": #if no normalization is chosen, return raw counts normalized by max count
        return total / np.nanmax(total)

    elif normalization == "p_joint": #joint probability normalization, most standard practice of correlation calculation
        P_joint = total / total_pairs #each element in total divided by total neighbor pairs
        for i in range(N): #for each phenotype pair, compute joint probability divided by independent probability
            for j in range(N):
                if P[i] > 0 and P[j] > 0:
                    corr[i, j] = P_joint[i, j] / (P[i] * P[j])

    elif normalization == "expected_uniform": #assume uniform distribution of phenotypes, good for when phenotype frequencies vary
        expected = np.outer(P, P) * total_pairs #expected counts based on phenotype frequencies * total pairs)
        corr = total / expected
        corr[expected == 0] = np.nan

    elif normalization == "density_corrected": #density-based normalization, requires spatial positions, good for non-uniform spatial distributions
        if coords is None:
            raise ValueError("coords must be provided for density_corrected normalization")

        tree = KDTree(coords) #build KDTree for spatial queries
        expected = np.zeros((N, N), dtype=float) #initialize expected counts matrix

        for i in range(N): #for each phenotype, compute local density and expected counts
            idx_i = np.where(labels == i)[0]
            if len(idx_i) == 0:
                continue

            neigh_idx_i = tree.query_radius(coords[idx_i], r=radius)
            density_i = np.mean([len(neigh) - 1 for neigh in neigh_idx_i])

            for j in range(N):
                expected[i, j] = density_i * P[j] * len(idx_i) #expected count of j around i cells, by taking the local density and multiplying it by the fraction of j cells. 

        with np.errstate(divide='ignore', invalid='ignore'):
            corr = total / expected
            corr[expected == 0] = np.nan

    else:
        raise ValueError(f"Unknown normalization type: {normalization}")

    return corr



def spatial_correlation_from_points(coords, labels, N, radius, normalization="p_joint"): #sends point coordinates and labels to counts and normalization functions
    total, _ = counts_from_points(coords, labels, N, radius)
    return normalize_correlation(total, labels, N, radius, normalization, coords=coords)


def spatial_correlation_from_dataframe(df, N, radius, normalization="p_joint"): #combines entire df into single correlation matrix
    required = {'x_microns', 'y_microns', 'phenotype'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    coords = df[['x_microns', 'y_microns']].to_numpy() #create numpy arrays of coordinates and labels
    labels = df['phenotype'].to_numpy().astype(int)

    return spatial_correlation_from_points(coords, labels, N, radius, normalization) #computes correlation matrix with normalization


def correlations_by_frame(df, frame_col, x_col, y_col, label_col, N, radius, normalization="p_joint"): #separates correlations by frame/time
    frames = sorted(df[frame_col].unique()) #sorts by time/frame
    all_corr_matrices = [] #initialize list to hold correlation matrices for each frame

    for f in frames:
        df_f = df[df[frame_col] == f] #filter dataframe for current frame
        coords = df_f[[x_col, y_col]].to_numpy() #create numpy arrays of coordinates and labels for current frame
        labels = df_f[label_col].to_numpy().astype(int)

        corr = spatial_correlation_from_points(coords, labels, N, radius, normalization) #compute correlation matrix for current frame
        all_corr_matrices.append(corr)  #append to list

    return frames, np.array(all_corr_matrices)



