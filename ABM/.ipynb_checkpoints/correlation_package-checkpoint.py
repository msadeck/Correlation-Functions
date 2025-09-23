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


def counts_matrix_for_snapshot_unordered(A, N, radius):
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


