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