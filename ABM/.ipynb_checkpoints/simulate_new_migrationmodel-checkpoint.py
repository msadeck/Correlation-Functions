import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import integrate
import matplotlib as mpl
from scipy import interpolate
import time
import os
import sys

from ABM_package import *



# Get rp from command-line
if len(sys.argv) < 2:
    print("Usage: python script_name.py <scale_factor>")
    sys.exit(1)


rp = 0.5
rd = rp/2
rm = 1.0
scale_factor = float(sys.argv[1])
den0 = .5
num_t = 100
n_sims = 5

for i in range(n_sims):
    # Run one simulation
    A_out, t_out, plot_list, interp_profiles = BDM_ABM(rp,rd,rm,scale_factor,den0, T_end=50.0)
    # Save each sim data to a .npy file
    F_values = np.array([compute_F(A) for A in plot_list])
    num_snapshots = len(plot_list)
    T_final = t_out[-1]
    snapshot_times = np.linspace(0, T_final, num_snapshots)
    F_out = interpolate.interp1d(snapshot_times, F_values, kind='linear')(t_out)
    save_data = {'variables': [t_out, A_out, F_out]}
    folder_path = "../data"
    # Create a unique filename for this simulation
    filename = f"{folder_path}/correlation_data_run_scale_{i}_{scale_factor}.npy"
    # Save to file
    np.save(filename, save_data)


n_sims = 50
folder_path = "../data"
# List to hold each A_out
A_list = []
F_list = []

for i in range(n_sims):
    filename = f"{folder_path}/correlation_data_run_scale_{i}_{scale_factor}.npy"
    mat = np.load(filename, allow_pickle=True, encoding='latin1').item()
    A_out = mat['variables'][1]
    A_list.append(A_out)
    F_out = mat['variables'][2]
    F_lit.append(F_out)
    

# Stack into 2D matrix: rows = simulations, cols = time points
A_matrix = np.vstack(A_list)
F_matrix = np.vstack(F_list)


# Average across simulations
avg_A = np.mean(A_matrix, axis=0)
avg_A = avg_A / (120*120)

# Average across simulations
avg_F = np.mean(F_matrix, axis=0)
avg_F = avg_F / (120*120)


# Compute derivative
t_out = mat['variables'][0]
ABM_t = compute_derivative(t_out, avg_A)
dF_dt = compute_derivative(t_out,avg_F)

#t_out2 = t_out.reshape(-1, 1)
#A_out2 = A_out.reshape(-1, 1)
#ABM_t2 = ABM_t.reshape(-1, 1)

save_data = {'variables': [t_out, avg_A, ABM_t, avg_F, dF_dt]}
folder_path = "../data"
# Create a unique filename for this simulation
filename = f"{folder_path}/correlation_data_run_scale_{scale_factor}_complete.npy"
# Save to file
np.save(filename, save_data)