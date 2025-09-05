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
    print("Usage: python script_name.py <rp_value>")
    sys.exit(1)


rp = float(sys.argv[1])
#rp = 0.5
rd = rp/2
rm = 1.0
scale_factor = 2
num_t = 100
n_sims = 50

for i in range(n_sims):
    # Run one simulation
    A_out,t_out,plot_list = BDM_ABM(rp,rd,rm,T_end=15.0, scale = scale_factor)
    # Save each sim data to a .npy file
    save_data = {'variables': [t_out, A_out]}
    folder_path = "../data"
    # Create a unique filename for this simulation
    filename = f"{folder_path}/modified_logistic_ABM_sim_rp_{rp}_rd_{rd}_rm_{rm:}_scale_2_{i}.npy"
    # Save to file
    np.save(filename, save_data)


n_sims = 50
folder_path = "../data"
# List to hold each A_out
A_list = []

for i in range(n_sims):
    filename = f"{folder_path}/modified_logistic_ABM_sim_rp_{rp}_rd_{rd}_rm_{rm:}_scale_2_{i}.npy"
    mat = np.load(filename, allow_pickle=True, encoding='latin1').item()
    A_out = mat['variables'][1]
    A_list.append(A_out)

# Stack into 2D matrix: rows = simulations, cols = time points
A_matrix = np.vstack(A_list)


# Average across simulations
avg_A = np.mean(A_matrix, axis=0)
avg_A = avg_A / (120*120)

# Compute derivative
t_out = mat['variables'][0]
ABM_t = compute_derivative(t_out, avg_A)

#t_out2 = t_out.reshape(-1, 1)
#A_out2 = A_out.reshape(-1, 1)
#ABM_t2 = ABM_t.reshape(-1, 1)

save_data = {'variables': [t_out, avg_A, ABM_t]}
folder_path = "../data"
# Create a unique filename for this simulation
filename = f"{folder_path}/modified_logistic_ABM_sim_rp_{rp}_rd_{rd}_rm_{rm:}_scale_2_complete.npy"
# Save to file
np.save(filename, save_data)