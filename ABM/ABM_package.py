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

def local_neighborhood_mask(A_shape, loc, distance=1):
    '''
    Create a sparse matrix with 1s in the neighborhood of a point (loc),
    and 0s elsewhere.

    Parameters:
    - A_shape: tuple (rows, cols) of the matrix
    - loc: tuple (x, y) center of the neighborhood
    - distance: neighborhood distance (default = 1)

    Returns:
    - mask: sparse lil_matrix with 1s in the neighborhood
    '''
    rows, cols = A_shape
    x, y = loc

    # Create an empty sparse matrix
    mask = lil_matrix((rows, cols), dtype=int)

    # Compute neighborhood bounds
    for i in range(max(0, x - distance), min(rows, x + distance + 1)):
        for j in range(max(0, y - distance), min(cols, y + distance + 1)):
            mask[i, j] = 1

    return mask
    
def SIR_ODE(t,y,q,desc):

    dydt = np.zeros((3,))

    dydt[0] = -q[0]*y[0]*y[1]
    dydt[1] = -q[1]*y[1] + q[0]*y[0]*y[1]
    dydt[2] = q[1]*y[1]
    
    return dydt

def ODE_sim(q,RHS,t,IC,description=None):
    
    #grids for numerical integration
    t_sim = np.linspace(t[0],t[-1],10000)
    
    #Initial condition
    y0 = IC
        
    #indices for integration steps to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    #make RHS a function of t,y
    def RHS_ty(t,y):
        return RHS(t,y,q,description)
            
    #initialize solution
    y = np.zeros((len(y0),len(t)))   
    y[:,0] = IC
    write_count = 1

    #integrate
    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t[0])   # initial values
    for i in range(1, t_sim.size):
        #write to y during write indices
        if np.any(i==t_sim_write_ind):
            y[:,write_count] = r.integrate(t_sim[i])
            write_count+=1
        else:
            #otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print("integration failed for parameter ")
            print(q)
            return 1e6*np.ones(y.shape)

    return y


def BDM_ABM(rp,rd,rm,scale,T_end=50.0):
    #rp is proliferation rate, rd is death rate, rm is migration rate and f is the factor at which those variables change depending on the neighborhood occupancy
    
    #number of lattice sites
    n = 120
    A = np.zeros((n**2,))

    #initial proportions of occupied sires
    A0 = 0.05
    
    #randomly place susceptible (1), infected (2), and recovered (3) agents
    A_num = int(np.ceil(A0*len(A)))
    A[:A_num] = 1
    #shuffle up
    A = A[np.random.permutation(n**2)]
    #make square
    A = A.reshape(n,n)

    #count number of susceptible, infected, and recovered agents.
    A_num = np.sum(A==1)

    #nondimensionalized time
    T_final = T_end/(rp-rd)

    #initialize time
    t = 0

    #track time, agent proportions, and snapshots of ABM in these lists
    t_list = [t]
    A_list = [A_num]
    plot_list = [A]
    
    #number of snapshots saved
    image_count = 1


    while t_list[-1] < T_final:
        # Select Random agent
        
        agent_loc = np.where(A!=0)
        agent_ind = np.random.permutation(len(agent_loc[0]))[0]
        loc = (agent_loc[0][agent_ind],agent_loc[1][agent_ind])
    
        mask = local_neighborhood_mask((n,n,),loc,distance=1)
        neigh_den = mask.multiply(A)
        result = np.sum(neigh_den==1)
       
        if result >= 4:
            rmf=scale*rm
            rpf=(1/scale)*rp
    
        else:
            rmf=(1/scale)*rm
            rpf=scale*rp
           
            
        a = rmf*A_num + rpf*A_num + rd*A_num
        tau = -np.log(np.random.uniform())/a
        t += tau

        Action = a*np.random.uniform()

        if Action <= rmf*(A_num):
            #agent movement
            
            #determine status
            agent_state = A[loc]

            ### Determine direction
            dir_select = np.ceil(np.random.uniform(high=4.0))

            #move right
            if dir_select == 1 and loc[0]<n-1:
                if A[(loc[0]+1,loc[1])] == 0:
                    A[(loc[0]+1,loc[1])] = agent_state
                    A[loc] = 0
            #move left
            elif dir_select == 2 and loc[0]>0:
                if A[(loc[0]-1,loc[1])] == 0:
                    A[(loc[0]-1,loc[1])] = agent_state
                    A[loc] = 0
            #move up
            elif dir_select == 3 and loc[1]<n-1:
                if A[(loc[0],loc[1]+1)] == 0:
                    A[(loc[0],loc[1]+1)] = agent_state
                    A[loc] = 0

            #move down                    
            elif dir_select == 4 and loc[1]>0:
                if A[(loc[0],loc[1]-1)] == 0:
                    A[(loc[0],loc[1]-1)] = agent_state
                    A[loc] = 0

        elif (Action <= rmf*(A_num) + rpf*A_num):
            #proliferation event
            
            ### Determine direction
            dir_select = np.ceil(np.random.uniform(high=4.0))

            #proliferate right
            if dir_select == 1 and loc[0]<n-1:
                if A[(loc[0]+1,loc[1])] == 0:
                    A[(loc[0]+1,loc[1])] = 1

            #proliferate left
            elif dir_select == 2 and loc[0]>0:
                if A[(loc[0]-1,loc[1])] == 0:
                    A[(loc[0]-1,loc[1])] = 1

            #proliferate up        
            elif dir_select == 3 and loc[1]<n-1:
                if A[(loc[0],loc[1]+1)] == 0:
                    A[(loc[0],loc[1]+1)] = 1

            #proliferate down
            elif dir_select == 4 and loc[1]>0:
                if A[(loc[0],loc[1]-1)] == 0:
                    A[(loc[0],loc[1]-1)] = 1

        elif (Action <= rmf*(A_num) + rpf*A_num + rd*A_num):
            #death event
            
            
            A[loc] = 0

        #count number of susceptible, infected, recovered agents
        A_num = np.sum(A==1)
        
        #append information to lists
        t_list.append(t)
        A_list.append(A_num)
        
        #sometimes save ABM snapshot
        if len(t_list) == 2:
            plot_list.append(np.copy(A))
            image_count+=1
        elif (t_list[-2] < image_count*T_final/50 and t_list[-1] >= image_count*T_final/50): 
            plot_list.append(np.copy(A))
            image_count+=1

    #interpolation to equispace grid
    t_out = np.linspace(0,T_final,100)

    f = interpolate.interp1d(t_list,A_list)
    A_out = f(t_out)

    return A_out,t_out,plot_list


def SIR_ABM(ri,rr,rm,T_end=5.0):

    #number of lattice sites
    n = 40

    A = np.zeros((n**2,))

    #initial proportions of susceptible, infected, and recovered agents
    s0 = 0.49
    i0 = 0.01
    r0 = 0.0

    #randomly place susceptible (1), infected (2), and recovered (3) agents
    s_num = int(np.ceil(s0*len(A)))
    i_num = int(np.ceil(i0*len(A)))
    r_num = int(np.ceil(r0*len(A)))
    A[:s_num] = 1
    A[s_num:s_num+i_num] = 2
    A[s_num+i_num:s_num+i_num+r_num] = 3
    #shuffle up
    A = A[np.random.permutation(n**2)]
    #make square
    A = A.reshape(n,n)

    #count number of susceptible, infected, and recovered agents.
    S_num = np.sum(A==1)
    I_num = np.sum(A==2)
    R_num = np.sum(A==3)
    total_num = S_num + I_num + R_num

    #Convert agent counts to proportions
    S = float(S_num)/float(total_num)
    I = float(I_num)/float(total_num)
    R = float(R_num)/float(total_num)

    #nondimensionalized time
    T_final = T_end/rr

    #initialize time
    t = 0

    #track time, agent proportions, and snapshots of ABM in these lists
    t_list = [t]
    S_list = [S]
    I_list = [I]
    R_list = [R]
    A_list = [A]
    #number of snapshots saved
    image_count = 1


    while t_list[-1] < T_final:

        a = rm*(S_num+I_num+R_num) + ri*I_num + rr*I_num
        tau = -np.log(np.random.uniform())/a
        t += tau

        Action = a*np.random.uniform()

        if Action <= rm*(S_num+I_num+R_num):
            #any agent movement
            
            # Select Random agent
            agent_loc = np.where(A!=0)
            agent_ind = np.random.permutation(len(agent_loc[0]))[0]
            loc = (agent_loc[0][agent_ind],agent_loc[1][agent_ind])
            
            #determine status
            agent_state = A[loc]

            ### Determine direction
            dir_select = np.ceil(np.random.uniform(high=4.0))

            #move right
            if dir_select == 1 and loc[0]<n-1:
                if A[(loc[0]+1,loc[1])] == 0:
                    A[(loc[0]+1,loc[1])] = agent_state
                    A[loc] = 0
            #move left
            elif dir_select == 2 and loc[0]>0:
                if A[(loc[0]-1,loc[1])] == 0:
                    A[(loc[0]-1,loc[1])] = agent_state
                    A[loc] = 0
            #move up
            elif dir_select == 3 and loc[1]<n-1:
                if A[(loc[0],loc[1]+1)] == 0:
                    A[(loc[0],loc[1]+1)] = agent_state
                    A[loc] = 0

            #move down                    
            elif dir_select == 4 and loc[1]>0:
                if A[(loc[0],loc[1]-1)] == 0:
                    A[(loc[0],loc[1]-1)] = agent_state
                    A[loc] = 0

        elif (rm*(S_num+I_num+R_num) < Action) and (Action <= rm*(S_num+I_num+R_num) + ri*I_num):
            #infection event
            
            ### Select Random infected agent
            I_ind = np.random.permutation(I_num)[0]
            loc = (np.where(A==2)[0][I_ind],np.where(A==2)[1][I_ind])

            ### Determine direction
            dir_select = np.ceil(np.random.uniform(high=4.0))

            #infect right
            if dir_select == 1 and loc[0]<n-1:
                if A[(loc[0]+1,loc[1])] == 1:
                    A[(loc[0]+1,loc[1])] = 2

            #infect left
            elif dir_select == 2 and loc[0]>0:
                if A[(loc[0]-1,loc[1])] == 1:
                    A[(loc[0]-1,loc[1])] = 2

            #infect up        
            elif dir_select == 3 and loc[1]<n-1:
                if A[(loc[0],loc[1]+1)] == 1:
                    A[(loc[0],loc[1]+1)] = 2

            #infect down
            elif dir_select == 4 and loc[1]>0:
                if A[(loc[0],loc[1]-1)] == 1:
                    A[(loc[0],loc[1]-1)] = 2

        elif (rm*(S_num+I_num+R_num) + ri*I_num < Action) and (Action <= rm*(S_num+I_num+R_num) + ri*I_num + rr*I_num):
            #Recovery event
            
            ### Select Random I
            I_ind = np.random.permutation(I_num)[0]
            loc = (np.where(A==2)[0][I_ind],np.where(A==2)[1][I_ind])
            A[loc] = 3

        #count number of susceptible, infected, recovered agents
        S_num = np.sum(A==1)
        I_num = np.sum(A==2)
        R_num = np.sum(A==3)
        #convert counts to proportions
        S = float(S_num)/float(total_num)
        I = float(I_num)/float(total_num)
        R = float(R_num)/float(total_num)

        #append information to lists
        t_list.append(t)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

        #sometimes save ABM snapshot
        if t_list[-2] < image_count*T_final/20 and t_list[-1] >= image_count*T_final/20:
            A_list.append(np.copy(A))
            image_count+=1

    #interpolation to equispace grid
    t_out = np.linspace(0,T_final,100)

    f = interpolate.interp1d(t_list,S_list)
    S_out = f(t_out)

    f = interpolate.interp1d(t_list,I_list)
    I_out = f(t_out)

    f = interpolate.interp1d(t_list,R_list)
    R_out = f(t_out)


    return S_out,I_out,R_out,t_out,A_list,total_num


def ABM_depict(A_list):
    cmaplist = [(1.0,1.0,1.0,1.0),(0.0,0.0,1.0,1.0),(0.0,1.0,0.0,1.0),(1.0,0.0,0.0,1.0)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, N = 4)

    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.matshow(A_list[6],cmap=cmap)
    ax = fig.add_subplot(1,3,2)
    ax.matshow(A_list[13],cmap=cmap)
    ax = fig.add_subplot(1,3,3)
    im = ax.matshow(A_list[-1],cmap=cmap)
    fig.colorbar(im,ax=ax)
    