# If you're using python 2?
# from __future__ import division
import numpy as np

# Definitions:
# pg: error rate per gate (20ns)
# pr: error rate per rest step (4250ns)
# sg: standard deviation in pg
# sr: standard deviation in pr

def make_difference_plot(x_vec,j1,j2):
    
    # Generates the difference plot
    
    # Function that I'm fitting to
    def pfunc(n_vec,pg,pr):
        
        # Boundary conditions in case you want to fit this via scipy.
        if pg < 0:
            pg == 0
        if pg > 1:
            pg == 1
        
        # P=0.5-(1-2pg)^n(0.5-pr)
        return [100*(0.5-(1-2*pg)**n*(0.5-pr)) for n in n_vec]
    
    # Generate individual curves using preset data defined below
    y_vec1 = pfunc(x_vec,pg_vec[j1],pr_fixed)
    y_vec2 = pfunc(x_vec,pg_vec[j2],pr_fixed)
    
    # Take difference, return
    return [y1-y2 for y1,y2 in zip(y_vec1,y_vec2)]

def make_std_plot(x_vec,j1,j2):
    
    # Generates the std plot
    
    # Function that I'm fitting to    
    def sfunc(n_vec,pg,pr,sg,sr):
        
        # Get variances
        sr2 = sr**2
        sg2 = sg**2
        
        # init return data list
        sm_vec = []
        for n in n_vec:
            
            # Calculate error probability
            # P=0.5-(1-2pg)^n(0.5-pr)
            P = (0.5-(1-2*pg)**n*(0.5-pr))
            
            # Calculate standard deviation of probability of error in
            # one round of error correction. 
            # Calculated assuming independent errors via:
            # s_{P}^2=(dP/dpg)^2s_{pg}^2+(dP/dpr)^2s_{pr}^2
            sp2 = (2*n*(1-2*pg)**(n-1)*(0.5-pr))**2*sg2 + (1-2*pg)**(2*n)*sr2
            
            # Combine variance from distribution of P and variance
            # from the binomial distribution that we take in a fairly crude manner.
            sm_vec.append(np.sqrt(P*(1-P)/N + sp2)*100)
        
        # Return to user
        return sm_vec
    
    # Generate individual curves using preset data defined below
    y_vec1 = sfunc(x_vec,pg_vec[j1],pr_fixed,sg_fixed,sr_fixed)
    y_vec2 = sfunc(x_vec,pg_vec[j2],pr_fixed,sg_fixed,sr_fixed)
    
    # Take difference, return
    return [np.sqrt(y1**2+y2**2) for y1,y2 in zip(y_vec1,y_vec2)]
    
    
# Fit coefficients obtained from fitting pfunc/sfunc to the experimental data,
# via scipy.optimize.curve_fit
# Placed here by hand so you don't have to import the data etc. Feel free to do otherwise.

# Please note, the below is stored as a fraction (i.e. 1=100%), otherwise the math
# doesn't work. 

# We have good reasons to average the rest error probability (pr_fixed)
# and standard deviation (sr_fixed) data, leaving only one free parameter between
# different data sets (I feel this is better).

# However, the standard deviation in gate error rates (sg_fixed) could 
# hypothetically vary as you detune the gates, so this isn't so cut and dry. It can
# also be set to zero without much effect; we only see something at around
# 1e-4.

pg_vec=[0.00098213056186337668, 0.0015671241839269017, 0.0025267070493921332, 0.0034871658898885118, 0.005803222326512896, 0.0070708654552344092, 0.011270378736976753, 0.016452382020151746]
sr_fixed = 0.0101053841726
sg_fixed = 1.1444357016e-05
pr_fixed = 0.1176079925
