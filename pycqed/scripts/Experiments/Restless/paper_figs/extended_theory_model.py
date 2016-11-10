import numpy as np
from math import exp, sqrt
from scipy import optimize

# Set experimental parameters (ta,tb fixed by Leo's measurement model + my
# interpretation thereof)
ta = 3.68  # Time between fixed measurement point and Cliffords
tb = 0.57  # Time between Cliffords and the fixed measurement point
tgate = 0.020  # Gate time
N = 8000  # Number of iterations

# The following functions are formulas for various terms (calculated analytically).
# I haven't yet completely explained this in the notes (doing that now), hope that
# these make some sense regardless. The idea is that we have the error rate p_e
# which is given as a function p_e(P_0(p_s^0(T_1),p_cl(T_1)),P_1(p_s^1(T_1),p_cl(T_1))),
# and so dpe/dT_1 can be written as a long combination of partial derivatives,
# which we calculate independently (because putting everything together would be
# horrible).

# Note that I treat pcl as a free variable here; in reality
# pcl = pcl_0+1-1/6(3+2exp(-tgate/2*T1)+exp(-tgate/T1)),
# and pcl_0 is the thing we need to adjust, but it is easier
# to ignore the T1 dependence and just fix pcl.
# However, we do not do this in calculating dpcl/dT1 (as you can see in the function below)
# Note: this then can allows for pcl smaller than the T1 limit, in which case
# we will get bad results for the calculated standard deviation.


def ps0(T1):
    return 1-exp(-tb/T1)


def ps1(T1):
    return (1-exp(-ta/T1))*exp(-tb/T1) + 0.02


def P0(T1, pcl, ncl):
    temp_val = ps0(T1)
    return temp_val + 0.5*(1-(1-2*pcl)**ncl)*(1-2*temp_val)


def P1(T1, pcl, ncl):
    temp_val = ps1(T1)
    return temp_val + 0.5*(1-(1-2*pcl)**ncl)*(1-2*temp_val)


def pe(T1, pcl, ncl):
    P0_val = P0(T1, pcl, ncl)
    P1_val = P1(T1, pcl, ncl)
    return (P0_val+P1_val-2*P0_val*P1_val)/(2-P0_val-P1_val)

#------------------


def dps0dT(T1):
    return -tb/(T1**2)*exp(-tb/T1)


def dps1dT(T1):
    return tb/(T1**2)*exp(-tb/T1) - (ta+tb)/(T1**2)*exp(-(ta+tb)/T1)


def dpcldT(T1):
    return -(1.875*tgate/(6*T1**2))*(exp(-1.875*tgate/(2*T1**2))+exp(-1.875*tgate/T1))

#------------------


def dP0dps0(T1, pcl, ncl):
    return (1-2*pcl)**ncl


def dP1dps1(T1, pcl, ncl):
    return (1-2*pcl)**ncl


def dP0dpcl(T1, pcl, ncl):
    return ncl/2*(1-2*pcl)**(ncl-1)*(1-2*ps0(T1))


def dP1dpcl(T1, pcl, ncl):
    return ncl/2*(1-2*pcl)**(ncl-1)*(1-2*ps1(T1))

#-----------------


def dP0dT(T1, pcl, ncl):
    return dP0dps0(T1, pcl, ncl)*dps0dT(T1) + dP0dpcl(T1, pcl, ncl)*dpcldT(T1)


def dP1dT(T1, pcl, ncl):
    return dP1dps1(T1, pcl, ncl)*dps1dT(T1) + dP1dpcl(T1, pcl, ncl)*dpcldT(T1)


def dpedP0(T1, pcl, ncl):
    P0_val = P0(T1, pcl, ncl)
    P1_val = P1(T1, pcl, ncl)
    term1 = (1-2*P1_val)/(2-P0_val-P1_val)
    term2 = (P0_val+P1_val-2*P0_val*P1_val)/(2-P0_val-P1_val)**2
    return term1+term2


def dpedP1(T1, pcl, ncl):
    P0_val = P0(T1, pcl, ncl)
    P1_val = P1(T1, pcl, ncl)
    term1 = (1-2*P0_val)/(2-P0_val-P1_val)
    term2 = (P0_val+P1_val-2*P0_val*P1_val)/(2-P0_val-P1_val)**2
    return term1+term2

#-----------------


def dpedT(T1, pcl, ncl):
    return dpedP0(T1, pcl, ncl)*dP0dT(T1, pcl, ncl) + dpedP1(T1, pcl, ncl)*dP1dT(T1, pcl, ncl)


def dpedpcl(T1, pcl, ncl):
    dP0dpcl_val = dP0dpcl(T1, pcl, ncl)
    dP1dpcl_val = dP1dpcl(T1, pcl, ncl)
    P0_val = P0(T1, pcl, ncl)
    P1_val = P1(T1, pcl, ncl)
    term_1 = (dP0dpcl_val+dP1dpcl_val-2 *
              (P0_val*dP1dpcl_val+P1_val*dP0dpcl_val))/(2-P0_val-P1_val)
    term_2 = (P0_val+P1_val-2*P0_val*P1_val) / \
        (2-P0_val-P1_val)**2 * (dP0dpcl_val+dP1dpcl_val)
    return term_1+term_2


def stder(T1, sT1, pcl, ncl):
    pe_val = pe(T1, pcl, ncl)
    dpedT_val = dpedT(T1, pcl, ncl)
    varpe = sT1**2*dpedT_val**2
    return sqrt((N-1)/N*varpe + pe_val*(1-pe_val)/N)

def pe_for_fit(ncls, T1, pcl):
    return [pe(T1,pcl,ncl) for ncl in ncls]


# We need to fit for pcl. As T1 here just determines the SPAM,
# we can just to a fit to T1 and pcl to extract the pcl value
# pcl_vec = []
# def pe_for_fit(ncls, T1, pcl):
#     return [pe(T1,pcl,ncl) for ncl in ncls]
# T1_vec = []
# for err_vec, std_vec in zip(err_vecs,std_vecs):
#     T1,pcl = optimize.curve_fit(pe_for_fit,ncl_vec,err_vec,[20,0.01])[0]
#     T1_vec.append(T1)
#     pcl_vec.append(pcl)
