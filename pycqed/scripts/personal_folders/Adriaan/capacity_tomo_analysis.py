#!/usr/bin/env python3

##########################################################################
# MODULE IMPORTS
##########################################################################

import numpy as np              # for fast numerical operations
import matplotlib.pyplot as plt  # for plots
import argparse                 # for command line arguments
import h5py                     # to read the HDF5 file format
import scipy.optimize as opt    # for evaluating the maximum in the bound
import sys                      # for determining smallest positive float
#     (lower bound for eta in the optimization
#     in the function "bound", see variable
#     "bnd")

##########################################################################
# PARAMETERS
##########################################################################

# If you set any of the following parameters to a value outside of [0,1], the
# user will be prompted for the value of that variable (see ACTUAL PROGRAM
# below)
q = 0.9         # preparation quality
ppass_x = 1/2  # assumed probability that the error rate in X is below e_x
ppass_z = 1/2  # assumed probability that the error rate in Z is below e_z

# Note: Luckily, it turns out that for reasonable values of N (total number of
# qubits), the capacity bound is extremely robust against unfavorable choices of
# ppass_x and ppass_z, so that setting them to 1/2 (or even a _way_ smaller
# value) is safe

# define bounds for the optimization problem in the function "bound" below
# sys.float_info[3] is the smallest positive float, None means no bound
bnd = ((sys.float_info[3], None),)

# During the optimization below, the optimization algorithm will pick
# invalid values as an argument for the log function (not exactly sure why,
# because it shouldn't, given the bounds I've set).
# To avoid warnings for that, I set the following:
np.seterr(invalid='ignore')

# Determine area to be plotted
epsExponentMin = -9  # start epsilon at 10^exponentmin
epsExponentMax = -1  # stop epsilon at 10^exponentmax

# set number of points calculated for each plot
epsSteps = 100

# determine number of shots for the green (1) and red (2) plot
shots1 = 10**5
shots2 = 5000

# set the difference in the error rates for the middle (1) and right (2) plots
diff1 = 0.01
diff2 = -0.01

##########################################################################
# FUNCTION DEFINITIONS
##########################################################################

# define function that extracts the dataset from an hdf5 file into a NumPy
# array


def extract(filename):
    # create file object that hooks to the file
    file = h5py.File(filename, 'r')
    dataset = file['Experimental Data']['Data']  # get dataset from that file
    return np.array(dataset)

# define function that takes an extracted NumPy array and returns the tuple
# (N, n, k, e_x, e_z), where
#     N:   total number of rounds,     e_x: error rate in X,
#     n:   number of X-rounds,         e_z: error rate in Z
#     k:   number of Z-rounds,


def getParameters(array):
    N = len(array)
    basis = np.array(array[:, 0], dtype=bool)
    prep = np.array(array[:, 1], dtype=bool)
    meas = np.array(array[:, 2], dtype=bool)
    error = np.bitwise_xor(prep, meas)
    n, k, xErr, zErr = 0, 0, 0, 0
    for i in range(N):
        if basis[i] == 0:
            zErr += error[i]
            k += 1
        elif basis[i] == 1:
            xErr += error[i]
            n += 1
    return (N, n, k, xErr / n, zErr / k)

# define binary entropy function


def h(p):
    if p == 0 or p == 1:
        return 0
    else:
        return - p * np.log2(p) - (1-p) * np.log2(1-p)

# define delta (deformed smoothing parameter)


def delta(epsilon, ppass_x, ppass_z, eta):
    return ( np.sqrt(epsilon/2) - eta ) / \
           (3 + 1/np.sqrt(ppass_z) + 4/np.sqrt(ppass_x))

# define mu_x (error rate correction term in x)


def mux(n, k, epsilon, ppass_x, ppass_z, eta):
    return np.sqrt(k*(n+1) / (n**2 * (n+k))
                    * np.log(1 / delta(epsilon, ppass_x, ppass_z, eta)))

# define mu_z (error rate correction term in z)


def muz(n, k, epsilon, ppass_x, ppass_z, eta):
    return np.sqrt(n*(k+1) / (k**2 * (n+k))
                    * np.log(1 / delta(epsilon, ppass_x, ppass_z, eta)))

# define the objective function that is optimized in the bound


def f(n, k, ex, ez, q, epsilon, ppass_x, ppass_z, eta):
    return (n+k) * (q - h(ex + mux(n, k, epsilon, ppass_x, ppass_z, eta))
                    - h(ez + muz(n, k, epsilon, ppass_x, ppass_z, eta)) ) \
        - 2*np.log( 2 / (delta(epsilon, ppass_x, ppass_z, eta))**2 ) \
        - 4*np.log(1 / eta) - 2

# define function that evaluates our bound on the one-shot quantum capacity


def bound(n, k, ex, ez, q, epsilon, ppass_x, ppass_z):
    res = opt.minimize(lambda x: -f(n, k, ex, ez, q, epsilon, ppass_x, ppass_z, x),
                       10**(-6), method='L-BFGS-B', bounds=bnd)
    return f(n, k, ex, ez, q, epsilon, ppass_x, ppass_z, res.x[0])

##########################################################################
# ACTUAL PROGRAM
##########################################################################

if __name__ == '__main__':

    # handle command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('filename', help='The hdf5 file to read from')
    # args = parser.parse_args()
    # fileName = args.filename
    # fileName='D:\\Experiments\\1607_Qcodes_5qubit\\data\\20160922\\192707_Capacity_tomo_idle_time_3e-07s_base_ZX\\192707_Capacity_tomo_idle_time_3e-07s_base_ZX.hdf5'
    # fileName='D:\\Experiments\\1607_Qcodes_5qubit\\data\\20160922\\193525_Capacity_tomo_idle_time_1e-06s_base_ZX\\193525_Capacity_tomo_idle_time_1e-06s_base_ZX.hdf5'
    # fileName='D:\\Experiments\\1607_Qcodes_5qubit\\data\\20160922\\194147_Capacity_tomo_idle_time_3e-07s_base_ZX\\194147_Capacity_tomo_idle_time_3e-07s_base_ZX.hdf5'
    # fileName='D:\\Experiments\\1607_Qcodes_5qubit\\data\\20160922\\212731_Capacity_tomo_idle_time_6e-07s_base_ZX\\212731_Capacity_tomo_idle_time_6e-07s_base_ZX.hdf5'
    fileName='D:\\Experiments\\1607_Qcodes_5qubit\\data\\20160922\\231341_Capacity_tomo_idle_time_1e-06s_base_ZX\\231341_Capacity_tomo_idle_time_1e-06s_base_ZX.hdf5'



    # load data as per command line argument


    array = extract(fileName)

    # get the parameters (N, n, k, e_x, e_z) that are determined in the
    # experiment, where
    #     N:   total number of rounds
    #     n:   number of X-rounds,
    #     k:   number of Z-rounds,
    #     e_x: error rate in X,
    #     e_z: error rate in Z
    (N, n, k, e_x, e_z) = getParameters(array)
    print("N (total number of rounds): " + str(N))
    print("n (number of X-rounds):     " + str(n))
    print("k (number of Z-rounds):     " + str(k))
    print("e_x (error rate in X):      " + str(e_x))
    print("e_z (error rate in Z):      " + str(e_z))

    # Ask the user for a value for the preparation quality q (if not already set
    # to a value in [0,1] in the PARAMETERS section above)
    while q < 0 or q > 1:
        q = float(input("Enter a value for q (preparation quality): "))
        if q < 0 or q > 1:
            print("Value for q must be between 0 and 1!")
        else:
            break

    # Ask the user for a value for the passing probability ppass_x (if not
    # already set to a value in [0,1] in the PARAMETERS section above)
    while ppass_x < 0 or ppass_x > 1:
        ppass_x = float(input("Enter a value for ppass_x (probability that " +
                              "the error rate is below e_x): "))
        if ppass_x < 0 or ppass_x > 1:
            print("Value for ppass_x must be between 0 and 1!")

    # Ask the user for a value for the passing probability ppass_z (if not
    # already set to a value in [0,1] in the PARAMETERS section above)
    while ppass_z < 0 or ppass_z > 1:
        ppass_z = float(input("Enter a value for ppass_z (probability that " +
                              "the error rate is below e_z): "))
        if ppass_z < 0 or ppass_z > 1:
            print("Value for ppass_z must be between 0 and 1!")

    # create figure instance (necessary because we have several subplots)
    fig = plt.figure(1)

    # set the array of values of epsilon for which you want to calculate a data
    # point in the plot (same for all 9 plots)
    eps = np.logspace(epsExponentMin, epsExponentMax, num=epsSteps)

    # LEFT PLOT for the error rates as in the actual data

    # calculate y-values (bounds on the one-shot capacity) for N as in the data,
    # N as in the variable shots1 (see PARAMETERS above) (1) and N as in the
    # variable shots2 (2)
    rate = [1/(n+k) * bound(n, k, e_x, e_z, q, e, ppass_x, ppass_z)
            for e in eps]
    rate1 = [1/(shots1) * bound(shots1/2, shots1/2, e_x, e_z, q, e, ppass_x, ppass_z)
             for e in eps]
    rate2 = [1/(shots2) * bound(shots2/2, shots2/2, e_x, e_z, q, e, ppass_x, ppass_z)
             for e in eps]
    # create subplot instance (axis) for the three plots
    ax1 = plt.subplot(131)
    # create the three plots
    ax1.plot(eps, rate, eps, rate1, eps, rate2)
    # set subplot title
    plt.title('$e_x$ = ' + "{:.2%}".format(e_x) +
              ', $e_z$ = ' + "{:.2%}".format(e_z))
    # make x-axis logarithmic
    plt.xscale('log')
    # put labels on the x- and y-axis
    plt.xlabel('decoding error tolerance $\\varepsilon$')
    plt.ylabel('lower bound on rate $\\frac{1}{N}Q^{\\varepsilon}(\Lambda)$')

    # PLOT IN THE MIDDLE for error rates that differ by diff1 (see the
    # variable in the PARAMETERS section above)

    # shift the error rates by diff1 (this is the only difference to the plot(s)
    # above)
    e_x1 = e_x + diff1
    e_z1 = e_z + diff1
    # calculate y-values (bounds on the one-shot capacity) for N as in the data,
    # N as in the variable shots1 (see PARAMETERS above) (1) and N as in the
    # variable shots2 (2)
    rate = [1/(n+k) * bound(n, k, e_x1, e_z1, q, e, ppass_x, ppass_z)
            for e in eps]
    rate1 = [1/(shots1) * bound(shots1/2, shots1/2, e_x1, e_z1, q, e, ppass_x, ppass_z)
             for e in eps]
    rate2 = [1/(shots2) * bound(shots2/2, shots2/2, e_x1, e_z1, q, e, ppass_x, ppass_z)
             for e in eps]
    # create subplot instance (axis) for the three plots
    ax2 = plt.subplot(132)
    # create the three plots
    ax2.plot(eps, rate, eps, rate1, eps, rate2)
    # set subplot title
    plt.title("diff: " + "{:.2%}".format(diff1))
    # make x-axis logarithmic
    plt.xscale('log')

    # PLOT ON THE RIGHT for error rates that differ by diff2 (see the variable
    # in the PARAMETERS section above)

    # shift the error rates by diff2 (this is the only difference to the plot(s)
    # above)
    e_x2 = e_x + diff2
    e_z2 = e_z + diff2
    # calculate y-values (bounds on the one-shot capacity) for N as in the data,
    # N as in the variable shots1 (see PARAMETERS above) (1) and N as in the
    # variable shots2 (2)
    rate = [1/(n+k) * bound(n, k, e_x2, e_z2, q, e, ppass_x, ppass_z)
            for e in eps]
    rate1 = [1/(shots1) * bound(shots1/2, shots1/2, e_x2, e_z2, q, e, ppass_x, ppass_z)
             for e in eps]
    rate2 = [1/(shots2) * bound(shots2/2, shots2/2, e_x2, e_z2, q, e, ppass_x, ppass_z)
             for e in eps]
    # create subplot instance (axis) for the three plots
    ax3 = plt.subplot(133)
    # create the three plots
    ax3.plot(eps, rate, eps, rate1, eps, rate2)
    # put a legend for the three plots (indicating the different values of N)
    ax3.legend(['$N$ as in data', '$N$ = ' + str(shots1), '$N$ = ' +
                str(shots2)], loc=10)
    # set subplot title
    plt.title("diff: " + "{:.2%}".format(diff2))
    # make x-axis logarithmic
    plt.xscale('log')

    # show the plots
    plt.show()
