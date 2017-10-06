import numpy as np
import scipy
import lmfit
import matplotlib.pyplot as plt


###
# Fitting
###
def fitTwoGaussians(data, nbins = 200, plot=False):
    """
    Fits two gaussians to the data using lmfit
    """
    if plot is True:
        n, bins, patches = plt.hist(data, nbins, facecolor = 'blue', alpha=0.3)
    else:
        n, bins = np.histogram(data, bins = nbins)
    x =bins[0:len(bins)-1] +0.5* np.diff(bins)[0]

    g1 = lmfit.models.GaussianModel(prefix = 'g1_')
    g2 = lmfit.models.GaussianModel(prefix = 'g2_')
    pars = g1.guess(n, x=x)
    pars.update(g2.make_params())

    pars['g2_center'].set(0.02)
    pars['g2_sigma'].set(0.02)
    pars['g2_amplitude'].set(10)

    model = g1 + g2

    init = model.eval(pars, x=x)
    out =  model.fit(n, pars, x=x)
    if plot is True:
        plt.plot(x, init, 'k--')
        plt.plot(x, out.best_fit, 'r-')
        plt.legend(['initial guess', 'fit'], loc = 'best' )
    return out


def fitGaussian(data, nbins, plot=False, bin_threshold=None):
    """
    Fits a single gaussian to the data. Requires 1D array
    """
    if plot is True:
        n, bins, patches = plt.hist(data, nbins, facecolor = 'blue', alpha=0.3)
    else:
        n, bins = np.histogram(data, bins = nbins)

    x =bins[0:len(bins)-1] +0.5* np.diff(bins)[0]


    if bin_threshold is not None:
        n_threshold = bin_threshold * max(n)
        bin_indices = np.where(n>n_threshold)
    else:
        bin_indices = np.arange(len(x))


    model = lmfit.models.GaussianModel()
    pars = model.guess(n[bin_indices], x=x[bin_indices])
    out = model.fit(n[bin_indices], pars, x=x[bin_indices])

    if plot:
        plt.plot(x, model.eval(params=out.params, x=x), '-')
        # if bin_threshold is not None:
        #     plt.axhline(y=bin_threshold * max(n), color='r')

    return out

def find_pre_thresh_based_on_00_only(pre_data, eps, nbins, plot=False):
    """
    """



    out = fitGaussian(pre_data, nbins, plot, bin_threshold=0.3)

    popt = out.params.valuesdict()
    #since we only threshold on a single side we multply by 2
    a = scipy.special.erfcinv(eps)

    dx = np.sqrt(2)*popt['sigma']*a

    pre_thresh_left = popt['center'] - np.sign(np.median(pre_data)-popt['center']) * dx
    pre_thresh_right = popt['center'] + np.sign(np.median(pre_data)-popt['center']) * dx
    if plot is True:
        plt.axvline(x=pre_thresh_left, color='r')
        plt.axvline(x=pre_thresh_right, color='r')

    return popt['center'] , dx



def find_pre_thresh(epsilon, p, bindiff):
    """finds the threshold using the inverse error complement function
    param: epsilon: remaining prob of finding excited state in ssro after selection
    (Based on 2 fit integrals)
            p: ordered dict of lmfit results
            bindiff: normalization factor for scaling counts to gaussian values

    """
    if(p['g1_amplitude'] > p['g2_amplitude']):
        x = epsilon * (p['g1_amplitude'] + p['g2_amplitude']) / (p['g2_amplitude']) / bindiff
        a = scipy.special.erfcinv(x)
        #the first is the higher gaussian
        if(p['g1_center'] < p['g2_center']):
            #the higher is before the lower
             t = p['g2_center'] - a *  p['g2_sigma']
        else:
             t = p['g2_center'] + a *  p['g2_sigma']
    else:
        #the second() is the higher gaussian
        x = epsilon * (p['g1_amplitude'] + p['g2_amplitude']) / (p['g1_amplitude']) / bindiff
        a = scipy.special.erfcinv(x)
        if(p['g1_center'] > p['g2_center']):
            #the higher is before the lower
             t = p['g1_center'] - a *  p['g1_sigma']
        else:
             t = p['g1_center'] + a *  p['g1_sigma']
    if(x > 0.5):
        print('WARNING Threshold unreliable, epsilon value too large, tries to get %d percent of g2' % (x * 100))
    return t


def calc_entanglement_threshold(radius,
                                data_points,
                                normalize=True,
                                plot=False,
                                center_threshold_offset = 0.0,
                                save_dict = None,
                                zerozero_oneone=False):
    """
    Expects list of data sets in order 00 01 10 11
    returns the two thresholds and populations
    if normalize is true: returns the expected rho element instead of remaining part of single gaussian
    """
    g = [fitGaussian(Q, 200, plot=plot, bin_threshold=0.3) for Q in data_points]

    if(save_dict is not None):
        save_dict['entanglement_threshold_gaussian_fits'] = g

    if zerozero_oneone is True:
        center_threshold = np.mean(
            [g[1].best_values['center'], g[2].best_values['center']]) \
            + center_threshold_offset
    else:
        center_threshold = np.mean(
            [g[0].best_values['center'], g[3].best_values['center']]) \
            + center_threshold_offset

    threshold_left = center_threshold - radius
    threshold_right = center_threshold + radius

    n = np.zeros(4)
    for i in range(0,4):
        if i == 0 or i == 3:
            #edges
            n[i] =  np.abs(0.5 + np.sign(center_threshold - g[i].best_values['center']) * 0.5 * scipy.special.erf((g[i].best_values['center'] - threshold_left) / g[i].best_values['sigma'] * (1 / np.sqrt(2))) \
            - (0.5 + np.sign(center_threshold - g[i].best_values['center']) * 0.5 * scipy.special.erf((g[i].best_values['center'] - threshold_right) / g[i].best_values['sigma'] * (1 / np.sqrt(2)))))
        else:
            #get the centers
            # if(threshold_right < g[i].best_values['center']) or (threshold_left > g[i].best_values['center']):
            #     print('Warning threshold too narrow incorrect populations calculated')
            n[i] =  1-(0.5 - 0.5 * scipy.special.erf((g[i].best_values['center'] - threshold_left) / g[i].best_values['sigma'] * (1 / np.sqrt(2))) \
                    +  0.5 + 0.5 * scipy.special.erf((g[i].best_values['center'] - threshold_right) / g[i].best_values['sigma'] * (1 / np.sqrt(2))) )

    if normalize:
        n = n / sum(n)
    if plot:
        plt.axvline(x=center_threshold, color='r')
        plt.axvline(x=threshold_left, color='r')
        plt.axvline(x=threshold_right, color='r')
        labels = ['00', '01', '10', '11']
        plt.legend(['n_%s %.3f' % (labels[i], n[i]) for i in range(0,len(n))], loc='best')
        plt.title('Remaining Populations and thresholds for entangling cal points')

    return [threshold_left, center_threshold, threshold_right], n


def calc_entanglement_threshold_percentage(
                                percentage,
                                data_points,
                                normalize=True,
                                plot=False,
                                save_dict = None,
                                zerozero_oneone=False,
                                nbins = 200):
    """
    Expects list of data sets in order 00 01 10 11
    returns the two thresholds and populations
    if normalize is true: returns the expected rho element instead of remaining part of single gaussian
    """
    gaussians = [fitGaussian(Q, nbins, plot=plot, bin_threshold=0.3) for Q in data_points]

    if(save_dict is not None):
        save_dict['entanglement_threshold_gaussian_fits'] = gaussians

    if zerozero_oneone is False:
        center_threshold = np.mean([gaussians[1].best_values['center'], gaussians[2].best_values['center']])
    else:
        center_threshold = np.mean([gaussians[0].best_values['center'], gaussians[3].best_values['center']])

    sigma = gaussians[1].best_values['sigma']
    threshold_left = center_threshold - sigma
    threshold_right = center_threshold + sigma

    def get_percentages_from_thresholds(threshold_left,
                                        threshold_right,
                                        gaussians
                                        ):
        n = np.zeros(4)

        if threshold_left<threshold_right:
            for i in range(0,4):
                if i == 0 or i == 3:
                    #edges
                    n[i] =  np.abs(0.5 + np.sign(center_threshold - gaussians[i].best_values['center']) * 0.5 * scipy.special.erf((gaussians[i].best_values['center'] - threshold_left) / gaussians[i].best_values['sigma'] * (1 / np.sqrt(2))) \
                    - (0.5 + np.sign(center_threshold - gaussians[i].best_values['center']) * 0.5 * scipy.special.erf((gaussians[i].best_values['center'] - threshold_right) / gaussians[i].best_values['sigma'] * (1 / np.sqrt(2)))))
                else:
                    #get the centers
                    n[i] =  1-(0.5 - 0.5 * scipy.special.erf((gaussians[i].best_values['center'] - threshold_left) / gaussians[i].best_values['sigma'] * (1 / np.sqrt(2))) \
                            +  0.5 + 0.5 * scipy.special.erf((gaussians[i].best_values['center'] - threshold_right) / gaussians[i].best_values['sigma'] * (1 / np.sqrt(2))) )

        else:
            pass
        # print 'threshold left %s' %threshold_left
        # print 'threshold right %s' %threshold_right
        # print 'n is %s' %n

        return n

    def cost_function_percentage_0110(x, percentage, gaussians):
        threshold_left = x[0]
        threshold_right = x[1]
        percentages = get_percentages_from_thresholds(
                        threshold_left,
                        threshold_right,
                        gaussians)
        return np.abs(np.sum(percentages)-4*percentage) \
            + percentages[0] + percentages[3]

    def cost_function_percentage_0011(x, percentage, gaussians):
        threshold_left = x[0]
        threshold_right = x[1]
        percentages = get_percentages_from_thresholds(
                        threshold_left,
                        threshold_right,
                        gaussians)
        return np.abs(np.sum(percentages)-4*percentage) \
            + percentages[1] + percentages[2]


    cons = ({'type': 'ineq', 'fun': lambda x:  x[1] - x[0]})

    if zerozero_oneone is False:
        cost_function_percentage = cost_function_percentage_0110
    else:
        cost_function_percentage = cost_function_percentage_0011

    res = scipy.optimize.minimize(
        cost_function_percentage,
        [threshold_left, threshold_right],
        args=(percentage, gaussians),
        method='Nelder-Mead',
        # constraints=cons,
        # bounds=[(center_threshold - 3*gaussians[1].best_values['sigma'],
        #          center_threshold + 3*gaussians[1].best_values['sigma']),
        #         (center_threshold - 3*gaussians[1].best_values['sigma'],
        #          center_threshold + 3*gaussians[1].best_values['sigma'])],
        tol=0.01,
        options={'maxiter':100,
                 'initial_simplex':np.array(
                    [[threshold_left, threshold_right],
                     [threshold_left-0.1*sigma, threshold_right+0.1*sigma],
                     [threshold_left+0.1*sigma, threshold_right-0.1*sigma]])})

    [threshold_left, threshold_right] = res.x

    percentages = get_percentages_from_thresholds(threshold_left,
                                        threshold_right,
                                        gaussians
                                        )

    center_threshold = np.mean([threshold_left, threshold_right])

    if plot:
        plt.axvline(x=center_threshold, color='r')
        plt.axvline(x=threshold_left, color='r')
        plt.axvline(x=threshold_right, color='r')
        labels = ['00', '01', '10', '11']
        plt.legend(['n_%s %.3f' % (labels[i], percentages[i])
                   for i in range(0,len(percentages))], loc='best')
        plt.title('Remaining Populations and thresholds for entangling cal points')

    return [threshold_left, center_threshold, threshold_right], percentages


#####rotation functions

def rotate_data(theta, I, Q, list_shape=False):
    """
    Rotates the data, expects [n_single_shots,n_points] data
    theta should be in radians
    arg: list_shape set to true if you have a list of arrays as data
    """
    if list_shape:
        Inew = [I[i] * np.cos(theta) - Q[i] * np.sin(theta) for i in range(len(I))]
        Qnew = [I[i] * np.sin(theta) + Q[i] * np.cos(theta) for i in range(len(I))]
    else:

        Inew = I * np.cos(theta) - Q * np.sin(theta)
        Qnew = I * np.sin(theta) + Q * np.cos(theta)
    return Inew, Qnew

def calc_optimal_rotation_angle(x_0, x_1):
    """
    REturns optimal angle that Rotates the data to one quadrature(the second dimension), expects [n_single_shots,n_points] data
    x_0, x_1 should be vectors of the me
    ans of two data sets[I,Q] will rotate to Q
    """
    diff =  x_0 - x_1
    return np.arctan(diff[0] / diff[1])

def rotate_optimally(I,Q, x_0, x_1, list_shape =False):
    """
    Combines the above two functions(rotate_data nd calc_optimal rotation angle) into one.
    Optimally rotates a set of data to the Q-Quadrature.  expects [n_single_shots,n_points] data
    x_0, x_1 should be vectors of the means of two data sets[I,Q] will rotate to Q
    """
    theta = calc_optimal_rotation_angle(x_0, x_1)
    return rotate_data(theta, I, Q, list_shape)


def find_optimal_rotation_angle_based_cal_points_means(data_array_list):
    """
    Tries to rotate optimally by minimizing the difference between the means of each data set
    data: list of tuples (I, Q) of arrays of voltages of each cal point
    """
    theta = scipy.optimize.minimize(calc_mean_difference, 0, args=(data_array_list), method="Nelder-Mead")
    return theta['x'][0]


def find_optimal_rotation_angle_by_sigma(I, Q):
    theta = scipy.optimize.minimize(calc_sigma, 0, args=(I, Q), method="Nelder-Mead")
    return theta['x'][0]

def calc_sigma(theta,I,Q):
    """ calculates standard deviation on data or on a gaussian fit of the data"""
    Inew, Qnew = rotate_data(theta, I, Q)
    sigma = np.std(Inew)
    return sigma

def calc_mean_difference(theta, data_array_list):
    """
    Calculates the difference between the means of I-quadrature data of each element in the list
    """
    meansI = []
    for I, Q in data_array_list:
        Inew, Qnew = rotate_data(theta, I, Q)
        meansI.append(np.mean(Inew))
    return np.std(meansI)









def reject_outliers_IQ(I, Q, n_sweeps):
    """
    rejects the first two points of each n_avg and returns a numpy array like the original dat
    """
    #skip the first two points of every n_avg
    n_points = np.shape(I)[0]
    n_avg = n_points / n_sweeps
    n_avg = int(n_avg)
    ind = []
    for i in range(n_avg):
        ind.extend([j + i * n_sweeps for j in range(2,n_sweeps)])
    Inew = I[ind,:]
    Qnew = Q[ind,:]
    return Inew, Qnew

def find_optimal_threshold(Q_0, Q_1, ranges=(-0.2,0.2) , nbins=1000, plot=False, legend=['00','01']):
    """
    finds an optimal threshold for distinguishing two histograms
    """
    n_00, bin_edges = np.histogram(Q_0, bins=nbins, range=ranges, density=True)
    n_10, bin_edges = np.histogram(Q_1, bins=nbins, range=ranges, density=True)
    delta = (bin_edges[1]-bin_edges[0])
    cum_00 = np.cumsum(n_00) * delta
    cum_10 = np.cumsum(n_10) * delta
    diff = abs(cum_10-cum_00)
    threshold = np.argmax(diff)
    threshold_voltage = bin_edges[threshold] + delta/2

    if plot is True:
        plt.plot(bin_edges[0:len(bin_edges)-1], cum_00)
        plt.plot(bin_edges[0:len(bin_edges)-1], cum_10)
        plt.axvline(x=threshold_voltage, color='r')
        plt.legend(legend, loc='best')

    return diff[threshold], threshold_voltage

def select_data(indices, I, Q, list_shape = False):
    """selects data based on a set of indices.
     Returns a list of arrays since each trace will now have a different amount of points
     """

    if list_shape:
        n = len(I)
        Inew = [I[i][indices[i]] for i in range(n)]
        Qnew = [Q[i][indices[i]] for i in range(n)]
    else:
        n = np.shape(I)[1]
        Inew = [I[indices[i],i] for i in range(n)]
        Qnew = [Q[indices[i],i] for i in range(n)]
    return Inew, Qnew

def select_single_data(indices, Q, list_shape = False):
    """selects data based on a set of indices. But now only a single quadrature
     Returns a list of arrays since each trace will now have a different amount of points
     """

    if list_shape:
        n = len(Q)
        Qnew = [Q[i][indices[i]] for i in range(n)]
    else:
        n = np.shape(Q)[1]
        Qnew = [Q[indices[i],i] for i in range(n)]
    return Qnew

def select_from_double_threshold(left_border, right_border, Q, inside=True):
    """
    Expects Q to be lists of arrays. Only selects a single quadrature since you should rotate first.
    params
        inside: inside or outside the borders
    returns the selected array and a list of arrays of indices
    """
    if inside:
        ind = [(Q[i] > left_border) & (Q[i] < right_border) for i in range(len(Q))]
    else:
        ind=  [(Q[i] < left_border) & (Q[i] > right_border) for i in range(len(Q))]
    return [Q[i][ind[i]] for i in range(len(Q))], ind

def getOptimumThresholdsTwoQubits(Q_cal, nbins=400, plot=False, legend=['00','10', '01', '11'], swap_01_10 = True):
    """
    Finds the optimal (4) thresholds based on a list of four calibration points arrays of the 00 01 10 and 11 states
    It puts the thresholds at the voltages where neighbouring cumulative histograms have the largest difference.
    """
    domain = (min(map(np.min,Q_cal)), max(map(np.max, Q_cal)))
    n = [[] for i in range(4)]
    c = [[] for i in range(4)]
    thresholds = [[] for i in range(3)]
    threshold_voltages = [[] for i in range(3)]
    if swap_01_10:
        i_01 = 2
        i_10 = 1
    else:
        i_01 = 1
        i_10 = 2
    n[0], bin_edges = np.histogram(Q_cal[0], bins=nbins, range=domain, density=True)
    n[i_01], bin_edges = np.histogram(Q_cal[1], bins=nbins, range=domain, density=True)
    n[i_10], bin_edges = np.histogram(Q_cal[2], bins=nbins, range=domain, density=True)
    n[3], bin_edges = np.histogram(Q_cal[3], bins=nbins, range=domain, density=True)
    delta = (bin_edges[1]-bin_edges[0])
    for i in range(len(n)):
        c[i] = np.cumsum(n[i]) * delta
        if(i > 0):
            diff = abs(c[i-1]-c[i])
            thresholds[i-1] = np.argmax(diff)
            threshold_voltages[i-1] = bin_edges[thresholds[i-1]] + delta/2
            if plot:
                a = 0;#plt.axvline(x=threshold_voltages[i-1], color='r')
        if plot:
            colors = ['blue', 'red', 'green', 'teal']
            plt.plot(bin_edges[0:len(bin_edges)-1], c[i], color=colors[i])
    if plot:
        plt.legend(legend, loc='best')
        k=0
        for tes in threshold_voltages:
            print (colors[k])
            #plt.axvline(tes, color=colors[k])
            k = k+1

    return thresholds, threshold_voltages

def getOptimum3binThresholdsTwoQubits(Q_cal, nbins=400, plot=False):
    """
    Finds the optimal thresholds for 3 bins
    """
    domain = (min(map(np.min,Q_cal)), max(map(np.max, Q_cal)))
    n = [[] for i in range(4)]
    c = [[] for i in range(4)]
    thresholds = [[] for i in range(2)]
    threshold_voltages = [[] for i in range(2)]

    n[0], bin_edges = np.histogram(Q_cal[0], bins=nbins, range=domain, density=True)
    #all states except 00
    n[1], bin_edges = np.histogram(np.concatenate([Q_cal[1],Q_cal[2],Q_cal[3]]), bins=nbins, range=domain, density=True)
    #all states except 11
    n[2], bin_edges = np.histogram(np.concatenate([Q_cal[0],Q_cal[1],Q_cal[2]]), bins=nbins, range=domain, density=True)

    n[3], bin_edges = np.histogram(Q_cal[3], bins=nbins, range=domain, density=True)
    delta = (bin_edges[1]-bin_edges[0])

    #get cumsums from histograms
    for i in range(len(n)):
        c[i] = np.cumsum(n[i]) * delta

    #compare 00 to all other states
    diff = abs(c[0]-c[1])
    thresholds[0] = np.argmax(diff)
    threshold_voltages[0] = bin_edges[thresholds[0]] + delta/2
    #now do the same for 11
    diff = abs(c[3]-c[2])
    thresholds[1] = np.argmax(diff)
    threshold_voltages[1] = bin_edges[thresholds[1]] + delta/2


    return thresholds, threshold_voltages


def getCountsFromThresholds(Q, thresholds):
    """
    Returns the counts in each bin defined by a number of n ordered thresholds from left to right
    if swap 01 10 is true it will swap the counts in the middle bins.
    """
    #check for ordering!
    temp = thresholds[0]
    counts = []
    #get the leftmost count
    if thresholds[0] < thresholds[1]:
        counts.append(sum(Q < thresholds[0]))
    else:
        counts.append(sum(Q > thresholds[0]))
    #get intermediate counts
    for i in range(1,len(thresholds)):
        #check the order of the thresholds
        if thresholds[0] < thresholds[1]:
            counts.append(sum((Q < thresholds[i]) & (Q > thresholds[i-1])))
        else:
            counts.append(sum((Q > thresholds[i]) & (Q < thresholds[i-1])))
    #get the rightmost bin
    if thresholds[0] < thresholds[1]:
        counts.append(sum(Q > thresholds[i]))
    else:
        counts.append(sum(Q < thresholds[i]))

    return counts