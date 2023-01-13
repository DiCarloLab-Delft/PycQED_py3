"""
File containing analysis for LRU-related experiments.
"""
import os
import pycqed.analysis_v2.base_analysis as ba
import matplotlib.pyplot as plt
import numpy as np
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.measurement.hdf5_data as h5d
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors
import itertools


def rotate_and_center_data(I, Q, vec0, vec1, phi=0):
    vector = vec1-vec0
    angle = np.arctan(vector[1]/vector[0])
    rot_matrix = np.array([[ np.cos(-angle+phi),-np.sin(-angle+phi)],
                           [ np.sin(-angle+phi), np.cos(-angle+phi)]])
    proc = np.array((I, Q))
    proc = np.dot(rot_matrix, proc)
    return proc.transpose()

def _calculate_fid_and_threshold(x0, n0, x1, n1):
    """
    Calculate fidelity and threshold from histogram data:
    x0, n0 is the histogram data of shots 0 (value and occurences),
    x1, n1 is the histogram data of shots 1 (value and occurences).
    """
    # Build cumulative histograms of shots 0 
    # and 1 in common bins by interpolation.
    all_x = np.unique(np.sort(np.concatenate((x0, x1))))
    cumsum0, cumsum1 = np.cumsum(n0), np.cumsum(n1)
    ecumsum0 = np.interp(x=all_x, xp=x0, fp=cumsum0, left=0)
    necumsum0 = ecumsum0/np.max(ecumsum0)
    ecumsum1 = np.interp(x=all_x, xp=x1, fp=cumsum1, left=0)
    necumsum1 = ecumsum1/np.max(ecumsum1)
    # Calculate optimal threshold and fidelity
    F_vs_th = (1-(1-abs(necumsum0 - necumsum1))/2)
    opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
    opt_idx = int(round(np.average(opt_idxs)))
    F_assignment_raw = F_vs_th[opt_idx]
    threshold_raw = all_x[opt_idx]
    return F_assignment_raw, threshold_raw

def _fit_double_gauss(x_vals, hist_0, hist_1,
                      _x0_guess=None, _x1_guess=None):
    '''
    Fit two histograms to a double gaussian with
    common parameters. From fitted parameters,
    calculate SNR, Pe0, Pg1, Teff, Ffit and Fdiscr.
    '''
    from scipy.optimize import curve_fit
    # Double gaussian model for fitting
    def _gauss_pdf(x, x0, sigma):
        return np.exp(-((x-x0)/sigma)**2/2)
    global double_gauss
    def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
        _dist0 = A*( (1-r)*_gauss_pdf(x, x0, sigma0) + r*_gauss_pdf(x, x1, sigma1) )
        return _dist0
    # helper function to simultaneously fit both histograms with common parameters
    def _double_gauss_joint(x, x0, x1, sigma0, sigma1, A0, A1, r0, r1):
        _dist0 = double_gauss(x, x0, x1, sigma0, sigma1, A0, r0)
        _dist1 = double_gauss(x, x1, x0, sigma1, sigma0, A1, r1)
        return np.concatenate((_dist0, _dist1))
    # Guess for fit
    pdf_0 = hist_0/np.sum(hist_0) # Get prob. distribution
    pdf_1 = hist_1/np.sum(hist_1) # 
    if _x0_guess == None:
        _x0_guess = np.sum(x_vals*pdf_0) # calculate mean
    if _x1_guess == None:
        _x1_guess = np.sum(x_vals*pdf_1) #
    _sigma0_guess = np.sqrt(np.sum((x_vals-_x0_guess)**2*pdf_0)) # calculate std
    _sigma1_guess = np.sqrt(np.sum((x_vals-_x1_guess)**2*pdf_1)) #
    _r0_guess = 0.01
    _r1_guess = 0.05
    _A0_guess = np.max(hist_0)
    _A1_guess = np.max(hist_1)
    p0 = [_x0_guess, _x1_guess, _sigma0_guess, _sigma1_guess, _A0_guess, _A1_guess, _r0_guess, _r1_guess]
    # Bounding parameters
    _x0_bound = (-np.inf,np.inf)
    _x1_bound = (-np.inf,np.inf)
    _sigma0_bound = (0,np.inf)
    _sigma1_bound = (0,np.inf)
    _r0_bound = (0,1)
    _r1_bound = (0,1)
    _A0_bound = (0,np.inf)
    _A1_bound = (0,np.inf)
    bounds = np.array([_x0_bound, _x1_bound, _sigma0_bound, _sigma1_bound, _A0_bound, _A1_bound, _r0_bound, _r1_bound])
    # Fit parameters within bounds
    popt, pcov = curve_fit(
        _double_gauss_joint, x_vals,
        np.concatenate((hist_0, hist_1)),
        p0=p0, bounds=bounds.transpose())
    popt0 = popt[[0,1,2,3,4,6]]
    popt1 = popt[[1,0,3,2,5,7]]
    # Calculate quantities of interest
    SNR = abs(popt0[0] - popt1[0])/((abs(popt0[2])+abs(popt1[2]))/2)
    P_e0 = popt0[5]*popt0[2]/(popt0[2]*popt0[5] + popt0[3]*(1-popt0[5]))
    P_g1 = popt1[5]*popt1[2]/(popt1[2]*popt1[5] + popt1[3]*(1-popt1[5]))
    # Fidelity from fit
    _range = x_vals[0], x_vals[-1]
    _x_data = np.linspace(*_range, 10001)
    _h0 = double_gauss(_x_data, *popt0)# compute distrubition from
    _h1 = double_gauss(_x_data, *popt1)# fitted parameters.
    Fid_fit, threshold_fit = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
    # Discrimination fidelity
    _h0 = double_gauss(_x_data, *popt0[:-1], 0)# compute distrubition without residual
    _h1 = double_gauss(_x_data, *popt1[:-1], 0)# excitation of relaxation.
    Fid_discr, threshold_discr = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
    # return results
    qoi = { 'SNR': SNR,
            'P_e0': P_e0, 'P_g1': P_g1, 
            'Fid_fit': Fid_fit, 'Fid_discr': Fid_discr }
    return popt0, popt1, qoi

def _twoD_Gaussian(x, y, x0, y0, sigma_x, sigma_y, theta=0):
    '''
    Funtion that paramatrizes a 
    single 2D gaussian distribution.
    '''
    x0 = float(x0)
    y0 = float(y0)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g

def triple_gauss(x, y,
    x0, x1, x2,
    y0, y1, y2,
    sigma_x_0, sigma_x_1, sigma_x_2,
    sigma_y_0, sigma_y_1, sigma_y_2,
    A, r_1, r_2):
    '''
    Function that paramtrizes a 2D
    3-gaussian mixture.
    '''
    _dist = A*((1-r_1-r_2)*_twoD_Gaussian(x, y, x0, y0, sigma_x_0, sigma_y_0)+
               r_1*_twoD_Gaussian(x, y, x1, y1, sigma_x_1, sigma_y_1)+
               r_2*_twoD_Gaussian(x, y, x2, y2, sigma_x_2, sigma_y_2))
    return _dist

def _triple_gauss_joint(data,
    x0, x1, x2,
    y0, y1, y2,
    sigma_x_0, sigma_x_1, sigma_x_2,
    sigma_y_0, sigma_y_1, sigma_y_2,
    A0, A1, A2,
    r0_1, r1_1, r2_1,
    r0_2, r1_2, r2_2):
    '''
    Helper function to fit 3 simultatneous mixtures
    of 3 gaussians each in 2D.
    '''
    x, y = data
    _dist0 = triple_gauss(x, y,
                          x0, x1, x2,
                          y0, y1, y2,
                          sigma_x_0, sigma_x_1, sigma_x_2,
                          sigma_y_0, sigma_y_1, sigma_y_2,
                          A0, r0_1, r0_2)
    _dist1 = triple_gauss(x, y,
                          x1, x0, x2,
                          y1, y0, y2,
                          sigma_x_1, sigma_x_0, sigma_x_2,
                          sigma_y_1, sigma_y_0, sigma_y_2,
                          A1, r1_1, r1_2)
    _dist2 = triple_gauss(x, y,
                          x2, x1, x0,
                          y2, y1, y0,
                          sigma_x_2, sigma_x_1, sigma_x_0,
                          sigma_y_2, sigma_y_1, sigma_y_0,
                          A2, r2_1, r2_2)
    return np.concatenate((_dist0.ravel(), _dist1.ravel(), _dist2.ravel()))

def _fit_triple_gauss(x_vals, y_vals, hist_0, hist_1, hist_2):
    '''
    Fit three 2D histograms to a 3-gaussian mixture with
    common parameters. From fitted parameters, calculate
    P0i, P1i and P2i for each input state i=g,e,f.
    '''
    from scipy.optimize import curve_fit
    #####################
    # Guesses for fit
    #####################
    # Get prob. distribution
    pdf_0 = hist_0/np.sum(hist_0)
    pdf_1 = hist_1/np.sum(hist_1)
    pdf_2 = hist_2/np.sum(hist_2)
    # calculate mean
    _x0_guess, _y0_guess = np.sum(np.dot(pdf_0, x_vals)), np.sum(np.dot(pdf_0.T, y_vals)) 
    _x1_guess, _y1_guess = np.sum(np.dot(pdf_1, x_vals)), np.sum(np.dot(pdf_1.T, y_vals))
    _x2_guess, _y2_guess = np.sum(np.dot(pdf_2, x_vals)), np.sum(np.dot(pdf_2.T, y_vals))
    # calculate standard deviation
    _sigma_x_0_guess, _sigma_y_0_guess = np.sqrt(np.sum(np.dot(pdf_0, (x_vals-_x0_guess)**2))), np.sqrt(np.sum(np.dot(pdf_0.T, (y_vals-_y0_guess)**2)))
    _sigma_x_1_guess, _sigma_y_1_guess = np.sqrt(np.sum(np.dot(pdf_1, (x_vals-_x1_guess)**2))), np.sqrt(np.sum(np.dot(pdf_1.T, (y_vals-_y1_guess)**2)))
    _sigma_x_2_guess, _sigma_y_2_guess = np.sqrt(np.sum(np.dot(pdf_2, (x_vals-_x2_guess)**2))), np.sqrt(np.sum(np.dot(pdf_2.T, (y_vals-_y2_guess)**2)))
    # residual populations
    _r0_1_guess, _r0_2_guess = 0.05, 0.05
    _r1_1_guess, _r1_2_guess = 0.05, 0.05
    _r2_1_guess, _r2_2_guess = 0.05, 0.05
    # Amplitudes
    _A0_guess = np.max(hist_0)
    _A1_guess = np.max(hist_1)
    _A2_guess = np.max(hist_2)
    # Guess array
    p0 = [_x0_guess, _x1_guess, _x2_guess,
          _y0_guess, _y1_guess, _y2_guess,
          _sigma_x_0_guess, _sigma_x_1_guess, _sigma_x_2_guess,
          _sigma_y_0_guess, _sigma_y_1_guess, _sigma_y_2_guess,
          _A0_guess, _A1_guess, _A2_guess,
          _r0_1_guess, _r1_1_guess, _r2_1_guess,
          _r0_2_guess, _r1_2_guess, _r2_2_guess]
    # Bounding parameters
    _x0_bound, _y0_bound = (-np.inf,np.inf), (-np.inf,np.inf)
    _x1_bound, _y1_bound = (-np.inf,np.inf), (-np.inf,np.inf)
    _x2_bound, _y2_bound = (-np.inf,np.inf), (-np.inf,np.inf)
    _sigma_x_0_bound, _sigma_y_0_bound = (0,np.inf), (0,np.inf)
    _sigma_x_1_bound, _sigma_y_1_bound = (0,np.inf), (0,np.inf)
    _sigma_x_2_bound, _sigma_y_2_bound = (0,np.inf), (0,np.inf)
    _r0_bound = (0,1)
    _r1_bound = (0,1)
    _r2_bound = (0,1)
    _A0_bound = (0,np.inf)
    _A1_bound = (0,np.inf)
    _A2_bound = (0,np.inf)
    bounds = np.array([_x0_bound, _x1_bound, _x2_bound,
                       _y0_bound, _y1_bound, _y2_bound,
                       _sigma_x_0_bound, _sigma_x_1_bound, _sigma_x_2_bound,
                       _sigma_y_0_bound, _sigma_y_1_bound, _sigma_y_2_bound,
                       _A0_bound, _A1_bound, _A2_bound,
                       _r0_bound, _r1_bound, _r2_bound,
                       _r0_bound, _r1_bound, _r2_bound])
    # Create meshgrid of points
    _X, _Y = np.meshgrid(x_vals, y_vals)
    # Fit parameters within bounds
    popt, pcov = curve_fit(
        _triple_gauss_joint, (_X, _Y),
        np.concatenate((hist_0.ravel(), hist_1.ravel(), hist_2.ravel())),
        p0=p0, bounds=bounds.transpose())
    #               x0   x1   x2   y0   y1   y2  sx0  sx1  sx2  sy0  sy1  sy2    A   r1   r2
    popt0 = popt[[   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  15,  18]]
    popt1 = popt[[   1,   0,   2,   4,   3,   5,   7,   6,   8,  10,   9,  11,  13,  16,  19]]
    popt2 = popt[[   2,   1,   0,   5,   4,   3,   8,   7,   6,  11,  10,   9,  14,  17,  20]]
    ##################################
    # Calculate quantites of interest
    ##################################
    # ground state distribution
    sx0, sx1, sx2 = popt0[6], popt1[6], popt2[6]
    sy0, sy1, sy2 = popt0[9], popt1[9], popt2[9]
    r1 = popt0[13]
    r2 = popt0[14]
    P_0g = ((1-r1-r2)*sx0*sy0)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    P_1g = (r1*sx1*sy1)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    P_2g = (r2*sx2*sy2)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    # excited state distribution
    r1 = popt1[13]
    r2 = popt1[14]
    P_0e = ((1-r1-r2)*sx0*sy0)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    P_1e = (r1*sx1*sy1)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    P_2e = (r2*sx2*sy2)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    # f state distribution
    r1 = popt2[13]
    r2 = popt2[14]
    P_0f = ((1-r1-r2)*sx0*sy0)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    P_1f = (r1*sx1*sy1)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    P_2f = (r2*sx2*sy2)/((1-r1-r2)*sx0*sy0 + r1*sx1*sy1 + r2*sx2*sy2)
    # qoi dictionary
    qoi = {'P_0g':P_0g, 'P_1g':P_1g, 'P_2g':P_2g,
           'P_0e':P_0e, 'P_1e':P_1e, 'P_2e':P_2e,
           'P_0f':P_0f, 'P_1f':P_1f, 'P_2f':P_2f}
    return popt0, popt1, popt2, qoi

def _decision_boundary_points(coefs, intercepts):
    '''
    Find points along the decision boundaries of 
    LinearDiscriminantAnalysis (LDA).
    This is performed by finding the interception
    of the bounds of LDA. For LDA, these bounds are
    encoded in the coef_ and intercept_ parameters
    of the classifier.
    Each bound <i> is given by the equation:
    y + coef_i[0]/coef_i[1]*x + intercept_i = 0
    Note this only works for LinearDiscriminantAnalysis.
    Other classifiers might have diferent bound models.
    '''
    points = {}
    # Cycle through model coeficients
    # and intercepts.
    n = len(intercepts)
    if n == 3:
        _bounds = [[0,1], [1,2], [0,2]]
    if n == 4:
        _bounds = [[0,1], [1,2], [2,3], [3,0]]
    for i, j in _bounds:
        c_i = coefs[i]
        int_i = intercepts[i]
        c_j = coefs[j]
        int_j = intercepts[j]
        x =  (- int_j/c_j[1] + int_i/c_i[1])/(-c_i[0]/c_i[1] + c_j[0]/c_j[1])
        y = -c_i[0]/c_i[1]*x - int_i/c_i[1]
        points[f'{i}{j}'] = (x, y)
    # Find mean point
    points['mean'] = np.mean([ [x, y] for (x, y) in points.values()], axis=0)
    return points

class LRU_experiment_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for LRU experiment.
    """
    def __init__(self,
                 qubit: str,
                 h_state: bool = False,
                 heralded_init: bool = False,
                 fit_3gauss: bool = False,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.qubit = qubit
        self.h_state = h_state
        self.heralded_init = heralded_init
        self.fit_3gauss = fit_3gauss
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        # Perform measurement post-selection
        _cycle = 4
        if self.h_state:
            _cycle += 2
        if self.heralded_init:
            _cycle *= 2
        ############################################
        # Rotate shots in IQ plane
        ############################################
        # Sort shots
        _raw_shots = self.raw_data_dict['data'][:,1:]
        if self.heralded_init:
            _shots_0 = _raw_shots[1::_cycle]
            _shots_1 = _raw_shots[3::_cycle]
            _shots_2 = _raw_shots[5::_cycle]
            _shots_lru = _raw_shots[7::_cycle]
            if self.h_state:
                _shots_3 = _raw_shots[9::_cycle]
                _shots_hlru = _raw_shots[11::_cycle]
        else:
            _shots_0 = _raw_shots[0::_cycle]
            _shots_1 = _raw_shots[1::_cycle]
            _shots_2 = _raw_shots[2::_cycle]
            _shots_lru = _raw_shots[3::_cycle]
            if self.h_state:
                _shots_3 = _raw_shots[4::_cycle]
                _shots_hlru = _raw_shots[5::_cycle]
        # Save raw shots
        self.proc_data_dict['shots_0_IQ'] = _shots_0
        self.proc_data_dict['shots_1_IQ'] = _shots_1
        self.proc_data_dict['shots_2_IQ'] = _shots_2
        self.proc_data_dict['shots_lru_IQ'] = _shots_lru
        if self.h_state:
            self.proc_data_dict['shots_3_IQ'] = _shots_3
            self.proc_data_dict['shots_hlru_IQ'] = _shots_hlru
        # Rotate data
        center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
        center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
        raw_shots = rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
        #####################################################
        # From this point onward raw shots has shape 
        # (nr_shots, nr_quadratures).
        # Post select based on heralding measurement result.
        #####################################################
        if self.heralded_init:
            # estimate post-selection threshold
            shots_0 = raw_shots[1::_cycle, 0]
            shots_1 = raw_shots[3::_cycle, 0]
            ps_th = (np.mean(shots_0)+np.mean(shots_1))/2
            # Sort heralding shots from experiment shots
            ps_shots = raw_shots[0::2,0] # only I quadrature needed for postselection
            exp_shots = raw_shots[1::2] # Here we want to keep both quadratures
            # create post-selection mask
            _mask = [ 1 if s<ps_th else np.nan for s in ps_shots ]
            for i, s in enumerate(_mask):
                exp_shots[i] *= s
            # Remove marked shots
            Shots_0 = exp_shots[0::int(_cycle/2)]
            Shots_1 = exp_shots[1::int(_cycle/2)]
            Shots_2 = exp_shots[2::int(_cycle/2)]
            Shots_lru = exp_shots[3::int(_cycle/2)]
            Shots_0 = Shots_0[~np.isnan(Shots_0[:,0])]
            Shots_1 = Shots_1[~np.isnan(Shots_1[:,0])]
            Shots_2 = Shots_2[~np.isnan(Shots_2[:,0])]
            Shots_lru = Shots_lru[~np.isnan(Shots_lru[:,0])]
            if self.h_state:
                Shots_3 = exp_shots[4::int(_cycle/2)]
                Shots_hlru = exp_shots[5::int(_cycle/2)]
                Shots_3 = Shots_3[~np.isnan(Shots_3[:,0])]
                Shots_hlru = Shots_hlru[~np.isnan(Shots_hlru[:,0])]
        else:
            # Sort 0 and 1 shots
            Shots_0 = raw_shots[0::_cycle]
            Shots_1 = raw_shots[1::_cycle]
            Shots_2 = raw_shots[2::_cycle]
            Shots_lru = raw_shots[3::_cycle]
            if self.h_state:
                Shots_3 = raw_shots[4::_cycle]
                Shots_hlru = raw_shots[5::_cycle]
        # Save rotated shots
        self.proc_data_dict['Shots_0'] = Shots_0
        self.proc_data_dict['Shots_1'] = Shots_1
        self.proc_data_dict['Shots_2'] = Shots_2
        self.proc_data_dict['Shots_lru'] = Shots_lru
        if self.h_state:
            self.proc_data_dict['Shots_3'] = Shots_3
            self.proc_data_dict['Shots_hlru'] = Shots_hlru
        ##############################################################
        # From this point onward Shots_<i> contains post-selected
        # shots of state <i> and has shape (nr_ps_shots, nr_quadtrs).
        # Next we will analyze shots projected along axis and 
        # therefore use a single quadrature. shots_<i> will be used
        # to denote that array of shots.
        ##############################################################
        self.qoi = {}
        if not self.h_state:
            ############################################
            # Use classifier to assign states in the 
            # IQ plane and calculate qutrit fidelity.
            ############################################
            # Parse data for classifier
            data = np.concatenate((Shots_0, Shots_1, Shots_2))
            labels = [0 for s in Shots_0]+[1 for s in Shots_1]+[2 for s in Shots_2]
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
            clf.fit(data, labels)
            dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
            Fid_dict = {}
            for state, shots in zip([    '0',     '1',     '2'],
                                    [Shots_0, Shots_1, Shots_2]):
                _res = clf.predict(shots)
                _fid = np.mean(_res == int(state))
                Fid_dict[state] = _fid
            Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
            # Get assignment fidelity matrix
            M = np.zeros((3,3))
            for i, shots in enumerate([Shots_0, Shots_1, Shots_2]):
                for j, state in enumerate(['0', '1', '2']):
                    _res = clf.predict(shots)
                    M[i][j] = np.mean(_res == int(state))
            # Get leakage removal fraction 
            _res = clf.predict(Shots_lru)
            _vec = np.array([np.mean(_res == int('0')),
                             np.mean(_res == int('1')),
                             np.mean(_res == int('2'))])
            M_inv = np.linalg.inv(M)
            pop_vec = np.dot(_vec, M_inv)
            self.proc_data_dict['classifier'] = clf
            self.proc_data_dict['dec_bounds'] = dec_bounds
            self.proc_data_dict['Fid_dict'] = Fid_dict
            self.qoi['Fid_dict'] = Fid_dict
            self.qoi['Assignment_matrix'] = M
            self.qoi['pop_vec'] = pop_vec
            self.qoi['removal_fraction'] = 1-pop_vec[2]
        else:
            ############################################
            # Use classifier to assign states in the 
            # IQ plane and calculate ququat fidelity.
            ############################################
            # Parse data for classifier
            data = np.concatenate((Shots_0, Shots_1, Shots_2, Shots_3))
            labels = [0 for s in Shots_0]+[1 for s in Shots_1]+\
                     [2 for s in Shots_2]+[3 for s in Shots_3]
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            hclf = LinearDiscriminantAnalysis()
            hclf.fit(data, labels)
            dec_bounds = _decision_boundary_points(hclf.coef_, hclf.intercept_)
            Fid_dict = {}
            for state, shots in zip([    '0',     '1',     '2',     '3'],
                                    [Shots_0, Shots_1, Shots_2, Shots_3]):
                _res = hclf.predict(shots)
                _fid = np.mean(_res == int(state))
                Fid_dict[state] = _fid
            Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
            # Get assignment fidelity matrix
            M = np.zeros((4,4))
            for i, shots in enumerate([Shots_0, Shots_1, Shots_2, Shots_3]):
                for j, state in enumerate(['0', '1', '2', '3']):
                    _res = hclf.predict(shots)
                    M[i][j] = np.mean(_res == int(state))
            # Get leakage removal fraction 
            _res_f = hclf.predict(Shots_lru)
            _vec_f = np.array([np.mean(_res_f == int('0')),
                             np.mean(_res_f == int('1')),
                             np.mean(_res_f == int('2')),
                             np.mean(_res_f == int('3'))])
            M_inv = np.linalg.inv(M)
            pop_vec_f = np.dot(_vec_f, M_inv)
            # Get leakage removal fraction 
            _res_h = hclf.predict(Shots_hlru)
            _vec_h = np.array([np.mean(_res_h == int('0')),
                             np.mean(_res_h == int('1')),
                             np.mean(_res_h == int('2')),
                             np.mean(_res_h == int('3'))])
            M_inv = np.linalg.inv(M)
            pop_vec_h = np.dot(_vec_h, M_inv)
            self.proc_data_dict['classifier'] = hclf
            self.proc_data_dict['dec_bounds'] = dec_bounds
            self.proc_data_dict['Fid_dict'] = Fid_dict
            self.qoi['Fid_dict'] = Fid_dict
            self.qoi['Assignment_matrix'] = M
            self.qoi['pop_vec'] = pop_vec_f
            self.qoi['pop_vec_h'] = pop_vec_h
            self.qoi['removal_fraction'] = 1-pop_vec_f[2]
            self.qoi['removal_fraction_h'] = 1-pop_vec_h[3]
        #########################################
        # Project data along axis perpendicular
        # to the decision boundaries.
        #########################################
        ############################
        # Projection along 10 axis.
        ############################
        # Rotate shots over 01 decision boundary axis
        shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1], dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
        shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1], dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
        # Take relavant quadrature
        shots_0 = shots_0[:,0]
        shots_1 = shots_1[:,0]
        n_shots_1 = len(shots_1)
        # find range
        _all_shots = np.concatenate((shots_0, shots_1))
        _range = (np.min(_all_shots), np.max(_all_shots))
        # Sort shots in unique values
        x0, n0 = np.unique(shots_0, return_counts=True)
        x1, n1 = np.unique(shots_1, return_counts=True)
        Fid_01, threshold_01 = _calculate_fid_and_threshold(x0, n0, x1, n1)
        # Histogram of shots for 1 and 2
        h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
        h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
        # Save processed data
        self.proc_data_dict['projection_01'] = {}
        self.proc_data_dict['projection_01']['h0'] = h0
        self.proc_data_dict['projection_01']['h1'] = h1
        self.proc_data_dict['projection_01']['bin_centers'] = bin_centers
        self.proc_data_dict['projection_01']['popt0'] = popt0
        self.proc_data_dict['projection_01']['popt1'] = popt1
        self.proc_data_dict['projection_01']['SNR'] = params_01['SNR']
        self.proc_data_dict['projection_01']['Fid'] = Fid_01
        self.proc_data_dict['projection_01']['threshold'] = threshold_01
        ############################
        # Projection along 12 axis.
        ############################
        # Rotate shots over 12 decision boundary axis
        shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
        shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
        # Take relavant quadrature
        shots_1 = shots_1[:,0]
        shots_2 = shots_2[:,0]
        n_shots_2 = len(shots_2)
        # find range
        _all_shots = np.concatenate((shots_1, shots_2))
        _range = (np.min(_all_shots), np.max(_all_shots))
        # Sort shots in unique values
        x1, n1 = np.unique(shots_1, return_counts=True)
        x2, n2 = np.unique(shots_2, return_counts=True)
        Fid_12, threshold_12 = _calculate_fid_and_threshold(x1, n1, x2, n2)
        # Histogram of shots for 1 and 2
        h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
        h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        popt1, popt2, params_12 = _fit_double_gauss(bin_centers, h1, h2)
        # Save processed data
        self.proc_data_dict['projection_12'] = {}
        self.proc_data_dict['projection_12']['h1'] = h1
        self.proc_data_dict['projection_12']['h2'] = h2
        self.proc_data_dict['projection_12']['bin_centers'] = bin_centers
        self.proc_data_dict['projection_12']['popt1'] = popt1
        self.proc_data_dict['projection_12']['popt2'] = popt2
        self.proc_data_dict['projection_12']['SNR'] = params_12['SNR']
        self.proc_data_dict['projection_12']['Fid'] = Fid_12
        self.proc_data_dict['projection_12']['threshold'] = threshold_12

        if not self.h_state:
            ############################
            # Projection along 02 axis.
            ############################
            # Rotate shots over 02 decision boundary axis
            shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            # Take relavant quadrature
            shots_0 = shots_0[:,0]
            shots_2 = shots_2[:,0]
            n_shots_2 = len(shots_2)
            # find range
            _all_shots = np.concatenate((shots_0, shots_2))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x0, n0 = np.unique(shots_0, return_counts=True)
            x2, n2 = np.unique(shots_2, return_counts=True)
            Fid_02, threshold_02 = _calculate_fid_and_threshold(x0, n0, x2, n2)
            # Histogram of shots for 1 and 2
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt0, popt2, params_02 = _fit_double_gauss(bin_centers, h0, h2)
            # Save processed data
            self.proc_data_dict['projection_02'] = {}
            self.proc_data_dict['projection_02']['h0'] = h0
            self.proc_data_dict['projection_02']['h2'] = h2
            self.proc_data_dict['projection_02']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_02']['popt0'] = popt0
            self.proc_data_dict['projection_02']['popt2'] = popt2
            self.proc_data_dict['projection_02']['SNR'] = params_02['SNR']
            self.proc_data_dict['projection_02']['Fid'] = Fid_02
            self.proc_data_dict['projection_02']['threshold'] = threshold_02
        else:
            ############################
            # Projection along 23 axis.
            ############################
            # Rotate shots over 23 decision boundary axis
            shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['23'], phi=np.pi/2)
            shots_3 = rotate_and_center_data(Shots_3[:,0],Shots_3[:,1],dec_bounds['mean'],dec_bounds['23'], phi=np.pi/2)
            # Take relavant quadrature
            shots_3 = shots_3[:,0]
            shots_2 = shots_2[:,0]
            n_shots_2 = len(shots_2)
            # find range
            _all_shots = np.concatenate((shots_3, shots_2))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x3, n3 = np.unique(shots_3, return_counts=True)
            x2, n2 = np.unique(shots_2, return_counts=True)
            Fid_23, threshold_23 = _calculate_fid_and_threshold(x3, n3, x2, n2)
            # Histogram of shots for 1 and 2
            h3, bin_edges = np.histogram(shots_3, bins=100, range=_range)
            h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt3, popt2, params_23 = _fit_double_gauss(bin_centers, h3, h2)
            # Save processed data
            self.proc_data_dict['projection_23'] = {}
            self.proc_data_dict['projection_23']['h3'] = h3
            self.proc_data_dict['projection_23']['h2'] = h2
            self.proc_data_dict['projection_23']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_23']['popt3'] = popt3
            self.proc_data_dict['projection_23']['popt2'] = popt2
            self.proc_data_dict['projection_23']['SNR'] = params_23['SNR']
            self.proc_data_dict['projection_23']['Fid'] = Fid_23
            self.proc_data_dict['projection_23']['threshold'] = threshold_23
            ############################
            # Projection along 30 axis.
            ############################
            # Rotate shots over 30 decision boundary axis
            shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['30'], phi=np.pi/2)
            shots_3 = rotate_and_center_data(Shots_3[:,0],Shots_3[:,1],dec_bounds['mean'],dec_bounds['30'], phi=np.pi/2)
            # Take relavant quadrature
            shots_3 = shots_3[:,0]
            shots_0 = shots_0[:,0]
            n_shots_3 = len(shots_3)
            # find range
            _all_shots = np.concatenate((shots_3, shots_0))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x3, n3 = np.unique(shots_3, return_counts=True)
            x0, n0 = np.unique(shots_0, return_counts=True)
            Fid_30, threshold_30 = _calculate_fid_and_threshold(x3, n3, x0, n0)
            # Histogram of shots for 1 and 2
            h3, bin_edges = np.histogram(shots_3, bins=100, range=_range)
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt3, popt0, params_30 = _fit_double_gauss(bin_centers, h3, h0)
            # Save processed data
            self.proc_data_dict['projection_30'] = {}
            self.proc_data_dict['projection_30']['h3'] = h3
            self.proc_data_dict['projection_30']['h0'] = h0
            self.proc_data_dict['projection_30']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_30']['popt3'] = popt3
            self.proc_data_dict['projection_30']['popt0'] = popt0
            self.proc_data_dict['projection_30']['SNR'] = params_30['SNR']
            self.proc_data_dict['projection_30']['Fid'] = Fid_30
            self.proc_data_dict['projection_30']['threshold'] = threshold_30

        ###############################
        # Fit 3-gaussian mixture
        ###############################
        if self.fit_3gauss:
            _all_shots = np.concatenate((self.proc_data_dict['shots_0_IQ'], 
                                         self.proc_data_dict['shots_1_IQ'], 
                                         self.proc_data_dict['shots_2_IQ']))
            _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1,
                            np.max(np.abs(_all_shots[:,1]))*1.1 ])
            def _histogram_2d(x_data, y_data, lim):
                _bins = (np. linspace(-lim, lim, 201), np.linspace(-lim, lim, 201))
                n, _xbins, _ybins = np.histogram2d(x_data, y_data, bins=_bins)
                return n.T, _xbins, _ybins
            # 2D histograms
            n_0, xbins, ybins = _histogram_2d(*self.proc_data_dict['shots_0_IQ'].T, lim=_lim)
            n_1, xbins, ybins = _histogram_2d(*self.proc_data_dict['shots_1_IQ'].T, lim=_lim)
            n_2, xbins, ybins = _histogram_2d(*self.proc_data_dict['shots_2_IQ'].T, lim=_lim)
            n_3, xbins, ybins = _histogram_2d(*self.proc_data_dict['shots_lru_IQ'].T, lim=_lim)
            # bin centers
            xbins_c, ybins_c = (xbins[1:]+xbins[:-1])/2, (ybins[1:]+ybins[:-1])/2 
            popt0, popt1, popt2, qoi = _fit_triple_gauss(xbins_c, ybins_c, n_0, n_1, n_2)
            popt3, _popt1, _popt2, _qoi = _fit_triple_gauss(xbins_c, ybins_c, n_3, n_1, n_2)
            self.proc_data_dict['3gauss_fit'] = {}
            self.proc_data_dict['3gauss_fit']['xbins'] = xbins
            self.proc_data_dict['3gauss_fit']['ybins'] = ybins
            self.proc_data_dict['3gauss_fit']['n0'] = n_0
            self.proc_data_dict['3gauss_fit']['n1'] = n_1
            self.proc_data_dict['3gauss_fit']['n2'] = n_2
            self.proc_data_dict['3gauss_fit']['n3'] = n_3
            self.proc_data_dict['3gauss_fit']['popt0'] = popt0
            self.proc_data_dict['3gauss_fit']['popt1'] = popt1
            self.proc_data_dict['3gauss_fit']['popt2'] = popt2
            self.proc_data_dict['3gauss_fit']['popt3'] = popt3
            self.proc_data_dict['3gauss_fit']['qoi'] = _qoi
            self.proc_data_dict['3gauss_fit']['removal_fraction'] = 1-_qoi['P_2g']
            self.qoi['removal_fraction_gauss_fit'] = 1-_qoi['P_2g']

    def prepare_plots(self):
        self.axs_dict = {}
        fig = plt.figure(figsize=(8,4), dpi=100)
        if self.h_state:
            axs = [fig.add_subplot(121),
                   fig.add_subplot(422),
                   fig.add_subplot(424),
                   fig.add_subplot(426),
                   fig.add_subplot(428)]
        else:
            axs = [fig.add_subplot(121),
                   fig.add_subplot(322),
                   fig.add_subplot(324),
                   fig.add_subplot(326)]
        # fig.patch.set_alpha(0)
        self.axs_dict['SSRO_plot'] = axs[0]
        self.figs['SSRO_plot'] = fig
        self.plot_dicts['SSRO_plot'] = {
            'plotfn': ssro_IQ_projection_plotfn,
            'ax_id': 'SSRO_plot',
            'shots_0': self.proc_data_dict['Shots_0'],
            'shots_1': self.proc_data_dict['Shots_1'],
            'shots_2': self.proc_data_dict['Shots_2'],
            'shots_3': self.proc_data_dict['Shots_3'] if self.h_state \
                       else None,
            'projection_01': self.proc_data_dict['projection_01'],
            'projection_12': self.proc_data_dict['projection_12'],
            'projection_02': None if self.h_state else\
                             self.proc_data_dict['projection_02'],
            'projection_23': None if not self.h_state else\
                             self.proc_data_dict['projection_23'],
            'projection_30': None if not self.h_state else\
                             self.proc_data_dict['projection_30'],
            'classifier': self.proc_data_dict['classifier'],
            'dec_bounds': self.proc_data_dict['dec_bounds'],
            'Fid_dict': self.proc_data_dict['Fid_dict'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, ax = plt.subplots(figsize=(3.25,3.25), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Leakage_histogram_f_state'] = ax
        self.figs['Leakage_histogram_f_state'] = fig
        self.plot_dicts['Leakage_histogram_f_state'] = {
            'plotfn': leakage_hist_plotfn,
            'ax_id': 'Leakage_histogram_f_state',
            'shots_0': self.proc_data_dict['Shots_0'],
            'shots_1': self.proc_data_dict['Shots_1'],
            'shots_2': self.proc_data_dict['Shots_2'],
            'shots_3': self.proc_data_dict['Shots_3'] if self.h_state \
                       else None,
            'shots_lru': self.proc_data_dict['Shots_lru'],
            'classifier': self.proc_data_dict['classifier'],
            'dec_bounds': self.proc_data_dict['dec_bounds'],
            'pop_vec': self.qoi['pop_vec'],
            'state' : '2',
            'qoi': self.qoi,
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        if self.h_state:
            fig, ax = plt.subplots(figsize=(3.25,3.25), dpi=100)
            # fig.patch.set_alpha(0)
            self.axs_dict['Leakage_histogram_h_state'] = ax
            self.figs['Leakage_histogram_h_state'] = fig
            self.plot_dicts['Leakage_histogram'] = {
                'plotfn': leakage_hist_plotfn,
                'ax_id': 'Leakage_histogram_h_state',
                'shots_0': self.proc_data_dict['Shots_0'],
                'shots_1': self.proc_data_dict['Shots_1'],
                'shots_2': self.proc_data_dict['Shots_2'],
                'shots_3': self.proc_data_dict['Shots_3'],
                'shots_lru': self.proc_data_dict['Shots_hlru'],
                'classifier': self.proc_data_dict['classifier'],
                'dec_bounds': self.proc_data_dict['dec_bounds'],
                'pop_vec': self.qoi['pop_vec_h'],
                'state' : '3',
                'qoi': self.qoi,
                'qubit': self.qubit,
                'timestamp': self.timestamp
            }
        fig, ax = plt.subplots(figsize=(3,3), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Assignment_matrix'] = ax
        self.figs['Assignment_matrix'] = fig
        self.plot_dicts['Assignment_matrix'] = {
            'plotfn': assignment_matrix_plotfn,
            'ax_id': 'Assignment_matrix',
            'M': self.qoi['Assignment_matrix'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        if self.fit_3gauss:
            fig = plt.figure(figsize=(12,4), dpi=100)
            axs = [None,None,None]
            axs[0] = fig.add_subplot(1, 3, 1)
            axs[2] = fig.add_subplot(1, 3, 3 , projection='3d', elev=20)
            axs[1] = fig.add_subplot(1, 3, 2)
            # fig.patch.set_alpha(0)
            self.axs_dict['Three_gauss_fit'] = axs[0]
            self.figs['Three_gauss_fit'] = fig
            self.plot_dicts['Three_gauss_fit'] = {
                'plotfn': gauss_fit2D_plotfn,
                'ax_id': 'Three_gauss_fit',
                'fit_dict': self.proc_data_dict['3gauss_fit'],
                'qubit': self.qubit,
                'timestamp': self.timestamp
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def ssro_IQ_plotfn(
    shots_0, 
    shots_1,
    shots_2,
    shots_3,
    timestamp,
    qubit,
    ax, 
    dec_bounds=None,
    Fid_dict=None,
    **kw):
    fig = ax.get_figure()
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    # Plot stuff
    ax.plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.05)
    ax.plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.05)
    ax.plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    ax.plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    ax.plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    ax.plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    ax.plot(popt_0[1], popt_0[2], 'x', color='white')
    ax.plot(popt_1[1], popt_1[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    ax.add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    ax.add_patch(circle_1)
    _all_shots = np.concatenate((shots_0, shots_1))
    if type(shots_2) != type(None):
        popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
        ax.plot(shots_2[:,0], shots_2[:,1], '.', color='C2', alpha=0.05)
        ax.plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
        ax.plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
        ax.plot(popt_2[1], popt_2[2], 'x', color='white')
        # Draw 4sigma ellipse around mean
        circle_2 = Ellipse((popt_2[1], popt_2[2]),
                          width=4*popt_2[3], height=4*popt_2[4],
                          angle=-popt_2[5]*180/np.pi,
                          ec='white', fc='none', ls='--', lw=1.25, zorder=10)
        ax.add_patch(circle_2)
        _all_shots = np.concatenate((_all_shots, shots_2))
    if type(shots_3) != type(None):
        popt_3 = _fit_2D_gaussian(shots_3[:,0], shots_3[:,1])
        ax.plot(shots_3[:,0], shots_3[:,1], '.', color='gold', alpha=0.05)
        ax.plot([0, popt_3[1]], [0, popt_3[2]], '--', color='k', lw=.5)
        ax.plot(popt_3[1], popt_3[2], '.', color='gold', label='$3^\mathrm{rd}$ excited')
        ax.plot(popt_3[1], popt_3[2], 'x', color='white')
        # Draw 4sigma ellipse around mean
        circle_3 = Ellipse((popt_3[1], popt_3[2]),
                          width=4*popt_3[3], height=4*popt_3[4],
                          angle=-popt_3[5]*180/np.pi,
                          ec='white', fc='none', ls='--', lw=1.25, zorder=10)
        ax.add_patch(circle_3)
        _all_shots = np.concatenate((_all_shots, shots_3))

    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    ax.set_xlim(-_lim, _lim)
    ax.set_ylim(-_lim, _lim)
    ax.legend(frameon=False)
    ax.set_xlabel('Integrated voltage I')
    ax.set_ylabel('Integrated voltage Q')
    ax.set_title(f'{timestamp}\nIQ plot qubit {qubit}')
    if dec_bounds:
        # Plot decision boundary
        _bounds = list(dec_bounds.keys())
        _bounds.remove('mean')
        Lim_points = {}
        for bound in _bounds:
            dec_bounds['mean']
            _x0, _y0 = dec_bounds['mean']
            _x1, _y1 = dec_bounds[bound]
            a = (_y1-_y0)/(_x1-_x0)
            b = _y0 - a*_x0
            _xlim = 1e2*np.sign(_x1-_x0)
            _ylim = a*_xlim + b
            Lim_points[bound] = _xlim, _ylim
        # Plot classifier zones
        from matplotlib.patches import Polygon
        # Plot 0 area
        _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['30']]
        _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
        ax.add_patch(_patch)
        # Plot 1 area
        _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
        _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
        ax.add_patch(_patch)
        # Plot 2 area
        _points = [dec_bounds['mean'], Lim_points['23'], Lim_points['12']]
        _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
        ax.add_patch(_patch)
        if type(shots_3) != type(None):
            # Plot 3 area
            _points = [dec_bounds['mean'], Lim_points['23'], Lim_points['30']]
            _patch = Polygon(_points, color='gold', alpha=0.2, lw=0)
            ax.add_patch(_patch)
        for bound in _bounds:
            _x0, _y0 = dec_bounds['mean']
            _x1, _y1 = Lim_points[bound]
            ax.plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
        if Fid_dict:
            # Write fidelity textbox
            text = '\n'.join(('Assignment fidelity:',
                              f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                              f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                              f'$F_f$ : {Fid_dict["2"]*100:.1f}%',
                              f'$F_h$ : {Fid_dict["3"]*100:.1f}%',
                              f'$F_\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'))
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax.text(1.05, 1, text, transform=ax.transAxes,
                    verticalalignment='top', bbox=props)

def ssro_IQ_projection_plotfn(
    shots_0, 
    shots_1,
    shots_2,
    projection_01,
    projection_12,
    classifier,
    dec_bounds,
    Fid_dict,
    timestamp,
    qubit, 
    ax,
    shots_3=None,
    projection_02=None,
    projection_23=None,
    projection_30=None,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    if type(shots_3) != type(None):
        popt_3 = _fit_2D_gaussian(shots_3[:,0], shots_3[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.05)
    axs[0].plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.05)
    axs[0].plot(shots_2[:,0], shots_2[:,1], '.', color='C2', alpha=0.05)
    if type(shots_3) != type(None):
        axs[0].plot(shots_3[:,0], shots_3[:,1], '.', color='gold', alpha=0.05)
    axs[0].plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
    axs[0].plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    axs[0].plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    axs[0].plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
    axs[0].plot(popt_0[1], popt_0[2], 'x', color='white')
    axs[0].plot(popt_1[1], popt_1[2], 'x', color='white')
    axs[0].plot(popt_2[1], popt_2[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_1)
    circle_2 = Ellipse((popt_2[1], popt_2[2]),
                      width=4*popt_2[3], height=4*popt_2[4],
                      angle=-popt_2[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_2)
    _all_shots = np.concatenate((shots_0, shots_1, shots_2))
    if type(shots_3) != type(None):
        popt_3 = _fit_2D_gaussian(shots_3[:,0], shots_3[:,1])
        axs[0].plot([0, popt_3[1]], [0, popt_3[2]], '--', color='k', lw=.5)
        axs[0].plot(popt_3[1], popt_3[2], '.', color='gold', label='$3^\mathrm{rd}$ excited')
        axs[0].plot(popt_3[1], popt_3[2], 'x', color='w')
        # Draw 4sigma ellipse around mean
        circle_3 = Ellipse((popt_3[1], popt_3[2]),
                          width=4*popt_3[3], height=4*popt_3[4],
                          angle=-popt_3[5]*180/np.pi,
                          ec='white', fc='none', ls='--', lw=1.25, zorder=10)
        axs[0].add_patch(circle_3)
        _all_shots = np.concatenate((_all_shots, shots_3))
    # Plot classifier zones
    from matplotlib.patches import Polygon
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
    # Plot decision boundary
    _bounds = list(dec_bounds.keys())
    _bounds.remove('mean')
    Lim_points = {}
    for bound in _bounds:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot classifier zones
    from matplotlib.patches import Polygon
    if type(shots_3) != type(None):
        boundaries = [('30', '01'), ('01', '12'), ('12', '23'), ('23', '30')]
        colors = ['C0', 'C3', 'C2', 'gold']
    else:
        boundaries = [('02', '01'), ('01', '12'), ('12', '02')]
        colors = ['C0', 'C3', 'C2']
    for _bds, color in zip(boundaries, colors):
        _points = [dec_bounds['mean'], Lim_points[_bds[0]], Lim_points[_bds[1]]]
        _patch = Polygon(_points, color=color, alpha=0.2, lw=0)
        axs[0].add_patch(_patch)
    for bound in _bounds:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
    axs[0].legend(frameon=False)
    axs[0].set_xlabel('Integrated voltage I')
    axs[0].set_ylabel('Integrated voltage Q')
    axs[0].set_title(f'IQ plot qubit {qubit}')
    fig.suptitle(f'{timestamp}\n')
    ##########################
    # Plot projections
    ##########################
    # 01 projection
    _bin_c = projection_01['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[1].bar(_bin_c, projection_01['h0'], bin_width, fc='C0', alpha=0.4)
    axs[1].bar(_bin_c, projection_01['h1'], bin_width, fc='C3', alpha=0.4)
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt0']), '-C0')
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt1']), '-C3')
    axs[1].axvline(projection_01['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_01["Fid"]*100:.1f}%',
                      f'SNR : {projection_01["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[1].text(.775, .9, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[1].text(projection_01['popt0'][0], projection_01['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[1].text(projection_01['popt1'][0], projection_01['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[1].set_xticklabels([])
    axs[1].set_xlim(_bin_c[0], _bin_c[-1])
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Projection of data')
    # 12 projection
    _bin_c = projection_12['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[2].bar(_bin_c, projection_12['h1'], bin_width, fc='C3', alpha=0.4)
    axs[2].bar(_bin_c, projection_12['h2'], bin_width, fc='C2', alpha=0.4)
    axs[2].plot(_bin_c, double_gauss(_bin_c, *projection_12['popt1']), '-C3')
    axs[2].plot(_bin_c, double_gauss(_bin_c, *projection_12['popt2']), '-C2')
    axs[2].axvline(projection_12['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_12["Fid"]*100:.1f}%',
                      f'SNR : {projection_12["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[2].text(.775, .9, text, transform=axs[2].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[2].text(projection_12['popt1'][0], projection_12['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[2].text(projection_12['popt2'][0], projection_12['popt2'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='C2')
    axs[2].set_xticklabels([])
    axs[2].set_xlim(_bin_c[0], _bin_c[-1])
    axs[2].set_ylim(bottom=0)
    if projection_02:
        # 02 projection
        _bin_c = projection_02['bin_centers']
        bin_width = _bin_c[1]-_bin_c[0]
        axs[3].bar(_bin_c, projection_02['h0'], bin_width, fc='C0', alpha=0.4)
        axs[3].bar(_bin_c, projection_02['h2'], bin_width, fc='C2', alpha=0.4)
        axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_02['popt0']), '-C0')
        axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_02['popt2']), '-C2')
        axs[3].axvline(projection_02['threshold'], ls='--', color='k', lw=1)
        text = '\n'.join((f'Fid.  : {projection_02["Fid"]*100:.1f}%',
                          f'SNR : {projection_02["SNR"]:.1f}'))
        props = dict(boxstyle='round', facecolor='gray', alpha=0)
        axs[3].text(.775, .9, text, transform=axs[3].transAxes,
                    verticalalignment='top', bbox=props, fontsize=7)
        axs[3].text(projection_02['popt0'][0], projection_02['popt0'][4]/2,
                    r'$|g\rangle$', ha='center', va='center', color='C0')
        axs[3].text(projection_02['popt2'][0], projection_02['popt2'][4]/2,
                    r'$|f\rangle$', ha='center', va='center', color='C2')
        axs[3].set_xticklabels([])
        axs[3].set_xlim(_bin_c[0], _bin_c[-1])
        axs[3].set_ylim(bottom=0)
        axs[3].set_xlabel('Integrated voltage')
    if projection_23:
        # 23 projection
        _bin_c = projection_23['bin_centers']
        bin_width = _bin_c[1]-_bin_c[0]
        axs[3].bar(_bin_c, projection_23['h3'], bin_width, fc='gold', alpha=0.4)
        axs[3].bar(_bin_c, projection_23['h2'], bin_width, fc='C2', alpha=0.4)
        axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_23['popt3']), '-', color='gold')
        axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_23['popt2']), '-C2')
        axs[3].axvline(projection_23['threshold'], ls='--', color='k', lw=1)
        text = '\n'.join((f'Fid.  : {projection_23["Fid"]*100:.1f}%',
                          f'SNR : {projection_23["SNR"]:.1f}'))
        props = dict(boxstyle='round', facecolor='gray', alpha=0)
        axs[3].text(.775, .9, text, transform=axs[3].transAxes,
                    verticalalignment='top', bbox=props, fontsize=7)
        axs[3].text(projection_23['popt3'][0], projection_23['popt3'][4]/2,
                    r'$|h\rangle$', ha='center', va='center', color='gold')
        axs[3].text(projection_23['popt2'][0], projection_23['popt2'][4]/2,
                    r'$|f\rangle$', ha='center', va='center', color='C2')
        axs[3].set_xticklabels([])
        axs[3].set_xlim(_bin_c[0], _bin_c[-1])
        axs[3].set_ylim(bottom=0)
    if projection_30:
        # 23 projection
        _bin_c = projection_30['bin_centers']
        bin_width = _bin_c[1]-_bin_c[0]
        axs[4].bar(_bin_c, projection_30['h3'], bin_width, fc='gold', alpha=0.4)
        axs[4].bar(_bin_c, projection_30['h0'], bin_width, fc='C0', alpha=0.4)
        axs[4].plot(_bin_c, double_gauss(_bin_c, *projection_30['popt3']), '-', color='gold')
        axs[4].plot(_bin_c, double_gauss(_bin_c, *projection_30['popt0']), '-C0')
        axs[4].axvline(projection_30['threshold'], ls='--', color='k', lw=1)
        text = '\n'.join((f'Fid.  : {projection_30["Fid"]*100:.1f}%',
                          f'SNR : {projection_30["SNR"]:.1f}'))
        props = dict(boxstyle='round', facecolor='gray', alpha=0)
        axs[4].text(.775, .9, text, transform=axs[4].transAxes,
                    verticalalignment='top', bbox=props, fontsize=7)
        axs[4].text(projection_30['popt3'][0], projection_30['popt3'][4]/2,
                    r'$|h\rangle$', ha='center', va='center', color='gold')
        axs[4].text(projection_30['popt0'][0], projection_30['popt0'][4]/2,
                    r'$|g\rangle$', ha='center', va='center', color='C0')
        axs[4].set_xticklabels([])
        axs[4].set_xlim(_bin_c[0], _bin_c[-1])
        axs[4].set_ylim(bottom=0)
        axs[4].set_xlabel('Integrated voltage')

    # Write fidelity textbox
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                      f'$F_f$ : {Fid_dict["2"]*100:.1f}%'))
    if type(shots_3) != type(None):
        text += f'\n$F_h$ : {Fid_dict["3"]*100:.1f}%'
    text += f'\n$F_\\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'
    props = dict(boxstyle='round', facecolor='w', alpha=1)
    axs[1].text(1.05, 1, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props)

def leakage_hist_plotfn(
    shots_0, 
    shots_1,
    shots_2,
    shots_lru,
    classifier,
    dec_bounds,
    pop_vec,
    qoi,
    timestamp,
    qubit, 
    ax,
    shots_3=None,
    state='2',
    **kw):
    fig = ax.get_figure()
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    if type(shots_3) != type(None):
        popt_3 = _fit_2D_gaussian(shots_3[:,0], shots_3[:,1])
    ax.plot(shots_lru[:,0], shots_lru[:,1], '.', color='C0', alpha=1, markersize=1)
    ax.plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    ax.plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    ax.plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
    ax.plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    ax.plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    ax.plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
    ax.plot(popt_0[1], popt_0[2], 'o', color='w') # change for ket state
    ax.plot(popt_1[1], popt_1[2], 'o', color='w') # change for ket state
    ax.plot(popt_2[1], popt_2[2], 'o', color='w') # change for ket state
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    ax.add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    ax.add_patch(circle_1)
    circle_2 = Ellipse((popt_2[1], popt_2[2]),
                      width=4*popt_2[3], height=4*popt_2[4],
                      angle=-popt_2[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    ax.add_patch(circle_2)
    _all_shots = np.concatenate((shots_0, shots_1, shots_2))
    if type(shots_3) != type(None):
        popt_3 = _fit_2D_gaussian(shots_3[:,0], shots_3[:,1])
        ax.plot([0, popt_3[1]], [0, popt_3[2]], '--', color='k', lw=.5)
        ax.plot(popt_3[1], popt_3[2], '.', color='gold', label='$3^\mathrm{rd}$ excited')
        ax.plot(popt_3[1], popt_3[2], 'o', color='w')
        # Draw 4sigma ellipse around mean
        circle_3 = Ellipse((popt_3[1], popt_3[2]),
                          width=4*popt_3[3], height=4*popt_3[4],
                          angle=-popt_3[5]*180/np.pi,
                          ec='white', fc='none', ls='--', lw=1.25, zorder=10)
        ax.add_patch(circle_3)
        _all_shots = np.concatenate((_all_shots, shots_3))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    ax.set_xlim(-_lim, _lim)
    ax.set_ylim(-_lim, _lim)
    ax.legend(frameon=False)
    ax.set_xlabel('Integrated voltage I')
    ax.set_ylabel('Integrated voltage Q')
    ax.set_title(f'{timestamp}\nIQ plot qubit {qubit}')
    # Plot decision boundary
    _bounds = list(dec_bounds.keys())
    _bounds.remove('mean')
    Lim_points = {}
    for bound in _bounds:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot classifier zones
    from matplotlib.patches import Polygon
    if type(shots_3) != type(None):
        boundaries = [('30', '01'), ('01', '12'), ('12', '23'), ('23', '30')]
        colors = ['C0', 'C3', 'C2', 'gold']
    else:
        boundaries = [('02', '01'), ('01', '12'), ('12', '02')]
        colors = ['C0', 'C3', 'C2']
    for _bds, color in zip(boundaries, colors):
        _points = [dec_bounds['mean'], Lim_points[_bds[0]], Lim_points[_bds[1]]]
        _patch = Polygon(_points, color=color, alpha=0.2, lw=0)
        ax.add_patch(_patch)
    for bound in _bounds:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        ax.plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)

    text = '\n'.join(('State population:',
                      '$P_\\mathrm{|0\\rangle}=$'+f'{pop_vec[0]*100:.2f}%',
                      '$P_\\mathrm{|1\\rangle}=$'+f'{pop_vec[1]*100:.2f}%',
                      '$P_\\mathrm{|2\\rangle}=$'+f'{pop_vec[2]*100:.2f}%',
                      ' ' if len(pop_vec)==3 else \
                      '$P_\\mathrm{|3\\rangle}=$'+f'{pop_vec[3]*100:.2f}%' ,
                     f'$|{state}\\rangle$'+f' removal fraction:\n\n\n'))
    if 'removal_fraction_gauss_fit' in qoi:
        text +=  '\n'+r'$|2\rangle$'+f' removal fraction\nfrom 3-gauss. fit:\n\n\n'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(1.05, .95, text, transform=ax.transAxes,
            verticalalignment='top', bbox=props, fontsize=9)
    if state == '2':
        removal_fraction = qoi['removal_fraction']
    else:
        removal_fraction = qoi['removal_fraction_h']
    text = f'{removal_fraction*100:.1f}%'
    ax.text(1.05, .48, text, transform=ax.transAxes,
            verticalalignment='top', fontsize=24)
    if 'removal_fraction_gauss_fit' in qoi:
        text = f'{qoi["removal_fraction_gauss_fit"]*100:.1f}%'
        ax.text(1.05, .25, text, transform=ax.transAxes,
                verticalalignment='top', fontsize=24)
    ax.set_xlabel('Integrated voltage I')
    ax.set_ylabel('Integrated voltage Q')
    ax.set_title(f'{timestamp}\n{state}-state removal fraction qubit {qubit}')

def gauss_fit2D_plotfn(
    fit_dict,
    timestamp,
    qubit, 
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    axs = np.array(axs)[[0,2,1]]

    xbins, ybins = fit_dict['xbins'], fit_dict['ybins']
    xbins_c, ybins_c = (xbins[1:]+xbins[:-1])/2, (ybins[1:]+ybins[:-1])/2 
    _X, _Y = np.meshgrid(xbins_c, ybins_c)
    # Plot measured histograms
    n_3_fit = triple_gauss(_X, _Y, *fit_dict['popt3'])
    axs[0].pcolormesh(xbins, ybins, fit_dict['n3'], alpha=1, vmax=2)
    # Draw 4sigma ellipse around mean
    for i in range(3):
        # Plot center of distributions 
        axs[0].plot(fit_dict[f'popt{i}'][0], fit_dict[f'popt{i}'][3], 'C3x')
        axs[1].plot(fit_dict[f'popt{i}'][0], fit_dict[f'popt{i}'][3], 'C3x')
        # Draw 4sigma ellipse around mean
        circle = patches.Ellipse((fit_dict[f'popt{i}'][0], fit_dict[f'popt{i}'][3]),
                    width=4*fit_dict[f'popt{i}'][6], height=4*fit_dict[f'popt{i}'][9],
                    angle=0,#-popt_0[5]*180/np.pi,
                    ec='white', fc='none', ls='--', lw=1.25, zorder=10)
        axs[1].add_patch(circle)
    # Plot fitted histograms
    axs[1].pcolormesh(xbins, ybins, n_3_fit, alpha=1, vmax=1)
    # Plot center of distributions 
    axs[1].plot(fit_dict['popt0'][0], fit_dict['popt0'][3], 'C3x')
    axs[1].plot(fit_dict['popt1'][0], fit_dict['popt1'][3], 'C3x')
    axs[1].plot(fit_dict['popt2'][0], fit_dict['popt2'][3], 'C3x')
    axs[0].set_xlabel('Integrated voltage, $I$')
    axs[0].set_ylabel('Integrated voltage, $Q$')
    axs[1].set_xlabel('Integrated voltage, $I$')
    axs[0].set_title('Measured shots')
    axs[1].set_title('Three gaussian mixture fit')
    # 3D plot
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ([(0, '#440154'), (.01,'#3b528b'),
                                                                     (.02,'#21918c'), (.3, '#fde725'), (1, 'w')]))
    norm = matplotlib.colors.Normalize(vmin=np.min(n_3_fit), vmax=np.max(n_3_fit))
    axs[2].axes.axes.set_position((.33, .15, .9, .8))
    axs[2].patch.set_visible(False)
    surf = axs[2].plot_surface(_X, _Y, n_3_fit, cmap=cmap, alpha=1,
                               linewidth=0, antialiased=True)
    axs[2].set_xticks([-1,-.5, 0, .5, 1])
    axs[2].set_xticklabels(['-1', '', '0', '', '1'])
    axs[2].set_yticks([-1,-.5, 0, .5, 1])
    axs[2].set_yticklabels(['-1', '', '0', '', '1'])
    axs[2].tick_params(axis='x', which='major', pad=-5)
    axs[2].tick_params(axis='y', which='major', pad=-5)
    axs[2].set_xlim(xbins[0], xbins[-1])
    axs[2].set_ylim(ybins[0], ybins[-1])
    axs[2].set_xlabel('Voltage, $I$', labelpad=-5)
    axs[2].set_ylabel('Voltage, $Q$', labelpad=-5)
    axs[1].text(1.2, 1.025, 'Three gaussian mixture 3D plot', transform=axs[1].transAxes, size=12)
    # horizontal colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cbar_ax = fig.add_axes([.67, .12, .22, .02])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    text = '\n'.join((r'$P_{|0\rangle}$'+f' = {fit_dict["qoi"]["P_0g"]*100:.2f}%',
                      r'$P_{|1\rangle}$'+f' = {fit_dict["qoi"]["P_1g"]*100:.2f}%',
                      r'$P_{|2\rangle}$'+f' = {fit_dict["qoi"]["P_2g"]*100:.2f}%'))
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[1].text(1.05, .95, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props, fontsize=11, zorder=100)
    fig.suptitle(f'{timestamp}\n2-state removal fraction qubit {qubit}', y=1.05)

def assignment_matrix_plotfn(
    M,
    qubit,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    im = ax.imshow(M, cmap=plt.cm.Reds, vmin=0, vmax=1)
    n = len(M)
    for i in range(n):
        for j in range(n):
            c = M[j,i]
            if abs(c) > .5:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center',
                             color = 'white')
            else:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f'$|{i}\\rangle$' for i in range(n)])
    ax.set_xlabel('Assigned state')
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([f'$|{i}\\rangle$' for i in range(n)])
    ax.set_ylabel('Prepared state')
    name = qubit
    if n==3:
        name = 'Qutrit'
    elif n==4:
        name = 'Ququat'
    ax.set_title(f'{timestamp}\n{name} assignment matrix qubit {qubit}')
    cbar_ax = fig.add_axes([.95, .15, .03, .7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('assignment probability')


class Repeated_LRU_experiment_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for LRU experiment.
    """
    def __init__(self,
                 qubit: str,
                 rounds: int,
                 heralded_init: bool = False,
                 h_state: bool = False,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.qubit = qubit
        self.rounds = rounds
        self.heralded_init = heralded_init
        self.h_state = h_state
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        # Perform measurement post-selection
        _cycle = 2*self.rounds+4
        if self.h_state:
            _cycle += 1
        if self.heralded_init:
            _cycle *= 2
        ############################################
        # Rotate shots in IQ plane
        ############################################
        # Sort shots
        _raw_shots = self.raw_data_dict['data'][:,1:]
        if self.heralded_init:
            _shots_0 = _raw_shots[2*self.rounds+1::_cycle]
            _shots_1 = _raw_shots[2*self.rounds+3::_cycle]
            _shots_2 = _raw_shots[2*self.rounds+5::_cycle]
            _shots_lru = _raw_shots[2*self.rounds+7::_cycle]
            if self.h_state:
                _shots_3 = _raw_shots[2*self.rounds+9::_cycle]
        else:
            _shots_0 = _raw_shots[2*self.rounds+0::_cycle]
            _shots_1 = _raw_shots[2*self.rounds+1::_cycle]
            _shots_2 = _raw_shots[2*self.rounds+2::_cycle]
            _shots_lru = _raw_shots[2*self.rounds+3::_cycle]
            if self.h_state:
                _shots_3 = _raw_shots[2*self.rounds+4::_cycle]
        # Save raw shots
        self.proc_data_dict['shots_0_IQ'] = _shots_0
        self.proc_data_dict['shots_1_IQ'] = _shots_1
        self.proc_data_dict['shots_2_IQ'] = _shots_2
        self.proc_data_dict['shots_lru_IQ'] = _shots_lru
        if self.h_state:
            self.proc_data_dict['shots_3_IQ'] = _shots_3
        # Rotate data
        center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
        center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
        center_2 = np.array([np.mean(_shots_2[:,0]), np.mean(_shots_2[:,1])])
        raw_shots = rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
        #####################################################
        # From this point onward raw shots has shape 
        # (nr_shots, nr_quadratures).
        # Post select based on heralding measurement result.
        #####################################################
        if self.heralded_init:
            raise NotImplementedError('Not implemented yet.')
            # # estimate post-selection threshold
            # shots_0 = raw_shots[1::_cycle, 0]
            # shots_1 = raw_shots[3::_cycle, 0]
            # ps_th = (np.mean(shots_0)+np.mean(shots_1))/2
            # # Sort heralding shots from experiment shots
            # ps_shots = raw_shots[0::2,0] # only I quadrature needed for postselection
            # exp_shots = raw_shots[1::2] # Here we want to keep both quadratures
            # # create post-selection mask
            # _mask = [ 1 if s<ps_th else np.nan for s in ps_shots ]
            # for i, s in enumerate(_mask):
            #     exp_shots[i] *= s
            # # Remove marked shots
            # Shots_0 = exp_shots[0::int(_cycle/2)]
            # Shots_1 = exp_shots[1::int(_cycle/2)]
            # Shots_2 = exp_shots[2::int(_cycle/2)]
            # Shots_lru = exp_shots[3::int(_cycle/2)]
            # Shots_0 = Shots_0[~np.isnan(Shots_0[:,0])]
            # Shots_1 = Shots_1[~np.isnan(Shots_1[:,0])]
            # Shots_2 = Shots_2[~np.isnan(Shots_2[:,0])]
            # Shots_lru = Shots_lru[~np.isnan(Shots_lru[:,0])]
        else:
            # Sort 0 and 1 shots
            Shots_0 = raw_shots[2*self.rounds+0::_cycle]
            Shots_1 = raw_shots[2*self.rounds+1::_cycle]
            Shots_2 = raw_shots[2*self.rounds+2::_cycle]
            Shots_lru = raw_shots[2*self.rounds+3::_cycle]
            if self.h_state:
                Shots_3 = raw_shots[2*self.rounds+4::_cycle]
            Shots_exp = {}
            Shots_ref = {}
            for r in range(self.rounds):
                Shots_exp[f'round {r+1}'] = raw_shots[r::_cycle]
                Shots_ref[f'round {r+1}'] = raw_shots[self.rounds+r::_cycle]
        ##############################################################
        # From this point onward Shots_<i> contains post-selected
        # shots of state <i> and has shape (nr_ps_shots, nr_quadtrs).
        # Next we will analyze shots projected along axis and 
        # therefore use a single quadrature. shots_<i> will be used
        # to denote that array of shots.
        ##############################################################
        self.qoi = {}
        if self.h_state:
            ############################################
            # Use classifier to assign states in the 
            # IQ plane and calculate qutrit fidelity.
            ############################################
            # Parse data for classifier
            data = np.concatenate((Shots_0, Shots_1, Shots_2, Shots_3))
            labels = [0 for s in Shots_0]+[1 for s in Shots_1]+\
                     [2 for s in Shots_2]+[3 for s in Shots_3]
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
            clf.fit(data, labels)
            dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
            Fid_dict = {}
            for state, shots in zip([    '0',     '1',     '2',     '3'],
                                    [Shots_0, Shots_1, Shots_2, Shots_3]):
                _res = clf.predict(shots)
                _fid = np.mean(_res == int(state))
                Fid_dict[state] = _fid
            Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
            # Get assignment fidelity matrix
            M = np.zeros((4,4))
            for i, shots in enumerate([Shots_0, Shots_1, Shots_2, Shots_3]):
                for j, state in enumerate(['0', '1', '2', '3']):
                    _res = clf.predict(shots)
                    M[i][j] = np.mean(_res == int(state))
            # Get leakage removal fraction 
            _res = clf.predict(Shots_lru)
            _vec = np.array([np.mean(_res == int('0')),
                             np.mean(_res == int('1')),
                             np.mean(_res == int('2')),
                             np.mean(_res == int('3'))])
            M_inv = np.linalg.inv(M)
            pop_vec = np.dot(_vec, M_inv)
            self.proc_data_dict['classifier'] = clf
            self.proc_data_dict['dec_bounds'] = dec_bounds
            self.proc_data_dict['Fid_dict'] = Fid_dict
            self.qoi['Fid_dict'] = Fid_dict
            self.qoi['Assignment_matrix'] = M
            self.qoi['pop_vec'] = pop_vec
            self.qoi['removal_fraction'] = 1-pop_vec[2]
        else:
            ############################################
            # Use classifier to assign states in the 
            # IQ plane and calculate qutrit fidelity.
            ############################################
            # Parse data for classifier
            data = np.concatenate((Shots_0, Shots_1, Shots_2))
            labels = [0 for s in Shots_0]+[1 for s in Shots_1]+[2 for s in Shots_2]
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
            clf.fit(data, labels)
            dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
            Fid_dict = {}
            for state, shots in zip([    '0',     '1',     '2'],
                                    [Shots_0, Shots_1, Shots_2]):
                _res = clf.predict(shots)
                _fid = np.mean(_res == int(state))
                Fid_dict[state] = _fid
            Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
            # Get assignment fidelity matrix
            M = np.zeros((3,3))
            for i, shots in enumerate([Shots_0, Shots_1, Shots_2]):
                for j, state in enumerate(['0', '1', '2']):
                    _res = clf.predict(shots)
                    M[i][j] = np.mean(_res == int(state))
            # Get leakage removal fraction 
            _res = clf.predict(Shots_lru)
            _vec = np.array([np.mean(_res == int('0')),
                             np.mean(_res == int('1')),
                             np.mean(_res == int('2'))])
            M_inv = np.linalg.inv(M)
            pop_vec = np.dot(_vec, M_inv)
            # Make it a 4x4 matrix
            M = np.append(M, [[0,0,0]], 0)
            M = np.append(M, [[0],[0],[0],[1]], 1)
            self.proc_data_dict['classifier'] = clf
            self.proc_data_dict['dec_bounds'] = dec_bounds
            self.proc_data_dict['Fid_dict'] = Fid_dict
            self.qoi['Fid_dict'] = Fid_dict
            self.qoi['Assignment_matrix'] = M
            self.qoi['pop_vec'] = pop_vec
            self.qoi['removal_fraction'] = 1-pop_vec[2]
        #########################################
        # Project data along axis perpendicular
        # to the decision boundaries.
        #########################################
        ############################
        # Projection along 10 axis.
        ############################
        # Rotate shots over 01 decision boundary axis
        shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1], dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
        shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1], dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
        # Take relavant quadrature
        shots_0 = shots_0[:,0]
        shots_1 = shots_1[:,0]
        n_shots_1 = len(shots_1)
        # find range
        _all_shots = np.concatenate((shots_0, shots_1))
        _range = (np.min(_all_shots), np.max(_all_shots))
        # Sort shots in unique values
        x0, n0 = np.unique(shots_0, return_counts=True)
        x1, n1 = np.unique(shots_1, return_counts=True)
        Fid_01, threshold_01 = _calculate_fid_and_threshold(x0, n0, x1, n1)
        # Histogram of shots for 1 and 2
        h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
        h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
        # Save processed data
        self.proc_data_dict['projection_01'] = {}
        self.proc_data_dict['projection_01']['h0'] = h0
        self.proc_data_dict['projection_01']['h1'] = h1
        self.proc_data_dict['projection_01']['bin_centers'] = bin_centers
        self.proc_data_dict['projection_01']['popt0'] = popt0
        self.proc_data_dict['projection_01']['popt1'] = popt1
        self.proc_data_dict['projection_01']['SNR'] = params_01['SNR']
        self.proc_data_dict['projection_01']['Fid'] = Fid_01
        self.proc_data_dict['projection_01']['threshold'] = threshold_01
        ############################
        # Projection along 12 axis.
        ############################
        # Rotate shots over 12 decision boundary axis
        shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
        shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
        # Take relavant quadrature
        shots_1 = shots_1[:,0]
        shots_2 = shots_2[:,0]
        n_shots_2 = len(shots_2)
        # find range
        _all_shots = np.concatenate((shots_1, shots_2))
        _range = (np.min(_all_shots), np.max(_all_shots))
        # Sort shots in unique values
        x1, n1 = np.unique(shots_1, return_counts=True)
        x2, n2 = np.unique(shots_2, return_counts=True)
        Fid_12, threshold_12 = _calculate_fid_and_threshold(x1, n1, x2, n2)
        # Histogram of shots for 1 and 2
        h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
        h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        popt1, popt2, params_12 = _fit_double_gauss(bin_centers, h1, h2)
        # Save processed data
        self.proc_data_dict['projection_12'] = {}
        self.proc_data_dict['projection_12']['h1'] = h1
        self.proc_data_dict['projection_12']['h2'] = h2
        self.proc_data_dict['projection_12']['bin_centers'] = bin_centers
        self.proc_data_dict['projection_12']['popt1'] = popt1
        self.proc_data_dict['projection_12']['popt2'] = popt2
        self.proc_data_dict['projection_12']['SNR'] = params_12['SNR']
        self.proc_data_dict['projection_12']['Fid'] = Fid_12
        self.proc_data_dict['projection_12']['threshold'] = threshold_12
        if not self.h_state:
            ############################
            # Projection along 02 axis.
            ############################
            # Rotate shots over 02 decision boundary axis
            shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            # Take relavant quadrature
            shots_0 = shots_0[:,0]
            shots_2 = shots_2[:,0]
            n_shots_2 = len(shots_2)
            # find range
            _all_shots = np.concatenate((shots_0, shots_2))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x0, n0 = np.unique(shots_0, return_counts=True)
            x2, n2 = np.unique(shots_2, return_counts=True)
            Fid_02, threshold_02 = _calculate_fid_and_threshold(x0, n0, x2, n2)
            # Histogram of shots for 1 and 2
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt0, popt2, params_02 = _fit_double_gauss(bin_centers, h0, h2)
            # Save processed data
            self.proc_data_dict['projection_02'] = {}
            self.proc_data_dict['projection_02']['h0'] = h0
            self.proc_data_dict['projection_02']['h2'] = h2
            self.proc_data_dict['projection_02']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_02']['popt0'] = popt0
            self.proc_data_dict['projection_02']['popt2'] = popt2
            self.proc_data_dict['projection_02']['SNR'] = params_02['SNR']
            self.proc_data_dict['projection_02']['Fid'] = Fid_02
            self.proc_data_dict['projection_02']['threshold'] = threshold_02
        else:
            ############################
            # Projection along 23 axis.
            ############################
            # Rotate shots over 23 decision boundary axis
            shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['23'], phi=np.pi/2)
            shots_3 = rotate_and_center_data(Shots_3[:,0],Shots_3[:,1],dec_bounds['mean'],dec_bounds['23'], phi=np.pi/2)
            # Take relavant quadrature
            shots_3 = shots_3[:,0]
            shots_2 = shots_2[:,0]
            n_shots_2 = len(shots_2)
            # find range
            _all_shots = np.concatenate((shots_3, shots_2))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x3, n3 = np.unique(shots_3, return_counts=True)
            x2, n2 = np.unique(shots_2, return_counts=True)
            Fid_23, threshold_23 = _calculate_fid_and_threshold(x3, n3, x2, n2)
            # Histogram of shots for 1 and 2
            h3, bin_edges = np.histogram(shots_3, bins=100, range=_range)
            h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt3, popt2, params_23 = _fit_double_gauss(bin_centers, h3, h2)
            # Save processed data
            self.proc_data_dict['projection_23'] = {}
            self.proc_data_dict['projection_23']['h3'] = h3
            self.proc_data_dict['projection_23']['h2'] = h2
            self.proc_data_dict['projection_23']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_23']['popt3'] = popt3
            self.proc_data_dict['projection_23']['popt2'] = popt2
            self.proc_data_dict['projection_23']['SNR'] = params_23['SNR']
            self.proc_data_dict['projection_23']['Fid'] = Fid_23
            self.proc_data_dict['projection_23']['threshold'] = threshold_23
            ############################
            # Projection along 30 axis.
            ############################
            # Rotate shots over 30 decision boundary axis
            shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['30'], phi=np.pi/2)
            shots_3 = rotate_and_center_data(Shots_3[:,0],Shots_3[:,1],dec_bounds['mean'],dec_bounds['30'], phi=np.pi/2)
            # Take relavant quadrature
            shots_3 = shots_3[:,0]
            shots_0 = shots_0[:,0]
            n_shots_3 = len(shots_3)
            # find range
            _all_shots = np.concatenate((shots_3, shots_0))
            _range = (np.min(_all_shots), np.max(_all_shots))
            # Sort shots in unique values
            x3, n3 = np.unique(shots_3, return_counts=True)
            x0, n0 = np.unique(shots_0, return_counts=True)
            Fid_30, threshold_30 = _calculate_fid_and_threshold(x3, n3, x0, n0)
            # Histogram of shots for 1 and 2
            h3, bin_edges = np.histogram(shots_3, bins=100, range=_range)
            h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
            popt3, popt0, params_30 = _fit_double_gauss(bin_centers, h3, h0)
            # Save processed data
            self.proc_data_dict['projection_30'] = {}
            self.proc_data_dict['projection_30']['h3'] = h3
            self.proc_data_dict['projection_30']['h0'] = h0
            self.proc_data_dict['projection_30']['bin_centers'] = bin_centers
            self.proc_data_dict['projection_30']['popt3'] = popt3
            self.proc_data_dict['projection_30']['popt0'] = popt0
            self.proc_data_dict['projection_30']['SNR'] = params_30['SNR']
            self.proc_data_dict['projection_30']['Fid'] = Fid_30
            self.proc_data_dict['projection_30']['threshold'] = threshold_30

        #########################################
        # Analyze repeated LRU experiment shots  #
        #########################################
        # Assign shots
        Shots_qutrit_exp = {}
        Shots_qutrit_ref = {}
        for r in range(self.rounds):
            Shots_qutrit_exp[f'round {r+1}'] = clf.predict(Shots_exp[f'round {r+1}'])
            Shots_qutrit_ref[f'round {r+1}'] = clf.predict(Shots_ref[f'round {r+1}'])
        # Calculate leakage in ancilla:
        Population_exp = {}
        Population_ref = {}
        def _get_pop_vector(Shots):
            p0 = np.mean(Shots==0)
            p1 = np.mean(Shots==1)
            p2 = np.mean(Shots==2)
            p3 = np.mean(Shots==3)
            return np.array([p0, p1, p2, p3])
        M_inv = np.linalg.inv(M)
        for r in range(self.rounds):
            _pop_vec = _get_pop_vector(Shots_qutrit_exp[f'round {r+1}'])
            Population_exp[f'round {r+1}'] = np.dot(_pop_vec, M_inv)
            _pop_vec = _get_pop_vector(Shots_qutrit_ref[f'round {r+1}'])
            Population_ref[f'round {r+1}'] = np.dot(_pop_vec, M_inv)
        Population_f_exp = np.array([Population_exp[k][2] for k in Population_exp.keys()])
        Population_f_ref = np.array([Population_ref[k][2] for k in Population_ref.keys()])
        self.proc_data_dict['Population_exp'] = Population_exp
        self.proc_data_dict['Population_f_exp'] = Population_f_exp
        self.proc_data_dict['Population_ref'] = Population_ref
        self.proc_data_dict['Population_f_ref'] = Population_f_ref
        if self.h_state:
            Population_h_exp = np.array([Population_exp[k][3] for k in Population_exp.keys()])
            Population_h_ref = np.array([Population_ref[k][3] for k in Population_ref.keys()])
            self.proc_data_dict['Population_h_exp'] = Population_h_exp
            self.proc_data_dict['Population_h_ref'] = Population_h_ref
        # Fit leakage and seepage rates
        from scipy.optimize import curve_fit
        def _func(n, L, S):
            return (1 - np.exp(-n*(S+L)))*L/(S+L) 
        _x = np.arange(0, self.rounds+1)
        _y = [0]+list(Population_f_exp)
        p0 = [.02, .5]
        popt, pcov = curve_fit(_func, _x, _y, p0=p0, bounds=((0,0), (1,1)))
        self.proc_data_dict['fit_res_exp'] = popt, pcov
        _y = [0]+list(Population_f_ref)
        popt, pcov = curve_fit(_func, _x, _y, p0=p0, bounds=((0,0), (1,1)))
        self.proc_data_dict['fit_res_ref'] = popt, pcov
        if self.h_state:
            _y = [0]+list(Population_h_exp)
            popt, pcov = curve_fit(_func, _x, _y, p0=p0, bounds=((0,0), (1,1)))
            self.proc_data_dict['fit_res_exp_h'] = popt, pcov
            _y = [0]+list(Population_h_ref)
            popt, pcov = curve_fit(_func, _x, _y, p0=p0, bounds=((0,0), (1,1)))
            self.proc_data_dict['fit_res_ref_h'] = popt, pcov

    def prepare_plots(self):

        self.axs_dict = {}
        fig = plt.figure(figsize=(8,4), dpi=100)
        if self.h_state:
            axs = [fig.add_subplot(121),
                   fig.add_subplot(422),
                   fig.add_subplot(424),
                   fig.add_subplot(426),
                   fig.add_subplot(428)]
        else:
            axs = [fig.add_subplot(121),
                   fig.add_subplot(322),
                   fig.add_subplot(324),
                   fig.add_subplot(326)]
        # fig.patch.set_alpha(0)
        self.axs_dict['SSRO_plot'] = axs[0]
        self.figs['SSRO_plot'] = fig
        self.plot_dicts['SSRO_plot'] = {
            'plotfn': ssro_IQ_projection_plotfn,
            'ax_id': 'SSRO_plot',
            'shots_0': self.proc_data_dict['shots_0_IQ'],
            'shots_1': self.proc_data_dict['shots_1_IQ'],
            'shots_2': self.proc_data_dict['shots_2_IQ'],
            'shots_3': self.proc_data_dict['shots_3_IQ'] if self.h_state \
                       else None,
            'projection_01': self.proc_data_dict['projection_01'],
            'projection_12': self.proc_data_dict['projection_12'],
            'projection_02': None if self.h_state else\
                             self.proc_data_dict['projection_02'],
            'projection_23': None if not self.h_state else\
                             self.proc_data_dict['projection_23'],
            'projection_30': None if not self.h_state else\
                             self.proc_data_dict['projection_30'],
            'classifier': self.proc_data_dict['classifier'],
            'dec_bounds': self.proc_data_dict['dec_bounds'],
            'Fid_dict': self.proc_data_dict['Fid_dict'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, ax = plt.subplots(figsize=(3.25,3.25), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Leakage_histogram'] = ax
        self.figs['Leakage_histogram'] = fig
        self.plot_dicts['Leakage_histogram'] = {
            'plotfn': leakage_hist_plotfn,
            'ax_id': 'Leakage_histogram',
            'shots_0': self.proc_data_dict['shots_0_IQ'],
            'shots_1': self.proc_data_dict['shots_1_IQ'],
            'shots_2': self.proc_data_dict['shots_2_IQ'],
            'shots_3': self.proc_data_dict['shots_3_IQ'] if self.h_state \
                       else None,
            'shots_lru': self.proc_data_dict['shots_lru_IQ'],
            'classifier': self.proc_data_dict['classifier'],
            'dec_bounds': self.proc_data_dict['dec_bounds'],
            'qoi': self.qoi,
            'pop_vec': self.qoi['pop_vec'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, ax = plt.subplots(figsize=(3,3), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Assignment_matrix'] = ax
        self.figs['Assignment_matrix'] = fig
        self.plot_dicts['Assignment_matrix'] = {
            'plotfn': assignment_matrix_plotfn,
            'ax_id': 'Assignment_matrix',
            'M': self.qoi['Assignment_matrix'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, axs = plt.subplots(figsize=(5,3), nrows=2, sharex=True, dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Population_vs_rounds'] = axs[0]
        self.figs['Population_vs_rounds'] = fig
        self.plot_dicts['Population_vs_rounds'] = {
            'plotfn': Population_vs_rounds_plotfn,
            'ax_id': 'Population_vs_rounds',
            'rounds': self.rounds,
            'Population_exp': self.proc_data_dict['Population_exp'],
            'fit_res_exp': self.proc_data_dict['fit_res_exp'],
            'Population_ref': self.proc_data_dict['Population_ref'],
            'fit_res_ref': self.proc_data_dict['fit_res_ref'],
            'fit_res_exp_h': self.proc_data_dict['fit_res_exp_h'] if self.h_state \
                             else None,
            'fit_res_ref_h': self.proc_data_dict['fit_res_ref_h'] if self.h_state \
                             else None,
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def Population_vs_rounds_plotfn(
    rounds,
    Population_exp,
    fit_res_exp,
    Population_ref,
    fit_res_ref,
    timestamp,
    qubit,
    ax,
    fit_res_exp_h=None,
    fit_res_ref_h=None,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    def _func(n, L, S):
        return (1 - np.exp(-n*(S+L)))*L/(S+L)
    popt_exp, pcov_exp = fit_res_exp
    perr_exp = np.sqrt(np.abs(np.diag(pcov_exp)))
    popt_ref, pcov_ref = fit_res_ref
    perr_ref = np.sqrt(np.abs(np.diag(pcov_ref)))
    Population_g_ref = np.array([Population_ref[k][0] for k in Population_ref.keys()])
    Population_e_ref = np.array([Population_ref[k][1] for k in Population_ref.keys()])
    Population_f_ref = np.array([Population_ref[k][2] for k in Population_ref.keys()])
    Population_g_exp = np.array([Population_exp[k][0] for k in Population_exp.keys()])
    Population_e_exp = np.array([Population_exp[k][1] for k in Population_exp.keys()])
    Population_f_exp = np.array([Population_exp[k][2] for k in Population_exp.keys()])
    if not (fit_res_exp_h is None):
        Population_h_ref = np.array([Population_ref[k][3] for k in Population_ref.keys()])
        Population_h_exp = np.array([Population_exp[k][3] for k in Population_exp.keys()])
        popt_exp_h, pcov_exp_h = fit_res_exp_h
        perr_exp_h = np.sqrt(np.abs(np.diag(pcov_exp_h)))
        popt_ref_h, pcov_ref_h = fit_res_ref_h
        perr_ref_h = np.sqrt(np.abs(np.diag(pcov_ref_h)))
    _rounds_arr = np.arange(rounds)+1
    axs[0].plot(_rounds_arr, Population_g_ref, 'C0-', alpha=.5, label='$|g\\rangle_\\mathrm{{Ref.}}$')
    axs[0].plot(_rounds_arr, Population_e_ref, 'C3-', alpha=.5, label='$|e\\rangle_\\mathrm{{Ref.}}$')
    axs[0].plot(_rounds_arr, Population_g_exp, 'C0-', label='$|g\\rangle_\\mathrm{{Gate}}$')
    axs[0].plot(_rounds_arr, Population_e_exp, 'C3-', label='$|e\\rangle_\\mathrm{{Gate}}$')
    axs[1].plot(_rounds_arr, _func(_rounds_arr, *popt_ref), 'k--')
    axs[1].plot(_rounds_arr, _func(_rounds_arr, *popt_exp), 'k--')
    if not (fit_res_exp_h is None):
        axs[1].plot(_rounds_arr, _func(_rounds_arr, *popt_ref_h), '--', color='goldenrod')
        axs[1].plot(_rounds_arr, _func(_rounds_arr, *popt_exp_h), '--', color='goldenrod')
    axs[1].plot(_rounds_arr, Population_f_ref, 'C2-', alpha=.5, label='$|f\\rangle_\\mathrm{{Ref.}}$')
    axs[1].plot(_rounds_arr, Population_f_exp, 'C2-', label='$|f\\rangle_\\mathrm{{Gate}}$')
    if not (fit_res_exp_h is None):
        axs[1].plot(_rounds_arr, Population_h_ref, '-', color='gold', alpha=.5, label='$|h\\rangle_\\mathrm{{Ref.}}$')
        axs[1].plot(_rounds_arr, Population_h_exp, '-', color='gold', label='$|h\\rangle_\\mathrm{{Gate}}$')
    txtstr = 'Ref.:\n'+\
             f'$L_1={popt_ref[0]*100:.2f} \\pm {perr_ref[0]:.2f}\\%$\n'+\
             f'$L_2={popt_ref[1]*100:.2f} \\pm {perr_ref[1]:.2f}\\%$\n'
    if not (fit_res_exp_h is None):
        txtstr+= f'$L_1^h={popt_ref_h[0]*100:.2f} \\pm {perr_ref_h[0]:.2f}\\%$\n'+\
                 f'$L_2^h={popt_ref_h[1]*100:.2f} \\pm {perr_ref_h[1]:.2f}\\%$\n'
    txtstr+= 'Gate:\n'+\
             f'$L_1={popt_exp[0]*100:.2f} \\pm {perr_exp[0]:.2f}\\%$\n'+\
             f'$L_2={popt_exp[1]*100:.2f} \\pm {perr_exp[1]:.2f}\\%$'
    if not (fit_res_exp_h is None):
        txtstr+= f'\n$L_1^h={popt_exp_h[0]*100:.2f} \\pm {perr_exp_h[0]:.2f}\\%$\n'+\
                 f'$L_2^h={popt_exp_h[1]*100:.2f} \\pm {perr_exp_h[1]:.2f}\\%$'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[1].text(1.3, 1, txtstr, transform=axs[0].transAxes,
                verticalalignment='top', bbox=props)
    axs[0].legend(loc=2, frameon=False, bbox_to_anchor=(1.01,1))
    axs[1].legend(loc=2, frameon=False, bbox_to_anchor=(1.01,1))
    axs[0].set_ylabel('Population')
    axs[1].set_ylabel('Population')
    axs[1].set_xlabel('Rounds')
    axs[0].set_title(f'{timestamp}\nRepeated LRU experiment {qubit}')


def _get_expected_value(operator, state, n):
    m = 1
    for i in range(n):
        if operator[i] == 'Z' and state[i] == '1':
            m *= -1
    return m

def _gen_M_matrix(n):
    # List of different Operators
    ops = ['I','Z']
    Operators = [''.join(op) for op in itertools.product(ops, repeat=n)]
    # List of calibration points
    states = ['0','1']
    Cal_points = [''.join(s) for s in itertools.product(states, repeat=n)]
    # Calculate M matrix
    M = np.zeros((2**n, 2**n), dtype=int)
    for j, state in enumerate(Cal_points):
        Betas = np.ones(len(Operators))
        for i in range(2**n):
            Betas[i] = _get_expected_value(Operators[i], state, n)
        M[j] = Betas
    M = np.linalg.pinv(M) # invert matrix
    return M

def _get_Beta_matrix(Cal_shots_dig, n):
    '''
    Calculate RO error model (Beta) matrix.
    <Cal_shots_dig> should be a dictionary with the format:
    Cal_shots_dig[<qubit>][<state>] = array of shots with +1 and -1.
    '''
    # List of different Operators
    ops = ['I','Z']
    Operators = [''.join(op) for op in itertools.product(ops, repeat=n)]
    # List of qubits
    Qubits = list(Cal_shots_dig.keys())
    # Calculate Beta matrix
    H = {}
    B = {}
    M = _gen_M_matrix(n)
    for op in Operators[1:]:
        H[op] = np.zeros(2**n)
        for i, state in enumerate(Cal_shots_dig[Qubits[0]].keys()):
            correlator = 1
            for j, qubit in enumerate(Qubits):
                if op[j] == 'Z':
                    correlator *= np.array(Cal_shots_dig[Qubits[j]][state])
            H[op][i] = np.mean(correlator)
        B[op] = np.dot(M, H[op])
    return B

def _correct_pauli_vector(Beta_matrix, P_vector):
    '''
    Applies readout correction from Beta matrix
    to a single qubit pauli vector.
    '''
    B_matrix = np.array([Beta_matrix[key][1:] for key in Beta_matrix.keys()])
    B_0 = np.array([Beta_matrix[key][0] for key in Beta_matrix.keys()])
    iB_matrix = np.linalg.inv(B_matrix)
    # This part is ony valid for single qubit pauli vectors
    _p_vec_corrected = (P_vector[1:]-B_0)*iB_matrix
    P_corrected = np.concatenate(([1], _p_vec_corrected[0]))
    return P_corrected

def _gen_density_matrix(mx, my, mz):
    Pauli_ops = {}
    Pauli_ops['I'] = np.array([[  1,  0],
                               [  0,  1]])
    Pauli_ops['Z'] = np.array([[  1,  0],
                               [  0, -1]])
    Pauli_ops['X'] = np.array([[  0,  1],
                               [  1,  0]])
    Pauli_ops['Y'] = np.array([[  0,-1j],
                               [ 1j,  0]])
    rho = (Pauli_ops['I'] + mx*Pauli_ops['X'] + my*Pauli_ops['Y'] + mz*Pauli_ops['Z'])/2
    return rho

def _get_choi_from_PTM(PTM, dim=2):
    Pauli_ops = {}
    Pauli_ops['I'] = np.array([[  1,  0],
                               [  0,  1]])
    Pauli_ops['Z'] = np.array([[  1,  0],
                               [  0, -1]])
    Pauli_ops['X'] = np.array([[  0,  1],
                               [  1,  0]])
    Pauli_ops['Y'] = np.array([[  0,-1j],
                               [ 1j,  0]])
    paulis = [Pauli_ops['I'],
              Pauli_ops['X'], 
              Pauli_ops['Y'], 
              Pauli_ops['Z']]
    choi_state = np.zeros([dim**2, dim**2], dtype='complex')
    for i in range(dim**2):
        for j in range(dim**2):
            choi_state += 1/dim**2 * PTM[i,j] * np.kron(paulis[j].transpose(), paulis[i]) 
    return choi_state

def _get_pauli_transfer_matrix(Pauli_0, Pauli_1,
                              Pauli_p, Pauli_m,
                              Pauli_ip, Pauli_im, 
                              M_in = None):
    if type(M_in) == type(None):
        M_in = np.array([[1, 0, 0, 1],  # 0
                         [1, 0, 0,-1],  # 1
                         [1, 1, 0, 0],  # +
                         [1,-1, 0, 0],  # -
                         [1, 0, 1, 0],  # +i
                         [1, 0,-1, 0]]) # -i
    M_in=np.transpose(M_in)
    M_out= np.array([Pauli_0,
                     Pauli_1,
                     Pauli_p,
                     Pauli_m,
                     Pauli_ip,
                     Pauli_im])
    M_out=np.transpose(M_out)
    R = np.matmul(M_out, np.linalg.pinv(M_in))
    # Check for physicality
    choi_state = _get_choi_from_PTM(R)
    if (np.real(np.linalg.eig(choi_state)[0]) < 0).any():
        print('PTM is unphysical')
    else:
        print('PTM is physical')
    return R

def _PTM_angle(angle):
    angle *= np.pi/180
    R = np.array([[ 1,             0,             0, 0],
                  [ 0, np.cos(angle),-np.sin(angle), 0],
                  [ 0, np.sin(angle), np.cos(angle), 0],
                  [ 0,             0,             0, 1]])
    return R

def _PTM_fidelity(R, R_id):
    return (np.trace(np.matmul(np.transpose(R_id), R))/2+1)/3

class LRU_process_tomo_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for LRU process tomography experiment.
    """
    def __init__(self,
                 qubit: str,
                 post_select_2state: bool = False,
                 fit_3gauss: bool = False,
                 t_start: str = None, 
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.qubit = qubit
        self.post_select_2state = post_select_2state
        self.fit_3gauss = fit_3gauss
        if auto:
            self.run_analysis()

    def extract_data(self):
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        _cycle = 18+4
        ################################
        # Rotate shots in IQ plane
        ################################
        # Sort shots
        _raw_shots = self.raw_data_dict['data'][:,1:]
        _shots_0 = _raw_shots[18::_cycle]
        _shots_1 = _raw_shots[19::_cycle]
        _shots_2 = _raw_shots[20::_cycle]
        # Rotate data
        center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
        center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
        center_2 = np.array([np.mean(_shots_2[:,0]), np.mean(_shots_2[:,1])])
        raw_shots = rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
        Shots_0 = raw_shots[18::_cycle]
        Shots_1 = raw_shots[19::_cycle]
        Shots_2 = raw_shots[20::_cycle]
        Shots_3 = raw_shots[21::_cycle]
        self.proc_data_dict['shots_0_IQ'] = Shots_0
        self.proc_data_dict['shots_1_IQ'] = Shots_1
        self.proc_data_dict['shots_2_IQ'] = Shots_2
        self.proc_data_dict['shots_3_IQ'] = Shots_3
        # Use classifier for data
        data = np.concatenate((Shots_0, Shots_1, Shots_2))
        labels = [0 for s in Shots_0]+[1 for s in Shots_1]+[2 for s in Shots_2]
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis()
        clf.fit(data, labels)
        dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
        Fid_dict = {}
        for state, shots in zip([    '0',     '1',     '2'],
                                [Shots_0, Shots_1, Shots_2]):
            _res = clf.predict(shots)
            _fid = np.mean(_res == int(state))
            Fid_dict[state] = _fid
        Fid_dict['avg'] = np.mean([f for f in Fid_dict.values()])
        # Get assignment fidelity matrix
        M = np.zeros((3,3))
        for i, shots in enumerate([Shots_0, Shots_1, Shots_2]):
            for j, state in enumerate(['0', '1', '2']):
                _res = clf.predict(shots)
                M[i][j] = np.mean(_res == int(state))
        # Get leakage removal fraction 
        _res = clf.predict(Shots_3)
        _vec = np.array([np.mean(_res == int('0')),
                         np.mean(_res == int('1')),
                         np.mean(_res == int('2'))])
        M_inv = np.linalg.inv(M)
        pop_vec = np.dot(_vec, M_inv)
        self.proc_data_dict['classifier'] = clf
        self.proc_data_dict['dec_bounds'] = dec_bounds
        self.proc_data_dict['Fid_dict'] = Fid_dict
        self.qoi = {}
        self.qoi['Fid_dict'] = Fid_dict
        self.qoi['Assignment_matrix'] = M
        self.qoi['pop_vec'] = pop_vec
        self.qoi['removal_fraction'] = 1-pop_vec[2]
        #########################################
        # Project data along axis perpendicular
        # to the decision boundaries.
        #########################################
        ############################
        # Projection along 01 axis.
        ############################
        # Rotate shots over 01 axis
        shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
        shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['01'], phi=np.pi/2)
        # Take relavant quadrature
        shots_0 = shots_0[:,0]
        shots_1 = shots_1[:,0]
        n_shots_1 = len(shots_1)
        # find range
        _all_shots = np.concatenate((shots_0, shots_1))
        _range = (np.min(_all_shots), np.max(_all_shots))
        # Sort shots in unique values
        x0, n0 = np.unique(shots_0, return_counts=True)
        x1, n1 = np.unique(shots_1, return_counts=True)
        Fid_01, threshold_01 = _calculate_fid_and_threshold(x0, n0, x1, n1)
        # Histogram of shots for 1 and 2
        h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
        h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
        # Save processed data
        self.proc_data_dict['projection_01'] = {}
        self.proc_data_dict['projection_01']['h0'] = h0
        self.proc_data_dict['projection_01']['h1'] = h1
        self.proc_data_dict['projection_01']['bin_centers'] = bin_centers
        self.proc_data_dict['projection_01']['popt0'] = popt0
        self.proc_data_dict['projection_01']['popt1'] = popt1
        self.proc_data_dict['projection_01']['SNR'] = params_01['SNR']
        self.proc_data_dict['projection_01']['Fid'] = Fid_01
        self.proc_data_dict['projection_01']['threshold'] = threshold_01
        ############################
        # Projection along 12 axis.
        ############################
        # Rotate shots over 12 axis
        shots_1 = rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
        shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
        # Take relavant quadrature
        shots_1 = shots_1[:,0]
        shots_2 = shots_2[:,0]
        n_shots_2 = len(shots_2)
        # find range
        _all_shots = np.concatenate((shots_1, shots_2))
        _range = (np.min(_all_shots), np.max(_all_shots))
        # Sort shots in unique values
        x1, n1 = np.unique(shots_1, return_counts=True)
        x2, n2 = np.unique(shots_2, return_counts=True)
        Fid_12, threshold_12 = _calculate_fid_and_threshold(x1, n1, x2, n2)
        # Histogram of shots for 1 and 2
        h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
        h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        popt1, popt2, params_12 = _fit_double_gauss(bin_centers, h1, h2)
        # Save processed data
        self.proc_data_dict['projection_12'] = {}
        self.proc_data_dict['projection_12']['h1'] = h1
        self.proc_data_dict['projection_12']['h2'] = h2
        self.proc_data_dict['projection_12']['bin_centers'] = bin_centers
        self.proc_data_dict['projection_12']['popt1'] = popt1
        self.proc_data_dict['projection_12']['popt2'] = popt2
        self.proc_data_dict['projection_12']['SNR'] = params_12['SNR']
        self.proc_data_dict['projection_12']['Fid'] = Fid_12
        self.proc_data_dict['projection_12']['threshold'] = threshold_12
        ############################
        # Projection along 02 axis.
        ############################
        # Rotate shots over 02 axis
        shots_0 = rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
        shots_2 = rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
        # Take relavant quadrature
        shots_0 = shots_0[:,0]
        shots_2 = shots_2[:,0]
        n_shots_2 = len(shots_2)
        # find range
        _all_shots = np.concatenate((shots_0, shots_2))
        _range = (np.min(_all_shots), np.max(_all_shots))
        # Sort shots in unique values
        x0, n0 = np.unique(shots_0, return_counts=True)
        x2, n2 = np.unique(shots_2, return_counts=True)
        Fid_02, threshold_02 = _calculate_fid_and_threshold(x0, n0, x2, n2)
        # Histogram of shots for 1 and 2
        h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
        h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        popt0, popt2, params_02 = _fit_double_gauss(bin_centers, h0, h2)
        # Save processed data
        self.proc_data_dict['projection_02'] = {}
        self.proc_data_dict['projection_02']['h0'] = h0
        self.proc_data_dict['projection_02']['h2'] = h2
        self.proc_data_dict['projection_02']['bin_centers'] = bin_centers
        self.proc_data_dict['projection_02']['popt0'] = popt0
        self.proc_data_dict['projection_02']['popt2'] = popt2
        self.proc_data_dict['projection_02']['SNR'] = params_02['SNR']
        self.proc_data_dict['projection_02']['Fid'] = Fid_02
        self.proc_data_dict['projection_02']['threshold'] = threshold_02
        ###############################
        # Fit 3-gaussian mixture
        ###############################
        if self.fit_3gauss:
            _all_shots = np.concatenate((self.proc_data_dict['shots_0_IQ'], 
                                         self.proc_data_dict['shots_1_IQ'], 
                                         self.proc_data_dict['shots_2_IQ']))
            _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1,
                            np.max(np.abs(_all_shots[:,1]))*1.1 ])
            def _histogram_2d(x_data, y_data, lim):
                _bins = (np. linspace(-lim, lim, 201), np.linspace(-lim, lim, 201))
                n, _xbins, _ybins = np.histogram2d(x_data, y_data, bins=_bins)
                return n.T, _xbins, _ybins
            # 2D histograms
            n_0, xbins, ybins = _histogram_2d(*self.proc_data_dict['shots_0_IQ'].T, lim=_lim)
            n_1, xbins, ybins = _histogram_2d(*self.proc_data_dict['shots_1_IQ'].T, lim=_lim)
            n_2, xbins, ybins = _histogram_2d(*self.proc_data_dict['shots_2_IQ'].T, lim=_lim)
            n_3, xbins, ybins = _histogram_2d(*self.proc_data_dict['shots_3_IQ'].T, lim=_lim)
            # bin centers
            xbins_c, ybins_c = (xbins[1:]+xbins[:-1])/2, (ybins[1:]+ybins[:-1])/2 
            popt0, popt1, popt2, qoi = _fit_triple_gauss(xbins_c, ybins_c, n_0, n_1, n_2)
            popt3, _popt1, _popt2, _qoi = _fit_triple_gauss(xbins_c, ybins_c, n_3, n_1, n_2)
            self.proc_data_dict['3gauss_fit'] = {}
            self.proc_data_dict['3gauss_fit']['xbins'] = xbins
            self.proc_data_dict['3gauss_fit']['ybins'] = ybins
            self.proc_data_dict['3gauss_fit']['n0'] = n_0
            self.proc_data_dict['3gauss_fit']['n1'] = n_1
            self.proc_data_dict['3gauss_fit']['n2'] = n_2
            self.proc_data_dict['3gauss_fit']['n3'] = n_3
            self.proc_data_dict['3gauss_fit']['popt0'] = popt0
            self.proc_data_dict['3gauss_fit']['popt1'] = popt1
            self.proc_data_dict['3gauss_fit']['popt2'] = popt2
            self.proc_data_dict['3gauss_fit']['popt3'] = popt3
            self.proc_data_dict['3gauss_fit']['qoi'] = _qoi
            self.proc_data_dict['3gauss_fit']['removal_fraction'] = 1-_qoi['P_2g']
            self.qoi['removal_fraction_gauss_fit'] = 1-_qoi['P_2g']

        ################################
        # Process tomo analysis
        ################################
        _thresh = self.proc_data_dict['projection_01']['threshold']
        if self.post_select_2state:
            _dig_shots = clf.predict(raw_shots)
        else:
            _dig_shots = [0 if s < _thresh else 1 for s in raw_shots[:,0]]
        # Beta matrix for readout corrections
        cal_shots_dig = {self.qubit:{}}
        cal_shots_dig[self.qubit]['0'] = [+1 if s < _thresh else -1 for s in Shots_0[:,0]]
        cal_shots_dig[self.qubit]['1'] = [+1 if s < _thresh else -1 for s in Shots_1[:,0]]
        Beta_matrix = _get_Beta_matrix(cal_shots_dig, n=1)
        # Calculate expectation values for each state
        States = ['0', '1', '+', '-', '+i', '-i']
        Operators = ['Z', 'X', 'Y']
        # Parse shots and post-select leakage
        dig_shots = {}
        leak_frac = 0
        for i, state in enumerate(States):
            dig_shots[state] = {}
            for j, op in enumerate(Operators):
                _shot_list = _dig_shots[i*3+j::_cycle]
                # Post-select on leakage
                _shot_list = [ s for s in _shot_list if s!=2 ]
                _leak_frac = 1-len(_shot_list)/len(_dig_shots[i*3+j::_cycle])
                leak_frac += _leak_frac/18
                # Turn in meas outcomes (+1/-1)
                dig_shots[state][op] = 1 - 2*np.array(_shot_list)
        # Build density matrices and pauli vector for each input state
        # Ideal states
        Density_matrices_ideal = {}
        Density_matrices_ideal['0'] = _gen_density_matrix(mx=0, my=0, mz=+1)
        Density_matrices_ideal['1'] = _gen_density_matrix(mx=0, my=0, mz=-1)
        Density_matrices_ideal['+'] = _gen_density_matrix(mx=+1, my=0, mz=0)
        Density_matrices_ideal['-'] = _gen_density_matrix(mx=-1, my=0, mz=0)
        Density_matrices_ideal['+i'] = _gen_density_matrix(mx=0, my=+1, mz=0)
        Density_matrices_ideal['-i'] = _gen_density_matrix(mx=0, my=-1, mz=0)
        Density_matrices = {}
        Pauli_vectors = {}
        for state in States:
            mz = np.mean(dig_shots[state]['Z'])
            mx = np.mean(dig_shots[state]['X'])
            my = np.mean(dig_shots[state]['Y'])
            p_vector = np.array([1, mx, my, mz])
            Pauli_vectors[state] = _correct_pauli_vector(Beta_matrix, p_vector)
            rho = _gen_density_matrix(*Pauli_vectors[state][1:])
            Density_matrices[state] = rho
        # Get PTM
        PTM = _get_pauli_transfer_matrix(Pauli_0 = Pauli_vectors['0'], 
                                         Pauli_1 = Pauli_vectors['1'],
                                         Pauli_p = Pauli_vectors['+'], 
                                         Pauli_m = Pauli_vectors['-'],
                                         Pauli_ip = Pauli_vectors['+i'], 
                                         Pauli_im = Pauli_vectors['-i'])
        # Calculate angle of plus state
        angle_p = np.arctan2(Pauli_vectors['+'][2],Pauli_vectors['+'][1])*180/np.pi
        # get ideal PTM with same angle and
        # calculate fidelity of extracted PTM.
        PTM_id = np.eye(4)
        F_PTM = _PTM_fidelity(PTM, PTM_id)
        PTM_id = _PTM_angle(angle_p)
        F_PTM_rotated = _PTM_fidelity(PTM, PTM_id)
        self.proc_data_dict['PS_frac'] = leak_frac
        self.proc_data_dict['Density_matrices'] = Density_matrices
        self.proc_data_dict['Density_matrices_ideal'] = Density_matrices_ideal
        self.proc_data_dict['PTM'] = PTM
        self.proc_data_dict['angle_p'] = angle_p
        self.proc_data_dict['F_PTM'] = F_PTM
        self.proc_data_dict['F_PTM_r'] = F_PTM_rotated

    def prepare_plots(self):
        self.axs_dict = {}
        fig = plt.figure(figsize=(8,4), dpi=100)
        axs = [fig.add_subplot(121),
               fig.add_subplot(322),
               fig.add_subplot(324),
               fig.add_subplot(326)]
        # fig.patch.set_alpha(0)
        self.axs_dict['SSRO_plot'] = axs[0]
        self.figs['SSRO_plot'] = fig
        self.plot_dicts['SSRO_plot'] = {
            'plotfn': ssro_IQ_projection_plotfn,
            'ax_id': 'SSRO_plot',
            'shots_0': self.proc_data_dict['shots_0_IQ'],
            'shots_1': self.proc_data_dict['shots_1_IQ'],
            'shots_2': self.proc_data_dict['shots_2_IQ'],
            'projection_01': self.proc_data_dict['projection_01'],
            'projection_12': self.proc_data_dict['projection_12'],
            'projection_02': self.proc_data_dict['projection_02'],
            'classifier': self.proc_data_dict['classifier'],
            'dec_bounds': self.proc_data_dict['dec_bounds'],
            'Fid_dict': self.proc_data_dict['Fid_dict'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, ax = plt.subplots(figsize=(3,3), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Assignment_matrix'] = ax
        self.figs['Assignment_matrix'] = fig
        self.plot_dicts['Assignment_matrix'] = {
            'plotfn': assignment_matrix_plotfn,
            'ax_id': 'Assignment_matrix',
            'M': self.qoi['Assignment_matrix'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, ax = plt.subplots(figsize=(3.25,3.25), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Leakage_histogram'] = ax
        self.figs['Leakage_histogram'] = fig
        self.plot_dicts['Leakage_histogram'] = {
            'plotfn': leakage_hist_plotfn,
            'ax_id': 'Leakage_histogram',
            'shots_0': self.proc_data_dict['shots_0_IQ'],
            'shots_1': self.proc_data_dict['shots_1_IQ'],
            'shots_2': self.proc_data_dict['shots_2_IQ'],
            'shots_lru': self.proc_data_dict['shots_3_IQ'],
            'classifier': self.proc_data_dict['classifier'],
            'dec_bounds': self.proc_data_dict['dec_bounds'],
            'pop_vec': self.qoi['pop_vec'],
            'qoi': self.qoi,
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig = plt.figure(figsize=(3*1.5, 6*1.5), dpi=200)
        axs =[]
        for i in range(6):
            axs.append(fig.add_subplot(3, 2, i+1 , projection='3d', azim=-35, elev=35))
        # fig.patch.set_alpha(0)
        self.axs_dict['Density_matrices'] = axs[0]
        self.figs['Density_matrices'] = fig
        self.plot_dicts['Density_matrices'] = {
            'plotfn': density_matrices_plotfn,
            'ax_id': 'Density_matrices',
            'Density_matrices': self.proc_data_dict['Density_matrices'],
            'Density_matrices_ideal': self.proc_data_dict['Density_matrices_ideal'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, ax = plt.subplots(figsize=(3.5,3.5), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Pauli_transfer_matrix'] = ax
        self.figs['Pauli_transfer_matrix'] = fig
        self.plot_dicts['Pauli_transfer_matrix'] = {
            'plotfn': PTM_plotfn,
            'ax_id': 'Pauli_transfer_matrix',
            'R': self.proc_data_dict['PTM'],
            'F_PTM': self.proc_data_dict['F_PTM'],
            'angle': self.proc_data_dict['angle_p'],
            'F_PTM_r': self.proc_data_dict['F_PTM_r'],
            'title': r'PTM, $\mathcal{R}_{\mathrm{LRU}}$'+f' {self.qubit}',
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        if self.fit_3gauss:
            fig = plt.figure(figsize=(12,4), dpi=100)
            axs = [None,None,None]
            axs[0] = fig.add_subplot(1, 3, 1)
            axs[2] = fig.add_subplot(1, 3, 3 , projection='3d', elev=20)
            axs[1] = fig.add_subplot(1, 3, 2)
            # fig.patch.set_alpha(0)
            self.axs_dict['Three_gauss_fit'] = axs[0]
            self.figs['Three_gauss_fit'] = fig
            self.plot_dicts['Three_gauss_fit'] = {
                'plotfn': gauss_fit2D_plotfn,
                'ax_id': 'Three_gauss_fit',
                'fit_dict': self.proc_data_dict['3gauss_fit'],
                'qubit': self.qubit,
                'timestamp': self.timestamp
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def _plot_density_matrix(rho, f, ax, rho_ideal=None, state=None, cbar=True):
    if state is None:
        pass
    else:
        ax.set_title('Logical state '+state, pad=10, fontsize=5)
    ax.set_xticks([-.1, .9])
    ax.set_yticks([-.1, .9])
    ax.set_xticklabels(['$0$', '$1$'], rotation=0, fontsize=4.5)
    ax.set_yticklabels(['$0$', '$1$'], rotation=0, fontsize=4.5)
    ax.tick_params(axis='x', which='major', pad=-6)
    ax.tick_params(axis='y', which='major', pad=-7)
    ax.tick_params(axis='z', which='major', pad=-4)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")
    ax.set_zticks(np.linspace(0, 1, 3))
    ax.set_zticklabels(['0', '0.5', '1'], fontsize=4)
    ax.set_zlim(0, 1)

    xedges = np.arange(-.75, 2, 1)
    yedges = np.arange(-.75, 2, 1)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = .8
    dz = np.abs(rho).ravel()
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C3",'darkseagreen',"C0",
                                                                    'antiquewhite',"C3"])
    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    color=cmap(norm([np.angle(e) for e in rho.ravel()]))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='max',
             color=color, alpha=1 , edgecolor='black', linewidth=.1)
    if rho_ideal is not None:
        dz1 = np.abs(rho_ideal).ravel()
        color1=cmap(norm([np.angle(e) for e in rho_ideal.ravel()]))
        # selector
        s = [k for k in range(len(dz1)) if dz1[k] > .15]
        ax.bar3d(xpos[s], ypos[s], dz[s], dx, dy, dz=dz1[s]-dz[s], zsort='min',
                 color=color1[s], alpha=.25, edgecolor='black', linewidth=.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if cbar:
        cb = f.colorbar(sm)
        cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.set_ticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
        cb.set_label('arg', fontsize=3)
        cb.ax.tick_params(labelsize=3)
        cb.outline.set_linewidth(.5)
        cb.ax.tick_params(labelsize=3, width=.5, length=1, pad=1)
        f.axes[-1].set_position([.85, .2, .05, .6])
        f.axes[-1].get_yaxis().labelpad=-4
    ax.set_zlabel(r'$|\rho|$', fontsize=5, labelpad=-51)
    
def density_matrices_plotfn(
    Density_matrices, 
    Density_matrices_ideal,
    timestamp,
    qubit,
    ax,
    title=None,
    **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    States = [s for s in Density_matrices.keys()]
    for i, state in enumerate(States):
        axs[i].axes.axes.set_position((.2+(i//2)*.225, .43-(i%2)*.1, .15*1.5, .075*1.1))
        _plot_density_matrix(Density_matrices[state], fig, axs[i], state=None, cbar=False,
                             rho_ideal=Density_matrices_ideal[state])
        axs[i].set_title(r'Input state $|{}\rangle$'.format(state), fontsize=5, pad=-5)
        axs[i].patch.set_visible(False)
    # vertical colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cbar_ax = fig.add_axes([.9, .341, .01, .15])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['C3','darkseagreen','C0','antiquewhite','C3'])
    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
    cb.set_label(r'arg($\rho$)', fontsize=5, labelpad=-8)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(.5)
    cb.ax.tick_params(labelsize=5, width=.5, length=2, pad=3)
    _title = timestamp
    if title:
        _title += '\n'+title
    fig.suptitle(_title, y=.55, x=.55, size=6.5)
    
def PTM_plotfn(
    R,
    F_PTM,
    angle,
    F_PTM_r,
    timestamp,
    qubit, 
    ax,
    title=None,
    **kw):
    im = ax.imshow(R, cmap=plt.cm.PiYG, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([ '$I$', '$X$', '$Y$', '$Z$'])
    ax.set_yticklabels([ '$I$', '$X$', '$Y$', '$Z$'])
    for i in range(4):
        for j in range(4):
                c = R[j,i]
                if abs(c) > .5:
                    ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center',
                                 color = 'white')
                else:
                    ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center')
    _title = f'{timestamp}'
    if title:
        _title += '\n'+title
    ax.set_title(_title, pad=4)
    ax.set_xlabel('input Pauli operator')
    ax.set_ylabel('Output Pauli operator')
    ax.axes.tick_params(width=.5, length=2)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['top'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    if F_PTM:
        text = '\n'.join(('Average gate fidelity:',
                          f'$F_\mathrm{"{avg}"}=${F_PTM*100:.1f}%',
                          'Average gate fidelity\nafter rotation:',
                          f'$\phi=${angle:.1f}$^\mathrm{"{o}"}$',
                          f'$F_\mathrm{"{avg}"}^\phi=${F_PTM_r*100:.1f}%'))
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(1.05, 1., text, transform=ax.transAxes,
                verticalalignment='top', bbox=props)