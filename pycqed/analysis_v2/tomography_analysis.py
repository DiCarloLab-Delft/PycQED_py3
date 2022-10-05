"""
File containing analysis for tomography-related experiments.
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
    for i, j in [[0,1], [1,2], [0,2]]:
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

class Gate_process_tomo_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for LRU process tomography experiment.
    """
    def __init__(self,
                 qubit: str,
                 f_state: bool = True,
                 post_select_2state: bool = False,
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
        self.f_state = f_state
        self.post_select_2state = post_select_2state
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
        _cycle = 2*18+2
        if self.f_state:
            _cycle += 1
        ################################
        # Rotate shots in IQ plane
        ################################
        # Sort shots
        _raw_shots = self.raw_data_dict['data'][:,1:]
        _shots_0 = _raw_shots[36::_cycle]
        _shots_1 = _raw_shots[37::_cycle]
        if self.f_state:
	        _shots_2 = _raw_shots[38::_cycle]
        # Rotate data
        center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
        center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
        raw_shots = rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
        Shots_0 = raw_shots[36::_cycle]
        Shots_1 = raw_shots[37::_cycle]
        self.proc_data_dict['shots_0_IQ'] = Shots_0
        self.proc_data_dict['shots_1_IQ'] = Shots_1
        if self.f_state:
        	Shots_2 = raw_shots[38::_cycle]
	        self.proc_data_dict['shots_2_IQ'] = Shots_2
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
        self.proc_data_dict['classifier'] = clf
        self.proc_data_dict['dec_bounds'] = dec_bounds
        self.proc_data_dict['Fid_dict'] = Fid_dict
        self.qoi = {}
        self.qoi['Fid_dict'] = Fid_dict
        self.qoi['Assignment_matrix'] = M
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
        ################################
        # Process tomo analysis
        ################################
        _thresh = self.proc_data_dict['projection_01']['threshold']
        
        if self.post_select_2state and self.f_state:
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
        # Parse tomography shots and post-select leakage
        dig_shots_idle = {}
        dig_shots_gate = {}
        leak_frac_idle = 0
        leak_frac_gate = 0
        for i, state in enumerate(States):
            dig_shots_idle[state] = {}
            dig_shots_gate[state] = {}
            for j, op in enumerate(Operators):
                _shot_list_idle = _dig_shots[i*3+j::_cycle]
                _shot_list_gate = _dig_shots[18+i*3+j::_cycle]
                # Post-select on leakage
                _shot_list_idle = [ s for s in _shot_list_idle if s!=2 ]
                _leak_frac_idle = 1-len(_shot_list_idle)/len(_dig_shots[i*3+j::_cycle])
                leak_frac_idle += _leak_frac_idle/18
                _shot_list_gate = [ s for s in _shot_list_gate if s!=2 ]
                _leak_frac_gate = 1-len(_shot_list_gate)/len(_dig_shots[i*3+j::_cycle])
                leak_frac_gate += _leak_frac_gate/18
                # Turn in meas outcomes (+1/-1)
                dig_shots_idle[state][op] = 1 - 2*np.array(_shot_list_idle)
                dig_shots_gate[state][op] = 1 - 2*np.array(_shot_list_gate)
        # Build density matrices and pauli vector for each input state
        # Ideal states
        Density_matrices_ideal = {}
        Density_matrices_ideal['0'] = _gen_density_matrix(mx=0, my=0, mz=+1)
        Density_matrices_ideal['1'] = _gen_density_matrix(mx=0, my=0, mz=-1)
        Density_matrices_ideal['+'] = _gen_density_matrix(mx=+1, my=0, mz=0)
        Density_matrices_ideal['-'] = _gen_density_matrix(mx=-1, my=0, mz=0)
        Density_matrices_ideal['+i'] = _gen_density_matrix(mx=0, my=+1, mz=0)
        Density_matrices_ideal['-i'] = _gen_density_matrix(mx=0, my=-1, mz=0)
        Density_matrices_idle = {}
        Density_matrices_gate = {}
        Pauli_vectors_idle = {}
        Pauli_vectors_gate = {}
        for state in States:
            # Idle
            mz = np.mean(dig_shots_idle[state]['Z'])
            mx = np.mean(dig_shots_idle[state]['X'])
            my = np.mean(dig_shots_idle[state]['Y'])
            p_vector = np.array([1, mx, my, mz])
            Pauli_vectors_idle[state] = _correct_pauli_vector(Beta_matrix, p_vector)
            rho = _gen_density_matrix(*Pauli_vectors_idle[state][1:])
            Density_matrices_idle[state] = rho
            # Gate
            mz = np.mean(dig_shots_gate[state]['Z'])
            mx = np.mean(dig_shots_gate[state]['X'])
            my = np.mean(dig_shots_gate[state]['Y'])
            p_vector = np.array([1, mx, my, mz])
            Pauli_vectors_gate[state] = _correct_pauli_vector(Beta_matrix, p_vector)
            rho = _gen_density_matrix(*Pauli_vectors_gate[state][1:])
            Density_matrices_gate[state] = rho
        # Get PTM
        PTM_idle = _get_pauli_transfer_matrix(
            Pauli_0 = Pauli_vectors_idle['0'], 
            Pauli_1 = Pauli_vectors_idle['1'],
            Pauli_p = Pauli_vectors_idle['+'], 
            Pauli_m = Pauli_vectors_idle['-'],
            Pauli_ip = Pauli_vectors_idle['+i'], 
            Pauli_im = Pauli_vectors_idle['-i'])
        PTM_gate = _get_pauli_transfer_matrix(
            Pauli_0 = Pauli_vectors_gate['0'], 
            Pauli_1 = Pauli_vectors_gate['1'],
            Pauli_p = Pauli_vectors_gate['+'], 
            Pauli_m = Pauli_vectors_gate['-'],
            Pauli_ip = Pauli_vectors_gate['+i'], 
            Pauli_im = Pauli_vectors_gate['-i'])
        # Calculate angle of rotation
        def _get_PTM_angles(PTM):
            angle_xy = np.arctan2(PTM[1,2], PTM[1,1])*180/np.pi
            angle_xz = np.arctan2(PTM[1,3], PTM[1,1])*180/np.pi
            angle_yz = np.arctan2(PTM[2,3], PTM[2,2])*180/np.pi
            angle_dict = {'xy':angle_xy, 'xz':angle_xz, 'yz':angle_yz}
            return angle_dict
        Angle_dict_idle = _get_PTM_angles(PTM_idle)
        Angle_dict_gate = _get_PTM_angles(PTM_gate)
        # get ideal PTM with same angle and
        # calculate fidelity of extracted PTM.
        # PTM_id = np.eye(4)
        # F_PTM_idle = _PTM_fidelity(PTM_idle, PTM_id)
        # PTM_id = _PTM_angle(angle_p)
        # F_PTM_rotated = _PTM_fidelity(PTM, PTM_id)
        # self.proc_data_dict['PS_frac'] = leak_frac
        self.proc_data_dict['Density_matrices_ideal'] = Density_matrices_ideal
        self.proc_data_dict['Density_matrices_idle'] = Density_matrices_idle
        self.proc_data_dict['Density_matrices_gate'] = Density_matrices_gate
        self.proc_data_dict['PTM_idle'] = PTM_idle
        self.proc_data_dict['PTM_gate'] = PTM_gate
        self.proc_data_dict['PTM_angles_idle'] = Angle_dict_idle
        self.proc_data_dict['PTM_angles_gate'] = Angle_dict_gate

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
        fig = plt.figure(figsize=(3*1.5, 6*1.5), dpi=200)
        axs =[]
        for i in range(6):
            axs.append(fig.add_subplot(3, 2, i+1 , projection='3d', azim=-35, elev=35))
        # fig.patch.set_alpha(0)
        self.axs_dict['Density_matrices_idle'] = axs[0]
        self.figs['Density_matrices_idle'] = fig
        self.plot_dicts['Density_matrices_idle'] = {
            'plotfn': density_matrices_plotfn,
            'ax_id': 'Density_matrices_idle',
            'Density_matrices': self.proc_data_dict['Density_matrices_idle'],
            'Density_matrices_ideal': self.proc_data_dict['Density_matrices_ideal'],
            'title': 'Idle density matrices',
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig = plt.figure(figsize=(3*1.5, 6*1.5), dpi=200)
        axs =[]
        for i in range(6):
            axs.append(fig.add_subplot(3, 2, i+1 , projection='3d', azim=-35, elev=35))
        # fig.patch.set_alpha(0)
        self.axs_dict['Density_matrices_gate'] = axs[0]
        self.figs['Density_matrices_gate'] = fig
        self.plot_dicts['Density_matrices_gate'] = {
            'plotfn': density_matrices_plotfn,
            'ax_id': 'Density_matrices_gate',
            'Density_matrices': self.proc_data_dict['Density_matrices_gate'],
            'Density_matrices_ideal': self.proc_data_dict['Density_matrices_idle'],
            'title': 'Gate density matrices',
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, ax = plt.subplots(figsize=(3.5,3.5), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Pauli_transfer_matrix_Idle'] = ax
        self.figs['Pauli_transfer_matrix_Idle'] = fig
        self.plot_dicts['Pauli_transfer_matrix_Idle'] = {
            'plotfn': PTM_plotfn,
            'ax_id': 'Pauli_transfer_matrix_Idle',
            'R': self.proc_data_dict['PTM_idle'],
            'R_angles': self.proc_data_dict['PTM_angles_idle'],
            'title': f'PTM of qubit {self.qubit} after idle',
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }
        fig, ax = plt.subplots(figsize=(3.5,3.5), dpi=100)
        # fig.patch.set_alpha(0)
        self.axs_dict['Pauli_transfer_matrix_Gate'] = ax
        self.figs['Pauli_transfer_matrix_Gate'] = fig
        self.plot_dicts['Pauli_transfer_matrix_Gate'] = {
            'plotfn': PTM_plotfn,
            'ax_id': 'Pauli_transfer_matrix_Gate',
            'R': self.proc_data_dict['PTM_gate'],
            'R_angles': self.proc_data_dict['PTM_angles_gate'],
            'title': f'PTM of qubit {self.qubit} after gate',
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

def ssro_IQ_projection_plotfn(
    shots_0, 
    shots_1,
    shots_2,
    projection_01,
    projection_12,
    projection_02,
    classifier,
    dec_bounds,
    Fid_dict,
    timestamp,
    qubit, 
    ax, **kw):
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
        p0 = [counts.max(), np.mean(X), np.mean(Y), .2, .2, 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.05)
    axs[0].plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.05)
    axs[0].plot(shots_2[:,0], shots_2[:,1], '.', color='C2', alpha=0.05)
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
    # Plot classifier zones
    from matplotlib import colors
    _all_shots = np.concatenate((shots_0, shots_1))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    X, Y = np.meshgrid(np.linspace(-_lim, _lim, 1001), np.linspace(-_lim, _lim, 1001))
    pred_labels = classifier.predict(np.c_[X.ravel(), Y.ravel()])
    pred_labels = pred_labels.reshape(X.shape)
    cmap = colors.LinearSegmentedColormap.from_list("", ["C0","C3","C2"])
    cs = axs[0].contourf(X, Y, pred_labels, cmap=cmap, alpha=0.2)
    # Plot decision boundary
    for bound in ['01', '12', '02']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        axs[0].plot([_x0, _xlim], [_y0, _ylim], 'k--', lw=1)
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
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
    # Write fidelity textbox
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                      f'$F_f$ : {Fid_dict["2"]*100:.1f}%',
                      f'$F_\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'))
    props = dict(boxstyle='round', facecolor='gray', alpha=.2)
    axs[1].text(1.05, 1, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props)

def assignment_matrix_plotfn(
    M,
    qubit,
    timestamp,
    ax, **kw):
    fig = ax.get_figure()
    im = ax.imshow(M, cmap=plt.cm.Reds, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            c = M[j,i]
            if abs(c) > .5:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center',
                             color = 'white')
            else:
                ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels([r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$'])
    ax.set_xlabel('Assigned state')
    ax.set_yticks([0,1,2])
    ax.set_yticklabels([r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$'])
    ax.set_ylabel('Prepared state')
    ax.set_title(f'{timestamp}\nQutrit assignment matrix qubit {qubit}')
    cbar_ax = fig.add_axes([.95, .15, .03, .7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('assignment probability')

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
    R_angles,
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
    if R_angles:
        text = '\n'.join(('PTM rotation angles',
                          '$\phi_{xy}=$'+f'{R_angles["xy"]:.1f} deg.',
                          '$\phi_{xz}=$'+f'{R_angles["xz"]:.1f} deg.',
                          '$\phi_{yz}=$'+f'{R_angles["yz"]:.1f} deg.'))
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(1.05, 1., text, transform=ax.transAxes,
                verticalalignment='top', bbox=props)