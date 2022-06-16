import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import cvxpy
import copy
import pycqed.analysis_v2.disturbancecalc as pb
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.measurement.hdf5_data as h5d
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import matplotlib
from matplotlib.colors import to_rgba
import itertools
plt.rcdefaults()

####################################
# HELPER FUNCTIONS
####################################
def estimate_threshold(P0, P1):
    bounds = np.min(list(P0)+list(P1)), np.max(list(P0)+list(P1))
    y0, x0 = np.histogram(P0, range=bounds, bins=200)
    x0 = (x0[1:]+x0[:-1])/2
    y1, x1 = np.histogram(P1, range=bounds, bins=200)
    x1 = (x1[1:]+x1[:-1])/2
    bounds = np.argmax(y0), np.argmax(y1)
    intersect0 = y0[bounds[0]:bounds[1]]
    intersect1 = y1[bounds[0]:bounds[1]]
    th_idx = np.argmin(np.abs(intersect0-intersect1))
    th = x0[bounds[0]:bounds[1]][th_idx]
    return th

def compute_thresholds(qubits, raw_shots, exception_qubits=[]):
    '''
    Computes thresholds based on P0000 and P1111 experiments.
    Exception qubits are used for data qubits belonging to UHFs
    that are only triggered once in the double parity experiment.
    '''
    P0 = { q : raw_shots[q][0::4] if q in exception_qubits else raw_shots[q][0::5] for q in qubits }
    P1 = { q : raw_shots[q][1::4] if q in exception_qubits else raw_shots[q][1::5] for q in qubits }
    Thresholds = { q : estimate_threshold(P0[q], P1[q]) for q in qubits }
    return Thresholds, P0, P1

def digitize_and_sort(qubits, raw_shots, thresholds, ancilla_qubit, exception_qubits=[]):
    Dig_shots = { q : (raw_shots[q]>thresholds[q])*1 for q in qubits }
    p0 = { q : Dig_shots[q][0::4] if q in exception_qubits else Dig_shots[q][0::5] for q in qubits }
    p1 = { q : Dig_shots[q][1::4] if q in exception_qubits else Dig_shots[q][1::5] for q in qubits }
    exp_1 = { q : Dig_shots[q][2::4] if q in exception_qubits else Dig_shots[q][2::5] for q in qubits }
    exp_2 = { q : (Dig_shots[q][3::5], Dig_shots[q][4::5]) if q==ancilla_qubit \
              else (Dig_shots[q][3::4] if q in exception_qubits else Dig_shots[q][4::5]) for q in qubits }
    return p0, p1, exp_1, exp_2

def get_distribution(p1, p2, p3, p4):
    outcomes = ([ str(i)+str(j)+str(k)+str(l) for i, j, k, l in zip(p1, p2, p3, p4)])
    outcome, freq = np.unique(outcomes, return_counts=True)
    distribution = { f'{i:04b}' : 0 for i in range(16) }
    for key in distribution.keys():
        if key in outcome:
            distribution[key] += freq[np.where(outcome==key)][0]/np.sum(freq)
    return distribution

class ProbabilityDistribution(np.ndarray):
    def __new__(cls, n_bits):
        self = super().__new__(cls, (2**n_bits,), float, np.zeros(2**n_bits, 'd'))
        self.n_bits = n_bits
        self.bs = ["".join(x) for x in itertools.product(*([('0', '1')] * n_bits))]
        self.bi = {s: i for i,s in enumerate(self.bs)}
        return self
  
    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(self.bi[key])
        else:
            return super().__getitem__(key)
    
    def __setitem__(self, key, val):
        if isinstance(key, str):
            return super().__setitem__(self.bi[key], val)
        else:
            return super().__setitem__(key, val)
    
    def __str__(self):
        return '\n'.join(["%s: %g" % (s, v) for s,v in zip(self.bs, self)]) + '\n'

def calculate_TVD(p, q):
    D_tvd = 0
    for s in p.keys():
        D_tvd += np.abs(p[s]-q[s])/2
    return D_tvd
    
def calculate_OVD(p, q, p_ideal):
    D_ovd = 0
    for s in p.keys():
        D_ovd += p_ideal[s]/p[s]*max( p[s]-q[s], 0 )
    return D_ovd

def compute_metrics(p, e1, e2, n_data_points):
    n_bits = 4
    SOLVER = 'SCS'
    p_ref = ProbabilityDistribution(n_bits)
    p_single = ProbabilityDistribution(n_bits)
    p_double = ProbabilityDistribution(n_bits)
    for state in p.keys():
        p_ref[state] = p[state]
        p_single[state] = e1[state]
        p_double[state] = e2[state]
    data_ref = np.array( p_ref * n_data_points, dtype='int')  # no finite sample error
    data_single = np.array(p_single * n_data_points, dtype='int')  # no finite sample error
    disturbances_single = pb.compute_disturbances(n_bits, data_ref, data_single, solver=SOLVER)
    data_double = np.array(p_double * n_data_points, dtype='int')  # no finite sample error
    disturbances_double = pb.compute_disturbances(n_bits, data_ref, data_double, solver=SOLVER)
    
    p_ideal = { s : 0.5 if s in ['0000', '1111'] else 0 for s in p.keys() }
    
    D_tvd_single = calculate_TVD(p, e1)
    D_ovd_single = calculate_OVD(p, e1, p_ideal)
    r_single = D_ovd_single/D_tvd_single
    Disturbances_ovd_single = [ (r_single*disturbances_single[i][0], r_single*disturbances_single[i][1]) for i in range(4) ]

    D_tvd_double = calculate_TVD(p, e2)
    D_ovd_double = calculate_OVD(p, e2, p_ideal)
    r_double = D_ovd_double/D_tvd_double
    Disturbances_ovd_double = [ (r_double*disturbances_double[i][0], r_double*disturbances_double[i][1]) for i in range(4) ]
    return (D_tvd_single, D_ovd_single, r_single, Disturbances_ovd_single,
            D_tvd_double, D_ovd_double, r_double, Disturbances_ovd_double)

class Sandia_parity_benchmark(ba.BaseDataAnalysis):

    def __init__(self,
                 ancilla_qubit:str,
                 data_qubits:list,
                 exception_qubits:list=[],
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

        self.ancilla_qubit = ancilla_qubit
        self.data_qubits = data_qubits
        self.exception_qubits = exception_qubits

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

        Qubits = [ self.raw_data_dict['value_names'][i].decode()[-2:] for i in range(5) ]
        Raw_shots = { self.raw_data_dict['value_names'][i].decode()[-2:] : self.raw_data_dict['data'][:,i+1] for i in range(5) }
        Thresholds, p0, p1 = compute_thresholds(Qubits, Raw_shots, exception_qubits=self.exception_qubits)

        self.proc_data_dict['Qubits'] = Qubits
        self.proc_data_dict['Thresholds'] = Thresholds
        self.proc_data_dict['p0'] = p0
        self.proc_data_dict['p1'] = p1

        Init_0, Init_1, Exp_1, Exp_2 = digitize_and_sort(Qubits, Raw_shots, Thresholds,
                                                         ancilla_qubit=self.ancilla_qubit,
                                                         exception_qubits=self.exception_qubits)
        P0 = get_distribution(Init_0[self.data_qubits[0]], Init_0[self.data_qubits[1]], Init_0[self.data_qubits[2]], Init_0[self.data_qubits[3]])
        P1 = get_distribution(Init_1[self.data_qubits[0]], Init_1[self.data_qubits[1]], Init_1[self.data_qubits[2]], Init_1[self.data_qubits[3]])
        P = { key : (P0[key]+P1[key])/2 for key in P0.keys() }
        E1 = get_distribution(Exp_1[self.data_qubits[0]] , Exp_1[self.data_qubits[1]] , Exp_1[self.data_qubits[2]] , Exp_1[self.data_qubits[3]] )
        E2 = get_distribution(Exp_2[self.data_qubits[0]] , Exp_2[self.data_qubits[1]] , Exp_2[self.data_qubits[2]] , Exp_2[self.data_qubits[3]] )
        M1 = np.mean(Exp_2[self.ancilla_qubit][0])
        M2 = np.mean(Exp_2[self.ancilla_qubit][1])

        D_tvd_single, D_ovd_single, r_single, Disturbances_ovd_single,\
        D_tvd_double, D_ovd_double, r_double, Disturbances_ovd_double = compute_metrics(P, E1, E2, len(Init_0[self.ancilla_qubit]))

        self.proc_data_dict['P0'] = P0
        self.proc_data_dict['P1'] = P1
        self.proc_data_dict['P']  = P
        self.proc_data_dict['E1'] = E1
        self.proc_data_dict['E2'] = E2
        self.proc_data_dict['M1'] = M1
        self.proc_data_dict['M2'] = M2

        self.quantities_of_interest = {}
        self.quantities_of_interest['D_tvd_single']            = D_tvd_single
        self.quantities_of_interest['D_ovd_single']            = D_ovd_single
        self.quantities_of_interest['r_single']                = r_single
        self.quantities_of_interest['Disturbances_ovd_single'] = Disturbances_ovd_single

        self.quantities_of_interest['D_tvd_double']            = D_tvd_double
        self.quantities_of_interest['D_ovd_double']            = D_ovd_double
        self.quantities_of_interest['r_double']                = r_double
        self.quantities_of_interest['Disturbances_ovd_double'] = Disturbances_ovd_double

    def prepare_plots(self):

        self.axs_dict = {}
        fig = plt.figure(figsize=(6.5, 9), dpi=200)
        axs = [fig.add_subplot(521),
               fig.add_subplot(522),
               fig.add_subplot(512),
               fig.add_subplot(513),
               fig.add_subplot(527)]
        # fig.patch.set_alpha(0)
        self.axs_dict['main'] = axs
        self.figs['main'] = fig
        self.plot_dicts['main'] = {
            'plotfn': plot_function,
            'P0': self.proc_data_dict['P0'],
            'P1': self.proc_data_dict['P1'],
            'P': self.proc_data_dict['P'],
            'E1': self.proc_data_dict['E1'],
            'E2': self.proc_data_dict['E2'],
            'M1': self.proc_data_dict['M1'],
            'M2': self.proc_data_dict['M2'],
            'r_single': self.quantities_of_interest['r_single'],
            'Disturbances_ovd_single': self.quantities_of_interest['Disturbances_ovd_single'],
            'r_double': self.quantities_of_interest['r_double'],
            'Disturbances_ovd_double': self.quantities_of_interest['Disturbances_ovd_double'],
            'timestamp': self.timestamp}

        fig, axs = plt.subplots(figsize=(10,2), ncols=5, dpi=200)
        fig.patch.set_alpha(0)
        self.axs_dict['calibration'] = axs
        self.figs['calibration'] = fig
        self.plot_dicts['calibration'] = {
            'plotfn': plot_calibration,
            'qubits': self.proc_data_dict['Qubits'],
            'p0': self.proc_data_dict['p0'],
            'p1': self.proc_data_dict['p1'],
            'thresholds': self.proc_data_dict['Thresholds']}

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))


def get_expected_value(operator, state, n):
    m = 1
    for i in range(n):
        if operator[i] == 'Z' and state[i] == '1':
            m *= -1
    return m
    
def gen_M_matrix(n):
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
            Betas[i] = get_expected_value(Operators[i], state, n)
        M[j] = Betas
    M = np.linalg.pinv(M) # invert matrix
    return M

def get_Beta_matrix(Cal_shots_dig, n):
    # List of different Operators
    ops = ['I','Z']
    Operators = [''.join(op) for op in itertools.product(ops, repeat=n)]
    # List of qubits
    Qubits = list(Cal_shots_dig.keys())
    # Calculate Beta matrix
    H = {}
    B = {}
    M = gen_M_matrix(n)
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

def gen_gate_order(n):
    # Gate order in experiment
    tomo_gates = ['Z', 'X', 'Y']
    Gate_order = [''.join(op)[::-1] for op in itertools.product(tomo_gates, repeat=n)]
    return np.array(Gate_order)

def gen_n_Q_pauli(n):
    # Single qubit pauli terms
    Pauli_operators = {}
    Pauli_operators['I'] = np.array([[  1,  0],
                                     [  0,  1]])
    Pauli_operators['Z'] = np.array([[  1,  0],
                                     [  0, -1]])
    Pauli_operators['X'] = np.array([[  0,  1],
                                     [  1,  0]])
    Pauli_operators['Y'] = np.array([[  0,-1j],
                                     [ 1j,  0]])
    # Four qubit pauli terms
    pauli_ops = ['I', 'X', 'Y', 'Z']
    Pauli_terms = {}
    Operators = [''.join(op) for op in itertools.product(pauli_ops, repeat=n)]
    for Op in Operators:
        Pauli_terms[Op] = Pauli_operators[Op[0]]
        for op in Op[1:]:
            Pauli_terms[Op]=np.kron(Pauli_terms[Op], Pauli_operators[op])
    return Pauli_terms

def get_Pauli_expectation_values(Beta_matrix, Gate_order, Mask, Tomo_shots_dig):
    '''
    Calculates Pauli expectation values (PEVs) in three steps:
        1. Calculate raw PEVs.
        2. Condition (post-select) data on no errors in stabilizers.
        3. Apply readout corrections to PEVs based on Beta matarix.
    '''
    Qubits = list(Tomo_shots_dig.keys())[1:]
    n = len(Qubits)
    
    B_matrix = np.array([Beta_matrix[key][1:] for key in Beta_matrix.keys()])
    B_0 = np.array([Beta_matrix[key][0] for key in Beta_matrix.keys()])
    iB_matrix = np.linalg.inv(B_matrix)
    pauli_ops = ['I', 'X', 'Y', 'Z']
    P_values = {''.join(op):[] for op in itertools.product(pauli_ops, repeat=n)}
    P_frac = copy.deepcopy(P_values)
    for i, pre_rotation in enumerate(Gate_order[:]):
        combs = [('I', op) for op in pre_rotation ]
        P_vector = {''.join(o):1 for o in itertools.product(*combs)}
        for correlator in P_vector.keys():
            # Calculate raw PEVs
            C = 1
            for j, qubit in enumerate(Qubits):
                if correlator[n-j-1] != 'I':
                    C *= np.array(Tomo_shots_dig[qubit][i], dtype=float)
            # Post-select data on stabilizer measurements
            C = C*Mask[i]
            n_total = len(C)
            C = C[~np.isnan(C)]
            n_selec = len(C)
            P_vector[correlator] = np.mean(C)
            P_frac[correlator] = n_selec/n_total
        # Aplly readout corrections
        P = np.array([P_vector[key] for key in list(P_vector.keys())[1:]])
        P_corrected = np.dot(P-B_0, iB_matrix)
        P_vec_corr = { key: P_corrected[i-1] if i!=0 else 1 for i, key in enumerate(list(P_vector.keys()))}
        # Fill main pauli vector with corresponding expectation values
        for key in P_vec_corr.keys():
            P_values[key].append(P_vec_corr[key])
    # Average all repeated pauli terms
    for key in P_values:
        P_values[key] = np.mean(P_values[key])
    # Calculate density matrix
    Pauli_terms_n = gen_n_Q_pauli(n)
    rho = np.zeros((2**n,2**n))*(1+0*1j)
    for op in Pauli_terms_n.keys():
        rho += P_values[op]*Pauli_terms_n[op]/2**n
    return P_values, rho, P_frac

def fidelity(rho_1, rho_2, trace_conserved = False):
    if trace_conserved:
        if np.round(np.trace(rho_1), 3) !=1:
            raise ValueError('rho_1 unphysical, trace =/= 1, but ', np.trace(rho_1))
        if np.round(np.trace(rho_2), 3) !=1:
            raise ValueError('rho_2 unphysical, trace =/= 1, but ', np.trace(rho_2))
    sqrt_rho_1 = linalg.sqrtm(rho_1)
    eig_vals = linalg.eig(np.dot(np.dot(sqrt_rho_1,rho_2),sqrt_rho_1))[0]
    pos_eig = [vals for vals in eig_vals if vals > 0]
    return float(np.sum(np.real(np.sqrt(pos_eig))))**2

class Weight_n_parity_tomography(ba.BaseDataAnalysis):
    def __init__(self,
                 sim_measurement: bool,
                 exception_qubits: list = [],
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

        self.sim_measurement = sim_measurement
        self.exception_qubits = exception_qubits

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

        self.proc_data_dict = {}
        Qubits = [ name.decode().split(' ')[-1] for name in self.raw_data_dict['value_names'] ]
        Data_qubits = [ q for q in Qubits if 'D' in q ]
        Anc_qubit = [ q for q in Qubits if ('X' in q) or ('Z' in q) ][0]
        n = len(Data_qubits)
        self.Qubits = Qubits
        ############################
        # Sort calibration Shots
        ############################
        Cal_shots = {q : {} for q in Qubits}
        Cal_shots_dig = {q : {} for q in Data_qubits}
        states = ['0','1']
        combinations = [''.join(s) for s in itertools.product(states, repeat=n+1)]

        if self.sim_measurement:
            cycle = 3**n
        else:
            cycle = 3**n*2
        Thresholds = {}
        self.proc_data_dict['Shots_0'] = {}
        self.proc_data_dict['Shots_1'] = {}
        for i, qubit in enumerate(Qubits):
            Shots_0 = []
            Shots_1 = []
            for j, comb in enumerate(combinations):
                Cal_shots[qubit][comb] = self.raw_data_dict['data'][:,i+1][cycle+j::cycle+2**(n+1)]
                if comb[i] == '0':
                    Shots_0+=list(Cal_shots[qubit][comb])
                else:
                    Shots_1+=list(Cal_shots[qubit][comb])
            Thresholds[qubit] = estimate_threshold(Shots_0, Shots_1)
            self.proc_data_dict['Shots_0'][qubit] = Shots_0
            self.proc_data_dict['Shots_1'][qubit] = Shots_1
            # Digitize data qubit shots
            def digitize(shots, threshold):
                dig_shots = [ +1 if s<threshold else -1 for s in shots ]
                return np.array(dig_shots)
            if qubit in Data_qubits:
                states = ['0','1']
                combs = [''.join(s) for s in itertools.product(states, repeat=n)]
                
                for comb in combs:
                    tot_shots = np.concatenate((Cal_shots[qubit]['0'+comb], Cal_shots[qubit]['1'+comb]))
                    Cal_shots_dig[qubit][comb] = digitize(tot_shots, Thresholds[qubit])
        self.proc_data_dict['Thresholds'] = Thresholds
        # Get RO Beta matrix
        B = get_Beta_matrix(Cal_shots_dig, n)
        self.proc_data_dict['Beta_matrix'] = B
        ############################
        # Sort tomography Shots
        ############################
        Tomo_shots = {q: [None for i in range(3**n)] for q in Qubits}
        Tomo_shots_dig = {q: [None for i in range(3**n)] for q in Qubits}
        if self.sim_measurement:
            for i, qubit in enumerate(Qubits):
                for j in range(3**n):
                    Tomo_shots[qubit][j] = list(self.raw_data_dict['data'][:,i+1][j::(3**n)+2**(n+1)])
                    Tomo_shots_dig[qubit][j] = list(digitize(self.raw_data_dict['data'][:,i+1][j::(3**n)+2**(n+1)], Thresholds[qubit]))
        else:
            for i, qubit in enumerate(Data_qubits):
                for j in range(3**n):
                    if qubit in self.exception_qubits:
                        Tomo_shots[qubit][j] = list(self.raw_data_dict['data'][:,i+2][2*j::(3**n)*2+2**(n+1)])
                        Tomo_shots_dig[qubit][j] = list(digitize(self.raw_data_dict['data'][:,i+2][2*j::(3**n)*2+2**(n+1)], Thresholds[qubit]))
                    else:
                        Tomo_shots[qubit][j] = list(self.raw_data_dict['data'][:,i+2][2*j+1::(3**n)*2+2**(n+1)])
                        Tomo_shots_dig[qubit][j] = list(digitize(self.raw_data_dict['data'][:,i+2][2*j+1::(3**n)*2+2**(n+1)], Thresholds[qubit]))
            for j in range(3**n):
                Tomo_shots[Anc_qubit][j] = list(self.raw_data_dict['data'][:,1][2*j::(3**n)*2+32])
                Tomo_shots_dig[Anc_qubit][j] = list(digitize(self.raw_data_dict['data'][:,1][2*j::(3**n)*2+2**(n+1)], Thresholds[Anc_qubit]))
        ###########################
        # Get post-selection masks
        ###########################
        def get_mask(Shots, result):
            Mask = [None for i in range(3**n)]
            for i in range(3**n):
                Mask[i] = np.array([ 1 if s == result else np.nan for s in Shots[i] ])
            return Mask
        ps_mask_0 = get_mask(Tomo_shots_dig[Anc_qubit], result=+1)
        ps_mask_1 = get_mask(Tomo_shots_dig[Anc_qubit], result=-1)
        #####################################
        # Calculate Pauli expectation values
        #####################################
        Pauli_terms_0, rho_0, P_frac_0 = get_Pauli_expectation_values(B, gen_gate_order(n), ps_mask_0, 
                                                                      Tomo_shots_dig=Tomo_shots_dig)
        Pauli_terms_1, rho_1, P_frac_1 = get_Pauli_expectation_values(B, gen_gate_order(n), ps_mask_1,
                                                                      Tomo_shots_dig=Tomo_shots_dig)
        R_0 = np.zeros(rho_0.shape)
        R_0[ 0, 0] = .5
        R_0[ 0,-1] = .5
        R_0[-1, 0] = .5
        R_0[-1,-1] = .5
        R_1 = np.zeros(rho_1.shape)
        R_1[ 0, 0] = .5
        R_1[ 0,-1] = -.5
        R_1[-1, 0] = -.5
        R_1[-1,-1] = .5
        self.proc_data_dict['Pauli_terms_0'] = Pauli_terms_0
        self.proc_data_dict['Pauli_terms_1'] = Pauli_terms_1
        self.proc_data_dict['rho_0'] = rho_0
        self.proc_data_dict['rho_1'] = rho_1
        self.proc_data_dict['ps_frac_0'] = np.mean(list(P_frac_0.values()))
        self.proc_data_dict['ps_frac_1'] = np.mean(list(P_frac_1.values()))
        self.proc_data_dict['Fid_0'] = fidelity(rho_0, R_0)
        self.proc_data_dict['Fid_1'] = fidelity(rho_1, R_1)
        self.proc_data_dict['angle_0'] = np.angle(rho_0[0,-1])*180/np.pi
        self.proc_data_dict['angle_1'] = np.angle(rho_1[0,-1])*180/np.pi
        self.proc_data_dict['nr_shots'] = len(Tomo_shots_dig[Anc_qubit][0])

    def prepare_plots(self):
        n = len(self.Qubits)
        ancilla = self.Qubits[0]
        data_qubits = self.Qubits[1:]
        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(4*n, 3), ncols=n)
        self.axs_dict['calibration_histograms'] = axs
        self.figs['calibration_histograms'] = fig
        self.plot_dicts['calibration_histograms'] = {
            'plotfn': plot_shots_histogram,
            'Qubits': self.Qubits,
            'Shots_0': self.proc_data_dict['Shots_0'],
            'Shots_1': self.proc_data_dict['Shots_1'],
            'Thresholds': self.proc_data_dict['Thresholds'],
            'timestamp': self.timestamp
        }

        R_0 = np.zeros(self.proc_data_dict['rho_0'].shape)
        R_0[ 0, 0] = .5
        R_0[ 0,-1] = .5
        R_0[-1, 0] = .5
        R_0[-1,-1] = .5
        fig = plt.figure(figsize=(6, 5.7), dpi = 200)
        ax = fig.add_subplot(221, projection='3d', azim=-35, elev=30)
        self.axs_dict['Tomography_condition_0'] = ax
        self.figs['Tomography_condition_0'] = fig
        self.plot_dicts['Tomography_condition_0'] = {
            'plotfn': plot_density_matrix,
            'rho': self.proc_data_dict['rho_0'],
            'rho_id': R_0,
            'title': rf'{self.timestamp}'+'\n'+\
                rf'tomography of qubits {" ".join(data_qubits)}'+'\n'+\
                rf'condition $|0\rangle_{"{"+ancilla+"}"}$',
            'Fid': self.proc_data_dict['Fid_0'],
            'Ps_frac': self.proc_data_dict['ps_frac_0'],
            'angle': self.proc_data_dict['angle_0'],
            'nr_shots': self.proc_data_dict['nr_shots']
        }

        R_1 = np.zeros(self.proc_data_dict['rho_1'].shape)
        R_1[ 0, 0] = .5
        R_1[ 0,-1] = -.5
        R_1[-1, 0] = -.5
        R_1[-1,-1] = .5
        fig = plt.figure(figsize=(6, 5.7), dpi = 200)
        ax = fig.add_subplot(221, projection='3d', azim=-35, elev=30)
        self.axs_dict['Tomography_condition_1'] = ax
        self.figs['Tomography_condition_1'] = fig
        self.plot_dicts['Tomography_condition_1'] = {
            'plotfn': plot_density_matrix,
            'rho': self.proc_data_dict['rho_1'],
            'rho_id': R_1,
            'title': rf'{self.timestamp}'+'\n'+\
                rf'tomography of qubits {" ".join(data_qubits)}'+'\n'+\
                rf'condition $|1\rangle_{"{"+ancilla+"}"}$',
            'Fid': self.proc_data_dict['Fid_1'],
            'Ps_frac': self.proc_data_dict['ps_frac_1'],
            'angle': self.proc_data_dict['angle_1'],
            'nr_shots': self.proc_data_dict['nr_shots']
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def plot_shots_histogram(Qubits, Shots_0, Shots_1, Thresholds, timestamp,
                         ax, **kw):
    fig = ax[0].get_figure()
    for i, qubit in enumerate(Qubits):
        ax[i].hist(Shots_0[qubit], bins=100, color='C0', alpha=0.5)
        ax[i].hist(Shots_1[qubit], bins=100, color='C3', alpha=0.5)
        ax[i].axvline(Thresholds[qubit], color='k', ls='--', lw=1)
        ax[i].set_yticks([])
        ax[i].set_title(qubit)
    fig.suptitle(f'{timestamp} Calibration points shots', y=1.1)

def plot_density_matrix(rho, rho_id, title,
                        Fid, Ps_frac, angle,
                        nr_shots,
                        ax, **kw):
    fig = ax.get_figure()
    n = len(rho)
    # xedges = np.arange(-.75, n, 1)
    # yedges = np.arange(-.75, n, 1)
    xedges = np.linspace(0, 1, n+1)
    yedges = np.linspace(0, 1, n+1)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 1/n*0.8
    dz = np.abs(rho).ravel()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C3",'darkseagreen',"C0",'antiquewhite',"C3"])
    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    color=cmap(norm([np.angle(e) for e in rho.ravel()]))
    color_id=cmap(norm([np.angle(e) for e in rho_id.ravel()]))
    dz1 = np.abs(rho_id).ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='max',
             color=color, alpha=1 , edgecolor='black', linewidth=.1)
    # selector
    s = [k for k in range(len(dz1)) if dz1[k] > .15]
    colors = [ to_rgba(color_id[k], .25) if dz1[k] > dz[k] else to_rgba(color_id[k], 1) for k in s ]
    Z = [ dz[k] if dz1[k] > dz[k] else dz1[k] for k in s ]
    DZ= [ dz1[k]-dz[k] if dz1[k] > dz[k] else dz[k]-dz1[k] for k in s ]
    ax.bar3d(xpos[s], ypos[s], Z, dx, dy, dz=DZ, zsort='min',
             color=colors, edgecolor=to_rgba('black', .25), linewidth=.4)
    N = int(np.log2(n))
    states = ['0', '1']
    combs = [''.join(s) for s in itertools.product(states, repeat=N)]
    tick_period = n//(3-2*(N%2)) - N%2
    ax.set_xticks(xpos[::n][::tick_period]+1/n/2)
    ax.set_yticks(ypos[:n:tick_period]+1/n/2)
    ax.set_xticklabels(combs[::tick_period], rotation=20, fontsize=6, ha='right')
    ax.set_yticklabels(combs[::tick_period], rotation=-40, fontsize=6)
    ax.tick_params(axis='x', which='major', pad=-6, labelsize=6)
    ax.tick_params(axis='y', which='major', pad=-6, labelsize=6)
    ax.tick_params(axis='z', which='major', pad=-2, labelsize=6)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")
    ax.set_zticks(np.linspace(0, .5, 5))
    ax.set_zticklabels(['0', '', '0.25', '', '0.5'])
    ax.set_zlim(0, .5)
    ax.set_zlabel(r'$|\rho|$', labelpad=-8, size=7)
    ax.set_title(title, size=7)
    # Text box
    s = ''.join((r'$F_{|\psi\rangle}='+fr'{Fid*100:.1f}\%$', '\n',
                 r'$\mathrm{arg}(\rho_{0,15})='+fr'{angle:.1f}^\circ$', '\n',
                 r'$P_\mathrm{ps}='+fr'{Ps_frac*100:.1f}\%$', '\n',
                 f'# shots per Pauli {nr_shots}'))
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=1)
    ax.text(.5, 1, .5, s, size=5, bbox=props, va='bottom')
    # colorbar
    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.55, 0.56, 0.01, 0.275])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C3",'darkseagreen',"C0",'antiquewhite',"C3"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                          orientation='vertical')
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
    cb.set_label(r'arg$(\rho)$', fontsize=7, labelpad=-10)
    cb.ax.tick_params(labelsize=7)

def plot_calibration(qubits, p0, p1, thresholds,
                     ax, **kw):
    fig = ax[0].get_figure()
    for i, qubit in enumerate(qubits):
        ax[i].hist(p0[qubit], bins=100, color='C0', alpha=.5)
        ax[i].hist(p1[qubit], bins=100, color='C3', alpha=.5)
        ax[i].axvline(thresholds[qubit], c='k', ls='--', lw=.75)
        ax[i].set_title(qubit)
        ax[i].set_yticks([])
    fig.tight_layout()

def plot_function(P0, P1, P, E1, E2, M1, M2,
                  Disturbances_ovd_single, r_single,
                  Disturbances_ovd_double, r_double,
                  timestamp,
                  ax, **kw):
    fig = ax[0].get_figure()
    # Calibration 0
    ax[0].set_title(r'Reference $|0000\rangle$')
    ax[0].axhline(1., color='black', alpha=.5, linestyle='--')
    ax[0].bar(np.arange(0,16), [P0[k] for k in P0.keys()], color='C0')
    ax[0].set_ylim(0, 1.05)
    ax[0].set_xticks([0,5,10,15])
    ax[0].set_yticks([0, .5, 1])
    ax[0].set_xticklabels(['{:04b}'.format(5*i) for i in range(4)], rotation=45, fontsize=8)
    ax[0].set_yticklabels([0, 0.5, 1])
    ax[0].set_ylabel('Fraction')
    # Calibration 1
    ax[1].set_title(r'Reference $|1111\rangle$')
    ax[1].axhline(1., color='black', alpha=.5, linestyle='--')
    ax[1].bar(np.arange(0,16), [P1[k] for k in P1.keys()], color='C0')
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xticks([0,5,10,15])
    ax[1].set_yticks([0, .5, 1])
    ax[1].set_xticklabels(['{:04b}'.format(5*i) for i in range(4)], rotation=45, fontsize=8)
    ax[1].set_yticklabels(['', '', ''])
    # Single parity
    ax[2].set_title('Single parity check')
    ax[2].axhline(.5, color='black', alpha=.5, linestyle='--')
    ax[2].bar(np.arange(0,16), [P[k] for k in P.keys()], color='C0', alpha=.25, label='calibration')
    ax[2].bar(np.arange(0,16), [E1[k] for k in E1.keys()], color='C0', label='parity check')
    ax[2].set_ylim(0, .525)
    ax[2].set_yticks([0, .25, .5])
    ax[2].set_xticks(np.arange(0,16))
    ax[2].set_xticklabels(['{:04b}'.format(i) for i in range(16)], rotation=45, fontsize=8)
    ax[2].set_yticklabels([0, 0.25, 0.5])
    ax[2].set_xlabel('measured state')
    ax[2].set_ylabel('Fraction')
    ax[2].legend(bbox_to_anchor=(1.025, 1), loc='upper left')
    # Repeated parity
    ax[3].set_title('Repeated parity check')
    ax[3].axhline(.5, color='black', alpha=.5, linestyle='--')
    ax[3].bar(np.arange(0,16), [P[k] for k in P.keys()], color='C0', alpha=.25, label='calibration')
    ax[3].bar(np.arange(0,16), [E1[k] for k in E1.keys()], color='C1', label='single parity check')
    ax[3].bar(np.arange(0,16), [E2[k] for k in E2.keys()], color='C0', label='double parity check')
    ax[3].set_ylim(0, .525)
    ax[3].set_yticks([0, .25, .5])
    ax[3].set_xticks(np.arange(0,16))
    ax[3].set_xticklabels(['{:04b}'.format(i) for i in range(16)], rotation=45, fontsize=8)
    ax[3].set_yticklabels([0, 0.25, 0.5])
    ax[3].set_xlabel('measured state')
    ax[3].set_ylabel('Fraction')
    ax[3].legend(bbox_to_anchor=(1.025, 1), loc='upper left', fontsize=6)
    # Parity outcome results
    ax[4].set_title('Parity results')
    ax[4].axhline(1, color='black', alpha=.5, linestyle='--')
    ax[4].axhline(-1, color='black', alpha=.5, linestyle='--')
    ax[4].bar([1, 2], [M1*2-1, M2*2-1])
    ax[4].set_ylim(-1.1, 1.1)
    ax[4].set_xticks([1,2])
    ax[4].set_xticklabels([r'$\langle m_1\rangle$', r'$\langle m_2\rangle$'])
    textstr1 = '\n'.join(('',
                          '$D_1^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_single[0][0], Disturbances_ovd_single[0][1]),
                          '$D_2^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_single[1][0], Disturbances_ovd_single[1][1]),
                          '$\Delta^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_single[2][0]+Disturbances_ovd_single[3][0], Disturbances_ovd_single[2][1]+Disturbances_ovd_single[3][1]),
                          '$r$  =  %f' % (r_single)))
    
    textstr2 = '\n'.join(('Repeatability  =  %.1f%%' % (M2*100),
                          '$D_1^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_double[0][0], Disturbances_ovd_double[0][1]),
                          '$D_2^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_double[1][0], Disturbances_ovd_double[1][1]),
                          '$\Delta^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_double[2][0]+Disturbances_ovd_double[3][0], Disturbances_ovd_double[2][1]+Disturbances_ovd_double[3][1]),
                          '$r$  =  %f' % (r_double)))

    props = dict(boxstyle='round', facecolor='gray', alpha=0.15)
    fig.tight_layout()
    ax[4].text(1.08, 1.25, 'Single parity', transform=ax[4].transAxes, fontsize=12,
            verticalalignment='top')
    ax[4].text(1.1, 0.95, textstr1, transform=ax[4].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax[4].text(2.25, 1.25, 'Repeated parity', transform=ax[4].transAxes, fontsize=12,
            verticalalignment='top')
    ax[4].text(2.27, 0.95, textstr2, transform=ax[4].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    fig.suptitle(f'Sandia parity benchmark {timestamp}', y=1.01, x=.43)



