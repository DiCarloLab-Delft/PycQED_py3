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
    ax[4].bar([1, 2], [1-M1*2, 1-M2*2])
    ax[4].set_ylim(-1.1, 1.1)
    ax[4].set_xticks([1,2])
    ax[4].set_xticklabels([r'$\langle m_1\rangle$', r'$\langle m_2\rangle$'])
    textstr1 = '\n'.join(('',
                          '$D_1^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_single[0][0], Disturbances_ovd_single[0][1]),
                          '$D_2^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_single[1][0], Disturbances_ovd_single[1][1]),
                          '$\Delta^{ovd}$  =  %f $\pm$ %f' % (Disturbances_ovd_single[2][0]+Disturbances_ovd_single[3][0], Disturbances_ovd_single[2][1]+Disturbances_ovd_single[3][1]),
                          '$r$  =  %f' % (r_single)))
    
    textstr2 = '\n'.join(('Repeatability  =  %.1f%%' % ((1-M2)*100),
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
                 n_rounds: int,
                 post_selection: bool,
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
        self.n_rounds = n_rounds
        self.post_selection = post_selection
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
            cycle = 3**n*self.n_rounds
        else:
            cycle = 3**n*(self.n_rounds+1)

        ## NB: Ps is not yet implemented 
        if self.post_selection:
            cycle*=2

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
            _s_0, _s_1 = np.mean(Shots_0), np.mean(Shots_1)
            if _s_1 < _s_0:
                Shots_0, Shots_1 = -np.array(Shots_0), -np.array(Shots_1)
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
                    if (qubit in self.exception_qubits) or (qubit==Anc_qubit):
                        _idx = self.n_rounds*j
                        Tomo_shots[qubit][j] = list(self.raw_data_dict['data'][:,i+1][_idx::cycle+2**(n+1)])
                    else:
                        _idx = self.n_rounds*j + self.n_rounds-1
                        Tomo_shots[qubit][j] = list(self.raw_data_dict['data'][:,i+1][_idx::cycle+2**(n+1)])
                    Tomo_shots_dig[qubit][j] = list(digitize(Tomo_shots[qubit][j], Thresholds[qubit]))
        else: # Sequential measurement    
            for i, qubit in enumerate(Data_qubits):
                for j in range(3**n):
                    if qubit in self.exception_qubits:
                        _idx = (self.n_rounds+1)*j
                        Tomo_shots[qubit][j] = list(self.raw_data_dict['data'][:,i+2][_idx::cycle+2**(n+1)])
                    else:
                        _idx = (self.n_rounds+1)*j+self.n_rounds
                        Tomo_shots[qubit][j] = list(self.raw_data_dict['data'][:,i+2][_idx::cycle+2**(n+1)])
                    Tomo_shots_dig[qubit][j] = list(digitize(Tomo_shots[qubit][j], Thresholds[qubit]))
            for j in range(3**n):
                _idx = (self.n_rounds+1)*j
                Tomo_shots[Anc_qubit][j] = list(self.raw_data_dict['data'][:,1][_idx::cycle+2**(n+1)])
                Tomo_shots_dig[Anc_qubit][j] = list(digitize(Tomo_shots[Anc_qubit][j], Thresholds[Anc_qubit]))
        # Calculate repeatability
        if self.n_rounds == 2:
            M1_dig = []
            M2_dig = []
            if self.sim_measurement:
                for j in range(3**n):
                    _idx = self.n_rounds*j
                    _M1 = list(self.raw_data_dict['data'][:,1][_idx::cycle+2**(n+1)])
                    _M2 = list(self.raw_data_dict['data'][:,1][_idx+1::cycle+2**(n+1)])
                    M1_dig += list(digitize(_M1, Thresholds[Anc_qubit]))
                    M2_dig += list(digitize(_M2, Thresholds[Anc_qubit]))
            else:
                for j in range(3**n):
                    _idx = (self.n_rounds+1)*j
                    _M1 = list(self.raw_data_dict['data'][:,1][_idx::cycle+2**(n+1)])
                    _M2 = list(self.raw_data_dict['data'][:,1][_idx+1::cycle+2**(n+1)])
                    M1_dig += list(digitize(_M1, Thresholds[Anc_qubit]))
                    M2_dig += list(digitize(_M2, Thresholds[Anc_qubit]))
            self.proc_data_dict['repeatability'] = (1+np.mean(M2_dig))/2
            self.proc_data_dict['M1'] = np.mean(M1_dig)
            self.proc_data_dict['M2'] = np.mean(M2_dig)

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
        if self.n_rounds==2:
            diag_0 = np.diagonal(np.real(self.proc_data_dict['rho_0']))
            diag_1 = np.diagonal(np.real(self.proc_data_dict['rho_1']))
            self.proc_data_dict['P_dist'] = self.proc_data_dict['ps_frac_0']*diag_0+\
                                            self.proc_data_dict['ps_frac_1']*diag_1

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
        title = rf'{self.timestamp}'+'\n'+\
                rf'tomography of qubits {" ".join(data_qubits)}'+'\n'+\
                rf'condition $|0\rangle_{"{"+ancilla+"}"}$'
        if self.sim_measurement:
            title += ' (sim-msmt)'
        else:
            title += ' (seq-msmt)'
        self.plot_dicts['Tomography_condition_0'] = {
            'plotfn': plot_density_matrix,
            'rho': self.proc_data_dict['rho_0'],
            'rho_id': R_0,
            'title': title,
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
        title = rf'{self.timestamp}'+'\n'+\
                rf'tomography of qubits {" ".join(data_qubits)}'+'\n'+\
                rf'condition $|1\rangle_{"{"+ancilla+"}"}$'
        if self.sim_measurement:
            title += ' (sim-msmt)'
        else:
            title += ' (seq-msmt)'
        self.plot_dicts['Tomography_condition_1'] = {
            'plotfn': plot_density_matrix,
            'rho': self.proc_data_dict['rho_1'],
            'rho_id': R_1,
            'title': title,
            'Fid': self.proc_data_dict['Fid_1'],
            'Ps_frac': self.proc_data_dict['ps_frac_1'],
            'angle': self.proc_data_dict['angle_1'],
            'nr_shots': self.proc_data_dict['nr_shots']
        }
        if self.n_rounds == 2:
            fig, axs = plt.subplots(figsize=(10,2), ncols=2,
                                    gridspec_kw={'width_ratios':[1,2]})
            self.axs_dict['Repeatability_analysis'] = axs
            self.figs['Repeatability_analysis'] = fig
            self.plot_dicts['Repeatability_analysis'] = {
                'plotfn': plot_repeatabilityfn,
                'M1': self.proc_data_dict['M1'],
                'M2': self.proc_data_dict['M2'],
                'repeatability': self.proc_data_dict['repeatability'],
                'P_dist': self.proc_data_dict['P_dist'],
                'ancilla' : ancilla,
                'data_qubits' : data_qubits,
                'timestamp': self.timestamp
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
                 r'$\mathrm{arg}(\rho_{0,'+f'{n-1}'+'})='+fr'{angle:.1f}^\circ$', '\n',
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

def plot_repeatabilityfn(M1, M2, repeatability, P_dist,
                         ancilla, data_qubits, timestamp,
                         ax, **kw):
    fig = ax[0].get_figure()
    ax[0].bar([r'$\langle M_1\rangle$', r'$\langle M_2\rangle$'], [M1, M2])
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].set_yticks([-1, -.5, 0, .5, 1])
    ax[0].set_yticklabels(['-1', '', '0', '', '1'])
    ax[0].set_title(f'{ancilla} measurement results')
    ax[0].text(-.4, -.9, f'Repeatability : {repeatability*100:.1f}%')

    states = ['0', '1']
    n = len(data_qubits)
    combs = np.array([''.join(s) for s in itertools.product(states, repeat=n)])
    idx_sort = np.argsort([ s.count('1') for s in combs ])
    ax[1].bar(combs[idx_sort], P_dist[idx_sort])
    ax[1].set_xticklabels(combs[idx_sort], rotation=90)
    ax[1].set_title(f'{" ".join(data_qubits)} measurement probability')
    fig.suptitle(f'{timestamp}\nRepeatability measurement {ancilla}', y=1.2)


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

def _get_threshold(Shots_0, Shots_1):
    # Take relavant quadrature
    shots_0 = Shots_0[:,0]
    shots_1 = Shots_1[:,0]
    # find range
    _all_shots = np.concatenate((shots_0, shots_1))
    _range = (np.min(_all_shots), np.max(_all_shots))
    # Sort shots in unique values
    x0, n0 = np.unique(shots_0, return_counts=True)
    x1, n1 = np.unique(shots_1, return_counts=True)
    Fid, threshold = _calculate_fid_and_threshold(x0, n0, x1, n1)
    return threshold

def _gauss_pdf(x, x0, sigma):
    return np.exp(-((x-x0)/sigma)**2/2)

def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
    _dist0 = A*( (1-r)*_gauss_pdf(x, x0, sigma0) + r*_gauss_pdf(x, x1, sigma1) )
    return _dist0

def _double_gauss_joint(x, x0, x1, sigma0, sigma1, A0, A1, r0, r1):
    _dist0 = double_gauss(x, x0, x1, sigma0, sigma1, A0, r0)
    _dist1 = double_gauss(x, x1, x0, sigma1, sigma0, A1, r1)
    return np.concatenate((_dist0, _dist1))

def _fit_double_gauss(x_vals, hist_0, hist_1):
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
    _x0_guess = np.sum(x_vals*pdf_0) # calculate mean
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
    bounds = np.array([_x0_bound, _x1_bound, _sigma0_bound, _sigma1_bound,
                       _A0_bound, _A1_bound, _r0_bound, _r1_bound])
    # Fit parameters within bounds
    popt, pcov = curve_fit(
        _double_gauss_joint, x_vals,
        np.concatenate((hist_0, hist_1)),
        p0=p0, bounds=bounds.transpose())
    popt0 = popt[[0,1,2,3,4,6]]
    popt1 = popt[[1,0,3,2,5,7]]
    # Calculate quantities of interest
    SNR = abs(popt0[0] - popt1[0])/((abs(popt0[2])+abs(popt1[2]))/2)
    P_e0 = popt0[5]
    P_g1 = popt1[5]
    # Fidelity from fit
    _range = (np.min(x_vals), np.max(x_vals))
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
    n = len(intercepts)
    if n == 3:
        _bounds = [[0,1], [1,2], [0,2]]
    if n == 4:
        _bounds = [[0,1], [1,2], [2,3], [0,3]]
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

def _rotate_and_center_data(I, Q, vec0, vec1, phi=0):
    vector = vec1-vec0
    angle = np.arctan(vector[1]/vector[0])
    rot_matrix = np.array([[ np.cos(-angle+phi),-np.sin(-angle+phi)],
                           [ np.sin(-angle+phi), np.cos(-angle+phi)]])
    proc = np.array((I, Q))
    proc = np.dot(rot_matrix, proc)
    return proc.transpose()

def _calculate_deffect_rate(Shots, n_rounds, with_reset=False):
    '''
    Shots must be a dictionary with format:
                                  |<---nr_shots--->|
    Shots['round <i>'] = np.array([0/1,......., 0/1])
    '''
    Deffect_rate = {}
    nr_shots = len(Shots['round 1'])
    if not with_reset:
        # M array is measured data
        # P array is parity data
        # D array is defect data
        M_values = np.ones((nr_shots, n_rounds))
        for r in range(n_rounds):
            # Convert to +1 and -1 values
            M_values[:,r] *= 1-2*(Shots[f'round {r+1}']) 

        P_values = np.hstack( (np.ones((nr_shots, 2)), M_values) )
        P_values = P_values[:,1:] * P_values[:,:-1]
    else:
        # P array is parity data
        # D array is defect data
        P_values = np.ones((nr_shots, n_rounds))
        for r in range(n_rounds):
            # Convert to +1 and -1 values
            P_values[:,r] *= 1-2*(Shots[f'round {r+1}']) 
    D_values = P_values[:,1:] * P_values[:,:-1]
    Deffect_rate = [ np.mean(1-D_values[:,i])/2 for i in range(n_rounds)]
    return Deffect_rate

class Repeated_stabilizer_measurement_analysis(ba.BaseDataAnalysis):
    def __init__(self,
                 qubit: str,
                 n_rounds: int,
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
        self.n_rounds = n_rounds

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
        ######################################
        # Sort shots and assign them
        ######################################
        n_rounds = self.n_rounds
        _cycle = n_rounds*4 + 3

        _raw_shots = self.raw_data_dict['data'][:,1:]# remove shot number
        _shots_0 = _raw_shots[4*n_rounds+0::_cycle]
        _shots_1 = _raw_shots[4*n_rounds+1::_cycle]
        _shots_2 = _raw_shots[4*n_rounds+2::_cycle]
        # Rotate data
        center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
        center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
        center_2 = np.array([np.mean(_shots_2[:,0]), np.mean(_shots_2[:,1])])
        raw_shots = _rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
        Shots_0 = raw_shots[4*n_rounds+0::_cycle]
        Shots_1 = raw_shots[4*n_rounds+1::_cycle]
        Shots_2 = raw_shots[4*n_rounds+2::_cycle]
        self.proc_data_dict['Shots_0'] = Shots_0
        self.proc_data_dict['Shots_1'] = Shots_1
        self.proc_data_dict['Shots_2'] = Shots_2
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
        self.proc_data_dict['dec_bounds'] = dec_bounds
        self.proc_data_dict['classifier'] = clf
        self.proc_data_dict['Fid_dict'] = Fid_dict
        self.proc_data_dict['Assignment_matrix'] = M
        #########################################
        # Project data along axis perpendicular
        # to the decision boundaries.
        #########################################
        ############################
        # Projection along 01 axis.
        ############################
        # Rotate shots over 01 axis
        shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['01'],phi=np.pi/2)
        shots_1 = _rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'],dec_bounds['01'],phi=np.pi/2)
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
        shots_1 = _rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
        shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
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
        shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
        shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
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
        ########################################
        # Analyze experiment shots, post-select
        # on leakage and calculate deffect rate
        ########################################
        # Sort experimental shots
        # 0-Normal experiment
        # 1-LRU on data experiment
        # 2-LRU on ancilla experiment
        # 3-LRU on data and ancilla experiment
        shots_exp_0 = {}
        Shots_qubit_0 = {}
        Shots_qutrit_0 = {}
        shots_exp_1 = {}
        Shots_qubit_1 = {}
        Shots_qutrit_1 = {}
        shots_exp_2 = {}
        Shots_qubit_2 = {}
        Shots_qutrit_2 = {}
        shots_exp_3 = {}
        Shots_qubit_3 = {}
        Shots_qutrit_3 = {}
        _zero_lvl = np.mean(self.proc_data_dict['Shots_0'][:,0])
        _one_lvl = np.mean(self.proc_data_dict['Shots_1'][:,1])
        threshold = self.proc_data_dict['projection_01']['threshold']
        for r in range(n_rounds):
            # Note we are using the rotated shots already
            shots_exp_0[f'round {r+1}'] = raw_shots[r+0*n_rounds::_cycle]
            shots_exp_1[f'round {r+1}'] = raw_shots[r+1*n_rounds::_cycle]
            shots_exp_2[f'round {r+1}'] = raw_shots[r+2*n_rounds::_cycle]
            shots_exp_3[f'round {r+1}'] = raw_shots[r+3*n_rounds::_cycle]
            # Perform Qubit assignment
            if _zero_lvl < threshold: # zero level is left of threshold
                Shots_qubit_0[f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_0[f'round {r+1}'][:,0]])
                Shots_qubit_1[f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_1[f'round {r+1}'][:,0]])
                Shots_qubit_2[f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_2[f'round {r+1}'][:,0]])
                Shots_qubit_3[f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_3[f'round {r+1}'][:,0]])
            else: # zero level is right of threshold
                Shots_qubit_0[f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_0[f'round {r+1}'][:,0]])
                Shots_qubit_1[f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_1[f'round {r+1}'][:,0]])
                Shots_qubit_2[f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_2[f'round {r+1}'][:,0]])
                Shots_qubit_3[f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_3[f'round {r+1}'][:,0]])
            # Perform Qutrit assignment
            Shots_qutrit_0[f'round {r+1}'] = clf.predict(shots_exp_0[f'round {r+1}'])
            Shots_qutrit_1[f'round {r+1}'] = clf.predict(shots_exp_1[f'round {r+1}'])
            Shots_qutrit_2[f'round {r+1}'] = clf.predict(shots_exp_2[f'round {r+1}'])
            Shots_qutrit_3[f'round {r+1}'] = clf.predict(shots_exp_3[f'round {r+1}'])
        # Calculate leakage in ancilla:
        Population_0 = {}
        Population_1 = {}
        Population_2 = {}
        Population_3 = {}
        def _get_pop_vector(Shots):
            p0 = np.mean(Shots==0)
            p1 = np.mean(Shots==1)
            p2 = np.mean(Shots==2)
            return np.array([p0, p1, p2])
        M_inv = np.linalg.inv(M)
        for r in range(n_rounds):
            _pop_vec_0 = _get_pop_vector(Shots_qutrit_0[f'round {r+1}'])
            _pop_vec_1 = _get_pop_vector(Shots_qutrit_1[f'round {r+1}'])
            _pop_vec_2 = _get_pop_vector(Shots_qutrit_2[f'round {r+1}'])
            _pop_vec_3 = _get_pop_vector(Shots_qutrit_3[f'round {r+1}'])
            Population_0[f'round {r+1}'] = np.dot(_pop_vec_0, M_inv)
            Population_1[f'round {r+1}'] = np.dot(_pop_vec_1, M_inv)
            Population_2[f'round {r+1}'] = np.dot(_pop_vec_2, M_inv)
            Population_3[f'round {r+1}'] = np.dot(_pop_vec_3, M_inv)
        Population_0 = np.array([Population_0[k][2] for k in Population_0.keys()])
        Population_1 = np.array([Population_1[k][2] for k in Population_1.keys()])
        Population_2 = np.array([Population_2[k][2] for k in Population_2.keys()])
        Population_3 = np.array([Population_3[k][2] for k in Population_3.keys()])
        # Fit leakage and seepage rates
        from scipy.optimize import curve_fit
        def _func(n, L, S):
            return (1 - np.exp(-n*(S+L)))*L/(S+L)
        _x = np.arange(0, self.n_rounds)+1
        popt_0, pcov_0 = curve_fit(_func, _x, Population_0)
        popt_1, pcov_1 = curve_fit(_func, _x, Population_1)
        popt_2, pcov_2 = curve_fit(_func, _x, Population_2)
        popt_3, pcov_3 = curve_fit(_func, _x, Population_3)
        self.proc_data_dict['Population_0'] = Population_0
        self.proc_data_dict['Population_1'] = Population_1
        self.proc_data_dict['Population_2'] = Population_2
        self.proc_data_dict['Population_3'] = Population_3
        # Perform post-selection on Qutrit readout
        Shots_qutrit_ps_0 = {}
        Shots_qutrit_ps_1 = {}
        Shots_qutrit_ps_2 = {}
        Shots_qutrit_ps_3 = {}
        nr_shots = len(Shots_qutrit_0['round 1'])
        _mask_0 = np.ones(nr_shots)
        _mask_1 = np.ones(nr_shots)
        _mask_2 = np.ones(nr_shots)
        _mask_3 = np.ones(nr_shots)
        Ps_fraction_0 = np.ones(n_rounds)
        Ps_fraction_1 = np.ones(n_rounds)
        Ps_fraction_2 = np.ones(n_rounds)
        Ps_fraction_3 = np.ones(n_rounds)
        # get post selection mask and ps fraction for each round
        for r in range(n_rounds):
            _mask_0 *= np.array([1 if s != 2 else np.nan for s in Shots_qutrit_0[f'round {r+1}']])
            _mask_1 *= np.array([1 if s != 2 else np.nan for s in Shots_qutrit_1[f'round {r+1}']])
            _mask_2 *= np.array([1 if s != 2 else np.nan for s in Shots_qutrit_2[f'round {r+1}']])
            _mask_3 *= np.array([1 if s != 2 else np.nan for s in Shots_qutrit_3[f'round {r+1}']])
            Ps_fraction_0[r] = np.nansum(_mask_0)/nr_shots
            Ps_fraction_1[r] = np.nansum(_mask_1)/nr_shots
            Ps_fraction_2[r] = np.nansum(_mask_2)/nr_shots
            Ps_fraction_3[r] = np.nansum(_mask_3)/nr_shots
        # remove leakage detection events
        for r in range(n_rounds):
            Shots_qutrit_ps_0[f'round {r+1}'] = Shots_qutrit_0[f'round {r+1}'][~np.isnan(_mask_0)]
            Shots_qutrit_ps_1[f'round {r+1}'] = Shots_qutrit_1[f'round {r+1}'][~np.isnan(_mask_1)]
            Shots_qutrit_ps_2[f'round {r+1}'] = Shots_qutrit_2[f'round {r+1}'][~np.isnan(_mask_2)]
            Shots_qutrit_ps_3[f'round {r+1}'] = Shots_qutrit_3[f'round {r+1}'][~np.isnan(_mask_3)]
        ###########################
        # Calculate defect rate
        ###########################
        print(f'Post-selected fraction normal: {Ps_fraction_0[-1]*100:.5f} %')
        print(f'Post-selected fraction LRU data: {Ps_fraction_1[-1]*100:.5f} %')
        print(f'Post-selected fraction LRU ancilla: {Ps_fraction_2[-1]*100:.5f} %')
        print(f'Post-selected fraction LRU both: {Ps_fraction_3[-1]*100:.5f} %')
        deffect_rate_0 = _calculate_deffect_rate(Shots_qubit_0, n_rounds)
        deffect_rate_0_ps = _calculate_deffect_rate(Shots_qutrit_ps_0, n_rounds)
        deffect_rate_1 = _calculate_deffect_rate(Shots_qubit_1, n_rounds)
        deffect_rate_1_ps = _calculate_deffect_rate(Shots_qutrit_ps_1, n_rounds)
        deffect_rate_2 = _calculate_deffect_rate(Shots_qubit_2, n_rounds)
        deffect_rate_2_ps = _calculate_deffect_rate(Shots_qutrit_ps_2, n_rounds)
        deffect_rate_3 = _calculate_deffect_rate(Shots_qubit_3, n_rounds)
        deffect_rate_3_ps = _calculate_deffect_rate(Shots_qutrit_ps_3, n_rounds)
        self.qoi = {}
        self.qoi['deffect_rate_normal'] = deffect_rate_0
        self.qoi['deffect_rate_LRU_data'] = deffect_rate_1
        self.qoi['deffect_rate_LRU_ancilla'] = deffect_rate_2
        self.qoi['deffect_rate_LRU_data_ancilla'] = deffect_rate_3
        self.qoi['deffect_rate_normal_ps'] = deffect_rate_0_ps
        self.qoi['deffect_rate_LRU_data_ps'] = deffect_rate_1_ps
        self.qoi['deffect_rate_LRU_ancilla_ps'] = deffect_rate_2_ps
        self.qoi['deffect_rate_LRU_data_ancilla_ps'] = deffect_rate_3_ps
        self.qoi['Ps_fraction_normal'] = Ps_fraction_0
        self.qoi['Ps_fraction_LRU_data'] = Ps_fraction_1
        self.qoi['Ps_fraction_LRU_ancilla'] = Ps_fraction_2
        self.qoi['Ps_fraction_LRU_data_ancilla'] = Ps_fraction_3
        self.qoi['Population_normal'] = Population_0
        self.qoi['Population_LRU_data'] = Population_1
        self.qoi['Population_LRU_ancilla'] = Population_2
        self.qoi['Population_LRU_data_ancilla'] = Population_3

    def prepare_plots(self):
        self.axs_dict = {}
        fig = plt.figure(figsize=(8,4), dpi=100)
        axs = [fig.add_subplot(121),
               fig.add_subplot(322),
               fig.add_subplot(324),
               fig.add_subplot(326)]
        # fig.patch.set_alpha(0)
        self.axs_dict['IQ_readout_histogram'] = axs[0]
        self.figs['IQ_readout_histogram'] = fig
        self.plot_dicts['IQ_readout_histogram'] = {
            'plotfn': ssro_IQ_projection_plotfn,
            'ax_id': 'IQ_readout_histogram',
            'shots_0': self.proc_data_dict['Shots_0'],
            'shots_1': self.proc_data_dict['Shots_1'],
            'shots_2': self.proc_data_dict['Shots_2'],
            'projection_01': self.proc_data_dict['projection_01'],
            'projection_12': self.proc_data_dict['projection_12'],
            'projection_02': self.proc_data_dict['projection_02'],
            'classifier': self.proc_data_dict['classifier'],
            'dec_bounds': self.proc_data_dict['dec_bounds'],
            'Fid_dict': self.proc_data_dict['Fid_dict'],
            'qubit': self.qubit,
            'timestamp': self.timestamp
        }

        fig, axs = plt.subplots(figsize=(6*1.5,2.5*1.5), ncols=2, dpi=100)
        self.axs_dict['Deffect_rate_plot'] = axs[0]
        self.figs['Deffect_rate_plot'] = fig
        self.plot_dicts['Deffect_rate_plot'] = {
            'plotfn': deffect_rate_plotfn,
            'ax_id': 'Deffect_rate_plot',
            'n_rounds': self.n_rounds,
            'deffect_rate_0': self.qoi['deffect_rate_normal'],
            'deffect_rate_0_ps': self.qoi['deffect_rate_normal_ps'],
            'deffect_rate_1': self.qoi['deffect_rate_LRU_data'],
            'deffect_rate_1_ps': self.qoi['deffect_rate_LRU_data_ps'], 
            'deffect_rate_2': self.qoi['deffect_rate_LRU_ancilla'],
            'deffect_rate_2_ps': self.qoi['deffect_rate_LRU_ancilla_ps'], 
            'deffect_rate_3': self.qoi['deffect_rate_LRU_data_ancilla'],
            'deffect_rate_3_ps': self.qoi['deffect_rate_LRU_data_ancilla_ps'], 
            'ps_0': self.qoi['Ps_fraction_normal'], 
            'ps_1': self.qoi['Ps_fraction_LRU_data'], 
            'ps_2': self.qoi['Ps_fraction_LRU_ancilla'], 
            'ps_3': self.qoi['Ps_fraction_LRU_data_ancilla'],
            'p_0': self.qoi['Population_normal'], 
            'p_1': self.qoi['Population_LRU_data'], 
            'p_2': self.qoi['Population_LRU_ancilla'], 
            'p_3': self.qoi['Population_LRU_data_ancilla'], 
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
    shots_3,
    projection_01,
    projection_12,
    projection_03,
    projection_23,
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
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    popt_3 = _fit_2D_gaussian(shots_3[:,0], shots_3[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.025)
    axs[0].plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.025)
    axs[0].plot(shots_2[:,0], shots_2[:,1], '.', color='C2', alpha=0.025)
    axs[0].plot(shots_3[:,0], shots_3[:,1], '.', color='gold', alpha=0.025)
    axs[0].plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_3[1]], [0, popt_3[2]], '--', color='k', lw=.5)
    axs[0].plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    axs[0].plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    axs[0].plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
    axs[0].plot(popt_3[1], popt_3[2], '.', color='gold', label='$3^\mathrm{nd}$ excited')
    axs[0].plot(popt_0[1], popt_0[2], 'x', color='white')
    axs[0].plot(popt_1[1], popt_1[2], 'x', color='white')
    axs[0].plot(popt_2[1], popt_2[2], 'x', color='white')
    axs[0].plot(popt_3[1], popt_3[2], 'x', color='white')
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
    circle_3 = Ellipse((popt_3[1], popt_3[2]),
                      width=4*popt_3[3], height=4*popt_3[4],
                      angle=-popt_3[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_3)
    # Plot classifier zones
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    Lim_points = {}
    for bound in ['01', '12', '03', '23']:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot 0 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['03']]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['12'], Lim_points['23']]
    _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['03'], Lim_points['23']]
    _patch = Polygon(_points, color='gold', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    for bound in ['01', '12', '03', '23']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
    axs[0].legend(frameon=False, loc=1)
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
    # 03 projection
    _bin_c = projection_03['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[3].bar(_bin_c, projection_03['h0'], bin_width, fc='C0', alpha=0.4)
    axs[3].bar(_bin_c, projection_03['h3'], bin_width, fc='gold', alpha=0.4)
    axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_03['popt0']), '-C0')
    axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_03['popt3']), '-C1')
    axs[3].axvline(projection_03['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_03["Fid"]*100:.1f}%',
                      f'SNR : {projection_03["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[3].text(.775, .9, text, transform=axs[3].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[3].text(projection_03['popt0'][0], projection_03['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[3].text(projection_03['popt3'][0], projection_03['popt3'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='gold')
    axs[3].set_xticklabels([])
    axs[3].set_xlim(_bin_c[0], _bin_c[-1])
    axs[3].set_ylim(bottom=0)
    axs[3].set_xlabel('Integrated voltage')
    # 23 projection
    _bin_c = projection_23['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[4].bar(_bin_c, projection_23['h2'], bin_width, fc='C2', alpha=0.4)
    axs[4].bar(_bin_c, projection_23['h3'], bin_width, fc='gold', alpha=0.4)
    axs[4].plot(_bin_c, double_gauss(_bin_c, *projection_23['popt2']), '-C2')
    axs[4].plot(_bin_c, double_gauss(_bin_c, *projection_23['popt3']), '-C1')
    axs[4].axvline(projection_23['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_23["Fid"]*100:.1f}%',
                      f'SNR : {projection_23["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[4].text(.775, .9, text, transform=axs[4].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[4].text(projection_23['popt2'][0], projection_23['popt2'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[4].text(projection_23['popt3'][0], projection_23['popt3'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='gold')
    axs[4].set_xticklabels([])
    axs[4].set_xlim(_bin_c[0], _bin_c[-1])
    axs[4].set_ylim(bottom=0)
    axs[4].set_xlabel('Integrated voltage')
    # Write fidelity textbox
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                      f'$F_f$ : {Fid_dict["2"]*100:.1f}%',
                      f'$F_h$ : {Fid_dict["3"]*100:.1f}%' if '3' in Fid_dict.keys() else '',
                      f'$F_\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'))
    props = dict(boxstyle='round', facecolor='gray', alpha=.2)
    axs[1].text(1.12, 1, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props)

def deffect_rate_plotfn(
    n_rounds,
    deffect_rate_0,
    deffect_rate_0_ps,
    deffect_rate_1,
    deffect_rate_1_ps, 
    deffect_rate_2,
    deffect_rate_2_ps, 
    deffect_rate_3,
    deffect_rate_3_ps, 
    ps_0, 
    ps_1, 
    ps_2, 
    ps_3, 
    p_0, 
    p_1, 
    p_2, 
    p_3, 
    timestamp,
    qubit, 
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_0[1:], 'C0-', label='Normal')
    # axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_0_ps[1:], 'C0-', alpha=.5)
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_1[1:], 'C1-', label='LRU data')
    # axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_1_ps[1:], 'C1-', alpha=.5)
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_2[1:], 'C2-', label='LRU ancilla')
    # axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_2_ps[1:], 'C2-', alpha=.5)
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_3[1:], 'C3-', label='LRU data ancilla')
    # axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_3_ps[1:], 'C3-', alpha=.5)
    axs[0].grid(ls='--')
    axs[0].set_ylabel('error probability')
    axs[0].set_xlabel('rounds')
    axs[0].legend(frameon=False, bbox_to_anchor = (1.01, 1))
    axs[0].set_title('Deffect rate')

    axs[1].plot((np.arange(n_rounds)+1), p_0*100, 'C0-', label='Normal')
    axs[1].plot((np.arange(n_rounds)+1), p_1*100, 'C1-', label='LRU data')
    axs[1].plot((np.arange(n_rounds)+1), p_2*100, 'C2-', label='LRU ancilla')
    axs[1].plot((np.arange(n_rounds)+1), p_3*100, 'C3-', label='LRU data ancilla')
    axs[1].set_ylabel(r'$|f\rangle$ population (%)')
    axs[1].set_xlabel('rounds')
    axs[1].set_title('Leakage population')
    axs[1].grid(ls='--')

    fig.suptitle(f'{timestamp}\n{qubit} repeated stabilizer experiment', y=1.01)
    fig.tight_layout()


class Repeated_stabilizer_measurement_with_data_measurement_analysis(ba.BaseDataAnalysis):
    def __init__(self,
                 qubit: str,
                 Rounds: list,
                 heralded_init: bool = False,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 with_reset: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.qubit = qubit
        self.Rounds = Rounds
        self.with_reset = with_reset
        self.heralded_init = heralded_init

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
        ######################################
        # Sort shots and assign them
        ######################################
        Rounds = self.Rounds
        _total_rounds = np.sum(Rounds)
        if self.heralded_init:
            _total_rounds += len(Rounds)
        _cycle = _total_rounds*4 + 4
        if self.heralded_init:
            _cycle += 4
        # Get qubit names in channel order
        names = [ name.decode().split(' ')[-2] for name in self.raw_data_dict['value_names'] ]
        self.Qubits = names[::2]
        # Dictionary that will store raw shots
        # so that they can later be sorted.
        raw_shots = {q: {} for q in self.Qubits}
        for q_idx, qubit in enumerate(self.Qubits):
            self.proc_data_dict[qubit] = {}
            _ch_I, _ch_Q = 2*q_idx+1, 2*q_idx+2
            _raw_shots = self.raw_data_dict['data'][:,[_ch_I, _ch_Q]]
            if self.heralded_init:
                _shots_0 = _raw_shots[4*_total_rounds+1::_cycle]
                _shots_1 = _raw_shots[4*_total_rounds+3::_cycle]
                _shots_2 = _raw_shots[4*_total_rounds+5::_cycle]
                _shots_3 = _raw_shots[4*_total_rounds+7::_cycle]
            else:
                _shots_0 = _raw_shots[4*_total_rounds+0::_cycle]
                _shots_1 = _raw_shots[4*_total_rounds+1::_cycle]
                _shots_2 = _raw_shots[4*_total_rounds+2::_cycle]
                _shots_3 = _raw_shots[4*_total_rounds+3::_cycle]
            # Rotate data
            center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
            center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
            center_2 = np.array([np.mean(_shots_2[:,0]), np.mean(_shots_2[:,1])])
            center_3 = np.array([np.mean(_shots_3[:,0]), np.mean(_shots_3[:,1])])
            raw_shots[qubit] = _rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
            if self.heralded_init:
                Shots_0 = raw_shots[qubit][4*_total_rounds+1::_cycle]
                Shots_1 = raw_shots[qubit][4*_total_rounds+3::_cycle]
                Shots_2 = raw_shots[qubit][4*_total_rounds+5::_cycle]
                Shots_3 = raw_shots[qubit][4*_total_rounds+7::_cycle]
            else:
                Shots_0 = raw_shots[qubit][4*_total_rounds+0::_cycle]
                Shots_1 = raw_shots[qubit][4*_total_rounds+1::_cycle]
                Shots_2 = raw_shots[qubit][4*_total_rounds+2::_cycle]
                Shots_3 = raw_shots[qubit][4*_total_rounds+3::_cycle]
            self.proc_data_dict[qubit]['Shots_0'] = Shots_0
            self.proc_data_dict[qubit]['Shots_1'] = Shots_1
            self.proc_data_dict[qubit]['Shots_2'] = Shots_2
            self.proc_data_dict[qubit]['Shots_3'] = Shots_3
            if 'Z' in qubit:
                # Use classifier for data
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
            else:
                # Use classifier for data
                data = np.concatenate((Shots_0, Shots_1, Shots_2))
                labels = [0 for s in Shots_0]+[1 for s in Shots_1]+\
                         [2 for s in Shots_2]
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
                # Make it a 4x4 matrix
                M = np.append(M, [[0,0,0]], 0)
                M = np.append(M, [[0],[0],[0],[1]], 1)
            self.proc_data_dict[qubit]['classifier'] = clf
            self.proc_data_dict[qubit]['Fid_dict'] = Fid_dict
            self.proc_data_dict[qubit]['Assignment_matrix'] = M
            self.proc_data_dict[qubit]['dec_bounds'] = dec_bounds

            ##################
            # Post selection
            ##################
            if self.heralded_init:
                _ps_shots_0 = raw_shots[qubit][4*_total_rounds+0::_cycle]
                _ps_shots_1 = raw_shots[qubit][4*_total_rounds+2::_cycle]
                _ps_shots_2 = raw_shots[qubit][4*_total_rounds+4::_cycle]
                _ps_shots_3 = raw_shots[qubit][4*_total_rounds+6::_cycle]

                def _post_select(shots, ps_shots):
                    _ps_shots = clf.predict(ps_shots)
                    _mask = np.array([1 if s == 0 else np.nan for s in _ps_shots])
                    # print(np.nansum(_mask)/ len(_mask))
                    shots = shots[~np.isnan(_mask)]
                    return shots

                Shots_0 = _post_select(Shots_0, _ps_shots_0)
                Shots_1 = _post_select(Shots_1, _ps_shots_1)
                Shots_2 = _post_select(Shots_2, _ps_shots_2)
                Shots_3 = _post_select(Shots_3, _ps_shots_3)
                self.proc_data_dict[qubit]['Shots_0'] = Shots_0
                self.proc_data_dict[qubit]['Shots_1'] = Shots_1
                self.proc_data_dict[qubit]['Shots_2'] = Shots_2
                self.proc_data_dict[qubit]['Shots_3'] = Shots_3
                if 'Z' in qubit:
                    # Use classifier for data
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
                else:
                    # Use classifier for data
                    data = np.concatenate((Shots_0, Shots_1, Shots_2))
                    labels = [0 for s in Shots_0]+[1 for s in Shots_1]+\
                             [2 for s in Shots_2]
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
                    # Make it a 4x4 matrix
                    M = np.append(M, [[0,0,0]], 0)
                    M = np.append(M, [[0],[0],[0],[1]], 1)
                self.proc_data_dict[qubit]['classifier'] = clf
                self.proc_data_dict[qubit]['Fid_dict'] = Fid_dict
                self.proc_data_dict[qubit]['Assignment_matrix'] = M
                self.proc_data_dict[qubit]['dec_bounds'] = dec_bounds
            #########################################
            # Project data along axis perpendicular
            # to the decision boundaries.
            #########################################
            ############################
            # Projection along 01 axis.
            ############################
            # Rotate shots over 01 axis
            shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['01'],phi=np.pi/2)
            shots_1 = _rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'],dec_bounds['01'],phi=np.pi/2)
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
            self.proc_data_dict[qubit]['projection_01'] = {}
            self.proc_data_dict[qubit]['projection_01']['h0'] = h0
            self.proc_data_dict[qubit]['projection_01']['h1'] = h1
            self.proc_data_dict[qubit]['projection_01']['bin_centers'] = bin_centers
            self.proc_data_dict[qubit]['projection_01']['popt0'] = popt0
            self.proc_data_dict[qubit]['projection_01']['popt1'] = popt1
            self.proc_data_dict[qubit]['projection_01']['SNR'] = params_01['SNR']
            self.proc_data_dict[qubit]['projection_01']['Fid'] = Fid_01
            self.proc_data_dict[qubit]['projection_01']['threshold'] = threshold_01
            ############################
            # Projection along 12 axis.
            ############################
            # Rotate shots over 12 axis
            shots_1 = _rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
            shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
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
            self.proc_data_dict[qubit]['projection_12'] = {}
            self.proc_data_dict[qubit]['projection_12']['h1'] = h1
            self.proc_data_dict[qubit]['projection_12']['h2'] = h2
            self.proc_data_dict[qubit]['projection_12']['bin_centers'] = bin_centers
            self.proc_data_dict[qubit]['projection_12']['popt1'] = popt1
            self.proc_data_dict[qubit]['projection_12']['popt2'] = popt2
            self.proc_data_dict[qubit]['projection_12']['SNR'] = params_12['SNR']
            self.proc_data_dict[qubit]['projection_12']['Fid'] = Fid_12
            self.proc_data_dict[qubit]['projection_12']['threshold'] = threshold_12
            
            if 'Z' in qubit:
                ############################
                # Projection along 03 axis.
                ############################
                # Rotate shots over 03 axis
                shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['03'], phi=np.pi/2)
                shots_3 = _rotate_and_center_data(Shots_3[:,0],Shots_3[:,1],dec_bounds['mean'],dec_bounds['03'], phi=np.pi/2)
                # Take relavant quadrature
                shots_0 = shots_0[:,0]
                shots_3 = shots_3[:,0]
                n_shots_3 = len(shots_3)
                # find range
                _all_shots = np.concatenate((shots_0, shots_3))
                _range = (np.min(_all_shots), np.max(_all_shots))
                # Sort shots in unique values
                x0, n0 = np.unique(shots_0, return_counts=True)
                x3, n3 = np.unique(shots_3, return_counts=True)
                Fid_03, threshold_03 = _calculate_fid_and_threshold(x0, n0, x3, n3)
                # Histogram of shots for 1 and 2
                h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
                h3, bin_edges = np.histogram(shots_3, bins=100, range=_range)
                bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
                popt0, popt3, params_03 = _fit_double_gauss(bin_centers, h0, h3)
                # Save processed data
                self.proc_data_dict[qubit]['projection_03'] = {}
                self.proc_data_dict[qubit]['projection_03']['h0'] = h0
                self.proc_data_dict[qubit]['projection_03']['h3'] = h3
                self.proc_data_dict[qubit]['projection_03']['bin_centers'] = bin_centers
                self.proc_data_dict[qubit]['projection_03']['popt0'] = popt0
                self.proc_data_dict[qubit]['projection_03']['popt3'] = popt3
                self.proc_data_dict[qubit]['projection_03']['SNR'] = params_03['SNR']
                self.proc_data_dict[qubit]['projection_03']['Fid'] = Fid_03
                self.proc_data_dict[qubit]['projection_03']['threshold'] = threshold_03
                ############################
                # Projection along 23 axis.
                ############################
                # Rotate shots over 23 axis
                shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['23'], phi=np.pi/2)
                shots_3 = _rotate_and_center_data(Shots_3[:,0],Shots_3[:,1],dec_bounds['mean'],dec_bounds['23'], phi=np.pi/2)
                # Take relavant quadrature
                shots_2 = shots_2[:,0]
                shots_3 = shots_3[:,0]
                n_shots_3 = len(shots_3)
                # find range
                _all_shots = np.concatenate((shots_2, shots_3))
                _range = (np.min(_all_shots), np.max(_all_shots))
                # Sort shots in unique values
                x2, n2 = np.unique(shots_2, return_counts=True)
                x3, n3 = np.unique(shots_3, return_counts=True)
                Fid_23, threshold_23 = _calculate_fid_and_threshold(x2, n2, x3, n3)
                # Histogram of shots for 1 and 2
                h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
                h3, bin_edges = np.histogram(shots_3, bins=100, range=_range)
                bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
                popt2, popt3, params_23 = _fit_double_gauss(bin_centers, h2, h3)
                # Save processed data
                self.proc_data_dict[qubit]['projection_23'] = {}
                self.proc_data_dict[qubit]['projection_23']['h2'] = h2
                self.proc_data_dict[qubit]['projection_23']['h3'] = h3
                self.proc_data_dict[qubit]['projection_23']['bin_centers'] = bin_centers
                self.proc_data_dict[qubit]['projection_23']['popt2'] = popt2
                self.proc_data_dict[qubit]['projection_23']['popt3'] = popt3
                self.proc_data_dict[qubit]['projection_23']['SNR'] = params_23['SNR']
                self.proc_data_dict[qubit]['projection_23']['Fid'] = Fid_23
                self.proc_data_dict[qubit]['projection_23']['threshold'] = threshold_23
            else:
                ############################
                # Projection along 02 axis.
                ############################
                # Rotate shots over 02 axis
                shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
                shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
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
                self.proc_data_dict[qubit]['projection_02'] = {}
                self.proc_data_dict[qubit]['projection_02']['h0'] = h0
                self.proc_data_dict[qubit]['projection_02']['h2'] = h2
                self.proc_data_dict[qubit]['projection_02']['bin_centers'] = bin_centers
                self.proc_data_dict[qubit]['projection_02']['popt0'] = popt0
                self.proc_data_dict[qubit]['projection_02']['popt2'] = popt2
                self.proc_data_dict[qubit]['projection_02']['SNR'] = params_02['SNR']
                self.proc_data_dict[qubit]['projection_02']['Fid'] = Fid_02
                self.proc_data_dict[qubit]['projection_02']['threshold'] = threshold_02

        ########################################
        # Analyze experiment shots, post-select
        # on leakage and calculate deffect rate
        ########################################
        # Sort experimental shots
        # 0-Normal experiment
        # 1-LRU on data experiment
        # 2-LRU on ancilla experiment
        # 3-LRU on data and ancilla experiment
        shots_exp_0 = {q: {} for q in self.Qubits}
        Shots_qubit_0 = {q: {} for q in self.Qubits}
        Shots_qutrit_0 = {q: {} for q in self.Qubits}
        shots_exp_1 = {q: {} for q in self.Qubits}
        Shots_qubit_1 = {q: {} for q in self.Qubits}
        Shots_qutrit_1 = {q: {} for q in self.Qubits}
        shots_exp_2 = {q: {} for q in self.Qubits}
        Shots_qubit_2 = {q: {} for q in self.Qubits}
        Shots_qutrit_2 = {q: {} for q in self.Qubits}
        shots_exp_3 = {q: {} for q in self.Qubits}
        Shots_qubit_3 = {q: {} for q in self.Qubits}
        Shots_qutrit_3 = {q: {} for q in self.Qubits}
        for q in self.Qubits:
            # threshold = _get_threshold(Shots_0, Shots_1)
            _zero_lvl = np.mean(self.proc_data_dict[q]['Shots_0'][:,0])
            _one_lvl = np.mean(self.proc_data_dict[q]['Shots_1'][:,1])
            threshold = self.proc_data_dict[q]['projection_01']['threshold']
            _clf = self.proc_data_dict[q]['classifier']
            for r_idx, n_rounds in enumerate(Rounds):
                shots_exp_0[q][f'{n_rounds}_R'] = {}
                Shots_qubit_0[q][f'{n_rounds}_R'] = {}
                Shots_qutrit_0[q][f'{n_rounds}_R'] = {}
                shots_exp_1[q][f'{n_rounds}_R'] = {}
                Shots_qubit_1[q][f'{n_rounds}_R'] = {}
                Shots_qutrit_1[q][f'{n_rounds}_R'] = {}
                shots_exp_2[q][f'{n_rounds}_R'] = {}
                Shots_qubit_2[q][f'{n_rounds}_R'] = {}
                Shots_qutrit_2[q][f'{n_rounds}_R'] = {}
                shots_exp_3[q][f'{n_rounds}_R'] = {}
                Shots_qubit_3[q][f'{n_rounds}_R'] = {}
                Shots_qutrit_3[q][f'{n_rounds}_R'] = {}
                # counter for number of shots in previous rounds
                _aux = int(4*np.sum(Rounds[:r_idx]))
                if self.heralded_init:
                    _aux = int(4*(np.sum(Rounds[:r_idx])+len(Rounds[:r_idx])))
                for r in range(n_rounds):
                    # Note we are using the rotated shots already
                    shots_exp_0[q][f'{n_rounds}_R'][f'round {r+1}'] = \
                        raw_shots[q][r+0*(n_rounds+self.heralded_init)+self.heralded_init+_aux::_cycle]
                    shots_exp_1[q][f'{n_rounds}_R'][f'round {r+1}'] = \
                        raw_shots[q][r+1*(n_rounds+self.heralded_init)+self.heralded_init+_aux::_cycle]
                    shots_exp_2[q][f'{n_rounds}_R'][f'round {r+1}'] = \
                        raw_shots[q][r+2*(n_rounds+self.heralded_init)+self.heralded_init+_aux::_cycle]
                    shots_exp_3[q][f'{n_rounds}_R'][f'round {r+1}'] = \
                        raw_shots[q][r+3*(n_rounds+self.heralded_init)+self.heralded_init+_aux::_cycle]
                    # Perform Qubit assignment
                    if _zero_lvl < threshold: # zero level is left of threshold
                        Shots_qubit_0[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_0[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_1[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_1[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_2[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_2[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_3[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_3[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                    else: # zero level is right of threshold
                        Shots_qubit_0[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_0[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_1[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_1[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_2[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_2[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_3[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_3[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                    # Perform Qutrit assignment
                    Shots_qutrit_0[q][f'{n_rounds}_R'][f'round {r+1}'] = _clf.predict(shots_exp_0[q][f'{n_rounds}_R'][f'round {r+1}'])
                    Shots_qutrit_1[q][f'{n_rounds}_R'][f'round {r+1}'] = _clf.predict(shots_exp_1[q][f'{n_rounds}_R'][f'round {r+1}'])
                    Shots_qutrit_2[q][f'{n_rounds}_R'][f'round {r+1}'] = _clf.predict(shots_exp_2[q][f'{n_rounds}_R'][f'round {r+1}'])
                    Shots_qutrit_3[q][f'{n_rounds}_R'][f'round {r+1}'] = _clf.predict(shots_exp_3[q][f'{n_rounds}_R'][f'round {r+1}'])
                # Post selection
                if self.heralded_init:
                    # Sort heralding shots
                    shots_exp_0[q][f'{n_rounds}_R']['ps'] = \
                        raw_shots[q][0*(n_rounds+self.heralded_init)+_aux::_cycle]
                    shots_exp_1[q][f'{n_rounds}_R']['ps'] = \
                        raw_shots[q][1*(n_rounds+self.heralded_init)+_aux::_cycle]
                    shots_exp_2[q][f'{n_rounds}_R']['ps'] = \
                        raw_shots[q][2*(n_rounds+self.heralded_init)+_aux::_cycle]
                    shots_exp_3[q][f'{n_rounds}_R']['ps'] = \
                        raw_shots[q][3*(n_rounds+self.heralded_init)+_aux::_cycle]
                    # Classify heralding shots
                    Shots_qutrit_0[q][f'{n_rounds}_R']['round 0'] = _clf.predict(shots_exp_0[q][f'{n_rounds}_R']['ps'])
                    Shots_qutrit_1[q][f'{n_rounds}_R']['round 0'] = _clf.predict(shots_exp_1[q][f'{n_rounds}_R']['ps'])
                    Shots_qutrit_2[q][f'{n_rounds}_R']['round 0'] = _clf.predict(shots_exp_2[q][f'{n_rounds}_R']['ps'])
                    Shots_qutrit_3[q][f'{n_rounds}_R']['round 0'] = _clf.predict(shots_exp_3[q][f'{n_rounds}_R']['ps'])
                    # Compute post-selection mask
                    Shots_qutrit_0[q][f'{n_rounds}_R']['ps'] = np.array([ 1 if s == 0 else np.nan for s in Shots_qutrit_0[q][f'{n_rounds}_R']['round 0'] ])
                    Shots_qutrit_1[q][f'{n_rounds}_R']['ps'] = np.array([ 1 if s == 0 else np.nan for s in Shots_qutrit_1[q][f'{n_rounds}_R']['round 0'] ])
                    Shots_qutrit_2[q][f'{n_rounds}_R']['ps'] = np.array([ 1 if s == 0 else np.nan for s in Shots_qutrit_2[q][f'{n_rounds}_R']['round 0'] ])
                    Shots_qutrit_3[q][f'{n_rounds}_R']['ps'] = np.array([ 1 if s == 0 else np.nan for s in Shots_qutrit_3[q][f'{n_rounds}_R']['round 0'] ])
        # Perform post-selection
        if self.heralded_init:
            for R in Rounds:
                _n_shots = len(Shots_qutrit_0[q][f'{R}_R']['ps']) 
                _mask_0 = np.ones(_n_shots)
                _mask_1 = np.ones(_n_shots)
                _mask_2 = np.ones(_n_shots)
                _mask_3 = np.ones(_n_shots)
                for q in self.Qubits:
                    _mask_0 *= Shots_qutrit_0[q][f'{R}_R']['ps']
                    _mask_1 *= Shots_qutrit_1[q][f'{R}_R']['ps']
                    _mask_2 *= Shots_qutrit_2[q][f'{R}_R']['ps']
                    _mask_3 *= Shots_qutrit_3[q][f'{R}_R']['ps']
                print(f'{R}_R Percentage of post-selected shots 0: {np.nansum(_mask_0)/len(_mask_0)*100:.2f}%')
                print(f'{R}_R Percentage of post-selected shots 1: {np.nansum(_mask_1)/len(_mask_1)*100:.2f}%')
                print(f'{R}_R Percentage of post-selected shots 2: {np.nansum(_mask_2)/len(_mask_2)*100:.2f}%')
                print(f'{R}_R Percentage of post-selected shots 3: {np.nansum(_mask_3)/len(_mask_3)*100:.2f}%')
                for q in self.Qubits:
                    for r in range(R):
                        # Remove marked shots in qubit shots
                        Shots_qubit_0[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qubit_0[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_0)]
                        Shots_qubit_1[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qubit_1[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_1)]
                        Shots_qubit_2[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qubit_2[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_2)]
                        Shots_qubit_3[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qubit_3[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_3)]
                        # Remove marked shots in qutrit shots
                        Shots_qutrit_0[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qutrit_0[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_0)]
                        Shots_qutrit_1[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qutrit_1[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_1)]
                        Shots_qutrit_2[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qutrit_2[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_2)]
                        Shots_qutrit_3[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qutrit_3[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_3)]
        self.proc_data_dict['Shots_qubit_0'] = Shots_qubit_0
        self.proc_data_dict['Shots_qubit_1'] = Shots_qubit_1
        self.proc_data_dict['Shots_qubit_2'] = Shots_qubit_2
        self.proc_data_dict['Shots_qubit_3'] = Shots_qubit_3
        self.proc_data_dict['Shots_qutrit_0'] = Shots_qutrit_0
        self.proc_data_dict['Shots_qutrit_1'] = Shots_qutrit_1
        self.proc_data_dict['Shots_qutrit_2'] = Shots_qutrit_2
        self.proc_data_dict['Shots_qutrit_3'] = Shots_qutrit_3
        ####################
        # Calculate leakage
        ####################
        Population_0 = {q:{} for q in self.Qubits}
        Population_1 = {q:{} for q in self.Qubits}
        Population_2 = {q:{} for q in self.Qubits}
        Population_3 = {q:{} for q in self.Qubits}
        Population_f_0 = {q:{} for q in self.Qubits}
        Population_f_1 = {q:{} for q in self.Qubits}
        Population_f_2 = {q:{} for q in self.Qubits}
        Population_f_3 = {q:{} for q in self.Qubits}
        Population_h_0 = {q:{} for q in self.Qubits}
        Population_h_1 = {q:{} for q in self.Qubits}
        Population_h_2 = {q:{} for q in self.Qubits}
        Population_h_3 = {q:{} for q in self.Qubits}
        def _get_pop_vector(Shots):
            p0 = np.mean(Shots==0)
            p1 = np.mean(Shots==1)
            p2 = np.mean(Shots==2)
            p3 = np.mean(Shots==3)
            return np.array([p0, p1, p2, p3])
        for q in self.Qubits:
            M_inv = np.linalg.inv(self.proc_data_dict[q]['Assignment_matrix'])
            if q == self.qubit:
                # For the ancilla qubit we'll calculate  
                # leakage in every measurement round.
                for n_rounds in Rounds:
                    Population_0[q][f'{n_rounds}_R'] = {}
                    Population_1[q][f'{n_rounds}_R'] = {}
                    Population_2[q][f'{n_rounds}_R'] = {}
                    Population_3[q][f'{n_rounds}_R'] = {}
                    for r in range(n_rounds):
                        _pop_vec_0 = _get_pop_vector(Shots_qutrit_0[q][f'{n_rounds}_R'][f'round {r+1}'])
                        _pop_vec_1 = _get_pop_vector(Shots_qutrit_1[q][f'{n_rounds}_R'][f'round {r+1}'])
                        _pop_vec_2 = _get_pop_vector(Shots_qutrit_2[q][f'{n_rounds}_R'][f'round {r+1}'])
                        _pop_vec_3 = _get_pop_vector(Shots_qutrit_3[q][f'{n_rounds}_R'][f'round {r+1}'])
                        Population_0[q][f'{n_rounds}_R'][f'round {r+1}'] = np.dot(_pop_vec_0, M_inv)
                        Population_1[q][f'{n_rounds}_R'][f'round {r+1}'] = np.dot(_pop_vec_1, M_inv)
                        Population_2[q][f'{n_rounds}_R'][f'round {r+1}'] = np.dot(_pop_vec_2, M_inv)
                        Population_3[q][f'{n_rounds}_R'][f'round {r+1}'] = np.dot(_pop_vec_3, M_inv)
                Population_f_0[q] = np.array([Population_0[q][f'{Rounds[-1]}_R'][k][2] for k in Population_0[q][f'{Rounds[-1]}_R'].keys()])
                Population_f_1[q] = np.array([Population_1[q][f'{Rounds[-1]}_R'][k][2] for k in Population_1[q][f'{Rounds[-1]}_R'].keys()])
                Population_f_2[q] = np.array([Population_2[q][f'{Rounds[-1]}_R'][k][2] for k in Population_2[q][f'{Rounds[-1]}_R'].keys()])
                Population_f_3[q] = np.array([Population_3[q][f'{Rounds[-1]}_R'][k][2] for k in Population_3[q][f'{Rounds[-1]}_R'].keys()])
                Population_h_0[q] = np.array([Population_0[q][f'{Rounds[-1]}_R'][k][3] for k in Population_0[q][f'{Rounds[-1]}_R'].keys()])
                Population_h_1[q] = np.array([Population_1[q][f'{Rounds[-1]}_R'][k][3] for k in Population_1[q][f'{Rounds[-1]}_R'].keys()])
                Population_h_2[q] = np.array([Population_2[q][f'{Rounds[-1]}_R'][k][3] for k in Population_2[q][f'{Rounds[-1]}_R'].keys()])
                Population_h_3[q] = np.array([Population_3[q][f'{Rounds[-1]}_R'][k][3] for k in Population_3[q][f'{Rounds[-1]}_R'].keys()])
            else:
                # For the data qubit we'll only calculate  
                # leakage in the last measurement round.
                for n_rounds in Rounds:
                    _pop_vec_0 = _get_pop_vector(Shots_qutrit_0[q][f'{n_rounds}_R'][f'round {n_rounds}'])
                    _pop_vec_1 = _get_pop_vector(Shots_qutrit_1[q][f'{n_rounds}_R'][f'round {n_rounds}'])
                    _pop_vec_2 = _get_pop_vector(Shots_qutrit_2[q][f'{n_rounds}_R'][f'round {n_rounds}'])
                    _pop_vec_3 = _get_pop_vector(Shots_qutrit_3[q][f'{n_rounds}_R'][f'round {n_rounds}'])
                    Population_0[q][f'{n_rounds}_R'] = np.dot(_pop_vec_0, M_inv)
                    Population_1[q][f'{n_rounds}_R'] = np.dot(_pop_vec_1, M_inv)
                    Population_2[q][f'{n_rounds}_R'] = np.dot(_pop_vec_2, M_inv)
                    Population_3[q][f'{n_rounds}_R'] = np.dot(_pop_vec_3, M_inv)
                Population_f_0[q] = np.array([Population_0[q][k][2] for k in Population_0[q].keys()])
                Population_f_1[q] = np.array([Population_1[q][k][2] for k in Population_1[q].keys()])
                Population_f_2[q] = np.array([Population_2[q][k][2] for k in Population_2[q].keys()])
                Population_f_3[q] = np.array([Population_3[q][k][2] for k in Population_3[q].keys()])
                Population_h_0[q] = np.array([Population_0[q][k][3] for k in Population_0[q].keys()])
                Population_h_1[q] = np.array([Population_1[q][k][3] for k in Population_1[q].keys()])
                Population_h_2[q] = np.array([Population_2[q][k][3] for k in Population_2[q].keys()])
                Population_h_3[q] = np.array([Population_3[q][k][3] for k in Population_3[q].keys()])
        self.proc_data_dict['Population_0'] = Population_0
        self.proc_data_dict['Population_1'] = Population_1
        self.proc_data_dict['Population_2'] = Population_2
        self.proc_data_dict['Population_3'] = Population_3
        self.proc_data_dict['Population_f_0'] = Population_f_0
        self.proc_data_dict['Population_f_1'] = Population_f_1
        self.proc_data_dict['Population_f_2'] = Population_f_2
        self.proc_data_dict['Population_f_3'] = Population_f_3
        self.proc_data_dict['Population_h_0'] = Population_h_0
        self.proc_data_dict['Population_h_1'] = Population_h_1
        self.proc_data_dict['Population_h_2'] = Population_h_2
        self.proc_data_dict['Population_h_3'] = Population_h_3
        ###########################
        # Calculate defect rate
        ###########################
        deffect_rate_0 = {}
        deffect_rate_1 = {}
        deffect_rate_2 = {}
        deffect_rate_3 = {}
        for n_rounds in Rounds:
            deffect_rate_0[f'{n_rounds}_R'] = _calculate_deffect_rate(Shots_qubit_0[self.qubit][f'{n_rounds}_R'], n_rounds, with_reset=self.with_reset)
            deffect_rate_1[f'{n_rounds}_R'] = _calculate_deffect_rate(Shots_qubit_1[self.qubit][f'{n_rounds}_R'], n_rounds, with_reset=self.with_reset)
            deffect_rate_2[f'{n_rounds}_R'] = _calculate_deffect_rate(Shots_qubit_2[self.qubit][f'{n_rounds}_R'], n_rounds, with_reset=self.with_reset)
            deffect_rate_3[f'{n_rounds}_R'] = _calculate_deffect_rate(Shots_qubit_3[self.qubit][f'{n_rounds}_R'], n_rounds, with_reset=self.with_reset)
        self.qoi = {}
        self.qoi['deffect_rate_normal'] = deffect_rate_0
        self.qoi['deffect_rate_LRU_data'] = deffect_rate_1
        self.qoi['deffect_rate_LRU_ancilla'] = deffect_rate_2
        self.qoi['deffect_rate_LRU_data_ancilla'] = deffect_rate_3
        self.qoi['Population_normal'] = Population_f_0
        self.qoi['Population_LRU_data'] = Population_f_1
        self.qoi['Population_LRU_ancilla'] = Population_f_2
        self.qoi['Population_LRU_data_ancilla'] = Population_f_3
        self.qoi['Population_normal_h'] = Population_h_0
        self.qoi['Population_LRU_data_h'] = Population_h_1
        self.qoi['Population_LRU_ancilla_h'] = Population_h_2
        self.qoi['Population_LRU_data_ancilla_h'] = Population_h_3

    def prepare_plots(self):
        self.axs_dict = {}

        for qubit in self.Qubits:
            if 'D' in qubit:
                fig = plt.figure(figsize=(10,5), dpi=100)
                axs = [fig.add_subplot(121),
                       fig.add_subplot(322),
                       fig.add_subplot(324),
                       fig.add_subplot(326)]
                # fig.patch.set_alpha(0)
                self.axs_dict[f'IQ_readout_histogram_{qubit}'] = axs[0]
                self.figs[f'IQ_readout_histogram_{qubit}'] = fig
                self.plot_dicts[f'IQ_readout_histogram_{qubit}'] = {
                    'plotfn': ssro_IQ_projection_plotfn_2,
                    'ax_id': f'IQ_readout_histogram_{qubit}',
                    'shots_0': self.proc_data_dict[qubit]['Shots_0'],
                    'shots_1': self.proc_data_dict[qubit]['Shots_1'],
                    'shots_2': self.proc_data_dict[qubit]['Shots_2'],
                    'projection_01': self.proc_data_dict[qubit]['projection_01'],
                    'projection_12': self.proc_data_dict[qubit]['projection_12'],
                    'projection_02': self.proc_data_dict[qubit]['projection_02'],
                    'classifier': self.proc_data_dict[qubit]['classifier'],
                    'dec_bounds': self.proc_data_dict[qubit]['dec_bounds'],
                    'Fid_dict': self.proc_data_dict[qubit]['Fid_dict'],
                    'qubit': qubit,
                    'timestamp': self.timestamp
                }
            else:
                fig = plt.figure(figsize=(10,5), dpi=100)
                axs = [fig.add_subplot(121),
                       fig.add_subplot(422),
                       fig.add_subplot(424),
                       fig.add_subplot(426),
                       fig.add_subplot(428)]
                # fig.patch.set_alpha(0)
                self.axs_dict[f'IQ_readout_histogram_{qubit}'] = axs[0]
                self.figs[f'IQ_readout_histogram_{qubit}'] = fig
                self.plot_dicts[f'IQ_readout_histogram_{qubit}'] = {
                    'plotfn': ssro_IQ_projection_plotfn,
                    'ax_id': f'IQ_readout_histogram_{qubit}',
                    'shots_0': self.proc_data_dict[qubit]['Shots_0'],
                    'shots_1': self.proc_data_dict[qubit]['Shots_1'],
                    'shots_2': self.proc_data_dict[qubit]['Shots_2'],
                    'shots_3': self.proc_data_dict[qubit]['Shots_3'],
                    'projection_01': self.proc_data_dict[qubit]['projection_01'],
                    'projection_12': self.proc_data_dict[qubit]['projection_12'],
                    'projection_03': self.proc_data_dict[qubit]['projection_03'],
                    'projection_23': self.proc_data_dict[qubit]['projection_23'],
                    'classifier': self.proc_data_dict[qubit]['classifier'],
                    'dec_bounds': self.proc_data_dict[qubit]['dec_bounds'],
                    'Fid_dict': self.proc_data_dict[qubit]['Fid_dict'],
                    'qubit': qubit,
                    'timestamp': self.timestamp
                }

        fig = plt.figure(figsize=(11, 3))
        gs = fig.add_gridspec(1, 5)
        axs = []
        axs.append(fig.add_subplot(gs[0, 0:2]))
        axs.append(fig.add_subplot(gs[0, 2:3]))
        axs.append(fig.add_subplot(gs[0, 3:4]))
        axs.append(fig.add_subplot(gs[0, 4:5]))
        self.axs_dict['Deffect_rate_plot'] = axs[0]
        self.figs['Deffect_rate_plot'] = fig
        self.plot_dicts['Deffect_rate_plot'] = {
            'plotfn': deffect_rate_plotfn2,
            'ax_id': 'Deffect_rate_plot',
            'Rounds': self.Rounds,
            'deffect_rate_0': self.qoi['deffect_rate_normal'][f'{self.Rounds[-1]}_R'],
            'deffect_rate_1': self.qoi['deffect_rate_LRU_data'][f'{self.Rounds[-1]}_R'], 
            'deffect_rate_2': self.qoi['deffect_rate_LRU_ancilla'][f'{self.Rounds[-1]}_R'],
            'deffect_rate_3': self.qoi['deffect_rate_LRU_data_ancilla'][f'{self.Rounds[-1]}_R'],
            'p_0': self.qoi['Population_normal'], 
            'p_1': self.qoi['Population_LRU_data'], 
            'p_2': self.qoi['Population_LRU_ancilla'], 
            'p_3': self.qoi['Population_LRU_data_ancilla'],
            'p_0_h': self.qoi['Population_normal_h'], 
            'p_1_h': self.qoi['Population_LRU_data_h'], 
            'p_2_h': self.qoi['Population_LRU_ancilla_h'], 
            'p_3_h': self.qoi['Population_LRU_data_ancilla_h'],  
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

def deffect_rate_plotfn2(
    Rounds,
    deffect_rate_0,
    deffect_rate_1,
    deffect_rate_2,
    deffect_rate_3,
    p_0, 
    p_1, 
    p_2, 
    p_3, 
    p_0_h, 
    p_1_h, 
    p_2_h, 
    p_3_h,
    timestamp,
    qubit, 
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    n_rounds = Rounds[-1]
    
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_0[1:], 'C0-', label='Normal')
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_1[1:], 'C1-', label='LRU data')
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_2[1:], 'C2-', label='LRU ancilla')
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_3[1:], 'C3-', label='LRU data ancilla')
    axs[0].grid(ls='--')
    axs[0].set_ylabel('error probability')
    axs[0].set_xlabel('rounds')
    axs[0].set_title('Deffect rate')

    axs[1].plot((np.arange(n_rounds)+1), p_0[qubit]*100, 'C0-', label='Normal')
    axs[1].plot((np.arange(n_rounds)+1), p_1[qubit]*100, 'C1-', label='LRU data')
    axs[1].plot((np.arange(n_rounds)+1), p_2[qubit]*100, 'C2-', label='LRU ancilla')
    axs[1].plot((np.arange(n_rounds)+1), p_3[qubit]*100, 'C3-', label='LRU data ancilla')
    axs[1].plot((np.arange(n_rounds)+1), p_0_h[qubit]*100, 'C0--',alpha=0.4)
    axs[1].plot((np.arange(n_rounds)+1), p_1_h[qubit]*100, 'C1--',alpha=0.4)
    axs[1].plot((np.arange(n_rounds)+1), p_2_h[qubit]*100, 'C2--',alpha=0.4)
    axs[1].plot((np.arange(n_rounds)+1), p_3_h[qubit]*100, 'C3--',alpha=0.4)
    axs[1].set_ylabel(r'$|L_{1}\rangle$ population (%)')
    axs[1].set_xlabel('rounds')
    axs[1].set_title(qubit)
    axs[1].grid(ls='--')

    Data_qubits = [name for name in p_0.keys()]
    Data_qubits.remove(qubit)
    for i, q in enumerate(Data_qubits):
        axs[2+i].plot(Rounds, (p_0[q]+p_0_h[q])*100, 'C0.-', label='Normal')
        axs[2+i].plot(Rounds, (p_1[q]+p_1_h[q])*100, 'C1.-', label='LRU data')
        axs[2+i].plot(Rounds, (p_2[q]+p_2_h[q])*100, 'C2.-', label='LRU ancilla')
        axs[2+i].plot(Rounds, (p_3[q]+p_3_h[q])*100, 'C3.-', label='LRU data ancilla')
        # axs[2+i].plot(Rounds, p_0_h[q]*100, 'C0--',alpha=0.4)
        # axs[2+i].plot(Rounds, p_1_h[q]*100, 'C1--',alpha=0.4)
        # axs[2+i].plot(Rounds, p_2_h[q]*100, 'C2--',alpha=0.4)
        # axs[2+i].plot(Rounds, p_3_h[q]*100, 'C3--',alpha=0.4)
        axs[2+i].set_ylabel(r'Leakage population (%)')
        axs[2+i].set_xlabel('rounds')
        axs[2+i].set_title(q)
        axs[2+i].grid(ls='--')


    axs[3].legend(frameon=False, bbox_to_anchor = (1.01, 1))

    fig.suptitle(f'{timestamp}\n{qubit} repeated stabilizer experiment')
    fig.tight_layout()

def ssro_IQ_projection_plotfn_2(
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
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
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
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1, shots_2))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    Lim_points = {}
    for bound in ['01', '12', '02']:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot 0 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['02']]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['02'], Lim_points['12']]
    _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    for bound in ['01', '12', '02']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
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


class Repeated_stabilizer_measurements(ba.BaseDataAnalysis):
    def __init__(self,
                 qubit: str,
                 Rounds: list,
                 heralded_init: bool = False,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, 
                 extract_only: bool = False,
                 with_reset: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.qubit = qubit
        self.Rounds = Rounds
        self.with_reset = with_reset
        self.heralded_init = heralded_init

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
        ######################################
        # Sort shots and assign them
        ######################################
        Rounds = self.Rounds
        _total_rounds = np.sum(Rounds)
        if self.heralded_init:
            _total_rounds += len(Rounds)
        _cycle = _total_rounds*3 + 3
        if self.heralded_init:
            _cycle += 3
        # Get qubit names in channel order
        names = [ name.decode().split(' ')[-2] for name in self.raw_data_dict['value_names'] ]
        self.Qubits = names[::2]
        # Dictionary that will store raw shots
        # so that they can later be sorted.
        raw_shots = {q: {} for q in self.Qubits}
        for q_idx, qubit in enumerate(self.Qubits):
            self.proc_data_dict[qubit] = {}
            _ch_I, _ch_Q = 2*q_idx+1, 2*q_idx+2
            _raw_shots = self.raw_data_dict['data'][:,[_ch_I, _ch_Q]]
            if self.heralded_init:
                _shots_0 = _raw_shots[3*_total_rounds+1::_cycle]
                _shots_1 = _raw_shots[3*_total_rounds+3::_cycle]
                _shots_2 = _raw_shots[3*_total_rounds+5::_cycle]
            else:
                _shots_0 = _raw_shots[3*_total_rounds+0::_cycle]
                _shots_1 = _raw_shots[3*_total_rounds+1::_cycle]
                _shots_2 = _raw_shots[3*_total_rounds+2::_cycle]
            # Rotate data
            center_0 = np.array([np.mean(_shots_0[:,0]), np.mean(_shots_0[:,1])])
            center_1 = np.array([np.mean(_shots_1[:,0]), np.mean(_shots_1[:,1])])
            center_2 = np.array([np.mean(_shots_2[:,0]), np.mean(_shots_2[:,1])])
            raw_shots[qubit] = _rotate_and_center_data(_raw_shots[:,0], _raw_shots[:,1], center_0, center_1)
            if self.heralded_init:
                Shots_0 = raw_shots[qubit][3*_total_rounds+1::_cycle]
                Shots_1 = raw_shots[qubit][3*_total_rounds+3::_cycle]
                Shots_2 = raw_shots[qubit][3*_total_rounds+5::_cycle]
            else:
                Shots_0 = raw_shots[qubit][3*_total_rounds+0::_cycle]
                Shots_1 = raw_shots[qubit][3*_total_rounds+1::_cycle]
                Shots_2 = raw_shots[qubit][3*_total_rounds+2::_cycle]
            self.proc_data_dict[qubit]['Shots_0'] = Shots_0
            self.proc_data_dict[qubit]['Shots_1'] = Shots_1
            self.proc_data_dict[qubit]['Shots_2'] = Shots_2
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
            # # Make it a 4x4 matrix
            # M = np.append(M, [[0,0,0]], 0)
            # M = np.append(M, [[0],[0],[0],[1]], 1)
            self.proc_data_dict[qubit]['classifier'] = clf
            self.proc_data_dict[qubit]['Fid_dict'] = Fid_dict
            self.proc_data_dict[qubit]['Assignment_matrix'] = M
            self.proc_data_dict[qubit]['dec_bounds'] = dec_bounds

            ##################
            # Post selection
            ##################
            if self.heralded_init:
                _ps_shots_0 = raw_shots[qubit][3*_total_rounds+0::_cycle]
                _ps_shots_1 = raw_shots[qubit][3*_total_rounds+2::_cycle]
                _ps_shots_2 = raw_shots[qubit][3*_total_rounds+4::_cycle]

                def _post_select(shots, ps_shots):
                    _ps_shots = clf.predict(ps_shots)
                    _mask = np.array([1 if s == 0 else np.nan for s in _ps_shots])
                    # print(np.nansum(_mask)/ len(_mask))
                    shots = shots[~np.isnan(_mask)]
                    return shots

                Shots_0 = _post_select(Shots_0, _ps_shots_0)
                Shots_1 = _post_select(Shots_1, _ps_shots_1)
                Shots_2 = _post_select(Shots_2, _ps_shots_2)
                self.proc_data_dict[qubit]['Shots_0'] = Shots_0
                self.proc_data_dict[qubit]['Shots_1'] = Shots_1
                self.proc_data_dict[qubit]['Shots_2'] = Shots_2
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
                # # Make it a 4x4 matrix
                # M = np.append(M, [[0,0,0]], 0)
                # M = np.append(M, [[0],[0],[0],[1]], 1)
                self.proc_data_dict[qubit]['classifier'] = clf
                self.proc_data_dict[qubit]['Fid_dict'] = Fid_dict
                self.proc_data_dict[qubit]['Assignment_matrix'] = M
                self.proc_data_dict[qubit]['dec_bounds'] = dec_bounds
            #########################################
            # Project data along axis perpendicular
            # to the decision boundaries.
            #########################################
            ############################
            # Projection along 01 axis.
            ############################
            # Rotate shots over 01 axis
            shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['01'],phi=np.pi/2)
            shots_1 = _rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'],dec_bounds['01'],phi=np.pi/2)
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
            self.proc_data_dict[qubit]['projection_01'] = {}
            self.proc_data_dict[qubit]['projection_01']['h0'] = h0
            self.proc_data_dict[qubit]['projection_01']['h1'] = h1
            self.proc_data_dict[qubit]['projection_01']['bin_centers'] = bin_centers
            self.proc_data_dict[qubit]['projection_01']['popt0'] = popt0
            self.proc_data_dict[qubit]['projection_01']['popt1'] = popt1
            self.proc_data_dict[qubit]['projection_01']['SNR'] = params_01['SNR']
            self.proc_data_dict[qubit]['projection_01']['Fid'] = Fid_01
            self.proc_data_dict[qubit]['projection_01']['threshold'] = threshold_01
            ############################
            # Projection along 12 axis.
            ############################
            # Rotate shots over 12 axis
            shots_1 = _rotate_and_center_data(Shots_1[:,0],Shots_1[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
            shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'], dec_bounds['12'], phi=np.pi/2)
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
            self.proc_data_dict[qubit]['projection_12'] = {}
            self.proc_data_dict[qubit]['projection_12']['h1'] = h1
            self.proc_data_dict[qubit]['projection_12']['h2'] = h2
            self.proc_data_dict[qubit]['projection_12']['bin_centers'] = bin_centers
            self.proc_data_dict[qubit]['projection_12']['popt1'] = popt1
            self.proc_data_dict[qubit]['projection_12']['popt2'] = popt2
            self.proc_data_dict[qubit]['projection_12']['SNR'] = params_12['SNR']
            self.proc_data_dict[qubit]['projection_12']['Fid'] = Fid_12
            self.proc_data_dict[qubit]['projection_12']['threshold'] = threshold_12
            
            ############################
            # Projection along 02 axis.
            ############################
            # Rotate shots over 02 axis
            shots_0 = _rotate_and_center_data(Shots_0[:,0],Shots_0[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
            shots_2 = _rotate_and_center_data(Shots_2[:,0],Shots_2[:,1],dec_bounds['mean'],dec_bounds['02'], phi=np.pi/2)
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
            self.proc_data_dict[qubit]['projection_02'] = {}
            self.proc_data_dict[qubit]['projection_02']['h0'] = h0
            self.proc_data_dict[qubit]['projection_02']['h2'] = h2
            self.proc_data_dict[qubit]['projection_02']['bin_centers'] = bin_centers
            self.proc_data_dict[qubit]['projection_02']['popt0'] = popt0
            self.proc_data_dict[qubit]['projection_02']['popt2'] = popt2
            self.proc_data_dict[qubit]['projection_02']['SNR'] = params_02['SNR']
            self.proc_data_dict[qubit]['projection_02']['Fid'] = Fid_02
            self.proc_data_dict[qubit]['projection_02']['threshold'] = threshold_02

        ########################################
        # Analyze experiment shots, post-select
        # on leakage and calculate deffect rate
        ########################################
        # Sort experimental shots
        # 0-Individual stabilizer experiment
        # 1-Same-type stabilizer experiment
        # 2-Sim stabilizer experiment
        shots_exp_0 = {q: {} for q in self.Qubits}
        Shots_qubit_0 = {q: {} for q in self.Qubits}
        Shots_qutrit_0 = {q: {} for q in self.Qubits}
        shots_exp_1 = {q: {} for q in self.Qubits}
        Shots_qubit_1 = {q: {} for q in self.Qubits}
        Shots_qutrit_1 = {q: {} for q in self.Qubits}
        shots_exp_2 = {q: {} for q in self.Qubits}
        Shots_qubit_2 = {q: {} for q in self.Qubits}
        Shots_qutrit_2 = {q: {} for q in self.Qubits}
        shots_exp_3 = {q: {} for q in self.Qubits}
        for q in self.Qubits:
            # threshold = _get_threshold(Shots_0, Shots_1)
            _zero_lvl = np.mean(self.proc_data_dict[q]['Shots_0'][:,0])
            _one_lvl = np.mean(self.proc_data_dict[q]['Shots_1'][:,1])
            threshold = self.proc_data_dict[q]['projection_01']['threshold']
            _clf = self.proc_data_dict[q]['classifier']
            for r_idx, n_rounds in enumerate(Rounds):
                shots_exp_0[q][f'{n_rounds}_R'] = {}
                Shots_qubit_0[q][f'{n_rounds}_R'] = {}
                Shots_qutrit_0[q][f'{n_rounds}_R'] = {}
                shots_exp_1[q][f'{n_rounds}_R'] = {}
                Shots_qubit_1[q][f'{n_rounds}_R'] = {}
                Shots_qutrit_1[q][f'{n_rounds}_R'] = {}
                shots_exp_2[q][f'{n_rounds}_R'] = {}
                Shots_qubit_2[q][f'{n_rounds}_R'] = {}
                Shots_qutrit_2[q][f'{n_rounds}_R'] = {}
                # counter for number of shots in previous rounds
                _aux = int(3*np.sum(Rounds[:r_idx]))
                if self.heralded_init:
                    _aux = int(3*(np.sum(Rounds[:r_idx])+len(Rounds[:r_idx])))
                for r in range(n_rounds):
                    # Note we are using the rotated shots already
                    shots_exp_0[q][f'{n_rounds}_R'][f'round {r+1}'] = \
                        raw_shots[q][r+0*(n_rounds+self.heralded_init)+self.heralded_init+_aux::_cycle]
                    shots_exp_1[q][f'{n_rounds}_R'][f'round {r+1}'] = \
                        raw_shots[q][r+1*(n_rounds+self.heralded_init)+self.heralded_init+_aux::_cycle]
                    shots_exp_2[q][f'{n_rounds}_R'][f'round {r+1}'] = \
                        raw_shots[q][r+2*(n_rounds+self.heralded_init)+self.heralded_init+_aux::_cycle]                    # Perform Qubit assignment
                    if _zero_lvl < threshold: # zero level is left of threshold
                        Shots_qubit_0[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_0[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_1[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_1[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_2[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s<threshold else 1 for s in shots_exp_2[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                    else: # zero level is right of threshold
                        Shots_qubit_0[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_0[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_1[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_1[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                        Shots_qubit_2[q][f'{n_rounds}_R'][f'round {r+1}'] = np.array([0 if s>threshold else 1 for s in shots_exp_2[q][f'{n_rounds}_R'][f'round {r+1}'][:,0]])
                    # Perform Qutrit assignment
                    Shots_qutrit_0[q][f'{n_rounds}_R'][f'round {r+1}'] = _clf.predict(shots_exp_0[q][f'{n_rounds}_R'][f'round {r+1}'])
                    Shots_qutrit_1[q][f'{n_rounds}_R'][f'round {r+1}'] = _clf.predict(shots_exp_1[q][f'{n_rounds}_R'][f'round {r+1}'])
                    Shots_qutrit_2[q][f'{n_rounds}_R'][f'round {r+1}'] = _clf.predict(shots_exp_2[q][f'{n_rounds}_R'][f'round {r+1}'])
                # Post selection
                if self.heralded_init:
                    # Sort heralding shots
                    shots_exp_0[q][f'{n_rounds}_R']['ps'] = \
                        raw_shots[q][0*(n_rounds+self.heralded_init)+_aux::_cycle]
                    shots_exp_1[q][f'{n_rounds}_R']['ps'] = \
                        raw_shots[q][1*(n_rounds+self.heralded_init)+_aux::_cycle]
                    shots_exp_2[q][f'{n_rounds}_R']['ps'] = \
                        raw_shots[q][2*(n_rounds+self.heralded_init)+_aux::_cycle]
                    # Classify heralding shots
                    Shots_qutrit_0[q][f'{n_rounds}_R']['round 0'] = _clf.predict(shots_exp_0[q][f'{n_rounds}_R']['ps'])
                    Shots_qutrit_1[q][f'{n_rounds}_R']['round 0'] = _clf.predict(shots_exp_1[q][f'{n_rounds}_R']['ps'])
                    Shots_qutrit_2[q][f'{n_rounds}_R']['round 0'] = _clf.predict(shots_exp_2[q][f'{n_rounds}_R']['ps'])
                    # Compute post-selection mask
                    Shots_qutrit_0[q][f'{n_rounds}_R']['ps'] = np.array([ 1 if s == 0 else np.nan for s in Shots_qutrit_0[q][f'{n_rounds}_R']['round 0'] ])
                    Shots_qutrit_1[q][f'{n_rounds}_R']['ps'] = np.array([ 1 if s == 0 else np.nan for s in Shots_qutrit_1[q][f'{n_rounds}_R']['round 0'] ])
                    Shots_qutrit_2[q][f'{n_rounds}_R']['ps'] = np.array([ 1 if s == 0 else np.nan for s in Shots_qutrit_2[q][f'{n_rounds}_R']['round 0'] ])
        # Perform post-selection
        if self.heralded_init:
            for R in Rounds:
                _n_shots = len(Shots_qutrit_0[q][f'{R}_R']['ps']) 
                _mask_0 = np.ones(_n_shots)
                _mask_1 = np.ones(_n_shots)
                _mask_2 = np.ones(_n_shots)
                for q in self.Qubits:
                    _mask_0 *= Shots_qutrit_0[q][f'{R}_R']['ps']
                    _mask_1 *= Shots_qutrit_1[q][f'{R}_R']['ps']
                    _mask_2 *= Shots_qutrit_2[q][f'{R}_R']['ps']
                print(f'{R}_R Percentage of post-selected shots 0: {np.nansum(_mask_0)/len(_mask_0)*100:.2f}%')
                print(f'{R}_R Percentage of post-selected shots 1: {np.nansum(_mask_1)/len(_mask_1)*100:.2f}%')
                print(f'{R}_R Percentage of post-selected shots 2: {np.nansum(_mask_2)/len(_mask_2)*100:.2f}%')
                for q in self.Qubits:
                    for r in range(R):
                        # Remove marked shots in qubit shots
                        Shots_qubit_0[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qubit_0[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_0)]
                        Shots_qubit_1[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qubit_1[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_1)]
                        Shots_qubit_2[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qubit_2[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_2)]
                        # Remove marked shots in qutrit shots
                        Shots_qutrit_0[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qutrit_0[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_0)]
                        Shots_qutrit_1[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qutrit_1[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_1)]
                        Shots_qutrit_2[q][f'{R}_R'][f'round {r+1}'] = \
                            Shots_qutrit_2[q][f'{R}_R'][f'round {r+1}'][~np.isnan(_mask_2)]
        self.proc_data_dict['Shots_qubit_0'] = Shots_qubit_0
        self.proc_data_dict['Shots_qubit_1'] = Shots_qubit_1
        self.proc_data_dict['Shots_qubit_2'] = Shots_qubit_2
        self.proc_data_dict['Shots_qutrit_0'] = Shots_qutrit_0
        self.proc_data_dict['Shots_qutrit_1'] = Shots_qutrit_1
        self.proc_data_dict['Shots_qutrit_2'] = Shots_qutrit_2
        # Calculate leakage
        ####################
        Population_0 = {q:{} for q in self.Qubits}
        Population_1 = {q:{} for q in self.Qubits}
        Population_2 = {q:{} for q in self.Qubits}
        Population_f_0 = {q:{} for q in self.Qubits}
        Population_f_1 = {q:{} for q in self.Qubits}
        Population_f_2 = {q:{} for q in self.Qubits}
        def _get_pop_vector(Shots):
            p0 = np.mean(Shots==0)
            p1 = np.mean(Shots==1)
            p2 = np.mean(Shots==2)
            return np.array([p0, p1, p2])
        for q in self.Qubits:
            M_inv = np.linalg.inv(self.proc_data_dict[q]['Assignment_matrix'])
            if q == self.qubit:
                # For the ancilla qubit we'll calculate  
                # leakage in every measurement round.
                for n_rounds in Rounds:
                    Population_0[q][f'{n_rounds}_R'] = {}
                    Population_1[q][f'{n_rounds}_R'] = {}
                    Population_2[q][f'{n_rounds}_R'] = {}
                    for r in range(n_rounds):
                        _pop_vec_0 = _get_pop_vector(Shots_qutrit_0[q][f'{n_rounds}_R'][f'round {r+1}'])
                        _pop_vec_1 = _get_pop_vector(Shots_qutrit_1[q][f'{n_rounds}_R'][f'round {r+1}'])
                        _pop_vec_2 = _get_pop_vector(Shots_qutrit_2[q][f'{n_rounds}_R'][f'round {r+1}'])
                        Population_0[q][f'{n_rounds}_R'][f'round {r+1}'] = np.dot(_pop_vec_0, M_inv)
                        Population_1[q][f'{n_rounds}_R'][f'round {r+1}'] = np.dot(_pop_vec_1, M_inv)
                        Population_2[q][f'{n_rounds}_R'][f'round {r+1}'] = np.dot(_pop_vec_2, M_inv)
                Population_f_0[q] = np.array([Population_0[q][f'{Rounds[-1]}_R'][k][2] for k in Population_0[q][f'{Rounds[-1]}_R'].keys()])
                Population_f_1[q] = np.array([Population_1[q][f'{Rounds[-1]}_R'][k][2] for k in Population_1[q][f'{Rounds[-1]}_R'].keys()])
                Population_f_2[q] = np.array([Population_2[q][f'{Rounds[-1]}_R'][k][2] for k in Population_2[q][f'{Rounds[-1]}_R'].keys()])
            else:
                # For the data qubit we'll only calculate  
                # leakage in the last measurement round.
                for n_rounds in Rounds:
                    _pop_vec_0 = _get_pop_vector(Shots_qutrit_0[q][f'{n_rounds}_R'][f'round {n_rounds}'])
                    _pop_vec_1 = _get_pop_vector(Shots_qutrit_1[q][f'{n_rounds}_R'][f'round {n_rounds}'])
                    _pop_vec_2 = _get_pop_vector(Shots_qutrit_2[q][f'{n_rounds}_R'][f'round {n_rounds}'])
                    Population_0[q][f'{n_rounds}_R'] = np.dot(_pop_vec_0, M_inv)
                    Population_1[q][f'{n_rounds}_R'] = np.dot(_pop_vec_1, M_inv)
                    Population_2[q][f'{n_rounds}_R'] = np.dot(_pop_vec_2, M_inv)
                Population_f_0[q] = np.array([Population_0[q][k][2] for k in Population_0[q].keys()])
                Population_f_1[q] = np.array([Population_1[q][k][2] for k in Population_1[q].keys()])
                Population_f_2[q] = np.array([Population_2[q][k][2] for k in Population_2[q].keys()])
        self.proc_data_dict['Population_0'] = Population_0
        self.proc_data_dict['Population_1'] = Population_1
        self.proc_data_dict['Population_2'] = Population_2
        self.proc_data_dict['Population_f_0'] = Population_f_0
        self.proc_data_dict['Population_f_1'] = Population_f_1
        self.proc_data_dict['Population_f_2'] = Population_f_2
        ###########################
        # Calculate defect rate
        ###########################
        deffect_rate_0 = {}
        deffect_rate_1 = {}
        deffect_rate_2 = {}
        for n_rounds in Rounds:
            deffect_rate_0[f'{n_rounds}_R'] = _calculate_deffect_rate(Shots_qubit_0[self.qubit][f'{n_rounds}_R'], n_rounds, with_reset=self.with_reset)
            deffect_rate_1[f'{n_rounds}_R'] = _calculate_deffect_rate(Shots_qubit_1[self.qubit][f'{n_rounds}_R'], n_rounds, with_reset=self.with_reset)
            deffect_rate_2[f'{n_rounds}_R'] = _calculate_deffect_rate(Shots_qubit_2[self.qubit][f'{n_rounds}_R'], n_rounds, with_reset=self.with_reset)
        self.qoi = {}
        self.qoi['deffect_rate_normal'] = deffect_rate_0
        self.qoi['deffect_rate_LRU_data'] = deffect_rate_1
        self.qoi['deffect_rate_LRU_ancilla'] = deffect_rate_2
        self.qoi['Population_normal'] = Population_f_0
        self.qoi['Population_LRU_data'] = Population_f_1
        self.qoi['Population_LRU_ancilla'] = Population_f_2

    def prepare_plots(self):
        self.axs_dict = {}

        for qubit in self.Qubits:
            fig = plt.figure(figsize=(10,5), dpi=100)
            axs = [fig.add_subplot(121),
                   fig.add_subplot(322),
                   fig.add_subplot(324),
                   fig.add_subplot(326)]
            # fig.patch.set_alpha(0)
            self.axs_dict[f'IQ_readout_histogram_{qubit}'] = axs[0]
            self.figs[f'IQ_readout_histogram_{qubit}'] = fig
            self.plot_dicts[f'IQ_readout_histogram_{qubit}'] = {
                'plotfn': ssro_IQ_projection_plotfn_3,
                'ax_id': f'IQ_readout_histogram_{qubit}',
                'shots_0': self.proc_data_dict[qubit]['Shots_0'],
                'shots_1': self.proc_data_dict[qubit]['Shots_1'],
                'shots_2': self.proc_data_dict[qubit]['Shots_2'],
                'projection_01': self.proc_data_dict[qubit]['projection_01'],
                'projection_12': self.proc_data_dict[qubit]['projection_12'],
                'projection_02': self.proc_data_dict[qubit]['projection_02'],
                'classifier': self.proc_data_dict[qubit]['classifier'],
                'dec_bounds': self.proc_data_dict[qubit]['dec_bounds'],
                'Fid_dict': self.proc_data_dict[qubit]['Fid_dict'],
                'qubit': qubit,
                'timestamp': self.timestamp
            }
            

        fig = plt.figure(figsize=(11, 3))
        gs = fig.add_gridspec(1, 5)
        axs = []
        axs.append(fig.add_subplot(gs[0, 0:2]))
        axs.append(fig.add_subplot(gs[0, 2:3]))
        axs.append(fig.add_subplot(gs[0, 3:4]))
        axs.append(fig.add_subplot(gs[0, 4:5]))
        self.axs_dict['Deffect_rate_plot'] = axs[0]
        self.figs['Deffect_rate_plot'] = fig
        self.plot_dicts['Deffect_rate_plot'] = {
            'plotfn': deffect_rate_plotfn3,
            'ax_id': 'Deffect_rate_plot',
            'Rounds': self.Rounds,
            'deffect_rate_0': self.qoi['deffect_rate_normal'][f'{self.Rounds[-1]}_R'],
            'deffect_rate_1': self.qoi['deffect_rate_LRU_data'][f'{self.Rounds[-1]}_R'], 
            'deffect_rate_2': self.qoi['deffect_rate_LRU_ancilla'][f'{self.Rounds[-1]}_R'],
            'p_0': self.qoi['Population_normal'], 
            'p_1': self.qoi['Population_LRU_data'], 
            'p_2': self.qoi['Population_LRU_ancilla'],
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

def deffect_rate_plotfn3(
    Rounds,
    deffect_rate_0,
    deffect_rate_1,
    deffect_rate_2,
    p_0, 
    p_1, 
    p_2, 
    timestamp,
    qubit, 
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()

    n_rounds = Rounds[-1]
    
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_0[1:], 'C0-', label='Individ. stab')
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_1[1:], 'C1-', label='Same_type stab')
    axs[0].plot((np.arange(n_rounds)+1)[1:], deffect_rate_2[1:], 'C2-', label='Sim. stab')
    axs[0].grid(ls='--')
    axs[0].set_ylabel('error probability')
    axs[0].set_xlabel('rounds')
    axs[0].set_title('Deffect rate')

    axs[1].plot((np.arange(n_rounds)+1), p_0[qubit]*100, 'C0-', label='Individ. stab')
    axs[1].plot((np.arange(n_rounds)+1), p_1[qubit]*100, 'C1-', label='Same_type stab')
    axs[1].plot((np.arange(n_rounds)+1), p_2[qubit]*100, 'C2-', label='Sim. stab')
    axs[1].set_ylabel(r'$|L_{1}\rangle$ population (%)')
    axs[1].set_xlabel('rounds')
    axs[1].set_title(qubit)
    axs[1].grid(ls='--')

    Data_qubits = [name for name in p_0.keys()]
    Data_qubits.remove(qubit)
    for i, q in enumerate(Data_qubits):
        axs[2+i].plot(Rounds, (p_0[q])*100, 'C0.-', label='Individ. stab')
        axs[2+i].plot(Rounds, (p_1[q])*100, 'C1.-', label='Same_type stab')
        axs[2+i].plot(Rounds, (p_2[q])*100, 'C2.-', label='Sim. stab')
        axs[2+i].set_ylabel(r'Leakage population (%)')
        axs[2+i].set_xlabel('rounds')
        axs[2+i].set_title(q)
        axs[2+i].grid(ls='--')


    axs[3].legend(frameon=False, bbox_to_anchor = (1.01, 1))

    fig.suptitle(f'{timestamp}\n{qubit} repeated stabilizer experiment')
    fig.tight_layout()

def ssro_IQ_projection_plotfn_3(
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
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
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
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1, shots_2))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    Lim_points = {}
    for bound in ['01', '12', '02']:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot 0 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['02']]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['02'], Lim_points['12']]
    _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    for bound in ['01', '12', '02']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
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