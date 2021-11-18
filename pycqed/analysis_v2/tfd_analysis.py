"""
Analysis for Thermal Field Double state VQE experiment
"""

import os
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
    cmap_to_alpha, cmap_first_to_alpha
import pycqed.measurement.hdf5_data as h5d
from pycqed.analysis import analysis_toolbox as a_tools
import pandas as pd
from scipy import linalg
import cmath as cm


class TFD_Analysis_Pauli_Strings(ba.BaseDataAnalysis):
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 g: float = 1, T=1,
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):
        """
        Analysis for the Thermal Field Double state QAOA experiment.

        Args:
            g (float):
                coupling strength (in theorist units)
            T (float):
                temperature (in theorist units)
        """

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.g = g
        self.T = T
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
        param_spec = {
            'data': ('Experimental Data/Data', 'dset'),
            'combinations':  ('Experimental Data/Experimental Metadata/combinations', 'dset'),
            'value_names': ('Experimental Data', 'attr:value_names')}

        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)

        # For some reason the list is stored a list of length 1 arrays...
        self.raw_data_dict['combinations'] = [
            c[0] for c in self.raw_data_dict['combinations']]

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}
        # combinations = ['X', 'Z', '0000', '1111']
        combinations = self.raw_data_dict['combinations']
        raw_shots = self.raw_data_dict['data'][:, 1:]
        value_names = self.raw_data_dict['value_names']


        binned_data = {}
        for i, ch_name in enumerate(value_names):
            ch_data = raw_shots[:, i]  # select shots per channel
            binned_data[ch_name] = {}
            for j, comb in enumerate(combinations):
                binned_data[ch_name][comb] = np.mean(
                    ch_data[j::len(combinations)])  #get average for shots per combination
        #data per combination is stored with index steps of len(combinations) starting from j.

        # Calculate mean voltages to determine threshold
        mn_voltages = {}
        for i, ch_name in enumerate(value_names):
            ch_data = binned_data[ch_name]  # select per channel
            mn_voltages[ch_name] = {'0': [], '1': []}
            for c in combinations:
                if c == '0000':
                    mn_voltages[ch_name]['0'].append(ch_data[c])
                elif c == '1111':
                    mn_voltages[ch_name]['1'].append(ch_data[c])
            mn_voltages[ch_name]['0'] = np.mean(mn_voltages[ch_name]['0'])
            mn_voltages[ch_name]['1'] = np.mean(mn_voltages[ch_name]['1'])
            mn_voltages[ch_name]['threshold'] = np.mean(
                [mn_voltages[ch_name]['0'], mn_voltages[ch_name]['1']])

        self.proc_data_dict['mn_voltages'] = mn_voltages

        # Digitize data
        digitized_data = np.zeros(raw_shots.shape)
        for i, vn in enumerate(value_names):
            digitized_data[:, i] = np.array(
                raw_shots[:, i] > mn_voltages[vn]['threshold'], dtype=int)
        # Calculating correlations when values are expressed as
        # eigenvalues (+- 1) is easier
        digitized_data_pm = digitized_data
        digitized_data_pm[digitized_data_pm < .5] = -1

        # Bin the Pauli Terms
        pauli_terms = {'ZZII': 0, 'XIII': 0, 'IXII': 0,
                       'IIZZ': 0, 'IIXI': 0, 'IIIX': 0,
                       'ZIZI': 0, 'IZIZ': 0, 'XIXI': 0, 'IXIX': 0}
        x_cnt = 0
        z_cnt = 0
        for i, row in enumerate(digitized_data_pm):
            comb = combinations[i % len(combinations)]
            if comb == 'X' or comb == 'X-IIII':
                x_cnt += 1
                pauli_terms['XIII'] += row[0]
                pauli_terms['IXII'] += row[1]
                pauli_terms['IIXI'] += row[2]
                pauli_terms['IIIX'] += row[3]
                pauli_terms['XIXI'] += row[0]*row[2]
                pauli_terms['IXIX'] += row[1]*row[3]

            elif comb == 'Z' or comb == 'Z-IIII':
                z_cnt += 1
                pauli_terms['ZZII'] += row[0]*row[1]
                pauli_terms['IIZZ'] += row[2]*row[3]
                pauli_terms['ZIZI'] += row[0]*row[2]
                pauli_terms['IZIZ'] += row[1]*row[3]

        # Normalize the pauli terms
        for key, val in pauli_terms.items():
            pauli_terms[key] = val/x_cnt
        self.proc_data_dict['pauli_terms'] = pauli_terms

        self.proc_data_dict['energy_terms'] = calc_tfd_hamiltonian(
            pauli_terms, g=self.g, T=self.T)
        self.proc_data_dict['quantities_of_interest'] = {
            'g': self.g, 'T': self.T,
            **self.proc_data_dict['pauli_terms'],
            **self.proc_data_dict['energy_terms']}

    def prepare_plots(self):
        self.plot_dicts['pauli_operators_Strings'] = {
            'plotfn': plot_pauli_ops,
            'pauli_terms': self.proc_data_dict['pauli_terms'],
            'energy_terms': self.proc_data_dict['energy_terms']
        }


def calc_tfd_hamiltonian(pauli_terms: dict, g: float = 1, T=1):
    """
    Calculate the thermal field double Hamiltonian expectation value.

    Args:
        pauli_terms (dict):
            dictionary containing the expectation values.
            Keys are of the form "XIII", "ZIZI"

    Hamiltonian is given by
        H = H_A + H_B  - T H_AB

    Individual terms H_A:
        H_A = (Z_1^A * Z_2^A) + g (X_1^A* I_2^A) + g(I_1^A * X_2^A)
        <H_A>  = ZZII + g XIII + g IXII


    Individual terms H_B:
        H_B = (Z_1^B * Z_2^B) + g (X_1^B* I_2^B) + g(I_1^B * X_2^B)
        <H_A>  = IIZZ + g IIXI + g IIIX


    Individual terms H_AB:
        H_AB = (Z_1^A * Z_1^B)+(Z_2^A * Z_2^B) + (X_1^A* X_1^B)+(X_2^A * X_2^B)
        <H_AB>  = ZIZI + IZIZ + XIXI + IXIX
    """
    factor=1.0
    H_A = factor*pauli_terms['ZZII'] + g*pauli_terms['XIII'] + g*pauli_terms['IXII']
    H_B = factor*pauli_terms['IIZZ'] + g*pauli_terms['IIXI'] + g*pauli_terms['IIIX']
    H_AB = pauli_terms['ZIZI'] + pauli_terms['IZIZ'] + \
        pauli_terms['XIXI'] + pauli_terms['IXIX']

    if np.isinf(T):
        H = -1*H_AB
    else:
        H = H_A + H_B - (T**factor)*H_AB

    return {'H': H, 'H_A': H_A, 'H_B': H_B, 'H_AB': H_AB}


def plot_pauli_ops(pauli_terms, energy_terms, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()

    labels = pauli_terms.keys()
    for i, label in enumerate(labels):
        if i < 3:
            c = 'r'
        elif i < 6:
            c = 'b'
        else:
            c = 'purple'
        ax.bar(i, pauli_terms[label], color=c, align='center')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)

    ax.text(1, .5, '$H_A=${:.2f}'.format(energy_terms['H_A']))
    ax.text(4, .6, '$H_B=${:.2f}'.format(energy_terms['H_B']))
    ax.text(7, .7, '$H_{AB}=$'+'{:.2f}'.format(energy_terms['H_AB']))

    ax.set_ylabel('Expectation value')
    ax.set_ylim(-1.05, 1.05)
    ax.set_title('Digitized pauli expectation values')


def plot_all_pauli_ops(full_dict, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()

    labels = full_dict.keys()
    for i, label in enumerate(labels):
        if 'ZZII' in label or 'IIZZ' in label or 'XXII' in label or 'IIXX' in label:
            c = 'r'
        elif 'ZIZI' in label or 'IZIZ' in label or 'XIXI' in label or 'IXIX' in label:
            c = 'b'
        else:
            c = 'purple'
        ax.bar(i, full_dict[label], color=c, align='center')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=60)
        ax.text(1, -.5, '$Inter=${:.2f}'.format(np.abs(full_dict['ZIZI'])+np.abs(full_dict['IZIZ'])+
                                                np.abs(full_dict['XIXI'])+np.abs(full_dict['IXIX'])))
        ax.text(15, -.5, '$Intra=${:.2f}'.format(np.abs(full_dict['ZZII'])+np.abs(full_dict['IIZZ'])+
                                                np.abs(full_dict['XXII'])+np.abs(full_dict['IIXX'])))
    ax.set_ylabel('Expectation value')
    ax.set_ylim(-1.05, 1.05)
    ax.set_title('All pauli expectation values')

############################################
# Addition from 18-02-2020
############################################


def plot_expectation_values_TFD(full_dict, qubit_order=['D1', 'Z1', 'X1', 'D3'],
                                system_A_qubits=['X1', 'D3'],
                                system_B_qubits=['D1', 'Z1'], bases = ['Z', 'X'],
                                ax=None, T:float = None,
                                exact_dict: dict = None, **kw):
    if ax is None:
        f, ax = plt.subplots(figsize=(12,5))
    else:
        f = ax.get_figure()

    f.set_figwidth(12)
    f.set_figheight(10)
    operators = full_dict.keys()
    color_dict = dict()
    labels = ['IIII']
    color_dict['IIII'] = 'purple'
    for i, operator in enumerate(operators):
        for j, basis in enumerate(bases):
            if basis in operator:
                correlators = ','.join([qubit_order[i] for i, j in enumerate(operator) if j != 'I'])
                label = r'{}-${}$'.format(basis, correlators)
                labels.append(label)
                if len(label) < 10:
                    if (system_A_qubits[0] in label and system_A_qubits[1] in label):
                        color_dict[label] = 'r'
                    elif (system_B_qubits[0] in label and system_B_qubits[1] in label):
                        color_dict[label] = 'r'
                    elif (system_A_qubits[0] in label and system_B_qubits[0] in label):
                        color_dict[label] = 'b'
                    elif (system_A_qubits[1] in label and system_B_qubits[1] in label):
                        color_dict[label] = 'b'
                    else:
                        color_dict[label] = 'purple'
                else:
                    color_dict[label] = 'purple'

    for i, operator in enumerate(operators):
        ax.bar(i, full_dict[operator], color=color_dict[labels[i]], align='center', zorder = 1)
        if exact_dict is not None:
            T_idx = exact_dict['T'].index(T)
            ax.bar(list(full_dict).index(operator), exact_dict[operator][T_idx], fill=False, linestyle='--', edgecolor='black', align='center', zorder = 2)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=75)
        ax.text(1, -.5, '$Inter=${:.2f}'.format(np.abs(full_dict['ZIZI'])+np.abs(full_dict['IZIZ'])+
                                                np.abs(full_dict['XIXI'])+np.abs(full_dict['IXIX'])))
        ax.text(15, -.5, '$Intra=${:.2f}'.format(np.abs(full_dict['ZZII'])+np.abs(full_dict['IIZZ'])+
                                                 np.abs(full_dict['XXII'])+np.abs(full_dict['IIXX'])))
    ax.set_ylabel('Expectation value')
    ax.set_ylim(-1.05, 1.05)
    ax.set_title('Expectation values for pauli operators')
    return f, ax


class TFD_versus_temperature_analysis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True, operators=None, exact_dict: dict = None):
        """
        Analysis for the Thermal Field Double QAOA experiment. Plots expectation values versus temperatures.

        Args:
            g (float):
                coupling strength (in theorist units)
            T (float):
                temperature (in theorist units)
        """

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        if operators is not None:
            self.operators = operators
        else:
            self.operators = None

        if exact_dict is not None:
            self.exact_dict = exact_dict
        else:
            self.exact_dict = None

        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Extract pauli terms from multiple hd5 files.
        """
        self.raw_data_dict = {}
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)
        for ts in self.timestamps:
            data_fp = get_datafilepath_from_timestamp(ts)
            param_spec = {'TFD_dict': ('Analysis/quantities_of_interest', 'attr:all_attr'),
                         'tomo_dict': ('Analysis/quantities_of_interest/full_tomo_dict', 'attr:all_attr')}
            self.raw_data_dict[ts] = h5d.extract_pars_from_datafile(data_fp, param_spec)

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}
        self.proc_data_dict['timestamps'] = self.raw_data_dict['timestamps']
        self.proc_data_dict['T'] = [self.raw_data_dict[ts]['TFD_dict']['T'] for ts in self.proc_data_dict['timestamps']]
        for i, operator in enumerate(self.operators):
            if '+' in operator:
                seperate_operators = operator.split('+')
                self.proc_data_dict[operator] = np.zeros(len(self.proc_data_dict['timestamps']))
                for sep in seperate_operators:
                    self.proc_data_dict[operator] += np.array([self.raw_data_dict[ts]['tomo_dict'][sep] for ts in self.proc_data_dict['timestamps']])
                self.proc_data_dict[operator] = list(self.proc_data_dict[operator])
            else:
                self.proc_data_dict[operator] = [self.raw_data_dict[ts]['tomo_dict'][operator] for ts in self.proc_data_dict['timestamps']]

    def prepare_plots(self):
        self.plot_dicts['pauli_vs_temperature'] = {
            'plotfn': plot_TFD_versus_T,
            'tomo_dict': self.proc_data_dict,
            'operators': self.operators,
            'exact_dict': self.exact_dict,
            'numplotsy': len(self.operators),
            'presentation_mode': True
        }


def plot_TFD_versus_T(tomo_dict, operators=None, beta=False, ax=None, ax_dict=None, figsize=(10, 10), exact_dict=None, **kw):
    if ax is None:
        fig, ax = plt.subplots(len(operators), figsize=figsize)
    else:
        fig = ax[0].get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(15)
    if beta == True:
        x_label = 'Beta'
        x = [1/T for T in tomo_dict['T']]
        if exact_dict is not None:
            x_exact = [1/T for T in exact_dict['T']]
    else:
        x_label = 'Temperature'
        x = tomo_dict['T']
        if exact_dict is not None:
            x_exact = exact_dict['T']
    for i, operator in enumerate(operators):
        ax[i].plot(x, tomo_dict[operator], color='red', label='experiment')
        ax[i].scatter(x, tomo_dict[operator], facecolor='red')
        if exact_dict is not None:
            ax[i].plot(x_exact, exact_dict[operator], color = 'black', label='exact')
            ax[i].scatter(x_exact, exact_dict[operator], facecolor = 'black')
        ax[i].set_xlabel(x_label)
        ax[i].set_ylabel(operator)
        ax[i].legend()
        if '+' in operator:
            ax[i].set_ylim(-2, 2)
        else:
            ax[i].set_ylim(-1, 1)
    return fig, ax


"""
Fidelity analysis functions added 02-25-2020
"""


def operator(operator_string):
    """
    Inputs a string consisting of I, X, Y and Z characters
    Outputs an array of the crossproduct of the corresponding operators
    """
    #Pauli matrices
    I=np.array([[1.,0],[0,1.]])/2
    X=np.array([[0,1.],[1.,0]])/2
    Y=np.array([[0,0+-1.j],[0+1.j,0]])/2
    Z=np.array([[1,0],[0,-1]])/2
    full_operator=1
    for operator in operator_string:
        if operator == 'I':
            operator = I
        elif operator == 'X':
            operator = X
        elif operator == 'Y':
            operator = Y
        elif operator == 'Z':
            operator = Z
        elif operator == 'H':
            operator = 1/np.sqrt(2)*(X+Z)
        else:
            raise ValueError('operator_string should contain only I, X, Y or Z')
        full_operator=np.kron(full_operator,operator)
    return full_operator

def vec2dm(vec):
    vec = vec.reshape(len(vec), 1)
    vec_transpose = vec.reshape(1, len(vec))
    rho = np.dot(vec, vec_transpose)
    return rho

def vecs2mat(vec1,vec2):
    if len(vec1) != len(vec2):
        raise ValueError('Vectors must be same length')
    vec1 = vec1.reshape(len(vec1), 1)
    vec2 = vec2.reshape(1, len(vec2))
    rho = np.dot(vec1,vec2)
    return rho

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

def trace_distance(rho_1, rho_2):
    """
    To be constructed
    """
    return

def tomo2dm(tomo_dict):
    num_qubits = len(list(tomo_dict.keys())[0])
    dim = 2**num_qubits
    dm = np.zeros((dim,dim), dtype=np.complex128)
    for op, value in tomo_dict.items():
        dm += value*operator(op)
    return dm

class Hamiltonian:
    def __init__(self, hamiltonian=operator('ZZ')+operator('XI')+operator('IX')):
        self.hamiltonian = hamiltonian
        self.dim = len(hamiltonian)

    def eigen_values(self):
        eigen_values = np.linalg.eig(self.hamiltonian)[0]
        return eigen_values

    def eigen_vectors(self):
        eigen_vectors = np.linalg.eig(self.hamiltonian)[1]
        return eigen_vectors

    def eigen_dict(self):
        eigen_dict=dict()
        eigen_values, eigen_vectors = np.linalg.eig(self.hamiltonian)
        for n in range(len(eigen_values)):
            eigen_dict[n]=[]
            eigen_dict[n].append(eigen_values[n])
            eigen_dict[n].append(eigen_vectors[:,n])
        return eigen_dict

    def thermal_gibbs_rho(self, T):
        if np.round(T, 6) == 0:
            raise ValueError('Temperature can not be zero')
        rho = np.zeros((self.dim, self.dim))
        for n in range(self.dim):
            vec = self.eigen_vectors()[:,n].reshape(self.dim,1)
            rho += np.exp(-self.eigen_values()[n]/T)*np.dot(vec,np.transpose(vec))
        return np.round(rho/np.trace(rho),6)

    def TFD_state(self, T):
        if np.round(T, 6) == 0:
            raise ValueError('Temperature can not be zero')
        psi = np.zeros((1,self.dim**2))
        for n in range(self.dim):
            vec = self.eigen_vectors()[:,n]
            psi += np.exp(-self.eigen_values()[n]/(2*T))*np.kron(vec,vec)
        psi_norm=np.linalg.norm(psi)
        return np.transpose(psi)/psi_norm

    def TFD_rho(self, T):
        vec=self.TFD_state(T).reshape(self.dim**2,1)
        vec_transpose=self.TFD_state(T).reshape(1,self.dim**2)
        rho = np.dot(vec,vec_transpose)
        return rho

    def plot_non_zero_pauli_terms(self, pauli_dict,T):
        new_dict = pauli_dict.copy()
        pauli_set = ['I', 'X', 'Y', 'Z']
        if len(pauli_dict) == 16:
            PiPj = []
            for Pi in pauli_set:
                for Pj in pauli_set:
                    PiPj.append(Pi+Pj)
            for i, term in enumerate(PiPj):
                if np.round(np.sum(np.abs(new_dict[term])),6) == 0:
                    del new_dict[term]
        elif len(pauli_dict) == 256:
            PiPjPkPl = []
            for Pi in pauli_set:
                for Pj in pauli_set:
                    for Pk in pauli_set:
                        for Pl in pauli_set:
                            PiPjPkPl.append(Pi+Pj+Pk+Pl)
            for i, term in enumerate(PiPjPkPl):
                if np.round(np.sum(np.abs(new_dict[term])),6) == 0:
                    del new_dict[term]
        else:
            raise ValueError('Not all pauli terms in dictionary')
        fig, axs = plt.subplots(1,figsize=(10,10))
        for i, term in enumerate(new_dict.keys()):
            axs.plot(1/T, new_dict[term], label=term)
        axs.legend()
        axs.set_ylabel('Pauli terms')
        axs.set_xlabel('1/T')

    def expectation_value(self, operator, rho):
        if len(operator) != len(rho):
            raise ValueError('Operator and density matrix must be have same dimensions')
        return np.round(np.real(np.trace(np.dot(operator, rho))),6)

    def pauli_vector_gibbs(self, T, plot=False):
        if self.dim != 4:
            raise ValueError('Only for 4x4 Hamiltonian')
        T=np.array(T)
        pauli_dict=dict()
        pauli_set=['I', 'X', 'Y', 'Z']
        for Pi in pauli_set:
            for Pj in pauli_set:
                PiPj = operator(Pi+Pj)
                pauli_dict[Pi+Pj] = []
                if np.sqrt(np.size(T)) > 1:
                    for Ti in T:
                        pauli_dict[Pi+Pj].append(self.expectation_value(PiPj, self.thermal_gibbs_rho(Ti)))
                else:
                    pauli_dict[Pi+Pj].append(self.expectation_value(PiPj, self.thermal_gibbs_rho(T)))
        if plot:
            self.plot_non_zero_pauli_terms(pauli_dict, T)
        return pauli_dict

    def pauli_vector_TFD(self, T, plot=False):
        if self.dim**2 != 16:
            raise ValueError('Only for 16x16 Hamiltonian')
        T=np.array(T)
        pauli_dict=dict()
        pauli_set=['I', 'X', 'Y', 'Z']
        for Pi in pauli_set:
            for Pj in pauli_set:
                for Pk in pauli_set:
                    for Pl in pauli_set:
                        PiPjPkPl = operator(Pi+Pj+Pk+Pl)
                        pauli_dict[Pi+Pj+Pk+Pl] = []
                        if np.sqrt(np.size(T)) > 1:
                            for Ti in T:
                                pauli_dict[Pi+Pj+Pk+Pl].append(self.expectation_value(PiPjPkPl, np.dot(self.TFD_state(Ti),np.transpose(self.TFD_state(Ti)))))
                        else:
                            pauli_dict[Pi+Pj+Pk+Pl].append(self.expectation_value(PiPjPkPl, np.dot(self.TFD_state(Ti),np.transpose(self.TFD_state(T)))))
        if plot:
            self.plot_non_zero_pauli_terms(pauli_dict, T)
        return pauli_dict



def plot_fidelities_versus_T(fid_dict, data_label=None, data_marker=None, data_color='black', beta=False, ax=None, ax_dict=None, figsize=(10, 10), **kw):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    print(fid_dict)
    if beta == True:
        x_label = 'Beta'
        x = [1/T for T in list(fid_dict.keys())]
    else:
        x_label = 'Temperature'
        x = list(fid_dict.keys())
    ax.plot(x, list(fid_dict.values()), color=data_color)
    ax.scatter(x, list(fid_dict.values()), marker=data_marker, facecolor=data_color, label=data_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Fidelity')
    ax.set_xscale('symlog')
    ax.legend()
    ax.set_ylim(0.7, 1.01)
    return fig, ax

class Gibbs_fidelity_analysis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str = None, t_stop: str = None, ts_list=None,
                 label: str = '',
                 g: float = 1, T=1,
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):
        """
        Analysis for the Thermal Field Double state QAOA experiment.

        Args:
            g (float):
                coupling strength (in theorist units)
            T (float):
                temperature (in theorist units)
        """

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.g = g
        self.T = T
        self.ts_list = ts_list
        if options_dict == None:
            self.options_dict = {'beta': False,
                                 'data_label': 'Data',
                                 'data_color': 'black',
                                 'save_figs': True}
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Extract two qubit tomography terms.
        """
        self.raw_data_dict = {}
        if self.ts_list is None:
            self.timestamps = a_tools.get_timestamps_in_range(self.t_start,
                                                              self.t_stop,
                                                              label=self.labels)
        else:
            self.timestamps = self.ts_list
        for ts in self.timestamps:
            data_fp = get_datafilepath_from_timestamp(ts)
            param_spec = {'TFD_dict': ('Analysis/quantities_of_interest', 'attr:all_attr'),
                          'tomo_dict': ('Analysis/quantities_of_interest/full_tomo_dict', 'attr:all_attr')}
            self.raw_data_dict[ts] = h5d.extract_pars_from_datafile(data_fp, param_spec)

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        """
        Create density matrix, find exact density matrix for given T and calculate fidelity.
        """
        self.proc_data_dict = {}
        self.hamiltonian = Hamiltonian()
        self.proc_data_dict['timestamps'] = self.raw_data_dict['timestamps']
        self.proc_data_dict['operators'] = self.raw_data_dict[self.raw_data_dict['timestamps'][0]]['tomo_dict'].keys()
        self.proc_data_dict['T'] = [self.raw_data_dict[ts]['TFD_dict']['T'] for ts in self.proc_data_dict['timestamps']]
        for i, operator in enumerate(self.proc_data_dict['operators']):
            self.proc_data_dict[operator] = [self.raw_data_dict[ts]['tomo_dict'][operator] for ts in self.proc_data_dict['timestamps']]
        for i, Ti in enumerate(self.proc_data_dict['T']):
            self.proc_data_dict[Ti] = {}
            self.proc_data_dict[Ti]['density_matrices'] = {}
            self.proc_data_dict[Ti]['density_matrices']['experiment'] = tomo2dm(self.raw_data_dict[self.proc_data_dict['timestamps'][i]]['tomo_dict'])
            self.proc_data_dict[Ti]['density_matrices']['theory'] = self.hamiltonian.thermal_gibbs_rho(T=Ti)
            self.proc_data_dict[Ti]['fidelity'] = fidelity(self.proc_data_dict[Ti]['density_matrices']['experiment'],self.proc_data_dict[Ti]['density_matrices']['theory'])
        fid_df = pd.DataFrame.from_dict({T:self.proc_data_dict[T]['fidelity'] for T in self.proc_data_dict['T']}, orient='index',columns=['F']).transpose()
        operator_df = pd.DataFrame.from_dict({op:values for op, values in self.proc_data_dict.items() if op in self.proc_data_dict['operators']}, orient='index',columns=list(self.proc_data_dict['T']))
        self.proc_data_dict['dataframe'] = pd.concat([fid_df, operator_df])

    def prepare_plots(self):
        if len(self.timestamps) == 1:
            self.options_dict['data_label'] = str(round(self.proc_data_dict[self.proc_data_dict['T'][0]]['fidelity'],3))
        self.plot_dicts['fidelities_vs_temperature'] = {
            'plotfn': plot_fidelities_versus_T,
            'fid_dict': {T:self.proc_data_dict[T]['fidelity'] for T in self.proc_data_dict['T']},
            'beta': self.options_dict['beta'],
            'data_label': self.options_dict['data_label'],
            'data_color': self.options_dict['data_color']
        }
        self.plot_dicts['Gibbs_tomograph'] = {
            'plotfn': plot_expectation_values_Gibbs,
            'full_dict': self.raw_data_dict[list(self.raw_data_dict)[0]]['tomo_dict'],
            'T':self.raw_data_dict[list(self.raw_data_dict)[0]]['TFD_dict']['T'],
            'beta': self.options_dict['beta'],
        }

def plot_expectation_values_Gibbs(full_dict, qubit_order=['D1', 'Z1', 'X1', 'D3'], system_A_qubits=['X1','D3'],
                                  system_B_qubits=['D1', 'Z1'], gibbs_qubits=['X1','D3'], bases = ['Z','X','Y'], ax=None, T:float = None,
                                  exact_dict:bool = True, non_zero_only=False, **kw):
    if ax is None:
        f, ax = plt.subplots(figsize=(12,5))
    else:
        f = ax.get_figure()

    f.set_figwidth(12)
    f.set_figheight(10)

    hamiltonian = Hamiltonian()
    if non_zero_only:
        operators = [key for key in full_dict.keys() if np.round(full_dict[key], 3) != 0]
    else:
        operators = full_dict.keys()
    exact_dict = {operator_string:hamiltonian.expectation_value(operator(operator_string), hamiltonian.thermal_gibbs_rho(T=T)) for operator_string in operators}
    color_dict = dict()
    labels = [gibbs_qubits[0]+' '+gibbs_qubits[1]]
    color_dict[gibbs_qubits[0]+' '+gibbs_qubits[1]] = 'purple'
    for i, op in enumerate(operators):
        if op != 'II':
            labels.append(op)
        if op == 'XX':
            color_dict[i] = 'r'
        elif op == 'YY':
            color_dict[i] = 'yellow'
        elif op == 'ZZ':
            color_dict[i] = 'b'
        elif op == 'IX' or op == 'XI':
            color_dict[i] = 'green'
        else:
            color_dict[i] = 'purple'
    for i, op in enumerate(operators):
        ax.bar(i, full_dict[op], color=color_dict[i], align='center', zorder=1)
        if exact_dict is not None:
            ax.bar(list(full_dict).index(op), exact_dict[op]*4, fill=False, linestyle='--', edgecolor='black', align='center', zorder = 2)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=75)
    ax.set_ylabel('Expectation value')
    ax.set_ylim(-1.05, 1.05)
    ax.set_title('Expectation values for pauli operators')
    
    return f, ax
