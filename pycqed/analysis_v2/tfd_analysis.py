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


class TFD_3CZ_Analysis_Pauli_Strings(ba.BaseDataAnalysis):
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 g: float = 1, T: float = 1,
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):
        """
        Analysis for 3CZ version of the Thermal Field Double VQE circuit.

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
            ch_data = raw_shots[:, i]  # select per channel
            binned_data[ch_name] = {}
            for j, comb in enumerate(combinations):

                binned_data[ch_name][comb] = np.mean(
                    ch_data[j::len(combinations)])  # start at

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
    H_A = 1.57*pauli_terms['ZZII'] + g*pauli_terms['XIII'] + g*pauli_terms['IXII']
    H_B = 1.57*pauli_terms['IIZZ'] + g*pauli_terms['IIXI'] + g*pauli_terms['IIIX']
    H_AB = pauli_terms['ZIZI'] + pauli_terms['IZIZ'] + \
        pauli_terms['XIXI'] + pauli_terms['IXIX']

    H = H_A + H_B - (T**1.57)*H_AB

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
