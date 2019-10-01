import os
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
    cmap_to_alpha, cmap_first_to_alpha
from pycqed.utilities.general import int2base
import pycqed.measurement.hdf5_data as h5d


class Multiplexed_Readout_Analysis(ba.BaseDataAnalysis):
    """
    Multiplexed readout analysis.

    Does data binning and creates histograms of data.
    Threshold is auto determined as the mean of the data.
    Used to construct a assignment probability matris.
    """

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 options_dict: dict = None, extract_only: bool = False,
                 extract_combinations: bool = False,
                 auto=True):
        """
        Inherits from BaseDataAnalysis.

        extract_combinations (bool):
            if True, tries to extract combinations used in experiment
            from the experimental metadata.
        """

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)
        self.extract_combinations = extract_combinations
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
        param_spec = {'data':
                      ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        if self.extract_combinations:
            param_spec['combinations'] = \
                ('Experimental Data/Experimental Metadata/combinations', 'dset')
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

        if self.extract_combinations:
            # For some reason the list is stored a list of length 1 arrays...
            self.raw_data_dict['combinations'] = [
                c[0] for c in self.raw_data_dict['combinations']]

    def process_data(self):
        self.proc_data_dict = {}

        # determine combinations iterated over (e.g., 000, 001, 010, 011 etc.)
        value_names = self.raw_data_dict['value_names']
        nr_qubits = len(value_names)

        if 'combinations' in self.raw_data_dict.keys():
            combinations = self.raw_data_dict['combinations']
        else:
            base = 2
            combinations = [int2base(i, base=base, fixed_length=nr_qubits) for
                            i in range(base**nr_qubits)]
        self.proc_data_dict['combinations'] = combinations

        # Not all experiments are valid calibration points.
        valid_combinations = []
        for c in combinations:
            if set(c) <= {'0', '1'}:
                valid_combinations += [c]
        self.proc_data_dict['valid_combinations'] = valid_combinations

        # Get qubit labels
        qubit_labels = [v[-2:].decode('utf-8').strip() for v in value_names]
        self.proc_data_dict['qubit_labels'] = qubit_labels

        # Bin data in histograms
        raw_shots = self.raw_data_dict['data'][:, 1:]
        hist_data = {}
        for i, ch_name in enumerate(value_names):
            ch_data = raw_shots[:, i]  # select per channel
            hist_data[ch_name] = {}
            for j, comb in enumerate(combinations):
                cnts, bin_edges = np.histogram(
                    ch_data[j::len(combinations)],
                    bins=100, range=(np.min(ch_data), np.max(ch_data)))
                bin_centers = bin_edges[:-1]+(bin_edges[1]-bin_edges[0])/2
                hist_data[ch_name][comb] = (cnts, bin_centers)
        self.proc_data_dict['hist_data'] = hist_data

        # Calculate mean voltages (used for threshold )
        binned_data = {}
        for i, ch_name in enumerate(value_names):
            ch_data = raw_shots[:, i]  # select per channel
            binned_data[ch_name] = {}
            for j, comb in enumerate(combinations):

                binned_data[ch_name][comb] = np.mean(
                    ch_data[j::len(combinations)])  # start at

        mn_voltages = {}
        for i, ch_name in enumerate(value_names):
            ch_data = binned_data[ch_name]  # select per channel
            mn_voltages[ch_name] = {'0': [], '1': []}
            for c in combinations:
                if c in valid_combinations:
                    if c[i] == '0':
                        mn_voltages[ch_name]['0'].append(ch_data[c])
                    elif c[i] == '1':
                        mn_voltages[ch_name]['1'].append(ch_data[c])
            mn_voltages[ch_name]['0'] = np.mean(mn_voltages[ch_name]['0'])
            mn_voltages[ch_name]['1'] = np.mean(mn_voltages[ch_name]['1'])
            mn_voltages[ch_name]['threshold'] = np.mean(
                [mn_voltages[ch_name]['0'],
                 mn_voltages[ch_name]['1']])
        self.proc_data_dict['mn_voltages'] = mn_voltages

        # Digitize data
        digitized_data = np.zeros(raw_shots.shape)
        for i, vn in enumerate(value_names):
            digitized_data[:, i] = np.array(
                raw_shots[:, i] > mn_voltages[vn]['threshold'], dtype=int)

        # Bin digitized data
        binned_dig_data = {}
        for i, ch_name in enumerate(value_names):
            ch_data = digitized_data[:, i]  # select per channel
            binned_dig_data[ch_name] = {}
            for j, comb in enumerate(combinations):

                binned_dig_data[ch_name][comb] = np.mean(
                    ch_data[j::len(combinations)])  # start at

        self.proc_data_dict['binned_dig_data'] = binned_dig_data
        # Calculate assignment probability matrix
        assignment_prob_matrix = calc_assignment_prob_matrix(
            combinations, digitized_data, valid_combinations=valid_combinations)
        self.proc_data_dict['assignment_prob_matrix'] = assignment_prob_matrix

    def prepare_plots(self):
        self.plot_dicts['assignment_probability_matrix'] = {
            'plotfn': plot_assignment_prob_matrix,
            'assignment_prob_matrix':
                self.proc_data_dict['assignment_prob_matrix'],
            'combinations': self.proc_data_dict['combinations'],
            'valid_combinations': self.proc_data_dict['valid_combinations'],
            'qubit_labels': self.proc_data_dict['qubit_labels'],
            'plotsize': np.array(np.shape(self.proc_data_dict['assignment_prob_matrix'].T))*.8
        }
        for i, value_name in enumerate(self.raw_data_dict['value_names']):
            qubit_label = self.proc_data_dict['qubit_labels'][i]

            self.plot_dicts['mux_ssro_histogram_{}'.format(qubit_label)] = {
                'plotfn': plot_mux_ssro_histograms,
                'hist_data': self.proc_data_dict['hist_data'][value_name],
                'qubit_idx': i,
                'value_name': value_name,
                'combinations': self.proc_data_dict['combinations'],
                'qubit_labels': self.proc_data_dict['qubit_labels'],
                'threshold': self.proc_data_dict['mn_voltages'][value_name]['threshold']
            }


def calc_assignment_prob_matrix(combinations, digitized_data,
        valid_combinations=None):

    if valid_combinations is None:
        valid_combinations = combinations
    assignment_prob_matrix = np.zeros((len(combinations),
        len(valid_combinations)))
    # for input_state in combinations:

    for i, outcome in enumerate(digitized_data):
        decl_state = ''.join([str(int(s)) for s in outcome])
        # check what combination the declared state corresponds to
        decl_state_idx = valid_combinations.index(decl_state)

        # row -> input state
        # column -> declared state
        # increment the count of the declared state for the input state by 1
        assignment_prob_matrix[i % len(combinations), decl_state_idx] += 1

    # Normalize the matrix
    assignment_prob_matrix /= np.sum(assignment_prob_matrix, axis=1)[0]
    return assignment_prob_matrix


def plot_assignment_prob_matrix(assignment_prob_matrix,
                                combinations, qubit_labels, ax=None,
                                valid_combinations=None, **kw):
    if ax is None:
        figsize = np.array(np.shape(assignment_prob_matrix))*.7
        f, ax = plt.subplots(figsize=figsize)
    else:
        f = ax.get_figure()

    if valid_combinations is None:
        valid_combinations = combinations

    alpha_reds = cmap_to_alpha(cmap=pl.cm.Reds)
    colors = [(0.6, 0.76, 0.98), (0, 0, 0)]
    cm = LinearSegmentedColormap.from_list('my_blue', colors)
    alpha_blues = cmap_first_to_alpha(cmap=cm)

    red_im = ax.matshow(assignment_prob_matrix*100,
                        cmap=alpha_reds, clim=(0., 10))
    blue_im = ax.matshow(assignment_prob_matrix*100,
                         cmap=alpha_blues, clim=(50, 100))

    caxb = f.add_axes([0.9, 0.6, 0.02, 0.3])

    caxr = f.add_axes([0.9, 0.15, 0.02, 0.3])
    ax.figure.colorbar(red_im, ax=ax, cax=caxr)
    ax.figure.colorbar(blue_im, ax=ax, cax=caxb)

    rows, cols = np.shape(assignment_prob_matrix)
    for i in range(rows):
        for j in range(cols):
            c = assignment_prob_matrix[i, j]
            if c > .05:
                col = 'white'
            else:
                col = 'black'
            ax.text(j, i, '{:.2f}'.format(c),
                    va='center', ha='center', color=col)

    ax.set_xticklabels(valid_combinations)
    ax.set_xticks(np.arange(len(valid_combinations)))

    ax.set_yticklabels(combinations)
    ax.set_yticks(np.arange(len(combinations)))
    ax.set_ylim(len(combinations)-.5, -.5)
    ax.set_ylabel('Input state')
    ax.set_xlabel('Declared state')
    ax.xaxis.set_label_position('top')

    qubit_labels_str = ', '.join(qubit_labels)
    ax.set_title('Assignment probability matrix\n qubits: [{}]'.format(
        qubit_labels_str))


def plot_mux_ssro_histograms(
        hist_data, combinations,
        qubit_idx, value_name,
        qubit_labels, threshold,
        ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()

    colors_R = pl.cm.Reds
    colors_B = pl.cm.Blues
    colors_G = pl.cm.Greens
    iR = 0.1  # Do not start at the complete white/transparent end
    iB = 0.1
    iG = 0.1
    for i, (key, (cnts, bin_centers)) in enumerate(hist_data.items()):

        if set(key) <= {'0', '1'}:
            if key[qubit_idx] == '0':
                # increment the blue colorscale
                col = colors_B(iB)
                iB += 0.8/(len(combinations)/2)  # .8 to not span full colorscale
            elif key[qubit_idx] == '1':
                # Increment the red colorscale
                col = colors_R(iR)
                iR += 0.8/(len(combinations)/2)
            else:
                raise ValueError('{}  {}'.format(
                    combinations, combinations[qubit_idx]))
        else:
            # increment the green colorscale
            col = colors_G(iG)
            iG += 0.8/(len(combinations)/2)  # .8 to not span full colorscale
        ax.plot(bin_centers, cnts, label=key, color=col)

    ax.set_xlabel(value_name.decode('utf-8'))
    ax.set_ylabel('Counts')
    ax.axvline(x=threshold,
               ls='--', color='grey', label='threshold')
    ax.legend(loc=(1.05, .01), title='Prepared state\n{}'.format(
        qubit_labels))
