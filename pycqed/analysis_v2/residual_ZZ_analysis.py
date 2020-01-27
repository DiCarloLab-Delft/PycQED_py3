import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import os
from pycqed.analysis import analysis_toolbox as a_tools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pl
from matplotlib import cm
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
    cmap_to_alpha, cmap_first_to_alpha

class Residual_Crosstalk(ba.BaseDataAnalysis):
    """
    Residual crosstalk analysis.

    Produces residual crosstalk matrix from multiple crosstalk measurements.
    """

    def __init__(self, qubits: list = None, t_start: str = None, t_stop: str = None,
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
        self.qubits = qubits
        if auto:
            self.run_analysis()
        
    def extract_data(self):
        self.raw_data_dict = {}
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)
        for ts in self.timestamps:
            data_fp = get_datafilepath_from_timestamp(ts)
            param_spec = {
                'data': ('Experimental Data/Data', 'dset'),
                'target_qubit':  ('Experimental Data/Experimental Metadata', 'attr:target_qubit'),
                'spectator_qubits':  ('Experimental Data/Experimental Metadata', 'attr:spectator_qubits'),
                'spectator_state':  ('Experimental Data/Experimental Metadata', 'attr:spectator_state')}
            self.raw_data_dict[ts] = h5d.extract_pars_from_datafile(
                data_fp, param_spec)
        
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]
        
    def process_data(self):
            self.proc_data_dict = {}
            residual_ZZ_matrix = np.zeros((len(self.qubits),len(self.qubits)))
            for ts in self.timestamps:
                self.timestamp = ts
                target_qubit = self.raw_data_dict[ts]['target_qubit']
                spectators = self.raw_data_dict[ts]['spectator_qubits'] 
                state = self.raw_data_dict[ts]['spectator_state'] 
                spectator_list = spectators.strip('[').strip(']').split(', ')
                for i, bit in enumerate(state):
                    if int(bit) == 1:
                        active_spectator = spectator_list[i]
                        active_spectator = active_spectator
                x, y = self.raw_data_dict[ts]['data'][:-2,0], self.raw_data_dict[ts]['data'][:-2,1]
                fit_res = fit_residual_coupling_oscillation(x, y)
                frequency = fit_res.best_values['frequency']
                if y[1]-y[0] < 0:
                    frequency = -1*frequency
                if np.std(y)/np.mean(y) < 0.1:
                    frequency = 0
                residual_ZZ_matrix[qubits.index(target_qubit),qubits.index(active_spectator[1:-1])] = frequency
                print('{}:Setting {},{} to {}'.format(ts, target_qubit, active_spectator[1:-1], frequency))
            for i in range(len(self.qubits)):
                residual_ZZ_matrix[i,i] = np.nan
            self.proc_data_dict['quantities_of_interest'] = {
                'matrix': residual_ZZ_matrix,
                'qubit_labels': self.qubits}
    def prepare_plots(self):
        self.plot_dicts['residual_matrix'] = {
            'plotfn': plot_crosstalk_matrix,
            'crosstalk_matrix': self.proc_data_dict['quantities_of_interest']['matrix'],
            'qubit_labels': self.qubits
        }

def fit_residual_coupling_oscillation(x, y, **kw):
    average = np.mean(y)
    ft_of_data = np.fft.fft(y)
    index_of_fourier_maximum = np.argmax(np.abs(
    ft_of_data[1:len(ft_of_data) // 2])) + 1
    max_ramsey_delay = x[-1] - x[0]
    fft_axis_scaling = 1 / (max_ramsey_delay)
    freq_est = fft_axis_scaling * index_of_fourier_maximum
    est_number_of_periods = index_of_fourier_maximum
    
    
    damped_osc_mod = lmfit.Model(fit_mods.ExpDampOscFunc)
    damped_osc_mod.set_param_hint('frequency',
                                  value=freq_est,
                                  vary=True)
    damped_osc_mod.set_param_hint('phase',
                                  value=np.pi/2,vary=False)
    damped_osc_mod.set_param_hint('amplitude',
                                  value=0.5*(max(y)-min(y)),
                                  min=min(y),
                                  max=max(y),
                                  vary=True)
    damped_osc_mod.set_param_hint('tau',
                                  value=x[1]*10,
#                                   min=x[1],
#                                   max=x[1]*1000,
                                  vary=True)
    damped_osc_mod.set_param_hint('exponential_offset',
                                  value=average,
                                  min=min(y),
                                  max=max(y),
                                  vary=True)
    damped_osc_mod.set_param_hint('oscillation_offset',
                                  value=0,
                                  vary=False)
    damped_osc_mod.set_param_hint('n',
                                  value=1,
                                  vary=False)
    params = damped_osc_mod.make_params()

    fit_res = damped_osc_mod.fit(data=y,
                                 t=x,
                                 params=params)
    return fit_res

    def plot_crosstalk_matrix(crosstalk_matrix,
                          qubit_labels, combinations=None, ax=None,
                          valid_combinations=None, **kw):
    crosstalk_matrix = np.copy(crosstalk_matrix)/1000
    if ax is None:
        figsize = np.array(np.shape(crosstalk_matrix))
        f = plt.figure(figsize=figsize)
        ax = f.add_axes([0.2, 0.2, 0.6, 0.6])
    else:
        f = ax.get_figure()
    if combinations is None:
        combinations = qubit_labels
    if valid_combinations is None:
        valid_combinations = combinations
    
    max_range = 500*np.ceil(np.nanmax(np.abs(crosstalk_matrix))/500)
    min_range = -1*max_range
    mat_plot = ax.matshow(crosstalk_matrix,
                        cmap=cm.RdBu, clim=(-max_range, max_range))

    caxr = f.add_axes([0.85, 0.25, 0.02, 0.5])
    ax.figure.colorbar(mat_plot, ax=ax, cax=caxr, label='Frequency shift (kHz)')

    rows, cols = np.shape(crosstalk_matrix)
    for i in range(rows):
        for j in range(cols):
            c = crosstalk_matrix[i, j]
            if c > 0.5*max_range or c < 0.5*min_range:
                col = 'white'
            else:
                col = 'black'
            text_filtered = '{:.0f}'.format(c)
            if np.isnan(c):
                text_filtered = '-'
            ax.text(j, i, text_filtered,
                    va='center', ha='center', color=col)

    ax.set_xticklabels(valid_combinations)
    ax.set_xticks(np.arange(len(valid_combinations)))
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticklabels(combinations)
    ax.set_yticks(np.arange(len(combinations)))
    ax.set_ylim(len(combinations)-.5, -.5)
    ax.set_ylabel('Echo qubit')
    ax.set_xlabel('Control qubit')
#     ax.xaxis.set_label_position('top')

    qubit_labels_str = ', '.join(qubit_labels)
    ax.set_title('Cross-talk matrix')
    return f, ax