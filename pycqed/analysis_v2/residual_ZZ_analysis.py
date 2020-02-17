import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import os
from pycqed.analysis import analysis_toolbox as a_tools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pl
from matplotlib import cm
import pycqed.measurement.hdf5_data as h5d
import lmfit
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
    cmap_to_alpha, cmap_first_to_alpha

class Residual_Crosstalk(ba.BaseDataAnalysis):
    """
    Residual crosstalk analysis.

    Produces residual crosstalk matrix from multiple crosstalk measurements.
    Arguments:
            t_start & t_stop: the first and last timestamps of a sequence of
            residual ZZ measurement. The residual ZZ measurement is defined as
            the function 'residual_coupling_sequence' in repository ./pycqed/measurement/openql_experiments/multi_qubit_oql.py
            qubits: list of qubit names that are involved in the experiment as
            either Echo/Target qubit or Spectator/Control qubit. Echo/Target qubit is the qubit
            that performs the echo sequence, while the Spectator/Control qubit is the qubit
            that is excited halfway during that sequence. 
    """

    def __init__(self, qubits: list = None, t_start: str = None, t_stop: str = None,
                 label: str = '', extract_only: bool = False,
                 options_dict: dict = None, 
                 auto=True):
        """
        Inherits from BaseDataAnalysis.

        """

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict)
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
            self.proc_data_dict['quantities_of_interest'] = {}
            ts_dict = {}
            matrix_labels = {}
            residual_ZZ_matrix = np.zeros((len(self.qubits),len(self.qubits)))
            for ts in self.timestamps:
                ts_dict[ts] = {}
                self.timestamp = ts
                target_qubit = self.raw_data_dict[ts]['target_qubit']
                spectators = self.raw_data_dict[ts]['spectator_qubits'] 
                state = self.raw_data_dict[ts]['spectator_state'] 
                spectator_list = spectators.strip('[').strip(']').split(', ')
                for i, bit in enumerate(state):
                    if int(bit) == 1:
                        active_spectator = spectator_list[i]
                        active_spectator = active_spectator[1:-1]
                x, y = self.raw_data_dict[ts]['data'][:-2,0], self.raw_data_dict[ts]['data'][:-2,1]
                low_cal = self.raw_data_dict[ts]['data'][-2,1]
                high_cal = self.raw_data_dict[ts]['data'][-1,1]
                y = (y-low_cal)/(high_cal-low_cal)
                fit_res = fit_residual_coupling_oscillation(x, y)
                frequency = fit_res.best_values['frequency']
                if y[1]-y[0] < 0:
                    frequency = -1*frequency
                if np.std(y)/np.mean(y) < 0.1:
                    print('No oscillation found for {},{}'.format(target_qubit, active_spectator))
                    frequency = 0
                residual_ZZ_matrix[self.qubits.index(target_qubit), self.qubits.index(active_spectator)] = frequency
                ts_dict[ts]['frequency'] = frequency
                ts_dict[ts]['fit_res'] = fit_res
                ts_dict[ts]['spectator'] = active_spectator
                ts_dict[ts]['target'] = target_qubit
                print('{}:Setting {},{} to {}'.format(ts, target_qubit, active_spectator, frequency))
            for i in range(len(self.qubits)):
                residual_ZZ_matrix[i,i] = np.nan
            self.proc_data_dict['quantities_of_interest'] = {
                'matrix': residual_ZZ_matrix,
                'qubit_labels': self.qubits,
                'fit_per_ts': ts_dict}
            
    def prepare_plots(self):
        self.plot_dicts['residual_matrix'] = {
            'plotfn': plot_crosstalk_matrix,
            'crosstalk_matrix': self.proc_data_dict['quantities_of_interest']['matrix'],
            'qubit_labels': self.qubits
        }
        self.plot_dicts['fit_matrix'] = {
            'plotfn': plot_residual_fit_matrix,
            'fit_per_ts': self.proc_data_dict['quantities_of_interest']['fit_per_ts'],
            'qubit_labels': self.qubits
        }
        
def plot_residual_fit_matrix(fit_per_ts, qubit_labels, ax=None, fig_size = None, **kw):
    axes_length = 5*len(qubit_labels)
    fig_size = (axes_length, axes_length)
    fig, axs = plt.subplots(len(qubit_labels),len(qubit_labels), figsize=fig_size)
    for i, qubit in enumerate(qubit_labels):
        axs[i,i].set_axis_off()
        axs[i,i].text(0.5, 0.5, qubit, ha='center', va='center', fontsize=24)
    for ts in fit_per_ts.keys():
        spectator = fit_per_ts[ts]['spectator']
        target = fit_per_ts[ts]['target']
        i,j = qubit_labels.index(target), qubit_labels.index(spectator)
        plot_residual_coupling_fit(fit_per_ts[ts]['fit_res'], axs[i,j], i, j, fit_per_ts[ts]['frequency'], ts)
    fig.text(0, 0.5, 'Echo qubit', ha='center', va='center', rotation = 'vertical')
    fig.text(0.5, 0, 'Control qubit', ha='center', va='center')
    for i, qubit in enumerate(qubit_labels):
        fig.text(0.05, 0.8*(1-(i)/4), qubit)
        fig.text(0.8*(i+1)/4, 0.05, qubit)
    return fig, axs

def fit_residual_coupling_oscillation(x, y, **kw):
    """
    Fit damped oscillator for the residual ZZ measurement defined as
    the function 'residual_coupling_sequence' in repository 
    ./pycqed/measurement/openql_experiments/multi_qubit_oql.py

    """
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


def plot_residual_coupling_fit(fit_res, ax, row, column, frequency, ts):
    """
    Plot function for the fitting result of the fit function 'fit_residual_coupling_oscillation'
    Arguments: 
                fit_res: ModelResult type file containing fitting result.
                ax: axis of a Matplotlib subplot.
                row, column: row and column of ax within corresponding Matplotlib figure.
                frequency: frequency found in the fitting result.
                ts: timestamp used to acquire data for fitting.

    """
    y_data = fit_res.data
    y_fit = fit_res.best_fit
    x_data = fit_res.userkws['t']*1e6
    ax.scatter(x_data, y_data)
    ax.plot(x_data, y_fit, color='green')
    ax.set_xlabel(r'$\tau$/2 ($\mu$s)')
    if column == 0:
        ax.set_ylabel('Echo qubit population')
    ax.set_xlim(min(x_data), max(x_data))
    ax.set_ylim(0,1)
    ax.legend([str(round(frequency/1e3,1))+' kHz'])
    ax.set_title(ts)


def plot_crosstalk_matrix(crosstalk_matrix,
                          qubit_labels, ax=None, **kw):
    """
    Plot function for nxn crosstalk matrix aqcuired by Residual_Crosstalk analysis. 
    """
    crosstalk_matrix = np.copy(crosstalk_matrix)/1000
    if ax is None:
        figsize = np.array(np.shape(crosstalk_matrix))
        f = plt.figure(figsize=figsize)
        ax = f.add_axes([0.2, 0.2, 0.6, 0.6])
    else:
        f = ax.get_figure()
    
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

    ax.set_xticklabels(qubit_labels)
    ax.set_xticks(np.arange(len(qubit_labels)))
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticklabels(qubit_labels)
    ax.set_yticks(np.arange(len(qubit_labels)))
    ax.set_ylim(len(qubit_labels)-.5, -.5)
    ax.set_ylabel('Echo qubit')
    ax.set_xlabel('Control qubit')
#     ax.xaxis.set_label_position('top')

    qubit_labels_str = ', '.join(qubit_labels)
    ax.set_title('Cross-talk matrix')
    return f, ax