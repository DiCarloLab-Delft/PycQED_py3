import lmfit
from copy import deepcopy
from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old
from pycqed.analysis.tools import cryoscope_tools as ct
import pycqed.analysis_v2.base_analysis as ba
import pycqed.measurement.waveform_control_CC.waveform as wf
import pycqed.analysis.fitting_models as fit_mods
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.stats import sem
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

class RandomizedBenchmarking_SingleQubit_Analyasis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None,auto=True, close_figs=True):
        if options_dict is None:
            options_dict = dict()
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict, close_figs=close_figs)
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
                self.t_start, self.t_stop,
                label=self.labels)

        a = ma_old.MeasurementAnalysis(timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        if 'bins' in a.data_file['Experimental Data']['Experimental Metadata'].keys():
            bins = a.data_file['Experimental Data']['Experimental Metadata']['bins'].value
            self.raw_data_dict['ncl'] = bins[:-6:2]
            self.raw_data_dict['bins'] = bins

            self.raw_data_dict['value_names'] = a.value_names
            self.raw_data_dict['value_units'] = a.value_units
            self.raw_data_dict['measurementstring'] = a.measurementstring
            self.raw_data_dict['timestamp_string'] =a.timestamp_string

            self.raw_data_dict['binned_vals']=OrderedDict()
            self.raw_data_dict['cal_pts_zero'] = OrderedDict()
            self.raw_data_dict['cal_pts_one'] = OrderedDict()
            self.raw_data_dict['cal_pts_two'] = OrderedDict()
            self.raw_data_dict['measured_values_I'] = OrderedDict()
            self.raw_data_dict['measured_values_X'] = OrderedDict()
            for i, val_name in enumerate(a.value_names):
                binned_yvals = np.reshape(a.measured_values[i], (len(bins), -1), order='F')
                self.raw_data_dict['binned_vals'][val_name] = binned_yvals
                self.raw_data_dict['cal_pts_zero'][val_name] =\
                     binned_yvals[-6:-4, :].flatten()
                self.raw_data_dict['cal_pts_one'][val_name] =\
                     binned_yvals[-4:-2, :].flatten()
                self.raw_data_dict['cal_pts_two'][val_name] =\
                     binned_yvals[-2:, :].flatten()
                self.raw_data_dict['measured_values_I'][val_name] =\
                     binned_yvals[:-6:2, :]
                self.raw_data_dict['measured_values_X'][val_name] =\
                     binned_yvals[1:-6:2, :]

        else:
            bins = None

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish() # closes data file

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)


        for key in ['V0', 'V1', 'V2', 'SI', 'SX', 'P0', 'P1', 'P2', 'M_inv']:
            self.proc_data_dict[key] = OrderedDict()

        for val_name in self.raw_data_dict['value_names']:
            V0 = np.mean(
                self.raw_data_dict['cal_pts_zero'][val_name])
            V1 = np.mean(
                self.raw_data_dict['cal_pts_one'][val_name])
            V2 = np.mean(
                self.raw_data_dict['cal_pts_two'][val_name])

            self.proc_data_dict['V0'][val_name] =V0
            self.proc_data_dict['V1'][val_name] =V1
            self.proc_data_dict['V2'][val_name] =V2

            SI = np.mean(
                self.raw_data_dict['measured_values_I'][val_name], axis=1)
            SX = np.mean(
                self.raw_data_dict['measured_values_X'][val_name], axis=1)
            self.proc_data_dict['SI'][val_name] = SI
            self.proc_data_dict['SX'][val_name] = SX

            P0, P1, P2, M_inv = populations_using_rate_equations(
                SI, SX, V0, V1, V2)
            self.proc_data_dict['P0'][val_name] = P0
            self.proc_data_dict['P1'][val_name] = P1
            self.proc_data_dict['P2'][val_name] = P2
            self.proc_data_dict['M_inv'][val_name] = M_inv

    def prepare_plots(self):
        val_names = self.raw_data_dict['value_names']

        for i, val_name in enumerate(val_names):
            self.plot_dicts['binned_data_{}'.format(val_name)] = {
                    'plotfn': self.plot_line,
                    'xvals': self.raw_data_dict['bins'],
                    'yvals': np.mean(self.raw_data_dict['binned_vals'][val_name], axis=1),
                    'yerr':  sem(self.raw_data_dict['binned_vals'][val_name], axis=1) ,
                    'xlabel': 'Number of Cliffrods',
                    'xunit': '#',
                    'ylabel': val_name,
                    'yunit': self.raw_data_dict['value_units'][i],
                'title': self.raw_data_dict['timestamp_string']+'\n'+self.raw_data_dict['measurementstring'],
            }

        fs = plt.rcParams['figure.figsize']
        self.plot_dicts['cal_points_hexbin'] = {
            'plotfn':plot_cal_points_hexbin,
            'shots_0': (self.raw_data_dict['cal_pts_zero'][val_names[0]],
                        self.raw_data_dict['cal_pts_zero'][val_names[1]]),
            'shots_1': (self.raw_data_dict['cal_pts_one'][val_names[0]],
                        self.raw_data_dict['cal_pts_one'][val_names[1]]),
            'shots_2': (self.raw_data_dict['cal_pts_two'][val_names[0]],
                        self.raw_data_dict['cal_pts_two'][val_names[1]]),
            'xlabel': val_names[0],
            'xunit': self.raw_data_dict['value_units'][0],
            'ylabel': val_names[1],
            'yunit': self.raw_data_dict['value_units'][1],
            'title':self.raw_data_dict['timestamp_string']+'\n'+self.raw_data_dict['measurementstring'] +' hexbin plot',
            'plotsize':(fs[0]*1.5, fs[1])
            }


        for i, val_name in enumerate(val_names):
            self.plot_dicts['raw_RB_curve_data_{}'.format(val_name)] = {
                    'plotfn': plot_raw_RB_curve,
                    'ncl': self.proc_data_dict['ncl'],
                    'SI': self.proc_data_dict['SI'][val_name],
                    'SX': self.proc_data_dict['SX'][val_name],
                    'V0': self.proc_data_dict['V0'][val_name],
                    'V1': self.proc_data_dict['V1'][val_name],
                    'V2': self.proc_data_dict['V2'][val_name],

                    'xlabel': 'Number of Cliffrods',
                    'xunit': '#',
                    'ylabel': val_name,
                    'yunit': self.proc_data_dict['value_units'][i],
                'title': self.proc_data_dict['timestamp_string']+'\n'+self.proc_data_dict['measurementstring'],
            }



class InterleavedTwoQubitRB_Analyasis(ba.BaseDataAnalysis):
    def __init__(self, t_start: str, t_stop: str, label='arc',
                 options_dict: dict=None,
                 f_demod: float=0, demodulate: bool=False, auto=True):
        if options_dict is None:
            options_dict = dict()
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict)
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.raw_data_dict = OrderedDict()

        self.timestamps = a_tools.get_timestamps_in_range(
                self.t_start, self.t_stop,
                label=self.labels)

        a = ma_old.MeasurementAnalysis(timestamp=self.timestamps[0], auto=False, close_file=False)
        a.get_naming_and_values()

        self.raw_data_dict['ncl'] = a.sweep_points
        self.raw_data_dict['q0_base'] = a.measured_values[0]
        self.raw_data_dict['q0_inter'] = a.measured_values[1]
        self.raw_data_dict['q1_base'] = a.measured_values[2]
        self.raw_data_dict['q1_inter'] = a.measured_values[3]
        self.raw_data_dict['data'] = []
        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish() # closes data file


    def process_data(self):
        dd = self.raw_data_dict
        # converting to survival probabilities
        for frac in ['q0_base', 'q1_base', 'q0_inter', 'q1_inter']:
            self.proc_data_dict['p_{}'.format(frac)] = 1-dd[frac]
            self.proc_data_dict['p_{}'.format(frac)] = 1-dd[frac]
        self.proc_data_dict['p_00_base'] = (1-dd['q0_base'])*(1-dd['q1_base'])
        self.proc_data_dict['p_00_inter'] = (1-dd['q0_inter'])*(1-dd['q1_inter'])



    def prepare_plots(self):
        self.plot_dicts['freqs'] = {
            'plotfn': self.dac_arc_ana.plot_freqs,
            'title':"Cryoscope arc \n"+self.timestamps[0]+' - '+self.timestamps[-1]}

        self.plot_dicts['FluxArc'] = {
            'plotfn': self.dac_arc_ana.plot_ffts,
            'title':"Cryoscope arc \n"+self.timestamps[0]+' - '+self.timestamps[-1]}

def plot_cal_points_hexbin(shots_0,
                            shots_1,
                            shots_2,
                            xlabel:str, xunit:str,
                            ylabel:str, yunit:str,
                            title:str,
                            ax, **kw):
    # Choose colormap
    alpha_cmaps = []
    for cmap in [pl.cm.Blues, pl.cm.Reds, pl.cm.Greens]:
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        alpha_cmaps.append(my_cmap)

    f=plt.gcf()
    hb = ax.hexbin(x=shots_2[0], y=shots_2[1], cmap=alpha_cmaps[2])
    clim = hb.get_clim()

    cb = f.colorbar(hb, ax=ax)
    cb.set_label(r'Counts $|2\rangle$')
    hb = ax.hexbin(x=shots_1[0], y=shots_1[1], cmap=alpha_cmaps[1])
    hb.set_clim(clim)
    cb = f.colorbar(hb, ax=ax)
    cb.set_label(r'Counts $|1\rangle$')
    hb = ax.hexbin(x=shots_0[0], y=shots_0[1], cmap=alpha_cmaps[0])
    hb.set_clim(clim)
    cb = f.colorbar(hb, ax=ax)
    cb.set_label(r'Counts $|0\rangle$')

    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.set_title(title)

def plot_raw_RB_curve(ncl, SI, SX, V0, V1, V2, title, ax,
                      xlabel, xunit, ylabel, yunit,**kw):
    ax.plot(ncl,SI, label='SI', marker='o')
    ax.plot(ncl,SX, label='SX', marker='o')
    ax.plot(ncl[-1]+.5, V0, label='V0', marker='o', c='C0')
    ax.plot(ncl[-1]+1.5, V1, label='V1', marker='o', c='C1')
    ax.plot(ncl[-1]+2.5, V2, label='V2', marker='o', c='C2')
    ax.set_title(title)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    ax.legend()




def populations_using_rate_equations(SI:np.array , SX:np.array ,
                                     V0:float , V1:float , V2:float):
    """
    Args:
        SI (float): signal value for signal with I (Identity) added
        SX (float): signal value for signal with X (π-pulse) added
        V0 (float):
        V1 (float):
        V2 (float):

    Based on equation (S1) from Asaad & Dickel et al. npj Quant. Info. (2016)

    To quantify leakage, we monitor the populations Pi of the three lowest
    energy states (i ∈ {0, 1, 2}) and calculate the average
    values <Pi>. To do this, we calibrate the average signal levels Vi for
    the transmons in level i, and perform each measurement twice, the second
    time with an added final π pulse on the 0–1 transition. This final π
    pulse swaps P0 and P1, leaving P2 unaffected. Under the assumption that higher levels are unpopulated (P0 +P1 +P2 = 1),

     [V0 −V2,   V1 −V2] [P0]  = [S −V2]
     [V1 −V2,   V0 −V2] [P1]  = [S' −V2]

    where S (S') is the measured signal level without (with) final π pulse. The populations are extracted by matrix inversion.
    """
    M =np.array([[V0-V2, V1-V2], [V1-V2, V0-V2]])
    M_inv = np.linalg.inv(M)

    P0 = np.zeros(len(SI))
    P1 = np.zeros(len(SX))
    for i, (sI, sX) in enumerate(zip(SI, SX)):
        p0, p1 = np.dot(np.array([sI-V2, sX-V2]), M_inv)
        p0, p1 = np.dot(M_inv, np.array([sI-V2, sX-V2]))
        P0[i] = p0
        P1[i] = p1

    P2 = 1- P0 - P1

    return P0, P1, P2, M_inv

