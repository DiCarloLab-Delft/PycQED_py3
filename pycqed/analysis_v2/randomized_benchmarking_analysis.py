import lmfit
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
                self.raw_data_dict['cal_pts_zero'][val_name] = binned_yvals[-6:-4, :]
                self.raw_data_dict['cal_pts_one'][val_name] = binned_yvals[-4:-2, :]
                self.raw_data_dict['cal_pts_two'][val_name] = binned_yvals[-2:, :]
                self.raw_data_dict['measured_values_I'][val_name] = binned_yvals[:-6:2, :]
                self.raw_data_dict['measured_values_X'][val_name] = binned_yvals[1:-6:2, :]

        else:
            bins = None

        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish() # closes data file

    def prepare_plots(self):
        for i, val_name in enumerate(self.raw_data_dict['value_names']):
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
