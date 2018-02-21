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

class RamZFluxArc(ba.BaseDataAnalysis):
    """
    Analysis for the 2D scan that is used to calibrate the FluxArc.

    There exist two variant
        TwoD -> single experiment
        multiple 1D -> combination of several linescans

    This analysis only implements the second variant (as of Feb 2018)
    """

    def __init__(self, t_start: str, t_stop: str, label='arc',
                 options_dict: dict=None,
                 f_demod: float=0, demodulate: bool=False, auto=True):
        if options_dict is None:
            options_dict = dict()

        self.numeric_params = []
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict)

        # Now actually extract the parameters and the data
        self.params_dict = {
            'xlabel': 'sweep_name',
            'xunit': 'sweep_unit',
            'ylabel': 'sweep_name_2D',
            'yunit': 'sweep_unit_2D',
            'measurementstring': 'measurementstring',
            'xvals': 'sweep_points',
            'yvals': 'sweep_points_2D',
            'value_names': 'value_names',
            'value_units': 'value_units',
            'measured_values': 'measured_values',

            # # Qubit parameters
            # 'f_max': '{}.f_max'.format(self.data_dict['qubit_name']),
            # 'E_c': '{}.E_c'.format(self.data_dict['qubit_name']),
            # 'V_offset': '{}.V_offset'.format(self.data_dict['qubit_name']),
            # 'V_per_phi0':
            #     '{}.V_per_phi0'.format(self.data_dict['qubit_name']),
            # 'asymmetry':
            #     '{}.asymmetry'.format(self.data_dict['qubit_name']),
        }
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.

        Overwrite this method if you wnat to

        """
        self.timestamps = a_tools.get_timestamps_in_range(
                self.t_start, self.t_stop,
                label=self.labels)

        self.raw_data_dict = OrderedDict()

        # FIXME: this is hardcoded and should be an argument in options dict
        amp_key = 'Snapshot/instruments/AWG8_8005/parameters/awgs_0_outputs_1_amplitude'

        self.raw_data_dict['amps'] = []
        self.raw_data_dict['data'] = []

        for t in self.timestamps:
            a = ma_old.MeasurementAnalysis(timestamp=t, auto=False, close_file=False)
            a.get_naming_and_values()
            amp = a.data_file[amp_key].attrs['value']
            data = a.measured_values[2] +1j* a.measured_values[3]
            # hacky but required for data saving
            self.raw_data_dict['folder'] = a.folder
            self.raw_data_dict['amps'].append(amp)
            self.raw_data_dict['data'].append(data)


        self.raw_data_dict['times'] = a.sweep_points
        self.raw_data_dict['timestamps'] = self.timestamps


    def process_data(self):
        self.dac_arc_ana = ct.DacArchAnalysis(
            self.raw_data_dict['times'],
            self.raw_data_dict['amps'],
            self.raw_data_dict['data'],
            poly_fit_order=2, plot_fits=False)
        self.proc_data_dict['dac_arc_ana'] = self.dac_arc_ana

        # this is the infamous dac arc conversion method
        # we would like this to be directly accessible
        self.freq_to_amp = self.dac_arc_ana.freq_to_amp



    def prepare_plots(self):
        self.plot_dicts['freqs'] = {
            'plotfn': self.dac_arc_ana.plot_freqs,
            'title':"Cryoscope arc \n"+self.timestamps[0]+' - '+self.timestamps[-1]}

        self.plot_dicts['FluxArc'] = {
            'plotfn': self.dac_arc_ana.plot_ffts,
            'title':"Cryoscope arc \n"+self.timestamps[0]+' - '+self.timestamps[-1]}