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
from pycqed.analysis import fitting_models as fit_mods
import lmfit
from copy import deepcopy
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor


class ResZZAnalysis(ba.BaseDataAnalysis):
    def __init__(
            self,
            ts: str = None,
            label: str = "Residual_ZZ_",
            data_file_path: str = None,
            options_dict: dict = None,
            extract_only: bool = False,
            close_figs=True,
            do_fitting: bool = True,
            auto=True,
            artificial_detuning: float = None
    ):
        super().__init__(t_start=ts, t_stop=ts,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         close_figs=close_figs,
                         extract_only=extract_only, do_fitting=do_fitting)

        # if artificial_detuning is None:
        #     artificial_detuning = 0
        # self.artificial_detuning = artificial_detuning
        self.get_timestamps()
        for ts in self.timestamps:
            self.timestamp = ts
            if auto:
                self.run_analysis()

    def extract_data(self):
        """
        Extracts the data from the hdf5 file and saves it in a dictionary under the key "data".
        The dictionary self.raw_data_dict also contains entries for the timestamp, folder and value names.
        """

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}

        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

        self.raw_data_dict["sweep_points"] = self.raw_data_dict['data'][:, 0]

    def process_data(self):
        """
        Use the calibration points to rotate and normalise the data for both qubits.
        """
        self.proc_data_dict = {}
        for i in np.arange(1, int((len(self.raw_data_dict['value_names'])+1)), 2):
            qubit_name = self.raw_data_dict['value_names'][i-1][-4:-2].decode('ascii')
            self.proc_data_dict[qubit_name] = {}
            self.proc_data_dict[qubit_name]['qubit_type'] = 'control' if i == 1 else 'spec'
            self.proc_data_dict[qubit_name]['times'] = self.raw_data_dict['data'][:, 0]
            self.proc_data_dict[qubit_name]['data_I'] = self.raw_data_dict['data'][:,i]
            self.proc_data_dict[qubit_name]['data_Q'] = self.raw_data_dict['data'][:, i + 1]

            cal_zero_points = slice(-4, -3)
            cal_one_points = slice(-2, -1)

            self.proc_data_dict[qubit_name]['normalised_data'] = ma.a_tools.rotate_and_normalize_data(
                [self.proc_data_dict[qubit_name]['data_I'], self.proc_data_dict[qubit_name]['data_Q']], cal_zero_points,
                cal_one_points)[0]

    def prepare_fitting(self):
        """
        Create initial guess for the fits
        :return:
        """
        for qubit_name in self.proc_data_dict.keys():
            if self.proc_data_dict[qubit_name]['qubit_type'] is not 'control':
                continue

            ft_of_data = np.fft.fft(self.proc_data_dict[qubit_name]['normalised_data'][:-4])
            index_of_fourier_maximum = np.argmax(np.abs(
                ft_of_data[1:len(ft_of_data) // 2])) + 1
            max_delay = self.proc_data_dict[qubit_name]['times'][:-4][-1] - self.proc_data_dict[qubit_name]['times'][:-4][0]

            fft_axis_scaling = 1 / max_delay
            freq_est = fft_axis_scaling * index_of_fourier_maximum

            if (np.average(self.proc_data_dict[qubit_name]['normalised_data'][:4]) >
                    np.average(self.proc_data_dict[qubit_name]['normalised_data'][4:8])):
                phase_estimate = np.pi / 2
            else:
                phase_estimate = - np.pi / 2

            guess_dict = {}
            guess_dict['amplitude'] = {'value': max(self.proc_data_dict[qubit_name]['normalised_data'][:-4]),
                                          'min': 0,
                                          'max':1,
                                          'vary': True}
            guess_dict['oscillation_offset'] = {'value': 0,
                                                 'vary': False}
            guess_dict['n'] = {'value': 1,
                               'vary': False}
            guess_dict['exponential_offset'] = {'value': 0.5,
                                                'min': 0.4,
                                                'max': 0.6,
                                                'vary': True}
            guess_dict['phase'] = {'value': phase_estimate,
                                      'min': phase_estimate-np.pi/4,
                                      'max': phase_estimate+np.pi/4,
                                      'vary': True}
            guess_dict['frequency'] = {'value': freq_est,
                                          'min': (1/(100 * self.proc_data_dict[qubit_name]['times'][:-4][-1])),
                                          'max': (20/self.proc_data_dict[qubit_name]['times'][:-4][-1]),
                                          'vary': True}
            guess_dict['tau'] = {'value': self.proc_data_dict[qubit_name]['times'][1]*10,
                                    'min': self.proc_data_dict[qubit_name]['times'][1],
                                    'max': self.proc_data_dict[qubit_name]['times'][1]*1000,
                                    'vary': True}

            self.fit_dicts['Residual_ZZ_fit'] = {
                'fit_fn': fit_mods.ExpDampOscFunc,
                'guess_dict': guess_dict,
                'fit_xvals': {'t': self.proc_data_dict[qubit_name]['times'][:-4]},
                'fit_yvals': {'data': self.proc_data_dict[qubit_name]['normalised_data'][:-4]},
                'fitting_type':'minimize'
            }

    def prepare_plots(self):
        """
        Create plots
        :return:
        """

        self.raw_data_dict["xlabel"] = r'Idling time before $\pi$ pulse'
        self.raw_data_dict["ylabel"] = "Excited state population"
        self.raw_data_dict["xunit"] = 'us'

        control_qubit = [q for q in self.proc_data_dict.keys() if self.proc_data_dict[q]['qubit_type'] == 'control'][0]
        spec_qubits = [q for q in self.proc_data_dict.keys() if self.proc_data_dict[q]['qubit_type'] == 'spec']
        self.raw_data_dict["measurementstring"] = f'Residual ZZ\necho: {control_qubit}\nspectators: {spec_qubits}'

        for qubit_name in self.proc_data_dict.keys():

            if qubit_name == control_qubit:
                plot_name = f"Control_{qubit_name}"
            else:
                plot_name = f"Spectator_{qubit_name}"
            self.plot_dicts[plot_name] = {
                "plotfn": self.plot_line,
                "xvals": self.raw_data_dict["sweep_points"],
                "xlabel": self.raw_data_dict["xlabel"],
                "xunit": self.raw_data_dict["xunit"],  # does not do anything yet
                "yvals": self.proc_data_dict[qubit_name]["normalised_data"],
                "ylabel": self.raw_data_dict["ylabel"] + f' {qubit_name}',
                "yunit": "",
                "setlabel": "Measured data",
                "title": (
                    self.raw_data_dict["timestamps"][0]
                    + " "
                    + self.raw_data_dict["measurementstring"]
                ),
                "do_legend": True,
                "legend_pos": "upper right",
            }

        self.plot_dicts['osc_exp_fit'] = {
            'ax_id': f"Control_{control_qubit}",
            'plotfn': self.plot_fit,
            'fit_res': self.fit_dicts['Residual_ZZ_fit']['fit_res'],
            'setlabel': 'Oscillation with exponential decay fit',
            'do_legend': True,
            'legend_pos': 'best'}

        fit_res_params = self.fit_dicts['Residual_ZZ_fit']['fit_res'].params
        scale_frequency, unit_frequency = SI_prefix_and_scale_factor(fit_res_params['frequency'].value, 'Hz')
        plot_frequency = fit_res_params['frequency'].value * scale_frequency
        scale_amplitude, unit_amplitude = SI_prefix_and_scale_factor(fit_res_params['amplitude'].value)
        plot_amplitude = fit_res_params['amplitude'].value * scale_amplitude
        scale_tau, unit_tau = SI_prefix_and_scale_factor(fit_res_params['tau'].value, 's')
        plot_tau = fit_res_params['tau'].value * scale_tau
        scale_offset, unit_offset = SI_prefix_and_scale_factor(fit_res_params['exponential_offset'].value)
        plot_offset = fit_res_params['exponential_offset'].value * scale_offset
        scale_phase, unit_phase = SI_prefix_and_scale_factor(fit_res_params['phase'].value, 'rad')
        plot_phase = fit_res_params['phase'].value * scale_phase

        if plot_phase >= 0:
            self.resZZ = -1*plot_frequency
        else:
            self.resZZ = plot_frequency

        self.plot_dicts['ResZZ_box'] = {
            'ax_id': f"Control_{control_qubit}",
            'ypos': .7,
            'xpos': 1.04,
            'plotfn': self.plot_text,
            'dpi': 200,
            'box_props': 'fancy',
            'horizontalalignment': 'left',
            # 'text_string': 'Chi = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].chisqr),
            'text_string': 'Residual ZZ coupling = %.2f ' % (self.resZZ) + unit_frequency
        }

        self.plot_dicts['Parameters'] = {
            'ax_id': f"Control_{control_qubit}",
            'ypos': .5,
            'xpos': 1.04,
            'plotfn': self.plot_text,
            'dpi': 200,
            'box_props': 'fancy',
            'horizontalalignment': 'left',
            # 'text_string': 'Chi = ' + str(self.fit_dicts['ExpGaussDecayCos']['fit_res'].chisqr),
            'text_string': 'Fit results:' + '\n' + '\n'
                           + 'f = %.2f ' % (plot_frequency) + unit_frequency + '\n'
                           + '$\mathrm{\chi}^2$ = %.3f' % (self.fit_dicts['Residual_ZZ_fit']['fit_res'].chisqr) + '\n'
                           + '$\mathrm{T}$ = %.2f ' % (plot_tau) + unit_tau + '\n'
                           + 'A = %.2f ' % (plot_amplitude) + unit_amplitude + '\n'
                           + 'Offset = %.2f ' % (plot_offset) + unit_offset + '\n'
                           + 'Phase = %.2f ' % (plot_phase) + unit_phase
        }

