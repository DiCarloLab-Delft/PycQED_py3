import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.fitting_models import Qubit_dac_to_freq
import lmfit
from copy import deepcopy
from pycqed.analysis_v2.base_analysis import plot_scatter_errorbar_fit, plot_scatter_errorbar


class BasicDACvsFrequency(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 auto: bool = True,
                 data_file_path: str = None,
                 close_figs: bool = True,
                 options_dict: dict = None, extract_only: bool = False,
                 do_fitting: bool = False):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only)

        self.params_dict = {'freq_label': self.options_dict.get('sweep_name_key', 'sweep_name'),
                            'freq_unit': self.options_dict.get('sweep_unit_key', 'sweep_unit'),
                            'measurementstring': 'measurementstring',
                            'freq': self.options_dict.get('sweep_points_key', 'sweep_points'),
                            'amp': self.options_dict.get('amp_key', 'amp'),
                            'phase': self.options_dict.get('phase_key', 'phase'),
                            'dac': self.options_dict.get('dac_key', 'Instrument settings.fluxcurrent.VFCQ6'),
                            }
        self.numeric_params = ['freq', 'amp', 'phase', 'dac']

        self.extract_fitparams = self.options_dict.get('fitparams', True)
        if self.extract_fitparams:
            p = self.options_dict.get('fitparams_key', 'Fitted Params distance.f0.value')
            self.params_dict.update({'fitparams': p})
            self.numeric_params.append('fitparams')

        temp_keys = self.options_dict.get('temp_keys', {})
        self.temperature_plots = len(temp_keys) > 1
        if self.temperature_plots:
            for temp_key in temp_keys:
                self.params_dict[temp_key] = temp_keys[temp_key]
                self.numeric_params.append(temp_key)

        if auto:
            self.run_analysis()

    def process_data(self):
        # sort data
        self.proc_data_dict = {}
        dac_values_unsorted = np.array(self.raw_data_dict['dac'])
        sorted_indices = dac_values_unsorted.argsort()
        self.proc_data_dict['dac_values'] = np.array(dac_values_unsorted[sorted_indices], dtype=float)
        self.proc_data_dict['amplitude_values'] = np.array(self.raw_data_dict['amp'][sorted_indices], dtype=float)
        self.proc_data_dict['phase_values'] = np.array(self.raw_data_dict['phase'][sorted_indices], dtype=float)
        self.proc_data_dict['frequency_values'] = np.array(self.raw_data_dict['freq'][sorted_indices], dtype=float)

        deg_factor = np.pi / 180
        rad = self.proc_data_dict['phase_values'] * deg_factor
        compl = self.proc_data_dict['amplitude_values'] * (np.cos(rad) + 1j * np.sin(rad))
        self.proc_data_dict['distance_values'] = a_tools.calculate_distance_ground_state(
            data_real=compl.real, data_imag=compl.imag, percentile=70, normalize=True)

        if self.extract_fitparams:
            self.proc_data_dict['fit_frequencies'] = np.array(self.raw_data_dict['fitparams'][sorted_indices],
                                                              dtype=float)

        if self.temperature_plots:
            self.proc_data_dict['T_mc'] = np.array(self.raw_data_dict['T_mc'][sorted_indices], dtype=float)
            self.proc_data_dict['T_cp'] = np.array(self.raw_data_dict['T_cp'][sorted_indices], dtype=float)

        # Smooth data and find peeks
        for i, dac_value in enumerate(self.proc_data_dict['dac_values']):
            peaks_x, peaks_z, smoothed_z = a_tools.peak_finder_v3(self.proc_data_dict['frequency_values'][i],
                                                                  self.proc_data_dict['amplitude_values'][i],
                                                                  smoothing=self.options_dict.get('smoothing', False),
                                                                  perc=self.options_dict.get('preak_perc', 99),
                                                                  window_len=self.options_dict.get('smoothing_win_len',
                                                                                                   False),
                                                                  factor=self.options_dict.get('data_factor', 1))
            peaks_x, peaks_z, smoothed_z = a_tools.peak_finder_v3(self.proc_data_dict['frequency_values'][i],
                                                                  self.proc_data_dict['phase_values'][i],
                                                                  smoothing=self.options_dict.get('smoothing', False),
                                                                  perc=self.options_dict.get('preak_perc', 99),
                                                                  window_len=self.options_dict.get('smoothing_win_len',
                                                                                                   False),
                                                                  factor=self.options_dict.get('data_factor', 1))
            # Fixme: smooth and peak-find s21 distance
            # Fixme: save smoothed data and peaks

            # save peaks and smoothed data
            # smoothed_amplitude_values[i, :] = smoothed_z
            # peak_frequencies[i] = peaks_x
            # peak_amplitudes[i] = peaks_z

    def run_fitting(self):
        self.fit_dicts = {}
        fit_result = fit_qubit_dac_arch(freq=self.proc_data_dict['fit_frequencies'],
                                        dac=self.proc_data_dict['dac_values'])
        self.fit_dicts['fit_result'] = fit_result
        # self.fit_dicts['E_c'] = fit_result.params['E_c']
        # self.fit_dicts['f_max'] = fit_result.params['f_max']
        # self.fit_dicts['dac_sweet_spot'] = fit_result.params['dac_sweet_spot']
        # self.fit_dicts['dac_per_phi0'] = fit_result.params['V_per_phi0']
        # self.fit_dicts['asymmetry'] = fit_result.params['asymmetry']

    def prepare_plots(self):
        fit_result = self.fit_dicts['fit_result']
        if self.options_dict.get('plot_vs_flux', False):
            factor = fit_result.params['V_per_phi0']
        else:
            factor = 1

        x = self.proc_data_dict['dac_values'] * self.options_dict.get('current_multiplier', 1)
        y = self.proc_data_dict['frequency_values']

        twoDPlot = {'plotfn': self.plot_colorx,
                    'xvals': x,
                    'yvals': y,
                    'title': 'Flux Current Spectroscopy Sweep',
                    'xlabel': r'Flux bias current, I',
                    'xunit': 'A',
                    'ylabel': r'Frequency',
                    'yunit': 'Hz',
                    # 'zrange': [smoothed_amplitude_values.min(), smoothed_amplitude_values.max()],
                    'xrange': [np.min(x), np.max(x)],
                    'yrange': [np.min(y), np.max(y)],
                    'plotsize': self.options_dict.get('plotsize', None),
                    'cmap': self.options_dict.get('cmap', 'YlGn_r'),
                    'plot_transpose': self.options_dict.get('plot_transpose', False),
                    }

        if self.do_fitting:
            fit = {
                'plotfn': self.plot_fit,
                'fit_res': fit_result,
                'xvals': self.proc_data_dict['dac_values'],
                'yvals': self.proc_data_dict['fit_frequencies'],
                'marker': '',
                'linestyle': '-',
            }

        scatter = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['dac_values'],
            'yvals': self.proc_data_dict['fit_frequencies'],
            'marker': 'x',
            'linestyle': 'None',
        }

        for ax in ['amplitude', 'phase', 'distance']:
            z = self.proc_data_dict['%s_values' % ax]
            td = deepcopy(twoDPlot)
            td['zvals'] = z
            td['zlabel'] = ax
            self.plot_dicts[ax] = td

            sc = deepcopy(scatter)
            sc['ax_id'] = ax
            self.plot_dicts[ax + '_scatter'] = sc

            if self.do_fitting:
                f = deepcopy(fit)
                f['ax_id'] = ax
                self.plot_dicts[ax + '_fit'] = f


def fit_qubit_dac_arch(freq, dac):
    arch_model = lmfit.Model(Qubit_dac_to_freq)
    arch_model.set_param_hint('E_c', value=260e6, min=100e6, max=350e6)
    arch_model.set_param_hint('f_max', value=6e9, min=0.1e9, max=10e9)
    arch_model.set_param_hint('dac_sweet_spot', value=0, min=-0.5, max=0.5)
    arch_model.set_param_hint('V_per_phi0', value=0.1, min=0)
    arch_model.set_param_hint('asymmetry', value=0)

    arch_model.make_params()

    fit_result = arch_model.fit(freq, dac_voltage=dac)
    return fit_result
