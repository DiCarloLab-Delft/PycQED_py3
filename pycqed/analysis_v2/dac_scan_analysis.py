import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.fitting_models import Qubit_dac_to_freq
import lmfit
from copy import deepcopy
from pycqed.analysis_v2.base_analysis import plot_scatter_errorbar_fit, plot_scatter_errorbar


class FluxFrequency(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 auto: bool = True,
                 data_file_path: str = None,
                 close_figs: bool = True,
                 options_dict: dict = None, extract_only: bool = False,
                 do_fitting: bool = False,
                 is_spectroscopy: bool = True,
                 extract_fitparams: bool = True,
                 temp_keys: dict = None,
                 ):
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

        self.is_spectroscopy = is_spectroscopy
        self.extract_fitparams = extract_fitparams
        if extract_fitparams:
            if is_spectroscopy:
                default_key = 'Fitted Params distance.f0.value'
            else:
                default_key = 'Fitted Params HM.f0.value'

            p = self.options_dict.get('fitparams_key', default_key)
            self.params_dict['fitparams'] = p
            self.numeric_params.append('fitparams')

        self.temp_keys = temp_keys
        self.temperature_plots = False if temp_keys is None else len(temp_keys) >= 1
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
        self.proc_data_dict['datetime'] = np.array(self.raw_data_dict['datetime'])[sorted_indices]
        # Do we have negative angles?
        negative_angles = np.min(self.proc_data_dict['phase_values']) < 0
        if negative_angles:
            tpi = np.pi
        else:
            tpi = 2 * np.pi
        angle_type_deg_guess = np.max(np.abs(self.proc_data_dict['phase_values'])) > tpi

        if self.options_dict.get('phase_in_rad', False):
            deg_factor = 1
            if angle_type_deg_guess:
                print('Warning: Assuming degrees as unit for Phase, but it does not seem to be in radians, '
                      + 'consider changing the  phase_in_rad entry in the options dict accordingly')
        else:
            deg_factor = np.pi / 180
            if not angle_type_deg_guess:
                print('Warning: Assuming degrees as unit for Phase, but it might not be - '
                      + 'consider changing the  phase_in_rad entry in the options dict accordingly')

        rad = self.proc_data_dict['phase_values'] * deg_factor
        real = self.proc_data_dict['amplitude_values'] * np.cos(rad)
        imag = self.proc_data_dict['amplitude_values'] * np.sin(rad)
        self.proc_data_dict['distance_values'] = [a_tools.calculate_distance_ground_state(
            data_real=real[i], data_imag=imag[i], percentile=self.options_dict.get('s21_percentile', 70),
            normalize=self.options_dict.get('s21_normalize_per_dac', False)) for i, v in
            enumerate(self.proc_data_dict['dac_values'])]

        if self.options_dict.get('s21_normalize_global', True):
            temp = self.proc_data_dict['distance_values'] = self.proc_data_dict['distance_values']
            temp = temp / np.max(temp)
            del temp

        if self.extract_fitparams:
            corr_f = self.options_dict.get('fitparams_corr_fact', 1)
            self.proc_data_dict['fit_frequencies'] = self.raw_data_dict['fitparams'][sorted_indices] * corr_f

        if self.temperature_plots:
            for k in self.temp_keys:
                self.proc_data_dict[k] = np.array(self.raw_data_dict[k][sorted_indices], dtype=float)

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
        self.fit_result = {}
        if self.is_spectroscopy:
            default_fitfunc = fit_qubit_dac_arch
        else:
            default_fitfunc = fit_resonator_dac_arch

        fitfunc = self.options_dict.get('fitfunc', default_fitfunc)
        fit_result = fitfunc(freq=self.proc_data_dict['fit_frequencies'], dac=self.proc_data_dict['dac_values'])

        self.fit_result['dac_arc'] = fit_result
        # self.fit_dicts['E_c'] = fit_result.params['E_c']
        # self.fit_dicts['f_max'] = fit_result.params['f_max']
        # self.fit_dicts['dac_sweet_spot'] = fit_result.params['dac_sweet_spot']
        # self.fit_dicts['dac_per_phi0'] = fit_result.params['V_per_phi0']
        # self.fit_dicts['asymmetry'] = fit_result.params['asymmetry']

    def prepare_plots(self):
        if self.options_dict.get('plot_vs_flux', False) and hasattr(self,
                                                                    'fit_result') and 'dac_arc' in self.fit_result:
            factor = self.fit_result['dac_arc'].params['V_per_phi0']
        else:
            factor = 1

        x = self.proc_data_dict['dac_values'] * self.options_dict.get('current_multiplier', 1)
        y = self.proc_data_dict['frequency_values']

        if self.is_spectroscopy:
            s = 'Spectroscopy'
        else:
            s = 'Resonator'

        twoDPlot = {'plotfn': self.plot_colorx,
                    'xvals': x,
                    'yvals': y,
                    'title': 'Flux Current ' + s + ' Sweep',
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

        scatter = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['dac_values'],
            'yvals': self.proc_data_dict['fit_frequencies'],
            'marker': 'x',
            'linestyle': 'None',
        }

        if self.do_fitting:
            fit_result = self.fit_result['dac_arc']
            fit = {
                'plotfn': self.plot_fit,
                'fit_res': fit_result,
                'xvals': self.proc_data_dict['dac_values'],
                'yvals': self.proc_data_dict['fit_frequencies'],
                'marker': '',
                'linestyle': '-',
            }

        for ax in ['amplitude', 'phase', 'distance']:
            z = self.proc_data_dict['%s_values' % ax]
            td = deepcopy(twoDPlot)
            td['zvals'] = z
            unit = ' (a.u.)'
            if ax == 'phase':
                if self.options_dict.get('phase_in_rad', False):
                    unit = ' (rad.)'
                else:
                    unit = ' (deg.)'
            elif ax == 'distance':
                if self.options_dict.get('s21_normalize_global', False):
                    unit = ' (norm.)'
            td['zlabel'] = ax + unit
            td['ax_id'] = ax
            self.plot_dicts[ax] = td

            sc = deepcopy(scatter)
            sc['ax_id'] = ax
            self.plot_dicts[ax + '_scatter'] = sc

            if self.do_fitting:
                f = deepcopy(fit)
                f['ax_id'] = ax
                self.plot_dicts[ax + '_fit'] = f

        # Now plot temperatures
        if self.temperature_plots:
            for k in self.temp_keys:
                temp_dict = {
                    'plotfn': self.plot_line,
                    'xvals': x,
                    'yvals': self.proc_data_dict[k],
                    'title': 'Fridge Temperature during Flux Current Sweep',
                    'xlabel': r'Flux bias current, I',
                    'xunit': 'A',
                    'ylabel': r'Temperature',
                    'yunit': 'K',
                    'marker': 'x',
                    'linestyle': '-',
                    'do_legend': True,
                    'setlabel': k,
                }
                t = deepcopy(temp_dict)
                t['ax_id'] = 'temperature_dac_relation'
                self.plot_dicts['temperature_' + k + '_dac_relation'] = t

                t = deepcopy(temp_dict)
                t['xvals'] = self.proc_data_dict['datetime']
                t['ax_id'] = 'temperature_time_relation'
                t['xlabel'] = r'Time in Delft'
                t['xunit'] = ''
                self.plot_dicts['temperature_' + k + '_time_relation'] = t


def fit_qubit_dac_arch(freq, dac):
    arch_model = lmfit.Model(Qubit_dac_to_freq)
    arch_model.set_param_hint('E_c', value=260e6, min=100e6, max=400e6)
    fmax = np.max(freq)
    dacs_ss = dac[np.where(freq >= 0.99 * fmax)]
    dac_ss = dac[np.argmin(np.abs(dacs_ss))]
    arch_model.set_param_hint('f_max', value=fmax, min=0.8 * fmax, max=1.2 * fmax)
    arch_model.set_param_hint('dac_sweet_spot', value=dac_ss, min=-0.5, max=0.5)
    arch_model.set_param_hint('V_per_phi0', value=0.04, min=0.001, max=100)
    arch_model.set_param_hint('asymmetry', value=0)

    arch_model.make_params()

    fit_result = arch_model.fit(freq, dac_voltage=dac)
    return fit_result


def fit_resonator_dac_arch(freq, dac):
    # fixme!!!!
    arch_model = lmfit.Model(Qubit_dac_to_freq)
    arch_model.set_param_hint('E_c', value=260e6, min=100e6, max=350e6)
    arch_model.set_param_hint('f_max', value=5e9, min=0.1e9, max=10e9)
    arch_model.set_param_hint('dac_sweet_spot', value=0, min=-0.5, max=0.5)
    arch_model.set_param_hint('V_per_phi0', value=0.1, min=0)
    arch_model.set_param_hint('asymmetry', value=0)

    arch_model.make_params()

    fit_result = arch_model.fit(freq, dac_voltage=dac)
    return fit_result
