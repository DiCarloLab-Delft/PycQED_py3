'''
Hacked together by Rene Vollmer
'''
import datetime

import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.fitting_models import Qubit_dac_to_freq, Resonator_dac_to_freq, Qubit_dac_arch_guess, \
    Resonator_dac_arch_guess
import lmfit
from copy import deepcopy


class FluxFrequency(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 auto: bool = True,
                 data_file_path: str = None,
                 close_figs: bool = True,
                 options_dict: dict = None, extract_only: bool = False,
                 do_fitting: bool = True,
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
        self.proc_data_dict['dac_values'] = dac_values_unsorted[sorted_indices]

        temp = {
            'amp': 'amplitude_values',
            'phase': 'phase_values',
            'freq': 'frequency_values'
        }
        for k in temp:
            self.proc_data_dict[temp[k]] = self._sort_by_axis0(self.raw_data_dict[k], sorted_indices)
        self.proc_data_dict['datetime'] = self._sort_by_axis0(self.raw_data_dict['datetime'], sorted_indices,
                                                              type=datetime.datetime)
        # Do we have negative angles?
        negative_angles = self._globalmin(self.proc_data_dict['phase_values']) < 0
        if negative_angles:
            tpi = np.pi
        else:
            tpi = 2 * np.pi
        angle_type_deg_guess = np.max([np.max(np.abs(i)) for i in self.proc_data_dict['phase_values']]) > tpi

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

        rad = [i * deg_factor for i in self.proc_data_dict['phase_values']]
        real = [self.proc_data_dict['amplitude_values'][j] * np.cos(i) for j, i in enumerate(rad)]
        imag = [self.proc_data_dict['amplitude_values'][j] * np.sin(i) for j, i in enumerate(rad)]
        self.proc_data_dict['distance_values'] = [a_tools.calculate_distance_ground_state(
            data_real=real[i], data_imag=imag[i], percentile=self.options_dict.get('s21_percentile', 70),
            normalize=self.options_dict.get('s21_normalize_per_dac', False)) for i, v in
            enumerate(self.proc_data_dict['dac_values'])]

        if self.options_dict.get('s21_normalize_global', True):
            self.proc_data_dict['distance_values'] = [temp / np.max(temp) for temp in
                                                      self.proc_data_dict['distance_values']]

        if self.extract_fitparams:
            corr_f = self.options_dict.get('fitparams_corr_fact', 1)
            self.proc_data_dict['fit_frequencies'] = self.raw_data_dict['fitparams'][sorted_indices] * corr_f

        if self.temperature_plots:
            for k in self.temp_keys:
                self.proc_data_dict[k] = np.array(self.raw_data_dict[k][sorted_indices], dtype=float)

        # Smooth data and find peeks
        smooth = self.options_dict.get('smoothing', False)
        freqs = self.proc_data_dict['frequency_values']
        for k in ['amplitude_values', 'phase_values', 'distance_values']:
            self.proc_data_dict[k + '_smooth'] = {}
            for i, dac_value in enumerate(self.proc_data_dict['dac_values']):
                peaks_x, peaks_z, smoothed_z = a_tools.peak_finder_v3(freqs[i],
                                                                      self.proc_data_dict[k][i],
                                                                      smoothing=smooth,
                                                                      perc=self.options_dict.get('peak_perc', 99),
                                                                      window_len=self.options_dict.get(
                                                                          'smoothing_win_len',
                                                                          False),
                                                                      factor=self.options_dict.get('data_factor', 1))
                self.proc_data_dict[k + '_smooth'][i] = smoothed_z

                # Fixme: save peaks

    def run_fitting(self):
        self.fit_dicts = {}
        self.fit_result = {}

        dac_vals = self.proc_data_dict['dac_values']
        freq_vals = self.proc_data_dict['fit_frequencies']

        f_q = self.options_dict.get('qubit_freq', None)
        ext = f_q is not None
        if self.is_spectroscopy:
            fitmod = lmfit.Model(Qubit_dac_to_freq)
            fitmod.guess = Qubit_dac_arch_guess.__get__(fitmod, fitmod.__class__)
            fitmod.guess(freq=freq_vals, dac_voltage=dac_vals)
        else:
            if f_q is None and self.verbose:
                print('Specify qubit_freq in the options_dict to obtain a better fit!')

            # Todo: provide alternative fit?
            fitmod = lmfit.Model(Resonator_dac_to_freq)
            fitmod.guess = Resonator_dac_arch_guess.__get__(fitmod, fitmod.__class__)
            fitmod.guess(freq=freq_vals, dac_voltage=dac_vals, f_max_qubit=f_q)

        fit_result = fitmod.fit(freq_vals, dac_voltage=dac_vals)

        self.fit_result['dac_arc'] = fit_result
        EC = fit_result.params['E_c']
        if self.is_spectroscopy:
            f0 = fit_result.params['f_max']
        else:
            f0 = fit_result.params['f_max_qubit']

        self.fit_dicts['E_C'] = EC.value
        self.fit_dicts['E_J'] = (f0.value ** 2 + 2 * EC.value * f0.value + EC.value ** 2) / (8 * EC.value)
        self.fit_dicts['f_sweet_spot'] = f0.value
        self.fit_dicts['dac_sweet_spot'] = fit_result.params['dac_sweet_spot'].value
        self.fit_dicts['dac_per_phi0'] = fit_result.params['V_per_phi0'].value
        self.fit_dicts['asymmetry'] = fit_result.params['asymmetry'].value

        self.fit_dicts['E_C_std'] = EC.stderr
        self.fit_dicts['E_J_std'] = -1  # (f0 ** 2 + 2 * EC * f0 + EC ** 2) / (8 * EC)
        self.fit_dicts['f_sweet_spot_std'] = f0.stderr
        self.fit_dicts['dac_sweet_spot_std'] = fit_result.params['dac_sweet_spot'].stderr
        self.fit_dicts['dac_per_phi0_std'] = fit_result.params['V_per_phi0'].stderr
        self.fit_dicts['asymmetry_std'] = fit_result.params['asymmetry'].stderr

        if not self.is_spectroscopy:
            g = fit_result.params['coupling']
            fr = fit_result.params['f_0_res']
            self.fit_dicts['coupling'] = g.value
            self.fit_dicts['coupling_std'] = g.stderr
            self.fit_dicts['f_0_res'] = fr.value
            self.fit_dicts['f_0_res_std'] = fr.stderr

    def prepare_plots(self):
        plot_vs_flux = self.options_dict.get('plot_vs_flux', False)
        custom_multiplier = self.options_dict.get('current_multiplier', 1)
        fitted = hasattr(self, 'fit_result') and 'dac_arc' in self.fit_result
        plot_vs_flux = plot_vs_flux and fitted and (custom_multiplier == 1)

        flux_factor = 1
        if plot_vs_flux:
            flux_factor = self.fit_result['dac_arc'].params['V_per_phi0']

        if plot_vs_flux:
            cm = flux_factor
        else:
            cm = custom_multiplier

        current_label = 'Flux bias current, I'
        current_unit = 'A'
        if plot_vs_flux:
            current_label = 'Flux'
            current_unit = r'$\Phi_0$'

        x = self.proc_data_dict['dac_values'] * cm
        y = self.proc_data_dict['frequency_values']

        if self.is_spectroscopy:
            s = 'Spectroscopy'
        else:
            s = 'Resonator'

        twoDPlot = {'plotfn': self.plot_colorx,
                    'xvals': x,
                    'yvals': y,
                    'title': 'Flux Current ' + s + ' Sweep',
                    'xlabel': current_label,
                    'xunit': current_unit,
                    'ylabel': r'Frequency',
                    'yunit': 'Hz',
                    # 'zrange': [smoothed_amplitude_values.min(), smoothed_amplitude_values.max()],
                    'xrange': [self._globalmin(x), self._globalmax(x)],
                    'yrange': [self._globalmin(y), self._globalmax(y)],
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
                'xvals': self.proc_data_dict['dac_values'] * cm,
                'yvals': self.proc_data_dict['fit_frequencies'],
                'marker': '',
                'linestyle': '-',
            }

        ext = self.options_dict.get('qubit_freq', None) is not None
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

            if hasattr(self, 'fit_dicts') and self.options_dict.get('print_fit_result_plot', True):
                dac_fit_text = ''
                # if ext or self.is_spectroscopy:
                dac_fit_text += '$E_C/2 \pi = %.2f(\pm %.3f)$ MHz\n' % (
                    self.fit_dicts['E_C'] * 1e-6, self.fit_dicts['E_C_std'] * 1e-6)
                dac_fit_text += '$E_J/\hbar = %.2f$ GHz\n' % (
                        self.fit_dicts['E_J'] * 1e-9)  # , self.fit_dicts['E_J_std'] * 1e-9
                dac_fit_text += '$\omega_{ss}/2 \pi = %.2f(\pm %.3f)$ GHz\n' % (
                    self.fit_dicts['f_sweet_spot'] * 1e-9, self.fit_dicts['f_sweet_spot_std'] * 1e-9)
                dac_fit_text += '$I_{ss}/2 \pi = %.2f(\pm %.3f)$ mA\n' % (
                    self.fit_dicts['dac_sweet_spot'] * custom_multiplier * 1e3,
                    self.fit_dicts['dac_sweet_spot_std'] * custom_multiplier * 1e3)
                dac_fit_text += '$I/\Phi_0 = %.2f(\pm %.3f)$ mA/$\Phi_0$' % (
                    self.fit_dicts['dac_per_phi0'] * custom_multiplier * 1e3,
                    self.fit_dicts['dac_per_phi0_std'] * custom_multiplier * 1e3)

                if not self.is_spectroscopy:
                    dac_fit_text += '\n$g/2 \pi = %.2f(\pm %.3f)$ MHz\n' % (
                        self.fit_dicts['coupling'] * 1e-6, self.fit_dicts['coupling_std'] * 1e-6)
                    dac_fit_text += '$\omega_{r,0}/2 \pi = %.2f(\pm %.3f)$ GHz' % (
                        self.fit_dicts['f_0_res'] * 1e-9, self.fit_dicts['f_0_res_std'] * 1e-9)
                self.plot_dicts['text_msg_' + ax] = {
                    'ax_id': ax,
                    # 'ypos': 0.15,
                    'plotfn': self.plot_text,
                    'box_props': 'fancy',
                    'text_string': dac_fit_text,
                }

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
