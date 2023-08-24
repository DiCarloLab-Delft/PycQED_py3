'''
Hacked together by Rene Vollmer
'''
import datetime
from pycqed.analysis import measurement_analysis as ma
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis_v2.simple_analysis as sa
import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.fitting_models import Qubit_dac_to_freq, Resonator_dac_to_freq, Qubit_dac_arch_guess, \
    Resonator_dac_arch_guess
import lmfit
import matplotlib.pyplot as plt
from scipy import optimize
from collections import OrderedDict
from copy import deepcopy, copy
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis.tools import data_manipulation as dm_tools
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel)
import logging

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
        """
        Class for analysing DAC archs (Qubit or Resonator Frequency vs. DAC current
        or Flux). Fitting is not super stable, so it might be adviseable to manually
        specify a fit_guess dict (inside the options_dict).
        TODO: Implement a rejection/filtering mechanism for outliers (of the fitted frequencies)
        TODO: Use the already implemented peak finder (see process_data)
        TODO: Allow to pass a custom peak finder
        TODO: Allow to not specify the fitparams_key parameter and solely rely on internal fitting
        TODO: Make guess functions better

        :param options_dict: - fit_guess (dict): allows to specify parameters for the inital guess.
                             - plot_guess (bool): if True plot the guess as well
                             - dac_key (str): string for extracting the DAC values
                                    (e.g. 'Instrument settings.fluxcurrent.Q')
                             - amp_key (str): string for extracting the measured amplitude values
                             - phase_key (str): string for extracting the measured phase values
                             - fitparams_key (str): string for extracting the fitted frequency values
                             - phase_in_rad (bool)
                             - s21_normalize_per_dac (bool)
                             - s21_normalize_global (bool)
                             - s21_percentile (float)
                             - plot_vs_flux (bool): plot against flux quanta rather than current?
                             - (+inherited from BaseDataAnalysis)
        :param is_spectroscopy: (bool) Spectoscropy or Resonator measurement?
        :param temp_keys: (dict) dict of strings for extracting temperatures for each DAC value
        (+params inherited from BaseDataAnalysis)
        """
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
                            'dac': self.options_dict.get('dac_key', 'Instrument settings.fluxcurrent.Q'),
                            }
        self.numeric_params = ['freq', 'amp', 'phase', 'dac']

        self.is_spectroscopy = is_spectroscopy
        self.extract_fitparams = extract_fitparams
        if extract_fitparams:
            if is_spectroscopy:
                default_key = 'Analysis.Fitted Params distance.f0.value'
            else:
                default_key = 'Fitted Params HM.f0.value'

            p = self.options_dict.get('fitparams_key', default_key)
            self.params_dict['fitparams'] = p
            self.numeric_params.append('fitparams')

        self.temp_keys = temp_keys
        self.temperature_plots = False if temp_keys is None else len(
            temp_keys) >= 1
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
            self.proc_data_dict[temp[k]] = self._sort_by_axis0(
                self.raw_data_dict[k], sorted_indices)
        self.proc_data_dict['datetime'] = [
            self.raw_data_dict['datetime'][i] for i in sorted_indices]
        # Do we have negative angles?
        negative_angles = self._globalmin(
            self.proc_data_dict['phase_values']) < 0
        if negative_angles:
            tpi = np.pi
        else:
            tpi = 2 * np.pi
        angle_type_deg_guess = np.max(
            [np.max(np.abs(i)) for i in self.proc_data_dict['phase_values']]) > tpi

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
        real = [self.proc_data_dict['amplitude_values']
                [j] * np.cos(i) for j, i in enumerate(rad)]
        imag = [self.proc_data_dict['amplitude_values']
                [j] * np.sin(i) for j, i in enumerate(rad)]
        self.proc_data_dict['distance_values'] = [a_tools.calculate_distance_ground_state(
            data_real=real[i], data_imag=imag[i], percentile=self.options_dict.get(
                's21_percentile', 70),
            normalize=self.options_dict.get('s21_normalize_per_dac', False)) for i, v in
            enumerate(self.proc_data_dict['dac_values'])]

        if self.options_dict.get('s21_normalize_global', True):
            self.proc_data_dict['distance_values'] = [temp / np.max(temp) for temp in
                                                      self.proc_data_dict['distance_values']]

        if self.extract_fitparams:
            corr_f = self.options_dict.get('fitparams_corr_fact', 1)
            self.proc_data_dict['fit_frequencies'] = self.raw_data_dict['fitparams'][sorted_indices] * corr_f

        if np.max(self.proc_data_dict['fit_frequencies']) < 1e9:
            self.proc_data_dict['fit_frequencies'] = self.proc_data_dict['fit_frequencies']*1e9

        if self.temperature_plots:
            for k in self.temp_keys:
                self.proc_data_dict[k] = np.array(
                    self.raw_data_dict[k][sorted_indices], dtype=float)

        # Smooth data and find peeks
        smooth = self.options_dict.get('smoothing', False)
        freqs = self.proc_data_dict['frequency_values']
        # self.proc_data_dict['peaks'] = {}
        for k in ['amplitude_values', 'phase_values', 'distance_values']:
            self.proc_data_dict[k + '_smooth'] = {}
            for i, dac_value in enumerate(self.proc_data_dict['dac_values']):
                peaks_x, peaks_z, smoothed_z = a_tools.peak_finder_v3(
                    freqs[i], self.proc_data_dict[k][i], smoothing=smooth,
                    perc=self.options_dict.get('peak_perc', 99),
                    window_len=self.options_dict.get('smoothing_win_len',
                                                     False),
                    factor=self.options_dict.get('data_factor', 1))
                self.proc_data_dict[k + '_smooth'][i] = smoothed_z

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        dac_vals = self.proc_data_dict['dac_values']
        freq_vals = self.proc_data_dict['fit_frequencies']
        if max(freq_vals) < 1e9:
            freq_vals *= 1e9

        guess = self.options_dict.get('fit_guess', {})
        f_q = guess.get(
            'f_max_qubit', self.options_dict.get('f_max_qubit', None))
        guess['f_max_qubit'] = f_q
        ext = f_q is not None
        if self.is_spectroscopy:
            fitmod = lmfit.Model(Qubit_dac_to_freq)
            fitmod.guess = Qubit_dac_arch_guess.__get__(
                fitmod, fitmod.__class__)
        else:
            if f_q is None and self.verbose:
                print('Specify f_max_qubit in the options_dict to obtain a better fit!')
                # Todo: provide alternative fit?
            fitmod = lmfit.Model(Resonator_dac_to_freq)
            fitmod.guess = Resonator_dac_arch_guess.__get__(
                fitmod, fitmod.__class__)

#        fit_result = fitmod.fit(freq_vals, dac_voltage=dac_vals)

        self.fit_dicts['dac_arc'] = {
            'model': fitmod,
            'fit_xvals': {'dac_voltage': dac_vals},
            'fit_yvals': {'data': freq_vals},
            'guessfn_pars': {'values': guess}
        }

    def analyze_fit_results(self):
        fit_result = self.fit_res['dac_arc']

        EC = fit_result.params['E_c']
        if self.is_spectroscopy:
            f0 = fit_result.params['f_max']
        else:
            f0 = fit_result.params['f_max_qubit']

        # TODO: This is very dirty code! Derived values like E_J should be set as
        # fitmod.set_param_hint('E_J', expr='(f0 ** 2 + 2 * EC * f0 + EC ** 2) / (8 * EC)', vary=False)
        # And fit_res_dicts should not be used!
        self.fit_res_dicts = {}
        self.fit_res_dicts['E_C'] = EC.value
        self.fit_res_dicts['E_J'] = (
            f0.value ** 2 + 2 * EC.value * f0.value + EC.value ** 2) / (8 * EC.value)
        self.fit_res_dicts['f_sweet_spot'] = f0.value
        self.fit_res_dicts['dac_sweet_spot'] = fit_result.params['dac_sweet_spot'].value
        self.fit_res_dicts['dac_per_phi0'] = fit_result.params['V_per_phi0'].value
        self.fit_res_dicts['asymmetry'] = fit_result.params['asymmetry'].value

        self.fit_res_dicts['E_C_std'] = EC.stderr
        # (f0 ** 2 + 2 * EC * f0 + EC ** 2) / (8 * EC)
        self.fit_res_dicts['E_J_std'] = -1
        self.fit_res_dicts['f_sweet_spot_std'] = f0.stderr
        self.fit_res_dicts['dac_sweet_spot_std'] = fit_result.params['dac_sweet_spot'].stderr
        self.fit_res_dicts['dac_per_phi0_std'] = fit_result.params['V_per_phi0'].stderr
        self.fit_res_dicts['asymmetry_std'] = fit_result.params['asymmetry'].stderr

        if not self.is_spectroscopy:
            g = fit_result.params['coupling']
            fr = fit_result.params['f_0_res']
            self.fit_res_dicts['coupling'] = g.value
            self.fit_res_dicts['coupling_std'] = g.stderr
            self.fit_res_dicts['f_0_res'] = fr.value
            self.fit_res_dicts['f_0_res_std'] = fr.stderr

    def prepare_plots(self):
        plot_vs_flux = self.options_dict.get('plot_vs_flux', False)
        custom_multiplier = self.options_dict.get('current_multiplier', 1)
        fitted = hasattr(self, 'fit_result') and 'dac_arc' in self.fit_result
        plot_vs_flux = plot_vs_flux and fitted and (custom_multiplier == 1)

        flux_factor = 1
        if plot_vs_flux:
            flux_factor = self.fit_res_dicts['dac_arc'].params['V_per_phi0']

        if plot_vs_flux:
            cm = flux_factor
        else:
            cm = custom_multiplier

        current_label = self.params_dict['dac'].split('.')[-1]
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

        self.qubit_name = self.raw_data_dict['measurementstring'][0].split("_")[
            2]
        ext = self.options_dict.get('qubit_freq', None) is not None
        for ax in ['amplitude', 'phase', 'distance']:
            z = self.proc_data_dict['%s_values' % ax]

            td = {'plotfn': self.plot_colorx,
                  'zorder': 0,
                  'xvals': x,
                  'yvals': y,
                  'zvals': z,
                  'title': 'Flux Current ' + s + ' Sweep\nqubit %s\nTimestamp %s -> %s' % (self.qubit_name, self.t_start, self.t_stop),
                  'xlabel': current_label,
                  'xunit': current_unit,
                  'ylabel': r'Frequency',
                  'yunit': 'Hz',
                  # 'zrange': [smoothed_amplitude_values.min(), smoothed_amplitude_values.max()],
                  # 'xrange': [self._globalmin(x), self._globalmax(x)],
                  # 'yrange': [self._globalmin(y), self._globalmax(y)],
                  'plotsize': self.options_dict.get('plotsize', None),
                  'cmap': self.options_dict.get('cmap', 'YlGn_r'),
                  'plot_transpose': self.options_dict.get('plot_transpose', False),
                  }

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
            self.plot_dicts[ax+'_2D'] = td

            if self.options_dict.get('show_fitted_peaks', True):
                sc = {
                    'plotfn': self.plot_line,
                    'zorder': 5,
                    'xvals': self.proc_data_dict['dac_values'],
                    'yvals': self.proc_data_dict['fit_frequencies'],
                    'marker': 'x',
                    'linestyle': 'None',
                }
                sc['ax_id'] = ax
                self.plot_dicts[ax + '_scatter'] = sc

            if self.do_fitting:
                fit_result = self.fit_res['dac_arc']
                self.plot_dicts[ax + '_fit'] = {
                    'plotfn': self.plot_fit,
                    'plot_init': self.options_dict.get('plot_guess', False),
                    'ax_id': ax,
                    'zorder': 10,
                    'fit_res': fit_result,
                    'xvals': self.proc_data_dict['dac_values'] * cm,
                    'yvals': self.proc_data_dict['fit_frequencies'],
                    'marker': '',
                    'setlabel': 'Fit',
                    'linestyle': '-',
                }

            if hasattr(self, 'fit_dicts') and hasattr(self, 'fit_res_dicts') and self.options_dict.get('print_fit_result_plot', True):
                dac_fit_text = ''
                # if ext or self.is_spectroscopy:
                dac_fit_text += '$E_C/2 \pi = %.2f(\pm %.3f)$ MHz\n' % (
                    self.fit_res_dicts['E_C'] * 1e-6, self.fit_res_dicts['E_C_std'] * 1e-6)
                dac_fit_text += '$E_J/\hbar = %.2f$ GHz\n' % (
                    self.fit_res_dicts['E_J'] * 1e-9)  # , self.fit_res_dicts['E_J_std'] * 1e-9
                dac_fit_text += '$\omega_{ss}/2 \pi = %.2f(\pm %.3f)$ GHz\n' % (
                    self.fit_res_dicts['f_sweet_spot'] * 1e-9, self.fit_res_dicts['f_sweet_spot_std'] * 1e-9)
                dac_fit_text += '$I_{ss} = %.2f(\pm %.3f)$ mA\n' % (
                    self.fit_res_dicts['dac_sweet_spot'] *
                    custom_multiplier * 1e3,
                    self.fit_res_dicts['dac_sweet_spot_std'] * custom_multiplier * 1e3)
                dac_fit_text += '$I/\Phi_0 = %.2f(\pm %.3f)$ mA/$\Phi_0$' % (
                    self.fit_res_dicts['dac_per_phi0'] *
                    custom_multiplier * 1e3,
                    self.fit_res_dicts['dac_per_phi0_std'] * custom_multiplier * 1e3)

                if not self.is_spectroscopy:
                    dac_fit_text += '\n$g/2 \pi = %.2f(\pm %.3f)$ MHz\n' % (
                        self.fit_res_dicts['coupling'] * 1e-6, self.fit_res_dicts['coupling_std'] * 1e-6)
                    dac_fit_text += '$\omega_{r,0}/2 \pi = %.2f(\pm %.3f)$ GHz' % (
                        self.fit_res_dicts['f_0_res'] * 1e-9, self.fit_res_dicts['f_0_res_std'] * 1e-9)
                self.plot_dicts['text_msg_' + ax] = {
                    'ax_id': ax,
                    'xpos': 1.3,
                    'ypos': 0.6,
                    'plotfn': self.plot_text,
                    'box_props': 'fancy',
                    'text_string': dac_fit_text,
                    'horizontalalignment': 'left'
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
                temp_dict['ax_id'] = 'temperature_dac_relation'
                self.plot_dicts['temperature_' +
                                k + '_dac_relation'] = temp_dict

                # Do not attempt to use deepcopy, that will use huge amounts of RAM!
                temp_dict2 = {
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

                temp_dict2['xvals'] = self.proc_data_dict['datetime']
                temp_dict2['ax_id'] = 'temperature_time_relation'
                temp_dict2['xlabel'] = r'Time in Delft'
                temp_dict2['xunit'] = ''
                self.plot_dicts['temperature_' +
                                k + '_time_relation'] = temp_dict2


class Susceptibility_to_Flux_Bias(sa.Basic2DInterpolatedAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 data_file_path: str = None,
                 close_figs: bool = True,
                 options_dict: dict = None,
                 extract_only: bool = False,
                 do_fitting: bool = True,
                 measurement_channel: int = 0,
                 correlation_distance: int = 1,
                 ):
        """
        Class for extracting local susceptibility the qubit frequency on DC
        flux parameter. Local means that  qubit frequency is assumed to be
        linearly dependent on the flux parameter.
        The input dataset needs to be 2D, frequency (x-axis) vs flux parameter (y-axis).

        The final result in units of Hz per unit-of-DC-flux-parameter is stored in
        self.proc_data_dict['susceptibility'].

        TODO: Add plotting to verify the extraction of susceptibility is correct
        """

        self.measurement_channel = measurement_channel
        self.correlation_distance = correlation_distance

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only)

    def process_data(self):
        # load and reshape arrays
        f = self.raw_data_dict['x']
        dac = self.raw_data_dict['y']
        cols = np.unique(f).shape[0]
        f = f.reshape(-1, cols)
        dac = dac.reshape(-1, cols)

        f_offset = f[0]-np.mean(f[0])
        ddac = dac[1, 0]-dac[0, 0]
        m = self.raw_data_dict['measured_values']
        m = np.reshape(m, (m.shape[0], f.shape[0], -1)
                       )[self.measurement_channel]

        # calculate correlation between cuts at different values of DC flux parameter
        # sum all corelations
        step = self.correlation_distance
        cor_tot = copy(m[0])*0
        for m1, m2 in zip(m[:-step], m[step:]):
            m1 -= np.mean(m1)
            m2 -= np.mean(m2)
            cor = np.correlate(m2, m1, mode='same')
            cor_tot += cor

        self.freq_diff, self.correlation = f_offset, cor_tot

        # fit baussian to the sum of the correlations
        self.gauss = lambda f, f0, A, w, o: A*np.exp(-(f-f0)**2/2/w**2)+o
        pk = a_tools.peak_finder_v2(f_offset, cor_tot)
        p0 = (pk[0], np.max(cor_tot), 3 *
              (f_offset[1]-f_offset[0]), -np.max(cor_tot)/100)
        popt, perr = optimize.curve_fit(self.gauss, f_offset, cor_tot, p0=p0)
        self.correlation_fit_parameters = popt

        # save susceptinility in the proc_data_dict
        self.proc_data_dict['freq'] = f
        self.proc_data_dict['dac'] = dac
        self.proc_data_dict['measured_value'] = m
        self.proc_data_dict['susceptibility'] = popt[0]/ddac/step

        self.proc_data_dict['correlation_msg'] = '$\Delta I=${:.3g} A ' \
            '(order {})\n' \
            '$\Delta f_0=${:.3} Hz\n' \
            'susc. {:.3g} Hz/A'.format(ddac/step,
                                       self.correlation_distance, popt[0], popt[0]/ddac/step)

    def run_fitting(self):
        pass

    def prepare_plots(self):
        self.figs['slope'], axs = plt.subplots(figsize=(5, 5), nrows=2,
                                               gridspec_kw={'height_ratios': (2, 1)})
        self.axs['slope'] = axs[0]
        self.axs['correlations'] = axs[1]

        self.plot_dicts['slope'] = {'plotfn': self.plot_colorxy,
                                    'zorder': 0,
                                    'xvals': self.proc_data_dict['freq'][0, :],
                                    'yvals': self.proc_data_dict['dac'][:, 0],
                                    'zvals': self.proc_data_dict['measured_value'],
                                    'title': 'Frequency Susceptibility to Fluxcurrent',
                                    'xlabel': 'Frequency',
                                    'xunit': 'Hz',
                                    'ylabel': 'Current',
                                    'yunit': 'A',
                                    'zlabel': '',
                                    }

        self.plot_dicts['correlations'] = {'plotfn': self.plot_line,
                                           'xvals': self.freq_diff,
                                           'yvals': self.correlation,
                                           'title': '',
                                           'xlabel': '$\Delta f$',
                                           'xunit': 'Hz',
                                           'ylabel': 'Current',
                                           'yunit': 'A',
                                           'marker': '.',
                                           }

        self.plot_dicts['correlations_fit'] = {'plotfn': self.plot_line,
                                               'xvals': self.freq_diff,
                                               'yvals': self.gauss(self.freq_diff, *self.correlation_fit_parameters),
                                               'title': '',
                                               'xlabel': '$\Delta f$',
                                               'xunit': 'Hz',
                                               'ylabel': 'Correlation',
                                               'yunit': '',
                                               'ax_id': 'correlations',
                                               'marker': '',
                                               }

        self.plot_dicts['rb_text'] = {
            'plotfn': self.plot_text,
            'text_string': self.proc_data_dict['correlation_msg'],
            'xpos': 0.98, 'ypos': 0.92,
            'horizontalalignment': 'right',
            'ax_id': 'correlations', }


class DACarcPolyFit(ba.BaseDataAnalysis):
    '''
    Uses a series of spectroscopy measurements at different flux biases
    to perform a polynomial fit and find approximate dependence f(\Phi).

    Args:
        t_start (str):
            timestamp of the first measurement

        t_start (str):
            timestamp of the last measurement

        dac_key (str):
            key pointing to the parameter that stores the current
            applied to the flux bias line

        frequency_key (str):
            key pointing to the parameter that stores the fitted
            qubit frequency

        degree (int): order of the polynomial to be fitted

    '''
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = 'spectroscopy',
                 options_dict: dict = None, extract_only: bool = False,
                 dac_key='Instrument settings.fluxcurrent.Q',
                 frequency_key='Analysis.Fitted Params HM.f0.value',
                 do_fitting=True, degree=2
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only,
                         do_fitting=do_fitting)

        self.params_dict = {'dac': dac_key,
                            'qfreq': frequency_key,
                            'measurementstring': 'measurementstring'}

        self.numeric_params = ['dac', 'qfreq']
        self.degree = degree

        self.run_analysis()
        # return self.proc_data_dict

    def run_fitting(self):
        dac = self.raw_data_dict['dac']
        freq = self.raw_data_dict['qfreq']

        polycoeffs = np.polyfit(dac, freq, self.degree)

        self.fit_res = {}
        self.fit_res['fit_polycoeffs'] = polycoeffs
        self.fit_res['fit_polynomial'] = np.poly1d(polycoeffs)

    def save_fit_results(self):
        # todo: if you want to save some results to a hdf5, do it here
        pass

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['dac'],
            'xlabel': 'Current',
            'xunit': 'A',
            'yvals': self.raw_data_dict['qfreq'],
            'ylabel': 'Frequency',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'linestyle': '',
            'marker': 'o',
            'setlabel': 'data',
            'color': 'C0',
        }

        xs = np.linspace(min(self.raw_data_dict['dac']), max(
            self.raw_data_dict['dac']), 301)
        self.plot_dicts['fit'] = {
            'plotfn': self.plot_line,
            'xvals': xs,
            'xlabel': 'Current',
            'xunit': 'A',
            'yvals': self.fit_res['fit_polynomial'](xs),
            'ylabel': 'Frequency',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'linestyle': '-',
            'marker': '',
            'setlabel': 'fit',
            'color': 'C0',
            'ax_id': 'main'
        }


class DAC_analysis(ma.TwoD_Analysis):
    """
    Analyzes 2D qubit DAC arcs. Fits a Lorentzian to each linecut to extract
    the qubit frequency at each DAC value.

    Fits a 2nd degree polynomial through the extracted qubit frequencies with
    the np.polyfit method.

    This function can be called with the timestamp of the DAC arc as its only
    argument. It is heavily inspired by the VNA_DAC_Analysis in
    analysis_v2.spectroscopy_analysis.

    Important returned variables are:
    self.dac_fit_res['fit_polycoeffs']: the coefficients describing the DAC arc
                                        parabola
    self.sweet_spot_value: The current at which the parabola is at a maximum,
                           obtained by -b/2a (parabola = ax^2 + bx +c)
    """

    def __init__(self,
                 options_dict=None,
                 do_fitting=True,
                 extract_only=False,
                 auto=True,
                 use_phase:bool=False,
                 **kw):
        super(ma.TwoD_Analysis, self).__init__(options_dict=options_dict,
                                               extract_only=extract_only,
                                               auto=auto,
                                               do_fitting=do_fitting,
                                               **kw)
        linecut_fit_result = self.fit_linecuts(use_phase)
        self.linecut_fit_result = linecut_fit_result
        f0s = []
        for res in self.linecut_fit_result:
            f0s.append(res.values['f0'])
        self.f0s = np.array(f0s)
        self.run_full_analysis(**kw)
        self.dac_fit_res = self.fit_dac_arc()
        self.sweet_spot_value = self.dac_fit_res['sweetspot_dac']
        self.plot_fit_result(**kw)

    def fit_linecuts(self, use_phase):
        if use_phase:
            linecut_mag = np.array(self.measured_values)[1].T[:]
        else:
            linecut_mag = np.array(self.measured_values)[0].T[:]
        sweep_points = self.sweep_points[:]
        fit_result = []
        for linecut in linecut_mag:
            fit_result.append(self.qubit_fit(sweep_points, linecut))
        return fit_result

    def qubit_fit(self, sweep_points, linecut_mag, **kw):
        """
        This is basically a modified copy of the 'fit_data' function of the
        Qubit_Spectroscopy_Analysis method.
        Does not support 2nd peak fitting, as it does not seem necessary.
        """
        frequency_guess = kw.get('frequency_guess', None)
        percentile = kw.get('percentile', 2.5)
        num_sigma_threshold = kw.get('num_sigma_threshold', 4)
        window_len_filter = kw.get('window_len_filter', 3)
        optimize = kw.pop('optimize', True)
        verbose = kw.get('verbose', False)

        self.data_dist = linecut_mag

        data_dist_smooth = a_tools.smooth(self.data_dist,
                                          window_len=window_len_filter)
        self.peaks = a_tools.peak_finder(sweep_points,
                                         data_dist_smooth,
                                         percentile=percentile,
                                         num_sigma_threshold=num_sigma_threshold,
                                         optimize=optimize,
                                         window_len=0)

        # extract highest peak -> ge transition
        if frequency_guess is not None:
            f0 = frequency_guess
            kappa_guess = (max(self.sweep_points)-min(self.sweep_points))/20
            key = 'peak'
        elif self.peaks['dip'] is None:
            f0 = self.peaks['peak']
            kappa_guess = self.peaks['peak_width'] / 4
            key = 'peak'
        elif self.peaks['peak'] is None:
            f0 = self.peaks['dip']
            kappa_guess = self.peaks['dip_width'] / 4
            key = 'dip'
        # elif self.peaks['dip'] < self.peaks['peak']:
        elif np.abs(data_dist_smooth[self.peaks['dip_idx']]) < \
                np.abs(data_dist_smooth[self.peaks['peak_idx']]):
            f0 = self.peaks['peak']
            kappa_guess = self.peaks['peak_width'] / 4
            key = 'peak'
        # elif self.peaks['peak'] < self.peaks['dip']:
        elif np.abs(data_dist_smooth[self.peaks['dip_idx']]) > \
                np.abs(data_dist_smooth[self.peaks['peak_idx']]):
            f0 = self.peaks['dip']
            kappa_guess = self.peaks['dip_width'] / 4
            key = 'dip'
        else:  # Otherwise take center of range and raise warning
            f0 = np.median(self.sweep_points)
            kappa_guess = 0.005 * 1e9
            logging.warning('No peaks or dips have been found. Initial '
                            'frequency guess taken '
                            'as median of sweep points (f_guess={}), '
                            'initial linewidth '
                            'guess was taken as kappa_guess={}'.format(
                                f0, kappa_guess))
            key = 'peak'

        tallest_peak = f0  # the ge freq
        if verbose:
            print('Largest ' + key + ' is at ', tallest_peak)
        if f0 == self.peaks[key]:
            tallest_peak_idx = self.peaks[key + '_idx']
            if verbose:
                print('Largest ' + key + ' idx is ', tallest_peak_idx)

        amplitude_guess = np.pi * kappa_guess * \
            abs(max(self.data_dist) - min(self.data_dist))
        if key == 'dip':
            amplitude_guess = -amplitude_guess

        LorentzianModel = fit_mods.LorentzianModel

        LorentzianModel.set_param_hint('f0',
                                       min=min(self.sweep_points),
                                       max=max(self.sweep_points),
                                       value=f0)
        LorentzianModel.set_param_hint('A',
                                       value=amplitude_guess)

        LorentzianModel.set_param_hint('offset',
                                       value=np.mean(self.data_dist),
                                       vary=True)
        LorentzianModel.set_param_hint('kappa',
                                       value=kappa_guess,
                                       min=1,
                                       vary=True)
        LorentzianModel.set_param_hint('Q',
                                       expr='f0/kappa',
                                       vary=False)
        self.params = LorentzianModel.make_params()

        fit_res = LorentzianModel.fit(data=self.data_dist,
                                      f=self.sweep_points,
                                      params=self.params)

        return fit_res

    def fit_dac_arc(self):
        DAC_values = self.sweep_points_2D[1:-1]
        f0s = self.f0s[1:-1]
        # DAC_values = self.sweep_points_2D
        # f0s = self.f0s
        polycoeffs = np.polyfit(DAC_values, f0s, 2)
        sweetspot_dac = -polycoeffs[1]/(2*polycoeffs[0])
        fit_res = {}
        fit_res['fit_polycoeffs'] = polycoeffs
        fit_res['fit_polynomial'] = np.poly1d(polycoeffs)
        fit_res['sweetspot_dac'] = sweetspot_dac
        return fit_res

    def plot_fit_result(self, normalize=False, plot_linecuts=True,
                        linecut_log=False, colorplot_log=False,
                        plot_all=True, save_fig=True,
                        transpose=False, figsize=None, filtered=False,
                        subtract_mean_x=False, subtract_mean_y=False,
                        **kw):
        fig, ax = plt.subplots(figsize=figsize)
        self.fig_array.append(fig)
        self.ax_array.append(ax)
        # print "unransposed",meas_vals
        # print "transposed", meas_vals.transpose()
        self.ax_array.append(ax)
        savename = 'Heatmap_{}_fit'.format(self.value_names[0])
        fig_title = '{} {} \n{}'.format(
            self.timestamp_string, self.measurementstring,
            self.value_names[0])

        if "xlabel" not in kw:
            kw["xlabel"] = self.parameter_names[0]
        if "ylabel" not in kw:
            kw["ylabel"] = self.parameter_names[1]
        if "zlabel" not in kw:
            kw["zlabel"] = self.value_names[0]
        if "x_unit" not in kw:
            kw["x_unit"] = self.parameter_units[0]
        if "y_unit" not in kw:
            kw["y_unit"] = self.parameter_units[1]
        if "z_unit" not in kw:
            kw["z_unit"] = self.value_units[0]

        # subtract mean from each row/column if demanded
        plot_zvals = self.measured_values[0].transpose()
        if subtract_mean_x:
            plot_zvals = plot_zvals - np.mean(plot_zvals, axis=1)[:, None]
        if subtract_mean_y:
            plot_zvals = plot_zvals - np.mean(plot_zvals, axis=0)[None, :]

        a_tools.color_plot(x=self.sweep_points,
                           y=self.sweep_points_2D,
                           z=plot_zvals,
                           # zlabel=self.zlabels[0],
                           fig=fig, ax=ax,
                           log=colorplot_log,
                           transpose=transpose,
                           normalize=normalize,
                           **kw)

        plot_dacs = np.linspace(min(self.sweep_points_2D),
                                max(self.sweep_points_2D), 101)

        fit_plot = self.dac_fit_res['fit_polynomial'](plot_dacs)

        # set_xlabel(ax, kw["xlabel"], kw["x_unit"])
        # set_ylabel(ax, kw["ylabel"], kw["y_unit"])

        ax.plot(self.f0s, self.sweep_points_2D, 'ro-')
        ax.plot(fit_plot, plot_dacs, 'b')

        ax.set_title(fig_title)

        if save_fig:
            self.save_fig(fig, figname=savename, **kw)

    def run_full_analysis(self, normalize=False, plot_linecuts=True,
                          linecut_log=False, colorplot_log=False,
                          plot_all=True, save_fig=True,
                          transpose=False, figsize=None, filtered=False,
                          subtract_mean_x=False, subtract_mean_y=False,
                          **kw):
        '''
        Args:
          linecut_log (bool):
              log scale for the line cut?
              Remember to set the labels correctly.
          colorplot_log (string/bool):
              True/False for z axis scaling, or any string containing any
              combination of letters x, y, z for scaling of the according axis.
              Remember to set the labels correctly.

        '''
        close_file = kw.pop('close_file', True)
        self.fig_array = []
        self.ax_array = []

        for i, meas_vals in enumerate(self.measured_values[:1]):
            if filtered:
                # print(self.measured_values)
                # print(self.value_names)
                if self.value_names[i] == 'Phase':
                    self.measured_values[i] = dm_tools.filter_resonator_visibility(
                                                      x=self.sweep_points,
                                                      y=self.sweep_points_2D,
                                                      z=self.measured_values[i],
                                                      **kw)

            if (not plot_all) & (i >= 1):
                break
            # Linecuts are above because somehow normalization applies to both
            # colorplot and linecuts otherwise.
            if plot_linecuts:
                fig, ax = plt.subplots(figsize=figsize)
                self.fig_array.append(fig)
                self.ax_array.append(ax)
                savename = 'linecut_{}'.format(self.value_names[i])
                fig_title = '{} {} \nlinecut {}'.format(
                    self.timestamp_string, self.measurementstring,
                    self.value_names[i])
                a_tools.linecut_plot(x=self.sweep_points,
                                     y=self.sweep_points_2D,
                                     z=self.measured_values[i],
                                     y_name=self.parameter_names[1],
                                     y_unit=self.parameter_units[1],
                                     log=linecut_log,
                                     # zlabel=self.zlabels[i],
                                     fig=fig, ax=ax, **kw)
                ax.set_title(fig_title)
                set_xlabel(ax, self.parameter_names[0],
                           self.parameter_units[0])
                # ylabel is value units as we are plotting linecuts
                set_ylabel(ax, self.value_names[i],
                           self.value_units[i])

                if save_fig:
                    self.save_fig(fig, figname=savename,
                                  fig_tight=False, **kw)
            # Heatmap
            fig, ax = plt.subplots(figsize=figsize)
            self.fig_array.append(fig)
            self.ax_array.append(ax)
            if normalize:
                print("normalize on")
            self.ax_array.append(ax)
            savename = 'Heatmap_{}'.format(self.value_names[i])
            fig_title = '{} {} \n{}'.format(
              self.timestamp_string, self.measurementstring,
              self.value_names[i])

            if "xlabel" not in kw:
                kw["xlabel"] = self.parameter_names[0]
            if "ylabel" not in kw:
                kw["ylabel"] = self.parameter_names[1]
            if "zlabel" not in kw:
                kw["zlabel"] = self.value_names[0]
            if "x_unit" not in kw:
                kw["x_unit"] = self.parameter_units[0]
            if "y_unit" not in kw:
                kw["y_unit"] = self.parameter_units[1]
            if "z_unit" not in kw:
                kw["z_unit"] = self.value_units[0]

            # subtract mean from each row/column if demanded
            plot_zvals = meas_vals.transpose()
            if subtract_mean_x:
                plot_zvals = plot_zvals - np.mean(plot_zvals, axis=1)[:, None]
            if subtract_mean_y:
                plot_zvals = plot_zvals - np.mean(plot_zvals, axis=0)[None, :]

            a_tools.color_plot(x=self.sweep_points,
                               y=self.sweep_points_2D,
                               z=plot_zvals,
                               # zlabel=self.sweept_val[i],u
                               fig=fig, ax=ax,
                               log=colorplot_log,
                               transpose=transpose,
                               normalize=normalize,
                               **kw)
            ax.set_title(fig_title)
            set_xlabel(ax, kw["xlabel"], kw["x_unit"])
            set_ylabel(ax, kw["ylabel"], kw["y_unit"])
            if save_fig:
                self.save_fig(fig, figname=savename,
                              fig_tight=False, **kw)
