'''
Collection of classes to analyse Quantum efficiency measurements.
General procedure is:
 - find factor 'a' of the quadratic scaling of the Single-Shot-Readout as a function of scaling_amp (see SSROAnalysis class)
 - find sigma from the Gaussian fit of the Dephasing as a function of scaling_amp (see dephasingAnalysis class)
 - Calculate eta = a * sigma**2 / 2 (see QuantumEfficiencyAnalysis class)
For details, see https://arxiv.org/abs/1711.05336

Lastly, the QuantumEfficiencyAnalysisTWPA class allows for analysing the efficiency
as a function of TWPA power and frequency.

Hacked together by Rene Vollmer
'''

import datetime
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis_v2.base_analysis import plot_scatter_errorbar_fit, plot_scatter_errorbar

import numpy as np
import lmfit
import os

from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict

import copy
from pycqed.analysis.measurement_analysis import MeasurementAnalysis
from copy import deepcopy

class QuantumEfficiencyAnalysisTWPA(ba.BaseDataAnalysis):
    '''
    Analyses Quantum efficiency measurements as a function of TWPA Pump frequency and power.
    '''

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label_dephasing: str = '_dephasing',
                 label_ssro: str = '_SSRO', label: str = '',
                 options_dict: dict = None,
                 extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True,
                 twpa_pump_freq_key: str = 'Instrument settings.TWPA_Pump.frequency',
                 twpa_pump_power_key: str = 'Instrument settings.TWPA_Pump.power',
                 use_prefit: bool = False):
        '''

        :param t_start: start time of scan as a string of format YYYYMMDD_HHmmss
        :param t_stop: end time of scan as a string of format YYYYMMDD_HHmmss
        :param options_dict: Available options are the ones from the base_analysis and:
                                - individual_plots : plot all the individual fits?
                                - cmap : colormap for 2D plots
                                - plotsize : plotsize for 2D plots
                                - (todo)
        :param auto: Execute all steps automatically
        :param close_figs: Close the figure (do not display)
        :param extract_only: Should we also do the plots?
        :param do_fitting: Should the run_fitting method be executed?
        :param label_dephasing: the label that was used to name the dephasing measurements
        :param label_ssro: the label that was used to name the SSRO measurements
        :param label: (Optional) common label that was used to name all measurements
        :param twpa_pump_freq_key: key for the TWPA Pump Frequency, e.g. 'Instrument settings.TWPA_Pump.frequency'
        :param twpa_pump_power_key: key for the TWPA Pump Power, e.g. 'Instrument settings.TWPA_Pump.power'
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only)
        self.use_prefit = use_prefit
        self.label_dephasing = label_dephasing
        self.label_ssro = label_ssro

        self.params_dict = {'TWPA_freq': twpa_pump_freq_key,
                            'measurementstring': 'measurementstring',
                            'TWPA_power': twpa_pump_power_key}

        self.numeric_params = ['TWPA_freq', 'TWPA_power']
        if use_prefit:
            self.params_dict['a'] = 'Analysis.coherence_analysis.a'
            self.params_dict['a_std'] = 'Analysis.coherence_analysis.a_std'
            self.params_dict['sigma'] = 'Analysis.coherence_analysis.sigma'
            self.params_dict['sigma_std'] = 'Analysis.coherence_analysis.sigma_std'
            self.params_dict['eta'] = 'Analysis.coherence_analysis.eta'
            self.params_dict['u_eta'] = 'Analysis.coherence_analysis.u_eta'
            self.numeric_params.append('a')
            self.numeric_params.append('a_std')
            self.numeric_params.append('sigma')
            self.numeric_params.append('sigma_std')
            self.numeric_params.append('eta')
            self.numeric_params.append('u_eta')

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
        # Sort data by frequencies and power
        self.proc_data_dict = {}
        twpa_freqs_unsorted = np.array(self.raw_data_dict['TWPA_freq'], dtype=float)
        twpa_freqs = np.unique(twpa_freqs_unsorted)
        twpa_freqs.sort()
        twpa_powers_unsorted = np.array(self.raw_data_dict['TWPA_power'], dtype=float)
        twpa_powers = np.unique(twpa_powers_unsorted)
        twpa_powers.sort()

        self.proc_data_dict['TWPA_freqs'] = twpa_freqs
        self.proc_data_dict['TWPA_powers'] = twpa_powers
        if self.verbose:
            print('Found %d twpa freqs and %d amplitudes' % (len(twpa_freqs), len(twpa_powers)))
            print(twpa_freqs, twpa_powers)

        dates = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        # date_limits = np.array([[(None, None)] * len(twpa_powers)] * len(twpa_freqs))
        datetimes = np.array(self.raw_data_dict['datetime'], dtype=datetime.datetime)
        for i, twpa_freq in enumerate(twpa_freqs):
            freq_indices = np.where(twpa_freqs_unsorted == twpa_freq)
            for j, twpa_power in enumerate(twpa_powers):
                power_indices = np.where(twpa_powers_unsorted == twpa_power)
                indices = np.array(np.intersect1d(freq_indices, power_indices), dtype=int)
                if self.use_prefit:
                    if len(indices) > 1:
                        print("Warning: more than one efficiency value found for freq %.3f and power %.3f"%(twpa_freq*1e-9,twpa_power))
                    elif len(indices) == 1:
                        print("Warning:no efficiency value found for freq %.3f and power %.3f"%(twpa_freq*1e-9,twpa_power))
                    dates = indices[0]

                else:
                    dts = datetimes[indices]
                    dates[i, j] = dts
                # date_limits[i, j][0] = np.min(dts)
                # date_limits[i, j][1] = np.max(dts)

        if self.use_prefit:
            self.proc_data_dict['sorted_indices'] = np.array(dates, dtype=int)
        else:
            self.proc_data_dict['sorted_datetimes'] = dates
            # self.proc_data_dict['sorted_date_limits'] = date_limits

    def process_data(self):
        twpa_freqs = self.proc_data_dict['TWPA_freqs']
        twpa_powers = self.proc_data_dict['TWPA_powers']
        dates = self.proc_data_dict['sorted_datetimes']
        sorted_indices = self.proc_data_dict['sorted_indices']
        # date_limits = self.proc_data_dict['sorted_date_limits']

        eta = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        u_eta = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        sigma = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        u_sigma = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        a = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)
        u_a = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=float)


        objects = np.array([[None] * len(twpa_powers)] * len(twpa_freqs), dtype=QuantumEfficiencyAnalysis)
        d = copy.deepcopy(self.options_dict)
        d['save_figs'] = False
        for i, freq in enumerate(twpa_freqs):
            for j, power in enumerate(twpa_powers):
                if self.use_prefit:
                    index = sorted_indices[i, j]
                    a[i, j] = self.raw_data_dict['a'][index]
                    u_a[i, j] = self.raw_data_dict['a_std'][index]
                    sigma[i, j] = self.raw_data_dict['sigma'][index]
                    u_sigma[i, j] = self.raw_data_dict['sigma_std'][index]
                    eta[i, j] = self.raw_data_dict['eta'][index]
                    u_eta[i, j] = self.raw_data_dict['u_eta'][index]
                else:
                    t_start = [d.strftime("%Y%m%d_%H%M%S") for d in dates[i, j]]  # date_limits[i, j][0]
                    t_stop = None  # date_limits[i, j][1]
                    # print(t_start, t_stop)
                    qea = QuantumEfficiencyAnalysis(t_start=t_start, t_stop=t_stop, label_dephasing=self.label_dephasing,
                                                    label_ssro=self.label_ssro, options_dict=d, auto=False,
                                                    extract_only=True)
                    qea.run_analysis()

                    a[i, j] = qea.fit_dicts['a']
                    u_a[i, j] = qea.fit_dicts['a_std']

                    sigma[i, j] = qea.fit_dicts['sigma']
                    u_sigma[i, j] = qea.fit_dicts['sigma_std']

                    eta[i, j] = qea.fit_dicts['eta']
                    u_eta[i, j] = qea.fit_dicts['u_eta']

                    objects[i, j] = qea

            if not self.use_prefit:
                self.proc_data_dict['analysis_objects'] = objects

        self.proc_data_dict['as'] = a
        self.proc_data_dict['as_std'] = u_a
        self.proc_data_dict['sigmas'] = sigma
        self.proc_data_dict['sigmas_std'] = u_sigma
        self.proc_data_dict['etas'] = eta
        self.proc_data_dict['etas_std'] = u_eta

        self.proc_data_dict['as'] = a
        self.proc_data_dict['as_std'] = u_a
        self.proc_data_dict['sigmas'] = sigma
        self.proc_data_dict['sigmas_std'] = u_sigma
        self.proc_data_dict['etas'] = eta
        self.proc_data_dict['etas_std'] = u_eta

    def prepare_plots(self):
        title = ('\n' + self.timestamps[0] + ' - "' +
                 self.raw_data_dict['measurementstring'] + '"')

        twpa_powers = self.proc_data_dict['TWPA_powers']
        twpa_freqs = self.proc_data_dict['TWPA_freqs']

        # Quantum Efficiency
        self.plot_dicts['quantum_eff'] = {
            'plotfn': self.plot_colorxy,
            'title': title,
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['etas'].transpose() * 100,
            'zlabel': r'Quantum efficiency $\eta$ (%)',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['quantum_eff_vari'] = {
            'plotfn': self.plot_colorxy,
            'title': '' + title,
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['etas_std'].transpose(),
            'zlabel': r'Quantum efficiency Deviation $\delta \eta$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }

        # SSRO Slope
        self.plot_dicts['ssro_slope'] = {
            'plotfn': self.plot_colorxy,
            'title': '' + title,  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['as'].transpose(),
            'zlabel': r'SSRO slope $a$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['ssro_slope_vari'] = {
            'plotfn': self.plot_colorxy,
            'title': '' + title,  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['as_std'].transpose(),
            'zlabel': r'SSRO slope variance $\delta a$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }

        # Dephasing Gauss Width
        self.plot_dicts['dephasing_gauss_width'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['sigmas'].transpose(),
            'zlabel': r'dephasing Gauss width $\sigma$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['dephasing_gauss_width_vari'] = {
            'plotfn': self.plot_colorxy,
            'title': '' + title,  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['sigmas_std'].transpose(),
            'zlabel': r'dephasing Gauss width variance $\delta\sigma$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }

        if self.options_dict.get('individual_plots', False):
            # todo: add 1D plot from QuantumEfficiencyAnalysis
            for i, twpa_freq in enumerate(twpa_freqs):
                for j, twpa_power in enumerate(twpa_powers):
                    pre = 'freq_%.3f-power_%.3f-' % (twpa_freq, twpa_power)
                    obj = self.proc_data_dict['analysis_objects'][i, j]
                    for k in ['amp_vs_dephasing_coherence', 'amp_vs_dephasing_fit', ]:
                        self.plot_dicts[pre + k] = obj.ra.plot_dicts[k]
                        self.plot_dicts[pre + k]['ax_id'] = pre + 'snr_analysis'
                    for k in ['amp_vs_SNR_fit', 'amp_vs_SNR_scatter', ]:
                        self.plot_dicts[pre + k] = obj.ssro.plot_dicts[k]
                        self.plot_dicts[pre + k]['ax_id'] = pre + 'snr_analysis'


class QuantumEfficiencyAnalysis(ba.BaseDataAnalysis):
    '''
    Analyses one set of Quantum efficiency measurements
    '''

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label_dephasing: str = '_dephasing', label_ssro: str = '_SSRO',
                 options_dict: dict = None, options_dict_ssro: dict = None,
                 options_dict_dephasing: dict = None,
                 extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True,
                 use_sweeps: bool = True):
        '''

        :param t_start: start time of scan as a string of format YYYYMMDD_HHmmss
        :param t_stop: end time of scan as a string of format YYYYMMDD_HHmmss
        :param options_dict: Available options are the ones from the base_analysis and:
                                - individual_plots : plot all the individual fits?
                                - cmap : colormap for 2D plots
                                - plotsize : plotsize for 2D plots
                                - (todo)
        :param options_dict_dephasing: same as options_dict, but exclusively for
                                    the dephasing analysis.
        :param options_dict_ssro: same as options_dict, but exclusively for
                                  the ssro analysis.
        :param auto: Execute all steps automatically
        :param close_figs: Close the figure (do not display)
        :param extract_only: Should we also do the plots?
        :param do_fitting: Should the run_fitting method be executed?
        :param label_dephasing: the label to identify the dephasing measurements
        :param label_ssro: the label to identify the SSRO measurements
        :param use_sweeps: True: Use the datat from one sweep folder.
                           False: Collect the results from the individual
                                  measurements (not tested yet)
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )
        if options_dict_dephasing is None:
             options_dict_dephasing = {}
        if options_dict_ssro is None:
            options_dict_ssro = {}
        d = copy.deepcopy(self.options_dict)
        d['save_figs'] = False
        dr = {**d, **options_dict_dephasing}
        ds = {**d, **options_dict_ssro}

        if use_sweeps:
            self.ra = DephasingAnalysisSweep(t_start=t_start, t_stop=t_stop,
                                          label=label_dephasing,
                                          options_dict=dr, auto=False,
                                          extract_only=True)
            self.ssro = SSROAnalysisSweep(t_start=t_start, t_stop=t_stop,
                                          label=label_ssro,
                                          options_dict=ds, auto=False,
                                          extract_only=True)
        else:
            self.ra = DephasingAnalysisSingleScans(t_start=t_start, t_stop=t_stop,
                                                label=label_dephasing,
                                                options_dict=dr, auto=False,
                                                extract_only=True)
            self.ssro = SSROAnalysisSingleScans(t_start=t_start, t_stop=t_stop,
                                                label=label_ssro,
                                                options_dict=ds, auto=False,
                                                extract_only=True)

        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()

        self.ra.extract_data()
        self.ssro.extract_data()

        youngest = max(np.max(np.array(self.ra.raw_data_dict['datetime'], dtype=datetime.datetime)),
                       np.max(np.array(self.ssro.raw_data_dict['datetime'], dtype=datetime.datetime)))

        youngest += datetime.timedelta(seconds=1)

        self.raw_data_dict['datetime'] = [youngest]
        self.raw_data_dict['timestamps'] = [youngest.strftime("%Y%m%d_%H%M%S")]
        self.timestamps = [youngest.strftime("%Y%m%d_%H%M%S")]

        f = '%s_quantum_efficiency_analysis' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.raw_data_dict['measurementstring'] = f
        self.options_dict['analysis_result_file'] = os.path.join(folder, f + '.hdf5')

    def run_fitting(self):
        self.ra.run_analysis()
        self.ssro.run_analysis()

        self.fit_dicts = OrderedDict()
        self.fit_dicts['sigma'] = self.ra.fit_res['coherence_fit'].params['sigma'].value
        self.fit_dicts['sigma_std'] = self.ra.fit_res['coherence_fit'].params['sigma'].stderr
        # self.raw_data_dict['scale'] = self.ra.fit_dicts['coherence_fit']['scale']
        self.fit_dicts['a'] = self.ssro.fit_res['snr_fit'].params['a'].value
        self.fit_dicts['a_std'] = self.ssro.fit_res['snr_fit'].params['a'].stderr

        sigma = self.fit_dicts['sigma']
        u_sigma = self.fit_dicts['sigma_std']
        a = self.fit_dicts['a']
        u_a = self.fit_dicts['a_std']

        eta = (a * sigma) ** 2 / 2
        u_eta = 2 * (u_a / a + u_sigma / sigma) * eta

        if self.verbose:
            print('eta = %.4f +- %.4f' % (eta, u_eta))

        self.fit_dicts['eta'] = eta
        self.fit_dicts['u_eta'] = u_eta

        # For saving
        self.fit_res = OrderedDict()
        self.fit_res['quantum_efficiency'] = OrderedDict()
        self.fit_res['quantum_efficiency']['eta'] = eta
        self.fit_res['quantum_efficiency']['u_eta'] = u_eta
        self.fit_res['quantum_efficiency']['sigma'] = sigma
        self.fit_res['quantum_efficiency']['sigma_std'] = u_sigma
        self.fit_res['quantum_efficiency']['a'] = a
        self.fit_res['quantum_efficiency']['a_std'] = u_a

    def prepare_plots(self):
        title = ('\n' + self.timestamps[0] + ' - "' +
                 self.raw_data_dict['measurementstring'] + '"')

        self.ra.prepare_plots()
        dicts = OrderedDict()
        for d in self.ra.plot_dicts:
            dicts[d] = self.ra.plot_dicts[d]
        self.ssro.prepare_plots()
        for d in self.ssro.plot_dicts:
            dicts[d] = self.ssro.plot_dicts[d]

        if self.options_dict.get('subplots', True):
            self.plot_dicts = deepcopy(dicts)

        for k in ['amp_vs_dephasing_fitted',
                  'amp_vs_dephasing_not_fitted',
                  'amp_vs_dephasing_fit',
                  'amp_vs_SNR_scatter_fitted',
                  'amp_vs_SNR_scatter_not_fitted',
                  'amp_vs_SNR_fit',]:
            if k in dicts:
                k2 = 'quantum_eff_analysis_' + k
                self.plot_dicts[k2] = dicts[k]
                self.plot_dicts[k2]['ax_id'] = 'quantum_eff_analysis'
                self.plot_dicts[k2]['ylabel'] = 'SNR, coherence'
                self.plot_dicts[k2]['yunit'] = '(-)'
                self.plot_dicts[k2]['title'] = ''

        self.plot_dicts['amp_vs_SNR_fit']['do_legend'] = True
        self.plot_dicts['amp_vs_dephasing_fit']['do_legend'] = True

        res = self.fit_res['quantum_efficiency']
        t = '$\sigma=%.3f\pm%.3f$'%(res['sigma'], res['sigma_std'])
        self.plot_dicts['quantum_eff_analysis_sigma_text'] = {
            'ax_id': 'quantum_eff_analysis',
            'plotfn': self.plot_line,
            'xvals': [0,0], 'yvals': [0,0],
            'marker': None,
            'linestyle': '',
            'setlabel': t,
            'do_legend': True,
        }
        t = '$a=%.3f\pm%.3f$'%(res['a'], res['a_std'])
        self.plot_dicts['quantum_eff_analysis_a_text'] = {
            'ax_id': 'quantum_eff_analysis',
            'plotfn': self.plot_line,
            'xvals': [0,0], 'yvals': [0,0],
            'marker': None,
            'linestyle': '',
            'setlabel': t,
            'do_legend': True,
        }
        t = '$\eta=%.3f\pm%.3f$'%(res['eta'], res['u_eta'])
        self.plot_dicts['quantum_eff_analysis_qeff_text'] = {
            'ax_id': 'quantum_eff_analysis',
            'plotfn': self.plot_line,
            'xvals': [0,0], 'yvals': [0,0],
            'marker': None,
            'linestyle': '',
            'setlabel': t,
            'do_legend': True,
        }

class DephasingAnalysis(ba.BaseDataAnalysis):
    '''

    options_dict options:
     - fit_phase_offset (bool) - Fit the phase offset?
     - default_phase_offset (float) - Fixed value for the phase offset.
                                        Ignored if fit_phase_offset=True
     - amp_threshold: (float) - maximal amplitude to fit.
                                 Do not set or set to False to fit all data.
    '''

    def process_data(self):
        # Remove None entries
        dephasing = self.raw_data_dict['dephasing']
        amps = self.raw_data_dict['scaling_amp']
        mask = np.intersect1d(np.where(dephasing != None), np.where(amps != None))

        self.proc_data_dict['scaling_amp'] = amps[mask]
        self.proc_data_dict['coherence'] = dephasing[mask]
        self.proc_data_dict['phase'] = self.raw_data_dict['phase'][mask]

        # Fitting mask
        mask = range(0, len(amps))
        inv_mask = []
        if self.options_dict.get('amp_threshold', False):
            mask = np.where(amps < self.options_dict['amp_threshold'])
            inv_mask = np.where(amps >= self.options_dict['amp_threshold'])

        self.proc_data_dict['fitting_mask'] = mask
        self.proc_data_dict['fitting_mask_inv'] = inv_mask

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        coherence = self.proc_data_dict['coherence']
        amps = self.proc_data_dict['scaling_amp']

        mask = self.proc_data_dict['fitting_mask']
        amps = amps[mask]
        coherence = coherence[mask]

        def gaussian(x, sigma, scale):
            return scale * np.exp(-(x) ** 2 / (2 * sigma ** 2))

        gmodel = lmfit.models.Model(gaussian)
        gmodel.set_param_hint('sigma', value=0.5, min=0, max=10)
        gmodel.set_param_hint('scale', value=np.max(coherence))  # , min=0.1, max=100)
        gpara = gmodel.make_params()

        self.fit_dicts['coherence_fit'] = {
            'model': gmodel,
            'fit_xvals': {'x': amps},
            'fit_yvals': {'data': coherence},
            'guess_pars': gpara,
        }


    def run_fitting(self):
        super().run_fitting()

        sigma = self.fit_res['coherence_fit'].params['sigma'].value
        cexp = '-1/(2*%.5f*s**2)'%((sigma**2),)

        def square(x, s, b):
            return ((x * s) ** 2 + b) % 360

        def minimizer_function_vec(params, x, data):
            s = params['s']
            b = params['b']
            return np.abs(((square(x, s, b)-data+180)%360)-180)

        def minimizer_function(params, x, data):
            return np.sum(minimizer_function_vec(params, x, data))

        phase = self.proc_data_dict['phase']
        amps = self.proc_data_dict['scaling_amp']

        mask = self.proc_data_dict['fitting_mask']
        amps = amps[mask]
        phase = phase[mask]

        def fit_phase(amps, phase):
            params = lmfit.Parameters()
            params.add('s', value=3/sigma, min=0.01, max=200, vary=True)
            i = max(int(round(len(phase)/10)), 1)
            fit_offset = self.options_dict.get('fit_phase_offset', False)
            dpo = self.options_dict.get('default_phase_offset', 180)
            phase_guess = np.mean(phase[0:i]) if fit_offset else dpo

            params.add('b', value=phase_guess, min=0, max=360, vary=True)
            params.add('c', expr=cexp)
            mini = lmfit.Minimizer(minimizer_function, params=params, fcn_args=(amps, phase))
            res = mini.minimize(method='differential_evolution')

            if not fit_offset:
                return res, res

            params2 = lmfit.Parameters()
            params2.add('s', value=res.params['s'].value, min=0.01, max=200, vary=False)
            params2.add('b', value=res.params['b'].value, min=0, max=360, vary=True)
            params2.add('c', expr=cexp)
            mini2 = lmfit.Minimizer(minimizer_function, params=params2, fcn_args=(amps, phase))
            res2 = mini2.minimize(method='differential_evolution')

            if res.chisqr < res2.chisqr:
                return res, res
            else:
                return res2, res

        res, res_old = fit_phase(amps, phase)
        self.fit_res['coherence_phase_fit'] = res

        fit_amps = np.linspace(min(amps), max(amps), 300)
        fit_phase = square(x=fit_amps, s=res.params['s'].value, b=res.params['b'].value)
        guess_phase = square(x=fit_amps, s=res_old.params['s'].init_value,
                             b=res_old.params['b'].init_value)
        self.proc_data_dict['coherence_phase_fit'] = {'amps': fit_amps,
                                                      'phase': fit_phase,
                                                      'phase_guess' : guess_phase}

    def prepare_plots(self):
        t = self.timestamps[0]

        phase_fit_params = self.fit_res['coherence_phase_fit'].params
        amps = self.proc_data_dict['scaling_amp']
        fit_text = "$\sigma = %.3f \pm %.3f$" % (
                        self.fit_res['coherence_fit'].params['sigma'].value,
                        self.fit_res['coherence_fit'].params['sigma'].stderr)
        fit_text += '\n$c=%.5f$'%(phase_fit_params['c'].value)

        self.plot_dicts['text_msg_amp_vs_dephasing'] = {
                    'ax_id': 'amp_vs_dephasing',
                    # 'ypos': 0.15,
                    'plotfn': self.plot_text,
                    'box_props': 'fancy',
                    'text_string': fit_text,
        }
        'dirty hack to rescale y-axis in the plots'
        b=self.fit_res['coherence_fit']
        scale_amp=b.best_values['scale']
                


        self.plot_dicts['amp_vs_dephasing_fit'] = {
            'plotfn': self.plot_fit,
            #'plot_init' : True,
            'ax_id': 'amp_vs_dephasing',
            'plot_normed':True,
            'zorder': 5,
            'fit_res': self.fit_res['coherence_fit'],
            'xvals': amps,
            'marker': '',
            'linestyle': '-',
            'ylabel': r'Relative contrast', #r'Coherence, $\left| \rho_{01} \right|$'
            'yunit': '',
            'xlabel': 'scaling amplitude',
            'xunit': 'rel. amp.',
            'setlabel': 'coherence fit',
            'color': 'red',
        }
        fit_text = 'Fit Result:\n$y=(x \cdot s)^2 + \\varphi$\n'
        fit_text += '$s=%.2f$, '%(phase_fit_params['s'].value) #, phase_fit_params['s'].stderr
        fit_text += '$\\varphi=%.1f$\n'%(phase_fit_params['b'].value) #, phase_fit_params['b'].stderr
        fit_text += '$\Rightarrow c=%.5f$'%(phase_fit_params['c'].value)
        self.plot_dicts['text_msg_amp_vs_dephasing_phase'] = {
                    'ax_id': 'amp_vs_dephasing_phase',
                    'xpos': 1.05,
                    'plotfn': self.plot_text,
                    'box_props': 'fancy',
                    'horizontalalignment': 'left',
                    'text_string': fit_text,
        }
        self.plot_dicts['amp_vs_dephasing_phase_fit'] = {
            'plotfn': self.plot_line,
            'ax_id': 'amp_vs_dephasing_phase',
            'zorder': 5,
            'xvals': self.proc_data_dict['coherence_phase_fit']['amps'],
            'yvals': self.proc_data_dict['coherence_phase_fit']['phase'],
            'marker': '',
            'linestyle': '-',
            'ylabel': r'Phase',
            'yunit': 'Deg.',
            'xlabel': 'scaling amplitude',
            'xunit': 'rel. amp.',
            'setlabel': 'phase fit',
            'color': 'red',
        }
        if self.options_dict.get('plot_guess', False):
            self.plot_dicts['amp_vs_dephasing_phase_fit_guess'] = {
                'plotfn': self.plot_line,
                'ax_id': 'amp_vs_dephasing_phase',
                'zorder': 1,
                'xvals': self.proc_data_dict['coherence_phase_fit']['amps'],
                'yvals': self.proc_data_dict['coherence_phase_fit']['phase_guess'],
                'marker': '',
                'linestyle': '-',
                'ylabel': r'Phase',
                'yunit': 'Deg.',
                'xlabel': 'scaling amplitude',
                'xunit': 'rel. amp.',
                'setlabel': 'phase fit (guess)',
                'color': 'lightgray',
                'alpha' : 0.1,
            }

        fit_mask = self.proc_data_dict['fitting_mask']
        fit_mask_inv = self.proc_data_dict['fitting_mask_inv']
        use_ext = len(fit_mask) > 0 and len(fit_mask_inv) > 0
        if len(fit_mask) > 0:
            label1 = 'Coherence data'
            label2 = 'Phase'
            if use_ext:
                label1 += ' (used in fitting)'
                label2 += ' (used in fitting)'
            self.plot_dicts['amp_vs_dephasing_fitted'] = {
                'title' : t,
                'plotfn': self.plot_line,
                'ax_id': 'amp_vs_dephasing',
                'zorder': 0,
                'xvals': amps[fit_mask],
                'yvals': self.proc_data_dict['coherence'][fit_mask]/scale_amp,
                'marker': 'o',
                'linestyle': '',
                'setlabel': label1,
                'color': 'red',
            }
            self.plot_dicts['amp_vs_dephasing_phase_not_fitted'] = {
                'title' : t,
                'plotfn': self.plot_line,
                'ax_id': 'amp_vs_dephasing_phase',
                'xvals': amps[fit_mask],
                'yvals': self.proc_data_dict['phase'][fit_mask],
                'marker': 'x',
                'linestyle': '',
                'ylabel': label2,
                'yunit': 'deg.',
                'xlabel': 'scaling amplitude',
                'xunit': 'rel. amp.',
                'setlabel': 'dephasing phase data',
            }
        if len(fit_mask_inv) > 0:
            label1 = 'Coherence data'
            label2 = 'Phase'
            if use_ext:
                label1 += ' (not fitted)'
                label2 += ' (not fitted)'
            self.plot_dicts['amp_vs_dephasing_not_fitted'] = {
                'title' : t,
                'plotfn': self.plot_line,
                'ax_id': 'amp_vs_dephasing',
                'zorder': 1,
                'xvals': amps[fit_mask_inv],
                'yvals': self.proc_data_dict['coherence'][fit_mask_inv],
                'marker': 'x',
                'linestyle': '',
                'setlabel': label1,
                'color': 'red',
                }
            self.plot_dicts['amp_vs_dephasing_phase_not_fitted'] = {
                'title' : t,
                'plotfn': self.plot_line,
                'ax_id': 'amp_vs_dephasing_phase',
                'xvals': amps[fit_mask_inv],
                'yvals': self.proc_data_dict['phase'][fit_mask_inv],
                'marker': 'x',
                'linestyle': '',
                'ylabel': label2,
                'yunit': 'deg.',
                'xlabel': 'scaling amplitude',
                'xunit': 'rel. amp.',
                'setlabel': 'dephasing phase data',
            }


class DephasingAnalysisSweep(DephasingAnalysis):
    '''
    Gathers/Loads data from a single coherence/dephasing (e.g. Ramsey/) sweep scan
    and analyses it (see DephasingAnalysis).
    '''
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '_ro_amp_sweep_dephasing',
                 options_dict: dict = None, extract_only: bool = False,
                 auto: bool = True, close_figs: bool = True,
                 do_fitting: bool = True):
        super().__init__(t_start=t_start, t_stop=t_start,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )
        self.single_timestamp = True
        ts = a_tools.get_timestamps_in_range(timestamp_start=t_start,
                                        timestamp_end=t_stop, label=label,
                                        exact_label_match=True)
        if self.verbose:
            print('DephasingAnalysisSweep', ts)
        assert len(ts) == 1, 'Expected a single match, found %d'%len(ts)
        self.timestamp = ts[0]

        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        data_file = MeasurementAnalysis(label=self.labels[0],
                                        timestamp=self.timestamp,
                                        auto=True, TwoD=False)

        dateobj = a_tools.datetime_from_timestamp(self.timestamp)
        self.timestamps = [self.timestamp]
        self.raw_data_dict['timestamps'] = [self.timestamp]
        self.raw_data_dict['datetime'] = np.array([dateobj], dtype=datetime.datetime)

        temp = data_file.load_hdf5data()
        data_file.get_naming_and_values()
        self.raw_data_dict['scaling_amp'] = data_file.sweep_points
        self.raw_data_dict['dephasing'] = np.array(data_file.measured_values[0], dtype=float)
        self.raw_data_dict['phase'] = np.array(data_file.measured_values[1], dtype=float)
        self.raw_data_dict['folder'] = data_file.folder


class DephasingAnalysisSingleScans(DephasingAnalysis):
    '''
    Gathers/Loads data from a range of single coherence/dephasing scans (e.g. Ramsey/)
    and analyses it (see DephasingAnalysis).

    options_dict options:
     - Inherited option from DephasingAnalysis
     - scaling_amp_key_dephasing (string) - key of the scaling amp in the hdf5 file
                                             e.g. 'Instrument settings.RO_lutman.M_amp_R0'
     - dephasing_amplitude_key (string) - key of the coherence amp in the hdf5 file
                                           e.g. 'Analysis.Fitted Params lin_trans w0.amplitude.value'
     - dephasing_phase_key: (string) - key of the coherence phase in the hdf5 file
                                        e.g. 'Analysis.Fitted Params lin_trans w0.phase.value'
    '''

    def __init__(self, t_start: str = None, t_stop: str = None, label: str = '_dephasing',
                 options_dict: dict = None, extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only
                         )
        sa = self.options_dict.get('scaling_amp_key_dephasing', 'Instrument settings.RO_lutman.M_amp_R0')
        rak = self.options_dict.get('dephasing_amplitude_key', 'Analysis.Fitted Params lin_trans w0.amplitude.value')
        rap = self.options_dict.get('dephasing_phase_key', 'Analysis.Fitted Params lin_trans w0.phase.value')
        self.params_dict = {'scaling_amp': sa,
                            'dephasing': rak,
                            'phase': rap,
                            }
        self.numeric_params = ['scaling_amp', 'dephasing', 'phase']

        if auto:
            self.run_analysis()

    def extract_data(self):
        # Load data
        super().extract_data()
        #todo: we need an option to remove outliers and the reference point

        # Set output paths
        youngest = np.max(self.raw_data_dict['datetime'])
        youngest += datetime.timedelta(seconds=1)

        f = '%s_amp_sweep_dephasing' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = os.path.join(folder, f + '.hdf5')


class SSROAnalysis(ba.BaseDataAnalysis):
    def process_data(self):

        # Remove None entries
        snr = np.array(self.raw_data_dict['SNR'], dtype=float)
        amps = np.array(self.raw_data_dict['scaling_amp'], dtype=float)
        mask = np.intersect1d(np.where(snr != None), np.where(amps != None))
        self.proc_data_dict['scaling_amp'] = amps[mask]
        self.proc_data_dict['SNR'] = snr[mask]
        self.proc_data_dict['F_a'] = np.array(self.raw_data_dict['F_a'], dtype=float)[mask]
        self.proc_data_dict['F_d'] = np.array(self.raw_data_dict['F_d'], dtype=float)[mask]

        # Fitting masks
        mask = range(0, len(amps))
        inv_mask = []
        if self.options_dict.get('amp_threshold', False):
            mask = np.where(amps < self.options_dict['amp_threshold'])
            inv_mask = np.where(amps >= self.options_dict['amp_threshold'])
        self.proc_data_dict['fitting_mask'] = mask
        self.proc_data_dict['fitting_mask_inv'] = inv_mask

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        SNR = self.proc_data_dict['SNR']
        amps = self.proc_data_dict['scaling_amp']

        mask = self.proc_data_dict['fitting_mask']
        amps = amps[mask]
        SNR = SNR[mask]

        def line(x, a):
            return a * x

        gmodel = lmfit.models.Model(line)
        gmodel.set_param_hint('a', value=1, min=1e-5, max=100)
        para = gmodel.make_params()

        self.fit_dicts['snr_fit'] = {
            'model': gmodel,
            'fit_xvals': {'x': amps},
            'fit_yvals': {'data': SNR},
            'guess_pars': para,
        }

    def prepare_plots(self):
        t = self.timestamps[0]

        amps = self.proc_data_dict['scaling_amp']
        name = ''
        self.plot_dicts[name + 'amp_vs_SNR_fit'] = {
            'title' : t,
            'plotfn': self.plot_fit,
            'ax_id': name + 'amp_vs_SNR',
            'zorder': 5,
            'fit_res': self.fit_res['snr_fit'],
            'xvals': self.proc_data_dict['scaling_amp'],
            'marker': '',
            'linestyle': '-',
            'setlabel': 'SNR fit',
            'do_legend': True,
            'color': 'blue',
        }
        fit_mask = self.proc_data_dict['fitting_mask']
        fit_mask_inv = self.proc_data_dict['fitting_mask_inv']
        use_ext = len(fit_mask) > 0 and len(fit_mask_inv) > 0
        if len(fit_mask) > 0:
            label = 'SNR data'
            if use_ext:
                label += ' (used in fitting)'
            self.plot_dicts[name + 'amp_vs_SNR_scatter_fitted'] = {
                'title' : t,
                'plotfn': self.plot_line,
                'ax_id': name + 'amp_vs_SNR',
                'zorder': 0,
                'xvals': amps[fit_mask],
                'xlabel': 'scaling amplitude',
                'xunit': 'rel. amp.',
                'yvals': self.proc_data_dict['SNR'][fit_mask],
                'ylabel': 'SNR',
                'yunit': '-',
                'marker': 'o',
                'linestyle': '',
                'setlabel': label,
                'do_legend': True,
                'color': 'blue',
            }

        if len(fit_mask_inv) > 0:
            label = 'SNR data'
            if use_ext:
                label += ' (not fitted)'
            self.plot_dicts[name + 'amp_vs_SNR_scatter_not_fitted'] = {
                'title' : t,
                'plotfn': self.plot_line,
                'ax_id': name + 'amp_vs_SNR',
                'zorder': 0,
                'xvals': amps[fit_mask_inv],
                'xlabel': 'scaling amplitude',
                'xunit': 'rel. amp.',
                'yvals': self.proc_data_dict['SNR'][fit_mask_inv],
                'ylabel': 'SNR',
                'yunit': '-',
                'marker': 'x',
                'linestyle': '',
                'setlabel': label,
                'do_legend': True,
                'color': 'blue',
        }
        self.plot_dicts[name + 'amp_vs_Fa'] = {
            'title' : t,
            'plotfn': self.plot_line,
            'zorder': 0,
            'ax_id': name + 'amp_vs_F',
            'xvals': amps,
            'yvals': self.proc_data_dict['F_a'],
            'marker': 'x',
            'linestyle': '',
            'setlabel': '$F_a$ data',
            'do_legend': True,
        }
        self.plot_dicts[name + 'amp_vs_Fd'] = {
            'title' : t,
            'plotfn': self.plot_line,
            'zorder': 1,
            'ax_id': name + 'amp_vs_F',
            'xvals': amps,
            'yvals': self.proc_data_dict['F_d'],
            'marker': 'x',
            'linestyle': '',
            'ylabel': 'Fidelity',
            'yunit': '-',
            'xlabel': 'scaling amplitude',
            'xunit': 'rel. amp.',
            'setlabel': '$F_d$ data',
            'do_legend': True,
        }


class SSROAnalysisSweep(SSROAnalysis):
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '_ro_amp_sweep_SNR',
                 options_dict: dict = None, extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True):
        super().__init__(t_start=t_start, t_stop=t_start,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )
        self.single_timestamp = True

        ts = a_tools.get_timestamps_in_range(timestamp_start=t_start,
                                             timestamp_end=t_stop,
                                             label=label,
                                             exact_label_match=True)
        if self.verbose:
            print('SSROAnalysisSweep', ts)
        assert len(ts) == 1, 'Expected a single match, found %d'%len(ts)
        self.timestamp = ts[0]

        if auto:
            self.run_analysis()

    def extract_data(self):
        self.raw_data_dict = OrderedDict()
        data_file = MeasurementAnalysis(timestamp=self.timestamp,
                                        auto=True, TwoD=False)

        dateobj = a_tools.datetime_from_timestamp(self.timestamp)
        self.timestamps = [self.timestamp]
        self.raw_data_dict['timestamps'] = [self.timestamp]
        self.raw_data_dict['datetime'] = np.array([dateobj], dtype=datetime.datetime)

        temp = data_file.load_hdf5data()
        data_file.get_naming_and_values()
        self.raw_data_dict['scaling_amp'] = data_file.sweep_points
        self.raw_data_dict['SNR'] = np.array(data_file.measured_values[0], dtype=float)
        self.raw_data_dict['F_d'] = np.array(data_file.measured_values[1], dtype=float)
        self.raw_data_dict['F_a'] = np.array(data_file.measured_values[2], dtype=float)
        self.raw_data_dict['folder'] = data_file.folder


class SSROAnalysisSingleScans(SSROAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '_SSRO', options_dict: dict = None,
                 extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )

        sa = self.options_dict.get('scaling_amp_key_ssro',
                                   'Instrument settings.RO_lutman.M_amp_R0')

        self.params_dict = {'scaling_amp': sa,
                            'SNR': 'Analysis.SSRO_Fidelity.SNR',
                            'F_a': 'Analysis.SSRO_Fidelity.F_a',
                            'F_d': 'Analysis.SSRO_Fidelity.F_d',
                            }
        self.numeric_params = ['scaling_amp', 'SNR', 'F_a', 'F_d']

        if auto:
            self.run_analysis()

    def extract_data(self):
        # Load data
        super().extract_data()

        #todo: we need an option to remove outliers and the reference point
        # Set output paths
        youngest = np.max(self.raw_data_dict['datetime'])
        youngest += datetime.timedelta(seconds=1)
        f = '%s_amp_sweep_SNR_optimized' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = os.path.join(folder, f + '.hdf5')


