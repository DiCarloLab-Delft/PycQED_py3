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
            'title': '' + title,  # todo
            'yvals': twpa_powers, 'ylabel': r'TWPA Power', 'yunit': 'dBm',
            'xvals': twpa_freqs, 'xlabel': 'TWPA Frequency', 'xunit': 'Hz',
            'zvals': self.proc_data_dict['etas'].transpose() * 100,
            'zlabel': r'Quantum efficiency $\eta$ (%)',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['quantum_eff_vari'] = {
            'plotfn': self.plot_colorxy,
            'title': '' + title,  # todo
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
        self.options_dict['analysis_result_file'] = os.path.join(folder, f + '.hdf5')

    def run_fitting(self):
        self.ra.run_analysis()
        self.ssro.run_analysis()

        self.fit_dicts = OrderedDict()
        self.fit_dicts['sigma'] = self.ra.fit_dicts['coherence_fit']['sigma']
        self.fit_dicts['sigma_std'] = self.ra.fit_dicts['coherence_fit']['sigma_std']
        # self.raw_data_dict['scale'] = self.ra.fit_dicts['coherence_fit']['scale']
        self.fit_dicts['a'] = self.ssro.fit_dicts['snr_fit']['a']
        self.fit_dicts['a_std'] = self.ssro.fit_dicts['snr_fit']['a_std']

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
        # self.ra.plot_dicts['amp_vs_dephasing_fit']
        # self.ra.plot_dicts['amp_vs_dephasing_Phase']
        # self.ra.plot_dicts['amp_vs_dephasing_coherence']

        # self.ssro.plot_dicts['amp_vs_SNR_fit']
        # self.ssro.plot_dicts['amp_vs_SNR_scatter']
        # self.ssro.plot_dicts['amp_vs_Fa']
        # self.ssro.plot_dicts['amp_vs_Fd']
        self.ra.prepare_plots()
        for d in self.ra.plot_dicts:
            self.plot_dicts[d] = self.ra.plot_dicts[d]
        self.ssro.prepare_plots()
        for d in self.ssro.plot_dicts:
            self.plot_dicts[d] = self.ssro.plot_dicts[d]

        if self.options_dict.get('subplots', True):
            for k in ['amp_vs_dephasing_coherence_fitted',
                      'amp_vs_dephasing_coherence_not_fitted',
                      'amp_vs_dephasing_fit',
                      'amp_vs_SNR_scatter_fitted',
                      'amp_vs_SNR_scatter_not_fitted',
                      'amp_vs_SNR_fit',]:
                self.plot_dicts[k]['ax_id'] = 'snr_analysis'
                self.plot_dicts[k]['ylabel'] = 'SNR, coherence'
                self.plot_dicts[k]['yunit'] = '(-)'
                self.plot_dicts[k]['title'] = r'$\eta = (%.4f \pm %.4f)$ %%' % (
                    100 * self.fit_dicts['eta'], 100 * self.fit_dicts['u_eta'])
                + title

            #self.plot_dicts['amp_vs_dephasing_fit']['color'] = 'red'
            #self.plot_dicts['amp_vs_dephasing_coherence_fitted']['color'] = 'red'
            #self.plot_dicts['amp_vs_dephasing_coherence_not_fitted']['color'] = 'red'
            #self.plot_dicts['amp_vs_SNR_fit']['color'] = 'blue'
            #self.plot_dicts['amp_vs_SNR_scatter_fitted']['color'] = 'blue'
            #self.plot_dicts['amp_vs_SNR_scatter_not_fitted']['color'] = 'blue'

            # self.plot_dicts['amp_vs_dephasing_coherence']['marker'] = 'o'
            # self.plot_dicts['amp_vs_SNR_scatter']['marker'] = 'o'

            self.plot_dicts['amp_vs_SNR_fit']['do_legend'] = True
            self.plot_dicts['amp_vs_dephasing_fit']['do_legend'] = True


class DephasingAnalysis(ba.BaseDataAnalysis):

    def process_data(self):
        # Remove None entries
        dephasing = self.raw_data_dict['dephasing']
        amps = self.raw_data_dict['scaling_amp']
        mask = np.intersect1d(np.where(dephasing != None), np.where(amps != None))

        self.proc_data_dict['scaling_amp'] = amps[mask]
        self.proc_data_dict['coherence'] = dephasing[mask]
        self.proc_data_dict['phase'] = self.raw_data_dict['phase'][mask]

    def run_fitting(self):
        self.fit_res = OrderedDict()
        self.fit_dicts['coherence_fit'] = OrderedDict()
        coherence = self.proc_data_dict['coherence']
        amps = self.proc_data_dict['scaling_amp']

        mask = range(0, len(amps))
        inv_mask = []
        if self.options_dict.get('amp_threshold', False):
            mask = np.where(amps < self.options_dict['amp_threshold'])
            inv_mask = np.where(amps >= self.options_dict['amp_threshold'])
            amps = amps[mask]
            coherence = coherence[mask]
        self.fit_dicts['coherence_fit']['mask'] = mask
        self.fit_dicts['coherence_fit']['inv_mask'] = inv_mask

        def gaussian(x, sigma, scale):
            return scale * np.exp(-(x) ** 2 / (2 * sigma ** 2))

        gmodel = lmfit.models.Model(gaussian)
        gmodel.set_param_hint('sigma', value=0.5, min=0, max=10)
        gmodel.set_param_hint('scale', value=np.max(coherence))  # , min=0.1, max=100)
        para = gmodel.make_params()
        coherence_fit = gmodel.fit(coherence, x=amps, **para)
        self.fit_res['coherence_fit'] = coherence_fit
        self.fit_dicts['coherence_fit']['sigma'] = coherence_fit.params['sigma'].value
        self.fit_dicts['coherence_fit']['sigma_std'] = coherence_fit.params['sigma'].stderr
        self.fit_dicts['coherence_fit']['scale'] = coherence_fit.params['scale'].value
        self.fit_dicts['coherence_fit']['scale_std'] = coherence_fit.params['scale'].stderr

    def prepare_plots(self):
        amps = self.proc_data_dict['scaling_amp']
        name = ''
        fit_text = "$\sigma = %.3f \pm %.3f$" % (
                        self.fit_dicts['coherence_fit']['sigma'],
                        self.fit_dicts['coherence_fit']['sigma_std'])
        self.plot_dicts['text_msg_' + name + 'amp_vs_dephasing'] = {
                    'ax_id': name + 'amp_vs_dephasing',
                    # 'ypos': 0.15,
                    'plotfn': self.plot_text,
                    'box_props': 'fancy',
                    'text_string': fit_text,
        }
        self.plot_dicts[name + 'amp_vs_dephasing_fit'] = {
            'plotfn': self.plot_fit,
            'ax_id': name + 'amp_vs_dephasing',
            'zorder': 5,
            'fit_res': self.fit_res['coherence_fit'],
            'xvals': amps,
            'marker': '',
            'linestyle': '-',
            'ylabel': r'Coherence, $\left| \rho_{01} \right|$',
            'yunit': '',
            'xlabel': 'scaling amplitude',
            'xunit': 'rel. amp.',
            'setlabel': 'coherence fit',
            'color': 'red',
        }

        fit_mask = self.fit_dicts['coherence_fit']['mask']
        fit_mask_inv = self.fit_dicts['coherence_fit']['inv_mask']
        self.plot_dicts[name + 'amp_vs_dephasing_coherence_fitted'] = {
            'plotfn': self.plot_line,
            'ax_id': name + 'amp_vs_dephasing',
            'zorder': 0,
            'xvals': amps[fit_mask],
            'yvals': self.proc_data_dict['coherence'][fit_mask],
            'marker': 'o',
            'linestyle': '',
            'setlabel': 'coherence data',
            'color': 'red',
        }
        self.plot_dicts[name + 'amp_vs_dephasing_coherence_not_fitted'] = {
            'plotfn': self.plot_line,
            'ax_id': name + 'amp_vs_dephasing',
            'zorder': 1,
            'xvals': amps[fit_mask_inv],
            'yvals': self.proc_data_dict['coherence'][fit_mask_inv],
            'marker': 'x',
            'linestyle': '',
            'setlabel': 'coherence data (not fitted)',
            'color': 'red',
        }
        self.plot_dicts[name + 'amp_vs_dephasing_Phase'] = {
            'plotfn': self.plot_line,
            'xvals': amps,
            'yvals': self.proc_data_dict['phase'],
            'marker': 'x',
            'linestyle': '',
            'ylabel': 'Phase',
            'yunit': 'deg.',
            'xlabel': 'scaling amplitude',
            'xunit': 'rel. amp.',
            'setlabel': 'dephasing phase data',
        }


class DephasingAnalysisSweep(DephasingAnalysis):
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

    def run_fitting(self):
        self.fit_res = OrderedDict()
        self.fit_dicts['snr_fit'] = OrderedDict()
        SNR = self.proc_data_dict['SNR']
        amps = self.proc_data_dict['scaling_amp']

        mask = range(0, len(amps))
        inv_mask = []
        if self.options_dict.get('amp_threshold', False):
            mask = np.where(amps < self.options_dict['amp_threshold'])
            inv_mask = np.where(amps >= self.options_dict['amp_threshold'])
            amps = amps[mask]
            SNR = SNR[mask]
        self.fit_dicts['snr_fit']['mask'] = mask
        self.fit_dicts['snr_fit']['inv_mask'] = inv_mask

        def line(x, a):
            return a * x

        gmodel = lmfit.models.Model(line)
        gmodel.set_param_hint('a', value=1, min=1e-5, max=100)
        para = gmodel.make_params()
        snr_fit = gmodel.fit(SNR, x=amps, **para)
        self.fit_res['snr_fit'] = snr_fit

        self.fit_dicts['snr_fit']['a'] = snr_fit.params['a'].value
        self.fit_dicts['snr_fit']['a_std'] = snr_fit.params['a'].stderr

    def prepare_plots(self):
        amps = self.proc_data_dict['scaling_amp']
        name = ''
        self.plot_dicts[name + 'amp_vs_SNR_fit'] = {
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
        fit_mask = self.fit_dicts['snr_fit']['mask']
        fit_mask_inv = self.fit_dicts['snr_fit']['inv_mask']
        self.plot_dicts[name + 'amp_vs_SNR_scatter_fitted'] = {
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
            'setlabel': 'SNR data',
            'do_legend': True,
            'color': 'blue',
        }
        self.plot_dicts[name + 'amp_vs_SNR_scatter_not_fitted'] = {
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
            'setlabel': 'SNR data (not fitted)',
            'do_legend': True,
            'color': 'blue',
        }
        self.plot_dicts[name + 'amp_vs_Fa'] = {
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

        # Set output paths
        youngest = np.max(self.raw_data_dict['datetime'])
        youngest += datetime.timedelta(seconds=1)
        f = '%s_amp_sweep_SNR_optimized' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = os.path.join(folder, f + '.hdf5')
