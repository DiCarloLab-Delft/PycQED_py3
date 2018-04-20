'''
Collection of classes to analyse Quantum efficiency measurements as a function of TWPA Pump frequency and power.
General procedure is: for each TWPA Pump frequency and power value,
 - find factor 'a' of the quadratic scaling of the Single-Shot-Readout as a function of clear_scaling_amp (see SSROAnalysis class)
 - find sigma from the Gaussian fit of the Ramsey as a function of clear_scaling_amp (see RamseyAnalysis class)
 - Calculate eta = a * sigma**2 / 2 (see QuantumEfficiencyAnalysis class)
For details, see https://arxiv.org/abs/1711.05336

Hacked together by Rene Vollmer
'''

import datetime
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis_v2.base_analysis import plot_scatter_errorbar_fit, plot_scatter_errorbar

import numpy as np
import lmfit
import os

from pycqed.analysis import analysis_toolbox as a_tools


class QuantumEfficiencyAnalysis(ba.BaseDataAnalysis):
    '''
    Analyses Quantum efficiency measurements as a function of TWPA Pump frequency and power.
    '''

    def __init__(self, t_start: str = None, t_stop: str = None, label_ramsey: str = '_Ramsey',
                 label_ssro: str = '_SSRO', options_dict: dict = None,
                 extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True,
                 twpa_pump_freq_key: str = 'Instrument settings.TWPA_Pump.frequency',
                 twpa_pump_power_key: str = 'Instrument settings.TWPA_Pump.power'):
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
        :param label_ramsey: the label that was used to name the ramsey measurements
        :param label_ssro: the label that was used to name the SSRO measurements
        :param twpa_pump_freq_key: key for the TWPA Pump Frequency, e.g. 'Instrument settings.TWPA_Pump.frequency'
        :param twpa_pump_power_key: key for the TWPA Pump Power, e.g. 'Instrument settings.TWPA_Pump.power'
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )
        self.ra = RamseyAnalysis(t_start=t_start, t_stop=t_stop, label=label_ramsey, options_dict=options_dict,
                                 auto=False, twpa_pump_freq_key=twpa_pump_freq_key,
                                 twpa_pump_power_key=twpa_pump_power_key)
        self.ssro = SSROAnalysis(t_start=t_start, t_stop=t_stop, label=label_ssro, options_dict=options_dict,
                                 auto=False, twpa_pump_freq_key=twpa_pump_freq_key,
                                 twpa_pump_power_key=twpa_pump_power_key)

        if auto:
            self.run_analysis()

    def extract_data(self):
        self.ra.run_analysis()
        self.ssro.run_analysis()
        self.raw_data_dict = {}
        self.raw_data_dict['TWPA_freq_ramesy'] = np.array(self.ra.proc_data_dict['TWPA_freqs'], dtype=float)
        self.raw_data_dict['TWPA_freq_ssro'] = np.array(self.ssro.proc_data_dict['TWPA_freqs'], dtype=float)
        self.raw_data_dict['TWPA_power_ramesy'] = np.array(self.ra.proc_data_dict['TWPA_powers'], dtype=float)
        self.raw_data_dict['TWPA_power_ssro'] = np.array(self.ssro.proc_data_dict['TWPA_powers'], dtype=float)

        self.raw_data_dict['sigmas'] = self.ra.fit_dicts['coherence_fit']['sigma']
        self.raw_data_dict['sigmas_std'] = self.ra.fit_dicts['coherence_fit']['sigma_std']
        # self.raw_data_dict['scales'] = self.ra.fit_dicts['coherence_fit']['scale']
        self.raw_data_dict['as'] = self.ssro.fit_dicts['snr_fit']['a']
        self.raw_data_dict['as_std'] = self.ssro.fit_dicts['snr_fit']['a_std']

        youngest = max(np.max(self.ra.raw_data_dict['datetime']), np.max(self.ssro.raw_data_dict['datetime']))
        youngest += datetime.timedelta(seconds=1)
        self.raw_data_dict['datetime'] = [youngest]
        self.raw_data_dict['timestamps'] = [youngest.strftime("%Y%m%d_%H%M%S")]
        self.timestamps = [youngest]
        folder = a_tools.datadir + '/%s_quantum_efficiency_analysis' % (youngest.strftime("%Y%m%d/%H%M%S"))
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = os.path.join(folder, f + '.hdf5')

    def process_data(self):
        self.proc_data_dict = {}
        sigma = self.raw_data_dict['sigmas']
        u_sigma = self.raw_data_dict['sigmas_std']
        a = self.raw_data_dict['as']
        u_a = self.raw_data_dict['as_std']

        eta = a * sigma ** 2 / 2
        u_eta = (u_a / a + 2 * u_sigma / sigma) * eta

        self.proc_data_dict['etas'] = np.array(eta, dtype=float)
        self.proc_data_dict['u_etas'] = np.array(u_eta, dtype=float)

    def prepare_plots(self):
        self.plot_dicts['quantum_eff'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': self.raw_data_dict['TWPA_power_ramesy'],
            'ylabel': r'TWPA Power',
            'yunit': 'dBm',
            'xvals': self.raw_data_dict['TWPA_freq_ramesy'],
            'xlabel': 'TWPA Frequency',
            'xunit': 'Hz',
            'zvals': self.proc_data_dict['etas'].transpose() * 100,
            'zlabel': r'Quantum efficiency $\eta$ (%)',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }
        self.plot_dicts['quantum_eff_vari'] = {
            'plotfn': self.plot_colorxy,
            'title': '',  # todo
            'yvals': self.raw_data_dict['TWPA_power_ramesy'],
            'ylabel': r'TWPA Power',
            'yunit': 'dBm',
            'xvals': self.raw_data_dict['TWPA_freq_ramesy'],
            'xlabel': 'TWPA Frequency',
            'xunit': 'Hz',
            'zvals': self.proc_data_dict['u_etas'].transpose(),
            'zlabel': r'Quantum efficiency Deviation $\delta \eta$',
            'plotsize': self.options_dict.get('plotsize', None),
            'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        }


class RamseyAnalysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None, label: str = '_Ramsey',
                 options_dict: dict = None, extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True,
                 twpa_pump_freq_key: str = 'Instrument settings.TWPA_Pump.frequency',
                 twpa_pump_power_key: str = 'Instrument settings.TWPA_Pump.power'):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only
                         )

        self.params_dict = {'clear_scaling_amp': 'Instrument settings.RO_lutman.M_amp_R0',
                            'dephasing': 'Analysis.Fitted Params lin_trans w0.amplitude.value',
                            'phase': 'Analysis.Fitted Params lin_trans w0.phase.value',
                            'TWPA_freq': twpa_pump_freq_key,
                            'TWPA_power': twpa_pump_power_key,
                            }
        self.numeric_params = ['clear_scaling_amp', 'dephasing', 'phase', 'TWPA_freq', 'TWPA_power']

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
        # todo: factor 2?
        self.raw_data_dict['coherence'] = self.raw_data_dict['dephasing'] / 2

        youngest = np.max(self.raw_data_dict['datetime'])
        youngest += datetime.timedelta(seconds=1)

        f = '%s_CLEAR_amp_sweep_ramsey' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = self.options_dict.get('analysis_result_file',
                                                                          os.path.join(folder, f + '.hdf5'))

    def process_data(self):
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

        coherence = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        phase = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        ampl = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))

        for i, twpa_freq in enumerate(twpa_freqs):
            freq_indices = np.where(twpa_freqs_unsorted == twpa_freq)
            for j, twpa_power in enumerate(twpa_powers):
                power_indices = np.where(twpa_powers_unsorted == twpa_power)
                indices = np.intersect1d(freq_indices, power_indices)
                coherence[i, j] = self.raw_data_dict['coherence'][indices]
                phase[i, j] = self.raw_data_dict['phase'][indices]
                ampl[i, j] = self.raw_data_dict['clear_scaling_amp'][indices]

        self.proc_data_dict['coherence'] = coherence
        self.proc_data_dict['phase'] = phase
        self.proc_data_dict['clear_scaling_amp'] = ampl

    def run_fitting(self):
        twpa_powers = self.proc_data_dict['TWPA_powers']
        twpa_freqs = self.proc_data_dict['TWPA_freqs']
        self.fit_res = {}
        self.fit_res['coherence_fit'] = {}
        self.fit_dicts['coherence_fit'] = {}
        self.fit_dicts['coherence_fit']['sigma'] = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        self.fit_dicts['coherence_fit']['sigma_std'] = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        self.fit_dicts['coherence_fit']['scale'] = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        self.fit_dicts['coherence_fit']['scale_std'] = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        for i, twpa_freq in enumerate(twpa_freqs):
            self.fit_res['coherence_fit'][twpa_freq] = {}
            for j, twpa_power in enumerate(twpa_powers):
                coherence = self.proc_data_dict['coherence'][i, j]
                clear_scaling_amp = self.proc_data_dict['clear_scaling_amp'][i, j]

                def gaussian(x, sigma, scale):
                    return scale * np.exp(-(x) ** 2 / (2 * sigma ** 2))

                gmodel = lmfit.models.Model(gaussian)
                gmodel.set_param_hint('sigma', value=0.07, min=-1, max=1)
                gmodel.set_param_hint('scale', value=0.9)  # , min=0.1, max=100)
                para = gmodel.make_params()
                coherence_fit = gmodel.fit(coherence, x=clear_scaling_amp, **para)
                self.fit_res['coherence_fit'][twpa_freq][twpa_power] = coherence_fit
                self.fit_dicts['coherence_fit']['sigma'][i, j] = coherence_fit.params['sigma'].value
                self.fit_dicts['coherence_fit']['sigma_std'][i, j] = coherence_fit.params['sigma'].stderr
                self.fit_dicts['coherence_fit']['scale'][i, j] = coherence_fit.params['scale'].value
                self.fit_dicts['coherence_fit']['scale_std'][i, j] = coherence_fit.params['scale'].stderr

    def save_fit_results(self):
        # todo: if you want to save some results to a hdf5, do it here
        pass

    def prepare_plots(self):
        # todo: add 2D plot
        if self.options_dict.get('individual_plots', False):
            for i, twpa_freq in enumerate(self.proc_data_dict['TWPA_freqs']):
                for j, twpa_power in enumerate(self.proc_data_dict['TWPA_powers']):
                    name = 'f_%.3f-p_%.1f' % (twpa_freq * 1e-9, twpa_power)
                    if self.proc_data_dict['clear_scaling_amp'][i, j] is not None:
                        self.plot_dicts[name + 'CLEAR_vs_Ramsey_fit'] = {
                            'plotfn': self.plot_fit,
                            'ax_id': name + 'CLEAR_vs_Ramsey',
                            'zorder': 5,
                            'fit_res': self.fit_res['coherence_fit'][twpa_freq][twpa_power],
                            'xvals': self.proc_data_dict['clear_scaling_amp'][i, j],
                            'marker': '',
                            'linestyle': '-',
                            'ylabel': 'Coherence',
                            'yunit': 'a.u.',
                            'xlabel': 'CLEAR scaling amplitude',
                            'xunit': 'V',
                        }
                        self.plot_dicts[name + 'CLEAR_vs_Ramsey_coherence'] = {
                            'plotfn': self.plot_line,
                            'ax_id': name + 'CLEAR_vs_Ramsey',
                            'zorder': 0,
                            'xvals': self.proc_data_dict['clear_scaling_amp'][i, j],
                            'yvals': self.proc_data_dict['coherence'][i, j],
                            'marker': 'x',
                            'linestyle': '',
                        }
                        self.plot_dicts[name + 'CLEAR_vs_Ramsey_Phase'] = {
                            'plotfn': self.plot_line,
                            'xvals': self.proc_data_dict['clear_scaling_amp'][i, j],
                            'yvals': self.proc_data_dict['phase'][i, j],
                            'marker': 'x',
                            'linestyle': '',
                            'ylabel': 'Phase',
                            'yunit': 'deg.',
                            'xlabel': 'CLEAR scaling amplitude',
                            'xunit': 'V',
                        }


class SSROAnalysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None, t_stop: str = None, label: str = '_SSRO',
                 options_dict: dict = None, extract_only: bool = False, auto: bool = True,
                 close_figs: bool = True, do_fitting: bool = True,
                 twpa_pump_freq_key: str = 'Instrument settings.TWPA_Pump.frequency',
                 twpa_pump_power_key: str = 'Instrument settings.TWPA_Pump.power'):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         do_fitting=do_fitting,
                         close_figs=close_figs,
                         extract_only=extract_only,
                         )

        self.params_dict = {'clear_scaling_amp': 'Instrument settings.RO_lutman.M_amp_R0',
                            'SNR': 'Analysis.SSRO_Fidelity.SNR',
                            'F_a': 'Analysis.SSRO_Fidelity.F_a',
                            'F_d': 'Analysis.SSRO_Fidelity.F_d',
                            'TWPA_freq': twpa_pump_freq_key,
                            'TWPA_power': twpa_pump_power_key,
                            }
        self.numeric_params = ['clear_scaling_amp', 'SNR', 'F_a', 'F_d', 'TWPA_freq', 'TWPA_power']

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()

        youngest = np.max(self.raw_data_dict['datetime'])
        youngest += datetime.timedelta(seconds=1)

        f = '%s_CLEAR_amp_sweep_SNR_optimized' % (youngest.strftime("%H%M%S"))
        d = '%s' % (youngest.strftime("%Y%m%d"))
        folder = os.path.join(a_tools.datadir, d, f)
        self.raw_data_dict['folder'] = [folder]
        self.options_dict['analysis_result_file'] = self.options_dict.get('analysis_result_file',
                                                                          os.path.join(folder, f + '.hdf5'))

    def process_data(self):
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

        snrs = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        fa = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        fd = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))
        ampl = np.array([[None] * len(twpa_powers)] * len(twpa_freqs))

        for i, twpa_freq in enumerate(twpa_freqs):
            freq_indices = np.where(twpa_freqs_unsorted == twpa_freq)
            for j, twpa_power in enumerate(twpa_powers):
                power_indices = np.where(twpa_powers_unsorted == twpa_power)
                reference_mask = np.where(self.raw_data_dict['clear_scaling_amp'] < 0.05)
                indices = np.intersect1d(freq_indices, power_indices)
                indices = np.intersect1d(indices, reference_mask)
                snrs[i, j] = self.raw_data_dict['SNR'][indices]
                fa[i, j] = self.raw_data_dict['F_a'][indices]
                fd[i, j] = self.raw_data_dict['F_d'][indices]
                ampl[i, j] = self.raw_data_dict['clear_scaling_amp'][indices]

        self.proc_data_dict['SNR'] = snrs
        self.proc_data_dict['F_a'] = fa
        self.proc_data_dict['F_d'] = fd
        self.proc_data_dict['clear_scaling_amp'] = ampl

    def run_fitting(self):
        self.fit_res = {}
        self.fit_res['snr_fit'] = {}
        self.fit_dicts['snr_fit'] = {}
        self.fit_dicts['snr_fit']['a'] = np.array(
            [[None] * len(self.proc_data_dict['TWPA_powers'])] * len(self.proc_data_dict['TWPA_freqs']))
        self.fit_dicts['snr_fit']['a_std'] = np.array(
            [[None] * len(self.proc_data_dict['TWPA_powers'])] * len(self.proc_data_dict['TWPA_freqs']))
        for i, twpa_freq in enumerate(self.proc_data_dict['TWPA_freqs']):
            self.fit_res['snr_fit'][twpa_freq] = {}
            for j, twpa_power in enumerate(self.proc_data_dict['TWPA_powers']):
                SNR = self.proc_data_dict['SNR'][i, j]
                clear_scaling_amp_dephasing = self.proc_data_dict['clear_scaling_amp'][i, j]

                def line(x, a):
                    return a * (x ** 2)

                gmodel = lmfit.models.Model(line)
                snr_fit = gmodel.fit(SNR, x=clear_scaling_amp_dephasing, a=1)
                self.fit_res['snr_fit'][twpa_freq][twpa_power] = snr_fit
                self.fit_dicts['snr_fit']['a'][i, j] = snr_fit.params['a'].value
                self.fit_dicts['snr_fit']['a_std'][i, j] = snr_fit.params['a'].stderr

    def save_fit_results(self):
        # todo: if you want to save some results to a hdf5, do it here
        pass

    def prepare_plots(self):
        # self.plot_dicts['CLEAR_vs_slope'] = {
        #     'plotfn': self.plot_colorxy,
        #     'title' : '', #todo
        #     'xvals': self.proc_data_dict['TWPA_freqs'] ,
        #     'yvals': self.proc_data_dict['TWPA_powers'] ,
        #     'zvals' : self.fit_dicts['snr_fit']['a'].transpose(),
        #     'xlabel': 'TWPA Frequency',
        #     'xunit': 'Hz',
        #     'ylabel': r'TWPA Power',
        #     'yunit': 'dBm',
        #     'zlabel' : 'SNR slope (1/V)',
        #     'plotsize': self.options_dict.get('plotsize', None),
        #     'cmap': self.options_dict.get('cmap', 'YlGn_r'),
        # }
        # todo: fix

        if self.options_dict.get('individual_plots', False):
            for i, twpa_freq in enumerate(self.proc_data_dict['TWPA_freqs']):
                for j, twpa_power in enumerate(self.proc_data_dict['TWPA_powers']):
                    name = 'f_%.3f-p_%.1f' % (twpa_freq * 1e-9, twpa_power)
                    if self.proc_data_dict['clear_scaling_amp'][i, j] is not None:
                        self.plot_dicts[name + 'CLEAR_vs_SNR_fit'] = {
                            'plotfn': self.plot_fit,
                            'ax_id': name + 'CLEAR_vs_Ramsey',
                            'zorder': 5,
                            'fit_res': self.fit_res['snr_fit'][twpa_freq][twpa_power],
                            'xvals': self.proc_data_dict['clear_scaling_amp'][i, j],
                            'marker': '',
                            'linestyle': '-',
                        }
                        self.plot_dicts[name + 'CLEAR_vs_SNR_scatter'] = {
                            'plotfn': self.plot_line,
                            'ax_id': name + 'CLEAR_vs_Ramsey',
                            'zorder': 0,
                            'xvals': self.proc_data_dict['clear_scaling_amp'][i, j],
                            'xlabel': 'CLEAR scaling amplitude',
                            'xunit': 'V',
                            'yvals': self.proc_data_dict['SNR'][i, j],
                            'ylabel': 'SNR',
                            'yunit': '-',
                            'marker': 'x',
                            'linestyle': '',
                        }
                        self.plot_dicts[name + 'CLEAR_vs_Fa'] = {
                            'plotfn': self.plot_line,
                            'zorder': 0,
                            'ax_id': name + 'CLEAR_vs_F',
                            'xvals': self.proc_data_dict['clear_scaling_amp'][i, j],
                            'yvals': self.proc_data_dict['F_a'][i, j],
                            'marker': 'x',
                            'linestyle': '',
                            'setlabel': '$F_a$',
                            'do_legend': True,
                        }
                        self.plot_dicts[name + 'CLEAR_vs_Fd'] = {
                            'plotfn': self.plot_line,
                            'zorder': 1,
                            'ax_id': name + 'CLEAR_vs_F',
                            'xvals': self.proc_data_dict['clear_scaling_amp'][i, j],
                            'yvals': self.proc_data_dict['F_d'][i, j],
                            'marker': 'x',
                            'linestyle': '',
                            'ylabel': 'Fidelity',
                            'yunit': '-',
                            'xlabel': 'CLEAR scaling amplitude',
                            'xunit': 'V',
                            'setlabel': '$F_d$',
                            'do_legend': True,
                        }
