import lmfit
from collections import OrderedDict
import pycqed.analysis_v2.base_analysis as ba
import pycqed.measurement.waveform_control_CC.waveform as wf
import pycqed.analysis.fitting_models as fit_mods
import numpy as np
from numpy.fft import fft, ifft, fftfreq


# This is the version from before Nov 22 2017
# Should be replaced by the Brian's cryoscope tools (analysis/tools/cryoscope_tools)

class RamZFluxArc(ba.BaseDataAnalysis):
    """
    Analysis for the 2D scan that is used to calibrate the FluxArc.

    There exist two variant
        TwoD -> single experimetn
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True,
                 f_demod: float=0, demodulate: bool=False):
        if options_dict is None:
            options_dict = dict()

        self.numeric_params = []
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = True

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

    def process_data(self):
        FFT = np.zeros(np.shape(self.raw_data_dict['measured_values'][0]))
        for i, vals in enumerate(self.raw_data_dict['measured_values'][0]):
            FFT[i, 1:] = np.abs(np.fft.fft(vals))[1:]
        dt = self.raw_data_dict['yvals'][1] - self.raw_data_dict['yvals'][0]
        freqs = np.fft.fftfreq(len(vals), dt)

        self.proc_data_dict['FFT'] = FFT
        self.proc_data_dict['FFT_yvals'] = freqs
        self.proc_data_dict['FFT_ylabel'] = 'Frequency'
        self.proc_data_dict['FFT_yunit'] = 'Hz'
        self.proc_data_dict['FFT_zlabel'] = 'Magnitude'
        self.proc_data_dict['FFT_zunit'] = 'a.u.'

        FFT_peak_idx = []
        for i in range(np.shape(self.proc_data_dict['FFT'])[0]):
            FFT_peak_idx.append(np.argmax(
                self.proc_data_dict['FFT'][i, :len(freqs)//2]))
        self.proc_data_dict['FFT_peak_idx'] = FFT_peak_idx
        self.proc_data_dict['FFT_peak_freqs'] = freqs[FFT_peak_idx]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        start_fit_idx = self.options_dict.get('start_fit_idx', 0)
        stop_fit_idx = self.options_dict.get('stop_fit_idx', -1)

        self.fit_dicts['parabola_fit'] = {
            'model': lmfit.models.PolynomialModel(degree=2),
            'fit_xvals': {'x': self.raw_data_dict['xvals'][start_fit_idx:stop_fit_idx]},
            'fit_yvals': {'data':
                self.proc_data_dict['FFT_peak_freqs'][start_fit_idx:stop_fit_idx]}}

    def prepare_plots(self):
        self.plot_dicts['raw_data'] = {
            'plotfn': self.plot_colorxy,
            'title': self.timestamps[0] + ' raw data',
            'xvals': self.raw_data_dict['xvals'],
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': self.raw_data_dict['xunit'],
            'yvals': self.raw_data_dict['yvals'],
            'ylabel': self.raw_data_dict['ylabel'],
            'yunit': self.raw_data_dict['yunit'],
            'zvals': self.raw_data_dict['measured_values'][0],
            'zlabel': self.raw_data_dict['value_names'][0],
            'zunit': self.raw_data_dict['value_units'][0],
            'do_legend': True, }

        self.plot_dicts['fourier_data'] = {
            'plotfn': self.plot_colorxy,
            'title': self.timestamps[0] + ' fourier transformed data',
            'xvals': self.raw_data_dict['xvals'],
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': self.raw_data_dict['xunit'],

            'yvals': self.proc_data_dict['FFT_yvals'],
            'ylabel': self.proc_data_dict['FFT_ylabel'],
            'yunit': self.proc_data_dict['FFT_yunit'],
            'zvals': self.proc_data_dict['FFT'],
            'zlabel': self.proc_data_dict['FFT_zlabel'],
            'zunit': self.proc_data_dict['FFT_zunit'],
            'do_legend': True, }

        self.plot_dicts['fourier_peaks'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['xvals'],
            'yvals': self.proc_data_dict['FFT_peak_freqs'],
            'ax_id': 'fourier_data',
            'marker': 'o',
            'line_kws': {'color': 'C1', 'markersize': 5,
                         'markeredgewidth': .2,
                         'markerfacecolor': 'None'},
            'linestyle': '',
            'setlabel': 'Fourier maxima', 'do_legend': True,
            'legend_pos': 'right'}

        if self.do_fitting:
            self.plot_dicts['parabola_fit'] = {
                'ax_id': 'fourier_data',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['parabola_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'parabola fit',
                'do_legend': True,
                'line_kws': {'color': 'C3'},

                'legend_pos': 'upper right'}


class RamZAnalysisInterleaved(ba.BaseDataAnalysis):
    '''
    Analysis for the Ram-Z measurement (interleaved case).
    '''

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True,
                 f_demod: float=0, demodulate: bool=False):
        if options_dict is None:
            options_dict = dict()
        options_dict['scan_label'] = options_dict.get('scan_label', 'Ram_Z')
        options_dict['tight_fig'] = False
        options_dict['apply_default_fig_settings'] = False

        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = True

        # Extract metadata to know where to extract the parameters from
        self.params_dict = {
            'qubit_name':
                'Experimental Data.Experimental Metadata.qubit_name',
            'QWG_channel':
                'Experimental Data.Experimental Metadata.QWG_channel',
            'flux_LutMan':
                'Experimental Data.Experimental Metadata.flux_LutMan',
        }
        self.numeric_params = []
        self.extract_data()

        # Now actually extract the parameters and the data
        self.params_dict = {
            # Data
            'xlabel': 'sweep_name',
            'xunit': 'sweep_unit',
            'measurementstring': 'measurementstring',
            'sweep_points': 'sweep_points',
            'value_names': 'value_names',
            'value_units': 'value_units',
            'measured_values': 'measured_values',

            # Pulse parameters
            'wave_dict_unit':
                '{}.wave_dict_unit'.format(self.data_dict['flux_LutMan']),
            'sampling_rate':
                '{}.sampling_rate'.format(self.data_dict['flux_LutMan']),
            'QWG_amp':
                'QWG.ch{}_amp'.format(self.data_dict['QWG_channel']),
            'F_amp':
                '{}.F_amp'.format(self.data_dict['flux_LutMan']),

            # Qubit parameters
            'f_max': '{}.f_max'.format(self.data_dict['qubit_name']),
            'E_c': '{}.E_c'.format(self.data_dict['qubit_name']),
            'V_offset': '{}.V_offset'.format(self.data_dict['qubit_name']),
            'V_per_phi0':
                '{}.V_per_phi0'.format(self.data_dict['qubit_name']),
            'asymmetry':
                '{}.asymmetry'.format(self.data_dict['qubit_name']),
        }
        self.demod = demodulate
        self.f_demod = f_demod

        self.numeric_params = [
            'sampling_rate',
            'QWG_amp',
            'F_amp',
            'f_max',
            'E_c',
            'V_offset',
            'V_per_phi0',
            'asymmetry',
        ]

        if auto:
            self.run_analysis()

    def process_data(self):
        I = -(self.data_dict['measured_values'][0, ::2] - 0.5) * 2
        Q = -(self.data_dict['measured_values'][0, 1::2] - 0.5) * 2

        if self.demod:
            I, Q = self.demodulate(I, Q, self.f_demod,
                                   self.data_dict['sweep_points'][::2])

        raw_phase = np.arctan2(Q, I)
        phase = np.unwrap(raw_phase)
        df = np.gradient(phase) / (2 * np.pi)

        if self.demod:
            df += self.f_demod

        if self.data_dict['wave_dict_unit'] == 'V':
            flux_amp = self.data_dict['F_amp']
        else:
            flux_amp = self.data_dict['F_amp'] * self.data_dict['QWG_amp'] / 2

        dV = fit_mods.Qubit_freq_to_dac(
            frequency=self.data_dict['f_max']-df,
            f_max=self.data_dict['f_max'],
            E_c=self.data_dict['E_c'],
            dac_sweet_spot=self.data_dict['V_offset'],
            V_per_phi0=self.data_dict['V_per_phi0'],
            asymmetry=self.data_dict['asymmetry']) / flux_amp

        # Save results
        self.data_dict['I'] = I
        self.data_dict['Q'] = Q
        self.data_dict['raw_phase'] = raw_phase
        self.data_dict['phase'] = phase
        self.data_dict['detuning'] = df
        self.data_dict['waveform'] = dV

    @classmethod
    def demodulate(self, I, Q, f_demod, t_pts):
        '''
        Demodulate signal in I and Q, sampled at points t_pts, with frequency
        f_demod.
        '''
        cosDemod = np.cos(2 * np.pi * f_demod * t_pts)
        sinDemod = np.sin(2 * np.pi * f_demod * t_pts)
        Iout = I * cosDemod + Q * sinDemod
        Qout = Q * cosDemod - I * sinDemod

        return Iout, Qout

    def prepare_plots(self):
        self.plot_dicts['raw_data'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' raw data',
            'xvals': self.data_dict['sweep_points'][::2],
            'xlabel': self.data_dict['xlabel'],
            'xunit': self.data_dict['xunit'][0],
            'yvals': [self.data_dict['measured_values'][0, ::2],
                      self.data_dict['measured_values'][0, 1::2]],
            'multiple': True,
            'setlabel': ['cos', 'sin'],
            'do_legend': True,
        }

        self.plot_dicts['demodulated_data'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' demodulated data',
            'xvals': self.data_dict['sweep_points'][::2],
            'xlabel': self.data_dict['xlabel'],
            'xunit': self.data_dict['xunit'][0],
            'yvals': [self.data_dict['I'],
                      self.data_dict['Q']],
            'multiple': True,
            'setlabel': ['I', 'Q'],
            'do_legend': True,
        }

        self.plot_dicts['raw_phase'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' raw phase',
            'xvals': self.data_dict['sweep_points'][::2],
            'xlabel': self.data_dict['xlabel'],
            'xunit': self.data_dict['xunit'][0],
            'yvals': self.data_dict['raw_phase'],
            'ylabel': 'phase',
            'yunit': 'deg',
        }

        self.plot_dicts['phase'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' phase',
            'xvals': self.data_dict['sweep_points'][::2],
            'xlabel': self.data_dict['xlabel'],
            'xunit': self.data_dict['xunit'][0],
            'yvals': self.data_dict['phase'],
            'ylabel': 'phase',
            'yunit': 'deg',
        }

        self.plot_dicts['detuning'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' detuning',
            'xvals': self.data_dict['sweep_points'][::2],
            'xlabel': self.data_dict['xlabel'],
            'xunit': self.data_dict['xunit'][0],
            'yvals': self.data_dict['detuning'],
            'ylabel': '$\\Delta f$',
            'yunit': 'Hz',
        }

        self.plot_dicts['waveform'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' waveform',
            'xvals': self.data_dict['sweep_points'][::2],
            'xlabel': self.data_dict['xlabel'],
            'xunit': self.data_dict['xunit'][0],
            'yvals': self.data_dict['waveform'],
            'ylabel': 'amplitude',
            'yunit': 'V',
        }

    def run_fitting(self):
        pass


class DistortionFineAnalysis(ba.BaseDataAnalysis):
    '''
    Analysis for the enhanced cryogenic oscilloscope.
    '''

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        if options_dict is None:
            options_dict = dict()
        options_dict['scan_label'] = 'distortion_scope_fine'
        options_dict['tight_fig'] = False
        options_dict['apply_default_fig_settings'] = False

        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        # Extract metadata to know where to extract the parameters from
        self.params_dict = {
            'qubit_name':
                'Experimental Data.Experimental Metadata.qubit_name',
            'QWG_channel':
                'Experimental Data.Experimental Metadata.QWG_channel',
            'flux_LutMan':
                'Experimental Data.Experimental Metadata.flux_LutMan',
        }
        self.numeric_params = []
        self.extract_data()

        # Now actually extract the parameters and the data
        self.params_dict = {
            # Data
            'xlabel': 'sweep_name',
            'xunit': 'sweep_unit',
            'measurementstring': 'measurementstring',
            'sweep_points': 'sweep_points',
            'value_names': 'value_names',
            'value_units': 'value_units',
            'measured_values': 'measured_values',

            # Pulse parameters
            'gauss_sigma':
                '{}.S_gauss_sigma'.format(self.data_dict['flux_LutMan'][0]),
            'gauss_amp': '{}.S_amp'.format(self.data_dict['flux_LutMan'][0]),
            'wave_dict_unit':
                '{}.wave_dict_unit'.format(self.data_dict['flux_LutMan'][0]),
            'sampling_rate':
                '{}.sampling_rate'.format(self.data_dict['flux_LutMan'][0]),
            'QWG_amp':
                'QWG.ch{}_amp'.format(self.data_dict['QWG_channel'][0]),
            'distortions_amp':
                '{}.F_amp'.format(self.data_dict['flux_LutMan'][0]),

            # Qubit parameters
            'f_max': '{}.f_max'.format(self.data_dict['qubit_name'][0]),
            'E_c': '{}.E_c'.format(self.data_dict['qubit_name'][0]),
            'V_offset': '{}.V_offset'.format(self.data_dict['qubit_name'][0]),
            'V_per_phi0':
                '{}.V_per_phi0'.format(self.data_dict['qubit_name'][0]),
            'asymmetry':
                '{}.asymmetry'.format(self.data_dict['qubit_name'][0]),
        }
        self.numeric_params = [
            'gauss_sigma',
            'gauss_amp',
            'sampling_rate',
            'QWG_amp',
            'distortions_amp',
            'f_max',
            'E_c',
            'V_offset',
            'V_per_phi0',
            'asymmetry',
        ]

        if auto:
            self.run_analysis()

    def process_data(self):
        # Convert lists of duplicated params to floats
        for k in self.numeric_params:
            self.data_dict[k] = self.data_dict[k][1]

        # Background
        cos = (self.data_dict['measured_values'][0][0, ::2] - 0.5) * 2
        sin = (self.data_dict['measured_values'][0][0, 1::2] - 0.5) * 2
        bg_phase = np.unwrap(np.arctan2(sin, cos))
        self.data_dict['background phase'] = np.mean(bg_phase)
        n = fft(bg_phase)
        self.data_dict['noise power spectrum'] = np.conj(n) * n

        # Data
        self.data_dict['cos'] = (
            self.data_dict['measured_values'][1][0, ::2] - 0.5) * 2
        self.data_dict['sin'] = (
            self.data_dict['measured_values'][1][0, 1::2] - 0.5) * 2

        self.data_dict['raw phase'] = np.arctan2(self.data_dict['sin'],
                                                 self.data_dict['cos'])
        # Unwrap phase back-to-front, because it makes more sense to assume
        # it's closer to zero at later times.
        self.data_dict['phase'] = (
            np.unwrap(self.data_dict['raw phase'][::-1])[::-1] -
            self.data_dict['background phase'])

        # After unwrapping we can convert phases to degrees
        self.data_dict['raw phase'] = np.rad2deg(self.data_dict['raw phase'])
        self.data_dict['phase'] = np.rad2deg(self.data_dict['phase'])
        self.data_dict['background phase'] = np.rad2deg(
            self.data_dict['background phase'])

        self.data_dict['tau'] = self.data_dict['sweep_points'][1][::2]

        # Calculate integral of the scoping pulse
        data_dt = (self.data_dict['tau'][1] -
                   self.data_dict['tau'][0])
        gauss_pulse = wf.gauss_pulse(
            self.data_dict['gauss_amp'],
            self.data_dict['gauss_sigma'],
            nr_sigma=4,
            sampling_rate=1/data_dt,
            axis='x', phase=0, motzoi=0, delay=0,
            subtract_offset='first')[0]  # Returns I and Q

        if self.data_dict['wave_dict_unit'] == 'frac':
            # Convert to volts
            gauss_pulse *= self.data_dict['QWG_amp'] / 2

        # t0: time from start to center of scoping pulse
        t0 = self.data_dict['gauss_sigma'] * 2

        # N = self.data_dict['noise power spectrum']
        y = self.data_dict['phase']

        # Calculate convolution kernel
        # Kernel needs to be sampled at the same rate as data.
        h = 360 * fit_mods.Qubit_dac_sensitivity(
            dac_voltage=gauss_pulse,
            f_max=self.data_dict['f_max'],
            E_c=self.data_dict['E_c'],
            dac_sweet_spot=self.data_dict['V_offset'],
            V_per_phi0=self.data_dict['V_per_phi0'],
            asymmetry=self.data_dict['asymmetry'])

        h = np.hstack((h, np.zeros(len(y) - len(h))))
        freqs = fftfreq(len(h), d=data_dt)
        # Multiply phase to get the spectrum for a Gaussian centered around 0.
        H = fft(h) * np.exp(2j * np.pi * freqs * t0)
        X = fft(y) / H

        # Cut off spectrum outisde width of Gaussian, because sampling
        # artefacts dominate there and we're not even really interested in
        # these really high frequencies.
        # TODO: This needs to be understood better. Cutting off does not work
        # right now.
        cut_off_freq_indices = np.where(
            2 * np.pi * np.abs(freqs) > 4 / self.data_dict['gauss_sigma'])[0]

        for ind in cut_off_freq_indices:
            X[ind] = 0

        x = np.real(ifft(X)) * (freqs[1] - freqs[0])
        # Wiener deconvolution
        # x = np.real(ifft(fft(y) * np.conj(H) /
        #                  (np.conj(H) * H + N / (np.conj(y) * y))))

        self.cut_off_inds = cut_off_freq_indices
        self.t0 = t0
        self.y = y
        self.h = h
        self.H = H
        self.X = X
        self.freqs = freqs

        self.data_dict['tau'] += t0
        self.data_dict['dV'] = x  # [t0_samples:-t0_samples]

    def prepare_plots(self):
        self.plot_dicts['raw_data'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' raw data',
            'xvals': self.data_dict['sweep_points'][1][::2],
            'xlabel': self.data_dict['xlabel'][1],
            'xunit': self.data_dict['xunit'][1][0],
            'yvals': [self.data_dict['cos'], self.data_dict['sin']],
            'multiple': True,
            'setlabel': ['cos', 'sin'],
            'do_legend': True,
        }

        self.plot_dicts['raw_phase'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' raw phase',
            'xvals': self.data_dict['sweep_points'][1][::2],
            'xlabel': self.data_dict['xlabel'][1],
            'xunit': self.data_dict['xunit'][1][0],
            'yvals': self.data_dict['raw phase'],
            'ylabel': 'phase',
            'yunit': 'deg',
        }

        self.plot_dicts['phase'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' phase',
            'xvals': self.data_dict['sweep_points'][1][::2],
            'xlabel': self.data_dict['xlabel'][1],
            'xunit': self.data_dict['xunit'][1][0],
            'yvals': self.data_dict['phase'],
            'ylabel': 'phase',
            'yunit': 'deg',
        }

        self.plot_dicts['dV'] = {
            'plotfn': self.plot_line,
            'title': self.timestamps[0] + ' $\\delta V$',
            'xvals': self.data_dict['tau'],
            'xlabel': self.data_dict['xlabel'][1],
            'xunit': self.data_dict['xunit'][1][0],
            'yvals': self.data_dict['dV'],
            'ylabel': '$\\delta V$',
            'yunit': 'V',
        }

    def run_fitting(self):
        pass
