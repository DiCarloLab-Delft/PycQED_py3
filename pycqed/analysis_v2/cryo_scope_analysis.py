import matplotlib.pyplot as plt
from typing import Union
from copy import deepcopy
from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old
from pycqed.analysis.tools import cryoscope_tools as ct
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from scipy.stats import sem
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from matplotlib import gridspec, ticker
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

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
                 ch_amp_key: str='Snapshot/instruments/AWG8_8005'
                 '/parameters/awgs_0_outputs_1_amplitude',
                 ch_range_key: str='Snapshot/instruments/AWG8_8005'
                 '/parameters/sigouts_0_range',
                 waveform_amp_key: str='Snapshot/instruments/FL_LutMan_QR'
                 '/parameters/sq_amp',
                 close_figs=True,
                 nyquist_calc: str= 'auto',
                 exclusion_indices: list=None,
                 ch_idx_cos: int=0,
                 ch_idx_sin: int=1,
                 f_demod: float=0, demodulate: bool=False, auto=True):
        if options_dict is None:
            options_dict = dict()

        self.ch_amp_key = ch_amp_key
        # ch_range_keycan also be set to `None`, then the value will
        # default to 1 (no rescaling)
        self.ch_range_key = ch_range_key
        self.waveform_amp_key = waveform_amp_key
        self.exclusion_indices = exclusion_indices
        self.exclusion_indices = exclusion_indices \
            if exclusion_indices is not None else []
        self.nyquist_calc = nyquist_calc
        self.ch_idx_cos = ch_idx_cos
        self.ch_idx_sin = ch_idx_sin

        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict, close_figs=close_figs)
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        self.raw_data_dict = OrderedDict()

        self.raw_data_dict['amps'] = []
        self.raw_data_dict['data'] = []

        for t in self.timestamps:
            a = ma_old.MeasurementAnalysis(
                timestamp=t, auto=False, close_file=False)
            a.get_naming_and_values()

            ch_amp = a.data_file[self.ch_amp_key].attrs['value']
            if self.ch_range_key is None:
                ch_range = 2  # corresponds to a scale factor of 1
            else:
                ch_range = a.data_file[self.ch_range_key].attrs['value']
            waveform_amp = a.data_file[self.waveform_amp_key].attrs['value']
            amp = ch_amp*ch_range/2*waveform_amp
            # amp = ch_amp
            data = a.measured_values[self.ch_idx_cos] + 1j * \
                a.measured_values[self.ch_idx_sin]
            # hacky but required for data saving
            self.raw_data_dict['folder'] = a.folder
            self.raw_data_dict['amps'].append(amp)
            self.raw_data_dict['data'].append(data)
            a.finish()

        self.raw_data_dict['times'] = a.sweep_points
        self.raw_data_dict['timestamps'] = self.timestamps

    def process_data(self):
        self.dac_arc_ana = ct.DacArchAnalysis(
            self.raw_data_dict['times'],
            self.raw_data_dict['amps'],
            self.raw_data_dict['data'],
            exclusion_indices=self.exclusion_indices,
            nyquist_calc=self.nyquist_calc,
            poly_fit_order=2, plot_fits=False)
        self.proc_data_dict['dac_arc_ana'] = self.dac_arc_ana
        self.proc_data_dict['poly_coeffs'] = self.dac_arc_ana.poly_fit

        # this is the dac arc conversion method
        # we would like this to be directly accessible
        self.freq_to_amp = self.dac_arc_ana.freq_to_amp
        self.amp_to_freq = self.dac_arc_ana.amp_to_freq

    def prepare_plots(self):
        self.plot_dicts['freqs'] = {
            'plotfn': self.dac_arc_ana.plot_freqs,
            'title': "Cryoscope arc \n" +
            self.timestamps[0]+' - ' + self.timestamps[-1]}

        self.plot_dicts['FluxArc'] = {
            'plotfn': self.dac_arc_ana.plot_ffts,
            'title': "Cryoscope arc \n" +
            self.timestamps[0]+' - '+self.timestamps[-1]}


class Cryoscope_Analysis(ba.BaseDataAnalysis):
    """
    Cryoscope analysis. Requires a function to convert frequency to amp
    for the final step of the analysis.
    """

    def __init__(
            self, t_start: str,
            t_stop: str =None,
            label='cryoscope',
            derivative_window_length: float=5e-9,
            norm_window_size: int=31,
            nyquist_order: int =0,
            ch_amp_key: str='Snapshot/instruments/AWG8_8005'
        '/parameters/awgs_0_outputs_1_amplitude',
            ch_range_key: str='Snapshot/instruments/AWG8_8005'
        '/parameters/sigouts_0_range',
            polycoeffs_freq_conv: Union[list, str] =
        'Snapshot/instruments/FL_LutMan_QR/parameters/polycoeffs_freq_conv/value',
            ch_idx_cos: int=0,
            ch_idx_sin: int=1,
            input_wf_key: str=None,
            options_dict: dict=None,
            close_figs: bool=True,
            auto=True):
        """
        Cryoscope analysis for an arbitrary waveform.
        """
        if options_dict is None:
            options_dict = dict()

        self.polycoeffs_freq_conv = polycoeffs_freq_conv

        self.ch_amp_key = ch_amp_key
        # ch_range_keycan also be set to `None`, then the value will
        # default to 1 (no rescaling)
        self.ch_range_key = ch_range_key

        self.derivative_window_length = derivative_window_length
        self.norm_window_size = norm_window_size
        self.nyquist_order = nyquist_order

        self.ch_idx_cos = ch_idx_cos
        self.ch_idx_sin = ch_idx_sin

        super().__init__(
            t_start=t_start, t_stop=t_stop, label=label,
            options_dict=options_dict, close_figs=close_figs)
        if auto:
            self.run_analysis()

    def amp_to_freq(self, amp):
        return np.polyval(self.polycoeffs_freq_conv, amp)

    def freq_to_amp(self, freq, positive_branch=True):
        return ct.freq_to_amp_root_parabola(freq,
                                            poly_coeffs=self.polycoeffs_freq_conv,
                                            positive_branch=positive_branch)

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)
        self.timestamp = self.timestamps[0]
        self.raw_data_dict = OrderedDict()

        self.raw_data_dict['amps'] = []
        self.raw_data_dict['data'] = []

        a = ma_old.MeasurementAnalysis(
            timestamp=self.timestamp, auto=False, close_file=False)
        a.get_naming_and_values()

        ch_amp = a.data_file[self.ch_amp_key].attrs['value']
        if self.ch_range_key is None:
            ch_range = 2  # corresponds to a scale factor of 1
        else:
            ch_range = a.data_file[self.ch_range_key].attrs['value']
        amp = ch_amp*ch_range/2
        # amp = ch_amp

        # read conversion polynomial from the datafile if not provided as input
        if isinstance(self.polycoeffs_freq_conv, str):
            self.polycoeffs_freq_conv = np.array(
                a.data_file[self.polycoeffs_freq_conv])
            print(np.array(self.polycoeffs_freq_conv))

        self.raw_data_dict['data'] = a.measured_values[self.ch_idx_cos] + 1j * \
            a.measured_values[self.ch_idx_sin]

        # hacky but required for data saving
        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['amps'].append(amp)
        a.finish()

        self.raw_data_dict['times'] = a.sweep_points
        self.raw_data_dict['timestamps'] = self.timestamps

    def process_data(self):
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        self.proc_data_dict['derivative_window_length'] = \
            self.derivative_window_length
        self.proc_data_dict['norm_window_size'] = self.norm_window_size
        self.proc_data_dict['nyquist_order'] = self.nyquist_order

        self.ca = ct.CryoscopeAnalyzer(
            self.proc_data_dict['times'], self.proc_data_dict['data'],
            derivative_window_length=self.proc_data_dict['derivative_window_length'],
            norm_window_size=self.proc_data_dict['norm_window_size'],
            demod_smooth=None)

        self.ca.freq_to_amp = self.freq_to_amp
        self.ca.nyquist_order = self.nyquist_order

    def prepare_plots(self):
        # pass
        self.plot_dicts['raw_data'] = {
            'plotfn': self.ca.plot_raw_data,
            'title': self.timestamp+'\nRaw cryoscope data'}

        self.plot_dicts['demod_data'] = {
            'plotfn': self.ca.plot_demodulated_data,
            'title': self.timestamp+'\nDemodulated data'}

        self.plot_dicts['norm_data_circ'] = {
            'plotfn': self.ca.plot_normalized_data_circle,
            'title': self.timestamp+'\nNormalized cryoscope data'}

        self.plot_dicts['demod_phase'] = {
            'plotfn': self.ca.plot_phase,
            'title': self.timestamp+'\nDemodulated phase'}

        self.plot_dicts['frequency_detuning'] = {
            'plotfn': self.ca.plot_frequency,
            'title': self.timestamp+'\nDetuning frequency'}

        self.plot_dicts['cryoscope_amplitude'] = {
            'plotfn': self.ca.plot_amplitude,
            'title': self.timestamp+'\nCryoscope amplitude'}

        self.plot_dicts['short_time_fft'] = {
            'plotfn': self.ca.plot_short_time_fft,
            'title': self.timestamp+'\nShort time Fourier Transform'}


        self.plot_dicts['zoomed_cryoscope_amplitude'] = {
            'plotfn': make_zoomed_cryoscope_fig,
            't': self.ca.time,
            'amp': self.ca.get_amplitudes(),
            'title': self.timestamp+'\n Zoomed cryoscope amplitude'}


class SlidingPulses_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for the sliding pulses experiment.

    For noise reasons this is expected to be acquired as a TwoD in a single experiment.
    There exist two variant
        TwoD -> single experiment
        multiple 1D -> combination of several linescans

    This analysis only implements the second variant (as of Feb 2018)
    """

    def __init__(self, t_start: str=None, t_stop: str=None, label='',
                 options_dict: dict=None,
                 sliding_pulse_duration=220e-9,
                 freq_to_amp=None, amp_to_freq=None,
                 f_demod: float=0, demodulate: bool=False, auto=True):
        if options_dict is None:
            options_dict = dict()
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict)

        self.freq_to_amp = freq_to_amp
        self.amp_to_freq = amp_to_freq
        self.sliding_pulse_duration = sliding_pulse_duration

        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.
        """
        self.timestamps = a_tools.get_timestamps_in_range(
            self.t_start, self.t_stop,
            label=self.labels)

        self.raw_data_dict = OrderedDict()
        # auto is True for the TwoD analysis as the heatmap can be useful
        # for debugging the data
        a = ma_old.TwoD_Analysis(timestamp=self.timestamps[0], auto=True,
                                 close_file=False)
        a.get_naming_and_values_2D()
        # FIXME: this is hardcoded and should be an argument in options dict
        amp_key = 'Snapshot/instruments/AWG8_8005/parameters/awgs_0_outputs_1_amplitude'
        amp = a.data_file[amp_key].attrs['value']

        self.raw_data_dict['amp'] = amp
        self.raw_data_dict['phases'] = a.measured_values[0]
        self.raw_data_dict['times'] = a.sweep_points

        # hacky but required for data saving
        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamps'] = self.timestamps
        a.finish()

    def process_data(self):
        phase = np.nanmean(np.unwrap(self.raw_data_dict['phases'][::-1],
                                     discont=180, axis=1)[::-1], axis=1)
        phase_err = sem(np.unwrap(self.raw_data_dict['phases'],
                                  discont=180, axis=1), axis=1)

        self.proc_data_dict['phase'] = phase
        self.proc_data_dict['phase_err'] = phase_err

        self.proc_data_dict['t'] = self.raw_data_dict['times']

        if self.amp_to_freq is not None and self.freq_to_amp is not None:
            mean_phase = np.nanmean(phase[len(phase)//2:])
            # mean_phase = np.nanmean(phase[:])

            detuning_rad_s = (np.deg2rad(phase-mean_phase) /
                              self.sliding_pulse_duration)
            detuning = detuning_rad_s/(2*np.pi)

            mod_frequency = self.amp_to_freq(self.raw_data_dict['amp'])
            real_detuning = mod_frequency + detuning
            amp = self.freq_to_amp(real_detuning)

            self.proc_data_dict['amp'] = amp

    def prepare_plots(self):
        self.plot_dicts['phase_plot'] = {
            'plotfn': make_phase_plot,
            't': self.proc_data_dict['t'],
            'phase': self.proc_data_dict['phase'],
            'phase_err': self.proc_data_dict['phase_err'],
            'title': "Sliding pulses\n"+self.timestamps[0]}
        if self.amp_to_freq is not None and self.freq_to_amp is not None:
            self.plot_dicts['normalized_amp_plot'] = {
                'plotfn': make_amp_err_plot,
                't': self.proc_data_dict['t'],
                'amp': self.proc_data_dict['amp'],
                'timestamp': self.timestamps[0]}


def make_phase_plot(t, phase, phase_err, title,  ylim=None, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()

    ax.errorbar(t, phase, phase_err, marker='.')
    ax.set_title(title)
    set_xlabel(ax, 'Gate separtion', 's')
    set_ylabel(ax, 'Phase', 'deg')

    mean_phase_tail = np.nanmean(phase[10:])

    ax.axhline(mean_phase_tail, ls='-', c='grey', linewidth=.5)
    ax.axhline(mean_phase_tail+10, ls=':', c='grey',
               label=r'$\pm$10 deg', linewidth=0.5)
    ax.axhline(mean_phase_tail-10, ls=':', c='grey', linewidth=0.5)
    ax.axhline(mean_phase_tail+5, ls='--', c='grey',
               label=r'$\pm$5 deg', linewidth=0.5)
    ax.axhline(mean_phase_tail-5, ls='--', c='grey', linewidth=0.5)
    ax.legend()
    if ylim is None:
        ax.set_ylim(mean_phase_tail-60, mean_phase_tail+40)
    else:
        ax.set_ylim(ylim[0], ylim[1])


def make_amp_err_plot(t, amp, timestamp, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()

    mean_amp = np.nanmean(amp[len(amp)//2])
    ax.plot(t, amp/mean_amp, marker='.')

    ax.axhline(1.001, ls='--', c='grey', label=r'$\pm$0.1%')
    ax.axhline(0.999, ls='--', c='grey')
    ax.axhline(1.0, ls='-', c='grey', linewidth=.5)

    ax.axhline(1.0001, ls=':', c='grey', label=r'$\pm$ 0.01%')
    ax.axhline(0.9999, ls=':', c='grey')
    ax.legend(loc=(1.05, 0.5))
    ax.set_title('Normalized to {:.2f}\n {}'.format(mean_amp, timestamp))
    set_xlabel(ax, 'Time', 's')
    set_ylabel(ax, 'Normalized Amplitude')



def make_zoomed_cryoscope_fig(t, amp, title, ax=None, **kw):

    # x = ca.time
    x = t
    y = amp
    # y = ca.get_amplitudes()
    gc = np.mean(y[len(y)//5:4*len(y)//5])

    if ax is not None:
        ax=ax
        f=plt.gcf()
    else:
        f, ax = plt.subplots()
    ax.plot(x,y/gc,  label='Signal')
    ax.axhline(1.01, ls='--', c='grey', label=r'$\pm$1%')
    ax.axhline(0.99, ls='--', c='grey')
    ax.axhline(1.0, ls='-', c='grey', linewidth=.5)

    ax.axhline(1.001, ls=':', c='grey', label=r'$\pm$ 0.1%')
    ax.axhline(0.999, ls=':', c='grey')
    # ax.axvline(10e-9, ls='--', c='k')

    ax.set_ylim(.95, 1.02)
    # ax.set_xlim(-0e-9, 480e-9)
    set_xlabel(ax, 'Time', 's')
    set_ylabel(ax, 'Normalized Amplitude', '')

    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax2 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, [.29, .14, 0.65, .4])
    ax2.set_axes_locator(ip)

    mark_inset(ax, ax2, 1,3, color='grey')
    ax2.axhline(1.0, ls='-', c='grey')
    ax2.axhline(1.01, ls='--', c='grey', label=r'$\pm$1%')
    ax2.axhline(0.99, ls='--', c='grey')
    ax2.axhline(1.001, ls=':', c='grey', label=r'$\pm$ 0.1%')
    ax2.axhline(0.999, ls=':', c='grey')
    ax2.plot(x, y/gc, '-')

    formatter = ticker.FuncFormatter(lambda x, pos: round(x*1e9,3))
    ax2.xaxis.set_major_formatter(formatter)

    ax2.set_ylim(0.998, 1.002)
    ax2.set_xlim(0, min(150e-9, max(t)))
    ax.legend(loc=1)

    ax.set_title(title)
    ax.text(.02, .93, '(a)', color='black', transform=ax.transAxes)

