
import matplotlib.pyplot as plt
from copy import deepcopy
from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old
from pycqed.analysis.tools import cryoscope_tools as ct
import pycqed.analysis_v2.base_analysis as ba
import numpy as np
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from pycqed.analysis.tools.spectralfac import SpectralFactorization


class Cryospec_Analysis(ba.BaseDataAnalysis):
    """
    Analysis for the cryoscope frequency domain experiments, also known as the
    cryogenic spectrum analyzer.
    """

    def __init__(
            self, t_start: str=None, t_stop: str =None, label='',
            ch_amp_key: str='Snapshot/instruments/AWG8_8027'
        '/parameters/awgs_0_outputs_1_amplitude',
            ch_range_key: str='Snapshot/instruments/AWG8_8027'
        '/parameters/sigouts_0_range',
            ch_idx_cos: int=0, ch_idx_sin: int=1,
            options_dict: dict=None,
            close_figs: bool=True,
            delta_peak_integral:float =0.5,
            auto=True):
        """
        Cryoscope analysis for an arbitrary waveform.
        """
        if options_dict is None:
            options_dict = dict()

        self.ch_amp_key = ch_amp_key
        # ch_range_keycan also be set to `None`, then the value will
        # default to 1 (no rescaling)
        self.ch_range_key = ch_range_key

        self.ch_idx_cos = ch_idx_cos
        self.ch_idx_sin = ch_idx_sin

        self.delta_peak_integral=delta_peak_integral

        super().__init__(
            t_start=t_start, t_stop=t_stop, label=label,
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
        self.timestamp = self.timestamps[0]
        self.raw_data_dict = OrderedDict()

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
        self.raw_data_dict['amp'] = amp

        self.raw_data_dict['data'] = (
            a.measured_values[self.ch_idx_cos] +
            1j * a.measured_values[self.ch_idx_sin])
        self.raw_data_dict['mod_freqs'] = a.sweep_points
        # hacky but required for data saving
        self.raw_data_dict['folder'] = a.folder
        self.raw_data_dict['timestamp_string'] = a.timestamp_string
        self.raw_data_dict['measurementstring'] = a.measurementstring
        a.finish()

        self.raw_data_dict['times'] = a.sweep_points
        self.raw_data_dict['timestamps'] = self.timestamps

    def process_data(self):

        self.proc_data_dict = deepcopy(self.raw_data_dict)

        data_n = ct.normalize_sincos(
            self.proc_data_dict['data'],
            window_size=len(self.proc_data_dict['data']))
        phases = np.angle(data_n, deg=False)
        self.proc_data_dict['phases'] = np.rad2deg(phases)
        unwrapped_phases = np.rad2deg(np.unwrap(phases[::-1])[::-1])
        unwrapped_phases -= unwrapped_phases[0]  # add offset
        unwrapped_phases[unwrapped_phases < 0] = 0  # remove negative phases

        self.proc_data_dict['unwrapped_phases'] = unwrapped_phases

        # 0.5 is hardcoded as the amplitude of the waveform, integral of the
        # shape is ignored for now.
        amp = self.proc_data_dict['amp']*self.delta_peak_integral
        self.proc_data_dict['transfer_func_naive'] = \
            self.calc_transfer_func_naive(unwrapped_phases,
                                          delta_peak_integral=amp)
        self.proc_data_dict['power_spectrum'] = self.proc_data_dict['transfer_func_naive']**2
        delta_f = abs(self.raw_data_dict['mod_freqs']
                      [1]-self.raw_data_dict['mod_freqs'][0])
        self.spec_fact = SpectralFactorization(
            self.proc_data_dict['power_spectrum'][::-1], delta_f=delta_f,
            suppress_small_elements_warning=True)

    def calc_transfer_func_naive(self, unwrapped_phases, delta_peak_integral=1):
        """
        Based on the following formula:
            d phi =  sum_j |u_i(w_j)|^2 |h(w_j)|^2 (delta_w/2pi)

        The input waveform u_i (t) is approximated as an inf length
        cosine, having a delta function as it's Fourier transform:
            u_i(w_j) =  delta(w_i - w_j).

        Giving:
            d phi_i =  sum_j | delta(w_i-w_j)|^2 |h(w_j)|^2 (delta_w/2pi)
            d phi_i = |h(w_i)|^2 (delta_w /2pi)

        And thus:
            h(w_i) = sqrt(2*np.pi*d phi_i/ delta_w)*1/sqrt(delta_peak_integral)

        """
        scale_fact = 1 / \
            (delta_peak_integral)  # TODO fix dimensionality constant
        return np.sqrt(unwrapped_phases)*scale_fact

    def prepare_plots(self):

        fs = plt.rcParams['figure.figsize']
        # define figure and axes here to have custom layout
        self.figs['main_transfer_func'], axs = plt.subplots(
            ncols=2, nrows=2, figsize=(fs[0]*2, fs[1]*2))
        self.figs['main_transfer_func'].patch.set_alpha(0)


        self.figs['main_transfer_func'].suptitle(
            self.proc_data_dict['timestamp_string']+'\n' +
            'Spectral factorization '+
            self.proc_data_dict['measurementstring'], y=1.05)
        self.axs['impulse_response'] = axs[0, 0]
        self.axs['step_response'] = axs[1, 0]
        self.axs['transfer_func'] = axs[0, 1]
        self.axs['power_spectrum'] = axs[1, 1]

        self.plot_dicts['impulse_response'] = {
            'ylabel': 'Impulse response $h(t)$',
            'yunit': 'a.u.',
            'xlabel': 'Time',
            'xunit': 's',
            'y': self.spec_fact.impulse_response.real,
            'dx': self.spec_fact.delta_t,
            'plotfn': plot_result}
        self.plot_dicts['step_response'] = {
            'ylabel': 'Step response $s(t)$',
            'yunit': 'a.u.',
            'xlabel': 'Time',
            'xunit': 's',
            'y': self.spec_fact.step_response.real,
            'dx': self.spec_fact.delta_t,
            'plotfn': plot_result}

        self.plot_dicts['transfer_func'] = {
            'ylabel': 'Transfer function $h(f)$',
            'yunit': 'a.u.',
            'xlabel': 'Frequency',
            'xunit': 'Hz',
            'y': self.spec_fact.transfer_function,
            'dx': self.spec_fact.delta_f,
            'plotfn': plot_result}
        self.plot_dicts['power_spectrum'] = {
            'ylabel': 'Power spectrum $|h(f)|²$',
            'yunit': 'a.u.',
            'xlabel': 'Frequency',
            'xunit': 'Hz',
            'y': self.spec_fact.two_sided_power_spectrum,
            'dx': self.spec_fact.delta_f,
            'plotfn': plot_result}

        self.plot_dicts['raw_data'] = {
            'data': self.proc_data_dict['data'],
            'mod_freqs': self.proc_data_dict['mod_freqs'],
            'plotfn': plot_raw_data,
            'title': self.timestamp+'\nRaw cryospec data'}

        self.plot_dicts['unit_circle_data'] = {
            'data': self.proc_data_dict['data'],
            'mod_freqs': self.proc_data_dict['mod_freqs'],
            'plotfn': plot_phase_circle,
            'title': self.timestamp+'\nCryospec data unit circle'}

        self.plot_dicts['unit_circle_data'] = {
            'data': self.proc_data_dict['data'],
            'mod_freqs': self.proc_data_dict['mod_freqs'],
            'plotfn': plot_phase_circle,
            'title': self.timestamp+'\nCryospec data unit circle'}

        self.plot_dicts['phases_raw'] = {
            'phases': self.proc_data_dict['phases'],
            'mod_freqs': self.proc_data_dict['mod_freqs'],
            'plotfn': plot_phases,
            'title': self.timestamp+'\nCryospec raw phases'}

        self.plot_dicts['phases_unwrapped'] = {
            'phases': self.proc_data_dict['unwrapped_phases'],
            'mod_freqs': self.proc_data_dict['mod_freqs'],
            'plotfn': plot_phases,
            'title': self.timestamp+'\nCryospec phases'}

        self.plot_dicts['transfer_func_naive'] = {
            'transfunc': self.proc_data_dict['transfer_func_naive'],
            'mod_freqs': self.proc_data_dict['mod_freqs'],
            'plotfn': plot_transferfunc,
            'title': self.timestamp+'\nCryospec transfer function naive'}


def plot_raw_data(mod_freqs, data, title='', ax=None, **kw):
    if ax == None:
        f, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(mod_freqs, np.real(data), label='I', marker='.')
    ax.plot(mod_freqs, np.imag(data), label='Q', marker='.')
    set_xlabel(ax, 'Modulation frequency', 'Hz')
    ax.legend(loc=(1.05, .5))


def plot_phase_circle(mod_freqs, data, title='', ax=None, **kw):
    if ax == None:
        f, ax = plt.subplots()
    ax.set_title(title)
    ax.scatter(np.real(data), np.imag(data),
               c=mod_freqs, cmap=plt.cm.Blues,
               marker='.')
#     ax.plot(mod_freqs, data_Q, label='Q', marker='.')
    set_xlabel(ax, 'real', '')
    set_ylabel(ax, 'imag', '')
    ax.set_aspect('equal')


def plot_phases(mod_freqs, phases, title='', ax=None, **kw):
    if ax == None:
        f, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(mod_freqs, phases, marker='.')  # , c=mod_freqs, cmap=plt.cm.Blues)
    set_xlabel(ax, 'Modulation frequency', 'Hz')
    set_ylabel(ax, 'Phase', 'deg')


def plot_transferfunc(mod_freqs, transfunc, title='', ax=None,
                      plot_kw={}, **kw):
    if ax == None:
        f, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(mod_freqs, transfunc, marker='.', **plot_kw)
    set_xlabel(ax, 'Modulation frequency', 'Hz')
    set_ylabel(ax, 'h($\omega$)', 'a.u.')


def plot_result(y, dx, title='', ax=None, mode='half',
                xlabel='', xunit='',
                ylabel='', yunit='',
                **kw):
    if ax == None:
        f, ax = plt.subplots()
    ax.set_title(title)
    _plot_series(ax=ax, y=y, dx=dx, mode=mode)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)


def _plot_series(ax, y, dx, mode='raw'):
    """
    Plot a series y(t), sampled at y[n] = y(n*dx).

    `mode`: how to plot
        `raw`: plot from 0 to len(y)*dx
        `shift`: assume fft-shifted: plot from -n/2*dx to n/2*dx
        `half`: plot from 0 to n/2*dx
    """

    if np.iscomplexobj(y):
        _plot_series(ax, y.real, dx, mode)
        _plot_series(ax, y.imag, dx, mode)
    else:
        n = len(y)

        if mode == 'raw':
            xx = np.arange(n) * dx
            ax.plot(xx, y)

        if mode == 'shift':
            xx = (np.arange(n) - n // 2) * dx
            ax.plot(xx, np.fft.fftshift(y))

        if mode == 'half':
            xx = (np.arange(n // 2)) * dx
            ax.plot(xx, y[:n // 2])
