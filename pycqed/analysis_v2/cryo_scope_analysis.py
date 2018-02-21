import lmfit
from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old
from pycqed.analysis.tools import cryoscope_tools as ct
import pycqed.analysis_v2.base_analysis as ba
import pycqed.measurement.waveform_control_CC.waveform as wf
import pycqed.analysis.fitting_models as fit_mods
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.stats import sem
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel


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
                 f_demod: float=0, demodulate: bool=False, auto=True):
        if options_dict is None:
            options_dict = dict()
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         options_dict=options_dict)
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Custom data extraction for this specific experiment.

        Overwrite this method if you wnat to

        """
        self.timestamps = a_tools.get_timestamps_in_range(
                self.t_start, self.t_stop,
                label=self.labels)

        self.raw_data_dict = OrderedDict()

        # FIXME: this is hardcoded and should be an argument in options dict
        amp_key = 'Snapshot/instruments/AWG8_8005/parameters/awgs_0_outputs_1_amplitude'

        self.raw_data_dict['amps'] = []
        self.raw_data_dict['data'] = []

        for t in self.timestamps:
            a = ma_old.MeasurementAnalysis(timestamp=t, auto=False, close_file=False)
            a.get_naming_and_values()
            amp = a.data_file[amp_key].attrs['value']
            data = a.measured_values[2] +1j* a.measured_values[3]
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
            poly_fit_order=2, plot_fits=False)
        self.proc_data_dict['dac_arc_ana'] = self.dac_arc_ana

        # this is the infamous dac arc conversion method
        # we would like this to be directly accessible
        self.freq_to_amp = self.dac_arc_ana.freq_to_amp
        self.amp_to_freq = self.dac_arc_ana.amp_to_freq



    def prepare_plots(self):
        self.plot_dicts['freqs'] = {
            'plotfn': self.dac_arc_ana.plot_freqs,
            'title':"Cryoscope arc \n"+self.timestamps[0]+' - '+self.timestamps[-1]}

        self.plot_dicts['FluxArc'] = {
            'plotfn': self.dac_arc_ana.plot_ffts,
            'title':"Cryoscope arc \n"+self.timestamps[0]+' - '+self.timestamps[-1]}



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

        a=ma_old.TwoD_Analysis(timestamp=self.timestamps[0], auto=False)
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
        phase = np.mean(np.unwrap(self.raw_data_dict['phases'][::-1],
                                  discont=180, axis=1)[::-1], axis=1)
        phase_err = sem(np.unwrap(self.raw_data_dict['phases'],
                                  discont=180, axis=1), axis=1)

        self.proc_data_dict['phase'] = phase
        self.proc_data_dict['phase_err'] = phase_err

        self.proc_data_dict['t'] = self.raw_data_dict['times']

        if self.amp_to_freq is not None and self.freq_to_amp is not None:
            mean_phase = np.mean(phase[len(phase)//2:])
            # mean_phase = np.mean(phase[:])

            detuning_rad_s = (np.deg2rad(phase-mean_phase)/
                              self.sliding_pulse_duration)
            detuning =  detuning_rad_s/(2*np.pi)

            mod_frequency = self.amp_to_freq(self.raw_data_dict['amp'])
            real_detuning = mod_frequency + detuning
            amp = self.freq_to_amp(real_detuning)

            self.proc_data_dict['amp'] = amp

    def prepare_plots(self):
        self.plot_dicts['phase_plot'] = {
            'plotfn': make_phase_plot,
            't':self.proc_data_dict['t'],
            'phase': self.proc_data_dict['phase'],
            'phase_err': self.proc_data_dict['phase_err'],
            'title':"Sliding pulses\n"+self.timestamps[0]}
        if self.amp_to_freq is not None and self.freq_to_amp is not None:
            self.plot_dicts['FluxArc'] = {
                'plotfn': make_amp_err_plot,
                't': self.proc_data_dict['t'],
                'amp': self.proc_data_dict['amp'],
                'timestamp':self.timestamps[0]}




def make_phase_plot(t, phase, phase_err, title,  ylim=None, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()

    ax.errorbar(t, phase, phase_err, marker='.')
    ax.set_title(title)
    set_xlabel(ax, 'Gate separtion', 's')
    set_ylabel(ax, 'Phase', 'deg')

    mean_phase_tail = np.mean(phase[10:])

    ax.axhline(mean_phase_tail, ls='-', c='grey', linewidth=.5)
    ax.axhline(mean_phase_tail+10, ls=':', c='grey', label=r'$\pm$10 deg', linewidth=0.5)
    ax.axhline(mean_phase_tail-10, ls=':', c='grey', linewidth=0.5)
    ax.axhline(mean_phase_tail+5, ls='--', c='grey', label=r'$\pm$5 deg', linewidth=0.5)
    ax.axhline(mean_phase_tail-5, ls='--', c='grey', linewidth=0.5)
    ax.legend()
    if ylim == None:
        ax.set_ylim(mean_phase_tail-60, mean_phase_tail+40)
    else:
        ax.set_ylim(ylim[0], ylim[1])


def make_amp_err_plot(t, amp, timestamp, ax=None, **kw):
    if ax ==None:
        f, ax =plt.subplots()

    mean_amp = np.mean(amp[len(amp)//2])
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


