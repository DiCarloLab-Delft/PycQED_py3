import lmfit
import numpy as np
from collections import OrderedDict
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba

from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old


class Timing_Cal_Flux_Coarse(ba.BaseDataAnalysis):
    """
    Manual analysis for a coarse calibration of timings.
    Used to mark latencies in a 2D timing calibration experiment.

    Args:
        ch_idx      (int) : from what channel to use the acquired data
        ro_latency  (int) : Determines the point from which the pulses overlap
                            with the readout. A negative latency for the ro
                            means that the ro should be triggered before.
                            The microwave pulses
        flux_latency (int):
        mw_latency  (int) :


    The experiment consists of the following sequence.

        ------------ RO              - DIO1 (RO)
        ----- flux ----              - DIO3 (flux)
        --X90-----X90--              - DIO4/5 (mw)

    From this sequence the delays of the flux (x-axis) and microwave( y-axis)
    are varied.

    Below is some ASCII art to show what the data should look like.

            ----------------
    DIO4    ----------------   <-- overlap with the readout signal
    (mw)    _________---____         corresponds to flux-mw lat =0
            ________---_____
            _______---______   <-- moving feature = overlap with flux
            ______---_______
            _____---________

                DIO3 (flux)

    By specifying expected timings (in seconds) the latencies can be inferred.


    """

    def __init__(self, t_start: str=None,
                 t_stop: str=None,
                 label='',
                 close_figs: bool=True,
                 ch_idx: int=0,
                 ro_latency: float=0,
                 flux_latency: float=0,
                 mw_pulse_separation: float=100e-9,
                 options_dict: dict=None,
                 mw_duration: int=20e-9,
                 auto=True):
        if options_dict is None:
            options_dict = dict()

        self.ch_idx = ch_idx
        self.ro_latency = ro_latency
        self.flux_latency = flux_latency
        self.mw_pulse_separation = mw_pulse_separation
        self.options_dict = options_dict
        self.mw_duration = mw_duration
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         close_figs=close_figs,
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

        for t in self.timestamps:
            a = ma_old.TwoD_Analysis(
                timestamp=t, auto=False, close_file=False)
            a.get_naming_and_values_2D()

            self.raw_data_dict['folder'] = a.folder

            self.raw_data_dict['xvals'] = a.sweep_points
            self.raw_data_dict['yvals'] = a.sweep_points_2D
            self.raw_data_dict['zvals'] = a.measured_values[self.ch_idx].T

            self.raw_data_dict['xlabel'] = a.parameter_names[0]
            self.raw_data_dict['ylabel'] = a.parameter_names[1]
            self.raw_data_dict['zlabel'] = a.value_names[self.ch_idx]
            self.raw_data_dict['xunit'] = a.parameter_units[0]
            self.raw_data_dict['yunit'] = a.parameter_units[1]
            self.raw_data_dict['zunit'] = a.value_units[self.ch_idx]
            self.raw_data_dict['measurementstring'] = a.measurementstring

            self.raw_data_dict['timestamp_string'] = a.timestamp_string
            a.finish()

        self.raw_data_dict['times'] = a.sweep_points
        self.raw_data_dict['timestamps'] = self.timestamps

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_colorxy,
            'title': '{}\n{}'.format(self.raw_data_dict['timestamp_string'],
                                     self.raw_data_dict['measurementstring']),
            'xvals': self.raw_data_dict['xvals'],
            'yvals': self.raw_data_dict['yvals'],
            'zvals': self.raw_data_dict['zvals'],
            'xlabel': self.raw_data_dict['xlabel'],
            'ylabel': self.raw_data_dict['ylabel'],
            'zlabel': self.raw_data_dict['zlabel'],
            'xunit': self.raw_data_dict['xunit'],
            'yunit': self.raw_data_dict['yunit'],
            'zunit': self.raw_data_dict['zunit']}

        self.plot_dicts['annotation'] = {
            'plotfn': annotate_timing_coarse_cal,
            'flux_latency': self.flux_latency,
            'ro_latency': self.ro_latency,
            'mw_pulse_separation': self.mw_pulse_separation,
            'ax_id': 'main'}


class Timing_Cal_Flux_Fine(ba.BaseDataAnalysis):
    """
    Manual analysis for a coarse calibration of timings.
    Used to mark latencies in a 2D timing calibration experiment.

    Args:
        ch_idx      (int) : from what channel to use the acquired data
        ro_latency  (int) : Determines the point from which the pulses overlap
                            with the readout. A negative latency for the ro
                            means that the ro should be triggered before.
                            The microwave pulses
        flux_latency (int):
        mw_latency  (int) :


    The experiment consists of the following sequence.

        ------------ RO              -  (RO)
        ----- flux ----              -  (flux)
        --X90-----X90--              -  (mw)

    From this sequence the delays of the flux (x-axis) and microwave( y-axis)
    are varied.

    Below is some ASCII art to show what the data should look like.

            ----------------
            ----------------   <-- overlap with the readout signal
    (mw)    _________---____         corresponds to flux-mw lat =0
    latency ________---_____
            _______---______   <-- moving feature = overlap with flux
            ______---_______
            _____---________

                (flux latency)

    By specifying expected timings (in seconds) the latencies can be inferred.


    """

    def __init__(self, t_start: str=None,
                 t_stop: str=None,
                 label='',
                 close_figs: bool=True,
                 ch_idx: int=0,
                 ro_latency: float=0,
                 flux_latency: float=0,
                 mw_pulse_separation: float=100e-9,
                 options_dict: dict=None,
                 mw_duration: int=20e-9,
                 auto=True):
        if options_dict is None:
            options_dict = dict()

        self.ch_idx = ch_idx
        self.ro_latency = ro_latency
        self.flux_latency = flux_latency
        self.mw_pulse_separation = mw_pulse_separation
        self.options_dict = options_dict
        self.mw_duration = mw_duration
        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         close_figs=close_figs,
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

        for t in self.timestamps:
            a = ma_old.TwoD_Analysis(
                timestamp=t, auto=False, close_file=False)
            a.get_naming_and_values_2D()

            self.raw_data_dict['folder'] = a.folder

            self.raw_data_dict['xvals'] = a.sweep_points
            self.raw_data_dict['yvals'] = a.sweep_points_2D
            self.raw_data_dict['zvals'] = a.measured_values[self.ch_idx].T

            self.raw_data_dict['xlabel'] = a.parameter_names[0]
            self.raw_data_dict['ylabel'] = a.parameter_names[1]
            self.raw_data_dict['zlabel'] = a.value_names[self.ch_idx]
            self.raw_data_dict['xunit'] = a.parameter_units[0]
            self.raw_data_dict['yunit'] = a.parameter_units[1]
            self.raw_data_dict['zunit'] = a.value_units[self.ch_idx]
            self.raw_data_dict['measurementstring'] = a.measurementstring

            self.raw_data_dict['timestamp_string'] = a.timestamp_string
            a.finish()

        self.raw_data_dict['times'] = a.sweep_points
        self.raw_data_dict['timestamps'] = self.timestamps

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_colorxy,
            'title': '{}\n{}'.format(self.raw_data_dict['timestamp_string'],
                                     self.raw_data_dict['measurementstring']),
            'xvals': self.raw_data_dict['xvals'],
            'yvals': self.raw_data_dict['yvals'],
            'zvals': self.raw_data_dict['zvals'],
            'xlabel': self.raw_data_dict['xlabel'],
            'ylabel': self.raw_data_dict['ylabel'],
            'zlabel': self.raw_data_dict['zlabel'],
            'xunit': self.raw_data_dict['xunit'],
            'yunit': self.raw_data_dict['yunit'],
            'zunit': self.raw_data_dict['zunit']}

        self.plot_dicts['annotation'] = {
            'plotfn': annotate_timing_fine_cal,
            'flux_latency': self.flux_latency,
            'ro_latency': self.ro_latency,
            'mw_pulse_separation': self.mw_pulse_separation,
            'ax_id': 'main'}


def annotate_timing_coarse_cal(ax, flux_latency, ro_latency,
                               mw_pulse_duration=20e-9,
                               flux_pulse_duration=40e-9,
                               mw_pulse_separation=100e-9,
                               **kw):
    """
    """
    ro_latency_clocks = ro_latency/20e-9
    flux_latency_clocks = flux_latency/20e-9
    mw_pulse_separation_clocks = mw_pulse_separation/20e-9
    mw_pulse_duration_clocks = mw_pulse_duration/20e-9

    x = np.arange(40)-4

    # intersect happens when mw_latency = ro_latency,
    # mw_latency is along the y-axis.
    ax.plot(x, -ro_latency_clocks*np.ones(len(x)), c='r',
            label='mw-ro overlap', ls='-')

    y = (x-flux_latency_clocks)
    ax.plot(x, y+mw_pulse_duration_clocks,
            color='r', label='mw-flux overlap 1', ls='--')
    ax.plot(x, y, color='r', ls='--')

    y = (x-flux_latency_clocks-mw_pulse_separation_clocks)
    ax.plot(x, y, c='r', ls='-.', label='mw-flux overlap 2')
    ax.plot(x, y-mw_pulse_duration_clocks, c='r', ls='-.')

    timing_info = '{: <28}{:.2f} ns\n{: <28}{:.2f} ns\n{: <24}{:.2f} ns'.format(
        'flux latency:', flux_latency*1e9,
        'ro latency:', ro_latency*1e9,
        'mw pulse separation:', mw_pulse_separation*1e9)
    ax.text(1.25, .85, timing_info, transform=ax.transAxes)

    ax.legend()

def annotate_timing_fine_cal(ax, flux_latency, ro_latency,
                               mw_pulse_duration=20e-9,
                               flux_pulse_duration=40e-9,
                               mw_pulse_separation=100e-9,
                               **kw):
    """
    """
    x = np.arange(-300e-9,301e-9,10e-9)

    # intersect happens when mw_latency = ro_latency,
    # mw_latency is along the y-axis.
    ax.plot(x, -ro_latency*np.ones(len(x)), c='r',
            label='mw-ro overlap', ls='-')

    y = (x-flux_latency)
    ax.plot(x, y+flux_pulse_duration,
            color='r', label='mw-flux overlap 1', ls='--')
    ax.plot(x, y, color='orange', ls='--')

    y = (x-flux_latency-mw_pulse_separation+mw_pulse_duration)
    ax.plot(x, y, c='r', ls='-.', label='mw-flux overlap 2')
    ax.plot(x, y-flux_pulse_duration, c='r', ls='-.')

    timing_info = '{: <28}{:.2f} ns\n{: <28}{:.2f} ns\n{: <24}{:.2f} ns'.format(
        'flux latency:', flux_latency*1e9,
        'ro latency:', ro_latency*1e9,
        'mw pulse separation:', mw_pulse_separation*1e9)
    ax.text(1.25, .85, timing_info, transform=ax.transAxes)

    ax.legend()
