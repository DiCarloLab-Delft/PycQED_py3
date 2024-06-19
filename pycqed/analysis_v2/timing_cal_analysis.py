import os
import lmfit
import pycqed.measurement.hdf5_data as hd5
import numpy as np
from typing import Dict, Callable, Any, Optional
import matplotlib.pyplot as plt
from collections import OrderedDict
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis import analysis_toolbox as a_tools
from collections import OrderedDict
from pycqed.analysis import measurement_analysis as ma_old
from qce_utils.control_interfaces.intrf_channel_identifier import IQubitID, QubitIDObj
from qce_utils.addon_pycqed.deserialize_xarray_to_obj import DeserializeBootstrap
from qce_utils.addon_pycqed.object_factories.factory_latency_landscape import LatencyAmplitudeIdentifierFactory
from qce_utils.control_interfaces.datastorage_control.analysis_factories.factory_latency_transmission import (
    LatencyAmplitudeIdentifierAnalysis,
    LatencyAmplitudeIdentifier,
    LatencyExperimentType,
)
from qce_interp.visualization.plotting_functionality import (
    IFigureAxesPair,
    SubplotKeywordEnum,
    LabelFormat,
)


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
                 mw_duration: float=20e-9,
                 flux_pulse_duration:float =10e-9, 
                 auto=True):
        if options_dict is None:
            options_dict = dict()

        self.ch_idx = ch_idx
        self.ro_latency = ro_latency
        self.flux_latency = flux_latency
        self.mw_pulse_separation = mw_pulse_separation
        self.options_dict = options_dict
        self.mw_duration = mw_duration
        self.flux_pulse_duration = flux_pulse_duration
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
            'flux_pulse_duration': self.flux_pulse_duration, 
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


class TimingMicrowaveFluxAnalysis(ba.BaseDataAnalysis):

    # region Class Constructor
    def __init__(self, qubit_id: str, flux_pulse_duration: float, microwave_pulse_duration: float, microwave_pulse_separation: float, t_start: str = None, t_stop: str = None, label: str = '', data_file_path: str = None, close_figs: bool = True, options_dict: dict = None, extract_only: bool = False, do_fitting: bool = False, save_qois: bool = True):
        super().__init__(t_start, t_stop, label, data_file_path, close_figs, options_dict, extract_only, do_fitting, save_qois)
        # Data allocation
        self._qubit_id: IQubitID = QubitIDObj(qubit_id)
        self.object_factory = LatencyAmplitudeIdentifierFactory()
        self.analysis_factory = LatencyAmplitudeIdentifierAnalysis(
            qubit_id=self._qubit_id,
            flux_pulse_duration=flux_pulse_duration,
            microwave_pulse_duration=microwave_pulse_duration,
            buffer_duration=microwave_pulse_separation - flux_pulse_duration,
            experiment_type=LatencyExperimentType.DELAY_MICROWAVE_FIX_FLUX,  # FIXME: Currently hardcoded option
        )
        # Required attributes
        self.params_dict: Dict = {}
        self.numeric_params: Dict = {}
        # Obtain data file path
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        self.data_file_path = a_tools.get_datafilepath_from_timestamp(self.timestamp)
        # Specify data keys
        self._raw_data_key: str = 'data'
        self._raw_value_names_key: str = 'value_names'
    # endregion

    # region Class Methods
    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        param_spec = {
            self._raw_data_key: ('Experimental Data/Data', 'dset'),
            self._raw_value_names_key: ('Experimental Data', 'attr:value_names'),
        }
        self.raw_data_dict = hd5.extract_pars_from_datafile(self.data_file_path, param_spec)
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(self.data_file_path)[0]
        # Construct datastructure
        self.latency_identifier: LatencyAmplitudeIdentifier = self.object_factory.construct(
            source=DeserializeBootstrap.from_path(self.data_file_path)
        )

    def process_data(self):
        """
        process_data: overloaded in child classes,
        takes care of mundane tasks such as binning filtering etc
        """
        pass

    def prepare_plots(self):
        """
        Defines a default plot by setting up the plotting dictionaries to
        specify what is to be plotted
        """
        # Data allocation
        self.axs_dict = {}
        self.plot_dicts = {}
        timestamp: str = self.timestamp

        # (MW) Latency vs (MW) pulse amplitude
        title: str = 'latency_vs_amplitude'
        fig, ax = plt.subplots()
        self.axs_dict[title] = ax
        self.figs[title] = fig
        self.plot_dicts[title] = {
            'plotfn': plot_function_wrapper(self.plot_latency_vs_amplitude_intersection),
            'qubit_id': self._qubit_id,
            'latency_identifier': self.latency_identifier,
            'analysis_factory': self.analysis_factory,
            'timestamp': timestamp,
        }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', True):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

    def analyze_fit_results(self):
        """
        Do analysis on the results of the fits to extract quantities of
        interest.
        """
        # raise NotImplemented
        pass

    @staticmethod
    def plot_latency_vs_amplitude_intersection(qubit_id: IQubitID, latency_identifier: LatencyAmplitudeIdentifier, analysis_factory: LatencyAmplitudeIdentifierAnalysis, **kwargs) -> IFigureAxesPair:
        """Plot wrapper."""
        kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = LabelFormat(
            x_label=f'{qubit_id.id} MW latency [ns]',
            y_label=f'{qubit_id.id} Integrated value [a.u.]',
        )
        fig, ax = analysis_factory.plot_latency_vs_amplitude_intersection(
            identifier=latency_identifier,
            qubit_id=qubit_id,
            **kwargs,
        )

        if analysis_factory.analyzable_latency:
            transition_latencies: np.ndarray = analysis_factory.calculate_relative_transition_latencies(identifier=latency_identifier)
            relative_latency: float = analysis_factory.calculate_relative_latency(identifier=latency_identifier)

            kwargs[SubplotKeywordEnum.HOST_AXES.value] = fig, ax
            fig, ax = analysis_factory.plot_latency_transition_detection(
                transition_latencies=transition_latencies,
                relative_latency=relative_latency,
                **kwargs,
            )
        return fig, ax
    # endregion


def plot_function_wrapper(plot_function: Callable[[Any], Any]) -> Callable[[Optional[plt.Axes], Any], Any]:
    def method(ax: Optional[plt.Axes] = None, *args, **kwargs) -> IFigureAxesPair:
        # Data allocation
        timestamp: str = kwargs.pop("timestamp", "not defined")
        fig = ax.get_figure()
        axs = fig.get_axes()
        fig.suptitle(f'ts: {timestamp}\n')

        kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, axs)
        if len(axs) == 1:
            kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, axs[0])
        return plot_function(
            *args,
            **kwargs,
        )

    return method
