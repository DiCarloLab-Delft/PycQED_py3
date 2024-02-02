# -------------------------------------------
# Module describing the measurement functionality used for AC flux-crosstalk experiment
# -------------------------------------------
from abc import ABC, abstractmethod
import os
import functools
import numpy as np
import warnings
from typing import TypeVar, Type, Dict, Optional
from pycqed.instrument_drivers.meta_instrument.device_object_CCL import DeviceCCL as Device
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon import CCLight_Transmon as Transmon
from pycqed.measurement.measurement_control import MeasurementControl as MeasurementControl
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC as CentralControl
from pycqed.instrument_drivers.meta_instrument.LutMans.flux_lutman_vcz import HDAWG_Flux_LutMan as FluxLutMan
from pycqed.measurement.sweep_functions import (
    FLsweep as FluxSweepFunctionObject,
    OpenQL_Sweep as OpenQLSweep,
)
from pycqed.measurement.flux_crosstalk_ac.schedule import (
    OqlProgram,
    schedule_flux_crosstalk,
    schedule_ramsey,
)
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from pycqed.analysis_v2.base_analysis import BaseDataAnalysis
import pycqed.measurement.hdf5_data as hd5
import matplotlib.pyplot as plt
from pycqed.qce_utils.custom_exceptions import InterfaceMethodException


class IBaseDataAnalysis(ABC):
    
    # region Interface Methods
    @abstractmethod
    def process_data(self):
        """
        process_data: overloaded in child classes,
        takes care of mundane tasks such as binning filtering etc
        """
        raise InterfaceMethodException

    @abstractmethod
    def prepare_plots(self):
        """
        Defines a default plot by setting up the plotting dictionaries to
        specify what is to be plotted
        """
        raise InterfaceMethodException

    @abstractmethod
    def analyze_fit_results(self):
        """
        Do analysis on the results of the fits to extract quantities of
        interest.
        """
        raise InterfaceMethodException
    # endregion


def plot_example(x_array: np.ndarray, y_array: np.ndarray, ax: Optional[plt.Axes] = None, **kwargs):
    # Get keyword argument defaults
    x_label: str = kwargs.pop('x_label', 'Default Label [a.u.]')
    y_label: str = kwargs.pop('y_label', 'Default Label [a.u.]')
    # Plot figure
    print(kwargs)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        x_array,
        y_array,
        '.-',
    )
    ax.grid(True, alpha=0.5, linestyle='dashed')
    ax.set_axisbelow(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

TBaseDataAnalysis = TypeVar('TBaseDataAnalysis', bound=BaseDataAnalysis)

class FluxCrosstalkAnalysis(BaseDataAnalysis, IBaseDataAnalysis):
    
    # region Class Constructor
    def __init__(self, t_start: str = None, t_stop: str = None, label: str = '', data_file_path: str = None, close_figs: bool = True, options_dict: dict = None, extract_only: bool = False, do_fitting: bool = False, save_qois: bool = True):
        super().__init__(t_start, t_stop, label, data_file_path, close_figs, options_dict, extract_only, do_fitting, save_qois)
        self.params_dict: Dict = {}
        self.numeric_params: Dict = {}
    # endregion
    
    # region Interface Methods
    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_filepath = get_datafilepath_from_timestamp(self.timestamp)
        self._raw_data_key: str = 'data'
        param_spec = {
            self._raw_data_key: ('Experimental Data/Data', 'dset'),
            'value_names': ('Experimental Data', 'attr:value_names'),
        }
        self.raw_data_dict = hd5.extract_pars_from_datafile(data_filepath, param_spec)
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_filepath)[0]
        
    def process_data(self):
        """
        process_data: overloaded in child classes,
        takes care of mundane tasks such as binning filtering etc
        """
        self.data_array: np.ndarray = self.raw_data_dict[self._raw_data_key]

    def prepare_plots(self):
        """
        Defines a default plot by setting up the plotting dictionaries to
        specify what is to be plotted
        """
        self.plot_dicts['Example_plot'] = {
            'plotfn': plot_example,
            'x_array': self.data_array[:, 0],
            'y_array': self.data_array[:, 1],
            'x_label': 'Times [s]',
        }

    def analyze_fit_results(self):
        """
        Do analysis on the results of the fits to extract quantities of
        interest.
        """
        # raise NotImplemented
        pass
    # endregion


def decorator_run_analysis(analysis_class: Type[TBaseDataAnalysis], filter_label: str):
    
    def decorator(func):
        """
        Decorator that constructs analysis and triggers execution.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Construct analysis class
            instance: TBaseDataAnalysis = analysis_class(label=filter_label)
            instance.run_analysis()
            
            return result
        return wrapper
    
    return decorator


def decorator_pyplot_dataset(dataset_key: str, array_dimensions: int, show_figure: bool, *plot_args, **plot_kwargs):
    
    def decorator(func):
        """
        Decorator to measure and record the execution time of the decorated function or method.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Assumes dictionary structure
            dataset_key_exists: bool = dataset_key in result and isinstance(result[dataset_key], np.ndarray)
            # Guard clause, if dataset key does not exists
            if not dataset_key_exists:
                warnings.warn(f"Key: {dataset_key} is not present in function output.")
                return result
            
            data_array: np.ndarray = result[dataset_key]
            # Get keyword argument defaults
            x_label: str = plot_kwargs.pop('x_label', 'Default Label [a.u.]')
            y_label: str = plot_kwargs.pop('y_label', 'Default Label [a.u.]')
            # Plot figure
            fig, ax = plt.subplots(*plot_args, **plot_kwargs)
            ax.plot(
                data_array[:, 0],
                data_array[:, 1],
                '.-',
            )
            ax.grid(True, alpha=0.5, linestyle='dashed')
            ax.set_axisbelow(True)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if show_figure:
                plt.show()
            else:
                plt.close(fig)
            
            return result
        return wrapper
    
    return decorator


@decorator_run_analysis(
    analysis_class=FluxCrosstalkAnalysis,
    filter_label='FluxCrosstalk',
)
def measure_ac_flux_crosstalk(device: Device, qubit_echo_id: str, prepare_for_timedomain: bool = False, disable_metadata: bool = False):
    """
    Performs an experiment
    """
    # Data allocation
    qubit_echo: Transmon = device.find_instrument(qubit_echo_id)
    flux_lutman: FluxLutMan = qubit_echo.instr_LutMan_Flux.get_instr()
    max_duration_ns: int = 10  # [ns]
    flux_pulse_duration: np.ndarray = np.arange(0e-9, max_duration_ns * 1e-9, 1/2.4e9)
    # flux_pulse_amplitude: np.ndarray = np.asarray([flux_lutman.sq_amp()])
    flux_cw = "sf_square"
    meas_control: MeasurementControl = device.instr_MC.get_instr()
    meas_control_nested: MeasurementControl = device.instr_nested_MC.get_instr()
    central_control: CentralControl = device.instr_CC.get_instr()

    # Prepare for time-domain if requested
    if prepare_for_timedomain:
        device.prepare_for_timedomain(qubits=[qubit_echo_id])

    schedule: OqlProgram = schedule_flux_crosstalk(
        qubit_echo_index=qubit_echo.cfg_qubit_nr(),
        flux_pulse_cw=flux_cw,  # Square pulse?
        platf_cfg=device.cfg_openql_platform_fn(),
        half_echo_delay_ns=max_duration_ns,  # [ns]
    )
    sweep_function: FluxSweepFunctionObject = FluxSweepFunctionObject(
        flux_lutman,
        flux_lutman.sq_length,  # Square length (qcodes) parameter
        amp_for_generation=0.0,
        waveform_name="square",  # Meaning we are going to sweep the square-pulse parameters only
    )
    # flux_pulse_duration = np.concatenate([flux_pulse_duration, np.zeros(shape=4)])  # Include calibration points
    central_control.eqasm_program(schedule.filename)
    central_control.start()
    
    detector = device.get_int_avg_det(
        qubits=[qubit_echo_id],
        single_int_avg=True,
        always_prepare=True,
    )

    meas_control.set_sweep_function(sweep_function)
    meas_control.set_sweep_points(flux_pulse_duration)
    meas_control.set_detector_function(detector)
    label = f'FluxCrosstalk_{qubit_echo.name}'
    result = meas_control.run(label, disable_snapshot_metadata=disable_metadata)

    return result
