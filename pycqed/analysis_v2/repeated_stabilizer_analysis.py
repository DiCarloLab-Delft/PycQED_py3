# -------------------------------------------
# Module containing base-analysis implementation for repeated stabilizer analysis.
# -------------------------------------------
from abc import ABC, ABCMeta, abstractmethod
import os
from typing import List, Union, Dict, Callable, Any, Optional
from enum import Enum
import pycqed.measurement.hdf5_data as hd5
from pycqed.analysis_v2.base_analysis import BaseDataAnalysis
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from qce_circuit.library.repetition_code_circuit import (
    InitialStateContainer,
    InitialStateEnum,
)
from qce_circuit.connectivity.intrf_channel_identifier import (
    IQubitID,
    QubitIDObj,
)
from qce_interp import (
    DataManager,
    QubitIDObj,
    ParityType,
    Surface17Layer,
    IErrorDetectionIdentifier,
    ILabeledErrorDetectionIdentifier,
    ErrorDetectionIdentifier,
    LabeledErrorDetectionIdentifier,
    ISyndromeDecoder,
    ILabeledSyndromeDecoder,
    Distance5LookupTableDecoder,
    LabeledSyndromeDecoder,
    StateAcquisitionContainer,
)
from qce_interp.visualization import (
    plot_state_classification,
    plot_defect_rate,
    plot_all_defect_rate,
    plot_pij_matrix,
)
from qce_interp.visualization.plotting_functionality import (
    SubplotKeywordEnum,
    IFigureAxesPair,
)
import matplotlib.pyplot as plt


class RepeatedStabilizerAnalysis(BaseDataAnalysis):
    
    # region Class Constructor
    def __init__(self, involved_qubit_names: List[str], qec_cycles: List[int], initial_state: InitialStateContainer, t_start: str = None, t_stop: str = None, label: str = '', data_file_path: str = None, close_figs: bool = True, options_dict: dict = None, extract_only: bool = False, do_fitting: bool = False, save_qois: bool = True):
        super().__init__(t_start, t_stop, label, data_file_path, close_figs, options_dict, extract_only, do_fitting, save_qois)
        # Store arguments
        self.involved_qubit_ids: List[IQubitID] = [QubitIDObj(name) for name in involved_qubit_names]
        self.involved_data_qubit_ids: List[IQubitID] = [qubit_id for qubit_id in self.involved_qubit_ids if qubit_id in Surface17Layer().data_qubit_ids]
        self.involved_ancilla_qubit_ids: List[IQubitID] = [qubit_id for qubit_id in self.involved_qubit_ids if qubit_id in Surface17Layer().ancilla_qubit_ids]
        self.qec_cycles: List[int] = qec_cycles
        self.initial_state: InitialStateContainer = initial_state
        # Required attributes
        self.params_dict: Dict = {}
        self.numeric_params: Dict = {}
        # Obtain data file path
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        self.data_file_path = get_datafilepath_from_timestamp(self.timestamp)
        # Specify data keys
        self._raw_data_key: str = 'data'
        self._raw_value_names_key: str = 'value_names'
    # endregion
    
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

    def process_data(self):
        """
        process_data: overloaded in child classes,
        takes care of mundane tasks such as binning filtering etc
        """
        self.data_manager: DataManager = DataManager.from_file_path(
            file_path=self.data_file_path,
            rounds=self.qec_cycles,
            heralded_initialization=True,
            qutrit_calibration_points=True,
            involved_data_qubit_ids=self.involved_data_qubit_ids,
            involved_ancilla_qubit_ids=self.involved_ancilla_qubit_ids,
            expected_parity_lookup=self.initial_state_to_expected_parity(
                initial_state=self.initial_state,
                involved_ancilla_qubit_ids=self.involved_ancilla_qubit_ids,
            ),
            device_layout=Surface17Layer(),
        )
        error_identifier: IErrorDetectionIdentifier = self.data_manager.get_error_detection_classifier(
            use_heralded_post_selection=True,
            use_computational_parity=True,
        )
        self.labeled_error_identifier: ILabeledErrorDetectionIdentifier = LabeledErrorDetectionIdentifier(
            error_identifier,
        )

    def prepare_plots(self):
        """
        Defines a default plot by setting up the plotting dictionaries to
        specify what is to be plotted
        """
        # Data allocation
        self.axs_dict = {}
        self.plot_dicts = {}
        
        # Pij matrix
        title: str = 'pij_matrix'
        fig, ax = plt.subplots()
        self.axs_dict[title] = ax
        self.figs[title] = fig
        self.plot_dicts[title] = {
            'plotfn': plot_function_wrapper(plot_pij_matrix),
            'error_identifier': self.labeled_error_identifier,
            'included_rounds': self.data_manager.rounds,
        }
        # Defect rates (individual)
        for qubit_id in self.involved_ancilla_qubit_ids:
            title: str = f'defect_rate_{qubit_id.id}'
            fig, ax = plt.subplots()
            self.axs_dict[title] = ax
            self.figs[title] = fig
            self.plot_dicts[title] = {
                'plotfn': plot_function_wrapper(plot_defect_rate),
                'error_identifier': self.labeled_error_identifier,
                'qubit_id': qubit_id,
                'qec_cycles': self.data_manager.rounds[-1],
            }
        # Defect rates (all)
        title: str = 'all_defect_rates'
        fig, ax = plt.subplots()
        self.axs_dict[title] = ax
        self.figs[title] = fig
        self.plot_dicts[title] = {
            'plotfn': plot_function_wrapper(plot_all_defect_rate),
            'error_identifier': self.labeled_error_identifier,
            'included_rounds': self.data_manager.rounds[-1],
        }
        # IQ readout (individual)
        for qubit_id in self.involved_qubit_ids:
            title: str = f'IQ_readout_histogram_{qubit_id.id}'
            fig, ax = plt.subplots()
            self.axs_dict[title] = ax
            self.figs[title] = fig
            self.plot_dicts[title] = {
                'plotfn': plot_function_wrapper(plot_state_classification),
                'state_classifier': self.data_manager.get_state_acquisition(qubit_id=qubit_id),
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
    def initial_state_to_expected_parity(initial_state: InitialStateContainer, involved_ancilla_qubit_ids: List[IQubitID]) -> Dict[IQubitID, ParityType]:
        assert initial_state.distance == len(involved_ancilla_qubit_ids) + 1, f"Expects N number of initial states and N-1 number of ancilla's. instead {initial_state.distance} != {len(involved_ancilla_qubit_ids)-1}."
        result: Dict[IQubitID, ParityType] = {}
        
        for i, qubit_id in enumerate(involved_ancilla_qubit_ids):
            state_a: int = initial_state.as_array[i]
            state_b: int = initial_state.as_array[i+1]
            even_parity: bool = state_a == state_b
            if even_parity:
                result[qubit_id] = ParityType.EVEN
            else:
                result[qubit_id] = ParityType.ODD
        return result


def plot_function_wrapper(plot_function: Callable[[Any], Any]) -> Callable[[Optional[plt.Axes], Any], Any]:
    
    def method(ax: Optional[plt.Axes] = None, *args, **kwargs) -> IFigureAxesPair:
        fig = ax.get_figure()
        kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, ax)
        return plot_function(
            *args,
            **kwargs,
        )
    return method
        

if __name__ == "__main__":
    from pycqed.analysis_v2 import repeated_stabilizer_analysis as repsa
    reload(repsa)
    from pycqed.analysis_v2.repeated_stabilizer_analysis import (
        RepeatedStabilizerAnalysis,
        InitialStateContainer,
        InitialStateEnum,
    )
    
    analysis = RepeatedStabilizerAnalysis(
        involved_qubit_names=["D5", "Z4", "D6", "Z2", "D3"],
        qec_cycles=list(range(1, 11, 1)),
        initial_state=InitialStateContainer.from_ordered_list([
            InitialStateEnum.ZERO,
            InitialStateEnum.ONE,
            InitialStateEnum.ZERO,
        ]),
        label="Repeated_stab_meas",
    )
    analysis.run_analysis()