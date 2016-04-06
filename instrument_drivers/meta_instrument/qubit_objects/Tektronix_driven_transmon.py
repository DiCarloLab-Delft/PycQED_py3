import logging
import numpy as np
from scipy.optimize import brent

from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.measurement import mc_parameter_wrapper as pw

from modules.measurement import sweep_functions as swf
from modules.measurement import CBox_sweep_functions as cb_swf
from modules.measurement import awg_sweep_functions as awg_swf
from modules.analysis import measurement_analysis as ma
from modules.measurement.pulse_sequences import standard_sequences as st_seqs
import modules.measurement.randomized_benchmarking.randomized_benchmarking as rb
from modules.measurement.calibration_toolbox import mixer_carrier_cancellation_CBox
from modules.measurement.calibration_toolbox import mixer_skewness_cal_CBox_adaptive

from modules.measurement.optimization import nelder_mead

# from .qubit_object import Transmon
from .CBox_driven_transmon import CBox_driven_transmon
# It would be better to inherit from Transmon directly and put all the common
# stuff in there but for now I am inheriting from what I already have
# MAR april 2016


class Tektronix_driven_transmon(CBox_driven_transmon):
    '''
    Setup configuration:
        Drive:                 Tektronix 5014 AWG
        Acquisition:           CBox
                    (in the future to be compatible with both CBox and ATS)
        Readout pulse configuration: LO modulated using AWG
    '''
    def __init__(self, name,
                 LO, cw_source, td_source,
                 IVVI, AWG,
                 heterodyne_instr,
                 MC):
        super().__init__(name)