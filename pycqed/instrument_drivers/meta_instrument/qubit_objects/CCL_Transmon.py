'''
File:               CC_transmon.py
Author:             Adriaan Rol
Purpose:            Multiple qubit objects for CC controlled transmons
Usage:
Notes:
Bugs:
'''
import time
import logging
import numpy as np
import copy

import os
from pycqed.measurement.waveform_control_CC import qasm_compiler as qcx
from scipy.optimize import brent
from pycqed.measurement.optimization import nelder_mead
import pygsti

from .qubit_object import Transmon
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2
import pycqed.analysis.analysis_toolbox as a_tools

from pycqed.analysis.tools.data_manipulation import rotation_matrix
from pycqed.measurement.calibration_toolbox import (
    mixer_carrier_cancellation, mixer_skewness_calibration_CBoxV3)

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sqqs
import pycqed.measurement.CBox_sweep_functions as cbs
from pycqed.measurement.waveform_control_CC import qasm_helpers as qh
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta
from pycqed.measurement.waveform_control_CC import instruction_lib as ins_lib

from pycqed.measurement.waveform_control_CC import QWG_fluxing_seqs as qwfs
from pycqed.measurement.waveform_control_CC.instruction_lib import convert_to_clocks

from pycqed.measurement import detector_functions as det
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
import pycqed.measurement.gate_set_tomography.gate_set_tomography_CC as gstCC


class CCLight_Transmon(Transmon):

    '''
    Setup configuration:
        Drive:                 CBox AWGs
        Acquisition:           CBox
        Readout pulse configuration: LO modulated using AWG
    '''

    def __init__(self, name, **kw):
        t0 = time.time()
        super().__init__(name, **kw)
        self.add_instrument_ref_parameters()
        self.add_parameters()
        self.connect_message(begin_time=t0)


    def add_parameters(self):
        pass

    def add_instrument_ref_parameters(self):
        pass

    def prepare_for_continuous_wave(self):
        pass

    def prepare_readout(self):
        pass

    def prepare_for_timedomain(self):
        pass

    def prepare_for_fluxing(self, reset=True):
        pass

    def _get_acquisition_instr(self):
        pass

    def _set_acquisition_instr(self, acq_instr_name):
        pass

