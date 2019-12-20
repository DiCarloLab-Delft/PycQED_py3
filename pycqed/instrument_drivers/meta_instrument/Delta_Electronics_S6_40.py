
import time
import logging
import numpy as np
# from scipy.optimize import brent
# from math import gcd
# from qcodes import Instrument
from qcodes.utils import validators as vals
# from qcodes.instrument.parameter import ManualParameter
from pycqed.analysis import analysis_toolbox as atools
# from pycqed.utilities.general import add_suffix_to_dict_keys

from pycqed.measurement import detector_functions as det
# from pycqed.measurement import composite_detector_functions as cdet
# from pycqed.measurement import mc_parameter_wrapper as pw

from pycqed.measurement import sweep_functions as swf
# from pycqed.measurement import awg_sweep_functions as awg_swf
# from pycqed.analysis import measurement_analysis as ma
# from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_5014
# from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_UHFQC
# from pycqed.measurement.calibration_toolbox import mixer_skewness_calibration_5014
# from pycqed.measurement.optimization import nelder_mead
from pycqed.analysis import analysis_toolbox as a_tools
# import pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts as sq
import logging
import numpy as np
from copy import deepcopy,copy

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter




class Delta_Electronics_S6_40(Instrument):
    # Instruments will be a list of RemoteInstrument objects, which can be
    # given to a server on creation but not later on, so it needs to be
    # listed in shared_kwargs

    def __init__(self, name, IVVI_dac, Keithley_instr, **kwargs):
        super().__init__(name, **kwargs)
        self.keithley = Keithley_instr
        self.IVVI_dac = IVVI_dac
        self.max_I = 40. #A
        self.max_V = 6.  #V
        self.calibration_val = 1.
        self.add_parameter('I',
                           set_cmd=self._set_I,
                           get_cmd=self._get_I,
                           label='Current',
                           vals=vals.Numbers(max_value=self.max_I),
                           unit='A')

        self.add_parameter('calibration',
                           set_cmd=self._set_calibration,
                           get_cmd=self._get_calibration)



    def _get_I(self):
        return convert_I(self.keithley.amplitude())

    def _set_I(self,value):

        if value<0:
            value = 0

        mv = ItomV(value)
        if value == 0:
            mv = 0
        self.IVVI_dac(mv)

    def _get_calibration(self):
        return self.calibration_val

    def _set_calibration(self,conversion_R):
        self.calibration_val = conversion_R

    def seti(self,value):
        self.I(value)

    def measurei(self):
        return self.I()

    def measureR(self):
        eps = 1e-4
        if abs(self.I()-0)>eps:
            return 0.3
        else:
            return 1e3


def mVtoI(dacmV):
    Imvmax = 1.28258444
    Imax = 10.17
    offset = -0.0013610744675000002
    dac0 =  6.8234155278397051
    convR = 0.0016264508908275281
    return ((convR*(dacmV-dac0))*Imax/Imvmax)+offset

def ItomV(I):
    Imvmax = 1.28258444
    Imax = 10.17
    offset = -0.0013610744675000002
    dac0 =  6.8234155278397051
    convR = 0.0016264508908275281
    return (((I-offset)/(Imax/Imvmax))/convR + dac0)

def convert_I(dacV):
    Imvmax = 1.28258444
    Imax = 10.17
    return dacV*Imax/Imvmax