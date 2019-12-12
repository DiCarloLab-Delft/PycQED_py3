import time
import logging
import numpy as np
from qcodes.utils import validators as vals
from pycqed.analysis import analysis_toolbox as atools

from pycqed.measurement import detector_functions as det

from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import analysis_toolbox as a_tools
import logging
from copy import deepcopy,copy

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter


class IVVI_current_source(Instrument):
    def __init__(self, name, IVVI_dac, amps_per_mV, **kwargs):
        super().__init__(name, **kwargs)
        self.add_parameter('channel', parameter_class=InstrumentParameter)
        self.IVVI_dac = IVVI_dac
        self.V_to_I = amps_per_mV*1.
        self.max_I = 2000.*self.V_to_I
        self.add_parameter('I',
                           set_cmd=self._set_I,
                           get_cmd=self._get_I,
                           label='Current',
                           vals=vals.Numbers(min_value=-self.max_I,
                                             max_value=self.max_I),
                           unit='A')

    def _get_I(self):
        return self.mVtoI(self.IVVI_dac())

    def _set_I(self,value):
        mV = self.ItomV(value)
        self.IVVI_dac(mV)

    def seti(self,value):
        self.I(value)

    def ItomV(self,I):
        return I/self.V_to_I

    def mVtoI(self,mV):
        return mV*self.V_to_I

    def measurei(self):
        return self.I()

    def measureR(self):
        return 0.01


