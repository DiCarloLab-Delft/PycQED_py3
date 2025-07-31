import time
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals


class Kepco_BOP_20_20_M(Instrument):
    # Instruments will be a list of RemoteInstrument objects, which can be
    # given to a server on creation but not later on, so it needs to be
    # listed in shared_kwargs

    def __init__(self, name, IVVI_dac, Keithley_I_mon, Keithley_V_mon, **kwargs):
        super().__init__(name, **kwargs)
        self.keithley_Imon = Keithley_I_mon
        self.keithley_Imon.mode('dc voltage')
        self.keithley_Vmon = Keithley_V_mon
        self.keithley_Vmon.mode('dc voltage')
        self.IVVI_dac = IVVI_dac
        self.max_I = 10. #A
        self.max_V = 6.  #V
        self.calibration_val = 1.
        self.add_parameter('I',
                           set_cmd=self._set_I,
                           get_cmd=self._get_I,
                           label='Current',
                           vals=vals.Numbers(min_value=-self.max_I, max_value=self.max_I),
                           unit='A')



    def _get_I(self):
        return convert_I(self.keithley_Imon.amplitude())

    def _set_I(self,value):
        mv = ItomV(value)
        self.IVVI_dac(mv)

    def seti(self,value):
        self.I(value)

    def measurei(self):
        return self.I()

    def measureR(self):
        current = self.I()
        if np.abs(current)<1e-3:
            return 0
        else:
            volt = self.measurev()
            return volt/current

    def measurev(self):
        time.sleep(0.5)
        return self.keithley_Vmon.amplitude()



def ItomV(I):
    # To calculate the dac value such that appropriate current is supplied
    # Current = A*mV+B
    A = 0.00909832
    B = -0.00123296
    dac_offset = -1.35
    return (I-B)/A + dac_offset

def convert_I(dacV):
    # To calculate the current given a monitored voltage
    # Voltage = a*Amps+b
    a = 0.001989925815
    b = 0.0
    return (dacV-b)/a