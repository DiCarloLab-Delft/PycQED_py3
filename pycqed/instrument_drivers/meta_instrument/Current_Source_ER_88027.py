import time

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals


class Current_Source_ER_88027(Instrument):
    # Instruments will be a list of RemoteInstrument objects, which can be
    # given to a server on creation but not later on, so it needs to be
    # listed in shared_kwargs

    def __init__(self, name, Keithley_Vsource, Keithley_instr, **kwargs):
        super().__init__(name, **kwargs)
        self.keithley = Keithley_instr
        self.Vsource = Keithley_Vsource
        self.max_I = 10. #A
        self.max_V = 6.  #V

        Keithley_Vsource.seti(10e-3)
        self.add_parameter('I',
                           set_cmd=self._set_I,
                           get_cmd=self._get_I,
                           label='Current',
                           vals=vals.Numbers(max_value=self.max_I),
                           unit='A')



    def _get_I(self):
        return convert_I(self.keithley.amplitude())

    def _set_I(self,value):
        self.Vsource.setv(value)
        time.sleep(0.250)

    def seti(self,value):
        self.I(value)

    def measurei(self):
        return self.I()

    def measureR(self):
        eps = 1e-4
        if abs(self.I()-0)>eps:
            return 0.3
        else:
            return 0.3


def convert_I(dacV):
    return dacV