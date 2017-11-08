"""
Driver for Minicircuits RF switches

This version is being developed for a RC-4SDPT-A18,
but probably carries over to other models without effort.
"""
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals

from pyvisa import VisaIOError

def dec_to_binstring(x, n=4):
    x = int(x)
    res = [1 if x & (1<<i) else 0 for i in range(n)]
    return res

def binstring_to_dec(s):
    return sum(int(b)*(1<<i) for i, b in enumerate(s))

class MinicircuitsRFSwitch(VisaInstrument):
    '''
    Driver for Minicircuits RF switch
    '''

    def __init__(self, name, address, **kwargs):

        super().__init__(name, address, terminator="\n\r", **kwargs)

        self.add_parameter("switch_setting",
                           get_cmd="SWPORT?",
                           set_cmd=self._set_switch,
                           get_parser=dec_to_binstring,
                           set_parser=binstring_to_dec,
                           vals=vals.Anything())

        # the device sends a single '\n' as a hello, messing everything up...
        self.visa_handle.read_termination = "\n"
        try:
            self.visa_handle.read_raw()
        except VisaIOError as e:
            print("Minicircuit Switch did not send telnet handshake")
        self.visa_handle.read_termination = "\n\r"

        #this has to be read differently from the device. todo
        #self.connect_message()

    def _set_switch(self, value):
        """
        The switch acknowledges switching by sending a 1,
        so setting is a 'ask' process.
        """
        ret = self.ask("SETP={}".format(value))
        assert ret=="1"
