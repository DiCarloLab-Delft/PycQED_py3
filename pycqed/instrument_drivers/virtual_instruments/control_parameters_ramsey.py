from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

import numpy as np



class ControlParametersRamsey(Instrument):
    '''
    
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Instrument parameters
        self.add_parameter('ramsey',
                           label='whether we are doing a ram-Z or echo-Z',
                           parameter_class=ManualParameter,
                           vals=vals.Bool())
        self.add_parameter('sigma', unit='Phi_0',
                           label='width of the Gaussian (mean is 0)',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('detuning_ramsey', unit='Hz',
                           label='how much the qubit is detuned from the sweetspot',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('pulse_length', unit='s',
                           label='TOTAL length for both ram-Z and echo-Z',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())