
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
import time


class DummyParHolder(Instrument):
    '''
    Holds dummy parameters which are get and set able as well as provides
    some basic functions that depends on these parameters for testing
    purposes.

    Located in physical instruments because it mimics a instrument that
    talks directly to the hardware.
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Instrument parameters
        for parname in ['x', 'y', 'z']:
            self.add_parameter(parname, unit='m',
                               parameter_class=ManualParameter,
                               vals=vals.Numbers(), initial_value=0)

        self.add_parameter('noise', unit='V',
                           label='white noise amplitude',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=0)

        self.add_parameter('delay', unit='s',
                           label='Sampling delay',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=0)

        self.add_parameter('parabola', unit='V',
                           get_cmd=self._measure_parabola)
        self.add_parameter('skewed_parabola', unit='V',
                           get_cmd=self._measure_skewed_parabola)

        self.add_parameter('array_like', unit='a.u.',
                           parameter_class=ManualParameter,
                           vals=vals.Arrays())

        self.add_parameter('dict_like', unit='a.u.',
                           parameter_class=ManualParameter,
                           vals=vals.Dict())
        self.add_parameter('status', vals=vals.Anything(),
                           parameter_class=ManualParameter)

    def get_idn(self):
        return 'dummy'

    def _measure_parabola(self):
        time.sleep(self.delay())
        return (self.x()**2 + self.y()**2 + self.z()**2 +
                self.noise()*np.random.rand(1))

    def _measure_skewed_parabola(self):
        '''
        Adds a -x term to add a corelation between the parameters.
        '''
        time.sleep(self.delay())
        return ((self.x()**2 + self.y()**2 +
                 self.z()**2)*(1 + abs(self.y()-self.x())) +
                self.noise()*np.random.rand(1))
