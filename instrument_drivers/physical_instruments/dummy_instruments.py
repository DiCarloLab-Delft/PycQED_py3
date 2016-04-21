
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


class DummyParHolder(Instrument):
    '''
    Holds dummy parameters which are get and set able aswell as provides
    some basic functions that depends on these parameters for testing
    purposes.

    Located in physical instruments because it mimicks a instrument that
    talks directly to the hardware.
    '''

    def __init__(self, name):
        super().__init__(name)

        # Instrument parameters
        for parname in ['x', 'y', 'z']:
            self.add_parameter(parname, units='a.u.',
                               parameter_class=ManualParameter,
                               vals=vals.Numbers(), initial_value=0)

        self.add_parameter('noise', units='a.u.',
                           label='white noise amplitude',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(), initial_value=0)

        self.add_parameter('parabola', units='a.u.',
                           get_cmd=self._measure_parabola)
        self.add_parameter('skewed_parabola', units='a.u.',
                           get_cmd=self._measure_skewwed_parabola)

    def _measure_parabola(self):
        return (self.x.get()**2 + self.y.get()**2 + self.z.get()**2 +
                self.noise.get()*np.random.rand(1))

    def _measure_skewwed_parabola(self):
        '''
        Adds a -x term to add a corelation between the parameters.
        '''
        return ((self.x.get()**2 + self.y.get()**2 +
                self.z.get()**2)*(1 + abs(self.y.get()-self.x.get())) +
                self.noise.get()*np.random.rand(1))


