
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.analysis.fitting_models import LorentzFunc
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
        for parname in ['x', 'y', 'z', 'x0', 'y0', 'z0']:
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

        self.add_parameter('parabola_list', unit='V',
                           get_cmd=self._measure_parabola_list)

        self.add_parameter('skewed_parabola', unit='V',
                           get_cmd=self._measure_skewed_parabola)
        self.add_parameter('cos_mod_parabola', unit='V',
                           get_cmd=self._measure_cos_mod_parabola)

        self.add_parameter('lorentz_dip', unit='V',
                           get_cmd=self._measure_lorentz_dip)

        self.add_parameter('lorentz_dip_cos_mod', unit='V',
                           get_cmd=self._measure_lorentz_dip_cos_mod)

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

    def _measure_lorentz_dip(self):
        time.sleep(self.delay())
        y0 = LorentzFunc(self.x(), -1, center=self.x0(), sigma=5)
        y1 = LorentzFunc(self.y(), -1, center=self.y0(), sigma=5)
        y2 = LorentzFunc(self.z(), -1, center=self.z0(), sigma=5)

        y = y0+y1+y2 + self.noise()*np.random.rand(1)
        return y

    def _measure_lorentz_dip_cos_mod(self):
        time.sleep(self.delay())
        y = self._measure_lorentz_dip()
        cos_val = np.cos(self.x()*10+self.y()*10 + self.z()*10)/200
        return y + cos_val

    def _measure_parabola(self):
        time.sleep(self.delay())
        return ((self.x()-self.x0())**2 +
                (self.y()-self.y0())**2 +
                (self.z()-self.z0())**2 +
                self.noise()*np.random.rand(1))

    def _measure_parabola_list(self):
        # Returns same as measure parabola but then as a list of list
        # This corresponds to a natural format for e.g., the
        # UHFQC single int avg detector.
        # Where the outer list would be lenght 1 (seq of 1 segment)
        # with 1 entry (only one value logged)
        return [self._measure_parabola()]

    def _measure_cos_mod_parabola(self):
        time.sleep(self.delay())
        cos_val = np.cos(self.x()/10+self.y()/10 + self.z() /
                         10)**2  # ensures always larger than 1
        par = self._measure_parabola()
        n = self.noise()*np.random.rand(1)
        return cos_val*par + n + par/10

    def _measure_skewed_parabola(self):
        '''
        Adds a -x term to add a corelation between the parameters.
        '''
        time.sleep(self.delay())
        return ((self.x()**2 + self.y()**2 +
                 self.z()**2)*(1 + abs(self.y()-self.x())) +
                self.noise()*np.random.rand(1))
