import time

from qcodes import Instrument, validators as vals
from qcodes import MultiParameter, Parameter
from functools import partial
from qcodes.instrument.parameter import ManualParameter


class virtual_ZNB20(Instrument):
    """
    Very crude version of VNA to work with mock qubit.
    Basically just imports all parameters and discards all functions.

    Hacked the parameters by making them ManualParameters, to ensure they are
    gettable and settable (There may be a more elegant way of doing so, but 
    this works for now)
    """

    def __init__(self, name, **kwargs):

        super().__init__(name=name, **kwargs)

        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(-150, 25))

        self.add_parameter(name='avg',
                           label='Averages',
                           unit='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, 1000))

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('auto', 'flatten', 'reduce', 'moving'))

        self.add_parameter(name='average_state',
                           parameter_class=ManualParameter,
                           vals=vals.OnOff())

        self.add_parameter(name='bandwidth',
                           label='Bandwidth',
                           unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, 1e6))

        self.add_parameter(name='center_frequency',
                           unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(100e3, 20e9))

        self.add_parameter(name='span_frequency',
                           unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 20e9))

        self.add_parameter(name='start_frequency',
                           unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(100e3, 20e9))

        self.add_parameter(name='stop_frequency',
                           unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(100e3, 20e9))

        self.add_parameter(name='number_sweeps_all',
                           parameter_class=ManualParameter,
                           vals=vals.Ints(1, 100000))

        self.add_parameter(name='npts',
                           parameter_class=ManualParameter,
                           vals=vals.Ints(1, 100001))

        self.add_parameter(name='min_sweep_time',
                           parameter_class=ManualParameter,
                           vals=vals.OnOff())

        self.add_parameter(name='sweep_time',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 1e5))

        self.add_parameter(name='sweep_type',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('lin', 'linear', 'log', 'logarithmic', 'pow', 'power',
                                          'cw', 'poin', 'point', 'segm', 'segment'))
