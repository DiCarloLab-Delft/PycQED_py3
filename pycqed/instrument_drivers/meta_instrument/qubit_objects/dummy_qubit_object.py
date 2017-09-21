from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object import Qubit
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class DummyQubit(Qubit):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_parameter('T1', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T2', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T2_echo', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('f_qubit', label='qubit frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_rb', label='qubit frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_ssro', label='qubit frequency', unit='Hz',
                           parameter_class=ManualParameter)
