from .qubit_object import CCLight_Transmon


from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det

from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object import Qubit
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class Mock_CCLight_Transmon(CCLight_Transmon):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_mock_params()


    def add_mock_params(self):
        """
        Add qubit parameters that are used to mock the system.

        These parameters are
            - prefixed with `mock_`
            - describe "hidden" parameters to mock real experiments.
        """


        self.add_parameter(
            'mock_freq_qubit', label='qubit frequency', unit='Hz',
            docstring='A fixed value, can be made fancier by making it depend on Flux through E_c, E_j and flux',
            parameter_class=ManualParameter)

        self.add_parameter('mock_freq_res', label='qubit frequency', unit='Hz',
                           parameter_class=ManualParameter)


    def measure_rabi(self, MC=None, amps=None,
                     analyze=True, close_fig=True, real_imag=True,
                     prepare_for_timedomain=True, all_modules=False):
        if MC is None:
            MC = self.instr_MC.get_instr()
        s = swf.None_sweep()
        d = det.Mock_detector() # Does not exist yet

