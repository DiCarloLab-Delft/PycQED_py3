from .CCL_Transmon import CCLight_Transmon

import numpy as np
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as ma
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

        self.add_parameter('mock_mw_amp180', label='Pi-pulse amplitude', unit='V',
                           parameter_class=ManualParameter)

    def measure_rabi(self, MC=None, amps=None,
                     analyze=True, close_fig=True, real_imag=True,
                     prepare_for_timedomain=True, all_modules=False):
        if MC is None:
            MC = self.instr_MC.get_instr()
        if amps is None:
            amps = np.linspace(0.1, 1, 31)
        s = swf.None_Sweep()
        f_rabi = 1
        mocked_values = np.cos(2*np.pi*f_rabi*amps)
        # Add (0,0.5) normal distributed noise
        mocked_values += np.random.normal(0, 0.5, np.size(mocked_values))
        d = det.Mock_Detector(value_names=['Rabi Amplitude'], value_units=['-'],
                              detector_control='soft', mock_values=mocked_values)  # Does not exist yet

        MC.set_sweep_function(s)
        MC.set_sweep_points(amps)
        MC.set_detector_function(d)
        MC.run('mock_rabi_')
        ma.Rabi_Analysis(label='mock_')
        return True
