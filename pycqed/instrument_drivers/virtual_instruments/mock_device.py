import numpy as np
from pycqed.analysis.fitting_models import hanger_func_complex_SI
from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers, Enum, Ints
from qcodes.instrument.parameter import ManualParameter
from pycqed.analysis.tools.data_manipulation import count_error_fractions


class Mock_Device(Instrument):
    '''
    A mock device object that is designed to mock experimental.
    This instrument should eventually be able to mock both
    spectroscopy and timedomain experiments.





    Currently it supports the following experiments
        -


    It contains the parameters to
        -

    Changelog
        - initial creation April 2018 MAR
    '''

    def __init__(self, name, **kw):
        super().__init__(name=name, **kw)
        self.add_parameter('res_freq', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('res_Qi', initial_value=1e5,
                           parameter_class=ManualParameter)
        self.add_parameter('res_Qe', initial_value=1e5,
                           parameter_class=ManualParameter)
        self.add_parameter('res_theta', unit='rad',
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('res_phi_0', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('res_phi_v', initial_value=0,
                           parameter_class=ManualParameter)


        self.add_parameter('mw_freq', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('mw_pow', unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('acq_type', initial_value='magn_phase',
                           vals=Enum('real_imag',
                                          'magn_phase', 'magn'),
                           parameter_class=ManualParameter)

        self.add_parameter('cw_noise_level', initial_value=0,
                           parameter_class=ManualParameter)

        self.add_parameter('S21', unit='V', get_cmd=self.measure_transmission)

    def measure_transmission(self):

        # TODO: add attenuation and gain
        transmission = dBm_to_Vpeak(self.mw_pow())

        Ql = Ql_from_Qi_Qe(self.res_Qi(), self.res_Qe())

        S21 = hanger_func_complex_SI(f=self.mw_freq(),
                                        A=transmission,
                                        f0=self.res_freq(),
                                        Ql=Ql,
                                        Qe=self.res_Qe(),
                                        theta=self.res_theta(),
                                        phi_0=self.res_phi_0(),
                                        phi_v=self.res_phi_v())

        noise = self.cw_noise_level() * np.random.rand(2)
        S21 += noise[0]+1j*noise[1]
        if self.acq_type()=='real_imag':
            return np.real(S21), np.imag(S21)
        elif self.acq_type() =='magn_phase':
            return np.abs(S21), np.angle(S21)
        elif self.acq_type() == 'magn':
            return np.abs(S21)
        else:
            raise ValueError()

def Ql_from_Qi_Qe(Qi, Qe):
    """
    1/Ql = 1/Qi+1/Qe
    """
    Ql = 1/(1/Qi+1/Qe)
    return Ql

def dBm_to_Vpeak(P_dBm):
    Vpeak = 10**((P_dBm-10)/20)
    return Vpeak

