import numpy as np
import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
s_pars = Instrument('Simulation parameters')

# adding all the manual parameters relevant for my simulation so that I
# can track and loop over them.
s_pars.add_parameter('pulse_length', unit='s',
                     parameter_class=ManualParameter, initial_value=20e-9)
s_pars.add_parameter('lambda_coeffs', unit='a.u.',
                     parameter_class=ManualParameter,
                     initial_value=np.zeros(4), vals=vals.Arrays())

s_pars.add_parameter('theta_f', unit='rad',
                     parameter_class=ManualParameter, initial_value=0.5*np.pi)
s_pars.add_parameter('f_01_max', unit='Hz',
                     parameter_class=ManualParameter, initial_value=8e9)
s_pars.add_parameter('f_interaction', unit='Hz',
                     parameter_class=ManualParameter, initial_value=6.8e9)
s_pars.add_parameter('J_2', unit='Hz',
                     parameter_class=ManualParameter, initial_value=40e6)
s_pars.add_parameter('E_c', unit='Hz',
                     parameter_class=ManualParameter, initial_value=300e6)
s_pars.add_parameter('dac_flux_coefficient', unit='a.u.',
                     parameter_class=ManualParameter, initial_value=1)
s_pars.add_parameter('asymmetry', unit='a.u.',
                     parameter_class=ManualParameter, initial_value=0)
s_pars.add_parameter('sampling_rate', unit='Hz',
                     parameter_class=ManualParameter, initial_value=2e9)



params_dict = {'length': s_pars.pulse_length,
               'lambda_coeffs': s_pars.lambda_coeffs,
               'theta_f': s_pars.theta_f,
               'f_01_max': s_pars.f_01_max,
               'f_interaction': s_pars.f_interaction,
               'J_2': s_pars.J_2,
               'E_c': s_pars.E_c,
               'dac_flux_coefficient': s_pars.dac_flux_coefficient,
               'asymmetry': s_pars.asymmetry,
               'sampling_rate': s_pars.sampling_rate,
               'return_all': False}


# These are the params of the 5 qubit chip for swapping with the bus resonator
s_pars.E_c(300e6)
s_pars.J_2(np.sqrt(2)*50e6)
s_pars.lambda_coeffs(np.array([1, 0, 0]))
s_pars.theta_f(np.pi/2)
s_pars.f_interaction(4.8e9)
s_pars.f_01_max(5.94e9)

s_pars.print_readable_snapshot()
