# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from pycqed.scripts.personal_folders.Adriaan.CZ_sims import CZ_sims as czs
import qcodes as qc
from pycqed.measurement import detector_functions as det
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from pycqed.measurement import measurement_control as mc
from qcodes.utils import validators as vals


station = qc.Station()
MC = mc.MeasurementControl('MC')
station.add_component(MC)
MC.station = station
station.MC = MC

s_pars = Instrument('Simulation parameters')
s_pars.add_parameter('pulse_length', units='s',
                     parameter_class=ManualParameter, initial_value=20)
s_pars.add_parameter('lambda_coeffs', units='a.u.',
                     parameter_class=ManualParameter,
                     initial_value=np.zeros(4), vals=vals.Arrays())

s_pars.add_parameter('theta_f', units='rad',
                     parameter_class=ManualParameter, initial_value=0.5*np.pi)
s_pars.add_parameter('f_01_max', units='Hz',
                     parameter_class=ManualParameter, initial_value=8e9)
s_pars.add_parameter('f_interaction', units='Hz',
                     parameter_class=ManualParameter, initial_value=6.8e9)
s_pars.add_parameter('J2', units='Hz',
                     parameter_class=ManualParameter, initial_value=40e6)
s_pars.add_parameter('E_c', units='Hz',
                     parameter_class=ManualParameter, initial_value=300e6)
s_pars.add_parameter('dac_flux_coefficient', units='a.u.',
                     parameter_class=ManualParameter, initial_value=1)
s_pars.add_parameter('asymmetry', units='a.u.',
                     parameter_class=ManualParameter, initial_value=0)
s_pars.add_parameter('sampling_rate', units='Hz',
                     parameter_class=ManualParameter, initial_value=2e9)
station.add_component(s_pars)


params_dict = {'length': s_pars.pulse_length,
               'lambda_coeffs': s_pars.lambda_coeffs,
               'theta_f': s_pars.theta_f,
               'f_01_max': s_pars.f_01_max,
               'f_interaction': s_pars.f_interaction,
               'J2': s_pars.J2,
               'E_c': s_pars.E_c,
               'dac_flux_coefficient': s_pars.dac_flux_coefficient,
               'asymmetry': s_pars.asymmetry,
               'sampling_rate': s_pars.sampling_rate,
               'return_all': False}


sim_det = det.Function_Detector(czs.simulate_CZ_trajectory, params_dict,
                                value_names=['Conditional Phase', 'leakage'],
                                value_units=['deg', ' '])

MC.set_sweep_function(s_pars.pulse_length)
MC.set_sweep_points(np.arange(10e-9, 50e-9, 5e-9))
MC.set_detector_function(sim_det)
MC.run('test_simulation')
