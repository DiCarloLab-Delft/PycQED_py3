# %matplotlib inline
import scipy
from importlib import reload
import numpy as np
import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

from pycqed.measurement.optimization import nelder_mead
from pycqed.measurement import measurement_control as mc
from pycqed.measurement import detector_functions as det
from pycqed.measurement import mc_parameter_wrapper
from pycqed.scripts.personal_folders.Adriaan.CZ_sims import CZ_sims as czs
from pycqed.analysis import measurement_analysis as ma

station = qc.Station()
MC = mc.MeasurementControl('MC')
station.add_component(MC)
MC.station = station
station.MC = MC

# MC_outer = mc.MeasurementControl('MC_outer')
# station.add_component(MC_outer)
# MC_outer.station = station


# I create an empty instrument here that does nothing and will contain
# the parameters relevant for my simulation
s_pars = Instrument('Simulation parameters')

# adding all the manual parameters relevant for my simulation so that I
# can track and loop over them.
s_pars.add_parameter('pulse_length', unit='s',
                     parameter_class=ManualParameter, initial_value=20)
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
station.add_component(s_pars)


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
s_pars.J_2(np.sqrt(2)*25e6)
s_pars.lambda_coeffs(np.array([1., 0, 0]))
s_pars.theta_f(np.pi/2)
s_pars.f_interaction(4.8e9+s_pars.E_c())
s_pars.f_01_max(5.94e9)


sim_det = det.Function_Detector(czs.simulate_CZ_trajectory, params_dict,
                                value_names=['Conditional Phase', 'leakage'],
                                value_units=['deg', ' '])

# Quickly defining a cost function
def CZ_cost(**params_dict):
    cond_phase, leakage = czs.simulate_CZ_trajectory(**params_dict)
    phase_cost = abs(cond_phase-180)
    leakage_cost = leakage*100  # leakage weighted more heavily
    return phase_cost+0, cond_phase, leakage


sim_cost_det = det.Function_Detector(
    CZ_cost, params_dict,
    value_names=['Cost function', 'Conditional Phase', 'Leakage'],
    value_units=['a.u.', 'deg', ' '])


l1_swf = mc_parameter_wrapper.wrap_vector_par_to_swf(s_pars.lambda_coeffs, 0)
l1_swf.parameter_name = 'Lambda 1'
l2_swf = mc_parameter_wrapper.wrap_vector_par_to_swf(s_pars.lambda_coeffs, 1)
l2_swf.parameter_name = 'Lambda 2'

s_pars.pulse_length(50e-9)
MC.set_sweep_functions([s_pars.theta_f, l1_swf])
MC.set_sweep_points(np.linspace(.4*np.pi/2, 1.2*np.pi/2, 31))
MC.set_sweep_points_2D(np.linspace(0, 1, 11))
MC.set_detector_function(sim_cost_det)
MC.run('heatmap_{:.2f}ns'.format(s_pars.pulse_length()*1e9),
       mode='2D')
a = ma.MeasurementAnalysis(TwoD=True)
