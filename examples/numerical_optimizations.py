# import numpy as np
# import scipy
# from pycqed.measurement import sweep_functions as swf
# from sweep_functions import (Sweep_function, Soft_Sweep)
# from pycqed.measurement import AWG_sweep_functions as awg_swf
# from pycqed.measurement import detector_functions as det
# import matplotlib.pyplot as plt
# from pycqed.analysis import measurement_analysis as MA
'''
Warning: This code will not run as it is written in python2 for qtlab.
However it is here to serve as an example of using the different options
in the numerical optimizations.
'''


FakeSample = qt.instruments.create('FakeSample', 'Bart_parameter_holder',
                                   dummy_instrument=True)
MC = qt.instruments['MC']
try:
    MC.remove()
except:
    pass
MC = qt.instruments.create('MC', 'MeasurementControl', dummy_instrument=True)
'''
Plan for fixing issue 154
1. clean up passing of arguments V
2. Make MC work with arbitrary optimization functions  V
3. set mode as argument to MC (no longer global setting) V
4. get rid of the "scaling" parameter use stepsize instead -V
5. add termination condition V
'''


class sweep_function1(Soft_Sweep):
    def __init__(self, **kw):
        super(sweep_function1, self).__init__()
        # From Soft_Sweep the self.sweep_control = 'soft'
        self.name = 'Sweep_function1'
        self.parameter_name = 'x'
        self.unit = 'unit_x'
        self.FakeSample = qt.instruments['FakeSample']

    def set_parameter(self, val):
        self.FakeSample.set_x(val)

class sweep_function2(Soft_Sweep):
    def __init__(self, **kw):
        super(sweep_function2, self).__init__()
        self.name = 'Sweep_function2'
        self.parameter_name = 'y'
        self.unit = 'unit_y'
        self.FakeSample = qt.instruments['FakeSample']

    def set_parameter(self, val):
        self.FakeSample.set_y(val)

class test_detector(object):
    def __init__(self, **kw):
        self.FakeSample = qt.instruments['FakeSample']
        self.detector_control = 'soft'

    def acquire_data_point(self, **kw):
        return self.FakeSample.measure_convexity()

    def prepare(self,**kw):
        pass

    def finish(self,**kw):
        pass

class test_parabolic_detector(test_detector):
    def __init__(self, **kw):
        super(test_parabolic_detector, self).__init__()
        self.value_names = 'F'
        self.value_units = 'unit_F'

    def acquire_data_point(self, **kw):
        return self.FakeSample.measure_2D_sinc()

#sweepfunctions and detector
sweepfunctions = [sweep_function1(), sweep_function2()]
detector = test_parabolic_detector()

start_val = np.array([1, 1])
initial_stepsize = np.array([-.1, .2])

# Initial guess
x0 = start_val/initial_stepsize
# Scaling parameters
x_scale = 1/initial_stepsize

bounds0 = np.array([(-100, 100), (-100, 100)]).T/initial_stepsize # needs to be rearranged
bounds = np.zeros((len(bounds0[0]),2)) # Bounds for parameters (unused in Powell)
for i in range(len(bounds0[0])):
    bounds[i][0] = bounds0[0][i]
    bounds[i][1] = bounds0[1][i]
ftol = 1e-3
xtol = 1e-4
maxiter = 500 # Maximum No. iterations
maxfun = 500 # Maximum No. function evaluations
factr = 1e7     #1e7
pgtol = 1e-1       #2e-2
epsilon = 1e-1     #1e-2
epsilon_COBYLA = 0.2 # Initial step length
accuracy_COBYLA = 1e-2 # Convergence tolerance
constraints = np.array([100, 100])
minimize = True

noise = 0.01 # Maximum amplitude of the Zero-Mean White Noise
FakeSample.set_noise(noise)

# Example 1 find optimum using Powell method:
name = 'Powell method'
MC.set_sweep_functions(sweepfunctions) # sets swf1 and swf2
MC.set_detector_function(detector) # sets test_detector
ad_func_pars = {'adaptive_function': 'Powell',
                'x0': x0, 'x_scale': x_scale,  'ftol': ftol,
                'xtol': xtol, 'maxiter': maxiter, 'maxfun': maxfun,
                'minimize': minimize}
MC.set_adaptive_function_parameters(ad_func_pars)
MC.run(name=name, mode='adaptive')
MA.OptimizationAnalysis(auto=True, label=name)

# Example 2 find optimum by using a function that got passed by hand.
name = 'Nelder method'
ad_func_pars = {'adaptive_function': scipy.optimize.fmin,
                'x0': x0, 'x_scale': x_scale,  'ftol': ftol,
                'xtol': xtol, 'maxiter': maxiter, 'maxfun': maxfun,
                'minimize': minimize}
MC.set_adaptive_function_parameters(ad_func_pars)
MC.run(name=name, mode='adaptive')
MA.OptimizationAnalysis(auto=True, label=name)

# Example 3 make use of termination condition
name = 'Nelder_with_termination'
ad_func_pars = {'adaptive_function': scipy.optimize.fmin,
                'x0': x0, 'x_scale': x_scale,  'ftol': ftol,
                'xtol': xtol, 'maxiter': maxiter, 'maxfun': maxfun,
                'minimize': minimize,
                'f_termination': -1.95}
MC.set_adaptive_function_parameters(ad_func_pars)
MC.run(name=name, mode='adaptive')
MA.OptimizationAnalysis(auto=True, label=name)

# Example 4 make use of termination condition with maximize
name = 'Nelder_maximize'
ad_func_pars = {'adaptive_function': scipy.optimize.fmin,
                'x0': x0, 'x_scale': x_scale,  'ftol': ftol,
                'xtol': xtol, 'maxiter': maxiter, 'maxfun': maxfun,
                'minimize': False,
                'f_termination': .4}
MC.set_adaptive_function_parameters(ad_func_pars)
MC.run(name=name, mode='adaptive')
MA.OptimizationAnalysis(auto=True, label=name)

# Example 5 testing direc argument of Powell.
name = 'Powel_direction scale =1 no direc'
ad_func_pars = {'adaptive_function': 'Powell',
                'x0': [1, 1], 'x_scale': 1,  'ftol': ftol,
                'xtol': xtol, 'maxiter': maxiter, 'maxfun': maxfun}
MC.set_adaptive_function_parameters(ad_func_pars)
MC.run(name=name, mode='adaptive')
MA.OptimizationAnalysis(auto=True, label=name)

name = 'Powel_direction scale = 1 direc'
ad_func_pars = {'adaptive_function': 'Powell',
                'x0': [1, 1], 'ftol': ftol,
                'direc': ([.2, 0], [0, .1]),  # direc is a tuple of vectors
                'xtol': xtol, 'maxiter': maxiter, 'maxfun': maxfun}
MC.set_adaptive_function_parameters(ad_func_pars)
MC.run(name=name, mode='adaptive')
MA.OptimizationAnalysis(auto=True, label=name)

