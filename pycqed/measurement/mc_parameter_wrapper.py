'''
Module containing functions that wrap a QCodes parameter into a sweep or
detector function
'''
import qcodes as qc
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det


def wrap_par_to_swf(parameter, retrieve_value=False):
    '''
     - only soft sweep_functions
    '''
    sweep_function = swf.Sweep_function()
    sweep_function.sweep_control = 'soft'
    sweep_function.name = parameter.name
    sweep_function.parameter_name = parameter.label
    sweep_function.unit = parameter.unit

    sweep_function.prepare = pass_function
    sweep_function.finish = pass_function
    if retrieve_value:
        def set_par(val):
            parameter.set(val)
            parameter.get()
        sweep_function.set_parameter = set_par
    else:
        sweep_function.set_parameter = parameter.set
    sweep_function.get = parameter.get
    return sweep_function


def wrap_pars_to_swf(parameters, retrieve_value=False):
    # FIXME: Shouldn't this be removed?
    '''
     - only soft sweep_functions
    '''
    sweep_function = swf.Sweep_function()
    sweep_function.sweep_control = 'soft'
    sweep_function.name = parameter.name
    sweep_function.parameter_name = parameter.label
    sweep_function.unit = parameter.unit

    def wrapped_set(val):
        old_val = parameter.get()
        vector_value = old_val
        vector_value[index] = val
        parameter.set(vector_value)

    sweep_function.prepare = pass_function
    sweep_function.finish = pass_function
    def set_par(val):
        for par in parameters:
            par.set(val)
            if retrieve_value:
                par.get()

    sweep_function.set_parameter = set_par

    return sweep_function


def wrap_pars_to_swf(parameters, retrieve_value=False):
    '''
     - only soft sweep_functions
    '''
    sweep_function = swf.Sweep_function()
    sweep_function.sweep_control = 'soft'
    sweep_function.name = parameters[0].name
    sweep_function.parameter_name = parameters[0].label
    sweep_function.unit = parameters[0].unit

    sweep_function.prepare = pass_function
    sweep_function.finish = pass_function

    def set_par(val):
        for par in parameters:
            par.set(val)
            if retrieve_value:
                par.get()

    sweep_function.set_parameter = set_par

    return sweep_function


def wrap_par_to_det(parameter, control='soft'):
    '''
    Takes in a QCoDeS Parameter instance and returns a PycQED DetectorFunction
    that wraps around the Parameter.

    The following attributes of the QCoDes parameter are used
        par.name    -> detector.name
        par.label   -> detector.value_names (either string or list of strings)
        par.unit    ->  detector.value_units
        par.get     -> detector.acquire_data_point
                    -> detector.get_values

    The following attributes are not taken from the parameter
        det.prepare             <- pass_function
        det.finish              <- pass_function
        det.detector_control    <- input argument of this function

    '''
    detector_function = det.Detector_Function()
    detector_function.detector_control = control
    detector_function.name = parameter.name
    if isinstance(parameter.label, list):
        detector_function.value_names = parameter.label
        detector_function.value_units = parameter.unit
    else:
        detector_function.value_names = [parameter.label]
        detector_function.value_units = [parameter.unit]

    detector_function.prepare = pass_function
    detector_function.finish = pass_function
    detector_function.acquire_data_point = parameter.get
    detector_function.get_values = parameter.get
    return detector_function


def pass_function(**kw):
    pass


def wrap_func_to_det(func, name, value_names, units, control='soft', **kw):
    detector_function = det.Detector_Function()
    detector_function.detector_control = control
    detector_function.name = name
    detector_function.value_names = value_names
    detector_function.value_units = units

    detector_function.prepare = pass_function
    detector_function.finish = pass_function

    def wrapped_func():
        return func(**kw)

    detector_function.acquire_data_point = wrapped_func
    detector_function.get_values = wrapped_func
    return detector_function


def wrap_par_remainder(par, remainder=1):
    new_par = qc.Parameter(name=par.name, label=par.label, unit=par.unit)

    def wrap_set(val):
        val = val % remainder
        par.set(val)
        par.get()
    new_par.set = wrap_set
    return new_par


def wrap_par_set_get(par):
    new_par = qc.Parameter(name=par.name, label=par.label, unit=par.unit)

    def wrap_set(val):
        par.set(val)
        par.get()
    new_par.set = wrap_set
    return new_par
