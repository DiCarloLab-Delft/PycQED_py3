'''
Module containing a collection of detector functions used by the
Measurement Control.
FIXME: split off hardware-specific detectors in separate files
'''

import numpy as np
import logging
import time
from deprecated import deprecated

from pycqed.analysis.fit_toolbox import functions as fn

# import instruments for type annotations
# from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
# from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController import UHFQC

# compatibility imports for functions that were moved under directory det_funcs. New code should use new locations
from pycqed.measurement.det_fncs.Base import Detector_Function, Mock_Detector, Multi_Detector, Soft_Detector, \
    Hard_Detector
from pycqed.measurement.det_fncs.hard.UHFQC import Multi_Detector_UHF, \
    UHFQC_input_average_detector, UHFQC_demodulated_input_avg_det, \
    UHFQC_spectroscopy_detector, UHFQC_integrated_average_detector, UHFQC_correlation_detector, \
    UHFQC_integration_logging_det, UHFQC_statistics_logging_det, UHFQC_single_qubit_statistics_logging_det
from pycqed.measurement.det_fncs.hard.SignalHound import Signal_Hound_fixed_frequency, Signal_Hound_sweeped_frequency, \
    SH_mixer_skewness_det


from qcodes.instrument.parameter import _BaseParameter

log = logging.getLogger(__name__)


##########################################################################
##########################################################################
####################     Hardware Controlled Detectors     ###############
##########################################################################
##########################################################################


class Dummy_Detector_Hard(Hard_Detector):

    def __init__(self, delay=0, noise=0, **kw):
        super(Dummy_Detector_Hard, self).__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = ['distance', 'Power']
        self.value_units = ['m', 'W']
        self.delay = delay
        self.noise = noise
        self.times_called = 0

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points

    def get_values(self):
        x = self.sweep_points
        noise = self.noise * (np.random.rand(2, len(x)) - .5)
        data = np.array([np.sin(x / np.pi),
                         np.cos(x / np.pi)])
        data += noise
        time.sleep(self.delay)
        # Counter used in test suite to test how many times data was acquired.
        self.times_called += 1

        return data


class Dummy_Shots_Detector(Hard_Detector):

    def __init__(self, max_shots=10, **kw):
        super().__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = ['shots']
        self.value_units = ['m']
        self.max_shots = max_shots
        self.times_called = 0

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points

    def get_values(self):
        x = self.sweep_points

        start_idx = self.times_called * self.max_shots % len(x)

        dat = x[start_idx:start_idx + self.max_shots]
        self.times_called += 1
        return dat


class Sweep_pts_detector(Detector_Function):
    """
    Returns the sweep points, used for testing purposes
    """

    def __init__(self, params, chunk_size=80):
        self.detector_control = 'hard'
        self.value_names = []
        self.value_units = []
        self.chunk_size = chunk_size
        self.i = 0
        for par in params:
            self.value_names += [par.name]
            self.value_units += [par.units]

    def prepare(self, sweep_points):
        self.i = 0
        self.sweep_points = sweep_points

    def get_values(self):
        return self.get()

    def acquire_data_point(self):
        return self.get()

    def get(self):
        print('passing chunk {}'.format(self.i))
        start_idx = self.i * self.chunk_size
        end_idx = start_idx + self.chunk_size
        self.i += 1
        time.sleep(.2)
        if len(np.shape(self.sweep_points)) == 2:
            return self.sweep_points[start_idx:end_idx, :].T
        else:
            return self.sweep_points[start_idx:end_idx]


class ZNB_VNA_detector(Hard_Detector):

    def __init__(self, VNA, **kw):
        '''
        Detector function for the Rohde & Schwarz ZNB VNA
        '''
        super(ZNB_VNA_detector, self).__init__()
        self.VNA = VNA
        # self.value_names = ['ampl', 'phase',
        #                    'real', 'imag', 'ampl_dB']
        self.value_names = ['ampl_dB', 'phase']
        # self.value_units = ['', 'radians',
        #                    '', '', 'dB']
        self.value_units = ['dB', 'radians']

    def get_values(self):
        '''
        Start a measurement, wait untill the end and retrive data.
        Return real and imaginary transmission coefficients +
        amplitude (linear) and phase (deg or radians)
        '''
        self.VNA.start_sweep_all()  # start a measurement
        # wait untill the end of measurement before moving on
        self.VNA.wait_to_continue()
        # for visualization on the VNA screen (no effect on data)
        self.VNA.autoscale_trace()
        # get data and process them
        real_data, imag_data = self.VNA.get_real_imaginary_data()

        complex_data = np.add(real_data, 1j * imag_data)
        ampl_linear = np.abs(complex_data)
        ampl_dB = 20 * np.log10(ampl_linear)
        phase_radians = np.arctan2(imag_data, real_data)

        # return ampl_linear, phase_radians, real_data, imag_data, ampl_dB
        return ampl_dB, phase_radians


##############################################################################
##############################################################################
####################     Software Controlled Detectors     ###################
##############################################################################
##############################################################################


class Dummy_Detector_Soft(Soft_Detector):

    def __init__(self, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']
        self.i = 0
        # self.x can be used to set x value externally
        self.x = None

    def acquire_data_point(self, **kw):
        if self.x is None:
            x = self.i / 15.
        self.i += 1
        time.sleep(self.delay)
        return np.array([np.sin(x / np.pi), np.cos(x / np.pi)])


class Dummy_Detector_Soft_diff_shape(Soft_Detector):
    # For testing purpose, returns data in a slightly different shape

    def __init__(self, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']
        self.i = 0
        # self.x can be used to set x value externally
        self.x = None

    def acquire_data_point(self, **kw):
        if self.x is None:
            x = self.i / 15.
        self.i += 1
        time.sleep(self.delay)
        # This is the format an N-D detector returns data in.
        return np.array([[np.sin(x / np.pi), np.cos(x / np.pi)]]).reshape(2, -1)


class Function_Detector(Soft_Detector):
    """
    Defines a detector function that wraps around an user-defined function.
    Inputs are:
        get_function (callable) : function used for acquiring values
        value_names (list) : names of the elements returned by the function
        value_units (list) : units of the elements returned by the function
        result_keys (list) : keys of the dictionary returned by the function
                             if not None
        msmt_kw   (dict)   : kwargs for the get_function, dict items can be
            values or parameters. If they are parameters the output of the
            get method will be used for each get_function evaluation.

        prepare_function (callable): function used as the prepare method
        prepare_kw (dict)   : kwargs for the prepare function
        always_prepare (bool) : if True calls prepare every time data is
            acquried

    The input function get_function must return a dictionary.
    The contents(keys) of this dictionary are going to be the measured
    values to be plotted and stored by PycQED
    """

    def __init__(self, get_function, value_names=None,
                 detector_control: str = 'soft',
                 value_units: list = None, msmt_kw: dict = {},
                 result_keys: list = None,
                 prepare_function=None, prepare_function_kwargs: dict = {},
                 always_prepare: bool = False, **kw):
        super().__init__()
        self.get_function = get_function
        self.result_keys = result_keys
        self.value_names = value_names
        self.value_units = value_units
        self.msmt_kw = msmt_kw
        self.detector_control = detector_control
        if self.value_names is None:
            self.value_names = result_keys
        if self.value_units is None:
            self.value_units = ['a.u.'] * len(self.value_names)

        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs
        self.always_prepare = always_prepare

    def prepare(self, **kw):
        if self.prepare_function is not None:
            self.prepare_function(**self.prepare_function_kwargs)

    def acquire_data_point(self, **kw):
        if self.always_prepare:
            self.prepare()
        measurement_kwargs = {}
        # If an entry has a get method that will be used to set the value.
        # This makes parameters work in this context.
        for key, item in self.msmt_kw.items():
            if isinstance(item, _BaseParameter):
                value = item.get()
            else:
                value = item
            measurement_kwargs[key] = value

        # Call the function
        result = self.get_function(**measurement_kwargs)
        if self.result_keys is None:
            return result
        else:
            results = [result[key] for key in self.result_keys]
            if len(results) == 1:
                return results[0]  # for a single entry we don't want a list
            return results

    def get_values(self):
        return self.acquire_data_point()


class Detect_simulated_hanger_Soft(Soft_Detector):

    def __init__(self, **kw):
        self.set_kw()

        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['mV', 'mV']

    def acquire_data_point(self, **kw):
        f = self.source.get_frequency()
        f0 = 5.e9
        Q = 10000.
        Qe = 12000.
        theta = 0.2
        A = 50.
        Inoise = np.random.randn()
        Qnoise = np.random.randn()

        IQ = fn.disp_hanger_S21_complex(*(f, f0, Q, Qe, A, theta))
        return IQ.real + Inoise, IQ.imag + Qnoise


class Heterodyne_probe(Soft_Detector):

    def __init__(self, HS, threshold=1.75, trigger_separation=10e-6,
                 demod_mode='double', RO_length=2000e-9, **kw):
        super().__init__(**kw)
        self.HS = HS
        self.name = 'Heterodyne probe'
        self.value_names = ['S21', 'S21 angle']  # , 'Re{S21}', 'Im{S21}']
        self.value_units = ['V', 'deg']
        self.first = True
        self.last_frequency = 0.
        self.threshold = threshold
        self.last = 1.
        self.HS.trigger_separation(trigger_separation)
        if 'double' in demod_mode:
            HS.single_sideband_demod(False)
        else:
            HS.single_sideband_demod(True)

        self.trigger_separation = trigger_separation
        self.demod_mode = demod_mode
        self.RO_length = RO_length

    def prepare(self):
        self.HS.RO_length(self.RO_length)
        self.HS.trigger_separation(self.trigger_separation)

        self.HS.prepare()

    def acquire_data_point(self, **kw):
        passed = False
        c = 0
        while (not passed):
            S21 = self.HS.probe()
            cond_a = ((abs(S21) / self.last) >
                      self.threshold) or ((self.last / abs(S21)) > self.threshold)
            cond_b = self.HS.frequency() >= self.last_frequency
            if cond_a and cond_b:
                passed = False
            else:
                passed = True
            if self.first or c > 3:
                passed = True
            # if not passed:
            #     print('retrying HS probe')
            c += 1
        self.last_frequency = self.HS.frequency()
        self.first = False
        self.last = abs(S21)
        return abs(S21), np.angle(S21) / (2 * np.pi) * 360,  # S21.real, S21.imag

    def finish(self):
        self.HS.finish()


class Heterodyne_probe_soft_avg(Soft_Detector):

    def __init__(self, HS, threshold=1.75, Navg=10, **kw):
        super().__init__(**kw)
        self.HS = HS
        self.name = 'Heterodyne probe'
        self.value_names = ['S21', 'S21 angle']  # , 'Re{S21}', 'Im{S21}']
        self.value_units = ['mV', 'deg']  # , 'a.u.', 'a.u.']
        self.first = True
        self.last_frequency = 0.
        self.threshold = threshold
        self.last = 1.
        self.Navg = Navg

    def prepare(self):
        self.HS.prepare()

    def acquire_data_point(self, **kw):
        accum_real = 0.
        accum_imag = 0.
        for i in range(self.Navg):
            measure = self.acquire_single_data_point(**kw)
            accum_real += measure[0]
            accum_imag += measure[1]
        S21 = (accum_real + 1j * accum_imag) / float(self.Navg)

        return abs(S21), np.angle(S21) / (2 * np.pi) * 360

    def acquire_single_data_point(self, **kw):
        passed = False
        c = 0
        while (not passed):
            S21 = self.HS.probe()
            cond_a = (
                             abs(S21) / self.last > self.threshold) or (self.last / abs(S21) > self.threshold)
            cond_b = self.HS.frequency() > self.last_frequency
            if cond_a and cond_b:
                passed = False
            else:
                passed = True
            if self.first or c > 3:
                passed = True
            if not passed:
                print('retrying HS probe')
            c += 1
        self.last_frequency = self.HS.frequency()
        self.first = False
        self.last = abs(S21)
        return S21.real, S21.imag

    def finish(self):
        self.HS.finish()



# --------------------------------------------
# Fake detectors
# --------------------------------------------

@deprecated(version='0.4', reason="broken code")
class Chevron_sim(Hard_Detector):
    """
    Returns a simulated chevron as if it was measured.
    """

    def __init__(self, simulation_dict, **kw):
        super(Chevron_sim, self).__init__(**kw)
        self.simulation_dict = simulation_dict
        self.name = 'Simulated Chevron'
        self.value_names = []
        self.value_units = []

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points
        self.t_start = self.sweep_points[0]
        self.dt = self.sweep_points[1] - self.sweep_points[0]

    # FIXME: missing chev_lib
    # def get_values(self):
    #     return chev_lib.chevron_slice(self.simulation_dict['detuning'],
    #                                   self.simulation_dict['dist_step'],
    #                                   self.simulation_dict['g'],
    #                                   self.t_start,
    #                                   self.dt,
    #                                   self.simulation_dict['dist_step'])


@deprecated(version='0.4', reason="use Function_Detector")
class Function_Detector_list(Soft_Detector):
    """
    Defines a detector function that wraps around an user-defined function.
    Inputs are:
        sweep_function, function that is going to be wrapped around
        result_keys, keys of the dictionary returned by the function
        value_names, names of the elements returned by the function
        value_units, units of the elements returned by the function
        msmt_kw, kw arguments for the function
    The input function sweep_function must return a dictionary.
    The contents(keys) of this dictionary are going to be the measured
    values to be plotted and stored by PycQED
    """

    def __init__(self, sweep_function, result_keys, value_names=None,
                 value_unit=None, msmt_kw=None, **kw):
        log.warning("Deprecation warning. Function_Detector_list "
                    "is deprecated, use Function_Detector")
        super(Function_Detector_list, self).__init__()
        self.sweep_function = sweep_function
        self.result_keys = result_keys
        self.value_names = value_names
        self.value_units = value_unit
        self.msmt_kw = msmt_kw or {}
        if self.value_names is None:
            self.value_names = result_keys
        if self.value_units is None:
            self.value_units = [""] * len(result_keys)

    def acquire_data_point(self, **kw):
        return self.sweep_function(**self.msmt_kw)
