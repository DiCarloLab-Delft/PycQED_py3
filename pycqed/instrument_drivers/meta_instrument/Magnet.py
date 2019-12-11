# Magnet.py

import time
import logging
import numpy as np
from scipy.optimize import brent
from math import gcd
from qcodes import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.utilities.general import add_suffix_to_dict_keys

from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.measurement import mc_parameter_wrapper as pw

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_5014
from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_UHFQC
from pycqed.measurement.calibration_toolbox import mixer_skewness_calibration_5014
from pycqed.measurement.optimization import nelder_mead

import pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts as sq
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter

class Magnet(Instrument):

    shared_kwargs = ['Isource']
    def __init__(self, name, Isource, **kw):
        super().__init__(name, **kw)

        self.add_parameter('Isource', parameter_class=InstrumentParameter)
        self.Isource = Isource

        self.add_parameter('field',
                           get_cmd=self.measure_field,
                           get_parser=float,
                           set_cmd =self.step_magfield_to_value,
                           label='Measure or set magnetic field',
                           unit='T',
                           vals=vals.Numbers(min_value=0., max_value=0.251))

        self.add_parameter('resistance',
                           get_cmd=self.Isource.measureR,
                           get_parser=float,
                           label = 'Magnet Resistance',
                           unit='Ohm')

        # self.add_parameter('amount_of_steps')

        # self.add_parameter('acquisition_instr',
        #                    set_cmd=self._do_set_acquisition_instr,
        #                    get_cmd=self._do_get_acquisition_instr,
        #                    vals=vals.Strings())
    def BtoI(self, magfield):
        MagCurrRatio = 0.0516 # Tesla/Ampere
        I = magfield/MagCurrRatio
        return I

    def ItoB(self, current):
        MagCurrRatio = 0.0516 # Tesla/Ampere
        B = current*MagCurrRatio
        return B


    def measure_field(self):
        if self.Isource is not None:
            I = self.Isource.measurei()
            B = self.ItoB(I)
            return B
        else:
            print('no Isource')

    def step_magfield_to_value(self, field):
        MagCurrRatio = 0.0516 # Tesla/Ampere
        Ramp_rate_I = 0.01 # A/s max = 0.36
        mag_field_rate = MagCurrRatio * Ramp_rate_I # Tesla/s

        step_time = 0.01
        # mag_field_step = mag_field_rate*step_time
        current_step = Ramp_rate_I  * step_time
        I_now = self.Isource.measurei()
        current_target = self.BtoI(field)
        if current_target >= I_now:
            current_step *= +1
        if current_target < I_now:
            current_step *= -1
        num_steps = int(1.*(current_target-I_now)/(current_step))
        # mag_now = self.Isource.measurei() * MagCurrRatio
        # num_steps = int(1.*(field-mag_now)/(mag_field_step))
        sweep_time = step_time*num_steps
        print(np.abs(sweep_time))

        for tt in range(num_steps):

            time.sleep(step_time)
            self.Isource.seti(I_now)
            I_now += current_step

        if self.Isource.measureR() > 1:
            self.Isource.seti(0)
            return 'non-superconducting'

        self.Isource.seti(self.BtoI(field))