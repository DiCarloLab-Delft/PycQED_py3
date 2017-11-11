import numpy as np
import qcodes as qc
import os
import logging
from copy import deepcopy
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
from pycqed.analysis import multiplexed_RO_analysis as mra
from pycqed.measurement import detector_functions as det
from pycqed.measurement import sweep_functions as swf
from pycqed.analysis import measurement_analysis as ma
import pycqed.measurement.gate_set_tomography.gate_set_tomography_CC as gstCC
import pygsti

from collections import defaultdict

class DeviceCCL(Instrument):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        self.msmt_suffix  = '_' + name

        self.add_parameter('qubits',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Lists(elt_validator=vals.Strings()))

        self.add_parameter('RO_acq_averages',
                           initial_value=1024,
                           vals=vals.Ints(),
                           parameter_class=ManualParameter)

        self.add_parameter(
            'ro_lo_freq', unit='Hz',
            docstring=('Frequency of the common LO for all RO pulses.'),
            parameter_class=ManualParameter)

        self.add_parameter('instr_MC', label='MeasurementControl',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter('ro_acq_weight_type',
                           initial_value='DSB',
                           vals=vals.Enum('SSB', 'DSB', 'optimal'),
                           docstring=ro_acq_docstr,
                           parameter_class=ManualParameter)


    def prepare_for_timedomain(self):
        pass

    def _setup_qubits(self):

        if self.ro_acq_weight_type() == 'optimal':
            nr_of_acquisition_channels_per_qubit = 1
        else:
            nr_of_acquisition_channels_per_qubit = 2


        for qb_name in self.qubits():

            qb = self.find_instrument(qubit_name)


        # setup RO pulse
        # allocate different resonators in lutman for different qubits

        # setup RO acq channels
        # allocate different channels in uhfli for different qubits (1 or 2)
