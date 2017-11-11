import numpy as np
import qcodes as qc
import os
import logging
from copy import deepcopy
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
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

        self.msmt_suffix = '_' + name

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

        self.add_parameter('ro_pow_LO', label='RO power LO',
                           unit='dBm', initial_value=20,
                           parameter_class=ManualParameter)

        self.add_parameter('instr_MC', label='MeasurementControl',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_VSM', label='Vector Switch Matrix',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter(
            'instr_CC', label='Central Controller',
            docstring=('Device responsible for controlling the experiment'
                       ' using eQASM generated using OpenQL, in the near'
                       ' future will be the CC_Light.'),
            parameter_class=InstrumentRefParameter)

        ro_acq_docstr = (
            'Determines what type of integration weights to use: '
            '\n\t SSB: Single sideband demodulation\n\t'
            'DSB: Double sideband demodulation\n\t'
            'optimal: waveforms specified in "RO_acq_weight_func_I" '
            '\n\tand "RO_acq_weight_func_Q"')

        self.add_parameter('ro_acq_weight_type',
                           initial_value='DSB',
                           vals=vals.Enum('SSB', 'DSB', 'optimal'),
                           docstring=ro_acq_docstr,
                           parameter_class=ManualParameter)

        self.add_parameter('ro_qubits_list',
                           vals=vals.Lists(elt_validator=vals.Strings()),
                           parameter_class=ManualParameter,
                           docstring="Select which qubits to readout \
                                by default detector. Empty list means all.",
                           initial_value=[])

    def prepare_for_timedomain(self):
        pass

    def _setup_qubits(self):
        """
        set the parameters of the individual qubits to be compatible
        with multiplexed readout
        """

        if self.ro_acq_weight_type() == 'optimal':
            nr_of_acquisition_channels_per_qubit = 1
        else:
            nr_of_acquisition_channels_per_qubit = 2

        used_acq_channels = defaultdict(int)

        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)

            # all qubits use the same acquisition type
            qb.ro_acq_weight_type(self.ro_acq_weight_type())

            acq_device = qb.instr_acquisition()

            # allocate different acquisition channels
            # use next available channel as I
            index = used_acq_channels[acq_device]
            used_acq_channels[acq_device] += 1
            qb.ro_acq_weight_chI(index)

            # use next available channel as Q if needed
            if nr_of_acquisition_channels_per_qubit > 1:
                index = used_acq_channels[acq_device]
                used_acq_channels[acq_device] += 1
                qb.ro_acq_weight_chQ(index)

            # set RO modulation to use common LO frequency
            qb.ro_freq_mod(qb.ro_freq() - self.ro_lo_freq())

    def _setup_ro_lutmans(self):
        """
        set the combinations for all involved lutmans
        to involve the qubits that are supposed to be read out.
        """

        combs = defaultdict(lambda: [[]])

        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            lutman_name = qb.instr_ro_lutman()
            if self.ro_qubits_list() and qb_name not in self.ro_qubits_list():
                continue
            combs[lutman_name] = (combs[lutman_name] +
                                  [c + [qb.ro_pulse_res_nr()]
                                   for c in combs[lutman_name]])

        for lutman_name in combs:
            lutman = self.find_instrument(lutman_name)
            lutman.resonator_combinations(combs[lutman_name])

    def _prep_ro_instantiate_detectors(self):
        # collect which channels belong to which qubit and make a detector

        channels_list = []  # tuples (instrumentname, channel, description)

        for qb_name in self.qubits():
            if self.ro_qubits_list() and qb_name not in self.ro_qubits_list():
                continue

            qb = self.find_instrument(qb_name)
            acq_instr_name = qb.instr_acquisition()

            # one channel per qb
            if self.ro_acq_weight_type() == 'optimal':
                ch_idx = qb.ro_acq_weight_chI()
                channels_list.append((acq_instr_name, ch_idx, qb_name))
            else:
                ch_idx = qb.ro_acq_weight_chI()
                channels_list.append((acq_instr_name, ch_idx, qb_name + " I"))
                ch_idx = qb.ro_acq_weight_chQ()
                channels_list.append((acq_instr_name, ch_idx, qb_name + " Q"))

        # for now, implement only working with one UHFLI
        acq_instruments = list(set([inst for inst, _, _ in channels_list]))
        if len(acq_instruments) != 1:
            raise NotImplementedError("Only one acquisition"
                                      "instrument supported so far")

        if self.ro_acq_weight_type() == 'optimal':
            # todo: digitized mode
            result_logging_mode = 'lin_trans'
        else:
            result_logging_mode = 'raw'

        ro_ch_idx = [ch for _, ch, _ in channels_list]
        value_names = [n for _, _, n in channels_list]

        if 'UHFQC' in self.instr_acquisition():
            UHFQC = self.find_instrument(acq_instruments[0])

            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=UHFQC,
                AWG=self.instr_CC.get_instr(),
                nr_averages=self.ro_acq_averages(),
                nr_samples=int(self.ro_acq_input_average_length()*1.8e9))

            self.int_avg_det = det.UHFQC_integrated_average_detector(
                UHFQC=UHFQC, AWG=self.instr_CC.get_instr(),
                channels=ro_ch_idx,
                result_logging_mode=result_logging_mode,
                nr_averages=self.ro_acq_averages(),
                integration_length=self.ro_acq_integration_length())

            self.int_avg_det.value_names = value_names

            self.int_avg_det_single = det.UHFQC_integrated_average_detector(
                UHFQC=UHFQC, AWG=self.instr_CC.get_instr(),
                channels=ro_ch_idx,
                result_logging_mode=result_logging_mode,
                nr_averages=self.ro_acq_averages(),
                real_imag=True, single_int_avg=True,
                integration_length=self.ro_acq_integration_length())

            self.int_avg_det_single.value_names = value_names

            self.int_log_det = det.UHFQC_integration_logging_det(
                UHFQC=UHFQC, AWG=self.instr_CC.get_instr(),
                channels=ro_ch_idx,
                result_logging_mode=result_logging_mode,
                integration_length=self.ro_acq_integration_length())

            self.int_log_det.value_names = value_names

    def _prep_ro_sources(self):
        LO = self.instr_LO_ro.get_instr()
        LO.frequency.set(self.ro_lo_freq())
        LO.on()
        LO.power(self.ro_pow_LO())

    def _prep_ro_pulses(self):
        pass

    def prepare_for_timedomain(self):
        pass