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
import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo
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

        # actually, it should be possible to build the integration
        # weights obeying different settings for different
        # qubits, but for now we use a fixed common value.

        self.add_parameter('ro_acq_integration_length', initial_value=500e-9,
                           vals=vals.Numbers(min_value=0, max_value=20e6),
                           parameter_class=ManualParameter)

        self.add_parameter('ro_pow_LO', label='RO power LO',
                           unit='dBm', initial_value=20,
                           parameter_class=ManualParameter)
        self.add_parameter('ro_acq_averages', initial_value=1024,
                           vals=vals.Numbers(min_value=0, max_value=1e6),
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
                           initial_value='SSB',
                           vals=vals.Enum('SSB', 'DSB', 'optimal'),
                           docstring=ro_acq_docstr,
                           parameter_class=ManualParameter)

        self.add_parameter('ro_qubits_list',
                           vals=vals.Lists(elt_validator=vals.Strings()),
                           parameter_class=ManualParameter,
                           docstring="Select which qubits to readout \
                                by default detector. Empty list means all.",
                           initial_value=[])

        self.add_parameter('cfg_openql_platform_fn',
                           label='OpenQL platform configuration filename',
                           parameter_class=ManualParameter,
                           vals=vals.Strings())

    def _grab_instruments_from_qb(self):
        """
        initialize instruments that should only exist once from the first
        qubit. Maybe must be done in a more elegant way (at least check
        uniqueness).
        """

        qb = self.find_instrument(self.qubits()[0])
        self.instr_MC(qb.instr_MC())
        self.instr_VSM(qb.instr_VSM())
        self.instr_CC(qb.instr_CC())
        self.cfg_openql_platform_fn(qb.cfg_openql_platform_fn())

    def prepare_readout(self):
        self._prep_ro_setup_qubits()
        self._prep_ro_instantiate_detectors()
        self._prep_ro_sources()
        self._prep_ro_pulses()
        self._prep_ro_integration_weights()

    def _prep_ro_setup_qubits(self):
        """
        set the parameters of the individual qubits to be compatible
        with multiplexed readout.
        """

        ro_qb_list = self.ro_qubits_list()
        if ro_qb_list == []:
            ro_qb_list = self.qubits()

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

    def _prep_ro_integration_weights(self):
        """
        Set the acquisition integration weights on each channel
        """
        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            qb._prep_ro_integration_weights()

    def _prep_ro_instantiate_detectors(self):
        """
        collect which channels are being used for which qubit and make
        detectors.
        """


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

        if 'UHFQC' in acq_instruments[0]:
            UHFQC = self.find_instrument(acq_instruments[0])

            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=UHFQC,
                AWG=self.instr_CC.get_instr(),
                nr_averages=self.ro_acq_averages(),
                nr_samples=int(self.ro_acq_integration_length()*1.8e9))

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
        """
        turn on and configure the RO LO's of all qubits to be measured.
        """
        ro_qb_list = self.ro_qubits_list()
        if ro_qb_list == []:
            ro_qb_list = self.qubits()

        for qb_name in ro_qb_list:
            LO = self.find_instrument(qb_name).instr_LO_ro.get_instr()
            LO.frequency.set(self.ro_lo_freq())
            LO.power(self.ro_pow_LO())
            LO.on()

    def _prep_ro_pulses(self):
        """
        configure lutmans to measure the qubits and
        let the lutman configure the readout AWGs.
        """
        # these are the qubits that should be possible to read out
        ro_qb_list = self.ro_qubits_list()
        if ro_qb_list == []:
            ro_qb_list = self.qubits()

        lutmans_to_configure = {}

        # calculate the combinations that each ro_lutman should be able to do
        combs = defaultdict(lambda: [[]])

        for qb_name in ro_qb_list:
            qb = self.find_instrument(qb_name)

            ro_lm = qb.instr_LutMan_RO.get_instr()
            lutmans_to_configure[ro_lm.name] = ro_lm
            res_nr = qb.ro_pulse_res_nr()

            # extend the list of combinations to be set for the lutman

            combs[ro_lm.name] = (combs[ro_lm.name] +
                                 [c + [res_nr]
                                  for c in combs[ro_lm.name]])

            # These parameters affect all resonators.
            # Should not be part of individual qubits

            ro_lm.set('pulse_type', 'M_' + qb.ro_pulse_type())
            ro_lm.set('mixer_alpha',
                      qb.ro_pulse_mixer_alpha())
            ro_lm.set('mixer_phi',
                      qb.ro_pulse_mixer_phi())
            ro_lm.set('mixer_offs_I', qb.ro_pulse_mixer_offs_I())
            ro_lm.set('mixer_offs_Q', qb.ro_pulse_mixer_offs_Q())
            ro_lm.acquisition_delay(qb.ro_acq_delay())

            # configure the lutman settings for the pulse on the resonator
            # of this qubit

            ro_lm.set('M_modulation_R{}'.format(res_nr), qb.ro_freq_mod())
            ro_lm.set('M_length_R{}'.format(res_nr),
                      qb.ro_pulse_length())
            ro_lm.set('M_amp_R{}'.format(res_nr),
                      qb.ro_pulse_amp())
            ro_lm.set('M_phi_R{}'.format(res_nr),
                      qb.ro_pulse_phi())
            ro_lm.set('M_down_length0_R{}'.format(res_nr),
                      qb.ro_pulse_down_length0())
            ro_lm.set('M_down_amp0_R{}'.format(res_nr),
                      qb.ro_pulse_down_amp0())
            ro_lm.set('M_down_phi0_R{}'.format(res_nr),
                      qb.ro_pulse_down_phi0())
            ro_lm.set('M_down_length1_R{}'.format(res_nr),
                      qb.ro_pulse_down_length1())
            ro_lm.set('M_down_amp1_R{}'.format(res_nr),
                      qb.ro_pulse_down_amp1())
            ro_lm.set('M_down_phi1_R{}'.format(res_nr),
                      qb.ro_pulse_down_phi1())

        for ro_lm_name, ro_lm in lutmans_to_configure.items():
            ro_lm.resonator_combinations(combs[ro_lm_name][1:])
            ro_lm.load_DIO_triggered_sequence_onto_UHFQC()
            ro_lm.set_mixer_offsets()

    def _prep_td_configure_VSM(self):
        """
        turn off all VSM channels and then use qubit settings to
        turn on the required channels again.
        """

        # turn all channels on all VSMS off
        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            VSM = qb.instr_VSM.get_instr()
            VSM.set_all_switches_to('OFF')

        # turn the desired channels on
        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)

            # Configure VSM
            # N.B. This configure VSM block is geared specifically to the
            # Duplexer/BlueBox VSM
            VSM = qb.instr_VSM.get_instr()
            Gin = qb.mw_vsm_ch_Gin()
            Din = qb.mw_vsm_ch_Din()
            out = qb.mw_vsm_ch_out()

            VSM.set('in{}_out{}_switch'.format(Gin, out), qb.mw_vsm_switch())
            VSM.set('in{}_out{}_switch'.format(Din, out), qb.mw_vsm_switch())

            VSM.set('in{}_out{}_att'.format(Gin, out), qb.mw_vsm_G_att())
            VSM.set('in{}_out{}_att'.format(Din, out), qb.mw_vsm_D_att())
            VSM.set('in{}_out{}_phase'.format(Gin, out), qb.mw_vsm_G_phase())
            VSM.set('in{}_out{}_phase'.format(Din, out), qb.mw_vsm_D_phase())

            self.instr_CC.get_instr().set(
                'vsm_channel_delay{}'.format(qb.cfg_qubit_nr()),
                qb.mw_vsm_delay())

    def prepare_for_timedomain(self):
        self.prepare_readout()

        for qb_name in self.qubits():
            qb = self.find_instrument(qb_name)
            qb._prep_td_sources()
            qb._prep_mw_pulses()

        self._prep_td_configure_VSM()

    def measure_two_qubit_AllXY(self, q0: str, q1: str,
                                sequence_type='sequential',
                                replace_q1_pulses_X180: bool=False,
                                analyze=True, close_fig=True,
                                prepare_for_timedomain=True, MC=None):

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        assert q0 in self.qubits()
        assert q1 in self.qubits()

        q0idx = self.find_instrument(q0).cfg_qubit_nr()
        q1idx = self.find_instrument(q1).cfg_qubit_nr()

        p = mqo.two_qubit_AllXY(q0idx, q1idx,
                                platf_cfg=self.cfg_openql_platform_fn(),
                                sequence_type=sequence_type,
                                replace_q1_pulses_X180=replace_q1_pulses_X180,
                                double_points=True)
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(42))
        MC.set_detector_function(d)
        MC.run('TwoQubitAllXY_{}_{}_{}'.format(q0, q1, self.msmt_suffix))
        if analyze:
            a = ma.MeasurementAnalysis(close_main_fig=close_fig)
