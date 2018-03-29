import logging
import numpy as np
import matplotlib.pyplot as plt

import qcodes as qc

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_swf2
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.measurement.pulse_sequences import calibration_elements as cal_elts
from pycqed.analysis import measurement_analysis as ma
import pycqed.analysis.randomized_benchmarking_analysis as rbma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit
from pycqed.measurement import optimization as opti
from pycqed.measurement import mc_parameter_wrapper


class QuDev_transmon(Qubit):
    def __init__(self, name, MC,
                 heterodyne=None,  # metainstrument for cw spectroscopy
                 cw_source=None,  # MWG for driving the qubit continuously
                 readout_DC_LO=None,  # MWG for downconverting RO signal
                 readout_UC_LO=None,  # MWG for upconverting RO signal
                 readout_RF=None,  # MWG for driving the readout resonator
                 drive_LO=None,
                 AWG=None,  # for generating IQ modulated drive pulses and
                            # triggering instruments
                 UHFQC=None,  # For readout
                 DC_source=None,  # DC voltage source for changing flux bias
                 **kw):
        super().__init__(name, **kw)

        self.MC = MC
        self.heterodyne = heterodyne
        self.cw_source = cw_source
        self.AWG = AWG
        self.UHFQC = UHFQC
        self.readout_DC_LO = readout_DC_LO
        self.readout_UC_LO = readout_UC_LO
        self.readout_RF = readout_RF
        self.drive_LO = drive_LO

        self.add_parameter('f_RO_resonator', label='RO resonator frequency',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('Q_RO_resonator', label='RO resonator Q factor',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('ssro_contrast', unit='arb.', initial_value=0,
                           label='integrated g-e trace contrast',
                           parameter_class=ManualParameter)
        self.add_parameter('optimal_acquisition_delay', label='Optimal '
                           'acquisition delay', unit='s', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('f_qubit', label='Qubit frequency', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('f_ef_qubit', label='Qubit ef frequency', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('T1', label='Qubit relaxation', unit='s',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('T1_ef', label='Qubit relaxation', unit='s',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('T2', label='Qubit dephasing Echo', unit='s',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('T2_ef', label='Qubit dephasing Echo', unit='s',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('T2_star', label='Qubit dephasing', unit='s',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('T2_star_ef', label='Qubit dephasing', unit='s',
                           initial_value=0, parameter_class=ManualParameter)
        # self.add_parameter('amp180', label='Qubit pi pulse amp', unit='V',
        #                    initial_value=0, parameter_class=ManualParameter)
        # self.add_parameter('amp90', label='Qubit pi/2 pulse amp', unit='V',
        #                    initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('anharmonicity', label='Qubit anharmonicity',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('dynamic_phase', label='CZ dynamic phase',
                           unit='deg', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('EC_qubit', label='Qubit EC', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('EJ_qubit', label='Qubit EJ', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('spec_pow', unit='dBm', initial_value=-20,
                           parameter_class=ManualParameter,
                           label='Qubit spectroscopy power')
        self.add_parameter('f_RO', unit='Hz', parameter_class=ManualParameter,
                           label='Readout frequency')
        self.add_parameter('drive_LO_pow', unit='dBm',
                           parameter_class=ManualParameter,
                           label='Qubit drive pulse mixer LO power')
        self.add_parameter('pulse_I_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the drive line I channel')
        self.add_parameter('pulse_Q_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the drive line Q channel')
        self.add_parameter('mw_uc_switch', initial_value=None,
                           label='Drive upconversion board',
                           vals=vals.Enum(*(['UC'+str(i) for i in range(1,6)])),
                           parameter_class=ManualParameter)
        self.add_parameter('dc_source_channel', initial_value=None,
                           label='Channel name of the DC source for flux bias',
                           vals=vals.Strings(), parameter_class=ManualParameter)
        self.add_parameter('RO_pulse_power', unit='dBm',
                           parameter_class=ManualParameter,
                           label='Readout signal power')
        self.add_parameter('RO_I_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the readout I channel')
        self.add_parameter('RO_Q_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the readout Q channel')
        self.add_parameter('RO_acq_averages', initial_value=1024,
                           vals=vals.Ints(0, 1000000),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_integration_length', initial_value=2.2e-6,
                           vals=vals.Numbers(min_value=10e-9, max_value=2.2e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_I', initial_value=0,
                           vals=vals.Enum(0, 1, 2, 3, 4, 5, 6, 7, 8),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_Q', initial_value=1,
                           vals=vals.Enum(None, 0, 1, 2, 3, 4, 5, 6, 7, 8),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_shots', initial_value=4094,
                           docstring='Number of single shot measurements to do'
                                     'in single shot experiments.',
                           vals=vals.Ints(0, 4095),
                           parameter_class=ManualParameter)

        self.add_parameter('RO_IQ_angle', initial_value=0,
                           docstring='The phase of the integration weights when'
                                     'using SSB, DSB or square_rot integration '
                                     'weights', label='RO IQ angle', unit='rad',
                           parameter_class=ManualParameter)

        self.add_parameter('ro_acq_weight_func_I', vals=vals.Arrays(),
                           label='Optimized weights for I channel',
                           parameter_class=ManualParameter)
        self.add_parameter('ro_acq_weight_func_Q', vals=vals.Arrays(),
                           label='Optimized weights for Q channel',
                           parameter_class=ManualParameter)
        self.add_parameter('ro_acq_input_average_length', unit='s',
                           initial_value=2.275e-6, docstring='The measurement '
                               'time in input averaging mode',
                           label='Input average measurement time',
                           vals=vals.Numbers(0, 2.275e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('ro_acq_weight_type', initial_value='SSB',
                           vals=vals.Enum('SSB', 'DSB', 'optimal',
                                          'square_rot', 'manual'),
                           docstring=(
                               'Determines what type of integration weights to '
                               'use: \n\tSSB: Single sideband demodulation\n\t'
                               'DSB: Double sideband demodulation\n\toptimal: '
                               'waveforms specified in "ro_acq_weight_func_I" '
                               'and "ro_acq_weight_func_Q"\n\tsquare_rot: uses '
                               'a single integration channel with boxcar '
                               'weights'),
                           parameter_class=ManualParameter)

        self.add_parameter('ro_alpha', initial_value=1,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('ro_phi_skew', initial_value=0, unit='deg',
                           docstring='Readout upconversion mixer phase '
                                     'imbalance',
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)

        # add pulsed spectroscopy pulse parameters
        self.add_operation('Spec')
        self.add_pulse_parameter('Spec', 'spec_pulse_type', 'pulse_type',
                                 vals=vals.Strings(),
                                 initial_value='SquarePulse')
        self.add_pulse_parameter('Spec', 'spec_pulse_marker_channel', 'channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('Spec', 'spec_pulse_amp', 'amplitude',
                                 vals=vals.Numbers(), initial_value=1)
        self.add_pulse_parameter('Spec', 'spec_pulse_length', 'length',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('Spec', 'spec_pulse_depletion_time',
                                 'pulse_delay', vals=vals.Numbers(),
                                 initial_value=None)

        # add readout pulse parameters
        self.add_operation('RO')
        self.add_pulse_parameter('RO', 'RO_pulse_type', 'pulse_type',
                                 vals=vals.Strings(),
                                 initial_value='MW_IQmod_pulse_UHFQC')
        self.add_pulse_parameter('RO', 'RO_I_channel', 'I_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('RO', 'RO_Q_channel', 'Q_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('RO', 'RO_pulse_marker_channel',
                                 'RO_pulse_marker_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('RO', 'RO_amp', 'amplitude',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'RO_pulse_length', 'length',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'RO_pulse_delay', 'pulse_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'f_RO_mod', 'mod_frequency',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'RO_acq_marker_delay',
                                 'acq_marker_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'RO_acq_marker_channel',
                                 'acq_marker_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('RO', 'RO_pulse_phase', 'phase',
                                 initial_value=None, vals=vals.Numbers())
        # add drive pulse parameters
        self.add_operation('X180')
        self.add_pulse_parameter('X180', 'pulse_type', 'pulse_type',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180', 'pulse_I_channel', 'I_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180', 'pulse_Q_channel', 'Q_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180', 'amp180', 'amplitude',
                                 initial_value=1, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'amp90_scale', 'amp90_scale',
                                 initial_value=0.5, vals=vals.Numbers(0, 1))
        self.add_pulse_parameter('X180', 'pulse_delay', 'pulse_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'gauss_sigma', 'sigma',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'nr_sigma', 'nr_sigma',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'motzoi', 'motzoi',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'f_pulse_mod', 'mod_frequency',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'phi_skew', 'phi_skew',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'alpha', 'alpha',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'X_pulse_phase', 'phase',
                                 initial_value=None, vals=vals.Numbers())

        # add drive pulse parameters for ef transition
        self.add_operation('X180_ef')
        self.add_pulse_parameter('X180_ef', 'pulse_type_ef', 'pulse_type',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180_ef', 'amp180_ef', 'amplitude',
                                 initial_value=1, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'amp90_scale_ef', 'amp90_scale',
                                 initial_value=0.5, vals=vals.Numbers(0, 1))
        self.add_pulse_parameter('X180_ef', 'pulse_delay_ef', 'pulse_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'gauss_sigma_ef', 'sigma',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'nr_sigma_ef', 'nr_sigma',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'motzoi_ef', 'motzoi',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'X_pulse_phase_ef', 'phase',
                                 initial_value=None, vals=vals.Numbers())

        # add flux pulse parameters
        self.add_operation('flux')
        self.add_pulse_parameter('flux', 'flux_pulse_type', 'pulse_type',
                                 initial_value='SquarePulse',
                                 vals=vals.Enum('SquarePulse'))
        self.add_pulse_parameter('flux', 'flux_pulse_channel', 'channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('flux', 'flux_pulse_amp', 'amplitude',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('flux', 'flux_pulse_length', 'length',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('flux', 'flux_pulse_delay', 'pulse_delay',
                                 initial_value=None, vals=vals.Numbers())

        self.update_detector_functions()

    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name}

    def update_detector_functions(self):

        if self.RO_acq_weight_function_Q() is None or \
           self.ro_acq_weight_type() not in ['SSB', 'DSB']:
            channels = [self.RO_acq_weight_function_I()]
        else:
            channels = [self.RO_acq_weight_function_I(),
                        self.RO_acq_weight_function_Q()]

        self.int_log_det = det.UHFQC_integration_logging_det(
            UHFQC=self.UHFQC, AWG=self.AWG, channels=channels,
            integration_length=self.RO_acq_integration_length(),
            nr_shots=self.RO_acq_shots())

        self.int_avg_det = det.UHFQC_integrated_average_detector(
            self.UHFQC, self.AWG, nr_averages=self.RO_acq_averages(),
            channels=channels,
            integration_length=self.RO_acq_integration_length())

        self.dig_avg_det = det.UHFQC_integrated_average_detector(
            self.UHFQC, self.AWG, nr_averages=self.RO_acq_averages(),
            channels=channels,
            integration_length=self.RO_acq_integration_length(),
            result_logging_mode='digitized')

        self.inp_avg_det = det.UHFQC_input_average_detector(
            UHFQC=self.UHFQC, AWG=self.AWG, nr_averages=self.RO_acq_averages(),
            nr_samples=int(self.ro_acq_input_average_length()*1.8e9))

        self.dig_log_det = det.UHFQC_integration_logging_det(
            UHFQC=self.UHFQC, AWG=self.AWG, channels=channels,
            integration_length=self.RO_acq_integration_length(),
            nr_shots=self.RO_acq_shots(), result_logging_mode='digitized')

        self.int_avg_det_spec = det.UHFQC_integrated_average_detector(
            UHFQC=self.UHFQC, result_logging_mode='raw',
            channels=[self.RO_acq_weight_function_I(),
                      self.RO_acq_weight_function_Q()],
            nr_averages=self.RO_acq_averages(),
            real_imag=False, single_int_avg=True,
            integration_length=self.RO_acq_integration_length())

    def prepare_for_continuous_wave(self):
        self.heterodyne.auto_seq_loading(True)
        self.heterodyne._awg_seq_parameters_changed = True
        self.heterodyne._UHFQC_awg_parameters_changed = True
        if self.cw_source is not None:
            self.cw_source.off()
            self.cw_source.pulsemod_state('Off')
            self.cw_source.power.set(self.spec_pow())
        if self.readout_RF is not None:
            self.readout_RF.pulsemod_state('Off')
        if self.readout_DC_LO is not None:
            self.readout_DC_LO.pulsemod_state('Off')
        if self.readout_UC_LO is not None:
            self.readout_UC_LO.pulsemod_state('Off')


    def prepare_for_pulsed_spec(self):

        if self.cw_source is not None:
            self.cw_source.pulsemod_state('On')
            self.cw_source.on()
            self.cw_source.power.set(self.spec_pow())
        if self.f_RO() is None:
            f_RO = self.f_RO_resonator()
        else:
            f_RO = self.f_RO()

        eval('self.UHFQC.sigouts_{}_offset({})'.format(
            self.RO_I_channel(), self.RO_I_offset()))
        eval('self.UHFQC.sigouts_{}_offset({})'.format(
            self.RO_Q_channel(), self.RO_Q_offset()))
        self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(
            f_RO_mod=self.f_RO_mod(), RO_amp=self.RO_amp(),
            RO_pulse_length=self.RO_pulse_length(),
            alpha=self.ro_alpha(), phi_skew=self.ro_phi_skew())
        self.readout_UC_LO.pulsemod_state('Off')
        self.readout_UC_LO.frequency(f_RO - self.f_RO_mod())
        self.readout_UC_LO.on()

        self.heterodyne.auto_seq_loading(False)
        # self.heterodyne._awg_seq_parameters_changed = False
        # self.heterodyne._UHFQC_awg_parameters_changed = False

    def prepare_for_timedomain(self, multiplexed=False):
        # cw source
        if self.cw_source is not None:
            self.cw_source.off()

        # drive LO
        self.drive_LO.pulsemod_state('Off')
        self.drive_LO.frequency(self.f_qubit() - self.f_pulse_mod())
        self.drive_LO.power(self.drive_LO_pow())
        self.drive_LO.on()

        # drive modulation
        self.AWG.set(self.pulse_I_channel() + '_offset',
                     self.pulse_I_offset())
        self.AWG.set(self.pulse_Q_channel() + '_offset',
                     self.pulse_Q_offset())

        # readout LO
        if self.f_RO() is None:
            f_RO = self.f_RO_resonator()
        else:
            f_RO = self.f_RO()
        self.readout_DC_LO.pulsemod_state('Off')
        self.readout_DC_LO.frequency(f_RO - self.f_RO_mod())
        self.readout_DC_LO.on()

        # readout pulse
        if not multiplexed:
            if self.RO_pulse_type() == 'MW_IQmod_pulse_UHFQC':
                eval('self.UHFQC.sigouts_{}_offset({})'.format(
                    self.RO_I_channel(), self.RO_I_offset()))
                eval('self.UHFQC.sigouts_{}_offset({})'.format(
                    self.RO_Q_channel(), self.RO_Q_offset()))
                self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(
                    f_RO_mod=self.f_RO_mod(), RO_amp=self.RO_amp(),
                    RO_pulse_length=self.RO_pulse_length(),
                    alpha=self.ro_alpha(), phi_skew=self.ro_phi_skew())
                self.readout_UC_LO.pulsemod_state('Off')
                self.readout_UC_LO.frequency(f_RO - self.f_RO_mod())
                self.readout_UC_LO.on()
            elif self.RO_pulse_type() == 'Gated_MW_RO_pulse':
                self.readout_RF.pulsemod_state('On')
                self.readout_RF.frequency(f_RO)
                self.readout_RF.power(self.RO_pulse_power())
                self.readout_RF.on()
                self.UHFQC.awg_sequence_acquisition()
        else:
            # setting up the UHFQC awg sequence must be done externally by a
            # readout manager
            self.readout_UC_LO.pulsemod_state('Off')
            self.readout_UC_LO.frequency(f_RO - self.f_RO_mod())
            self.readout_UC_LO.on()

        self.update_detector_functions()
        self.set_readout_weights()

    def prepare_for_mixer_calibration(self, suppress, dc_if=None):
        """
        Sets up MWG and UHFQC settings for mixer calibration. It is up to the
        user to set the switch configuration to the correct position.
        Args:
            suppress: 'drive/readout LO/sideband'. Determines the readout
                      demodulation frequency.
        """
        if dc_if is None or 'drive' in suppress:
            dc_if=self.f_RO_mod()

        if self.cw_source is not None:
            self.cw_source.off()

        self.update_detector_functions()

        # drive LO
        if 'drive' in suppress:
            self.drive_LO.pulsemod_state('Off')
            self.drive_LO.frequency(self.f_qubit() - self.f_pulse_mod())
            self.drive_LO.power(self.drive_LO_pow())
            self.drive_LO.on()
        else:
            self.drive_LO.off()

        # drive offsets
        self.AWG.set(self.pulse_I_channel() + '_offset',
                     self.pulse_I_offset())
        self.AWG.set(self.pulse_Q_channel() + '_offset',
                     self.pulse_Q_offset())

        # readout frequency
        if suppress == 'drive LO':
            f_RO = self.drive_LO.frequency()
        elif suppress == 'drive sideband':
            f_RO = self.drive_LO.frequency() - self.f_pulse_mod()
        elif suppress == 'readout LO':
            f_RO = self.f_RO() - self.f_RO_mod()
        elif suppress == 'readout sideband':
            f_RO = self.f_RO() - 2*self.f_RO_mod()
        self.readout_DC_LO.frequency(f_RO - dc_if)
        self.readout_DC_LO.pulsemod_state('Off')
        self.readout_DC_LO.on()

        if 'readout' in suppress:
            self.readout_UC_LO.pulsemod_state('Off')
            self.readout_UC_LO.frequency(self.f_RO() - self.f_RO_mod())
            self.readout_UC_LO.on()

        # UHFQC settings
        if 'drive' in suppress:
            self.UHFQC.awg_sequence_acquisition()
            self.set_readout_weights('SSB')
        else:
            self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(
                f_RO_mod=self.f_RO_mod(), RO_amp=self.RO_amp(),
                RO_pulse_length=self.RO_pulse_length(),
                alpha=self.ro_alpha(), phi_skew=self.ro_phi_skew())
            self.set_readout_weights('SSB', f_mod=dc_if)

    def set_readout_weights(self, type=None, f_mod=None):
        if type is None:
            type = self.ro_acq_weight_type()
        if f_mod is None:
            f_mod = self.f_RO_mod()

        if type == 'manual':
            pass
        elif type == 'optimal':
            if (self.ro_acq_weight_func_I() is None or
                        self.ro_acq_weight_func_Q() is None):
                logging.warning('Optimal weights are None, not setting '
                                'integration weights')
            else:
                # When optimal weights are used, only the RO I weight
                # channel is used
                self.UHFQC.set('quex_wint_weights_{}_real'.format(
                               self.RO_acq_weight_function_I()),
                               self.ro_acq_weight_func_I().copy())
                self.UHFQC.set('quex_wint_weights_{}_imag'.format(
                               self.RO_acq_weight_function_I()),
                               self.ro_acq_weight_func_Q().copy())

                self.UHFQC.set('quex_rot_{}_real'.format(
                               self.RO_acq_weight_function_I()), 1.0)
                self.UHFQC.set('quex_rot_{}_imag'.format(
                               self.RO_acq_weight_function_I()), -1.0)
        else:
            tbase = np.arange(0, 4096 / 1.8e9, 1 / 1.8e9)
            theta = self.RO_IQ_angle()
            cosI = np.array(np.cos(2 * np.pi * f_mod * tbase + theta))
            sinI = np.array(np.sin(2 * np.pi * f_mod * tbase + theta))
            c1 = self.RO_acq_weight_function_I()
            c2 = self.RO_acq_weight_function_Q()
            if type == 'SSB':
                self.UHFQC.set('quex_wint_weights_{}_real'.format(c1), cosI)
                self.UHFQC.set('quex_rot_{}_real'.format(c1), 1)
                self.UHFQC.set('quex_wint_weights_{}_real'.format(c2), sinI)
                self.UHFQC.set('quex_rot_{}_real'.format(c2), 1)
                self.UHFQC.set('quex_wint_weights_{}_imag'.format(c1), sinI)
                self.UHFQC.set('quex_rot_{}_imag'.format(c1), 1)
                self.UHFQC.set('quex_wint_weights_{}_imag'.format(c2), cosI)
                self.UHFQC.set('quex_rot_{}_imag'.format(c2), -1)
            elif type == 'DSB':
                self.UHFQC.set('quex_wint_weights_{}_real'.format(c1), cosI)
                self.UHFQC.set('quex_rot_{}_real'.format(c1), 1)
                self.UHFQC.set('quex_wint_weights_{}_real'.format(c2), sinI)
                self.UHFQC.set('quex_rot_{}_real'.format(c2), 1)
                self.UHFQC.set('quex_rot_{}_imag'.format(c1), 0)
                self.UHFQC.set('quex_rot_{}_imag'.format(c2), 0)
            elif type == 'square_rot':
                self.UHFQC.set('quex_wint_weights_{}_real'.format(c1), cosI)
                self.UHFQC.set('quex_rot_{}_real'.format(c1), 1)
                self.UHFQC.set('quex_wint_weights_{}_imag'.format(c1), sinI)
                self.UHFQC.set('quex_rot_{}_imag'.format(c1), 1)

    def measure_resonator_spectroscopy(self, freqs=None, MC=None,
                                       analyze=True, close_fig=True):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for measure_resonator_"
                             "spectroscopy")
        if np.any(freqs<500e6):
            logging.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if MC is None:
            MC = self.MC

        previous_freq = self.heterodyne.frequency()

        self.prepare_for_continuous_wave()
        MC.set_sweep_function(pw.wrap_par_to_swf(
            self.heterodyne.frequency))
        MC.set_sweep_points(freqs)
        demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
            else 'double'
        MC.set_detector_function(det.Heterodyne_probe(
            self.heterodyne,
            trigger_separation=self.heterodyne.trigger_separation(),
            demod_mode=demod_mode))
        MC.run(name='resonator_scan'+self.msmt_suffix)

        self.heterodyne.frequency(previous_freq)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_resonator_spectroscopy_flux_sweep(self, freqs, voltages,
            flux_channel=None, MC=None, analyze=True, close_fig=True):
        if MC is None:
            MC == self.MC
        if flux_channel is None:
            flux_channel = self.dc_source_channel()
        self.prepare_for_continuous_wave()
        MC.set_sweep_function(self.heterodyne.frequency)
        MC.set_sweep_points(freqs)
        MC.set_sweep_function_2D(self.DC_source.parameters
                                 ['volt_' + flux_channel])
        MC.set_sweep_points_2D(voltages)
        demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
            else 'double'
        MC.set_detector_function(det.Heterodyne_probe(
            self.heterodyne,
            trigger_separation=self.heterodyne.trigger_separation(),
            demod_mode=demod_mode))
        MC.run_2D(name='resonator_spectoscopy_flux_sweep' + self.msmt_suffix)
        if analyze:
            ma.TwoD_Analysis(label='resonator_spectoscopy_flux_sweep',
                             close_fig=close_fig, close_file=True,
                             qb_name=self.name)
            # ma.MeasurementAnalysis(TwoD=True, close_fig=close_fig)

    def measure_qubit_spectroscopy_flux_sweep(self, freqs, voltages,
            flux_channel=None, pulsed=True, MC=None, analyze=True,
                                                  close_fig=True):
        if MC is None:
            MC = self.MC
        if flux_channel is None:
            flux_channel = self.dc_source_channel()
        if pulsed:
            self.prepare_for_pulsed_spec()
            spec_pars = self.get_spec_pars()
            RO_pars = self.get_RO_pars()
            self.cw_source.on()
            sq.Pulsed_spec_seq(spec_pars, RO_pars)
            MC.set_sweep_function(self.cw_source.frequency)
            MC.set_sweep_points(freqs)
            MC.set_sweep_function_2D(self.DC_source.parameters
                                     ['volt_' + flux_channel])
            MC.set_sweep_points_2D(voltages)
            demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
                else 'double'
            MC.set_detector_function(det.Heterodyne_probe(
                self.heterodyne,
                trigger_separation=self.heterodyne.trigger_separation(),
                demod_mode=demod_mode))
            self.AWG.start()
            MC.run_2D(name='qubit_spectroscopy_flux_sweep' + self.msmt_suffix)
            self.cw_source.off()
        else:
            raise NotImplementedError()
        if analyze:
            ma.TwoD_Analysis(label='qubit_spectroscopy_flux_sweep',
                             close_fig=close_fig, close_file=True,
                             qb_name=self.name)
            # ma.MeasurementAnalysis(TwoD=True, close_fig=close_fig)


    def measure_homodyne_acqusition_delay(self, delays=None, MC=None,
                                          analyze=True, close_fig=True):
        """
        Varies the delay between the homodyne modulation signal and
        acquisition. Measures the transmittance.
        """
        if delays is None:
            raise ValueError("Unspecified delays for measure_homodyne_"
                             "acquisition_delay")

        if MC is None:
            MC = self.MC

        # set number of averages to 1 due to a readout bug
        previous_nr_averages = self.heterodyne.nr_averages()
        self.heterodyne.nr_averages(1)
        previous_delay = self.heterodyne.acquisition_delay()

        self.prepare_for_continuous_wave()
        MC.set_sweep_function(pw.wrap_par_to_swf(
            self.heterodyne.acquisition_delay))
        MC.set_sweep_points(delays)
        demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
            else 'double'
        MC.set_detector_function(det.Heterodyne_probe(
            self.heterodyne,
            trigger_separation=self.heterodyne.trigger_separation(),
            demod_mode=demod_mode))
        MC.run(name='acquisition_delay_scan'+self.msmt_suffix)

        self.heterodyne.acquisition_delay(previous_delay)
        self.heterodyne.nr_averages(previous_nr_averages)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_spectroscopy(self, freqs=None, pulsed=False, MC=None,
                             analyze=True, close_fig=True,upload=True,
                             label=None):
        """ Varies qubit drive frequency and measures the resonator
        transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for measure_spectroscopy")
        if np.any(freqs<500e6):
            logging.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if MC is None:
            MC = self.MC

        if not pulsed:

            if label is None:
                label = 'spectroscopy'+self.msmt_suffix

            self.heterodyne.frequency(self.f_RO())
            self.prepare_for_continuous_wave()
            self.cw_source.on()

            MC.set_sweep_function(pw.wrap_par_to_swf(self.cw_source.frequency))
            MC.set_sweep_points(freqs)
            demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
                else 'double'
            MC.set_detector_function(det.Heterodyne_probe(
                self.heterodyne,
                trigger_separation=self.heterodyne.trigger_separation(),
                demod_mode=demod_mode))
            MC.run(name=label)

            self.cw_source.off()

        else:

            if label is None:
                label = 'pulsed_spec'+self.msmt_suffix

            self.prepare_for_pulsed_spec()

            spec_pars = self.get_spec_pars()
            RO_pars = self.get_RO_pars()

            self.cw_source.on()

            sq.Pulsed_spec_seq(spec_pars, RO_pars, upload=upload)

            self.AWG.start()

            MC.set_sweep_function(self.cw_source.frequency)
            MC.set_sweep_points(freqs)
            demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
                else 'double'
            MC.set_detector_function(det.Heterodyne_probe(
                self.heterodyne,
                trigger_separation=self.heterodyne.trigger_separation(),
                demod_mode=demod_mode))
            MC.run(name=label)

            self.cw_source.off()


        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_rabi(self, amps=None, MC=None, analyze=True,
                     close_fig=True, cal_points=True, no_cal_points=2,
                     upload=True, label=None,  n=1):

        """
        Varies the amplitude of the qubit drive pulse and measures the readout
        resonator transmission.

        Args:
            amps            the array of drive pulse amplitudes
            MC              the MeasurementControl object
            analyse         whether to create a (base) MeasurementAnalysis
                            object for this measurement; offers possibility to
                            manually analyse data using the classes in
                            measurement_analysis.py
            close_fig       whether or not to close the default analysis figure
            cal_points      whether or not to use calibration points
            no_cal_points   how many calibration points to use
            upload          whether or not to upload the sequence to the AWG
            label           the measurement label
            n               the number of times the drive pulses with the same
                            amplitude should be repeated in each measurement
        """

        if amps is None:
            raise ValueError("Unspecified amplitudes for measure_rabi")

        # Define the measurement label
        if label is None:
            label = 'Rabi-n{}'.format(n) + self.msmt_suffix

        # Prepare the physical instruments for a time domain measurement
        self.prepare_for_timedomain()

        # Define the MeasurementControl object for this measurement
        if MC is None:
            MC = self.MC

        if cal_points:
            step = np.abs(amps[-1]-amps[-2])
            if no_cal_points == 4:
                sweep_points = np.concatenate(
                    [amps, [amps[-1]+step, amps[-1]+2*step, amps[-1]+3*step,
                        amps[-1]+4*step]])
            elif no_cal_points == 2:
                sweep_points = np.concatenate(
                    [amps, [amps[-1]+step, amps[-1]+2*step]])
            else:
                sweep_points = amps
        else:
            sweep_points = amps

        # Specify the sweep function, the sweep points,
        # and the detector function, and run the measurement
        MC.set_sweep_function(awg_swf.Rabi(pulse_pars=self.get_drive_pars(),
                                           RO_pars=self.get_RO_pars(), n=n,
                                           cal_points=cal_points,
                                           no_cal_points=no_cal_points,
                                           upload=upload))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_rabi_2nd_exc(self, amps=None, n=1, MC=None, analyze=True,
                             label=None, last_ge_pulse=True,
                             close_fig=True, cal_points=True, no_cal_points=4,
                             upload=True):

        if amps is None:
            raise ValueError("Unspecified amplitudes for measure_rabi")

        if label is None:
            label = 'Rabi_2nd_exc-n{}'.format(n) + self.msmt_suffix

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        if self.cal_points:
            if cal_points:
                step = np.abs(amps[-1]-amps[-2])
                if no_cal_points == 6:
                    sweep_points = np.concatenate(
                        [amps, [amps[-1]+step, amps[-1]+2*step, amps[-1]+3*step,
                            amps[-1]+4*step, amps[-1]+5*step, amps[-1]+6*step]])
                elif no_cal_points == 4:
                    sweep_points = np.concatenate(
                        [amps, [amps[-1]+step, amps[-1]+2*step, amps[-1]+3*step,
                            amps[-1]+4*step]])
                elif no_cal_points == 2:
                    sweep_points = np.concatenate(
                        [amps, [amps[-1]+step, amps[-1]+2*step]])
                else:
                    sweep_points = amps
        else:
            sweep_points = amps

        MC.set_sweep_function(awg_swf.Rabi_2nd_exc(
                        pulse_pars=self.get_drive_pars(),
                        pulse_pars_2nd=self.get_ef_drive_pars(),
                        RO_pars=self.get_RO_pars(),
                        last_ge_pulse=last_ge_pulse,
                        n=n, upload=upload,
                        cal_points=cal_points, no_cal_points=no_cal_points))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_rabi_amp90(self, scales=np.linspace(0.3, 0.7, 31), n=1,
                           MC=None, analyze=True, close_fig=True, upload=True):

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Rabi_amp90(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(), n=n,
            upload=upload))
        MC.set_sweep_points(scales)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi_amp90_scales_n{}'.format(n)+self.msmt_suffix)


    def measure_T1(self, times=None, MC=None, analyze=True, upload=True,
                       close_fig=True, cal_points=True, label=None):

        if times is None:
            raise ValueError("Unspecified times for measure_T1")
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        # Define the measurement label
        if label is None:
            label = 'T1' + self.msmt_suffix

        if cal_points:
            step = np.abs(times[-1]-times[-2])
            sweep_points = np.concatenate(
                [times, [times[-1]+step,  times[-1]+2*step,
                    times[-1]+3*step, times[-1]+4*step]])
        else:
            sweep_points = times

        MC.set_sweep_function(awg_swf.T1(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            upload=upload, cal_points=cal_points))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_T1_2nd_exc(self, times=None, MC=None, analyze=True, upload=True,
                           close_fig=True, cal_points=True, no_cal_points=6,
                           label=None, last_ge_pulse=True):

        if times is None:
            raise ValueError("Unspecified times for measure_T1_2nd_exc")
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()

        if label is None:
            label = 'T1_2nd'+self.msmt_suffix

        if MC is None:
            MC = self.MC

        if cal_points:
            step = np.abs(times[-1]-times[-2])
            if no_cal_points == 6:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step,
                                 times[-1]+3*step, times[-1]+4*step,
                                 times[-1]+5*step, times[-1]+6*step]])
            elif no_cal_points == 4:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step,
                                 times[-1]+3*step, times[-1]+4*step]])
            elif no_cal_points == 2:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step]])
            else:
                sweep_points = times
        else:
            sweep_points = times

        MC.set_sweep_function(awg_swf.T1_2nd_exc(
                                pulse_pars=self.get_drive_pars(),
                                pulse_pars_2nd=self.get_ef_drive_pars(),
                                RO_pars=self.get_RO_pars(),
                                upload=upload,
                                cal_points=cal_points,
                                no_cal_points=no_cal_points,
                                last_ge_pulse=last_ge_pulse))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)


    def measure_qscale(self, qscales=None, MC=None, analyze=True, upload=True,
                       close_fig=True, label=None, cal_points=True):

        if qscales is None:
            raise ValueError("Unspecified qscale values for measure_qscale")
        uniques = np.unique(qscales[range(3)])
        if uniques.size>1:
            raise ValueError("The values in the qscales array are not repeated "
                             "3 times.")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        if label is None:
            label = 'QScale'+self.msmt_suffix

        if cal_points:
            step = np.abs(qscales[-1] - qscales[-4])
            sweep_points = np.concatenate(
                [qscales, [qscales[-1] + step, qscales[-1] + 2*step,
                    qscales[-1] + 3*step, qscales[-1] + 4*step]])
        else:
            sweep_points = qscales

        MC.set_sweep_function(awg_swf.QScale(
                pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
                upload=upload, cal_points=cal_points))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_qscale_2nd_exc(self, qscales=None, MC=None, analyze=True,
                               upload=True, close_fig=True, label=None,
                               cal_points=True, no_cal_points=6,
                               last_ge_pulse=True):

        if qscales is None:
            raise ValueError("Unspecified qscale values for"
                             " measure_qscale_2nd_exc")
        uniques = np.unique(qscales[range(3)])
        if uniques.size>1:
            raise ValueError("The values in the qscales array are not repeated "
                             "3 times.")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        if label is None:
            label = 'QScale_2nd_exc'+self.msmt_suffix

        if cal_points:
            step = np.abs(qscales[-1] - qscales[-4])
            if no_cal_points == 6:
                sweep_points = np.concatenate(
                    [qscales, [qscales[-1] + step, qscales[-1] + 2*step,
                               qscales[-1] + 3*step, qscales[-1] + 4*step,
                               qscales[-1] + 5*step, qscales[-1] + 6*step]])
            elif no_cal_points == 4:
                sweep_points = np.concatenate(
                    [qscales, [qscales[-1] + step, qscales[-1] + 2*step,
                               qscales[-1] + 3*step, qscales[-1] + 4*step]])
            elif no_cal_points == 2:
                sweep_points = np.concatenate(
                    [qscales, [qscales[-1] + step, qscales[-1] + 2*step]])
            else:
                sweep_points = qscales
        else:
            sweep_points = qscales

        MC.set_sweep_function(awg_swf.QScale_2nd_exc(
            qscales=sweep_points,
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            upload=upload, cal_points=cal_points, no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_ramsey_multiple_detunings(self, times=None,
                                          artificial_detunings=None, label='',
                                          MC=None, analyze=True, close_fig=True,
                                          cal_points=True, upload=True):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detunings is None:
            logging.warning('Artificial detuning is 0.')
        uniques = np.unique(times[range(len(artificial_detunings))])
        if uniques.size>1:
            raise ValueError("The values in the times array are not repeated "
                             "len(artificial_detunings) times.")
        if np.any(np.asarray(np.abs(artificial_detunings))<1e3):
            logging.warning('The artificial detuning is too small. The units '
                            'should be Hz.')
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # Define the measurement label
        if label == '':
            label = 'Ramsey_mult_det' + self.msmt_suffix

        if cal_points:
            len_art_det = len(artificial_detunings)
            step = np.abs(times[-1] - times[-len_art_det-1])
            sweep_points = np.concatenate(
                [times, [times[-1] + step, times[-1] + 2*step,
                    times[-1] + 3*step, times[-1] + 4*step]])
        else:
            sweep_points = times

        Rams_swf = awg_swf.Ramsey_multiple_detunings(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            artificial_detunings=artificial_detunings, cal_points=cal_points,
            upload=upload)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)


    def measure_ramsey(self, times=None, artificial_detuning=0, label='',
                       MC=None, analyze=True, close_fig=True, cal_points=True,
                       upload=True):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detuning is None:
            logging.warning('Artificial detuning is 0.')
        if np.abs(artificial_detuning) < 1e3:
            logging.warning('The artificial detuning is too small. The units'
                            'should be Hz.')
        if np.any(times > 1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # Define the measurement label
        if label == '':
            label = 'Ramsey' + self.msmt_suffix

        if cal_points:
            step = np.abs(times[-1]-times[-2])
            sweep_points = np.concatenate(
                [times, [times[-1]+step,  times[-1]+2*step,
                             times[-1]+3*step, times[-1]+4*step]])
        else:
            sweep_points = times

        Rams_swf = awg_swf.Ramsey(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            artificial_detuning=artificial_detuning, cal_points=cal_points,
            upload=upload)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_ramsey_2nd_exc(self, times=None, artificial_detuning=0, label=None,
                       MC=None, analyze=True, close_fig=True, cal_points=True,
                       n=1, upload=True, last_ge_pulse=True, no_cal_points=6):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detuning is None:
            logging.warning('Artificial detuning is 0.')
        if np.abs(artificial_detuning)<1e3:
            logging.warning('The artificial detuning is too small. The units'
                            'should be Hz.')
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        if label is None:
            label = 'Ramsey_2nd'+self.msmt_suffix

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        if cal_points:
            step = np.abs(times[-1]-times[-2])
            if no_cal_points == 6:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step,
                                 times[-1]+3*step, times[-1]+4*step,
                                 times[-1]+4*step, times[-1]+6*step]])
            elif no_cal_points == 4:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step,
                                 times[-1]+3*step, times[-1]+4*step]])
            elif no_cal_points == 2:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step]])
            else:
                sweep_points = times
        else:
            sweep_points = times

        Rams_2nd_swf = awg_swf.Ramsey_2nd_exc(
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            artificial_detuning=artificial_detuning,
            cal_points=cal_points, n=n, upload=upload,
            no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse)
        MC.set_sweep_function(Rams_2nd_swf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_ramsey_2nd_exc_multiple_detunings(self, times=None,
                               artificial_detunings=None, label=None,
                               MC=None, analyze=True, close_fig=True,
                               cal_points=True, n=1, upload=True,
                               last_ge_pulse=True, no_cal_points=6):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detunings is None:
            logging.warning('Artificial detunings were not given.')
        if np.any(np.asarray(np.abs(artificial_detunings))<1e3):
            logging.warning('The artificial detuning is too small. The units '
                            'should be Hz.')
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        if label is None:
            label = 'Ramsey_mult_det_2nd'+self.msmt_suffix

        if cal_points:
            len_art_det = len(artificial_detunings)
            step = np.abs(times[-1] - times[-len_art_det-1])
            if no_cal_points == 6:
                sweep_points = np.concatenate(
                    [times, [times[-1] + step, times[-1] + 2*step,
                             times[-1] + 3*step, times[-1] + 4*step,
                             times[-1] + 5*step, times[-1] + 6*step]])
            elif no_cal_points == 4:
                sweep_points = np.concatenate(
                    [times, [times[-1] + step, times[-1] + 2*step,
                             times[-1] + 3*step, times[-1] + 4*step]])
            elif no_cal_points == 2:
                sweep_points = np.concatenate(
                    [times, [times[-1] + step, times[-1] + 2*step]])
            else:
                sweep_points = times
        else:
            sweep_points = times

        Rams_2nd_swf = awg_swf.Ramsey_2nd_exc_multiple_detunings(
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            artificial_detunings=artificial_detunings,
            cal_points=cal_points, n=n, upload=upload,
            no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse)
        MC.set_sweep_function(Rams_2nd_swf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)


    def measure_echo(self, times=None, MC=None, artificial_detuning=None,
                     upload=True, analyze=True, close_fig=True):

        if times is None:
            raise ValueError("Unspecified times for measure_echo")

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        Echo_swf = awg_swf.Echo(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            artificial_detuning=artificial_detuning, upload=upload)
        MC.set_sweep_function(Echo_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Echo'+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_allxy(self, double_points=True, MC=None, upload=True,
                      analyze=True, close_fig=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.AllXY(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            double_points=double_points, upload=upload))
        MC.set_detector_function(self.int_avg_det)
        MC.run('AllXY'+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_randomized_benchmarking(self, nr_cliffords=None, nr_seeds=50,
                                        thresholded=False, RO_pars=None,
                                        MC=None, close_fig=True,
                                        upload=False, analyze=True,
                                        gate_decomp='HZ', label=None,
                                        cal_points=True,
                                        interleaved_gate=None,
                                        det_func=None):
        '''
        Performs a randomized benchmarking experiment on 1 qubit.
        type(nr_cliffords) == array
        type(nr_seeds) == int
        '''

        if nr_cliffords is None:
            raise ValueError("Unspecified nr_cliffords for measure_echo")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        if RO_pars is None:
            RO_pars = self.get_RO_pars()

        if label is None:
            if interleaved_gate is None:
                label = 'RB_{}_{}_seeds_{}_cliffords'.format(
                    gate_decomp, nr_seeds, nr_cliffords[-1]) + self.msmt_suffix
            else:
                label = 'IRB_{}_{}_{}_seeds_{}_cliffords'.format(
                    interleaved_gate, gate_decomp,
                    nr_seeds, nr_cliffords[-1]) \
                        + self.msmt_suffix

        if thresholded:
            if int(self.UHFQC.get('quex_thres_{}_level'.format(
                    self.RO_acq_weight_function_I()))) == 0:
                raise ValueError('The threshold value is not set.')

        nr_seeds_arr = np.arange(nr_seeds)
        if cal_points:
            step = np.abs(nr_seeds_arr [-1] - nr_seeds_arr [-2])
            sweep_points1D = np.concatenate(
                [nr_seeds_arr,
                 [nr_seeds_arr[-1]+step, nr_seeds_arr[-1]+2*step,
                  nr_seeds_arr[-1]+3*step, nr_seeds_arr[-1]+4*step]])
        else:
            sweep_points1D = nr_seeds_arr

        RB_sweepfunction = awg_swf.Randomized_Benchmarking_one_length(
            pulse_pars=self.get_drive_pars(), RO_pars=RO_pars,
            cal_points=cal_points, gate_decomposition=gate_decomp,
            nr_cliffords_value=nr_cliffords[0], upload=upload,
            interleaved_gate=interleaved_gate)

        RB_sweepfunction_2D = awg_swf.Randomized_Benchmarking_nr_cliffords(
            RB_sweepfunction=RB_sweepfunction)

        MC.set_sweep_function(RB_sweepfunction)
        MC.set_sweep_points(sweep_points1D)
        MC.set_sweep_function_2D(RB_sweepfunction_2D)
        MC.set_sweep_points_2D(nr_cliffords)
        if det_func is None:
            if thresholded:
                MC.set_detector_function(self.dig_avg_det)
            else:
                MC.set_detector_function(self.int_avg_det)
        else:
            MC.set_detector_function(det_func)

        MC.run(label, mode='2D')

        if analyze:
            # ma.TwoD_Analysis(label=label,
            #                  close_fig=close_fig,
            #                  qb_name=self.name,)
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name, TwoD=True)
        return MC

    def measure_transients(self, MC=None, cases=('off', 'on'), upload=True,
                           analyze=True, **kw):
        """
        If the resulting transients will be used to caclulate the optimal
        weight functions, then it is important that the UHFQC iavg_delay and
        wint_delay are calibrated such that the weights and traces are
        aligned: iavg_delay = 2*wint_delay.
        """
        if MC is None:
            MC = self.MC

        self.prepare_for_timedomain()
        npoints = self.inp_avg_det.nr_samples
        if 'off' in cases:
            MC.set_sweep_function(awg_swf.OffOn(
                pulse_pars=self.get_drive_pars(),
                RO_pars=self.get_RO_pars(),
                pulse_comb='OffOff',
                upload=upload))
            MC.set_sweep_points(np.linspace(0, npoints/1.8e9, npoints,
                                            endpoint=False))
            MC.set_detector_function(self.inp_avg_det)
            MC.run(name='timetrace_off' + self.msmt_suffix)
            if analyze:
                ma.MeasurementAnalysis(auto=True, qb_name=self.name, **kw)

        if 'on' in cases:
            MC.set_sweep_function(awg_swf.OffOn(
                pulse_pars=self.get_drive_pars(),
                RO_pars=self.get_RO_pars(),
                pulse_comb='OnOn',
                upload=upload))
            MC.set_sweep_points(np.linspace(0, npoints/1.8e9, npoints,
                                            endpoint=False))
            MC.set_detector_function(self.inp_avg_det)
            MC.run(name='timetrace_on' + self.msmt_suffix)
            if analyze:
                ma.MeasurementAnalysis(auto=True, qb_name=self.name, **kw)

    def measure_readout_pulse_scope(self, delays, freqs, RO_separation=None,
                                    comm_freq=225e6, analyze=True, label=None,
                                    close_fig=True, upload=True, verbose=False,
                                    cal_points=((-4, -3), (-2, -1)), MC=None):
        """
        From the documentation of the used sequence function:

        Prepares the AWGs for a readout pulse shape and timing measurement.

        The sequence consists of two readout pulses where the drive pulse start
        time is swept through the first readout pulse. Because the photons in
        the readout resonator induce an ac-Stark shift of the qubit frequency,
        we can determine the readout pulse shape by sweeping the drive frequency
        in an outer loop to determine the qubit frequency.

        Important: This sequence includes two readouts per segment. For this
        reason the calibration points are also duplicated.

        Args:
            delays: A list of delays between the start of the first readout pulse
                    and the center of the drive pulse.
            RO_separation: Separation between the starts of the two readout pulses.
                           If the comm_freq parameter is not None, the used value
                           is increased to satisfy the commensurability constraint.
            cal_points: True for default calibration points, False for no
                        calibration points or a list of two lists, containing
                        the indices of the calibration segments for the ground
                        and excited state.
            comm_freq: The readout pulse separation will be a multiple of
                       1/comm_freq
        """

        if delays is None:
            raise ValueError("Unspecified amplitudes for "
                             "measure_readout_pulse_scope")
        if label is None:
            label = 'Readout_pulse_scope' + self.msmt_suffix
        if MC is None:
            MC = self.MC
        if freqs is None:
            freqs = self.f_qubit() + np.linspace(-50e6, 50e6, 201)
        if RO_separation is None:
            RO_separation = np.abs(self.RO_acq_marker_delay())
            RO_separation += 2 * self.RO_pulse_length()
            RO_separation += np.max(delays)
            RO_separation += 200e-9  # for slack

        self.prepare_for_timedomain()
        MC.set_sweep_function(awg_swf.Readout_pulse_scope_swf(
            delays=delays,
            pulse_pars=self.get_drive_pars(),
            RO_pars=self.get_RO_pars(),
            RO_separation=RO_separation,
            cal_points=cal_points,
            comm_freq=comm_freq,
            verbose=verbose,
            upload=upload))
        MC.set_sweep_points(delays)
        MC.set_sweep_function_2D(swf.Offset_Sweep(
            mc_parameter_wrapper.wrap_par_to_swf(self.drive_LO.frequency),
            -self.f_pulse_mod(),
            parameter_name=self.name + ' drive frequency'))
        MC.set_sweep_points_2D(freqs)

        d = det.UHFQC_integrated_average_detector(
            self.UHFQC, self.AWG, nr_averages=self.RO_acq_averages(),
            channels=self.int_avg_det.channels,
            integration_length=self.RO_acq_integration_length(),
            values_per_point=2, values_per_point_suffex=['_probe', '_measure'])
        MC.set_detector_function(d)
        MC.run(label)

        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            ma.MeasurementAnalysis(TwoD=True, auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def calibrate_drive_mixer_carrier(self, MC=None, update=True, x0=(0., 0.),
                                      initial_stepsize=0.01, trigger_sep=5e-6):
        if MC is None:
            MC = self.MC
        self.prepare_for_mixer_calibration(suppress='drive LO')
        cal_elts.mixer_calibration_sequence(
            trigger_sep, 0, self.RO_acq_marker_channel(),
            self.pulse_I_channel(), self.pulse_Q_channel())
        detector = self.int_avg_det_spec
        ad_func_pars = {'adaptive_function': opti.nelder_mead,
                        'x0': x0,
                        'initial_step': [initial_stepsize, initial_stepsize],
                        'no_improv_break': 15,
                        'minimize': True,
                        'maxiter': 500}
        chI_par = self.AWG.parameters['{}_offset'.format(
            self.pulse_I_channel())]
        chQ_par = self.AWG.parameters['{}_offset'.format(
            self.pulse_Q_channel())]
        MC.set_sweep_functions([chI_par, chQ_par])
        MC.set_detector_function(det.IndexDetector(detector, 0))
        MC.set_adaptive_function_parameters(ad_func_pars)
        self.AWG.start()
        MC.run(name='drive_carrier_calibration' + self.msmt_suffix,
               mode='adaptive')
        a = ma.OptimizationAnalysis(label='drive_carrier_calibration')
        # v2 creates a pretty picture of the optimizations
        ma.OptimizationAnalysis_v2(label='drive_carrier_calibration')

        ch_1_min = a.optimization_result[0][0]
        ch_2_min = a.optimization_result[0][1]
        if update:
            self.pulse_I_offset(ch_1_min)
            self.pulse_Q_offset(ch_2_min)
        return ch_1_min, ch_2_min

    def calibrate_drive_mixer_skewness(self, MC=None, update=True,
                                       amplitude=0.1, trigger_sep=5e-6,
                                       initial_stepsize=None):
        if initial_stepsize is None:
            initial_stepsize = [0.15, 10]
        if MC is None:
            MC = self.MC
        self.prepare_for_mixer_calibration(suppress='drive sideband')
        detector = det.UHFQC_mixer_skewness_det(
            self.UHFQC, qc.station, [self.RO_acq_weight_function_I(),
                                     self.RO_acq_weight_function_Q()],
            self.pulse_I_channel(), self.pulse_Q_channel(),
            self.alpha, self.phi_skew, self.f_pulse_mod(),
            self.RO_acq_marker_channel(),
            amplitude=amplitude, nr_averages=self.RO_acq_averages(),
            RO_trigger_separation=trigger_sep, verbose=False)
        # detector = det.UHFQC_readout_mixer_skewness_det(
        #     self.UHFQC, qc.station, [self.RO_acq_weight_function_I(),
        #                              self.RO_acq_weight_function_Q()],
        #     self.ro_alpha, self.ro_phi_skew, self.f_RO_mod(),
        #     self.RO_acq_marker_channel(),
        #     nr_averages=self.RO_acq_averages(),
        #     RO_pulse_length=self.RO_pulse_length(),
        #     verbose=False)
        ad_func_pars = {'adaptive_function': opti.nelder_mead,
                        'x0': [self.alpha(), self.phi_skew()],
                        'initial_step': initial_stepsize,
                        'no_improv_break': 12,
                        'minimize': True,
                        'maxiter': 500}
        MC.set_sweep_functions([self.alpha, self.phi_skew])
        MC.set_detector_function(det.IndexDetector(detector, 0))
        MC.set_adaptive_function_parameters(ad_func_pars)
        MC.run(name='drive_skewness_calibration' + self.msmt_suffix,
               mode='adaptive')
        a = ma.OptimizationAnalysis(label='drive_skewness_calibration')
        # v2 creates a pretty picture of the optimizations
        ma.OptimizationAnalysis_v2(label='drive_skewness_calibration')

        # phi and alpha are the coefficients that go in the predistortion matrix
        alpha = a.optimization_result[0][0]
        phi = a.optimization_result[0][1]
        if update:
            self.alpha(alpha)
            self.phi_skew(phi)
        return alpha, phi

    def calibrate_readout_mixer_skewness(self, dc_if=50e6, MC=None, update=True,
                                         trigger_sep=5e-6,
                                         initial_stepsize=(0.15, 10)):
        initial_stepsize = list(initial_stepsize)
        if MC is None:
            MC = self.MC
        self.prepare_for_mixer_calibration('readout sideband', dc_if=dc_if)
        detector = det.UHFQC_readout_mixer_skewness_det(
            self.UHFQC, None, [self.RO_acq_weight_function_I(),
                                   self.RO_acq_weight_function_Q()],
            self.ro_alpha, self.ro_phi_skew, self.f_RO_mod(), self.RO_amp(),
            self.RO_pulse_length(), self.RO_acq_averages(), verbose=False)
        ad_func_pars = {'adaptive_function': opti.nelder_mead,
                        'x0': [self.ro_alpha(), self.ro_phi_skew()],
                        'initial_step': initial_stepsize,
                        'no_improv_break': 12,
                        'minimize': True,
                        'maxiter': 500}
        MC.set_sweep_functions([self.ro_alpha, self.ro_phi_skew])
        MC.set_detector_function(det.IndexDetector(detector, 0))
        MC.set_adaptive_function_parameters(ad_func_pars)
        cal_elts.mixer_calibration_sequence(
            trigger_sep, 0, self.RO_acq_marker_channel())
        self.AWG.start()
        MC.run(name='readout_skewness_calibration' + self.msmt_suffix,
               mode='adaptive')
        self.AWG.stop()
        a = ma.OptimizationAnalysis(label='readout_skewness_calibration')
        # v2 creates a pretty picture of the optimizations
        ma.OptimizationAnalysis_v2(label='readout_skewness_calibration')

        # phi and alpha are the coefficients that go in the predistortion matrix
        ro_alpha = a.optimization_result[0][0]
        ro_phi = a.optimization_result[0][1]
        if update:
            self.ro_alpha(ro_alpha)
            self.ro_phi_skew(ro_phi)
        return ro_alpha, ro_phi

    def measure_drive_mixer_spectrum(self, if_freqs, MC=None, amplitude=0.1,
                                     trigger_sep=5e-6):
        if MC is None:
            MC = self.MC
        self.prepare_for_mixer_calibration('drive LO')
        cal_elts.mixer_calibration_sequence(
            trigger_sep, amplitude, self.RO_acq_marker_channel(),
            self.pulse_I_channel(), self.pulse_Q_channel(), self.f_pulse_mod(),
            self.phi_skew(), self.alpha()
        )
        DC_LO_freqs = (if_freqs*trigger_sep).astype(np.int)/trigger_sep + \
                      self.drive_LO.frequency() - self.f_RO_mod()
        MC.set_sweep_function(self.readout_DC_LO.frequency)
        MC.set_sweep_points(DC_LO_freqs)
        MC.set_detector_function(self.int_avg_det_spec)
        self.AWG.start()
        MC.run('mw_uc_spectrum' + self.msmt_suffix)
        ma.MeasurementAnalysis(plot_args=dict(log=True, marker=''))

    def measure_readout_mixer_spectrum(self, if_freqs, dc_if=50e6, MC=None,
                                       amplitude=0.1, trigger_sep=5e-6):
        if MC is None:
            MC = self.MC
        self.prepare_for_mixer_calibration('readout LO', dc_if=dc_if)
        cal_elts.mixer_calibration_sequence(
            trigger_sep, 0, self.RO_acq_marker_channel())
        DC_LO_freqs = if_freqs + self.readout_UC_LO.frequency() - dc_if
        MC.set_sweep_function(self.readout_DC_LO.frequency)
        MC.set_sweep_points(DC_LO_freqs)
        MC.set_detector_function(self.int_avg_det_spec)
        self.AWG.start()
        MC.run('ro_uc_spectrum' + self.msmt_suffix)
        ma.MeasurementAnalysis(plot_args=dict(log=True, marker=''))

    def find_optimized_weights(self, MC=None, update=True, measure=True, **kw):
        # FIXME: Make a proper analysis class for this (Ants, 04.12.2017)
        if measure:
            self.measure_transients(MC, analyze=False, **kw)
        MAon = ma.MeasurementAnalysis(label='timetrace_on')
        MAoff = ma.MeasurementAnalysis(label='timetrace_off')
        don = MAon.measured_values[0] + 1j * MAon.measured_values[1]
        doff = MAoff.measured_values[0] + 1j * MAoff.measured_values[1]
        if update:
            wre = np.real(don - doff)
            wim = np.imag(don - doff)
            k = max(np.max(np.abs(wre)), np.max(np.abs(wim)))
            wre /= k
            wim /= k
            self.ro_acq_weight_func_I(wre)
            self.ro_acq_weight_func_Q(wim)
        if kw.get('plot', True):
            npoints = len(MAon.sweep_points)
            tbase = np.linspace(0, npoints/1.8e9, npoints, endpoint=False)
            modulation = np.exp(2j * np.pi * self.f_RO_mod() * tbase)
            plt.subplot(311)
            plt.plot(tbase / 1e-9, np.real(don * modulation), '-', label='I')
            plt.plot(tbase / 1e-9, np.imag(don * modulation), '-', label='Q')
            plt.ylabel('d.c. voltage,\npi pulse (V)')
            plt.xlim(0, kw.get('tmax', 300))
            plt.legend()
            plt.subplot(312)
            plt.plot(tbase / 1e-9, np.real(doff * modulation), '-', label='I')
            plt.plot(tbase / 1e-9, np.imag(doff * modulation), '-', label='Q')
            plt.ylabel('d.c. voltage,\nno pi pulse (V)')
            plt.xlim(0, kw.get('tmax', 300))
            plt.legend()
            plt.subplot(313)
            plt.plot(tbase / 1e-9, np.real((don - doff) * modulation), '-',
                     label='I')
            plt.plot(tbase / 1e-9, np.imag((don - doff) * modulation), '-',
                     label='Q')
            plt.ylabel('d.c. voltage\ndifference (V)')
            plt.xlim(0, kw.get('tmax', 300))
            plt.legend()
            plt.xlabel('Time (ns)')
            MAoff.save_fig(plt.gcf(), 'timetraces', xlabel='time',
                           ylabel='voltage')
            plt.close()

    def find_ssro_fidelity(self, nreps=1, MC=None, analyze=True, close_fig=True,
                           no_fits=False, upload=True, preselection_pulse=True,
                           thresholded=False, RO_comm=1/225e6):
        """
        Conduct an off-on measurement on the qubit recording single-shot
        results and determine the single shot readout fidelity.

        Calculates the assignment fidelity `F_a` which is the average
        probability of correctly guessing the state that was prepared. If
        `no_fits` is `False` also finds the discrimination fidelity F_d, that
        takes into account the probability of an bit flip after state
        preparation, by fitting double gaussians to both |0> prepared and |1>
        prepared datasets.

        Args:
            reps: Number of repetitions. If greater than 1, a 2D sweep will be
                  made with the second sweep function a NoneSweep with number of
                  sweep points equal to reps. Default 1.
            MC: MeasurementControl object to use for the measurement. Defaults
                to `self.MC`.
            analyze: Boolean flag, whether to analyse the measurement results.
                     Default `True`.
            close_fig: Boolean flag to close the matplotlib's figure. If
                       `False`, then the plots can be viewed with `plt.show()`
                       Default `True`.
            no_fits: Boolean flag to disable finding the discrimination
                     fidelity. Default `False`.
            preselection_pulse: Whether to do an additional readout pulse
                                before state preparation. Default `True`.
        Returns:
            If `no_fits` is `False` returns assigment fidelity, discrimination
            fidelity and SNR = 2 |mu00 - mu11| / (sigma00 + sigma11). Else
            returns just assignment fidelity.
        """

        if MC is None:
            MC = self.MC

        label = 'SSRO_fidelity'
        prev_shots = self.RO_acq_shots()
        if preselection_pulse:
            self.RO_acq_shots(4*(self.RO_acq_shots()//4))
        else:
            self.RO_acq_shots(2*(self.RO_acq_shots()//2))

        self.prepare_for_timedomain()

        RO_spacing = self.UHFQC.quex_wint_delay()*2/1.8e9
        RO_spacing += self.RO_acq_integration_length()
        RO_spacing += 10e-9 # for slack
        RO_spacing -= self.gauss_sigma()*self.nr_sigma()
        RO_spacing -= self.RO_pulse_delay()
        RO_spacing -= self.pulse_delay()
        RO_spacing = max(0, RO_spacing)
        delta_comm = RO_spacing + self.gauss_sigma()*self.nr_sigma() + \
                     self.RO_pulse_delay() + self.pulse_delay() + \
                     self.RO_pulse_length()
        delta_comm %= RO_comm
        RO_spacing += RO_comm - delta_comm

        MC.set_sweep_function(awg_swf2.n_qubit_off_on(
            pulse_pars_list=[self.get_drive_pars()],
            RO_pars=self.get_RO_pars(),
            upload=upload,
            preselection=preselection_pulse,
            RO_spacing=RO_spacing))
        spoints = np.arange(self.RO_acq_shots())
        if preselection_pulse:
            spoints //= 2
        MC.set_sweep_points(np.arange(self.RO_acq_shots()))
        if thresholded:
            MC.set_detector_function(self.dig_log_det)
        else:
            MC.set_detector_function(self.int_log_det)
        prev_avg = MC.soft_avg()
        MC.soft_avg(1)

        mode = '1D'
        if nreps > 1:
            label += '_nreps{}'.format(nreps)
            MC.set_sweep_function_2D(swf.None_Sweep())
            MC.set_sweep_points_2D(np.arange(nreps))
            mode = '2D'

        MC.run(name=label+self.msmt_suffix, mode=mode)

        MC.soft_avg(prev_avg)
        self.RO_acq_shots(prev_shots)

        if analyze:
            rotate = self.ro_acq_weight_type() in {'SSB', 'DSB'}
            if thresholded:
                channels = self.dig_log_det.value_names
            else:
                channels = self.int_log_det.value_names
            if preselection_pulse:
                nr_samples = 4
                sample_0 = 0
                sample_1 = 2
            else:
                nr_samples = 2
                sample_0 = 0
                sample_1 = 1
            ana = ma.SSRO_Analysis(auto=True, close_fig=close_fig,
                                   rotate=rotate, no_fits=no_fits,
                                   channels=channels, nr_samples=nr_samples,
                                   sample_0=sample_0, sample_1=sample_1,
                                   preselection=preselection_pulse)
            if not no_fits:
                return ana.F_a, ana.F_d, ana.SNR
            else:
                return ana.F_a

    def find_readout_angle(self, MC=None, upload=True, close_fig=True, update=True, nreps=10):
        """
        Finds the optimal angle on the IQ plane for readout (optimal phase for
        the boxcar integration weights)
        If the Q wint channel is set to `None`, sets it to the next channel
        after I.

        Args:
            MC: MeasurementControl object to use. Default `None`.
            upload: Whether to update the AWG sequence. Default `True`.
            close_fig: Wheter to close the figures in measurement analysis.
                       Default `True`.
            update: Whether to update the integration weights and the  Default `True`.
            nreps: Default 10.
        """
        if MC is None:
            MC = self.MC

        label = 'RO_theta'
        if self.RO_acq_weight_function_Q() is None:
            self.RO_acq_weight_function_Q(
                (self.RO_acq_weight_function_I() + 1) % 9)
        self.set_readout_weights(type='SSB')
        prev_shots = self.RO_acq_shots()
        self.RO_acq_shots(2*(self.RO_acq_shots()//2))
        self.prepare_for_timedomain()
        MC.set_sweep_function(awg_swf.OffOn(
            pulse_pars=self.get_drive_pars(),
            RO_pars=self.get_RO_pars(),
            upload=upload,
            preselection=False))
        spoints = np.arange(self.RO_acq_shots())
        MC.set_sweep_points(np.arange(self.RO_acq_shots()))
        MC.set_detector_function(self.int_log_det)
        prev_avg = MC.soft_avg()
        MC.soft_avg(1)

        mode = '1D'
        if nreps > 1:
            MC.set_sweep_function_2D(swf.None_Sweep())
            MC.set_sweep_points_2D(np.arange(nreps))
            mode = '2D'

        MC.run(name=label+self.msmt_suffix, mode=mode)

        MC.soft_avg(prev_avg)
        self.RO_acq_shots(prev_shots)

        rotate = self.RO_acq_weight_function_Q() is not None
        channels = self.int_log_det.value_names
        ana = ma.SSRO_Analysis(auto=True, close_fig=close_fig,
                               rotate=rotate, no_fits=True,
                               channels=channels,
                               preselection=False)
        if update:
            self.RO_IQ_angle(self.RO_IQ_angle() + ana.theta)
        return ana.theta

    def measure_dynamic_phase(self,
                              flux_pulse_length=None, flux_pulse_amp=None,
                              flux_pulse_delay=None, flux_pulse_channel=None,
                              thetas=None, X90_separation=None,
                              MC=None, label=None, analyze=True):

        if flux_pulse_amp is None:
            raise ValueError('Unspecified flux_pulse_amp.')
        if flux_pulse_length is None:
            raise ValueError('Unspecified flux_pulse_length.')
        if thetas is None:
            raise ValueError('Unspecified thetas array.')

        if MC is None:
            MC = self.MC

        if flux_pulse_channel is not None:
            flux_pulse_channel_backup = self.flux_pulse_channel()
            self.flux_pulse_channel(flux_pulse_channel)
        if flux_pulse_delay is not None:
            flux_pulse_delay_backup = self.flux_pulse_delay()
            self.flux_pulse_delay(flux_pulse_delay)

        if label is None:
            label = 'Dynamic_phase_measurement_{}_{}_filter'.format(
                self.name, self.flux_pulse_channel())

        self.prepare_for_timedomain()

        flux_pulse_amp_backup = self.flux_pulse_amp()
        flux_pulse_length_backup = self.flux_pulse_length()
        self.flux_pulse_length(flux_pulse_length)

        if X90_separation is None:
            X90_separation = 2*self.flux_pulse_delay() + self.flux_pulse_length()

        ampls = np.array([0, flux_pulse_amp])

        s1 = awg_swf.Ramsey_interleaved_fluxpulse_sweep(
            self, X90_separation=X90_separation,
            upload=False)

        s2 = awg_swf.Ramsey_fluxpulse_ampl_sweep(self, s1)

        MC.soft_avg(1)
        MC.set_sweep_function(s1)
        MC.set_sweep_points(thetas)
        MC.set_sweep_function_2D(s2)
        MC.set_sweep_points_2D(ampls)
        MC.set_detector_function(self.int_avg_det)
        MC.run_2D(name=label)

        if analyze:
            MA = ma.TwoD_Analysis(label=label, qb_name=self.name)

        self.flux_pulse_length(flux_pulse_length_backup)
        self.flux_pulse_amp(flux_pulse_amp_backup)
        if flux_pulse_channel is not None:
            self.flux_pulse_channel(flux_pulse_channel_backup)
        if flux_pulse_delay is not None:
            self.flux_pulse_delay(flux_pulse_delay_backup)

        return MA

    def measure_flux_detuning(self, flux_params=None, n=1, ramsey_times=None,
                              artificial_detuning=0, MC=None,
                              analyze=True, close_fig=True, upload=True,
                              fluxing_channels=[]):
        """
        DCL code. Instead use qb.calibrate_flux_pulse_frequency()

        Sweep over flux pulse amplitudes; for each, perform a Ramsey to get the
        detuning from qubit parking position.

        :param flux_amps: flux pulse amplitudes (V)
        :param n: #ofPulses
        :param MC: Measurement Control object
        :param analyze: do you want to analyze your data?
        :param close_fig: Close figure?
        :param upload: Don't know
        :return: Detuning of qubit freq from parking position with the applied
        flux pulse amplitude.
        """
        if flux_params is None:
            raise ValueError("Unspecified flux amplitudes "
                             "for measure_flux_detuning")
        if ramsey_times is None:
            raise ValueError("Unspecified Ramsey times "
                             "for measure_flux_detuning")

        if MC is None:
            MC = self.MC

        self.prepare_for_timedomain()

        flux_det_sweep = awg_swf.awg_seq_swf(fsqs.Ram_Z_seq,
              awg_seq_func_kwargs={'operation_dict': self.get_operation_dict(),
                                   'q0': self,
                                   'operation_name': 'flux_detuning',
                                   'times': ramsey_times,
                                   'artificial_detuning': artificial_detuning,
                                   'distortion_dict': self.dist_dict()},
              fluxing_channels=fluxing_channels,
              parameter_name='times')


        MC.set_sweep_function(ram_Z_sweep)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Flux_Detuning'+label+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

        """------------------
        #Soft sweep over flux amplitudes using
        awg_swf.awg_seq_swf(fsqs.single_pulse_seq)
        pulse_pars = {'pulse_type': 'SquarePulse',
                      'pulse_delay': .1e-6,
                      'channel': 'ch3',
                      'amplitude': 0.5,
                      'length': .1e-6,
                      'dead_time_length': 10e-6}
        square_pulse_seq=fsqs.single_pulse_seq(comp_pulse=False,return_seq=True)

        awg_square_sweep_fcn=awg_swf.awg_seq_swf(
            square_pulse_seq,
            unit='V',
            awg_seq_func_kwargs={'comp_pulse': False,
                                 'q0': self.qubit.name,
                                 'cal_points':self.cal_points,
                                 'distortion_dict': self.dist_dict,
                                 'upload': True})
        MC.set_sweep_function(awg_square_sweep_fcn)
        --------------------
        #Set the sweep points=sweep over flux amplitudes
        MC.set_sweep_points(flux_params['Amplitudes'])

        #Use the composite_detector_function
        flux_detuning_det=cdet.Flux_Detuning(
            flux_params,
            ramsey_times,
            artificial_detuning,
            self,
            self.AWG,
            MC,
            upload=upload,
            pulse_pars=self.get_drive_pars(),
            RO_pars=self.get_RO_pars())
        MC.set_detector_function(flux_detuning_det)

        MC.run('Flux_Detuning-n{}'.format(n) + self.msmt_suffix)

        #if analyze=True, instantiate a MeasurementAnalysis object,
        #which will contain the Flux_Detuning_Analysis class, that
        #will analyze the data and produce nice plots.
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)"""

    def find_resonator_frequency(self, freqs=None, update=False, MC=None,
                                 close_fig=True, fitting_model='hanger', **kw):
        """
        Finds the resonator frequency by performing a heterodyne experiment
        if freqs == None it will determine a default range dependent on the
        last known frequency of the resonator.
        WARNING: Does not automatically update the RO resonator parameters.
        Set update=True if you want this!
        """

        if not update:
            logging.warning("Does not automatically update the RO "
                            "resonator parameters. "
                            "Set update=True if you want this!")
        if freqs is None:
            if self.f_RO_resonator() != 0 and self.Q_RO_resonator() != 0:
                fmin = self.f_RO_resonator()*(1-10/self.Q_RO_resonator())
                fmax = self.f_RO_resonator()*(1+10/self.Q_RO_resonator())
                freqs = np.linspace(fmin, fmax, 100)
            else:
                raise ValueError("Unspecified frequencies for find_resonator_"
                                 "frequency and no previous value exists")

        if np.any(freqs<500e6):
            logging.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if MC is None:
            MC = self.MC

        self.measure_resonator_spectroscopy(freqs, MC, analyze=False)
        label = 'resonator_scan' + self.msmt_suffix
        HA = ma.Homodyne_Analysis(qb_name=self.name,
                                  label=label, close_fig=close_fig,
                                  fitting_model=fitting_model,**kw)
        f0 = HA.fit_res.params['f0'].value
        df0 = HA.fit_res.params['f0'].stderr
        Q = HA.fit_res.params['Q'].value
        dQ = HA.fit_res.params['Q'].stderr
        if f0 > max(freqs) or f0 < min(freqs):
            logging.warning('extracted frequency outside of range of scan')
        elif df0 > f0:
            logging.warning('resonator frequency uncertainty greater than '
                            'value')
        elif dQ > Q:
            logging.warning('resonator Q factor uncertainty greater than '
                            'value')
        elif update:  # don't update if there was trouble
            self.f_RO_resonator(f0)
            self.Q_RO_resonator(Q)
            self.heterodyne.frequency(f0)
        return f0

    def find_homodyne_acqusition_delay(self, delays=None, update=False, MC=None,
                                         close_fig=True):
        """
        Finds the acquisition delay for a homodyne experiment that corresponds
        to maximal signal strength.
        WARNING: Does not automatically update the qubit acquisition delay.
        Set update=True if you want this!
        """

        if not update:
            logging.warning("Does not automatically update the qubit "
                            "acquisition delay. "
                            "Set update=True if you want this!")

        if delays is None:
            delays = np.linspace(0,1e-6,100)

        if MC is None:
            MC = self.MC

        self.measure_homodyne_acqusition_delay(delays, MC, analyze=False)

        DA = ma.Acquisition_Delay_Analysis(label=self.msmt_suffix,
                                           close_fig=close_fig)
        d = DA.max_delay

        if update:
            self.optimal_acquisition_delay(d)
            self.heterodyne.acquisition_delay(d)
        return d

    def find_frequency(self, freqs, method='cw_spectroscopy', update=False,
                       MC=None, close_fig=True, analyze_ef=False, analyze=True,
                       upload=True, label=None,
                       **kw):
        """
        WARNING: Does not automatically update the qubit frequency parameter.
        Set update=True if you want this!

        Args:
            method:                   the spectroscopy type; options: 'pulsed',
                                      'spectrsocopy'
            update:                   whether to update the relevant qubit
                                      parameters with the found frequency(ies)
            MC:                       the measurement control object
            close_fig:                whether or not to close the figure
            analyze_ef:               whether or not to also look for the gf/2

        Keyword Args:
            interactive_plot:        (default=False)
                whether to plot with plotly or not
            analyze_ef:              (default=False)
                whether to look for another f_ge/2 peak/dip
            percentile:              (default=20)
                percentile of the data that is considered background noise
            num_sigma_threshold:     (default=5)
                used to define the threshold above(below) which to look for
                peaks(dips); threshold = background_mean +
                num_sigma_threshold * background_std
            window_len              (default=3)
                filtering window length; uses a_tools.smooth
            analysis_window         (default=10)
                how many data points (calibration points) to remove before
                sending data to peak_finder; uses a_tools.cut_edges,
                data = data[(analysis_window//2):-(analysis_window//2)]
            amp_only                (default=False)
                whether only I data exists
            save_name               (default='Source Frequency')
                figure name with which it will be saved
            auto                    (default=True)
                automatically perform the entire analysis upon call
            label                   (default=none?)
                label of the analysis routine
            folder                  (default=working folder)
                working folder
            NoCalPoints             (default=4)
                number of calibration points
            print_fit_results       (default=True)
                print the fit report
            print_frequency         (default=False)
                whether to print the f_ge and f_gf/2
            make_fig          {default=True)
                    whether or not to make a figure
            show                    (default=True)
                show the plots
            show_guess              (default=False)
                plot with initial guess values
            close_file              (default=True)
                close the hdf5 file

        Returns:
            the peak frequency(ies).
        """
        if not update:
            logging.warning("Does not automatically update the qubit "
                            "frequency parameter. "
                            "Set update=True if you want this!")
        if np.any(freqs<500e6):
            logging.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if MC is None:
            MC = self.MC

        if freqs is None:
            f_span = kw.get('f_span', 100e6)
            f_mean = kw.get('f_mean', self.f_qubit())
            nr_points = kw.get('nr_points', 100)
            if f_mean == 0:
                logging.warning("find_frequency does not know where to "
                                "look for the qubit. Please specify the "
                                "f_mean or the freqs function parameter.")
                return 0
            else:
                freqs = np.linspace(f_mean - f_span/2, f_mean + f_span/2,
                                    nr_points)

        if 'pulse' not in method.lower():
            if label is None:
                label = 'spectroscopy' + self.msmt_suffix
            if analyze_ef:
                label = 'high_power_' + label

            self.measure_spectroscopy(freqs, pulsed=False, MC=MC, label=label,
                                      close_fig=close_fig)
        else:
            if label is None:
                label = 'pulsed_spec' + self.msmt_suffix
            if analyze_ef:
                label = 'high_power_' + label

            self.measure_spectroscopy(freqs, pulsed=True, MC=MC, label=label,
                                      close_fig=close_fig,upload=upload)


        if analyze:
            amp_only = hasattr(self.heterodyne, 'RF')
            SpecA = ma.Qubit_Spectroscopy_Analysis(
                qb_name=self.name,
                analyze_ef=analyze_ef,
                label=label,
                close_fig=close_fig,**kw)

            f0 = SpecA.fitted_freq
            if update:
                self.f_qubit(f0)
            if analyze_ef:
                f0_ef = 2*SpecA.fitted_freq_gf_over_2 - f0
                if update:
                    self.f_ef_qubit(f0_ef)
            if analyze_ef:
                return f0, f0_ef
            else:
                return f0
        else:
            return

    def find_amplitudes(self, rabi_amps=None, label=None, for_ef=False,
                        update=False, MC=None, close_fig=True, cal_points=True,
                        no_cal_points=None, upload=True, last_ge_pulse=True,
                        analyze=True, **kw):

        """
            Finds the pi and pi/2 pulse amplitudes from the fit to a Rabi
            experiment. Uses the Rabi_Analysis(_new)
            class from measurement_analysis.py
            WARNING: Does not automatically update the qubit amplitudes.
            Set update=True if you want this!

            Analysis script for the Rabi measurement:
            1. The I and Q data are rotated and normalized based on the calibration
                points. In most analysis routines, the latter are typically 4:
                2 X180 measurements, and 2 identity measurements, which get
                averaged resulting in one X180 point and one identity point.
                However, the default for Rabi is 2 (2 identity measurements)
                because we typically do Rabi in order to find the correct amplitude
                for an X180 pulse. However, if a previous such value exists, this
                routine also accepts 4 cal pts. If X180_ef pulse was also
                previously calibrated, this routine also accepts 6 cal pts.
            2. The normalized data is fitted to a cosine function.
            3. The pi-pulse and pi/2-pulse amplitudes are calculated from the fit.
            4. The normalized data, the best fit results, and the pi and pi/2
                pulses are plotted.

            The ef analysis assumes the the e population is zero (because of the
            ge X180 pulse at the end).

            Arguments:
                rabi_amps:          amplitude sweep points for the
                                    Rabi experiment
                label:              label of the analysis routine
                for_ef:             find amplitudes for the ef transition
                update:             update the qubit amp180 and amp90 parameters
                MC:                 the measurement control object
                close_fig:          close the resulting figure?
                cal_points          whether to used calibration points of not
                no_cal_points       number of calibration points to use; if it's
                                    the first time rabi is run
                                    then 2 cal points (two I pulses at the end)
                                    should be used for the ge Rabi,
                                    and 4 (two I pulses and 2 ge X180 pulses at
                                    the end) for the ef Rabi
                last_ge_pulse       whether to map the population to the ground
                                    state after each run of the Rabi experiment
                                    on the ef level
            Keyword arguments:
                other keyword arguments. The Rabi sweep parameters 'amps_mean',
                 'amps_span', and 'nr_poinys' should be passed here. This will
                 result in a sweep over rabi_amps = np.linspace(amps_mean -
                 amps_span/2, amps_mean + amps_span/2, nr_points)

                auto              (default=True)
                    automatically perform the entire analysis upon call
                print_fit_results (default=True)
                    print the fit report
                make_fig          {default=True)
                    whether or not to make a figure
                show              (default=True)
                    show the plots
                show_guess        (default=False)
                    plot with initial guess values
                show_amplitudes   (default=True)
                    print the pi&piHalf pulses amplitudes
                plot_amplitudes   (default=True)
                    plot the pi&piHalf pulses amplitudes
                no_of_columns     (default=1)
                    number of columns in your paper; figure sizes will be adjusted
                    accordingly (1 col: figsize = ( 7in , 4in ) 2 cols: figsize =
                    ( 3.375in , 2.25in ), PRL guidelines)

            Returns:
                pi and pi/2 pulses amplitudes + their stderr as a dictionary with
                keys 'piPulse', 'piHalfPulse', 'piPulse_std', 'piHalfPulse_std'.
            """

        if not update:
            logging.warning("Does not automatically update the qubit pi and "
                            "pi/2 amplitudes. "
                            "Set update=True if you want this!")

        if MC is None:
            MC = self.MC

        if (cal_points) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if not cal_points:
            no_cal_points = 0

        #how many times to apply the Rabi pulse
        n = kw.get('n', 1)

        if rabi_amps is None:
            amps_span = kw.get('amps_span', 1.)
            amps_mean = kw.get('amps_mean', self.amp180())
            nr_points = kw.get('nr_points', 30)
            if amps_mean == 0:
                logging.warning("find_amplitudes does not know over which "
                                "amplitudes to do Rabi. Please specify the "
                                "amps_mean or the amps function parameter.")
                return 0
            else:
                rabi_amps = np.linspace(amps_mean - amps_span/2, amps_mean +
                                        amps_span/2, nr_points)

        if label is None:
            if for_ef:
                label = 'Rabi_2nd' + self.msmt_suffix
            else:
                label = 'Rabi' + self.msmt_suffix

        #Perform Rabi
        if for_ef is False:
            self.measure_rabi(amps=rabi_amps, n=n, MC=MC,
                              close_fig=close_fig,
                              label=label,
                              cal_points=cal_points,
                              no_cal_points=no_cal_points,
                              upload=upload)
        else:
            self.measure_rabi_2nd_exc(amps=rabi_amps, n=n, MC=MC,
                                      close_fig=close_fig, label=label,
                                      cal_points=cal_points,
                                      last_ge_pulse=last_ge_pulse,
                                      no_cal_points=no_cal_points,
                                      upload=upload)

        #get pi and pi/2 amplitudes from the analysis results
        if analyze:
            RabiA = ma.Rabi_Analysis(label=label, qb_name=self.name,
                                     NoCalPoints=no_cal_points,
                                     close_fig=close_fig, for_ef=for_ef,
                                     last_ge_pulse=last_ge_pulse, **kw)

            rabi_amps = RabiA.rabi_amplitudes   #This is a dict with keywords
                                                #'piPulse',  'piPulse_std',
                                                #'piHalfPulse', 'piHalfPulse_std

            amp180 = rabi_amps['piPulse']
            amp90 = rabi_amps['piHalfPulse']

            if update:
                if for_ef is False:
                    self.amp180(amp180)
                    self.amp90_scale(amp90/amp180)
                else:
                    self.amp180_ef(amp180)
                    self.amp90_scale_ef(amp90/amp180)
        else:
            return


    def find_T1(self, times, label=None, for_ef=False, update=False, MC=None,
                cal_points=True, no_cal_points=None, close_fig=True,
                last_ge_pulse=True, upload=True, **kw):

        """
        Finds the relaxation time T1 from the fit to an exponential
        decay function.
        WARNING: Does not automatically update the qubit T1 parameter.
        Set update=True if you want this!

        Routine:
            1. Apply pi pulse to get population in the excited state.
            2. Wait for different amounts of time before doing a measurement.

        Uses the T1_Analysis class from measurement_analysis.py.
        The ef analysis assumes the the e population is zero (because of the
        ge X180 pulse at the end).

        Arguments:
            times:                   array of times to wait before measurement
            label:                   label of the analysis routine
            for_ef:                  find T1 for the 2nd excitation (ef)
            update:                  update the qubit T1 parameter
            MC:                      the measurement control object
            close_fig:               close the resulting figure?

        Keyword Arguments:
            other keyword arguments. The the parameters times_mean, times_span,
            nr_points should be passed here. These are an alternative to
            passing the times array.

            auto              (default=True)
                automatically perform the entire analysis upon call
            print_fit_results (default=True)
                print the fit report
            make_fig          (default=True)
                whether to make the figures or not
            show_guess        (default=False)
                plot with initial guess values
            show_T1           (default=True)
                print the T1 and T1_stderr
            no_of_columns     (default=1)
                number of columns in your paper; figure sizes will be adjusted
                accordingly  (1 col: figsize = ( 7in , 4in ) 2 cols:
                figsize = ( 3.375in , 2.25in ), PRL guidelines)

        Returns:
            the relaxation time T1 + standard deviation as a dictionary with
            keys: 'T1', and 'T1_std'

        ! Specify either the times array or the times_mean value (defaults to
        5 micro-s) and the span around it (defaults to 10 micro-s) as kw.
        Then the script will construct the sweep points as
        np.linspace(times_mean - times_span/2, times_mean + times_span/2,
        nr_points)
        """

        if not update:
            logging.warning("Does not automatically update the qubit "
                            "T1 parameter. Set update=True if you want this!")
        if np.any(times>1e-3):
            logging.warning('Some of the values in the times array might be too '
                            'large.The units should be seconds.')

        if (cal_points) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if not cal_points:
            no_cal_points = 0

        if MC is None:
            MC = self.MC

        if label is None:
            if for_ef:
                label = 'T1_2nd' + self.msmt_suffix
            else:
                label = 'T1' + self.msmt_suffix

        if times is None:
            times_span = kw.get('times_span', 10e-6)
            times_mean = kw.get('times_mean', 5e-6)
            nr_points = kw.get('nr_points', 50)
            if times_mean == 0:
                logging.warning("find_T1 does not know how long to wait before"
                                "doing the read out. Please specify the "
                                "times_mean or the times function parameter.")
                return 0
            else:
                times = np.linspace(times_mean - times_span/2, times_mean +
                                    times_span/2, nr_points)

        #Perform measurement
        if for_ef:
            self.measure_T1_2nd_exc(times=times, MC=MC,
                                    close_fig=close_fig,
                                    cal_points=cal_points,
                                    no_cal_points=no_cal_points,
                                    last_ge_pulse=last_ge_pulse,
                                    upload=upload)

        else:
            self.measure_T1(times=times, MC=MC,
                            close_fig=close_fig,
                            cal_points=cal_points,
                            upload=upload)

        #Extract T1 and T1_stddev from ma.T1_Analysis
        if kw.pop('analyze', True):
            T1_Analysis = ma.T1_Analysis(label=label, qb_name=self.name,
                                         NoCalPoints=no_cal_points,
                                         for_ef=for_ef,
                                         last_ge_pulse=last_ge_pulse, **kw)
            T1_dict = T1_Analysis.T1_dict
            T1_value = T1_dict['T1']

            if update:
                if for_ef:
                    self.T1_ef(T1_value)
                else:
                    self.T1(T1_value)

            return T1_dict
        else:
            return

    def find_RB_gate_fidelity(self, nr_cliffords, label=None, nr_seeds=10,
                              MC=None, cal_points=True, gate_decomposition='HZ',
                              thresholded=False,
                              no_cal_points=None, close_fig=True,
                              upload=True, det_func=None, **kw):

        for_ef = kw.pop('for_ef', False)
        last_ge_pulse = kw.pop('last_ge_pulse', False)
        analyze = kw.pop('analyze', True)
        show = kw.pop('show', False)
        interleaved_gate = kw.pop('interleaved_gate', None)
        T1 = kw.pop('T1', None)
        T2 = kw.pop('T1', None)

        if det_func is None:
            det_func = self.int_avg_det

        if T1 is None and self.T1() is not None:
            T1 = self.T1()
        if T2 is None:
            if self.T2() is not None:
                T2 = self.T2()
            elif self.T2_star() is not None:
                print('T2 is None. Using T2_star.')
                T2 = self.T2_star()

        if type(nr_cliffords) is int:
            every_other = kw.pop('every_other', 5)
            nr_cliffords = np.asarray([j for j in
                                       list(range(0, nr_cliffords[0]+1,
                                                  every_other))])

        # if not update:
        #     logging.warning("Does not automatically update the qubit "
        #                     "parameter. Set update=True if you want this!")

        if (cal_points) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if not cal_points:
            no_cal_points = 0

        if MC is None:
            MC = self.MC

        if nr_cliffords is None:
            raise ValueError("Unspecified nr_cliffords")

        if label is None:
            if interleaved_gate is None:
                if for_ef:
                    label = 'RB_2nd_{}_{}_seeds_{}_cliffords'.format(
                        gate_decomposition, nr_seeds,
                        nr_cliffords[-1]) + self.msmt_suffix
                else:
                    label = 'RB_{}_{}_seeds_{}_cliffords'.format(
                        gate_decomposition, nr_seeds,
                        nr_cliffords[-1]) + self.msmt_suffix
            else:
                if for_ef:
                    label = 'IRB_2nd_{}_{}_{}_seeds_{}_cliffords'.format(
                        interleaved_gate, gate_decomposition,
                        nr_seeds, nr_cliffords[-1]) \
                            + self.msmt_suffix
                else:
                    label = 'IRB_{}_{}_{}_seeds_{}_cliffords'.format(
                        interleaved_gate, gate_decomposition,
                        nr_seeds, nr_cliffords[-1]) \
                            + self.msmt_suffix

        if thresholded:
            label += '_thresh'
            if int(self.UHFQC.get('quex_thres_{}_level'.format(
                    self.RO_acq_weight_function_I()))) == 0:
                raise ValueError('The threshold value is not set.')

        #Perform measurement
        self.measure_randomized_benchmarking(nr_cliffords=nr_cliffords,
                                             nr_seeds=nr_seeds, MC=MC,
                                             close_fig=close_fig,
                                             gate_decomp=gate_decomposition,
                                             thresholded=thresholded,
                                             cal_points=cal_points,
                                             label=label,
                                             analyze=analyze,
                                             upload=upload,
                                             interleaved_gate=interleaved_gate,
                                             det_func=det_func)

        #Analysis
        if analyze:
            pulse_delay = self.gauss_sigma() * self.nr_sigma()
            RB_Analysis = rbma.RandomizedBenchmarking_Analysis(label=label,
                                         qb_name=self.name,
                                         T1=T1, T2=T2, pulse_delay=pulse_delay,
                                         NoCalPoints=no_cal_points,
                                         for_ef=for_ef, show=show,
                                         gate_decomp=gate_decomposition,
                                         last_ge_pulse=last_ge_pulse, **kw)

        return

    def find_frequency_T2_ramsey(self, times, artificial_detuning=0,
                                 upload=True, MC=None, label=None,
                                 cal_points=True, no_cal_points=None,
                                 analyze=True, close_fig=True, update=False,
                                 for_ef=False, last_ge_pulse=False, **kw):
        """
        Finds the real qubit GE or EF transition frequencies and the dephasing
        rates T2* or T2*_ef from the fit to a Ramsey experiment.

        Uses the Ramsey_Analysis class for Ramsey with one artificial detuning,
        and the Ramsey_Analysis_multiple_detunings class for Ramsey with 2
        artificial detunings.

        Has support only for 1 or 2 artifical detunings.

        WARNING: Does not automatically update the qubit freq and T2_star
        parameters. Set update=True if you want this!

        Arguments:
            times                    array of times over which to sweep in
                                        the Ramsey measurement
            artificial_detuning:     difference between drive frequency and
                                        qubit frequency estimated from
                                        qubit spectroscopy. Must be a list with
                                        one or two entries.
            upload:                  upload sequence to AWG
            update:                  update the qubit frequency and T2*
                                        parameters
            MC:                      the measurement control object
            label:                   measurement label
            cal_points:              use calibration points or not
            no_cal_points:           number of cal_points (4 for ge;
                                        2,4,6 for ef)
            analyze:                 perform analysis
            close_fig:               close the resulting figure
            update:                  update relevant parameters
            for_ef:                  perform msmt and analysis on ef transition
            last_ge_pulse:           ge pi pulse at the end of each sequence

        Keyword arguments:
            For one artificial detuning, the Ramsey sweep time delays array
            'times', or the parameter 'times_mean' should be passed
            here (in seconds).

        Returns:
            The real qubit frequency + stddev, the dephasing rate T2* + stddev.

        For 1 artificial_detuning:
            ! Specify either the times array or the times_mean value (defaults
            to 2.5 micro-s) and the span around it (times_mean; defaults to 5
            micro-s) as kw. Then the script will construct the sweep points as
            times = np.linspace(times_mean - times_span/2, times_mean +
            times_span/2, nr_points).
        """
        if not update:
            logging.warning("Does not automatically update the qubit frequency "
                            "and T2_star parameters. "
                            "Set update=True if you want this!")
        if artificial_detuning == None:
            logging.warning('Artificial_detuning is None; qubit driven at "%s" '
                            'estimated with '
                            'spectroscopy' %self.f_qubit())
        if np.any(np.asarray(np.abs(artificial_detuning))<1e3):
            logging.warning('The artificial detuning is too small.')
        if np.any(times>1e-3):
            logging.warning('The values in the times array might be too large.')

        if (cal_points is True) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if '
                            'for_ef==False, or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if cal_points is False:
            no_cal_points = 0

        if MC is None:
            MC = self.MC

        if label is None:
            if for_ef:
                label = 'Ramsey_2nd' +self.msmt_suffix
            else:
                label = 'Ramsey' + self.msmt_suffix

        # Check if one or more artificial detunings
        if (type(artificial_detuning) is list) and (len(artificial_detuning)>1):
            # 2 ARTIFICIAL_DETUNING VALUES

            if times is None:
                logging.warning("find_frequency_T2_ramsey does not know over "
                                "which times to do Ramsey. Please specify the "
                                "times_mean or the times function parameter.")

            # Each time value must be repeated len(artificial_detunings) times
            # to correspond to the logic in Ramsey_seq_multiple_detunings
            # sequence
            len_art_det = len(artificial_detuning)
            # temp_array = np.zeros((times.size-no_cal_points)*len_art_det)
            temp_array = np.zeros(times.size*len_art_det)
            for i in range(len_art_det):
                np.put(temp_array, list(range(i, temp_array.size, len_art_det)),
                       times)
            # times = np.append(temp_array,times[-no_cal_points::])
            times = temp_array

            #Perform Ramsey multiple detunings
            if for_ef is False:
                self.measure_ramsey_multiple_detunings(
                    times=times,
                    artificial_detunings=artificial_detuning,
                    MC=MC,
                    label=label,
                    cal_points=cal_points,
                    close_fig=close_fig, upload=upload)

            else:
                self.measure_ramsey_2nd_exc_multiple_detunings(
                    times=times,
                    artificial_detunings=artificial_detuning,
                    cal_points=cal_points, no_cal_points=no_cal_points,
                    close_fig=close_fig, upload=upload,
                    last_ge_pulse=last_ge_pulse, MC=MC, label=label)

        else:
            # 1 ARTIFICIAL_DETUNING VALUE

            if type(artificial_detuning) is list:
                artificial_detuning = artificial_detuning[0]

            if times is None:
                times_span = kw.get('times_span', 5e-6)
                times_mean = kw.get('times_mean', 2.5e-6)
                nr_points = kw.get('nr_points', 50)
                if times_mean == 0:
                    logging.warning("find_frequency_T2_ramsey does not know "
                                    "over which times to do Ramsey. Please "
                                    "specify the times_mean or the times "
                                    "function parameter.")
                    return 0
                else:
                    times = np.linspace(times_mean - times_span/2,
                                        times_mean + times_span/2,
                                        nr_points)

            #Perform Ramsey one detuning
            if for_ef is False:
                self.measure_ramsey(times=times,
                                    artificial_detuning=artificial_detuning,
                                    MC=MC,
                                    cal_points=cal_points,
                                    close_fig=close_fig,
                                    upload=upload, label=label)

            else:
                self.measure_ramsey_2nd_exc(
                    times=times,
                    artificial_detuning=artificial_detuning,
                    MC=MC, cal_points=cal_points,
                    close_fig=close_fig, upload=upload,
                    last_ge_pulse=last_ge_pulse,
                    no_cal_points=no_cal_points, label=label)

        if analyze:

            RamseyA = ma.Ramsey_Analysis(
                auto=True,
                label=label,
                qb_name=self.name,
                NoCalPoints=no_cal_points,
                for_ef=for_ef,
                last_ge_pulse=last_ge_pulse,
                artificial_detuning=artificial_detuning, **kw)

            #get new freq and T2* from analysis results
            new_qubit_freq = RamseyA.qubit_frequency    #value
            fitted_freq = RamseyA.ramsey_freq           #dict
            T2_star = RamseyA.T2_star                   #dict

            print('New qubit frequency = {:.10f} \t stderr = {:.10f}'.format(
                new_qubit_freq, RamseyA.ramsey_freq['freq_stderr']))
            print('T2_Star = {:.5f} \t stderr = {:.5f}'.format(
                T2_star['T2_star'],T2_star['T2_star_stderr']))

            if update:
                if for_ef:
                    try:
                        self.f_ef_qubit(new_qubit_freq)
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)
                    try:
                        self.T2_star_ef(T2_star['T2_star'])
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)
                else:
                    try:
                        self.f_qubit(new_qubit_freq)
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)
                    try:
                        self.T2_star(T2_star['T2_star'])
                    except AttributeError as e:
                        logging.warning('%s. This parameter will not be '
                                        'updated.'%e)

            return new_qubit_freq, fitted_freq, T2_star

        else:
            return

    def find_qscale(self, qscales, label=None, for_ef=False, update=False,
                    MC=None, close_fig=True, last_ge_pulse=True, upload=True,
                    cal_points=True, no_cal_points=None, **kw):

        '''
        Performs the QScale calibration measurement ( (xX)-(xY)-(xmY) ) and
        extracts the optimal QScale parameter
        from the fits (ma.QScale_Analysis).
        WARNING: Does not automatically update the qubit qscale parameter. Set
        update=True if you want this!

        ma.QScale_Analysis:
        1. The I and Q data are rotated and normalized based on the calibration
            points. In most
            analysis routines, the latter are typically 4: 2 X180 measurements,
            and 2 identity measurements, which get averaged resulting in one
            X180 point and one identity point.
        2. The data points for the same qscale value are extracted (every other
            3rd point because the sequence
            used for this measurement applies the 3 sets of pulses
            ( (xX)-(xY)-(xmY) ) consecutively for each qscale value).
        3. The xX data is fitted to a lmfit.models.ConstantModel(), and the
            other 2 to an lmfit.models.LinearModel().
        4. The data and the resulting fits are all plotted on the same graph
            (self.make_figures).
        5. The optimal qscale parameter is obtained from the point where the 2
            linear fits intersect.

        Other possible  input parameters:
            qscales
                array of qscale values over which to sweep...
            or qscales_mean and qscales_span
                ...or the mean qscale value and the span around it
                (defaults to 3) as kw. Then the script will construct the sweep
                points as np.linspace(qscales_mean - qscales_span/2,
                qscales_mean + qscales_span/2, nr_points)

        Keyword parameters:
            label             (default=none?)
                label of the analysis routine
            for_ef            (default=False)
                whether to obtain the drag_qscale_ef parameter
            update            (default=True)
                whether or not to update the qubit drag_qscale parameter with
                the found value
            MC                (default=self.MC)
                the measurement control object
            close_fig         (default=True)
                close the resulting figure
            last_ge_pulse     (default=True)
                whether to apply an X180 ge pulse at the end

            Keyword parameters:
                qscale_mean       (default=self.drag_qscale()
                    mean of the desired qscale sweep values
                qscale_span       (default=3)
                    span around the qscale mean
                nr_points         (default=30)
                    number of sweep points between mean-span/2 and mean+span/2
                auto              (default=True)
                    automatically perform the entire analysis upon call
                folder            (default=working folder)
                    Working folder
                NoCalPoints       (default=4)
                    Number of calibration points
                cal_points        (default=[[-4, -3], [-2, -1]])
                    The indices of the calibration points
                show              (default=True)
                    show the plot
                show_guess        (default=False)
                    plot with initial guess values
                plot_title        (default=measurementstring)
                    the title for the plot as a string
                xlabel            (default=self.xlabel)
                    the label for the x axis as a string
                ylabel            (default=r'$F|1\rangle$')
                    the label for the x axis as a string
                close_file        (default=True)
                    close the hdf5 file

        Returns:
            the optimal DRAG QScale parameter + its stderr as a dictionary with
            keys 'qscale' and 'qscale_std'.
        '''

        if not update:
            logging.warning("Does not automatically update the qubit qscale "
                            "parameter. "
                            "Set update=True if you want this!")

        if (cal_points) and (no_cal_points is None):
            logging.warning('no_cal_points is None. Defaults to 4 if for_ef==False,'
                            'or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if not cal_points:
            no_cal_points = 0

        if MC is None:
            MC = self.MC

        if label is None:
            label = 'QScale' + self.msmt_suffix

        if qscales is None:
            # qscales_span = kw.get('qscales_span', 3)
            # qscales_mean = kw.get('qscales_mean', self.drag_qscale())
            # nr_points = kw.get('nr_points', 30)
            # if qscales_mean == 0:
            logging.warning("find_qscale does not know over which "
                            "qscale values to sweep. Please specify the "
                            "qscales_mean or the qscales function"
                            " parameter.")
            #     return 0
            # else:
            #     qscales = np.linspace(qscales_mean - qscales_span/2,
            #                           qscales_mean + qscales_span/2, nr_points)

        # Each qscale value must be repeated 3 times to correspoond to the
        # logic in QScale sequence
        # temp_array = np.zeros(3*(qscales.size-no_cal_points))
        # np.put(temp_array,list(range(0,temp_array.size,3)),qscales)
        # np.put(temp_array,list(range(1,temp_array.size,3)),qscales)
        # np.put(temp_array,list(range(2,temp_array.size,3)),qscales)
        # qscales = np.append(temp_array,qscales[-no_cal_points::])
        temp_array = np.zeros(3*qscales.size)
        np.put(temp_array,list(range(0,temp_array.size,3)),qscales)
        np.put(temp_array,list(range(1,temp_array.size,3)),qscales)
        np.put(temp_array,list(range(2,temp_array.size,3)),qscales)
        qscales = temp_array

        #Perform the qscale calibration measurement
        if for_ef:
            # Run measuremet
            self.measure_qscale_2nd_exc(qscales=qscales, MC=MC, upload=upload,
                                        close_fig=close_fig, label=label,
                                        last_ge_pulse=last_ge_pulse,
                                        cal_points=cal_points,
                                        no_cal_points=no_cal_points)
        else:
            self.measure_qscale(qscales=qscales, MC=MC, upload=upload,
                                close_fig=close_fig, label=label)

        # Perform analysis and extract the optimal qscale parameter
        # Returns the optimal qscale parameter
        if kw.pop('analyze',True):
            QscaleA = ma.QScale_Analysis(auto=True, qb_name=self.name,
                                         label=label,
                                         NoCalPoints=no_cal_points,
                                         for_ef=for_ef,
                                         last_ge_pulse=last_ge_pulse, **kw)

            Qscale_dict = QscaleA.optimal_qscale #dictionary of value, stderr
            Qscale_value = Qscale_dict['qscale']

            if update:
                if for_ef:
                    self.motzoi_ef(Qscale_value)
                else:
                    self.motzoi(Qscale_value)

            return Qscale_dict
        else:
            return

    def calculate_anharmonicity(self, update=False):

        """
        Computes the qubit anaharmonicity using f_ef (self.f_ef_qubit)
        and f_ge (self.f_qubit).
        It is assumed that the latter values exist.
        WARNING: Does not automatically update the qubit anharmonicity
        parameter. Set update=True if you want this!
        """
        if not update:
            logging.warning("Does not automatically update the qubit "
                            "anharmonicity parameter. "
                            "Set update=True if you want this!")

        if self.f_qubit() == 0:
            logging.warning('f_ge = 0. Run qubit spectroscopy or Ramsey.')
        if self.f_ef_qubit() == 0:
            logging.warning('f_ef = 0. Run qubit spectroscopy or Ramsey.')

        anharmonicity = self.f_ef_qubit() - self.f_qubit()

        if update:
            self.anharmonicity(anharmonicity)

        return  anharmonicity

    def calculate_EC_EJ(self, update=True, **kw):

        """
        Extracts EC and EJ from a least squares fit to the transmon
        Hamiltonian solutions. It uses a_tools.calculate_transmon_transitions,
        f_ge and f_ef.
        WARNING: Does not automatically update the qubit EC and EJ parameters.
        Set update=True if you want this!

        Keyword Arguments:
            asym:           (default=0)
                asymmetry d (Koch (2007), eqn 2.18) for asymmetric junctions
            reduced_flux:   (default=0)
                reduced magnetic flux through SQUID
            no_transitions  (default=2)
                how many transitions (levels) are you interested in
            dim:            (default=None)
                dimension of Hamiltonian will  be (2*dim+1,2*dim+1)
        """
        if not update:
            logging.warning("Does not automatically update the qubit EC and EJ "
                            "parameters. "
                            "Set update=True if you want this!")

        (EC,EJ) = a_tools.fit_EC_EJ(self.f_qubit(), self.f_ef_qubit(), **kw)

        if update:
            self.EC_qubit(EC)
            self.EJ_qubit(EJ)

        return EC, EJ

    def find_readout_frequency(self, freqs=None, update=False, MC=None, **kw):
        """
        You need a working pi-pulse for this to work. Also, if your
        readout pulse length is much longer than the T1, the results will not
        be nice as the excited state spectrum will be mixed with the ground
        state spectrum.
        """

        # FIXME: Make proper analysis class for this (Ants, 04.12.2017)
        if not update:
            logging.info("Does not automatically update the RO resonator "
                         "parameters. Set update=True if you want this!")
        if freqs is None:
            if self.f_RO() is not None:
                f_span = kw.pop('f_span', 20e6)
                fmin = self.f_RO() - f_span
                fmax = self.f_RO() + f_span
                n_freq = kw.pop('n_freq', 401)
                freqs = np.linspace(fmin, fmax, n_freq)
            else:
                raise ValueError("Unspecified frequencies for find_resonator_"
                                 "frequency and no previous value exists")
        if np.any(freqs < 500e6):
            logging.warning('Some of the values in the freqs array might be '
                            'too small. The units should be Hz.')
        if MC is None:
            MC = self.MC

        self.measure_dispersive_shift(freqs, MC=MC, analyze=False, **kw)
        MAon = ma.MeasurementAnalysis(label='on-spec' + self.msmt_suffix)
        MAoff = ma.MeasurementAnalysis(label='off-spec' + self.msmt_suffix)
        cdaton = MAon.measured_values[0] * \
                 np.exp(1j * np.pi * MAon.measured_values[1] / 180)
        cdatoff = MAoff.measured_values[0] * \
                  np.exp(1j * np.pi * MAoff.measured_values[1] / 180)
        fmax = freqs[np.argmax(np.abs(cdaton - cdatoff))]
        if update:
            self.f_RO(fmax)
        if kw.pop('plot', True):
            plt.plot(freqs / 1e9, np.abs(cdatoff),
                     label='qubit in $|g\\rangle$')
            plt.plot(freqs / 1e9, np.abs(cdaton),
                     label='qubit in $|e\\rangle$')
            plt.plot(freqs / 1e9, np.abs(cdaton - cdatoff),
                     label='difference')
            plt.vlines(fmax / 1e9, 0,
                       max(np.abs(cdatoff).max(), np.abs(cdaton).max()),
                       label='$\\nu_{{RO}} = {:.4f}$ GHz'.format(fmax / 1e9))
            plt.xlabel(r'Frequency, $f$ (GHz)')
            plt.ylabel(r'Transmission amplitude, $|S_{21}|$ (arb.)')
            plt.title(r'{} $\chi$ shift. {} and {}'.format(
                self.name, MAon.timestamp_string, MAoff.timestamp_string))
            plt.legend()
            MAoff.save_fig(plt.gcf(), 'chishift', ylabel='trans-amp')
        return fmax


    def measure_dispersive_shift(self, freqs, MC=None, analyze=True, **kw):
            # FIXME: Remove dependancy on heterodyne!
            if np.any(freqs < 500e6):
                logging.warning(('Some of the values in the freqs array might be '
                                 'too small. The units should be Hz.'))
            if MC is None:
                MC = self.MC

            heterodyne = self.heterodyne
            heterodyne.f_RO_mod(self.f_RO_mod())
            heterodyne.RO_length(self.RO_pulse_length())
            heterodyne.mod_amp(self.RO_amp())
            self.prepare_for_pulsed_spec()
            self.drive_LO.pulsemod_state('off')
            self.drive_LO.power(self.drive_LO_pow())
            self.UHFQC.quex_wint_length(int(self.RO_acq_integration_length()*1.8e9))
            heterodyne.nr_averages(self.RO_acq_averages())

            for mode in ('on', 'off'):
                sq.OffOn_seq(pulse_pars=self.get_drive_pars(),
                             RO_pars=self.get_RO_pars(),
                             pulse_comb='O{0}O{0}'.format(mode[1:]))
                MC.set_sweep_function(heterodyne.frequency)
                MC.set_sweep_points(freqs)
                demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
                    else 'double'
                MC.set_detector_function(det.Heterodyne_probe(
                    self.heterodyne,
                    trigger_separation=self.heterodyne.trigger_separation(),
                    demod_mode=demod_mode))
                self.AWG.start()
                MC.run(name='{}-spec{}'.format(mode, self.msmt_suffix))
                if analyze:
                    ma.MeasurementAnalysis(qb_name=self.name, **kw)

    def get_spec_pars(self):
        return self.get_operation_dict()['Spec ' + self.name]

    def get_RO_pars(self):
        return self.get_operation_dict()['RO ' + self.name]

    def get_drive_pars(self):
        return self.get_operation_dict()['X180 ' + self.name]

    def get_ef_drive_pars(self):
        return self.get_operation_dict()['X180_ef ' + self.name]

    def get_flux_pars(self):
        return self.get_operation_dict()['flux ' + self.name]

    def get_operation_dict(self, operation_dict=None):
        if operation_dict is None:
            operation_dict = {}
        operation_dict = super().get_operation_dict(operation_dict)
        operation_dict['Spec ' + self.name]['operation_type'] = 'MW'
        operation_dict['RO ' + self.name]['operation_type'] = 'RO'
        operation_dict['X180 ' + self.name]['operation_type'] = 'MW'
        operation_dict['X180_ef ' + self.name]['operation_type'] = 'MW'
        operation_dict['flux ' + self.name]['operation_type'] = 'Flux'
        operation_dict['X180_ef ' + self.name]['I_channel'] = \
            operation_dict['X180 ' + self.name]['I_channel']
        operation_dict['X180_ef ' + self.name]['Q_channel'] = \
            operation_dict['X180 ' + self.name]['Q_channel']
        operation_dict['X180_ef ' + self.name]['phi_skew'] = \
            operation_dict['X180 ' + self.name]['phi_skew']
        operation_dict['X180_ef ' + self.name]['alpha'] = \
            operation_dict['X180 ' + self.name]['alpha']

        if self.f_ef_qubit() == 0:
            operation_dict['X180_ef ' + self.name]['mod_frequency'] = None
        else:
            operation_dict['X180_ef ' + self.name]['mod_frequency'] = \
                self.f_ef_qubit() - self.f_qubit() + self.f_pulse_mod()

        operation_dict.update(add_suffix_to_dict_keys(
            sq.get_pulse_dict_from_pars(operation_dict['X180 ' + self.name]),
            ' ' + self.name))
        operation_dict.update(add_suffix_to_dict_keys(
            sq.get_pulse_dict_from_pars(operation_dict['X180_ef ' + self.name]),
            '_ef ' + self.name))

        return operation_dict



    def calibrate_flux_pulse_timing(self, MC=None, freqs=None, delays=None,
                                    analyze=False, update=False,**kw):
        """
        flux pulse timing calibration

        does a 2D measuement of the type:

                -------|X180|  ----------------  |RO|
                   <----->
                   | fluxpulse |

        where the flux pulse delay and the drive pulse frequency of the X180 pulse
        are swept.

        Args:
            MC: measurement control object
            freqs: numpy array frequencies in Hz for the flux pulse scope type experiment
            delays: numpy array with delays (in s) swept through
                    as delay of the drive pulse
            analyze: bool, if True, then the measured data
                        gets analyzed (for detailed documentation of the analysis see in
                        the FluxPulse_timing_calibration class update: bool, if True,
                        the AWG channel delay gets corrected, such that single qubit
                        gates and flux pulses have no relative delay

        Returns:
            fitted_delay: float, only returned, if analyze is True.
        """
        if MC is None:
            MC = self.MC

        channel = self.flux_pulse_channel()

        pulse_length = kw.pop('pulse_length', 100e-9)
        self.flux_pulse_length(pulse_length)
        amplitude = kw.pop('amplitude', 0.5)
        self.flux_pulse_amp(amplitude)

        measurement_string = 'Flux_pulse_delay_calibration_{}'.format(self.name)

        if freqs is None:
            freqs = self.f_qubit() + np.linspace(-50e6, 50e6, 20, endpoint=False)
        if delays is None:
            delays = np.linspace(-100e-9, pulse_length + 100e-9, 40, endpoint=False)

        self.prepare_for_timedomain()

        detector_fun = self.int_avg_det

        s1 = awg_swf.Fluxpulse_scope_swf(self)
        s2 = awg_swf.Fluxpulse_scope_drive_freq_sweep(self)

        MC.set_sweep_function(s1)
        MC.set_sweep_points(delays)
        MC.set_sweep_function_2D(s2)
        MC.set_sweep_points_2D(freqs)
        MC.set_detector_function(detector_fun)
        MC.run_2D(measurement_string)

        if analyze:
            flux_pulse_timing_ma = ma.FluxPulse_timing_calibration(
                        label=measurement_string,
                        flux_pulse_length=pulse_length,
                        qb_name=self.name,
                        auto=True,
                        plot=True)

            if update:
                new_delay = self.AWG.get('{}_delay'.format(channel)) + \
                            flux_pulse_timing_ma.fitted_delay
                self.AWG.set('{}_delay'.format(channel), new_delay)
                print('updated delay of channel {}.'.format(channel))
            else:
                logging.warning('Not updated, since update was disabled.')
            return flux_pulse_timing_ma.fitted_delay
        else:
            return



    def calibrate_flux_pulse_frequency(self,MC=None, thetas=None, ampls=None,
                                       analyze=False,
                                       plot=False,
                                       ampls_bidirectional = False,
                                       **kw):
        """
        flux pulse frequency calibration

        does a 2D measuement of the type:

                      X90_separation
                < -- ---- ----------- --->
                |X90|  --------------     |X90|  ---  |RO|
                       | fluxpulse |

        where the flux pulse amplitude and the angle of the second X90 pulse
        are swept.

        Args:
            MC: measurement control object
            thetas: numpy array with angles (in rad) for the Ramsey type
            ampls: numpy array with amplitudes (in V) swept through
                as flux pulse amplitudes
            ampls_bidirectional: bool, for use if the qubit is parked at sweetspot.
                                If true, the flux pulse amplitudes are swept to positive
                                and negative voltages and the frequency model fit is ]
                                performed on the combined dataset
            analyze: bool, if True, then the measured data
                     gets analyzed ( ma.fit_qubit_frequency() )


        """

        if MC is None:
            MC = self.MC

        channel = self.flux_pulse_channel()
        clock_rate = MC.station.pulsar.clock(channel)

        X90_separation = kw.pop('X90_separation', 200e-9)

        distorted = kw.pop('distorted', False)
        distortion_dict = kw.pop('distortion_dict', None)

        pulse_length = kw.pop('pulse_length', 30e-9)
        self.flux_pulse_length(pulse_length)

        pulse_delay = kw.pop('pulse_delay', 50e-9)
        self.flux_pulse_delay(pulse_delay)

        if thetas is None:
            thetas = np.linspace(0, 2*np.pi, 8, endpoint=False)

        if ampls is None:
            ampls = np.linspace(0, 1, 21)
            ampls_flag = True

        self.prepare_for_timedomain()
        detector_fun = self.int_avg_det

        s1 = awg_swf.Ramsey_interleaved_fluxpulse_sweep(
            self,
            X90_separation=X90_separation,
            distorted=distorted,
            distortion_dict=distortion_dict)
        s2 = awg_swf.Ramsey_fluxpulse_ampl_sweep(self, s1)

        MC.set_sweep_function(s1)
        MC.set_sweep_points(thetas)
        MC.set_sweep_function_2D(s2)
        MC.set_sweep_points_2D(ampls)
        MC.set_detector_function(detector_fun)

        measurement_string_1 = 'Flux_pulse_frequency_calibration_{}_1'.format(self.name)
        MC.run_2D(measurement_string_1)

        if ampls_bidirectional:
            MC.set_sweep_function(s1)
            MC.set_sweep_points(thetas)
            MC.set_sweep_function_2D(s2)
            MC.set_sweep_points_2D(-ampls)
            MC.set_detector_function(detector_fun)


            measurement_string_2 = 'Flux_pulse_frequency_calibration_{}_2'.format(self.name)
            MC.run_2D(measurement_string_2)

        if analyze:
            flux_pulse_ma_1 = ma.Fluxpulse_Ramsey_2D_Analysis(
                label=measurement_string_1,
                X90_separation=X90_separation,
                flux_pulse_length=pulse_length,
                qb_name=self.name,
                auto=False)
            flux_pulse_ma_1.fit_all(extrapolate_phase=True, plot=True)

            phases = flux_pulse_ma_1.fitted_phases
            ampls = flux_pulse_ma_1.sweep_points_2D


            if ampls_bidirectional:
                flux_pulse_ma_2 = ma.Fluxpulse_Ramsey_2D_Analysis(
                    label=measurement_string_2,
                    X90_separation=X90_separation,
                    flux_pulse_length=pulse_length,
                    qb_name=self.name,
                    auto=False)
                flux_pulse_ma_2.fit_all(extrapolate_phase=True, plot=True)

                phases = np.concatenate(flux_pulse_ma_2.fitted_phases[-1:0:-1],
                                        flux_pulse_ma_1.fitted_phases)
                ampls = np.concatenate(flux_pulse_ma_2.sweep_points_2D[-1:0:-1],
                                   flux_pulse_ma_1.sweep_points_2D)

            instrument_settings = flux_pulse_ma_1.data_file['Instrument settings']
            qubit_attrs = instrument_settings[self.name].attrs
            E_c = kw.pop('E_c', qubit_attrs.get('E_c', 0.3e9))
            f_max = kw.pop('f_max', qubit_attrs.get('f_max', self.f_qubit()))
            V_per_phi0 = kw.pop('V_per_phi0',
                                qubit_attrs.get('V_per_phi0', 1.))
            dac_sweet_spot = kw.pop('dac_sweet_spot',
                                    qubit_attrs.get('dac_sweet_spot', 0))


            freqs = f_max - phases/(2*np.pi*pulse_length)

            fit_res = ma.fit_qubit_frequency(ampls, freqs, E_c=E_c, f_max=f_max,
                                             V_per_phi0=V_per_phi0,
                                             dac_sweet_spot=dac_sweet_spot
                                             )
            print(fit_res.fit_report())

            if plot and ampls_bidirectional:
                fit_res.plot()
            if ampls_bidirectional:
                return fit_res


    def calibrate_CPhase_dynamic_phases(self,
                                        flux_pulse_length=None,
                                        flux_pulse_amp=None,
                                        flux_pulse_delay=None,
                                        thetas=None,
                                        X90_separation=None,
                                        flux_pulse_channel=None,
                                        MC=None, label=None,
                                        analyze=True, update=True, **kw):
        """
        CPhase dynamic phase calibration

        does a measuement of the type:

                      X90_separation
                < -- ---- ----------- --->
                |X90|  --------------     |X90|  ---  |RO|
                       | fluxpulse |

        where  the angle of the second X90 pulse is swept for
        the flux pulse amplitude  in [0,cphase_ampl].

        Args:
            MC: measurement control object
            thetas: numpy array with angles (in rad) for the Ramsey type
            ampls: numpy array with amplitudes (in V) swept through
                as flux pulse amplitudes
            analyze: bool, if True, then the measured data
                gets analyzed (


        """

        if MC is None:
            MC = self.MC

        if flux_pulse_amp is None:
            flux_pulse_amp = self.flux_pulse_amp()
            logging.warning('flux_pulse_amp is not specified. Using the value'
                            'in the flux_pulse_amp parameter.')
        if flux_pulse_length is None:
            flux_pulse_length = self.flux_pulse_length()
            logging.warning('flux_pulse_length is not specified. Using the value'
                            'in the flux_pulse_length parameter.')
        if flux_pulse_delay is None:
            flux_pulse_delay = self.flux_pulse_delay()
            logging.warning('flux_pulse_delay is not specified. Using the value'
                            'in the flux_pulse_delay parameter.')
        if flux_pulse_channel is None:
            flux_pulse_channel = self.flux_pulse_channel()
            logging.warning('flux_pulse_channel is not specified. Using the value'
                            'in the flux_pulse_channel parameter.')
        if thetas is None:
            thetas = np.linspace(0, 4*np.pi, 16)
            print('Sweeping over phases thata=np.linspace(0, 4*np.pi, 16).')

        if label is None:
            label = 'Dynamic_phase_measurement_{}_{}_filter'.format(
                self.name, self.flux_pulse_channel())

        self.measure_dynamic_phase(flux_pulse_length=flux_pulse_length,
                                   flux_pulse_amp=flux_pulse_amp,
                                   flux_pulse_channel=flux_pulse_channel,
                                   flux_pulse_delay=flux_pulse_delay,
                                   X90_separation=X90_separation,
                                   thetas=thetas,
                                   MC=MC,
                                   label=label)

        if analyze:
            MA = ma.Dynamic_phase_Analysis(
                    label=label,
                    X90_separation=X90_separation,
                    flux_pulse_amp=flux_pulse_amp,
                    flux_pulse_length=flux_pulse_length,
                    gauss_sigma=self.gauss_sigma(),
                    nr_gauss_sigma=self.nr_sigma(),
                    qb_name=self.name,
                    auto=False, **kw)

            dynamic_phase = MA.dyn_phase
            print('fitted dynamic phase on {}: {:0.3f} [deg]'.format(self.name,
                                                                dynamic_phase))
            if update:
                try:
                    self.dynamic_phase(dynamic_phase)
                except Exception:
                    logging.warning('Could not update '
                                    '{}.dynamic_phase().'.format(self.name))

            return dynamic_phase
        else:
            return

    def measure_cphase(self, qb_target, amps, lengths,
                       phases=None, spacing=100e-9,
                       MC=None, cal_points=None, plot=False,
                       return_population_loss=False,
                       upload_AWGs='all',
                       upload_channels='all',
                       prepare_for_timedomain=True
                       ):
        '''
        method to measure the phase acquired during a flux pulse conditioned on the state
        of the control qubit (self).
        In this measurement, the phase from two Ramsey type measurements
        on qb_target is measured, once with the control qubit in the excited state and once
        in the ground state. The conditional phase is calculated as the difference.


        Args:
            qb_target (QuDev_transmon): target qubit / non-fluxed qubit
            amps (list): list or array of flux pulse amplitudes
            lengths (list):  list or array of flux pulse lengths (must have same dimension as
                             amps)
            phases (array): phases used for the Ramsey type phase sweep
            spacing (float): spacing between flux pulse and Ramsey pulses in s
            MC (optional): measurement control
            cal_points (bool): if True, calibration points are measured
            plot (bool): if true, the phase fit is shown
            return_population_loss: if true, the population loss (loss of contrast when having
                                    the control qubit in the excited state is returned)
            upload_AWGs (list): list of the AWGs to be uploaded
            upload_channels (list): list of channels to be uploaded
            prepare_for_timedomain (bool): if False, the self.prepare_for_timedomain()
                                           is NOT called

        Returns:
            cphases (numpy array): array of the conditional phases measured at
                                    (amps[i], lengths[i])
        '''
        if len(amps) != len(lengths):
            logging.warning('amps and lengths must have the same '
                            'dimension.')

        if MC is None:
            MC = self.MC
        if phases is None:
            phases = np.linspace(0, 2*np.pi, 16, endpoint=False)
            phases = np.concatenate((phases,phases))

        cphase_all = []
        population_loss_all = []
        for amp, length in zip(amps, lengths):
            self.flux_pulse_amp(amp)
            self.flux_pulse_length(length)

            s1 = awg_swf.Flux_pulse_CPhase_meas_hard_swf(
                qb_control=self,
                qb_target=qb_target,
                sweep_mode='phase',
                cal_points=cal_points,
                reference_measurements=True,
                spacing=spacing,
                upload=False,
                upload_AWGs=upload_AWGs,
                upload_channels=upload_channels
            )
            s2 = awg_swf.Flux_pulse_CPhase_meas_2D(self, qb_target, s1,
                                                   sweep_mode='amplitude')
            if prepare_for_timedomain:
                qb_target.prepare_for_timedomain()

            MC.set_sweep_function(s1)
            MC.set_sweep_points(phases)
            MC.set_sweep_function_2D(s2)
            MC.set_sweep_points_2D([amp])
            MC.set_detector_function(self.int_avg_det)
            MC.run_2D('CPhase_measurement_{}_{}'.format(self.name,
                                                        qb_target.name))

            # ma.TwoD_Analysis(close_file=True)
            flux_pulse_ma = ma.Fluxpulse_Ramsey_2D_Analysis(
                label='CPhase_measurement_{}_{}'.format(self.name,
                                                        qb_target.name),
                qb_name=self.name, cal_points=cal_points,
                reference_measurements=True
            )
            fitted_phases, fitted_amps = \
                flux_pulse_ma.fit_all(plot=plot,
                                      cal_points=cal_points,
                                      return_ampl=True,
                                      )

            fitted_phases_exited = fitted_phases[:: 2]
            fitted_phases_ground = fitted_phases[1:: 2]

            cphases = fitted_phases_exited - fitted_phases_ground

            fitted_amps_exited = fitted_amps[:: 2]
            fitted_amps_ground = fitted_amps[1:: 2]

            pop_loss = np.abs(fitted_amps_ground - fitted_amps_exited) \
                       /fitted_amps_ground

            cphase_all.append(cphases[0])
            population_loss_all.append(pop_loss[0])

        plot_title = 'fitted CPhase: {:.3f} deg at' \
                     ' amp={:.2f}mV,' \
                     ' length={:.3f}ns'.format(cphases[0]/np.pi*180,
                                                amps[0]/1e-3, lengths[0]/1e-9)
        flux_pulse_ma.fit_all(plot=plot,
                              cal_points=cal_points,
                              return_ampl=True,
                              save_plot=True,
                              plot_title=plot_title
                              )
        cphase_all = np.array(cphase_all)
        population_loss_all = np.array(population_loss_all)
        if return_population_loss:
            return cphase_all, population_loss_all
        else:
            return cphase_all


    def measure_flux_pulse_scope(self, freqs, delays, pulse_length=None, pulse_amp=None,
                                 pulse_delay=None,
                                 MC=None,
                                 ):
        '''
        flux pulse scope measurement used to determine the shape of flux pulses
        set up as a 2D measurement (delay and drive pulse frequecy are being swept)
        pulse sequence:
                      <- delay ->
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- fluxpulse ----- |


        Args:
            freqs (numpy array): array of drive frequencies
            delays (numpy array): array of delays of the drive pulse w.r.t the flux pulse
            pulse_length (float): flux pulse length (if not specified, the
                                    self.flux_pulse_length() is taken)
            pulse_amp (float): flux pulse amplitude  (if not specified, the
                                    self.flux_pulse_amp() is taken)
            pulse_delay (float): flux pulse delay
            MC (MeasurementControl): if None, then the self.MC is taken

        Returns: None

        '''
        if pulse_length is not None:
            self.flux_pulse_length(pulse_length)
        if pulse_amp is not None:
            self.flux_pulse_amp(pulse_amp)
        if pulse_delay is not None:
            self.flux_pulse_delay(pulse_delay)

        if MC is None:
            MC = self.MC

        self.prepare_for_timedomain()

        s1 = awg_swf.Fluxpulse_scope_swf(self)
        s2 = awg_swf.Fluxpulse_scope_drive_freq_sweep(self)

        MC.set_sweep_function(s1)
        MC.set_sweep_points(delays)
        MC.set_sweep_function_2D(s2)
        MC.set_sweep_points_2D(freqs)
        MC.set_detector_function(self.int_avg_det)
        MC.run_2D('Flux_scope_{}'.format(self.name))

        ma.MeasurementAnalysis(TwoD=True)


def add_CZ_pulse(qbc, qbt):
    """
    Args:
        qbc: Control qubit. A QudevTransmon object corresponding to the qubit
             that we apply the flux pulse on.
        qbt: Target qubit. A QudevTransmon object corresponding to the qubit
             we induce the conditional phase on.
    """

    # add flux pulse parameters
    op_name = 'CZ ' + qbt.name
    ps_name = 'CZ_' + qbt.name

    if np.any([op_name == i for i in qbc.get_operation_dict().keys()]):
        # do not try to add it again if operation already exists
        raise ValueError('Operation {} already exists.'.format(op_name))
    else:
        qbc.add_operation(op_name)

        qbc.add_pulse_parameter(op_name, ps_name + '_target',  'qb_target',
                                initial_value=qbt.name,
                                vals=vals.Enum(qbt.name))
        qbc.add_pulse_parameter(op_name, ps_name + '_pulse_type', 'pulse_type',
                                initial_value='BufferedSquarePulse',
                                vals=vals.Enum('BufferedSquarePulse'))
        qbc.add_pulse_parameter(op_name, ps_name + '_channel', 'channel',
                                initial_value='', vals=vals.Strings())
        qbc.add_pulse_parameter(op_name, ps_name + '_amp', 'amplitude',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_length', 'pulse_length',
                                initial_value=0, vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_buf_start',
                                'buffer_length_start', initial_value=0,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_buf_end',
                                'buffer_length_end', initial_value=0,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_delay', 'pulse_delay',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_dynamic_phases',
                                'basis_rotation', initial_value={},
                                vals=vals.Dict())














