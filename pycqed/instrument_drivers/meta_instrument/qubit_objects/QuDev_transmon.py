import logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals

from pycqed.analysis_v2.readout_analysis import Singleshot_Readout_Analysis_Qutrit
from pycqed.measurement import detector_functions as det
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_swf2
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration_points import CalibrationPoints
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.analysis_v3 import pipeline_analysis as pla
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import timedomain_analysis as tda
import pycqed.analysis.randomized_benchmarking_analysis as rbma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.utilities.general import temporary_value
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit
from pycqed.measurement import optimization as opti
from pycqed.measurement import mc_parameter_wrapper
import pycqed.analysis_v2.spectroscopy_analysis as sa
from pycqed.utilities import math

try:
    import pycqed.simulations.readout_mode_simulations_for_CLEAR_pulse \
        as sim_CLEAR
except ModuleNotFoundError:
    log.warning('"readout_mode_simulations_for_CLEAR_pulse" not imported.')

class QuDev_transmon(Qubit):
    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        self.add_parameter('instr_mc', 
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_ge_lo', 
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_pulsar',
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_uhf',
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_ro_lo', 
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_trigger', 
            parameter_class=InstrumentRefParameter)
       
        # device parameters for user only
        # could be cleaned up
        self.add_parameter('f_RO_resonator', label='RO resonator frequency',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('f_RO_purcell', label='RO purcell filter frequency',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_purcell_kappa', label='Purcell filter kappa',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_J_coupling', label='J coupling of RO resonator'
                                                  'and purcell filter',
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
        self.add_parameter('chi', unit='Hz', parameter_class=ManualParameter,
                           label='Chi')
        
        # readout pulse parameters
        self.add_parameter('ro_freq', unit='Hz', 
                           parameter_class=ManualParameter, 
                           label='Readout frequency')
        self.add_parameter('ro_I_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the readout I channel')
        self.add_parameter('ro_Q_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the readout Q channel')
        self.add_parameter('ro_lo_power', unit='dBm',
                           parameter_class=ManualParameter,
                           label='Readout pulse upconversion mixer LO power')
        self.add_operation('RO')
        self.add_pulse_parameter('RO', 'ro_pulse_type', 'pulse_type',
                                 vals=vals.Enum('GaussFilteredCosIQPulse',
                                                'GaussFilteredCosIQPulseMultiChromatic'),
                                 initial_value='GaussFilteredCosIQPulse')
        self.add_pulse_parameter('RO', 'ro_I_channel', 'I_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('RO', 'ro_Q_channel', 'Q_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('RO', 'ro_amp', 'amplitude',
                                 initial_value=0.001,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_length', 'pulse_length',
                                 initial_value=2e-6, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_delay', 'pulse_delay',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_mod_freq', 'mod_frequency',
                                 initial_value=100e6,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_phase', 'phase',
                                 initial_value=0,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_phi_skew', 'phi_skew',
                                 initial_value=0,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_alpha', 'alpha',
                                 initial_value=1,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_sigma', 
                                 'gaussian_filter_sigma',
                                 initial_value=10e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_nr_sigma', 'nr_sigma',
                                 initial_value=5, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_phase_lock', 'phase_lock',
                                 initial_value=True, vals=vals.Bool())                                 
        self.add_pulse_parameter('RO', 'ro_basis_rotation',
                                 'basis_rotation', initial_value={},
                                 docstring='Dynamic phase acquired by other '
                                           'qubits due to a measurement tone on'
                                           ' this qubit.',
                                 label='RO pulse basis rotation dictionary',
                                 vals=vals.Dict())

        # acquisition parameters
        self.add_parameter('acq_I_channel', initial_value=0,
                           vals=vals.Enum(0, 1, 2, 3, 4, 5, 6, 7, 8),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_Q_channel', initial_value=1,
                           vals=vals.Enum(0, 1, 2, 3, 4, 5, 6, 7, 8, None),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_averages', initial_value=1024,
                           vals=vals.Ints(0, 1000000),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_shots', initial_value=4094,
                           docstring='Number of single shot measurements to do'
                                     'in single shot experiments.',
                           vals=vals.Ints(0, 1048576),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_length', initial_value=2.2e-6,
                           vals=vals.Numbers(min_value=1e-8,
                                             max_value=4097/1.2e9),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_IQ_angle', initial_value=0,
                           docstring='The phase of the integration weights '
                                     'when using SSB, DSB or square_rot '
                                     'integration weights', 
                                     label='Acquisition IQ angle', unit='rad',
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_I', vals=vals.Arrays(),
                           label='Optimized weights for I channel',
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_Q', vals=vals.Arrays(),
                           label='Optimized weights for Q channel',
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_type', initial_value='SSB',
                           vals=vals.Enum('SSB', 'DSB', 'optimal',
                                          'square_rot', 'manual',
                                          'optimal_qutrit'),
                           docstring=(
                               'Determines what type of integration weights to '
                               'use: \n\tSSB: Single sideband demodulation\n\t'
                               'DSB: Double sideband demodulation\n\toptimal: '
                               'waveforms specified in "ro_acq_weight_func_I" '
                               'and "ro_acq_weight_func_Q"\n\tsquare_rot: uses '
                               'a single integration channel with boxcar '
                               'weights'),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_I2', vals=vals.Arrays(),
                           label='Optimized weights for second integration '
                                 'channel I',
                           docstring=("Used for double weighted integration "
                                      "during qutrit readout"),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_Q2', vals=vals.Arrays(),
                           label='Optimized weights for second integration '
                                 'channel Q',
                           docstring=("Used for double weighted integration "
                                      "during qutrit readout"),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_basis', vals=vals.Lists(),
                           label="weight basis used",
                           docstring=("Used to log the weights basis for "
                                      "integration during qutrit readout. E.g."
                                      " ['ge', 'gf'] or ['ge', 'ortho']."),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_classifier_params', vals=vals.Dict(),
                           label='Parameters for the qutrit classifier.',
                           docstring=("Used in the int_avg_classif_det to "
                                      "classify single shots into g, e, f."),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_state_prob_mtx', vals=vals.Arrays(),
                           label='SSRO correction matrix.',
                           docstring=("Matrix of measured vs prepared qubit "
                                      "states."),
                           parameter_class=ManualParameter)

        # qubit drive pulse parameters
        self.add_parameter('ge_freq', label='Qubit drive frequency', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('ge_lo_power', unit='dBm',
                           parameter_class=ManualParameter,
                           label='Qubit drive pulse mixer LO power')
        self.add_parameter('ge_I_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the drive line I channel')
        self.add_parameter('ge_Q_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the drive line Q channel')
        # add drive pulse parameters
        self.add_operation('X180')
        self.add_pulse_parameter('X180', 'ge_pulse_type', 'pulse_type',
                                 initial_value='SSB_DRAG_pulse', 
                                 vals=vals.Enum('SSB_DRAG_pulse'))
        self.add_pulse_parameter('X180', 'ge_I_channel', 'I_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180', 'ge_Q_channel', 'Q_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180', 'ge_amp180', 'amplitude',
                                 initial_value=0.001, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'ge_amp90_scale', 'amp90_scale',
                                 initial_value=0.5, vals=vals.Numbers(0, 1))
        self.add_pulse_parameter('X180', 'ge_delay', 'pulse_delay',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'ge_sigma', 'sigma',
                                 initial_value=10e-9, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'ge_nr_sigma', 'nr_sigma',
                                 initial_value=5, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'ge_motzoi', 'motzoi',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'ge_mod_freq', 'mod_frequency',
                                 initial_value=-100e6, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'ge_phi_skew', 'phi_skew',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'ge_alpha', 'alpha',
                                 initial_value=1, vals=vals.Numbers())
        self.add_pulse_parameter('X180', 'ge_X_phase', 'phase',
                                 initial_value=0, vals=vals.Numbers())
        
        # qubit 2nd excitation drive pulse parameters
        self.add_parameter('ef_freq', label='Qubit ef drive frequency', 
                           unit='Hz', initial_value=0, 
                           parameter_class=ManualParameter)
        self.add_operation('X180_ef')
        self.add_pulse_parameter('X180_ef', 'ef_pulse_type', 'pulse_type',
                                 initial_value='SSB_DRAG_pulse', 
                                 vals=vals.Enum('SSB_DRAG_pulse'))
        self.add_pulse_parameter('X180_ef', 'ef_amp180', 'amplitude',
                                 initial_value=0.001, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'ef_amp90_scale', 'amp90_scale',
                                 initial_value=0.5, vals=vals.Numbers(0, 1))
        self.add_pulse_parameter('X180_ef', 'ef_delay', 'pulse_delay',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'ef_sigma', 'sigma',
                                 initial_value=10e-9, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'ef_nr_sigma', 'nr_sigma',
                                 initial_value=5, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'ef_motzoi', 'motzoi',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'ef_X_phase', 'phase',
                                 initial_value=0, vals=vals.Numbers())

        # add qubit spectroscopy parameters
        self.add_parameter('spec_power', unit='dBm', initial_value=-20,
                           parameter_class=ManualParameter,
                           label='Qubit spectroscopy power')
        self.add_operation('Spec')
        self.add_pulse_parameter('Spec', 'spec_pulse_type', 'pulse_type',
                                 initial_value='SquarePulse',
                                 vals=vals.Enum('SquarePulse'))
        self.add_pulse_parameter('Spec', 'spec_marker_channel', 'channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('Spec', 'spec_marker_amp', 'amplitude',
                                 vals=vals.Numbers(), initial_value=1)
        self.add_pulse_parameter('Spec', 'spec_marker_length', 'length',
                                 initial_value=5e-6, vals=vals.Numbers())
        self.add_pulse_parameter('Spec', 'spec_marker_delay', 'pulse_delay', 
                                 vals=vals.Numbers(), initial_value=0)

        # dc flux parameters
        self.add_parameter('dc_flux_parameter', initial_value=None,
                           label='QCoDeS parameter to sweep the dc flux',
                           parameter_class=ManualParameter)

        # Pulse preparation parameters
        DEFAULT_PREP_PARAMS = dict(preparation_type='wait',
                                   post_ro_wait=1e-6, reset_reps=1,
                                   final_reset_pulse=True,
                                   threshold_mapping={
                                       self.name: {0: 'g', 1: 'e'}})

        self.add_parameter('preparation_params', parameter_class=ManualParameter,
                            initial_value=DEFAULT_PREP_PARAMS, vals=vals.Dict())


    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name}

    def update_detector_functions(self):
        if self.acq_Q_channel() is None or \
           self.acq_weights_type() not in ['SSB', 'DSB', 'optimal_qutrit']:
            channels = [self.acq_I_channel()]
        else:
            channels = [self.acq_I_channel(), self.acq_Q_channel()]

        self.int_log_det = det.UHFQC_integration_logging_det(
            UHFQC=self.instr_uhf.get_instr(), 
            AWG=self.instr_pulsar.get_instr(), 
            channels=channels, nr_shots=self.acq_shots(),
            integration_length=self.acq_length(), 
            result_logging_mode='raw')

        self.int_avg_classif_det = det.UHFQC_classifier_detector(
            UHFQC=self.instr_uhf.get_instr(), 
            AWG=self.instr_pulsar.get_instr(),  
            channels=channels, nr_shots=self.acq_averages(),
            integration_length=self.acq_length(),
            get_values_function_kwargs={
                'classifier_params': self.acq_classifier_params(),
                'state_prob_mtx': self.acq_state_prob_mtx()
            })

        self.int_avg_det = det.UHFQC_integrated_average_detector(
            UHFQC=self.instr_uhf.get_instr(), 
            AWG=self.instr_pulsar.get_instr(), 
            channels=channels, nr_averages=self.acq_averages(),
            integration_length=self.acq_length(), 
            result_logging_mode='raw')

        self.dig_avg_det = det.UHFQC_integrated_average_detector(
            UHFQC=self.instr_uhf.get_instr(), 
            AWG=self.instr_pulsar.get_instr(), 
            channels=channels, nr_averages=self.acq_averages(),
            integration_length=self.acq_length(),
            result_logging_mode='digitized')

        nr_samples = int(self.acq_length() *
                         self.instr_uhf.get_instr().clock_freq())
        self.inp_avg_det = det.UHFQC_input_average_detector(
            UHFQC=self.instr_uhf.get_instr(), 
            AWG=self.instr_pulsar.get_instr(), 
            nr_averages=self.acq_averages(),
            nr_samples=nr_samples)

        self.dig_log_det = det.UHFQC_integration_logging_det(
            UHFQC=self.instr_uhf.get_instr(), 
            AWG=self.instr_pulsar.get_instr(), 
            channels=channels, nr_shots=self.acq_shots(),
            integration_length=self.acq_length(), 
            result_logging_mode='digitized')

        self.int_avg_det_spec = det.UHFQC_integrated_average_detector(
            UHFQC=self.instr_uhf.get_instr(), 
            AWG=self.instr_uhf.get_instr(), 
            channels=[self.acq_I_channel(), self.acq_Q_channel()], 
            nr_averages=self.acq_averages(),
            integration_length=self.acq_length(),
            result_logging_mode='raw', real_imag=False, single_int_avg=True)

    def prepare(self, drive='timedomain'):
        # configure readout local oscillators
        lo = self.instr_ro_lo
        if lo() is not None:
            lo.get_instr().pulsemod_state('Off')
            lo.get_instr().power(self.ro_lo_power())
            # in case of multichromatic readout, take first ro freq, else just
            # wrap the frequency in a list and take the first
            if np.ndim(self.ro_freq()) == 0:
                ro_freq = [self.ro_freq()]
            else:
                ro_freq = self.ro_freq()
            if np.ndim(self.ro_mod_freq()) == 0:
                ro_mod_freq = [self.ro_mod_freq()]
            else:
                ro_mod_freq = self.ro_mod_freq()
            lo.get_instr().frequency(ro_freq[0] - ro_mod_freq[0])

            lo.get_instr().on()

        # configure qubit drive local oscillator
        lo = self.instr_ge_lo
        if lo() is not None:
            if drive is None:
                lo.get_instr().off()
            elif drive == 'continuous_spec':
                lo.get_instr().pulsemod_state('Off')
                lo.get_instr().power(self.spec_power())
                lo.get_instr().frequency(self.ge_freq())
                lo.get_instr().on()
            elif drive == 'pulsed_spec':
                lo.get_instr().pulsemod_state('On')
                lo.get_instr().power(self.spec_power())
                lo.get_instr().frequency(self.ge_freq())
                lo.get_instr().on()
            elif drive == 'timedomain':
                lo.get_instr().pulsemod_state('Off')
                lo.get_instr().power(self.ge_lo_power())
                lo.get_instr().frequency(self.ge_freq() - self.ge_mod_freq())
                lo.get_instr().on()
            else:
                raise ValueError("Invalid drive parameter '{}'".format(drive)
                                 + ". Valid options are None, 'continuous_spec"
                                 + "', 'pulsed_spec' and 'timedomain'.")

        # set awg channel dc offsets
        offset_list = [('ro_I_channel', 'ro_I_offset'), 
                       ('ro_Q_channel', 'ro_Q_offset')]
        if drive == 'timedomain':
            offset_list += [('ge_I_channel', 'ge_I_offset'), 
                            ('ge_Q_channel', 'ge_Q_offset')]
        for channel_par, offset_par in offset_list:
            self.instr_pulsar.get_instr().set(
                self.get(channel_par) + '_offset', self.get(offset_par))

        # other preparations
        self.update_detector_functions()
        self.set_readout_weights()

    def set_readout_weights(self, weights_type=None, f_mod=None):
        if weights_type is None:
            weights_type = self.acq_weights_type()
        if f_mod is None:
            f_mod = self.ro_mod_freq()
        if weights_type == 'manual':
            pass
        elif weights_type == 'optimal':
            if (self.acq_weights_I() is None or self.acq_weights_Q() is None):
                log.warning('Optimal weights are None, not setting '
                                'integration weights')
                return
            # When optimal weights are used, only the RO I weight
            # channel is used
            self.instr_uhf.get_instr().set('qas_0_integration_weights_{}_real'.format(
                self.acq_I_channel()), self.acq_weights_I().copy())
            self.instr_uhf.get_instr().set('qas_0_integration_weights_{}_imag'.format(
                self.acq_I_channel()), self.acq_weights_Q().copy())
            self.instr_uhf.get_instr().set('qas_0_rotations_{}'.format(
                self.acq_I_channel()), 1.0-1.0j)
        elif weights_type == 'optimal_qutrit':
            for w_f in [self.acq_weights_I, self.acq_weights_Q,
                        self.acq_weights_I2, self.acq_weights_Q2]:
                if w_f() is None:
                    log.warning('The optimal weights {} are None. '
                                    '\nNot setting integration weights.'
                                    .format(w_f.name))
                    return
            # if all weights are not None, set first integration weights (real 
            # and imag) on channel I amd second integration weights on channel 
            # Q.
            self.instr_uhf.get_instr().set('qas_0_integration_weights_{}_real'.format(
                self.acq_I_channel()),
                self.acq_weights_I().copy())
            self.instr_uhf.get_instr().set('qas_0_integration_weights_{}_imag'.format(
                self.acq_I_channel()),
                self.acq_weights_Q().copy())
            self.instr_uhf.get_instr().set('qas_0_integration_weights_{}_real'.format(
                self.acq_Q_channel()),
                self.acq_weights_I2().copy())
            self.instr_uhf.get_instr().set('qas_0_integration_weights_{}_imag'.format(
                self.acq_Q_channel()),
                self.acq_weights_Q2().copy())

            self.instr_uhf.get_instr().set('qas_0_rotations_{}'.format(
                self.acq_I_channel()), 1.0-1.0j)
            self.instr_uhf.get_instr().set('qas_0_rotations_{}'.format(
                self.acq_Q_channel()), 1.0-1.0j)
                
        else:
            tbase = np.arange(0, 4097 / 1.8e9, 1 / 1.8e9)
            theta = self.acq_IQ_angle()
            cosI = np.array(np.cos(2 * np.pi * f_mod * tbase + theta))
            sinI = np.array(np.sin(2 * np.pi * f_mod * tbase + theta))
            c1 = self.acq_I_channel()
            c2 = self.acq_Q_channel()
            uhf = self.instr_uhf.get_instr()
            if weights_type == 'SSB':
                uhf.set('qas_0_integration_weights_{}_real'.format(c1), cosI)
                uhf.set('qas_0_rotations_{}'.format(c1), 1.0+1.0j)
                uhf.set('qas_0_integration_weights_{}_real'.format(c2), sinI)
                uhf.set('qas_0_rotations_{}'.format(c2), 1.0-1.0j)
                uhf.set('qas_0_integration_weights_{}_imag'.format(c1), sinI)
                uhf.set('qas_0_integration_weights_{}_imag'.format(c2), cosI)
            elif weights_type == 'DSB':
                uhf.set('qas_0_integration_weights_{}_real'.format(c1), cosI)
                uhf.set('qas_0_rotations_{}'.format(c1), 1.0+0j)
                uhf.set('qas_0_integration_weights_{}_real'.format(c2), sinI)
                uhf.set('qas_0_rotations_{}'.format(c2), 1.0+0j)
            elif weights_type == 'square_rot':
                uhf.set('qas_0_integration_weights_{}_real'.format(c1), cosI)
                uhf.set('qas_0_rotations_{}'.format(c1), 1.0+1.0j)
                uhf.set('qas_0_integration_weights_{}_imag'.format(c1), sinI)
            else:
                raise KeyError('Invalid weights type: {}'.format(weights_type))

    def get_spec_pars(self):
        return self.get_operation_dict()['Spec ' + self.name]

    def get_ro_pars(self):
        return self.get_operation_dict()['RO ' + self.name]
    
    def get_acq_pars(self):
        return self.get_operation_dict()['Acq ' + self.name]

    def get_ge_pars(self):
        return self.get_operation_dict()['X180 ' + self.name]

    def get_ef_pars(self):
        return self.get_operation_dict()['X180_ef ' + self.name]

    def get_operation_dict(self, operation_dict=None):
        if operation_dict is None:
            operation_dict = {}
        operation_dict = super().get_operation_dict(operation_dict)
        operation_dict['Spec ' + self.name]['operation_type'] = 'Other'
        operation_dict['RO ' + self.name]['operation_type'] = 'RO'
        operation_dict['X180 ' + self.name]['operation_type'] = 'MW'
        operation_dict['X180_ef ' + self.name]['operation_type'] = 'MW'
        operation_dict['X180 ' + self.name]['basis'] = self.name
        operation_dict['X180_ef ' + self.name]['basis'] = self.name + \
                                                                   '_ef'
        operation_dict['X180_ef ' + self.name]['I_channel'] = \
            operation_dict['X180 ' + self.name]['I_channel']
        operation_dict['X180_ef ' + self.name]['Q_channel'] = \
            operation_dict['X180 ' + self.name]['Q_channel']
        operation_dict['X180_ef ' + self.name]['phi_skew'] = \
            operation_dict['X180 ' + self.name]['phi_skew']
        operation_dict['X180_ef ' + self.name]['alpha'] = \
            operation_dict['X180 ' + self.name]['alpha']
        operation_dict['Acq ' + self.name] = deepcopy(
            operation_dict['RO ' + self.name])
        operation_dict['Acq ' + self.name]['amplitude'] = 0

        if self.ef_freq() == 0:
            operation_dict['X180_ef ' + self.name]['mod_frequency'] = None
        else:
            operation_dict['X180_ef ' + self.name]['mod_frequency'] = \
                self.ef_freq() - self.ge_freq() + self.ge_mod_freq()

        operation_dict.update(add_suffix_to_dict_keys(
            sq.get_pulse_dict_from_pars(
                operation_dict['X180 ' + self.name]), ' ' + self.name))
        operation_dict.update(add_suffix_to_dict_keys(
            sq.get_pulse_dict_from_pars(
                operation_dict['X180_ef ' + self.name]), '_ef ' + self.name))
        if np.ndim(self.ro_freq()) != 0:
            delta_freqs = np.diff(self.ro_freq(), prepend=self.ro_freq()[0])
            mods = [self.ro_mod_freq() + d for d in delta_freqs]
            operation_dict['RO ' + self.name]['mod_frequency'] = mods

        return operation_dict

    def swf_ro_freq_lo(self):
        return swf.Offset_Sweep(
            self.instr_ro_lo.get_instr().frequency, 
            -self.ro_mod_freq(),
            name='Readout frequency', 
            parameter_name='Readout frequency')
    def swf_ro_mod_freq(self):
        return swf.Offset_Sweep(
            self.ro_mod_freq,
            self.instr_ro_lo.get_instr().frequency(),
            name='Readout frequency',
            parameter_name='Readout frequency')
    def measure_resonator_spectroscopy(self, freqs, sweep_points_2D=None,
                                       sweep_function_2D=None,
                                       trigger_separation=3e-6, 
                                       upload=True, analyze=True, 
                                       close_fig=True, label=None):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """
        if np.any(freqs < 500e6):
            log.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))
    
        if label is None:
            if sweep_function_2D is not None:
                label = 'resonator_scan_2d' + self.msmt_suffix
            else:
                label = 'resonator_scan' + self.msmt_suffix

        self.prepare(drive=None)
        if upload:
            sq.pulse_list_list_seq([[self.get_ro_pars()]])

        MC = self.instr_mc.get_instr()
        MC.set_sweep_function(self.swf_ro_freq_lo())
        if sweep_function_2D is not None:
            MC.set_sweep_function_2D(sweep_function_2D)
            mode = '2D'
        else:
            mode = '1D'
        MC.set_sweep_points(freqs)
        if sweep_points_2D is not None:
            MC.set_sweep_points_2D(sweep_points_2D)
        MC.set_detector_function(self.int_avg_det_spec)

        with temporary_value(self.instr_trigger.get_instr().pulse_period, 
                             trigger_separation):
            self.instr_pulsar.get_instr().start(exclude=[self.instr_uhf()])
            MC.run(name=label, mode=mode)
            self.instr_pulsar.get_instr().stop()

        if analyze:
            ma.MeasurementAnalysis(close_fig=close_fig, qb_name=self.name, 
                                   TwoD=(mode == '2D'))

    def measure_qubit_spectroscopy(self, freqs, sweep_points_2D=None,
            sweep_function_2D=None, pulsed=True, trigger_separation=13e-6, 
            upload=True, analyze=True, close_fig=True, label=None):
        """ Varies qubit drive frequency and measures the resonator
        transmittance """
        if np.any(freqs < 500e6):
            log.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))
        if pulsed:
            if label is None:
                if sweep_function_2D is not None:
                    label = 'pulsed_spec_2d' + self.msmt_suffix
                else:
                    label = 'pulsed_spec' + self.msmt_suffix
            self.prepare(drive='pulsed_spec')
            if upload:
                sq.pulse_list_list_seq([[self.get_spec_pars(),
                                         self.get_ro_pars()]])
        else:
            if label is None:
                if sweep_function_2D is not None:
                    label = 'continuous_spec_2d' + self.msmt_suffix
                else:
                    label = 'continuous_spec' + self.msmt_suffix
            self.prepare(drive='continuous_spec')
            if upload:
                sq.pulse_list_list_seq([[self.get_ro_pars()]])
        
        MC = self.instr_mc.get_instr()
        MC.set_sweep_function(self.instr_ge_lo.get_instr().frequency)
        if sweep_function_2D is not None:
            MC.set_sweep_function_2D(sweep_function_2D)
            mode = '2D'
        else:
            mode = '1D'
        MC.set_sweep_points(freqs)
        if sweep_points_2D is not None:
            MC.set_sweep_points_2D(sweep_points_2D)
        MC.set_detector_function(self.int_avg_det_spec)

        with temporary_value(self.instr_trigger.get_instr().pulse_period, 
                             trigger_separation):
            self.instr_pulsar.get_instr().start(exclude=[self.instr_uhf()])
            MC.run(name=label, mode=mode)
            self.instr_pulsar.get_instr().stop()

        if analyze:
            ma.MeasurementAnalysis(close_fig=close_fig, qb_name=self.name, 
                                   TwoD=(mode == '2D'))

    def measure_rabi(self, amps, analyze=True, upload=True, label=None, n=1,
                     last_ge_pulse=False, n_cal_points_per_state=2,
                     cal_states='auto', for_ef=False, classified_ro=False,
                     prep_params=None, exp_metadata=None):

        """
        Varies the amplitude of the qubit drive pulse and measures the readout
        resonator transmission.

        Args:
            amps            the array of drive pulse amplitudes
            analyze         whether to create a (base) MeasurementAnalysis
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
        if prep_params is None:
            prep_params = self.preparation_params()

        # Define the measurement label
        if label is None:
            label = 'Rabi_ef' if for_ef else 'Rabi'
            if n != 1:
                label += f'-n{n}'
            if classified_ro:
                label += '_classified'
            if 'active' in prep_params['preparation_type']:
                label += '_reset'
            label += self.msmt_suffix

        # Prepare the physical instruments for a time domain measurement
        self.prepare(drive='timedomain')

        MC = self.instr_mc.get_instr()
        cal_states = CalibrationPoints.guess_cal_states(cal_states, for_ef)
        cp = CalibrationPoints.single_qubit(self.name, cal_states,
                                            n_per_state=n_cal_points_per_state)
        seq, sweep_points = sq.rabi_seq_active_reset(
            amps=amps, qb_name=self.name, cal_points=cp, n=n, for_ef=for_ef,
            operation_dict=self.get_operation_dict(), upload=False,
            last_ge_pulse=last_ge_pulse, prep_params=prep_params)
        # Specify the sweep function, the sweep points,
        # and the detector function, and run the measurement
        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Amplitude', unit='V'))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_classif_det if classified_ro else
                                 self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: amps},
                             'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'rotate': False if classified_ro else
                                len(cp.states) != 0,
                             'last_ge_pulses': [last_ge_pulse],
                             'data_to_fit': {self.name: 'pf' if for_ef else 'pe'},
                             "sweep_name": "Amplitude",
                             "sweep_unit": "V"})
        MC.run(label, exp_metadata=exp_metadata)

        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            tda.MultiQubit_TimeDomain_Analysis(qb_names=[self.name])

    def measure_rabi_amp90(self, scales=np.linspace(0.3, 0.7, 31), n=1,
                           MC=None, analyze=True, close_fig=True, upload=True):

        self.prepare(drive='timedomain')

        if MC is None:
            MC = self.instr_mc.get_instr()

        MC.set_sweep_function(awg_swf.Rabi_amp90(
            pulse_pars=self.get_ge_pars(), RO_pars=self.get_ro_pars(), n=n,
            upload=upload))
        MC.set_sweep_points(scales)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi_amp90_scales_n{}'.format(n)+ self.msmt_suffix)

    def measure_T1(self, times=None, analyze=True, upload=True,
                   last_ge_pulse=False, n_cal_points_per_state=2,
                   cal_states='auto', for_ef=False, classified_ro=False,
                   prep_params=None, label=None,
                   exp_metadata=None):

        if times is None:
            raise ValueError("Unspecified times for measure_T1")
        if np.any(times > 1e-3):
            log.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

        self.prepare(drive='timedomain')
        MC = self.instr_mc.get_instr()
        if prep_params is None:
            prep_params = self.preparation_params()

        # Define the measurement label
        if label is None:
            label = f'T1{"_ef" if for_ef else ""}' + self.msmt_suffix

        cal_states = CalibrationPoints.guess_cal_states(cal_states, for_ef)
        cp = CalibrationPoints.single_qubit(self.name, cal_states,
                                            n_per_state=n_cal_points_per_state)

        seq, sweep_points = sq.t1_active_reset(
            times=times, qb_name=self.name, cal_points=cp, for_ef=for_ef,
            operation_dict=self.get_operation_dict(), upload=False,
            last_ge_pulse=last_ge_pulse, prep_params=prep_params)

        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Time', unit='s'))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_classif_det if classified_ro else
                                 self.int_avg_det)

        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: times},
                             'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'rotate': False if classified_ro else
                                len(cp.states) != 0,
                             'last_ge_pulses': [last_ge_pulse],
                             'data_to_fit': {self.name: 'pf' if for_ef else 'pe'},
                             "sweep_name": "Time",
                             "sweep_unit": "s"})

        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            tda.MultiQubit_TimeDomain_Analysis(qb_names=[self.name])

    def measure_qscale(self, qscales=None, analyze=True, upload=True, label=None,
                       cal_states="auto", n_cal_points_per_state=2,
                       last_ge_pulse=False, for_ef=False, classified_ro=False,
                       prep_params=None, exp_metadata=None):
        if qscales is None:
            raise ValueError("Unspecified qscale values for measure_qscale")
        uniques = np.unique(qscales[range(3)])
        if uniques.size > 1:
            raise ValueError("The values in the qscales array are not repeated "
                             "3 times.")

        if prep_params is None:
            prep_params = self.preparation_params()
            log.debug(f"Preparation Parameters:\n{prep_params}")

        # Define the measurement label
        if label is None:
            label = f'Qscale{"_ef" if for_ef else ""}'
            if classified_ro:
                label += '_classified'
            if 'active' in prep_params['preparation_type']:
                label += '_reset'
            label += self.msmt_suffix

        MC = self.instr_mc.get_instr()

        self.prepare(drive='timedomain')

        # create cal points
        cal_states = CalibrationPoints.guess_cal_states(cal_states, for_ef)
        cp = CalibrationPoints.single_qubit(self.name, cal_states,
                                            n_per_state=n_cal_points_per_state)
        # create sequence
        seq, sweep_points = sq.qscale_active_reset(qscales=qscales,
            qb_name=self.name, cal_points=cp,  for_ef=for_ef,
            operation_dict=self.get_operation_dict(), upload=False,
            last_ge_pulse=last_ge_pulse, prep_params=prep_params)

        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Qscale factor'))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_classif_det if classified_ro else
                                 self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: qscales},
             'sweep_name': 'Qscale factor',
             'sweep_unit': '',
             'preparation_params': prep_params,
             'cal_points': repr(cp),
             'rotate': False if classified_ro else len(cp.states) != 0,
             'last_ge_pulses': [last_ge_pulse],
             'data_to_fit': {self.name: 'pf' if for_ef else 'pe'}})
        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            tda.MultiQubit_TimeDomain_Analysis(qb_names=[self.name])


    def measure_ramsey_multiple_detunings(self, times=None,
                                          artificial_detunings=None, label='',
                                          MC=None, analyze=True, close_fig=True,
                                          cal_points=True, upload=True,
                                          exp_metadata=None):
        log.error("This function is deprecated, please use measure_ramsey()")
        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detunings is None:
            log.warning('Artificial detuning is 0.')
        uniques = np.unique(times[range(len(artificial_detunings))])
        if uniques.size>1:
            raise ValueError("The values in the times array are not repeated "
                             "len(artificial_detunings) times.")
        if np.any(np.asarray(np.abs(artificial_detunings))<1e3):
            log.warning('The artificial detuning is too small. The units '
                            'should be Hz.')
        if np.any(times>1e-3):
            log.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        self.prepare(drive='timedomain')
        if MC is None:
            MC = self.instr_mc.get_instr()

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
            pulse_pars=self.get_ge_pars(), RO_pars=self.get_ro_pars(),
            artificial_detunings=artificial_detunings, cal_points=cal_points,
            upload=upload)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: sweep_points},
                             'use_cal_points': cal_points,
                             'artificial_detunings': artificial_detunings})
        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_ramsey_dyn_decoupling(self, times=None, artificial_detuning=0,
                                      label='', MC=None, analyze=True,
                                      close_fig=True, cal_points=True,
                                      upload=True, nr_echo_pulses=4,
                                      seq_func=None, cpmg_scheme=True,
                                      exp_metadata=None):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if np.any(times > 1e-3):
            log.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        if artificial_detuning is None:
            log.warning('Artificial detuning is 0.')
        if np.abs(artificial_detuning) < 1e3:
            log.warning('The artificial detuning is too small. The units'
                            'should be Hz.')

        if seq_func is None:
            seq_func = sq.ramsey_seq

        self.prepare(drive='timedomain')
        if MC is None:
            MC = self.instr_mc.get_instr()

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

        Rams_swf = awg_swf.Ramsey_decoupling_swf(
            seq_func=seq_func,
            pulse_pars=self.get_ge_pars(), RO_pars=self.get_ro_pars(),
            artificial_detuning=artificial_detuning, cal_points=cal_points,
            upload=upload, nr_echo_pulses=nr_echo_pulses, cpmg_scheme=cpmg_scheme)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: sweep_points},
                             'use_cal_points': cal_points,
                             'cpmg_scheme': cpmg_scheme,
                             'nr_echo_pulses': nr_echo_pulses,
                             'seq_func': seq_func,
                             'artificial_detuning': artificial_detuning})
        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            RamseyA = ma.Ramsey_Analysis(
                auto=True,
                label=label,
                qb_name=self.name,
                NoCalPoints=4,
                artificial_detuning=artificial_detuning,
                close_fig=close_fig)


    def measure_ramsey(self, times, artificial_detunings=None, label=None,
                       analyze=True, close_fig=True,
                       cal_states="auto", n_cal_points_per_state=2,
                       n=1, upload=True, last_ge_pulse=False, for_ef=False,
                       classified_ro=False, prep_params=None, exp_metadata=None):

        if prep_params is None:
            prep_params = self.preparation_params()

        # Define the measurement label
        if label is None:
            label = f'Ramsey{"_ef" if for_ef else ""}'
            if classified_ro:
                label += '_classified'
            if 'active' in prep_params['preparation_type']:
                label += '_reset'
            label += self.msmt_suffix

        MC = self.instr_mc.get_instr()
        self.prepare(drive='timedomain')

        # create cal points
        cal_states = CalibrationPoints.guess_cal_states(cal_states, for_ef)
        cp = CalibrationPoints.single_qubit(self.name, cal_states,
                                            n_per_state=n_cal_points_per_state)
        # create sequence
        seq, sweep_points = sq.ramsey_active_reset(
            times=times, artificial_detunings=artificial_detunings,
            qb_name=self.name, cal_points=cp, n=n, for_ef=for_ef,
            operation_dict=self.get_operation_dict(), upload=False,
            last_ge_pulse=last_ge_pulse, prep_params=prep_params)

        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Delay', unit='s'))
        MC.set_sweep_points(sweep_points)

        MC.set_detector_function(self.int_avg_classif_det if classified_ro else
                                 self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update(
            {'sweep_points_dict': {self.name: times},
             'sweep_name': 'Delay',
             'sweep_unit': 's',
             'cal_points': repr(cp),
             'preparation_params': prep_params,
             'last_ge_pulses': [last_ge_pulse],
             'artificial_detuning': artificial_detunings,
             'rotate': False if classified_ro else len(cp.states) != 0,
             'data_to_fit': {self.name: 'pf' if for_ef else 'pe'}})

        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            tda.MultiQubit_TimeDomain_Analysis(qb_names=[self.name])

    def measure_echo(self, times=None, artificial_detuning=None,
                     upload=True, analyze=True, close_fig=True, cal_points=True,
                     label=None, exp_metadata=None):

        if times is None:
            raise ValueError("Unspecified times for measure_echo")

        # Define the measurement label
        if label == '':
            label = 'Echo' + self.msmt_suffix

        if cal_points:
            step = np.abs(times[-1]-times[-2])
            sweep_points = np.concatenate(
                [times, [times[-1]+step,  times[-1]+2*step,
                         times[-1]+3*step, times[-1]+4*step]])
            cal_states_dict = {'g': [-4, -3], 'e': [-2, -1]}
            cal_states_rotations = {'g': 0, 'e': 1}
        else:
            sweep_points = times
            cal_states_dict = None
            cal_states_rotations = {}

        self.prepare(drive='timedomain')
        MC = self.instr_mc.get_instr()

        Echo_swf = awg_swf.Echo(
            pulse_pars=self.get_ge_pars(), RO_pars=self.get_ro_pars(),
            artificial_detuning=artificial_detuning, upload=upload)
        MC.set_sweep_function(Echo_swf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_classif_det if
                                 self.acq_weights_type() == 'optimal_qutrit'
                                 else self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: sweep_points},
                             'use_cal_points': cal_points,
                             'rotate': cal_points,
                             'cal_states_dict': cal_states_dict,
                             'cal_states_rotations': cal_states_rotations if
                                self.acq_weights_type() != 'optimal_qutrit'
                                else None,
                             'data_to_fit': {self.name: 'pe'},
                             'artificial_detuning': artificial_detuning})
        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            tda.MultiQubit_TimeDomain_Analysis(qb_names=[self.name])

    def measure_echo_2nd_exc(self, times=None, artificial_detuning=None,
                             label=None, analyze=True,
                             cal_points=True, no_cal_points=6, upload=True,
                             last_ge_pulse=False, exp_metadata=None):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detuning is None:
            log.warning('Artificial detuning is 0.')
        if np.abs(artificial_detuning) < 1e3:
            log.warning('The artificial detuning is too small. The units'
                            'should be Hz.')
        if np.any(times > 1e-3):
            log.warning('The values in the times array might be too large.'
                            'The units should be seconds.')

        if label is None:
            label = 'Echo_ef' + self.msmt_suffix

        self.prepare(drive='timedomain')
        MC = self.instr_mc.get_instr()

        cal_states_dict = None
        cal_states_rotations = {}
        if cal_points:
            step = np.abs(times[-1]-times[-2])
            if no_cal_points == 6:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step,
                                 times[-1]+3*step, times[-1]+4*step,
                                 times[-1]+5*step, times[-1]+6*step]])
                cal_states_dict = {'g': [-6, -5], 'e': [-4, -3], 'f': [-2, -1]}
                cal_states_rotations = {'g': 0, 'f': 1} if last_ge_pulse else \
                    {'e': 0, 'f': 1}
            elif no_cal_points == 4:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step,
                                 times[-1]+3*step, times[-1]+4*step]])
                cal_states_dict = {'g': [-4, -3], 'e': [-2, -1]}
                cal_states_rotations = {'g': 0, 'e': 1}
            elif no_cal_points == 2:
                sweep_points = np.concatenate(
                    [times, [times[-1]+step,  times[-1]+2*step]])
                cal_states_dict = {'g': [-2, -1]}
                cal_states_rotations = {'g': 0}
            else:
                sweep_points = times
        else:
            sweep_points = times

        Echo_2nd_swf = awg_swf.Echo_2nd_exc(
            pulse_pars=self.get_ge_pars(),
            pulse_pars_2nd=self.get_ef_pars(),
            RO_pars=self.get_ro_pars(),
            artificial_detuning=artificial_detuning,
            cal_points=cal_points, upload=upload,
            no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse)
        MC.set_sweep_function(Echo_2nd_swf)
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_classif_det if
                                 self.acq_weights_type() == 'optimal_qutrit'
                                 else self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: sweep_points},
                             'rotate': cal_points,
                             'last_ge_pulse': last_ge_pulse,
                             'data_to_fit': {self.name: 'pf'},
                             'cal_states_dict': cal_states_dict,
                             'cal_states_rotations': cal_states_rotations if
                                self.acq_weights_type() != 'optimal_qutrit'
                                else None,
                             'artificial_detuning': artificial_detuning})
        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            tda.MultiQubit_TimeDomain_Analysis(qb_names=[self.name])

    def measure_randomized_benchmarking(
            self, cliffords, nr_seeds,
            gate_decomp='HZ', interleaved_gate=None,
            n_cal_points_per_state=2, cal_states=(),
            classified_ro=False, thresholded=True, label=None,
            upload=True, analyze=True, prep_params=None,
            exp_metadata=None, **kw):
        '''
        Performs a randomized benchmarking experiment on 1 qubit.
        '''

        # Define the measurement label
        if label is None:
            if interleaved_gate is None:
                label = 'RB_{}_{}_seeds_{}_cliffords'.format(
                    gate_decomp, nr_seeds, cliffords[-1]) + self.msmt_suffix
            else:
                label = 'IRB_{}_{}_{}_seeds_{}_cliffords'.format(
                    interleaved_gate, gate_decomp, nr_seeds, cliffords[-1]) + \
                        self.msmt_suffix

        if prep_params is None:
            prep_params = self.preparation_params()
        # Prepare the physical instruments for a time domain measurement
        self.prepare(drive='timedomain')

        MC = self.instr_mc.get_instr()
        cal_states = CalibrationPoints.guess_cal_states(cal_states)
        cp = CalibrationPoints.single_qubit(self.name, cal_states,
                                            n_per_state=n_cal_points_per_state)

        sequences, hard_sweep_points, soft_sweep_points = \
            sq.randomized_renchmarking_seqs(
                qb_name=self.name, operation_dict=self.get_operation_dict(),
                cliffords=cliffords, nr_seeds=np.arange(nr_seeds),
                gate_decomposition=gate_decomp,
                interleaved_gate=interleaved_gate, upload=False,
                cal_points=cp, prep_params=prep_params)

        hard_sweep_func = awg_swf.SegmentHardSweep(
            sequence=sequences[0], upload=upload,
            parameter_name='Nr. Cliffords', unit='')
        MC.set_sweep_function(hard_sweep_func)
        MC.set_sweep_points(hard_sweep_points)

        MC.set_sweep_function_2D(awg_swf.SegmentSoftSweep(
            hard_sweep_func, sequences, 'Nr. Seeds', ''))
        MC.set_sweep_points_2D(soft_sweep_points)
        if thresholded:
            det_func = self.dig_avg_det
        elif classified_ro:
            det_func = self.int_avg_classif_det
        else:
            det_func = self.int_avg_det
        MC.set_detector_function(det_func)

        # create sweep points object
        sp = SweepPoints('nr_seeds', np.arange(nr_seeds), '', 'Nr. Seeds')
        sp.add_sweep_dimension()
        sp.add_sweep_parameter('cliffords', cliffords, '',
                               'Number of applied Cliffords, $m$')
        # create analysis pipeline object
        pp = ProcessingPipeline(
            'average',  keys_in=det_func.value_names,
            num_bins=[len(cliffords)]*len(det_func.value_names))
        pp.add_node(
            'get_std_deviation', keys_in=det_func.value_names,
            num_bins=[len(cliffords)]*len(det_func.value_names))
        pp.add_node('SingleQubitRBAnalysis', keys_in='previous',
                    std_keys=[k+' std' for k in det_func.value_names],
                    meas_obj_name=self.name, do_plotting=True)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'sweep_points': sp,
                             'meas_obj_sweep_points_map':
                                 sp.get_sweep_points_map([self.name]),
                             'meas_obj_value_names_map': {
                                 self.name: det_func.value_names},
                             'processing_pipe': pp})
        MC.run_2D(label, exp_metadata=exp_metadata)

        if analyze:
            pla.PipelineDataAnalysis()

    def measure_transients(self, states=('g', 'e'), upload=True,
                           analyze=True, acq_length=4097/1.8e9,
                           prep_params=None, exp_metadata=None, **kw):
        """
        If the resulting transients will be used to caclulate the optimal
        weight functions, then it is important that the UHFQC iavg_delay and
        wint_delay are calibrated such that the weights and traces are
        aligned: iavg_delay = 2*wint_delay.

        """
        MC = self.instr_mc.get_instr()
        name_extra = kw.get('name_extra', None)

        if prep_params is None:
            prep_params = self.preparation_params()
        if exp_metadata is None:
            exp_metadata = dict()
        exp_metadata.update(
            {'sweep_name': 'time',
             'sweep_unit': ['s']})

        with temporary_value(self.acq_length, acq_length):
            self.prepare(drive='timedomain')
            npoints = self.inp_avg_det.nr_samples
            sweep_points = np.linspace(0, npoints / 1.8e9, npoints,
                                                endpoint=False)
            for state in states:
                if state not in ['g', 'e', 'f']:
                    raise ValueError("Unrecognized state: {}. Must be 'g', 'e' "
                                     "or 'f'.".format(state))
                base_name = 'timetrace_{}'.format(state)
                name = base_name + "_" + name_extra if name_extra is not None \
                    else base_name
                seq, _ = sq.single_state_active_reset(
                    operation_dict=self.get_operation_dict(),
                    qb_name=self.name, state=state, prep_params=prep_params,
                    upload=False)
                # set sweep function and run measurement
                MC.set_sweep_function(awg_swf.SegmentHardSweep(sequence=seq,
                                                               upload=upload))
                MC.set_sweep_points(sweep_points)
                MC.set_detector_function(self.inp_avg_det)
                exp_metadata.update(dict(sweep_points_dict=sweep_points))
                MC.run(name=name + self.msmt_suffix, exp_metadata=exp_metadata)

    def measure_readout_pulse_scope(self, delays, freqs, RO_separation=None,
                                    prep_pulses=None, comm_freq=225e6,
                                    analyze=True, label=None,
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
            raise ValueError("Unspecified delays for "
                             "measure_readout_pulse_scope")
        if label is None:
            label = 'Readout_pulse_scope' + self.msmt_suffix
        if MC is None:
            MC = self.instr_mc.get_instr()
        if freqs is None:
            freqs = self.f_qubit() + np.linspace(-50e6, 50e6, 201)
        if RO_separation is None:
            RO_separation = 2 * self.ro_length()
            RO_separation += np.max(delays)
            RO_separation += 200e-9  # for slack

        self.prepare(drive='timedomain')
        MC.set_sweep_function(awg_swf.Readout_pulse_scope_swf(
            delays=delays,
            pulse_pars=self.get_ge_pars(),
            RO_pars=self.get_ro_pars(),
            RO_separation=RO_separation,
            cal_points=cal_points,
            prep_pulses=prep_pulses,
            comm_freq=comm_freq,
            verbose=verbose,
            upload=upload))
        MC.set_sweep_points(delays)
        MC.set_sweep_function_2D(swf.Offset_Sweep(
            mc_parameter_wrapper.wrap_par_to_swf(
                self.instr_ge_lo.get_instr().frequency),
            -self.ge_mod_freq(),
            parameter_name=self.name + ' drive frequency'))
        MC.set_sweep_points_2D(freqs)

        d = det.UHFQC_integrated_average_detector(
            self.instr_uhf.get_instr(), self.instr_pulsar.get_instr(),
            nr_averages=self.acq_averages(),
            channels=self.int_avg_det.channels,
            integration_length=self.acq_length(),
            values_per_point=2, values_per_point_suffex=['_probe', '_measure'])
        MC.set_detector_function(d)
        MC.run_2D(label)

        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            ma.MeasurementAnalysis(TwoD=True, auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_residual_readout_photons(
            self, delays_to_relax, ramsey_times, delay_buffer=0,
            cal_points=((-4, -3), (-2, -1)), verbose=False,
            artificial_detuning=None, analyze=True,
            label=None, close_fig=True, MC=None):
        """
        From the documentation of the used sequence function:

        The sequence consists of two readout pulses sandwitching two ramsey
        pulses inbetween. The delay between the first readout pulse and first
        ramsey pulse is swept, to measure the ac stark shift and dephasing
        from any residual photons.

        Important: This sequence includes two readouts per segment. For this
        reason the calibration points are also duplicated.

        Args:
            delays_to_relax: delay between the end of the first readout
                            pulse and the start of the first ramsey pulse.

            pulse_pars: Pulse dictionary for the ramsey pulse.
            RO_pars: Pulse dictionary for the readout pulse.
            ramsey_times: delays between ramsey pulses
            delay_buffer: delay between the start of the last ramsey pulse and
                          the start of the second readout pulse.
            cal_points: True for default calibration points, False for no
                        calibration points or a list of two lists,
                        containing the indices of the calibration
                        segments for the ground and excited state.
        """


        if label is None:
            label = 'residual_readout_photons' + self.msmt_suffix
        if MC is None:
            MC = self.instr_mc.get_instr()
        # duplicate sweep points for the two preparation states
        ramsey_times = np.vstack((ramsey_times, ramsey_times)).\
                       reshape((-1,), order='F')

        self.prepare(drive='timedomain')
        sf1 = awg_swf.readout_photons_in_resonator_swf(
            delay_to_relax=delays_to_relax[0],
            delay_buffer=delay_buffer,
            ramsey_times=ramsey_times,
            pulse_pars=self.get_ge_pars(),
            RO_pars=self.get_ro_pars(),
            cal_points=cal_points,
            verbose=verbose,
            artificial_detuning=artificial_detuning,
            upload=False)
        MC.set_sweep_function(sf1)
        MC.set_sweep_points(ramsey_times)
        sf2 = awg_swf.readout_photons_in_resonator_soft_swf(sf1)
        MC.set_sweep_function_2D(sf2)
        MC.set_sweep_points_2D(delays_to_relax)

        d = det.UHFQC_integrated_average_detector(
            self.instr_uhf.get_instr(), self.instr_pulsar.get_instr(),
            nr_averages=self.acq_averages(),
            channels=self.int_avg_det.channels,
            integration_length=self.acq_length(),
            values_per_point=2, values_per_point_suffex=['_test', '_measure'])
        MC.set_detector_function(d)
        MC.run_2D(label)
        self.artificial_detuning = artificial_detuning
        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            kappa = list(map(lambda w:0.5*(self.RO_purcell_kappa() - np.real(
                np.sqrt(-16*self.RO_J_coupling()*self.RO_J_coupling() +
                        (self.RO_purcell_kappa()-2j*(np.abs(w-self.f_RO_purcell())))*
                        (self.RO_purcell_kappa()-2j*(np.abs(w-self.f_RO_purcell())))
                        ))),
                             [self.f_RO_resonator() - self.chi(),
                              self.f_RO_resonator() + self.chi()]))
            if not (self.T2_star_ef() == 0):
                T2star = self.T2_star_ef()
            else:
                if self.T2_star() == 0:
                   raise ValueError('T2star is not given.')
                else:
                    T2star = self.T2_star()
            tda.ReadoutROPhotonsAnalysis(t_start=None,
                  close_figs=close_fig, options_dict={
                      'f_qubit': self.f_qubit(),
                      'chi': self.chi(),
                      'kappa-effective': kappa,
                      'T2echo': T2star ,
                      'do_analysis': True,
                      'artif_detuning': self.artificial_detuning() },
                  do_fitting=True)

    def measure_multi_element_segment_timing(
            self, phases, ramsey_time=4e-6, nr_wait_elems=16,
            elem_type='interleaved', cal_points=((-4, -3), (-2, -1)),
            label=None, MC=None, upload=True, analyze=True, close_fig=True):

        if label is None:
            label = 'Multi_element_segment_timing' + self.msmt_suffix
        if MC is None:
            MC = self.instr_mc.get_instr()

        self.prepare(drive='timedomain')
        MC.set_sweep_function(awg_swf.MultiElemSegmentTimingSwf(
            phases=phases,
            qbn=self.name,
            op_dict=self.get_operation_dict(),
            ramsey_time=ramsey_time,
            nr_wait_elems=nr_wait_elems,
            elem_type=elem_type,
            cal_points=cal_points,
            upload=upload))
        MC.set_sweep_points(phases)
        d = det.UHFQC_integrated_average_detector(
            self.instr_uhf.get_instr(), self.instr_pulsar.get_instr(), nr_averages=self.acq_averages(),
            channels=self.int_avg_det.channels,
            integration_length=self.acq_length(),
            values_per_point=2, values_per_point_suffex=['_single_elem',
                                                         '_multi_elem'])
        MC.set_detector_function(d)

        metadata = dict(
            ramsey_time=ramsey_time,
            nr_wait_elems=nr_wait_elems,
            elem_type=elem_type,
            cal_points=cal_points
        )
        MC.run(label, exp_metadata=metadata)

        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name)
    
    def measure_drive_mixer_spectrum(self, if_freqs, amplitude=0.5,
                                     trigger_sep=5e-6, align_frequencies=True):
        MC = self.instr_mc.get_instr()
        if align_frequencies:
            if_freqs = (if_freqs*trigger_sep).astype(np.int)/trigger_sep
        s = swf.Offset_Sweep(
            self.instr_ro_lo.get_instr().frequency, 
            self.ge_freq() - self.ro_mod_freq() - self.ge_mod_freq(),
            name='Drive intermediate frequency', 
            parameter_name='Drive intermediate frequency')
        MC.set_sweep_function(s)
        MC.set_sweep_points(if_freqs)
        MC.set_detector_function(self.int_avg_det_spec)
        drive_pulse = dict(
                pulse_type='GaussFilteredCosIQPulse',
                pulse_length=self.acq_length(),
                ref_point='start',
                amplitude=amplitude,
                I_channel=self.ge_I_channel(),
                Q_channel=self.ge_Q_channel(),
                mod_frequency=self.ge_mod_freq(),
                phase_lock=True,
            )
        sq.pulse_list_list_seq([[self.get_acq_pars(), drive_pulse]])
            
        with temporary_value(
            (self.acq_weights_type, 'SSB'),
            (self.instr_trigger.get_instr().pulse_period, trigger_sep),
        ):
            self.prepare(drive='timedomain')
            self.instr_pulsar.get_instr().start()
            MC.run('ge_uc_spectrum' + self.msmt_suffix)

        a = ma.MeasurementAnalysis(plot_args=dict(log=True, marker=''))
        return a

    def calibrate_drive_mixer_carrier(self, update=True, x0=(0., 0.),
                                      initial_stepsize=0.01, trigger_sep=5e-6):
        MC = self.instr_mc.get_instr()
        ad_func_pars = {'adaptive_function': opti.nelder_mead,
                        'x0': x0,
                        'initial_step': [initial_stepsize, initial_stepsize],
                        'no_improv_break': 15,
                        'minimize': True,
                        'maxiter': 500}
        chI_par = self.instr_pulsar.get_instr().parameters['{}_offset'.format(
            self.ge_I_channel())]
        chQ_par = self.instr_pulsar.get_instr().parameters['{}_offset'.format(
            self.ge_Q_channel())]
        MC.set_sweep_functions([chI_par, chQ_par])
        MC.set_adaptive_function_parameters(ad_func_pars)
        sq.pulse_list_list_seq([[self.get_acq_pars(), dict(
                            pulse_type='GaussFilteredCosIQPulse',
                            pulse_length=self.acq_length(),
                            ref_point='start',
                            amplitude=0,
                            I_channel=self.ge_I_channel(),
                            Q_channel=self.ge_Q_channel(),
                        )]])
            
        with temporary_value(
            (self.ro_freq, self.ge_freq() - self.ge_mod_freq()),
            (self.acq_weights_type, 'SSB'),
            (self.instr_trigger.get_instr().pulse_period, trigger_sep),
        ):
            self.prepare(drive='timedomain')
            MC.set_detector_function(det.IndexDetector(
                self.int_avg_det_spec, 0))
            self.instr_pulsar.get_instr().start(exclude=[self.instr_uhf()])
            MC.run(name='drive_carrier_calibration' + self.msmt_suffix,
                mode='adaptive')
        
        a = ma.OptimizationAnalysis(label='drive_carrier_calibration')
        # v2 creates a pretty picture of the optimizations
        ma.OptimizationAnalysis_v2(label='drive_carrier_calibration')

        ch_1_min = a.optimization_result[0][0]
        ch_2_min = a.optimization_result[0][1]
        if update:
            self.ge_I_offset(ch_1_min)
            self.ge_Q_offset(ch_2_min)
        return ch_1_min, ch_2_min

    def calibrate_drive_mixer_skewness(self, update=True, amplitude=0.5, 
                                       trigger_sep=5e-6,
                                       initial_stepsize=(0.15, 10)):
        MC = self.instr_mc.get_instr()
        ad_func_pars = {'adaptive_function': opti.nelder_mead,
                        'x0': [self.ge_alpha(), self.ge_phi_skew()],
                        'initial_step': initial_stepsize,
                        'no_improv_break': 12,
                        'minimize': True,
                        'maxiter': 500}
        MC.set_sweep_functions([self.ge_alpha, self.ge_phi_skew])
        MC.set_adaptive_function_parameters(ad_func_pars)

        with temporary_value(
            (self.ge_alpha, self.ge_alpha()),
            (self.ge_phi_skew, self.ge_phi_skew()),
            (self.ro_freq, self.ge_freq() - 2*self.ge_mod_freq()),
            (self.acq_weights_type, 'SSB'),
            (self.instr_trigger.get_instr().pulse_period, trigger_sep),
        ):
            self.prepare(drive='timedomain')
            detector = self.int_avg_det_spec
            detector.always_prepare = True
            detector.AWG = self.instr_pulsar.get_instr()
            detector.prepare_function = lambda \
                alphaparam=self.ge_alpha, skewparam=self.ge_phi_skew: \
                    sq.pulse_list_list_seq([[self.get_acq_pars(), dict(
                            pulse_type='GaussFilteredCosIQPulse',
                            pulse_length=self.acq_length(),
                            ref_point='start',
                            amplitude=amplitude,
                            I_channel=self.ge_I_channel(),
                            Q_channel=self.ge_Q_channel(),
                            mod_frequency=self.ge_mod_freq(),
                            phase_lock=True,
                            alpha=alphaparam(),
                            phi_skew=skewparam(),
                        )]])
            MC.set_detector_function(det.IndexDetector(detector, 0))
            MC.run(name='drive_skewness_calibration' + self.msmt_suffix,
                   mode='adaptive')
        
        a = ma.OptimizationAnalysis(label='drive_skewness_calibration')
        # v2 creates a pretty picture of the optimizations
        ma.OptimizationAnalysis_v2(label='drive_skewness_calibration')

        # phi and alpha are the coefficients that go in the predistortion matrix
        alpha = a.optimization_result[0][0]
        phi = a.optimization_result[0][1]
        if update:
            self.ge_alpha(alpha)
            self.ge_phi_skew(phi)
        return alpha, phi

    def calibrate_drive_mixer_skewness_NN(
            self, update=True,make_fig=True, meas_grid=None, n_meas=100,
            amplitude=0.1, trigger_sep=5e-6, two_rounds=False,
            estimator='GRNN_neupy', hyper_parameter_dict=None, 
            first_round_limits=(0.6, 1.2, -50, 35), **kwargs):
        if not len(first_round_limits) == 4:
            log.error('Input variable `first_round_limits` in function call '
                      '`calibrate_drive_mixer_skewness_NN` needs to be a list '
                      'or 1D array of length 4.\nFound length '
                      '{} object instead!'.format(len(first_round_limits)))
        if hyper_parameter_dict is None:
            log.warning('No hyperparameters passed to predictive mixer '
                        'calibration routine. Default values for the estimator'
                        'will be used!\n')
            hyper_parameter_dict = {'hidden_layers': [10],
                                    'learning_rate': 1e-3,
                                    'regularization_coefficient': 0.,
                                    'std_scaling': 0.6,
                                    'learning_steps': 5000,
                                    'cv_n_fold': 5,
                                    'polynomial_dimension': 2}
        std_devs = kwargs.get('std_devs', [0.1, 10])
        c = kwargs.pop('second_round_std_scale', 0.4)
        
        # Could make sample size variable (maxiter) for better adapting)
        if isinstance(std_devs, (list, np.ndarray)):
            if len(std_devs) != 2:
                log.error('std_devs passed in kwargs of `calibrate_drive_'
                          'mixer_NN` is of length: {}. '
                          'Requires length 2 instead.'.format(len(std_devs)))

        MC = self.instr_mc.get_instr()
        _alpha = self.ge_alpha()
        _phi = self.ge_phi_skew()
        for runs in range(3 if two_rounds else 2):
            if runs == 0:
                # half as many points from a uniform distribution at first run
                meas_grid = np.array([
                    np.random.uniform(first_round_limits[0], 
                                      first_round_limits[1], n_meas//2),
                    np.random.uniform(first_round_limits[2], 
                                      first_round_limits[3], n_meas//2)])
            else:
                k = 1. if runs == 1 else c
                meas_grid = np.array([
                    np.random.normal(_alpha, k*std_devs[0], n_meas),
                    np.random.normal(_phi, k*std_devs[1], n_meas)])

            s1 = swf.Hard_Sweep()
            s1.name = 'Amplitude ratio hardware sweep'
            s1.label = r'Amplitude ratio, $\alpha$'
            s1.unit = ''
            s2 = swf.Hard_Sweep()
            s2.name = 'Phase skew hardware sweep'
            s2.label = r'Phase skew, $\phi$'
            s2.unit = 'deg'
            MC.set_sweep_functions([s1, s2])
            MC.set_sweep_points(meas_grid.T)
            
            pulse_list_list = []
            for alpha, phi_skew in meas_grid.T:
                pulse_list_list.append([self.get_acq_pars(), dict(
                            pulse_type='GaussFilteredCosIQPulse',
                            pulse_length=self.acq_length(),
                            ref_point='start',
                            amplitude=amplitude,
                            I_channel=self.ge_I_channel(),
                            Q_channel=self.ge_Q_channel(),
                            mod_frequency=self.ge_mod_freq(),
                            phase_lock=True,
                            alpha=alpha,
                            phi_skew=phi_skew,
                        )])
            sq.pulse_list_list_seq(pulse_list_list)

            with temporary_value(
                (self.ro_freq, self.ge_freq() - 2*self.ge_mod_freq()),
                (self.acq_weights_type, 'SSB'),
                (self.instr_trigger.get_instr().pulse_period, trigger_sep),
            ):
                self.prepare(drive='timedomain')
                MC.set_detector_function(self.int_avg_det)
                MC.run(name='drive_skewness_calibration' + self.msmt_suffix)

            a = ma.OptimizationAnalysisNN(
                label='drive_skewness_calibration',
                hyper_parameter_dict=hyper_parameter_dict,
                meas_grid=meas_grid.T,
                estimator=estimator,
                two_rounds=two_rounds,
                round=runs, make_fig=make_fig)

            _alpha = a.optimization_result[0]
            _phi = a.optimization_result[1]
        
            if update:
                self.ge_alpha(_alpha)
                self.ge_phi_skew(_phi)

        return _alpha, _phi, a

    def find_optimized_weights(self, update=True, measure=True,
                               qutrit=False, acq_length=4097/1.8e9, **kw):
        # FIXME: Make a proper analysis class for this (Ants, 04.12.2017)
        # I agree (Christian, 07.11.2018 -- around 1 year later)

        levels = ('g', 'e', 'f') if qutrit else ('g', 'e')
        if measure:
            self.measure_transients(analyze=True, states=levels,
                                    acq_length=acq_length, **kw)

        # create label, measurement analysis and data for each level
        if kw.get("name_extra", False):
            labels = {l: 'timetrace_{}_'.format(l) + kw.get('name_extra')
                         + "_{}".format(self.name) for l in levels}
        else:
            labels = {l: 'timetrace_{}'.format(l)
                         + "_{}".format(self.name) for l in levels}
        m_a = {l: ma.MeasurementAnalysis(label=labels[l]) for l in levels}
        iq_traces = {l: m_a[l].measured_values[0]
                        + 1j * m_a[l].measured_values[1] for l in levels}
        final_basis_labels = ['ge'] # default basis vector if only qubit ro
        if qutrit:
            ref_state = kw.get('ref_state', 'g')
            basis = [iq_traces[l] - iq_traces[ref_state] for l in levels
                     if l != ref_state]
            basis_labels = [l + ref_state for l in levels if l != ref_state]
            final_basis = math.gram_schmidt(np.array(basis).transpose())
            final_basis = final_basis.transpose()  # obtain basis vect as rows
            # basis using second vector as primary vector
            basis_2nd = list(reversed(basis))
            final_basis_2nd = math.gram_schmidt(np.array(basis_2nd).transpose())
            final_basis_2nd = final_basis_2nd.transpose()
            if kw.get('non_ortho_basis', False):
                final_basis_labels = basis_labels
                final_basis = np.array([final_basis[0], final_basis_2nd[0]])
            elif kw.get('basis_2nd', False):
                final_basis_labels = [basis_labels[1]] + ['ortho']
                final_basis = final_basis_2nd
            else:
                final_basis_labels = [basis_labels[0]] + ['ortho']

            log.info(f"Setting Basis: {final_basis_labels}")
        if update:
            # FIXME: could merge qutrit and non qutrit although normalization is not
            #  the same but would be a good thing to do. First test if qutrit works
            #  well. idem in plot
            if qutrit:
                self.acq_weights_I(final_basis[0].real)
                self.acq_weights_Q(final_basis[0].imag)
                self.acq_weights_I2(final_basis[1].real)
                self.acq_weights_Q2(final_basis[1].imag)
                self.acq_weights_basis(final_basis_labels)
            else:
                wre = np.real(iq_traces['e'] - iq_traces['g'])
                wim = np.imag(iq_traces['e'] - iq_traces['g'])
                k = max(np.max(np.abs(wre)), np.max(np.abs(wim)))
                wre /= k
                wim /= k
                self.acq_weights_I(wre)
                self.acq_weights_Q(wim)
        if kw.get('plot', True):
            # TODO: Nathan: plot amplitude instead of I, Q ?
            npoints = len(m_a['g'].sweep_points)
            plot_ylabels = dict(g='d.c. voltage,\nNo pulse (V)',
                                e='d.c. voltage,\nPi_ge pulse (V)',
                                f='d.c. voltage,\nPi_gf pulse (V)')
            tbase = np.linspace(0, npoints/1.8e9, npoints, endpoint=False)
            modulation = np.exp(2j * np.pi * self.ro_mod_freq() * tbase)
            fig, ax = plt.subplots(len(levels) + 1, figsize=(20,20))
            ax[0].set_title('optimized weights ' + self.name +
                        "".join('\n' + m_a[l].timestamp_string for l in levels)
                        + f'\nWeight Basis: {final_basis_labels}')
            for i, l in enumerate(levels):
                ax[i].plot(tbase / 1e-9, np.real(iq_traces[l] * modulation), '-',
                         label='I_' + l)
                ax[i].plot(tbase / 1e-9, np.imag(iq_traces[l] * modulation), '-',
                         label='Q_' + l)
                ax[i].set_ylabel(plot_ylabels[l])
                ax[i].set_xlim(0, kw.get('tmax', 300))
                ax[i].legend(loc='upper right')
            if qutrit:
                for i, vect in enumerate(final_basis):
                    ax[-1].plot(tbase / 1e-9, np.real(vect * modulation), '-',
                                label='I_' + str(i))
                    ax[-1].plot(tbase / 1e-9, np.imag(vect * modulation), '-',
                                label='Q_' + str(i))
            else:
                ax[-1].plot(tbase / 1e-9,
                            np.real((iq_traces['e'] - iq_traces['g']) * modulation), '-',
                            label='I')
                ax[-1].plot(tbase / 1e-9,
                            np.imag((iq_traces['e'] - iq_traces['g']) * modulation), '-',
                            label='Q')
            ax[-1].set_ylabel('d.c. voltage\ndifference (V)')
            ax[-1].set_xlim(0, kw.get('tmax', 300))
            ax[-1].legend(loc='upper right')
            ax[-1].set_xlabel('Time (ns)')
            m_a['g'].save_fig(plt.gcf(), 'timetraces', xlabel='time',
                           ylabel='voltage')
            plt.tight_layout()
            plt.close()

    def find_ssro_fidelity(self, analyze=True, close_fig=True, no_fits=False,
                           upload=True, thresholded=False, label=None,
                           RO_comm=3 / 225e6, RO_slack=150e-9,
                           qutrit=False, update=False, prep_params=None):
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
            analyze: Boolean flag, whether to analyse the measurement results.
                     Default `True`.
            close_fig: Boolean flag to close the matplotlib's figure. If
                       `False`, then the plots can be viewed with `plt.show()`
                       Default `True`.
            no_fits: Boolean flag to disable finding the discrimination
                     fidelity. Default `False`.
            preselection_pulse: Whether to do an additional readout pulse
                                before state preparation. Default `True`.
            qutrit: SSRO for 3 levels readout
        Returns:
            If `no_fits` is `False` returns assigment fidelity, discrimination
            fidelity and SNR = 2 |mu00 - mu11| / (sigma00 + sigma11). Else
            returns just assignment fidelity.
        """
        MC = self.instr_mc.get_instr()
        if label is None:
            label = 'SSRO_fidelity'
        if thresholded:
            label += '_thresh'

        if prep_params is None:
            prep_params = self.preparation_params()

        self.prepare(drive='timedomain')

        RO_spacing = self.instr_uhf.get_instr().qas_0_delay() / 1.8e9
        RO_spacing += self.acq_length()
        RO_spacing += RO_slack  # for slack
        RO_spacing = np.ceil(RO_spacing / RO_comm) * RO_comm

        if prep_params['preparation_type'] not in ['preselection', 'wait']:
            raise NotImplementedError()
        preselection = prep_params['preparation_type'] == 'preselection'

        if thresholded:
            det_func = self.dig_log_det
        else:
            det_func = self.int_log_det

        if qutrit:
            states = ('g', 'e', 'f')
            for state in states:
                seq, swp = sq.single_state_active_reset(
                    operation_dict=self.get_operation_dict(),
                    qb_name=self.name, state=state,
                    prep_params=prep_params, upload=False)
                # set sweep function and run measurement
                MC.set_sweep_function(awg_swf.SegmentHardSweep(sequence=seq,
                                                               upload=upload))
                MC.set_sweep_points(swp)
                MC.set_detector_function(det_func)
                with temporary_value(MC.soft_avg, 1):
                    MC.run(name=label + '_{}'.format(state) + self.msmt_suffix)

        else:
            MC.set_sweep_function(awg_swf2.n_qubit_off_on(
                pulse_pars_list=[self.get_ge_pars()],
                RO_pars_list=[self.get_ro_pars()],
                upload=upload,
                preselection=preselection,
                RO_spacing=RO_spacing))
            MC.set_sweep_points(np.arange(4 if preselection else 2))
            MC.set_detector_function(det_func)
            with temporary_value(MC.soft_avg, 1):
                MC.run(name=label + self.msmt_suffix)

        if analyze:
            if qutrit:
                # TODO Nathan: could try and merge this with no qutrit to
                #  avoid logical branching
                options = \
                    dict(classif_method='threshold' if thresholded else 'gmm',
                         pre_selection=preselection)
                # options = 'gmm'
                labels = [label + '_{}'.format(l) for l in states]
                ssqtro = \
                    Singleshot_Readout_Analysis_Qutrit(label=labels,
                                                          options_dict=options)
                state_prob_mtx = ssqtro.proc_data_dict[
                    'analysis_params']['state_prob_mtx_masked']
                classifier_params = ssqtro.proc_data_dict[
                    'analysis_params'].get('classifier_params', None)
                if update:
                    self.acq_classifier_params(classifier_params)
                    self.acq_state_prob_mtx(state_prob_mtx)
                return state_prob_mtx, classifier_params
            else:
                rotate = self.acq_weights_type() in {'SSB', 'DSB'}
                preselection = prep_params['preparation_type'] == 'preselection'
                channels = det_func.value_names
                if preselection:
                    nr_samples = 4
                    sample_0 = 0
                    sample_1 = 2
                else:
                    nr_samples = 2
                    sample_0 = 0
                    sample_1 = 1
                ana = ma.SSRO_Analysis(auto=True, close_fig=close_fig,
                                       qb_name=self.name,
                                       rotate=rotate, no_fits=no_fits,
                                       channels=channels, nr_samples=nr_samples,
                                       sample_0=sample_0, sample_1=sample_1,
                                       preselection=preselection)
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
            MC = self.instr_mc.get_instr()

        label = 'RO_theta'
        if self.acq_weights_Q() is None:
            self.acq_weights_Q(
                (self.acq_weights_I() + 1) % 9)
        self.set_readout_weights(weights_type='SSB')
        prev_shots = self.acq_shots()
        self.acq_shots(2*(self.acq_shots()//2))
        self.prepare(drive='timedomain')
        MC.set_sweep_function(awg_swf.SingleLevel(
            pulse_pars=self.get_ge_pars(),
            RO_pars=self.get_ro_pars(),
            upload=upload,
            preselection=False))
        spoints = np.arange(self.acq_shots())
        MC.set_sweep_points(np.arange(self.acq_shots()))
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
        self.acq_shots(prev_shots)

        rotate = self.acq_weights_Q() is not None
        channels = self.int_log_det.value_names
        ana = ma.SSRO_Analysis(auto=True, close_fig=close_fig,
                               rotate=rotate, no_fits=True,
                               channels=channels,
                               preselection=False)
        if update:
            self.acq_IQ_angle(self.acq_IQ_angle() + ana.theta)
        return ana.theta

    def find_qubit_frequency(self, freqs, method='cw_spectroscopy',
                             update=False, trigger_separation=3e-6,
                             close_fig=True, analyze_ef=False, analyze=True,
                             upload=True, label=None, **kw):
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
            log.warning("Does not automatically update the qubit "
                            "frequency parameter. "
                            "Set update=True if you want this!")
        if np.any(freqs<500e6):
            log.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if freqs is None:
            f_span = kw.get('f_span', 100e6)
            f_mean = kw.get('f_mean', self.f_qubit())
            nr_points = kw.get('nr_points', 100)
            if f_mean == 0:
                log.warning("find_frequency does not know where to "
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

            self.measure_qubit_spectroscopy(freqs, pulsed=False,
                                      trigger_separation=trigger_separation,
                                      label=label, close_fig=close_fig)
        else:
            if label is None:
                label = 'pulsed_spec' + self.msmt_suffix
            if analyze_ef:
                label = 'high_power_' + label

            self.measure_qubit_spectroscopy(freqs, pulsed=True, label=label,
                                      close_fig=close_fig, upload=upload)


        if analyze:
            SpecA = ma.Qubit_Spectroscopy_Analysis(
                qb_name=self.name,
                analyze_ef=analyze_ef,
                label=label,
                close_fig=close_fig, **kw)

            f0 = SpecA.fitted_freq
            if update:
                if not analyze_ef:
                    self.ge_freq(f0)
                else:
                    f0_ef = 2*SpecA.fitted_freq_gf_over_2 - f0
                    self.ef_freq(f0_ef)
            if analyze_ef:
                return f0, f0_ef
            else:
                return f0
        else:
            return

    def find_amplitudes(self, rabi_amps=None, label=None, for_ef=False,
                        n_cal_points_per_state=2, cal_states='auto',
                        upload=True, last_ge_pulse=False, classified_ro=False,
                        prep_params=None, analyze=True, update=False,
                        exp_metadata=None, **kw):
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
            log.warning("Does not automatically update the qubit pi and "
                        "pi/2 amplitudes. Set update=True if you want this!")

        #how many times to apply the Rabi pulse
        n = kw.get('n', 1)

        if rabi_amps is None:
            raise ValueError('rabi_amps is None.')

        #Perform Rabi
        self.measure_rabi(amps=rabi_amps, analyze=False,
                          upload=upload, label=label, n=n,
                          n_cal_points_per_state=n_cal_points_per_state,
                          cal_states=cal_states, last_ge_pulse=last_ge_pulse,
                          for_ef=for_ef, classified_ro=classified_ro,
                          prep_params=prep_params, exp_metadata=exp_metadata)

        #get pi and pi/2 amplitudes from the analysis results
        if analyze:
            rabi_ana = tda.RabiAnalysis(qb_names=[self.name])
            if update:
                amp180 = rabi_ana.proc_data_dict['analysis_params_dict'][
                    self.name]['piPulse']
                if not for_ef:
                    self.ge_amp180(amp180)
                    self.ge_amp90_scale(0.5)
                else:
                    self.ef_amp180(amp180)
                    self.ef_amp90_scale(0.5)

        return

    def find_T1(self, times, n_cal_points_per_state=2, cal_states='auto',
                upload=True, last_ge_pulse=False, classified_ro=False,
                prep_params=None, analyze=True, update=False, label=None,
                for_ef=False, exp_metadata=None, **kw):

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
            log.warning("Does not automatically update the qubit "
                        "T1 parameter. Set update=True if you want this!")
        if np.any(times > 1e-3):
            raise ValueError('Some of the values in the times array might be too '
                             'large. The units should be seconds.')

        if times is None:
            times_span = kw.get('times_span', 10e-6)
            times_mean = kw.get('times_mean', 5e-6)
            nr_points = kw.get('nr_points', 50)
            if times_mean == 0:
                log.warning("find_T1 does not know how long to wait before"
                            "doing the read out. Please specify the "
                            "times_mean or the times function parameter.")
                return 0
            else:
                times = np.linspace(times_mean - times_span / 2, times_mean +
                                    times_span / 2, nr_points)

        # Perform measurement
        self.measure_T1(times=times,
                        analyze=False, upload=upload,
                        last_ge_pulse=last_ge_pulse, for_ef=for_ef,
                        n_cal_points_per_state=n_cal_points_per_state,
                        cal_states=cal_states, classified_ro=classified_ro,
                        prep_params=prep_params, label=label,
                        exp_metadata=exp_metadata)

        # Extract T1 and T1_stddev from ma.T1_Analysis
        if analyze:
            T1_ana = tda.T1Analysis(qb_names=[self.name])
            if update:
                T1 = T1_ana.proc_data_dict['analysis_params_dict'][
                    self.name]['T1']
                if for_ef:
                    self.T1_ef(T1)
                else:
                    self.T1(T1)
            return T1

    def find_rb_gate_fidelity(self, cliffords, nr_seeds, label=None,
                              gate_decomposition='HZ', interleaved_gate=None,
                              thresholded=True, classified_ro=False,
                              n_cal_points_per_state=2, cal_states=(),
                              upload=True, analyze=True,
                              prep_params=None, exp_metadata=None, **kw):

        if cliffords is None:
            raise ValueError("Unspecified cliffords array")

        if label is None:
            if interleaved_gate is None:
                label = 'RB_{}_{}_seeds_{}_cliffords'.format(
                    gate_decomposition, nr_seeds,
                    cliffords[-1]) + self.msmt_suffix
            else:
                label = 'IRB_{}_{}_{}_seeds_{}_cliffords'.format(
                    interleaved_gate, gate_decomposition,
                    nr_seeds, cliffords[-1]) + self.msmt_suffix

        #Perform measurement
        self.measure_randomized_benchmarking(
            cliffords=cliffords, nr_seeds=nr_seeds,
            gate_decomp=gate_decomposition, interleaved_gate=interleaved_gate,
            n_cal_points_per_state=n_cal_points_per_state, cal_states=cal_states,
            classified_ro=classified_ro, thresholded=thresholded, label=label,
            upload=upload, analyze=False, prep_params=prep_params,
            exp_metadata=exp_metadata)

        #Analysis
        if analyze:
            pla.PipelineDataAnalysis()

    def find_frequency_T2_ramsey(self, times, artificial_detunings=None,
                                 upload=True, label=None, n=1,
                                 cal_states="auto", n_cal_points_per_state=2,
                                 analyze=True, update=False, for_ef=False,
                                 last_ge_pulse=False, classified_ro=False,
                                 prep_params=None, exp_metadata=None, **kw):
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
            artificial_detunings:     difference between drive frequency and
                                        qubit frequency estimated from
                                        qubit spectroscopy. Must be a list with
                                        one or two entries.
            upload:                  upload sequence to AWG
            update:                  update the qubit frequency and T2*
                                        parameters
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
            log.warning("Does not automatically update the qubit frequency "
                            "and T2_star parameters. "
                            "Set update=True if you want this!")
        if artificial_detunings is None:
            log.warning('Artificial_detuning is None; qubit driven at "%s" '
                        'estimated with spectroscopy' %self.f_qubit())
        if np.any(np.asarray(np.abs(artificial_detunings)) < 1e3):
            log.warning('The artificial detuning is too small.')
        if np.any(times > 1e-3):
            log.warning('The values in the times array might be too large.')

        self.measure_ramsey(times, artificial_detunings=artificial_detunings,
                            label=label, cal_states=cal_states, n=n,
                            n_cal_points_per_state=n_cal_points_per_state,
                            last_ge_pulse=last_ge_pulse, for_ef=for_ef,
                            classified_ro=classified_ro, upload=upload,
                            prep_params=prep_params, exp_metadata=exp_metadata,
                            analyze=False)

        # # Check if one or more artificial detunings
        if (hasattr(artificial_detunings, '__iter__') and
                (len(artificial_detunings) > 1)):
            multiple_detunings = True
        else:
            multiple_detunings = False

        if analyze:
            if multiple_detunings:
                ramsey_ana = ma.Ramsey_Analysis(
                    auto=True,
                    label=label,
                    qb_name=self.name,
                    NoCalPoints=len(cal_states)*n_cal_points_per_state,
                    for_ef=for_ef,
                    last_ge_pulse=last_ge_pulse,
                    artificial_detuning=artificial_detunings, **kw)

                # get new freq and T2* from analysis results
                new_qubit_freq = ramsey_ana.qubit_frequency  # value
                T2_star = ramsey_ana.T2_star['T2_star']  # dict

            else:
                ramsey_ana = tda.RamseyAnalysis(
                    qb_names=[self.name], options_dict=dict(
                        fit_gaussian_decay=kw.get('fit_gaussian_decay', True)))
                new_qubit_freq = ramsey_ana.proc_data_dict[
                    'analysis_params_dict'][self.name]['exp_decay_' + self.name][
                    'new_qb_freq']
                T2_star = ramsey_ana.proc_data_dict[
                    'analysis_params_dict'][self.name]['exp_decay_' + self.name][
                    'T2_star']

            if update:
                if for_ef:
                    try:
                        self.ef_freq(new_qubit_freq)
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                        'updated.'%e)
                    try:
                        self.T2_star_ef(T2_star)
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                        'updated.'%e)
                else:
                    try:
                        self.ge_freq(new_qubit_freq)
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                        'updated.'%e)
                    try:
                        self.T2_star(T2_star)
                    except AttributeError as e:
                        log.warning('%s. This parameter will not be '
                                        'updated.'%e)


    def find_T2_echo(self, times, artificial_detuning=None,
                     upload=True, label=None,
                     cal_points=True, no_cal_points=None,
                     analyze=True, for_ef=False,
                     close_fig=True, update=False,
                     last_ge_pulse=False, **kw):
        """
        Finds the qubit T2 Echo.
        Uses the EchoAnalysis class in timedomain_analysis.py.

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
            label:                   measurement label
            cal_points:              use calibration points or not
            analyze:                 perform analysis
            close_fig:               close the resulting figure
            update:                  update relevant parameters

        Keyword arguments:
            The time delays array 'times', or the parameter 'times_mean'
            should be passed here (in seconds).

        Returns:
            Nothing
        """
        if not update:
            log.warning("Does not automatically update the qubit "
                            "T2_echo parameter. "
                            "Set update=True if you want this!")
        if artificial_detuning == None:
            log.warning('Artificial_detuning is None; applying resonant '
                            'drive.')
        else:
            if np.any(np.asarray(np.abs(artificial_detuning)) < 1e3):
                log.warning('The artificial detuning is too small.')
        if np.any(times > 1e-3):
            log.warning('The values in the times array might be too large.')

        if cal_points and no_cal_points is None:
            log.warning('no_cal_points is None. Defaults to 4 if '
                            'for_ef==False, or to 6 if for_ef==True.')
            if for_ef:
                no_cal_points = 6
            else:
                no_cal_points = 4

        if not cal_points:
            no_cal_points = 0

        if label is None:
            if for_ef:
                label = 'Echo_ef' + self.msmt_suffix
            else:
                label = 'Echo' + self.msmt_suffix

        if times is None:
            times_span = kw.get('times_span', 5e-6)
            times_mean = kw.get('times_mean', 2.5e-6)
            nr_points = kw.get('nr_points', 50)
            if times_mean == 0:
                log.warning("find_T2_echo does not know "
                                "over which times to do Ramsey. Please "
                                "specify the times_mean or the times "
                                "function parameter.")
                return 0
            else:
                times = np.linspace(times_mean - times_span/2,
                                    times_mean + times_span/2,
                                    nr_points)

        # perform measurement
        if for_ef:
            self.measure_echo_2nd_exc(times=times,
                                      artificial_detuning=artificial_detuning,
                                      label=label, cal_points=cal_points,
                                      no_cal_points=no_cal_points, upload=upload,
                                      last_ge_pulse=last_ge_pulse)
        else:
            self.measure_echo(
                times=times, artificial_detuning=artificial_detuning,
                cal_points=cal_points,
                close_fig=close_fig, upload=upload, label=label)

        if analyze:
            echo_ana = tda.EchoAnalysis(
                qb_names=[self.name],
                options_dict={
                    'artificial_detuning': artificial_detuning,
                    'fit_gaussian_decay':
                                  kw.get('fit_gaussian_decay', True)})
            if update:
                T2_echo = echo_ana.proc_data_dict[
                    'analysis_params_dict'][self.name]['T2_echo']
                try:
                    self.T2(T2_echo)
                except AttributeError as e:
                    log.warning('%s. This parameter will not be '
                                    'updated.'%e)

        return

    def find_qscale(self, qscales, label=None, for_ef=False,
                    last_ge_pulse=False, upload=True, analyze=True,
                    cal_states="auto", n_cal_points_per_state=2,
                    classified_ro=False, prep_params=None,
                    exp_metadata=None, update=False, **kw):

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
            log.warning("Does not automatically update the qubit qscale "
                            "parameter. "
                            "Set update=True if you want this!")

        qscales = np.repeat(qscales, 3)

        #Perform the qscale calibration measurement
        self.measure_qscale(qscales=qscales, upload=upload, label=label,
                            cal_states=cal_states, exp_metadata=exp_metadata,
                            n_cal_points_per_state=n_cal_points_per_state,
                            last_ge_pulse=last_ge_pulse, for_ef=for_ef,
                            classified_ro=classified_ro,
                            prep_params=prep_params, analyze=False)

        # Perform analysis and extract the optimal qscale parameter
        # Returns the optimal qscale parameter
        if analyze:
            qscale_ana = tda.QScaleAnalysis(qb_names=[self.name])
            if update:
                qscale = qscale_ana.proc_data_dict['analysis_params_dict'][
                    self.name]['qscale']
                if for_ef:
                    self.ef_motzoi(qscale)
                else:
                    self.ge_motzoi(qscale)
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
            log.warning("Does not automatically update the qubit "
                            "anharmonicity parameter. "
                            "Set update=True if you want this!")

        if self.ge_freq() == 0:
            log.warning('f_ge = 0. Run qubit spectroscopy or Ramsey.')
        if self.ef_freq() == 0:
            log.warning('f_ef = 0. Run qubit spectroscopy or Ramsey.')

        anharmonicity = self.ef_freq() - self.ge_freq()

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
            log.warning("Does not automatically update the qubit EC and EJ "
                            "parameters. "
                            "Set update=True if you want this!")

        (EC,EJ) = a_tools.fit_EC_EJ(self.f_qubit(), self.f_ef_qubit(), **kw)

        if update:
            self.EC_qubit(EC)
            self.EJ_qubit(EJ)

        return EC, EJ

    def find_readout_frequency(self, freqs=None, update=False, MC=None,
                               qutrit=False, **kw):
        """
        Find readout frequency at which contrast between the states of the
        qubit is the highest.
        You need a working pi-pulse for this to work, as well as a pi_ef
        pulse if you intend to use `for_3_level_ro`. Also, if your
        readout pulse length is much longer than the T1, the results will not
        be nice as the excited state spectrum will be mixed with the ground
        state spectrum.

        Args:
            freqs: frequencies to sweep
            qutrit (bool): find optimal frequency for 3-level readout.
                                    Default is False.
            **kw:

        Returns:

        """
        # FIXME: Make proper analysis class for this (Ants, 04.12.2017)
        if not update:
            log.info("Does not automatically update the RO resonator "
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
            log.warning('Some of the values in the freqs array might be '
                            'too small. The units should be Hz.')
        if MC is None:
            MC = self.instr_mc.get_instr()

        levels = ('g', 'e', 'f') if qutrit else ('g', 'e')

        self.measure_dispersive_shift(freqs, states=levels, analyze=False)
        labels = {l: '{}-spec'.format(l) + self.msmt_suffix for l in levels}
        m_a = {l: ma.MeasurementAnalysis(label=labels[l]) for l in levels}
        trace = {l: m_a[l].measured_values[0] *
                    np.exp(1j * np.pi * m_a[l].measured_values[1] / 180.)
                 for l in levels}
        # FIXME: make something that doesn't require a conditional branching
        if qutrit:
            total_dist = np.abs(trace['e'] - trace['g']) + \
                         np.abs(trace['f'] - trace['g']) + \
                         np.abs(trace['f'] - trace['e'])
            fmax = freqs[np.argmax(total_dist)]
            # FIXME: just as debug plotting for now
            fig, ax = plt.subplots(2)
            ax[0].plot(freqs, np.abs(trace['g']), label='g')
            ax[0].plot(freqs, np.abs(trace['e']), label='e')
            ax[0].plot(freqs, np.abs(trace['f']), label='f')
            ax[0].set_ylabel('Amplitude')
            ax[0].legend()
            ax[1].plot(freqs, np.abs(trace['e'] - trace['g']), label='eg')
            ax[1].plot(freqs, np.abs(trace['f'] - trace['g']), label='fg')
            ax[1].plot(freqs, np.abs(trace['e'] - trace['f']), label='ef')
            ax[1].plot(freqs, total_dist, label='total distance')
            ax[1].set_xlabel("Freq. [Hz]")
            ax[1].set_ylabel('Distance in IQ plane')
            ax[0].set_title("Current RO_freq: {} Hz\nOptimal Freq: {} Hz".format(
                self.ro_freq(),
                                                                          fmax))
            plt.legend()

            m_a['g'].save_fig(fig, 'IQplane_distance')
            plt.show()
            if kw.get('analyze', True):
                sa.ResonatorSpectroscopy_v2(labels=[l for l in labels.values()])
        else:
            fmax = freqs[np.argmax(np.abs(trace['e'] - trace['g']))]

        log.info("Optimal RO frequency to distinguish states {}: {} Hz"
                     .format(levels, fmax))

        if kw.get('analyze', True):
            SA = sa.ResonatorSpectroscopy(t_start=[m_a['g'].timestamp_string,
                                                   m_a['e'].timestamp_string],
                                          options_dict=dict(simultan=True,
                                                            fit_options=dict(
                                                            model='hanger_with_pf'),
                                                            scan_label=''),
                                          do_fitting=True)
            # FIXME Nathan: remove 3 level dependency; fix this analysis:
            # if qutrit:
            #     SA2 = sa.ResonatorSpectroscopy(t_start=m_a['f'].timestamp_string,
            #                               options_dict=dict(simultan=False,
            #                                                 fit_options = dict(
            #                                                 model='hanger_with_pf'),
            #                                                 scan_label=''),
            #                               do_fitting=True)

            if update:
                # FIXME Nathan: update parameters accordingly
                self.ro_freq(SA.f_RO if not qutrit else fmax)
                self.chi(SA.chi)
                self.f_RO_resonator(SA.f_RO_res)
                self.f_RO_purcell(SA.f_PF)
                self.RO_purcell_kappa(SA.kappa)
                self.RO_J_coupling(SA.J_)
                if kw.pop('get_CLEAR_params', False):
                    if self.ro_CLEAR_segment_length is None:
                        self.ro_CLEAR_segment_length = self.ro_length/10
                    if kw.get('max_amp_difference', False) :
                        '''this gives the ratio of the maximal hight for'''
                        '''the segments to the base amplitude'''
                        max_diff = kw.pop('max_amp_difference')
                    else:
                        max_diff = 3
                    self.ro_CLEAR_delta_amp_segment = \
                        sim_CLEAR.get_CLEAR_amplitudes(
                            self.f_RO_purcell, self.f_RO_resonator,
                            self.ro_freq, self.RO_purcell_kappa,
                            self.RO_J_coupling, self.chi, 1, self.ro_length,
                            length_segments=self.ro_CLEAR_segment_length,
                            sigma=self.ro_sigma,
                            max_amp_diff=max_diff) * self.ro_amp


    def measure_dispersive_shift(self, freqs, analyze=True, close_fig=True,
                                 upload=True, states=('g','e'), prep_params=None):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """

        if freqs is None:
            raise ValueError("Unspecified frequencies for "
                             "measure_resonator_spectroscopy")
        if np.any(freqs < 500e6):
            log.warning(('Some of the values in the freqs array '
                         'might be too small. The units should be Hz.'))
        if prep_params is None:
            prep_params = self.preparation_params()

        assert isinstance(states, tuple), \
            "states should be a tuple, not {}".format(type(states))

        self.prepare(drive='timedomain')
        MC = self.instr_mc.get_instr()

        for state in states:
            sq.single_state_active_reset(
                    operation_dict=self.get_operation_dict(),
                    qb_name=self.name,
                    state=state, prep_params=prep_params, upload=upload)

            MC.set_sweep_function(self.swf_ro_freq_lo()) 
            MC.set_sweep_points(freqs)
            MC.set_detector_function(self.int_avg_det_spec)

            self.instr_pulsar.get_instr().start(exclude=[self.instr_uhf()])
            MC.run(name=f"{state}-spec" + self.msmt_suffix)
            self.instr_pulsar.get_instr().stop()

            if analyze:
                ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                    qb_name=self.name)

    def calibrate_flux_pulse_timing(self, freqs=None, delays=None, MC=None,
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
            MC = self.instr_mc.get_instr()

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

        self.prepare(drive='timedomain')

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
                new_delay = self.instr_pulsar.get_instr().get('{}_delay'.format(channel)) + \
                            flux_pulse_timing_ma.fitted_delay
                self.instr_pulsar.get_instr().set('{}_delay'.format(channel), new_delay)
                print('updated delay of channel {}.'.format(channel))
            else:
                log.warning('Not updated, since update was disabled.')
            return flux_pulse_timing_ma.fitted_delay
        else:
            return



    def calibrate_flux_pulse_frequency(self, MC=None, thetas=None, ampls=None,
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
            MC = self.instr_mc.get_instr()

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

        self.prepare(drive='timedomain')
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
            MC = self.instr_mc.get_instr()

        if flux_pulse_amp is None:
            flux_pulse_amp = self.flux_pulse_amp()
            log.warning('flux_pulse_amp is not specified. Using the value'
                            'in the flux_pulse_amp parameter.')
        if flux_pulse_length is None:
            flux_pulse_length = self.flux_pulse_length()
            log.warning('flux_pulse_length is not specified. Using the value'
                            'in the flux_pulse_length parameter.')
        if flux_pulse_delay is None:
            flux_pulse_delay = self.flux_pulse_delay()
            log.warning('flux_pulse_delay is not specified. Using the value'
                            'in the flux_pulse_delay parameter.')
        if flux_pulse_channel is None:
            flux_pulse_channel = self.flux_pulse_channel()
            log.warning('flux_pulse_channel is not specified. Using the value'
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
                    TwoD=True,
                    flux_pulse_amp=flux_pulse_amp,
                    flux_pulse_length=flux_pulse_length,
                    qb_name=self.name, **kw)

            dynamic_phase = MA.dyn_phase
            print('fitted dynamic phase on {}: {:0.3f} [deg]'.format(self.name,
                                                                dynamic_phase))
            if update:
                try:
                    self.dynamic_phase(dynamic_phase)
                except Exception:
                    log.warning('Could not update '
                                    '{}.dynamic_phase().'.format(self.name))

            return dynamic_phase
        else:
            return

    def measure_flux_pulse_scope(self, freqs, delays, cz_pulse_name,
                                 analyze=True, cal_points=True,
                                 upload=True, label=None,
                                 n_cal_points_per_state=2, cal_states='auto',
                                 prep_params=None, exp_metadata=None):
        '''
        flux pulse scope measurement used to determine the shape of flux pulses
        set up as a 2D measurement (delay and drive pulse frequecy are
        being swept)
        pulse sequence:
                      <- delay ->
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- fluxpulse ----- |


        Args:
            freqs (numpy array): array of drive frequencies
            delays (numpy array): array of delays of the drive pulse w.r.t
                the flux pulse
            pulse_length (float): flux pulse length (if not specified, the
                                    self.flux_pulse_length() is taken)
            pulse_amp (float): flux pulse amplitude  (if not specified, the
                                    self.flux_pulse_amp() is taken)
            pulse_delay (float): flux pulse delay
            MC (MeasurementControl): if None, then the self.MC is taken

        Returns: None

        '''
        if label is None:
            label = 'Flux_scope_{}'.format(self.name)
        MC = self.instr_mc.get_instr()
        self.prepare(drive='timedomain')

        if cal_points:
            cal_states = CalibrationPoints.guess_cal_states(cal_states)
            cp = CalibrationPoints.single_qubit(
                self.name, cal_states, n_per_state=n_cal_points_per_state)
        else:
            cp = None
        if prep_params is None:
            prep_params = self.preparation_params()
        seq, sweep_points, sweep_points_2D = \
            fsqs.fluxpulse_scope_sequence(
                delays=delays, freqs=freqs, qb_name=self.name,
                operation_dict=self.get_operation_dict(),
                cz_pulse_name=cz_pulse_name, cal_points=cp,
                prep_params=prep_params, upload=False)
        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Delay', unit='s'))
        MC.set_sweep_points(sweep_points)
        MC.set_sweep_function_2D(swf.Offset_Sweep(
            self.instr_ge_lo.get_instr().frequency,
            -self.ge_mod_freq(),
            name='Drive frequency',
            parameter_name='Drive frequency', unit='Hz'))
        MC.set_sweep_points_2D(sweep_points_2D)
        MC.set_detector_function(self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: delays},
                             'sweep_points_dict_2D': {self.name: freqs},
                             'use_cal_points': cal_points,
                             'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'rotate': cal_points,
                             'data_to_fit': {self.name: 'pe'},
                             "sweep_name": "Delay",
                             "sweep_unit": "s"})
        MC.run_2D(label, exp_metadata=exp_metadata)

        if analyze:
            try:
                tda.MultiQubit_TimeDomain_Analysis(qb_names=[self.name],
                                                   options_dict=dict(TwoD=True))
            except Exception:
                ma.MeasurementAnalysis(TwoD=True)

    def measure_flux_pulse_scope_nzcz_alpha(
            self, nzcz_alphas, delays, CZ_pulse_name=None,
            cal_points=True, upload=True, upload_all=True,
            spacing=30e-9, MC=None):

        if MC is None:
            MC = self.instr_mc.get_instr()

        self.prepare(drive='timedomain')

        if cal_points:
            step = np.abs(delays[-1] - delays[-2])
            sweep_points = np.concatenate(
                [delays, [delays[-1]+step, delays[-1]+2*step,
                          delays[-1]+3*step, delays[-1]+4*step]])
        else:
            sweep_points = delays

        s1 = awg_swf.Fluxpulse_scope_nzcz_alpha_hard_swf(
            qb_name=self.name, nzcz_alpha=nzcz_alphas[0],
            CZ_pulse_name=CZ_pulse_name,
            operation_dict=self.get_operation_dict(),
            cal_points=cal_points, upload=False,
            upload_all=upload_all, spacing=spacing)
        s2 = awg_swf.Fluxpulse_scope_nzcz_alpha_soft_sweep(
            s1, upload=upload)

        MC.set_sweep_function(s1)
        MC.set_sweep_points(sweep_points)
        MC.set_sweep_function_2D(s2)
        MC.set_sweep_points_2D(nzcz_alphas)
        MC.set_detector_function(self.int_avg_det)
        MC.run_2D('Flux_scope_nzcz_alpha' + self.msmt_suffix)

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
    op_name = 'upCZ ' + qbt.name
    ps_name = 'upCZ_' + qbt.name

    if np.any([op_name == i for i in qbc.get_operation_dict().keys()]):
        # do not try to add it again if operation already exists
        raise ValueError('Operation {} already exists.'.format(op_name))
    else:
        qbc.add_operation(op_name)

        qbc.add_pulse_parameter(op_name, ps_name + '_cz_target_qb',
                                'cz_target_qb',
                                initial_value=qbt.name,
                                vals=vals.Enum(qbt.name))
        qbc.add_pulse_parameter(op_name, ps_name + '_pulse_type', 'pulse_type',
                                initial_value='BufferedCZPulse',
                                vals=vals.Enum('BufferedSquarePulse',
                                               'BufferedCZPulse',
                                               'NZBufferedCZPulse',
                                               'BufferedCZPulseEffectiveTime'))
        qbc.add_pulse_parameter(op_name, ps_name + '_channel', 'channel',
                                initial_value='', vals=vals.Strings())
        qbc.add_pulse_parameter(op_name, ps_name + '_aux_channels_dict',
                                'aux_channels_dict',
                                initial_value={}, vals=vals.Dict())
        qbc.add_pulse_parameter(op_name, ps_name + '_amplitude', 'amplitude',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_frequency', 'frequency',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_phase', 'phase',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_pulse_length',
                                'pulse_length',
                                initial_value=0, vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_alpha', 'alpha',
                                initial_value=1, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_buffer_length_start',
                                'buffer_length_start', initial_value=10e-9,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_buffer_length_end',
                                'buffer_length_end', initial_value=10e-9,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_extra_buffer_aux_pulse',
                                'extra_buffer_aux_pulse', initial_value=5e-9,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_pulse_delay',
                                'pulse_delay',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_basis_rotation',
                                'basis_rotation', initial_value={},
                                vals=vals.Dict())
        qbc.add_pulse_parameter(op_name, ps_name + '_gaussian_filter_sigma',
                                'gaussian_filter_sigma', initial_value=2e-9,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_chevron_func',
                                'chevron_func', initial_value=None,
                                vals=vals.Callable(),
                                docstring="Callable required when using "
                                          "effective time CZ pulse to "
                                          "straighten Chevron.")


def add_CZ_MG_pulse(qbc, qbt):
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

        qbc.add_pulse_parameter(op_name, ps_name + '_cz_target_qb',
                                'cz_target_qb',
                                initial_value=qbt.name,
                                vals=vals.Enum(qbt.name))
        qbc.add_pulse_parameter(op_name, ps_name + '_pulse_type', 'pulse_type',
                                initial_value='NZMartinisGellarPulse',
                                vals=vals.Enum('NZMartinisGellarPulse',
                                               'BufferedCZPulse'))
        qbc.add_pulse_parameter(op_name, ps_name + '_channel', 'channel',
                                initial_value='', vals=vals.Strings())
        qbc.add_pulse_parameter(op_name, ps_name + '_aux_channels_dict',
                                'aux_channels_dict',
                                initial_value={}, vals=vals.Dict())
        qbc.add_pulse_parameter(op_name, ps_name + '_theta_f', 'theta_f',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_lambda_2', 'lambda_2',
                                initial_value=0, vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_qbc_freq', 'qbc_freq',
                                initial_value=qbc.ge_freq(),
                                vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_qbt_freq', 'qbt_freq',
                                initial_value=qbt.ge_freq(),
                                vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_pulse_length',
                                'pulse_length',
                                initial_value=0, vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_alpha', 'alpha',
                                initial_value=1, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_buffer_length_start',
                                'buffer_length_start', initial_value=10e-9,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_buffer_length_end',
                                'buffer_length_end', initial_value=10e-9,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_extra_buffer_aux_pulse',
                                'extra_buffer_aux_pulse', initial_value=5e-9,
                                vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_anharmonicity',
                                'anharmonicity',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_J', 'J',
                                initial_value=0, vals=vals.Numbers())
        qbc.add_pulse_parameter(op_name, ps_name + '_basis_rotation',
                                'basis_rotation', initial_value={},
                                vals=vals.Dict())
        qbc.add_pulse_parameter(op_name, ps_name + '_dv_dphi', 'dv_dphi',
                                initial_value=0, vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_loop_asym', 'loop_asym',
                                initial_value=0, vals=vals.Numbers(0))
        qbc.add_pulse_parameter(op_name, ps_name + '_wave_generation_func',
                                'wave_generation_func', initial_value=None,
                                vals=vals.Callable())
        qbc.add_pulse_parameter(op_name, ps_name + '_pulse_delay',
                                'pulse_delay',
                                initial_value=0, vals=vals.Numbers())










