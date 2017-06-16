import logging
import numpy as np

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit

class QuDev_transmon(Qubit):
    def __init__(self, name, MC,
                 heterodyne = None, # metainstrument for cw spectroscopy
                 cw_source = None, # MWG for driving the qubit continuously
                 readout_DC_LO = None, # MWG for downconverting RO signal
                 readout_UC_LO = None, # MWG for upconverting RO signal
                 readout_RF = None, # MWG for driving the readout resonator
                 drive_LO = None,
                 AWG = None,  # for generating IQ modulated drive pulses and
                              # triggering instruments
                 UHFQC = None, # For readout
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
                           vals=vals.Enum(0, 1, 2, 3, 4, 5),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_Q', initial_value=1,
                           vals=vals.Enum(None, 0, 1, 2, 3, 4, 5),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_shots', initial_value=4094,
                           docstring='Number of single shot measurements to do'
                                     'in single shot experiments.',
                           vals=vals.Ints(0, 4095),
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
        self.add_pulse_parameter('X180', 'amp90', 'amplitude_90',
                                 initial_value=0.5, vals=vals.Numbers())
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
        self.add_pulse_parameter('X180_ef', 'amp90_ef', 'amplitude_90',
                                 initial_value=0.5, vals=vals.Numbers())
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
        self.add_pulse_parameter('flux', 'flux_pulse_type', 'flux_pulse_type',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('flux', 'flux_pulse_I_channel', 'flux_I_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('flux', 'flux_pulse_Q_channel', 'flux_Q_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('flux', 'flux_pulse_amp', 'flux_amplitude',
                                 initial_value=1, vals=vals.Numbers())
        self.add_pulse_parameter('flux', 'flux_pulse_length', 'flux_length',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('flux', 'flux_pulse_delay', 'flux_pulse_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('flux', 'flux_f_pulse_mod', 'flux_mod_frequency',
                                 initial_value=None, vals=vals.Numbers())

        self.update_detector_functions()

    def update_detector_functions(self):
        if self.RO_acq_weight_function_Q() is None:
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

        self.inp_avg_det = det.UHFQC_input_average_detector(
            UHFQC=self.UHFQC, AWG=self.AWG, nr_averages=self.RO_acq_averages(),
            nr_samples=4096)

    def prepare_for_continuous_wave(self):
        self.heterodyne.auto_seq_loading(True)
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
        # Not working
        self.heterodyne.auto_seq_loading(False)
        if self.cw_source is not None:
            self.cw_source.pulsemod_state('On')
            self.cw_source.on()
            self.cw_source.power.set(self.spec_pow())

        if self.f_RO() is None:
            f_RO = self.f_RO_resonator()
        else:
            f_RO = self.f_RO()
        if self.RO_pulse_type() == 'Gated_MW_RO_pulse':
            self.readout_RF.frequency(f_RO)
            self.readout_RF.power(self.RO_pulse_power())
            self.readout_RF.on()
            self.UHFQC.awg_sequence_acquisition(acquisition_delay=0)
        elif self.RO_pulse_type() == 'MW_IQmod_pulse_UHFQC':
            eval('self.UHFQC.sigouts_{}_offset({})'.format(
                self.RO_I_channel(), self.RO_I_offset()))
            eval('self.UHFQC.sigouts_{}_offset({})'.format(
                self.RO_Q_channel(), self.RO_Q_offset()))
            self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(
                f_RO_mod=self.f_RO_mod(), RO_amp=self.RO_amp(),
                RO_pulse_length=self.RO_pulse_length(),
                acquisition_delay=0)
            self.readout_UC_LO.pulsemod_state('Off')
            self.readout_UC_LO.frequency(f_RO - self.f_RO_mod())
            self.readout_UC_LO.on()
        elif self.RO_pulse_type() is 'Multiplexed_UHFQC_pulse':
            # setting up the UHFQC awg sequence must be done externally by a
            # readout manager
            self.readout_UC_LO.pulsemod_state('Off')
            self.readout_UC_LO.frequency(f_RO - self.f_RO_mod())
            self.readout_UC_LO.on()



    def prepare_for_timedomain(self):
        # cw source
        if self.cw_source is not None:
            self.cw_source.off()

        self.update_detector_functions()

        # drive LO
        self.drive_LO.pulsemod_state('Off')
        self.drive_LO.frequency(self.f_qubit() - self.f_pulse_mod())
        self.drive_LO.power(self.drive_LO_pow())
        self.drive_LO.on()

        # drive modulation
        #self.AWG.set(self.pulse_I_channel() + '_offset',
        #             self.pulse_I_offset())
        #self.AWG.set(self.pulse_Q_channel() + '_offset',
        #             self.pulse_Q_offset())

        # readout LO
        if self.f_RO() is None:
            f_RO = self.f_RO_resonator()
        else:
            f_RO = self.f_RO()
        self.readout_DC_LO.pulsemod_state('Off')
        self.readout_DC_LO.frequency(f_RO - self.f_RO_mod())
        self.readout_DC_LO.on()

        # readout pulse
        if self.RO_pulse_type() is 'MW_IQmod_pulse_UHFQC':
            eval('self.UHFQC.sigouts_{}_offset({})'.format(
                self.RO_I_channel(), self.RO_I_offset()))
            eval('self.UHFQC.sigouts_{}_offset({})'.format(
                self.RO_Q_channel(), self.RO_Q_offset()))
            self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(
                f_RO_mod=self.f_RO_mod(), RO_amp=self.RO_amp(),
                RO_pulse_length=self.RO_pulse_length(),
                acquisition_delay=0)
            self.readout_UC_LO.pulsemod_state('Off')
            self.readout_UC_LO.frequency(f_RO - self.f_RO_mod())
            self.readout_UC_LO.on()
        elif self.RO_pulse_type() is 'Gated_MW_RO_pulse':
            self.readout_RF.pulsemod_state('On')
            self.readout_RF.frequency(f_RO)
            self.readout_RF.power(self.RO_pulse_power())
            self.readout_RF.on()
            self.UHFQC.awg_sequence_acquisition(acquisition_delay=0)
        elif self.RO_pulse_type() is 'Multiplexed_UHFQC_pulse':
            # setting up the UHFQC awg sequence must be done externally by a
            # readout manager
            self.readout_UC_LO.pulsemod_state('Off')
            self.readout_UC_LO.frequency(f_RO - self.f_RO_mod())
            self.readout_UC_LO.on()

    def measure_resonator_spectroscopy(self, freqs=None, MC=None,
                                        analyze=True, close_fig=True):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for measure_resonator_"
                             "spectroscopy")

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
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

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
                             analyze=True, close_fig=True):
        """ Varies qubit drive frequency and measures the resonator
        transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for measure_spectroscopy")

        if MC is None:
            MC = self.MC

        if not pulsed:

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
            MC.run(name='spectroscopy'+self.msmt_suffix)

            self.cw_source.off()

        else:
            self.prepare_for_pulsed_spec()

            spec_pars = self.get_spec_pars()
            RO_pars = self.get_RO_pars()

            self.cw_source.on()

            sq.Pulsed_spec_seq(spec_pars, RO_pars)

            self.AWG.start()

            MC.set_sweep_function(self.cw_source.frequency)
            MC.set_sweep_points(freqs)
            demod_mode = 'single' if self.heterodyne.single_sideband_demod() \
                else 'double'
            MC.set_detector_function(det.Heterodyne_probe(
                self.heterodyne,
                trigger_separation=self.heterodyne.trigger_separation(),
                demod_mode=demod_mode))
            MC.run(name='pulsed-spec' + self.msmt_suffix)

            self.cw_source.off()


        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

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

        # Specify the sweep function, the sweep points,
        # and the detector function, and run the measurement
        MC.set_sweep_function(awg_swf.Rabi(pulse_pars=self.get_drive_pars(),
                                           RO_pars=self.get_RO_pars(), n=n,
                                           cal_points=cal_points,
                                           no_cal_points=no_cal_points,
                                           upload=upload))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

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

        MC.set_sweep_function(awg_swf.Rabi_2nd_exc(
                        pulse_pars=self.get_drive_pars(),
                        pulse_pars_2nd=self.get_ef_drive_pars(),
                        RO_pars=self.get_RO_pars(),
                        last_ge_pulse=last_ge_pulse,
                        amps=amps, n=n, upload=upload,
                        cal_points=cal_points, no_cal_points=no_cal_points))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

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

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        # Define the measurement label
        if label is None:
            label = 'T1' + self.msmt_suffix

        MC.set_sweep_function(awg_swf.T1(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            upload=upload, cal_points=cal_points))
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_T1_2nd_exc(self, times=None, MC=None, analyze=True, upload=True,
                           close_fig=True, cal_points=True, no_cal_points=6,
                           label=None, last_ge_pulse=True):

        if times is None:
            raise ValueError("Unspecified times for measure_T1_2nd_exc")

        self.prepare_for_timedomain()

        if label is None:
            label = 'T1_2nd'+self.msmt_suffix

        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.T1_2nd_exc(
                                pulse_pars=self.get_drive_pars(),
                                pulse_pars_2nd=self.get_ef_drive_pars(),
                                RO_pars=self.get_RO_pars(),
                                upload=upload,
                                cal_points=cal_points,
                                no_cal_points=no_cal_points,
                                last_ge_pulse=last_ge_pulse))
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)


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

        MC.set_sweep_function(awg_swf.QScale(qscales=qscales,
                pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
                upload=upload, cal_points=cal_points))
        MC.set_sweep_points(qscales)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

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

        MC.set_sweep_function(awg_swf.QScale_2nd_exc(
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            upload=upload, cal_points=cal_points, no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse))
        MC.set_sweep_points(qscales)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_ramsey_multiple_detunings(self, times=None,
                                          artificial_detunings=None, label='',
                                          MC=None, analyze=True, close_fig=True,
                                          cal_points=True, upload=True):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        uniques = np.unique(times[range(len(artificial_detunings))])
        if uniques.size>1:
            raise ValueError("The values in the times array are not repeated "
                             "len(artificial_detunings) times.")

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        Rams_swf = awg_swf.Ramsey_multiple_detunings(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            artificial_detunings=artificial_detunings, cal_points=cal_points,
            upload=upload)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Ramsey_mult_det'+label+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)


    def measure_ramsey(self, times=None, artificial_detuning=0, label='',
                       MC=None, analyze=True, close_fig=True, cal_points=True,
                       upload=True):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # Define the measurement label
        if label is None:
            label = 'Ramsey' + self.msmt_suffix

        Rams_swf = awg_swf.Ramsey(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            artificial_detuning=artificial_detuning, cal_points=cal_points,
            upload=upload)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_ramsey_2nd_exc(self, times=None, artificial_detuning=0, label=None,
                       MC=None, analyze=True, close_fig=True, cal_points=True,
                       n=1, upload=True, last_ge_pulse=True, no_cal_points=6):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detuning is None:
            logging.warning('Artificial detuning is 0.')

        if label is None:
            label = 'Ramsey_2nd'+self.msmt_suffix

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        Rams_2nd_swf = awg_swf.Ramsey_2nd_exc(
            pulse_pars=self.get_drive_pars(),
            pulse_pars_2nd=self.get_ef_drive_pars(),
            RO_pars=self.get_RO_pars(),
            artificial_detuning=artificial_detuning,
            cal_points=cal_points, n=n, upload=upload,
            no_cal_points=no_cal_points,
            last_ge_pulse=last_ge_pulse)
        MC.set_sweep_function(Rams_2nd_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

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
                                        T1=None, MC=None, close_fig=True,
                                        upload=True, analyze=True):
        '''
        Performs a randomized benchmarking fidelity.
        Optionally specifying T1 also shows the T1 limited fidelity.
        '''

        if nr_cliffords is None:
            raise ValueError("Unspecified nr_cliffords for measure_echo")
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Randomized_Benchmarking(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            double_curves=True,
            nr_cliffords=nr_cliffords, nr_seeds=nr_seeds, upload=upload))
        MC.set_detector_function(self.int_avg_det)
        MC.run('RB_{}seeds'.format(nr_seeds)+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def set_default_readout_weights(self, channels = (0, 1)):
        """
        Sets the integration weights of the channels `RO_acq_weight_I` and
        `RO_acq_weight_Q` to the default sinusoidal values. The integration
        result of I channel is the integral of
        `cos(w*t)*ch0(t) + sin(w*t)*ch1(t)` and of Q channel is the integral of
        `cos(w*t)*ch0(t) - sin(w*t)*ch1(t)`.

        Args:
            channels: An iterable of UHFLI ports that are used for the input
                      signal. E.g. for using just the first input port one
                      would use `(0,)`, if using both input ports, one would use
                      the default value `(0, 1)`.
        """

        trace_length = 4096
        tbase = np.arange(0, trace_length / 1.8e9, 1 / 1.8e9)
        cosI = np.array(np.cos(2 * np.pi * self.f_RO_mod() * tbase))
        sinI = np.array(np.sin(2 * np.pi * self.f_RO_mod() * tbase))

        c1 = self.RO_acq_weight_function_I()
        c2 = self.RO_acq_weight_function_Q()

        if 0 in channels:
            self.UHFQC.set('quex_wint_weights_{}_real'.format(c1), cosI)
            self.UHFQC.set('quex_rot_{}_real'.format(c1), 1)
            if c2 is not None:
                self.UHFQC.set('quex_wint_weights_{}_real'.format(c2), cosI)
                self.UHFQC.set('quex_rot_{}_real'.format(c2), 1)
        else:
            self.UHFQC.set('quex_rot_{}_real'.format(c1), 0)
            if c2 is not None:
                self.UHFQC.set('quex_rot_{}_real'.format(c2), 0)
        if 1 in channels:
            self.UHFQC.set('quex_wint_weights_{}_imag'.format(c1), sinI)
            self.UHFQC.set('quex_rot_{}_imag'.format(c1), 1)
            if c2 is not None:
                self.UHFQC.set('quex_wint_weights_{}_imag'.format(c2), sinI)
                self.UHFQC.set('quex_rot_{}_imag'.format(c2), -1)
        else:
            self.UHFQC.set('quex_rot_{}_imag'.format(c1), 0)
            if c2 is not None:
                self.UHFQC.set('quex_rot_{}_imag'.format(c2), 0)

    def calibrate_readout_weights(self, MC=None, update=True, channels=(0, 1),
                                  close_fig=True, upload=True, filter_t=None,
                                  subtract_offset=False, **kw):
        """
        Sets the weight function of the UHFLI channel
        `self.RO_acq_weight_function_I` to the difference of the traces of the
        qubit prepared in the |0> and |1> state, which is optimal assuming that
        the standard deviation over different measurements of both traces is
        a constant. Sets `self.RO_acq_weight_function_Q` to `None`.

        Args:
            MC: MeasurementControl object to use for the measurement. Defaults
                to `self.MC`.
            update: Boolean flag, whether to update the UHFQC integration
                    weight. Default `True`.
            channels: An iterable of UHFLI ports that are used for the input
                      signal. E.g. for using just the first input port one
                      would use `(0,)`, if using both input ports, one would use
                      the default value `(0, 1)`.
            analyze: Boolean flag to run default default analysis generating
                     plots of the traces.
            close_fig: Boolean flag to close the matplotlib's figure. If
                       `False`, then the plots can be viewed with `plt.show()`
                       Default `True`.
            upload: Whether to reupload the AWG waveforms. Currently
                    unimplemented
            filter_t: The type of filter to apply to the measured difference
                      of the average g- and e-trace.

        Returns:
            Optimal weight(s) for state discrimination.
        """

        old_averages = self.RO_acq_averages()
        self.RO_acq_averages(2**15)

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.OffOn(
            pulse_pars=self.get_drive_pars(),
            RO_pars=self.get_RO_pars(),
            pulse_comb='OffOff'))
        MC.set_sweep_points(np.arange(2))
        MC.set_detector_function(self.inp_avg_det)
        MC.run(name='Weight_calib_0'+self.msmt_suffix)

        MA = ma.MeasurementAnalysis(auto=False)
        MA.get_naming_and_values()
        MA.data_file.close()
        data0 = MA.measured_values

        MC.set_sweep_function(awg_swf.OffOn(
            pulse_pars=self.get_drive_pars(),
            RO_pars=self.get_RO_pars(),
            pulse_comb='OnOn'))
        MC.set_sweep_points(np.arange(2))
        MC.set_detector_function(self.inp_avg_det)
        MC.run(name='Weight_calib_1' + self.msmt_suffix)

        MA = ma.MeasurementAnalysis(auto=False)
        MA.get_naming_and_values()
        MA.data_file.close()
        data1 = MA.measured_values

        self.RO_acq_averages(old_averages)

        weights_I = data1[0] - data0[0]
        weights_Q = data1[1] - data0[1]
        if subtract_offset:
            weights_I -= (np.min(weights_I) + np.max(weights_I))/2
            weights_Q -= (np.min(weights_Q) + np.max(weights_Q))/2

        contrast = np.sqrt(np.sum((data1[0] - data0[0])**2 + \
                                  (data1[1] - data0[1])**2))
        print('\ng-e trace contrast: {}'.format(contrast))

        if filter_t == 'gaussian':
            data = weights_I + 1j*weights_Q
            sigma = kw.get('filter_sigma', 10/self.f_RO_mod())

            kernel = np.arange(-5*sigma, 5*sigma, 1/1.8e9)
            kernel -= np.mean(kernel)
            kernel = np.exp(-kernel**2 / (2*sigma**2)) * \
                     np.exp(-2j*np.pi*kernel*self.f_RO_mod())

            pad_len = len(kernel)
            data = np.pad(data, (pad_len, pad_len), 'constant')
            data = np.convolve(data, kernel, mode='same')
            data = data[pad_len+1:-pad_len+1]

            weights_I = np.real(data)
            weights_Q = np.imag(data)
        elif filter_t == 'blackman-harris':
            data = weights_I + 1j*weights_Q
            sigma = kw.get('filter_sigma', 10/self.f_RO_mod())
            N = int(1.8e9*sigma/0.1385072985484252)
            n = np.arange(N)
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            envelope = -a3*np.cos(6*np.pi*n/(N-1))
            envelope += a2*np.cos(4*np.pi*n/(N-1))
            envelope -= a1*np.cos(2*np.pi*n/(N-1))
            envelope += a0
            times = n/1.8e9
            times -= np.mean(times)
            kernel = envelope * np.exp(-2j*np.pi*times*self.f_RO_mod())

            pad_len = len(kernel)
            data = np.pad(data, (pad_len, pad_len), 'constant')
            data = np.convolve(data, kernel, mode='same')
            data = data[pad_len+1:-pad_len+1]

            weights_I = np.real(data)
            weights_Q = np.imag(data)
        elif filter_t == 'uniform':
            data = weights_I + 1j*weights_Q
            tbase = np.linspace(0, len(data)/1.8, len(data), endpoint=False)
            p = np.abs(data)
            p /= np.sum(p)
            angle = np.sum(p*np.angle(data*np.exp(2j*np.pi*tbase*self.f_RO_mod())))
            data = np.exp(-2j*np.pi*tbase*self.f_RO_mod() + 1j*angle)

            weights_I = np.real(data)
            weights_Q = np.imag(data)
        elif filter_t == 'none':
            pass
        else:
            raise KeyError('Unknown filter type: {}'.format(filter_t))

        max_diff = np.max([np.max(np.abs(weights_I)),
                           np.max(np.abs(weights_Q))])
        weights_I /= max_diff
        weights_Q /= max_diff

        if update:
            c = self.RO_acq_weight_function_I()
            self.RO_acq_weight_function_Q(None)
            if 0 in channels:
                self.UHFQC.set('quex_wint_weights_{}_real'.format(c),
                               np.array(weights_I))
                self.UHFQC.set('quex_rot_{}_real'.format(c), 1)
            else:
                self.UHFQC.set('quex_rot_{}_real'.format(c), 0)
            if 1 in channels:
                self.UHFQC.set('quex_wint_weights_{}_imag'.format(c),
                               np.array(weights_Q))
                self.UHFQC.set('quex_rot_{}_imag'.format(c), 1)
            else:
                self.UHFQC.set('quex_rot_{}_imag'.format(c), 0)
            self.ssro_contrast(contrast)

        # format the return value to the same shape as channels.
        ret = []
        for chan in channels:
            if chan == 0:
                ret.append(weights_I)
            if chan == 1:
                ret.append(weights_Q)
        return tuple(ret)

    def find_ssro_fidelity(self, MC=None, analyze=True, close_fig=True,
                           no_fits=False, upload=True):
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
            MC: MeasurementControl object to use for the measurement. Defaults
                to `self.MC`.
            analyze: Boolean flag, whether to analyse the measurement results.
                     Default `True`.
            close_fig: Boolean flag to close the matplotlib's figure. If
                       `False`, then the plots can be viewed with `plt.show()`
                       Default `True`.
            no_fits: Boolean flag to disable finding the discrimination
                     fidelity. Default `False`.
        Returns:
            If `no_fits` is `False` returns assigment fidelity, discrimination
            fidelity and SNR = 2 |mu00 - mu11| / (sigma00 + sigma11). Else
            returns just assignment fidelity.
        """

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.OffOn(
            pulse_pars=self.get_drive_pars(),
            RO_pars=self.get_RO_pars(),
            upload=upload))
        MC.set_sweep_points(np.arange(self.RO_acq_shots()))
        MC.set_detector_function(self.int_log_det)

        MC.run(name='SSRO_fidelity'+self.msmt_suffix)

        if analyze:
            rotate = self.RO_acq_weight_function_Q() is not None
            channels = self.int_log_det.value_names
            ana = ma.SSRO_Analysis(auto=True, close_fig=close_fig,
                                   rotate=rotate, no_fits=no_fits,
                                   channels=channels)
            if not no_fits:
                return ana.F_a, ana.F_d, ana.SNR
            else:
                return ana.F_a

    def measure_flux_detuning(self, flux_params=None, n=1, ramsey_times=None,
                              artificial_detuning=0, MC=None,
                              analyze=True, close_fig=True, upload=True,
                              fluxing_channels=[]):
        """
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

        if MC is None:
            MC = self.MC

        self.measure_resonator_spectroscopy(freqs, MC, analyze=False)
        label = 'resonator_scan' + self.msmt_suffix
        HA = ma.Homodyne_Analysis(label=label, close_fig=close_fig,
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
                       MC=None, close_fig=True, analyze_ef=False, **kw):
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
            self.measure_spectroscopy(freqs, pulsed=False, MC=MC,
                                      close_fig=close_fig)
            label = 'spectroscopy'
        else:
            self.measure_spectroscopy(freqs, pulsed=True, MC=MC,
                                      close_fig=close_fig)
            label = 'pulsed-spec'

        if analyze_ef:
            label = 'high_power_' + label

        amp_only = hasattr(self.heterodyne, 'RF')
        SpecA = ma.Qubit_Spectroscopy_Analysis(
            analyze_ef=analyze_ef, label=label, amp_only=amp_only, close_fig=close_fig,**kw)
        self.f_qubit(SpecA.fitted_freq)
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

    def find_amplitudes(self, rabi_amps=None, label=None, for_ef=False,
                        update=False, MC=None, close_fig=True, cal_points=True,
                        no_cal_points=4, upload=True, last_ge_pulse=True, **kw):

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
                routine also accepts 4 cal pts.
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

        if not cal_points:
            no_cal_points = 0

            #how many times to apply the Rabi pulse
        n = kw.get('n',1)

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
        # TODO: might have to correct Rabi_Analysis_new to Rabi_Analysis
        # when we decide which version we stick to.

        RabiA = ma.Rabi_Analysis_new(label=label, NoCalPoints=no_cal_points,
                                     close_fig=close_fig, for_ef=for_ef,
                                     last_ge_pulse=last_ge_pulse, **kw)

        rabi_amps = RabiA.rabi_amplitudes    #This is a dict with keywords
        #'piPulse',  'piPulse_std',
        #'piHalfPulse', 'piHalfPulse_std

        amp180 = rabi_amps['piPulse']
        amp90 = rabi_amps['piHalfPulse']

        if update:
            if for_ef is False:
                self.amp180(amp180)
                self.amp90_scale(amp90/amp180)
                self.amp90(amp90)
            else:
                self.amp180_ef(amp180)
                self.amp90_scale_ef(amp90/amp180)
                self.amp90_ef(amp90)

        return rabi_amps


    def find_T1(self, times, label=None, for_ef=False, update=False, MC=None,
                cal_points=True, no_cal_points=6, close_fig=True,
                last_ge_pulse=True, **kw):

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
                                last_ge_pulse=last_ge_pulse)
        else:
            self.measure_T1(times=times, MC=MC,
                            close_fig=close_fig,
                            cal_points=cal_points)

        #Extract T1 and T1_stddev from ma.T1_Analysis
        if for_ef:
            NoCalPoints = 6
        else:
            NoCalPoints = 4
        T1_Analysis = ma.T1_Analysis(label=label, NoCalPoints=NoCalPoints, **kw)
        T1_dict = T1_Analysis.T1
        T1_value = T1_dict['T1']

        if update:
            if for_ef:
                self.T1_ef(T1_value)
            else:
                self.T1(T1_value)

        return T1_dict

    def find_frequency_T2_ramsey(self, times, for_ef=False, artificial_detuning=0, update=False, MC=None,
                                     cal_points=True, close_fig=True, upload=True,
                                     last_ge_pulse=True, label=None,
                                     no_cal_points=6,**kw):

        """
        Finds the real qubit frequency and the dephasing rate T2* from the fit
        to a Ramsey experiment.
        Uses the Ramsey_Analysis class from measurement_analysis.py
        The ef analysis assumes the the e population is zero (because of the ge
        X180 pulse at the end).

        WARNING: Does not automatically update the qubit freq and T2_star
        parameters. Set update=True if you want this!

        :param times                    array of times over which to sweep in
                                        the Ramsey measurement
        :param artificial_detuning:     difference between drive frequency and
                                        qubit frequency estimated from
                                        qubit spectroscopy
        :param update:                  update the qubit amp180 and amp90
                                        parameters
        :param MC:                      the measurement control object
        :param close_fig:               close the resulting figure?
        :param kw:                      other keyword arguments. The Rabi sweep
                                        time delays array 'times',
                                        or the parameter 'times_mean' should be
                                        passed here (in seconds)
        :return:                        the real qubit frequency
                                        (=self.f_qubit()+artificial_detuning-
                                        fitted_freq)
                                        + stddev, the dephasing rate T2* +
                                        stddev
        """
        if not update:
            logging.warning("Does not automatically update the qubit frequency "
                            "and T2_star parameters. "
                            "Set update=True if you want this!")

        if artificial_detuning == 0:
            logging.warning('Artificial_detuning=0; qubit driven at "%s" '
                            'estimated with '
                            'spectroscopy' %self.f_qubit())

        if MC is None:
            MC = self.MC

        if label is None:
            if for_ef:
                label = 'Ramsey_2nd' +self.msmt_suffix
            else:
                label = 'Ramsey' + self.msmt_suffix

        if times is None:
            times_span = kw.get('times_span', 5e-6)
            times_mean = kw.get('times_mean', 2.5e-6)
            nr_points = kw.get('nr_points', 50)
            if times_mean == 0:
                logging.warning("find_frequency_T2_ramsey does not know over "
                                "which times to do Ramsey. Please specify the "
                                "times_mean or the times function parameter.")
                return 0
            else:
                times = np.linspace(times_mean - times_span/2,
                                    times_mean + times_span/2,
                                    nr_points)
        #Perform Ramsey
        if for_ef is False:
            self.measure_ramsey(times=times,
                                artificial_detuning=artificial_detuning,
                                MC=MC,
                                cal_points=cal_points,
                                close_fig=close_fig, upload=upload)
            RamseyA = ma.Ramsey_Analysis(auto=True, label=label, **kw)
        else:
            self.measure_ramsey_2nd_exc(times=times, artificial_detuning=artificial_detuning, MC=MC,
                                        cal_points=cal_points, close_fig=close_fig, upload=upload,
                                        last_ge_pulse=last_ge_pulse, no_cal_points=no_cal_points)
            RamseyA = ma.Ramsey_Analysis(auto=True, NoCalPoints=6, label=label, **kw)

        #get new freq and T2* from analysis results

        fitted_freq = RamseyA.Ramsey_freq['freq']
        T2_star = RamseyA.T2_star

        qubit_freq = self.f_qubit() + artificial_detuning - fitted_freq

        print('New qubit frequency = {:.10f} \t stderr = {:.10f}'.format(
            qubit_freq,RamseyA.Ramsey_freq['freq_stderr']))
        print('T2_Star = {:.5f} \t stderr = {:.5f}'.format(
            T2_star['T2_star'],T2_star['T2_star_stderr']))

        if update:
            if for_ef:
                self.f_ef_qubit(qubit_freq)
                self.T2_star_ef(T2_star['T2_star'])
            else:
                self.f_qubit(qubit_freq)
                self.T2_star(T2_star['T2_star'])

        return qubit_freq, T2_star


    def calibrate_ramsey(self, times, for_ef=False,
                         artificial_detunings=None, update=False,
                         MC=None, cal_points=True, close_fig=True,
                         upload=True, last_ge_pulse=True, **kw):

        """
        Finds the real qubit frequency and the dephasing rate T2* from the fit
        to a Ramsey experiment.
        Uses the Ramsey_Analysis class from measurement_analysis.py
        The ef analysis assumes the the e population is zero (because of the ge
        X180 pulse at the end).

        WARNING: Does not automatically update the qubit freq and T2_star
        parameters. Set update=True if you want this!

        :param times                    array of times over which to sweep in
                                        the Ramsey measurement
        :param artificial_detuning:     difference between drive frequency and
                                        qubit frequency estimated from
                                        qubit spectroscopy
        :param update:                  update the qubit amp180 and amp90
                                        parameters
        :param MC:                      the measurement control object
        :param close_fig:               close the resulting figure?
        :param kw:                      other keyword arguments. The Rabi sweep
                                        time delays array 'times',
                                        or the parameter 'times_mean' should be
                                        passed here (in seconds)
        :return:                        the real qubit frequency
                                        (=self.f_qubit()+artificial_detuning-
                                        fitted_freq)
                                        + stddev, the dephasing rate T2* +
                                        stddev
        """
        if not update:
            logging.warning("Does not automatically update the qubit frequency "
                            "and T2_star parameters. "
                            "Set update=True if you want this!")

        if artificial_detunings is None:
            logging.warning('Artificial_detuning=0; qubit driven at "%s" '
                            'estimated with '
                            'spectroscopy' %self.f_qubit())

        if MC is None:
            MC = self.MC

        if times is None:
            times_span = kw.get('times_span', 5e-6)
            times_mean = kw.get('times_mean', 2.5e-6)
            nr_points = kw.get('nr_points', 50)
            if times_mean == 0:
                logging.warning("find_frequency_T2_ramsey does not know over "
                                "which times to do Ramsey. Please specify the "
                                "times_mean or the times function parameter.")
                return 0
            else:
                times = np.linspace(times_mean - times_span/2,
                                    times_mean + times_span/2,
                                    nr_points)

        # Each time value must be repeated len(artificial_detunings) times to
        # correspoond to the logic in Ramsey_seq_multiple_detunings sequence
        len_art_det = len(artificial_detunings)
        temp_array = np.zeros(times.size*len_art_det)
        for i in range(len(artificial_detunings)):
            np.put(temp_array,list(range(i,temp_array.size,len_art_det)),times)
        times = temp_array

        #Perform Ramsey
        if for_ef is False:
            self.measure_ramsey_multiple_detunings(times=times,
                                artificial_detunings=artificial_detunings,
                                MC=MC,
                                cal_points=cal_points,
                                close_fig=close_fig, upload=upload)
            RamseyA = ma.Ramsey_Analysis_mult_det(auto=True,
                                qubit_frequency_spec=self.f_qubit(),
                                artificial_detunings=artificial_detunings, **kw)
        else:
            self.measure_ramsey_2nd_mult_det(times=times,
                                artificial_detunings=artificial_detunings,
                                MC=MC, cal_points=cal_points,
                                close_fig=close_fig, upload=upload,
                                last_ge_pulse=last_ge_pulse)
            RamseyA = ma.Ramsey_Analysis_mult_det(auto=True, NoCalPoints=6,
                                qubit_frequency_spec=self.f_ef_qubit(),
                                artificial_detunings=artificial_detunings, **kw)

        #get new freq and T2* from analysis results
        new_qubit_freq = RamseyA.qubit_frequency    #value
        fitted_freq = RamseyA.ramsey_freq           #dict
        T2_star = RamseyA.T2_star                   #dict

        print('New qubit frequency = {:.10f} \t stderr = {:.10f}'.format(
            new_qubit_freq,RamseyA.Ramsey_freq['freq_stderr']))
        print('T2_Star = {:.5f} \t stderr = {:.5f}'.format(
            T2_star['T2_star'],T2_star['T2_star_stderr']))

        if update:
            if for_ef:
                self.f_ef_qubit(new_qubit_freq)
                self.T2_star_ef(T2_star['T2_star'])
            else:
                self.f_qubit(new_qubit_freq)
                self.T2_star(T2_star['T2_star'])

        return new_qubit_freq, fitted_freq, T2_star

    def find_qscale(self, qscales, label=None, for_ef=False, update=False,
                    MC=None, close_fig=True, last_ge_pulse=True, upload=False,
                    cal_points=True, no_cal_points=6, **kw):

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

        if MC is None:
            MC = self.MC

        if label is None:
            label = 'QScale' + self.msmt_suffix

        if qscales is None:
            qscales_span = kw.get('qscales_span', 3)
            qscales_mean = kw.get('qscales_mean', self.drag_qscale())
            nr_points = kw.get('nr_points', 30)
            if qscales_mean == 0:
                logging.warning("find_qscale does not know over which "
                                "qscale values to sweep. Please specify the "
                                "qscales_mean or the qscales function"
                                " parameter.")
                return 0
            else:
                qscales = np.linspace(qscales_mean - qscales_span/2,
                                      qscales_mean + qscales_span/2, nr_points)

        # Each qscale value must be repeated 3 times to correspoond to the
        # logic in QScale sequence
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
            NoCalPoints = 6
        else:
            self.measure_qscale(qscales=qscales, MC=MC, upload=upload,
                                close_fig=close_fig, label=label)
            NoCalPoints = 4

        # Perform analysis and extract the optimal qscale parameter
        # Returns the optimal qscale parameter
        QscaleA = ma.QScale_Analysis(auto=True, label=label,
                                     NoCalPoints=NoCalPoints, **kw)

        Qscale_dict = QscaleA.optimal_qscale #dictionary of value, stderr
        Qscale_value = Qscale_dict['qscale']

        if update:
            self.motzoi(Qscale_value)

        return Qscale_dict

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

    def find_dispersive_shift(self, freqs=None, label = 'pulsed-spec',
                              update=False, **kw):
        """
        Finds the dispersive shift chi (in MHz) but doing 2 pulsed
        spectroscopies, one where a pi pulse is applied beforehand, and one
        where no pi pulse is applied.

        WARNING: Does not automatically update the qubit EC and EJ parameters.
        Set update=True if you want this!

        Arguments:
            freqs            frequency range over which to sweep
            label            label of the analysis routine
            update:          whether to update the qubit chi parameter or not

        Keyword Arguments:
            f_span
            f_mean
            nr_points

        Returns: the dispersive shift + stderr
        """

        if not update:
            logging.warning("Does not automatically update the qubit "
                            "dispersive shift parameter. "
                            "Set update=True if you want this!")

        if freqs is None:
            f_span = kw.get('f_span', 100e6)
            f_mean = kw.get('f_mean', self.f_qubit())
            nr_points = kw.get('nr_points', 100)
            if f_mean == 0:
                logging.warning("find_dispersive_shift does not know over "
                                "what frequency range to sweep. "
                                "Please specify the "
                                "f_mean or the freqs function parameter.")
                return 0
            else:
                freqs = np.linspace(f_mean - f_span/2, f_mean + f_span/2,
                                    nr_points)

        #Perform measurements


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

