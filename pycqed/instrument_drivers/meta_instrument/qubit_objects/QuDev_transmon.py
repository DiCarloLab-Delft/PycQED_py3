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
                 readout_LO = None, # MWG for down- (and up)converting RO signal
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
        self.readout_LO = readout_LO
        self.readout_RF = readout_RF
        self.drive_LO = drive_LO

        self.add_parameter('f_RO_resonator', label='RO resonator frequency',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('Q_RO_resonator', label='RO resonator Q factor',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('optimal_acquisition_delay', label='Optimal '
                           'acquisition delay', unit='s', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('f_qubit', label='Qubit frequency', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('f_ef_qubit', label='Qubit ef frequency', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('T1', label='Qubit relaxation', unit='s',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('T2_star', label='Qubit dephasing', unit='s',
                           initial_value=0, parameter_class=ManualParameter)
        # self.add_parameter('amp180', label='Qubit pi pulse amp', unit='V',
        #                    initial_value=0, parameter_class=ManualParameter)
        # self.add_parameter('amp90', label='Qubit pi/2 pulse amp', unit='V',
        #                    initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('anharmonicity', label='Qubit anharmonicity', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
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
                           vals=vals.Numbers(0, 1e6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_integration_length', initial_value=2.2e-6,
                           vals=vals.Numbers(min_value=10e-9, max_value=2.2e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_I', initial_value=0,
                           vals=vals.Enum(0, 1, 2, 3, 4, 5),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_Q', initial_value=1,
                           vals=vals.Enum(0, 1, 2, 3, 4, 5),
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
        self.add_pulse_parameter('RO', 'RO_pulse_marker_channel', 'RO_pulse_marker_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('RO', 'RO_amp', 'amplitude',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'RO_pulse_length', 'length',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'RO_pulse_delay', 'pulse_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'f_RO_mod', 'mod_frequency',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'RO_acq_marker_delay', 'acq_marker_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'RO_acq_marker_channel', 'acq_marker_channel',
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
        self.add_pulse_parameter('X180_ef', 'pulse_type', 'pulse_type',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180_ef', 'pulse_I_channel', 'I_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180_ef', 'pulse_Q_channel', 'Q_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('X180_ef', 'amp180', 'amplitude',
                                 initial_value=1, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'amp90_scale', 'amp90_scale',
                                 initial_value=0.5, vals=vals.Numbers(0, 1))
        self.add_pulse_parameter('X180_ef', 'pulse_delay', 'pulse_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'gauss_sigma', 'sigma',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'nr_sigma', 'nr_sigma',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'motzoi', 'motzoi',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'f_pulse_mod', 'mod_frequency',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'phi_skew', 'phi_skew',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'alpha', 'alpha',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('X180_ef', 'X_pulse_phase', 'phase',
                                 initial_value=None, vals=vals.Numbers())

        # add flux pulse parameters
        self.add_operation('flux')
        self.add_pulse_parameter('flux', 'flux_pulse_type', 'pulse_type',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('flux', 'pulse_I_channel', 'I_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('flux', 'pulse_Q_channel', 'Q_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('flux', 'flux_pulse_amp', 'amplitude',
                                 initial_value=1, vals=vals.Numbers())
        self.add_pulse_parameter('flux', 'flux_pulse_length', 'length',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('flux', 'flux_pulse_delay', 'pulse_delay',
                                 initial_value=None, vals=vals.Numbers())
        self.add_pulse_parameter('flux', 'f_pulse_mod', 'mod_frequency',
                                 initial_value=None, vals=vals.Numbers())

        self.update_detector_functions()

    def update_detector_functions(self):
        self.int_log_det = det.UHFQC_integration_logging_det(
            UHFQC=self.UHFQC, AWG=self.AWG, channels=[
                self.RO_acq_weight_function_I(),
                self.RO_acq_weight_function_Q()],
            integration_length=self.RO_acq_integration_length())

        self.int_avg_det = det.UHFQC_integrated_average_detector(
            self.UHFQC, self.AWG, nr_averages=self.RO_acq_averages(),
            channels=[self.RO_acq_weight_function_I(),
                      self.RO_acq_weight_function_Q()],
            integration_length=self.RO_acq_integration_length())

    def prepare_for_continuous_wave(self):
        self.heterodyne.auto_seq_loading(True)
        if self.cw_source is not None:
            self.cw_source.off()
            self.cw_source.pulsemod_state('Off')
            self.cw_source.power.set(self.spec_pow())
        if self.readout_RF is not None:
            self.readout_RF.pulsemod_state('Off')
        self.readout_LO.pulsemod_state('Off')

    def prepare_for_pulsed_spec(self):
        self.heterodyne.auto_seq_loading(False)
        if self.cw_source is not None:
            self.cw_source.pulsemod_state('On')
            self.cw_source.on()
            self.cw_source.power.set(self.spec_pow())
        self.readout_RF.pulsemod_state('On')

        #TODO add settings from prepare_for_timedomain?


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
        self.AWG.set(self.pulse_I_channel() + '_offset',
                     self.pulse_I_offset())
        self.AWG.set(self.pulse_Q_channel() + '_offset',
                     self.pulse_Q_offset())

        # readout LO
        if self.f_RO() is None:
            f_RO = self.f_RO_resonator()
        else:
            f_RO = self.f_RO()
        self.readout_LO.pulsemod_state('Off')
        self.readout_LO.frequency(f_RO - self.f_RO_mod())
        self.readout_LO.on()

        # readout pulse
        if self.RO_pulse_type() is 'MW_IQmod_pulse_UHFQC':
            eval('self._acquisition_instr.sigouts_{}_offset({})'.format(
                self.RO_I_channel(), self.RO_I_offset()))
            eval('self._acquisition_instr.sigouts_{}_offset({})'.format(
                self.RO_Q_channel(), self.RO_Q_offset()))
            self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(
                f_RO_mod=self.f_RO_mod(), RO_amp=self.RO_amp(),
                RO_pulse_length=self.RO_pulse_length(),
                acquisition_delay=0)
        elif self.RO_pulse_type() is 'Gated_MW_RO_pulse':
            self.readout_RF.pulsemod_state('On')
            self.readout_RF.frequency(f_RO)
            self.readout_RF.power(self.RO_pulse_power())
            self.readout_RF.on()
            self.UHFQC.awg_sequence_acquisition(acquisition_delay=0)

    def measure_heterodyne_spectroscopy(self, freqs=None, MC=None,
                                        analyze=True, close_fig=True):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """
        if freqs is None:
            raise ValueError("Unspecified frequencies for measure_heterodyne_"
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

    def measure_rabi(self, amps=None, n=1, MC=None, analyze=True,
                     close_fig=True, cal_points=True, upload=True):

        if amps is None:
            raise ValueError("Unspecified amplitudes for measure_rabi")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Rabi(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(), n=n,
            cal_points=cal_points, upload=upload))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi-n{}'.format(n) + self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_rabi_2nd_exc(self, amps=None, n=1, MC=None, analyze=True,
                     close_fig=True, cal_points=True, upload=True):

        if amps is None:
            raise ValueError("Unspecified amplitudes for measure_rabi")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Rabi_2nd_exc(
            pulse_pars=self.get_drive_pars(), pulse_pars_2nd=self.get_drive_pars(ef_transition=True),
            RO_pars=self.get_RO_pars(), amps=amps, n=n, cal_points=cal_points, upload=upload))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi_2nd_exc-n{}'.format(n) + self.msmt_suffix)

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
                   close_fig=True):

        if times is None:
            raise ValueError("Unspecified times for measure_T1")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.T1(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            upload=upload))
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('T1'+self.msmt_suffix)

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

        Rams_swf = awg_swf.Ramsey(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(),
            artificial_detuning=artificial_detuning, cal_points=cal_points,
            upload=upload)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Ramsey'+label+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_ramsey_2nd_exc(self, times=None, artificial_detuning=0, label='',
                       MC=None, analyze=True, close_fig=True, cal_points=True,
                       n=None, upload=True):

        if times is None:
            raise ValueError("Unspecified times for measure_ramsey")
        if artificial_detuning is None:
            logging.warning('Artificial detuning is 0.')

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        Rams_2nd_swf = awg_swf.Ramsey_2nd_exc(
            pulse_pars=self.get_drive_pars(), pulse_pars_2nd=self.get_drive_pars(ef_transition=True),
            RO_pars=self.get_RO_pars(), times=times,
            artificial_detuning=artificial_detuning, cal_points=cal_points,
            n=n, upload=upload)
        MC.set_sweep_function(Rams_2nd_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Ramsey_2nd_exc'+label+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_echo(self, times=None, MC=None, artificial_detuning=None,
                     upload=True, analyze=True):

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
                      analyze=True):
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

    def measure_ssro(self):
        raise NotImplementedError()

    def measure_flux_detuning(self, flux_params=None, n=1, ramsey_times=None, artificial_detuning=0, MC=None,
                              analyze=True, close_fig=True, upload=True,fluxing_channels=[]):
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
            raise ValueError("Unspecified flux amplitudes for measure_flux_detuning")
        if ramsey_times is None:
            raise ValueError("Unspecified Ramsey times for measure_flux_detuning")

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
        #Soft sweep over flux amplitudes using awg_swf.awg_seq_swf(fsqs.single_pulse_seq)
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

    def find_resonator_frequency(self, update=True, freqs=None, MC=None,
                                 close_fig=True):
        """
        Finds the resonator frequency by performing a heterodyne experiment
        if freqs == None it will determine a default range dependent on the
        last known frequency of the resonator.
        """
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

        self.measure_heterodyne_spectroscopy(freqs, MC, analyze=False)

        HA = ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig,
                                  fitting_model='lorentzian')
        f0 = HA.fit_results.params['f0'].value
        df0 = HA.fit_results.params['f0'].stderr
        Q = HA.fit_results.params['Q'].value
        dQ = HA.fit_results.params['Q'].stderr
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

    def find_homodyne_acqusition_delay(self, delays=None, update=True, MC=None,
                                         close_fig=True):
        """
        Finds the acquisition delay for a homodyne experiment that corresponds
        to maximal signal strength.
        """
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

    def find_frequency(self, method='cw_spectroscopy', update=True, MC=None,
                       close_fig=True, analyze_ef=False, **kw):
        if MC is None:
            MC = self.MC

        if 'spectroscopy' in method.lower():
            freqs = kw.get('freqs', None)
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

            amp_only = hasattr(self.heterodyne, 'RF')
            SpecA = ma.Qubit_Spectroscopy_Analysis(
                analyze_ef=analyze_ef, label=label, amp_only=amp_only, close_fig=close_fig)
            f0 = self.f_qubit(SpecA.fitted_freq)
            if analyze_ef:
                f0_ef = self.f_ef_qubit(2*SpecA.fitted_freq_gf_over_2 - f0)
            else:
                f0_ef = self.f_ef_qubit()
            if update:
                self.f_qubit(f0)
                self.f_ef_qubit(f0_ef)
            return f0, f0_ef
        else:
            raise ValueError("Unknown method '{}' for "
                             "find_frequency".format(method))

    def find_amplitudes(self, for_ef=False, update=True, MC=None, close_fig=True, **kw):

        """
        Finds the pi and pi/2 pulse amplitudes from the fit to a Rabi experiment. Uses the Rabi_Analysis(_new)
        class from measurement_analysis.py

        :param update:      update the qubit amp180 and amp90 parameters
        :param MC:          the measurement control object
        :param close_fig:   close the resulting figure?
        :param kw:          other keyword arguments. The Rabi sweep amplitudes array 'amps', or the parameter
                            'amps_mean' should be passed here
        :return:            the pi-pulse and pi/2-pulse amplitudes and standard deviations
        """

        if MC is None:
            MC = self.MC

        n = kw.get('n',1)  #how many times to apply the Rabi pulse for each amplitude

        #get amplitude sweep from **kw
        amps = kw.get('amps',None)
        if amps is None:
            amps_span = kw.get('amps_span', 1.)
            amps_mean = kw.get('amps_mean', self.amp180())
            nr_points = kw.get('nr_points', 30)
            if amps_mean == 0:
                logging.warning("find_amplitudes does not know over which "
                                "amplitudes to do Rabi. Please specify the "
                                "amps_mean or the amps function parameter.")
                return 0
            else:
                amps = np.linspace(amps_mean - amps_span/2, amps_mean + amps_span/2,
                                    nr_points)
        #Perform Rabi
        if for_ef is False:
            self.measure_rabi(amps=amps, n=n, MC=MC, close_fig=close_fig)
        else:
            self.measure_rabi_2nd_exc(amps=amps, n=n, MC=MC, close_fig=close_fig)

        #get pi and pi/2 amplitudes from the analysis results
        # TODO: might have to correct Rabi_Analysis_new to Rabi_Analysis when we decide which version we stick to.
        RabiA = ma.Rabi_Analysis_new(close_fig=close_fig)
        rabi_amps = RabiA.rabi_amps    #This is a dict with keywords 'piPulse',  'piPulse_std',
                                       #'piHalfPulse', 'piHalfPulse_std

        amp180 = rabi_amps.pop('piPulse')
        amp90 = rabi_amps.pop('piHalfPulse')

        if update:
            self.amp180(amp180)
            self.amp90_scale(amp90)

        return rabi_amps

    def find_T1(self, for_ef=False, update=True, MC=None, close_fig=True, **kw):

        """
        Finds the relaxation time T1 from the fit to a Ramsey experiment.
        Uses the Ramsey_Analysis class from measurement_analysis.py

        :param artificial_detuning:     difference between drive frequency and qubit frequency estimated from
                                        qubit spectroscopy
        :param update:                  update the qubit amp180 and amp90 parameters
        :param MC:                      the measurement control object
        :param close_fig:               close the resulting figure?
        :param kw:                      other keyword arguments. The Rabi sweep time delays array 'times',
                                        or the parameter 'times_mean' should be passed here (in seconds)
        :return:                        the real qubit frequency (=self.f_qubit()+artificial_detuning-fitted_freq)
                                        + stddev, the dephasing rate T2* + stddev
        """

    def find_frequency_T2_ramsey(self, for_ef=False, artificial_detuning=0, update=True, MC=None,
                                 close_fig=True, **kw):

        """
        Finds the real qubit frequency and the dephasing rate T2* from the fit to a Ramsey experiment.
        Uses the Ramsey_Analysis class from measurement_analysis.py

        :param artificial_detuning:     difference between drive frequency and qubit frequency estimated from
                                        qubit spectroscopy
        :param update:                  update the qubit amp180 and amp90 parameters
        :param MC:                      the measurement control object
        :param close_fig:               close the resulting figure?
        :param kw:                      other keyword arguments. The Rabi sweep time delays array 'times',
                                        or the parameter 'times_mean' should be passed here (in seconds)
        :return:                        the real qubit frequency (=self.f_qubit()+artificial_detuning-fitted_freq)
                                        + stddev, the dephasing rate T2* + stddev
        """

        if artificial_detuning == 0:
            logging.warning('Artificial_detuning=0; qubit driven at "%s" estimated with '
                            'spectroscopy' %self.f_qubit())

        if MC is None:
            MC = self.MC

        #get times sweep from **kw
        times = kw.get('times',None)
        if times is None:
            times_span = kw.get('times_span', 5e-6)
            times_mean = kw.get('times_mean', 2.5e-6)
            nr_points = kw.get('nr_points', 50)
            if times_mean == 0:
                logging.warning("find_qubit_frequency_ramsey does not know over which"
                                "times to do Ramsey. Please specify the "
                                "times_mean or the times function parameter.")
                return 0
            else:
                times = np.linspace(times_mean - times_span/2, times_mean + times_span/2,
                                   nr_points)
        #Perform Ramsey
        if for_ef is False:
            self.measure_ramsey(times=times, artificial_detuning=artificial_detuning, MC=MC,
                            close_fig=close_fig)
        else:
            self.measure_ramsey_2nd_exc(times=times, artificial_detuning=artificial_detuning, MC=MC,
                                        close_fig=close_fig)

        #get new freq and T2* from analysis results
        RamseyA = ma.Ramsey_Analysis(auto=True)
        fitted_freq = RamseyA.get_measured_freq()       #two element list with [fitted_freq, stddev]
        T2_star = RamseyA.get_measured_T2_star()        #two element list with [T2_star, stddev]

        qubit_freq = self.f_qubit() + artificial_detuning - fitted_freq[0]

        if update:
            self.f_qubit(qubit_freq)
            self.T2_star(T2_star[0])

        return fitted_freq, T2_star

    def find_anharmonicity(self, update=True):

        """
        Computes the qubit anaharmonicity using f_ef and f_ge. It is assumed that
        the latter have already been found.
        """

        if self.f_qubit() == 0:
            logging.warning('f_ge = 0. Run qubit spectroscopy or ramsey.')
        if self.f_ef_qubit() == 0:
            logging.warning('f_ef = 0. Run qubit spectroscopy or ramsey.')

        anharmonicity = self.f_ef_qubit() - self.f_qubit()

        if update:
            self.anharmonicity(anharmonicity)

        return  anharmonicity

    def find_EC_EJ(self, update=True, **kw):

        """
        Extracts EC and EJ from a least squares fit to the transmon Hamiltonian solutions.
        It uses a_tools.calculate_transom_transitions, f_ge and f_ef.
        **kw should include the following optional keywords:
            asym:               asymmetry d (Koch (2007), eqn 2.18) for asymmetric junctions
            reduced_flux:       reduced magnetic flux through SQUID
            dim:                dimension of Hamiltonian will  be (2*dim+1,2*dim+1)
        """

        (EC,EJ) = a_tools.fit_EC_EJ(self.f_qubit(), self.f_ef_qubit(), **kw)

        if update:
            self.EC_qubit(EC)
            self.EJ_qubit(EJ)

        return EC, EJ

    def get_spec_pars(self):
        return self.get_operation_dict()['Spec ' + self.name]

    def get_RO_pars(self):
        return self.get_operation_dict()['RO ' + self.name]

    def get_drive_pars(self, ef_transition=False):
        if ef_transition:
            return self.get_operation_dict()['X180_ef ' + self.name]
        else:
            return self.get_operation_dict()['X180 ' + self.name]

    def get_flux_pars(self):
        return self.get_operation_dict()['flux ' + self.name]

    def get_operation_dict(self, operation_dict=None):
        if operation_dict is None:
            operation_dict = {}
        operation_dict = super().get_operation_dict(operation_dict)
        operation_dict['Spec ' + self.name]['operation_type'] = 'MW'
        operation_dict['RO ' + self.name]['operation_type'] = 'RO'
        operation_dict['X180 ' + self.name]['operation_type'] = 'MW'
        operation_dict.update(add_suffix_to_dict_keys(
            sq.get_pulse_dict_from_pars(operation_dict['X180 ' + self.name]),
            ' ' + self.name))
        return operation_dict

