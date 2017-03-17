import logging
import numpy as np

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.analysis import measurement_analysis as ma
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
        self.add_parameter('optimal_acquisition_delay', label='Optimal '
                           'acquisition delay', unit='s', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('f_qubit', label='Qubit frequency', unit='Hz',
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
        if self.readout_DC_LO is not None:
            self.readout_DC_LO.pulsemod_state('Off')
        if self.readout_UC_LO is not None:
            self.readout_UC_LO.pulsemod_state('Off')

    def prepare_for_pulsed_spec(self):
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
                     close_fig=True, upload=True):

        if amps is None:
            raise ValueError("Unspecified amplitudes for measure_rabi")

        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Rabi(
            pulse_pars=self.get_drive_pars(), RO_pars=self.get_RO_pars(), n=n,
            upload=upload))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi-n{}'.format(n) + self.msmt_suffix)

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

    def calibrate_ssro(self, no_fits=False,
                       return_detector=False,
                       MC=None,
                       analyze=True,
                       close_fig=True,
                       verbose=True, SSB=False,
                       one_weight_function_UHFQC=False,
                       multiplier=1, nr_shots=4095):
        """
        Calibrates the integration weights of the UHFLI for state neasurement.
        :param no_fits:
        :param return_detector:
        :param MC:
        :param analyze:
        :param close_fig:
        :param verbose:
        :param SSB:
        :param one_weight_function_UHFQC:
        :param multiplier:
        :param nr_shots:
        :return:
        """
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        d = cdet.SSRO_Fidelity_Detector_Tek(
            'SSRO'+self.msmt_suffix,
            analyze=analyze,
            raw=no_fits,
            MC=MC,
            AWG=self.AWG,
            acquisition_instr=self.UHFQC,
            pulse_pars=self.get_drive_pars(),
            RO_pars=self.get_RO_pars(),
            IF=self.f_RO_mod(),
            weight_function_I=self.RO_acq_weight_function_I(),
            weight_function_Q=self.RO_acq_weight_function_Q(),
            nr_shots=nr_shots,
            one_weight_function_UHFQC=one_weight_function_UHFQC,
            optimized_weights=True,
            integration_length=self.RO_acq_integration_length(),
            close_fig=close_fig,
            SSB=SSB,
            multiplier=multiplier,
            nr_averages=self.RO_acq_averages())
        if return_detector:
            return d
        d.prepare()
        d.acquire_data_point()

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
                       close_fig=True, **kw):
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
                label=label, amp_only=amp_only, close_fig=close_fig)
            f0 = self.f_qubit(SpecA.fitted_freq)
            if update:
                self.f_qubit(f0)
            return f0
        else:
            raise ValueError("Unknown method '{}' for "
                             "find_frequency".format(method))

    def get_spec_pars(self):
        return self.get_operation_dict()['Spec ' + self.name]

    def get_RO_pars(self):
        return self.get_operation_dict()['RO ' + self.name]

    def get_drive_pars(self):
        return self.get_operation_dict()['X180 ' + self.name]

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