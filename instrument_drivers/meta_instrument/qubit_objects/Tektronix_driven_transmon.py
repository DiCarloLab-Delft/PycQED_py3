import logging
import numpy as np
from scipy.optimize import brent
from math import gcd
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.measurement import mc_parameter_wrapper as pw

from modules.measurement import sweep_functions as swf
from modules.measurement import CBox_sweep_functions as cb_swf
from modules.measurement import awg_sweep_functions as awg_swf
from modules.analysis import measurement_analysis as ma
from modules.measurement.pulse_sequences import standard_sequences as st_seqs
import modules.measurement.randomized_benchmarking.randomized_benchmarking as rb
from modules.measurement.calibration_toolbox import mixer_carrier_cancellation_5014
from modules.measurement.calibration_toolbox import mixer_skewness_calibration_5014
from modules.measurement.optimization import nelder_mead

from .qubit_object import Transmon
from .CBox_driven_transmon import CBox_driven_transmon
# It would be better to inherit from Transmon directly and put all the common
# stuff in there but for now I am inheriting from what I already have
# MAR april 2016


class Tektronix_driven_transmon(CBox_driven_transmon):
    '''
    Setup configuration:
        Drive:                 Tektronix 5014 AWG
        Acquisition:           CBox
            (in the future to be compatible with both CBox and ATS)

    Readout pulse configuration:
        Set by parameter RO_pulse_type ['MW_IQmod_pulse', 'Gated_MW_RO_pulse']
        - LO modulated using AWG: 'MW_IQmod_pulse'
        - LO + RF-pulsed with marker: 'Gated_MW_RO_pulse'
        Depending on the RO_pulse_type some parameters are not used
    '''
    shared_kwargs = ['LO', 'cw_source', 'td_source', 'IVVI', 'AWG', 'CBox',
                     'heterodyne_instr', 'rf_RO_source', 'MC']

    def __init__(self, name,
                 LO, cw_source, td_source,
                 IVVI, AWG, CBox,
                 heterodyne_instr, MC, rf_RO_source=None, **kw):
        super(CBox_driven_transmon, self).__init__(name, **kw)
        # Change this when inheriting directly from Transmon instead of
        # from CBox driven Transmon.
        self.LO = LO
        self.cw_source = cw_source
        self.td_source = td_source
        self.rf_RO_source = rf_RO_source
        self.IVVI = IVVI
        self.heterodyne_instr = heterodyne_instr
        self.AWG = AWG
        self.CBox = CBox
        self.MC = MC

        self.add_parameter('mod_amp_cw', label='RO modulation ampl cw',
                           units='V', initial_value=0.5,
                           parameter_class=ManualParameter)

        self.add_parameter('RO_power_cw', label='RO power cw',
                           units='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pow', label='spectroscopy power',
                           units='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pow_pulsed',
                           label='pulsed spectroscopy power',
                           units='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('td_source_pow',
                           label='Time-domain power',
                           units='dBm',
                           parameter_class=ManualParameter)
        # Rename f_RO_mod
        # Time-domain parameters
        self.add_parameter('pulse_I_channel', initial_value='ch1',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_Q_channel', initial_value='ch2',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_I_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)
        self.add_parameter('pulse_Q_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)

        # These parameters are only relevant if using MW_IQmod_pulse type
        # RO
        self.add_parameter('RO_I_channel', initial_value='ch3',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_Q_channel', initial_value='ch4',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_I_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_Q_offset', initial_value=0.0,
                           vals=vals.Numbers(min_value=-0.1, max_value=0.1),
                           parameter_class=ManualParameter)

        self.add_parameter('RO_pulse_type', initial_value='MW_IQmod_pulse',
                           vals=vals.Enum('MW_IQmod_pulse',
                                          'Gated_MW_RO_pulse'),
                           parameter_class=ManualParameter)
        # Relevant when using a marker channel to gate a MW-RO tone.
        self.add_parameter('RO_pulse_marker_channel',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_pulse_power', units='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('f_pulse_mod',
                           initial_value=-100e6,
                           label='pulse-modulation frequency', units='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_RO_mod',
                           label='Readout-modulation frequency', units='Hz',
                           initial_value=-2e7,
                           parameter_class=ManualParameter)
        # Used in calculating the fixed point frequency, if set to 0 it
        # has no effect
        self.add_parameter('f_JPA_pump_mod',
                           label='JPA pump modulation frequency', units='Hz',
                           initial_value=0,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 1e9))
        self.add_parameter('amp180',
                           label='Pi-pulse amplitude', units='V',
                           initial_value=.25,
                           vals=vals.Numbers(min_value=-0.5, max_value=0.5),
                           parameter_class=ManualParameter)
        self.add_parameter('amp90_scale',
                           label='pulse amplitude scaling factor', units='',
                           initial_value=.5,
                           vals=vals.Numbers(min_value=0, max_value=1.0),
                           parameter_class=ManualParameter)
        self.add_parameter('gauss_sigma', units='s',
                           initial_value=10e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('motzoi', label='Motzoi parameter', units='',
                           initial_value=0,
                           parameter_class=ManualParameter)

        self.add_parameter('phi_skew', label='IQ phase skewness', units='deg',
                           vals=vals.Numbers(-180, 180),
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('alpha', label='QI amplitude skewness', units='',
                           vals=vals.Numbers(.1, 2),
                           initial_value=1,
                           parameter_class=ManualParameter)

        # Single shot readout specific parameters
        self.add_parameter('RO_threshold', units='dac-value',
                           initial_value=0,
                           parameter_class=ManualParameter)
        # CBox specific parameter
        self.add_parameter('signal_line', parameter_class=ManualParameter,
                           vals=vals.Enum(0, 1), initial_value=0)

        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily
        self.int_avg_det = det.CBox_integrated_average_detector(self.CBox,
                                                                self.AWG)
        self.int_log_det = det.CBox_integration_logging_det(self.CBox,
                                                            self.AWG)
        self.input_average_detector = det.CBox_input_average_detector(
            self.CBox, self.AWG)

    def prepare_for_timedomain(self):
        self.LO.on()
        self.cw_source.off()
        self.td_source.on()

        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        self.td_source.frequency.set(self.f_qubit.get()
                                     - self.f_pulse_mod.get())

        # Use resonator freq unless explicitly specified
        if self.f_RO.get() is None:
            f_RO = self.f_res.get()
        else:
            f_RO = self.f_RO.get()
        self.LO.frequency.set(f_RO - self.f_RO_mod.get())
        self.td_source.power.set(self.td_source_pow.get())
        self.get_pulse_pars()

        self.AWG.set(self.pulse_I_channel.get()+'_offset',
                     self.pulse_I_offset.get())
        self.AWG.set(self.pulse_Q_channel.get()+'_offset',
                     self.pulse_Q_offset.get())

        if self.RO_pulse_type.get() is 'MW_IQmod_pulse':
            self.AWG.set(self.RO_I_channel.get()+'_offset',
                         self.RO_I_offset.get())
            self.AWG.set(self.RO_Q_channel.get()+'_offset',
                         self.RO_Q_offset.get())
        elif self.RO_pulse_type.get() is 'Gated_MW_RO_pulse':
            self.rf_RO_source.on()
            self.rf_RO_source.pulsemod_state.set('on')
            self.rf_RO_source.frequency.set(self.f_RO.get())
            self.rf_RO_source.power.set(self.RO_pulse_power.get())

    def calibrate_mixer_offsets(self, signal_hound, offs_type='pulse',
                                update=True):
        '''
        input:
            signal_hound: instance of the SH instrument
            offs_type:         ['pulse' | 'RO'] whether to calibrate the
                                            RO or pulse IQ offsets
            update:        update the values in the qubit object

        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        signal hound needs to be given as it this is not part of the qubit
        object in order to reduce dependencies.
        '''
        # ensures freq is set correctly
        # Still need to test this, start by doing this in notebook
        self.prepare_for_timedomain()
        self.AWG.stop()  # Make sure no waveforms are played
        if offs_type == 'pulse':
            AWG_channel1 = self.pulse_I_channel.get()
            AWG_channel2 = self.pulse_Q_channel.get()
            source = self.td_source
        elif offs_type == 'RO':
            AWG_channel1 = self.RO_I_channel.get()
            AWG_channel2 = self.RO_Q_channel.get()
            source = self.LO
        else:
            raise ValueError('offs_type "{}" not recognized'.format(offs_type))

        offset_I, offset_Q = mixer_carrier_cancellation_5014(
            AWG=self.AWG, SH=signal_hound, source=source, MC=self.MC,
            AWG_channel1=AWG_channel1, AWG_channel2=AWG_channel2)

        if update:
            if offs_type == 'pulse':
                self.pulse_I_offset.set(offset_I)
                self.pulse_Q_offset.set(offset_Q)
            if offs_type == 'RO':
                self.RO_I_offset.set(offset_I)
                self.RO_Q_offset.set(offset_Q)

    def calibrate_mixer_skewness(self, signal_hound, station, update=True):
        '''
        Calibrates mixer skewness at the frequency relevant for qubit driving.

        Note: I don't like that you have to pass station here but I don't want
        to introduce extra variables at this point, it should be available to
        you in the notebook (MAR).
        '''
        self.prepare_for_timedomain()

        phi, alpha = mixer_skewness_calibration_5014(
            signal_hound, self.td_source, station,
            f_mod=self.f_pulse_mod.get(),
            I_ch=self.pulse_I_channel.get(), Q_ch=self.pulse_Q_channel.get(),
            name='Mixer_skewness'+self.msmt_suffix)
        if update:
            self.phi_skew.set(phi)
            self.alpha.set(alpha)

    def calibrate_RO_threshold(self, method='conventional',
                               MC=None, close_fig=True,
                               verbose=False, make_fig=True):
        raise NotImplementedError()

    def measure_rabi(self, amps=np.linspace(-.5, .5, 31), n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Rabi(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, n=n))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi-n{}'.format(n)+self.msmt_suffix)
        if analyze:
            ma.Rabi_Analysis(auto=True, close_fig=close_fig)

    def measure_rabi_amp90(self,
                    amp90_scales=np.linspace(-1.5, 1.5, 31), n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Rabi_amp90(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, n=n))
        MC.set_sweep_points(amp90_scales)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi_amp90_scales-n{}'.format(n)+self.msmt_suffix)
        if analyze:
            ma.Rabi_Analysis(auto=True, close_fig=close_fig)

    def measure_T1(self, times, MC=None,
                   analyze=True, close_fig=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.T1(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars))
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('T1'+self.msmt_suffix)
        if analyze:
            a = ma.T1_Analysis(auto=True, close_fig=close_fig)
            return a.T1

    def measure_ramsey(self, times, artificial_detuning=None,
                       f_qubit=None, label='',
                       MC=None, analyze=True, close_fig=True, verbose=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        if f_qubit is None:
            f_qubit = self.f_qubit.get()
        self.td_source.set('frequency', f_qubit - self.f_pulse_mod.get())
        Rams_swf = awg_swf.Ramsey(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            artificial_detuning=artificial_detuning)
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Ramsey'+label+self.msmt_suffix)

        if analyze:
            a = ma.Ramsey_Analysis(auto=True, close_fig=close_fig)
            if verbose:
                fitted_freq = a.fit_res.params['frequency'].value
                print('Artificial detuning: {:.2e}'.format(
                      artificial_detuning))
                print('Fitted detuning: {:.2e}'.format(fitted_freq))
                print('Actual detuning:{:.2e}'.format(
                      fitted_freq-artificial_detuning))

    def measure_echo(self, times, label='', MC=None,
                     analyze=True, close_fig=True, verbose=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        Echo_swf = awg_swf.Echo(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars)
        MC.set_sweep_function(Echo_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Echo'+label+self.msmt_suffix)

        if analyze:
            a = ma.Echo_analysis(auto=True, close_fig=close_fig)
            return a

    def measure_allxy(self, double_points=True,
                      MC=None,
                      analyze=True, close_fig=True, verbose=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.AllXY(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            double_points=double_points))
        MC.set_detector_function(self.int_avg_det)
        MC.run('AllXY'+self.msmt_suffix)

        if analyze:
            a = ma.AllXY_Analysis(close_main_fig=close_fig)
            return a

    def measure_randomized_benchmarking(self, nr_cliffords,
                                        nr_seeds=50, T1=None,
                                        MC=None, analyze=True, close_fig=True,
                                        verbose=False):
        '''
        Performs a randomized benchmarking fidelity.
        Optionally specifying T1 also shows the T1 limited fidelity.
        '''
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        MC.set_sweep_function(awg_swf.Randomized_Benchmarking(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            double_curves=True,
            nr_cliffords=nr_cliffords, nr_seeds=nr_seeds))
        MC.set_detector_function(self.int_avg_det)
        MC.run('RB_{}seeds'.format(nr_seeds)+self.msmt_suffix)
        ma.RB_double_curve_Analysis(
            close_main_fig=close_fig, T1=T1,
            pulse_delay=self.pulse_delay.get())

    def measure_ssro(self, no_fits=False,
                     return_detector=False,
                     MC=None,
                     analyze=True,
                     close_fig=True,
                     verbose=True, set_integration_weights=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        d = cdet.SSRO_Fidelity_Detector_Tek(
            'SSRO'+self.msmt_suffix,
            analyze=return_detector,
            raw=no_fits,
            MC=MC,
            AWG=self.AWG, CBox=self.CBox,
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            set_integration_weights=set_integration_weights, close_fig=close_fig)
        if return_detector:
            return d
        d.prepare()
        d.acquire_data_point()
        if analyze:
            return ma.SSRO_Analysis(label='SSRO'+self.msmt_suffix,
                                    no_fits=no_fits, close_fig=close_fig)

    def measure_butterfly(self, return_detector=False,
                          MC=None,
                          analyze=True,
                          close_fig=True,
                          verbose=True,
                          initialize=False,
                          post_msmt_delay=2e-6, case=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        MC.set_sweep_function(awg_swf.Butterfly(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            initialize=initialize, post_msmt_delay=post_msmt_delay))
        MC.set_detector_function(self.int_log_det)
        MC.run('Butterfly{}initialize_{}'.format(self.msmt_suffix, initialize))
        # first perform SSRO analysis to extract the optimal rotation angle theta
        a = ma.SSRO_discrimination_analysis(
            label='Butterfly',
            current_threshold=None,
            close_fig=close_fig,
            plot_2D_histograms=True)
        #the, run it a second time to determin the optimum threshold along the rotated I axis
        b = ma.SSRO_discrimination_analysis(
            label='Butterfly',
            current_threshold=None,
            close_fig=close_fig,
            plot_2D_histograms=True, theta_in=-a.theta)

        c = ma.butterfly_analysis(
            close_main_fig=close_fig, initialize=initialize,
            theta_in=-a.theta,
            threshold=b.opt_I_threshold, digitize=True, case=case)
        return c.butterfly_coeffs

    def measure_transients(self, return_detector=False,
                           MC=None,
                           analyze=True,
                           close_fig=True,
                           verbose=True,
                           set_integration_weights=False,
                           nr_samples = 512):
        if set_integration_weights:
            print("always using 512 samples to set the weightfunction")
            self.CBox.nr_samples.set(512)
        else:
            self.CBox.nr_samples.set(nr_samples)

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        self.MC.set_sweep_function(awg_swf.OffOn(pulse_pars=self.pulse_pars,
                                                 RO_pars=self.RO_pars,
                                                 pulse_comb='OffOff',
                                                 nr_samples=nr_samples))
        self.MC.set_detector_function(self.input_average_detector)
        self.MC.run('Measure_transients_{}_0'.format(self.msmt_suffix))
        a0 = ma.MeasurementAnalysis(auto=True, close_fig=close_fig)
        self.MC.set_sweep_function(awg_swf.OffOn(pulse_pars=self.pulse_pars,
                                                 RO_pars=self.RO_pars,
                                                 pulse_comb='OnOn',
                                                 nr_samples=nr_samples))
        self.MC.set_detector_function(self.input_average_detector)
        self.MC.run('Measure_transients_{}_1'.format(self.msmt_suffix))
        a1 = ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

        if set_integration_weights:
            transient0 = a0.data[1, :]
            transient1 = a1.data[1, :]
            optimized_weights = transient1-transient0
            optimized_weights = optimized_weights+np.mean(optimized_weights)
            self.CBox.sig0_integration_weights.set(optimized_weights)
            self.CBox.sig1_integration_weights.set(np.zeros(512)) #disabling the Q quadrature



        # first perform SSRO analysis to extract the optimal rotation angle theta
        #return c.butterfly_coeffs



    def measure_rb_vs_amp(self, amps, nr_cliff=1,
                      resetless=True,
                      MC=None, analyze=True, close_fig=True,
                      verbose=False):
        raise NotImplementedError()


    def measure_motoi_XY(self, motzois, MC=None, analyze=True, close_fig=True,
                         verbose=True, update=True):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        MC.set_sweep_function(awg_swf.Motzoi_XY(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, motzois=motzois))
        MC.set_detector_function(self.int_avg_det)
        MC.run('Motzoi_XY'+self.msmt_suffix)

        if analyze:
            a = ma.Motzoi_XY_analysis(close_fig=close_fig)
            if update:
                self.motzoi.set(a.optimal_motzoi)
            return a

    def get_pulse_pars(self):
        self.pulse_pars = {
            'I_channel': self.pulse_I_channel.get(),
            'Q_channel': self.pulse_Q_channel.get(),
            'amplitude': self.amp180.get(),
            'amp90_scale': self.amp90_scale(),
            'sigma': self.gauss_sigma.get(),
            'nr_sigma': 4,
            'motzoi': self.motzoi.get(),
            'mod_frequency': self.f_pulse_mod.get(),
            'pulse_delay': self.pulse_delay.get(),
            'phi_skew': self.phi_skew.get(),
            'alpha': self.alpha.get(),
            'phase': 0,
            'pulse_type': 'SSB_DRAG_pulse'}

        self.RO_pars = {
            'I_channel': self.RO_I_channel.get(),
            'Q_channel': self.RO_Q_channel.get(),
            'RO_pulse_marker_channel': self.RO_pulse_marker_channel.get(),
            'amplitude': self.RO_amp.get(),
            'length': self.RO_pulse_length.get(),
            'pulse_delay': self.RO_pulse_delay.get(),
            'mod_frequency': self.f_RO_mod.get(),
            'fixed_point_frequency': gcd(int(self.f_RO_mod()),
                                         int(self.f_JPA_pump_mod())),
            'acq_marker_delay': self.RO_acq_marker_delay.get(),
            'acq_marker_channel': self.RO_acq_marker_channel.get(),
            'phase': 0,
            'pulse_type': self.RO_pulse_type.get()}
        return self.pulse_pars, self.RO_pars
