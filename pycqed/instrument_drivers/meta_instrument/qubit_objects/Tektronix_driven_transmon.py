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

from .qubit_object import Transmon
from .CBox_driven_transmon import CBox_driven_transmon
# It would be better to inherit from Transmon directly and put all the common
# stuff in there but for now I am inheriting from what I already have
# MAR april 2016


class Tektronix_driven_transmon(CBox_driven_transmon):

    '''
    Setup configuration:
        Drive:                 Tektronix 5014 AWG
        Acquisition:           CBox or UHFQC
            (in the future to be compatible with both CBox and ATS)

    Readout pulse configuration:
        Set by parameter RO_pulse_type ['MW_IQmod_pulse', 'Gated_MW_RO_pulse']
        - LO modulated using AWG: 'MW_IQmod_pulse'
        - LO + RF-pulsed with marker: 'Gated_MW_RO_pulse'
        Depending on the RO_pulse_type some parameters are not used
    '''

    def __init__(self, name, **kw):
        super(CBox_driven_transmon, self).__init__(name, **kw)
        # Change this when inheriting directly from Transmon instead of
        # from CBox driven Transmon.

        # Adding instrument parameters
        self.add_parameter('LO', parameter_class=InstrumentParameter)
        self.add_parameter('cw_source', parameter_class=InstrumentParameter)
        self.add_parameter('td_source', parameter_class=InstrumentParameter)
        self.add_parameter('IVVI', parameter_class=InstrumentParameter)
        self.add_parameter('FluxCtrl', parameter_class=InstrumentParameter)

        self.add_parameter('AWG', parameter_class=InstrumentParameter)

        self.add_parameter('heterodyne_instr', parameter_class=InstrumentParameter)

        self.add_parameter('LutMan', parameter_class=InstrumentParameter)
        self.add_parameter('CBox', parameter_class=InstrumentParameter)
        self.add_parameter('MC', parameter_class=InstrumentParameter)

        self.add_parameter('RF_RO_source',
                           parameter_class=InstrumentParameter)


        self.add_parameter('mod_amp_cw', label='RO modulation ampl cw',
                           unit='V', initial_value=0.5,
                           parameter_class=ManualParameter)

        self.add_parameter('RO_power_cw', label='RO power cw',
                           unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pow', label='spectroscopy power',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pow_pulsed',
                           label='pulsed spectroscopy power',
                           unit='dBm',
                           parameter_class=ManualParameter)
        self.add_parameter('td_source_pow',
                           label='Time-domain power',
                           unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('spec_pulse_type', label='Pulsed spec pulse type',
                           parameter_class=ManualParameter,
                           initial_value='SquarePulse',
                           vals=vals.Enum('SquarePulse'))  # , SSB_DRAG_pulse))
        # we should also implement SSB_DRAG_pulse for pulsed spec
        self.add_parameter('spec_pulse_length',
                           label='Pulsed spec pulse duration',
                           unit='s',
                           vals=vals.Numbers(1e-9, 20e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pulse_marker_channel',
                           unit='s',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('spec_pulse_depletion_time',
                           unit='s',
                           vals=vals.Numbers(1e-9, 50e-6),
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
        # readout parameters for time domain
        self.add_parameter('RO_acq_averages', initial_value=1024,
                           vals=vals.Numbers(min_value=0, max_value=1e6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_integration_length', initial_value=1e-6,
                           vals=vals.Numbers(
                               min_value=10e-9, max_value=1000e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_I', initial_value=0,
                           vals=vals.Ints(0, 5),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_weight_function_Q', initial_value=1,
                           vals=vals.Ints(0, 5),
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

        self.add_parameter('RO_pulse_type', initial_value='MW_IQmod_pulse_tek',
                           vals=vals.Enum('MW_IQmod_pulse_tek',
                                          'MW_IQmod_pulse_UHFQC',
                                          'Gated_MW_RO_pulse'),
                           parameter_class=ManualParameter)
        # Relevant when using a marker channel to gate a MW-RO tone.
        self.add_parameter('RO_pulse_marker_channel',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('RO_pulse_power', unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('f_pulse_mod',
                           initial_value=-100e6,
                           label='pulse-modulation frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_RO_mod',
                           label='Readout-modulation frequency', unit='Hz',
                           initial_value=-2e7,
                           parameter_class=ManualParameter)

        self.add_parameter('amp180',
                           label='Pi-pulse amplitude', unit='V',
                           initial_value=.25,
                           vals=vals.Numbers(min_value=-2.25, max_value=2.25),
                           parameter_class=ManualParameter)
        self.add_parameter('amp90_scale',
                           label='pulse amplitude scaling factor', unit='',
                           initial_value=.5,
                           vals=vals.Numbers(min_value=0, max_value=1.0),
                           parameter_class=ManualParameter)
        self.add_parameter('gauss_sigma', unit='s',
                           initial_value=10e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('motzoi', label='Motzoi parameter', unit='',
                           initial_value=0,
                           parameter_class=ManualParameter)

        self.add_parameter('phi_skew', label='IQ phase skewness', unit='deg',
                           vals=vals.Numbers(-180, 180),
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('alpha', label='QI amplitude skewness', unit='',
                           vals=vals.Numbers(.1, 2),
                           initial_value=1,
                           parameter_class=ManualParameter)

        # Single shot readout specific parameters
        self.add_parameter('RO_threshold', unit='dac-value',
                           initial_value=0,
                           parameter_class=ManualParameter)
        # CBox specific parameter
        self.add_parameter('signal_line', parameter_class=ManualParameter,
                           vals=vals.Enum(0, 1), initial_value=0)
        self.add_parameter('acquisition_instr',
                           set_cmd=self._do_set_acquisition_instr,
                           get_cmd=self._do_get_acquisition_instr,
                           vals=vals.Strings())

        self.add_parameter('flux_pulse_buffer',
                           label='Flux pulse buffer', unit='s',
                           initial_value=0.,
                           vals=vals.Numbers(min_value=0., max_value=50e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('fluxing_channel', initial_value='ch1',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        self.add_parameter('fluxing_amp',
                           label='SWAP resolution', unit='V',
                           initial_value=.5,
                           vals=vals.Numbers(min_value=-1., max_value=1.),
                           parameter_class=ManualParameter)
        self.add_parameter('SWAP_amp',
                           label='SWAP amplitude', unit='V',
                           initial_value=0.02,
                           vals=vals.Numbers(min_value=0.02, max_value=4.5),
                           parameter_class=ManualParameter)
        self.add_parameter('SWAP_time',
                           label='SWAP Time', unit='s',
                           initial_value=0.,
                           vals=vals.Numbers(min_value=0., max_value=1e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('flux_dead_time',
                           label='Time between flux pulse and comp.', unit='s',
                           initial_value=0.,
                           vals=vals.Numbers(min_value=0., max_value=50e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('mw_to_flux_delay',
                           label='time between and mw pulse and start of flux pulse', unit='s',
                           initial_value=0.,
                           vals=vals.Numbers(min_value=0., max_value=50e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('dist_dict',
                           get_cmd=self.get_dist_dict,
                           set_cmd=self.set_dist_dict,
                           vals=vals.Anything())

    def get_dist_dict(self):
        return self._dist_dict

    def set_dist_dict(self, dist_dict):
        self._dist_dict = dist_dict

    def prepare_for_continuous_wave(self):
        # makes sure the settings of the acquisition instrument are reloaded
        self.acquisition_instr(self.acquisition_instr())
        self.heterodyne_instr.get_instr().acquisition_instr(self.acquisition_instr())
        # Heterodyne tone configuration
        if not self.f_RO():
            RO_freq = self.f_res()
        else:
            RO_freq = self.f_RO()

        self.heterodyne_instr.get_instr()._disable_auto_seq_loading = False

        self.heterodyne_instr.get_instr().RF.on()
        self.heterodyne_instr.get_instr().LO.on()
        if hasattr(self.heterodyne_instr.get_instr(), 'mod_amp'):
            self.heterodyne_instr.get_instr().set('mod_amp', self.mod_amp_cw.get())
        else:
            self.heterodyne_instr.get_instr().RF_power(self.RO_power_cw())
        self.heterodyne_instr.get_instr().set('f_RO_mod', self.f_RO_mod.get())
        self.heterodyne_instr.get_instr().frequency.set(RO_freq)
        self.heterodyne_instr.get_instr().RF.power(self.RO_power_cw())
        self.heterodyne_instr.get_instr().RF_power(self.RO_power_cw())
        self.heterodyne_instr.get_instr().nr_averages(self.RO_acq_averages())
        # Turning of TD source
        if self.td_source.get_instr() is not  None:
            self.td_source.get_instr().off()

        # Updating Spec source
        if self.cw_source.get_instr() is not None:
            self.cw_source.get_instr().power(self.spec_pow())
            self.cw_source.get_instr().frequency(self.f_qubit())
            self.cw_source.get_instr().off()
            if hasattr(self.cw_source.get_instr(), 'pulsemod_state'):
                self.cw_source.get_instr().pulsemod_state('off')
            if hasattr(self.RF_RO_source.get_instr(), 'pulsemod_state'):
                self.RF_RO_source.get_instr().pulsemod_state('Off')
        else:
            logging.warning('No spectrocscopy source (cw_source) specified')

    def prepare_for_pulsed_spec(self):
        # TODO: fix prepare for pulsed spec
        # TODO: make measure pulsed spec
        self.prepare_for_timedomain()
        if self.td_source.get_instr() != None:
            self.td_source.get_instr().off()
        self.cw_source.get_instr().frequency(self.f_qubit())
        self.cw_source.get_instr().power(self.spec_pow_pulsed())
        if hasattr(self.cw_source.get_instr(), 'pulsemod_state'):
            self.cw_source.get_instr().pulsemod_state('On')
        else:
            RuntimeError(
                'Spec source for pulsed spectroscopy does not support pulsing!')
        self.cw_source.get_instr().on()

    def prepare_for_timedomain(self, input_averaging=False):
        # makes sure the settings of the acquisition instrument are reloaded
        self.acquisition_instr(self.acquisition_instr())
        if self.td_source.get_instr() != None:
            self.td_source.get_instr().pulsemod_state('Off')
        self.LO.get_instr().on()
        if self.cw_source.get_instr() != None:
            self.cw_source.get_instr().off()
        if self.td_source.get_instr() != None:
            self.td_source.get_instr().on()
        # Ensures the self.pulse_pars and self.RO_pars get created and updated
        self.get_pulse_pars()

        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        if self.td_source.get_instr() != None:
            self.td_source.get_instr().frequency.set(self.f_qubit.get()
                                                     - self.f_pulse_mod.get())

        # Use resonator freq unless explicitly specified
        if self.f_RO.get() is None:
            f_RO = self.f_res.get()
        else:
            f_RO = self.f_RO.get()
        self.LO.get_instr().frequency.set(f_RO - self.f_RO_mod.get())
        if self.td_source.get_instr() != None:
            self.td_source.get_instr().power.set(self.td_source_pow.get())

        # # makes sure dac range is used optimally, 20% overhead for mixer skew
        # # use 60% of based on linear range in mathematica
        self.AWG.get_instr().set('{}_amp'.format(self.pulse_I_channel()),
                                 self.amp180()*3.0)
        self.AWG.get_instr().set('{}_amp'.format(self.pulse_Q_channel()),
                                 self.amp180()*3.0)

        self.AWG.get_instr().set(self.pulse_I_channel.get()+'_offset',
                                 self.pulse_I_offset.get())
        self.AWG.get_instr().set(self.pulse_Q_channel.get()+'_offset',
                                 self.pulse_Q_offset.get())

        if self.RO_pulse_type() is 'MW_IQmod_pulse_tek':
            self.AWG.get_instr().set(self.RO_I_channel.get()+'_offset',
                                     self.RO_I_offset.get())
            self.AWG.get_instr().set(self.RO_Q_channel.get()+'_offset',
                                     self.RO_Q_offset.get())
        elif self.RO_pulse_type() is 'MW_IQmod_pulse_UHFQC':
            eval('self._acquisition_instr.sigouts_{}_offset({})'.format(
                self.RO_I_channel(), self.RO_I_offset()))
            eval('self._acquisition_instr.sigouts_{}_offset({})'.format(
                self.RO_Q_channel(), self.RO_Q_offset()))
            # This is commented out as doing this by default breaks multiplexed readout
            # it should instead be done using the lutmanman
            # self._acquisition_instr.awg_sequence_acquisition_and_pulse_SSB(
            #     f_RO_mod=self.f_RO_mod(), RO_amp=self.RO_amp(),
            # RO_pulse_length=self.RO_pulse_length(), acquisition_delay=270e-9)
        elif self.RO_pulse_type.get() is 'Gated_MW_RO_pulse':
            self.RF_RO_source.get_instr().pulsemod_state('On')
            self.RF_RO_source.get_instr().frequency(self.f_RO.get())
            self.RF_RO_source.get_instr().power(self.RO_pulse_power.get())
            self.RF_RO_source.get_instr().frequency(self.f_RO())
            self.RF_RO_source.get_instr().on()
            if 'UHFQC' in self.acquisition_instr():
                # self._acquisition_instr.awg_sequence_acquisition()
                # temperarliy removed for debugging
                pass

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
        self.AWG.get_instr().stop()  # Make sure no waveforms are played
        if offs_type == 'pulse':
            AWG_channel1 = self.pulse_I_channel.get()
            AWG_channel2 = self.pulse_Q_channel.get()
            source = self.td_source.get_instr()
        elif offs_type == 'RO':
            AWG_channel1 = self.RO_I_channel.get()
            AWG_channel2 = self.RO_Q_channel.get()
            source = self.LO.get_instr()
        else:
            raise ValueError('offs_type "{}" not recognized'.format(offs_type))

        offset_I, offset_Q = mixer_carrier_cancellation_5014(
            AWG=self.AWG.get_instr(), SH=signal_hound, source=source, MC=self.MC.get_instr(),
            AWG_channel1=AWG_channel1, AWG_channel2=AWG_channel2, xtol=0.0003)

        if update:
            if offs_type == 'pulse':
                self.pulse_I_offset.set(offset_I)
                self.pulse_Q_offset.set(offset_Q)
            if offs_type == 'RO':
                self.RO_I_offset.set(offset_I)
                self.RO_Q_offset.set(offset_Q)

    def calibrate_mixer_offsets_IQ_mod_RO_UHFQC(self, signal_hound,
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
        self.AWG.get_instr().stop()  # Make sure no waveforms are played
        AWG_channel1 = self.RO_I_channel.get()
        AWG_channel2 = self.RO_Q_channel.get()
        source = self.LO.get_instr()
        offset_I, offset_Q = mixer_carrier_cancellation_UHFQC(
            UHFQC=self._acquisition_instr, SH=signal_hound, source=source, MC=self.MC.get_instr(
            ),
            AWG_channel1=AWG_channel1, AWG_channel2=AWG_channel2)

        if update:
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
            signal_hound, self.td_source.get_instr(), station,
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

    def measure_heterodyne_spectroscopy(self, freqs, MC=None,
                                        analyze=True, close_fig=True):
        self.prepare_for_continuous_wave()

        # sqts.Pulsed_spec_seq(spec_pars, RO_pars)
        if MC is None:
            MC = self.MC.get_instr()
        MC.set_sweep_function(pw.wrap_par_to_swf(
                              self.heterodyne_instr.get_instr().frequency, retrieve_value=True))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(
            det.Heterodyne_probe(self.heterodyne_instr.get_instr(),
                                 trigger_separation=self.RO_acq_integration_length()+5e-6, RO_length=self.RO_acq_integration_length()))
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_spectroscopy(self, freqs, pulsed=False, MC=None,
                             analyze=True, close_fig=True,
                             force_load=True, use_max=False, update=True):
        self.prepare_for_continuous_wave()
        self.cw_source.get_instr().on()
        if MC is None:
            MC = self.MC.get_instr()
        if pulsed:
            # Redirect to the pulsed spec function
            return self.measure_pulsed_spectroscopy(freqs=freqs,
                                                    MC=MC,
                                                    analyze=analyze,
                                                    close_fig=close_fig,
                                                    update=update,
                                                    upload=force_load)

        MC.set_sweep_function(pw.wrap_par_to_swf(
                              self.cw_source.get_instr().frequency, retrieve_value=True))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(
            det.Heterodyne_probe(
                self.heterodyne_instr.get_instr(),
                trigger_separation=5e-6 + self.RO_acq_integration_length(),
                RO_length=self.RO_acq_integration_length()))
        MC.run(name='spectroscopy'+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)
        self.cw_source.get_instr().off()

    def measure_pulsed_spectroscopy(self, freqs, MC=None, analyze=True,
                                    return_detector=False,
                                    close_fig=True, upload=True, update=True,
                                    use_max=False):
        """
        Measure pulsed spec with the qubit.

            Accepts a manual sequence parameters, which has to be a call to a
            pulse generation allowing for alternative sequences to be played
            instead of the standard one

        """

        self.prepare_for_pulsed_spec()
        self.heterodyne_instr.get_instr()._disable_auto_seq_loading = True

        self.cw_source.get_instr().pulsemod_state.set('On')
        self.cw_source.get_instr().power.set(self.spec_pow_pulsed.get())
        self.cw_source.get_instr().on()

        if MC is None:
            MC = self.MC.get_instr()

        spec_pars, RO_pars = self.get_spec_pars()
        # Upload the AWG sequence
        sq.Pulsed_spec_seq(spec_pars, RO_pars)

        self.AWG.get_instr().start()
        if return_detector:
            return det.Heterodyne_probe(self.heterodyne_instr.get_instr())

        else:
            MC.set_sweep_function(pw.wrap_par_to_swf(
                self.cw_source.get_instr().frequency, retrieve_value=True))
            MC.set_sweep_points(freqs)
            MC.set_detector_function(
                det.Heterodyne_probe(self.heterodyne_instr.get_instr()))
            MC.run(name='pulsed-spec'+self.msmt_suffix)
            if analyze or update:
                ma_obj = ma.Qubit_Spectroscopy_Analysis(
                    auto=True, label='pulsed', close_fig=close_fig)
                if use_max:
                    f_qubit = ma_obj.peaks['peak']
                else:
                    f_qubit = ma_obj.fitted_freq
                if update:
                    self.f_qubit(f_qubit)
            self.cw_source.get_instr().off()
            return f_qubit

    def measure_rabi(self, amps=np.linspace(-.5, .5, 31), n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False, upload=True):
        # prepare for timedomain takes care of rescaling
        self.prepare_for_timedomain()
        # # Extra rescaling only happens if the amp180 was far too low for the Rabi
        if max(abs(amps))*2 > self.AWG.get_instr().get('{}_amp'.format(self.pulse_I_channel())):
            logging.warning('Auto rescaling AWG amplitude as amp180 {}'.format(
                            self.amp180()) +
                            ' was set very low in comparison to Rabi range')
            self.AWG.get_instr().set('{}_amp'.format(self.pulse_I_channel()),
                                     np.max(abs(amps))*3.0)
            self.AWG.get_instr().set('{}_amp'.format(self.pulse_Q_channel()),
                                     np.max(abs(amps))*3.0)
        if MC is None:
            MC = self.MC.get_instr()

        MC.set_sweep_function(awg_swf.Rabi(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, n=n, upload=upload))
        MC.set_sweep_points(amps)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi-n{}'.format(n)+self.msmt_suffix)
        if analyze:
            ma.Rabi_Analysis(auto=True, close_fig=close_fig)

    def measure_rabi_amp90(self,
                           scales=np.linspace(-0.7, 0.7, 31), n=1,
                           MC=None, analyze=True, close_fig=True,
                           verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        MC.set_sweep_function(awg_swf.Rabi_amp90(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, n=n))
        MC.set_sweep_points(scales)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Rabi_amp90_scales_n{}'.format(n)+self.msmt_suffix)
        if analyze:
            ma.Rabi_Analysis(auto=True, close_fig=close_fig)

    def measure_T1(self, times, MC=None,
                   analyze=True, upload=True, close_fig=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        MC.set_sweep_function(awg_swf.T1(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, upload=upload))
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('T1'+self.msmt_suffix)
        if analyze:
            a = ma.T1_Analysis(auto=True, close_fig=close_fig)
            return a.T1

    def measure_ramsey(self, times, artificial_detuning=0,
                       f_qubit=None, label='',
                       MC=None, analyze=True, close_fig=True, verbose=True,
                       upload=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        if f_qubit is None:
            f_qubit = self.f_qubit.get()
        self.td_source.get_instr().set(
            'frequency', f_qubit - self.f_pulse_mod.get())
        Rams_swf = awg_swf.Ramsey(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            artificial_detuning=artificial_detuning, upload=upload)
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
                     artificial_detuning=None, upload=True,
                     analyze=True, close_fig=True, verbose=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        Echo_swf = awg_swf.Echo(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            artificial_detuning=artificial_detuning, upload=upload)
        MC.set_sweep_function(Echo_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(self.int_avg_det)
        MC.run('Echo'+label+self.msmt_suffix)

        if analyze:
            a = ma.Ramsey_Analysis(
                auto=True, close_fig=close_fig, label='Echo')
            return a

    def measure_allxy(self, double_points=True,
                      MC=None,
                      analyze=True, close_fig=True, verbose=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

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
                                        verbose=False, upload=True):
        '''
        Performs a randomized benchmarking fidelity.
        Optionally specifying T1 also shows the T1 limited fidelity.
        '''
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()
        MC.set_sweep_function(awg_swf.Randomized_Benchmarking(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            double_curves=True,
            nr_cliffords=nr_cliffords, nr_seeds=nr_seeds, upload=upload))
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
                     verbose=True, optimized_weights=False, SSB=False,
                     one_weight_function_UHFQC=False,
                     multiplier=1, nr_shots=4095):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()
        d = cdet.SSRO_Fidelity_Detector_Tek(
            'SSRO'+self.msmt_suffix,
            analyze=analyze,
            raw=no_fits,
            MC=MC,
            AWG=self.AWG.get_instr(), acquisition_instr=self._acquisition_instr,
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, IF=self.f_RO_mod(),
            weight_function_I=self.RO_acq_weight_function_I(),
            weight_function_Q=self.RO_acq_weight_function_Q(),
            nr_shots=nr_shots, one_weight_function_UHFQC=one_weight_function_UHFQC,
            optimized_weights=optimized_weights,
            integration_length=self.RO_acq_integration_length(),
            close_fig=close_fig, SSB=SSB, multiplier=multiplier,
            nr_averages=self.RO_acq_averages())
        if return_detector:
            return d
        d.prepare()
        d.acquire_data_point()
        # if analyze:
        #     return ma.SSRO_Analysis(rotate=soft_rotate, label='SSRO'+self.msmt_suffix,
        #                             no_fits=no_fits, close_fig=close_fig)

    def measure_butterfly(self, return_detector=False,
                          MC=None,
                          analyze=True,
                          close_fig=True,
                          verbose=True,
                          initialize=False,
                          post_msmt_delay=2e-6, case=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()
        MC.set_sweep_function(awg_swf.Butterfly(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
            initialize=initialize, post_msmt_delay=post_msmt_delay))
        MC.set_detector_function(self.int_log_det)
        MC.run('Butterfly{}initialize_{}'.format(self.msmt_suffix, initialize))
        # first perform SSRO analysis to extract the optimal rotation angle
        # theta
        a = ma.SSRO_discrimination_analysis(
            label='Butterfly',
            current_threshold=None,
            close_fig=close_fig,
            plot_2D_histograms=True)

        # the, run it a second time to determin the optimum threshold along the
        # rotated I axis
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
                           nr_samples=512):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        self.MC.get_instr().set_sweep_function(awg_swf.OffOn(pulse_pars=self.pulse_pars,
                                                             RO_pars=self.RO_pars,
                                                             pulse_comb='OffOff',
                                                             nr_samples=nr_samples))
        self.MC.get_instr().set_sweep_points(np.arange(nr_samples))
        self.input_average_detector.nr_samples = nr_samples
        self.MC.get_instr().set_detector_function(self.input_average_detector)
        self.MC.get_instr().run(
            'Measure_transients_{}_0'.format(self.msmt_suffix))
        a0 = ma.MeasurementAnalysis(auto=True, close_fig=close_fig)
        self.MC.get_instr().set_sweep_function(awg_swf.OffOn(pulse_pars=self.pulse_pars,
                                                             RO_pars=self.RO_pars,
                                                             pulse_comb='OnOn',
                                                             nr_samples=nr_samples))
        # self.MC.get_instr().set_sweep_points(np.arange(nr_samples))
        self.input_average_detector.nr_samples = nr_samples
        self.MC.get_instr().set_detector_function(self.input_average_detector)
        self.MC.get_instr().run(
            'Measure_transients_{}_1'.format(self.msmt_suffix))
        a1 = ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_rb_vs_amp(self, amps, nr_cliff=1,
                          resetless=True,
                          MC=None, analyze=True, close_fig=True,
                          verbose=False):
        raise NotImplementedError()

    def measure_motzoi_XY(self, motzois, MC=None, analyze=True, close_fig=True,
                          verbose=True, update=True):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        MC.set_sweep_function(awg_swf.Motzoi_XY(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, motzois=motzois))
        MC.set_detector_function(self.int_avg_det)
        MC.run('Motzoi_XY'+self.msmt_suffix)

        if analyze:
            a = ma.Motzoi_XY_analysis(close_fig=close_fig)
            if update:
                self.motzoi.set(a.optimal_motzoi)
            return a

    def measure_freq_XY(self, f_span, n_f, MC=None, analyze=True, close_fig=True,
                        verbose=True, update=True):

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC.get_instr()

        freqs = np.linspace(-f_span*0.5, f_span*0.5, n_f) + \
            self.f_pulse_mod.get()

        MC.set_sweep_function(awg_swf.Freq_XY(
            pulse_pars=self.pulse_pars, RO_pars=self.RO_pars, freqs=freqs))
        MC.set_detector_function(self.int_avg_det)
        MC.run('Freq_XY'+self.msmt_suffix)

        # if analyze:
        #     a = ma.Motzoi_XY_analysis(close_fig=close_fig)
        #     if update:
        #         self.motzoi.set(a.optimal_motzoi)
        #     return a

    def measure_BusT1(self, times, MC=None):

        if MC is None:
            MC = self.MC.get_instr()

        cal_points = 4
        lengths_cal = times[-1] + \
            np.arange(1, 1+cal_points)*(times[1]-times[0])
        lengths_vec = np.concatenate((times, lengths_cal))

        mw_pulse_pars, RO_pars = self.get_pulse_pars()
        flux_pulse_pars, dist_dict = self.get_flux_pars()
        BusT1 = awg_swf.BusT1(times,
                              mw_pulse_pars,
                              RO_pars,
                              flux_pulse_pars,
                              dist_dict=dist_dict,
                              AWG=self.AWG.get_instr(),
                              upload=False, return_seq=True)

        exec('self.AWG.get_instr().ch%d_amp(2.)' % self.fluxing_channel())
        seq = BusT1.pre_upload()

        MC.set_sweep_function(BusT1)
        MC.set_sweep_points(lengths_vec)

        MC.set_detector_function(self.int_avg_det)
        self.AWG.get_instr().ch4_amp(flux_pulse_pars['swap_amp'])
        MC.run('Bus_T1')
        ma.T1_Analysis(auto=True, label='Bus_T1')

    def measure_BusT2(self, times, MC=None):

        if MC is None:
            MC = self.MC.get_instr()

        cal_points = 4
        lengths_cal = times[-1] + \
            np.arange(1, 1+cal_points)*(times[1]-times[0])
        lengths_vec = np.concatenate((times, lengths_cal))

        mw_pulse_pars, RO_pars = self.get_pulse_pars()
        flux_pulse_pars, dist_dict = self.get_flux_pars()
        BusT2 = awg_swf.BusT2(times_vec=times,
                              mw_pulse_pars=mw_pulse_pars,
                              RO_pars=RO_pars,
                              # artificial_detuning=artificial_detuning,
                              flux_pulse_pars=flux_pulse_pars,
                              dist_dict=dist_dict,
                              AWG=self.AWG.get_instr(),
                              upload=False, return_seq=True)

        exec('self.AWG.get_instr().ch%d_amp(2.)' % self.fluxing_channel())
        seq = BusT2.pre_upload()

        MC.set_sweep_function(BusT2)
        MC.set_sweep_points(lengths_vec)

        MC.set_detector_function(self.int_avg_det)
        self.AWG.get_instr().ch4_amp(flux_pulse_pars['swap_amp'])
        MC.run('Bus_Echo')
        ma.Ramsey_Analysis(auto=True, label='Bus_T2')

    def measure_BusEcho(self, times, artificial_detuning, MC=None):

        if MC is None:
            MC = self.MC.get_instr()

        cal_points = 4
        lengths_cal = times[-1] + \
            np.arange(1, 1+cal_points)*(times[1]-times[0])
        lengths_vec = np.concatenate((times, lengths_cal))

        mw_pulse_pars, RO_pars = self.get_pulse_pars()
        flux_pulse_pars, dist_dict = self.get_flux_pars()
        BusEcho = awg_swf.BusEcho(times_vec=times,
                                  mw_pulse_pars=mw_pulse_pars,
                                  RO_pars=RO_pars,
                                  artificial_detuning=artificial_detuning,
                                  flux_pulse_pars=flux_pulse_pars,
                                  dist_dict=dist_dict,
                                  AWG=self.AWG.get_instr(),
                                  upload=False, return_seq=True)

        exec('self.AWG.get_instr().ch%d_amp(2.)' % self.fluxing_channel())
        seq = BusEcho.pre_upload()

        MC.set_sweep_function(BusEcho)
        MC.set_sweep_points(lengths_vec)

        MC.set_detector_function(self.int_avg_det)
        self.AWG.get_instr().ch4_amp(flux_pulse_pars['swap_amp'])
        MC.run('Bus_Echo')
        ma.Ramsey_Analysis(auto=True, label='Bus_Echo')

    def _do_get_acquisition_instr(self):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily
        return self._acquisition_instr.name

    def _do_set_acquisition_instr(self, acquisition_instr):
        # Specifying the int_avg det here should allow replacing it with ATS
        # or potential digitizer acquisition easily

        self._acquisition_instr = self.find_instrument(acquisition_instr)
        if 'CBox' in acquisition_instr:
            logging.info("setting CBox acquisition")
            self.int_avg_det = det.CBox_integrated_average_detector(self._acquisition_instr,
                                                                    self.AWG.get_instr(),
                                                                    nr_averages=self.RO_acq_averages(),
                                                                    integration_length=self.RO_acq_integration_length(
                                                                    ),
                                                                    normalize=True)
            self.int_avg_det_rot = det.CBox_integrated_average_detector(self._acquisition_instr,
                                                                        self.AWG.get_instr(),
                                                                        nr_averages=self.RO_acq_averages(),
                                                                        integration_length=self.RO_acq_integration_length(
                                                                        ),
                                                                        normalize=True)
            self.int_log_det = det.CBox_integration_logging_det(self._acquisition_instr,
                                                                self.AWG.get_instr(), integration_length=self.RO_acq_integration_length())

            self.input_average_detector = det.CBox_input_average_detector(
                self._acquisition_instr,
                self.AWG.get_instr(), nr_averages=self.RO_acq_averages())

        elif 'UHFQC' in acquisition_instr:
            logging.info("setting UHFQC acquisition")
            self.input_average_detector = det.UHFQC_input_average_detector(
                UHFQC=self._acquisition_instr,
                AWG=self.AWG.get_instr(), nr_averages=self.RO_acq_averages())

            self.int_avg_det = det.UHFQC_integrated_average_detector(
                UHFQC=self._acquisition_instr, AWG=self.AWG.get_instr(),
                channels=[self.RO_acq_weight_function_I(),
                          self.RO_acq_weight_function_Q()],
                nr_averages=self.RO_acq_averages(),
                integration_length=self.RO_acq_integration_length())

            self.int_avg_det_rot = det.UHFQC_integrated_average_detector(
                UHFQC=self._acquisition_instr, AWG=self.AWG.get_instr(),
                channels=[self.RO_acq_weight_function_I(),
                          self.RO_acq_weight_function_Q()],
                nr_averages=self.RO_acq_averages(),
                integration_length=self.RO_acq_integration_length(), rotate=True)

            self.int_log_det = det.UHFQC_integration_logging_det(
                UHFQC=self._acquisition_instr, AWG=self.AWG.get_instr(),
                channels=[
                    self.RO_acq_weight_function_I(), self.RO_acq_weight_function_Q()],
                integration_length=self.RO_acq_integration_length())
        elif 'DDM' in acquisition_instr:
            logging.info("setting DDM acquisition")
            self.input_average_detector = det.DDM_input_average_detector(
                DDM=self._acquisition_instr,
                AWG=self.AWG, nr_averages=self.RO_acq_averages())

            self.int_avg_det = det.DDM_integrated_average_detector(
                    DDM=self._acquisition_instr, AWG=self.AWG,
                    channels=[self.RO_acq_weight_function_I(),
                              self.RO_acq_weight_function_Q()],
                    nr_averages=self.RO_acq_averages(),
                    integration_length=self.RO_acq_integration_length())

            self.int_log_det = det.DDM_integration_logging_det(
                DDM=self._acquisition_instr, AWG=self.AWG,
                channels=[
                    self.RO_acq_weight_function_I(), self.RO_acq_weight_function_Q()],
                integration_length=self.RO_acq_integration_length())

        elif 'ATS' in acquisition_instr:
            logging.info("setting ATS acquisition")
            # self.int_avg_det = det.ATS_integrated_average_continuous_detector(
            #     ATS=self._acquisition_instr.card,
            #     ATS_acq=self._acquisition_instr.controller, AWG=self.AWG.get_instr(),
            #     nr_averages=self.RO_acq_averages())

    def get_pulse_dict(self, pulse_dict={}):
        '''
        Returns a dictionary containing the pulse parameters of the qubit.
        This function is intended to replace the old get_pulse_pars.
        Dictionary contains the keys formatted as follows:
            operation self.name

        Input args:
            pulse_dict (dict):  Optionally specify an existing pulse dict to update

        (currently only contains single qubit pulses)
        '''
        drive_pars, RO_pars = self.get_pulse_pars()
        pulse_dict.update(add_suffix_to_dict_keys(
            sq.get_pulse_dict_from_pars(drive_pars), ' ' + self.name))
        pulse_dict.update({'RO {}'.format(self.name): RO_pars})

        spec_pars, RO_pars = self.get_spec_pars()
        pulse_dict.update({'Spec {}'.format(self.name): spec_pars})

        return pulse_dict

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
            'operation_type': 'MW',
            'target_qubit': self.name,
            'pulse_type': 'SSB_DRAG_pulse'}

        self.RO_pars = {
            'I_channel': self.RO_I_channel.get(),
            'Q_channel': self.RO_Q_channel.get(),
            'RO_pulse_marker_channel': self.RO_pulse_marker_channel.get(),
            'amplitude': self.RO_amp.get(),
            'length': self.RO_pulse_length.get(),
            'pulse_delay': self.RO_pulse_delay.get(),
            'mod_frequency': self.f_RO_mod.get(),
            'acq_marker_delay': self.RO_acq_marker_delay.get(),
            'acq_marker_channel': self.RO_acq_marker_channel.get(),
            'phase': 0,
            'operation_type': 'RO',
            'target_qubit': self.name,
            'pulse_type': self.RO_pulse_type.get()}
        return self.pulse_pars, self.RO_pars

    def get_spec_pars(self):
        # logging.warning('deprecated use get_operation_dict')
        pulse_pars, RO_pars = self.get_pulse_pars()
        spec_pars = {'pulse_type': 'SquarePulse',
                     'length': self.spec_pulse_length.get(),
                     'amplitude': 1,
                     'operation_type': 'MW',
                     'target_qubit': self.name,
                     'channel': self.spec_pulse_marker_channel.get()}

        RO_pars['pulse_delay'] += spec_pars['length']
        spec_pars['pulse_delay'] = (RO_pars['length'] +
                                    self.spec_pulse_depletion_time.get())
        return spec_pars, RO_pars

    def get_operation_dict(self, operation_dict={}):
        operation_dict = super().get_operation_dict(operation_dict)
        operation_dict['SpecPulse '+self.name] = self.get_spec_pars()[0]
        self.get_pulse_dict(operation_dict)
        return operation_dict
