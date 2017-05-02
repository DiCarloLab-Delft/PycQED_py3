import logging
import numpy as np
from scipy.optimize import brent

from .qubit_object import Transmon
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.measurement import mc_parameter_wrapper as pw

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import CBox_sweep_functions as cb_swf
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
import pycqed.measurement.randomized_benchmarking.randomized_benchmarking as rb
from pycqed.measurement.calibration_toolbox import mixer_carrier_cancellation_CBox
from pycqed.measurement.calibration_toolbox import mixer_skewness_cal_CBox_adaptive

from pycqed.measurement.optimization import nelder_mead


class CBox_driven_transmon(Transmon):
    '''
    Setup configuration:
        Drive:                 CBox AWGs
        Acquisition:           CBox
        Readout pulse configuration: LO modulated using AWG
    '''
    shared_kwargs = ['LO', 'cw_source', 'td_source', 'IVVI', 'AWG', 'LutMan',
                     'CBox',
                     'heterodyne_instr',  'MC']

    def __init__(self, name,
                 LO, cw_source, td_source,
                 IVVI, AWG, LutMan,
                 CBox, heterodyne_instr,
                 MC, **kw):
        super().__init__(name, **kw)
        '''
        Adds the parameters to the qubit insrument, it provides initial values
        for some parameters but not for all. Powers have to be set by hand as
        a safety measure.
        '''
        # MW-sources
        self.LO = LO
        self.cw_source = cw_source
        self.td_source = td_source
        self.IVVI = IVVI
        self.LutMan = LutMan
        self.heterodyne_instr = heterodyne_instr
        self.AWG = AWG
        self.CBox = CBox
        self.MC = MC
        self.add_parameter('mod_amp_cw', label='RO modulation ampl cw',
                           unit='V', initial_value=0.5,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_power_cw', label='RO power cw',
                           unit='dBm',
                           parameter_class=ManualParameter)

        self.add_parameter('mod_amp_td', label='RO modulation ampl td',
                           unit='V', initial_value=0.5,
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
        self.add_parameter('f_RO_mod',
                           label='Readout-modulation frequency', unit='Hz',
                           initial_value=-2e7,
                           parameter_class=ManualParameter)
        # Time-domain parameters
        self.add_parameter('f_pulse_mod',
                           initial_value=-50e6,
                           label='pulse-modulation frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('awg_nr', label='CBox awg nr', unit='#',
                           parameter_class=ManualParameter)

        self.add_parameter('amp180',
                           label='Pi-pulse amplitude', unit='mV',
                           initial_value=300,
                           parameter_class=ManualParameter)
        # Amp 90 is hardcoded to be half amp180
        self.add_parameter('amp90',
                           label='Pi/2-pulse amplitude', unit='mV',
                           get_cmd=self._get_amp90)
        self.add_parameter('gauss_width', unit='s',
                           initial_value=40e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('motzoi', label='Motzoi parameter', unit='',
                           initial_value=0,
                           parameter_class=ManualParameter)

        # Single shot readout specific parameters
        self.add_parameter('RO_threshold', unit='dac-value',
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('signal_line', parameter_class=ManualParameter,
                           vals=vals.Enum(0, 1), initial_value=0)

        # Mixer skewness correction
        self.add_parameter('phi', unit='deg',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('alpha', unit='',
                           parameter_class=ManualParameter, initial_value=1)
        # Mixer offsets correction, qubit drive
        self.add_parameter('mixer_offs_drive_I',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_drive_Q',
                           parameter_class=ManualParameter, initial_value=0)


    def prepare_for_continuous_wave(self):

        self.heterodyne_instr._disable_auto_seq_loading = False
        self.LO.on()
        self.td_source.off()
        if hasattr(self.heterodyne_instr, 'mod_amp'):
            self.heterodyne_instr.set('mod_amp', self.mod_amp_cw.get())
        else:
            self.heterodyne_instr.RF_power(self.RO_power_cw())
        # TODO: Update IF to f_RO_mod in heterodyne instr
        self.heterodyne_instr.set('f_RO_mod', self.f_RO_mod.get())
        self.heterodyne_instr.frequency.set(self.f_res.get())

        if hasattr(self.cw_source, 'pulsemod_state'):
            self.cw_source.pulsemod_state('off')
        self.cw_source.power.set(self.spec_pow.get())

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
        self.AWG.set('ch3_amp', self.mod_amp_td.get())
        self.AWG.set('ch4_amp', self.mod_amp_td.get())
        self.CBox.set('AWG{:.0g}_mode'.format(self.awg_nr.get()),
                      'segmented tape')
        # Mixer offsets correction
        self.CBox.set('AWG{:.0g}_dac0_offset'.format(self.awg_nr.get()),
                      self.mixer_offs_drive_I.get())
        self.CBox.set('AWG{:.0g}_dac1_offset'.format(self.awg_nr.get()),
                      self.mixer_offs_drive_Q.get())

        self.LutMan.amp180.set(self.amp180.get())
        self.LutMan.amp90.set(self.amp90.get())
        self.LutMan.gauss_width.set(self.gauss_width.get()*1e9)  # s to ns
        self.LutMan.motzoi_parameter.set(self.motzoi.get())
        self.LutMan.f_modulation.set(self.f_pulse_mod.get()*1e-9)

        # Mixer skewness correction
        self.LutMan.IQ_phase_skewness.set(0)
        print('self.LutMan type: ', type(self.LutMan))
        self.LutMan.QI_amp_ratio.set(1)
        self.LutMan.apply_predistortion_matrix.set(True)
        self.LutMan.alpha.set(self.alpha.get())
        self.LutMan.phi.set(self.phi.get())

        self.LutMan.load_pulses_onto_AWG_lookuptable(self.awg_nr.get())

        self.CBox.set('sig{}_threshold_line'.format(
                      int(self.signal_line.get())),
                      int(self.RO_threshold.get()))


    def get_resetless_rb_detector(self, nr_cliff, starting_seed=1,
                                  nr_seeds='max', pulse_p_elt='min',
                                  MC=None,
                                  upload=True):
        if MC is None:
            MC = self.MC

        if pulse_p_elt == 'min':
            safety_factor = 5 if nr_cliff < 8 else 3
            pulse_p_elt = int(safety_factor*nr_cliff)
        if nr_seeds == 'max':
            nr_seeds = 29184//pulse_p_elt

        if nr_seeds*pulse_p_elt > 29184:
            raise ValueError(
                'Too many pulses ({}), {} seeds, {} pulse_p_elt'.format(
                    nr_seeds*pulse_p_elt, nr_seeds, pulse_p_elt))

        resetless_interval = (
            np.round(pulse_p_elt*self.pulse_delay.get()*1e6)+2.5)*1e-6

        combined_tape = []
        for i in range(nr_seeds):
            if starting_seed is not None:
                seed = starting_seed*1000*i
            else:
                seed = None
            rb_seq = rb.randomized_benchmarking_sequence(nr_cliff,
                                                         desired_net_cl=3,
                                                         seed=seed)
            tape = rb.convert_clifford_sequence_to_tape(
                rb_seq, self.LutMan.lut_mapping.get())
            if len(tape) > pulse_p_elt:
                raise ValueError(
                    'Too many pulses ({}), {} pulse_p_elt'.format(
                        len(tape),  pulse_p_elt))
            combined_tape += [0]*(pulse_p_elt-len(tape))+tape

        # Rename IF in awg_swf_resetless tape
        s = awg_swf.Resetless_tape(
            n_pulses=pulse_p_elt, tape=combined_tape,
            IF=self.f_RO_mod.get(),
            pulse_delay=self.pulse_delay.get(),
            resetless_interval=resetless_interval,
            RO_pulse_delay=self.RO_pulse_delay.get(),
            RO_pulse_length=self.RO_pulse_length.get(),
            RO_trigger_delay=self.RO_acq_marker_delay.get(),
            AWG=self.AWG, CBox=self.CBox, upload=upload)

        d = cdet.CBox_trace_error_fraction_detector(
            'Resetless rb det',
            MC=MC, AWG=self.AWG, CBox=self.CBox,
            sequence_swf=s,
            threshold=self.RO_threshold.get(),
            save_raw_trace=False)
        return d

    def calibrate_pulse_parameters(self, method='resetless_rb', nr_cliff=10,
                                   parameters=['amp', 'motzoi', 'frequency'],
                                   amp_guess=None, motzoi_guess=None,
                                   frequency_guess=None,
                                   a_step=30, m_step=.1, f_step=20e3,
                                   MC=None, nested_MC=None,
                                   update=False, close_fig=True,
                                   verbose=True):
        '''
        Calibrates single qubit pulse parameters currently only using
        the resetless rb method (requires reasonable (80%+?) discrimination
        fidelity)

        If it there is only one parameter to sweep it will use brent's method
        instead.

        The function returns the values it found for the optimization.
        '''
        if method is not 'resetless_rb':
            raise NotImplementedError()

        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        if nested_MC is None:
            nested_MC = self.nested_MC

        d = self.get_resetless_rb_detector(nr_cliff=nr_cliff, MC=nested_MC)

        name = 'RB_{}cl_numerical'.format(nr_cliff)
        MC.set_detector_function(d)

        if amp_guess is None:
            amp_guess = self.amp180.get()
        if motzoi_guess is None:
            motzoi_guess = self.motzoi.get()
        if frequency_guess is None:
            frequency_guess = self.f_qubit.get()
        # Because we are sweeping the source and not the qubit frequency
        start_freq = frequency_guess - self.f_pulse_mod.get()

        sweep_functions = []
        x0 = []
        init_steps = []
        if 'amp' in parameters:
            sweep_functions.append(cb_swf.LutMan_amp180_90(self.LutMan))
            x0.append(amp_guess)
            init_steps.append(a_step)
        if 'motzoi' in parameters:
            sweep_functions.append(
                pw.wrap_par_to_swf(self.LutMan.motzoi_parameter))
            x0.append(motzoi_guess)
            init_steps.append(m_step)
        if 'frequency' in parameters:
            sweep_functions.append(
                pw.wrap_par_to_swf(self.td_source.frequency))
            x0.append(start_freq)
            init_steps.append(f_step)
        if len(sweep_functions) == 0:
            raise ValueError(
                'parameters "{}" not recognized'.format(parameters))

        MC.set_sweep_functions(sweep_functions)

        if len(sweep_functions) != 1:
            # noise ensures no_improv_break sets the termination condition
            ad_func_pars = {'adaptive_function': nelder_mead,
                            'x0': x0,
                            'initial_step': init_steps,
                            'no_improv_break': 10,
                            'minimize': False,
                            'maxiter': 500}
        elif len(sweep_functions) == 1:
            # Powell does not work for 1D, use brent instead
            brack = (x0[0]-5*init_steps[0], x0[0])
            # Ensures relative change in parameter is relevant
            if parameters == ['frequency']:
                tol = 1e-9
            else:
                tol = 1e-3
            print('Tolerance:', tol, init_steps[0])
            print(brack)
            ad_func_pars = {'adaptive_function': brent,
                            'brack': brack,
                            'tol': tol,  # Relative tolerance in brent
                            'minimize': False}
        MC.set_adaptive_function_parameters(ad_func_pars)
        MC.run(name=name, mode='adaptive')
        if len(sweep_functions) != 1:
            a = ma.OptimizationAnalysis(auto=True, label=name,
                                        close_fig=close_fig)
            if verbose:
                # Note printing can be made prettier
                print('Optimization converged to:')
                print('parameters: {}'.format(parameters))
                print(a.optimization_result[0])
            if update:
                for i, par in enumerate(parameters):
                    if par == 'amp':
                        self.amp180.set(a.optimization_result[0][i])
                    elif par == 'motzoi':
                        self.motzoi.set(a.optimization_result[0][i])
                    elif par == 'frequency':
                        self.f_qubit.set(a.optimization_result[0][i] +
                                         self.f_pulse_mod.get())
            return a
        else:
            a = ma.MeasurementAnalysis(label=name, close_fig=close_fig)
            print('Optimization for {} converged to: {}'.format(
                parameters[0], a.sweep_points[-1]))
            if update:
                if parameters == ['amp']:
                    self.amp180.set(a.sweep_points[-1])
                elif parameters == ['motzoi']:
                    self.motzoi.set(a.sweep_points[-1])
                elif parameters == ['frequency']:
                    self.f_qubit.set(a.sweep_points[-1]+self.f_pulse_mod.get())
            return a.sweep_points[-1]

    def calibrate_mixer_offsets(self, signal_hound, update=True):
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        signal hound needs to be given as it this is not part of the qubit
        object in order to reduce dependencies.
        '''
        # ensures freq is set correctly
        self.prepare_for_timedomain()
        self.AWG.stop()  # Make sure no waveforms are played
        offset_I, offset_Q = mixer_carrier_cancellation_CBox(
            CBox=self.CBox, SH=signal_hound, source=self.td_source,
            MC=self.MC, awg_nr=self.awg_nr.get())
        if update:
            self.mixer_offs_drive_I.set(offset_I)
            self.mixer_offs_drive_Q.set(offset_Q)

    def calibrate_mixer_skewness(self, signal_hound, update=True):
        '''
        Calibrates the mixer skewness using mixer_skewness_cal_CBox_adaptive
        see calibration toolbox for details
        '''
        self.prepare_for_timedomain()
        phi, alpha = mixer_skewness_cal_CBox_adaptive(
            CBox=self.CBox, SH=signal_hound, source=self.td_source,
            LutMan=self.LutMan, AWG=self.AWG, MC=self.MC,
            awg_nrs=[self.awg_nr.get()], calibrate_both_sidebands=True)
        if update:
            self.phi.set(phi)
            self.alpha.set(alpha)

    def calibrate_RO_threshold(self, method='conventional',
                               MC=None, close_fig=True,
                               verbose=False, make_fig=True):
        '''
        Calibrates the RO threshold and applies the correct rotation to the
        data either using a conventional SSRO experiment or by using the
        self-consistent method.

        For details see measure_ssro() and measure_discrimination_fid()

        method: 'conventional' or 'self-consistent

        '''
        self.prepare_for_timedomain()

        if method.lower() == 'conventional':
            self.CBox.lin_trans_coeffs.set([1, 0, 0, 1])
            self.measure_ssro(MC=MC, analyze=False, close_fig=close_fig,
                              verbose=verbose)
            a = ma.SSRO_Analysis(auto=True, close_fig=True,
                                 label='SSRO', no_fits=True,
                                 close_file=True)
            # SSRO analysis returns the angle to rotate by
            theta = a.theta  # analysis returns theta in rad

            rot_mat = [np.cos(theta), -np.sin(theta),
                       np.sin(theta), np.cos(theta)]
            self.CBox.lin_trans_coeffs.set(rot_mat)
            self.threshold = a.V_opt_raw  # allows
            self.RO_threshold.set(int(a.V_opt_raw))

        elif method.lower() == 'self-consistent':
            self.CBox.lin_trans_coeffs.set([1, 0, 0, 1])
            discr_vals = self.measure_discrimination_fid(
                MC=MC, close_fig=close_fig, make_fig=make_fig, verbose=verbose)

            # hardcoded indices correspond to values in CBox SSRO discr det
            theta = discr_vals[2] * 2 * np.pi/360

            # Discr returns the current angle, rotation is - that angle
            rot_mat = [np.cos(-1*theta), -np.sin(-1*theta),
                       np.sin(-1*theta), np.cos(-1*theta)]
            self.CBox.lin_trans_coeffs.set(rot_mat)

            # Measure it again to determine the threshold after rotating
            discr_vals = self.measure_discrimination_fid(
                MC=MC, close_fig=close_fig, make_fig=make_fig, verbose=verbose)

            # hardcoded indices correspond to values in CBox SSRO discr det
            theta = discr_vals[2]
            self.threshold = int(discr_vals[3])

            self.RO_threshold.set(int(self.threshold))
        else:
            raise ValueError('method %s not recognized, can be' % method +
                             ' either "conventional" or "self-consistent"')

    def measure_heterodyne_spectroscopy(self, freqs, MC=None,
                                        analyze=True, close_fig=True, RO_length=2000e-9):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        MC.set_sweep_function(pw.wrap_par_to_swf(
                              self.heterodyne_instr.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr, trigger_separation=2.8e-6, RO_length=2274e-9))
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_spectroscopy(self, freqs, pulsed=False, MC=None,
                             analyze=True, close_fig=True, mode='ROGated_SpecGate',
                             force_load=False):
        self.prepare_for_continuous_wave()
        self.cw_source.on()
        if MC is None:
            MC = self.MC
        if pulsed:
            # Redirect to the pulsed spec function
            return self.measure_pulsed_spectroscopy(freqs=freqs,
                                                    MC=MC,
                                                    analyze=analyze,
                                                    close_fig=close_fig,
                                                    mode=mode, force_load=force_load)

        MC.set_sweep_function(pw.wrap_par_to_swf(
                              self.cw_source.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(
            det.Heterodyne_probe(self.heterodyne_instr, trigger_separation=2.8e-6))
        MC.run(name='spectroscopy'+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)
        self.cw_source.off()

    def measure_pulsed_spectroscopy(self, freqs, mode='ROGated_SpecGate', MC=None,
                                    analyze=True, close_fig=True, force_load=False):
        # This is a trick so I can reuse the heterodyne instr
        # to do pulsed-spectroscopy
        self.heterodyne_instr._disable_auto_seq_loading = True

        if mode=='ROMod_SpecGated':
            if ('Pulsed_spec_with_RF_mod' not in self.AWG.setup_filename.get()) or force_load:
                st_seqs.Pulsed_spec_seq_RF_mod(
                    IF=self.f_RO_mod.get(),
                    spec_pulse_length=spec_pulse_length, marker_interval=30e-6,
                    RO_pulse_delay=self.RO_pulse_delay.get())
        elif mode=='ROGated_SpecGate':
            if ('Pulsed_spec_with_RF_gated' not in self.AWG.setup_filename.get()) or force_load:
                st_seqs.Pulsed_spec_seq_RF_gated(self.RO_pars,
                                                 self.pulse_pars)
        else:
            NotImplementedError('Pulsed Spec mode not supported. Only ROMod_SpecGated and ROGated_SpecGate are avaible right now.\n')

        self.cw_source.pulsemod_state.set('on')
        self.cw_source.power.set(self.spec_pow_pulsed.get())

        self.AWG.start()
        if hasattr(self.heterodyne_instr, 'mod_amp'):
            self.heterodyne_instr.set('mod_amp', self.mod_amp_cw.get())
        else:
            self.heterodyne_instr.RF.power(self.RO_power_cw())
        MC.set_sweep_function(pw.wrap_par_to_swf(
                              self.cw_source.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        MC.run(name='pulsed-spec'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_resonator_power(self, freqs, powers,
                                MC=None, analyze=True, close_fig=True):
        '''
        N.B. This one does not use powers but varies the mod-amp.
        Need to find a way to keep this function agnostic to that
        '''
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        MC.set_sweep_functions(
            [pw.wrap_par_to_swf(self.heterodyne_instr.frequency),
             pw.wrap_par_to_swf(self.heterodyne_instr.RF_power)])
        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(powers)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        MC.run(name='Resonator_power_scan'+self.msmt_suffix, mode='2D')
        if analyze:
            ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_resonator_dac(self, freqs, dac_voltages,
                              MC=None, analyze=True, close_fig=True):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        MC.set_sweep_functions(
            [self.heterodyne_instr.frequency,
             self.IVVI.parameters['dac{}'.format(self.dac_channel())]])
        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(dac_voltages)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        MC.run(name='Resonator_dac_scan'+self.msmt_suffix, mode='2D')
        if analyze:
            ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_rabi(self, amps, n=1,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        cal_points = [0, 0]
        amps = cal_points + list(amps)
        self.CBox.AWG0_mode('Codeword-trigger mode')
        self.CBox.AWG1_mode('Codeword-trigger mode')
        self.CBox.AWG2_mode('Codeword-trigger mode')
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.load_instructions('CBox_v3_test_program\Rabi.asm')
        self.CBox.set_master_controller_working_state(1, 0, 0)
        MC.set_sweep_function(pw.wrap_par_to_swf(self.LutMan.amp180))
        MC.set_sweep_points(amps)
        MC.set_detector_function(det.CBox_v3_single_int_avg_with_LutReload(
                                 self.CBox, self.LutMan,
                                 awg_nrs=[self.awg_nr.get()]))
        MC.run('Rabi-n{}'.format(n)+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_T1(self, times, MC=None,
                   analyze=True, close_fig=True):
        '''
        if update is True will update self.T1 with the measured value
        '''
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        # append the calibration points, times are for location in plot
        times = np.concatenate([times,
                               (times[-1]+times[0],
                                times[-1]+times[1],
                                times[-1]+times[2],
                                times[-1]+times[3])])
        MC.set_sweep_function(
            awg_swf.CBox_v3_T1(CBox=self.CBox, upload=True))
        MC.set_sweep_points(times)
        MC.set_detector_function(det.CBox_v3_integrated_average_detector(
                                 self.CBox))
        MC.run('T1'+self.msmt_suffix)
        if analyze:
            a = ma.T1_Analysis(auto=True, close_fig=True)
            return a.T1

    def measure_ramsey(self, times, artificial_detuning=0, f_qubit=None,
                       label='',
                       MC=None, analyze=True, close_fig=True, verbose=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC

        # This is required because I cannot change the phase in the pulses
        if not all([np.round(t*1e9) % (1/self.f_pulse_mod.get()*1e9)
                   == 0 for t in times]):
            raise ValueError('timesteps must be multiples of modulation freq')

        if f_qubit is None:
            f_qubit = self.f_qubit.get()
        # this should have no effect if artificial detuning = 0
        self.td_source.set('frequency', f_qubit - self.f_pulse_mod.get() +
                           artificial_detuning)
        Rams_swf = awg_swf.CBox_Ramsey(
            AWG=self.AWG, CBox=self.CBox, IF=self.f_RO_mod.get(), pulse_delay=0,
            RO_pulse_delay=self.RO_pulse_delay.get(),
            RO_trigger_delay=self.RO_acq_marker_delay.get(),
            RO_pulse_length=self.RO_pulse_length.get())
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(det.CBox_integrated_average_detector(
                                 self.CBox, self.AWG))
        MC.run('Ramsey'+label+self.msmt_suffix)

        if analyze:
            a = ma.Ramsey_Analysis(auto=True, close_fig=True)

            if verbose:
                fitted_freq = a.fit_res.params['frequency'].value
                print('Artificial detuning: {:.2e}'.format(
                      artificial_detuning))
                print('Fitted detuning: {:.2e}'.format(fitted_freq))
                print('Actual detuning:{:.2e}'.format(
                      fitted_freq-artificial_detuning))

    def measure_allxy(self, MC=None,
                      analyze=True, close_fig=True, verbose=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        d = cdet.AllXY_devition_detector_CBox(
            'AllXY'+self.msmt_suffix, MC=MC,
            AWG=self.AWG, CBox=self.CBox, IF=self.f_RO_mod.get(),
            pulse_delay=self.pulse_delay.get(),
            RO_pulse_delay=self.RO_pulse_delay.get(),
            RO_trigger_delay=self.RO_acq_marker_delay.get(),
            RO_pulse_length=self.RO_pulse_length.get())
        d.prepare()
        d.acquire_data_point()
        if analyze:
            a = ma.AllXY_Analysis(close_main_fig=close_fig)
            return a

    def measure_ssro(self, no_fits=False,
                     return_detector=False,
                     MC=None,
                     analyze=True, close_fig=True, verbose=True):
        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC
        d = cdet.SSRO_Fidelity_Detector_CBox(
            'SSRO'+self.msmt_suffix,
            analyze=return_detector,
            raw=no_fits,
            MC=MC,
            AWG=self.AWG, CBox=self.CBox, IF=self.f_RO_mod.get(),
            pulse_delay=self.pulse_delay.get(),
            RO_pulse_delay=self.RO_pulse_delay.get(),
            RO_trigger_delay=self.RO_acq_marker_delay.get(),
            RO_pulse_length=self.RO_pulse_length.get())

        if return_detector:
            return d
        d.prepare()
        d.acquire_data_point()
        if analyze:
            ma.SSRO_Analysis(label='SSRO'+self.msmt_suffix,
                             no_fits=no_fits, close_fig=close_fig)

    def measure_discrimination_fid(self, no_fits=False,
                                   return_detector=False,
                                   MC=None,
                                   analyze=True,
                                   close_fig=True, make_fig=True,
                                   verbose=True):
        '''
        Measures the single shot discrimination fidelity.
        Uses whatever sequence is currently loaded and takes 8000 single shots
        Constructs histograms based on those and uses it to extract the
        single-shot discrimination fidelity.
        '''
        self.prepare_for_timedomain()

        if MC is None:
            MC = self.MC

        # If I return the detector to use it must do analysis internally
        # Otherwise I do it here in the qubit object so that I can pass args
        analysis_in_det = return_detector
        d = cdet.CBox_SSRO_discrimination_detector(
            'SSRO-disc'+self.msmt_suffix,
            analyze=analysis_in_det,
            MC=MC, AWG=self.AWG, CBox=self.CBox,
            sequence_swf=swf.None_Sweep(sweep_control='hard',
                                        sweep_points=np.arange(10)))
        if return_detector:
            return d
        d.prepare()
        discr_vals = d.acquire_data_point()
        if analyze:
            current_threshold = self.CBox.sig0_threshold_line.get()
            a = ma.SSRO_discrimination_analysis(
                label='SSRO-disc'+self.msmt_suffix,
                current_threshold=current_threshold,
                close_fig=close_fig,
                plot_2D_histograms=make_fig)

            return (a.F_discr_curr_t*100, a.F_discr*100,
                    a.theta, a.opt_I_threshold,
                    a.relative_separation, a.relative_separation_I)
        return discr_vals

    def measure_rb_vs_amp(self, amps, nr_cliff=1,
                          resetless=True,
                          MC=None, analyze=True, close_fig=True,
                          verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        if resetless:
            d = self.get_resetless_rb_detector(nr_cliff=nr_cliff)
        else:
            raise NotImplementedError()
        MC.set_detector_function(d)
        MC.set_sweep_functions([cb_swf.LutMan_amp180_90(self.LutMan)])
        MC.set_sweep_points(amps)
        MC.run('RB-vs-amp_{}cliff'.format(nr_cliff) + self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(close_fig=close_fig)

    def _get_amp90(self):
        return self.amp180.get()/2
