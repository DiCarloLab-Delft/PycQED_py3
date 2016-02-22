import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals

from modules.analysis.analysis_toolbox import calculate_transmon_transitions
from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.measurement import mc_parameter_wrapper as pw
from modules.measurement import awg_sweep_functions as awg_swf
from modules.analysis import measurement_analysis as ma
from modules.measurement.pulse_sequences import standard_sequences as st_seqs


class Qubit(Instrument):
    '''
    Abstract base class for the qubit object.
    Contains a template for all the functions a qubit should have.
    N.B. This is not intended to be initialized.

    Specific types of qubits should inherit from this class, different setups
    can inherit from those to further specify the functionality.

    Possible inheritance tree
    - Qubit (general template class)
        - GateMon
        - Transmon (contains qubit specific functions)
            - transmon in setup a (contains setup specific functions)


    Naming conventions for functions
        The qubit object is a combination of a parameter holder and a
        convenient way of performing measurements. As a convention the qubit
        object contains the following types of functions designated by a prefix
        - measure_
            common measurements such as "spectroscopy" and "ramsey"
        - find_
            used to extract a specific quantity such as the qubit frequency
            these functions should always call "measure_" functions off the
            qubit objects (merge with calibrate?)
        - calibrate_
            used to find the optimal parameters of some quantity (merge with
            find?).
        - calculate_
            calculates a quantity based on parameters in the qubit object and
            those specified
            e.g. calculate_frequency


    Open for discussion:
        - is a split at the level below qubit really required?
        - Hz vs GHz (@cdickel)
        - is the name "find_" a good name or should it be merged with measure
            or calibrate?
        - Should the pulse-parameters be grouped here in some convenient way?
            (e.g. parameter prefixes)


    Futre music (too complicated for now):
        - Instead of having a parameter resonator, attach a resonator object
          that has it's own frequency parameter, attach a mixer object that has
          it's own calibration routines.
    '''
    def __init__(self, name):
        super().__init__(name)
        self.msmt_suffix = '_' + name  # used to append to measuremnet labels

    def measure_T1(self):
        # Note: I made all functions lowercase but for T1 it just looks too
        # ridiculous
        raise NotImplementedError()

    def measure_rabi(self):
        raise NotImplementedError()

    def measure_ramsey(self):
        raise NotImplementedError()

    def measure_echo(self):
        raise NotImplementedError()

    def measure_allxy(self):
        raise NotImplementedError()

    def measure_ssro(self):
        raise NotImplementedError()

    def measure_spectroscopy(self):
        '''
        Discuss: here I mean a single tone (resonator spectroscpy), for a qubit
        spectroscopy one should use heterodyne_spectroscopy.

        Is this confusing? Should it be the other way round?
        '''
        raise NotImplementedError()

    def measure_heterodyne_spectroscopy(self):
        raise NotImplementedError()


class Transmon(Qubit):
    '''
    circuit-QED Transmon as used in DiCarlo Lab.
    Adds transmon specific parameters as well
    '''
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter('EC', units='Hz',
                           get_cmd=self._get_EC,
                           set_cmd=self._set_EC,
                           vals=vals.Numbers())

        self.add_parameter('EJ', units='Hz',
                           get_cmd=self._get_EJ,
                           set_cmd=self._set_EJ,
                           vals=vals.Numbers())
        self.add_parameter('assymetry',
                           get_cmd=self._get_assym,
                           set_cmd=self._set_assym)

        self.add_parameter('dac_voltage', units='mV',
                           get_cmd=self._get_dac_voltage,
                           set_cmd=self._set_dac_voltage)
        self.add_parameter('dac_sweet_spot', units='mV',
                           get_cmd=self._get_dac_sw_spot,
                           set_cmd=self._set_dac_sw_spot)
        self.add_parameter('dac_channel', vals=vals.Ints(),
                           get_cmd=self._get_dac_channel,
                           set_cmd=self._set_dac_channel)
        self.add_parameter('flux',
                           get_cmd=self._get_flux,
                           set_cmd=self._set_flux)

        self.add_parameter('f_qubit', label='qubit frequency', units='Hz',
                           get_cmd=self._get_freq,
                           set_cmd=self._set_freq)
        self.add_parameter('f_res', label='resonator frequency', units='Hz',
                           get_cmd=self._get_f_res,
                           set_cmd=self._set_f_res)

        # Sequence/pulse parameters
        self.add_parameter('RO_pulse_delay', units='s',
                           get_cmd=self._get_RO_pulse_delay,
                           set_cmd=self._set_RO_pulse_delay)
        self.add_parameter('RO_pulse_length', units='s',
                           get_cmd=self._get_RO_pulse_length,
                           set_cmd=self._set_RO_pulse_length)
        self.add_parameter('RO_trigger_delay', units='s',
                           get_cmd=self._get_RO_trigger_delay,
                           set_cmd=self._set_RO_trigger_delay)
        self.add_parameter('pulse_separation', units='s',
                           get_cmd=self._get_pulse_separation,
                           set_cmd=self._set_pulse_separation)

    def calculate_frequency(self, EC=None, EJ=None, assymetry=None,
                            dac_voltage=None, flux=None,
                            no_transitions=1):
        '''
        Calculates transmon energy levels from the full transmon qubit
        Hamiltonian.

        Parameters of the qubit object are used unless specified.
        Flux can be specified both in terms of dac voltage or flux but not
        both.

        If specified in terms of a dac voltage it will convert the dac voltage
        to a flux using convert dac to flux (NOT IMPLEMENTED)
        '''
        if EC is None:
            EC = self.EC.get()
        if EJ is None:
            EJ = self.EJ.get()
        if assymetry is None:
            assymetry = self.assymetry.get()

        if dac_voltage is not None and flux is not None:
            raise ValueError('Specify either dac voltage or flux but not both')
        if flux is None:
            flux = self.flux.get()
        # Calculating with dac-voltage not implemented yet

        freq = calculate_transmon_transitions(
            EC, EJ, asym=assymetry, reduced_flux=flux, no_transitions=1)
        return freq

    def calculate_flux(self, frequency):
        raise NotImplementedError()

    def prepare_for_timedomain(self):
        raise NotImplementedError()

    def prepare_for_continuous_wave(self):
        raise NotImplementedError()

    def find_frequency(self, method='spectroscopy',
                       steps=[1, 3, 10, 30, 100, 300, 1000],
                       verbose=True,
                       update=False,
                       close_fig=False):

        if method.lower() == 'spectroscopy':
            self.measure_spectroscopy()
        elif method.lower() == 'ramsey':

            stepsize = abs(1/self.f_pulse_mod.get())
            cur_freq = self.f_qubit.get()
            # Steps don't double to be more robust against aliasing
            for n in [1, 3, 10, 30, 100, 300, 1000]:
                times = np.arange(self.pulse_separation.get(),
                                  50*n*stepsize, n*stepsize)
                artificial_detuning = 4/times[-1]
                self.measure_ramsey(times,
                                    artificial_detuning=artificial_detuning,
                                    f_qubit=cur_freq,
                                    analyze=False)
                a = ma.Ramsey_Analysis(auto=True, close_fig=close_fig)
                fitted_freq = a.fit_res.params['frequency'].value
                measured_detuning = fitted_freq-artificial_detuning
                cur_freq -= measured_detuning
                if verbose:
                    print('Measured detuning:{:.2e}'.format(measured_detuning))
                    print('Setting freq to: {:.9e}, \n'.format(cur_freq))

                if times[-1] > 3*a.T2_star:
                    if verbose:
                        print('Breaking of measurement because of T2*')
                    break
        if verbose:
            print('Converged to: {:.9e}'.format(cur_freq))
        if update:
            self.f_qubit.set(cur_freq)
        return cur_freq

    def find_resonator_frequency(self, **kw):
        raise NotImplementedError()

    # All the functions below should be absorbed into the new
    # "holder parameter" get and set function of QCodes that does not exist yet
    def _set_EJ(self, val):
        self._EJ = val

    def _get_EJ(self):
        return self._EJ

    def _set_EC(self, val):
        self._EC = val

    def _get_EC(self):
        return self._EC

    def _set_assym(self, val):
        self._assym = val

    def _get_assym(self):
        return self._assym

    def _set_dac_voltage(self, val):
        self._dac_voltage = val

    def _get_dac_voltage(self):
        return self._dac_voltage

    def _set_dac_sw_spot(self, val):
        self._dac_sw_spot = val

    def _get_dac_sw_spot(self):
        return self._dac_sw_spot

    def _set_dac_channel(self, val):
        self._dac_channel = val

    def _get_dac_channel(self):
        return self._dac_channel

    def _set_flux(self, val):
        self._flux = val

    def _get_flux(self):
        return self._flux

    def _set_freq(self, val):
        self._freq = val

    def _get_freq(self):
        return self._freq

    def _set_f_res(self, val):
        self._f_res = val

    def _get_f_res(self):
        return self._f_res

    def _set_RO_pulse_delay(self, val):
        self._RO_pulse_delay = val

    def _get_RO_pulse_delay(self):
        return self._RO_pulse_delay

    def _set_RO_pulse_length(self, val):
        self._RO_pulse_length = val

    def _get_RO_pulse_length(self):
        return self._RO_pulse_length

    def _set_RO_trigger_delay(self, val):
        self._RO_trigger_delay = val

    def _get_RO_trigger_delay(self):
        return self._RO_trigger_delay

    def _set_pulse_separation(self, val):
        self._pulse_separation = val

    def _get_pulse_separation(self):
        return self._pulse_separation


class CBox_driven_transmon(Transmon):
    '''
    Setup configuration:
        Drive:                 CBox AWGs
        Acquisition:           CBox
        Readout pulse configuration: LO modulated using AWG
    '''
    def __init__(self, name,
                 LO, cw_source, td_source,
                 IVVI, AWG, LutMan,
                 CBox, heterodyne_instr,
                 MC):
        super().__init__(name)
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
                           units='V',
                           get_cmd=self._get_mod_amp_cw,
                           set_cmd=self._set_mod_amp_cw)
        self.add_parameter('mod_amp_td', label='RO modulation ampl td',
                           units='V',
                           get_cmd=self._get_mod_amp_td,
                           set_cmd=self._set_mod_amp_td)

        self.add_parameter('spec_pow', label='spectroscopy power',
                           units='dBm',
                           get_cmd=self._get_spec_pow,
                           set_cmd=self._set_spec_pow)
        self.add_parameter('spec_pow_pulsed',
                           label='pulsed spectroscopy power',
                           units='dBm',
                           get_cmd=self._get_spec_pow_pulsed,
                           set_cmd=self._set_spec_pow_pulsed)
        self.add_parameter('td_source_pow',
                           label='Time-domain power',
                           units='dBm',
                           get_cmd=self._get_td_source_pow,
                           set_cmd=self._set_td_source_pow)

        self.add_parameter('IF',
                           label='inter-modulation frequency', units='Hz',
                           get_cmd=self._get_IF,
                           set_cmd=self._set_IF)
        # Time-domain parameters
        self.add_parameter('f_pulse_mod',
                           label='pulse-modulation frequency', units='Hz',
                           get_cmd=self._get_f_pulsemod,
                           set_cmd=self._set_f_pulsemod)
        self.add_parameter('awg_nr', label='CBox awg nr', units='#',
                           get_cmd=self._get_awg_nr,
                           set_cmd=self._set_awg_nr)

        self.add_parameter('amp180',
                           label='Pi-pulse amplitude', units='mV',
                           get_cmd=self._get_amp180,
                           set_cmd=self._set_amp180)
        # amp90 is hard-linked to amp180
        self.add_parameter('amp90',
                           label='Pi/2-pulse amplitude', units='mV',
                           get_cmd=self._get_amp90)
        self.add_parameter('gauss_width', units='s',
                           get_cmd=self._get_gauss_width,
                           set_cmd=self._set_gauss_width)
        self.add_parameter('motzoi', label='Motzoi parameter', units='',
                           get_cmd=self._get_motzoi,
                           set_cmd=self._set_motzoi)

    def prepare_for_continuous_wave(self):

        self.heterodyne_instr._disable_auto_seq_loading = False
        self.LO.on()
        self.td_source.off()
        self.cw_source.on()
        self.heterodyne_instr.set('mod_amp', self.mod_amp_cw.get())
        self.heterodyne_instr.set('IF', self.IF.get())
        self.heterodyne_instr.frequency.set(self.f_res.get())
        self.cw_source.pulsemod_state.set('off')
        self.cw_source.power.set(self.spec_pow.get())

    def prepare_for_timedomain(self):
        self.LO.on()
        self.cw_source.off()
        self.td_source.on()
        # Set source to fs =f-f_mod such that pulses appear at f = fs+f_mod
        self.td_source.frequency.set(self.f_qubit.get()
                                     - self.f_pulse_mod.get())
        self.td_source.power.set(self.td_source_pow.get())
        self.AWG.set('ch3_amp', self.mod_amp_td.get())
        self.AWG.set('ch4_amp', self.mod_amp_td.get())
        self.CBox.set('AWG{:.0g}_mode'.format(self.awg_nr.get()), 'tape')

        self.LutMan.amp180.set(self.amp180.get())
        self.LutMan.amp90.set(self.amp90.get())
        self.LutMan.gauss_width.set(self.gauss_width.get()*1e9)  # s to ns
        self.LutMan.motzoi_parameter.set(self.motzoi.get())
        self.LutMan.f_modulation.set(self.f_pulse_mod.get()*1e-9)
        self.LutMan.load_pulses_onto_AWG_lookuptable(self.awg_nr.get())

    def find_resonator_frequency(self, use_min=False,
                                 update=True,
                                 freqs=None,
                                 MC=None, close_fig=False):
        '''
        Finds the resonator frequency by performing a heterodyne experiment
        if freqs == None it will determine a default range dependent on the
        last known frequency of the resonator.
        '''
        if freqs is None:
            f_center = self.f_res.get()
            f_span = 10e6
            f_step = 50e3
            freqs = np.arange(f_center-f_span/2, f_center+f_span/2, f_step)
        self.measure_heterodyne_spectroscopy(freqs, MC, analyze=False)
        a = ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)
        if use_min:
            f_res = a.min_frequency
        else:
            f_res = a.fit_results.params['f0'].value
        if f_res > max(freqs) or f_res < min(freqs):
            logging.warning('exracted frequency outside of range of scan')
        elif update:  # don't update if the value is out of the scan range
            self.f_res.set(f_res)
        return f_res

    def measure_heterodyne_spectroscopy(self, freqs, MC=None,
                                        analyze=True, close_fig=False):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        MC.set_sweep_function(pw.wrap_par_to_swf(
                              self.heterodyne_instr.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_spectroscopy(self, freqs, pulsed=False, MC=None,
                             analyze=True, close_fig=False):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        if pulsed:
            # Redirect to the pulsed spec function
            return self.measure_pulsed_spectroscopy(freqs,
                                                    MC, analyze, close_fig)

        MC.set_sweep_function(pw.wrap_par_to_swf(
                              self.cw_source.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(
            det.Heterodyne_probe(self.heterodyne_instr))
        MC.run(name='spectroscopy'+self.msmt_suffix)

        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_pulsed_spectroscopy(self, freqs, MC=None,
                                    analyze=True, close_fig=False):
        # This is a trick so I can reuse the heterodyne instr
        # to do pulsed-spectroscopy
        self.heterodyne_instr._disable_auto_seq_loading = True
        if ('Pulsed_spec' not in self.AWG.setup_filename.get()):
            st_seqs.Pulsed_spec_seq_RF_mod(
                IF=self.IF.get(),
                spec_pulse_length=16e-6, marker_interval=30e-6,
                RO_pulse_delay=self.RO_pulse_delay.get())
        self.cw_source.pulsemod_state.set('on')
        self.cw_source.power.set(self.spec_pow_pulsed.get())

        self.AWG.start()
        self.heterodyne_instr.mod_amp.set(self.mod_amp_td.get())
        MC.set_sweep_function(pw.wrap_par_to_swf(self.cw_source.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        MC.run(name='pulsed-spec'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_resonator_power(self, freqs, mod_amps,
                                MC=None, analyze=True, close_fig=False):
        '''
        N.B. This one does not use powers but varies the mod-amp.
        Need to find a way to keep this function agnostic to that
        '''
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        MC.set_sweep_functions(
            [pw.wrap_par_to_swf(self.heterodyne_instr.frequency),
             pw.wrap_par_to_swf(self.heterodyne_instr.mod_amp)])
        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(mod_amps)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        MC.run(name='Resonator_power_scan'+self.msmt_suffix, mode='2D')
        if analyze:
            ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_resonator_dac(self, freqs, dac_voltages,
                              MC=None, analyze=True, close_fig=False):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        MC.set_sweep_functions(
            [pw.wrap_par_to_swf(self.heterodyne_instr.frequency),
             pw.wrap_par_to_swf(
                self.IVVI['dac{}'.format(self.dac_channel.get())])
             ])
        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(dac_voltages)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_instr))
        MC.run(name='Resonator_dac_scan'+self.msmt_suffix, mode='2D')
        if analyze:
            ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_rabi(self, pulse_amps, n=1,
                     MC=None, analyze=True, close_fig=False,
                     verbose=False):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        st_seqs.CBox_multi_pulse_seq(
            IF=self.IF.get(), n_pulses=n,
            pulse_separation=self.pulse_separation.get(),
            RO_pulse_delay=self.RO_pulse_delay.get(),
            RO_trigger_delay=self.RO_trigger_delay.get(),
            RO_pulse_length=self.RO_pulse_length.get(), verbose=verbose)
        self.AWG.set('ch3_amp', self.mod_amp_td.get())
        self.AWG.set('ch4_amp', self.mod_amp_td.get())
        self.AWG.start()

        cal_points = [0, 0]
        pulse_amps = cal_points + list(pulse_amps)
        self.CBox.set('AWG0_tape', [1, 1])
        self.CBox.set('AWG1_tape', [1, 1])
        MC.set_sweep_function(pw.wrap_par_to_swf(self.LutMan.amp180))
        MC.set_sweep_points(pulse_amps)
        MC.set_detector_function(det.CBox_single_int_avg_with_LutReload(
                                 self.CBox, self.LutMan,
                                 awg_nrs=[self.awg_nr.get()]))
        MC.run('Rabi-n{}'.format(n)+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_T1(self, times, MC=None,
                   analyze=True, close_fig=False):
        '''
        if update is True will update self.T1 with the measured value
        '''
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        # append the calibration points, times are for location in plot
        times = np.concatenate([times,
                               (times[-1]+times[0],
                                times[-1]+times[0],
                                times[-1]+times[1],
                                times[-1]+times[1])])
        self.CBox.set('nr_samples', len(times))
        MC.set_sweep_function(
            awg_swf.CBox_T1(IF=self.IF.get(),
                            RO_pulse_delay=self.RO_pulse_delay.get(),
                            RO_trigger_delay=self.RO_trigger_delay.get(),
                            mod_amp=self.mod_amp_td.get(),
                            AWG=self.AWG,
                            upload=True))
        MC.set_sweep_points(times)
        MC.set_detector_function(det.CBox_integrated_average_detector(
                                 self.CBox, self.AWG))
        MC.run('T1'+self.msmt_suffix)
        if analyze:
            a = ma.T1_Analysis(auto=True, close_fig=False)
            return a.T1

    def measure_ramsey(self, times, artificial_detuning=0, f_qubit=None,
                       MC=None, analyze=True, close_fig=False, verbose=True):
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
            AWG=self.AWG, CBox=self.CBox, IF=self.IF.get(), pulse_separation=0,
            RO_pulse_delay=self.RO_pulse_delay.get(),
            RO_trigger_delay=self.RO_trigger_delay.get(),
            RO_pulse_length=self.RO_pulse_length.get())
        MC.set_sweep_function(Rams_swf)
        MC.set_sweep_points(times)
        MC.set_detector_function(det.CBox_integrated_average_detector(
                                 self.CBox, self.AWG))
        MC.run('Ramsey'+self.msmt_suffix)

        if analyze:
            a = ma.Ramsey_Analysis(auto=True, close_fig=False)

            if verbose:
                fitted_freq = a.fit_res.params['frequency'].value
                print('Artificial detuning: {:.2e}'.format(
                      artificial_detuning))
                print('Fitted detuning: {:.2e}'.format(fitted_freq))
                print('Actual detuning:{:.2e}'.format(
                      fitted_freq-artificial_detuning))

    def measure_allxy(self, MC=None,
                      analyze=True, close_fig=False, verbose=True):
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.MC
        d = cdet.AllXY_devition_detector_CBox(
                'AllXY'+self.msmt_suffix, MC=MC,
                AWG=self.AWG, CBox=self.CBox, IF=self.IF.get(),
                pulse_separation=self.pulse_separation.get(),
                RO_pulse_delay=self.RO_pulse_delay.get(),
                RO_trigger_delay=self.RO_trigger_delay.get(),
                RO_pulse_length=self.RO_pulse_length.get())
        d.prepare()
        d.acquire_data_point()
        if analyze:
            ma.AllXY_Analysis(close_main_fig=close_fig)

    ###########################
    # Parameter get set commands, should be removed in a later version.
    ############################
    def _get_mod_amp_cw(self):
        return self._mod_amp_cw

    def _set_mod_amp_cw(self, val):
        self._mod_amp_cw = val

    def _get_IF(self):
        return self._IF

    def _set_IF(self, val):
        self._IF = val

    def _get_mod_amp_cw(self):
        return self._mod_amp_cw

    def _set_mod_amp_cw(self, val):
        self._mod_amp_cw = val

    def _get_mod_amp_cw(self):
        return self._mod_amp_cw

    def _set_mod_amp_cw(self, val):
        self._mod_amp_cw = val

    def _get_mod_amp_cw(self):
        return self._mod_amp_cw

    def _set_mod_amp_cw(self, val):
        self._mod_amp_cw = val

    def _get_mod_amp_cw(self):
        return self._mod_amp_cw

    def _set_mod_amp_cw(self, val):
        self._mod_amp_cw = val

    def _get_mod_amp_td(self):
        return self._mod_amp_td

    def _set_mod_amp_td(self, val):
        self._mod_amp_td = val

    def _get_spec_pow(self):
        return self._spec_pow

    def _set_spec_pow(self, val):
        self._spec_pow = val

    def _get_spec_pow_pulsed(self):
        return self._spec_pow_pulsed

    def _set_spec_pow_pulsed(self, val):
        self._spec_pow_pulsed = val

    def _get_td_source_pow(self):
        return self._td_source_pow

    def _set_td_source_pow(self, val):
        self._td_source_pow = val

    def _get_f_pulsemod(self):
        return self._f_pulsemod

    def _set_f_pulsemod(self, val):
        self._f_pulsemod = val

    def _get_awg_nr(self):
        return self._awg_nr

    def _set_awg_nr(self, val):
        self._awg_nr = val

    def _get_amp180(self):
        return self._amp180

    def _set_amp180(self, val):
        self._amp180 = val

    def _get_amp90(self):
        return self.amp180.get()/2

    def _get_gauss_width(self):
        return self._gauss_width

    def _set_gauss_width(self, val):
        self._gauss_width = val

    def _get_motzoi(self):
        return self._motzoi

    def _set_motzoi(self, val):
        self._motzoi = val
