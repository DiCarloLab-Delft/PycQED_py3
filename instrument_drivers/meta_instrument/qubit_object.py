import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals

from modules.analysis.analysis_toolbox import calculate_transmon_transitions
from modules.measurement import detector_functions as det
from modules.measurement import mc_parameter_wrapper as pw
from modules.analysis import measurement_analysis as ma
from modules.measurement.pulse_sequences import standard_sequences as st_seqs


class Qubit(Instrument):
    '''
    Base class for the qubit object.
    Contains a template for all the functions a qubit should have.

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
        self.add_parameter('T1', units='s',
                           get_cmd=self.measure_T1)
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

    def find_frequency(self, method='spectroscopy', **kw):
        raise NotImplementedError()
        if method == 'spectroscopy':
            self.measure_spectroscopy(**kw)
        else:
            # Paste the Ramsey loop here
            self.measure_Ramsey(**kw)

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
        st_seqs.CBox_multi_pulse_seq(
            IF=self.IF.get(), n_pulses=n,
            pulse_separation=self.pulse_separation.get(),
            RO_pulse_delay=self.RO_pulse_delay.get(),
            RO_trigger_delay=self.RO_trigger_delay.get(),
            RO_pulse_length=self.RO_pulse_length.get(), verbose=verbose)
        self.AWG.start()

        cal_points = [0, 0]
        pulse_amps = cal_points + list(pulse_amps)
        self.CBox.set('AWG0_tape', [1, 1])
        self.CBox.set('AWG1_tape', [1, 1])
        self.MC.set_sweep_function(pw.wrap_par_to_swf(self.LutMan.amp180))
        self.MC.set_sweep_points(pulse_amps)
        self.MC.set_detector_function(det.CBox_single_int_avg_with_LutReload(
                                      self.CBox, self.LutMan,
                                      awg_nrs=[self.awg_nr.get()]))
        self.MC.run('Rabi-n{}'.format(n)+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

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