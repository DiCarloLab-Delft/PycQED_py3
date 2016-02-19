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
        raise NotImplementedError()

    def measure_Rabi(self):
        raise NotImplementedError()

    def measure_Ramsey(self):
        raise NotImplementedError()

    def measure_Echo(self):
        raise NotImplementedError()

    def measure_AllXY(self):
        raise NotImplementedError()

    def measure_SSRO(self):
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


class CBox_driven_transmon(Transmon):
    '''
    Setup configuration:
        Drive:                 CBox AWGs
        Acquisition:           CBox
        Readout pulse configuration: LO modulated using AWG
    '''
    def __init__(self, name,
                 LO, cw_source, td_source,
                 IVVI,
                 CBox, heterodyne_source,
                 MC):
        super().__init__(name)
        # MW-sources
        self.LO = LO
        self.cw_source = cw_source
        self.td_source = td_source
        self.IVVI = IVVI
        self.heterodyne_source = heterodyne_source
        self.CBox = CBox
        self.MC = MC
        self.add_parameter('mod_amp_cw', label='RO modulation ampl cw',
                           units='V',
                           get_cmd=self._get_mod_amp_cw,
                           set_cmd=self._set_mod_amp_cw)

    def prepare_for_continuous_wave(self):
        # TODO: Currently hardcoded here. This has to disappear
        self.LO.on()
        self.td_source.off()
        self.cw_source.on()
        self.CBox.set('nr_averages', 2**11)
        self.heterodyne_source.set('mod_amp', self.mod_amp_cw.get())
        self.heterodyne_source.set('IF', -20e6)

        self.heterodyne_source.frequency.set(self.f_res.get())

        self.cw_source.pulsemod_state.set('off')


    def prepare_for_timedomain(self):
        # TODO: Currently hardcoded here. This has to disappear
        self.LO.on()
        self.cw_source.off()
        self.td_source.on()
        self.AWG.set('ch3_amp', .13) #0.125)
        self.AWG.set('ch4_amp', .13) #0.125)

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
                              self.heterodyne_source.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_source))
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    def measure_spectroscopy(self, freqs, pulsed=False, MC=None,
                             analyze=True, close_fig=False):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        if not pulsed:
            MC.set_sweep_function(pw.wrap_par_to_swf(
                                  self.cw_source.frequency))
            MC.set_sweep_points(freqs)
            MC.set_detector_function(
                det.Heterodyne_probe(self.heterodyne_source))
            MC.run(name='spectroscopy'+self.msmt_suffix)
        elif pulsed:
            if not ('Pulsed_spec' in self.AWG.setup_filename.get()):
                st_seqs.Pulsed_spec_seq_RF_mod(
                    IF=self.IF.get(),
                    spec_pulse_length=16e-6, marker_interval=30e-6,
                    RO_pulse_delay=300e-9)
            self.cw_source.pulsemod_state.set('on')


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
            [pw.wrap_par_to_swf(self.heterodyne_source.frequency),
             pw.wrap_par_to_swf(self.heterodyne_source.mod_amp)])
        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(mod_amps)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_source))
        MC.run(name='Resonator_power_scan'+self.msmt_suffix, mode='2D')
        if analyze:
            ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_resonator_dac(self, freqs, dac_voltages,
                              MC=None, analyze=True, close_fig=False):
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.MC
        MC.set_sweep_functions(
            [pw.wrap_par_to_swf(self.heterodyne_source.frequency),
             pw.wrap_par_to_swf(
                self.IVVI['dac{}'.format(self.dac_channel.get())])
             ])
        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(dac_voltages)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_source))
        MC.run(name='Resonator_dac_scan'+self.msmt_suffix, mode='2D')
        if analyze:
            ma.MeasurementAnalysis(auto=True, TwoD=True, close_fig=close_fig)

    def measure_rabi(self, pulse_amps, n=1,
                     MC=None, analyze=True, close_fig=False,
                     verbose=False):
        logging.warning('Not tested warning')
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
        MC.set_sweep_function(pw.wrap_par_to_swf(self.LutMan.amp180))
        MC.set_sweep_points(pulse_amps)
        MC.set_detector_function(det.CBox_single_int_avg_with_LutReload(
                                 self.CBox, self.LutMan, awg_nrs=[0, 1]))
        MC.run('Rabi-flipping-{}'.format(n)+self.msmt_suffix)
        ma.MeasurementAnalysis(auto=True, close_fig=close_fig)

    ###########################
    # Parameter get set commands, should be removed in a later version.
    ############################
    def _get_mod_amp_cw(self):
        return self._mod_amp_cw

    def _set_mod_amp_cw(self, val):
        self._mod_amp_cw = val
