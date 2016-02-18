import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals

from modules.analysis.analysis_toolbox import calculate_transmon_transitions
from modules.measurement import detector_functions as det
from modules.measurement import mc_parameter_wrapper as pw
from modules.analysis import measurement_analysis as ma


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
                 CBox, heterodyne_source,
                 MC):
        super().__init__(name)
        # MW-sources
        self.LO = LO
        self.cw_source = cw_source
        self.td_source = td_source

        self.heterodyne_source = heterodyne_source
        self.CBox = CBox
        self.MC = MC

    def prepare_for_continuous_wave(self):
        # Currently a lot hardcoded here. This has to disappear
        self.LO.on()
        self.td_source.off()
        self.cw_source.on()
        self.CBox.set('nr_averages', 2**11)
        self.heterodyne_source.set('mod_amp', .08)
        self.heterodyne_source.set('IF', -20e6)

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
        if MC is None:
            MC = self.MC
        MC.set_sweep_function(pw.wrap_par_to_swf(
                              self.heterodyne_source.frequency))
        MC.set_sweep_points(freqs)
        MC.set_detector_function(det.Heterodyne_probe(self.heterodyne_source))
        MC.run(name='Resonator_scan'+self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True, close_fig=close_fig)
