from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals


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

    def measure_heterodyne_spectroscopy(self, pulsed=False):
        raise NotImplementedError()


class Transmon(Qubit):
    '''
    circuit-QED Transmon as used in DiCarlo Lab.
    Adds transmon specific parameters as well
    '''
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter('E_c', units='Hz',
                           get_cmd=self._get_Ec,
                           set_cmd=self._set_Ec,
                           vals=vals.Numbers())

        self.add_parameter('E_j', units='Hz',
                           get_cmd=self._get_Ec,
                           set_cmd=self._set_Ec,
                           vals=vals.Numbers())
        self.add_parameter('assymetry')

        self.add_parameter('dac_sweet_spot', units='mV')
        self.add_parameter('dac_channel', units='mV')
        self.add_parameter('frequency', units='Hz')


    def calculate_frequency(self, dac=None, flux=None):
        raise NotImplementedError()

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


class CBox_driven_transmon(Transmon):
    def __init__(self, name):
        super().__init__(name)
        print('Hello')
