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
        - Transmon (contains qubit specific functions)
            - transmon in setup a (contains setup specific functions)
    '''
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter('T1', units='s',
                           get_cmd=self._measure_T1())

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



