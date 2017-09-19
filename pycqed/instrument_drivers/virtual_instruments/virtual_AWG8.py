import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


class VirtualAWG8(Instrument):
    """
    Dummy instrument that implements some of the interface of the AWG8.
    Most notably the codewords.

    We could also code generate a dummy interface from the json files and
    have an abstract ZI virtual instrument. This is right now beyond
    the scope.
    """

    def __init__(self, name):
        super().__init__(name)
        self.add_parameter('timeout', unit='s', initial_value=5,
                           parameter_class=ManualParameter,
                           vals=vals.MultiType(vals.Numbers(min_value=0),
                                               vals.Enum(None)))

        self._num_channels = 8
        self._num_codewords = 256

        self._add_codeword_parameters()

    def _add_codeword_parameters(self):
        """
        Adds manual parameters that are used for the codewords.
        It also contains initial values for each codeword to ensure
        that the "upload_codeword_program"

        """
        docst = ('Specifies a waveform to for a specific codeword. ' +
                 'The waveforms must be uploaded using ' +
                 '"upload_codeword_program". The channel number corresponds' +
                 ' to the channel as indicated on the device (1 is lowest).')
        for ch in range(self._num_channels):
            for cw in range(self._num_codewords):
                parname = 'wave_ch{}_cw{:03}'.format(ch+1, cw)
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(
                        ch+1, cw),
                    vals=vals.Arrays(),  # min_value, max_value = unknown
                    parameter_class=ManualParameter,
                    initial_value=np.zeros(32),
                    docstring=docst)

    def stop(self):
        pass

    def start(self):
        pass