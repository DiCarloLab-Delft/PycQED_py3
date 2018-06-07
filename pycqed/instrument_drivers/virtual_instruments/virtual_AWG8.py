import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from zlib import crc32

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
        self.add_dummy_parameters()

    def add_dummy_parameters(self):
        parnames = []
        for i in range(8):
            parnames.append('sigouts_{}_offset'.format(i))
            parnames.append('sigouts_{}_on'.format(i))
            self.add_parameter(
                'awgs_{}_outputs_{}_amplitude'.format(i//2, i % 2),
                initial_value=.5, parameter_class=ManualParameter)

            self.add_parameter('sigouts_{}_direct'.format(i),
                               initial_value=1,
                               parameter_class=ManualParameter)

            self.add_parameter('sigouts_{}_range'.format(i),
                               initial_value=0.8,
                               parameter_class=ManualParameter)
        for par in parnames:
            self.add_parameter(par, parameter_class=ManualParameter)

        for i in range(4):
            self.add_parameter(
                'awgs_{}_sequencer_program_crc32_hash'.format(i),
                parameter_class=ManualParameter,
                initial_value=0, vals=vals.Ints())

    def snapshot_base(self, update=False, params_to_skip_update=None):
        if params_to_skip_update is None:
            params_to_skip_update = self._params_to_skip_update
        snap = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update)
        return snap

    def _add_codeword_parameters(self):
        """
        Adds parameters parameters that are used for uploading codewords.
        It also contains initial values for each codeword to ensure
        that the "upload_codeword_program"

        """
        docst = ('Specifies a waveform to for a specific codeword. ' +
                 'The waveforms must be uploaded using ' +
                 '"upload_codeword_program". The channel number corresponds' +
                 ' to the channel as indicated on the device (1 is lowest).')
        self._params_to_skip_update = []
        for ch in range(self._num_channels):
            for cw in range(self._num_codewords):
                parname = 'wave_ch{}_cw{:03}'.format(ch+1, cw)
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(
                        ch+1, cw),
                    vals=vals.Arrays(),  # min_value, max_value = unknown
                    parameter_class=ManualParameter,
                    docstring=docst)
                self._params_to_skip_update.append(parname)

    def upload_codeword_program(self, awgs=np.arange(4)):
        for awg_nr in awgs:
            program = 'some dummy program_{}'.format(awg_nr)
            self.configure_awg_from_string(awg_nr=int(awg_nr),
                                           program_string=program,
                                           timeout=self.timeout())

    def configure_awg_from_string(self, awg_nr: int, program_string: str,
                                  timeout: float=15):
        # Actual uploading does not exist in the dummy AWG8
        hash = crc32(program_string.encode('utf-8'))
        self.set('awgs_{}_sequencer_program_crc32_hash'.format(awg_nr),
                 hash)

    def stop(self):
        pass

    def start(self):
        pass

    def upload_waveform_realtime(self, w0, w1, awg_nr: int, wf_nr: int =1):
        """
        Arguments:
            w0   (array): waveform for ch0 of the awg pair.
            w1   (array): waveform for ch1 of the awg pair.
            awg_nr (int): awg_nr indicating what awg pair to use.
            wf_nr  (int): waveform in memory to overwrite, default is 1.

        There are a few important notes when using this method
        - w0 and w1 must be of the same length
        - any parts of a waveform longer than w0/w1 will not be overwritten.
        - loading speed depends on the size of w0 and w1 and is ~80ms for 20us.

        """
        # these two attributes are added for debugging purposes.
        # they allow checking what the realtime loaded waveforms are.
        self._realtime_w0 = w0
        self._realtime_w1 = w1
        # stacking is here to mimic the full realtime loading
        c = np.vstack((w0, w1)).reshape((-2,), order='F')
        self._realtime_wf_c = c
