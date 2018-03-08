import os
import numpy as np
import logging

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

import ctypes
from ctypes.wintypes import MAX_PATH

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
        self._awg_str = [''] * 4
        self._waveforms = [{}, {}, {}, {}]
        self._devname = 'virt_awg8'

        if os.name == 'nt':
            dll = ctypes.windll.shell32
            buf = ctypes.create_unicode_buffer(MAX_PATH + 1)
            if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
                _basedir = buf.value
            else:
                logging.warning('Could not extract my documents folder')
        else:
            _basedir = os.path.expanduser('~')
        self.lab_one_webserver_path = os.path.join(
            _basedir, 'Zurich Instruments', 'LabOne', 'WebServer')

    def add_dummy_parameters(self):
        parnames = []
        for i in range(8):
            parnames.append('sigouts_{}_offset'.format(i))
            parnames.append('sigouts_{}_on'.format(i))
            parnames.append('sigouts_{}_direct'.format(i))
            parnames.append('sigouts_{}_range'.format(i))
            parnames.append('triggers_in_{}_imp50'.format(i))
            parnames.append('triggers_in_{}_level'.format(i))
        for i in range(4):
            parnames.append('awgs_{}_enable'.format(i))
            parnames.append('awgs_{}_auxtriggers_0_channel'.format(i))
            parnames.append('awgs_{}_auxtriggers_0_slope'.format(i))
            parnames.append('awgs_{}_dio_mask_value'.format(i))
            parnames.append('awgs_{}_dio_mask_shift'.format(i))
            parnames.append('awgs_{}_dio_strobe_index'.format(i))
            parnames.append('awgs_{}_dio_strobe_slope'.format(i))
            parnames.append('awgs_{}_dio_valid_index'.format(i))
            parnames.append('awgs_{}_dio_valid_polarity'.format(i))
        parnames.append('system_extclk')

        for par in parnames:
            self.add_parameter(par, parameter_class=ManualParameter)

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

    def clock_freq(self, awg_nr):
        return 2.4e9

    def upload_codeword_program(self):
        pass

    def stop(self):
        pass

    def start(self):
        pass

    def configure_awg_from_string(self, awg_nr, awg_str, timeout):
        self._awg_str[awg_nr] = awg_str

    def awg_update_waveform(self, awg_nr, i, data1, data2):
        self._waveforms[awg_nr][i] = (data1.copy(), data2.copy())

    def _write_csv_waveform(self, wf_name: str, waveform):
        filename = os.path.join(
            self.lab_one_webserver_path, 'awg', 'waves',
            self._devname+'_'+wf_name+'.csv')
        with open(filename, 'w') as f:
            np.savetxt(filename, waveform, delimiter=",")

    def _read_csv_waveform(self, wf_name: str):
        filename = os.path.join(
            self.lab_one_webserver_path, 'awg', 'waves',
            self._devname+'_'+wf_name+'.csv')
        try:
            return np.genfromtxt(filename, delimiter=',')
        except OSError as e:
            # if the waveform does not exist yet dont raise exception
            logging.warning(e)
            print(e)
            return None