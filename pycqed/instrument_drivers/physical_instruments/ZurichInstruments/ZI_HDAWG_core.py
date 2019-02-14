"""
Changelog:

20190112 WJV
- separated off application independent stuff into this ZI_HDAWG_core class,
  file ZI_HDAWG8.py will keep application dependent stuff. The software
  interface remains unchanged.
- addressed many warnings identified by PyCharm
- started adding more type annotations

"""

from . import zishell_NH as zs
from .ZI_base_instrument import ZI_base_instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
import logging
import os
import time
import numpy as np
import ctypes
from ctypes.wintypes import MAX_PATH
from zlib import crc32


class ZI_HDAWG_core(ZI_base_instrument):
    """
    This is PycQED/QCoDeS driver driver for the Zurich Instruments HDAWG.

    Parameter files are generated from the python API of the instrument
    using the "create_parameter_file" method in the ZI_base_instrument class.
    These are used to add parameters to the instrument.
    """

    ##########################################################################
    # 'public' functions: device control
    ##########################################################################

    def __init__(self, name, device: str,
                 server: str = 'localhost', port=8004,
                 num_codewords: int = 32, **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument as seen by the user
            device          (str) the name of the device e.g., "dev8008"
            server          (str) the ZI data server
            port            (int) the port to connect to
        """
        t0 = time.time()
        self._num_codewords = num_codewords

        if os.name == 'nt':
            dll = ctypes.windll.shell32
            buf = ctypes.create_unicode_buffer(MAX_PATH + 1)
            if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
                _basedir = buf.value
            else:
                raise Exception('Could not extract my documents folder')
        else:
            _basedir = os.path.expanduser('~')
        self._lab_one_webserver_path = os.path.join(
            _basedir, 'Zurich Instruments', 'LabOne', 'WebServer')

        super().__init__(name=name, **kw)
        self._devname = device
        self._dev = zs.ziShellDevice()
        self._dev.connect_server(server, port)
        print("Trying to connect to device {}".format(self._devname))
        self._dev.connect_device(self._devname, '1GbE')

        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_fn = os.path.join(dir_path, 'zi_parameter_files')
        filename = os.path.join(base_fn, 'node_doc_HDAWG8.json')
        try:
            self.add_parameters_from_file(filename=filename)
        except FileNotFoundError:
            logging.error("parameter file for data parameters"
                            " {} not found".format(filename))
            raise

        self._add_ZIshell_device_methods_to_instrument()

        dev_type = self.get('features_devtype')
        if dev_type == 'HDAWG8':
            self._num_channels = 8
        elif dev_type == 'HDAWG4':
            self._num_channels = 4
        else:
            raise Exception("Unknown device type '{}'".format(dev_type))

        self._add_codeword_parameters()
        self._add_extra_parameters()
        self.connect_message(begin_time=t0)

    def get_idn(self) -> dict:
        # FIXME, update using new parameters for this purpose
        idn_dict = {'vendor': 'ZurichInstruments',
                    'model': self._dev.daq.getByte(
                        '/{}/features/devtype'.format(self._devname)),
                    'serial': self._devname,
                    'firmware': self._dev.geti('system/fwrevision'),
                    'fpga_firmware': self._dev.geti('system/fpgarevision')
                    }
        return idn_dict

    def assure_ext_clock(self) -> None:
        """
        Make sure the instrument is using an external reference clock

        Based on: AWG8_V2_DIO_Calibrarion.ipynb
        """

        # get source:
        #   1: external
        #   0: internal (commanded so, or because of failure to sync to external clock)
        source = self.system_clocks_referenceclock_source()
        if source == 1:
            return

        print('Switching to external clock. This could take a while!')
        while True:
            self.system_clocks_referenceclock_source(1)
            while True:
                # get status:
                #   0: synced
                #   1: sync failed (will force clock source to internal and retry)
                #   2: syncing
                status = self.system_clocks_referenceclock_status()
                if status == 0:             # synced
                    break
                elif status == 2:           # syncing
                    print('.', end='')
                else:                       # sync failed
                    print('X', end='')
                time.sleep(0.1)
            if self.system_clocks_referenceclock_source() != 1:
                print(' Switching to external clock failed. Trying again.')
            else:
                break
        print('\nDone')

    # from IPython notebook AWG8_staircase_test.ipynb FIXME: return result
    def check_timing_error(self, awgs):
        for awg in awgs:
            timing_error = self.geti('awgs/' + str(awg) + '/dio/error/timing')
            if timing_error != 0:
                print('Timing error detected on DIO of AWG {}: 0x{:08x}'.format(awg, timing_error))
            else:
                print('No timing error detected on DIO of AWG {}'.format(awg))

    def stop(self) -> None:
        """
        Stops the program on all AWG's part of this HDAWG unit
        """
        for i in range(int(self._num_channels/2)):
            self.set('awgs_{}_enable'.format(i), 0)

    def start(self) -> None:
        """
        Starts the program on all AWG's part of this HDAWG unit
        """
        for i in range(int(self._num_channels/2)):
            self.set('awgs_{}_enable'.format(i), 1)

    def snapshot_base(self, update=False, params_to_skip_update=None):
        if params_to_skip_update is None:
            params_to_skip_update = self._params_to_skip_update
        snap = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update)
        return snap

    ##########################################################################
    # 'public' functions: generic AWG/waveform support
    ##########################################################################

    def configure_awg_from_string(self, awg_nr: int, program_string: str,
                                  timeout: float = 15) -> None:
        """
        Uploads a program string to one of the AWGs of the HDAWG.
        """
        self._dev.configure_awg_from_string(awg_nr=awg_nr,
                                            program_string=program_string,
                                            timeout=timeout)
        hash_val = crc32(program_string.encode('utf-8'))
        self.set('awgs_{}_sequencer_program_crc32_hash'.format(awg_nr),
                 hash_val)

    def upload_waveform_realtime(self, w0, w1, awg_nr: int, wf_nr: int = 1) -> None:
        """
        Warning! This method should be used with care.
        Uploads a waveform to the awg in realtime, note that this gets
        overwritten if a new program is uploaded.

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

        c = np.vstack((w0, w1)).reshape((-2,), order='F')
        self._dev.seti('awgs/{}/waveform/index'.format(awg_nr), wf_nr)
        self._dev.setv('awgs/{}/waveform/data'.format(awg_nr), c)
        self._dev.seti('awgs/{}/enable'.format(awg_nr), wf_nr)

    ##########################################################################
    # 'private' functions, internal to the driver
    ##########################################################################

    def _add_ZIshell_device_methods_to_instrument(self) -> None:
        """
        Some methods defined in the zishell are convenient as public
        methods of the instrument. These are added here.
        """
        self.reconnect = self._dev.reconnect
        self.restart_device = self._dev.restart_device
        self.poll = self._dev.poll
        self.sync = self._dev.sync
        self.read_from_scope = self._dev.read_from_scope
        self.restart_scope_module = self._dev.restart_scope_module
        self.restart_awg_module = self._dev.restart_awg_module

    def _add_extra_parameters(self) -> None:
        self.add_parameter('timeout', unit='s',
                           initial_value=10,
                           parameter_class=ManualParameter)
        self.add_parameter(
            'cfg_num_codewords', label='Number of used codewords', docstring=(
                'This parameter is used to determine how many codewords to '
                'upload in "self.upload_codeword_program".'),
            initial_value=self._num_codewords,
            # FIXME: commented out numbers larger than self._num_codewords
            # see also issue #358
            vals=vals.Enum(2, 4, 8, 16, 32),  # , 64, 128, 256, 1024),
            parameter_class=ManualParameter)

        self.add_parameter(
            'cfg_codeword_protocol', initial_value='identical',
            vals=vals.Enum('identical', 'microwave', 'flux'), docstring=(
                'Used in the configure codeword method to determine what DIO'
                ' pins are used in for which AWG numbers.'),
            parameter_class=ManualParameter)

        for i in range(4):
            self.add_parameter(
                'awgs_{}_sequencer_program_crc32_hash'.format(i),
                parameter_class=ManualParameter,
                initial_value=0, vals=vals.Ints())

    def _add_codeword_parameters(self) -> None:
        """
        Adds parameters that are used for uploading codewords.
        It also contains initial values for each codeword to ensure
        that the "upload_codeword_program" ... FIXME: comment ends

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
                    set_cmd=self._gen_write_csv(parname),
                    get_cmd=self._gen_read_csv(parname),
                    docstring=docst)
                self._params_to_skip_update.append(parname)

    def _gen_write_csv(self, wf_name):
        def write_func(waveform):
            # The length of HDAWG waveforms should be a multiple of 8 samples.
            if (len(waveform) % 8) != 0:
                extra_zeros = 8-(len(waveform) % 8)
                waveform = np.concatenate([waveform, np.zeros(extra_zeros)])
            return self._write_csv_waveform(
                wf_name=wf_name, waveform=waveform)
        return write_func

    def _write_csv_waveform(self, wf_name: str, waveform) -> None:
        filename = os.path.join(
            self._lab_one_webserver_path, 'awg', 'waves',
            self._devname+'_'+wf_name+'.csv')
        with open(filename, 'w') as f:  # FIXME: unused
            np.savetxt(filename, waveform, delimiter=",")

    def _gen_read_csv(self, wf_name):
        def read_func():
            return self._read_csv_waveform(
                wf_name=wf_name)
        return read_func

    def _read_csv_waveform(self, wf_name: str):
        filename = os.path.join(
            self._lab_one_webserver_path, 'awg', 'waves',
            self._devname+'_'+wf_name+'.csv')
        try:
            return np.genfromtxt(filename, delimiter=',')
        except OSError as e:
            # if the waveform does not exist yet dont raise exception
            logging.warning(e)
            print(e)
            return None
