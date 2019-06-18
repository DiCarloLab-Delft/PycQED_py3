"""
Notes:
- this is the 'core' driver for the ZI HDAWG: it contains the generic parts of
  the driver. Application dependent parts (e.g. codeword handling) reside in ZI_HDAWG8.py

To do:
- replace print() by logging

Changelog:

20190212 WJV
- separated off application independent stuff into this ZI_HDAWG_core class,
  file ZI_HDAWG8.py will keep application dependent stuff. The software
  interface remains unchanged.
- addressed many warnings identified by PyCharm
- started adding more type annotations

20190214 WJV
- added check_timing_error()
- moved out _add_extra_parameters() and _add_codeword_parameters()
- added load_default_settings()
- moved in _set_dio_delay()
- replaced some warnings by Exceptions
- made some instance variables 'private'

20190219 WJV
- removed unused open from _write_csv_waveform()

20190429 WJV
- merged branch 'QCC_testing' into 'feature/cc', changes:
    pulled in changed upload_waveform_realtime from ZI_HDAWG8.py again

20190618 WJV
- merged branch 'develop' into 'feature/cc', changes:
    pulled in changed upload_waveform_realtime from ZI_HDAWG8.py again

"""

import logging
import os
import time
import numpy as np
import ctypes
from ctypes.wintypes import MAX_PATH
from zlib import crc32

from . import zishell_NH as zs
from .ZI_base_instrument import ZI_base_instrument

log = logging.getLogger(__name__)

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

    def __init__(self, name: str,
                 device: str,
                 server: str = 'localhost', port = 8004,
                 **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument as seen by the user
            device          (str) the name of the device e.g., "dev8008"
            server          (str) the ZI data server
            port            (int) the port to connect to
        """

        super().__init__(name=name, **kw)

        # determine path for LabOne web server
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

        # save some parameters
        self._devname = device

        # connect to data server and device
        self._dev = zs.ziShellDevice()
        self._dev.connect_server(server, port)
        print("Trying to connect to device {}".format(self._devname))
        self._dev.connect_device(self._devname, '1GbE')

        # add qcodes parameters based on JSON parameter file
        # FIXME: we might want to skip/remove/(add  to _params_to_skip_update) entries like AWGS/*/ELF/DATA,
        #       AWGS/*/SEQUENCER/ASSEMBLY, AWGS/*/DIO/DATA
        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_fn = os.path.join(dir_path, 'zi_parameter_files')
        filename = os.path.join(base_fn, 'node_doc_HDAWG8.json')
        try:
            self.add_parameters_from_file(filename=filename)  # NB: defined in parent class
        except FileNotFoundError:
            log.error("parameter file for data parameters"
                            " {} not found".format(filename))
            raise
            # FIXME: we need to be capable to generate file if none exists

        self._add_ZIshell_device_methods_to_instrument()

        # determine number of channels
        dev_type = self.get('features_devtype')
        if dev_type == 'HDAWG8':
            self._num_channels = 8
        elif dev_type == 'HDAWG4':
            self._num_channels = 4
        else:
            raise Exception("Unknown device type '{}'".format(dev_type))

        # show some info
        serial = self.get('features_serial')
        options = self.get('features_options') # FIXME: check that we have what we need
        fw_revision = self.get('system_fwrevision') # Revision of the device internal controller software FIXME: check against minimum we need
        fpga_revision = self.get('system_fpgarevision') # HDL firmware revision FIXME: check against minimum we need
        log.info('AWG8: serial={}, options={}, fw_revision={}, fpga_revision={}'
                 .format(serial, options.replace('\n','|'), fw_revision, fpga_revision))
        log.info('DIO interface found in mode {} (0=CMOS, 1=LVDS)'.
                 format(self.get('dios_0_interface'))) # NB: mode is persistent across device restarts

        # NB: we don't want to load defaults automatically, but leave it up to the user

    def load_default_settings(self):
        """
        bring device into known state
        FIXME: incomplete
        """
        # clear output

        # clear AWGs

        # reset DIO parameters (bits, timing)
        for awg in range(4):
            self._set_dio_delay(awg, 0, 0xFFFFFFFF, 0)  # set all delays to 0

        # interesting nodes:
        # SYSTEM/AWG/CHANNELGROUPING
        # SYSTEM/CLOCKS/SAMPLECLOCK/FREQ
        #


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
                # get status (see docstring):
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
        # FIXME: also look at SYSTEM/CLOCKS/SAMPLECLOCK/STATUS ?

    # FIXME: add check_virt_mem_use(self)
    # AWGS/0/SEQUENCER/MEMORYUSAGE
    # AWGS/0/WAVEFORM/MEMORYUSAGE

    # FIXME: add support for nodes: RAW/ERROR/*, see https://people.zhinst.com/%7Eniels/

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
        # FIXME: side effect: the function above touches '/dio/strobe/slope'
        hash_val = crc32(program_string.encode('utf-8'))
        self.set('awgs_{}_sequencer_program_crc32_hash'.format(awg_nr),
                 hash_val)

    def upload_waveform_realtime(self, w0, w1, awg_nr: int, wf_nr: int = 1):
        """
        Warning! This method should be used with care.
        Uploads a waveform to the awg in realtime, note that this get's
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

        # Checked and everything matches
        # print(self)
        # print(hex(id(self)))
        # print(self._dev)
        # print(hex(id(self._dev)))

        c = np.vstack((w0, w1)).reshape((-2,), order='F')
        self._dev.seti('awgs/{}/enable'.format(awg_nr), 0)
        self._dev.subs('awgs/{}/ready'.format(awg_nr))
        self._dev.seti('awgs/{}/waveform/index'.format(awg_nr), wf_nr)
        # self._dev.setv('awgs/{}/waveform/data'.format(awg_nr), c)
        # Try as float32 instead
        self._dev.setv('awgs/{}/waveform/data'.format(awg_nr), c.astype(np.float32))

        # Commented out checking if ready.
        # creates too much time overhead.
        # data = self._dev.poll()
        # t0 = time.time()
        # while not data:
        #     data = self._dev.poll()
        #     if time.time()-t0> self.timeout():
        #         raise TimeoutError
        # self._dev.unsubs('awgs/{}/ready'.format(awg_nr))
        self._dev.seti('awgs/{}/enable'.format(awg_nr), 1)

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
#        with open(filename, 'w') as f:  # FIXME: unused
#            np.savetxt(filename, waveform, delimiter=",")
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
            log.warning(e)
            print(e)
            return None

    def _set_dio_delay(self, awg, strb_mask, data_mask, delay):
        """
        The function sets the DIO delay for a given FPGA. The valid delay range is
        0 to 6. The delays are created by either delaying the data bits or the strobe
        bit. The data_mask input represents all bits that are part of the codeword or
        the valid bit. The strb_mask input represents the bit that define the strobe.
        """
        if delay < 0:
            print('WARNING: Clamping delay to 0')
        if delay > 6:
            print('WARNING: Clamping delay to 6')
            delay = 6

        strb_delay = 0
        data_delay = 0
        if delay > 3:
            strb_delay = delay-3
        else:
            data_delay = 3-delay

        for i in range(32):
            self._dev.seti('awgs/{}/dio/delay/index'.format(awg), i)
            if strb_mask & (1 << i):
                self._dev.seti(
                    'awgs/{}/dio/delay/value'.format(awg), strb_delay)
            elif data_mask & (1 << i):
                self._dev.seti(
                    'awgs/{}/dio/delay/value'.format(awg), data_delay)
            else:
                self._dev.seti('awgs/{}/dio/delay/value'.format(awg), 0)
