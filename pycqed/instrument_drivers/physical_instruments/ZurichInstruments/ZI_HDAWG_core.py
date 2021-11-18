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

20190709 NCH
- Updated to use the improved ZI_base_instrument functionality, which now
    makes a lot of methods common to both the HDAWG and the UHF-QA.
- Removed zishell_NH

"""

import logging
import time
import json
import copy

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase

log = logging.getLogger(__name__)

class ZI_HDAWG_core(zibase.ZI_base_instrument):
    """
    This is PycQED/QCoDeS driver driver for the Zurich Instruments HDAWG.

    Parameter files are generated from the python API of the instrument
    using the "create_parameter_file" method in the ZI_base_instrument class.
    These are used to add parameters to the instrument.
    """

    # Define minimum required revisions
    MIN_FWREVISION = 62730
    MIN_FPGAREVISION = 62832
    MIN_SLAVEREVISION = 62659

    ##########################################################################
    # 'public' functions: device control
    ##########################################################################

    def __init__(self,
                 name: str,
                 device: str,
                 interface: str = '1GbE',
                 server: str = 'localhost',
                 port: int = 8004,
                 num_codewords: int = 32,
                 **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument as seen by the user
            device          (str) the name of the device e.g., "dev8008"
            interface       (str) the name of the interface to use ('1GbE' or 'USB')
            server          (str) the host where the ziDataServer is running
            port            (int) the port to connect to for the ziDataServer (don't change)
            num_codewords   (int) the number of codeword-based waveforms to prepare
        """
        t0 = time.time()
        super().__init__(name=name, device=device, interface=interface, server=server, port=port, num_codewords=num_codewords, **kw)
        self._default_waveform_length = 48  # Override default waveform length to 20 ns at 2.4 GSa/s

        # NB: we don't want to load defaults automatically, but leave it up to the user

        # Configure instrument to blink forever
        self.seti('raw/error/blinkseverity', 1)
        self.seti('raw/error/blinkforever', 1)

        t1 = time.time()
        log.info('{}: Initialized ZI_HDAWG_core in {}s'.format(self.devname, t1-t0))

    def _check_devtype(self):
        if self.devtype != 'HDAWG8' and self.devtype != 'HDAWG4':
            raise zibase.ziDeviceError('Device {} of type {} is not a HDAWG instrument!'.format(self.devname, self.devtype))

    def _check_options(self):
        """
        Checks that the correct options are installed on the instrument.
        """
        options = self.gets('features/options').split('\n')
        if 'PC' not in options:
            raise zibase.ziOptionsError('Device {} is missing the PC option!'.format(self.devname))

    def _check_awg_nr(self, awg_nr):
        """
        Checks that the given AWG index is valid for the device.
        """
        if self.devtype == 'HDAWG8' and (awg_nr < 0 or awg_nr > 3):
            raise zibase.ziValueError('Invalid AWG index of {} detected!'.format(awg_nr))
        elif self.devtype == 'HDAWG4' and (awg_nr < 0 or awg_nr > 1):
            raise zibase.ziValueError('Invalid AWG index of {} detected!'.format(awg_nr))

    def _check_versions(self):
        """
        Checks that sufficient versions of the firmware are available.
        """
        if self.geti('system/fwrevision') < ZI_HDAWG_core.MIN_FWREVISION:
            raise zibase.ziVersionError('Insufficient firmware revision detected! Need {}, got {}!'.format(ZI_HDAWG_core.MIN_FWREVISION, self.geti('system/fwrevision')))

        if self.geti('system/fpgarevision') < ZI_HDAWG_core.MIN_FPGAREVISION:
            raise zibase.ziVersionError('Insufficient FPGA revision detected! Need {}, got {}!'.format(ZI_HDAWG_core.MIN_FPGAREVISION, self.geti('system/fpgarevision')))

        if self.geti('system/slaverevision') < ZI_HDAWG_core.MIN_SLAVEREVISION:
            raise zibase.ziVersionError('Insufficient FPGA Slave revision detected! Need {}, got {}!'.format(ZI_HDAWG_core.MIN_SLAVEREVISION, self.geti('system/slaverevision')))

    def _num_channels(self):
        if self.devtype == 'HDAWG8':
            return 8
        elif self.devtype == 'HDAWG4':
            return 4
        else:
            raise Exception("Unknown device type '{}'".format(self.devtype))

    def load_default_settings(self):
        """
        bring device into known state
        """

        log.warning('{}: loading default settings (FIXME: still incomplete)'
                    .format(self.devname))

        # Setting the clock to external
        self.assure_ext_clock()

        # clear output

        # clear AWGs

        # reset DIO parameters (bits, timing)
        self._set_dio_delay(0)  # set all delays to 0

        # interesting nodes:
        # SYSTEM/AWG/CHANNELGROUPING
        # SYSTEM/CLOCKS/SAMPLECLOCK/FREQ
        #

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
            log.info(f'{self.devname}: Already using external clock')
            return

        log.info(f'{self.devname}: Switching to external clock. This could take a while')
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
                log.error(f'{self.devname}: Switching to external clock failed. Trying again')
            else:
                break
        log.info(f'{self.devname}: Successfully switched to external clock')

    # FIXME: add check_virt_mem_use(self)
    # AWGS/0/SEQUENCER/MEMORYUSAGE
    # AWGS/0/WAVEFORM/MEMORYUSAGE

    def get_idn(self) -> dict:
        idn_dict = super().get_idn()
        idn_dict['slave_firmware'] = self.geti('system/slaverevision')
        return idn_dict

    ##########################################################################
    # 'public' functions: generic AWG/waveform support
    ##########################################################################

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
        raise NotImplementedError('Please use the waveform parameters ("wave_chN_cwM") of the object to change waveforms!')

    ##########################################################################
    # 'public' functions: DIO debug support
    ##########################################################################

    def plot_dio_snapshot(self, bits=range(32)):
        zibase.plot_timing_diagram(self.getv('raw/dios/0/data'), bits, 64)

    def plot_awg_codewords(self, awg_nr=0, range=None):
        ts = []
        cws = []
        for d in self.getv('awgs/{}/dio/data'.format(awg_nr)):
            cws.append(d & 0x3ff)
            ts.append((d >> 10) & 0x3fffff)
        zibase.plot_codeword_diagram(ts, cws, range)

    ##########################################################################
    # 'private' functions, internal to the driver
    ##########################################################################

    # FIXME: no longer used
    def _set_dio_delay(self, delay):
        """
        The function sets the DIO delay for the instrument. The valid delay range is
        0 to 15. The delays are applied to all bits of the DIO bus.
        """
        if delay < 0:
            log.warning('{}: Clamping delay to 0'.format(self.devname))
        if delay > 15:
            log.warning('{}: Clamping delay to 15'.format(self.devname))
            delay = 15

        self.seti('raw/dios/0/delays/*/value', delay)
