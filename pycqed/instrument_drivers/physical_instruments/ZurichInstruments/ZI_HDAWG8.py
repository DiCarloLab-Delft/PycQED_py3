"""
Notes:
- this is the application dependent part of the ZI HDAWG driver: it handles all codeword related
  functionality including DIO configuration. Generic parts reside in ZI_HDAWG_core.py

To do:
- replace print() by logging
-

Changelog:

20190206 WJV
- started this Changelog
- manually checked against diverted branch HDAWG_V2_Verification:
    - the following functions match:
        - _find_valid_delays
        - _set_dio_delay
        - ensure_symmetric_strobe
        - calibrate_dio_protocol(self, awgs_and_sequences, verbose=False)
        - _get_edges
        - _is_dio_strb_symmetric
        - _analyze_dio_data
    - the following were already commented out here:
        - _check_protocol
        - _print_check_protocol_error_message
        - calibrate_dio
        - calibrate_dio_protocol(self)
    So we conclude all relevant changes of HDAWG_V2_Verification made it here,
    albeit in a different order that clutters the diff.
- removed the above mentioned 4 functions that were commented out
- added comments, organized code into sections
- made some functions 'private'
- NB: none of the above should change anything for real
- moved enabling of outputs to end in configure_codeword_protocol

20190207 WJV
- added assure_ext_clock()

20190212 WJV
- separated off application independent stuff into ZI_HDAWG_core class, this
  file will keep application dependent stuff
- addressed many warnings identified by PyCharm

20190214 WJV
- added activate_new_dio_triggering()
- moved in _add_extra_parameters() and _add_codeword_parameters()
- moved out _set_dio_delay()

20190417 WJV
- merged branch 'develop' into 'feature/cc'

20190429 WJV
- merged branch 'QCC_testing' into 'feature/cc', changes:
    upload_waveform_realtime was updated, moved it to ZI_HDAWG_core.py again

20190618 WJV
- merged branch 'develop' into 'feature/cc', changes:
    upload_waveform_realtime was updated, moved it to ZI_HDAWG_core.py again

20190627 WJV
- removed DIO calibration support, which will shortly be replaced

20190709 NCH
- Github PR #578

20191001 WJV
- removed duplicates from __init__
- cleanup
- changed _configure_codeword_protocol() to use table of modes
- split off calibrate_dio_protocol from calibrate_CC_dio_protocol for use with CC
- removed unused parameters cfg_num_codewords and cfg_codeword_protocol from upload_codeword_program()
- removed unused parameter default_dio_timing from _configure_codeword_protocol()

"""

import time
import logging
import numpy as np
import re
import os
import pycqed

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG_core as zicore

from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.helpers import full_class

log = logging.getLogger(__name__)

##########################################################################
# Exceptions
##########################################################################

class ziDIOActivityError(Exception):
    """Exception raised when no activity is found on the DIO bus during calibration."""
    pass

class ziDIOCalibrationError(Exception):
    """Exception raised when DIO calibration fails."""
    pass

##########################################################################
# Class
##########################################################################

class ZI_HDAWG8(zicore.ZI_HDAWG_core):

    def __init__(self,
                 name: str,
                 device: str,
                 interface: str = '1GbE',
                 server: str = 'localhost',
                 port = 8004,
                 num_codewords: int = 32, **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument as seen by the user
            device          (str) the name of the device e.g., "dev8008"
            interface       (str) the name of the interface to use ('1GbE' or 'USB')
            server          (str) the ZI data server
            port            (int) the port to connect to
            num_codewords   (int) the number of codeword-based waveforms to prepare
        """
        t0 = time.time()
        super().__init__(name=name, device=device, interface=interface, server=server, port=port, num_codewords=num_codewords, **kw)
        # Set default waveform length to 20 ns at 2.4 GSa/s
        self._default_waveform_length = 48

        # Holds the DIO calibration delay
        self._dio_calibration_delay = 0

        # show some info
        log.info('{}: DIO interface found in mode {}'
                 .format(self.devname, 'CMOS' if self.get('dios_0_interface') == 0 else 'LVDS')) # NB: mode is persistent across device restarts

        # Ensure snapshot is fairly small for HDAWGs
        self._snapshot_whitelist = {
            'IDN',
            'clockbase',
            'system_clocks_referenceclock_source',
            'system_clocks_referenceclock_status',
            'system_clocks_referenceclock_freq'}
        for i in range(4):
            self._snapshot_whitelist.update({
                'awgs_{}_enable'.format(i),
                'awgs_{}_outputs_0_amplitude'.format(i),
                'awgs_{}_outputs_1_amplitude'.format(i)})

        for i in range(8):
            self._snapshot_whitelist.update({
                'sigouts_{}_direct'.format(i), 'sigouts_{}_offset'.format(i),
                'sigouts_{}_on'.format(i) , 'sigouts_{}_range'.format(i)})

        self._params_to_exclude = set(self.parameters.keys()) - self._snapshot_whitelist

        t1 = time.time()
        log.info(f'{self.devname}: Initialized ZI_HDAWG in {t1 - t0}s')

    def _add_extra_parameters(self):
        """
        We add a few additional custom parameters on top of the ones defined in the device files. These are:
        timeout - A specific timeout value in seconds used for the various timeconsuming operations on the device
          such as waiting for an upload to complete.
        cfg_num_codewords - determines the maximum number of codewords to be supported by the program that will
          be uploaded using the "upload_codeword_program" method.
        cfg_codeword_protocol - determines the specific codeword protocol to use, for example 'microwave' or 'flux'.
          It determines which bits transmitted over the 32-bit DIO interface are used as actual codeword bits.
        awgs_[0-3]_sequencer_program_crc32_hash - CRC-32 hash of the currently uploaded sequencer program to enable
          changes in program to be detected.
        dio_calibration_delay - the delay that is programmed on the DIO lines as part of the DIO calibration
            process in order for the instrument to reliably sample data from the CC. Can be used to detect
            unexpected changes in timing of the entire system. The parameter can also be used to force a specific
            delay to be used on the DIO although that is not generally recommended.
        """
        super()._add_extra_parameters()

        self.add_parameter(
            'cfg_codeword_protocol', initial_value='identical',
            vals=validators.Enum('identical', 'microwave', 'new_novsm_microwave', 'flux'), docstring=(
                'Used in the configure codeword method to determine what DIO'
                ' pins are used in for which AWG numbers.'),
            parameter_class=ManualParameter)

        self.add_parameter('dio_calibration_delay',
                    set_cmd=self._set_dio_calibration_delay,
                    get_cmd=self._get_dio_calibration_delay,
                    unit='',
                    label='DIO Calibration delay',
                    docstring='Configures the internal delay in 300 MHz cycles (3.3 ns) '
                    'to be applied on the DIO interface in order to achieve reliable sampling'
                    ' of the codewords. The valid range is 0 to 15.',
                    vals=validators.Ints())

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update =None,
                      params_to_exclude = None ):
        """
        State of the instrument as a JSON-compatible dict.
        Args:
            update: If True, update the state by querying the
                instrument. If False, just use the latest values in memory.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac)
        Returns:
            dict: base snapshot
        """


        if params_to_exclude is None:
            params_to_exclude = self._params_to_exclude

        snap = {
            "functions": {name: func.snapshot(update=update)
                          for name, func in self.functions.items()},
            "submodules": {name: subm.snapshot(update=update)
                           for name, subm in self.submodules.items()},
            "__class__": full_class(self)
        }

        snap['parameters'] = {}
        for name, param in self.parameters.items():
            if params_to_exclude and name in params_to_exclude:
                pass
            elif params_to_skip_update and name in params_to_skip_update:
                update_par = False
            else:
                update_par = update
                try:
                    snap['parameters'][name] = param.snapshot(update=update_par)
                except:
                    logging.info("Snapshot: Could not update parameter: {}".format(name))
                    snap['parameters'][name] = param.snapshot(update=False)

        for attr in set(self._meta_attrs):
            if hasattr(self, attr):
                snap[attr] = getattr(self, attr)
        return snap

    ##########################################################################
    # 'public' functions: application specific/codeword support
    ##########################################################################

    def upload_codeword_program(self, awgs=np.arange(4)):
        """
        Generates a program that plays the codeword waves for each channel.

        awgs (array): the awg numbers to which to upload the codeword program.
        """
        self._configure_codeword_protocol()

        # Type conversion to ensure lists do not produce weird results
        awgs = np.array(awgs)
        if awgs.shape == ():
            awgs = np.array([awgs])

        for awg_nr in awgs:
            self._awg_program[awg_nr] = '''
while (1) {
    // Wait for a trigger on the DIO interface
    waitDIOTrigger();
    // Play a waveform from the table based on the DIO code-word
    playWaveDIO();
}'''
            self._awg_needs_configuration[awg_nr] = True

    ##########################################################################
    # 'private' functions: application specific/codeword support
    ##########################################################################

    def _get_waveform_table(self, awg_nr: int) -> list:
        """
        Returns the waveform table.

        The waveform table determines the mapping of waveforms to DIO codewords.
        The index of the table corresponds to the DIO codeword.
        The entry is a tuple of waveform names.

        Example:
            ["wave_ch7_cw000", "wave_ch8_cw000",
            "wave_ch7_cw001", "wave_ch8_cw001",
            "wave_ch7_cw002", "wave_ch8_cw002"]

        The waveform table generated depends on the awg_nr and the codeword
        protocol.
        """
        ch = awg_nr*2
        wf_table = []
        if 'flux' in self.cfg_codeword_protocol():
            for cw_r in range(8):
                for cw_l in range(8):
                    wf_table.append((zibase.gen_waveform_name(ch, cw_l),
                                     zibase.gen_waveform_name(ch+1, cw_r)))
        else:
            for dio_cw in range(self._num_codewords):
                wf_table.append((zibase.gen_waveform_name(ch, dio_cw),
                                 zibase.gen_waveform_name(ch+1, dio_cw)))
        return wf_table

    def _codeword_table_preamble(self, awg_nr):
        """
        Defines a snippet of code to use in the beginning of an AWG program in order to define the waveforms.
        The generated code depends on the instrument type. For the HDAWG instruments, we use the seWaveDIO
        function.
        """
        program = ''

        wf_table = self._get_waveform_table(awg_nr=awg_nr)
        for dio_cw, (wf_l, wf_r) in enumerate(wf_table):
            csvname_l = self.devname + '_' + wf_l
            csvname_r = self.devname + '_' + wf_r

            program += 'setWaveDIO({}, \"{}\", \"{}\");\n'.format(
                dio_cw, csvname_l, csvname_r)

        return program

    def _configure_codeword_protocol(self):
        """
        This method configures the AWG-8 codeword protocol.
        The qcodes parameter "cfg_codeword_protocol" defines what protocol is used.

        The final step enables the signal output of each AWG and sets
        it to the right mode.
        """
        # Check overall configuration
        if self.system_awg_channelgrouping() != 0:
            log.warning(f'{self.devname}: Instrument not in 4 x 2 channel mode! Switching...')
            self.system_awg_channelgrouping(0)
            self.sync()

        # Use 50 MHz DIO clocking
        self.seti('raw/dios/0/extclk', 1)

        # Configure the DIO interface and the waveforms
        for awg_nr in range(int(self._num_channels()//2)):
            # Set the bit index of the valid bit
            self.set('awgs_{}_dio_valid_index'.format(awg_nr), 31)

            # Set polarity of the valid bit:
            # 2: 'high', 1: 'low', 0: 'no valid needed'
            self.set('awgs_{}_dio_valid_polarity'.format(awg_nr), 2)

            # Set the bit index of the strobe signal (TOGGLE_DS),
            self.set('awgs_{}_dio_strobe_index'.format(awg_nr), 30)

            # Configure edge triggering for the strobe/toggle bit signal:
            # 0: no edges 1: rising edge, 2: falling edge or 3: both edges
            self.set('awgs_{}_dio_strobe_slope'.format(awg_nr), 0)

            # No special requirements regarding waveforms by default
            self._clear_readonly_waveforms(awg_nr)

            if 1:   # FIXME: new
                num_codewords = int(2 ** np.ceil(np.log2(self._num_codewords)))
                dio_mode_list = {
                    'identical':            { 'mask': 0xFF, 'shift': [0,  0,  0,  0] },
                    'microwave':            { 'mask': 0xFF, 'shift': [0,  0,  16, 16] },    # bits [7:0] and [23:16]
                    'new_novsm_microwave':  { 'mask': 0x7F, 'shift': [0,  7,  16, 23] },    # bits [6:0], [13:7], [22:16] and [29:23]
                    'flux':                 { 'mask': 0x3F, 'shift': [0,  6,  16, 22] },    # FIXME: mask for 2 channels
                }
                # FIXME: define DIO modes centrally in device independent way (lsb, width, channelCount)
                dio_mode = dio_mode_list.get(self.cfg_codeword_protocol())
                if dio_mode is None:
                    raise ValueError("Unsupported value '{}' for parameter cfg_codeword_protocol".format(self.cfg_codeword_protocol))
                mask = dio_mode['mask']
                self.set(f'awgs_{awg_nr}_dio_mask_value', mask)
                shift = dio_mode['shift'][awg_nr]
                self.set(f'awgs_{awg_nr}_dio_mask_shift', shift)
                # FIXME: flux mode sets mask, using 6 bits=2channels
                # FIXME: check _num_codewords against mode
                # FIXME: derive amp vs direct mode from dio_mode_list
            else:
                # the mask determines how many bits will be used in the protocol
                # e.g., mask 3 will mask the bits with bin(3) = 00000011 using
                # only the 2 Least Significant Bits.
                num_codewords = int(2 ** np.ceil(np.log2(self._num_codewords)))
                self.set('awgs_{}_dio_mask_value'.format(awg_nr), num_codewords - 1)

                # set mask and shift for codeword protocol
                # N.B. The shift is applied before the mask
                # The relevant bits can be selected by first shifting them
                # and then masking them.

                if self.cfg_codeword_protocol() == 'identical':
                    # In the identical protocol all bits are used to trigger
                    # the same codewords on all AWG's
                    self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)

                # NEW
                # In the new mw protocol bits [0:7] -> CW0 and bits [23:16] -> CW1
                elif self.cfg_codeword_protocol() == 'microwave':
                    if awg_nr in [0, 1]:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)
                    elif awg_nr in [2, 3]:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 16)

                # NEW
                # In the NO-VSM mw protocol bits [0:6] -> CW0, bits [13, 7] -> CW1,
                # bits [22:16] -> CW2 and bits [29:23] -> CW4
                elif self.cfg_codeword_protocol() == 'new_novsm_microwave':
                    if awg_nr == 0:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)
                    elif awg_nr == 1:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 7)
                    elif awg_nr == 2:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 16)
                    elif awg_nr == 3:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 23)

                # NEW
                # Proper use of flux AWG to allow independent triggering of flux
                # bits[0:2] for awg0_ch0, bits[3:5] for awg0_ch1,
                # bits[6:8] for awg0_ch2, bits[9:11] for awg0_ch3,
                # bits[16:18] for awg0_ch4, bits[19:21] for awg0_ch5,
                # bits[22:24] for awg0_ch6, bits[25:27] for awg0_ch7
                elif self.cfg_codeword_protocol() == 'flux':
                    self.set('awgs_{}_dio_mask_value'.format(awg_nr), 2**6-1)

                    if awg_nr == 0:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)
                    elif awg_nr == 1:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 6)
                    elif awg_nr == 2:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 16)
                    elif awg_nr == 3:
                        self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 22)

        ####################################################
        # Turn on device
        ####################################################
        time.sleep(.05)
        for awg_nr in range(int(self._num_channels()//2)):
            self.set('awgs_{}_enable'.format(awg_nr), 1)

        # Disable all function generators
        for param in [key for key in self.parameters.keys() if
                      re.match(r'sines_\d+_enables_\d+', key)]:
            self.set(param, 0)

        # Set amp or direct mode
        if self.cfg_codeword_protocol() == 'flux':
            # when doing flux pulses, set everything to amp mode
            for ch in range(8):
                self.set('sigouts_{}_direct'.format(ch), 0)
                self.set('sigouts_{}_range'.format(ch), 5)
        else:
            # Switch all outputs into direct mode when not using flux pulses
            for ch in range(8):
                self.set('sigouts_{}_direct'.format(ch), 1)
                self.set('sigouts_{}_range'.format(ch), .8)

        # Turn on all outputs
        for param in [key for key in self.parameters.keys() if re.match(r'sigouts_\d+_on', key)]:
            self.set(param, 1)

    def _debug_report_dio(self):
        # FIXME: only DIO 0 for now
        log.info('DIO bits with timing errors:  0x%08X' % self.geti('awgs/0/dio/error/timing'))
        log.info('DIO bits detected high:       0x%08X' % self.geti('awgs/0/dio/highbits'))
        log.info('DIO bits detected low:        0x%08X' % self.geti('awgs/0/dio/lowbits'))
        # AWGS/0/DIO/ERROR/WIDTH
        # AWGS/0/DIO/DATA

    ##########################################################################
    # 'private' functions: parameter support for DIO calibration delay
    ##########################################################################

    def _set_dio_calibration_delay(self, value):
        # Sanity check the value
        if value < 0 or value > 15:
            raise zibase.ziValueError('Trying to set DIO calibration delay to invalid value! Expected value in range 0 to 15. Got {}.'.format(value))

        log.info('Setting DIO calibration delay to {}'.format(value))
        # Store the value
        self._dio_calibration_delay = value

        # And configure the delays
        self.setd('raw/dios/0/delays/*', self._dio_calibration_delay)

    def _get_dio_calibration_delay(self):
        return self._dio_calibration_delay

    ##########################################################################
    # 'private' functions: DIO calibration
    ##########################################################################

    def _get_awg_dio_data(self, awg):
        data = self.getv('awgs/' + str(awg) + '/dio/data')
        ts = len(data)*[0]
        cw = len(data)*[0]
        for n, d in enumerate(data):
            ts[n] = d >> 10
            cw[n] = (d & ((1 << 10)-1))
        return (ts, cw)

    def _ensure_activity(self, awg_nr, mask_value=None, timeout=5, verbose=False):
        """
        Record DIO data and test whether there is activity on the bits activated in the DIO protocol for the given AWG.
        """
        if verbose: print("Testing DIO activity for AWG {}".format(awg_nr))

        vld_mask     = 1 << self.geti('awgs/{}/dio/valid/index'.format(awg_nr))
        vld_polarity = self.geti('awgs/{}/dio/valid/polarity'.format(awg_nr))
        strb_mask    = (1 << self.geti('awgs/{}/dio/strobe/index'.format(awg_nr)))
        strb_slope   = self.geti('awgs/{}/dio/strobe/slope'.format(awg_nr))

        if mask_value is None:
            mask_value = self.geti('awgs/{}/dio/mask/value'.format(awg_nr))

        cw_mask      = mask_value << self.geti('awgs/{}/dio/mask/shift'.format(awg_nr))

        for i in range(timeout):
            valid = True

            data = self.getv('raw/dios/0/data')
            if data is None:
                raise zibase.ziValueError('Failed to get DIO snapshot!')

            vld_activity = 0
            strb_activity = 0
            cw_activity = 0
            for d in data:
                cw_activity |= (d & cw_mask)
                vld_activity |= (d & vld_mask)
                strb_activity |= (d & strb_mask)

            if cw_activity != cw_mask:
                print("Did not see all codeword bits toggle! Got 0x{:08x}, expected 0x{:08x}.".format(cw_activity, cw_mask))
                valid = False

            if vld_polarity != 0 and vld_activity != vld_mask:
                print("Did not see valid bit toggle!")
                valid = False

            if strb_slope != 0 and strb_activity != strb_mask:
                print("Did not see valid bit toggle!")
                valid = False

            if valid:
                return True

        return False

    def _find_valid_delays(self, awgs_and_sequences, repetitions=1, verbose=False):
        """Finds valid DIO delay settings for a given AWG by testing all allowed delay settings for timing violations on the
        configured bits. In addition, it compares the recorded DIO codewords to an expected sequence to make sure that no
        codewords are sampled incorrectly."""
        if verbose: print("  Finding valid delays")
        valid_delays= []
        for delay in range(16):
            if verbose: print('   Testing delay {}'.format(delay))
            self.setd('raw/dios/0/delays/*/value', delay)
            time.sleep(1)
            valid_sequence = True
            for awg, sequence in awgs_and_sequences:
                if self.geti('awgs/' + str(awg) + '/dio/error/timing') == 0:
                    ts, cws = self._get_awg_dio_data(awg)
                    index = None
                    last_index = None
                    for n, cw in enumerate(cws):
                        if n == 0:
                            if cw not in sequence:
                                if verbose: print("WARNING: Codeword {} with value {} not in expected sequence {}!".format(n, cw, sequence))
                                if verbose: print("Detected codeword sequence: {}".format(cws))
                                valid_sequence = False
                                break
                            else:
                                index = sequence.index(cw)
                        else:
                            last_index = index
                            index = (index + 1) % len(sequence)
                            if cw != sequence[index]:
                                if verbose: print("WARNING: Codeword {} with value {} not expected to follow codeword {} in expected sequence {}!".format(n, cw, sequence[last_index], sequence))
                                if verbose: print("Detected codeword sequence: {}".format(cws))
                                valid_sequence = False
                                break
                else:
                    valid_sequence = False

            if valid_sequence:
                valid_delays.append(delay)

        return set(valid_delays)

    def _prepare_QCC_dio_calibration(self, QCC, verbose=False):
        """
        Prepares the appropriate program to calibrate DIO and returns
        expected sequence.
        N.B. only works for microwave on DIO4 and for Flux on DIO3
            (TODO add support for microwave on DIO5)
        """
        log.info('Calibrating DIO delays')
        if verbose: print("Calibrating DIO delays")

        cs_filepath = os.path.join(pycqed.__path__[0],
            'measurement',
            'openql_experiments',
            's17', 'cs.txt')

        opc_filepath = os.path.join(pycqed.__path__[0],
            'measurement',
            'openql_experiments',
            's17', 'qisa_opcodes.qmap')

        # Configure QCC
        QCC.control_store(cs_filepath)
        QCC.qisa_opcode(opc_filepath)

        if self.cfg_codeword_protocol() == 'flux':
            test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..',
                'examples','QCC_example',
                'qisa_test_assembly','flux_calibration.qisa'))

            sequence_length = 8
            staircase_sequence = np.arange(1, sequence_length)

            # expected sequence should be ([9, 18, 27, 36, 45, 54, 63])
            expected_sequence = [(0, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (1, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (2, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (3, list(staircase_sequence+ (staircase_sequence << 3)))]

        elif self.cfg_codeword_protocol() == 'microwave':

            test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..',
                'examples','QCC_example',
                'qisa_test_assembly','withvsm_calibration.qisa'))

            sequence_length = 32
            staircase_sequence = range(1, sequence_length)
            expected_sequence =  [(0, list(staircase_sequence)), \
                                 (1, list(staircase_sequence)), \
                                 (2, list(reversed(staircase_sequence))), \
                                 (3, list(reversed(staircase_sequence)))]


        elif self.cfg_codeword_protocol() == 'new_novsm_microwave':

            test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..','examples','QCC_example',
                'qisa_test_assembly','novsm_calibration.qisa'))

            sequence_length = 32
            staircase_sequence = range(1, sequence_length)
            expected_sequence = [(0, list(staircase_sequence)), \
                                 (1, list(reversed(staircase_sequence))), \
                                 (2, list(staircase_sequence)), \
                                 (3, list(reversed(staircase_sequence))) ]

        else:
            zibase.ziConfigurationError("Can only calibrate DIO protocol for 'flux' or 'microwave' mode!")

        # Start the QCC with the program configured above
        QCC.eqasm_program(test_fp)
        QCC.start()
        return expected_sequence

    def _prepare_CC_dio_calibration(self, CC, verbose=False):
        """
        Prepares the appropriate program to calibrate DIO and returns
        expected sequence.
        """
        log.info('Calibrating DIO delays')
        if verbose: print("Calibrating DIO delays")

        if self.cfg_codeword_protocol() == 'flux':
            test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..',
                'examples','CC_examples',
                'flux_calibration.vq1asm'))

            sequence_length = 8
            staircase_sequence = np.arange(1, sequence_length)

            # expected sequence should be ([9, 18, 27, 36, 45, 54, 63])
            expected_sequence = [(0, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (1, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (2, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (3, list(staircase_sequence+ (staircase_sequence << 3)))]

        elif self.cfg_codeword_protocol() == 'microwave':

            test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                                      '..', 'examples','CC_examples',
                                      'old_hdawg_calibration.vq1asm'))

            sequence_length = 32
            staircase_sequence = range(0, sequence_length)
            expected_sequence = [(0, list(staircase_sequence)), \
                                 (1, list(staircase_sequence)), \
                                 (2, list(reversed(staircase_sequence))), \
                                 (3, list(reversed(staircase_sequence)))]

        elif self.cfg_codeword_protocol() == 'new_novsm_microwave':
            raise NotImplementedError

        else:
            raise zibase.ziConfigurationError("Can only calibrate DIO protocol for 'flux' or 'microwave' mode!")

        # Start the QCC with the program configured above
        CC.eqasm_program(test_fp)
        CC.start()
        return expected_sequence

    def _prepare_CCL_dio_calibration(self, CCL, verbose=False):
        """
        Prepares the appropriate program to calibrate DIO and returns
        expected sequence.
        N.B. only works for microwave on DIO4 and for Flux on DIO3
            (TODO add support for microwave on DIO5)
        """
        log.info('Calibrating DIO delays')
        if verbose: print("Calibrating DIO delays")

        cs_filepath = os.path.join(pycqed.__path__[0],
            'measurement',
            'openql_experiments',
            'output', 'cs.txt')

        opc_filepath = os.path.join(pycqed.__path__[0],
            'measurement',
            'openql_experiments',
            'output', 'qisa_opcodes.qmap')

        # Configure CCL
        CCL.control_store(cs_filepath)
        CCL.qisa_opcode(opc_filepath)

        if self.cfg_codeword_protocol() == 'flux':
            test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..',
                'examples','CCLight_example',
                'qisa_test_assembly','calibration_cws_flux.qisa'))

            sequence_length = 8
            staircase_sequence = np.arange(1, sequence_length)
            expected_sequence = [(0, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (1, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (2, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (3, list(staircase_sequence))]
        elif self.cfg_codeword_protocol() == 'microwave':
            test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..','examples','CCLight_example',
                'qisa_test_assembly','calibration_cws_mw.qisa'))

            sequence_length = 32
            staircase_sequence = np.arange(1, sequence_length)
            expected_sequence = [(0, list(reversed(staircase_sequence))), \
                                 (1, list(reversed(staircase_sequence))), \
                                 (2, list(reversed(staircase_sequence))), \
                                 (3, list(reversed(staircase_sequence)))]

        else:
            zibase.ziConfigurationError("Can only calibrate DIO protocol for 'flux' or 'microwave' mode!")

        # Start the CCL with the program configured above
        CCL.eqasm_program(test_fp)
        CCL.start()
        return expected_sequence


    def calibrate_CC_dio_protocol(self, CC, verbose=False, repetitions=1):
        """
        Calibrates the DIO communication between CC and HDAWG.
        Arguments:
            CC (instr) : an instance of a CCL or QCC
            verbose (bool): if True prints to stdout
        """
        idn_str = CC.IDN()
        if 'model' in idn_str.keys():
            model_key = 'model'
        elif 'Model' in idn_str.keys():
            model_key = 'Model'

        CC_model = idn_str[model_key]
        if 'QCC' in CC_model:
            expected_sequence = self._prepare_QCC_dio_calibration(
                QCC=CC, verbose=verbose)
        elif 'CCL' in CC_model:
            expected_sequence = self._prepare_CCL_dio_calibration(
                CCL=CC, verbose=verbose)
        elif 'cc' in CC_model:
            expected_sequence = self._prepare_CC_dio_calibration(
                CC=CC, verbose=verbose)
        else:
            raise ValueError('CC model ({}) not recognized.'.format(CC_model))

        # Make sure the configuration is up-to-date
        self.assure_ext_clock()
        self.upload_codeword_program()

        for awg, sequence in expected_sequence:
            if not self._ensure_activity(awg, mask_value=np.bitwise_or.reduce(sequence), verbose=verbose):
                raise ziDIOActivityError('No or insufficient activity found on the DIO bits associated with AWG {}'.format(awg))

        valid_delays = self._find_valid_delays(expected_sequence, repetitions, verbose=verbose)
        if len(valid_delays) == 0:
            raise ziDIOCalibrationError('DIO calibration failed! No valid delays found')

        min_valid_delay = min(valid_delays)

        # Print information
        if verbose: print("  Valid delays are {}".format(valid_delays))
        if verbose: print("  Setting delay to {}".format(min_valid_delay))

        # And configure the delays
        self._set_dio_calibration_delay(min_valid_delay)

        # If successful clear all errors and return True
        self.clear_errors()
        return True
