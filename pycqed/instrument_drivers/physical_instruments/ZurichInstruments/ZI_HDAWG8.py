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

20200214 WJV
- removed unused parameter repetitions from _find_valid_delays()
- also removed parameter repetitions from calibrate_CC_dio_protocol()
- split off calibrate_dio_protocol() from calibrate_CC_dio_protocol() to allow standalone use

20200217 WJV
- moved DIO calibration helpers to their respective drivers
- we now implement new interface CalInterface

"""

import time
import logging
import json
import numpy as np
import re
from typing import Tuple, List, Union

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG_core as zicore
import pycqed.instrument_drivers.library.DIO as DIO

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

class ZI_HDAWG8(zicore.ZI_HDAWG_core, DIO.CalInterface):

    def __init__(self,
                 name: str,
                 device: str,
                 interface: str = '1GbE',
                 server: str = 'localhost',
                 port = 8004,
                 num_codewords: int = 64,
                 **kw) -> None:
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
            'system_clocks_referenceclock_freq',
            'cfg_sideband_mode',
            'cfg_codeword_protocol'}
        for i in range(4):
            self._snapshot_whitelist.update({
                'awgs_{}_enable'.format(i),
                'awgs_{}_outputs_0_amplitude'.format(i),
                'awgs_{}_outputs_1_amplitude'.format(i)})

        for i in range(8):
            self._snapshot_whitelist.update({
                'sigouts_{}_direct'.format(i),
                'sigouts_{}_offset'.format(i),
                'sigouts_{}_on'.format(i),
                'sigouts_{}_range'.format(i)})

        self._params_to_exclude = set(self.parameters.keys()) - self._snapshot_whitelist

        t1 = time.time()
        log.info(f'{self.devname}: Initialized ZI_HDAWG in {t1 - t0}s')

    def _gen_set_awgs_outputs_amplitude(self, awg, ch):
        """
        Create a function for mapping setting awgs_N_outputs_M_amplitude to the new nodes.
        """
        def _set_awgs_outputs_amplitude(value):
            self.set(f'awgs_{awg}_outputs_{ch}_gains_{ch}', value)
        return _set_awgs_outputs_amplitude

    def _gen_get_awgs_outputs_amplitude(self, awg, ch):
        """
        Create a function for mapping getting awgs_N_outputs_M_amplitude to the new nodes.
        """
        def _get_awgs_outputs_amplitude():
            return self.get(f'awgs_{awg}_outputs_{ch}_gains_{ch}')
        return _get_awgs_outputs_amplitude

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
        awgs_[0-3]_outputs_[0-1]_amplitude - dummy node mapping to the awgs/[0-3]/outputs/[0-1]/gains/[0-1] node
            to maintain compatibility
        """
        super()._add_extra_parameters()

        self.add_parameter(
            'cfg_sideband_mode', initial_value='static',
            vals=validators.Enum('static', 'real-time'), docstring=(
                'Used in the _codeword_table_preamble method to determine what'
                'format to use for the setWaveDIO command in the AWG sequence.'),
            parameter_class=ManualParameter)

        self.add_parameter(
            'cfg_codeword_protocol', initial_value='identical',
            vals=validators.Enum('identical', 'microwave', 'novsm_microwave', 'flux'), docstring=(
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

        for i in range(4):
            for ch in range(2):
                self.add_parameter(f'awgs_{i}_outputs_{ch}_amplitude',
                            set_cmd=self._gen_set_awgs_outputs_amplitude(i, ch),
                            get_cmd=self._gen_get_awgs_outputs_amplitude(i, ch),
                            unit='FS',
                            label=f'AWG {i} output {ch} amplitude (legacy, deprecated)',
                            docstring=f'Configures the amplitude in full scale units of AWG {i} output {ch} (zero-indexed). Note: this parameter is deprecated, use awgs_{ch}_outputs_{ch}_gains_{ch} instead',
                            vals=validators.Numbers())

    # FIXME: why the override, does not seem necessary now QCoDeS PRs 1161/1163 have been merged
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


    def upload_commandtable(self, commandtable: Union[str, dict], awg_nr: int):
        """
        Uploads commandtable that is used to call phase increment instructions via DIO codewords,
        needed for single qubit phase corrections.

        commandtable (Union[str, dict]):
            The json string to be uploaded as the commandtable. 
            Will be converted to string if given as dict.
        """
        if isinstance(commandtable, dict):
            commandtable = json.dumps(commandtable, sort_keys=True, indent=2)

        # validate json (without schema)
        try:
            json.loads(commandtable)
        except json.decoder.JSONDecodeError:
            log.error(f"Invalid JSON in commandtable: {commandtable}")
        else:
            log.info("Commandtable has valid json format")
            # upload commandtable
            self.stop()
            self.setv(f"awgs/{awg_nr}/commandtable/data", commandtable)
            self.start()

        return commandtable, self.geti(f"awgs/{awg_nr}/commandtable/status")

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

            # FIXME: Unfortunately, 'static' here also refers to configuration required for flux HDAWG8
            if self.cfg_sideband_mode() == 'static' or self.cfg_codeword_protocol() == 'flux':
                # program += 'assignWaveIndex(\"{}\", \"{}\", {});\n'.format(
                #     csvname_l, csvname_r, dio_cw)
                program += 'setWaveDIO({}, \"{}\", \"{}\");\n'.format(
                    dio_cw, csvname_l, csvname_r)
            elif self.cfg_sideband_mode() == 'real-time' and self.cfg_codeword_protocol() == 'novsm_microwave':
                # program += 'setWaveDIO({}, 1, 2, \"{}\", 1, 2, \"{}\");\n'.format(
                #     dio_cw, csvname_l, csvname_r)
                program += 'assignWaveIndex(1, 2, \"{}\", 1, 2, \"{}\", {});\n'.format(
                    csvname_l, csvname_r, dio_cw)
            else:
                raise Exception("Unknown modulation type '{}' and codeword protocol '{}'" \
                                    .format(self.cfg_sideband_mode(), self.cfg_codeword_protocol()))

        if self.cfg_sideband_mode() == 'real-time':
            program += '// Initialize the phase of the oscillators\n'
            program += 'executeTableEntry(1023);\n'
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
        self.seti('dios/0/mode', 2)

        # Configure the DIO interface and the waveforms
        for awg_nr in range(int(self._num_channels()//2)):
            # Set the bit index of the valid bit
            self.set('awgs_{}_dio_valid_index'.format(awg_nr), 31)

            # Set polarity of the valid bit:
            # 2: 'high', 1: 'low', 0: 'no valid needed'
            self.set('awgs_{}_dio_valid_polarity'.format(awg_nr), 2)

            # Set the bit index of the strobe signal (TOGGLE_DS):
            self.set('awgs_{}_dio_strobe_index'.format(awg_nr), 30)

            # Configure edge triggering for the strobe/toggle bit signal:
            # 0: no edges 1: rising edge, 2: falling edge or 3: both edges
            self.set('awgs_{}_dio_strobe_slope'.format(awg_nr), 0)

            # No special requirements regarding waveforms by default
            self._clear_readonly_waveforms(awg_nr)

            # Set DIO shift and mask
            channels = [2*awg_nr, 2*awg_nr+1]
            shift,mask = DIO.get_shift_and_mask(self.cfg_codeword_protocol(), channels)
            self.set(f'awgs_{awg_nr}_dio_mask_value', mask)
            self.set(f'awgs_{awg_nr}_dio_mask_shift', shift)

            # FIXME: check _num_codewords against mode
            # FIXME: derive amp vs direct mode from dio_mode_list

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

    def _ensure_activity(self, awg_nr, mask_value=None, timeout=5):
        """
        Record DIO data and test whether there is activity on the bits activated in the DIO protocol for the given AWG.
        """
        log.debug(f"Testing DIO activity for AWG {awg_nr}")

        vld_mask     = 1 << self.geti('awgs/{}/dio/valid/index'.format(awg_nr))
        vld_polarity = self.geti('awgs/{}/dio/valid/polarity'.format(awg_nr))
        strb_mask    = (1 << self.geti('awgs/{}/dio/strobe/index'.format(awg_nr)))
        strb_slope   = self.geti('awgs/{}/dio/strobe/slope'.format(awg_nr))

        if mask_value is None:
            mask_value = self.geti('awgs/{}/dio/mask/value'.format(awg_nr))

        cw_mask      = mask_value #<< self.geti('awgs/{}/dio/mask/shift'.format(awg_nr))

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
                log.warning(f"Did not see all codeword bits toggle! Got 0x{cw_activity:08x}, expected 0x{cw_mask:08x}.")
                valid = False

            if vld_polarity != 0 and vld_activity != vld_mask:
                log.warning("Did not see valid bit toggle!")
                valid = False

            if strb_slope != 0 and strb_activity != strb_mask:
                log.warning("Did not see strobe bit toggle!")
                valid = False

            if valid:
                return True

        return False

    def _find_valid_delays(self, awgs_and_sequences):
        """Finds valid DIO delay settings for a given AWG by testing all allowed delay settings for timing violations on the
        configured bits. In addition, it compares the recorded DIO codewords to an expected sequence to make sure that no
        codewords are sampled incorrectly."""
        log.debug("  Finding valid delays")
        valid_delays= []
        for delay in range(16):
            log.debug(f'   Testing delay {delay}')
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
                                log.warning(f"Codeword {n} with value {cw} not in expected sequence {sequence}!")
                                log.debug(f"Detected codeword sequence: {cws}")
                                valid_sequence = False
                                break
                            else:
                                index = sequence.index(cw)
                        else:
                            last_index = index
                            index = (index + 1) % len(sequence)
                            if cw != sequence[index]:
                                log.warning("Codeword {} with value {} not expected to follow codeword {} in expected sequence {}!".format(n, cw, sequence[last_index], sequence))
                                log.info(f"Detected codeword sequence: {cws}")
                                valid_sequence = False
                                break
                else:
                    valid_sequence = False

            if valid_sequence:
                valid_delays.append(delay)

        return set(valid_delays)

    ##########################################################################
    # overrides for CalInterface interface
    ##########################################################################

    # NB: based on UHFQuantumController.py::_prepare_HDAWG8_dio_calibration
    # FIXME: also requires fiddling with DIO data direction
    # FIXME: is this guaranteed to be synchronous to 10 MHz?
    def output_dio_calibration_data(self, dio_mode: str, port: int=0) -> Tuple[int, List]:
        """
        Configures an HDAWG with a default program that generates data suitable for DIO calibration.
        Also starts the HDAWG.
        """
        program = '''
        var A = 0xffff0000;
        var B = 0x00000000;

        while (1) {
            setDIO(A);
            wait(2);
            setDIO(B);
            wait(2);
        }
        '''
        self.configure_awg_from_string(0, program)
        self.seti('awgs/0/enable', 1)

        dio_mask = 0x7fff0000
        expected_sequence = []
        return dio_mask,expected_sequence

    def calibrate_dio_protocol(self, dio_mask: int, expected_sequence: List, port: int=0):
        # FIXME: UHF driver does not use expected_sequence, why the difference
        self.assure_ext_clock()
        self.upload_codeword_program()

        for awg, sequence in expected_sequence:
            if not self._ensure_activity(awg, mask_value=dio_mask):
                raise ziDIOActivityError('No or insufficient activity found on the DIO bits associated with AWG {}'.format(awg))

        valid_delays = self._find_valid_delays(expected_sequence)
        if len(valid_delays) == 0:
            raise ziDIOCalibrationError('DIO calibration failed! No valid delays found')

        # Find center of first valid region
        subseq = [[]]
        for e in valid_delays:
            if not subseq[-1] or subseq[-1][-1] == e - 1:
                subseq[-1].append(e)
            else:
                subseq.append([e])

        subseq = max(subseq, key=len)
        delay = len(subseq)//2 + subseq[0]

        # subseq = [[]]
        # for e in valid_delays:
        #     if not subseq[-1] or subseq[-1][-1] == e - 1:
        #         subseq[-1].append(e)
        #     else:
        #         subseq.append([e])

        # subseq = max(subseq, key=len)
        # delay = len(subseq)//2 + subseq[0]

        # Print information
        log.info(f"Valid delays are {valid_delays}")
        log.info(f"Setting delay to {delay}")

        # And configure the delays
        self._set_dio_calibration_delay(delay)

        # If successful clear all errors and return True
        self.clear_errors()  # FIXME: also clears errors not relating to DIO

    ##########################################################################
    # DIO calibration functions for *CC*
    ##########################################################################

    def calibrate_CC_dio_protocol(self, CC, verbose=False) -> None:
        raise DeprecationWarning("calibrate_CC_dio_protocol is deprecated, use instrument_drivers.library.DIO.calibrate")
