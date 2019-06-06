"""
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


"""

import time
import logging
import numpy as np

from .ZI_HDAWG_core import ZI_HDAWG_core
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

class ZI_HDAWG8(ZI_HDAWG_core):

    def __init__(self, name: str,
                 device: str,
                 server: str = 'localhost', port = 8004,
                 num_codewords: int = 32, **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument as seen by the user
            device          (str) the name of the device e.g., "dev8008"
            server          (str) the ZI data server
            port            (int) the port to connect to
            FIXME: comment incomplete
        """

        t0 = time.time()
        super().__init__(name, device, server, port, num_codewords, **kw)
        self._add_extra_parameters()
        self._add_codeword_parameters()
        self.connect_message(begin_time=t0)

    ##########################################################################
    # 'public' functions: application specific/codeword support
    ##########################################################################

    def initialze_all_codewords_to_zeros(self):  # FIXME: typo, but used in some Notebooks
        """
        Generates all zeros waveforms for all codewords
        """
        t0 = time.time()
        wf = np.zeros(32)
        waveform_params = [value for key, value in self.parameters.items()
                           if 'wave_ch' in key.lower()]
        for par in waveform_params:
            par(wf)
        t1 = time.time()
        print('Set all zeros waveforms in {:.1f} s'.format(t1-t0))

    def upload_codeword_program(self, awgs=np.arange(4)):
        """
        Generates a program that plays the codeword waves for each channel.

        awgs (array): the awg numbers to which to upload the codeword program.
                    By default uploads to all channels but can be specific to
                    speed up the process.
        """
        # Type conversion to ensure lists do not produce weird results
        awgs = np.array(awgs)
        # because awg_channels come in pairs and are numbered from 1-8 in API
        awg_channels = awgs*2+1

        for awg_nr in awgs:
            # disable all AWG channels
            self.set('awgs_{}_enable'.format(int(awg_nr)), 0)

        codeword_mode_snippet = (
            'while (1) { \n '
            '\t// Wait for a trigger on the DIO interface\n'
            '\twaitDIOTrigger();\n'
            '\t// Play a waveform from the table based on the DIO code-word\n'
            '\tplayWaveDIO(); \n'
            '}')
        if self.cfg_codeword_protocol() != 'flux':
            # FIXME: this is a catchall
            for ch in awg_channels:
                waveform_table = '// Define the waveform table\n'
                for cw in range(self.cfg_num_codewords()):
                    wf0_name = '{}_wave_ch{}_cw{:03}'.format(
                        self._devname, ch, cw)
                    wf1_name = '{}_wave_ch{}_cw{:03}'.format(
                        self._devname, ch+1, cw)
                    waveform_table += 'setWaveDIO({}, "{}", "{}");\n'.format(
                        cw, wf0_name, wf1_name)
                program = waveform_table + codeword_mode_snippet
                # N.B. awg_nr in goes from 0 to 3 in API while in LabOne
                # it is 1 to 4
                awg_nr = ch//2  # channels are coupled in pairs of 2
                self.configure_awg_from_string(awg_nr=int(awg_nr),
                                               program_string=program,
                                               timeout=self.timeout())
        else:  # if protocol is flux
            for ch in awg_channels:
                waveform_table = '//Flux mode\n// Define the waveform table\n'
                mask_0 = 0b000111  # AWGx_ch0 uses lower bits for CW
                mask_1 = 0b111000  # AWGx_ch1 uses higher bits for CW

                # for cw in range(2**6):
                for cw in range(8):
                    cw0 = cw & mask_0
                    cw1 = (cw & mask_1) >> 3
                    if 1:
                        # FIXME: this is a hack because not all AWG8 channels support
                        # amp mode. It forces all AWGs of a pair to behave identical.
                        cw1 = cw0
                        # FIXME: the above is no longer true
                        logging.warning('applied outdated flux channel duplication hack')
                    # if both wfs are triggered play both
                    if (cw0 != 0) and (cw1 != 0):
                        # if both waveforms exist, upload
                        wf0_cmd = '"{}_wave_ch{}_cw{:03}"'.format(
                            self._devname, ch, cw0)
                        wf1_cmd = '"{}_wave_ch{}_cw{:03}"'.format(
                            self._devname, ch+1, cw1)

                    # if single wf is triggered fill the other with zeros
                    elif (cw0 == 0) and (cw1 != 0):
                        wf0_cmd = 'zeros({})'.format(len(self.get(
                            'wave_ch{}_cw{:03}'.format(ch, cw1))))
                        wf1_cmd = '"{}_wave_ch{}_cw{:03}"'.format(
                            self._devname, ch+1, cw1)

                    elif (cw0 != 0) and (cw1 == 0):
                        wf0_cmd = '"{}_wave_ch{}_cw{:03}"'.format(
                            self._devname, ch, cw0)
                        wf1_cmd = 'zeros({})'.format(len(self.get(
                            'wave_ch{}_cw{:03}'.format(ch, cw0))))
                    # if no wfs are triggered play only zeros
                    else:
                        wf0_cmd = 'zeros({})'.format(928) # this length is to account for #109
                        wf1_cmd = 'zeros({})'.format(928) # this length is to account for #109

                    waveform_table += 'setWaveDIO({}, {}, {});\n'.format(
                        cw, wf0_cmd, wf1_cmd)
                program = waveform_table + codeword_mode_snippet

                # N.B. awg_nr in goes from 0 to 3 in API while in LabOne it
                # is 1 to 4
                awg_nr = ch//2  # channels are coupled in pairs of 2
                self.configure_awg_from_string(awg_nr=int(awg_nr),
                                               program_string=program,
                                               timeout=self.timeout())
        self.configure_codeword_protocol()

    # FIXME: should probably be private as it works in tandem with upload_codeword_program
    def configure_codeword_protocol(self, default_dio_timing: bool=False):
        """
        This method configures the AWG-8 codeword protocol.
        The final step enables the signal output of each AWG and sets
        it to the right mode.

        The qcodes parameter "cfg_codeword_protocol" defines what protocol is used.
        There are three options:
            identical : all AWGs have the same configuration
            microwave : AWGs 0 and 1 share bits
            flux      : Each AWG pair is responsible for 2 flux channels.
                        this also affects the "codeword_program" and
                        setting "wave_chX_cwXXX" parameters.

        """

        # Configure the DIO interface
        for awg_nr in range(int(self._num_channels/2)):
            # Set the bit index of the valid bit
            self.set('awgs_{}_dio_valid_index'.format(awg_nr), 31)

            # Set polarity of the valid bit:
            # 2: 'high', 1: 'low', 0: 'no valid needed'
            self.set('awgs_{}_dio_valid_polarity'.format(awg_nr), 2)

            # Set the bit index of the strobe signal (TOGGLE_DS),
            self.set('awgs_{}_dio_strobe_index'.format(awg_nr), 30)

            # Configure edge triggering for the strobe/toggle bit signal:
            # 1: rising edge, 2: falling edge or 3: both edges
            self.set('awgs_{}_dio_strobe_slope'.format(awg_nr), 3)

            # the mask determines how many bits will be used in the protocol
            # e.g., mask 3 will mask the bits with bin(3) = 00000011 using
            # only the 2 Least Significant Bits.
            # N.B. cfg_num_codewords must be a power of 2
            self.set('awgs_{}_dio_mask_value'.format(awg_nr),
                     self.cfg_num_codewords()-1)

            if self.cfg_codeword_protocol() == 'identical':
                # In the identical protocol all bits are used to trigger
                # the same codewords on all AWG's

                # N.B. The shift is applied before the mask
                # The relevant bits can be selected by first shifting them
                # and then masking them.
                self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)
            elif self.cfg_codeword_protocol() == 'microwave':
                # In the mw protocol bits [0:7] -> CW0 and bits [(8+1):15] -> CW1
                # N.B. DIO bit 8 (first of 2nd byte)  not connected in AWG8!
                if awg_nr in [0, 1]:
                    self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)
                elif awg_nr in [2, 3]:
                    self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 9)    # FIXME: this is no longer true for HDAWG V2
            elif self.cfg_codeword_protocol() == 'flux':
                # bits[0:3] for awg0_ch0, bits[4:6] for awg0_ch1 etc.
                # self.set('awgs_{}_dio_mask_value'.format(awg_nr), 2**6-1)
                # self.set('awgs_{}_dio_mask_shift'.format(awg_nr), awg_nr*6)

                # FIXME: this is a protocol that does identical flux pulses
                # on each channel.
                self.set('awgs_{}_dio_mask_value'.format(awg_nr), 2**3-1)
                # self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 3)
                self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 0)
            else:
                logging.error('unknown value for cfg_codeword_protocol')
                # FIXME: exception?

        # Disable all function generators
        self._dev.daq.setInt('/' + self._dev.device +
                             '/sigouts/*/enables/*', 0)

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

        # Enable AWGs
        time.sleep(.05)  # FIXME: why?
        self._dev.daq.setInt('/' + self._dev.device +
                             '/awgs/*/enable', 1)

        # Turn on all outputs
        self._dev.daq.setInt('/' + self._dev.device + '/sigouts/*/on', 1)

    def calibrate_dio_protocol(self, awgs_and_sequences, verbose=False):
        if verbose:
            print("INFO   : Calibrating DIO delays")

        if not self._ensure_symmetric_strobe(verbose):
            if verbose:
                print("ERROR  : Strobe is not symmetric!")
            return False
        else:
            if verbose:
                print("INFO   : Strobe is symmetric")

        all_valid_delays = []
        for awg, sequence in awgs_and_sequences:
            valid_delays = self._find_valid_delays(awg, sequence, verbose)
            if valid_delays:
                all_valid_delays.append(valid_delays)
            else:
                if verbose:
                    print(
                        "ERROR  : Unable to find valid delays for AWG {}!".format(awg))
                return False

        # Figure out which delays are valid
        combined_valid_delays = set.intersection(*all_valid_delays)
        max_valid_delay = max(combined_valid_delays)

        # Print information
        if verbose:
            print("INFO   : Valid delays are {}".format(combined_valid_delays))
        if verbose:
            print("INFO   : Setting delay to {}".format(max_valid_delay))

        # And configure the delays
        for awg, _ in awgs_and_sequences:
            vld_mask = 1 << self._dev.geti(
                'awgs/{}/dio/valid/index'.format(awg))
            strb_mask = 1 << self._dev.geti(
                'awgs/{}/dio/strobe/index'.format(awg))
            cw_mask = self._dev.geti('awgs/{}/dio/mask/value'.format(awg))
            cw_shift = self._dev.geti('awgs/{}/dio/mask/shift'.format(awg))
            if verbose:
                print("INFO   : Setting delay of AWG {}".format(awg))
            self._set_dio_delay(
                awg, strb_mask,
                (cw_mask << cw_shift) | vld_mask, max_valid_delay)

        return True

    def activate_new_dio_triggering(self):
        # from: AWG8_V2_DIO_Calibration
        # FIXME: cleanup
        # Activate the new DIO triggering functionality where the instrument does not specifically use
        # the flanks of the toggle signal  signal for sampling the DIO values, but an internally generated
        # counter locked to the expected period of the toggle signal (in 300 MHz cycles) and the first edge
        # of the toggle signal
        TOGGLE_PERIOD = 12
        self._dev.seti('awgs/*/dio/strobe/width', 0)
        self._dev.seti('awgs/*/dio/strobe/width', TOGGLE_PERIOD)
        for i in range(4):
            if self._dev.geti('awgs/{}/dio/strobe/width'.format(i)) != TOGGLE_PERIOD:
                raise Exception(
                    'Expected strobe width not configured correctly for AWG {}! Please rerun this cell!\n'.format(i))

    def _debug_report_dio(self):
        # FIXME: only DIO 0 for now
        logging.info('DIO bits with timing errors:  0x%08X' % self._dev.geti('awgs/0/dio/error/timing'))
        logging.info('DIO bits detected high:       0x%08X' % self._dev.geti('awgs/0/dio/highbits'))
        logging.info('DIO bits detected low:        0x%08X' % self._dev.geti('awgs/0/dio/lowbits'))
        # AWGS/0/DIO/ERROR/WIDTH
        # AWGS/0/DIO/DATA

    # override for InstrumentBase
    def snapshot_base(self, update=False, params_to_skip_update=None):
        if params_to_skip_update is None:
            params_to_skip_update = self._params_to_skip_update
        snap = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update)
        return snap

    ##########################################################################
    # 'private' functions: parameter support for codewords
    ##########################################################################

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
                parname = 'wave_ch{}_cw{:03}'.format(ch+1, cw)  # NB: parameter naming identical to QWG
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(
                        ch+1, cw),
                    vals=vals.Arrays(),  # min_value, max_value = unknown
                    set_cmd=self._gen_write_csv(parname),
                    get_cmd=self._gen_read_csv(parname),
                    docstring=docst)
                self._params_to_skip_update.append(parname)

    ##########################################################################
    # 'private' functions: helpers for calibrate_dio_protocol()
    ##########################################################################

    def _ensure_symmetric_strobe(self, verbose=False):
        done = False
        good_shots = 0
        bad_shots = 0
        strb_bits = []
        for awg in range(0, 4):
            strb_bits.append(self._dev.geti(
                'awgs/{}/dio/strobe/index'.format(awg)))
        strb_bits = list(set(strb_bits))
        if verbose:
            print('INFO   : Analyzing strobe bits {}'.format(strb_bits))

        while not done:
            data = self._dev.getv('raw/dios/0/data')
            if _is_dio_strb_symmetric(data, strb_bits):
                if verbose:
                    print('INFO   : Found good shot')
                bad_shots = 0
                good_shots += 1
                if good_shots > 5:
                    done = True
            else:
                if verbose:
                    print('INFO   : Strobe bit(s) are not sampled symmetrically')
                if verbose:
                    print("INFO   :   Disabling AWG's")

                # save enabled state of AWGs, then disable them
                enables = 4*[0]
                for awg in range(0, 4):
                    enables[awg] = self._dev.geti('awgs/{}/enable'.format(awg))
                    self._dev.seti('awgs/{}/enable'.format(awg), 0)

                # switch clock to internal and back to external
                if verbose:
                    print("INFO   :   Switching to internal clock")
                self.system_clocks_referenceclock_source(0)
                time.sleep(5)
                if verbose:
                    print("INFO   :   Switching to external clock")
                self.system_clocks_referenceclock_source(1)
                time.sleep(5)
                # FIXME: check locking

                # restore enabled state of AWGs
                if verbose:
                    print("INFO   :   Enabling AWG's")
                for awg in range(0, 4):
                    self._dev.seti('awgs/{}/enable'.format(awg), enables[awg])

                good_shots = 0
                bad_shots += 1
                if bad_shots > 5:
                    done = True

        return (good_shots > 0) and (bad_shots == 0)

    def _find_valid_delays(self, awg, expected_sequence, verbose=False):
        """
        The function loops through the possible delay settings on the DIO interface
        and records and analyzes DIO data for each setting. It then determines whether
        a given delay setting results in valid DIO protocol data being recorded.
        In order for data to be correct, two conditions must be satisfied: First,
        no timing violations are allowed, and, second, the sequence of codewords
        detected on the interface must match the expected sequence.
        """
        if verbose:
            print("INFO   : Finding valid delays for AWG {}".format(awg))
        vld_mask = 1 << self._dev.geti('awgs/{}/dio/valid/index'.format(awg))
        vld_polarity = self._dev.geti('awgs/{}/dio/valid/polarity'.format(awg))
        strb_mask = 1 << self._dev.geti('awgs/{}/dio/strobe/index'.format(awg))
        strb_slope = self._dev.geti('awgs/{}/dio/strobe/slope'.format(awg))
        cw_mask = self._dev.geti('awgs/{}/dio/mask/value'.format(awg))
        cw_shift = self._dev.geti('awgs/{}/dio/mask/shift'.format(awg))

        if verbose:
            print('INFO   : vld_mask     = 0x{:08x}'.format(vld_mask))
            print('INFO   : vld_polarity =', vld_polarity)
            print('INFO   : strb_mask    = 0x{:08x}'.format(strb_mask))
            print('INFO   : strb_slope   =', strb_slope)
            print('INFO   : cw_mask      = 0x{:08x}'.format(cw_mask))
            print('INFO   : cw_shift     =', cw_shift)

        valid_delays = []
        for delay in range(0, 7):
            if verbose:
                print("INFO   : Testing delay {} on AWG {}...".format(delay, awg))
            self._set_dio_delay(awg, strb_mask,
                                (cw_mask << cw_shift) | vld_mask, delay)

            data = self._dev.getv('awgs/' + str(awg) + '/dio/data')
            codewords, timing_violations = _analyze_dio_data(
                data, strb_mask, strb_slope, vld_mask, vld_polarity, cw_mask, cw_shift)
            timeout_cnt = 0
            while (cw_mask != 0) and len(codewords) == 0:
                if timeout_cnt > 5:
                    break
                if verbose:
                    print("WARNING: No codewords detected, trying again!")
                data = self._dev.getv('awgs/' + str(awg) + '/dio/data')
                codewords, timing_violations = _analyze_dio_data(
                    data, strb_mask, strb_slope, vld_mask, vld_polarity, cw_mask, cw_shift)
                timeout_cnt += 1

            # Compare codewords against sequence
            if (cw_mask != 0) and len(codewords) == 0:
                if verbose:
                    print(
                        "WARNING: No codewords detected on AWG {} for delay {}".format(awg, delay))
                continue

            # Can't do nothing with timing violations
            if timing_violations:
                if verbose:
                    print("WARNING: Timing violation detected on AWG {} for delay {}!".format(
                        awg, delay))
                continue

            # Check against expected sequence
            valid_sequence = True
            for n, codeword in enumerate(codewords):
                if n == 0:
                    if codeword not in expected_sequence:
                        if verbose:
                            print("WARNING: Codeword {} with value {} not in expected sequence {}!".format(
                                n, codeword, expected_sequence))
                        if verbose:
                            print(
                                "INFO   : Detected codeword sequence: {}".format(codewords))
                        valid_sequence = False
                        break
                    else:
                        index = expected_sequence.index(codeword)
                else:
                    last_index = index
                    index = (index + 1) % len(expected_sequence)
                    if codeword != expected_sequence[index]:
                        if verbose:
                            print("WARNING: Codeword {} with value {} not expected to follow codeword {} in expected sequence {}!".format(
                                n, codeword, expected_sequence[last_index], expected_sequence))
                        if verbose:
                            print(
                                "INFO   : Detected codeword sequence: {}".format(codewords))
                        valid_sequence = False
                        break

            # If we get to this point the delay is valid
            if valid_sequence:
                valid_delays.append(delay)

        if verbose:
            print("INFO   : Found valid delays of {}".format(list(valid_delays)))
        return set(valid_delays)

##############################################################################
# non class functions: helpers for calibrate_dio_protocol()
##############################################################################


def _get_edges(value, last_value, mask):
    """
    Given two integer values representing a current and a past value,
    and a bit mask, this function will return two
    integer values representing the bits with rising (re) and falling (fe)
    edges.
    """
    changed = value ^ last_value
    re = changed & value & mask
    fe = changed & ~value & mask
    return re, fe


def _is_dio_strb_symmetric(data, bits):
    # FIXME: reports OK if there is no input
    count_ok = True

    for bit in bits:
        strobe_mask = 1 << bit
        count_low = False
        count_high = False
        strobe_low = 0
        strobe_high = 0
        last_strobe = None
        for n, d in enumerate(data):
            curr_strobe = (d & strobe_mask) != 0

            if count_high:
                if curr_strobe:
                    strobe_high += 1
                else:
                    if (strobe_low > 0) and (strobe_low != strobe_high):
                        count_ok = False
                        break

            if count_low:
                if not curr_strobe:
                    strobe_low += 1
                else:
                    if (strobe_high > 0) and (strobe_low != strobe_high):
                        count_ok = False
                        break

            if (last_strobe != None):
                if (curr_strobe and not last_strobe):
                    strobe_high = 0
                    count_high = True
                    count_low = False
                elif (not curr_strobe and last_strobe):
                    strobe_low = 0
                    count_low = True
                    count_high = False

            last_strobe = curr_strobe

        if not count_ok:
            break

    return count_ok


def _analyze_dio_data(data, strb_mask, strb_slope, vld_mask, vld_polarity, cw_mask, cw_shift):
    """
    Analyzes a list of integer values that represent samples recorded on the DIO interface.
    The function needs information about the protocol used on the DIO interface. Based
    on this information the function will return two lists: the detected codewords
    and the positions where 'timing violations' are found. The codewords are sampled
    according to the protocol configuration. Timing violations occur when a codeword
    bit or the valid bit changes value at the same time as the strobe signal.
    """
    timing_violations = []
    codewords = []
    last_d = None
    for n, d in enumerate(data):
        if n > 0:
            strb_re = False
            strb_fe = False
            if strb_slope == 0:
                strb_re = True
                strb_fe = True
            elif strb_slope == 1:
                strb_re, _ = _get_edges(d, last_d, strb_mask)
            elif strb_slope == 2:
                _, strb_fe = _get_edges(d, last_d, strb_mask)
            else:
                strb_re, strb_fe = _get_edges(d, last_d, strb_mask)

            vld_re = False
            vld_fe = False
            if vld_polarity != 0:
                vld_re, vld_fe = _get_edges(d, last_d, vld_mask)

            d_re = False
            d_fe = False
            if cw_mask != 0:
                d_re, d_fe = _get_edges(d, last_d, cw_mask << cw_shift)

            vld_active = ((vld_polarity & 1) and ((d & vld_mask) == 0)) or (
                (vld_polarity & 2) and ((d & vld_mask) != 0))
            codeword = (d >> cw_shift) & cw_mask

            # Check for timing violation on vld
            if (strb_re or strb_fe) and (vld_re or vld_fe):
                timing_violations.append(n)
            elif (strb_re or strb_fe) and (d_re or d_fe):
                timing_violations.append(n)

            # Get the codewords
            if (strb_re or strb_fe) and vld_active:
                codewords.append(codeword)

        last_d = d

    return codewords, timing_violations
