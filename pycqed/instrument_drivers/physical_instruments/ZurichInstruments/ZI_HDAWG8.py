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
"""

import time
import logging
import numpy
import re
import os
import pycqed

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG_core as zicore
import pycqed.instrument_drivers.physical_instruments.QuTech_CCL as qtccl

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
        super().__init__(name=name, device=device, interface=interface, server=server, port=port, num_codewords=num_codewords, **kw)

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
                'awgs_{}_outputs_1_amplitude'.format(i), 
                'awgs_{}_sequencer_program_crc32_hash'.format(i)})

        for i in range(8): 
            self._snapshot_whitelist.update({
                'sigouts_{}_direct'.format(i), 'sigouts_{}_offset'.format(i),
                'sigouts_{}_on'.format(i) , 'sigouts_{}_range'.format(i)})

        self._params_to_exclude = set(self.parameters.keys()) - self._snapshot_whitelist
        
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

    def upload_codeword_program(self, awgs=numpy.arange(4), cfg_num_codewords=None, cfg_codeword_protocol=None):
        """
        Generates a program that plays the codeword waves for each channel.

        awgs (array): the awg numbers to which to upload the codeword program.
                    By default uploads to all channels but can be specific to
                    speed up the process.
        cfg_num_codewords (optional): Optionally specify the number of codewords. Uses the num_codewords
                    member if not specified.
        cfg_codeword_protocol (optional): Optionally specify the codeword protocol. Uses the cfg_codeword_protocol()
                    parameter if not specified.

        Note: Assumes 'system/awg/channelgrouping' to be '4x2' or an exception will be raised.
        """
        self._configure_codeword_protocol()

        # Type conversion to ensure lists do not produce weird results
        awgs = numpy.array(awgs)
    
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

    def _codeword_table_preamble(self, awg_nr):
        """
        Defines a snippet of code to use in the beginning of an AWG program in order to define the waveforms.
        The generated code depends on the instrument type. For the HDAWG instruments, we use the seWaveDIO
        function.
        """
        program = ''
        
        # because awg_channels come in pairs
        ch = awg_nr*2           

        for cw in range(self._num_codewords):
            parnames = 2*['']
            csvnames = 2*['']
            # Every AWG drives two channels
            for i in range(2):
                parnames[i] = zibase.gen_waveform_name(ch+i, cw)  # NB: parameter naming identical to QWG
                csvnames[i] = self.devname + '_' + parnames[i]
            
            program += 'setWaveDIO({}, \"{}\", \"{}\");\n'.format(cw, csvnames[0], csvnames[1])

        return program

    # FIXME: should probably be private as it works in tandem with upload_codeword_program
    def _configure_codeword_protocol(self, default_dio_timing: bool=False):
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
        # Check overall configuration
        if self.system_awg_channelgrouping() != 0:
            log.warning('{}: Instrument not in 4 x 2 channel mode! Switching...'.format(self.devname))
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

            # the mask determines how many bits will be used in the protocol
            # e.g., mask 3 will mask the bits with bin(3) = 00000011 using
            # only the 2 Least Significant Bits.
            num_codewords = int(2**numpy.ceil(numpy.log2(self._num_codewords)))

            self.set('awgs_{}_dio_mask_value'.format(awg_nr), num_codewords-1)

            # No special requirements regarding waveforms by default
            self._clear_readonly_waveforms(awg_nr)

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
                    # FIXME: this is no longer true for HDAWG V2
                    self.set('awgs_{}_dio_mask_shift'.format(awg_nr), 9)    
            elif self.cfg_codeword_protocol() == 'flux':
                # We need 64 codewords defined for the flux protocol
                if self._num_codewords != 64:
                    raise zibase.ziConfigurationError("Incorrect number of codewords defined! Need exactly 64 codewords for the 'flux' protocol.")

                # FIXME: Check that this configuration is correct when used with both
                # CC-Light and CC.
                # bits[0:3] for awg0_ch0, bits[4:6] for awg0_ch1 etc.
                self.set('awgs_{}_dio_mask_value'.format(awg_nr), 2**6-1)
                self.set('awgs_{}_dio_mask_shift'.format(awg_nr), awg_nr*6)

                mask_0 = 0b000111  # AWGx_ch0 uses lower bits for CW
                mask_1 = 0b111000  # AWGx_ch1 uses higher bits for CW

                for cw in range(2**6):
                    # If either codeword is zero, fill that waveform with zeros
                    # and mark it as read-only
                    if (cw & mask_0) == 0:
                        self.set(zibase.gen_waveform_name(2*awg_nr+0, cw), numpy.zeros(self._default_waveform_length))
                        self._set_readonly_waveform(2*awg_nr+0, cw)

                    if ((cw & mask_1) >> 3) == 0:
                        self.set(zibase.gen_waveform_name(2*awg_nr+1, cw), numpy.zeros(self._default_waveform_length))
                        self._set_readonly_waveform(2*awg_nr+1, cw)
            else:
                raise zibase.ziValueError('Invalid value {} selected for cfg_codeword_protocol!'.format(self.cfg_codeword_protocol()))

        # Disable all function generators
        for param in [key for key in self.parameters.keys() if re.match(r'sines_\d+_enables_\d+', key)]:
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
    # 'private' functions: parameter support for codewords
    ##########################################################################

    def _add_extra_parameters(self) -> None:
        super()._add_extra_parameters()

        self.add_parameter(
            'cfg_codeword_protocol', initial_value='identical',
            vals=validators.Enum('identical', 'microwave', 'flux'), docstring=(
                'Used in the configure codeword method to determine what DIO'
                ' pins are used in for which AWG numbers.'),
            parameter_class=ManualParameter)

    ##########################################################################
    # 'private' functions: DIO calibrration
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

    def calibrate_CCL_dio_protocol(self, CCL=None, verbose=False, repetitions=1):
        log.info('Calibrating DIO delays')
        if verbose: print("Calibrating DIO delays")

        if CCL is None:
            CCL = qtccl.CCL('CCL', address='192.168.0.11', port=5025)

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
            staircase_sequence = numpy.arange(1, sequence_length)
            expected_sequence = [(0, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (1, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (2, list(staircase_sequence + (staircase_sequence << 3))), \
                                 (3, list(staircase_sequence))]
        elif self.cfg_codeword_protocol() == 'microwave':
            test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..','examples','CCLight_example',
                'qisa_test_assembly','calibration_cws_mw.qisa'))

            sequence_length = 32
            staircase_sequence = range(1, sequence_length)
            expected_sequence = [(0, list(reversed(staircase_sequence))), \
                                 (1, list(reversed(staircase_sequence))), \
                                 (2, list(reversed(staircase_sequence))), \
                                 (3, list(reversed(staircase_sequence)))]

        else:
            zibase.ziConfigurationError("Can only calibrate DIO protocol for 'flux' or 'microwave' mode!")

        # Start the CCL with the program configured above
        CCL.eqasm_program(test_fp)
        CCL.start()

        # Make sure the configuration is up-to-date
        self.assure_ext_clock()
        self.upload_codeword_program()

        for awg, sequence in expected_sequence:
            if not self._ensure_activity(awg, mask_value=numpy.bitwise_or.reduce(sequence), verbose=verbose):
                raise ziDIOActivityError('No or insufficient activity found on the DIO bits associated with AWG {}'.format(awg))

        valid_delays = self._find_valid_delays(expected_sequence, repetitions, verbose=verbose)
        if len(valid_delays) == 0:
            raise ziDIOCalibrationError('DIO calibration failed! No valid delays found')
            return

        min_valid_delay = min(valid_delays)

        # Print information
        if verbose: print("  Valid delays are {}".format(valid_delays))
        if verbose: print("  Setting delay to {}".format(min_valid_delay))

        # And configure the delays
        self.setd('raw/dios/0/delays/*', min_valid_delay)