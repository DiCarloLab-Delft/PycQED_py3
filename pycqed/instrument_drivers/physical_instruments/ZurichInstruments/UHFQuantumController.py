"""
To do:
- split off application dependent code, as done for ZI_HDAWG8.py

Notes:


Changelog:

20190113 WJV
- started Changelog
- addressed many warnings identified by PyCharm
- started adding type annotations
- split of stuff into _add_node_pars()
- made some properties 'private'

20190219 WJV
- tagged some dead code with FIXM.

20190219:
- made _array_to_combined_vector_string() a @staticmethod

20190417 WJV
- merged branch 'develop' into 'feature/cc', changes:
    spec_mode_on
    spec_mode_off

20190429 WJV
- merged branch 'QCC_testing' into 'feature/cc', changes:
    load_default_settings(): awgs_0_dio_strobe_index changed from 31 (CCL) to 15 (QCC)

20190612 WJV
- merged branch 'QCC_testing' into 'feature/cc', changes:
    adds awg_sequence_acquisition_and_DIO_RED_test()

20190618 WJV
- merged branch 'develop' into 'feature/cc', changes:

20190813 NH
- merged branch 'develop' into 'feature/ZIupdateDrivers'
- Updated driver to use new UHFQA nodes
- Updated to support dynamic waveform upload properly. The AWG is configured when start() is called and the
    driver then chooses whether it is necessary to recompile the AWG program. The program will be recompiled
    if waveform lengths have changed. Otherwise, if waveforms have been updated they will just be downloaded
    directly to the instrument.

"""

import time
import os
import logging
import numpy as np
import pycqed

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
from pycqed.utilities.general import check_keyboard_interrupt

from qcodes.utils import validators
from qcodes.utils.helpers import full_class
from qcodes.instrument.parameter import ManualParameter

log = logging.getLogger(__name__)

##########################################################################
# Exceptions
##########################################################################


class ziUHFQCSeqCError(Exception):
    """Exception raised when the configured SeqC program does
       not match the structure needed for a given measurement in terms
       of number of samples, number of averages or the use of a delay."""
    pass


class ziUHFQCHoldoffError(Exception):
    """Exception raised when a holdoff error has occurred in either the
    input monitor or result logging unit. Increase the delay between triggers
    sent to these units to solve the problem."""
    pass

class ziUHFQCDIOActivityError(Exception):
    """Exception raised when insufficient activity is detected on the bits
    of the DIO to be used for controlling which qubits to measure."""
    pass

class ziUHFQCDIOCalibrationError(Exception):
    """Exception raised when the DIO calibration fails, meaning no signal
    delay can be found where no timing violations are detected."""
    pass

##########################################################################
# Module level functions
##########################################################################


def awg_sequence_acquisition_preamble():
    """
    This function defines a standard AWG program preamble, which is used
    regardless of the specific acquisition mode. The preamble defines standard
    functionality of the user registers, which are used for dynamically
    controlling e.g. number of iterations in a loop, etc.
    The preamble also defines a standard way of selecting between triggering
    the readout units or the time-domain input monitor.
    """
    preamble = """
// Reset error counter
setUserReg(4, 0);

// Define standard variables
var loop_cnt = getUserReg(0);
var ro_mode  = getUserReg(1);
var wait_dly = getUserReg(2);
var avg_cnt  = getUserReg(3);
var ro_arm;
var ro_trig;

// Configure readout mode
if (ro_mode) {
  ro_arm  = AWG_INTEGRATION_ARM;
  ro_trig = AWG_MONITOR_TRIGGER + AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER;
} else {
  ro_arm  = AWG_INTEGRATION_ARM;
  ro_trig = AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER;
}"""
    return preamble


def array2vect(array, name):
    # this function cuts up arrays into several vectors of maximum length 1024 that are joined.
    # this is to avoid python crashes (was found to crash for vectors of
    # length> 1490)
    if len(array) > 1024:
        splitted_array = np.array_split(array, len(array)//1024)
        string_array = ['\nvect(' + ','.join(['{:.8f}'.format(x)
                                              for x in sub_array]) + ')' for sub_array in splitted_array]
        return 'wave ' + name + ' = join(' + ','.join(string_array) + ');\n'
    else:
        return 'wave ' + name + ' = ' + 'vect(' + ','.join(['{:.8f}'.format(x) for x in array]) + ');\n'

##########################################################################
# Class
##########################################################################


class UHFQC(zibase.ZI_base_instrument):
    """
    This is the PycQED driver for the 1.8 Gsample/s UHFQA developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5/3.6 ucs4 19.05 for 64bit Windows from
        http://www.zhinst.com/downloads, https://people.zhinst.com/~niels/
    2. upload the latest firmware to the UHFQA usingthe LabOne GUI
    """

    # Define minimum required revisions
    MIN_FWREVISION = 63210
    MIN_FPGAREVISION = 63133

    # Define user registers
    USER_REG_LOOP_CNT = 0
    USER_REG_RO_MODE = 1
    USER_REG_WAIT_DLY = 2
    USER_REG_AVG_CNT = 3
    USER_REG_ERR_CNT = 4

    ##########################################################################
    # 'public' functions: device control
    ##########################################################################

    def __init__(self,
                 name,
                 device:                  str,
                 interface:               str = 'USB',
                 address:                 str = '127.0.0.1',
                 port:                    int = 8004,
                 use_dio:                 bool = True,
                 nr_integration_channels: int = 9,
                 server:                  str = '',
                 **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument
            device          (str) the name of the device e.g., "dev8008"
            interface       (str) the name of the interface to use ('1GbE' or 'USB')
            address         (str) the host where the ziDataServer is running (for compatibility)
            port            (int) the port to connect to for the ziDataServer (don't change)
            use_dio         (bool) assert to enable the DIO interface
            nr_integration_channels (int) the number of integration channels to use (max 10)
            server:         (str) the host where the ziDataServer is running (if not '' then used instead of address)
        """
        t0 = time.time()

        # Override server with the old-style address argument
        if server == '':
            server = address

        # save some parameters
        self._nr_integration_channels = nr_integration_channels
        self._use_dio = use_dio

        # Used for keeping track of which nodes we are monitoring for data
        self._acquisition_nodes = []

        # The following members define the characteristics of the configured
        # AWG program
        self._reset_awg_program_features()

        # The actual codeword cases used in a given program
        self._cases = None

        # Used for extra DIO output to CC for debugging
        self._diocws = None

        # Holds the DIO calibration delay
        self._dio_calibration_delay = 0

        # Define parameters that should not be part of the snapshot
        self._params_to_exclude = set(['features_code', 'system_fwlog', 'system_fwlogenable'])

        # Our base class includes all the functionality needed to initialize the parameters
        # of the object. Those parameters are read from instrument-specific JSON files stored
        # in the zi_parameter_files folder.
        super().__init__(name=name, device=device, interface=interface,
                         server=server, port=port, num_codewords=2**nr_integration_channels,
                         **kw)

        # Disable disfunctional parameters from snapshot
        self._params_to_exclude = set(['features_code', 'system_fwlog', 'system_fwlogenable'])

        # Set default waveform length to 20 ns at 1.8 GSa/s
        self._default_waveform_length = 32

        # Mask used for detecting codeword activity during DIO calibration
        self._dio_calibration_mask = None

        t1 = time.time()
        log.info(f'{self.devname}: Initialized UHFQC in {t1 - t0}s')

    ##########################################################################
    # Overriding Qcodes InstrumentBase methods
    ##########################################################################

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
    # Overriding ZI_base_instrument methods
    ##########################################################################

    def _check_devtype(self):
        if self.devtype != 'UHFQA':
            raise zibase.ziDeviceError(
                'Device {} of type {} is not a UHFQA instrument!'.format(self.devname, self.devtype))

    def _check_options(self):
        """
        Checks that the correct options are installed on the instrument.
        """
        options = self.gets('features/options').split('\n')
        if 'QA' not in options and 'QC' not in options:
            raise zibase.ziOptionsError(
                'Device {} is missing the QA or QC option!'.format(self.devname))
        if 'AWG' not in options:
            raise zibase.ziOptionsError(
                'Device {} is missing the AWG option!'.format(self.devname))

    def _check_awg_nr(self, awg_nr):
        """
        Checks that the given AWG index is valid for the device.
        """
        if (awg_nr != 0):
            raise zibase.ziValueError(
                'Invalid AWG index of {} detected!'.format(awg_nr))

    def _check_versions(self):
        """
        Checks that sufficient versions of the firmware are available.
        """
        if self.geti('system/fwrevision') < UHFQC.MIN_FWREVISION:
            raise zibase.ziVersionError('Insufficient firmware revision detected! Need {}, got {}!'.format(
                UHFQC.MIN_FWREVISION, self.geti('system/fwrevision')))

        if self.geti('system/fpgarevision') < UHFQC.MIN_FPGAREVISION:
            raise zibase.ziVersionError('Insufficient FPGA revision detected! Need {}, got {}!'.format(
                UHFQC.MIN_FPGAREVISION, self.geti('system/fpgarevision')))

    def _num_channels(self):
        return 2

    def _add_extra_parameters(self) -> None:
        """
        We add a few additional custom parameters on top of the ones defined in the device files. These are:
          qas_0_trans_offset_weightfunction - an offset correction parameter for all weight functions,
            this allows normalized calibration when performing cross-talk suppressed readout. The parameter
            is not actually used in this driver, but in some of the support classes that make use of the driver.
          AWG_file - allows the user to configure the AWG with a SeqC program from a specific file.
            Provided only because the old version of the driver had this parameter. It is discouraged to use
            it.
          wait_dly - a parameter that enables the user to set a delay in AWG clocks cycles (4.44 ns) to be
            applied between when the AWG starts playing the readout waveform, and when it triggers the
            actual readout.
          cases - a parameter that can be used to define which combination of readout waveforms to actually
            download to the instrument. As the instrument has a limited amount of memory available, it is
            not currently possible to store all 1024 possible combinations of readout waveforms that would
            be required to address the maximum number of qubits supported by the instrument (10). Therefore,
            the 'cases' mechanism is used to reduce that number to the combinations actually needed by
            an experiment.
          dio_calibration_delay - the delay that is programmed on the DIO lines as part of the DIO calibration
            process in order for the instrument to reliably sample data from the CC. Can be used to detect
            unexpected changes in timing of the entire system. The parameter can also be used to force a specific
            delay to be used on the DIO although that is not generally recommended.
        """
        super()._add_extra_parameters()

        # storing an offset correction parameter for all weight functions,
        # this allows normalized calibration when performing cross-talk suppressed
        # readout
        for i in range(self._nr_integration_channels):
            self.add_parameter(
                "qas_0_trans_offset_weightfunction_{}".format(i),
                unit='',  # unit is adc value
                label='RO normalization offset',
                initial_value=0.0,
                docstring='an offset correction parameter for all weight functions, '
                'this allows normalized calibration when performing cross-talk suppressed readout. The parameter '
                'is not actually used in this driver, but in some of the support classes that make use of the driver.',
                parameter_class=ManualParameter)

        self.add_parameter(
            'AWG_file',
            set_cmd=self._do_set_AWG_file,
            docstring='Configures the AWG with a SeqC program from a specific file. '
            'Provided only for backwards compatibility. It is discouraged to use '
            'this parameter unless you know what you are doing',
            vals=validators.Anything())

        self.add_parameter(
            'wait_dly',
            set_cmd=self._set_wait_dly,
            get_cmd=self._get_wait_dly,
            unit='',
            label='AWG cycle delay',
            docstring='Configures a delay in AWG clocks cycles (4.44 ns) to be '
            'applied between when the AWG starts playing the readout waveform, and when it triggers the '
            'actual readout.',
            vals=validators.Ints())

        self.add_parameter(
            'cases',
            set_cmd=self._set_cases,
            get_cmd=self._get_cases,
            docstring='Configures which combination of readout waveforms to actually '
            'download to the instrument. As the instrument has a limited amount of memory available, it is '
            'not currently possible to store all 1024 possible combinations of readout waveforms that would '
            'be required to address the maximum number of qubits supported by the instrument (10). Therefore, '
            'the \'cases\' mechanism is used to reduce that number to the combinations actually needed by '
            'an experiment. The parameter must be set to a list of integers. The list defines the codewords '
            'to be handled by the AWG program. For example, setting the parameter to [1, 5, 7] would result in '
            'an AWG program that handles only codewords 1, 5 and 7. When running, if the AWG receives a codeword '
            'that is not part of this list, an error will be triggered.',
            vals=validators.Lists())

        self.add_parameter('dio_calibration_delay',
            set_cmd=self._set_dio_calibration_delay,
            get_cmd=self._get_dio_calibration_delay,
            unit='',
            label='DIO Calibration delay',
            docstring='Configures the internal delay in 300 MHz cycles (3.3 ns) '
            'to be applied on the DIO interface in order to achieve reliable sampling '
            'of the codewords. The valid range is 0 to 15.',
            vals=validators.Ints())

    def _codeword_table_preamble(self, awg_nr):
        """
        Defines a snippet of code to use in the beginning of an AWG program in order to define the waveforms.
        The generated code depends on the instrument type. For the UHF-QA we simply define the raw waveforms.
        """
        program = ''

        # If the program doesn't need waveforms, just return here
        if not self._awg_program_features['waves']:
            return program

        # If the program needs cases, but none are defined, flag it as an error
        if self._awg_program_features['cases'] and self._cases is None:
            raise zibase.ziConfigurationError(
                'Missing definition of cases for AWG program!')

        wf_table = self._get_waveform_table(awg_nr)
        for dio_cw, (wf_l, wf_r) in enumerate(wf_table):
            csvname_l = self.devname + '_' + wf_l
            csvname_r = self.devname + '_' + wf_r
            program += 'wave {} = "{}";\n'.format(
                wf_l, csvname_l)
            program += 'wave {} = "{}";\n'.format(
                wf_r, csvname_r)
        return program

    ##########################################################################
    # 'public' overrides for ZI_base_instrument
    ##########################################################################

    def assure_ext_clock(self) -> None:
        """
        Make sure the instrument is using an external reference clock
        """
        # get source:
        #   1: external
        #   0: internal (commanded so, or because of failure to sync to external clock)
        source = self.system_extclk()
        if source == 1:
            return

        print('Switching to external clock. This could take a while!')
        while True:
            self.system_extclk(1)
            timeout = 10
            while timeout > 0:
                time.sleep(0.1)
                status = self.system_extclk()
                if status == 1:             # synced
                    break
                else:                       # sync failed
                    timeout -= 0.1
                    print('X', end='')
            if self.system_extclk() != 1:
                print(' Switching to external clock failed. Trying again.')
            else:
                break
        print('\nDone')

    def load_default_settings(self, upload_sequence=True) -> None:
        # standard configurations adapted from Haendbaek's notebook

        # The averaging-count is used to specify how many times the AWG program
        # should run
        LOG2_AVG_CNT = 10

        # Load an AWG program
        if upload_sequence:
            self.awg_sequence_acquisition()

        # Setting the clock to external
        self.system_extclk(1)

        # Turn on both outputs
        self.sigouts_0_on(1)
        self.sigouts_1_on(1)

        # Set the output channels to 50 ohm
        self.sigouts_0_imp50(True)
        self.sigouts_1_imp50(True)

        # Configure the analog trigger input 1 of the AWG to assert on a rising
        # edge on Ref_Trigger 1 (front-panel of the instrument)
        self.awgs_0_triggers_0_rising(1)
        self.awgs_0_triggers_0_level(0.000000000)
        self.awgs_0_triggers_0_channel(2)

        # Configure the digital trigger to be a rising-edge trigger
        self.awgs_0_auxtriggers_0_slope(1)

        # Straight connection, signal input 1 to channel 1, signal input 2 to
        # channel 2

        self.qas_0_deskew_rows_0_cols_0(1.0)
        self.qas_0_deskew_rows_0_cols_1(0.0)
        self.qas_0_deskew_rows_1_cols_0(0.0)
        self.qas_0_deskew_rows_1_cols_1(1.0)

        # Configure the codeword protocol
        if self._use_dio:
            self.dios_0_mode(2)  # QuExpress thresholds on DIO (mode == 2), AWG control of DIO (mode == 1)
            self.dios_0_drive(0x3)  # Drive DIO bits 15 to 0
            self.dios_0_extclk(2)  # 50 MHz clocking of the DIO
            self.awgs_0_dio_strobe_slope(0)  # no edge, replaced by dios_0_extclk(2)
            self.awgs_0_dio_strobe_index(15)  # NB: 15 for QCC (was 31 for CCL). Irrelevant now we use 50 MHz clocking
            self.awgs_0_dio_valid_polarity(2)  # high polarity
            self.awgs_0_dio_valid_index(16)

        # No rotation on the output of the weighted integration unit, i.e. take
        # real part of result
        for i in range(0, self._nr_integration_channels):
            self.set('qas_0_rotations_{}'.format(i), 1.0 + 0.0j)
            # remove offsets to weight function
            self.set('qas_0_trans_offset_weightfunction_{}'.format(i), 0.0)

        # No cross-coupling in the matrix multiplication (identity matrix)
        self.reset_crosstalk_matrix()

        # disable correlation mode on all channels
        self.reset_correlation_params()

        # Configure the result logger to not do any averaging
        self.qas_0_result_length(1000)
        self.qas_0_result_averages(pow(2, LOG2_AVG_CNT))
        # result_logging_mode 2 => raw (IQ)
        self.qas_0_result_source(2)

        # The custom firmware will feed through the signals on Signal Input 1 to Signal Output 1 and Signal Input 2 to Signal Output 2
        # when the AWG is OFF. For most practical applications this is not really useful. We, therefore, disable the generation of
        # these signals on the output here.
        self.sigouts_0_enables_0(0)
        self.sigouts_0_enables_1(0)
        self.sigouts_1_enables_0(0)
        self.sigouts_1_enables_1(0)

    ##########################################################################
    # Private methods
    ##########################################################################

    def _reset_awg_program_features(self):
        """
        Resets the self._awg_program_features to disable all features. The UHFQC can be configured with a number
        of application-specific AWG programs using this driver. However, all the programs share some characteristics that
        are described in the _awg_program_features dictionary. For example, all of the programs include a main loop
        that runs for a number of iterations given by a user register. This feature is indicated by the 'loop_cnt'
        item in the dictionary. In contrast, not all program include an extra loop for the number of averages that
        should be done. Therefore, the 'awg_cnt' item in the dictionary is not automatically set. The driver
        uses these features to keep track of what the current AWG program can do. It then raises errors in case
        the user tries to do something that is not supported.
        """
        self._awg_program_features = {
            'loop_cnt': False,
            'avg_cnt': False,
            'wait_dly': False,
            'waves': False,
            'cases': False,
            'diocws': False}

    def _set_dio_calibration_delay(self, value):
        # Sanity check the value
        if value < 0 or value > 15:
            raise zibase.ziValueError(
                'Trying to set DIO calibration delay to invalid value! Expected value in range 0 to 15. Got {}.'.format(
                    value))

        log.info('Setting DIO calibration delay to {}'.format(value))
        # Store the value
        self._dio_calibration_delay = value

        # And configure the delays
        self.setd('raw/dios/0/delay', self._dio_calibration_delay)

    def _get_dio_calibration_delay(self):
        return self._dio_calibration_delay

    def _set_wait_dly(self, value):
        self.set('awgs_0_userregs_{}'.format(UHFQC.USER_REG_WAIT_DLY), value)

    def _get_wait_dly(self):
        return self.get('awgs_0_userregs_{}'.format(UHFQC.USER_REG_WAIT_DLY))

    def _set_cases(self, value):
        # Generate error if we don't have an AWG program that supports cases
        if not self._awg_program_features['cases']:
            raise zibase.ziValueError(
                'Trying to define cases for an AWG program that does not support them!')

        # Check against number of codewords
        if len(value) > self._num_codewords:
            raise zibase.ziValueError('Trying to define a number of cases ({}) greater than configured number of codewords ({})!'.format(
                len(value), self._num_codewords))

        self._cases = value
        self._cw_mask = 0
        for case in self._cases:
            self._cw_mask |= case

        if self._awg_program_features['diocws'] and self._diocws is None:
            raise zibase.ziValueError(
                'AWG program defines DIO output, but no output values have been defined!')

        self._awg_program[0] = \
            awg_sequence_acquisition_preamble() + """
// Mask for selecting our codeword bits
const CW_MASK = (0x1ff << 17);
// Counts wrong codewords
var err_cnt = 0;
"""

        if self._awg_program_features['diocws']:
            self._awg_program[0] += \
                array2vect(self._diocws, "diocws") + """
// Loop once for each DIO codeword to output
for (cvar i = 0; i < {}; i = i + 1) {{""".format(len(self._diocws))
        else:
            self._awg_program[0] += """
// Loop for all measurements
repeat (loop_cnt) {"""

        self._awg_program[0] += """
    waitDIOTrigger();
    // Get codeword and apply mask
    var cw = getDIOTriggered() & CW_MASK;
    // Generate waveforms based on codeword output
    switch (cw) {"""
        # Add each of the cases
        for case in self._cases:
            self._awg_program[0] += """
        case 0x{:08x}: playWave({}, {});""".format(case << 17, zibase.gen_waveform_name(0, case), zibase.gen_waveform_name(1, case))

        # Add a default for ensuring we see something when the other cases fail
        self._awg_program[0] += """
        default: playWave(ones(32), ones(32)); err_cnt += 1;
    }
    wait(wait_dly);"""

        if self._awg_program_features['diocws']:
            self._awg_program[0] += """
    setDIO(diocws[i]);
"""
        self._awg_program[0] += """
    setTrigger(ro_trig);
    setTrigger(ro_arm);
}
wait(300);
setTrigger(0);
setUserReg(4, err_cnt);"""

        self._awg_needs_configuration[0] = True

    def _get_cases(self):
        return self._cases

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
        if self.cases() is not None:
            for case in self.cases():
                wf_table.append((zibase.gen_waveform_name(ch, case),
                                 zibase.gen_waveform_name(ch+1, case)))
        return wf_table

    ##########################################################################
    # 'public' functions
    ##########################################################################

    def clock_freq(self):
        return 1.8e9

    ##########################################################################
    # 'public' functions: utility
    ##########################################################################

    def reset_acquisition_params(self):
        log.info('Setting user registers to 0')
        for i in range(16):
            self.set('awgs_0_userregs_{}'.format(i), 0)

        self.reset_crosstalk_matrix()
        self.reset_correlation_params()
        self.reset_rotation_params()

    def reset_crosstalk_matrix(self):
        self.upload_crosstalk_matrix(np.eye(10))

    def reset_correlation_params(self):
        for i in range(10):
            self.set('qas_0_correlations_{}_enable'.format(i), 0)
            self.set('qas_0_correlations_{}_source'.format(i), 0)
        for i in range(10):
            self.set('qas_0_thresholds_{}_correlation_enable'.format(i), 0)
            self.set('qas_0_thresholds_{}_correlation_source'.format(i), 0)

    def reset_rotation_params(self):
        for i in range(10):
            self.set('qas_0_rotations_{}'.format(i), 1+1j)

    ##########################################################################
    # 'public' functions: generic AWG/waveform support
    ##########################################################################

    def load_awg_program_from_file(self, filename) -> None:
        """
        Loads an awg sequence onto the UHFQA from a text file.
        File needs to obey formatting specified in the manual.
        Only provided for backwards compatibility purposes.
        """
        print(filename)
        with open(filename, 'r') as awg_file:
            self._awg_program[0] = awg_file.read()
            self._awg_needs_configuration[0] = True

    def _do_set_AWG_file(self, filename) -> None:
        self.load_awg_program_from_file('UHFLI_AWG_sequences/'+filename)

    def awg_file(self, filename) -> None:
        """Only provided for backwards compatibility purposes."""
        self.load_awg_program_from_file(filename)

    def awg_update_waveform(self, index, data) -> None:
        raise NotImplementedError(
            'Method not implemented! Please use the corresponding waveform parameters \'wave_chN_cwM\' to update waveforms!')

    ##########################################################################
    # 'public' functions: acquisition support
    ##########################################################################

    def acquisition(self, samples=100, averages=1, acquisition_time=0.010, timeout=10,
                    channels=(0, 1), mode='rl') -> None:  # FIXME: wrong return type
        self.timeout(timeout)
        self.acquisition_initialize(samples, averages, channels, mode)
        data = self.acquisition_poll(samples, True, acquisition_time)
        self.acquisition_finalize()

        return data

    def acquisition_initialize(self, samples, averages, channels=(0, 1),
                               mode='rl') -> None:
        # Define the channels to use and subscribe to them
        self._acquisition_nodes = []

        # Loop counter of AWG
        loop_cnt = samples

        # Make some checks on the configured AWG program
        if samples > 1 and not self._awg_program_features['loop_cnt']:
            raise ziUHFQCSeqCError(
                'Trying to acquire {} samples using an AWG program that does not use \'loop_cnt\'.'.format(samples))

        if averages > 1 and not self._awg_program_features['avg_cnt']:
            # Adjust the AWG loop counter according to the configured program
            loop_cnt *= averages

        if mode == 'rl':
            for c in channels:
                path = self._get_full_path(
                    'qas/0/result/data/{}/wave'.format(c))
                self._acquisition_nodes.append(path)
                self.subs(path)
            # Enable automatic readout
            self.qas_0_result_reset(1)
            self.qas_0_result_enable(1)
            self.qas_0_result_length(samples)
            self.qas_0_result_averages(averages)
            ro_mode = 0
        else:
            for c in channels:
                path = self._get_full_path(
                    'qas/0/monitor/inputs/{}/wave'.format(c))
                self._acquisition_nodes.append(path)
                self.subs(path)
            # Enable automatic readout
            self.qas_0_monitor_reset(1)
            self.qas_0_monitor_enable(1)
            self.qas_0_monitor_length(samples)
            self.qas_0_monitor_averages(averages)
            ro_mode = 1

        self.set('awgs_0_userregs_{}'.format(UHFQC.USER_REG_LOOP_CNT), loop_cnt)
        self.set('awgs_0_userregs_{}'.format(UHFQC.USER_REG_RO_MODE), ro_mode)
        self.set('awgs_0_userregs_{}'.format(UHFQC.USER_REG_AVG_CNT), averages)
        if self.wait_dly() > 0 and not self._awg_program_features['wait_dly']:
            raise ziUHFQCSeqCError(
                'Trying to use a delay of {} using an AWG program that does not use \'wait_dly\'.'.format(self.wait_dly()))
        self.set('awgs_0_userregs_{}'.format(UHFQC.USER_REG_WAIT_DLY), self.wait_dly())
        self.subs(self._get_full_path('auxins/0/sample'))

        # Generate more dummy data
        self.auxins_0_averaging(8)

    def acquisition_arm(self, single=True) -> None:
        # time.sleep(0.01)
        self.awgs_0_single(single)
        self.start()

    def acquisition_poll(self, samples, arm=True,
                         acquisition_time=0.010) -> None:  # FIXME: wrong return type
        """
        Polls the UHFQC for data.

        Args:
            samples (int): the expected number of samples
            arm    (bool): if true arms the acquisition, disable when you
                           need synchronous acquisition with some external dev
            acquisition_time (float): time in sec between polls? # TODO check with Niels H
            timeout (float): time in seconds before timeout Error is raised.

        """
        data = {k: [] for k, dummy in enumerate(self._acquisition_nodes)}

        # Start acquisition
        if arm:
            self.acquisition_arm()

        # Acquire data
        gotem = [False]*len(self._acquisition_nodes)
        accumulated_time = 0

        while accumulated_time < self.timeout() and not all(gotem):
            dataset = self.poll(acquisition_time)

            # Enable the user to interrupt long (or buggy) acquisitions
            try:
                check_keyboard_interrupt()
            except KeyboardInterrupt as e:
                # Finalize acquisition before raising exception
                self.acquisition_finalize()
                raise e

            for n, p in enumerate(self._acquisition_nodes):
                if p in dataset:
                    for v in dataset[p]:
                        data[n] = np.concatenate((data[n], v['vector']))
                        if len(data[n]) >= samples:
                            gotem[n] = True
            accumulated_time += acquisition_time

        if not all(gotem):
            self.acquisition_finalize()
            for n, _c in enumerate(self._acquisition_nodes):
                if n in data:
                    print("\t: Channel {}: Got {} of {} samples".format(
                          n, len(data[n]), samples))
            raise TimeoutError("Error: Didn't get all results!")

        return data

    def acquisition_finalize(self) -> None:
        self.stop()

        for p in self._acquisition_nodes:
            self.unsubs(p)
        self.unsubs(self._get_full_path('auxins/0/sample'))

    def check_errors(self) -> None:
        """
        Checks the instrument for errors. As the UHFQA does not yet support the same error
        stack as the HDAWG instruments we do the checks by reading specific nodes
        in the system and then constructing similar messages as on the HDAWG.
        """
        # If this is the first time we are called, log the detected errors, but don't raise
        # any exceptions
        if self._errors is None:
            raise_exceptions = False
            self._errors = {}
        else:
            raise_exceptions = True

        # Stores the errors before processing
        errors = {'messages': []}

        # Now check for errors from the different functional units
        if self.qas_0_result_errors() > 0:
            errors['messages'].append({
                'code': 'RESHOLDOFF',
                'severity': 1.0,
                'count': self.qas_0_result_errors(),
                'message': 'Holdoff error detected when reading Quantum Analyzer Results! '
                'Increase the delay between trigger signals from the AWG!'})

        if self.qas_0_monitor_errors() > 0:
            errors['messages'].append({
                'code': 'MONHOLDOFF',
                'severity': 1.0,
                'count': self.qas_0_monitor_errors(),
                'message': 'Holdoff error detected when reading Quantum Analyzer Input Monitor! '
                'Increase the delay between trigger signals from the AWG!'})

        # Check optional codeword-based errors
        if self._awg_program_features['cases'] and self.get('awgs_0_userregs_{}'.format(UHFQC.USER_REG_ERR_CNT)) > 0:
            errors['messages'].append({
                'code': 'DIOCWCASE',
                'severity': 1.0,
                'count': self.get('awgs_0_userregs_{}'.format(UHFQC.USER_REG_ERR_CNT)),
                'message': 'AWG detected invalid codewords not covered by the configured cases!'})

        # Asserted in case errors were found
        found_errors = False

        # Go through the errors and update our structure, raise exceptions if anything changed
        for m in errors['messages']:
            code = m['code']
            count = m['count']
            severity = m['severity']
            message = m['message']

            if not raise_exceptions:
                self._errors[code] = {
                    'count': count,
                    'severity': severity,
                    'message': message}
                log.warning('{}: Code {}: "{}" ({})'.format(
                    self.devname, code, message, severity))
            else:
                # Optionally skip the error completely
                if code in self._errors_to_ignore:
                    continue

                # Check if there are new errors
                if code not in self._errors or count > self._errors[code]['count']:
                    log.error('{}: {} ({}/{})'.format(self.devname,
                                                      message, code, severity))
                    found_errors = True

                if code in self._errors:
                    self._errors[code]['count'] = count
                else:
                    self._errors[code] = {
                        'count': count,
                        'severity': severity,
                        'message': message}

        # if found_errors:
        #     raise zibase.ziRuntimeError('Errors detected during run-time!')

    def clear_errors(self) -> None:
        self.qas_0_result_reset(1)
        self.qas_0_monitor_reset(1)

    ##########################################################################
    # 'public' functions: DIO support
    ##########################################################################

    def plot_dio(self, bits=range(32), line_length=64):
        data = self.getv('awgs/0/dio/data')
        zibase.plot_timing_diagram(data, bits, line_length)

    ##########################################################################
    # 'public' functions: weight & matrix function helpers
    ##########################################################################

    def prepare_SSB_weight_and_rotation(self, IF,
                                        weight_function_I=0,
                                        weight_function_Q=1,
                                        rotation_angle=0,
                                        length=4096 / 1.8e9,
                                        scaling_factor=1) -> None:
        """
        Sets default integration weights for SSB modulation, beware does not
        load pulses or prepare the UFHQC progarm to do data acquisition
        """
        trace_length = 4096
        tbase = np.arange(0, trace_length / 1.8e9, 1 / 1.8e9)
        cosI = np.array(np.cos(2 * np.pi * IF * tbase + rotation_angle))
        sinI = np.array(np.sin(2 * np.pi * IF * tbase + rotation_angle))
        if length < 4096 / 1.8e9:
            max_sample = int(length * 1.8e9)
            # setting the samples beyond the length to 0
            cosI[max_sample:] = 0
            sinI[max_sample:] = 0
        self.set('qas_0_integration_weights_{}_real'.format(weight_function_I),
                 np.array(cosI))
        self.set('qas_0_integration_weights_{}_imag'.format(weight_function_I),
                 np.array(sinI))
        self.set('qas_0_rotations_{}'.format(
            weight_function_I), scaling_factor*(1.0 + 1.0j))
        if weight_function_Q != None:
            self.set('qas_0_integration_weights_{}_real'.format(weight_function_Q),
                     np.array(sinI))
            self.set('qas_0_integration_weights_{}_imag'.format(weight_function_Q),
                     np.array(cosI))
            self.set('qas_0_rotations_{}'.format(
                weight_function_Q), scaling_factor*(1.0 - 1.0j))

    def prepare_DSB_weight_and_rotation(self, IF, weight_function_I=0, weight_function_Q=1) -> None:
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.array(np.cos(2 * np.pi*IF*tbase))
        sinI = np.array(np.sin(2 * np.pi*IF*tbase))
        self.set('qas_0_integration_weights_{}_real'.format(weight_function_I),
                 np.array(cosI))
        self.set('qas_0_integration_weights_{}_real'.format(weight_function_Q),
                 np.array(sinI))
        # the factor 2 is needed so that scaling matches SSB downconversion
        self.set('qas_0_rotations_{}'.format(weight_function_I), 2.0 + 0.0j)
        self.set('qas_0_rotations_{}'.format(weight_function_Q), 2.0 + 0.0j)

    def upload_crosstalk_matrix(self, matrix) -> None:
        """
        Upload parameters for the 10*10 crosstalk suppression matrix.

        This method uses the 'qas_0_crosstalk_rows_*_cols_*' nodes.
        """
        for i in range(np.shape(matrix)[0]):  # looping over the rows
            for j in range(np.shape(matrix)[1]):  # looping over the colums
                self.set('qas_0_crosstalk_rows_{}_cols_{}'.format(
                    j, i), matrix[i][j])

    def download_crosstalk_matrix(self, nr_rows=10, nr_cols=10):
        """
        Upload parameters for the 10*10 crosstalk suppression matrix.

        This method uses the 'qas_0_crosstalk_rows_*_cols_*' nodes.
        """
        matrix = np.zeros([nr_rows, nr_cols])
        for i in range(np.shape(matrix)[0]):  # looping over the rows
            for j in range(np.shape(matrix)[1]):  # looping over the colums
                matrix[i][j] = self.get(
                    'qas_0_crosstalk_rows_{}_cols_{}'.format(j, i))
        return matrix

    ##########################################################################
    """
    'public' functions: sequencer functions
    Before acquisition can take place one of "awg_sequence_acquisition_and_"
    has to be called. These take care that the right program is uploaded.
    The variants are:
        awg_sequence_acquisition
            start acquisition after receiving a trigger, play no pulse
        awg_sequence_acquisition_and_pulse
            start acquisition after receiving a trigger,
            play the specified pulse
        awg_sequence_acquisition_and_pulse_SSB
            start acquisition after receiving a trigger,
            play an SSB pulse based on specified parameters
        awg_sequence_acquisition_and_DIO_triggered_pulse
            start acquisition after receiving a DIO trigger,
            play the pulse specified by the received DIO codeword
            cases argument specifies what codewords are supported.
        awg_sequence_acquisition_and_DIO_RED_test
            special DIO acquisition for testing real time error correction.
    """
    ##########################################################################

    def awg_sequence_acquisition_and_DIO_triggered_pulse(
            self, Iwaves=None, Qwaves=None, cases=None, acquisition_delay=0, timeout=5) -> None:
        """
        Loads the program for DIO acquisition on the AWG of the UHFQC.

        Arguments:
            Iwaves list of I waveforms (arrays) used (historical).
            Qwaves list of Q waveforms (arrays) used (historical).
            cases list of cases to include in the program.

        Uploads and compiles the AWG sequencer program.


        """
        # setting the acquisition delay samples
        delay_samples = int(acquisition_delay*1.8e9/8)
        self.wait_dly(delay_samples)

        # If no cases are defined, then we simply create all possible cases
        if cases is None:
            cases = np.arange(self._num_codewords)
        else:
            if len(cases) > self._num_codewords:
                raise zibase.ziConfigurationError('More cases ({}) defined than available codewords ({})!'.format(
                    len(cases), len(self._num_codewords)))

            # There is probably a more efficient way of doing this
            for case in cases:
                if (case < 0) or (case >= self._num_codewords):
                    raise zibase.ziConfigurationError(
                        'Case {} is out of range defined by the available codewords ({})!'.format(case, len(self._num_codewords)))

        # Sanity check on the parameters
        if Iwaves is not None and (len(Iwaves) != len(cases)):
            raise ziUHFQCSeqCError(
                'Number of I channel waveforms ({}) does not match number of cases ({})!'.format(len(Iwaves), len(cases)))

        if Qwaves is not None and (len(Qwaves) != len(cases)):
            raise ziUHFQCSeqCError(
                'Number of Q channel waveforms ({}) does not match number of cases ({})!'.format(len(Iwaves), len(cases)))

        # Sanity check on I channel waveforms
        if Iwaves is not None:
            for i, Iwave in enumerate(Iwaves):
                if np.max(Iwave) > 1.0 or np.min(Iwave) < -1.0:
                    raise KeyError(
                        "exceeding AWG range for I channel, all values should be within +/-1")
                if len(Iwave) > 16384:
                    raise KeyError(
                        "exceeding max AWG wave length of 16384 samples for I channel, trying to upload {} samples".format(len(Iwave)))

                # Update waveform table
                self.set(zibase.gen_waveform_name(0, cases[i]), Iwave)

        # Sanity check on Q channel waveforms
        if Qwaves is not None:
            for i, Qwave in enumerate(Qwaves):
                if np.max(Qwave) > 1.0 or np.min(Qwave) < -1.0:
                    raise KeyError(
                        "exceeding AWG range for Q channel, all values should be within +/-1")
                if len(Qwave) > 16384:
                    raise KeyError(
                        "exceeding max AWG wave length of 16384 samples for I channel, trying to upload {} samples".format(len(Qwave)))

                # Update waveform table
                self.set(zibase.gen_waveform_name(1, cases[i]), Qwave)

        # Define the behavior of our program
        self._reset_awg_program_features()
        self._awg_program_features['loop_cnt'] = True
        self._awg_program_features['wait_dly'] = True
        self._awg_program_features['waves'] = True
        self._awg_program_features['cases'] = True

        # Updating cases will cause our AWG program to update
        self.cases(cases)

    def awg_sequence_acquisition_and_DIO_RED_test(
            self, Iwaves=None, Qwaves=None, cases=None, acquisition_delay=0,
            codewords=None, timeout=5):

        # setting the acquisition delay samples
        delay_samples = int(acquisition_delay*1.8e9/8)
        # setting the delay in the instrument
        self.awgs_0_userregs_2(delay_samples)
        sequence = (
            'var wait_delay = getUserReg(2);\n' +
            'cvar i = 0;\n'+
            'const length = {};\n'.format(len(codewords))
            )
        sequence = sequence + array2vect(
                codewords, "codewords")
        # starting the loop and switch statement
        sequence = sequence +(
            ' setDIO(2048);\n'+
            'for (i = 0; i < length; i = i + 1) {\n'
            ' var codeword =  codewords[i];\n'+
            ' waitDIOTrigger();\n' +
            ' setDIO(codeword);\n'+
            ' wait(wait_delay);\n' +
            ' setDIO(2048);\n'+
            '}\n'
            )

        # Define the behavior of our program
        self._reset_awg_program_features()

        self._awg_program[0] = sequence
        self._awg_needs_configuration[0] = True
        # self.awg_string(sequence, timeout=timeout)

    def awg_sequence_acquisition_and_pulse(self, Iwave=None, Qwave=None, acquisition_delay=0, dig_trigger=True) -> None:
        if Iwave is not None and (np.max(Iwave) > 1.0 or np.min(Iwave) < -1.0):
            raise KeyError(
                "exceeding AWG range for I channel, all values should be within +/-1")

        if Qwave is not None and (np.max(Qwave) > 1.0 or np.min(Qwave) < -1.0):
            raise KeyError(
                "exceeding AWG range for Q channel, all values should be within +/-1")

        if Iwave is not None and (len(Iwave) > 16384):
            raise KeyError(
                "exceeding max AWG wave length of 16384 samples for I channel, trying to upload {} samples".format(len(Iwave)))

        if Qwave is not None and (len(Qwave) > 16384):
            raise KeyError(
                "exceeding max AWG wave length of 16384 samples for Q channel, trying to upload {} samples".format(len(Qwave)))

        # Check the we have sufficient codewords defined
        if self._num_codewords < 1:
            raise zibase.ziConfigurationError(
                'Insufficient number of codewords defined! Need at least 1 codeword.')

        # Configure the actual waveforms
        if Iwave is not None:
            self.set(zibase.gen_waveform_name(0, 0), Iwave)

        if Qwave is not None:
            self.set(zibase.gen_waveform_name(1, 0), Qwave)

        # Configure the delay
        self.set('awgs_0_userregs_{}'.format(UHFQC.USER_REG_WAIT_DLY),
                 int(acquisition_delay*1.8e9/8))

        delay_string = """
    wait(wait_dly);
"""

        playWave_string = """
    playWave({}, {});
        """.format(zibase.gen_waveform_name(0, 0), zibase.gen_waveform_name(1, 0))

        if dig_trigger:
            loop_start = """
repeat (loop_cnt) {
    waitDigTrigger(1, 1);
"""
        else:
            loop_start = """
repeat (loop_cnt) {
"""
        loop_end = """
    setTrigger(ro_trig);
    setTrigger(ro_arm);
    waitWave();
    wait(4000);
}
setTrigger(0);
"""

        self._reset_awg_program_features()
        self._awg_program_features['loop_cnt'] = True
        self._awg_program_features['wait_dly'] = True
        self._awg_program_features['waves'] = True

        self._awg_program[0] = \
            awg_sequence_acquisition_preamble() + \
            loop_start + \
            playWave_string + \
            delay_string + \
            loop_end

        self._awg_needs_configuration[0] = True

    def awg_sequence_acquisition(self):
        self._reset_awg_program_features()
        self._awg_program_features['loop_cnt'] = True

        self._awg_program[0] = awg_sequence_acquisition_preamble() + """
repeat (loop_cnt) {
    waitDigTrigger(1, 1);
    setTrigger(ro_trig);
    setTrigger(ro_arm);
}
setTrigger(0);
"""
        # Reset delay
        self.wait_dly(0)
        self._awg_needs_configuration[0] = True

    def awg_sequence_acquisition_and_pulse_SSB(
            self, f_RO_mod, RO_amp, RO_pulse_length, acquisition_delay, dig_trigger=True) -> None:
        f_sampling = 1.8e9
        samples = RO_pulse_length*f_sampling
        array = np.arange(int(samples))
        sinwave = RO_amp * np.sin(2 * np.pi*array*f_RO_mod/f_sampling)
        coswave = RO_amp * np.cos(2 * np.pi*array*f_RO_mod/f_sampling)
        Iwave = (coswave+sinwave) / np.sqrt(2)
        Qwave = (coswave-sinwave) / np.sqrt(2)
        self.awg_sequence_acquisition_and_pulse(
            Iwave, Qwave, acquisition_delay, dig_trigger=dig_trigger)

    def spec_mode_on(self, acq_length=1/1500, IF=20e6, ro_amp=0.1, wint_length=2**14) -> None:
        self._reset_awg_program_features()
        self._awg_program_features['loop_cnt'] = True
        self._awg_program_features['avg_cnt'] = True
        self._awg_program_features['waves'] = True

        # Reset delay
        self.wait_dly(0)

        # Check the we have sufficient codewords defined
        if self._num_codewords < 1:
            raise zibase.ziConfigurationError(
                'Insufficient number of codewords defined! Need at least 1 codeword.')

        # Define number of samples
        N = 16

        # Define alpha parameter
        alpha = 0.2

        # Define support parameters
        a0 = (1-alpha)/2
        a1 = 1/2
        a2 = alpha/2

        # Generate window function
        w = a0 - \
            a1 * np.cos(2 * np.pi * np.arange(N)/(N-1)) + \
            a2 * np.cos(4 * np.pi * np.arange(N)/(N-1))

        # Configure the actual waveforms
        self.set(zibase.gen_waveform_name(0, 0), w)
        self.set(zibase.gen_waveform_name(1, 0), w)

        playWave_string = """
    playWave({}, {});
        """.format(zibase.gen_waveform_name(0, 0), zibase.gen_waveform_name(1, 0))

        wait_string = """
    waitQAResultTrigger();
    wait(16);
        """

        self._awg_program[0] = awg_sequence_acquisition_preamble() + """
repeat (avg_cnt) {
  var wait_time = 0;

  repeat(loop_cnt) {
    wait_time = wait_time + 1;
    setTrigger(ro_trig);
    setTrigger(ro_arm);
    wait(wait_time);
""" + playWave_string + wait_string + """
  }
}
setTrigger(0);
"""

        # Also added by us
        self.awgs_0_outputs_0_mode(1)
        self.awgs_0_outputs_1_mode(1)

        # setting the internal oscillator to the IF
        self.oscs_0_freq(IF)

        self.sigouts_0_on(1)
        self.sigouts_1_on(1)

        # QuExpress thresholds on DIO (mode == 2), AWG control of DIO (mode == 1)
        self.dios_0_mode(2)
        # Drive DIO bits 31 to 16
        self.dios_0_drive(0xc)

        # setting the integration path to use the oscillator instead of
        # integration functions. Should be done before modifying the length.
        self.qas_0_integration_mode(1)

        self.qas_0_deskew_rows_0_cols_0(1.0)
        self.qas_0_deskew_rows_0_cols_1(0.0)
        self.qas_0_deskew_rows_1_cols_0(0.0)
        self.qas_0_deskew_rows_1_cols_1(1.0)
        self.qas_0_integration_length(wint_length)
        self.qas_0_delay(0)

        # Copy from the manual
        self.qas_0_rotations_0(1.0 + 0.0j)
        self.qas_0_rotations_1(0.0 + 1.0j)

        for i in range(0, 10):
            for j in range(0, 10):
                self.set('qas_0_crosstalk_rows_{0}_cols_{1}'.format(
                    i, j), 1.0*(i == j))

        # Configure some thresholds
        for i in range(0, 10):
            self.set('qas_0_thresholds_{}_level'.format(i), 0.01)

        # Also adder by us
        # result_source 0 => lin_trans readout(includes crosstalk corr)
        self.qas_0_result_source(0)
        self.qas_0_result_enable(1)
        self.qas_0_result_statistics_enable(0)

        self._awg_needs_configuration[0] = True

    def spec_mode_off(self) -> None:
        # Resetting To regular Mode
        # changing int length
        self.qas_0_integration_mode(0)

        # Default settings copied
        self.qas_0_rotations_0(1.0 + 0.0j)
        self.qas_0_rotations_1(1.0 + 0.0j)

        # setting to DSB by default
        self.qas_0_deskew_rows_0_cols_0(1.0)
        self.qas_0_deskew_rows_0_cols_1(0.0)
        self.qas_0_deskew_rows_1_cols_0(0.0)
        self.qas_0_deskew_rows_1_cols_1(1.0)

        # switching off the modulation tone
        self.awgs_0_outputs_0_mode(0)
        self.awgs_0_outputs_1_mode(0)

    def plot_dio_snapshot(self, bits=range(32)):
        zibase.plot_timing_diagram(self.getv('awgs/0/dio/data'), bits, 64)

    ##########################################################################
    # 'public' functions: print overview helpers
    ##########################################################################

    def print_correlation_overview(self):
        msg = '\tCorrelations overview \n'
        for i in range(10):
            enabled = self.get('qas_0_correlations_{}_enable'.format(i))
            source = self.get('qas_0_correlations_{}_source'.format(i))
            msg += "Correlations {}, enabled: {} \tsource: {}\n".format(
                i, enabled, source)
        msg += '\n\tThresholded correlations overview \n'
        for i in range(10):
            enabled = self.get(
                'qas_0_thresholds_{}_correlation_enable'.format(i))
            source = self.get(
                'qas_0_thresholds_{}_correlation_source'.format(i))
            msg += "Thresholds correlation {}, enabled: {} \tsource: {}\n".format(
                i, enabled, source)
        print(msg)

    def print_deskew_overview(self):
        msg = '\tDeskew overview \n'

        deskew_mat = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                deskew_mat[i, j] = self.get(
                    'qas_0_deskew_rows_{}_cols_{}'.format(i, j))
        msg += 'Deskew matrix: \n'
        msg += str(deskew_mat)
        print(msg)

    def print_crosstalk_overview(self):
        msg = '\tCrosstalk overview \n'
        msg += 'Bypass crosstalk: {} \n'.format(self.qas_0_crosstalk_bypass())

        crosstalk_mat = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                crosstalk_mat[i, j] = self.get(
                    'qas_0_crosstalk_rows_{}_cols_{}'.format(i, j))
        msg += 'Crosstalk matrix: \n'
        print(msg)
        print(crosstalk_mat)

    def print_integration_overview(self):
        msg = '\tIntegration overview \n'
        msg += 'Integration mode: {} \n'.format(
            self.qas_0_integration_mode())
        for i in range(10):
            msg += 'Integration source {}: {}\n'.format(
                i, self.get('qas_0_integration_sources_{}'.format(i)))
        print(msg)

    def print_rotations_overview(self):
        msg = '\tRotations overview \n'
        for i in range(10):
            msg += 'Rotations {}: {}\n'.format(
                i, self.get('qas_0_rotations_{}'.format(i)))
        print(msg)

    def print_thresholds_overview(self):
        msg = '\t Thresholds overview \n'
        for i in range(10):
            msg += 'Threshold {}: {}\n'.format(
                i, self.get('qas_0_thresholds_{}_level'.format(i)))
        print(msg)

    def print_user_regs_overview(self):
        msg = '\t User registers overview \n'
        user_reg_funcs = ['']*16
        user_reg_funcs[0] = 'Loop count'
        user_reg_funcs[1] = 'Readout mode'
        user_reg_funcs[2] = 'Wait delay'
        user_reg_funcs[3] = 'Average count'
        user_reg_funcs[4] = 'Error count'

        for i in range(16):
            msg += 'User reg {}: \t{}\t({})\n'.format(
                i, self.get('awgs_0_userregs_{}'.format(i)), user_reg_funcs[i])
        print(msg)

    def print_overview(self):
        """
        Print a readable overview of relevant parameters of the UHFQC.

        N.B. This overview is not complete, but combines different
        print helpers
        """
        self.print_correlation_overview()
        self.print_crosstalk_overview()
        self.print_deskew_overview()
        self.print_integration_overview()
        self.print_rotations_overview()
        self.print_thresholds_overview()
        self.print_user_regs_overview()

    ##########################################################################
    # DIO calibration functions
    ##########################################################################

    def _ensure_activity(self, awg_nr, timeout=5, verbose=False):
        """
        Record DIO data and test whether there is activity on the bits activated in the DIO protocol for the given AWG.
        """
        if verbose: print("Testing DIO activity for AWG {}".format(awg_nr))

        vld_mask     = 1 << self.geti('awgs/{}/dio/valid/index'.format(awg_nr))
        vld_polarity = self.geti('awgs/{}/dio/valid/polarity'.format(awg_nr))
        strb_mask    = (1 << self.geti('awgs/{}/dio/strobe/index'.format(awg_nr)))
        strb_slope   = self.geti('awgs/{}/dio/strobe/slope'.format(awg_nr))

        # Make sure the DIO calibration mask is configured
        if self._dio_calibration_mask is None:
            raise ValueError('DIO calibration bit mask not defined.')

        mask_value = self._dio_calibration_mask
        cw_mask = mask_value << 17

        for i in range(timeout):
            valid = True

            data = self.getv('awgs/0/dio/data')
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

    def _get_awg_dio_data(self, awg):
        data = self.getv('awgs/' + str(awg) + '/dio/data')
        ts = len(data)*[0]
        cw = len(data)*[0]
        for n, d in enumerate(data):
            ts[n] = d >> 10
            cw[n] = (d & ((1 << 10)-1))
        return (ts, cw)

    def _find_valid_delays(self, awg_nr, repetitions=1, verbose=False):
        """Finds valid DIO delay settings for a given AWG by testing all allowed delay settings for timing violations on the
        configured bits. In addition, it compares the recorded DIO codewords to an expected sequence to make sure that no
        codewords are sampled incorrectly."""
        if verbose: print("  Finding valid delays")

        vld_mask     = 1 << self.geti('awgs/{}/dio/valid/index'.format(awg_nr))
        vld_polarity = self.geti('awgs/{}/dio/valid/polarity'.format(awg_nr))
        strb_mask    = (1 << self.geti('awgs/{}/dio/strobe/index'.format(awg_nr)))
        strb_slope   = self.geti('awgs/{}/dio/strobe/slope'.format(awg_nr))

        # Make sure the DIO calibration mask is configured
        if self._dio_calibration_mask is None:
            raise ValueError('DIO calibration bit mask not defined.')

        mask_value = self._dio_calibration_mask
        cw_mask = mask_value << 17

        combined_mask = cw_mask
        if vld_polarity != 0:
            combined_mask |= vld_mask
        if strb_slope != 0:
            combined_mask |= strb_mask
        if verbose: print("  Using a mask value of 0x{:08x}".format(combined_mask))

        valid_delays= []
        for delay in range(16):
            if verbose: print('   Testing delay {}'.format(delay))
            self.setd('raw/dios/0/delay', delay)
            time.sleep(1)
            valid_sequence = True
            for awg in [0]:
                error_timing = self.geti('raw/dios/0/error/timing')
                if error_timing & combined_mask != 0:
                    valid_sequence = False

            if valid_sequence:
                valid_delays.append(delay)

        return set(valid_delays)

    ##########################################################################
    # DIO calibration functions for *CC*
    # FIXME: should not be in driver
    ##########################################################################

    def _prepare_CCL_dio_calibration(self, CCL, feedline=1, verbose=False):
        """Configures a CCL with a default program that generates data suitable for DIO calibration.
        Also starts the program."""
        cs_filepath = os.path.join(pycqed.__path__[0],
                'measurement',
                'openql_experiments',
                'output', 'cs.txt')

        opc_filepath = os.path.join(pycqed.__path__[0],
                'measurement',
                'openql_experiments',
                'output', 'qisa_opcodes.qmap')

        CCL.control_store(cs_filepath)
        CCL.qisa_opcode(opc_filepath)

        test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..',
                'examples','CCLight_example',
                'qisa_test_assembly','calibration_cws_ro.qisa'))

        # Start the CCL with the program configured above
        CCL.eqasm_program(test_fp)
        CCL.start()

        # Set the DIO calibration mask to enable 5 bit measurement
        if feedline == 1:
            self._dio_calibration_mask = 0x1f
        elif feedline == 2:
            self._dio_calibration_mask = 0x3
        else:
            raise ValueError('Invalid feedline {} selected for calibration.'.format(feedline))

    def _prepare_CC_dio_calibration(self, CC, verbose=False):
        test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                                      '..', 'examples','CC_examples',
                                      'uhfqc_calibration.vq1asm'))

        # Set the DIO calibration mask to enable 9 bit measurement
        self._dio_calibration_mask = 0x1ff
        CC.eqasm_program(test_fp)
        CC.start()

    def _prepare_QCC_dio_calibration(self, QCC, verbose=False):
        """Configures a QCC with a default program that generates data suitable for DIO calibration. Also starts the QCC."""

        cs_filepath = os.path.join(pycqed.__path__[0],
                'measurement',
                'openql_experiments',
                's17', 'cs.txt')

        opc_filepath = os.path.join(pycqed.__path__[0],
                'measurement',
                'openql_experiments',
                's17', 'qisa_opcodes.qmap')

        QCC.control_store(cs_filepath)
        QCC.qisa_opcode(opc_filepath)

        test_fp = os.path.abspath(os.path.join(pycqed.__path__[0],
                '..',
                'examples','QCC_example',
                'qisa_test_assembly','ro_calibration.qisa'))

        # Start the QCC with the program configured above
        QCC.stop()
        QCC.eqasm_program(test_fp)
        QCC.start()

        # Set the DIO calibration mask to enable 9 bit measurement
        self._dio_calibration_mask = 0x1ff

    def _prepare_HDAWG8_dio_calibration(self, HDAWG, verbose=False):
        """Configures an HDAWG with a default program that generates data suitable for DIO calibration. Also starts the HDAWG."""
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
        HDAWG.configure_awg_from_string(0, program)
        HDAWG.seti('awgs/0/enable', 1)

        self._dio_calibration_mask = 0x7fff

    def calibrate_CC_dio_protocol(self, CC, feedline=None, verbose=False, repetitions=1):
        log.info('Calibrating DIO delays')
        if verbose: print("Calibrating DIO delays")
        if feedline is None:
            raise ziUHFQCDIOCalibrationError('No feedline specified for calibration')

        CC_model = CC.IDN()['model']
        if 'QCC' in CC_model:
            self._prepare_QCC_dio_calibration(
                QCC=CC, verbose=verbose)
        elif 'CCL' in CC_model:
            self._prepare_CCL_dio_calibration(
                CCL=CC, feedline=feedline, verbose=verbose)
        elif 'HDAWG8' in CC_model:
            self._prepare_HDAWG8_dio_calibration(HDAWG=CC, verbose=verbose)
        elif 'cc' in CC_model:
            self._prepare_CC_dio_calibration(CC=CC, verbose=verbose)
        else:
            raise ValueError('CC model ({}) not recognized.'.format(CC_model))

        # Make sure the configuration is up-to-date
        self.assure_ext_clock()

        for awg in [0]:
            if not self._ensure_activity(awg, verbose=verbose):
                raise ziUHFQCDIOActivityError('No or insufficient activity found on the DIO bits associated with AWG {}'.format(awg))

        valid_delays = self._find_valid_delays(awg, repetitions, verbose=verbose)
        if len(valid_delays) == 0:
            raise ziUHFQCDIOCalibrationError('DIO calibration failed! No valid delays found')

        subseq = [[]]
        for e in valid_delays:
            if not subseq[-1] or subseq[-1][-1] == e - 1:
                subseq[-1].append(e)
            else:
                subseq.append([e])

        subseq = max(subseq, key=len)
        delay = len(subseq)//2 + subseq[0]

        # Print information
        if verbose: print("  Valid delays are {}".format(valid_delays))
        if verbose: print("  Setting delay to {}".format(delay))

        # And configure the delays
        self._set_dio_calibration_delay(delay)

        # Clear all detected errors (caused by DIO timing calibration)
        self.clear_errors()

