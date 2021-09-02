"""
    Base driver for the UHFQA instrument including all common functionality.
    Application dependent code can be found in the UHFQuantumController and in the
    UHFQA_qudev modules. 
"""

import time
import logging
import numpy as np

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
from pycqed.utilities.general import check_keyboard_interrupt

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
# Class
##########################################################################

class UHFQA_core(zibase.ZI_base_instrument):
    """
    This is the base PycQED driver for the 1.8 Gsample/s UHFQA developed
    by Zurich Instruments. The class implements functionality that os
    by both the DCL and QuDev versions of the UHFQA driver.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5/3.6 ucs4 19.05 for 64bit Windows from
        http://www.zhinst.com/downloads, https://people.zhinst.com/~niels/
    2. upload the latest firmware to the UHFQA using the LabOne GUI
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

    def __init__(self,
                 name,
                 device:                  str,
                 interface:               str = 'USB',
                 address:                 str = '127.0.0.1',
                 port:                    int = 8004,
                 nr_integration_channels: int = 10,
                 server:                  str = '',
                 **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument
            device          (str) the name of the device e.g., "dev8008"
            interface       (str) the name of the interface to use ('1GbE' or 'USB')
            address         (str) the host where the ziDataServer is running (for compatibility)
            port            (int) the port to connect to for the ziDataServer (don't change)
            nr_integration_channels (int) the number of integration channels to use (max 10)
            server:         (str) the host where the ziDataServer is running (if not '' then used instead of address)
        """
        t0 = time.time()

        # Override server with the old-style address argument
        if server == '':
            server = address

        # save some parameters
        self._nr_integration_channels = nr_integration_channels

        # Used for keeping track of which nodes we are monitoring for data
        self._acquisition_nodes = []

        # The following members define the characteristics of the configured
        # AWG program
        self._reset_awg_program_features()

        # Define parameters that should not be part of the snapshot
        self._params_to_exclude = set(['features_code', 'system_fwlog', 'system_fwlogenable'])

        # Set default waveform length to 20 ns at 1.8 GSa/s
        self._default_waveform_length = 32

        # Our base class includes all the functionality needed to initialize the parameters
        # of the object. Those parameters are read from instrument-specific JSON files stored
        # in the zi_parameter_files folder.
        super().__init__(name=name, device=device, interface=interface,
                         server=server, port=port, num_codewords=2**nr_integration_channels,
                         **kw)

        t1 = time.time()
        log.info(f'{self.devname}: Initialized UHFQA_core in {t1 - t0:.3f}s')

    ##########################################################################
    # Overriding private ZI_base_instrument methods
    ##########################################################################

    def _check_devtype(self) -> None:
        if self.devtype != 'UHFQA':
            raise zibase.ziDeviceError(
                'Device {} of type {} is not a UHFQA instrument!'.format(self.devname, self.devtype))

    def _check_options(self) -> None:
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

    def _check_versions(self) -> None:
        """
        Checks that sufficient versions of the firmware are available.
        """
        if self.geti('system/fwrevision') < UHFQA_core.MIN_FWREVISION:
            raise zibase.ziVersionError('Insufficient firmware revision detected! Need {}, got {}!'.format(
                UHFQA_core.MIN_FWREVISION, self.geti('system/fwrevision')))

        if self.geti('system/fpgarevision') < UHFQA_core.MIN_FPGAREVISION:
            raise zibase.ziVersionError('Insufficient FPGA revision detected! Need {}, got {}!'.format(
                UHFQA_core.MIN_FPGAREVISION, self.geti('system/fpgarevision')))

    def _check_awg_nr(self, awg_nr) -> None:
        """
        Checks that the given AWG index is valid for the device.
        """
        if (awg_nr != 0):
            raise zibase.ziValueError(
                'Invalid AWG index of {} detected!'.format(awg_nr))

    def _num_channels(self) -> int:
        return 2

    def _add_extra_parameters(self) -> None:
        """
        We add a few additional custom parameters on top of the ones defined in the device files. These are:
          qas_0_trans_offset_weightfunction - an offset correction parameter for all weight functions,
            this allows normalized calibration when performing cross-talk suppressed readout. The parameter
            is not actually used in this driver, but in some of the support classes that make use of the driver.
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

        log.info(f"{self.devname}: Switching to external clock.")
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
            if self.system_extclk() != 1:
                log.warning(f"{self.devname}: Switching to external clock failed. Trying again.")
            else:
                break
        log.info(f"{self.devname}: Switching to external clock done.")

    def clear_errors(self) -> None:
        super().clear_errors()
        self.qas_0_result_reset(1)
        self.qas_0_monitor_reset(1)

    def load_default_settings(self) -> None:
        # standard configurations adapted from Haandbaek's notebook

        # Setting the clock to external
        self.assure_ext_clock()

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

        # Configure the result logger to not do any averaging
        self.qas_0_result_length(1000)
        self.qas_0_result_averages(pow(2, 10))
        # result_logging_mode 2 => raw (IQ)
        self.qas_0_result_source(2)  # FIXME: not documented in "node_doc_UHFQA.json"

        self.reset_acquisition_params()

        # The custom firmware will feed through the signals on Signal Input 1 to Signal Output 1 and Signal Input 2 to Signal Output 2
        # when the AWG is OFF. For most practical applications this is not really useful. We, therefore, disable the generation of
        # these signals on the output here.
        self.sigouts_0_enables_0(0)
        self.sigouts_0_enables_1(0)
        self.sigouts_1_enables_0(0)
        self.sigouts_1_enables_1(0)

    ##########################################################################
    # 'public' functions
    ##########################################################################

    def clock_freq(self):
        return 1.8e9

    ##########################################################################
    # 'public' functions: utility
    ##########################################################################

    def reset_acquisition_params(self):
        log.info(f'{self.devname}: Setting user registers to 0')
        for i in range(16):
            self.set('awgs_0_userregs_{}'.format(i), 0)

        self.reset_crosstalk_matrix()
        self.reset_correlation_params()
        self.reset_rotation_params()

    def reset_crosstalk_matrix(self):
        self.upload_crosstalk_matrix(np.eye(self._nr_integration_channels))

    def reset_correlation_params(self):
        for i in range(self._nr_integration_channels):
            self.set('qas_0_correlations_{}_enable'.format(i), 0)
            self.set('qas_0_correlations_{}_source'.format(i), 0)
            self.set('qas_0_thresholds_{}_correlation_enable'.format(i), 0)
            self.set('qas_0_thresholds_{}_correlation_source'.format(i), 0)

    def reset_rotation_params(self):
        for i in range(self._nr_integration_channels):
            self.set('qas_0_rotations_{}'.format(i), 1+1j)

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
    # 'public' functions: print overview helpers
    ##########################################################################

    def print_correlation_overview(self) -> None:
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

    def print_deskew_overview(self) -> None:
        msg = '\tDeskew overview \n'

        deskew_mat = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                deskew_mat[i, j] = self.get(
                    'qas_0_deskew_rows_{}_cols_{}'.format(i, j))
        msg += 'Deskew matrix: \n'
        msg += str(deskew_mat)
        print(msg)

    def print_crosstalk_overview(self) -> None:
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

    def print_integration_overview(self) -> None:
        msg = '\tIntegration overview \n'
        msg += 'Integration mode: {} \n'.format(
            self.qas_0_integration_mode())
        for i in range(10):
            msg += 'Integration source {}: {}\n'.format(
                i, self.get('qas_0_integration_sources_{}'.format(i)))
        print(msg)

    def print_rotations_overview(self) -> None:
        msg = '\tRotations overview \n'
        for i in range(10):
            msg += 'Rotations {}: {}\n'.format(
                i, self.get('qas_0_rotations_{}'.format(i)))
        print(msg)

    def print_thresholds_overview(self) -> None:
        msg = '\t Thresholds overview \n'
        for i in range(10):
            msg += 'Threshold {}: {}\n'.format(
                i, self.get('qas_0_thresholds_{}_level'.format(i)))
        print(msg)

    def print_user_regs_overview(self) -> None:
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

    def print_overview(self) -> None:
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
    # 'public' functions: acquisition support
    ##########################################################################

    def acquisition(self, 
                    samples=100, 
                    averages=1, 
                    acquisition_time=0.010, 
                    timeout=10,
                    channels=(0, 1), 
                    mode='rl', 
                    poll=True):
        self.timeout(timeout)
        self.acquisition_initialize(samples, averages, channels, mode, poll)
        if poll:
            data = self.acquisition_poll(samples, True, acquisition_time)
        else:
            data = self.acquisition_get(samples, True, acquisition_time)
        self.acquisition_finalize()

        return data

    def acquisition_initialize(self, 
                               samples, 
                               averages,
                               loop_cnt = None,
                               channels=(0, 1),
                               mode='rl', 
                               poll=True) -> None:
        # Define the channels to use and subscribe to them
        self._acquisition_nodes = []

        # Loop counter of AWG
        if loop_cnt is None:
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
                if poll:
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
                if poll:
                    self.subs(path)

            # Enable automatic readout
            self.qas_0_monitor_reset(1)
            self.qas_0_monitor_enable(1)
            self.qas_0_monitor_length(samples)
            self.qas_0_monitor_averages(averages)
            ro_mode = 1

        self.set('awgs_0_userregs_{}'.format(UHFQA_core.USER_REG_LOOP_CNT), loop_cnt)
        self.set('awgs_0_userregs_{}'.format(UHFQA_core.USER_REG_RO_MODE), ro_mode)
        self.set('awgs_0_userregs_{}'.format(UHFQA_core.USER_REG_AVG_CNT), averages)
        if self.wait_dly() > 0 and not self._awg_program_features['wait_dly']:
            raise ziUHFQCSeqCError(
                'Trying to use a delay of {} using an AWG program that does not use \'wait_dly\'.'.format(self.wait_dly()))
        self.set('awgs_0_userregs_{}'.format(UHFQA_core.USER_REG_WAIT_DLY), self.wait_dly())
        if poll:
            self.subs(self._get_full_path('auxins/0/sample'))

        # Generate more dummy data
        self.auxins_0_averaging(8)
    
    def acquisition_arm(self, single=True) -> None:
        # time.sleep(0.01)
        self.awgs_0_single(single)
        self.start()

    def acquisition_poll(self, samples, arm=True,
                         acquisition_time=0.010):
        """
        Polls the UHFQC for data.

        Args:
            samples (int): the expected number of samples
            arm    (bool): if true arms the acquisition, disable when you
                           need synchronous acquisition with some external dev
            acquisition_time (float): time in sec between polls
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

    def acquisition_get(self, samples, arm=True,
                         acquisition_time=0.010):
        """
        Waits for the UHFQC to finish a measurement then reads the data.

        Args:
            samples (int): the expected number of samples
            arm    (bool): if true arms the acquisition, disable when you
                           need synchronous acquisition with some external dev
            acquisition_time (float): time in sec between polls
            timeout (float): time in seconds before timeout Error is raised.

        """
        data = {n: [] for n in range(len(self._acquisition_nodes))}

        # Start acquisition
        if arm:
            self.acquisition_arm()
            self.sync()

        done = False
        start = time.time()
        while (time.time()-start) < self.timeout():
            status = self.getdeep('awgs/0/sequencer/status')
            if status['value'][0] == 0:
                done = True
                break

        if not done:
            self.acquisition_finalize()
            raise TimeoutError("Error: Didn't get all results!")

        gotem = [False for _ in range(len(self._acquisition_nodes))]
        for n, p in enumerate(self._acquisition_nodes):
            data[n] = self.getv(p)
            if len(data[n]) >= samples:
                gotem[n] = True

        if not all(gotem):
            for n in data.keys():
                print("\t: Channel {}: Got {} of {} samples".format(
                      n, len(data[n]), samples))
            raise TimeoutError("Error: Didn't get all results!")

        return data

    def acquisition_finalize(self) -> None:
        self.stop()
        self.unsubs()

    ##########################################################################
    # Private methods
    ##########################################################################

    def _reset_awg_program_features(self) -> None:
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