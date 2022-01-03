"""
To do:

Notes:

Changelog:
"""

import time
import logging
import inspect
import numpy as np
from typing import Tuple,List

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.SHFQA_core as shf
import pycqed.instrument_drivers.library.DIO as DIO

from qcodes.utils import validators
from qcodes.utils.helpers import full_class

log = logging.getLogger(__name__)

##########################################################################
# Exceptions
##########################################################################

##########################################################################
# Class
##########################################################################

class SHFQA(shf.SHFQA_core, DIO.CalInterface):
    # TODO(TP): Adapt to SHFQA
    """
    This is the PycQED driver for the 2.0 Gsample/s SHFQA developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5/3.6 ucs4 19.05 for 64bit Windows from
        http://www.zhinst.com/downloads, https://people.zhinst.com/~niels/
    2. upload the latest firmware to the SHFQA using the LabOne GUI
    """

    # Constants definitions from "node_doc_SHFQA.json"
    DIOS_0_MODE_MANUAL = 0  # "0": "Manual setting of the DIO output value.",
    DIOS_0_MODE_AWG_SEQ = 1  # "1": "Enables setting of DIO output values by AWG sequencer commands.",
    DIOS_0_MODE_AWG_WAV = 2  # "2": "Enables the output of AWG waveform data as digital pattern on the DIO connector." FIXME: LabOne says: "QA result"
    # FIXME: comments in this file state: QuExpress thresholds on DIO (mode == 2)

    DIOS_0_EXTCLK_50MHZ = 2  # FIXME: not in "node_doc_SHFQA.json"

    AWGS_0_DIO_VALID_POLARITY_NONE = 0  # "0": "None: VALID bit is ignored.",
    AWGS_0_DIO_VALID_POLARITY_HIGH = 1  # "1": "High: VALID bit must be logical high.",
    AWGS_0_DIO_VALID_POLARITY_LOW = 2  # "2": "Low: VALID bit must be logical zero.",
    AWGS_0_DIO_VALID_POLARITY_BOTH = 3  # "3": "Both: VALID bit may be logical high or zero."

    SAMPLING_FREQUENCY = shf.SHFQA_core.SAMPLING_FREQUENCY

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
                 nr_integration_channels: int = 10,
                 server:                  str = '',
                 **kw) -> None:
        # TODO(TP): Adapt to SHFQA
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
        
        self._use_dio = use_dio

        # Used for extra DIO output to CC for debugging
        self._diocws = None

        # Holds the DIO calibration delay
        self._dio_calibration_delay = 0

        # Holds the number of configured cases
        self._cases = None

        super().__init__(name=name, device=device, interface=interface, address=address,
                         server=server, port=port, nr_integration_channels=nr_integration_channels,
                         **kw)

        t1 = time.time()
        log.info(f'{self.devname}: Initialized SHFQA in {t1 - t0:.3f}s')

    ##########################################################################
    # 'public' overrides for SHFQA_core
    ##########################################################################

    def load_default_settings(self, upload_sequence=True) -> None:
        # TODO(TP): Adapt to SHFQA
        super().load_default_settings()

        # Load an AWG program
        if upload_sequence:
            self.awg_sequence_acquisition()

        # Configure the codeword protocol
        if self._use_dio:
            self.dios_0_mode(self.DIOS_0_MODE_AWG_WAV)  # QuExpress thresholds on DIO (mode == 2), AWG control of DIO (mode == 1)
            self.dios_0_drive(0x3)  # Drive DIO bits 15 to 0
            self.dios_0_extclk(self.DIOS_0_EXTCLK_50MHZ)  # 50 MHz clocking of the DIO
            self.awgs_0_dio_strobe_slope(0)  # no edge, replaced by dios_0_extclk(2)
            self.awgs_0_dio_strobe_index(15)  # NB: 15 for QCC (was 31 for CCL). Irrelevant now we use 50 MHz clocking
            self.awgs_0_dio_valid_polarity(2)  # high polarity FIXME: does not match AWGS_0_DIO_VALID_POLARITY_HIGH
            self.awgs_0_dio_valid_index(16)

        # No rotation on the output of the weighted integration unit, i.e. take
        # real part of result
        for i in range(0, self._nr_integration_channels):
            self.set('qas_0_rotations_{}'.format(i), 1.0 + 0.0j)
            # remove offsets to weight function
            self.set('qas_0_trans_offset_weightfunction_{}'.format(i), 0.0)

    ##########################################################################
    # 'public' functions: generic AWG/waveform support
    ##########################################################################

    def load_awg_program_from_file(self, filename) -> None:
        # TODO(TP): Adapt to SHFQA
        """
        Loads an awg sequence onto the SHFQA from a text file.
        File needs to obey formatting specified in the manual.
        Only provided for backwards compatibility purposes.
        """
        print(filename)
        with open(filename, 'r') as awg_file:
            self._awg_program[0] = awg_file.read()
            self._awg_needs_configuration[0] = True

    def _do_set_AWG_file(self, filename) -> None:
        # TODO(TP): Adapt to SHFQA
        self.load_awg_program_from_file('UHFLI_AWG_sequences/'+filename)

    def awg_file(self, filename) -> None:
        # TODO(TP): Adapt to SHFQA
        """Only provided for backwards compatibility purposes."""
        self.load_awg_program_from_file(filename)

    def awg_update_waveform(self, index, data) -> None:
        # TODO(TP): Adapt to SHFQA
        raise NotImplementedError(
            'Method not implemented! Please use the corresponding waveform parameters \'wave_chN_cwM\' to update waveforms!')

    ##########################################################################
    # 'public' functions: DIO support
    ##########################################################################

    def plot_dio(self, bits=range(32), line_length=64) -> None:
        # TODO(TP): Adapt to SHFQA
        data = self.getv('awgs/0/dio/data')
        zibase.plot_timing_diagram(data, bits, line_length)

    ##########################################################################
    # 'public' functions: weight & matrix function helpers
    ##########################################################################

    def prepare_SSB_weight_and_rotation(self, IF,
                                        weight_function_I=0,
                                        weight_function_Q=1,
                                        rotation_angle=0,
                                        length=4096 / shf.SHFQA_core.SAMPLING_FREQUENCY,
                                        scaling_factor=1) -> None:
        # TODO(TP): Adapt to SHFQA
# FIXME: merge conflict 20200918
#=======
#    def check_errors(self, errors_to_ignore=None) -> None:
#>>>>>>> ee1ccf208faf635329ea2c979da5757ce4ce8e14
        """
        Sets default integration weights for SSB modulation, beware does not
        load pulses or prepare the UFHQC progarm to do data acquisition
        """
        trace_length = 4096
        tbase = np.arange(0, trace_length / SHFQA.SAMPLING_FREQUENCY, 1 / SHFQA.SAMPLING_FREQUENCY)
        cosI = np.array(np.cos(2 * np.pi * IF * tbase + rotation_angle))
        sinI = np.array(np.sin(2 * np.pi * IF * tbase + rotation_angle))
        if length < 4096 / SHFQA.SAMPLING_FREQUENCY:
            max_sample = int(length * SHFQA.SAMPLING_FREQUENCY)
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
        # TODO(TP): Adapt to SHFQA
        trace_length = 4096
        tbase = np.arange(0, trace_length/SHFQA.SAMPLING_FREQUENCY, 1/SHFQA.SAMPLING_FREQUENCY)
        cosI = np.array(np.cos(2 * np.pi*IF*tbase))
        sinI = np.array(np.sin(2 * np.pi*IF*tbase))
        self.set('qas_0_integration_weights_{}_real'.format(weight_function_I),
                 np.array(cosI))
        self.set('qas_0_integration_weights_{}_real'.format(weight_function_Q),
                 np.array(sinI))
        # the factor 2 is needed so that scaling matches SSB downconversion
        self.set('qas_0_rotations_{}'.format(weight_function_I), 2.0 + 0.0j)
        self.set('qas_0_rotations_{}'.format(weight_function_Q), 2.0 + 0.0j)

    ##########################################################################
    # Overriding private ZI_base_instrument methods
    ##########################################################################

    def _add_extra_parameters(self) -> None:
        # TODO(TP): Adapt to SHFQA
        """
        We add a few additional custom parameters on top of the ones defined in the device files. These are:
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

        self.add_parameter(
            'AWG_file',
            set_cmd=self._do_set_AWG_file,
            docstring='Configures the AWG with a SeqC program from a specific file. '
            'Provided only for backwards compatibility. It is discouraged to use '
            'this parameter unless you know what you are doing',
            vals=validators.Anything())

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

        self.add_parameter(
            'minimum_holdoff',
            get_cmd=self._get_minimum_holdoff,
            unit='s',
            label='Minimum hold-off',
            docstring='Returns the minimum allowed hold-off between two readout operations.',
            vals=validators.Numbers())

    def _codeword_table_preamble(self, awg_nr) -> str:
        # TODO(TP): Adapt to SHFQA
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

    def plot_dio_snapshot(self, bits=range(32)):
        # TODO(TP): Adapt to SHFQA
        zibase.plot_timing_diagram(self.getv('awgs/0/dio/data'), bits, 64)

    ##########################################################################
    # Overriding Qcodes InstrumentBase methods
    ##########################################################################

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update =None,
                      params_to_exclude = None ):
        # TODO(TP): Adapt to SHFQA
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
    # Private methods
    ##########################################################################

    def _reset_awg_program_features(self) -> None:
        # TODO(TP): Adapt to SHFQA
        """
        Resets the self._awg_program_features to disable all features. The SHFQA can be configured with a number
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

    def _set_dio_calibration_delay(self, value) -> None:
        # TODO(TP): Adapt to SHFQA
        # Sanity check the value
        if value < 0 or value > 15:
            raise zibase.ziValueError(
                'Trying to set DIO calibration delay to invalid value! Expected value in range 0 to 15. Got {}.'.format(
                    value))

        log.info(f"{self.devname}: Setting DIO calibration delay to {value}")
        # Store the value
        self._dio_calibration_delay = value

        # And configure the delays
        self.setd('raw/dios/0/delay', self._dio_calibration_delay)

    def _get_dio_calibration_delay(self):
        # TODO(TP): Adapt to SHFQA
        return self._dio_calibration_delay

    def _get_minimum_holdoff(self):
        # TODO(TP): Adapt to SHFQA
        if self.qas_0_result_averages() == 1:
            holdoff = np.max((800, self.qas_0_integration_length(), self.qas_0_delay()+16))/self.clock_freq()
        else:
            holdoff = np.max((2560, self.qas_0_integration_length(), self.qas_0_delay()+16))/self.clock_freq()

        return holdoff

    def _set_wait_dly(self, value) -> None:
        # TODO(TP): Adapt to SHFQA
        self.set('awgs_0_userregs_{}'.format(SHFQA.USER_REG_WAIT_DLY), value)

    def _get_wait_dly(self):
        # TODO(TP): Adapt to SHFQA
        return self.get('awgs_0_userregs_{}'.format(SHFQA.USER_REG_WAIT_DLY))

    def _set_cases(self, value) -> None:
        # TODO(TP): Adapt to SHFQA
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
""".format(self._cw_mask)

        if self._awg_program_features['diocws']:
            self._awg_program[0] += \
                _array2vect(self._diocws, "diocws") + """
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
        # FIXME: note that the actual wave timing (i.e. trigger latency) depends on the number of cases, because the
        #  switch statement generates a tree of if's internally. Consequentially, the maximum repetition rate also depends
        #  on the number of cases.
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
        # TODO(TP): Adapt to SHFQA
        return self._cases

    def _get_waveform_table(self, awg_nr: int) -> list:
        # TODO(TP): Adapt to SHFQA
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
    ##########################################################################
    # Application dependent code starts here:
    # - dedicated sequence programs
    # - DIO support
    # FIXME: move to separate class
    ##########################################################################
    ##########################################################################


    ##########################################################################
    # 'public' functions: sequencer functions
    ##########################################################################
    """
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

    def awg_sequence_acquisition_and_DIO_triggered_pulse(
            self, Iwaves=None, Qwaves=None, cases=None, acquisition_delay=0, timeout=5) -> None:
        # TODO(TP): Adapt to SHFQA
        """
        Loads the program for DIO acquisition on the AWG of the SHFQA.

        Arguments:
            Iwaves list of I waveforms (arrays) used (historical).
            Qwaves list of Q waveforms (arrays) used (historical).
            cases list of cases to include in the program.

        Uploads and compiles the AWG sequencer program.


        """
        # setting the acquisition delay samples
        delay_samples = int(acquisition_delay*SHFQA.SAMPLING_FREQUENCY/8)
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
            raise uhf.ziSHFQASeqCError(
                'Number of I channel waveforms ({}) does not match number of cases ({})!'.format(len(Iwaves), len(cases)))

        if Qwaves is not None and (len(Qwaves) != len(cases)):
            raise uhf.ziSHFQASeqCError(
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
            dio_out_vect=None, timeout=5):
        # TODO(TP): Adapt to SHFQA
        # setting the acquisition delay samples
        delay_samples = int(acquisition_delay*SHFQA.SAMPLING_FREQUENCY/8)
        # setting the delay in the instrument
        self.awgs_0_userregs_2(delay_samples)
        sequence = (
            'var wait_delay = getUserReg(2);\n' +
            'cvar i = 0;\n'+
            'const length = {};\n'.format(len(dio_out_vect))
            )
        sequence = sequence + _array2vect(dio_out_vect, "dio_out_vect")
        # starting the loop
        sequence = sequence +(
            'setDIO(2048); // FIXME: workaround because we cannot use setDIO(0)\n'+
            'for (i = 0; i < length; i = i + 1) {\n'
            ' var dio_out =  dio_out_vect[i];\n'+
            ' waitDIOTrigger();\n' +
            ' setDIO(dio_out);\n'+
            ' wait(wait_delay);\n' +
            ' setDIO(2048);\n'+
            '}\n'
            )

        # Define the behavior of our program
        self._reset_awg_program_features()
        self._awg_program[0] = sequence
        self._awg_needs_configuration[0] = True

    def awg_sequence_test_pattern(
            self,
            dio_out_vect=None):
        # TODO(TP): Adapt to SHFQA
        # setting the acquisition delay samples
        sequence = f"""
        cvar i = 0;
        const length = {len(dio_out_vect)};
        """
        sequence = sequence + _array2vect(dio_out_vect, "dio_out_vect")
        # starting the loop
        sequence = sequence + """
        setDIO(2048); // FIXME: workaround because we cannot use setDIO(0), still required in UHF firmware:65939
        for (i = 0; i < length; i = i + 1) {
          var dio_out =  dio_out_vect[i];
          waitDIOTrigger();
          setDIO(dio_out);
          wait(3);      // ~20 ns pulse time
          setDIO(2048);
        }
        """

        # Define the behavior of our program
        self._reset_awg_program_features()
        self._awg_program[0] = inspect.cleandoc(sequence)
        self._awg_needs_configuration[0] = True

    def awg_sequence_acquisition_and_pulse(self, Iwave=None, Qwave=None, acquisition_delay=0, dig_trigger=True) -> None:
        # TODO(TP): Adapt to SHFQA
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
        self.set('awgs_0_userregs_{}'.format(SHFQA.USER_REG_WAIT_DLY),
                 int(acquisition_delay*SHFQA.SAMPLING_FREQUENCY/8))

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
        # TODO(TP): Adapt to SHFQA
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

    def awg_debug_acquisition(self, dly=0):
        # TODO(TP): Adapt to SHFQA
        self._reset_awg_program_features()
        self._awg_program_features['avg_cnt']  = True
        self._awg_program_features['loop_cnt'] = True
        self._awg_program_features['wait_dly'] = True

        self._awg_program[0] = awg_sequence_acquisition_preamble() + """
repeat (avg_cnt) {
  repeat (loop_cnt) {
      setTrigger(ro_trig);
      setTrigger(ro_arm);
      wait(wait_dly);
  }
}
setTrigger(0);
"""
        # Reset delay
        self.wait_dly(dly)
        self._awg_needs_configuration[0] = True

    def awg_sequence_acquisition_and_pulse_SSB(
            self, f_RO_mod, RO_amp, RO_pulse_length, acquisition_delay, dig_trigger=True) -> None:
        # TODO(TP): Adapt to SHFQA
        samples = RO_pulse_length*SHFQA.SAMPLING_FREQUENCY
        array = np.arange(int(samples))
        sinwave = RO_amp * np.sin(2 * np.pi*array*f_RO_mod/SHFQA.SAMPLING_FREQUENCY)
        coswave = RO_amp * np.cos(2 * np.pi*array*f_RO_mod/SHFQA.SAMPLING_FREQUENCY)
        Iwave = (coswave+sinwave) / np.sqrt(2)
        Qwave = (coswave-sinwave) / np.sqrt(2)
        self.awg_sequence_acquisition_and_pulse(
            Iwave, Qwave, acquisition_delay, dig_trigger=dig_trigger)

    def spec_mode_on(self, acq_length=1/1500, IF=20e6, ro_amp=0.1, wint_length=2**14) -> None:
        # TODO(TP): Adapt to SHFQA
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
        self.dios_0_mode(self.DIOS_0_MODE_AWG_WAV)
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
        # TODO(TP): Adapt to SHFQA
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

    ##########################################################################
    # DIO calibration helpers
    ##########################################################################

    def _ensure_activity(self, awg_nr, mask_value: int, timeout=5):
        # TODO(TP): Adapt to SHFQA
        """
        Record DIO data and test whether there is activity on the bits activated in the DIO protocol for the given AWG.
        """
        log.debug(f"{self.devname}: Testing DIO activity for AWG {awg_nr}")

        vld_mask     = 1 << self.geti('awgs/{}/dio/valid/index'.format(awg_nr))
        vld_polarity = self.geti('awgs/{}/dio/valid/polarity'.format(awg_nr))
        strb_mask    = (1 << self.geti('awgs/{}/dio/strobe/index'.format(awg_nr)))
        strb_slope   = self.geti('awgs/{}/dio/strobe/slope'.format(awg_nr))

        cw_mask = mask_value  # FIXME: changed parameter to define mask that's already shifted in place << 17

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
                log.warning(f"{self.devname}: Did not see all codeword bits toggle! Got 0x{cw_activity:08x}, expected 0x{cw_mask:08x}.")
                valid = False

            if vld_polarity != 0 and vld_activity != vld_mask:
                log.warning("{self.devname}: Did not see valid bit toggle!")
                valid = False

            if strb_slope != 0 and strb_activity != strb_mask:
                log.warning("{self.devname}: Did not see strobe bit toggle!")
                valid = False

            if valid:
                return True

        return False

    def _get_awg_dio_data(self, awg):
        # TODO(TP): Adapt to SHFQA
        data = self.getv('awgs/' + str(awg) + '/dio/data')
        ts = len(data)*[0]
        cw = len(data)*[0]
        for n, d in enumerate(data):
            ts[n] = d >> 10
            cw[n] = (d & ((1 << 10)-1))
        return (ts, cw)

    def _find_valid_delays(self, awg_nr, mask_value: int):
        # TODO(TP): Adapt to SHFQA
        """Finds valid DIO delay settings for a given AWG by testing all allowed delay settings for timing violations on the
        configured bits. In addition, it compares the recorded DIO codewords to an expected sequence to make sure that no
        codewords are sampled incorrectly."""
        log.debug("{self.devname}: Finding valid delays")

        vld_mask     = 1 << self.geti('awgs/{}/dio/valid/index'.format(awg_nr))
        vld_polarity = self.geti('awgs/{}/dio/valid/polarity'.format(awg_nr))
        strb_mask    = (1 << self.geti('awgs/{}/dio/strobe/index'.format(awg_nr)))
        strb_slope   = self.geti('awgs/{}/dio/strobe/slope'.format(awg_nr))

        cw_mask = mask_value << 17

        combined_mask = cw_mask
        if vld_polarity != 0:
            combined_mask |= vld_mask
        if strb_slope != 0:
            combined_mask |= strb_mask
        log.debug(f"{self.devname}:   Using a mask value of 0x{combined_mask:08x}")

        valid_delays= []
        for delay in range(12):  # NB: 16 steps are available, but 2 periods of 20 ns should suffice
            log.debug(f'{self.devname}:    Testing delay {delay}')
            self.setd('raw/dios/0/delay', delay)  # in 1/300 MHz = 3.33 ns steps
            time.sleep(0.5)
            valid_sequence = True
            for awg in [0]:
                error_timing = self.geti('raw/dios/0/error/timing')
                if error_timing & combined_mask != 0:
                    valid_sequence = False

            if valid_sequence:
                valid_delays.append(delay)

        return set(valid_delays)

    ##########################################################################
    # overrides for CalInterface interface
    ##########################################################################

    def output_dio_calibration_data(self, dio_mode: str, port: int=0) -> Tuple[int, List]:
        # TODO(TP): Adapt to SHFQA
        # NB: ignoring dio_mode and port, because we have single mode only
        program = """
        // program: triggered upstream DIO calibration program
        const period = 18;          // 18*4.44 ns = 80 ns, NB: 40 ns is not attainable
        const n1 = 3;               // ~20 ns high time
        const n2 = period-n1-2-1;   // penalties: 2*setDIO, 1*loop
        waitDIOTrigger();
        while (1) {
            setDIO(0x000003FF);     // DV=0x0001, RSLT[8:0]=0x03FE.
            wait(n1);
            setDIO(0x00000000);
            wait(n2);
        }
        """
        self.configure_awg_from_string(0, program)
        # FIXME: set SHFQA0.dios_0_mode(SHFQA0.DIOS_0_MODE_AWG_SEQ), but reset after the calibration is done
        self.seti('awgs/0/enable', 1)  # FIXME: check success, use start()?

        dio_mask = 0x000003FF
        expected_sequence = []
        return dio_mask,expected_sequence

    def calibrate_dio_protocol(self, dio_mask: int, expected_sequence: List, port: int=0):
        # TODO(TP): Adapt to SHFQA
        log.info(f"{self.devname}: Calibrating DIO protocol")
        self.assure_ext_clock()

        # Get the integration length and result enable settings to be able to
        # restore them later
        integration_length = self.get('qas_0_integration_length')
        result_enable = self.get('qas_0_result_enable')
        monitor_enable = self.get('qas_0_monitor_enable')
        awg_enable = self.get('awgs_0_enable')

        try:
          self.set('qas_0_integration_length', 4)
          self.set('qas_0_result_enable', 0)
          self.set('qas_0_monitor_enable', 0)
          self.set('awgs_0_enable', 0)
          
          for awg in [0]:
              if not self._ensure_activity(awg, mask_value=dio_mask):
                  raise uhf.ziSHFQADIOActivityError('No or insufficient activity found on the DIO bits associated with AWG {}'.format(awg))

          valid_delays = self._find_valid_delays(awg, mask_value=dio_mask)
          if len(valid_delays) == 0:
              raise uhf.ziSHFQADIOCalibrationError('DIO calibration failed! No valid delays found')

          # Find center of first valid region
          subseq = [[]]
          for e in valid_delays:
              if not subseq[-1] or subseq[-1][-1] == e - 1:
                  subseq[-1].append(e)
              else:
                  subseq.append([e])

          subseq = max(subseq, key=len)
          delay = len(subseq)//2 + subseq[0]

          # Print information
          log.info(f"{self.devname}: Valid delays are {valid_delays}")

          # And configure the delays
          self._set_dio_calibration_delay(delay)

          # Clear all detected errors (caused by DIO timing calibration)
          self.check_errors(errors_to_ignore=['AWGDIOTIMING'])
        
        finally:
          # Restore settings either in case of an exception or if the DIO
          # routine finishes correctly
          self.set('qas_0_integration_length', integration_length)
          self.set('qas_0_result_enable', result_enable)
          self.set('qas_0_monitor_enable', monitor_enable)
          self.set('awgs_0_enable', awg_enable)

    ##########################################################################
    # DIO calibration functions for *CC*
    ##########################################################################

    def calibrate_CC_dio_protocol(self, CC, feedline=None, verbose=False) -> None:
        # TODO(TP): Adapt to SHFQA
        raise DeprecationWarning("calibrate_CC_dio_protocol is deprecated, use instrument_drivers.library.DIO.calibrate")


##########################################################################
# Module level functions
##########################################################################

def awg_sequence_acquisition_preamble():
    # TODO(TP): Adapt to SHFQA
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

def _array2vect(array, name):
    # TODO(TP): Adapt to SHFQA
    # this function cuts up arrays into several vectors of maximum length 1024 that are joined.
    # this is to avoid python crashes (was found to crash for vectors of
    # length> 1490)
    if len(array) > 1024:
        splitted_array = np.array_split(array, len(array) // 1024)
        string_array = ['\nvect(' + ','.join(['{:.8f}'.format(x)
                                              for x in sub_array]) + ')' for sub_array in splitted_array]
        return 'wave ' + name + ' = join(' + ','.join(string_array) + ');\n'
    else:
        return 'wave ' + name + ' = ' + 'vect(' + ','.join(['{:.8f}'.format(x) for x in array]) + ');\n'
