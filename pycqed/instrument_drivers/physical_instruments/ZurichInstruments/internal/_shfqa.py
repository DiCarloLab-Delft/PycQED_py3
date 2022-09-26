"""Module collecting implementation details of the SHFQA driver."""

import textwrap
import numpy as np
import logging

from dataclasses import dataclass
from abc import ABC, abstractmethod

import zhinst.deviceutils.shfqa as shfqa_utils
from zhinst.utils import wait_for_state_change

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument import (
    ziConfigurationError,
    ziValueError,
)

log = logging.getLogger(__name__)


class DeviceConstants:
    """
    Class collecting device constants under one common namespace.
    """

    class MinRevisions:
        """
        Minimum revision numbers required to operate the driver.
        """

        FW = 63210
        FPGA = 63133

    class GeneratorWaveforms:
        """
        Constants related to generator waveforms.
        """

        MAX_LENGTH = 4096
        GRANULARITY = 4

    class Averaging:
        """
        Constants relating hardware averaging algorithms to their corresponding integer encodings.
        """

        SEQUENTIAL = 1
        CYCLIC = 0

    class Readout:
        """
        Constants related to the readout mode ("rl" in driver lingo) of the SHFQA.
        """

        MAX_LENGTH = 4096
        GRANULARITY = 4
        QACHANNEL_MODE = 1
        RESULT_OF_INTEGRATION = 1
        RESULT_OF_DISCRIMINATION = 3

    class Scope:
        """
        Constants related to the scope monitor mode ("ro" in driver lingo) of the SHFQA.
        """

        MAX_LENGTH = 2 ** 18
        GRANULARITY = 32

    class Spectroscopy:
        """
        Constants related to the spectroscopy mode of the SHFQA.
        """

        MAX_LENGTH = 2 ** 25
        GRANULARITY = 4
        QACHANNEL_MODE = 0
        TRIGGER_SOURCE = 32  # hardwired to channel0_sequencer_trigger0

    class Sequencer:
        NUM_REGISTERS = 16

    SAMPLING_FREQUENCY = shfqa_utils.SHFQA_SAMPLING_FREQUENCY

    class Holdoff:
        """
        Constants related to measurement hold-off.
        """

        MARGIN = 72e-9
        PLAYZERO_GRANULARITY = 16
        MIN_LENGTH = 2048


class DioCalibration:
    """
    Class collecting constants relative to DIO calibration under one common namespace.
    """

    DELAYS = range(16)
    NUM_BITS = 32
    GENERATOR_INDEX = 0
    CC_TO_SHFQA_PROGRAM = "while(1){}"
    OUT_MASK = 0x7FFF
    SHFQA_TO_CC_PROGRAM = """
while (1) {
    setDIO(0x7FFF);
    wait(3);
    setDIO(0x00000000);
    wait(8);
}
        """


def preprocess_generator_waveform(waveform) -> None:
    """
    Validates a candidate complex array candidate for upload onto a generator slot of the device. The
    function additionally snaps the length of the array to the granularity of the generator.

    Args:
        waveform: complex array representing a candidate waveform to be uploaded to a device generator slot.
    """
    granularity = DeviceConstants.GeneratorWaveforms.GRANULARITY
    original_size = len(waveform)
    if (original_size % granularity) != 0:
        log.debug(
            f"Waveform is not a multiple of {granularity} samples, appending zeroes."
        )
        extra_zeroes = granularity - (original_size % granularity)
        waveform = np.concatenate([waveform, np.zeros(extra_zeroes)])

    if original_size > DeviceConstants.GeneratorWaveforms.MAX_LENGTH:
        raise ziValueError(
            f"Exceeding maximum generator wave length of {DeviceConstants.GeneratorWaveforms.MAX_LENGTH} samples "
            f"(trying to upload {len(waveform)} samples)"
        )
    return waveform


def hold_off_length(readout_duration: float) -> int:
    """
    Returns the number of samples to wait for between subsequent readouts to ensure no hold-off errors occur.

    Args:
        readout_duration: total duration of the readout operation.
    """
    readout_length = duration_to_length(
        readout_duration + DeviceConstants.Holdoff.MARGIN
    )
    granularity = DeviceConstants.Holdoff.PLAYZERO_GRANULARITY
    snapped_length = (
        ((readout_length + (granularity - 1))) // granularity
    ) * granularity
    if snapped_length < DeviceConstants.Holdoff.MIN_LENGTH:
        log.debug(
            f"The configured time between readout pulse generation and the end of the integration of {readout_duration} "
            f"seconds is shorter than the minimum holdoff length of {DeviceConstants.Holdoff.MIN_LENGTH} samples with a"
            f" sampling rate of {DeviceConstants.SAMPLING_FREQUENCY}."
        )
        return DeviceConstants.Holdoff.MIN_LENGTH
    return snapped_length


class UserRegisters:
    """
    Class assigning functionality to specific user registers.
    """

    INNER_LOOP = 0
    OUTER_LOOP = 1
    HOLDOFF_DELAY = 2
    NUM_ERRORS = 3


class SeqC:
    """
    Class collecting functionality relating to SHFQA sequencer programs under one common namespace.
    """

    _VAR_INNER_LOOP_SIZE = "inner_loop_size"
    _VAR_OUTER_LOOP_SIZE = "outer_loop_size"
    _VAR_INNER_LOOP_INDEX = "inner_loop_index"
    _VAR_OUTER_LOOP_INDEX = "outer_loop_index"
    _VAR_HOLDOFF_DELAY = "holdoff_delay"
    _VAR_NUM_ERRORS = "num_errors"
    _VAR_CODEWORD = "codeword"
    _VAR_CODEWORD_MASK = "codeword_mask"

    _SPECTROSCOPY_OSCILLATOR_INDEX = 0

    @dataclass(frozen=True)
    class Features:
        """
        Dataclass specifying the main features of a sequencer program. Used to ensure that uploaded sequencer
        programs are consistent with other device configurations at any given time.
        """

        inner_loop: bool
        outer_loop: bool
        spectroscopy: bool
        codewords: bool

    @dataclass(frozen=True)
    class LoopSizes:
        """
        Dataclass specifying the loop counts of the sequencer program. Used to ensure that uploaded sequencer
        programs are consistent with other device configurations at any given time.
        """

        inner: int
        outer: int

    @staticmethod
    def acquisition_and_DIO_triggered_pulse(
        mask: int, shift: int, cases: dict
    ) -> tuple:
        """
        Returns a tuple containing a SeqC.Features instance and a program string specifying an experiment where:

        The acquisition is started after receiving a DIO trigger, with different generator and integrator slots being
        triggered based on the value of the codeword stored within the DIO trigger value. The "num_errors" user register
        is incremented each time a codeword is received that is not included in the list specified by the "cases"
        argument. The total number of readouts is specified by the variable stored in the "num_samples" user register.

        Args:
            cases: dictionary whose entries {codeword: slot_indices} specify which generator and integrator slots to
                   trigger for each codeword.
        """
        features = SeqC.Features(
            inner_loop=True, outer_loop=False, spectroscopy=False, codewords=True
        )

        program = SeqC._preamble(features)
        program += SeqC._var_definition(SeqC._VAR_NUM_ERRORS, 0)
        program += SeqC._set_register(UserRegisters.NUM_ERRORS, 0)

        switch = []
        for case, indices in cases.items():
            startQA = SeqC._readout(
                generator_mask=SeqC._make_mask("QA_GEN_", indices),
                integrator_mask=SeqC._make_mask("QA_INT_", indices),
            )
            case = SeqC._scope(preamble=f"case {bin(case << 17)}:", body=startQA)
            switch.append(case)
        default = SeqC._scope(
            preamble="default:", body=f"{SeqC._VAR_NUM_ERRORS} += 1;\n"
        )
        switch.append(default)
        switch = SeqC._scope(preamble=f"switch({SeqC._VAR_CODEWORD})", body=switch)

        repeat = SeqC._play_zero(SeqC._VAR_HOLDOFF_DELAY)
        repeat += SeqC._wait_dio_trigger()
        repeat += SeqC._var_definition(
            SeqC._VAR_CODEWORD, SeqC._dio_codeword(mask, shift)
        )
        repeat += switch
        repeat = SeqC._scope(
            preamble=f"repeat ({SeqC._VAR_INNER_LOOP_SIZE})", body=repeat
        )

        program += repeat
        program += SeqC._set_register(UserRegisters.NUM_ERRORS, SeqC._VAR_NUM_ERRORS)

        return features, program

    @staticmethod
    def acquisition_and_pulse(
        slot: int,
        dio_trigger: bool = False,
    ) -> tuple:
        """
        Returns a tuple containing a SeqC.Features instance and a program string specifying an experiment where:

        The acquisition is started after optionally receiving a digital trigger, with the readout pulse
        and integration weights stored in the first generator and integrator slot, respectively.
        The total number of readouts is specified by the variable stored in the "num_samples" user
        register.

        Args:
            dio_trigger: specify whether or not the readout gets triggered by DIO.
        """
        features = SeqC.Features(
            inner_loop=True, outer_loop=False, spectroscopy=False, codewords=False
        )

        program = SeqC._preamble(features)

        repeat = SeqC._play_zero(SeqC._VAR_HOLDOFF_DELAY)
        if dio_trigger:
            repeat += SeqC._wait_dio_trigger()
        repeat += SeqC._readout(
            generator_mask=SeqC._make_mask("QA_GEN_", [slot]),
            integrator_mask=SeqC._make_mask("QA_INT_", [slot]),
        )
        repeat = SeqC._scope(
            preamble=f"repeat ({SeqC._VAR_INNER_LOOP_SIZE})",
            body=repeat,
        )

        program += repeat

        return features, program

    @staticmethod
    def acquisition(slot: int) -> tuple:
        """
        Returns a tuple containing a SeqC.Features instance and a program string specifying an experiment where:

        The acquisition is started after receiving a digital trigger, without playing any readout pulse and
        using the integration weights stored in the first integrator slot. The total number of readouts
        is specified by the variable stored in the "num_samples" user register.
        """
        features = SeqC.Features(
            inner_loop=True, outer_loop=False, spectroscopy=False, codewords=False
        )

        program = SeqC._preamble(features)

        repeat = SeqC._play_zero(SeqC._VAR_HOLDOFF_DELAY)
        repeat += SeqC._wait_dio_trigger()
        repeat += SeqC._readout(
            generator_mask="0",
            integrator_mask=SeqC._make_mask("QA_INT_", [slot]),
        )
        repeat = SeqC._scope(
            preamble=f"repeat ({SeqC._VAR_INNER_LOOP_SIZE})", body=repeat
        )

        program += repeat

        return features, program

    @staticmethod
    def spectroscopy(
        start_frequency: float, frequency_step: float, dio_trigger: bool
    ) -> tuple:
        """
        Returns a tuple containing a SeqC.Features instance and a program string specifying a sequencer-based
        spectroscopy experiment:

        The acquisition is started after receiving an optional DIO trigger. The number of frequency steps and averages
        is specified by the variables stored in the "num_samples" and "num_averages" user register, respectively.

        Args:
            start_frequency: starting point of the frequency sweep
            frequency_step: offset added to the previous frequency value at each new step
            dio_trigger: specify whether the different frequency steps are self-triggered, or via a DIO trigger.
        """
        features = SeqC.Features(
            inner_loop=True, outer_loop=True, spectroscopy=True, codewords=False
        )

        program = SeqC._preamble(features)
        program += SeqC._unset_trigger()
        program += SeqC._configure_frequency_sweep(start_frequency, frequency_step)

        repeat = SeqC._play_zero(SeqC._VAR_HOLDOFF_DELAY)
        if dio_trigger:
            repeat += SeqC._wait_dio_trigger()
        repeat += SeqC._set_sweep_step(SeqC._VAR_INNER_LOOP_INDEX)
        repeat += SeqC._set_trigger()
        repeat += SeqC._unset_trigger()

        loop_preamble = f"for(var {SeqC._VAR_INNER_LOOP_INDEX} = 0; {SeqC._VAR_INNER_LOOP_INDEX} < {SeqC._VAR_INNER_LOOP_SIZE}; {SeqC._VAR_INNER_LOOP_INDEX}++)"
        repeat = SeqC._scope(
            preamble=loop_preamble,
            body=repeat,
        )
        repeat = SeqC._scope(
            preamble=f"for(var {SeqC._VAR_OUTER_LOOP_INDEX} = 0; {SeqC._VAR_OUTER_LOOP_INDEX} < {SeqC._VAR_OUTER_LOOP_SIZE}; {SeqC._VAR_OUTER_LOOP_INDEX}++)",
            body=repeat,
        )

        program += repeat

        return features, program

    @staticmethod
    def _preamble(features) -> str:
        """
        Returns an SHFQA sequencer program preamble defining and initializing variables from values stored in user
        registers based on the features passed in as arguments.

        Args:
            features: dataclass instance specifying which variables to define and initialize.
        """
        preamble = ""
        if features.inner_loop:
            preamble += SeqC._var_definition(
                SeqC._VAR_INNER_LOOP_SIZE, SeqC._get_register(UserRegisters.INNER_LOOP)
            )
        if features.outer_loop:
            preamble += SeqC._var_definition(
                SeqC._VAR_OUTER_LOOP_SIZE, SeqC._get_register(UserRegisters.OUTER_LOOP)
            )
        preamble += SeqC._var_definition(
            SeqC._VAR_HOLDOFF_DELAY, SeqC._get_register(UserRegisters.HOLDOFF_DELAY)
        )
        return preamble

    @staticmethod
    def _dio_codeword(mask: int, shift: int) -> str:
        """
        Returns a SeqC string that corresponds to the value of the codeword currently present at the DIO interface.
        """
        return f"(getDIOTriggered() & {bin(mask)})"

    @staticmethod
    def _var_definition(name: str, value) -> str:
        """
        Returns a SeqC command string that defines a run-time variable.

        Args:
            name: name of the variable to define.
            value: initial value to assign to the variable.
        """
        return f"var {name} = {value};\n"

    @staticmethod
    def _set_register(index: int, value) -> str:
        """
        Returns a SeqC command string that sets the value of a user register.

        Args:
            index: index of the user register.
            value: value to assign to the above user register.
        """
        return f"setUserReg({index}, {value});\n"

    @staticmethod
    def _get_register(index: int) -> str:
        """
        Returns a SeqC string that corresponds to the value of a specific user register. Note: this string does not
        represent a command.

        Args:
            index: index of the user register from which to query the value
        """
        return f"getUserReg({index})"

    @staticmethod
    def _scope(preamble, body) -> str:
        """
        Returns a SeqC command string that encloses another program string inside a scope (brackets) with a preamble
        inside of parentheses. Can be used to build e.g. loops and switch statements.

        Args:
            preamble: string enclosed in parentheses before the start of the scope.
            body: string enclosed within brackets.
        """
        try:
            string = "".join([line for line in body])
            body = string
        except TypeError:
            pass
        body = textwrap.indent(body, "\t")

        return preamble + "\n" + f"{{\n{body}}}\n"

    @staticmethod
    def _readout(generator_mask: str, integrator_mask: str) -> str:
        """
        Returns a SeqC command string specifying a readout operation on the SHFQA. Note: this command also triggers the
        sequencer monitor used to trigger the scope.

        Args:
            generator_mask: specifies which generators to trigger
            integrator_mask: specifies which integration units to trigger
        """
        return f"startQA({generator_mask}, {integrator_mask}, true,  0, 0x0);\n"

    @staticmethod
    def _play_zero(num_samples: int) -> str:
        """
        Returns a SeqC command string specifying a blocking wait.

        Args:
            num_samples: number of samples to block until the next execution of '_play_zero'.
        """
        return f"playZero({num_samples});\n"

    @staticmethod
    def _configure_frequency_sweep(
        start_frequency: float, frequency_step: float
    ) -> str:
        """
        Returns a SeqC command string configuring a sequencer-based frequency sweep. Must be used in conjunction with
        '_set_sweep_step'.

        Args:
            start_frequency: starting point of the frequency sweep
            frequency_step: offset added to the previous frequency value at each new step
        """
        return f"configFreqSweep({SeqC._SPECTROSCOPY_OSCILLATOR_INDEX}, {start_frequency}, {frequency_step});\n"

    @staticmethod
    def _set_sweep_step(step_index: int) -> str:
        """
        Returns a SeqC command string specifying the setting of the IF frequency of the oscillator. For this command
        to have any effect, '_configure_frequency_sweep' must have been called previously.

        Args:
            step_index: index specifying the value of the frequency to set to the IF oscillator. The actual value is
                        determined at runtime and presupposes '_configure_frequency_sweep' has been called in the
                        current program.
        """
        command = (
            f"setSweepStep({SeqC._SPECTROSCOPY_OSCILLATOR_INDEX}, {step_index});\n"
        )
        command += "resetOscPhase();\n"

        return command

    @staticmethod
    def _wait_dio_trigger() -> str:
        """
        Returns a SeqC command string specifying a blocking wait for a DIO trigger.
        """
        return "waitDIOTrigger();\n"

    @staticmethod
    def _set_trigger() -> str:
        """
        Returns a SeqC command string setting the sequencer trigger.
        """
        return "setTrigger(1);\n"

    @staticmethod
    def _unset_trigger() -> str:
        """
        Returns a SeqC command string unsetting the sequencer trigger.
        """
        return "setTrigger(0);\n"

    @staticmethod
    def _make_mask(specifier: str, indices: list) -> str:
        """
        Returns a string specifying a mask variable based on indices that can be used as arguments for a readout
        command.

        Args:
            specifier: e.g. "QA_GEN_" for generators, and "QA_INT_" for integration units.
            indices: list of indices constituting the desired mask.
        """
        if not indices:
            return "0"
        mask = ""
        for i in indices[:-1]:
            mask += specifier + f"{i}|"
        mask += specifier + f"{indices[-1]}"

        return mask


def make_result_mode_manager(driver, result_mode: str):
    """
    Returns a concrete ResultModeManager object capable of managing the provided driver instance as specified by the
    provided result mode.

    Args:
        driver: SHFQA driver instance to be managed.
        result_mode: result mode on which to dispatch.

    Raises:
        ziValueError: the provided result mode is not supported.
    """
    if result_mode == "rl":
        return _ReadoutModeManager(driver)
    if result_mode == "ro":
        return _ScopeModeManager(driver)
    if result_mode == "spectroscopy":
        return _SpectroscopyModeManager(driver)

    _raise_unsupported_result_mode(result_mode)


class ResultModeManager(ABC):
    """
    Abstract base encapsulating the handling of different measurement modes of the SHFQA device.
    """

    def __init__(self, driver):
        self._driver = driver

    def result_paths(self) -> set:
        """
        Returns all the nodes paths expected to contain measurement data. Accumulates data returned by the abstract
        '_result_path' method over the currently active measurement units of the device.
        """
        node_paths = set()
        for ch, integrators in self._driver.active_slots.items():
            for integrator in integrators:
                node_paths.add(self._result_path(ch, integrator))
        return node_paths

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validates the driver configuration with the configured result mode. Can be used to e.g. enforce that sequencer
        programs currently cached in the driver are consistent with the result mode before actually pushing the whole
        configuration to the device.

        Raises:
            ziConfigurationError: the driver configuration is incompatible with the current result mode.
        """
        pass

    @abstractmethod
    def extract_data(self, samples: int, data: dict = None) -> dict:
        """
        Extracts measurement data corresponding to the configured result mode from the device.

        Args:
            samples: number of expected measurement samples
            data: optional raw data source to extract the data from
        """
        pass

    @abstractmethod
    def push_acquisition_unit_config(self) -> None:
        """
        Sets the measurement configuration to the device acquisition units corresponding to the current result mode.
        """
        pass

    @abstractmethod
    def start_acquisition_units(self) -> None:
        """
        Starts the device acquisition units corresponding to the current result mode.
        """
        pass

    @abstractmethod
    def wait_acquisition_units_finished(self) -> None:
        """
        Starts the device acquisition units corresponding to the current result mode.
        """
        pass

    def _acquisition_time_to_length(self) -> int:
        """
        Top level function applying device constraints to the acquisition time set by the user through the driver.
        """
        length = duration_to_length(self._driver._acquisition_time)
        new_length = (
            length // self._acquisition_granularity()
        ) * self._acquisition_granularity()
        new_length = min(new_length, self._max_acquisition_length())
        self._driver._acquisition_time = length_to_duration(new_length)
        if length != new_length:
            log.info(
                f"{self._driver.devname}: changed the acquisition time from {length_to_duration(length)} to "
                f"{self._driver._acquisition_time}."
            )
        return new_length

    @abstractmethod
    def _acquisition_granularity(self) -> int:
        """
        Returns the granularity for the given result mode. The value is used to clamp down the final number of samples
        to acquire.
        """
        pass

    @abstractmethod
    def _max_acquisition_length(self) -> int:
        """
        Returns the maximal number of samples that can be recorded for the given result mode.
        """
        pass

    @abstractmethod
    def _result_path(self, ch: int, integrator: int = None) -> str:
        """
        Returns a result path corresponding to the current result mode and the provided channel and integrator index.

        Args:
            ch: channel index
            integrator: optional integrator index.
        """
        pass


class _ReadoutModeManager(ResultModeManager):
    # Override
    def validate_config(self) -> None:
        if self._driver._seqc_features.spectroscopy:
            raise ziConfigurationError(
                "The configured sequencer program is only valid in the 'spectroscopy' result mode."
            )

    # Override
    def extract_data(self, samples: int, data: dict = None) -> dict:
        result = {}
        for ch, integrators in self._driver.active_slots.items():
            for integrator in integrators:
                path = self._result_path(ch, integrator)
                node_tree = data if data else self._driver.daq.get(path, flat=True)
                vector = _get_vector_from_node_tree(node_tree, path)
                if vector is not None:
                    new_entry = {integrator: vector}
                    try:
                        result[ch].update(new_entry)
                    except KeyError:
                        result[ch] = new_entry
        return result

    # Override
    def push_acquisition_unit_config(self) -> None:
        for ch in self._driver.active_channels:
            shfqa_utils.configure_result_logger_for_readout(
                self._driver.daq,
                self._driver.devname,
                channel_index=ch,
                result_source=self._driver._result_source,
                result_length=self._driver._samples,
                num_averages=self._driver._averages,
                averaging_mode=_averaging_mode_to_int(self._driver._averaging_mode),
            )
            self._driver.set(
                f"qachannels_{ch}_mode",
                DeviceConstants.Readout.QACHANNEL_MODE,
            )
            self._driver.set(
                f"qachannels_{ch}_readout_integration_delay",
                self._driver._wait_dly,
            )
            self._driver.set(
                f"qachannels_{ch}_readout_integration_length",
                self._acquisition_time_to_length(),
            )

    # Override
    def start_acquisition_units(self) -> None:
        for ch in self._driver.active_channels:
            shfqa_utils.enable_result_logger(
                self._driver.daq,
                self._driver.devname,
                channel_index=ch,
                mode="readout",
            )

    # Override
    def wait_acquisition_units_finished(self) -> None:
        for ch in self._driver.active_channels:
            path = f"/{self._driver.devname}/qachannels/{ch}/readout/result/enable"
            wait_for_state_change(
                self._driver.daq, path, 0, timeout=self._driver.timeout()
            )

    # Override
    def _acquisition_granularity(self) -> int:
        return DeviceConstants.Readout.GRANULARITY

    # Override
    def _max_acquisition_length(self) -> int:
        return DeviceConstants.Readout.MAX_LENGTH

    # Override
    def _result_path(self, ch: int, integrator: int = None) -> str:
        return f"/{self._driver.devname}/qachannels/{ch}/readout/result/data/{integrator}/wave"


class _ScopeModeManager(ResultModeManager):
    # Override
    def validate_config(self) -> None:
        if self._driver._seqc_features.spectroscopy:
            raise ziConfigurationError(
                "The configured sequencer program is only valid in the 'spectroscopy' result mode."
            )
        if self._driver._averaging_mode == "cyclic" and self._driver._averages > 1:
            raise ziConfigurationError(
                "Cyclic averaging is not supported in 'ro' mode."
            )

    # Override
    def extract_data(self, samples: int, data: dict = None) -> dict:
        result = {}
        for ch in self._driver.active_channels:
            path = self._result_path(ch)
            node_tree = data if data else self._driver.daq.get(path, flat=True)
            vector = _get_vector_from_node_tree(node_tree, path)
            if vector is not None:
                result[ch] = np.array_split(vector, samples)
        return result

    # Override
    def push_acquisition_unit_config(self) -> None:
        scope_trigger_input, scope_input_select = self._scope_configuration()
        shfqa_utils.configure_scope(
            self._driver.daq,
            self._driver.devname,
            input_select=scope_input_select,
            num_samples=self._acquisition_time_to_length(),
            trigger_input=scope_trigger_input,
            num_segments=self._driver._samples,
            num_averages=self._driver._averages,
            trigger_delay=self._driver._wait_dly,
        )

    # Override
    def start_acquisition_units(self) -> None:
        shfqa_utils.enable_scope(self._driver.daq, self._driver.devname, single=1)

    # Override
    def wait_acquisition_units_finished(self) -> None:
        path = f"/{self._driver.devname}/scopes/0/enable"
        wait_for_state_change(self._driver.daq, path, 0, timeout=self._driver.timeout())

    # Override
    def _acquisition_granularity(self) -> int:
        return DeviceConstants.Scope.GRANULARITY

    # Override
    def _max_acquisition_length(self) -> int:
        return (
            DeviceConstants.Scope.MAX_LENGTH // len(self._driver.active_channels)
        ) // self._driver._samples

    # Override
    def _result_path(self, ch: int, integrator: int = None) -> str:
        return f"/{self._driver.devname}/scopes/0/channels/{ch}/wave"

    def _scope_configuration(self) -> tuple:
        scope_input_select = {ch: ch for ch in self._driver.active_channels}
        sequencer_index = self._driver.active_channels[0]
        sequencer_monitor_start_index = 64
        trigger_input = (
            sequencer_monitor_start_index + sequencer_index
        )  # f"channel{sequencer_index}_sequencer_monitor0"
        return (trigger_input, scope_input_select)


class _SpectroscopyModeManager(ResultModeManager):
    # Override
    def validate_config(self) -> None:
        if not self._driver._seqc_features.spectroscopy:
            raise ziConfigurationError(
                "The configured sequencer program is incompatible with the 'spectroscopy' result mode."
            )

    # Override
    def extract_data(self, samples: int, data: dict = None) -> dict:
        result = {}
        for ch in self._driver.active_channels:
            path = self._result_path(ch)
            node_tree = data if data else self._driver.daq.get(path, flat=True)
            result[ch] = _get_vector_from_node_tree(node_tree, path)
        return result

    # Override
    def push_acquisition_unit_config(self) -> None:
        for ch in self._driver.active_channels:
            shfqa_utils.configure_result_logger_for_spectroscopy(
                self._driver.daq,
                self._driver.devname,
                channel_index=ch,
                result_length=self._driver._samples,
                num_averages=self._driver._averages,
                averaging_mode=_averaging_mode_to_int(self._driver._averaging_mode),
            )
            self._driver.set(
                f"qachannels_{ch}_mode",
                DeviceConstants.Spectroscopy.QACHANNEL_MODE,
            )
            self._driver.set(
                f"qachannels_{ch}_spectroscopy_trigger_channel",
                DeviceConstants.Spectroscopy.TRIGGER_SOURCE,
            )
            self._driver.set(
                f"qachannels_{ch}_spectroscopy_delay", self._driver._wait_dly
            )
            self._driver.set(
                f"qachannels_{ch}_spectroscopy_length",
                duration_to_length(self._driver._acquisition_time),
            )

    # Override
    def start_acquisition_units(self) -> None:
        for ch in self._driver.active_channels:
            shfqa_utils.enable_result_logger(
                self._driver.daq,
                self._driver.devname,
                channel_index=ch,
                mode="spectroscopy",
            )

    # Override
    def wait_acquisition_units_finished(self) -> None:
        for ch in self._driver.active_channels:
            path = f"/{self._driver.devname}/qachannels/{ch}/spectroscopy/result/enable"
            wait_for_state_change(
                self._driver.daq, path, 0, timeout=self._driver.timeout()
            )

    # Override
    def _acquisition_granularity(self) -> int:
        return DeviceConstants.Spectroscopy.GRANULARITY

    # Override
    def _max_acquisition_length(self) -> int:
        return DeviceConstants.Spectroscopy.MAX_LENGTH

    # Override
    def _result_path(self, ch: int, integrator: int = None) -> str:
        return f"/{self._driver.devname}/qachannels/{ch}/spectroscopy/result/data/wave"


def duration_to_length(duration: float) -> int:
    """
    Helper function converting a duration in seconds into number of samples on the device.
    """
    return int(round(duration * DeviceConstants.SAMPLING_FREQUENCY))


def length_to_duration(length: int) -> float:
    """
    Helper function converting a number of samples on the device to a duration in seconds.
    """
    return length / DeviceConstants.SAMPLING_FREQUENCY


def _get_vector_from_node_tree(node_tree: dict, path: str):
    try:
        return node_tree[path][0]["vector"]
    except KeyError:
        return None


def _averaging_mode_to_int(averaging_mode: str) -> int:
    if averaging_mode == "sequential":
        return DeviceConstants.Averaging.SEQUENTIAL
    if averaging_mode == "cyclic":
        return DeviceConstants.Averaging.CYCLIC
    _raise_unsupported_averaging_mode(averaging_mode)


def _raise_unsupported_result_mode(result_mode: str) -> None:
    raise ziValueError(
        f"Unsupported result mode: {result_mode}. Supported modes are 'rl', 'ro' and 'spectroscopy'."
    )


def _raise_unsupported_averaging_mode(averaging_mode: str) -> None:
    raise ziValueError(
        f"Unsupported readout mode: {averaging_mode}. Supported modes are 'sequential' and 'cyclic'."
    )
