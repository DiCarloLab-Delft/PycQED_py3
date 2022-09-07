"""
To do:

- Determine minimum revisions
- Implement DIO.CalInterface
- Calls to shfqa_utils also update Qcode parameters
- Finish docstrings

Notes:

Changelog:
"""

import time
import logging
import numpy as np
import os

from typing import Tuple, List

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument import (
    ZI_base_instrument,
    ziValueError,
    ziVersionError,
    ziDeviceError,
    ziConfigurationError,
)
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_uhfqc_compatibility as uhf_compatibility
from pycqed.instrument_drivers.library.DIO import CalInterface

from qcodes.utils import validators
from qcodes.utils.helpers import full_class

import zhinst.utils.shfqa as shfqa_utils
from zhinst.utils import wait_for_state_change

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.internal._shfqa import (
    DeviceConstants,
    UserRegisters,
    SeqC,
    DioCalibration,
    preprocess_generator_waveform,
    hold_off_length,
    make_result_mode_manager,
    duration_to_length,
)

log = logging.getLogger(__name__)


class Defaults:
    """
    Class specifying driver default configuration values.
    """

    NUM_RESULTS = uhf_compatibility.Dio.MAX_NUM_RESULTS
    NUM_MEASUREMENTS = 100
    NUM_AVERAGES = 1

    RESULT_MODE = "rl"
    RESULT_SOURCE = "result_of_integration"

    AVERAGING_MODE = "sequential"

    ACQUISITION_TIME = 1000e-9
    ACQUISITION_DELAY = 200e-9

    DIGITAL_TRIGGER_SOURCE = "software_trigger0"

    DIO_CALIBRATION_DELAY = 0

    CODEWORD_MANAGER_INSTANCE = uhf_compatibility.BruteForceCodewordManager()

    class SingleQubitExperiments:
        """
        Default values for single qubit experiments, (single readout pulse and integration weight).
        """

        CHANNEL = 0
        SLOT = 0


class SHFQA(ZI_base_instrument, CalInterface):
    """
    This is the PycQED driver for the 2.0 Gsample/s SHFQA developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5/3.6 ucs4 19.05 for 64bit Windows from
        http://www.zhinst.com/downloads, https://people.zhinst.com/~niels/
    2. upload the latest firmware to the SHFQA using the LabOne GUI
    """

    def __init__(
        self,
        name,
        device: str,
        interface: str = "USB",
        port: int = 8004,
        use_dio: bool = True,
        nr_integration_channels: int = Defaults.NUM_RESULTS,
        codeword_manager=Defaults.CODEWORD_MANAGER_INSTANCE,
        server: str = "127.0.0.1",
        **kw,
    ) -> None:
        t0 = time.time()

        self._codeword_manager = codeword_manager
        self._use_dio = use_dio

        super().__init__(
            name=name,
            device=device,
            interface=interface,
            server=server,
            port=port,
            num_codewords=2 ** nr_integration_channels,
            **kw,
        )

        self._seqc_features = None
        self._single = True
        self._poll = True
        self._result_mode_manager = None
        self._active_slots = {}

        self.load_default_settings(upload_sequence=False)

        log.info(f"{self.devname}: Initialized SHFQA in {time.time() - t0:.3f}s")

    ##########################################################################
    # 'public' overrides for ZI_base_instrument
    ##########################################################################

    def start(self) -> None:
        """
        Starts the driver. In particular, cached configurations are validated and pushed to the device before arming
        the acquisition units as well as the sequencers.
        """
        log.info(f"{self.devname}: Starting '{self.name}'")
        self.check_errors()
        if self._poll:
            self._subscribe_to_result_nodes()
        self.push_to_device()
        self._enable_channels()
        self._result_mode_manager.start_acquisition_units()
        self._start_sequencers()

    def stop(self) -> None:
        """
        Stops the driver. In particular, disarms active sequencers and checks for errors.
        """
        log.info(f"Stopping {self.name}")
        for ch in self.active_channels:
            self.set(f"qachannels_{ch}_generator_enable", 0)
        self.unsubs()
        self.check_errors()

    def load_default_settings(self, upload_sequence: bool = True) -> None:
        """
        Sets default values to the QCoDes extra parameters. Note: these default settings are not pushed to the device.
        They are only cached in the driver. Use the "push_to_device" method in order to upload the settings to the
        instrument.

        Args:
            upload_sequencer: specify whether or not to upload a default sequencer program.
        """
        if upload_sequence:
            self.awg_sequence_acquisition()

        self.wait_dly(Defaults.ACQUISITION_DELAY)
        self.dio_calibration_delay(Defaults.DIO_CALIBRATION_DELAY)

        self.result_mode(Defaults.RESULT_MODE)
        self.result_source(Defaults.RESULT_SOURCE)
        self.acquisition_time(Defaults.ACQUISITION_TIME)
        self.averages(Defaults.NUM_AVERAGES)
        self.samples(Defaults.NUM_MEASUREMENTS)
        self.averaging_mode(Defaults.AVERAGING_MODE)

    def configure_awg_from_string(
        self, awg_nr: int, program_string: str, timeout: float = 15
    ) -> None:
        """
        Uploads the provided program string to the specified sequencer.

        Args:
            awg_nr: index of the sequencer to configure.
            program_string: string specifying the SeqC program to upload.
            timeout: time interval after which the upload is considered to have failed.

        Raises:
            TimeoutError: the upload was not successful even after the specified timeout.
        """
        shfqa_utils.load_sequencer_program(
            self.daq,
            self.devname,
            channel_index=awg_nr,
            sequencer_program=program_string,
            timeout=self.timeout(),
        )

    def assure_ext_clock(self) -> None:
        """
        Attempts to lock the device to an external reference clock. Failure to lock does
        not raise an exception.
        """
        external_source = 1
        actual_source_path = "system/clocks/referenceclock/in/sourceactual"

        is_already_locked = self.geti(actual_source_path) == external_source
        if is_already_locked:
            return
        log.info(f"{self.devname}: Attempting to lock onto external reference clock...")
        self.set("system_clocks_referenceclock_in_source", external_source)
        try:
            wait_for_state_change(
                self.daq,
                f"/{self.devname}/system/clocks/referenceclock/in/sourceactual",
                value=external_source,
                timeout=self.timeout(),
            )
            log.info(
                f"{self.devname}: Successfully locked onto external reference clock."
            )
        except TimeoutError:
            log.info(
                f"{self.devname}: Failed to lock onto external reference clock within {self.timeout()}."
            )

    def clock_freq(self) -> float:
        """
        Returns the device clock frequency.
        """
        return DeviceConstants.SAMPLING_FREQUENCY

    def plot_dio_snapshot(self, bits=range(32)) -> None:
        raise NotImplementedError

    ##########################################################################
    # measurement setup methods ported from UHF
    ##########################################################################

    def acquisition(
        self,
        samples: int = Defaults.NUM_MEASUREMENTS,
        averages: int = Defaults.NUM_AVERAGES,
        acquisition_time: float = 0.010,
        timeout: float = 10,
        mode: str = Defaults.RESULT_MODE,
        poll: bool = True,
    ) -> dict:
        """
        Perform an acquisition from start to finish.

        Args:
            samples: number of measurement samples.
            averages: number of averages per measurement sample.
            acquisition_time: total time of the acquisition.
            timeout: time specifying at which point the absence of results is considered an error.
            mode: result mode with possible values 'rl', 'ro' or 'spectroscopy'.
            poll: specify whether measurement results will under the hood be acquired via poll or deep get.
        """
        self.timeout(timeout)

        self.acquisition_initialize(
            samples=samples, averages=averages, mode=mode, poll=False
        )

        if poll:
            data = self.acquisition_poll(
                samples=samples, arm=True, acquisition_time=acquisition_time
            )
        else:
            data = self.acquisition_get(
                samples=samples, arm=True, acquisition_time=acquisition_time
            )

        self.acquisition_finalize()

        return data

    def acquisition_initialize(
        self,
        samples: int,
        averages: int,
        loop_cnt: int = None,
        channels: tuple = (0, 1),  # ignored in this driver
        mode: str = Defaults.RESULT_MODE,
        poll: bool = True,
    ) -> None:
        """
        Initializes the acquisition units with the provided measurement configration.

        Args:
            samples: number of measurement samples.
            averages: number of averages per measurement sample.
            loop_cnt: number of loop points to be configured for the sequencer program.
            channels: (ignored)
            mode: result mode with possible values 'rl', 'ro' or 'spectroscopy'
            poll: specify whether to configure the device in a way that results may be collected using 'acquisition_poll'
                  at the end of the measurement.
        """
        self.samples(samples)
        self.averages(averages)
        self.result_mode(mode)
        self._poll = poll

    def acquisition_arm(self, single=True) -> None:
        """
        Arms the acquisition.

        Args:
            single: specifies whether to disarm sequencers once all results have been collected.
        """
        self._single = single
        self.start()

    def acquisition_poll(self, samples, arm=True, acquisition_time=0.010) -> dict:
        """
        Returns the collected measurement results using the poll command under the hood. Note: this method does not
        allow collecting partial results. Results will only be available if the acquisition units have been triggered
        as many times as specified in the "acquisition_initialize" method. As opposed to "acquisition_get", this way of
        acquiring results saves a redundant read request from the LabOne dataserver to the SHFQA at the end of the
        experiment.

        Args:
            samples: number of expected measurement results. Mainly used to split scope segments if the acquisition mode
                is set to "ro".
            arm: specify whether or not to arm the acquisition before attempting to collect results.
            acquisition_time: time during which to poll for results.

        Raises:
            TimeoutError: not all results as specified by the "acquisition_initialize" method could be collected.
        """
        if arm:
            self.acquisition_arm()

        accumulated_time = 0
        gotem = False
        while accumulated_time < self.timeout() and not gotem:
            poll_result = self.poll(acquisition_time)
            if poll_result:
                result = self._result_mode_manager.extract_data(
                    samples=samples, data=poll_result
                )
                if result:
                    gotem = True
            accumulated_time += acquisition_time

        if not gotem:
            self.acquisition_finalize()
            raise TimeoutError("Failed to retrieve all acquisition results.")

        return result

    def acquisition_get(self, samples, arm=True, acquisition_time=0.010) -> dict:
        """
        Returns the collected measurement results using the get command under the hood. Note: this method does not
        allow collecting partial results. Results will only be available if the acquisition units have been triggered
        as many times as specified in the "acquisition_initialize" method.

        Args:
            samples: number of expected measurement results. Mainly used to split scope segments if the acquisition mode
                     is set to "ro".
            arm: specify whether or not to arm the acquisition before attempting to collect results.
            acquisition_time: time during which to wait until the absence of results is considered an error.

        TimeoutError: not all results as specified by the "acquisition_initialize" method could be collected.
        """
        if arm:
            self.acquisition_arm()

        if self._single:
            try:
                self._wait_sequencers_finished()
            except TimeoutError:
                self.acquisition_finalize()
                raise TimeoutError(
                    "Failed to retrieve acquisition results because sequencers are still running."
                )

        try:
            self._result_mode_manager.wait_acquisition_units_finished()
            result = self._result_mode_manager.extract_data(samples=samples, data=None)
        except TimeoutError:
            self.acquisition_finalize()
            raise TimeoutError(
                "Failed to retrieve acquisition results because acquisition units are still running."
            )

        return result

    def acquisition_finalize(self) -> None:
        """
        Finalizes the current measurement.
        """
        self.stop()

    ##########################################################################
    # QCoDeS waveform parameters and their setters/getters
    ##########################################################################

    def _add_codeword_waveform_parameters(self, num_codewords) -> None:
        """
        Adds mutable QCoDeS parameters associating readout waveforms and codewords.

        Args:
            num_codewords: number specifying the range of codewords for which to define a waveform parameter.
        """
        log.info(f"{self.devname}: Adding codeword waveform parameters")
        for codeword in range(num_codewords):
            wf_name = _waveform_name(codeword)
            if wf_name not in self.parameters:
                self.add_parameter(
                    wf_name,
                    label=f"Waveform codeword {codeword}",
                    vals=validators.Arrays(valid_types=(complex,)),
                    set_cmd=self._write_waveform(codeword),
                    get_cmd=self._read_waveform(codeword),
                    docstring="Specifies a waveform for a specific codeword.",
                )
        self._num_codewords = num_codewords

    def _csv_filename(self, codeword):
        return os.path.join(
            self._get_awg_directory(),
            "waves",
            self.devname + "_" + _waveform_name(codeword) + ".csv",
        )

    def _write_csv(self, codeword, waveform) -> None:
        log.debug(f"{self.devname}: Writing waveform of codeword {codeword}")
        np.savetxt(self._csv_filename(codeword), waveform)

    def _read_csv(self, codeword):
        filename = self._csv_filename(codeword)
        try:
            log.debug(
                f"{self.devname}: reading codeword waveform {codeword} from csv '{filename}'"
            )
            waveform = np.genfromtxt(filename, dtype=np.complex128)
            return waveform
        except OSError as e:
            # if the waveform does not exist yet dont raise exception
            log.warning(e)
            return None

    def _write_waveform(self, codeword: int) -> None:
        """
        Returns the setter function for the QCoDeS waveform parameter associated to a given codeword.

        Args:
            codeword: codeword specifying which QCoDeS waveform parameter to associate the setter function to.
        """

        def write_func(waveform):
            ch, slot = self._generator_slot(codeword)
            waveform = preprocess_generator_waveform(waveform)
            self._write_csv(codeword, waveform)

        return write_func

    def _read_waveform(self, codeword):
        """
        Returns the getter function for the QCoDeS waveform parameter associated to a given codeword.

        Args:
            codeword: codeword specifying which QCoDeS waveform parameter to associate the getter function to.
        """

        def read_func():
            log.debug(f"{self.devname}: Reading waveform of codeword {codeword}")
            ch, slot = self._generator_slot(codeword)
            return self._read_csv(codeword)

        return read_func

    def _generator_slot(self, codeword: int) -> tuple:
        """
        Returns the channel and slot index specifying which generator memory slot a given codeword is associated with.

        Args:
            codeword: codeword from which to extract the memory slot.

        Raises:
            ziValueError: the codeword specifies either no generator memory slot at all, or multiple slots.
        """
        mapping = self._codeword_manager.codeword_slots(codeword)
        if len(mapping) == 0:
            raise ziValueError(
                f"Cannot read/write waveform of codeword {codeword} because it is not associated to any generator slot. "
                f"Only 'pure codeword waveforms', that is, codeword waveforms that are associated to a single generator, "
                f"can be uploaded to the device."
            )

        multiple_slots_error = (
            f"Cannot read/write waveform of codeword {codeword} because it is associated to multiple generator slots."
            f"Only 'pure codeword waveforms', that is, codeword waveforms that are associated to a single generator, "
            f"can be uploaded to the device."
        )
        spread_onto_multiple_channels = len(mapping) > 1
        if spread_onto_multiple_channels:
            raise ziValueError(multiple_slots_error)

        for ch, slots in mapping.items():
            spread_onto_multiple_slots = len(slots) > 1
            if spread_onto_multiple_slots:
                raise ziValueError(multiple_slots_error)
            return ch, slots[0]

    ##########################################################################
    # extra QCoDeS parameters and their setters/getters
    ##########################################################################

    def _add_extra_parameters(self) -> None:
        """
        Adds extra QCoDes parameters to the driver instance. For a detailed description, please refer to the code below.
        """
        super()._add_extra_parameters()

        ##########################################################################
        # ported from UHF
        ##########################################################################

        self.add_parameter(
            "wait_dly",
            set_cmd=self._set_wait_dly,
            get_cmd=self._get_wait_dly,
            unit="",
            label="Delay between the start of the output signal playback and integration at the input",
            docstring="Configures a delay in seconds to be "
            "applied between the start of the output signal playback and integration at the input",
            vals=validators.Numbers(),
        )

        self.add_parameter(
            "cases",
            set_cmd=self._set_codewords,
            get_cmd=self._get_codewords,
            docstring="List of integers defining the codewords "
            "to be handled by the sequencer program. For example, setting the parameter to [1, 5, 7] would result in "
            "an AWG program that handles only codewords 1, 5 and 7. When running, if the AWG receives a codeword "
            "that is not part of this list, an error will be triggered.",
        )

        self.add_parameter(
            "dio_calibration_delay",
            set_cmd=self._set_dio_calibration_delay,
            get_cmd=self._get_dio_calibration_delay,
            unit="",
            label="DIO Calibration delay",
            docstring="Configures the internal delay in 300 MHz cycles (3.3 ns) "
            "to be applied on the DIO interface in order to achieve reliable sampling "
            "of codewords. The valid range is 0 to 15.",
            vals=validators.Ints(),
        )

        ##########################################################################
        # specific to SHF
        ##########################################################################

        self.add_parameter(
            "result_mode",
            set_cmd=self._set_result_mode,
            get_cmd=self._get_result_mode,
            unit="",
            label="Result mode",
            docstring="Configures the result mode of the device. 'ro' records and returns the raw signal"
            "at the channel inputs using the scope. 'rl' performs a weighted integration using the readout"
            "units. 'spectroscopy' correlates the incoming signal with an internal oscillator whose frequency"
            "can be swept to perform continuous wave of pulsed spectroscopy experiments."
            " 'rl', 'spectroscopy'.",
            vals=validators.Enum("ro", "rl", "spectroscopy"),
        )

        self.add_parameter(
            "result_source",
            set_cmd=self._set_result_source,
            get_cmd=self._get_result_source,
            unit="",
            label="Result source",
            docstring="Configures the result source in 'rl' mode of the device.",
            vals=validators.Enum("result_of_integration", "result_of_discrimination"),
        )

        self.add_parameter(
            "acquisition_time",
            set_cmd=self._set_acquisition_time,
            get_cmd=self._get_acquisition_time,
            unit="",
            label="Acquisition time",
            docstring="Configures the acquisition time for each measurement in the experiment. The value will be "
            "set to the instrument based on the acquisition mode, but only at the execution of of 'start()'",
            vals=validators.Numbers(),
        )

        self.add_parameter(
            "samples",
            set_cmd=self._set_samples,
            get_cmd=self._get_samples,
            unit="",
            label="Number of measurement samples",
            docstring="Configures the number of measurement samples to acquire in the experiment.",
            vals=validators.Ints(),
        )

        self.add_parameter(
            "averages",
            set_cmd=self._set_averages,
            get_cmd=self._get_averages,
            unit="",
            label="Number of averages",
            docstring="Configures the number of averages to be performed on the device for each measurement sample.",
            vals=validators.Ints(),
        )

        self.add_parameter(
            "averaging_mode",
            set_cmd=self._set_averaging_mode,
            get_cmd=self._get_averaging_mode,
            unit="",
            label="Averaging mode",
            docstring="Configures the averaging algorithm performed on the device. Possible values: 'sequential' "
            "and 'cyclic'.",
            vals=validators.Enum("sequential", "cyclic"),
        )

    def _set_wait_dly(self, value: float) -> None:
        """
        Setter function for the "wait_dly" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        self._wait_dly = value

    def _get_wait_dly(self) -> float:
        """
        Getter function for the "wait_dly" QCoDeS parameter.
        """
        return self._wait_dly

    def _set_codewords(self, value) -> None:
        """
        Setter function for the "cases" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        self.awg_sequence_acquisition_and_DIO_triggered_pulse(
            Iwaves=None, Qwaves=None, cases=value
        )

    def _get_codewords(self) -> list:
        """
        Getter function for the "cases" QCoDeS parameter.
        """
        return self._codeword_manager.active_codewords

    def _set_dio_calibration_delay(self, value) -> None:
        """
        Setter function for the "dio_calibration_delay" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        if value not in DioCalibration.DELAYS:
            raise ziValueError(
                f"Trying to set DIO calibration delay to invalid value! Expected value in {DioCalibration.DELAYS}"
            )

        log.info(f"{self.devname}: Setting DIO calibration delay to {value}")
        self._dio_calibration_delay = value
        for bit in range(DioCalibration.NUM_BITS):
            self.seti(f"raw/dios/0/delays/{bit}/value", self._dio_calibration_delay)

    def _get_dio_calibration_delay(self) -> float:
        """
        Getter function for the "dio_calibration_delay" QCoDeS parameter.
        """
        return self._dio_calibration_delay

    def _set_result_mode(self, value: str) -> None:
        """
        Setter function for the "result_mode" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        self._result_mode_manager = make_result_mode_manager(self, value)
        self._result_mode = value

    def _get_result_mode(self) -> float:
        """
        Getter function for the "result_mode" QCoDeS parameter.
        """
        return self._result_mode

    def _set_result_source(self, value: str) -> None:
        """
        Setter function for the "result_source" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        self._result_source = value

    def _get_result_source(self) -> float:
        """
        Getter function for the "result_source" QCoDeS parameter.
        """
        return self._result_source

    def _set_acquisition_time(self, value: float) -> None:
        """
        Setter function for the "acquisition_time" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        self._acquisition_time = value

    def _get_acquisition_time(self) -> float:
        """
        Getter function for the "acquisition_time" QCoDeS parameter.
        """
        return self._acquisition_time

    def _set_samples(self, value: int) -> None:
        """
        Setter function for the "samples" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        self._samples = value

    def _get_samples(self) -> int:
        """
        Getter function for the "samples" QCoDeS parameter.
        """
        return self._samples

    def _set_averages(self, value: int) -> None:
        """
        Setter function for the "averages" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        self._averages = value

    def _get_averages(self) -> int:
        """
        Getter function for the "averages" QCoDeS parameter.
        """
        return self._averages

    def _set_averaging_mode(self, value: int) -> None:
        """
        Setter function for the "averaging_mode" QCoDeS parameter.

        Args:
            value: value to set the parameter to.
        """
        self._averaging_mode = value

    def _get_averaging_mode(self) -> str:
        """
        Getter function for the "averaging_mode" QCoDeS parameter.
        """
        return self._averaging_mode

    ##########################################################################
    # Overriding Qcodes InstrumentBase methods
    ##########################################################################

    def snapshot_base(
        self, update: bool = False, params_to_skip_update=None, params_to_exclude=None
    ):
        """
        Returns QCoDes snapshot instance.

        Args:
            update: specify whether to query parameters from the instrument itself, or to use cached values.
            params_to_skip_update: parameters for which to use cached values if update is set to True.
            params_to_exclude: parameters to exclude all together.
        """
        if params_to_exclude is None:
            params_to_exclude = set(
                ["features_code", "system_fwlog", "system_fwlogenable"]
            )

        snap = {
            "functions": {
                name: func.snapshot(update=update)
                for name, func in self.functions.items()
            },
            "submodules": {
                name: subm.snapshot(update=update)
                for name, subm in self.submodules.items()
            },
            "__class__": full_class(self),
        }

        snap["parameters"] = {}
        for name, param in self.parameters.items():
            if params_to_exclude and name in params_to_exclude:
                pass
            elif params_to_skip_update and name in params_to_skip_update:
                update_par = False
            else:
                update_par = update
                try:
                    snap["parameters"][name] = param.snapshot(update=update_par)
                except:
                    logging.info(
                        "Snapshot: Could not update parameter: {}".format(name)
                    )
                    snap["parameters"][name] = param.snapshot(update=False)

        for attr in set(self._meta_attrs):
            if hasattr(self, attr):
                snap[attr] = getattr(self, attr)
        return snap

    ##########################################################################
    # sequencer functions ported from UHFQC
    ##########################################################################

    def awg_sequence_acquisition_and_DIO_triggered_pulse(
        self, Iwaves=None, Qwaves=None, cases=None, acquisition_delay: float = 0
    ) -> None:
        """
        Configures the sequencers for a codeword experiment. The acquisition is started after receiving a DIO trigger,
        with different generator and integrator slots being triggered based on the value of the codeword stored in the
        DIO trigger value. The total number of readouts is specified by the variable stored in the "num_samples" user
        register. Note: sets the "wait_dly", "cases" and codeword waveform QCoDes parameters.

        Args:
            Iwaves: real number arrays representing the in-phase components of the codeword waveforms. Overrides waveform
                    parameters specified by the 'cases' argument. If cases is None, the waveforms will be mapped to
                    default cases specified by the CodewordMapper instance provided to the driver during its initialization.
            Qwaves: same as Iwaves, only for the quadrature components.
            cases: list specifying which codewords to consider in the uploaded sequence. This list must be consistent
                   with the provided waveforms, if provided.
            acquisition_delay: time between the reception of the integration trigger and the actual integration.

        Raises:
            ziValueError: the provided cases list is not consistent with the provided waveforms.
        """
        if cases is not None:
            self._codeword_manager.active_codewords = cases

        self._active_slots = self._codeword_manager.active_slots()
        codewords = self._codeword_manager.active_codewords

        waves = None
        if Iwaves is not None and Qwaves is not None:
            waves = []
            for i in range(len(Iwaves)):
                waves.append(
                    uhf_compatibility.check_and_convert_waveform(Iwaves[i], Qwaves[i])
                )
        elif Iwaves is not None or Qwaves is not None:
            raise ziValueError("Must provide both Iwaves and Qwaves arguments.")

        self.wait_dly(acquisition_delay)

        if waves is not None:
            if len(codewords) != len(waves):
                raise ziValueError(
                    f"Number of waves specified {len(waves)} does not match the number of currently active codewords "
                    f"{len(codewords)}."
                )
            for i, wave in enumerate(waves):
                self.set(_waveform_name(codewords[i]), wave)

        switch_cases = [None] * len(self.active_channels)

        for codeword in codewords:
            for ch, slots in self._codeword_manager.codeword_slots(codeword).items():
                try:
                    switch_cases[ch][codeword] = slots
                except TypeError:
                    switch_cases[ch] = {codeword: slots}

        for ch in self.active_channels:
            features, program = SeqC.acquisition_and_DIO_triggered_pulse(
                mask=uhf_compatibility.Dio.codeword_mask(),
                shift=uhf_compatibility.Dio.INPUT_SHIFT,
                cases=switch_cases[ch],
            )
            self._seqc_features = features
            self._awg_program[ch] = program

    ##########################################################################
    # single qubit experiments
    ##########################################################################

    def awg_sequence_acquisition_and_pulse(
        self,
        Iwave=None,
        Qwave=None,
        acquisition_delay=0,
        dig_trigger=True,
        ch: int = Defaults.SingleQubitExperiments.CHANNEL,
        slot: int = Defaults.SingleQubitExperiments.SLOT,
    ) -> None:
        """
        Configures the sequencers for an experiment where the acquisition is started after optionally receiving a
        DIO trigger, with the readout pulse and integration weights stored in the specified generator and integrator
        slot, respectively. The total number of readouts is specified by the variable stored in the "num_samples" user
        register. Note: sets the "wait_dly" QCoDes parameter.

        Args:
            Iwave: real numbers array representing the in-phase component of the readout pulse
            Qwave: real numbers array representing the quadrature component of the readout pulse
            acquisition_delay: time between the reception of the integration trigger and the actual integration.
            dig_trigger: specify whether or not the readout gets triggered through DIO.
            ch: index specifying which qachannel to perform the experiment on.
            slot: index specifying which generator and integrator slot to trigger for the experiment.
        """
        wave = uhf_compatibility.check_and_convert_waveform(Iwave, Qwave)

        self._active_slots = {ch: [slot]}
        self.wait_dly(acquisition_delay)

        if wave is not None:
            self.set(f"qachannels_{ch}_generator_waveforms_{slot}_wave", wave)

        features, program = SeqC.acquisition_and_pulse(slot, dio_trigger=dig_trigger)
        self._seqc_features = features
        self._awg_program[ch] = program

    def awg_sequence_acquisition(
        self,
        dly=0,
        ch: int = Defaults.SingleQubitExperiments.CHANNEL,
        slot: int = Defaults.SingleQubitExperiments.SLOT,
    ) -> None:
        """
        Configures the sequencers for an experiment where the acquisition is started after receiving a DIO trigger,
        without playing any readout pulse and using the integration weights stored in a specified integrator slot. The
        total number of readouts is specified by the variable stored in the "num_samples" user register. Note: sets
        the "wait_dly" QCoDes parameter.

        Args:
            dly: time between the reception of the trigger and the actual integration.
            ch: index specifying which qachannel to perform the experiment on.
            slot: index specifying which integrator slot to trigger for the experiment.
        """
        self._active_slots = {ch: [slot]}
        self.wait_dly(dly)

        shfqa_utils.configure_sequencer_triggering(
            self.daq,
            self.devname,
            channel_index=ch,
            aux_trigger=Defaults.DIGITAL_TRIGGER_SOURCE,
        )

        features, program = SeqC.acquisition(slot)
        self._seqc_features = features
        self._awg_program[ch] = program

    def awg_sequence_acquisition_and_pulse_SSB(
        self,
        f_RO_mod,
        RO_amp,
        RO_pulse_length,
        acquisition_delay,
        dig_trigger=True,
        ch: int = Defaults.SingleQubitExperiments.CHANNEL,
        slot: int = Defaults.SingleQubitExperiments.SLOT,
    ) -> None:
        """
        Same as 'awg_sequence_acquisition_and_pulse', only with an SSB readout pulse uploaded to the specified slot.
        Args:
            f_RO_mod: modulation frequency of the SSB readout pulse.
            RO_amp: amplitude of the SSB readout pulse.
            RO_pulse_length: length of the SSB readout pulse in seconds.
            acquisition_delay: time between the reception of the integration trigger and the actual integration.
            dig_trigger: specify whether or not the readout gets triggered through DIO.
            ch: index specifying which qachannel to perform the experiment on.
            slot: index specifying which generator and integrator slot to trigger for the experiment.
        """
        size = RO_pulse_length * DeviceConstants.SAMPLING_FREQUENCY
        array = np.arange(int(size))
        sin = RO_amp * np.sin(
            2 * np.pi * array * f_RO_mod / DeviceConstants.SAMPLING_FREQUENCY
        )
        cos = RO_amp * np.cos(
            2 * np.pi * array * f_RO_mod / DeviceConstants.SAMPLING_FREQUENCY
        )

        Iwave = (cos + sin) / np.sqrt(2)
        Qwave = (cos - sin) / np.sqrt(2)

        self.awg_sequence_acquisition_and_pulse(
            Iwave, Qwave, acquisition_delay, dig_trigger=dig_trigger, ch=ch, slot=slot
        )

    def configure_spectroscopy(
        self,
        start_frequency: float,
        frequency_step: float,
        settling_time: float,
        dio_trigger: bool = True,
        envelope=None,
        ch: int = Defaults.SingleQubitExperiments.CHANNEL,
    ) -> None:
        """
        Configures a sequencer-based continuous or pulsed spectroscopy experiment on the specified channel, where each step
        may be triggered via DIO. The number of frequency steps and averages is specified by the variables stored in the
        "num_samples" and "num_averages" user register, respectively.

        Args:
            start_frequency: starting point of the frequency sweep
            frequency_step: offset added to the previous frequency value at each new step
            settling_time: interval in seconds between the frequency setting and the start of the acquisition.
            dio_trigger: specify whether the different frequency steps are self-triggered, or via a DIO trigger.
            envelope: optional complex array parameter specifying the envelope to be applied to the resonator excitation.
                      If set to None, a continuous wave (CW) spectroscopy experiment will be performed.
            ch: index specifying which qachannel to perform the experiment on.
        """
        self.wait_dly(settling_time)
        self._active_slots = {ch: [Defaults.SingleQubitExperiments.SLOT]}

        if envelope is not None:
            self.set(
                f"qachannels_{ch}_spectroscopy_envelope_enable",
                1,
            )
            self.set(
                f"qachannels_{ch}_spectroscopy_envelope",
                envelope,
            )
        else:
            self.set(
                f"qachannels_{ch}_spectroscopy_envelope_enable",
                0,
            )

        features, program = SeqC.spectroscopy(
            start_frequency, frequency_step, dio_trigger=dio_trigger
        )
        self._seqc_features = features
        self._awg_program[ch] = program

    def awg_sequence_acquisition_and_DIO_RED_test(
        self,
        acquisition_delay=0,
        dio_out_vect=None,
    ) -> None:
        raise NotImplementedError

    def awg_sequence_test_pattern(self, dio_out_vect=None) -> None:
        raise NotImplementedError

    def spec_mode_on(
        self, acq_length=1 / 1500, IF=20e6, ro_amp=0.1, wint_length=2 ** 14
    ) -> None:
        raise NotImplementedError(
            "Use the method 'configure_spectroscopy' and the 'spectroscopy' tag in the corresponding 'acquisition_...' "
            "methods to specify a spectroscopy experiment."
        )

    ##########################################################################
    # weighted integration helpers ported from UHF
    ##########################################################################

    def prepare_SSB_weight_and_rotation(
        self,
        IF: float,
        rotation_angle: float = 0,
        length: float = DeviceConstants.Readout.MAX_LENGTH
        / DeviceConstants.SAMPLING_FREQUENCY,
        scaling_factor: float = 1,
        ch: int = Defaults.SingleQubitExperiments.CHANNEL,
        slot: int = Defaults.SingleQubitExperiments.SLOT,
    ) -> None:
        """
        Uploads SSB integration weights to a specified integrator slot.

        Args:
            IF: demodulation frequency.
            rotation_angle: rotation to apply to the weights.
            length: duration in seconds of the weights function.
            scaling_factor: additional scaling factor to apply to the weight function.
            ch: index specifying the qachannel.
            slot: index specifying which integrator slot to upload the weight function to.
        """
        # to be consistent in naming
        duration = length
        if duration_to_length(duration) > DeviceConstants.Readout.MAX_LENGTH:
            raise ziValueError(
                f"SSB integration weights of duration {duration} exceed the maximum readout length of "
                f"{DeviceConstants.Readout.MAX_LENGTH}"
            )
        tbase = np.arange(
            0,
            duration,
            1 / DeviceConstants.SAMPLING_FREQUENCY,
        )
        cos = scaling_factor * np.array(np.cos(2 * np.pi * IF * tbase + rotation_angle))
        sin = scaling_factor * np.array(np.sin(2 * np.pi * IF * tbase + rotation_angle))
        weight = cos - 1j * sin

        self.set(f"qachannels_{ch}_readout_integration_weights_{slot}_wave", weight)

    def prepare_DSB_weight_and_rotation(
        self,
        IF: float,
        ch: int = Defaults.SingleQubitExperiments.CHANNEL,
        slot: int = Defaults.SingleQubitExperiments.SLOT,
    ) -> None:
        raise NotImplementedError

    ##########################################################################
    # overrides for CalInterface interface
    ##########################################################################

    def output_dio_calibration_data(
        self, dio_mode: str, port: int = 0
    ) -> Tuple[int, List]:

        self.configure_awg_from_string(
            awg_nr=DioCalibration.GENERATOR_INDEX,
            program_string=DioCalibration.SHFQA_TO_CC_PROGRAM,
        )
        shfqa_utils.enable_sequencer(
            self.daq,
            self.devname,
            channel_index=DioCalibration.GENERATOR_INDEX,
            single=1,
        )

        DIO_SEQUENCER_1_OUTPUT = 32
        self.set("dios_0_mode", DIO_SEQUENCER_1_OUTPUT)
        generator_dio_path = (
            f"qachannels_{DioCalibration.GENERATOR_INDEX}_generator_dio_"
        )
        self.set(
            generator_dio_path + "valid_polarity",
            uhf_compatibility.Dio.VALID_POLARITY,
        )
        self.set(
            generator_dio_path + "valid_index",
            uhf_compatibility.Dio.VALID_INDEX,
        )
        dio_mask = 0x7FFF
        expected_sequence = []
        return dio_mask, expected_sequence

    def calibrate_dio_protocol(
        self, dio_mask: int, expected_sequence: List, port: int = 0
    ):
        self._ensure_activity(mask=dio_mask)

        log.info(f"{self.devname}: Finding valid delays...")

        valid_delays = []
        for delay in DioCalibration.DELAYS:
            self._set_dio_calibration_delay(delay)
            self.daq.setInt(f"/{self.devname}/raw/dios/0/error/timingclear", 1)
            time.sleep(3)
            timing_error = self.daq.getInt(
                f"/{self.devname}/raw/dios/0/error/timingsticky"
            )
            if timing_error == 0:
                valid_delays.append(delay)

        if not valid_delays:
            raise Exception("DIO calibration failed! No valid delays found")

        log.info(f"{self.devname}: Valid delays are {valid_delays}")

        subseqs = [[valid_delays[0]]]
        for delay in valid_delays:
            last_subseq = subseqs[-1]
            last_delay = last_subseq[-1]
            delay_following_sequence = not last_subseq or last_delay == delay - 1
            if delay_following_sequence:
                subseqs[-1].append(delay)
            else:
                subseqs.append([delay])

        longest_subseq = max(subseqs, key=len)
        delay = len(longest_subseq) // 2 + longest_subseq[0]

        self._set_dio_calibration_delay(delay)

        # Clear all detected errors (caused by DIO timing calibration)
        self.check_errors(errors_to_ignore=["AWGDIOTIMING"])

    def _ensure_activity(self, mask: int):
        """
        Record DIO data and test whether there is activity on the bits activated in the DIO protocol.
        """
        log.debug(f"{self.devname}: Testing DIO activity.")

        # The sequencer must be running in order for the RT logger to acquire data
        self.configure_awg_from_string(
            awg_nr=DioCalibration.GENERATOR_INDEX,
            program_string=DioCalibration.CC_TO_SHFQA_PROGRAM,
        )
        shfqa_utils.enable_sequencer(
            self.daq,
            self.devname,
            channel_index=DioCalibration.GENERATOR_INDEX,
            single=1,
        )

        rt_logger_path = (
            f"qachannels_{DioCalibration.GENERATOR_INDEX}_generator_rtlogger_"
        )
        self.set(rt_logger_path + "mode", 1)
        self.set(rt_logger_path + "starttimestamp", 0)
        self.set(rt_logger_path + "enable", 1)
        time.sleep(0.01)
        path = f"/{self.devname}/qachannels/0/generator/rtlogger/data"
        data = self.daq.get(path, settingsonly=False, flat=True)[path][0]["vector"]
        dio_snapshot = data[1::2]
        self.set(rt_logger_path + "enable", 0)
        self.set(rt_logger_path + "clear", 1)

        self.set(f"qachannels_{0}_generator_enable", 0)

        if dio_snapshot is None:
            raise ziValueError(f"{self.devname}: Failed to get DIO snapshot!")

        activity = 0
        for dio_value in dio_snapshot:
            activity |= int(dio_value) & mask

        if activity != mask:
            raise ziValueError(
                f"{self.devname}: Did not see activity on all bits! Got 0x{activity:08x}, expected 0x{mask:08x}."
            )

    ##########################################################################
    # Overriding private ZI_base_instrument methods
    ##########################################################################

    def _check_devtype(self) -> None:
        if "SHFQA" not in self.devtype:
            raise ziDeviceError(
                f"Device {self.devname} of type {self.devtype} is not a SHFQA instrument!"
            )

    def _check_options(self) -> None:
        pass

    def _check_versions(self) -> None:
        if self.geti("system/fwrevision") < DeviceConstants.MinRevisions.FW:
            raise ziVersionError(
                f"Insufficient firmware revision detected! Need {DeviceConstants.MinRevisions.FW}, "
                f'got {self.geti("system/fwrevision")}!'
            )
        if self.geti("system/fpgarevision") < DeviceConstants.MinRevisions.FPGA:
            raise ziVersionError(
                f"Insufficient FPGA revision detected! Need {DeviceConstants.MinRevisions.FPGA}, "
                f'got {self.geti("system/fpgarevision")}!'
            )

    def _check_awg_nr(self, awg_nr: int) -> None:
        if awg_nr > self._num_channels():
            raise ziValueError(f"Invalid AWG index of {awg_nr} detected!")

    def _num_channels(self) -> int:
        if self.devtype == "SHFQA4":
            return 4
        return 2

    def _num_awgs(self) -> int:
        return self._num_channels()

    ##########################################################################
    # properties
    ##########################################################################

    @property
    def active_slots(self) -> dict:
        """
        Returns the currently active slots in the driver.
        """
        if not self._active_slots:
            log.debug(
                log.debug(
                    f"{self.devname}: No slots are currently active. Make sure to setup an experimental "
                    f"sequence on the awg to pursue."
                )
            )
        return self._active_slots

    @property
    def active_channels(self) -> tuple:
        """
        Returns the currently active channels in the driver.
        """
        return tuple(ch for ch, slots in self.active_slots.items())

    ##########################################################################
    # startup helpers
    ##########################################################################

    def validate_config(self):
        """
        Top level function validating currently stored driver configuration. Can be used e.g. before actually pushing
        data to the device.
        """
        if self._seqc_features is None:
            raise ziConfigurationError("Missing awg sequence.")
        if not self._active_slots:
            raise ziConfigurationError("No readout units are active.")
        self._result_mode_manager.validate_config()

    def _push_sequencer_programs(self) -> None:
        """
        Uploads the cached sequencer programs corresponding to the currently active channels to the device.
        """
        for ch in self.active_channels:
            if self._awg_program[ch] is None:
                raise ziConfigurationError(
                    f"No sequencer program defined for active channel {ch}."
                )
            self.configure_awg_from_string(
                awg_nr=ch, program_string=self._awg_program[ch]
            )

    def _push_waveforms(self) -> None:
        """
        Uploads the currently active codewords to the device.
        """
        if not self._seqc_features.codewords:
            return
        for codeword in self._codeword_manager.active_codewords:
            try:
                waveform = self.get(_waveform_name(codeword))
                ch, slot = self._generator_slot(codeword)
                log.debug(
                    f"{self.devname}: Uploading {codeword} to slot {slot} of generator {ch}."
                )
                self.set(f"qachannels_{ch}_generator_waveforms_{slot}_wave", waveform)
            except ziValueError:
                pass

    def _push_hold_off_delay(self):
        for ch in self.active_channels:
            path = f"qachannels_{ch}_generator_userregs_"
            self.set(
                f"{path}{UserRegisters.HOLDOFF_DELAY}",
                hold_off_length(self._acquisition_time + self._wait_dly),
            )

    def _push_loop_config(self):
        loop_sizes = self._loop_sizes()
        for ch in self.active_channels:
            path = f"qachannels_{ch}_generator_userregs_"
            self.set(
                f"{path}{UserRegisters.INNER_LOOP}",
                loop_sizes.inner,
            )
            self.set(
                f"{path}{UserRegisters.OUTER_LOOP}",
                loop_sizes.outer,
            )

    def push_to_device(self):
        """
        Top level function responsible for pushing the shallow device configuration stored in the driver to the actual
        device.
        """
        self.validate_config()

        if self._use_dio:
            self._configure_codeword_protocol()

        self._push_sequencer_programs()
        self._push_waveforms()
        self._push_loop_config()
        self._push_hold_off_delay()

        self._result_mode_manager.push_acquisition_unit_config()

    def _enable_channels(self) -> None:
        """
        Enables the RF frontends corresponding to the currently active channels.
        """
        for ch in self.active_channels:
            path = f"qachannels_{ch}_"
            self.set(path + "input_on", 1)
            self.set(path + "output_on", 1)

    def _start_sequencers(self) -> None:
        """
        Starts the sequencers of the currently active channels.
        """
        for ch in self.active_channels:
            shfqa_utils.enable_sequencer(
                self.daq,
                self.devname,
                channel_index=ch,
                single=1 if self._single else 0,
            )

    ##########################################################################
    # experiment control helpers
    ##########################################################################

    def _wait_sequencers_finished(self) -> None:
        """
        Blocks until the currently active sequencers have stopped execution.

        Args:
            timeout: timeout specifying at which point sequencers that are still running are considered an error.

        Raises:
            TimeoutError: sequencers are still running after the specified timeout.
        """
        for ch in self.active_channels:
            generator_path = (
                f"/{self.devname}/qachannels/{ch}/generator/sequencer/status"
            )
            wait_for_state_change(self.daq, generator_path, 0, timeout=self.timeout())

    ##########################################################################
    # setup helpers
    ##########################################################################

    def _subscribe_to_result_nodes(self) -> None:
        """
        Subscribes to the device nodes containing measurement results specific to the current result mode.
        """
        self._subscribed_paths = []
        for path in self._result_mode_manager.result_paths():
            self.subs(path)
            self._subscribed_paths.append(path)

    def _configure_codeword_protocol(self) -> None:
        """
        Configures the device to work with the central controller DIO paradigm.
        """
        dio_path = "dios_0_"
        self.set(dio_path + "mode", uhf_compatibility.Dio.MODE)
        self.set(dio_path + "drive", uhf_compatibility.Dio.DRIVE)
        for ch in self.active_channels:
            generator_dio_path = f"qachannels_{ch}_generator_dio_"
            self.set(
                generator_dio_path + "valid_polarity",
                uhf_compatibility.Dio.VALID_POLARITY,
            )
            self.set(
                generator_dio_path + "valid_index", uhf_compatibility.Dio.VALID_INDEX
            )

    def _loop_sizes(self) -> SeqC.LoopSizes:
        if not self._seqc_features.outer_loop:
            return SeqC.LoopSizes(inner=self._samples * self._averages, outer=0)
        if self._averaging_mode == "sequential":
            return SeqC.LoopSizes(inner=self._averages, outer=self._samples)
        if self._averaging_mode == "cyclic":
            return SeqC.LoopSizes(inner=self._samples, outer=self._averages)
        raise ziConfigurationError

    ##########################################################################
    # 'public' utility functions ported from UHF
    ##########################################################################

    def reset_acquisition_params(self) -> None:
        self.reset_user_registers()
        self.reset_crosstalk_matrix()
        self.reset_correlation_params()

    def reset_user_registers(self) -> None:
        default_value = 0
        log.info(f"{self.devname}: Setting user registers to {default_value}")
        for ch in self.active_channels:
            for userreg_index in range(DeviceConstants.NUM_REGISTERS_PER_SEQUENCER):
                self.set(
                    f"qachannels_{ch}_generator_userregs_{userreg_index}",
                    default_value,
                )

    def reset_crosstalk_matrix(self) -> None:
        raise NotImplementedError("Crosstalk suppression is not available on SHFQA.")

    def reset_correlation_params(self) -> None:
        default_value = 0
        for ch, slots in self.active_slots.items():
            for slot in slots:
                self.set(
                    f"qachannels_{ch}_readout_discriminators_{slot}_threshold",
                    default_value,
                )
        log.info(f"{self.devname}: Correlation is not yet implemented")

    def reset_rotation_params(self) -> None:
        raise NotImplementedError(
            "Rotation of the integration weights is not supported by the SHFQA. Please include such corrections in the weights uploaded to the instrument."
        )

    def upload_crosstalk_matrix(self, matrix) -> None:
        raise NotImplementedError("Crosstalk suppression is not available on SHFQA.")

    def download_crosstalk_matrix(self, nr_rows=10, nr_cols=10) -> None:
        raise NotImplementedError("Crosstalk suppression is not available on SHFQA.")

    ##########################################################################
    # 'public' print overview functions ported from UHF
    ##########################################################################

    def print_correlation_overview(self) -> None:
        raise NotImplementedError("Crosstalk suppression is not available on SHFQA.")

    def print_deskew_overview(self) -> None:
        raise NotImplementedError(
            "Input deskew is not available on the SHFQA, as the rf down-conversion is performed in-the-box."
        )

    def print_crosstalk_overview(self) -> None:
        raise NotImplementedError("Crosstalk suppression is not available on SHFQA.")

    def print_integration_overview(self) -> None:
        raise NotImplementedError

    def print_rotations_overview(self) -> None:
        raise NotImplementedError(
            "Rotation of the integration weights is not supported by the SHFQA. Please include such corrections in the weights uploaded to the instrument."
        )

    def print_thresholds_overview(self) -> None:
        msg = "\t Thresholds overview \n"
        for ch, slots in self.active_slots.items():
            for slot in slots:
                path = f"qachannels/{ch}/readout/discriminators/{slot}/threshold"
                msg += f"Channel {ch} Threshold {slot}: {self.getd(path)}"
        print(msg)

    def print_user_regs_overview(self) -> None:
        msg = "\t User registers overview \n"
        user_reg_funcs = [""] * DeviceConstants.NUM_REGISTERS_PER_SEQUENCER
        user_reg_funcs[UserRegisters.INNER_LOOP] = SeqC._VAR_INNER_LOOP_SIZE
        user_reg_funcs[UserRegisters.OUTER_LOOP] = SeqC._VAR_OUTER_LOOP_SIZE
        user_reg_funcs[UserRegisters.NUM_ERRORS] = SeqC._VAR_NUM_ERRORS

        for ch in self.active_channels:
            for userreg_index in range(DeviceConstants.NUM_REGISTERS_PER_SEQUENCER):
                value = self.geti(f"qachannels/{ch}/generator/userregs/{userreg_index}")
                func = user_reg_funcs[userreg_index]
                msg += f"Sequencer {ch} User reg {userreg_index}: \t{value}\t({func})\n"
        print(msg)

    def print_overview(self) -> None:
        self.print_integration_overview()
        self.print_thresholds_overview()
        self.print_user_regs_overview()


def _waveform_name(codeword: int) -> str:
    """
    Returns a waveform name based on codeword number.

    Args:
        codeword: value of the codeword to use for the waveform name generation.
    """
    return "wave_cw{:03}".format(codeword)
