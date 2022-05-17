"""Module collecting functionality relative to easing the transition from UHFQC to SHFQA."""

import numpy as np
import logging

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument import (
    ziConfigurationError,
    ziValueError,
)
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.shfqa_codewords import (
    CodewordManager,
)
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.internal._shfqa import (
    preprocess_generator_waveform,
)

log = logging.getLogger(__name__)


class Dio:
    """
    Class collecting functionality relating to the UHFQA DIO compatibility mode under one namespace.
    """

    MODE = 17  # UHFQA compatibility mode
    VALID_POLARITY = 2  # high polarity
    VALID_INDEX = 16
    DRIVE = 0x3

    MAX_NUM_CHANNELS = 2
    MAX_NUM_RESULTS_PER_CHANNEL = 7
    MAX_NUM_RESULTS = MAX_NUM_CHANNELS * MAX_NUM_RESULTS_PER_CHANNEL
    MAX_NUM_CODEWORDS = 2 ** MAX_NUM_RESULTS

    "               <---- Input ----> <--- Output --->"
    _CW_MASK_STR = "01111111 11111110 00000000 00000000"
    "                <-ch2-> <-ch1->"
    INPUT_SHIFT = 17

    @staticmethod
    def codeword_mask() -> int:
        """
        Returns an integer mask allowing to extract a codeword from a DIO trigger value.
        """
        binary_mask_str = Dio._CW_MASK_STR.replace(" ", "")
        return int("".join(bit for bit in binary_mask_str), 2)

    @staticmethod
    def check_num_results(num_results: int) -> None:
        """
        Checks whether the provided number of results is compatible with the DIO compatibility mode.

        Args:
            num_results: total number of results to be checked

        Raises:
            ziConfigurationError: the number of results is incompatible.
        """
        if num_results > Dio.MAX_NUM_RESULTS:
            raise ziConfigurationError(
                f"The UHFQA DIO compatibility mode on the SHFQA only supports up to "
                f"{Dio.MAX_NUM_RESULTS} results!"
            )

    @staticmethod
    def check_codeword(codeword: int) -> None:
        """
        Checks whether the provided codeword is compatible with the DIO compatibility mode.

        Args:
            codeword: codeword to be checked

        Raises:
            ziConfigurationError: the provided codeword is incompatible.

        """
        if codeword > Dio.MAX_NUM_CODEWORDS:
            raise ziConfigurationError(
                f"The UHFQA DIO compatibility mode on the SHFQA only supports up to "
                f"{Dio.MAX_NUM_CODEWORDS} codewords!"
            )


class BruteForceCodewordManager(CodewordManager):
    """
    Default "brute force" mapping between codewords and generators: each specified
    codeword is assigned to a single generator in ascending order. Supports a maximum
    of Dio.MAX_NUM_RESULTS codewords simultaneously.
    """

    # Override
    def _codeword_slots(self, codeword: int) -> dict:
        Dio.check_codeword(codeword)
        try:
            result_index = self._active_codewords.index(codeword)
        except ValueError:
            raise CodewordManager.Error(
                "Trying to access a codeword in BruteForceCodewordManager that was not set."
            )
        ch, slot = divmod(result_index, Dio.MAX_NUM_RESULTS_PER_CHANNEL)
        return {ch: [slot]}

    # Override
    def _check_num_results(self, num_results: int) -> None:
        Dio.check_num_results(num_results)

    # Override
    def _check_codeword(self, codeword) -> None:
        Dio.check_codeword(codeword)

    # Override
    def _default_active_codewords(self) -> list:
        return range(Dio.MAX_NUM_RESULTS)


def check_and_convert_waveform(Iwave, Qwave) -> None:
    """
    Validates a pair of real arrays candidate for upload onto a generator slot of the SHFQA. The function
    additionally snaps the length of the array to the granularity of the generator.

    Args:
        Iwave: real numbers array representing the in-phase component of a candidate complex waveform to be uploaded
               to a device generator slot.
        Qwave: real numbers array representing the quadrature component of a candidate complex waveform to be uploaded
               to a device generator slot.
    """
    if Iwave is not None and Qwave is not None:
        Iwave = preprocess_generator_waveform(Iwave)
        Qwave = preprocess_generator_waveform(Qwave)
        if not (len(Iwave) == len(Iwave)):
            raise ziValueError(
                f"Length of I component {len(Iwave)} does not match length of Q component {len(Qwave)}"
            )
        return np.array([Iwave[i] + 1j * Qwave[i] for i in range(len(Iwave))])
    elif Iwave is not None:
        Iwave = preprocess_generator_waveform(Iwave)
        return np.array(Iwave)
    elif Qwave is not None:
        Qwave = preprocess_generator_waveform(Qwave)
        return np.array([1j * Qwave[i] for i in range(len(Qwave))])
