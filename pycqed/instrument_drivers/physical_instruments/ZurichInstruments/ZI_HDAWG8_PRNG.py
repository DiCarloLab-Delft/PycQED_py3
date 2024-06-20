# -------------------------------------------
# Module containing subclass of ZI_HDAWG8.
# Subclass overwrites upload-codeword-program
# Adds functionality to:
# - read dio 32-bit number
# - mask relevant part for AWG core
# - compare to pseudo random number generator (PRNG)
# - run codeword based on this (random) outcome
# -------------------------------------------
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8


@dataclass(frozen=True)
class StochasticTrigger:
    """
    Data class, containing information about a stochastic trigger.
    """
    awg_core_index: int
    """HDAWG core index (0, 1, 2 or 3)."""
    trigger_probability: float
    """Probability between 0 and 1."""
    default_codeword: int
    """Codeword played (by default) when trigger probability is not satisfied."""
    trigger_codeword: int
    """Codeword played when trigger probability is satisfied."""

    # region Class Properties
    @property
    def codeword_shift(self) -> int:
        """
        Example: 3221291008 => 11-0000000-0000001-00-0000000-0000000
                                  --AWG4- --AWG3-    --AWG2- --AWG1- (Note reverse order)
        :return: codeword shift based on 32-bit DIO conventions.
        """
        return {
            0: 0,
            1: 7,
            2: 16,
            3: 23,
        }[self.awg_core_index]

    @property
    def stochastic_trigger_bit(self) -> float:
        bit_size: int = 7
        return self.codeword_shift + bit_size - 1

    @property
    def codeword_mask(self) -> str:
        stochastic_trigger_bit: int = self.stochastic_trigger_bit
        return hex(2 ** stochastic_trigger_bit)

    @property
    def probability_range(self) -> Tuple[int, int]:
        """:return: Integer range for AWG PRNG range."""
        invalid_probability: bool = self.trigger_probability <= 0 or self.trigger_probability > 1
        if invalid_probability:
            raise ValueError(
                f"Probability must be between 0 and 1 (exclusive of 0.0, instead {self.trigger_probability}")
        # lb: int = 0
        # ub: int = max(
        #     1,
        #     min(
        #         round(1. / self.trigger_probability),
        #         2 ** 16,
        #     )
        # ) - 1  # PRNG range 0 to 2**16 - 1
        # return lb, ub
        return 0, 2**16 - 2

    @property
    def probability_cutoff(self) -> int:
        return int(round(self.trigger_probability * 2**16)) - 1
    # endregion

    # region Class Methods
    def construct_awg_stochastic_program(self) -> str:
        """:return: Program for AWG core to run stochastic trigger."""
        prng_lb, prng_ub = self.probability_range
        result: str = (
            "var dioIndex = 0;\n"
            f"setPRNGRange({prng_lb}, {prng_ub});\n"
            f"var prngCutoff = {self.probability_cutoff};\n"
            f"var dioMask = {self.codeword_mask};\n"
            f"var triggerCodeword = {self.trigger_codeword};\n"
            f"var defaultCodeword = {self.default_codeword};\n"
            "\n"
            "while (1) {\n"
            "    var prng_value = getPRNGValue();\n"
            "    // Wait for a trigger on the DIO interface\n"
            "    waitDIOTrigger();\n"
            "    // Process DIO trigger signal\n"
            "    if (getDIO() & dioMask) {\n"  # If PRNG bit is active [1XXXXXX]
            "        if (prng_value < prngCutoff) {\n"  # If PRNG value != 0
            "            executeTableEntry(triggerCodeword);\n"
            "        } else {\n"
            "            executeTableEntry(defaultCodeword);\n"
            "        }\n"
            "    } else {\n"
            "        playWaveDIO();\n"
            "    }\n"
            "}"
        )
        return result
    # endregion


class ZI_HDAWG8_PRNG(ZI_HDAWG8):
    """
    Behaviour class, driver for ZurichInstruments HDAWG8 instrument.
    Codeword program reserves last bit of 7-bit codeword for PRNG trigger.
    The behaviour of this trigger is set to do nothing by default and can be updated manually.
    An important difference between this class and the parent class is the change in codeword program.
    By adding processing logic to the AWG core, the minimal time to receive, process and execute pulses is extended.
    Note that the default behaviour of doing 'nothing' keeps the same processing step for timing consistencies.
    """

    # region Class Constructor
    def __init__(self, name: str, device: str, interface: str = '1GbE', server: str = 'localhost', port=8004, num_codewords: int = 64, **kw):
        super().__init__(name=name, device=device, interface=interface, server=server, port=port, num_codewords=num_codewords, **kw)
        identity_codeword: int = 0
        self.awg_stochastic_triggers: Dict[int, StochasticTrigger] = {
            0: StochasticTrigger(awg_core_index=0, trigger_probability=1.0, default_codeword=identity_codeword, trigger_codeword=identity_codeword),
            1: StochasticTrigger(awg_core_index=1, trigger_probability=1.0, default_codeword=identity_codeword, trigger_codeword=identity_codeword),
            2: StochasticTrigger(awg_core_index=2, trigger_probability=1.0, default_codeword=identity_codeword, trigger_codeword=identity_codeword),
            3: StochasticTrigger(awg_core_index=3, trigger_probability=1.0, default_codeword=identity_codeword, trigger_codeword=identity_codeword),
        }
    # endregion

    # region Class Methods
    def get_stochastic_trigger(self, awg_core_index: int) -> StochasticTrigger:
        """:return: Stochastic trigger dataclass if awg_core_index exists."""
        return self.awg_stochastic_triggers[awg_core_index]

    def set_awg_stochastic_trigger(self, stochastic_trigger: StochasticTrigger) -> None:
        """:sets: stochastic trigger dataclass."""
        awg_core_index: int = stochastic_trigger.awg_core_index
        allowed_awg_core_indices: List[int] = [0, 1, 2, 3]
        if awg_core_index not in allowed_awg_core_indices:
            raise ValueError(f"Choice must be within {allowed_awg_core_indices}, instead {awg_core_index}.")
        self.awg_stochastic_triggers[awg_core_index] = stochastic_trigger
        return None

    def upload_codeword_program(self, awgs: np.ndarray = np.arange(4)):
        """
        Generates a program that plays the codeword waves for each channel.
        :param awgs: (np.ndarray) the awg numbers to which to upload the codeword program.
        """
        self._configure_codeword_protocol()

        # Type conversion to ensure lists do not produce weird results
        awgs = np.array(awgs)
        if awgs.shape == ():
            awgs = np.array([awgs])

        for awg_nr in awgs:
            self._awg_program[awg_nr] = self.awg_stochastic_triggers[awg_nr].construct_awg_stochastic_program()
            self._awg_needs_configuration[awg_nr] = True
    # endregion
