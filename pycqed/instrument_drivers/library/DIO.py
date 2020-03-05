import sys
from abc import ABC, abstractmethod
from typing import Tuple,List


class CalInterface(ABC):
    # Abstract base class to define interface for DIO calibration
    # Use calibrate() to perform the calibration

    @abstractmethod
    def output_dio_calibration_data(self, dio_mode: str, port: int=0) -> Tuple[int, List]:
        """
        output DIO calibration pattern

        Args:
            dio_mode: the DIO mode for which calibration is requested
            port: the port on which to generate the data (other ports are also ALLOWED to produce data)

        Returns:
            dio_mask: mask defining bits that are actually toggled (codeword, trigger, toggle). On certain architectures
            this may be a subset of the bits used by dio_mode
            expected_sequence: list, may be empty
        """
        pass

    @abstractmethod
    def calibrate_dio_protocol(self, dio_mask: int, expected_sequence: List, port: int=0):
        """
        calibrate DIO protocol timing. Requires valid input signal on bits defined by dio_mask

        Args:
            dio_mask: mask defining bits that are actually toggled (codeword, trigger, toggle). On certain architectures
            this may be a subset of the bits used by dio_mode
            expected_sequence: list, may be empty
            port: the port on which to receive the data

        Returns:
        """
        pass


def calibrate(sender: CalInterface,
              receiver: CalInterface,
              sender_dio_mode: str='',
              sender_port: int=0,
              receiver_port: int=0
              ):
    """
    calibrate DIO timing between two physical instruments featuring DIO (i.e. implementing interface CalInterface)

    Args:
        sender: instrument driving DIO (Qutech CC/QCC/CC-light)
        sender_dio_mode: the DIO mode for which calibration is requested
        receiver: instrument receiving DIO (ZI UHFQA/HDAWG, QuTech CC/QWG)
        sender_port: the port on which to generate the data (other ports are also ALLOWED to produce data)
        receiver_port: the port on which to receive the data
    """
    # FIXME: allow list of senders or receivers
    dio_mask,expected_sequence = sender.output_dio_calibration_data(dio_mode=sender_dio_mode, port=sender_port)
    # FIXME: disable receiver connector outputs? And other receivers we're not aware of?
    receiver.calibrate_dio_protocol(dio_mask=dio_mask, expected_sequence=expected_sequence, port=receiver_port)
    sender.stop()  # FIXME: not in interface


_control_modes = {
    # control mode definition, compatible with OpenQL CC backend JSON syntax

    # preferred names
    "awg8-mw-vsm": {
        "control_bits": [
            [7,6,5,4,3,2,1,0],
            [23,22,21,20,19,18,17,16]
        ],
        "trigger_bits": [31]
    },
    "awg8-mw-direct-iq": {
        "control_bits": [
            [6,5,4,3,2,1,0],
            [13,12,11,10,9,8,7],
            [22,21,20,19,18,17,16],
            [29,28,27,26,25,24,23]
        ],
        "trigger_bits": [31]
    },
    "awg8-flux": {
        # NB: please note that internally one HDQWG AWG unit handles 2 channels, which requires special handling of the waveforms
        "control_bits": [
            [2,1,0],
            [5,4,3],
            [8,7,6],
            [11,10,9],
            [18,17,16],
            [21,20,19],
            [24,23,22],
            [27,26,25]
        ],
        "trigger_bits": [31]
    },

    ########################################
    # compatibility
    ########################################
    "microwave": {  # alias for "awg8-mw-vsm"
        "control_bits": [
            [7, 6, 5, 4, 3, 2, 1, 0],
            [23, 22, 21, 20, 19, 18, 17, 16]
        ],
        "trigger_bits": [31]
    },
    "novsm_microwave": {  # alias for "awg8-mw-direct-iq"
        "control_bits": [
            [6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7],
            [22, 21, 20, 19, 18, 17, 16],
            [29, 28, 27, 26, 25, 24, 23]
        ],
        "trigger_bits": [31]
    },
    "flux": {  # alias for "awg8-flux"
        # NB: please note that internally one HDQWG AWG unit handles 2 channels, which requires special handling of the waveforms
        "control_bits": [
            [2, 1, 0],
            [5, 4, 3],
            [8, 7, 6],
            [11, 10, 9],
            [18, 17, 16],
            [21, 20, 19],
            [24, 23, 22],
            [27, 26, 25]
        ],
        "trigger_bits": [31]
    }
}


def get_shift_and_mask(dio_mode: str, channels: List[int]) -> Tuple[int, int]:
    # extract information for dio_mode from _control_modes
    control_mode = _control_modes.get(dio_mode)
    if control_mode is None:
        raise ValueError(f"Unsupported DIO mode '{dio_mode}'")
    control_bits = control_mode['control_bits']
    # FIXME: also return trigger_bits
    # trigger_bits = control_mode['trigger_bits']

    # calculate mask
    nr_channels = 8  # fixed assumption for HDAWG and dual-QWG combo
    nr_groups = len(control_bits)
    ch_per_group = nr_channels/nr_groups
    mask = 0
    shift = sys.maxsize
    for ch in channels:
        if ch<0 or ch >= nr_channels:
            raise ValueError(f"Illegal channel {ch}")
        group = int(ch // ch_per_group)
        for bit in control_bits[group]:
            mask = mask | (1 << bit)
            if bit<shift:
                shift = bit  # find lowest bit used

    return shift,mask>>shift
