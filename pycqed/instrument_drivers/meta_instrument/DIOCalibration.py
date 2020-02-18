from abc import ABC, abstractmethod
from typing import Tuple,List


class DIOCalibration(ABC):
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
            dio_mask: mask defining bits that are actually toggled. On certain architectures this may be a subset of
            the bits used by dio_mode
            expected_sequence: list, may be empty
        """
        pass

    @abstractmethod
    def calibrate_dio_protocol(self, dio_mask: int, expected_sequence: List, port: int=0):
        """
        calibrate DIO protocol timing. Requires valid input signal on bits defined by dio_mask

        Args:
            dio_mask: mask defining bits that are actually toggled. On certain architectures this may be a subset of
            the bits used by dio_mode
            expected_sequence: list, may be empty
            port: the port on which to receive the data

        Returns:
        """
        pass


def calibrate(self,
              sender: DIOCalibration,
              sender_dio_mode: str,
              receiver: DIOCalibration,
              sender_port: int=0,
              receiver_port: int=0
              ):
    """
    calibrate DIO timing between two physical instruments featuring DIO (i.e. implementing interface DIOCalibration)

    Args:
        sender: instrument driving DIO (Qutech CC/QCC/CC-light)
        sender_dio_mode: the DIO mode for which calibration is requested
        receiver: instrument receiving DIO (ZI UHFQA/HDAWG, QuTech CC/QWG)
        sender_port: the port on which to generate the data (other ports are also ALLOWED to produce data)
        receiver_port: the port on which to receive the data
    """
    dio_mask,expected_sequence = sender.output_dio_calibration_data(dio_mode=sender_dio_mode, port=sender_port)
    # FIXME: disable receiver connector outputs? And other receivers we're not aware of?
    receiver.calibrate_dio_protocol(dio_mask=dio_mask, expected_sequence=expected_sequence, port=receiver_port)
    # FIXME: stop sender