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
    """
    This is the PycQED driver for the 2.0 Gsample/s SHFQA developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5/3.6 ucs4 19.05 for 64bit Windows from
        http://www.zhinst.com/downloads, https://people.zhinst.com/~niels/
    2. upload the latest firmware to the SHFQA using the LabOne GUI
    """

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
        # TODO: adapt arguments default-values to SHFQA
        # TODO: implement

        super().__init__(name=name, device=device, interface=interface, address=address,
                         server=server, port=port, nr_integration_channels=nr_integration_channels,
                         **kw)

    ##########################################################################
    # 'public' overrides for SHFQA_core
    ##########################################################################

    def load_default_settings(self, upload_sequence=True) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    ##########################################################################
    # 'public' functions: generic AWG/waveform support
    ##########################################################################

    def load_awg_program_from_file(self, filename) -> None:
        raise NotImplementedError

    def awg_file(self, filename) -> None:
        raise NotImplementedError

    def awg_update_waveform(self, index, data) -> None:
        raise NotImplementedError(
            'Method not implemented! Please use the corresponding waveform parameters \'wave_chN_cwM\' to update waveforms!')

    ##########################################################################
    # 'public' functions: DIO support
    ##########################################################################

    def plot_dio(self, bits=range(32), line_length=64) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    ##########################################################################
    # 'public' functions: weight & matrix function helpers
    ##########################################################################

    def prepare_SSB_weight_and_rotation(self, IF,
                                        weight_function_I=0,
                                        weight_function_Q=1,
                                        rotation_angle=0,
                                        length=4096 / 1.8e9,
                                        scaling_factor=1) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def prepare_DSB_weight_and_rotation(self, IF, weight_function_I=0, weight_function_Q=1) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError


    ##########################################################################
    # Overriding private ZI_base_instrument methods
    ##########################################################################

    def _add_extra_parameters(self) -> None:
        # TODO: implement
        pass
        
    def _codeword_table_preamble(self, awg_nr) -> str:
        raise NotImplementedError

    def plot_dio_snapshot(self, bits=range(32)):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    ##########################################################################
    # Overriding Qcodes InstrumentBase methods
    ##########################################################################

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update =None,
                      params_to_exclude = None ):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

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
    def awg_sequence_acquisition_and_DIO_triggered_pulse(
            self, Iwaves=None, Qwaves=None, cases=None, acquisition_delay=0, timeout=5) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def awg_sequence_acquisition_and_DIO_RED_test(
            self, Iwaves=None, Qwaves=None, cases=None, acquisition_delay=0,
            dio_out_vect=None, timeout=5):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def awg_sequence_test_pattern(
            self,
            dio_out_vect=None):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def awg_sequence_acquisition_and_pulse(self, Iwave=None, Qwave=None, acquisition_delay=0, dig_trigger=True) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def awg_sequence_acquisition(self):
        raise NotImplementedError

    def awg_debug_acquisition(self, dly=0):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError


    def awg_sequence_acquisition_and_pulse_SSB(
            self, f_RO_mod, RO_amp, RO_pulse_length, acquisition_delay, dig_trigger=True) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def spec_mode_on(self, acq_length=1/1500, IF=20e6, ro_amp=0.1, wint_length=2**14) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    ##########################################################################
    # overrides for CalInterface interface
    ##########################################################################

    def output_dio_calibration_data(self, dio_mode: str, port: int=0) -> Tuple[int, List]:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def calibrate_dio_protocol(self, dio_mask: int, expected_sequence: List, port: int=0):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    ##########################################################################
    # DIO calibration functions for *CC*
    ##########################################################################

    def calibrate_CC_dio_protocol(self, CC, feedline=None, verbose=False) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise DeprecationWarning("calibrate_CC_dio_protocol is deprecated, use instrument_drivers.library.DIO.calibrate")


##########################################################################
# Module level functions
##########################################################################

def awg_sequence_acquisition_preamble():
    raise NotImplementedError
