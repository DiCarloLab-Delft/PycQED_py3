"""
    Base driver for the SHFQA instrument including all common functionality.
    Application dependent code can be found in the SHFQA module. 
"""

import time
import logging
import numpy as np

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
from pycqed.utilities.general import check_keyboard_interrupt

from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter

log = logging.getLogger(__name__)

##########################################################################
# Exceptions
##########################################################################

class ziSHFQASeqCError(Exception):
    """Exception raised when the configured SeqC program does
       not match the structure needed for a given measurement in terms
       of number of samples, number of averages or the use of a delay."""
    pass


class ziSHFQAHoldoffError(Exception):
    """Exception raised when a holdoff error has occurred in either the
    input monitor or result logging unit. Increase the delay between triggers
    sent to these units to solve the problem."""
    pass

class ziSHFQADIOActivityError(Exception):
    """Exception raised when insufficient activity is detected on the bits
    of the DIO to be used for controlling which qubits to measure."""
    pass

class ziSHFQADIOCalibrationError(Exception):
    """Exception raised when the DIO calibration fails, meaning no signal
    delay can be found where no timing violations are detected."""
    pass

##########################################################################
# Class
##########################################################################

class SHFQA_core(zibase.ZI_base_instrument):
    """
    This is the base PycQED driver for the 2.0 Gsample/s SHFQA developed
    by Zurich Instruments. The class implements functionality that are used by 
    the SHFQA driver.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5/3.6 ucs4 19.05 for 64bit Windows from
        http://www.zhinst.com/downloads, https://people.zhinst.com/~niels/
    2. upload the latest firmware to the SHFQA using the LabOne GUI
    """

    def __init__(self,
                 name,
                 device:                  str,
                 interface:               str = 'USB',
                 address:                 str = '127.0.0.1',
                 port:                    int = 8004,
                 nr_integration_channels: int = 10,
                 server:                  str = '',
                 **kw) -> None:
        # TODO: adapt arguments default-values to SHFQA
        # TODO: implement

        # Our base class includes all the functionality needed to initialize the parameters
        # of the object. Those parameters are read from instrument-specific JSON files stored
        # in the zi_parameter_files folder.
        super().__init__(name=name, device=device, interface=interface,
                         server=server, port=port, num_codewords=2**nr_integration_channels,
                         **kw)

    ##########################################################################
    # Overriding private ZI_base_instrument methods
    ##########################################################################

    def _check_devtype(self) -> None:
        if not self.devtype.startswith('SHFQA'):
              raise zibase.ziDeviceError(
                  'Device {} of type {} is not a SHFQA instrument!'.format(self.devname, self.devtype))

    def _check_options(self) -> None:
        # TODO: implement
        pass

    def _check_versions(self) -> None:
        # TODO: implement
        pass

    def _check_awg_nr(self, awg_nr) -> None:
        raise NotImplementedError

    def _num_channels(self) -> int:
        return int(self.devtype.split('SHFQA')[1])
        
    def _add_extra_parameters(self) -> None:
        raise NotImplementedError

    ##########################################################################
    # 'public' overrides for ZI_base_instrument
    ##########################################################################

    def assure_ext_clock(self) -> None:
        raise NotImplementedError

    def clear_errors(self) -> None:
        raise NotImplementedError

    def load_default_settings(self) -> None:
        raise NotImplementedError

    ##########################################################################
    # 'public' functions
    ##########################################################################

    def clock_freq(self):
        return 2.0e9

    ##########################################################################
    # 'public' functions: utility
    ##########################################################################

    def reset_acquisition_params(self):
        raise NotImplementedError

    def reset_crosstalk_matrix(self):
        raise NotImplementedError

    def reset_correlation_params(self):
        raise NotImplementedError

    def reset_rotation_params(self):
        raise NotImplementedError

    def upload_crosstalk_matrix(self, matrix) -> None:
        raise NotImplementedError

    def download_crosstalk_matrix(self, nr_rows=10, nr_cols=10):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError


    ##########################################################################
    # 'public' functions: print overview helpers
    ##########################################################################

    def print_correlation_overview(self) -> None:
        raise NotImplementedError

    def print_deskew_overview(self) -> None:
        raise NotImplementedError

    def print_crosstalk_overview(self) -> None:
        raise NotImplementedError

    def print_integration_overview(self) -> None:
        raise NotImplementedError

    def print_rotations_overview(self) -> None:
        raise NotImplementedError

    def print_thresholds_overview(self) -> None:
        raise NotImplementedError

    def print_user_regs_overview(self) -> None:
        raise NotImplementedError

    def print_overview(self) -> None:
        raise NotImplementedError

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
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def acquisition_initialize(self, 
                               samples, 
                               averages,
                               loop_cnt = None,
                               channels=(0, 1),
                               mode='rl', 
                               poll=True) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError
    
    def acquisition_arm(self, single=True) -> None:
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def acquisition_poll(self, samples, arm=True,
                         acquisition_time=0.010):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def acquisition_get(self, samples, arm=True,
                         acquisition_time=0.010):
        # TODO: adapt arguments default-values to SHFQA
        raise NotImplementedError

    def acquisition_finalize(self) -> None:
        raise NotImplementedError

    