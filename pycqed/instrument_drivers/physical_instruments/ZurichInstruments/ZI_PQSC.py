"""
Driver for PQSC V1
Author: Michael Kerschbaum
Date: 2019/09
"""

import time
import sys
import os
import logging
import numpy as np
import pycqed

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase

log = logging.getLogger(__name__)

##########################################################################
# Exceptions
##########################################################################

##########################################################################
# Module level functions
##########################################################################

##########################################################################
# Class
##########################################################################


class ZI_PQSC(zibase.ZI_base_instrument):
    """
    This is the frist version of the PycQED driver for the Zurich Instruments 
    PQSC.
    """

    # Put in correct minimum required revisions
    #FIXME: put correct version
    MIN_FWREVISION = 63210
    MIN_FPGAREVISION = 63133

    ##########################################################################
    # 'public' functions: device control
    ##########################################################################

    def __init__(self,
                 name,
                 device: str,
                 interface: str = 'USB',
                 port: int = 8004,
                 server: str = '',
                 **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument
            device          (str) the name of the device e.g., "dev8008"
            interface       (str) the name of the interface to use 
                                  ('1GbE' or 'USB')
            port            (int) the port to connect to for the ziDataServer 
                                  (don't change)
            server:         (str) the host where the ziDataServer is running
        """
        t0 = time.time()

        # Our base class includes all the functionality needed to initialize
        # the parameters of the object. Those parameters are read from
        # instrument-specific JSON files stored in the zi_parameter_files
        # folder.
        super().__init__(
            name=name,
            device=device,
            interface=interface,
            server=server,
            port=port,
            **kw)

        t1 = time.time()
        print('Initialized PQSC', self.devname, 'in %.2fs' % (t1 - t0))

    ##########################################################################
    # Private methods
    ##########################################################################

    def _check_devtype(self):
        if self.devtype != 'PQSC':
            raise zibase.ziDeviceError('Device {} of type {} is not a PQSC \
                instrument!'.format(self.devname, self.devtype))

    def _check_options(self):
        """
        Checks that the correct options are installed on the instrument.
        """
        # FIXME
        # options = self.gets('features/options').split('\n')
        # if 'QA' not in options:
        #     raise zibase.ziOptionsError('Device {} is missing the QA option!'.format(self.devname))
        # if 'AWG' not in options:
        #     raise zibase.ziOptionsError('Device {} is missing the AWG option!'.format(self.devname))

    def _check_versions(self):
        """
        Checks that sufficient versions of the firmware are available.
        """
        if self.geti('system/fwrevision') < ZI_PQSC.MIN_FWREVISION:
            raise zibase.ziVersionError(
                'Insufficient firmware revision detected! Need {}, got {}!'.
                format(ZI_PQSC.MIN_FWREVISION, self.geti('system/fwrevision')))

        if self.geti('system/fpgarevision') < ZI_PQSC.MIN_FPGAREVISION:
            raise zibase.ziVersionError(
                'Insufficient FPGA revision detected! Need {}, got {}!'.format(
                    ZI_PQSC.MIN_FPGAREVISION,
                    self.geti('system/fpgarevision')))

    def _add_extra_parameters(self) -> None:
        """
        We add a few additional custom parameters on top of the ones defined in the device files. These are:
          qas_0_trans_offset_weightfunction - an offset correction parameter for all weight functions,
            this allows normalized calibration when performing cross-talk suppressed readout. The parameter
            is not actually used in this driver, but in some of the support classes that make use of the driver.
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
        """
        super()._add_extra_parameters()

    # FIXME: put in correct clock_freq
    def clock_freq(self):
        return 1.8e9

    ##########################################################################
    # 'public' functions:
    ##########################################################################

    def check_errors(self) -> None:
        """
        Checks the instrument for errors.
        """
        # If this is the first time we are called, log the detected errors,
        # but don't raise any exceptions
        if self._errors is None:
            raise_exceptions = False
            self._errors = {}
        else:
            raise_exceptions = True

        # Stores the errors before processing
        errors = {'messages': []}

        # Asserted in case errors were found
        found_errors = False

        # Go through the errors and update our structure, raise exceptions if
        # anything changed
        for m in errors['messages']:
            code = m['code']
            count = m['count']
            severity = m['severity']
            message = m['message']

            if not raise_exceptions:
                self._errors[code] = {
                    'count': count,
                    'severity': severity,
                    'message': message
                }
                log.warning('{}: Code {}: "{}" ({})'.format(
                    self.devname, code, message, severity))
            else:
                # Optionally skip the error completely
                if code in self._errors_to_ignore:
                    continue

                # Check if there are new errors
                if code not in self._errors or count > self._errors[code][
                        'count']:
                    log.error('{}: {} ({}/{})'.format(self.devname, message,
                                                      code, severity))
                    found_errors = True

                if code in self._errors:
                    self._errors[code]['count'] = count
                else:
                    self._errors[code] = {
                        'count': count,
                        'severity': severity,
                        'message': message
                    }

        if found_errors:
            raise zibase.ziRuntimeError('Errors detected during run-time!')

    def set_repetitions(self, num_reps: int):
        '''Sets the number of triggers to be generated.'''

        self.set('execution_repetitions', num_reps)

    def set_holdoff(self, holdoff: float):
        '''Sets the interval between triggers in seconds. Set to 1e-3 for 
        generating triggers at 1kHz, etc.'''

        self.set('execution_holdoff', holdoff)

    def get_progress(self):
        '''Returns a value between 0.0 and 1.0 indicating the progress as 
        triggers are generated.'''

        return self.get('execution_progress')

    def track_progress(self):
        '''Prints a progress bar.'''

        # TODO
