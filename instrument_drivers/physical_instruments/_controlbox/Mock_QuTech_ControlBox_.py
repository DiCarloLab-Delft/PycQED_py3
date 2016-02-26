# Note to Xiang, remove the imports that are not used :)
import time
import numpy as np
import sys
# import visa
import unittest
import logging

qcpath = 'D:\GitHubRepos\Qcodes'
if qcpath not in sys.path:
    sys.path.append(qcpath)

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals

# cython drivers for encoding and decoding
import pyximport
pyximport.install(setup_args={"script_args": ["--compiler=msvc"],
                              "include_dirs": np.get_include()},
                  reload_support=True)

from ._controlbox import codec as c
from . import QuTech_ControlBoxdriver as qcb


class Mock_QuTech_ControlBox(qcb.QuTech_ControlBox):
    '''
    TODO: This driver should disable the connection to VISA. Refer to Adriaan.
    '''
    def __init__(self, *args, **kwargs):
        super(Mock_QuTech_ControlBox, self).__init__(*args, **kwargs)

    def serial_write(self, command, verify_execution=True):
        '''
        Core write function used for communicating with the Box.

        Accepts either a bytes as input
        e.g. command = b'\x5A\x00\x7F'

        Writes data to the serial port and verifies the checksum to see if the
        message was received correctly.

        Returns: succes, message
        Succes is true if the checksum matches.
        '''
        if type(command) != bytes:
            raise TypeError('command must be type bytes')
        checksum = c.calculate_checksum(command)

        in_wait = self.visa_handle.bytes_in_buffer
        if in_wait > 0:  # Clear any leftover messages in the buffer
            self.visa_handle.clear()
            print("Extra flush! Flushed %s bytes" % in_wait)
        self.visa_handle.write_raw(command)
        # Done writing , verify message executed
        if verify_execution:
            message = self.serial_read()
            if bytes([message[0]]) == checksum:
                succes = True
            else:
                print(message[0], checksum)
                raise Exception('Checksum Error, Command not executed')
            return succes, message
        else:
            return True
