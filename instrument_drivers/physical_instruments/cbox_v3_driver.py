# Note to Xiang, remove the imports that are not used :)
import time
import numpy as np
import sys
# import visa
# import unittest
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

from ._controlbox import defHeaders  # File containing bytestring commands
from ._controlbox import codec as c
from ._controlbox import Assembler
from . import QuTech_ControlBoxdriver as qcb

'''
@author: Xiang Fu
The driver for ControlBox version 3. This is inherited from the driver of
ControlBox version 2.
'''

class QuTech_ControlBox_v3(qcb.QuTech_ControlBox):
    def __init__(self, *args, **kwargs):
        super(QuTech_ControlBox_v3, self).__init__(*args, **kwargs)

    def _set_touch_n_go_parameters(self):

        '''
        Touch 'n Go is only valid for ControlBox version 2, so this function is
        obselete.
        '''
        print("Warning: Touch 'n Go is only valid for ControlBox version 2. omit")

    def load_instructions(self, asm_file):
        '''
        set the weights of the integregration

        @param instructions : the instructions, an array of 32-bit instructions
        @return stat : 0 if the upload succeeded and 1 if the upload failed.
        '''

        asm = Assembler.Assembler(asm_file)

        instructions = asm.convert_to_instructions()

        if instructions == False :
            print("Error: the assembly file is of errors.")
            return False

        for instruction in instructions:
            print(format(instruction, 'x').zfill(8))

        # Check the instruction list length
        if len(instructions) == 0 :
            raise ValueError("The instruction list is empty.")

        cmd = defHeaders.LoadInstructionsHeader
        data_bytes = bytearray()

        instr_length = 32
        data_bytes.extend(c.encode_array(
                        self.convert_arrary_to_signed(instructions, instr_length),
                        data_bits_per_byte = 4,
                        bytes_per_value = 8))

        message = c.create_message(cmd, bytes(data_bytes))
        (stat, mesg) = self.serial_write(message)

        if not stat:
            raise Exception('Failed to load instructions')

        return (stat, mesg)


    def set_conditional_tape(self, awg_nr, tape_nr, tape):
        '''
        NOTE: The ControlBox does not support timing tape till now.

        set the conditional tape content for an awg

        @param awg : the awg of the dac, (0,1,2).
        @param tape_nr : the number of the tape, integer ranging (0~6)
        @param tape : the array of entries, with a maximum number of entries 512.
            Every entry is an integer has the following structure:
                |WaitingTime (9bits) | PUlse number (3 bits) | EndofSegment marker (1bit)|
            WaitingTime: The waiting time before the end of last pulse or trigger, in ns.
            Pulse number: 0~7, indicating which pulse to be output
            EndofSegment marker: 1 if the entry is the last entry of the tape, otherwise 0.
        @return stat : 0 if the upload succeeded and 1 if the upload failed.

        '''

        print("Timing tape is not supported yet.")
        return False

        length = len(tape)
        tape_addr_width = 9
        entry_length = 9 + 3 + 1

        # Check out of bounds
        if awg_nr < 0 or awg_nr > 2:
            raise ValueError
        if tape_nr < 0 or tape_nr > 6:
            raise ValueError
        if length < 1 or length > 512:
            raise ValueError("The conditional tape only supports a length from 1 to 512.")

        cmd = defHeaders.AwgCondionalTape
        data_bytes = bytearray()

        data_bytes.extend(c.encode_byte(awg_nr, 4,      # add AWG number
                                        expected_number_of_bytes = 1))
        data_bytes.extend(c.encode_byte(tape_nr, 4,     # add the tape number
                                        expected_number_of_bytes = 1))
        data_bytes.extend(c.encode_byte(length-1, 7,    # add the tape length
                          expected_number_of_bytes=np.ceil(tape_addr_width/7.0)))
        data_bytes.extend(c.encode_array(               # add the tape entries
                          convert_arrary_to_signed(tape, entry_length),
                          data_bits_per_byte = 7,
                          expected_number_of_bytes = np.ceil(entry_length/7.0)))

        message = c.create_message(cmd, data_bytes)
        (stat, mesg) = self.serial_write(message)
        return (stat, mesg)

    def set_segmented_tape(self, awg_nr, tape):
        '''
        NOTE: The ControlBox does not support timing tape till now.

        set the conditional tape content for an awg

        @param awg : the awg of the dac, (0,1,2).
        @param tape : the array of entries, with a maximum number of entries 29184.
            Every entry is an integer has the following structure:
                |WaitingTime (9bits) | PUlse number (3 bits) | EndofSegment marker (1bit)|
            WaitingTime: The waiting time before the end of last pulse or trigger, in ns.
            Pulse number: 0~7, indicating which pulse to be output
            EndofSegment marker: 1 if the entry is the last entry of a segment, otherwise 0.
        @return stat : 0 if the upload succeeded and 1 if the upload failed.

        '''

        print("Timing tape is not supported yet.")
        return False


        length = len(tape)
        tape_addr_width = 15
        entry_length = 9 + 3 + 1

        # Check out of bounds
        if awg_nr < 0 or awg_nr > 2:
            raise ValueError("Awg number error!")
        if length < 1 or length > 29184:
            raise ValueError("The segemented tape only supports a length from 1 to 29184.")

        cmd = defHeaders.AwgSegmentedTape
        data_bytes = bytearray()
        data_bytes.extend(c.encode_byte(awg_nr, 4,
                                        expected_number_of_bytes = 1))
        data_bytes.extend(c.encode_byte(length-1, 7,
                          expected_number_of_bytes=np.ceil(tape_addr_width / 7.0)))
        data_bytes.extend(c.encode_array(
                          convert_arrary_to_signed(tape, entry_length),
                          data_bits_per_byte = 7,
                          expected_number_of_bytes = np.ceil(entry_length/7.0)))

        message = self.create_message(cmd, data_bytes)
        (stat, mesg) = self.serial_write(message)
        return (stat, mesg)

    def create_entry(self, interval, pulse_num, end_of_marker):
        '''
        @param interval : The waiting time before the end of last pulse or trigger in ns,
                          ranging from 0ns to 2560ns with minimum step of 5ns.
        @param pulse_num : 0~7, indicating which pulse to be output
        @param end_of_marker : 1 if the entry is the last entry of a segment, otherwise 0.
        '''

        if interval < 0 or interval > 2560:
            raise ValueError
        if pulse_num < 0 or pulse_num > 7:
            raise ValueError
        if end_of_marker < 0 or end_of_marker > 1:
            raise ValueError

        entry_bits = BitArray(Bits(uint=interval, length=9))
        entry_bits.append(BitArray(Bits(uint=pulse_num, length=3)))
        entry_bits.append(BitArray(Bits(uint=end_of_marker, length=1)))
        # print "The entry generated is: ",
        # print entry_bits.uint

        return entry_bits.uint

    def convert_arrary_to_signed(self, unsigned_array, bit_width):
        '''
        Inteprete the input unsinged number array into a signed number array
        based on the given bitwidth.

        @param unsigned_array: the unsigned number array.
        @param bit_width: Bit width of the output signed number.
        '''

        signed_array = []
        for sample in unsigned_array:
            signed_array.append(self.convert_to_signed(sample, bit_width))

        return signed_array

    def convert_to_signed(self, unsigned_number, bit_width):
        '''
        Inteprete the input unsinged number into a signed number given the bitwidth.

        @param unsigned_number: the unsigned number.
        @param bit_width: Bit width of the output signed number.
        '''

        if (not isinstance(unsigned_number, (int))) or (unsigned_number < 0):
            raise ValueError("The number %d should be a positive integer." \
                              % unsigned_number)

        if unsigned_number < 0 or unsigned_number >= 2**bit_width:
            raise ValueError("Given number %d is too large in terms of the \
                              given bit_width %d." % unsigned_number, bit_width)

        if unsigned_number >= 2**(bit_width-1):
            signed_number = unsigned_number - 2**bit_width;
        else:
            signed_number = unsigned_number;

        return signed_number
