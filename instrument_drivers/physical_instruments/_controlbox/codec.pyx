from bitarray import bitarray
import numpy as np
cimport numpy as np  # cython internally fixes ambiguity
'''
@author: MAR
Cython module that contains encoder and decoder for the CBox v2
'''

def create_message(cmd=None, bytes data_bytes=bytes(),
                   bytes EOM=b"\x7F"):
    '''
    Input arguments:
                  cmd         = None
        bytes     data_bytes  = bytes()
        bytes     EOM         = b'\x7F'

    Creates bytes to send as a message.
    Starts with a command, then adds the data bytes and ends with EOM.
    '''
    cdef bytes message = bytes()
    if cmd != None:
        message+= cmd
    message +=data_bytes
    message += EOM # Because single char
    return message

cpdef encode_byte(int value, int data_bits_per_byte=7,
                  int expected_number_of_bytes=2):
    '''
    input arguments
    int value                    : value to be encoded
    int data_bits_per_byte       : specify bits/byte used in encoding
    int expected_number_of_bytes : number of bytes expected by CBox

    returns
    bytearray data_byte_array    : the encoded value


    From "250 MSPs Control Box Design Specification" version May 2015
    by Jacob de Sterke

    full_byte encoding:
    In this mode each protocol byte contains 7 bits or the data to be
    transferred. This results in a bit efficiency of about 85%. This mode
    is mostly used when large words (e.g. 28-bit words) need to be
    transferred. This improves the efficiency since transferring a 28-bit
    word only needs four protocol bytes instead of seven when using the
    nibble per byte mode.

    |7|6|5|4|3|2|1|0|
     | | > > > > > > > data bits
     |
     always 1


    nibble_byte encoding:
    In this mode each protocol byte contains only four bits (a nibble) of
    the data to be transferred. This results in a bit efficiency of 50%;
    to transfer one data byte two protocol bytes are needed.
    |7|6|5|4|3|2|1|0|
    | | | | | > > >  data bits
    | | > > > unused bits (should be set to zero for consistency)
    |
    always 1
    '''
    cdef int i
    cdef bytearray data_byte_array = bytearray(expected_number_of_bytes)
    cdef int MASK =((1<<(data_bits_per_byte)) -1); # Used to select the bits
    cdef unsigned char DataMASK = 1 << 7         # Sets initial bit to 1
    cdef unsigned char byte_val

    for i in range(expected_number_of_bytes):
        byte_val = (<char> (value>>(data_bits_per_byte*i) & MASK))
        byte_val =byte_val| DataMASK;
        data_byte_array[expected_number_of_bytes-(i+1)] = byte_val
    return bytes(data_byte_array)  # note, I might want to get rid of this

def encode_array(values, int data_bits_per_byte=7,
                 int bytes_per_value = 2):
    '''
    Input arguments
        int*   values                      : array of values to be encoded
        int    data_bits_per_byte = 7      : specify bits/byte used in encoding
        int    bytes_per_value    = 2      : number of bytes expected per value

    Takes an array of values and encodes every single one using
    the encode_byte function.
    '''
    cdef int nr_values = len(values)
    cdef bytearray data_bytes = bytearray()
    cdef int i
    for i in range(nr_values):
        data_bytes.extend(encode_byte(values[i], data_bits_per_byte=data_bits_per_byte,
                                    expected_number_of_bytes=bytes_per_value))
    return bytes(data_bytes)


def decode_message(data_bytes, int data_bits_per_byte=7,
                   int bytes_per_value=2):
    '''
    Input arguments:
        bytearray data_bytes         :
        int       data_bits_per_byte : 7
        int       bytes_per_value    : 2

    returns numpy array of type int
    Cythonized version of the CBox decoder
    '''
    message_bytes = data_bytes[:-1]
    cdef int message_length = len(message_bytes)/bytes_per_value
    # Preallocating the numpy array by cimport np gives another 30% speedup
    # But could not make it work easily with the compiler
    # cdef np.ndarray[int,ndim=1] values = np.empty(message_length, dtype=int)
    values = np.empty(message_length, dtype=int)
    cdef int i
    for i in range(message_length):
        iteration_bytes = message_bytes[i*bytes_per_value:
                                        (i+1)*bytes_per_value]
        values[i] = decode_byte(iteration_bytes,
                          data_bits_per_byte=data_bits_per_byte)
    return values

cpdef decode_byte(data_bytes, int data_bits_per_byte=7):
    '''
    Input arguments:
        bytearray data_bytes
        int       data_bits_per_byte : 7
    returns
        int       value

    Inverse function of encode byte. Protocol is described in docstring
    of encode_byte().
    '''
    cdef int MASK =((1<<(data_bits_per_byte)) -1);
    cdef int i=0;
    cdef int value =0 ;
    cdef int len_db = len(data_bytes)
    cdef int nr_bits_m1
    nr_bits_m1 = (data_bits_per_byte*len_db-1)
    # Combine relevant bits of different bytes
    for i in range(len_db):
        # Select desired bits
        bits = MASK & data_bytes[len_db-1-i];
        # Prepend next bits
        value |= (bits<<(data_bits_per_byte*i));
    # Convert to 32bit signed int
    if value &(1<<nr_bits_m1): # verify if negative
        value&= ~(1<<nr_bits_m1) # set msb of small int to 0
        value = value - 2**nr_bits_m1 # flip sign of 32bit int
    return value

def decode_boolean_array(data_bytes, int data_bits_per_byte=4):
    '''
    Used in the qubit state logging mode
    '''
    # Convert the raw bits to an array of booleans
    bitarr = bitarray()
    bitarr.frombytes(data_bytes[:-2]) # -2 to remove checksum and eom
    raw_vals = np.array(bitarr)
    # Slicing the array to only take the last 4 data bits per byte
    values = np.empty((len(data_bytes)-2)*4)
    for i in range((len(data_bytes)-2)):\
        # loop over all bytes (in steps of 8 bits)
        for j in range(data_bits_per_byte):
            # loop over all data bits in a byte
            values[data_bits_per_byte*i+j] = raw_vals[8*i+j+data_bits_per_byte]
    ch0_values = values[:len(values)//2]
    ch1_values = values[len(values)//2:]
    return ch0_values, ch1_values

def calculate_checksum(bytes input_command):
        '''
        Input arguments
            bytes input_command

        Calculates checksum by taking the XOR of all bytes in input_command
        '''
        checksum = 0
        for byte in input_command:
            checksum ^= byte
        checksum = checksum | 128
        checksum = bytes([checksum]) # Convert int to hexs and set MSbit

        return checksum
