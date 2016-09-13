import numpy as np
# cimport numpy as np
'''
@author: MAR
'''
def decode_message(data_bytes, int data_bits_per_byte=7,
                   int bytes_per_value=2):
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
    Inverse function of encode byte. Protocol is described in docstring
    of encode_byte().

    Note: does not support unsigned integers.
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
