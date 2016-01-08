~~~ python
def set_conditional_tape(self, awg_nr, tape_nr, tape):
    '''
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

    length = len(tape)
    tape_addr_width = 9
    entry_length = 9 + 3 + 1

    # Check out of bounds
    if awg_nr < 0 or awg_nr > 2:
        raise ValueError
    if tape_nr < 0 or tape_nr > 6:
        raise ValueError
    if length < 1 or length > 512:
        raise ValueError

    cmd = defHeaders.AwgCondionalTape
    data_bytes = []
    data_bytes.append(self.encode_byte(awg_nr, 4))
    data_bytes.append(self.encode_byte(tape_nr, 4))
    data_bytes.append(self.encode_byte(length-1, 7,
                      signed_integer_length=tape_addr_width,
                      expected_number_of_bytes=np.ceil(tape_addr_width/7.0)))
    for sample_data in tape:
        data_bytes.append(self.encode_byte(self.convert_to_signed(sample_data, entry_length), 7,
                      signed_integer_length=entry_length,
                      expected_number_of_bytes=np.ceil(entry_length/7.0)))

    message = self.create_message(cmd, data_bytes)
    (stat, mesg) = self.serial_write(message)
    return (stat, mesg)

def set_segmented_tape(self, awg_nr, tape):
    '''
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

    length = len(tape)
    tape_addr_width = 15
    entry_length = 9 + 3 + 1

    # Check out of bounds
    if awg_nr < 0 or awg_nr > 2:
        raise ValueError
    if length < 1 or length > 29184:
        raise ValueError

    cmd = defHeaders.AwgSegmentedTape
    data_bytes = []
    data_bytes.append(self.encode_byte(awg_nr, 4))
    data_bytes.append(self.encode_byte(length-1, 7,
                      signed_integer_length=tape_addr_width,
                      expected_number_of_bytes=np.ceil(tape_addr_width / 7.0)))
    for sample_data in tape:
        data_bytes.append(self.encode_byte(self.convert_to_signed(sample_data, entry_length), 7,
                      signed_integer_length=entry_length,
                      expected_number_of_bytes=np.ceil(entry_length / 7.0)))

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
