import logging
import sys


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_bin(x, n, Unsigned=True):
    '''
    Return the 2's complement of the integer number $x$
    for a given bitwidth $n$.
    '''
    if (not is_number(x)):
        raise ValueError('get_bin: parameter {} is not a number.'.format(x))

    if Unsigned is True:
        return '{0:{fill}{width}b}'.format(int(x), fill='0', width=n)
    else:
        return '{0:{fill}{width}b}'.format((int(x) + 2**n) % 2**n,
                                           fill='0', width=n)


def bin_to_hex(bin_val, hex_width):
    if (not is_number(bin_val)):
        raise ValueError('bin_to_hex: parameter {} is not a number'.format(
            bin_val))

    return format(int(bin_val, 2), 'X').zfill(hex_width)


def check_int(d, left_bound, right_bound):
    if not isinstance(d, int):
        raise ValueError("The input parameter {} is not an integer.".format(d))

    if d < left_bound or d > right_bound:
        raise ValueError("Parameter {} out of range: [{}, {}].".format(
            d, left_bound, right_bound))

    return True


def ToFourBytes(int_data):
    if not isinstance(int_data, int):
        raise ValueError(
            "The input data {} is not an integer".format(int_data))

    return int_data.to_bytes(4, byteorder='little', signed=False)


class CCLightMicrocode():
    """
    This is class is used to implement all functionalities required to generate
    the microcode stored in the control store of CCLight.
    """

    def __init__(self):
        self.microcode = [0]*256
        self.CS_header = "OpTypeLeft  CW_Left  OpTypeRight  CW_Right  Condition"
        self.CS_format = "{:5d}       {:3d}      {:6d}     {:6d}     {:5d} "
        self.type_dict = ["None", "Mw", "Flux", "Meas"]
        print("initialization finished.")

    def gen_control_store_line(self, condition, op_type_left, cw_left,
                               op_type_right, cw_right, present_format=False):
        """
        This function generates a single entry for the control store.
        The entry structure is:
            [cond. exe. bits | left u_instruction | right u_instruction]

        - The Conditional execution bits are 2-bit wide.
          - 0 means this is an unconditional operation.
          - 1~3 means both the left and right operation will be conditioned on
                the three different combinatorial results.

        - The microinstruction has the following structure:
            [Type | Codeword]
          - Type: a 2-bit value indicating which kind of control to trigger:
            - 0 this microinstruction is not a valid one. Ignored by QuMA.
            - 1 --> AWG-8 for microwave control.
            - 2 --> AWG-8 for flux control
            - 3 --> UHFQC for Measurement
          - Codeword: An 8-bit value.
            - For microwave control, all 8 bits can be used.
            - For flux control, only the low three bits are used.
            - For measurement, this field is ignored by QuMA.
        """
        check_int(condition, 0, 3)
        check_int(op_type_left, 0, 3)
        check_int(cw_left, 0, 255)
        check_int(op_type_right, 0, 3)
        check_int(cw_right, 0, 255)

        final_val = (condition << 20) + (op_type_left << 18) +\
            (cw_left << 10) + (op_type_right << 8) + cw_right

        # return final_val
        bin_str = get_bin(final_val, 32)
        # hex_str = bin_to_hex(bin_str, 8)  # commented as not used

        if present_format is True:
            bin_str = get_bin(condition, 2) + "|"
            bin_str += get_bin(op_type_left, 2) + " "
            bin_str += get_bin(cw_left, 8) + "|"
            bin_str += get_bin(op_type_right, 2) + " "
            bin_str += get_bin(cw_right, 8)
            bin_str = "0"*10 + bin_str

        return final_val

    def disa_cs_line(self, int_data):
        if not isinstance(int_data, int):
            raise ValueError(
                "The input data {} is not an integer".format(int_data))

        condition = (int_data >> 20) & 3
        op_type_left = (int_data >> 18) & 3
        cw_left = (int_data >> 10) & 0xFF
        op_type_right = (int_data >> 8) & 3
        cw_right = int_data & 0xFF
        return condition, op_type_left, cw_left, op_type_right, cw_right

    def disa_bin_microcode(self, microcode):
        cs_line_array = []
        if (len(microcode) % 4) != 0:
            raise ValueError("The input binary microcode should be a"
                             " multiple of 4.")

        for i in range(int(len(microcode) / 4)):
            cs_line_array.append(int.from_bytes(
                microcode[i*4: i*4 + 4], byteorder='little', signed=False))

        return cs_line_array

    def print_pure_cs_line(self, cs_line):
        (condition, op_type_left, cw_left, op_type_right, cw_right) = \
            self.disa_cs_line(cs_line)
        print(self.CS_format.format(condition, op_type_left, cw_left,
                                    op_type_right, cw_right))

    def print_compact_cs_line(self, cs_line):
        (condition, op_type_left, cw_left, op_type_right, cw_right) = \
            self.disa_cs_line(cs_line)

        print("{:3d} {:6s} ".format(cw_left, "(" +
                self.type_dict[op_type_left] + ")"),
                end = '')
        if op_type_right != 0:
            print("{:3d} {:6s}  ".format(cw_right, "(" +
                self.type_dict[op_type_right] + ")"),
                end = '')
        if condition != 0:
            print("condition: {}".format(condition), end='')

    def print_cs_line(self, line_number):
        print(self.CS_header)
        self.print_pure_cs_line(self.microcode[line_number])

    def print_cs_line_no_header(self, line_number):
        self.print_compact_cs_line(self.microcode[line_number])

    def dump_microcode(self, filename=None):
        if filename is not None:
            try:
                mc_config = open(filename, 'w', encoding='utf-8')
                logging.info("opened file {} successfully.".format(filename))
            except:
                raise OSError('\tError: Failed to open file ' +
                              self.filename + ".")
            # direct std output into the write file
            saveout = sys.stdout
            sys.stdout = mc_config

        print("     ", self.CS_header)
        for idx, cs_line in enumerate(self.microcode):
            print('  {:<3d}  '.format(idx), end='')
            self.print_pure_cs_line(cs_line)

        # restore the std output
        if filename is not None:
            mc_config.close()
            sys.stdout = saveout

    def load_microcode_array(self, microcode):
        if len(microcode) > 256:
            raise ValueError("The microcode can be at most 256 long. {} "
                             "are given".format(len(microcode)))

        for idx, cs_line in enumerate(microcode):
            if cs_line > (1 << 22) - 1:
                raise ValueError("The maximum value of a cs_line is: 2**22 -1."
                                 "{} is given at position {}.".format(
                                    cs_line, idx))

    def load_microcode(self, filename):
        try:
            mc_config = open(filename, 'r', encoding='utf-8')
            logging.info("opened file {} successfully.".format(filename))
        except:
            raise OSError('\tError: Failed to open file ' +
                          self.filename + ".")

        # throw away the header line
        mc_config.readline()

        for line in mc_config:
            # throw away empty lines
            if not line.rstrip():
                continue

            # remove the ':' character
            line = line.translate({ord(":"): None})

            # get each number in the line
            line_number, condition, op_type_left, cw_left,\
                op_type_right, cw_right = line.split()

            line_number = int(line_number)
            condition = int(condition)
            op_type_left = int(op_type_left)
            cw_left = int(cw_left)
            op_type_right = int(op_type_right)
            cw_right = int(cw_right)

            if line_number > 256:
                raise ValueError("line number ({}) in the file "
                                 "exceeds the maximum value (256).")

            self.microcode[line_number] = self.gen_control_store_line(
                condition, op_type_left, cw_left, op_type_right, cw_right)

    def write_to_bin(self, filename=None):
        bin_data = bytearray()
        for cs_line in self.microcode:
            bin_data.extend(ToFourBytes(cs_line))

        if filename is not None:
            try:
                bin_file = open(filename, 'wb')
                logging.info("open file", str(filename), "successfully.")
            except:
                raise OSError('\tError: Failed to open file ' +
                              self.filename + ".")

            bin_file.write(bin_data)
            bin_file.close()
        else:
            return bin_data

    def read_bin(self, filename):
        bin_file = open(filename, 'rb')
        bin_data = bin_file.read()
        bin_file.close()

        cs_line_array = self.disa_bin_microcode(bin_data)

        print("Condition  OpTypeLeft  CW_Left  OptypeRight  CW_Right")
        for i in range(len(cs_line_array)):
            self.print_pure_cs_line(cs_line_array[i])


    def insert_cs_line(self, line_number, condition, op_type_left,
                       cw_left, op_type_right=0, cw_right=0):
        if line_number > 256:
            raise ValueError("line number ({}) in the file "
                             "exceeds the maximum value (256).")

        cs_line = self.gen_control_store_line(condition, op_type_left, cw_left,
                                              op_type_right, cw_right)

        self.microcode[line_number] = cs_line


def gen_test_ctrl_store_lines(mc):
    for i in range(1, 128):
        mc.insert_cs_line(i, 0, 1, i)

    for i in range(128, 256):
        mc.insert_cs_line(i, 0, 2, (i-128) % 7 + 1, 2, (i-124) % 7 + 1)
