import string
from sys import exit


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_bin(x, n):
    if (not is_number(x)):
        raise ValueError('get_bin: parameter is not a number.')

    return '{0:{fill}{width}b}'.format((int(x) + 2**n) % 2**n,
                                       fill='0', width=n)


def bin_to_hex(bin_val, hex_width):
    if (not is_number(bin_val)):
        raise ValueError('bin_to_hex: parameter' + str(bin_val) +
                         'is not a number.')

    return format(int(bin_val, 2), 'x').zfill(hex_width)


def dec_to_bin_w4(x):
    if (not is_number(x)):
        raise ValueError('Parameter is not a number.')

    return '{0:04b}'.format(x)


def dec_to_bin_w8(x):
    if (not is_number(x)):
        raise ValueError('Parameter is not a number.')

    return '{0:08b}'.format(x)


class Assembler():

    def __init__(self, asm_filename):
        self.asmfilename = asm_filename

    InstOpCode = {'add':     '000000',
                  'sub':     '000000',
                  'beq':     '000100',
                  'bne':     '000101',
                  'addi':    '001000',
                  'ori':     '001101',
                  'lui':     '001111',
                  'waitreg': '000000',
                  'pulse':   '000000',
                  'measure': '000000',
                  'wait':    '110000',
                  'trigger': '101000',
                  'lw':      '100011',
                  'sw':      '101011'}

    InstfunctCode = {'add':     '100000',
                     'sub':     '100010',
                     'waitreg': '010000',
                     'pulse':   '000001',
                     'measure': '000010'}

    def get_reg_num(self, Register):
        '''
        It gets the register number from the input string.

        @param Register : the input string represents a register
        @return stat : 4-bit binary string representing the register number
        '''
        if (not is_number(Register.strip('r'))):
            raise ValueError("Register format is not correct.")

        reg_num = int(Register.strip('r'))

        if (reg_num < 0 or reg_num > 15):
            raise ValueError("Register number is out of range.")

        return dec_to_bin_w4(reg_num)

    def get_lui_pos(self, pos):

        if (not is_number(pos)):
            raise ValueError("Position input is not a number.")

        val_pos = int(pos)

        if (val_pos < 0 or val_pos > 3):
            raise ValueError("Position input is out of range ({0,1,2,3}\
                              expected).")

        elif (val_pos == 0):
            position = '0001'
        elif (val_pos == 1):
            position = '0010'
        elif (val_pos == 2):
            position = '0100'
        else:
            position = '1000'

        return position

    # lui rt, pos, byte_data
    def LuiFormat(self, Register, pos, putByte):
        try:
            opCode = self.InstOpCode['lui']
            FDC = '100'
            rt = self.get_reg_num(Register)
            position = self.get_lui_pos(pos)
            imm8 = dec_to_bin_w8(int(putByte))
            return opCode + FDC + rt + rt + position + '000' + imm8

        except ValueError as detail:
            print('Lui instruction format error:', detail.args)

    # mov rt, imm32
    def MovFormat(self, Register, imm32):
        try:
            bit32 = get_bin(imm32, 32)
            putByte0 = int(bit32[24:32], 2)
            putByte1 = int(bit32[16:24], 2)
            putByte2 = int(bit32[8:16], 2)
            putByte3 = int(bit32[0:8], 2)
            Luis = []
            Luis.append(self.LuiFormat(Register, 0, putByte0))
            Luis.append(self.LuiFormat(Register, 1, putByte1))
            Luis.append(self.LuiFormat(Register, 2, putByte2))
            Luis.append(self.LuiFormat(Register, 3, putByte3))
            return Luis

        except ValueError as detail:
            print('Lui instruction format error:', detail.args)

    # add rd, rs, rt
    def AddFormat(self, dst_reg, src_reg1, src_reg2):
        try:
            opCode = self.InstOpCode['add']
            FDC = '100'
            rd = self.get_reg_num(dst_reg)
            rs = self.get_reg_num(src_reg1)
            rt = self.get_reg_num(src_reg2)
            shamt = '00000'
            funct = self.InstfunctCode['add']
            return opCode + FDC + rs + rt + rd + shamt + funct

        except ValueError as detail:
            print('Add instruction format error: ', detail.args)

    # sub rd, rs, rt
    def SubFormat(self, dst_reg, src_reg1, src_reg2):
        try:
            opCode = self.InstOpCode['sub']
            FDC = '100'
            rd = self.get_reg_num(dst_reg)
            rs = self.get_reg_num(src_reg1)
            rt = self.get_reg_num(src_reg2)
            shamt = '00000'
            funct = self.InstfunctCode['sub']
            return opCode + FDC + rs + rt + rd + shamt + funct

        except ValueError as detail:
            print('Sub instruction format error: ', detail.args)

    # beq rs, rt, off
    def BeqFormat(self, src_reg1, src_reg2, offset15):
        try:
            opCode = self.InstOpCode['beq']
            FDC = '100'
            rs = self.get_reg_num(src_reg1)
            rt = self.get_reg_num(src_reg2)
            imm15 = get_bin(offset15, 15)
            return opCode + FDC + rs + rt + imm15

        except ValueError as detail:
            print('Beq instruction format error: ', detail.args)

    # bne rs, rt, off
    def BneFormat(self, src_reg1, src_reg2, offset15):
        try:
            opCode = self.InstOpCode['bne']
            FDC = '100'
            rs = self.get_reg_num(src_reg1)
            rt = self.get_reg_num(src_reg2)
            immValue = get_bin(offset15, 15)
            return opCode + FDC + rs + rt + immValue

        except ValueError as detail:
            print('Bne instruction format error: ', detail.args)

    # addi rt, rs, imm
    def AddiFormat(self, dst_reg, src_reg1, imm15):
        try:
            opCode = self.InstOpCode['addi']
            FDC = '100'
            rs = self.get_reg_num(src_reg1)
            rt = self.get_reg_num(dst_reg)
            immValue = get_bin(imm15, 15)
            return opCode + FDC + rs + rt + immValue

        except ValueError as detail:
            print('Addi instruction format error: ', detail.args)

    # ori rt, rs, imm
    def OriFormat(self, dst_reg, src_reg1, imm15):
        try:
            opCode = self.InstOpCode['ori']
            FDC = '100'
            rs = self.get_reg_num(src_reg1)
            rt = self.get_reg_num(dst_reg)
            immValue = get_bin(imm15, 15)
            return opCode + FDC + rs + rt + immValue

        except ValueError as detail:
            print('Ori instruction format error: ', detail.args)

    # waitreg rs
    def WaitRegFormat(self, src_reg):
        try:
            opCode = self.InstOpCode['waitreg']
            FDC = '001'
            rs = self.get_reg_num(src_reg)
            zero13 = '0000000000000'
            funct = self.InstfunctCode['waitreg']
            return opCode + FDC + rs + zero13 + funct

        except ValueError as detail:
            print('WaitReg instruction format error: ', detail.args)

    # pulse AWG0, AWG1, AWG2
    def PulseFormat(self, awg0, awg1, awg2):
        try:
            opCode = self.InstOpCode['pulse']
            FDC = '001'
            shamt = '00000'
            funct = self.InstfunctCode['pulse']
            return opCode + FDC + awg0 + awg1 + awg2 + shamt + funct

        except ValueError as detail:
            print('Pulse instruction format error: ', detail.args)

    # measure rt
    # def MeasureFormat(self, dst_reg):
    #     try:
    #         opCode = self.InstOpCode['measure']
    #         FDC = '011'
    #         zero4 = '0000'
    #         rt = self.get_reg_num(dst_reg)
    #         zero9 = '000000000'
    #         funct = self.InstfunctCode['measure']
    #         return opCode + FDC + zero4 + rt + zero9 + funct
    #
    #     except ValueError as detail:
    #         print('Measure instruction format error: ', detail.args)

    # measure
    def MeasureFormat(self):
        try:
            opCode = self.InstOpCode['measure']
            FDC = '011'
            zero4 = '0000'
            rt = '0000'
            zero9 = '000000000'
            funct = self.InstfunctCode['measure']
            return opCode + FDC + zero4 + rt + zero9 + funct

        except ValueError as detail:
            print('Measure instruction format error: ', detail.args)

    # wait imm
    def WaitFormat(self, imm15):
        try:
            opCode = self.InstOpCode['wait']
            FDC = '001'
            zero8 = '00000000'
            immValue = get_bin(imm15, 15)
            return opCode + FDC + zero8 + immValue

        except ValueError as detail:
            print('Ori instruction format error: ', detail.args)

    # trigger mask, duration
    def TriggerFormat(self, mask, imm11):
        if len(mask) != 7:
            raise ValueError("The mask should be 7 bits. \
                              With the MSb indicating marker 1, \
                              and the LSb indicating marker 7.")
        for b in mask:
            if (b != '0' and b != '1'):
                raise ValueError("The mask should only contain 1 or 0.")

        # In the core of 3.1.0, the MSb works for the trigger 7.
        # Reverse the string so that the MSb works for trigger 1.
        mask = mask[::-1]

        mask = "00000" + mask   # The mask should be 12-bit wide.

        if int(imm11) < 0 or int(imm11) > 2047:
            raise ValueError("the value of the duration time is out of range \
                              (accepted: integer in 0~2047).")
        try:
            opCode = self.InstOpCode['trigger']
            FDC = '001'
            immValue = get_bin(imm11, 12)       # TODO: Check Range
            return opCode + FDC + mask + immValue[1:]

        except ValueError as detail:
            print('Ori instruction format error: ', detail.args)

    def NopFormat(self):
        return "00000000000000000000000000000000"

    def convert_to_instructions(self):
        print("The old version assembler.")
        try:
            Asm_File = open(self.asmfilename, 'r', encoding="utf-8")
            print("open file", self.asmfilename, "successfully.")
        except:
            print('\tError: Fail to open file ' + self.asmfilename + ".")
            exit(0)

        tag_addr_dict = {}
        cur_addr = 0
        instructions = []

        for line in Asm_File:
            line = line.split('#', 1)[0]  # remove anything after '#' symbole
            line = line.strip(' \t\n\r')  # remove whitespace

            if (len(line) == 0):  # skip empty line and comment
                continue

            cur_addr = len(instructions) + 1

            head, sep, tail = line.partition(':')
            if (sep == ":"):
                # print("***************** head: ", head)
                tag_addr_dict[head.strip().lower()] = cur_addr
                instr = tail
            else:
                instr = head

            # the following translate function should be tested.
            elements = [rawEle.strip(string.punctuation.translate(
                        {ord('-'): None})) for rawEle in instr.split()]

            if (elements[0].lower() == 'lui'):     # lui rt, pos, byte_data
                # print('parsing lui instruction.')
                instructions.append(int(self.LuiFormat(elements[1],
                                                       elements[2],
                                                       elements[3]), 2))

            elif (elements[0].lower() == 'mov'):      # mov rt, imm32
                # print('parsing mov instruction.')
                instr4 = self.MovFormat(elements[1], elements[2])
                for i in instr4:
                    instructions.append(int(i, 2))

            elif (elements[0].lower() == 'add'):   # add rd, rs, rt
                # print('parsing add instruction.')
                if (elements[1][0] != 'r' or elements[2][0] != 'r' or
                        elements[3][0] != 'r'):
                    print('Error: Add instruction only receive three registers\
                           as input.')
                    exit()

                instructions.append(int(self.AddFormat(elements[1],
                                                       elements[2],
                                                       elements[3]), 2))

            elif (elements[0].lower() == 'sub'):   # sub rd, rs, rt
                # print('parsing sub instruction.')
                if (elements[1][0] != 'r' or elements[2][0] != 'r' or
                        elements[3][0] != 'r'):
                    print('Error: Sub instruction only receive three \
                           registers as input.')
                    exit()

                instructions.append(int(self.SubFormat(elements[1],
                                                       elements[2],
                                                       elements[3]), 2))

            elif (elements[0].lower() == 'beq'):   # beq rs, rt, off
                # print('parsing beq instruction.')

                if (elements[1][0] != 'r' or elements[2][0] != 'r'):
                    print('Error: beq instruction only receive registers as \
                           the first two parameter.')
                    exit()

                if elements[3].strip().lower() in tag_addr_dict:
                    target_addr = tag_addr_dict[elements[3].strip().lower()] -\
                                  (cur_addr + 1)
                else:
                    print("Error: beq. Cannot find the target: ",
                          elements[3].strip().lower())
                    exit()

                instructions.append(int(self.BeqFormat(elements[1],
                                                       elements[2],
                                                       target_addr), 2))

            elif (elements[0].lower() == 'bne'):   # bne rs, rt, off
                # print('parsing bne instruction.')

                if (elements[1][0] != 'r' or elements[2][0] != 'r'):
                    print('Error: bne instruction only receive registers as \
                           the first two parameter.')
                    exit()

                if elements[3].strip().lower() in tag_addr_dict:
                    target_addr = tag_addr_dict[elements[3].strip().lower()] -\
                                  (cur_addr + 1)
                    # print("bne, target_addr: ", target_addr)
                else:
                    print("Error: bne. Cannot find the target: ",
                          elements[3].strip().lower())
                    exit()

                instructions.append(int(self.BneFormat(elements[1],
                                                       elements[2],
                                                       target_addr), 2))

            elif (elements[0].lower() == 'addi'):  # addi rt, rs, imm
                # print('parsing addi instruction.')

                if (elements[1][0] != 'r' or elements[2][0] != 'r'):
                    print('Error: addi instruction only receive registers as \
                           the first two parameter.')
                    exit()

                instructions.append(int(self.AddiFormat(elements[1],
                                                        elements[2],
                                                        elements[3]), 2))

            elif (elements[0].lower() == 'ori'):   # ori rt, rs, imm
                # print('parsing ori instruction.')

                if (elements[1][0] != 'r' or elements[2][0] != 'r'):
                    print('Error: Ori instruction only receive registers as \
                           the first two parameter.')
                    exit()

                instructions.append(int(self.OriFormat(elements[1],
                                                       elements[2],
                                                       elements[3]), 2))

            elif (elements[0].lower() == 'waitreg'):   # WaitReg rs
                # print('parsing WaitReg instruction.')

                if (elements[1][0] != 'r'):
                    print('Error: WaitReg instruction only a register as the \
                           parameter.')
                    exit()

                instructions.append(int(self.WaitRegFormat(elements[1]), 2))

            elif (elements[0].lower() == 'pulse'):   # Pulse awg0, awg1, awg2
                # print('parsing Pulse instruction.')
                instructions.append(int(self.PulseFormat(elements[1],
                                                         elements[2],
                                                         elements[3]), 2))

            elif (elements[0].lower() == 'measure'):   # Measure
                # print('parsing Measure instruction.')

                # if (elements[1][0] != 'r'):
                #     print('Error: measure instruction only a register as the\
                #            parameter.')
                #     exit()

                instructions.append(int(self.MeasureFormat(), 2))

            elif (elements[0].lower() == 'wait'):   # Wait imm
                # print('parsing Wait instruction.')
                instructions.append(int(self.WaitFormat(elements[1]), 2))

            elif (elements[0].lower() == 'trigger'):   # Trigger mask, duration
                # print('parsing Trigger instruction.')
                instructions.append(int(self.TriggerFormat(elements[1],
                                                           elements[2]), 2))

            elif (elements[0].lower() == 'nop'):
                # print('parsing nop instruction.')
                instructions.append(int(self.NopFormat(), 2))

            else:
                print('Error: unsupported instruction %s found. Abort!' %
                      elements[0])
                Asm_File.close()
                return False

        print("tag_addr_dict: ", tag_addr_dict)
        Asm_File.close()

        return instructions
