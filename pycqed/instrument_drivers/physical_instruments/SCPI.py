'''
File:       SCPI.py
Author:     Wouter Vlothuizen, TNO/QuTech
Purpose:    base class for SCPI ('Standard Commands for Programmable
            Instruments') commands
Usage:      don't use directly, use a derived class (e.g. QWG)
Notes:
Bugs:
'''

from qcodes import IPInstrument
from qcodes import validators as vals
import socket

"""
FIXME: we would like to be able to choose the base class separately, so the
user can choose (e.g. use VISA for IEE488 bus units, and IpInstrument for
networked units). This would also make the inits cleaner
"""


class SCPI(IPInstrument):

    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port,
                         write_confirmation=False,  # required for QWG
                         **kwargs)

        # send things immediately
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # beef up buffer, to prevent socket.send() not sending all our data
        # in one go
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512*1024)

        # FIXME convert operation etc to parameters
        # IDN is implemented in the instrument base class

        # example of how the commands could look
        self.add_function('reset', call_cmd='*RST')

    def _recv(self):
        """
        Overwrites base IP recv command to ensuring read till EOM
        FIXME: should be in parent class
        """
        return self._socket.makefile().readline().rstrip()
    ###
    # Helpers
    ###

    def readBinary(self, size):
        data = self._socket.recv(size)
        actLen = len(data)
        expLen = size
        i = 1
        while (actLen != expLen):
            data += self._socket.recv(expLen-actLen)
            actLen = len(data)
            i = i+1
        return data

    def writeBinary(self, binMsg):
        self._socket.send(binMsg)       # FIXME: should be in parent class

    def ask_float(self, str):
        return float(self.ask(str))

    def ask_int(self, str):
        return int(self.ask(str))

    ###
    # Generic SCPI commands from IEEE 488.2 (IEC 625-2) standard
    ###

    def clearStatus(self):
        self.write('*CLS')

    def setEventStatusEnable(self, value):
        self.write('*ESE %d' % value)

    def getEventStatusEnable(self):
        return self.ask('*ESE?')

    def getEventStatusEnableRegister(self):
        return self.ask('*ESR?')

    def getIdentity(self):
        return self.ask('*IDN?')

    def operationComplete(self):
        self.write('*OPC')

    def getOperationComplete(self):
        return self.ask('*OPC?')

    def get_operation_complete(self):  # FIXME: PR #638, all naming should be changed to snake_case
        return self.ask('*OPC?')

    def getOptions(self):
        return self.ask('*OPT?')

    def serviceRequestEnable(self, value):
        self.write('*SRE %d' % value)

    def getServiceRequestEnable(self):
        return self.ask_int('*SRE?')

    def getStatusByte(self):
        return self.ask_int('*STB?')

    def getTestResult(self):
        # NB: result bits are device dependent
        return self.ask_int('*TST?')

    def trigger(self):
        self.write('*TRG')

    def wait(self):
        self.write('*WAI')

    def reset(self):
        self.write('*RST')

    ###
    # Required SCPI commands (SCPI std V1999.0 4.2.1)
    ###

    def getError(self):
        ''' Returns:    '0,"No error"' or <error message>
        '''
        return self.ask('system:err?')

    def getSystemErrorCount(self):
        return self.ask_int('system:error:count?')

    def getSystemVersion(self):
        return self.ask('system:version?')

    ###
    # IEEE 488.2 binblock handling
    ###

    def binBlockWrite(self, binBlock, header):
        '''
        write IEEE488.2 binblock

        Args:
            binBlock (bytearray): binary data to send

            header (string): command string to use
        '''
        totHdr = header + SCPI.buildHeaderString(len(binBlock))
        binMsg = totHdr.encode() + binBlock
        self.writeBinary(binMsg)
        self.write('')                  # add a Line Terminator

    def binBlockRead(self):
        # FIXME: untested
        ''' read IEEE488.2 binblock
        '''
        # get and decode header
        headerA = self.readBinary(2)                        # read '#N'
        headerAstr = headerA.decode()
        if(headerAstr[0] != '#'):
            s = 'SCPI header error: received {}'.format(headerA)
            raise RuntimeError(s)
        digitCnt = int(headerAstr[1])
        headerB = self.readBinary(digitCnt)
        byteCnt = int(headerB.decode())
        binBlock = self.readBinary(byteCnt)
        self.readBinary(2)                                  # consume <CR><LF>
        return binBlock

    @staticmethod
    def buildHeaderString(byteCnt):
        ''' generate IEEE488.2 binblock header
        '''
        byteCntStr = str(byteCnt)
        digitCntStr = str(len(byteCntStr))
        binHeaderStr = '#' + digitCntStr + byteCntStr
        return binHeaderStr
