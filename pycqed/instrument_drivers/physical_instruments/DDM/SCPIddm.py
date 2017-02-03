'''
    File:               SCPIdriver.py
    Author:             Wouter Vlothuizen, TNO/QuTech
    Purpose:            base class for SCPI ('Standard Commands for Programmable Instruments') devices talking via Ethernet
    Usage:              don't use directly, use a derived class (e.g. AWG520)
                        does not depend on other software or drivers (e.g. Visa, IVI, etc)
    Bugs:
        - ignores responses and status
        - some member variables still unused

'''

from qcodes import IPInstrument
from qcodes import validators as vals
import socket


class SCPIddm(IPInstrument):

    
    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port,
                         write_confirmation=False,  # required for QWG
                         **kwargs)

        # send things immediately
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # beef up buffer, to prevent socket.send() not sending all our data
        # in one go
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512*1024)
        
    def open(self, host, port=5025):
        ''' open connection, e.g. open('192.168.0.16', 4000)
        '''
            # first set timeout (before connect)
        self._socket.settimeout(10)
            # beef up buffer, to prevent socket.send() not sending all our data
            # in one go
            #self.socket.setsockopt(
            #    socket.SOL_SOCKET, socket.SO_RCVBUF , 1024*1024)
        self._socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF , 32*1024)
        print (self._socket.getsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF ))

            #self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.connect((host, port))
        # FIXME: log instrument info:
        # obj.getIdentity()
        # obj.getOptions()

    def close(self):
        ''' close connection
        '''
        self._socket.close()
    def _recv(self):
        """
        Overwrites base IP recv command to ensuring read till EOM
        FIXME: should be in parent class
        """
        return self._socket.makefile().readline().rstrip()
   
        # FIXME: logging
        # FIXME: check for SCPI errors (in debug mode)

    def writeBinary(self, binMsg):
        self._socket.send(binMsg)       # FIXME: should be in parent class

    #def writeBinary(self, data):
        ''' send binary data
                Input:
                        data    bytearray
        '''
       # expLen = len(data)
        #actLen = self._socket.send(data)
       # if(actLen != expLen):
                # FIXME: handle this case by calling send again. Or enlarge
                # socket.SO_SNDBUF even further
       #     raise UserWarning(
       #         'not all data sent: expected %d, actual %d' % (expLen, actLen))

        # FIXME: logging
        # FIXME: check for SCPI errors (in debug mode)

    def readBinary(self, byteCnt):
        ''' read binary data
        '''
        #if not self.simMode:
        data = self._socket.recv(byteCnt)
        
        expLen = byteCnt
        actLen = len(data)
        
        i=1
        while (actLen != expLen):
            data += self._socket.recv(expLen-actLen)
            actLen = len(data)
            i=i+1
            print('i=%d' % i)
           
                #raise UserWarning(
                    #'not all data sent: expected %d, actual %d' % (expLen, actLen))

        #else:
        #    data = zeros(byteCnt, 1)

        # FIXME: logging
        return data

                               # remove trailing white space, CR, LF

    def askDouble(self, str):
        resp = self.ask(str)
        return str2double(resp)

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

    def getOptions(self):
        return self.ask('*OPT?')

    def serviceRequestEnable(self, value):
        self.write('*SRE %d' % value)

    def getServiceRequestEnable(self):
        return self.ask('*SRE?')

    def getStatusByte(self):
        return self.ask('*STB?')

    def getTestResult(self):
        # NB: result bits are device dependent
        return self.ask('*TST?')

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
        return self.ask('system:error:count?')

    def getSystemVersion(self):
        return self.ask('system:version?')

    def binBlockWrite(self, binBlock, header):
        ''' write IEEE488.2 binblock
                Input:
                        binBlock    bytearray
                        header      string
        '''
        totHdr = header + SCPIddm.buildHeaderString(len(binBlock))
        binMsg = totHdr.encode() + binBlock
        self.writeBinary(binMsg)
        self.write('')                                # add a Line Terminator

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
        print(digitCnt)
        headerB = self.readBinary(digitCnt)
        byteCnt = int(headerB.decode())
        print('byteCnt%d' % byteCnt)
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

    @staticmethod
    def getByteCntFromHeader(headerStr):
        ''' decode IEEE488.2 binblock header
        '''
        # FIXME: old Matlab code
        digitCnt = sscanf(headerStr, '#%1d')
        formatString = sprintf('%%%dd', digitCnt)           # e.g. '%3d'
        # byteCnt = sscanf(headerStr(3:end), formatString)        # NB: skip
        # first '#N'
        return byteCnt

'''
        function flushInput(obj)
            % FIXME
            %   is = obj.socket.getInputStream();
            %   inCnt = is.available()
            %   is.read()
        end
'''
