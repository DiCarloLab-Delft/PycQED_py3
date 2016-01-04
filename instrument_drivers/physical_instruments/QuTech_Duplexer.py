import logging
import socket
import os
import qt
import numpy as np
from instrument import Instrument
from time import sleep, time
import select
import h5py


class QuTech_Duplexer(Instrument):
    '''
    This is the python driver for the QuTech duplexer made by TNO.
    Usage:
    Initialize with
    <name> = instruments.create('name', 'QuTech_Duplexer', address='<TCPIP address>')
    '''

    def __init__(self, name, address='TCPIP0::192.168.0.100', reset=False):
        '''
        Initializes the QuTech_Duplexer, and communicates with the wrapper.

        Input:
            name (string)    : name of the instrument
            address (string) : TCPIP address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])
        self.S = Socket((address[8:]), 5025)
        self.SCPI_command_pause = 0.1

        cal_file_path = os.path.join(
            qt.config['PycQEDdir'], 'instrument_drivers',
            'physical_instruments', '_Duplexer',
            'Duplexer_normalized_gain.hdf5')
        cal_file = h5py.File(cal_file_path, 'r')
        self.calibration_array = list(cal_file.values())[0][1]
        # Input 1
        self.add_parameter('in1_out1_switch', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            option_list=('ON', 'OFF', 'EXT'))
        self.add_parameter('in1_out1_attenuation', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536)
        self.add_parameter('in1_out1_phase', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536)
        self.add_parameter('in1_out2_switch', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            option_list=('ON', 'OFF', 'EXT'))
        self.add_parameter('in1_out2_attenuation', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536)
        self.add_parameter('in1_out2_phase', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536)

        # Input 2
        self.add_parameter('in2_out1_switch', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            option_list=('ON','OFF','EXT'))
        self.add_parameter('in2_out1_attenuation', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        self.add_parameter('in2_out1_phase', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        self.add_parameter('in2_out2_switch', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            option_list=('ON','OFF','EXT'))
        self.add_parameter('in2_out2_attenuation', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536)
        self.add_parameter('in2_out2_phase', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536)
        # Input 3
        self.add_parameter('in3_out1_switch', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            option_list=('ON', 'OFF', 'EXT'))
        self.add_parameter('in3_out1_attenuation', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        self.add_parameter('in3_out1_phase', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        self.add_parameter('in3_out2_switch', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            option_list=('ON', 'OFF', 'EXT'))
        self.add_parameter('in3_out2_attenuation', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        self.add_parameter('in3_out2_phase', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        # Input 4
        self.add_parameter('in4_out1_switch', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            option_list=('ON','OFF','EXT'))
        self.add_parameter('in4_out1_attenuation', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        self.add_parameter('in4_out1_phase', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        self.add_parameter('in4_out2_switch', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            option_list=('ON','OFF','EXT'))
        self.add_parameter('in4_out2_attenuation', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )
        self.add_parameter('in4_out2_phase', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0, maxval=65536 )

        #Initialize all to zero
        self.set_all_switches_to('OFF')
        self.set_all_phases_to(0)
        self.set_all_attenuations_to(0)

    ############################################
    #  Functions for the actual VISA commands  #
    ############################################
    def IDN(self):
        logging.warning('IDN currently not functional.' +
                        ' Assuming error is in Duplexer MAR-12-2014 ')

        idn = self.S.ask('*IDN?')
        return idn

    def set_all_switches_to(self, mode):
        for i in range(4):
            for j in range(2):
                self.set_switch(i+1, j+1, mode)

    def set_all_phases_to(self, val):
        for i in range(4):
            for j in range(2):
                self.set_phase(i+1, j+1, val)

    def set_all_attenuations_to(self, val):
        for i in range(4):
            for j in range(2):
                self.set_attenuation(i+1, j+1, val)

    def set_switch(self, input_ch, output_ch, mode):
        '''
        Public version
        Sets switch between 'input_ch' and 'output_ch' to 'mode'
            input_ch, int in '1-4'
            output_ch, int in '1-2'
            mode, str in  'on,off,ext'
                ext is external triggering mode
        '''
        return eval('self.set_in%s_out%s_switch("%s")' %
                    (input_ch, output_ch, mode))

    def get_switch(self, input_ch, output_ch):
        return eval('self.in%s_out%s_switch' %(input_ch, output_ch))

    def set_attenuation(self, input_ch, output_ch, value, mode='raw'):
        '''
        Public version
        Sets attenuation between 'input_ch' and 'output_ch' to 'value'.
            optional value 'mode' can be set to 'raw' or 'cal' (calibrated).
                'cal' mode is currently not implemented in duplexer.
            input_ch, int in '1-4'
            output_ch, int in '1-2'
            value, int in '0-(2^16-1)' (highest value is lowest attentuation)
        '''
        return eval('self.set_in%s_out%s_attenuation(%s)' %
                    (input_ch, output_ch, value))

    def get_attenuation(self, input_ch, output_ch):
        return eval('self.in%s_out%s_attenuation' % (input_ch, output_ch))

    def set_phase(self, input_ch, output_ch, phase, mode='raw'):
        return eval('self.set_in%s_out%s_phase(%s)' %
                    (input_ch, output_ch, phase))

    def get_phase(self, input_ch, output_ch):
        return eval('self.in%s_out%s_phase' % (input_ch, output_ch))

    def _set_switch(self, input_ch, output_ch, mode):
        '''
        Private version
        Sets switch between 'input_ch' and 'output_ch' to 'mode'
            input_ch, int in '1-4'
            output_ch, int in '1-2'
            mode, str in  'on,off,ext'
                ext is external triggering mode
        '''
        if (type(input_ch) != int):
            raise TypeError
        if input_ch not in [1, 2, 3, 4]:
            raise ValueError
        if (type(output_ch) != int):
            raise TypeError
        if output_ch not in [1, 2]:
            raise ValueError
        if (type(mode) != str):
            raise TypeError
        if (mode not in ['ON', 'OFF', 'EXT']):
            raise ValueError('Mode "%s" not in [on,off,ext]' % mode)
        self.S.write('ch:in%s:out%s:sw %s \n' % (input_ch, output_ch, mode))
        qt.msleep(self.SCPI_command_pause)
        return True

    def get_scaling_increment(self, scaling_factor):
        cal_slope = 2048  # dB/attenuation_dac_val \
        scaling_in_dB = 10 * np.log10(scaling_factor)
        scaling_increment = cal_slope * scaling_in_dB
        return scaling_increment

    def calculate_attenuation(self, current_dac_value, scaling_factor):

        try:
            current_val = self.calibration_array[current_dac_value]
        except:
            print('Setting wrong attenuation for Duplexer')
            if current_val < 0:
                current_val = self.calibration_array[-1]
            else:
                current_val = self.calibration_array[0]
        final_val = self.calibration_array - current_val * scaling_factor
        new_dac_val = np.argmin(np.abs(final_val))
        return new_dac_val

    def _set_attenuation(self, input_ch, output_ch, value, mode='raw'):
        '''
        Sets attenuation between 'input_ch' and 'output_ch' to 'value'.
            optional value 'mode' can be set to 'raw' or 'cal' (calibrated).
                'cal' mode is currently not implemented in duplexer.
            input_ch, int in '1-4'
            output_ch, int in '1-2'
            value, int in '0-(2^16-1)' (highest value is lowest attentuation)
        '''
        if (type(input_ch) != int):
            raise TypeError
        if input_ch not in [1, 2, 3, 4]:
            raise ValueError
        if (type(output_ch) != int):
            raise TypeError
        if output_ch not in [1, 2]:
            raise ValueError
        if (type(value) != int):
            raise TypeError
        if output_ch not in list(range(65536)):
            raise ValueError
        if (type(mode) != str):
            raise TypeError
        if (mode not in ['raw', 'cal']):
            raise ValueError

        self.S.write('ch:in%s:out%s:att:%s %s \n' %(input_ch,output_ch,mode,value))
        qt.msleep(self.SCPI_command_pause)
        return True


    def _set_phase(self,input_ch,output_ch,phase,mode ='raw'):
        if (type(input_ch)!=int):
            raise TypeError
        if input_ch not in [1,2,3,4]:
            raise ValueError
        if (type(output_ch)!=int):
            raise TypeError
        if output_ch not in [1,2]:
            raise ValueError
        if (type(phase)!=int):
            raise TypeError
        if phase not in list(range(65536)):
            raise ValueError

        if (type(mode)!=str):
            raise TypeError
        if (mode not in ['raw','cal']):
            raise ValueError

        self.S.write('ch:in%s:out%s:ph:%s %s \n' % (input_ch, output_ch, mode, phase))
        qt.msleep(self.SCPI_command_pause)
        return True



    ##############
    # get and set funcs
    # Input 1
    def _do_set_in1_out1_switch(self,mode):
        self._set_switch(1,1,mode)
        self.in1_out1_switch=mode
        return True

    def _do_get_in1_out1_switch(self):
        return self.in1_out1_switch

    def _do_set_in1_out1_attenuation(self,value):
        '''
        Set attenuation value, currently only supports "raw"
        '''
        self._set_attenuation(1,1,value)
        self.in1_out1_attenuation=value
        return True

    def _do_get_in1_out1_attenuation(self):
        return self.in1_out1_attenuation

    def _do_set_in1_out1_phase(self,value):
        '''
        Set phase value, currently only supports "raw"
        '''
        self._set_phase(1,1,value)
        self.in1_out1_phase=value
        return True

    def _do_get_in1_out1_phase(self):
        return self.in1_out1_phase

    def _do_set_in1_out2_switch(self,mode):
        self._set_switch(1,2,mode)
        self.in1_out2_switch=mode
        return True

    def _do_get_in1_out2_switch(self):
        return self.in1_out2_switch

    def _do_set_in1_out2_attenuation(self,value):
        '''
        Set attenuation value, currently only supports "raw"
        '''
        self._set_attenuation(1,2,value)
        self.in1_out2_attenuation=value
        return True

    def _do_get_in1_out2_attenuation(self):
        return self.in1_out2_attenuation

    def _do_set_in1_out2_phase(self,value):
        '''
        Set phase value, currently only supports "raw"
        '''
        self._set_phase(1, 2, value)
        self.in1_out2_phase=value
        return True

    def _do_get_in1_out2_phase(self):
        return self.in1_out2_phase

    # Input 2
    def _do_set_in2_out1_switch(self,mode):
        self._set_switch(2,1,mode)
        self.in2_out1_switch=mode
        return True

    def _do_get_in2_out1_switch(self):
        return self.in2_out1_switch

    def _do_set_in2_out1_attenuation(self,value):
        '''
        Set attenuation value, currently only supports "raw"
        '''
        self._set_attenuation(2,1,value)
        self.in2_out1_attenuation=value
        return True

    def _do_get_in2_out1_attenuation(self):
        return self.in2_out1_attenuation

    def _do_set_in2_out1_phase(self,value):
        '''
        Set phase value, currently only supports "raw"
        '''
        self._set_phase(2,1,value)
        self.in2_out1_phase=value
        return True

    def _do_get_in2_out1_phase(self):
        return self.in2_out1_phase

    def _do_set_in2_out2_switch(self,mode):
        self._set_switch(2,2,mode)
        self.in2_out2_switch=mode
        return True

    def _do_get_in2_out2_switch(self):
        return self.in2_out2_switch

    def _do_set_in2_out2_attenuation(self,value):
        '''
        Set attenuation value, currently only supports "raw"
        '''
        self._set_attenuation(2,2,value)
        self.in2_out2_attenuation=value
        return True

    def _do_get_in2_out2_attenuation(self):
        return self.in2_out2_attenuation

    def _do_set_in2_out2_phase(self,value):
        '''
        Set phase value, currently only supports "raw"
        '''
        self._set_phase(2,2,value)
        self.in2_out2_phase=value
        return True

    def _do_get_in2_out2_phase(self):
        return self.in2_out2_phase


    # Input 3
    def _do_set_in3_out1_switch(self,mode):
        self._set_switch(3,1,mode)
        self.in3_out1_switch=mode
        return True

    def _do_get_in3_out1_switch(self):
        return self.in3_out1_switch

    def _do_set_in3_out1_attenuation(self,value):
        '''
        Set attenuation value, currently only supports "raw"
        '''
        self._set_attenuation(3,1,value)
        self.in3_out1_attenuation=value
        return True

    def _do_get_in3_out1_attenuation(self):
        return self.in3_out1_attenuation

    def _do_set_in3_out1_phase(self,value):
        '''
        Set phase value, currently only supports "raw"
        '''
        self._set_phase(3,1,value)
        self.in3_out1_phase=value
        return True

    def _do_get_in3_out1_phase(self):
        return self.in3_out1_phase

    def _do_set_in3_out2_switch(self,mode):
        self._set_switch(3,2,mode)
        self.in3_out2_switch=mode
        return True

    def _do_get_in3_out2_switch(self):
        return self.in3_out2_switch

    def _do_set_in3_out2_attenuation(self,value):
        '''
        Set attenuation value, currently only supports "raw"
        '''
        self._set_attenuation(3,2,value)
        self.in3_out2_attenuation=value
        return True

    def _do_get_in3_out2_attenuation(self):
        return self.in3_out2_attenuation

    def _do_set_in3_out2_phase(self,value):
        '''
        Set phase value, currently only supports "raw"
        '''
        self._set_phase(3,2,value)
        self.in3_out2_phase=value
        return True

    def _do_get_in3_out2_phase(self):
        return self.in3_out2_phase

    # Input 4
    def _do_set_in4_out1_switch(self,mode):
        self._set_switch(4,1,mode)
        self.in4_out1_switch=mode
        return True

    def _do_get_in4_out1_switch(self):
        return self.in4_out1_switch

    def _do_set_in4_out1_attenuation(self,value):
        '''
        Set attenuation value, currently only supports "raw"
        '''
        self._set_attenuation(4,1,value)
        self.in4_out1_attenuation=value
        return True

    def _do_get_in4_out1_attenuation(self):
        return self.in4_out1_attenuation

    def _do_set_in4_out1_phase(self,value):
        '''
        Set phase value, currently only supports "raw"
        '''
        self._set_phase(4,1,value)
        self.in4_out1_phase=value
        return True

    def _do_get_in4_out1_phase(self):
        return self.in4_out1_phase

    def _do_set_in4_out2_switch(self,mode):
        self._set_switch(4,2,mode)
        self.in4_out2_switch=mode
        return True

    def _do_get_in4_out2_switch(self):
        return self.in4_out2_switch

    def _do_set_in4_out2_attenuation(self,value):
        '''
        Set attenuation value, currently only supports "raw"
        '''
        self._set_attenuation(4,2,value)
        self.in4_out2_attenuation=value
        return True

    def _do_get_in4_out2_attenuation(self):
        return self.in4_out2_attenuation

    def _do_set_in4_out2_phase(self,value):
        '''
        Set phase value, currently only supports "raw"
        '''
        self._set_phase(4,2,value)
        self.in4_out2_phase=value
        return True

    def _do_get_in4_out2_phase(self):
        return self.in1_out2_phase

class Socket:
    # Should be moved to qtlab source as it is currently being used in multiple user instruments
    def __init__(self, host, port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((host, port))
        self._socket.settimeout(20)

    def clear(self):
        rlist, wlist, xlist = select.select([self._socket], [], [], 0)
        if len(rlist) == 0:
            return
        ret = self.read()
        print('Unexpected data before ask(): %r' % (ret, ))

    def write(self, data):
        self.clear()
        if len(data) > 0 and data[-1] != '\r\n':
            data += '\n'
        self._socket.send(data)

    def read(self, timeouttime=20):
        start = time()
        try:
            ans = ''
            while len(ans) == 0 and (time() - start) < timeouttime or not has_newline(ans):
                ans2 = self._socket.recv(8192)
                ans += ans2
                if len(ans2) == 0:
                    sleep(0.01)
            AWGlastdataread=ans
        except socket.timeout as e:
            print('Timed out')
            return ''

        if len(ans) > 0:
            ans = ans.rstrip('\r\n')
        return ans

    def ask(self, data):
        self.clear()
        self.write(data)
        return self.read()
