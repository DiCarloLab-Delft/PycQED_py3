import zhinst.ziPython as zi
import zhinst.utils as zi_utils
import time
import json
import os
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from fnmatch import fnmatch



class UHFQC(Instrument):
    """
    This is the qcodes driver for the 1.8 Gsample/s UHF-QC developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5 ucs4 16.04 for 64bit Windows from http://www.zhinst.com/downloads
    2. pip install dependencies: httplib2, plotly, pyqtgraph
    3. upload the latest firmware to the UHFQC by opening reboot.bat in 'Transmon\Inventory\ZurichInstruments\firmware_Nielsb\firmware_x'. WIth x the highest available number.
    6. find out where sequences are stored by saving a sequence from the GUI and then check :"showLog" to see where it is stored. This is the location where AWG sequences can be loaded from.
    misc: when device crashes, check the log file in 
    EOM
    """

    def __init__(self, name, server_name, device='auto', interface='USB', address='127.0.0.1', port=8004, **kw):
        '''
        Input arguments: 
            name:           (str) name of the instrument 
            server_name:    (str) qcodes instrument server 
            address:        (int) the address of the data server e.g. 8006
        '''

        t0 = time.time()
        super().__init__(name, server_name)

        self._daq = zi.ziDAQServer(address, int(port), 5)
        if device.lower() == 'auto':
            self._device = zi_utils.autoDetect(self._daq)
        else:
            self._device = device
            self._daq.connectDevice(self._device, interface)
        self._device = zi_utils.autoDetect(self._daq)
        s_node_pars=[]
        d_node_pars=[]


        self._s_file_name ='zi_parameter_files/s_node_pars_{}.txt'.format(self._device)
        self._d_file_name = 'zi_parameter_files/d_node_pars_{}.txt'.format(self._device)

        print(self._s_file_name)
        print(self._d_file_name)

        try:
            f=open(self._s_file_name).read()
            s_node_pars = json.loads(f)
        except:
            print("parameter file for gettable parameters {} not found".format(self._s_file_name))
        try:
            f=open(self._d_file_name).read()
            d_node_pars = json.loads(f)
        except:
            print("parameter file for settable parameters {} not found".format(self._d_file_name))

        for parameter in s_node_pars:
            parname=parameter[0][9:].replace("/","_")
            if parameter[1]=='float':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setd, parameter[0]),
                    get_cmd=self._gen_get_func(self.getd, parameter[0]),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1]=='float_small':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setd, parameter[0]),
                    get_cmd=self._gen_get_func(self.getd, parameter[0]),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1]=='int_8bit':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parameter[0]),
                    get_cmd=self._gen_get_func(self.geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='int':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parameter[0]),
                    get_cmd=self._gen_get_func(self.geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='int_64':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parameter[0]),
                    get_cmd=self._gen_get_func(self.geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='bool':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parameter[0]),
                    get_cmd=self._gen_get_func(self.geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            else:
                print("parameter {} type {} from from s_node_pars not recognized".format(parname,parameter[1]))

        for parameter in d_node_pars:
            parname=parameter[0][9:].replace("/","_")
            if parameter[1]=='float':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(self.getd, parameter[0]))
            elif parameter[1]=='vector_g':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(self.getv, parameter[0]))
            elif parameter[1]=='vector_s':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setv, parameter[0]),
                    vals=vals.Anything())
            else:   
                print("parameter {} type {} from d_node_pars not recognized".format(parname,parameter[1]))


        self.add_parameter('AWG_file',
                           set_cmd=self._do_set_AWG_file,
                           vals=vals.Anything())

        self.load_default_settings()
        t1 = time.time()
        print('Initialized UHFQC', self._device,
              'in %.2fs' % (t1-t0))

    def load_default_settings(self):
        #standard configurations adapted from Haendbaek's notebook
        # Run this block to do some standard configuration

        # The averaging-count is used to specify how many times the AWG program should run
        LOG2_AVG_CNT = 6

        # This averaging count specifies how many measurements the result logger should average
        LOG2_RL_AVG_CNT = 0

        # Load an AWG program (from Zurich Instruments/LabOne/WebServer/awg/src)
        self.AWG_file('quexpress.seqc')

        # The AWG program uses userregs/0 to define the number o iterations in the loop
        self.seti('awgs/0/userregs/0', pow(2, LOG2_AVG_CNT)*pow(2, LOG2_RL_AVG_CNT))

        # Turn on both outputs
        self.seti('sigouts/0/on', 1)
        self.seti('sigouts/1/on', 1)

        # Configure the input averager length and averaging count
        self.seti('quex/iavg/length', 2048)
        self.seti('quex/iavg/avgcnt', LOG2_AVG_CNT)

        # QuExpress thresholds on DIO (mode == 2), AWG control of DIO (mode == 1)
        self.seti('dios/0/mode', 2)
        # Drive DIO bits 31 to 16
        self.seti('dios/0/drive', 0xc)

        # Configure the analog trigger input 1 of the AWG to assert on a rising edge on Ref/Trigger 1 (front-panel of the instrument)
        self.seti('awgs/0/triggers/0/rising', 1)
        self.setd('awgs/0/triggers/0/level', 0.000000000)
        self.seti('awgs/0/triggers/0/channel', 2)

        # Straight connection, signal input 1 to channel 1, signal input 2 to channel 2
        self.setd('quex/deskew/0/col/0', 1.0)
        self.setd('quex/deskew/0/col/1', 0.0)
        self.setd('quex/deskew/1/col/0', 0.0)
        self.setd('quex/deskew/1/col/1', 1.0)

        # Configure the weighted integration units with constant values, each channel gets
        # one, two, three and four non-zero weights, respectively
        self.setv('quex/wint/weights/0/real', np.array([1.0]*1 + [0.0]*127))
        self.setv('quex/wint/weights/0/imag', np.array([1.0]*1 + [0.0]*127))
        self.setv('quex/wint/weights/1/real', np.array([1.0]*2 + [0.0]*126))
        self.setv('quex/wint/weights/1/imag', np.array([1.0]*2 + [0.0]*126))
        self.setv('quex/wint/weights/2/real', np.array([1.0]*3 + [0.0]*125))
        self.setv('quex/wint/weights/2/imag', np.array([1.0]*3 + [0.0]*125))
        self.setv('quex/wint/weights/3/real', np.array([1.0]*12 + [0.0]*116))
        self.setv('quex/wint/weights/3/imag', np.array([1.0]*12 + [0.0]*116))

        # Length is set in units of 1 samples
        self.seti('quex/wint/length', 4)
        self.seti('quex/wint/delay', 0)
            
        # No rotation on the output of the weighted integration units, i.e. take real part of result
        for i in range(0, 4):
            self.setd('quex/rot/{0}/real'.format(i), 1.0)
            self.setd('quex/rot/{0}/imag'.format(i), 0.0)

        # No cross-coupling in the matrix multiplication (identity matrix)
        for i in range(0, 4):
            for j in range(0, 4):
                if i == j:
                    self.setd('quex/trans/{0}/col/{1}/real'.format(i,j), 1.0)
                else:
                    self.setd('quex/trans/{0}/col/{1}/real'.format(i,j), 0.0)
                    
        # Configure the result logger to not do any averaging
        self.seti('quex/rl/length', pow(2, LOG2_AVG_CNT))
        self.seti('quex/rl/avgcnt', LOG2_RL_AVG_CNT)
        self.seti('quex/rl/source', 2)

        # Ready for readout
        self.seti('quex/iavg/readout', 1)
        self.seti('quex/rl/readout', 1)

        # The custom firmware will feed through the signals on Signal Input 1 to Signal Output 1 and Signal Input 2 to Signal Output 2
        # when the AWG is OFF. For most practical applications this is not really useful. We, therefore, disable the generation of
        # these signals on the output here.
        for i in range(0, 2):
            for j in range(0, 4):
                self.seti('sigouts/{0}/enables/{1}'.format(i, j), 0)

    def _gen_set_func(self, dev_set_type, cmd_str):
        def set_func(val):
            dev_set_type(cmd_str, val)
            return dev_set_type(cmd_str, value=val)
        return set_func

    def _gen_get_func(self, dev_get_type, ch):
        def get_func():
            return dev_get_type(ch)
        return get_func

    def reconnect(self):
        zi_utils.autoDetect(self._daq)

    def _do_set_AWG_file(self, filename):
        h = self._daq.awgModule()
        h.set('awgModule/device', self._device)
        h.set('awgModule/index', 0)
        h.execute()
        h.set('awgModule/compiler/sourcefile', filename)
        h.set('awgModule/compiler/start', 1)
        h.set('awgModule/elf/file', '')        

        # code to upload AWG sequence as a string 
        # def awg(self, filename):
        #         for device in self._devices:
        #             h = self.daq.awgModule()
        #             h.set('awgModule/device', device)
        #             h.set('awgModule/index', 0)
        #             h.execute()
        #             h.set('awgModule/compiler/sourcefile', filename)
        #             h.set('awgModule/compiler/start', 1)
        #             h.set('awgModule/elf/file', '’)        

        # Now, if you would change it to:

        #  def awg(self, sourcestring):
        #         for device in self._devices:
        #             h = self.daq.awgModule()
        #             h.set('awgModule/device', device)
        #             h.set('awgModule/index', 0)
        #             h.execute()
        #             h.set('awgModule/compiler/sourcestring', sourcestring)
        #             h.set('awgModule/compiler/start', 1)
        #             h.set('awgModule/elf/file', '’)    

    def close(self): 
        self._daq.disconnectDevice(self._device)

    def find(self, *args):
        nodes = self._daq.listNodes('/', 7)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower() for k in nodes if fnmatch(k.lower(), m.lower())]

        return nodes

    def finds(self, *args):
        nodes = self._daq.listNodes('/', 15)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower() for k in nodes if fnmatch(k.lower(), m.lower())]

        return nodes


    def create_parameter_files(self):
        #this functions retrieves all possible settable and gettable parameters from the device.
        #Additionally, iot gets all minimum and maximum values for the parameters by trial and error

        s_node_pars=[]
        d_node_pars=[]
        patterns = ["awgs", "sigins", "sigouts", "quex"]
        for pattern in patterns:
            print("extracting parameters of type", pattern)
            all_nodes = set(self.find('*{}*'.format(pattern)))
            s_nodes = set(self.finds('*{}*'.format(pattern)))
            d_nodes = all_nodes.difference(s_nodes)
            print(len(all_nodes))
            # extracting info from the setting nodes
            s_nodes = list(s_nodes)
            default_values=self.getd(s_nodes, True)
            for s_node in s_nodes:
                self.setd(s_node,  1e12)
            max_values = self.getd(s_nodes, True)
            for s_node in s_nodes:
                self.setd(s_node, -1e12)
            min_values = self.getd(s_nodes, True)
            float_values = [np.pi]*len(s_nodes)
            for i, s_node in enumerate(s_nodes):
                if np.pi > max_values[i]:
                    float_values[i] = max_values[i]/np.pi;            
                self.setd(s_node, float_values[i])
            actual_float_values = self.getd(s_nodes, True)        
            node_types = ['']*len(s_nodes)
            for i, s_node in enumerate(s_nodes):
                #self.setd(node,default_values[i])
                fraction, integer = np.modf(actual_float_values[i])
                if fraction != 0:
                    node_types[i] = 'float'
                    if min_values[i]==max_values[i]:
                        node_types[i]='float_small'
                        min_values[i]=0
                    elif abs(min_values[i])<0.01:
                        min_values[i]=0
                else:
                    node_types[i] = 'int'
                    min_values[i]=0
                    if  max_values[i]==3567587328:
                        node_types[i] = 'int_64'
                        max_values[i]=4294967295
                    elif  max_values[i]==1:
                        node_types[i] = 'bool'
                    elif max_values[i]==0:
                        max_values[i]=255
                        node_types[i] = 'int_8bit'
                    elif max_values[i]>4294967295:
                        node_types[i] = 'float'
                line=[s_node, node_types[i], min_values[i], max_values[i], '\n']
                print(line)
                s_node_pars.append(line)
          
            #extracting info from the data nodes
            d_nodes = list(d_nodes)
            #default_values=self.getd(d_nodes, True)
            default_values=np.zeros(len(d_nodes))
            node_types = ['']*len(d_nodes)
            for i, d_node in enumerate(d_nodes):
                try: 
                    answer=self.getv(d_node)
                    if isinstance(answer, dict):
                        value=answer['value'][0]
                        node_types[i]='float'
                    elif  isinstance(answer, list):
                        value=answer[0]['vector']
                        node_types[i]='vector_g'
                    else:
                        print("unknown type")
                except:
                    node_types[i]='vector_s'
                line=[d_node, node_types[i]]#, default_values[i]]
                print(line)
                d_node_pars.append(line)
        f = open(self._s_file_name, 'w')
        json.dump(s_node_pars, f,default=int)
        f.close()

        f = open(self._d_file_name, 'w')
        json.dump(d_node_pars, f, default=int)
        f.close()


    def setd(self, path, value):
        # Handle absolute path
        if path[0] == '/':
            self._daq.setDouble(path, value)
        else:
            self._daq.setDouble('/' + self._device + '/' + path, value)

    def seti(self, path, value):
        # Handle absolute path
        if path[0] == '/':
            self._daq.setInt(path, value)
        else:
            self._daq.setInt('/' + self._device + '/' + path, value)

    def setv(self, path, value):
        # Handle absolute path
        if path[0] == '/':
            self._daq.vectorWrite(path, value)
        else:
            self._daq.vectorWrite('/' + self._device + '/' + path, value)

    def geti(self, paths, deep=True):
        if type(paths) is not list:
            paths = [ paths ]
            single = 1
        else:
            single = 0
        
        values = []
        for p in paths:
            if p[0] == '/':
                if deep:
                    self._daq.getAsEvent(p)
                    tmp = self._daq.poll(0.1, 500, 4, True)
                    if p in tmp:
                        values.append(tmp[p]['value'][0])
                else:
                    values.append(self._daq.getInt(p))
            else:
                tmp_p = '/' + self._device + '/' + p
                if deep:
                    self._daq.getAsEvent(tmp_p)
                    tmp = self._daq.poll(0.1, 500, 4, True)
                    if tmp_p in tmp:
                        values.append(tmp[tmp_p]['value'][0])
                else:
                    values.append(self._daq.getInt(tmp_p))
        if single:
            return values[0]
        else:
            return values

    def getd(self, paths, deep=True):
        if type(paths) is not list:
            paths = [ paths ]
            single = 1
        else:
            single = 0
        
        values = []
        for p in paths:
            if p[0] == '/':
                if deep:
                    self._daq.getAsEvent(p)
                    tmp = self._daq.poll(0.1, 500, 4, True)
                    if p in tmp:
                        values.append(tmp[p]['value'][0])
                else:
                    values.append(self._daq.getDouble(p))
            else:
                tmp_p = '/' + self._device + '/' + p
                if deep:
                    self._daq.getAsEvent(tmp_p)
                    tmp = self._daq.poll(0.1, 500, 4, True)
                    if tmp_p in tmp:
                        values.append(tmp[tmp_p]['value'][0])
                else:
                    values.append(self._daq.getDouble(tmp_p))
        if single:
            return values[0]
        else:
            return values

    def getv(self, paths):
        if type(paths) is not list:
            paths = [ paths ]
            single = 1
        else:
            single = 0
        
        values = []
        for p in paths:
            if p[0] == '/':
                self._daq.getAsEvent(p)
                tmp = self._daq.poll(0.5, 500, 4, True)
                if p in tmp:
                    values.append(tmp[p])
            else:
                tmp_p = '/' + self._device + '/' + p
                self._daq.getAsEvent(tmp_p)
                tmp = self._daq.poll(0.5, 500, 4, True)
                if tmp_p in tmp:
                    values.append(tmp[tmp_p])
        if single:
            return values[0]
        else:
            return values



