import zhinst.ziPython as zi
import zhinst.utils as zi_utils
import time
import json
import os
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from fnmatch import fnmatch
#from instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC



class UHFQC(Instrument):
    """
    This is the qcodes driver for the 1.8 Gsample/s UHF-QC developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5 ucs4 16.04 for 64bit Windows from http://www.zhinst.com/downloads, https://people.zhinst.com/~niels/
    2. pip install dependencies: httplib2, plotly, pyqtgraph
    3. upload the latest firmware to the UHFQC by opening reboot.bat in 'Transmon\Inventory\ZurichInstruments\firmware_Nielsb\firmware_x'. WIth x the highest available number.
    4. find out where sequences are stored by saving a sequence from the GUI and then check :"showLog" to see where it is stored. This is the location where AWG sequences can be loaded from.
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
        #self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) #suggestion W vlothuizen
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

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self._s_file_name = os.path.join(dir_path, 'zi_parameter_files', 's_node_pars.txt')
        self._d_file_name = os.path.join(dir_path, 'zi_parameter_files', 'd_node_pars.txt')

        init = True
        try:
            f = open(self._s_file_name).read()
            s_node_pars = json.loads(f)
        except:
            print("parameter file for gettable parameters {} not found".format(self._s_file_name))
            init=False
        try:
            f = open(self._d_file_name).read()
            d_node_pars = json.loads(f)
        except:
            print("parameter file for settable parameters {} not found".format(self._d_file_name))
            init = False

        for parameter in s_node_pars:
            parname=parameter[0].replace("/","_")
            parfunc="/"+device+"/"+parameter[0]
            if parameter[1] == 'float':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setd, parfunc),
                    get_cmd=self._gen_get_func(self.getd, parfunc),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1] == 'float_small':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setd, parfunc),
                    get_cmd=self._gen_get_func(self.getd, parfunc),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1] == 'int_8bit':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parfunc),
                    get_cmd=self._gen_get_func(self.geti, parfunc),
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            elif parameter[1]=='int':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parfunc),
                    get_cmd=self._gen_get_func(self.geti, parfunc),
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            elif parameter[1]=='int_64':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parfunc),
                    get_cmd=self._gen_get_func(self.geti, parfunc),
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            elif parameter[1]=='bool':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parfunc),
                    get_cmd=self._gen_get_func(self.geti, parfunc),
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            else:
                print("parameter {} type {} from from s_node_pars not recognized".format(parname,parameter[1]))

        for parameter in d_node_pars:
            parname=parameter[0].replace("/","_")
            parfunc="/"+device+"/"+parameter[0]
            if parameter[1]=='float':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(self.getd, parfunc))
            elif parameter[1]=='vector_g':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(self.getv, parfunc))
            elif parameter[1]=='vector_s':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setv, parfunc),
                    vals=vals.Anything())
            elif parameter[1]=='vector_gs':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setv, parfunc),
                    get_cmd=self._gen_get_func(self.getv, parfunc),
                    vals=vals.Anything())
            else:
                print("parameter {} type {} from d_node_pars not recognized".format(parname,parameter[1]))


        self.add_parameter('AWG_file',
                           set_cmd=self._do_set_AWG_file,
                           vals=vals.Anything())
        if init:
            self.load_default_settings()
        t1 = time.time()

        print('Initialized UHFQC', self._device,
              'in %.2fs' % (t1-t0))

    def load_default_settings(self):
        #standard configurations adapted from Haendbaek's notebook
        # Run this block to do some standard configuration

        # The averaging-count is used to specify how many times the AWG program should run
        LOG2_AVG_CNT = 10

        # This averaging count specifies how many measurements the result logger should average
        LOG2_RL_AVG_CNT = 0

        # Load an AWG program (from Zurich Instruments/LabOne/WebServer/awg/src)
        self.awg_sequence_acquisition()

        # Turn on both outputs
        self.sigouts_0_on(1)
        self.sigouts_1_on(1)

        # QuExpress thresholds on DIO (mode == 2), AWG control of DIO (mode == 1)
        self.dios_0_mode(2)
        # Drive DIO bits 31 to 16
        self.dios_0_drive(0xc)

        # Configure the analog trigger input 1 of the AWG to assert on a rising edge on Ref_Trigger 1 (front-panel of the instrument)
        self.awgs_0_triggers_0_rising(1)
        self.awgs_0_triggers_0_level(0.000000000)
        self.awgs_0_triggers_0_channel(2)


        # Straight connection, signal input 1 to channel 1, signal input 2 to channel 2
        self.quex_deskew_0_col_0(1.0)
        self.quex_deskew_0_col_1(0.0)
        self.quex_deskew_1_col_0(0.0)
        self.quex_deskew_1_col_1(1.0)

        self.quex_wint_delay(0)

        # Setting the clock to external
        self.system_extclk(1)

        # No rotation on the output of the weighted integration units, i.e. take real part of result
        for i in range(0, 4):
            eval('self.quex_rot_{0}_real(1.0)'.format(i))
            eval('self.quex_rot_{0}_imag(0.0)'.format(i))

        # No cross-coupling in the matrix multiplication (identity matrix)
        for i in range(0, 4):
            for j in range(0, 4):
                if i == j:
                    eval('self.quex_trans_{0}_col_{1}_real(1)'.format(i,j))
                else:
                    eval('self.quex_trans_{0}_col_{1}_real(0)'.format(i,j))

        # Configure the result logger to not do any averaging
        self.quex_rl_length(pow(2, LOG2_AVG_CNT)-1)
        self.quex_rl_avgcnt(LOG2_RL_AVG_CNT)
        self.quex_rl_source(2)

        # Ready for readout
        self.quex_iavg_readout(1)
        self.quex_rl_readout(1)

        # The custom firmware will feed through the signals on Signal Input 1 to Signal Output 1 and Signal Input 2 to Signal Output 2
        # when the AWG is OFF. For most practical applications this is not really useful. We, therefore, disable the generation of
        # these signals on the output here.
        self.sigouts_0_enables_3(0)
        self.sigouts_1_enables_7(0)



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

    def awg(self, filename):
        print(filename)
        with open(filename, 'r') as awg_file:
            sourcestring = awg_file.read()
            self.awg_string(sourcestring)

    def _do_set_AWG_file(self, filename):
        self.awg('UHFLI_AWG_sequences/'+filename)


    def awg_string(self, sourcestring):
        h = self._daq.awgModule()
        h.set('awgModule/device', self._device)
        h.set('awgModule/index', 0)
        h.execute()
        h.set('awgModule/compiler/sourcestring', sourcestring)
        h.set('awgModule/compiler/start', 1)
        h.set('awgModule/elf/file', '')


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

    def single_acquisition(self, samples, acquisition_time=0.010, timeout=0, channels=set([0, 1]), mode='rl'):
        # Define the channels to use
        paths = dict()
        data = dict()
        print('single acq')
        if mode == 'rl':
            for c in channels:
                paths[c] = '/' + self._device + '/quex/rl/data/{}'.format(c)
                data[c] = []
                self._daq.subscribe(paths[c])
                print("rl mode now")
        else:
            for c in channels:
                paths[c] = '/' + self._device + '/quex/iavg/data/{}'.format(c)
                data[c] = []
                self._daq.subscribe(paths[c])
                print("iavg mode now")


        #self._daq.setInt('/' + self._device + '/awgs/0/single', 1)
        #self._daq.setInt('/' + self._device + '/awgs/0/enable', 1)

        timeout = 0
        gotem = [False]*len(channels)
        while not all(gotem) and timeout < 100:
            dataset = self._daq.poll(acquisition_time, timeout, 4, True)
            for n, c in enumerate(channels):
                p = paths[c]
                if p in dataset:
                    for v in dataset[p]:
                        data[c] = np.concatenate((data[c], v['vector']))
                    if len(data[c]) >= samples:
                        gotem[n] = True

            timeout += 1

        if not all(gotem):
            print("Error: Didn't get all results!")
            for n, c in enumerate(channels):
                print("    : Channel {}: Got {} of {} samples", c, len(data[c]), samples)
            return (None, None)
        # print("data type {}".format(type(data)))
        return data

    def create_parameter_files(self):
        #this functions retrieves all possible settable and gettable parameters from the device.
        #Additionally, iot gets all minimum and maximum values for the parameters by trial and error

        s_node_pars=[]
        d_node_pars=[]
        patterns = ["awgs", "sigins", "sigouts", "quex", "dios","system/extclk"] #["quex/iavg", "quex/wint"]
        #json.dump([, s_file, default=int)
        #json.dump([, d_file, default=int)
        for pattern in patterns:
            print("extracting parameters of type", pattern)
            all_nodes = set(self.find('*{}*'.format(pattern)))
            s_nodes = set(self.finds('*{}*'.format(pattern)))
            d_nodes = all_nodes.difference(s_nodes)
            print(len(all_nodes))
            # extracting info from the setting nodes
            s_nodes = list(s_nodes)
            default_values=self.getd(s_nodes)
            for s_node in s_nodes:
                self.setd(s_node,  1e12)
            max_values = self.getd(s_nodes)
            for s_node in s_nodes:
                self.setd(s_node, -1e12)
            min_values = self.getd(s_nodes)
            float_values = dict.fromkeys(s_nodes)
            for s_node in s_nodes:
                if np.pi > max_values[s_node]:
                    float_values[s_node] = max_values[s_node]/np.pi;
                else:
                    float_values[s_node] = np.pi
                self.setd(s_node, float_values[s_node])
            actual_float_values = self.getd(s_nodes)

            node_types = dict.fromkeys(s_nodes)
            for s_node in sorted(s_nodes):
                #self.setd(node,default_values[s_node])
                fraction, integer = np.modf(actual_float_values[s_node])
                if fraction != 0:
                    node_types[s_node] = 'float'
                    if min_values[s_node]==max_values[s_node]:
                        node_types[s_node]='float_small'
                        min_values[s_node]=0
                    elif abs(min_values[s_node])<0.01:
                        min_values[s_node]=0
                else:
                    node_types[s_node] = 'int'
                    min_values[s_node]=0
                    if  max_values[s_node]==3567587328:
                        node_types[s_node] = 'int_64'
                        max_values[s_node]=4294967295
                    elif  max_values[s_node]==1:
                        node_types[s_node] = 'bool'
                    elif max_values[s_node]==0:
                        max_values[s_node]=255
                        node_types[s_node] = 'int_8bit'
                    elif max_values[s_node]>4294967295:
                        node_types[s_node] = 'float'

                line=[s_node.replace('/' + self._device + '/', ''), node_types[s_node], min_values[s_node], max_values[s_node]]
                print(line)
                s_node_pars.append(line)
                #json.dump(line, s_file, indent=2, default=int)


            #extracting info from the data nodes
            d_nodes = list(d_nodes)
            #default_values=self.getd(d_nodes)
            default_values=np.zeros(len(d_nodes))
            node_types = ['']*len(d_nodes)

            for i, d_node in enumerate(d_nodes):
                try:
                    answer=self.getv(d_node)
                    if isinstance(answer, dict):
                        value=answer['value'][0]
                        node_types[i]='float'
                    elif  isinstance(answer, list):
                        try:
                            self.setv(d_node,np.array([0,0,0]))
                            node_types[i]='vector_gs'
                        except:
                            value=answer[0]['vector']
                            node_types[i]='vector_g'
                    else:
                        print("unknown type")
                except:
                    node_types[i]='vector_s'
                line=[d_node.replace('/' + self._device + '/', ''), node_types[i]]#, default_values[i]]
                print(line)
                d_node_pars.append(line)
                #json.dump(line, d_file, indent=2, default=int)

        with open(self._s_file_name, 'w') as s_file:
            json.dump(s_node_pars, s_file, default=int, indent=2)

        with open(self._d_file_name, 'w') as d_file:
            json.dump(d_node_pars, d_file, default=int, indent=2)

    def prepare_SSB_weight_and_rotation(self, IF,  weight_function_I=0, weight_function_Q=1):
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.array(np.cos(2*np.pi*IF*tbase))
        sinI = np.array(np.sin(2*np.pi*IF*tbase))
        eval('self.quex_wint_weights_{}_real(np.array(cosI))'.format(weight_function_I))
        eval('self.quex_wint_weights_{}_imag(np.array(sinI))'.format(weight_function_I))
        eval('self.quex_wint_weights_{}_real(np.array(sinI))'.format(weight_function_Q))
        eval('self.quex_wint_weights_{}_real(np.array(cosI))'.format(weight_function_Q))
        eval('self.quex_rot_{}_real(1.0)'.format(weight_function_I))
        eval('self.quex_rot_{}_imag(1.0)'.format(weight_function_I))
        eval('self.quex_rot_{}_real(1.0)'.format(weight_function_Q))
        eval('self.quex_rot_{}_imag(-1.0)'.format(weight_function_Q))

    def prepare_DSB_weight_and_rotation(self, IF, weight_function_I=0, weight_function_Q=1):
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.array(np.cos(2*np.pi*IF*tbase))
        sinI = np.array(np.sin(2*np.pi*IF*tbase))
        eval('self.quex_wint_weights_{}_real(np.array(cosI))'.format(weight_function_I))
        eval('self.quex_wint_weights_{}_real(np.array(sinI))'.format(weight_function_I))
        eval('self.quex_wint_weights_{}_real(np.array(sinI))'.format(weight_function_Q))
        eval('self.quex_wint_weights_{}_real(np.array(cosI))'.format(weight_function_Q))
        eval('self.quex_rot_{}_real(1.0)'.format(weight_function_I))
        eval('self.quex_rot_{}_imag(0.0)'.format(weight_function_I))
        eval('self.quex_rot_{}_real(1.0)'.format(weight_function_Q))
        eval('self.quex_rot_{}_imag(0.0)'.format(weight_function_Q))

    def _make_full_path(self, paths):
        full_paths = []
        for p in paths:
            if p[0] == '/':
                full_paths.append(p)
            else:
                full_paths.append('/' + self._device + '/' + p)
        return full_paths

    def seti(self, path, value):
        # Handle absolute path
        if path[0] == '/':
            self._daq.setInt(path, int(value))
        else:
            self._daq.setInt('/' + self._device + '/' + path, int(value))

    def setd(self, path, value):
        # Handle absolute path
        if path[0] == '/':
            self._daq.setDouble(path, float(value))
        else:
            self._daq.setDouble('/' + self._device + '/' + path, float(value))

    def get(self, paths, convert=None):
        if type(paths) is not list:
            paths = [ paths ]
            single = 1
        else:
            single = 0

        paths = self._make_full_path(paths)
        values = {}

        for p in paths:
            self._daq.getAsEvent(p)

        while len(values) < len(paths):
            tmp = self._daq.poll(0.001, 500, 4, True)
            for p in tmp:
                if convert:
                    values[p] = convert(tmp[p]['value'][0])
                else:
                    values[p] = tmp[p]['value'][0]

        if single:
            return values[paths[0]]
        else:
            return values

    def geti(self, paths):
        return self.get(paths, int)

    def getd(self, paths):
        return self.get(paths, float)

    def getv(self, paths):
        if type(paths) is not list:
            paths = [ paths ]
            single = 1
        else:
            single = 0

        paths = self._make_full_path(paths)
        values = {}

        for p in paths:
            self._daq.getAsEvent(p)

        tries = 0
        while len(values) < len(paths) and tries < 10:
            try:
                tmp = self._daq.poll(0.001, 500, 4, True)
                for p in tmp:
                    values[p] = tmp[p]
            except ZIException:
                pass

        if single:
            return values[paths[0]]
        else:
            return values

    def setv(self, path, value):
        # Handle absolute path
        if path[0] == '/':
            self._daq.vectorWrite(path, value)
        else:
            self._daq.vectorWrite('/' + self._device + '/' + path, value)



    ## sequencer functions

    def awg_sequence_acquisition_and_pulse(self, Iwave, Qwave, acquisition_delay):
        if np.max(Iwave)>1.0 or np.min(Iwave)<-1.0:
            raise KeyError("exceeding AWG range for I channel, all values should be withing +/-1")
        elif np.max(Qwave)>1.0 or np.min(Qwave)<-1.0:
            raise KeyError("exceeding AWG range for Q channel, all values should be withing +/-1")
        elif len(Iwave)>1493:
            raise KeyError("exceeding max AWG wave lenght of 1493 samples for I channel, trying to upload {} samples".format(len(Iwave)))
        elif len(Qwave)>1493:
            raise KeyError("exceeding max AWG wave lenght of 1493 samples for Q channel, trying to upload {} samples".format(len(Qwave)))

        Iwave_strip=",".join(str(bit) for bit in Iwave)
        Qwave_strip=",".join(str(bit) for bit in Qwave)
        wave_I_string = "wave Iwave = vect("+Iwave_strip+");\n"
        wave_Q_string = "wave Qwave = vect("+Qwave_strip+");\n"

        delay_samples = int(acquisition_delay*1.8e9/8)
        delay_string='\twait({});\n'.format(delay_samples)


        preamble="""
const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x0f0000;

setTrigger(WINT_EN);
var loop_cnt = getUserReg(0);
var RO_TRIG;
if(getUserReg(1)){
  RO_TRIG=IAVG_TRIG;
}else{
  RO_TRIG=WINT_TRIG;
}\n"""

        loop_start="""
repeat(loop_cnt) {
\twaitDigTrigger(1, 0);
\twaitDigTrigger(1, 1);
\tplayWave(Iwave,Qwave);\n"""


        end_string="""
\tsetTrigger(WINT_EN +RO_TRIG);
\tsetTrigger(WINT_EN);
\twaitWave();
}
wait(300);
setTrigger(0);"""

        string = preamble+wave_I_string+wave_Q_string+loop_start+delay_string+end_string
        self.awg_string(string)

    def awg_sequence_acquisition(self):
        string="""
const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x0f0000;

setTrigger(WINT_EN);
var loop_cnt = getUserReg(0);
var RO_TRIG;
if(getUserReg(1)){
  RO_TRIG=IAVG_TRIG;
}else{
  RO_TRIG=WINT_TRIG;
}

repeat(loop_cnt) {
\twaitDigTrigger(1, 0);
\twaitDigTrigger(1, 1);\n
\tsetTrigger(WINT_EN +RO_TRIG);
\tsetTrigger(WINT_EN);
\twait(300);
}
setTrigger(0);"""
        self.awg_string(string)



    def awg_sequence_acquisition_and_pulse_SSB(self, f_RO_mod, RO_amp, RO_pulse_length, acquisition_delay):
        f_sampling=1.8e9
        samples=RO_pulse_length*f_sampling
        array=np.arange(int(samples))
        sinwave=RO_amp*np.sin(2*np.pi*array*f_RO_mod/f_sampling)
        coswave=RO_amp*np.cos(2*np.pi*array*f_RO_mod/f_sampling)
        Iwave = coswave+sinwave;
        Qwave = coswave-sinwave;
        # Iwave, Qwave = PG.mod_pulse(np.ones(samples), np.zeros(samples), f=f_RO_mod, phase=0, sampling_rate=f_sampling)
        self.awg_sequence_acquisition_and_pulse(Iwave, Qwave, acquisition_delay)


    def upload_transformation_matrix(self, matrix):
        for i in range(np.shape(matrix)[0]): #looping over the rows
            for j in range(np.shape(matrix)[1]): #looping over the colums
                #value =matrix[i,j]
                #print(value)
                eval('self.quex_trans_{}_col_{}_real(matrix[{}][{}])'.format(j,i,i,j))

    def download_transformation_matrix(self, nr_rows=4, nr_cols=4):
        matrix = np.zeros([nr_rows,nr_cols])
        for i in range(np.shape(matrix)[0]): #looping over the rows
            for j in range(np.shape(matrix)[1]): #looping over the colums
                matrix[i][j]=(eval('self.quex_trans_{}_col_{}_real()'.format(j,i)))
                #print(value)
                #matrix[i,j]=value
        return matrix