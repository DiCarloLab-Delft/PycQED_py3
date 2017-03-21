import zhinst.ziPython as zi
import zhinst.utils as zi_utils
import time
import json
import os
import sys
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from fnmatch import fnmatch
from qcodes.instrument.parameter import ManualParameter
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

    def __init__(self, name, device='auto', interface='USB', address='127.0.0.1', port=8004, **kw):
        '''
        Input arguments:
            name:           (str) name of the instrument
            server_name:    (str) qcodes instrument server
            address:        (int) the address of the data server e.g. 8006
        '''
        #self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) #suggestion W vlothuizen
        t0 = time.time()
        super().__init__(name, **kw)

        self._daq = zi.ziDAQServer(address, int(port), 5)
        if device.lower() == 'auto':
            self._device = zi_utils.autoDetect(self._daq)
        else:
            self._device = device
            self._daq.connectDevice(self._device, interface)
        #self._device = zi_utils.autoDetect(self._daq)
        self._awgModule = self._daq.awgModule()
        self._awgModule.set('awgModule/device', self._device)
        self._awgModule.execute()

        self.acquisition_paths = []

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
        #storing an offset correction parameter for all weight functions,
        #this allows normalized calibration when performing cross-talk suppressed
        #readout
        for i in range(5):
            self.add_parameter("quex_trans_offset_weightfunction_{}".format(i),
                   unit='V',
                   label='RO normalization offset (V)',
                   initial_value=0.0,
                   parameter_class=ManualParameter)
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

        # Configure the digital trigger to be a rising-edge trigger
        self.awgs_0_auxtriggers_0_slope(1);
        # Straight connection, signal input 1 to channel 1, signal input 2 to channel 2
        self.quex_deskew_0_col_0(1.0)
        self.quex_deskew_0_col_1(0.0)
        self.quex_deskew_1_col_0(0.0)
        self.quex_deskew_1_col_1(1.0)

        self.quex_wint_delay(0)

        # Setting the clock to external
        self.system_extclk(1)

        # No rotation on the output of the weighted integration unit, i.e. take real part of result
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

        # Ready for readout. Writing a '1' to these nodes activates the automatic readout of results.
        # This functionality should be used once the ziPython driver has been improved to handle
        # the 'poll' commands of these results correctly. Until then, we write a '0' to the nodes
        # to prevent automatic result readout. It is then necessary to poll e.g. the AWG in order to
        # detect when the measurement is complete, and then manually fetch the results using the 'get'
        # command. Disabling the automatic result readout speeds up the operation a bit, since we avoid
        # sending the same data twice.
        self.quex_iavg_readout(0)
        self.quex_rl_readout(0)




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
        """
        Loads an awg sequence onto the UHFQC from a text file.
        File needs to obey formatting specified in the manual.
        """
        print(filename)
        with open(filename, 'r') as awg_file:
            sourcestring = awg_file.read()
            self.awg_string(sourcestring)

    def _do_set_AWG_file(self, filename):
        self.awg('UHFLI_AWG_sequences/'+filename)

    def awg_file(self, filename):
        self._awgModule.set('awgModule/compiler/sourcefile', filename)
        self._awgModule.set('awgModule/compiler/start', 1)
        #self._awgModule.set('awgModule/elf/file', '')
        while self._awgModule.get('awgModule/progress')['progress'][0] < 1.0:
            time.sleep(0.1)
        print(self._awgModule.get('awgModule/compiler/statusstring')['compiler']['statusstring'][0])
        self._daq.sync()

    def awg_string(self, sourcestring):
        path = '/' + self._device + '/awgs/0/ready'
        self._daq.subscribe(path)
        self._awgModule.set('awgModule/compiler/sourcestring', sourcestring)
        #self._awgModule.set('awgModule/elf/file', '')
        while self._awgModule.get('awgModule/progress')['progress'][0] < 1.0:
            time.sleep(0.1)
        print(self._awgModule.get('awgModule/compiler/statusstring')['compiler']['statusstring'][0])
        while self._awgModule.get('awgModule/progress')['progress'][0] < 1.0:
            time.sleep(0.01)

        ready = False
        timeout = 0
        while not ready and timeout < 1.0:
            data = self._daq.poll(0.1, 1, 4, True)
            timeout += 0.1
            if path in data:
                if data[path]['value'][-1] == 1:
                    ready = True
        self._daq.unsubscribe(path)

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

    def sync(self):
        self._daq.sync()

    def acquisition_arm(self):
        # time.sleep(0.01)
        self._daq.asyncSetInt('/' + self._device + '/awgs/0/single', 1)
        self._daq.syncSetInt('/' + self._device + '/awgs/0/enable', 1)
        # t0=time.time()
        # time.sleep(0.001)
        #self._daq.sync()
        # deltat=time.time()-t0
        # print('UHFQC syncing took {}'.format(deltat))


    def acquisition_get(self, samples, acquisition_time=0.010, timeout=0, channels=set([0, 1]), mode='rl'):
        # Define the channels to use
        paths = dict()
        data = dict()
        if mode == 'rl':
            for c in channels:
                paths[c] = '/' + self._device + '/quex/rl/data/{}'.format(c)
                data[c] = []
                self._daq.subscribe(paths[c])
        else:
            for c in channels:
                paths[c] = '/' + self._device + '/quex/iavg/data/{}'.format(c)
                data[c] = []

        # Disable automatic readout
        self._daq.setInt('/' + self._device + '/quex/rl/readout', 0)
        # It would be better to move this call in to the initialization function
        # in order to save time here
        enable_path = '/' + self._device + '/awgs/0/enable'
        self._daq.subscribe(enable_path)

        # Added for testing purposes, remove again according to how the AWG is started
        self._daq.setInt('/' + self._device + '/awgs/0/single', 1)
        self._daq.setInt(enable_path, 1)

        # Wait for the AWG to finish
        gotit = False
        accumulated_time = 0
        while not gotit and accumulated_time < timeout:
            dataset = self._daq.poll(acquisition_time, 1, 4, True)
            if enable_path in dataset and dataset[enable_path]['value'][0] == 0:
                gotit = True
            else:
                accumulated_time += acquisition_time

        if not gotit:
            print("Error: AWG did not finish in time!")
            return None

        # Acquire data
        gotem = [False]*len(channels)
        for n, c in enumerate(channels):
            p = paths[c]
            dataset = self._daq.get(p, True, 0)
            if p in dataset:
                for v in dataset[p]:
                    data[c] = np.concatenate((data[c], v['vector']))
                if len(data[c]) >= samples:
                    gotem[n] = True

        if not all(gotem):
            print("Error: Didn't get all results!")
            for n, c in enumerate(channels):
                print("    : Channel {}: Got {} of {} samples", c, len(data[c]), samples)
            return None

        # print("data type {}".format(type(data)))
        return data

    def acquisition_poll(self, samples, arm=True,
                                acquisition_time=0.010, timeout=1.0):
        """
        Polls the UHFQC for data.

        Args:
            samples (int): the expected number of samples
            arm    (bool): if true arms the acquisition, disable when you
                           need synchronous acquisition with some external dev
            acquisition_time (float): time in sec between polls? # TODO check with Niels H
            timeout (float): time in unknown units before timeout Error is raised.

        """
        data = dict()

        # Start acquisition
        if arm:
            self.acquisition_arm()

        # Acquire data
        gotem = [False]*len(self.acquisition_paths)
        accumulated_time = 0

        while accumulated_time < timeout and not all(gotem):
            dataset = self._daq.poll(acquisition_time, 1, 4, True)
            #print(dataset)
            for n, p in enumerate(self.acquisition_paths):
                if p in dataset:
                    for v in dataset[p]:
                        if n in data:
                            data[n] = np.concatenate((data[n], v['vector']))
                        else:
                            data[n] = v['vector']
                        if len(data[n]) >= samples:
                            gotem[n] = True
            accumulated_time += acquisition_time

        if not all(gotem):
            self.acquisition_finalize()
            for n, c in enumerate(self.acquisition_paths):
                if n in data:
                    print("    : Channel {}: Got {} of {} samples", n, len(data[n]), samples)
            raise TimeoutError("Error: Didn't get all results!")

        return data

    def acquisition(self, samples, acquisition_time=0.010, timeout=0, channels=set([0, 1]), mode='rl'):
        self.acquisition_initialize(channels, mode)
        data = self.acquisition_poll(samples, acquisition_time, timeout)
        self.acquisition_finalize()

        return data

    def acquisition_initialize(self, channels=set([0, 1]), mode='rl'):
        # Define the channels to use
        self.acquisition_paths = []

        if mode == 'rl':
            for c in channels:
                self.acquisition_paths.append('/' + self._device + '/quex/rl/data/{}'.format(c))
            self._daq.subscribe('/' + self._device + '/quex/rl/data/*')
            # Enable automatic readout
            self._daq.setInt('/' + self._device + '/quex/rl/readout', 1)
        else:
            for c in channels:
                self.acquisition_paths.append('/' + self._device + '/quex/iavg/data/{}'.format(c))
            self._daq.subscribe('/' + self._device + '/quex/iavg/data/*')
            # Enable automatic readout
            self._daq.setInt('/' + self._device + '/quex/iavg/readout', 1)

        self._daq.subscribe('/' + self._device + '/auxins/0/sample')

        # Generate more dummy data
        self._daq.setInt('/' + self._device + '/auxins/0/averaging', 8);

    def acquisition_finalize(self):
        for p in self.acquisition_paths:
            self._daq.unsubscribe(p)
        self._daq.unsubscribe('/' + self._device + '/auxins/0/sample')

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

    def prepare_SSB_weight_and_rotation(self, IF,
                                        weight_function_I=0,
                                        weight_function_Q=1):
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.array(np.cos(2*np.pi*IF*tbase))
        sinI = np.array(np.sin(2*np.pi*IF*tbase))
        eval('self.quex_wint_weights_{}_real(np.array(cosI))'.format(weight_function_I))
        eval('self.quex_wint_weights_{}_imag(np.array(sinI))'.format(weight_function_I))
        eval('self.quex_wint_weights_{}_real(np.array(sinI))'.format(weight_function_Q))
        eval('self.quex_wint_weights_{}_imag(np.array(cosI))'.format(weight_function_Q))
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

    def _make_full_path(self, path):
        if path[0] == '/':
            return path
        else:
            return '/' + self._device + '/' + path

    def seti(self, path, value, async=False):
        if async:
            func = self._daq.asyncSetInt
        else:
            func = self._daq.setInt

        func(self._make_full_path(path), int(value))

    def setd(self, path, value, async=False):
        if async:
            func = self._daq.asyncSetDouble
        else:
            func = self._daq.setDouble

        func(self._make_full_path(path), float(value))

    def _get(self, paths, convert=None):
        if type(paths) is not list:
            paths = [ paths ]
            single = 1
        else:
            single = 0

        values = {}

        for p in paths:
            values[p] = convert(self._daq.getDouble(self._make_full_path(p)))

        if single:
            return values[paths[0]]
        else:
            return values

    def geti(self, paths):
        return self._get(paths, int)

    def getd(self, paths):
        return self._get(paths, float)

    def getv(self, paths):
        if type(paths) is not list:
            paths = [ paths ]
            single = 1
        else:
            single = 0

        paths = [self._make_full_path(p) for p in paths]
        values = {}

        for p in paths:
            timeout = 0
            while p not in values and timeout < 5:
                try:
                    tmp = self._daq.get(p, True, 0)
                    values[p] = tmp[p]
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    timeout += 1

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
        elif len(Iwave)>16384:
            raise KeyError("exceeding max AWG wave lenght of 16384 samples for I channel, trying to upload {} samples".format(len(Iwave)))
        elif len(Qwave)>16384:
            raise KeyError("exceeding max AWG wave lenght of 16384 samples for Q channel, trying to upload {} samples".format(len(Qwave)))

        wave_I_string = self.array_to_combined_vector_string(Iwave, "Iwave")
        wave_Q_string = self.array_to_combined_vector_string(Qwave, "Qwave")
        delay_samples = int(acquisition_delay*1.8e9/8)
        delay_string = '\twait(getUserReg(2));\n'
        self.awgs_0_userregs_2(delay_samples)

        preamble="""
const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x1f0000;
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

    def array_to_combined_vector_string(self, array, name):
        # this function cuts up arrays into several vectors of maximum length 1024 that are joined.
        # this is to avoid python crashes (was found to crash for vectors of lenght> 1490)
        string = 'vect('
        join = False
        n = 0
        while n < len(array):
            string += '{:.3f}'.format(array[n])
            if ((n+1) % 1024 != 0) and n < len(array)-1:
                string += ','

            if ((n+1) % 1024 == 0):
                string += ')'
                if n < len(array)-1:
                    string += ',\nvect('
                    join = True
            n += 1

        string += ')'
        if join:
            string = 'wave '+ name +' = join(' + string + ');\n'
        else:
            string = 'wave '+ name +' = '+ string + ';\n'
        return string

    def awg_sequence_acquisition(self):
        string="""
const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x1f0000;
setTrigger(WINT_EN);
var loop_cnt = getUserReg(0);
var RO_TRIG;
if(getUserReg(1)){
  RO_TRIG=IAVG_TRIG;
}else{
  RO_TRIG=WINT_TRIG;
}
repeat(loop_cnt) {
\twaitDigTrigger(1, 1);\n
\tsetTrigger(WINT_EN +RO_TRIG);
\twait(5);
\tsetTrigger(WINT_EN);
\twait(300);
}
wait(1000);
setTrigger(0);"""
        self.awg_string(string)


    def awg_update_waveform(self, index, data):
        self.awgs_0_waveform_index(index)
        self.awgs_0_waveform_data(data)
        self._daq.sync()

    def awg_sequence_acquisition_and_pulse_SSB(self, f_RO_mod, RO_amp, RO_pulse_length, acquisition_delay):
        f_sampling=1.8e9
        samples = RO_pulse_length*f_sampling
        array = np.arange(int(samples))
        sinwave = RO_amp*np.sin(2*np.pi*array*f_RO_mod/f_sampling)
        coswave = RO_amp*np.cos(2*np.pi*array*f_RO_mod/f_sampling)
        Iwave = (coswave+sinwave)/np.sqrt(2)
        Qwave = (coswave-sinwave)/np.sqrt(2)
        # Iwave, Qwave = PG.mod_pulse(np.ones(samples), np.zeros(samples), f=f_RO_mod, phase=0, sampling_rate=f_sampling)
        self.awg_sequence_acquisition_and_pulse(Iwave, Qwave, acquisition_delay)


    def upload_transformation_matrix(self, matrix):
        for i in range(np.shape(matrix)[0]): #looping over the rows
            for j in range(np.shape(matrix)[1]): #looping over the colums
                #value =matrix[i,j]
                #print(value)
                eval('self.quex_trans_{}_col_{}_real(matrix[{}][{}])'.format(j,i,i,j))

    def download_transformation_matrix(self, nr_rows=4, nr_cols=4):
        matrix = np.zeros([nr_rows, nr_cols])
        for i in range(np.shape(matrix)[0]): #looping over the rows
            for j in range(np.shape(matrix)[1]): #looping over the colums
                matrix[i][j] = (eval('self.quex_trans_{}_col_{}_real()'.format(j,i)))
                #print(value)
                #matrix[i,j]=value
        return matrix