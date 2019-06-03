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
import ctypes
from ctypes.wintypes import MAX_PATH


class UHFQC(Instrument):

    """
    This is the qcodes driver for the 1.8 Gsample/s UHF-QC developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5/3.6 ucs4 16.04 for 64bit Windows from
        http://www.zhinst.com/downloads, https://people.zhinst.com/~niels/
    2. pip install dependencies: httplib2, plotly, pyqtgraph
    3. upload the latest firmware to the UHFQC by opening reboot.bat in
    'Transmon\Inventory\ZurichInstruments\firmware_UHFLI\firmware_x\reboot_dev'.
        With x the highest available number and dev the device number.
    4. find out where sequences are stored by saving a sequence from the
        GUI and then check :"showLog" to see where it is stored. This is the
        location where AWG sequences can be loaded from.
    misc: when device crashes, check the log file in
    EOM
    """

    def __init__(self, name, device='auto', interface='USB',
                 address='127.0.0.1', port=8004, DIO=True,
                 nr_integration_channels=9, **kw):
        '''
        Input arguments:
            name:           (str) name of the instrument
            server_name:    (str) qcodes instrument server
            address:        (int) the address of the data server e.g. 8006
        '''
        # self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # #suggestion W vlothuizen
        t0 = time.time()
        super().__init__(name, **kw)
        self.nr_integration_channels = nr_integration_channels
        self.DIO = DIO
        self._daq = zi.ziDAQServer(address, int(port), 5)
        # self._daq.setDebugLevel(5)
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

        s_node_pars = []
        d_node_pars = []

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self._s_file_name = os.path.join(
            dir_path, 'zi_parameter_files', 's_node_pars.txt')
        self._d_file_name = os.path.join(
            dir_path, 'zi_parameter_files', 'd_node_pars.txt')

        init = True
        try:
            with open(self._s_file_name) as f:
                s_node_pars = json.loads(f.read())
        except:
            print("parameter file for gettable parameters {} not found".format(
                self._s_file_name))
            init = False
        try:
            with open(self._d_file_name) as f:
                d_node_pars = json.loads(f.read())
        except:
            print("parameter file for settable parameters {} not found".format(
                self._d_file_name))
            init = False

        self.add_parameter('timeout', unit='s',
                           initial_value=30,
                           parameter_class=ManualParameter)
        for parameter in s_node_pars:
            parname = parameter[0].replace("/", "_")
            parfunc = "/"+self._device+"/"+parameter[0]
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
            elif parameter[1] == 'int':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parfunc),
                    get_cmd=self._gen_get_func(self.geti, parfunc),
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            elif parameter[1] == 'int_64':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parfunc),
                    get_cmd=self._gen_get_func(self.geti, parfunc),
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            elif parameter[1] == 'bool':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.seti, parfunc),
                    get_cmd=self._gen_get_func(self.geti, parfunc),
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            else:
                print("parameter {} type {} from from s_node_pars not recognized".format(
                    parname, parameter[1]))

        for parameter in d_node_pars:
            parname = parameter[0].replace("/", "_")
            parfunc = "/"+self._device+"/"+parameter[0]
            if parameter[1] == 'float':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(self.getd, parfunc))
            elif parameter[1] == 'vector_g':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(self.getv, parfunc))
            elif parameter[1] == 'vector_s':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setv, parfunc),
                    vals=vals.Anything())
            elif parameter[1] == 'vector_gs':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(self.setv, parfunc),
                    get_cmd=self._gen_get_func(self.getv, parfunc),
                    vals=vals.Anything())
            else:
                print("parameter {} type {} from d_node_pars not recognized".format(
                    parname, parameter[1]))

        self.add_parameter('AWG_file',
                           set_cmd=self._do_set_AWG_file,
                           vals=vals.Anything())
        # storing an offset correction parameter for all weight functions,
        # this allows normalized calibration when performing cross-talk suppressed
        # readout
        for i in range(self.nr_integration_channels):
            self.add_parameter("quex_trans_offset_weightfunction_{}".format(i),
                               unit='',  # unit is adc value
                               label='RO normalization offset',
                               initial_value=0.0,
                               parameter_class=ManualParameter)
        if init:
            self.load_default_settings()
        t1 = time.time()

        print('Initialized UHFQC', self._device,
              'in %.2fs' % (t1-t0))
        
        if os.name == 'nt':
            dll = ctypes.windll.shell32
            buf = ctypes.create_unicode_buffer(MAX_PATH + 1)
            if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
                _basedir = buf.value
            else:
                logging.warning('Could not extract my documents folder')
        else:
            _basedir = os.path.expanduser('~')
        self.lab_one_webserver_path = os.path.join(
            _basedir, 'Zurich Instruments', 'LabOne', 'WebServer')

    def load_default_settings(self):
        # standard configurations adapted from Haendbaek's notebook
        # Run this block to do some standard configuration

        # The averaging-count is used to specify how many times the AWG program
        # should run
        LOG2_AVG_CNT = 10

        # This averaging count specifies how many measurements the result
        # logger should average
        LOG2_RL_AVG_CNT = 0

        # Load an AWG program (from Zurich
        # Instruments/LabOne/WebServer/awg/src)
        self.awg_sequence_acquisition()

        # Turn on both outputs
        self.sigouts_0_on(1)
        self.sigouts_1_on(1)

        # QuExpress thresholds on DIO (mode == 2), AWG control of DIO (mode ==
        # 1)
        self.dios_0_mode(2)
        # Drive DIO bits 15 to 0
        self.dios_0_drive(0x3)

        # Configure the analog trigger input 1 of the AWG to assert on a rising
        # edge on Ref_Trigger 1 (front-panel of the instrument)
        self.awgs_0_triggers_0_rising(1)
        self.awgs_0_triggers_0_level(0.000000000)
        self.awgs_0_triggers_0_channel(2)

        # Configure the digital trigger to be a rising-edge trigger
        self.awgs_0_auxtriggers_0_slope(1)
        # Straight connection, signal input 1 to channel 1, signal input 2 to
        # channel 2

        self.quex_deskew_0_col_0(1.0)
        self.quex_deskew_0_col_1(0.0)
        self.quex_deskew_1_col_0(0.0)
        self.quex_deskew_1_col_1(1.0)

        self.quex_wint_delay(0)

        # Setting the clock to external
        self.system_extclk(1)

        # Configure the codeword protocol
        if self.DIO:
            self.awgs_0_dio_strobe_index(31)
            self.awgs_0_dio_strobe_slope(1)  # rising edge
            self.awgs_0_dio_valid_index(16)
            self.awgs_0_dio_valid_polarity(2)  # high polarity

        # setting the output channels to 50 ohm
        self.sigouts_0_imp50(True)
        self.sigouts_1_imp50(True)

        # We probably need to adjust some delays here...
        # self.awgs_0_dio_delay_index(31)
        # self.awgs_0_dio_delay_value(1)

        # No rotation on the output of the weighted integration unit, i.e. take
        # real part of result
        for i in range(0, self.nr_integration_channels):
            self.set('quex_rot_{0}_real'.format(i), 1.0)
            self.set('quex_rot_{0}_imag'.format(i), 0.0)
            # remove offsets to weight function
            self.set('quex_trans_offset_weightfunction_{}'.format(i), 0.0)

        # No thresholding or correlation modes
        for i in range(0, self.nr_integration_channels):
            eval('self.quex_thres_{0}_level(0)'.format(i))
            eval('self.quex_corr_{0}_mode(0)'.format(i))
            eval('self.quex_corr_{0}_source(0)'.format(i))

        # No cross-coupling in the matrix multiplication (identity matrix)
        for i in range(0, self.nr_integration_channels):
            for j in range(0, self.nr_integration_channels):
                if i == j:
                    self.set('quex_trans_{0}_col_{1}_real'.format(i, j), 1)
                else:
                    self.set('quex_trans_{0}_col_{1}_real'.format(i, j), 0)

        # Configure the result logger to not do any averaging
        self.quex_rl_length(pow(2, LOG2_AVG_CNT)-1)
        self.quex_rl_avgcnt(LOG2_RL_AVG_CNT)
        self.quex_rl_source(2)

        # Ready for readout. Writing a '1' to these nodes activates the
        # automatic readout of results. This functionality should be used once
        # the ziPython driver has been improved to handle the 'poll' commands
        # of these results correctly. Until then, we write a '0' to the nodes
        # to prevent automatic result readout. It is then necessary to poll
        # e.g. the AWG in order to detect when the measurement is complete, and
        # then manually fetch the results using the 'get' command. Disabling
        # the automatic result readout speeds up the operation a bit, since we
        # avoid sending the same data twice.
        self.quex_iavg_readout(0)
        self.quex_rl_readout(0)

        # The custom firmware will feed through the signals on Signal Input 1
        # to Signal Output 1 and Signal Input 2 to Signal Output 2 when the AWG
        # is OFF. For most practical applications this is not really useful.
        # We, therefore, disable the generation of these signals on the output
        # here.
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

    def clock_freq(self):
        return 1.8e9/(2**self.awgs_0_time())

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
    
    def _write_csv_waveform(self, wf_name: str, waveform):
        filename = os.path.join(
            self.lab_one_webserver_path, 'awg', 'waves',
            self._device+'_'+wf_name+'.csv')
        # with open(filename, 'w') as f:
        np.savetxt(filename, waveform, delimiter=",")

    def _do_set_AWG_file(self, filename):
        self.awg('UHFLI_AWG_sequences/'+filename)

    def awg_file(self, filename):
        self._awgModule.set('awgModule/compiler/sourcefile', filename)
        self._awgModule.set('awgModule/compiler/start', 1)
        #self._awgModule.set('awgModule/elf/file', '')
        while self._awgModule.get('awgModule/progress')['progress'][0] < 1.0:
            time.sleep(0.1)
        print(self._awgModule.get('awgModule/compiler/statusstring')
              ['compiler']['statusstring'][0])
        self._daq.sync()

    def awg_string(self, program_string: str, timeout: float=5):
        t0 = time.time()
        awg_nr = 0  # hardcoded for UHFQC
        print('Configuring AWG of {}'.format(self.name))
        if not self._awgModule:
            raise(ziShellModuleError())

        self._awgModule.set('awgModule/index', awg_nr)
        self._awgModule.set('awgModule/compiler/sourcestring', program_string)

        t0 = time.time()

        succes_msg = 'File successfully uploaded'
        # Success is set to False when either a timeout or a bad compilation
        # message is encountered.
        success = True
        # while ("compilation not completed"):
        while len(self._awgModule.get('awgModule/compiler/sourcestring')
                  ['compiler']['sourcestring'][0]) > 0:
            time.sleep(0.01)
            comp_msg = (self._awgModule.get(
                'awgModule/compiler/statusstring')['compiler']
                ['statusstring'][0])
            if (time.time()-t0 >= timeout):
                success = False
                print('Timeout encountered during compilation.')
                break
            time.sleep(0.01)

        comp_msg = (self._awgModule.get(
            'awgModule/compiler/statusstring')['compiler']
            ['statusstring'][0])

        if not comp_msg.endswith(succes_msg):
            success = False

        if not success:
            # Printing is disabled because we put the waveform in the program
            # this should be changed when .csv waveforms are supported for UHFQC
            print("Compilation failed, printing program:")
            for i, line in enumerate(program_string.splitlines()):
                print(i+1, '\t', line)
            print('\n')
            raise ziShellCompilationError(comp_msg)
            print("Possible error:", comp)
            pass
        # If succesful the comipilation success message is printed
        t1 = time.time()
        print(self._awgModule.get('awgModule/compiler/statusstring')
              ['compiler']['statusstring'][0] + ' in {:.2f}s'.format(t1-t0))

        # path = '/' + self._device + '/awgs/0/ready'
        # self._daq.subscribe(path)
        # self._awgModule.set('awgModule/compiler/sourcestring', program_string)
        # #self._awgModule.set('awgModule/elf/file', '')
        # while self._awgModule.get('awgModule/progress')['progress'][0] < 1.0:
        #     time.sleep(0.1)
        # print(self._awgModule.get('awgModule/compiler/statusstring')
        #       ['compiler']['statusstring'][0])
        # while self._awgModule.get('awgModule/progress')['progress'][0] < 1.0:
        #     time.sleep(0.01)
        # ready = False
        # timeout = 0
        # while not ready and timeout < 1.0:
        #     data = self._daq.poll(0.1, 1, 4, True)
        #     timeout += 0.1
        #     if path in data:
        #         if data[path]['value'][-1] == 1:
        #             ready = True
        # self._daq.unsubscribe(path)

    def close(self):
        self._daq.disconnectDevice(self._device)
        super().close()

    def find(self, *args):
        nodes = self._daq.listNodes('/', 7)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower()
                         for k in nodes if fnmatch(k.lower(), m.lower())]

        return nodes

    def finds(self, *args):
        nodes = self._daq.listNodes('/', 15)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower()
                         for k in nodes if fnmatch(k.lower(), m.lower())]
        nodes = [k.lower() for k in nodes if 'weights' not in k.lower()]
        nodes = [k.lower() for k in nodes if 'statemap' not in k.lower()]
        return nodes

    def sync(self):
        self._daq.sync()

    def acquisition_arm(self, single=True):
        # time.sleep(0.01)
        self._daq.asyncSetInt('/' + self._device + '/awgs/0/single', single)
        self._daq.syncSetInt('/' + self._device + '/awgs/0/enable', 1)
        # t0=time.time()
        # time.sleep(0.001)
        # self._daq.sync()
        # deltat=time.time()-t0
        # print('UHFQC syncing took {}'.format(deltat))

    def start(self):
        """Tektronix-style start command"""
        self._daq.syncSetInt('/' + self._device + '/awgs/0/enable', 1)

    def stop(self):
        """Tektronix-style stop command"""
        self._daq.syncSetInt('/' + self._device + '/awgs/0/enable', 0)

    def acquisition_get(self, samples, acquisition_time=0.010,
                        timeout=0, channels=set([0, 1]), mode='rl'):
        logging.warning(
            "acquisition_get is deprecated (Nov 2017). Dont' use it!")
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

        # Added for testing purposes, remove again according to how the AWG is
        # started
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
                print("    : Channel {}: Got {} of {} samples",
                      c, len(data[c]), samples)
            return None

        # print("data type {}".format(type(data)))
        return data

    def acquisition_poll(self, samples, arm=True,
                         acquisition_time=0.010):
        """
        Polls the UHFQC for data.

        Args:
            samples (int): the expected number of samples
            arm    (bool): if true arms the acquisition, disable when you
                           need synchronous acquisition with some external dev
            acquisition_time (float): time in sec between polls? # TODO check with Niels H
            timeout (float): time in seconds before timeout Error is raised.

        """
        data = {k: [] for k, dummy in enumerate(self.acquisition_paths)}

        # Start acquisition
        if arm:
            self.acquisition_arm()

        # Acquire data
        gotem = [False]*len(self.acquisition_paths)
        accumulated_time = 0

        while accumulated_time < self.timeout() and not all(gotem):
            dataset = self._daq.poll(acquisition_time, 1, 4, True)

            for n, p in enumerate(self.acquisition_paths):
                if p in dataset:
                    for v in dataset[p]:
                        data[n] = np.concatenate((data[n], v['vector']))
                        if len(data[n]) >= samples:
                            gotem[n] = True
            accumulated_time += acquisition_time

        if not all(gotem):
            self.acquisition_finalize()
            for n, c in enumerate(self.acquisition_paths):
                if n in data:
                    print("\t: Channel {}: Got {} of {} samples".format(
                          n, len(data[n]), samples))
            raise TimeoutError("Error: Didn't get all results!")

        return data

    def acquisition(self, samples, acquisition_time=0.010, timeout=0,
                    channels=(0, 1), mode='rl'):
        self.timeout(timeout)
        self.acquisition_initialize(channels, mode)
        data = self.acquisition_poll(samples, True, acquisition_time)
        self.acquisition_finalize()

        return data

    def acquisition_initialize(self, channels=(0, 1), mode='rl'):
        # Define the channels to use and subscribe to them
        self.acquisition_paths = []

        if mode == 'rl':
            readout = 0
            for c in channels:
                self.acquisition_paths.append(
                    '/' + self._device + '/quex/rl/data/{}'.format(c))
                readout += (1 << c)
            self._daq.subscribe('/' + self._device + '/quex/rl/data/*')
            # Enable automatic readout
            self._daq.setInt('/' + self._device + '/quex/rl/readout', readout)
        else:
            for c in channels:
                self.acquisition_paths.append(
                    '/' + self._device + '/quex/iavg/data/{}'.format(c))
            self._daq.subscribe('/' + self._device + '/quex/iavg/data/*')
            # Enable automatic readout
            self._daq.setInt('/' + self._device + '/quex/iavg/readout', 1)

        self._daq.subscribe('/' + self._device + '/auxins/0/sample')

        # Generate more dummy data
        self._daq.setInt('/' + self._device + '/auxins/0/averaging', 8)

    def acquisition_finalize(self):
        for p in self.acquisition_paths:
            self._daq.unsubscribe(p)
        self._daq.unsubscribe('/' + self._device + '/auxins/0/sample')

    def create_parameter_files(self):
        # this functions retrieves all possible settable and gettable
        # parameters from the device. Additionally, iot gets all minimum and
        # maximum values for the parameters by trial and error

        s_node_pars = []
        d_node_pars = []
        # ["quex/iavg", "quex/wint"]
        patterns = [
            "awgs", "sigins", "sigouts", "quex", "dios", "system/extclk",
            'triggers/in']
        # json.dump([, s_file, default=int)
        # json.dump([, d_file, default=int)
        for pattern in patterns:
            print("extracting parameters of type", pattern)
            all_nodes = set(self.find('/{}/*{}*'.format(self._device, pattern)))
            s_nodes = set(self.finds('/{}/*{}*'.format(self._device, pattern)))
            d_nodes = all_nodes.difference(s_nodes)
            print(len(all_nodes))
            # extracting info from the setting nodes
            s_nodes = list(s_nodes)
            default_values = self.getd(s_nodes)
            for s_node in s_nodes:
                self.setd(s_node,  1e12)
            max_values = self.getd(s_nodes)
            for s_node in s_nodes:
                self.setd(s_node, -1e12)
            min_values = self.getd(s_nodes)
            float_values = dict.fromkeys(s_nodes)
            for s_node in s_nodes:
                if np.pi > max_values[s_node]:
                    float_values[s_node] = max_values[s_node]/np.pi
                else:
                    float_values[s_node] = np.pi
                self.setd(s_node, float_values[s_node])
            actual_float_values = self.getd(s_nodes)

            node_types = dict.fromkeys(s_nodes)
            for s_node in sorted(s_nodes):
                # self.setd(node,default_values[s_node])
                fraction, integer = np.modf(actual_float_values[s_node])
                if fraction != 0:
                    node_types[s_node] = 'float'
                    if min_values[s_node] == max_values[s_node]:
                        node_types[s_node] = 'float_small'
                        min_values[s_node] = 0
                    elif abs(min_values[s_node]) < 0.01:
                        min_values[s_node] = 0
                else:
                    node_types[s_node] = 'int'
                    min_values[s_node] = 0
                    if max_values[s_node] == 3567587328:
                        node_types[s_node] = 'int_64'
                        max_values[s_node] = 4294967295
                    elif max_values[s_node] == 1:
                        node_types[s_node] = 'bool'
                    elif max_values[s_node] == 0:
                        max_values[s_node] = 255
                        node_types[s_node] = 'int_8bit'
                    elif max_values[s_node] > 4294967295:
                        node_types[s_node] = 'float'

                line = [s_node.replace(
                    '/' + self._device + '/', ''), node_types[s_node], min_values[s_node], max_values[s_node]]
                print(line)
                s_node_pars.append(line)
                #json.dump(line, s_file, indent=2, default=int)

            # extracting info from the data nodes
            d_nodes = list(d_nodes)
            # default_values=self.getd(d_nodes)
            default_values = np.zeros(len(d_nodes))
            node_types = ['']*len(d_nodes)

            for i, d_node in enumerate(d_nodes):
                try:
                    answer = self.getv(d_node)
                    if isinstance(answer, dict):
                        value = answer['value'][0]
                        node_types[i] = 'float'
                    elif isinstance(answer, list):
                        try:
                            self.setv(d_node, np.array([0, 0, 0]))
                            node_types[i] = 'vector_gs'
                        except:
                            value = answer[0]['vector']
                            node_types[i] = 'vector_g'
                    else:
                        print("unknown type")
                except:
                    node_types[i] = 'vector_s'
                # , default_values[i]]
                line = [
                    d_node.replace('/' + self._device + '/', ''), node_types[i]]
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
        """
        Sets defualt integration weights for SSB modulation, beware does not
        load pulses or prepare the UFHQC progarm to do data acquisition
        """
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        print(len(tbase))
        cosI = np.array(np.cos(2*np.pi*IF*tbase))
        sinI = np.array(np.sin(2*np.pi*IF*tbase))
        self.set('quex_wint_weights_{}_real'.format(weight_function_I),
                 np.array(cosI))
        self.set('quex_wint_weights_{}_imag'.format(weight_function_I),
                 np.array(sinI))
        self.set('quex_wint_weights_{}_real'.format(weight_function_Q),
                 np.array(sinI))
        self.set('quex_wint_weights_{}_imag'.format(weight_function_Q),
                 np.array(cosI))
        self.set('quex_rot_{}_real'.format(weight_function_I), 1.0)
        self.set('quex_rot_{}_imag'.format(weight_function_I), 1.0)
        self.set('quex_rot_{}_real'.format(weight_function_Q), 1.0)
        self.set('quex_rot_{}_imag'.format(weight_function_Q), -1.0)

    def prepare_DSB_weight_and_rotation(self, IF, weight_function_I=0, weight_function_Q=1):
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.array(np.cos(2*np.pi*IF*tbase))
        sinI = np.array(np.sin(2*np.pi*IF*tbase))
        self.set('quex_wint_weights_{}_real'.format(weight_function_I),
                 np.array(cosI))
        self.set('quex_wint_weights_{}_real'.format(weight_function_Q),
                 np.array(sinI))
        # the factor 2 is needed so that scaling matches SSB downconversion
        self.set('quex_rot_{}_real'.format(weight_function_I), 2.0)
        self.set('quex_rot_{}_imag'.format(weight_function_I), 0.0)
        self.set('quex_rot_{}_real'.format(weight_function_Q), 2.0)
        self.set('quex_rot_{}_imag'.format(weight_function_Q), 0.0)

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
            paths = [paths]
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
            paths = [paths]
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

    # sequencer functions
    def awg_sequence_acquisition_and_DIO_triggered_pulse(
            self, Iwaves, Qwaves, cases, acquisition_delay, timeout=5):
        # setting the acquisition delay samples
        delay_samples = int(acquisition_delay*1.8e9/8)
        # setting the delay in the instrument
        self.awgs_0_userregs_2(delay_samples)

        sequence = (
            'const TRIGGER1  = 0x000001;\n' +
            'const WINT_TRIG = 0x000010;\n' +
            'const IAVG_TRIG = 0x000020;\n' +
            'const WINT_EN   = 0x1ff0000;\n' +
            'const DIO_VALID = 0x00010000;\n' +
            'setTrigger(WINT_EN);\n' +
            'var loop_cnt = getUserReg(0);\n' +
            'var wait_delay = getUserReg(2);\n' +
            'var RO_TRIG;\n' +
            'if(getUserReg(1)){\n' +
            ' RO_TRIG=IAVG_TRIG;\n' +
            '}else{\n' +
            ' RO_TRIG=WINT_TRIG;\n' +
            '}\n' +
            'var trigvalid = 0;\n' +
            'var dio_in = 0;\n' +
            'var cw = 0;\n')

        # loop to generate the wave list
        for i in range(len(Iwaves)):
            Iwave = Iwaves[i]
            Qwave = Qwaves[i]
            if np.max(Iwave) > 1.0 or np.min(Iwave) < -1.0:
                raise KeyError(
                    "exceeding AWG range for I channel, all values should be within +/-1")
            elif np.max(Qwave) > 1.0 or np.min(Qwave) < -1.0:
                raise KeyError(
                    "exceeding AWG range for Q channel, all values should be within +/-1")
            elif len(Iwave) > 16384:
                raise KeyError(
                    "exceeding max AWG wave lenght of 16384 samples for I channel, trying to upload {} samples".format(len(Iwave)))
            elif len(Qwave) > 16384:
                raise KeyError(
                    "exceeding max AWG wave lenght of 16384 samples for Q channel, trying to upload {} samples".format(len(Qwave)))
            wave_I_string = self.array_to_combined_vector_string(
                Iwave, "Iwave{}".format(i))
            wave_Q_string = self.array_to_combined_vector_string(
                Qwave, "Qwave{}".format(i))
            sequence = sequence+wave_I_string+wave_Q_string
        # starting the loop and switch statement
        sequence = sequence+(
            'repeat(loop_cnt) {\n' +
            ' waitDIOTrigger();\n' +
            ' var dio = getDIOTriggered();\n' +
            # now hardcoded for 7 bits (cc-light)
            ' cw = (dio >> 17) & 0x1f;\n' +
            '  switch(cw) {\n')
        # adding the case statements
        for i in range(len(Iwaves)):
            # generating the case statement string
            case = '  case {}:\n'.format(int(cases[i]))
            case_play = '   playWave(Iwave{}, Qwave{});\n'.format(i, i)
            # adding the individual case statements to the sequence
            # FIXME: this is a hack to work around missing timing in OpenQL
            # Oct 2017
            sequence = sequence + case+case_play

        # adding the final part of the sequence including a default wave
        sequence = (sequence +
                    '  default:\n' +
                    '   playWave(ones(36), ones(36));\n' +
                    ' }\n' +
                    ' wait(wait_delay);\n' +
                    ' setTrigger(WINT_EN + RO_TRIG);\n' +
                    ' setTrigger(WINT_EN);\n' +
                    ' wait(100);'
                    #' waitWave();\n'+ #removing this waitwave for now
                    '}\n' +
                    'wait(300);\n' +
                    'setTrigger(0);\n')
        self.awg_string(sequence, timeout=timeout)

    def awg_sequence_acquisition_and_pulse(self, Iwave, Qwave, trigger=True):
        if np.max(Iwave) > 1.0 or np.min(Iwave) < -1.0:
            raise KeyError(
                "exceeding AWG range for I channel, all values should be withing +/-1")
        elif np.max(Qwave) > 1.0 or np.min(Qwave) < -1.0:
            raise KeyError(
                "exceeding AWG range for Q channel, all values should be withing +/-1")
        elif len(Iwave) > 16384:
            raise KeyError(
                "exceeding max AWG wave lenght of 16384 samples for I channel, trying to upload {} samples".format(len(Iwave)))
        elif len(Qwave) > 16384:
            raise KeyError(
                "exceeding max AWG wave lenght of 16384 samples for Q channel, trying to upload {} samples".format(len(Qwave)))

        wave_I_string = self.array_to_combined_vector_string(Iwave, "Iwave")
        wave_Q_string = self.array_to_combined_vector_string(Qwave, "Qwave")
        preamble = """
const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x1ff0000;
setTrigger(WINT_EN);
var loop_cnt = getUserReg(0);
var wait_delay = getUserReg(2);
var RO_TRIG;
if(getUserReg(1)){
  RO_TRIG=IAVG_TRIG;
}else{
  RO_TRIG=WINT_TRIG;
}\n"""
        loop_start = """
repeat(loop_cnt) {
""" + ("\twaitDigTrigger(1, 1);" if trigger else "") + """
\tplayWave(Iwave, Qwave);\n"""
        end_string = """
\tsetTrigger(WINT_EN + RO_TRIG);
\tsetTrigger(WINT_EN);
\twaitWave();
}
wait(300);
setTrigger(0);"""

        string = preamble+wave_I_string+wave_Q_string + \
            loop_start+end_string
        self.awg_string(string)

    def awg_sequence_acquisition_and_pulse_multi_segment(self, readout_pulses):

        wave_defs = ""

        for i, pulse in enumerate(readout_pulses):
            Iwave = pulse.real.copy()
            Qwave = pulse.imag.copy()
            if np.max(Iwave) > 1.0 or np.min(Iwave) < -1.0:
                raise KeyError("exceeding AWG range for I channel, all values "
                               "should be withing +/-1")
            elif np.max(Qwave) > 1.0 or np.min(Qwave) < -1.0:
                raise KeyError("exceeding AWG range for Q channel, all values "
                               "should be withing +/-1")
            elif len(Iwave) > 16384:
                raise KeyError("exceeding max AWG wave lenght of 16384 samples "
                               "for I channel, trying to upload {} samples"
                               .format(len(Iwave)))
            elif len(Qwave) > 16384:
                raise KeyError("exceeding max AWG wave lenght of 16384 samples "
                               "for Q channel, trying to upload {} samples"
                               .format(len(Qwave)))
            wave_defs += self.array_to_combined_vector_string(
                Iwave, "Iwave{}".format(i))
            wave_defs += self.array_to_combined_vector_string(
                Qwave, "Qwave{}".format(i))

        preamble = """
        const TRIGGER1  = 0x0000001;
        const WINT_TRIG = 0x0000010;
        const IAVG_TRIG = 0x0000020;
        const WINT_EN   = 0x1ff0000;
        setTrigger(WINT_EN);
        var loop_cnt = getUserReg(0);
        var wait_delay = getUserReg(2);
        var RO_TRIG;
        if (getUserReg(1)) {{
          RO_TRIG = IAVG_TRIG;
        }} else {{
          RO_TRIG = WINT_TRIG;
        }}\n"""

        loop = """for (var i = 0; i < loop_cnt; i = i + {}) {{
        """.format(len(readout_pulses))

        for i, _ in enumerate(readout_pulses):
            loop += """
            \twaitDigTrigger(1, 1);
            \tplayWave(Iwave{0}, Qwave{0});\n""".format(i)
            loop += """
            \tsetTrigger(WINT_EN + RO_TRIG);
            \tsetTrigger(WINT_EN);
            \twaitWave();\n"""

        end_string = """
        }
        wait(300);
        setTrigger(0);\n"""

        string = preamble + wave_defs + loop + end_string
        self.awg_string(string)


    def array_to_combined_vector_string(self, array, name):
        # this function cuts up arrays into several vectors of maximum length 1024 that are joined.
        # this is to avoid python crashes (was found to crash for vectors of
        # lenght> 1490)
        string = 'vect('
        join = False
        n = 0
        while n < len(array):
            string += '{:.10f}'.format(array[n])
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
            string = 'wave ' + name + ' = join(' + string + ');\n'
        else:
            string = 'wave ' + name + ' = ' + string + ';\n'
        return string

    def awg_sequence_acquisition(self, trigger=True):
        string = """
const TRIGGER1  = 0x0000001;
const WINT_TRIG = 0x0000010;
const IAVG_TRIG = 0x0000020;
const WINT_EN   = 0x1ff0000;
setTrigger(WINT_EN);
var loop_cnt = getUserReg(0);
var RO_TRIG;
if(getUserReg(1)){
  RO_TRIG=IAVG_TRIG;
}else{
  RO_TRIG=WINT_TRIG;
}
repeat(loop_cnt) {
""" + ("\twaitDigTrigger(1, 1);" if trigger else "") + """
\tsetTrigger(WINT_EN + RO_TRIG);
\twait(5);
\tsetTrigger(WINT_EN);
""" + ("\twait(2250);" if not trigger else "") + """
}
wait(1000);
setTrigger(0);"""
        self.awg_string(string)

    def awg_update_waveform(self, index, data1, data2=None):
        """Immediately updates the waveform with the given index.

        The waveform in the waveform viewer in the LabOne web interface is not
        updated. If data for both AWG channels is given, then the lengths must
        match.

        Warning! This method should be used with care.
        Note that the waveform get's overwritten if a new program is uploaded.

        Args:
            index: Index of the waveform to update. Corresponds to the order of
                   waveforms in the AWG/Waveform tab in the web interface.
                   Starts from 0.
            data1: Waveform data for channel 1.
            data2: Optional waveform data for channel 2.
        """
        if data1 is None and data2 is None:
            return
        elif data1 is None:
            data = data2
        elif data2 is None:
            data = data1
        else:
            data = np.vstack((data1, data2,)).reshape((-1,), order='F')
        self.awgs_0_waveform_index(index)
        self.awgs_0_waveform_data(data)
        self._daq.sync()

    def awg_sequence_acquisition_and_pulse_SSB(
            self, f_RO_mod, RO_amp, RO_pulse_length,
            alpha=1, phi_skew=0):
        f_sampling = 1.8e9
        samples = RO_pulse_length*f_sampling
        sample_idxs = np.arange(int(samples))
        Iwave = RO_amp * alpha * np.cos(
            2 * np.pi * sample_idxs * f_RO_mod / f_sampling + phi_skew * np.pi / 180)
        Qwave = - RO_amp * np.sin(2 * np.pi * sample_idxs * f_RO_mod / f_sampling)
        self.awg_sequence_acquisition_and_pulse(Iwave, Qwave)


    def awg_sequence_acquisition_and_pulse_SSB_gaussian_filtered(
            self, f_RO_mod, RO_amp, RO_pulse_length, filter_sigma, nr_sigma,
            alpha=1, phi_skew=0):
        f_sampling = 1.8e9
        samples = RO_pulse_length*f_sampling

        wave = RO_amp*np.ones(int(samples))

        waveFiltered = gaussian_filter(wave, filter_sigma, nr_sigma,
                                       sampling_rate=f_sampling)

        Iwave, Qwave = IQ_split(waveFiltered, f_RO_mod, phi_skew=phi_skew,
                                alpha=alpha, sampling_rate=f_sampling)

        self.awg_sequence_acquisition_and_pulse(Iwave, Qwave)


    def awg_sequence_acquisition_and_pulse_SSB_CLEAR_pulse(
            self, amp_base, length_total, delta_amp_segments, length_segments, f_RO_mod,
            sampling_rate = 1.8e9, phase = 0, alpha=1):
        '''
        Generates the envelope of a CLEAR pulse.
            length_total in s
            length_segments list of length 4 in s
            amp_base in V
            delta_amp_segments list of length 4 in V
            sampling_rate in Hz
            empty delay in s
            phase in degrees
        '''

        amp_pulse = CLEAR_shape(  amp_base, length_total, delta_amp_segments,
                                length_segments, sampling_rate=sampling_rate)


        Iwave, Qwave = IQ_split(amp_pulse, f_RO_mod, phi_skew= phase, alpha = alpha,
                                sampling_rate =  sampling_rate)

        self.awg_sequence_acquisition_and_pulse(Iwave, Qwave)


    def awg_sequence_acquisition_and_pulse_SSB_gauss_CLEAR_pulse(self,
                          amp_base, length_total, delta_amp_segments,
                          length_segments, sigma, nr_sigma, f_RO_mod,
                          alpha=1, sampling_rate=1.8e9, phase=0):
        """
        Generates the envelope of a gaussian filtered CLEAR pulse.
            length_total in s
            length_segments list of length 4 in s
            amp_base in V
            delta_amp_segments list of length 4 in V
            gauss_amp in V
            sigma in s
            sampling_rate in Hz
            empty delay in s
            phase in degrees
        """
        amp_pulse = CLEAR_shape(amp_base, length_total, delta_amp_segments,
                                length_segments, sampling_rate=sampling_rate)

        amp_filtered = gaussian_filter( amp_pulse, sigma, nr_sigma,
                                        sampling_rate=sampling_rate)

        pulse_I, pulse_Q = IQ_split(amp_filtered, f_RO_mod, phi_skew=phase,
                                alpha=alpha, sampling_rate=sampling_rate)

        self.awg_sequence_acquisition_and_pulse(pulse_I, pulse_Q)


    def upload_transformation_matrix(self, matrix):
        for i in range(np.shape(matrix)[0]):  # looping over the rows
            for j in range(np.shape(matrix)[1]):  # looping over the colums
                self.set('quex_trans_{}_col_{}_real'.format(
                    j, i), matrix[i][j])

    def download_transformation_matrix(self, nr_rows=None, nr_cols=None):
        if not nr_rows or not nr_cols:
            nr_rows = self.nr_integration_channels
            nr_cols = self.nr_integration_channels
        matrix = np.zeros([nr_rows, nr_cols])
        for i in range(np.shape(matrix)[0]):  # looping over the rows
            for j in range(np.shape(matrix)[1]):  # looping over the colums
                matrix[i][j] = self.get(
                    'quex_trans_{}_col_{}_real'.format(j, i))
        return matrix

    def spec_mode_on(self, acq_length=1/1500, IF=20e6, ro_amp=0.1):
        awg_code = """
const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x1f0000;
setTrigger(WINT_EN);
var loop_cnt = getUserReg(0);

const Fsample = 1.8e9;
const triggerdelay = {}; //seconds

repeat(loop_cnt) {{
setTrigger(WINT_EN + WINT_TRIG + TRIGGER1);
wait(5);
setTrigger(WINT_EN);
wait(triggerdelay*Fsample/8 - 5);
}}
wait(1000);
setTrigger(0);
        """.format(acq_length)
        # setting the internal oscillator to the IF
        self.oscs_0_freq(IF)
        # setting the integration path to use the oscillator instead of integration functions
        self.quex_wint_mode(1)
        # just below the
        self.quex_wint_length(int(acq_length*0.99*1.8e9))
        # uploading the sequence
        self.awg_string(awg_code)
        # setting the integration rotation to single sideband
        self.quex_rot_0_real(1)
        self.quex_rot_0_imag(1)
        self.quex_rot_1_real(1)
        self.quex_rot_1_imag(-1)
        # setting the mixer deskewing to identity
        self.quex_deskew_0_col_0(1)
        self.quex_deskew_1_col_0(0)
        self.quex_deskew_0_col_1(0)
        self.quex_deskew_1_col_1(1)

        self.sigouts_0_enables_3(1)
        self.sigouts_1_enables_7(1)
        # setting
        self.sigouts_1_amplitudes_7(ro_amp)  # magic scale factor
        self.sigouts_0_amplitudes_3(ro_amp)

    def spec_mode_off(self):
        # Resetting To regular Mode
        # changing int length
        self.quex_wint_mode(0)
        # Default settings copied
        self.quex_rot_0_imag(0)
        self.quex_rot_0_real(1)
        self.quex_rot_1_imag(0)
        self.quex_rot_1_real(1)
        # setting to DSB by default
        self.quex_deskew_0_col_0(1)
        self.quex_deskew_1_col_0(0)
        self.quex_deskew_0_col_1(0)
        self.quex_deskew_1_col_1(1)
        # switching off the modulation tone
        self.sigouts_0_enables_3(0)
        self.sigouts_1_enables_7(0)


class ziShellError(Exception):
    """Base class for exceptions in this module."""
    pass


class ziShellDAQError(ziShellError):
    """Exception raised when no DAQ has been connected."""
    pass


class ziShellModuleError(ziShellError):
    """Exception raised when a module has not been started."""
    pass


class ziShellCompilationError(ziShellError):
    """
    Exception raised when the zi AWG-8 compiler encounters an error.
    """
    pass

####################
# Helper Functions #
####################
def gaussian_filter(wave, filter_sigma, nr_sigma, sampling_rate=1.8e9):

    filter_samples = int(filter_sigma*nr_sigma*sampling_rate)
    filter_sample_idxs = np.arange(filter_samples)
    gauss_filter = np.exp(-0.5*(filter_sample_idxs - filter_samples/2)**2 /
                          (filter_sigma*sampling_rate)**2)
    gauss_filter /= gauss_filter.sum()
    waveFiltered = np.convolve(wave, gauss_filter, mode='full')
    return waveFiltered


def IQ_split(wave, f_RO_mod, phi_skew=0, alpha=1, sampling_rate=1.8e9):
    Iwave = alpha * wave * np.cos(2 * np.pi *
                                  np.arange(len(wave)) * f_RO_mod /sampling_rate +
                                  phi_skew * np.pi / 180)
    Qwave = -wave * np.sin(2 * np.pi *
                           np.arange(len(wave)) * f_RO_mod /sampling_rate)

    return Iwave, Qwave


def CLEAR_shape(amp_base, length_total, delta_amp_segments,
                 length_segments, sampling_rate=1.8e9):
    
    delta_amp_segments = list(delta_amp_segments)
    if type(length_segments) == float:
        length_segments = [length_segments]*4
    elif type(length_segments) == list:
        pass
    else:
        raise TypeError('The type of length_segments needs to be list or float')

    if len(delta_amp_segments) == len( length_segments) == 4:
        pass
    else:
        raise ValueError(
            'delta_amp_segments and length_segments need to be lists of length 4')


    pulse_samples = (length_total)*sampling_rate
    segments_samples = list(map(lambda x: int(x*sampling_rate), length_segments))
    amp_pulse = np.concatenate(
        ((np.ones(segments_samples[0])*(amp_base+delta_amp_segments[0])),
         (np.ones(segments_samples[1])*(amp_base+delta_amp_segments[1])),
         (np.ones(int(pulse_samples-sum(segments_samples)))*amp_base),
         (np.ones(segments_samples[2])*(amp_base+delta_amp_segments[2])),
         (np.ones(segments_samples[3])*(amp_base+delta_amp_segments[3]))))

    return amp_pulse