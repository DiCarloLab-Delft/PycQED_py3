import zhinst.ziPython as zi
import zhinst.utils as zi_utils
import time
import json
import os
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from fnmatch import fnmatch
import zhinst.zishell as zis
#from instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC



class UHFQC(Instrument):
    """
    This is the qcodes driver for the 1.8 Gsample/s UHF-QC developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5 ucs4 16.04 for 64bit Windows from http://www.zhinst.com/downloads
    2. pip install dependencies: httplib2, plotly, pyqtgraph
    3. manually paste zishell.py one directory above the zhinst directory (C:/Anaconda3/side) (can be found in transmon/inventory/firmware_Nielsb)
    4. upload the latest firmware to the UHFQC by opening reboot.bat in 'Transmon\Inventory\ZurichInstruments\firmware_Nielsb\firmware_x'. WIth x the highest available number.
    5. find out where sequences are stored by saving a sequence from the GUI and then check :"showLog" to see where it is stored. This is the location where AWG sequences can be loaded from.
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
        print(self._device)

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self._s_file_name = dir_path+'\\zi_parameter_files\\s_node_pars.txt'
        self._d_file_name = dir_path+'\\zi_parameter_files\\d_node_pars.txt'

        init=True
        try:
            f=open(self._s_file_name).read()
            s_node_pars = json.loads(f)
        except:
            print("parameter file for gettable parameters {} not found".format(self._s_file_name))
            init=False
        try:
            f=open(self._d_file_name).read()
            d_node_pars = json.loads(f)
        except:
            print("parameter file for settable parameters {} not found".format(self._d_file_name))
            init=False

        for parameter in s_node_pars:
            parname=parameter[0].replace("/","_")
            parfunc="/"+device+"/"+parameter[0]
            if parameter[1]=='float':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(zis.setd, parfunc),
                    get_cmd=self._gen_get_func(zis.getd, parfunc),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1]=='float_small':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(zis.setd, parfunc),
                    get_cmd=self._gen_get_func(zis.getd, parfunc),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1]=='int_8bit':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(zis.seti, parfunc),
                    get_cmd=self._gen_get_func(zis.geti, parfunc),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='int':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(zis.seti, parfunc),
                    get_cmd=self._gen_get_func(zis.geti, parfunc),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='int_64':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(zis.seti, parfunc),
                    get_cmd=self._gen_get_func(zis.geti, parfunc),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='bool':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(zis.seti, parfunc),
                    get_cmd=self._gen_get_func(zis.geti, parfunc),
                    vals=vals.Ints(parameter[2], parameter[3]))
            else:
                print("parameter {} type {} from from s_node_pars not recognized".format(parname,parameter[1]))
        for parameter in d_node_pars:
            parname=parameter[0].replace("/","_")
            parfunc="/"+device+"/"+parameter[0]
            if parameter[1]=='float':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(zis.getd, parfunc))
            elif parameter[1]=='vector_g':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(zis.getv, parfunc))
            elif parameter[1]=='vector_s':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(zis.setv, parfunc),
                    vals=vals.Anything())
            elif parameter[1]=='vector_gs':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(zis.setv, parfunc),
                    get_cmd=self._gen_get_func(zis.getv, parfunc),
                    vals=vals.Anything())
            else:
                print("parameter {} type {} from d_node_pars not recognized".format(parname,parameter[1]))


        self.add_parameter('AWG_file',
                           set_cmd=self._do_set_AWG_file,
                           vals=vals.Anything())
        zis.connect_server('localhost', port)
        zis.connect_device(self._device, 'USB')
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
        self.AWG_file('traditional.seqc')

        # The AWG program uses userregs/0 to define the number o iterations in the loop
        self.awgs_0_userregs_0(pow(2, LOG2_AVG_CNT)*pow(2, LOG2_RL_AVG_CNT))

        # Turn on both outputs
        self.sigouts_0_on(1)
        self.sigouts_1_on(1)

        # Configure the input averager length and averaging count
        self.quex_iavg_length(4096)
        self.quex_iavg_avgcnt(LOG2_AVG_CNT)

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

        # Configure the weighted integration units with constant values, each channel gets
        # one, two, three and four non-zero weights, respectively
        self.quex_wint_weights_0_real(np.array([1.0]*1 + [0.0]*127))
        self.quex_wint_weights_0_imag(np.array([1.0]*1 + [0.0]*127))
        self.quex_wint_weights_1_real(np.array([1.0]*2 + [0.0]*126))
        self.quex_wint_weights_1_imag(np.array([1.0]*2 + [0.0]*126))
        self.quex_wint_weights_2_real(np.array([1.0]*3 + [0.0]*125))
        self.quex_wint_weights_2_imag(np.array([1.0]*3 + [0.0]*125))
        self.quex_wint_weights_3_real(np.array([1.0]*12 + [0.0]*116))
        self.quex_wint_weights_3_imag(np.array([1.0]*12 + [0.0]*116))

        # Length is set in units of 1 samples
        self.quex_wint_length(4096)
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

    def _do_set_AWG_file(self, filename):
        zis.awg(filename)

        #code to upload AWG sequence as a string
        # def awg(self, filename):
        #         for device in self._devices:
        #             h = self.daq.awgModule()
        #             h.set('awgModule/device', device)
        #             h.set('awgModule/index', 0)
        #             h.execute()
        #             h.set('awgModule/compiler/sourcefile', filename)
        #             h.set('awgModule/compiler/start', 1)
        #             h.set('awgModule/elf/file', '')

        # Now, if you would change it to:

         # def awg(self, sourcestring):
         #        for device in self._devices:
         #            h = self.daq.awgModule()
         #            h.set('awgModule/device', device)
         #            h.set('awgModule/index', 0)
         #            h.execute()
         #            h.set('awgModule/compiler/sourcestring', sourcestring)
         #            h.set('awgModule/compiler/start', 1)
         #            h.set('awgModule/elf/file', '')

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
        patterns = ["awgs", "sigins", "sigouts", "quex", "dios","system/extclk"] #["quex/iavg", "quex/wint"]
        s_file = open(self._s_file_name, 'w')
        d_file = open(self._d_file_name, 'w')
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
            default_values=zis.getd(s_nodes, True)
            for s_node in s_nodes:
                zis.setd(s_node,  1e12)
            max_values = zis.getd(s_nodes, True)
            for s_node in s_nodes:
                zis.setd(s_node, -1e12)
            min_values = zis.getd(s_nodes, True)
            float_values = [np.pi]*len(s_nodes)
            for i, s_node in enumerate(s_nodes):
                if np.pi > max_values[i]:
                    float_values[i] = max_values[i]/np.pi;
                zis.setd(s_node, float_values[i])
            actual_float_values = zis.getd(s_nodes, True)
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
                line=[s_node, node_types[i], min_values[i], max_values[i]]
                print(line)
                s_node_pars.append(line)
                #json.dump(line, s_file, indent=2, default=int)


            #extracting info from the data nodes
            d_nodes = list(d_nodes)
            #default_values=self.getd(d_nodes, True)
            default_values=np.zeros(len(d_nodes))
            node_types = ['']*len(d_nodes)

            for i, d_node in enumerate(d_nodes):
                try:
                    answer=zis.getv(d_node)
                    if isinstance(answer, dict):
                        value=answer['value'][0]
                        node_types[i]='float'
                    elif  isinstance(answer, list):
                        try:
                            zis.setv(d_node,np.array([0,0,0]))
                            node_types[i]='vector_gs'
                        except:
                            value=answer[0]['vector']
                            node_types[i]='vector_g'
                    else:
                        print("unknown type")
                except:
                    node_types[i]='vector_s'
                line=[d_node, node_types[i]]#, default_values[i]]
                print(line)
                d_node_pars.append(line)
                #json.dump(line, d_file, indent=2, default=int)

        json.dump(s_node_pars[9:], s_file, default=int, indent=2)
        json.dump(d_node_pars[9:], d_file, default=int, indent=2)
        s_file.close()
        d_file.close()

    def prepare_SSB_weight_and_rotation(self, IF):
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.cos(2*np.pi*IF*tbase)
        sinI = np.sin(2*np.pi*IF*tbase)
        self.quex_wint_weights_0_real(np.array(cosI))
        self.quex_wint_weights_0_imag(np.array(sinI))
        self.quex_wint_weights_1_real(np.array(sinI))
        self.quex_wint_weights_1_imag(np.array(cosI))
        self.quex_rot_0_real(1.0)
        self.quex_rot_0_imag(1.0)
        self.quex_rot_1_real(1.0)
        self.quex_rot_1_imag(-1.0)

    def prepare_DSB_weight_and_rotation(self, IF):
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
        cosI = np.cos(2*np.pi*IF*tbase)
        sinI = np.sin(2*np.pi*IF*tbase)
        self.quex_wint_weights_0_real(np.array(cosI))
        self.quex_wint_weights_0_imag(np.array(sinI)*0)
        self.quex_wint_weights_1_real(np.array(sinI))
        self.quex_wint_weights_1_imag(np.array(cosI)*0)
        self.quex_rot_0_real(1.0)
        self.quex_rot_0_imag(0.0)
        self.quex_rot_1_real(1.0)
        self.quex_rot_1_imag(0.0)


    # def render_weights(self, wave_name, show=True, time_units='lut_index',
    #             reload_pulses=True):
    #     if reload_pulses:
    #         self.generate_standard_pulses()
    #     fig, ax = plt.subplots(1, 1)
    #     if time_units == 'lut_index':
    #         x = np.arange(len(self._wave_dict[wave_name][0]))
    #         ax.set_xlabel('Lookuptable index (i)')
    #         ax.vlines(128, self._voltage_min, self._voltage_max, linestyle='--')
    #     elif time_units == 's':
    #         x = (np.arange(len(self._wave_dict[wave_name][0]))
    #              / self.sampling_rate.get())
    #         ax.set_xlabel('time (s)')
    #         ax.vlines(128 / self.sampling_rate.get(),
    #                   self._voltage_min, self._voltage_max, linestyle='--')

    #     ax.plot(x, self._wave_dict[wave_name][0],
    #             marker='o', label=wave_name+' chI')
    #     ax.plot(x, self._wave_dict[wave_name][1],
    #             marker='o', label=wave_name+' chQ')
    #     ax.set_ylabel('Amplitude (V)')
    #     ax.set_axis_bgcolor('gray')
    #     ax.axhspan(self._voltage_min, self._voltage_max, facecolor='w',
    #                linewidth=0)
    #     ax.legend()
    #     ax.set_ylim(self._voltage_min*1.1, self._voltage_max*1.1)
    #     ax.set_xlim(0, x[-1])
    #     if show:
    #         plt.show()
    #     return fig, ax

    # def setd(self, path, value):
    #     # Handle absolute path
    #     if path[0] == '/':
    #         self._daq.setDouble(path, value)
    #     else:
    #         self._daq.setDouble('/' + self._device + '/' + path, value)

    # def seti(self, path, value):
    #     # Handle absolute path
    #     if path[0] == '/':
    #         self._daq.setInt(path, value)
    #     else:
    #         self._daq.setInt('/' + self._device + '/' + path, value)

    # def setv(self, path, value):
    #     # Handle absolute path
    #     if path[0] == '/':
    #         self._daq.vectorWrite(path, value)
    #     else:
    #         self._daq.vectorWrite('/' + self._device + '/' + path, value)

    # def geti(self, paths, deep=True):
    #     if type(paths) is not list:
    #         paths = [ paths ]
    #         single = 1
    #     else:
    #         single = 0

    #     values = []
    #     for p in paths:
    #         if p[0] == '/':
    #             if deep:
    #                 self._daq.getAsEvent(p)
    #                 tmp = self._daq.poll(0.1, 500, 4, True)
    #                 if p in tmp:
    #                     values.append(tmp[p]['value'][0])
    #             else:
    #                 values.append(self._daq.getInt(p))
    #         else:
    #             tmp_p = '/' + self._device + '/' + p
    #             if deep:
    #                 self._daq.getAsEvent(tmp_p)
    #                 tmp = self._daq.poll(0.1, 500, 4, True)
    #                 if tmp_p in tmp:
    #                     values.append(tmp[tmp_p]['value'][0])
    #             else:
    #                 values.append(self._daq.getInt(tmp_p))
    #     if single:
    #         return values[0]
    #     else:
    #         return values

    # def getd(self, paths, deep=True):
    #     if type(paths) is not list:
    #         paths = [ paths ]
    #         single = 1
    #     else:
    #         single = 0

    #     values = []
    #     for p in paths:
    #         if p[0] == '/':
    #             if deep:
    #                 self._daq.getAsEvent(p)
    #                 tmp = self._daq.poll(0.1, 500, 4, True)
    #                 if p in tmp:
    #                     values.append(tmp[p]['value'][0])
    #             else:
    #                 values.append(self._daq.getDouble(p))
    #         else:
    #             tmp_p = '/' + self._device + '/' + p
    #             if deep:
    #                 self._daq.getAsEvent(tmp_p)
    #                 tmp = self._daq.poll(0.1, 500, 4, True)
    #                 if tmp_p in tmp:
    #                     values.append(tmp[tmp_p]['value'][0])
    #             else:
    #                 values.append(self._daq.getDouble(tmp_p))
    #     if single:
    #         return values[0]
    #     else:
    #         return values

    # def getv(self, paths):
    #     if type(paths) is not list:
    #         paths = [ paths ]
    #         single = 1
    #     else:
    #         single = 0

    #     values = []
    #     for p in paths:
    #         if p[0] == '/':
    #             self._daq.getAsEvent(p)
    #             tmp = self._daq.poll(0.5, 500, 4, True)
    #             if p in tmp:
    #                 values.append(tmp[p])
    #         else:
    #             tmp_p = '/' + self._device + '/' + p
    #             self._daq.getAsEvent(tmp_p)
    #             tmp = self._daq.poll(0.5, 500, 4, True)
    #             if tmp_p in tmp:
    #                 values.append(tmp[tmp_p])
    #     if single:
    #         return values[0]
    #     else:
    #         return values



