import zhinst.zishell as zis
import zhinst.ziPython as zi
import zhinst.utils as zi_utils
import time
import json
import os
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals



class UHFQC(Instrument):
    '''
    This is the qcodes driver for the 1.8 Gsample/s UHF-QC developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. install ziPython 3.5 ucs4 16.04 for 64bit Windows from http://www.zhinst.com/downloads
    2. pip install dependencies: httplib2, plotly, pyqtgraph
    3. manually paste zishell.py one directory above the zhinst directory (C:/Anaconda3/side) (can be found in transmon/inventory/firmware_Nielsb)
    4. add zishell.py to the zhinst __init__.py
    5. upload the latest firmware to the UHFQC by opening reboot.bat in "Transmon\Inventory\ZurichInstruments\firmware_Nielsb\firmware_x". WIth x the highest available number.
    6. find out where sequences are stored by saving a sequence from the GUI and then check :"showLog" to see where it is stored. This is the location where AWG sequences can be loaded from.
    todo:
    - write all fcuncions for data acquisition
    - write all functions for AWG control

    misc: when device crashes, check the log file in "D:\TUD207933"\My Documents\Zurich Instruments\LabOne\WebServer\Log
    '''
    def __init__(self, name, server_name, address=8004, **kw):
        '''
        Input arguments: 
            name:           (str) name of the instrument 
            server_name:    (str) qcodes instrument server 
            address:        (int) the address of the data server e.g. 8006
        '''

        t0 = time.time()
        super().__init__(name, server_name)
        # parameter structure: [ZI node path, data type, min value, max value], extracting all parameters from precalibrated text files

        setd = zis.setd
        getd = zis.getd
        seti = zis.seti
        geti = zis.geti
        setv = zis.setv
        getv = zis.getv

        #idea to automatically generate functions and min/max values:
        #find all nodes: with:        
        #for n in zis.find('*%s*' %self._device):
        #remove dev###
        # infer parameter type
        # get max and min by trial and error
        #create function
        # 
        self._daq = zi.ziDAQServer('127.0.0.1', address)
        self._device = zi_utils.autoDetect(self._daq)
        s_node_pars=[]
        d_node_pars=[]

        self._s_file_name ='zi_parameter_files/s_node_pars_{}.txt'.format(self._device)
        self._d_file_name = 'zi_parameter_files/d_node_pars_{}.txt'.format(self._device)

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
                    set_cmd=self._gen_set_func(setd, parameter[0]),
                    get_cmd=self._gen_get_func(getd, parameter[0]),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1]=='float_small':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(setd, parameter[0]),
                    get_cmd=self._gen_get_func(getd, parameter[0]),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1]=='int_8bit':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(seti, parameter[0]),
                    get_cmd=self._gen_get_func(geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='int':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(seti, parameter[0]),
                    get_cmd=self._gen_get_func(geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='int_64':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(seti, parameter[0]),
                    get_cmd=self._gen_get_func(geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='bool':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(seti, parameter[0]),
                    get_cmd=self._gen_get_func(geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            else:
                print("parameter {} type {} from from s_node_pars not recognized".format(parname,parameter[1]))

        for parameter in d_node_pars:
            parname=parameter[0][9:].replace("/","_")
            if parameter[1]=='float':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(getd, parameter[0]))
            elif parameter[1]=='vector_g':
                self.add_parameter(
                    parname,
                    get_cmd=self._gen_get_func(getv, parameter[0]))
            elif parameter[1]=='vector_s':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(setv, parameter[0]),
                    vals=vals.Anything())
            else:   
                print("parameter {} type {} from d_node_pars not recognized".format(parname,parameter[1]))


        # self.add_parameter('awg_sequence',
        #                    set_cmd=self._do_set_awg,
        #                    get_cmd=self._do_get_acquisition_mode,
        #                    vals=vals.Anything())



        zis.connect_server('localhost', address)
        zis.connect_device(self._device, 'USB')
        t1 = time.time()
        print('Initialized UHFQC', self._device,
              'in %.2fs' % (t1-t0))


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

    def load_AWG_sequence(self, sequence):
        zis.awg(sequence)
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
        zis.disconnect_device(self._device)


    def create_parameter_files(self):
        #this functions retrieves all possible settable and gettable parameters from the device.
        #Additionally, iot gets all minimum and maximum values for the parameters by trial and error

        s_node_pars=[]
        d_node_pars=[]
        patterns = ["awgs", "sigins", "sigouts", "quex"]
        for pattern in patterns:
            print("extracting parameters of type", pattern)
            all_nodes = set(zis.find('*{}*'.format(pattern)))
            s_nodes = set(zis.finds('*{}*'.format(pattern)))
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
                #zis.setd(node,default_values[i])
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
                line=[s_node, node_types[i], min_values[i], max_values[i], '\n']
                print(line)
                s_node_pars.append(line)
          
            #extracting info from the data nodes
            d_nodes = list(d_nodes)
            #default_values=zis.getd(d_nodes, True)
            default_values=np.zeros(len(d_nodes))
            node_types = ['']*len(d_nodes)
            for i, d_node in enumerate(d_nodes):
                try: 
                    answer=zis.getv(d_node)
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





