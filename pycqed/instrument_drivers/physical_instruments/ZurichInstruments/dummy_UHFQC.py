import time
import json
import os
import sys
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from fnmatch import fnmatch
from qcodes.instrument.parameter import ManualParameter


class dummy_UHFQC(Instrument):

    """
    Dummy version of the UHFQC driver.
    This is quickly put together and quite hacky.
    A proper elegant solution is feasible but not implemented here.
    """

    def __init__(self, name, **kw):
        t0 = time.time()
        super().__init__(name, **kw)

        # self._daq = zi.ziDAQServer(address, int(port), 5)
        # if device.lower() == 'auto':
        #     self._device = zi_utils.autoDetect(self._daq)
        # else:
        #     self._device = device
        #     self._daq.connectDevice(self._device, interface)
        # #self._device = zi_utils.autoDetect(self._daq)
        # self._awgModule = self._daq.awgModule()
        # self._awgModule.set('awgModule/device', self._device)
        # self._awgModule.execute()

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
            f = open(self._s_file_name).read()
            s_node_pars = json.loads(f)
        except Exception:
            print("parameter file for gettable parameters {} not found".format(
                self._s_file_name))
            init = False
        try:
            f = open(self._d_file_name).read()
            d_node_pars = json.loads(f)
        except Exception:
            print("parameter file for settable parameters {} not found".format(
                self._d_file_name))
            init = False

        self.add_parameter('timeout', unit='s',
                           initial_value=10,
                           parameter_class=ManualParameter)
        for parameter in s_node_pars:
            parname = parameter[0].replace("/", "_")
            if parameter[1] == 'float':
                self.add_parameter(
                    parname,
                    parameter_class=ManualParameter,
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1] == 'float_small':
                self.add_parameter(
                    parname,
                    parameter_class=ManualParameter,
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1] == 'int_8bit':
                self.add_parameter(
                    parname,
                    parameter_class=ManualParameter,
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            elif parameter[1] == 'int':
                self.add_parameter(
                    parname,
                    parameter_class=ManualParameter,
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            elif parameter[1] == 'int_64':
                self.add_parameter(
                    parname,
                    parameter_class=ManualParameter,
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            elif parameter[1] == 'bool':
                self.add_parameter(
                    parname,
                    parameter_class=ManualParameter,
                    vals=vals.Ints(int(parameter[2]), int(parameter[3])))
            else:
                print("parameter {} type {} from from s_node_pars not recognized".format(
                    parname, parameter[1]))

        for parameter in d_node_pars:
            parname = parameter[0].replace("/", "_")
            if parameter[1] == 'float':
                self.add_parameter(parname, parameter_class=ManualParameter)
            elif parameter[1] == 'vector_g':
                self.add_parameter(parname, parameter_class=ManualParameter)
            elif parameter[1] == 'vector_s':
                self.add_parameter(
                    parname,
                    parameter_class=ManualParameter,
                    vals=vals.Anything())
            elif parameter[1] == 'vector_gs':
                self.add_parameter(
                    parname,
                    parameter_class=ManualParameter,
                    vals=vals.Anything())
            else:
                print("parameter {} type {} from d_node_pars not recognized".format(
                    parname, parameter[1]))

        self.add_parameter('AWG_file',
                           parameter_class=ManualParameter,
                           vals=vals.Anything())
        # storing an offset correction parameter for all weight functions,
        # this allows normalized calibration when performing cross-talk suppressed
        # readout
        for i in range(5):
            self.add_parameter("quex_trans_offset_weightfunction_{}".format(i),
                               unit='',  # unit is adc value
                               label='RO normalization offset',
                               initial_value=0.0,
                               parameter_class=ManualParameter)
        if init:
            self.load_default_settings()
        t1 = time.time()

        print('Initialized dummy UHFQC', self.name,
              'in %.2fs' % (t1-t0))

    def load_default_settings(self, upload_sequence=True):
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
        if upload_sequence:
            self.awg_sequence_acquisition()

        # Turn on both outputs
        self.sigouts_0_on(1)
        self.sigouts_1_on(1)

        # QuExpress thresholds on DIO (mode == 2), AWG control of DIO (mode ==
        # 1)
        self.dios_0_mode(1)
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

        # No rotation on the output of the weighted integration unit, i.e. take
        # real part of result
        for i in range(0, 4):
            eval('self.quex_rot_{0}_real(1.0)'.format(i))
            eval('self.quex_rot_{0}_imag(0.0)'.format(i))

        # No cross-coupling in the matrix multiplication (identity matrix)
        for i in range(0, 4):
            for j in range(0, 4):
                if i == j:
                    eval('self.quex_trans_{0}_col_{1}_real(1)'.format(i, j))
                else:
                    eval('self.quex_trans_{0}_col_{1}_real(0)'.format(i, j))

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
        self.sigouts_0_enables_0(0)
        self.sigouts_1_enables_1(0)

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
        try:
            return 1.8e9/(2**self.awgs_0_time())
        except TypeError:
            # occurs if awgs_0_time is None instead of 0
            return 1.8e9/(2**0)

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
        pass

    def awg_file(self, filename):
        pass

    def awg_string(self, sourcestring):
        pass

    def sync(self):
        pass

    def acquisition_arm(self):
        pass

    def acquisition_get(self, samples, acquisition_time=0.010,
                        timeout=0, channels=set([0, 1]), mode='rl'):
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
        Dummy version of UHFQC acquisiton poll
        """
        data = dict()
        # puts dummy data in all channels of the expected length
        # channels are labeled as integers
        channels = self._acquisition_channels
        for ch in channels:
            data[ch] = np.random.rand(samples)
        return data

    def acquisition(self, samples, acquisition_time=0.010, timeout=0,
                    channels=set([0, 1]), mode='rl'):
        self.acquisition_initialize(channels, mode)
        data = self.acquisition_poll(samples, acquisition_time, timeout)
        self.acquisition_finalize()

        return data

    def acquisition_initialize(self, channels=set([0, 1]), mode='rl'):
        self._acquisition_channels = channels

    def acquisition_finalize(self):
        pass

    def prepare_SSB_weight_and_rotation(self, IF,
                                        weight_function_I=0,
                                        weight_function_Q=1):
        """
        Sets defualt integration weights for SSB modulation, beware does not
        load pulses or prepare the UFHQC progarm to do data acquisition
        """
        trace_length = 4096
        tbase = np.arange(0, trace_length/1.8e9, 1/1.8e9)
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

    def prepare_DSB_weight_and_rotation(
            self, IF, weight_function_I=0, weight_function_Q=1):
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

    def seti(self, path, value, asynchronous=False):
        if asynchronous:
            func = self._daq.asyncSetInt
        else:
            func = self._daq.setInt

        func(self._make_full_path(path), int(value))

    def setd(self, path, value, asynchronous=False):
        if asynchronous:
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
            'const WINT_EN   = 0x1f0000;\n' +
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
            'while (1) {\n' +
            ' waitDIOTrigger();\n' +
            ' var dio = getDIOTriggered();\n' +
            ' cw = (dio >> 17) & 0x1f;\n' +
            '  switch(cw) {\n')
        # adding the case statements
        for i in range(len(Iwaves)):
            # generating the case statement string
            case = '  case {}:\n'.format(cases[i])
            case_play = '   playWave(Iwave{}, Qwave{});\n'.format(i, i)
            # adding the individual case statements to the sequence
            sequence = sequence + case+case_play

        # adding the final part of the sequence including a default wave
        sequence = (sequence +
                    '  default:\n' +
                    '   playWave(ones(36), ones(36));\n' +
                    ' }\n' +
                    ' wait(wait_delay);\n' +
                    ' setTrigger(WINT_EN + RO_TRIG);\n' +
                    ' setTrigger(WINT_EN);\n' +
                    #' waitWave();\n'+ #removing this waitwave for now
                    '}\n' +
                    'wait(300);\n' +
                    'setTrigger(0);\n')
        self.awg_string(sequence)

    def awg_sequence_acquisition_and_pulse(self, Iwave, Qwave, acquisition_delay):
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
        delay_samples = int(acquisition_delay*1.8e9/8)
        delay_string = '\twait(wait_delay);\n'
        self.awgs_0_userregs_2(delay_samples)

        preamble = """
const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x1f0000;
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
\twaitDigTrigger(1, 1);
\tplayWave(Iwave, Qwave);\n"""

        end_string = """
\tsetTrigger(WINT_EN + RO_TRIG);
\tsetTrigger(WINT_EN);
\twaitWave();
}
wait(300);
setTrigger(0);"""

        string = preamble+wave_I_string+wave_Q_string + \
            loop_start+delay_string+end_string
        self.awg_string(string)

    def array_to_combined_vector_string(self, array, name):
        # this function cuts up arrays into several vectors of maximum length 1024 that are joined.
        # this is to avoid python crashes (was found to crash for vectors of
        # lenght> 1490)
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
            string = 'wave ' + name + ' = join(' + string + ');\n'
        else:
            string = 'wave ' + name + ' = ' + string + ';\n'
        return string

    def awg_sequence_acquisition(self):
        string = """
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

    def awg_sequence_acquisition_and_pulse_SSB(
            self, f_RO_mod, RO_amp, RO_pulse_length, acquisition_delay):
        f_sampling = 1.8e9
        samples = RO_pulse_length*f_sampling
        array = np.arange(int(samples))
        sinwave = RO_amp*np.sin(2*np.pi*array*f_RO_mod/f_sampling)
        coswave = RO_amp*np.cos(2*np.pi*array*f_RO_mod/f_sampling)
        Iwave = (coswave+sinwave)/np.sqrt(2)
        Qwave = (coswave-sinwave)/np.sqrt(2)
        # Iwave, Qwave = PG.mod_pulse(np.ones(samples), np.zeros(samples), f=f_RO_mod, phase=0, sampling_rate=f_sampling)
        self.awg_sequence_acquisition_and_pulse(
            Iwave, Qwave, acquisition_delay)

    def upload_transformation_matrix(self, matrix):
        for i in range(np.shape(matrix)[0]):  # looping over the rows
            for j in range(np.shape(matrix)[1]):  # looping over the colums
                eval(
                    'self.quex_trans_{}_col_{}_real(matrix[{}][{}])'.format(j, i, i, j))

    def download_transformation_matrix(self, nr_rows=4, nr_cols=4):
        matrix = np.zeros([nr_rows, nr_cols])
        for i in range(np.shape(matrix)[0]):  # looping over the rows
            for j in range(np.shape(matrix)[1]):  # looping over the colums
                matrix[i][j] = (
                    eval('self.quex_trans_{}_col_{}_real()'.format(j, i)))
                # print(value)
                # matrix[i,j]=value
        return matrix
