import logging
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.detector_functions as det
from pycqed.analysis.analysis_toolbox import rotate_and_normalize_data
from qcodes.instrument.parameter import ManualParameter
import pycqed.analysis.measurement_analysis as ma
import numpy as np
import os
import json


# this is here for backwards compatibility purposes, there is only one QASM swf
QASM_Sweep = swf.QASM_Sweep


class ASM_Sweep(swf.Hard_Sweep):

    def __init__(self, filename, CBox, upload=True):
        super().__init__()
        self.name = 'ASM'
        self.filename = filename
        self.upload = upload
        self.CBox = CBox

    def prepare(self, **kw):
        self.CBox.trigger_source('internal')
        if self.upload:
            self.CBox.load_instructions(self.filename)


class CBox_integrated_average_detector_CC(det.Hard_Detector):

    def __init__(self, CBox, seg_per_point=1,
                 nr_averages=1024, integration_length=1e-6,
                 cal_pts=None, **kw):
        '''
        Integration average detector.
        Defaults to averaging data in a number of segments equal to the
        nr of sweep points specificed.

        seg_per_point allows you to use more than 1 segment per sweeppoint.
        this is for example useful when doing a MotzoiXY measurement in which
        there are 2 datapoints per sweep point.
        Normalize/Rotate adds a third measurement with the rotated/normalized data.
        '''
        super().__init__(**kw)
        self.CBox = CBox
        self.name = 'CBox_integrated_average_detector_CC'
        self.cal_pts = cal_pts
        if self.cal_pts == None:
            self.value_names = ['I', 'Q']
            self.value_units = ['a.u.', 'a.u.']
        else:
            self.value_names = ['F']
            self.value_units = ['|1>']
        self.seg_per_point = seg_per_point
        self.nr_averages = nr_averages
        self.integration_length = integration_length

    def get_values(self):
        succes = False
        i = 0
        while not succes:
            try:
                self.CBox.set('run_mode', 'idle')
                self.CBox.core_state('idle')
                self.CBox.core_state('active')
                self.CBox.set('acquisition_mode', 'idle')
                self.CBox.set('acquisition_mode', 'integration averaging')
                self.CBox.set('run_mode', 'run')
                # does not restart AWG tape in CBox as we don't use it anymore
                data = self.CBox.get_integrated_avg_results()
                succes = True
            except Exception as e:
                logging.warning('Exception caught retrying')
                logging.warning(e)
            i += 1
            if i > 20:
                break
        if self.cal_pts:
            data = rotate_and_normalize_data(data, zero_coord=self.cal_pts[0],
                                             one_coord=self.cal_pts[1])[0]
        return data

    def acquire_data_point(self):
        return self.get_values()

    def prepare(self, sweep_points=[0]):
        self.CBox.set('nr_samples', self.seg_per_point*len(sweep_points))
        self.CBox.nr_averages(int(self.nr_averages))
        self.CBox.integration_length(int(self.integration_length/(5e-9)))

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')


class CBox_single_integration_average_det_CC(
        CBox_integrated_average_detector_CC):

    '''
    Detector used for acquiring single points of the CBox
    Soft version of the regular integrated avg detector.
    '''

    def __init__(self, CBox, seg_per_point=1,
                 nr_averages=1024, integration_length=1e-6,
                 cal_pts=None, **kw):
        super().__init__(CBox, seg_per_point,
                         nr_averages, integration_length, cal_pts, **kw)
        if self.seg_per_point == 1:
            self.detector_control = 'soft'
        else:  # this is already implicitly the case through inheritance
            self.detector_control = 'hard'

    def prepare(self, sweep_points=[0]):
        self.CBox.nr_samples(self.seg_per_point)
        self.CBox.nr_averages(int(self.nr_averages))
        self.CBox.integration_length(int(self.integration_length/(5e-9)))

    def acquire_data_point(self, **kw):
        return self.get_values()


class CBox_integration_logging_det_CC(det.Hard_Detector):

    def __init__(self, CBox, **kw):
        '''
        If you want AWG reloading you should give a LutMan and specify
        on what AWG nr to reload default is no reloading of pulses.
        '''
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_integration_logging_detector'
        self.value_names = ['I shots', 'Q shots']
        self.value_units = ['a.u.', 'a.u.']

    def get_values(self):
        succes = False
        i = 0
        while not succes:
            try:
                self.CBox.set('run_mode', 'idle')
                self.CBox.core_state('idle')
                self.CBox.core_state('active')
                self.CBox.set('acquisition_mode', 'idle')
                self.CBox.set('acquisition_mode', 'integration logging')
                self.CBox.set('run_mode', 'run')
                # does not restart AWG tape in CBox as we don't use it anymore
                data = self.CBox.get_integration_log_results()
                succes = True
            except Exception as e:
                logging.warning('Exception caught retrying')
                logging.warning(e)
            i += 1
            if i > 20:
                break
        return data

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')

class CBox_state_counters_det_CC(det.Hard_Detector):

    def __init__(self, CBox, **kw):
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_state_counters_detector'
        # A and B refer to the counts for the different weight functions
        self.value_names = ['no error A', 'single error A', 'double error A',
                            '|0> A', '|1> A',
                            'no error B', 'single error B', 'double error B',
                            '|0> B', '|1> B', ]
        self.value_units = ['#']*10

    def acquire_data_point(self):
        return self.get_values()

    def get_values(self):
        succes = False
        i = 0
        while not succes:
            try:
                self.CBox.set('run_mode', 'idle')
                self.CBox.core_state('idle')
                self.CBox.core_state('active')
                self.CBox.set('acquisition_mode', 'idle')
                self.CBox.set('acquisition_mode', 'integration logging')
                self.CBox.set('run_mode', 'run')
                # does not restart AWG tape in CBox as we don't use it anymore
                data = self.CBox.get_qubit_state_log_counters()
                succes = True
            except Exception as e:
                logging.warning('Exception caught retrying')
                logging.warning(e)
            i += 1
            if i > 20:
                break
        return np.concatenate(data)  # concatenates counters A and B

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')


class CBox_single_qubit_frac1_counter_CC(CBox_state_counters_det_CC):

    '''
    Based on the shot counters, returns the fraction of shots that corresponds
    to a specific state.
    Note that this is not corrected for RO-fidelity.

    Also note that depending on the side of the RO the F|1> and F|0> could be
    inverted
    '''

    def __init__(self, CBox):
        super().__init__(CBox=CBox)
        self.detector_control = 'soft'
        # self.CBox = CBox
        self.name = 'CBox_single_qubit_frac1_counter'
        # A and B refer to the counts for the different weight functions
        self.value_names = ['Frac_1']
        self.value_units = ['']

    def acquire_data_point(self):
        d = super().acquire_data_point()
        data = d[4]/(d[3]+d[4])
        return data

class CBox_err_frac_CC(CBox_state_counters_det_CC):

    '''
    Child of the state counters detector
    Returns fraction of event type s by using state counters 1 and 2
    Rescales the measured counts to percentages.
    '''

    def __init__(self, CBox):
        super().__init__(CBox=CBox)
        self.detector_control = 'soft'
        self.name = 'CBox_err_frac_CC'
        self.value_names = ['frac. err.']
        self.value_units = ['']

    def prepare(self, **kw):
        self.nr_shots = self.CBox.log_length.get()

    def acquire_data_point(self):
        d = super().acquire_data_point()
        data = d[1]/self.nr_shots*100
        return data


class CBox_single_qubit_event_s_fraction_CC(CBox_state_counters_det_CC):

    '''
    Child of the state counters detector
    Returns fraction of event type s by using state counters 1 and 2
    Rescales the measured counts to percentages.
    '''

    def __init__(self, CBox):
        super().__init__(CBox=CBox)
        self.detector_control = 'soft'
        self.name = 'CBox_single_qubit_event_s_fraction'
        self.value_names = ['frac. err.', 'frac. 2 or more', 'frac. event s']
        self.value_units = ['%', '%', '%']

    def prepare(self, **kw):
        self.nr_shots = self.CBox.log_length.get()

    def acquire_data_point(self):
        d = super().acquire_data_point()
        # data = d[1]/(d[1]+d[2])
        data = [
            d[1]/self.nr_shots*100,
            d[2]/self.nr_shots*100,
            (d[1]-d[2])/self.nr_shots*100]
        return data



class CBox_int_avg_func_prep_det_CC(CBox_integrated_average_detector_CC):

    def __init__(self, CBox, prepare_function, prepare_function_kwargs=None,
                 seg_per_point=1,
                 nr_averages=1024, integration_length=1e-6,
                 cal_pts=None, **kw):
        '''
        int avg detector with additional option to run a specific function
        everytime the prepare is called.
        '''
        super().__init__(CBox, seg_per_point, nr_averages,
                         integration_length, cal_pts, **kw)
        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs

    def prepare(self, sweep_points=[0]):
        super().prepare(sweep_points)
        if self.prepare_function_kwargs is not None:
            self.prepare_function(**self.prepare_function_kwargs)
        else:
            self.prepare_function()


class UHFQC_int_avg_func_prep_det_CC(det.UHFQC_integrated_average_detector):

    def __init__(self, prepare_function, prepare_function_kwargs=None,
                 **kw):
        '''
        int avg detector with additional option to run a specific function
        everytime the prepare is called.
        '''
        super().__init__(**kw)
        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs

    def prepare(self, sweep_points=[0]):
        super().prepare(sweep_points)
        if self.prepare_function_kwargs is not None:
            self.prepare_function(**self.prepare_function_kwargs)
        else:
            self.prepare_function()


def load_range_of_asm_files(asm_filenames, counter_param, CBox):
    asm_filename = asm_filenames[counter_param()]
    counter_param((counter_param()+1) % len(asm_filenames))
    CBox.load_instructions(asm_filename)


def extract_msmt_pts_from_config(config_filename):
    """
    This is a dummy version that
    """
    config_opened = open(config_filename, 'r')
    config_dict = json.load(config_opened)
    config_opened.close()
    return config_dict['measurement_points']


def measure_asm_files(asm_filenames, config_filename, qubit, MC):
    """
    Takes one or more asm_files as input and runs them on the hardware
    """
    qubit.prepare_for_timedomain()
    CBox = qubit.CBox  # The qubit object contains a reference to the CBox

    counter_param = ManualParameter('name_ctr', initial_value=0)

    if len(asm_filenames) > 1:
        MC.soft_avg(len(asm_filenames))
        nr_hard_averages = 256
    else:
        MC.soft_avg(8)
        nr_hard_averages = qubit.RO_acq_averages()//MC.soft_avg()

    if qubit.cal_pt_zero() is not None:
        cal_pts = (qubit.cal_pt_zero(), qubit.cal_pt_one())
    else:
        cal_pts = None

    prepare_function_kwargs = {
        'counter_param': counter_param,  'asm_filenames': asm_filenames,
        'CBox': CBox}

    detector = CBox_int_avg_func_prep_det_CC(
        CBox, prepare_function=load_range_of_asm_files,
        prepare_function_kwargs=prepare_function_kwargs,
        nr_averages=nr_hard_averages,
        cal_pts=cal_pts)

    measurement_points = extract_msmt_pts_from_config(config_filename)

    s = swf.None_Sweep()
    if 'rb' in asm_filenames[0]:
        s.parameter_name = 'Number of Cliffords'
        s.unit = '#'
    MC.set_sweep_function(s)
    MC.set_sweep_points(measurement_points)
    MC.set_detector_function(detector)

    MC.run('Demo {}'.format(os.path.split(asm_filenames[0])[-1]))
    if 'rb' in asm_filenames[0]:
        a = ma.RandomizedBenchmarking_Analysis(
               label='Demo rb', rotate_and_normalize=False, close_fig=True)
        p = a.fit_res.best_values['p']
        Fcl = p+(1-p)/2
        Fg = Fcl**(1/1.875)
        print('Clifford Fidelity:\t{:.4f}\nGate Fidelity: \t\t{:.4f}'.format(Fcl, Fg))
        Ncl = np.arange(a.sweep_points[0], a.sweep_points[-1])
        RB_fit = ma.fit_mods.RandomizedBenchmarkingDecay(Ncl, **a.fit_res.best_values)
        MC.main_QtPlot.add(x=Ncl, y=RB_fit, subplot=0)

def simulate_qasm_files(qasm_filenames, config_filename, qxc, MC,
                        error_rate = 0, nr_averages=int(1e4)):
    """
    Takes one or more asm_files as input and runs them on the hardware
    """

    if len(qasm_filenames) > 1:
        MC.soft_avg(len(qasm_filenames))
        nr_hard_averages = 256
    else:
        nr_hard_averages = nr_averages
        MC.soft_avg(1)

    qx_det = det.QX_Hard_Detector(qxc, qasm_filenames,
                                  p_error=error_rate, num_avg=nr_hard_averages)
    measurement_points = extract_msmt_pts_from_config(config_filename)

    MC.set_sweep_function(swf.None_Sweep())
    MC.set_detector_function(qx_det)
    MC.set_sweep_points(measurement_points)
    MC.run('Demo QX sim {}'.format(os.path.split(qasm_filenames[0])[-1]))


###########

def getFilePath(filename):
    '''
    get the absolute path to the directory containing the file
    '''
    return os.path.dirname(os.path.abspath(filename))


def getBaseName(filename):
    '''
    return the basename without path and extension
    '''
    return os.path.splitext(os.path.basename(filename))[0]


def getExtension(filename):
    '''
    get the extension starting with a dot.
    '''
    return os.path.splitext(filename)[1]


def removeExtension(filename):
    '''
    remove the extension from the file name. The path is kept.
    '''
    return os.path.splitext(filename)[0]


def RepresentsInt(s):
    return s.isdigit()


def list_files_in_dir(dirpath):
    return [f for f in os.listdir(dirpath) if
            os.path.isfile(os.path.join(dirpath, f))]


def getHomoFiles(fn):
    dirpath = getFilePath(fn)
    basename = getBaseName(fn)
    ext = getExtension(fn)
    name_core, sep, tail = basename.rpartition('_')
    if name_core == '':
        name_core = tail
    homo_files = []
    if not RepresentsInt(tail):
        homo_files.append(fn)
    else:
        files_in_dir = list_files_in_dir(dirpath)
        for f in files_in_dir:
            if (getExtension(f) != ext):
                continue
            head, sep, tail = getBaseName(f).rpartition('_')
            if (RepresentsInt(tail) and (head == name_core)):
                homo_files.append(dirpath + '\\'+f)

    config_file = dirpath + "\\{}_config.json".format(name_core)

    return (config_file, homo_files)



def convert_to_clocks(duration, f_sampling=200e6, rounding_period=None):
    """
    convert a duration in seconds to an integer number of clocks

        f_sampling: 200e6 is the CBox sampling frequency
    """
    if rounding_period is not None:
        duration = (duration//rounding_period+1)*rounding_period
    clock_duration = int(duration*f_sampling)
    return clock_duration


def get_operation_dict(qubit, operation_dict={}):

    pulse_period_clocks = convert_to_clocks(
        qubit.gauss_width()*4, rounding_period=1/abs(qubit.f_pulse_mod()))
    RO_length_clocks = convert_to_clocks(qubit.RO_pulse_length())
    RO_pulse_delay_clocks = convert_to_clocks(qubit.RO_pulse_delay())

    operation_dict['init_all'] = {'instruction':
                                  'WaitReg r0 \nWaitReg r0 \n'}
    operation_dict['I {}'.format(qubit.name)] = {
        'duration': pulse_period_clocks, 'instruction': 'wait {} \n'}
    operation_dict['X180 {}'.format(qubit.name)] = {
        'duration': pulse_period_clocks, 'instruction':
            'trigger 0000000, 1 \nwait 1\n'+
            'trigger 1000001, 1  \nwait {}\n'.format( #1001001
                pulse_period_clocks-1)}
    operation_dict['Y180 {}'.format(qubit.name)] = {
        'duration': pulse_period_clocks, 'instruction':
            'trigger 0100000, 1 \nwait 1\n'+
            'trigger 1100001, 1  \nwait {}\n'.format(
                pulse_period_clocks-1)}
    operation_dict['X90 {}'.format(qubit.name)] = {
        'duration': pulse_period_clocks, 'instruction':
            'trigger 0010000, 1 \nwait 1\n'+
            'trigger 1010000, 1  \nwait {}\n'.format(
                pulse_period_clocks-1)}
    operation_dict['Y90 {}'.format(qubit.name)] = {
        'duration': pulse_period_clocks, 'instruction':
            'trigger 0110000, 1 \nwait 1\n'+
            'trigger 1110000, 1  \nwait {}\n'.format(
                pulse_period_clocks-1)}
    operation_dict['mX90 {}'.format(qubit.name)] = {
        'duration': pulse_period_clocks, 'instruction':
            'trigger 0001000, 1 \nwait 1\n'+
            'trigger 1001000, 1  \nwait {}\n'.format(
                pulse_period_clocks-1)}
    operation_dict['mY90 {}'.format(qubit.name)] = {
        'duration': pulse_period_clocks, 'instruction':
            'trigger 0101000, 1 \nwait 1\n'+
            'trigger 1101000, 1  \nwait {}\n'.format(
                pulse_period_clocks-1)}

    if qubit.RO_pulse_type() == 'MW_IQmod_pulse':
        operation_dict['RO {}'.format(qubit.name)] = {
            'duration': RO_length_clocks,
            'instruction': 'wait {} \npulse 0000 1111 1111 '.format(
                RO_pulse_delay_clocks)
            + '\nwait {} \nmeasure \n'.format(RO_length_clocks)}
    elif qubit.RO_pulse_type() == 'Gated_MW_RO_pulse':
        operation_dict['RO {}'.format(qubit.name)] = {
            'duration': RO_length_clocks, 'instruction':
            'wait {} \ntrigger 1000000, {} \n measure \n'.format(
                RO_pulse_delay_clocks, RO_length_clocks)}

    return operation_dict
