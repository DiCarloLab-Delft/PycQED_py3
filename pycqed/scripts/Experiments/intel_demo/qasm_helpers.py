import logging
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.detector_functions as det
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta


class QASM_Sweep(swf.Hard_Sweep):

    def __init__(self, filename, CBox, op_dict, upload=True):
        super().__init__()
        self.name = 'QASM_Sweep'
        self.filename = filename
        self.upload = upload
        self.CBox = CBox
        self.op_dict = op_dict

    def prepare(self, **kw):
        self.CBox.trigger_source('internal')
        if self.upload:
            asm_file = qta.qasm_to_asm(self.filename, self.op_dict)
            self.CBox.load_instructions(asm_file.name)


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
                 nr_averages=1024, integration_length=1e-6, **kw):
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
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
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
                 nr_averages=1024, integration_length=1e-6, **kw):
        super().__init__(CBox, seg_per_point,
                         nr_averages, integration_length, **kw)
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
        self.value_names = ['I', 'Q']
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


class CBox_int_avg_func_prep_det_CC(CBox_integrated_average_detector_CC):

    def __init__(self, CBox, prepare_function, prepare_function_kwargs=None,
                 seg_per_point=1,
                 nr_averages=1024, integration_length=1e-6, **kw):
        '''
        int avg detector with additional option to run a specific function
        everytime the prepare is called.
        '''
        super().__init__(CBox, seg_per_point, nr_averages,
                         integration_length, **kw)
        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs

    def prepare(self, sweep_points=[0]):
        super().prepare(sweep_points)
        if self.prepare_function_kwargs is not None:
            self.prepare_function(**self.prepare_function_kwargs)
        else:
            self.prepare_function()


def create_CBox_op_dict(qubit_name, pulse_length=8, RO_length=50, RO_delay=10,
                        modulated_RO=True):
    operation_dict = {
        'init_all': {'instruction': 'WaitReg r0 \nWaitReg r0 \n'},
        'I {}'.format(qubit_name): {
            'duration': pulse_length, 'instruction': 'wait {} \n'},
        'X180 {}'.format(qubit_name): {
            'duration': pulse_length, 'instruction':
            'pulse 1001 0000 1001  \nwait {}\n'.format(pulse_length)},
        'Y180 {}'.format(qubit_name): {
            'duration': pulse_length, 'instruction':
            'pulse 1010 0000 1010  \nwait {}\n'.format(pulse_length)},
        'X90 {}'.format(qubit_name): {
            'duration': pulse_length, 'instruction':
            'pulse 1011 0000 1011  \nwait {}\n'.format(pulse_length)},
        'Y90 {}'.format(qubit_name): {
            'duration': pulse_length, 'instruction':
            'pulse 1100 0000 1100  \nwait {}\n'.format(pulse_length)},
        'mX90 {}'.format(qubit_name): {
            'duration': pulse_length, 'instruction':
            'pulse 1101 0000 1101  \nwait {}\n'.format(pulse_length)},
        'mY90 {}'.format(qubit_name): {
            'duration': pulse_length, 'instruction':
            'pulse 1110 0000 1110  \nwait {}\n'.format(pulse_length)}}
    if modulated_RO:
        operation_dict['RO {}'.format(qubit_name)] = {
            'duration': RO_length, 'instruction':
            'wait {} \npulse 0000 1111 1111 \nwait {} \nmeasure \n'.format(
                RO_delay, RO_length)}
    else:
        operation_dict['RO {}'.format(qubit_name)] = {
            'duration': RO_length, 'instruction':
            'wait {} \ntrigger 1000000, {} \n measure \n'.format(RO_delay,
                                                                 RO_length)}
    return operation_dict
