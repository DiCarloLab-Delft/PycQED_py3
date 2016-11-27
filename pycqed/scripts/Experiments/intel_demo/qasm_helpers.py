import logging
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.detector_functions as det
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta
from qcodes.instrument.parameter import ManualParameter
import numpy as np
import os
import json


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
        CBox.nr_averages
        nr_hard_averages = 512
        MC.soft_avg(5)

    prepare_function_kwargs = {
        'counter_param': counter_param,  'asm_filenames': asm_filenames,
        'CBox': CBox}

    detector = CBox_int_avg_func_prep_det_CC(
        CBox, prepare_function=load_range_of_asm_files,
        prepare_function_kwargs=prepare_function_kwargs,
        nr_averages=nr_hard_averages)

    measurement_points = extract_msmt_pts_from_config(config_filename)

    MC.set_sweep_function(swf.None_Sweep())
    MC.set_sweep_points(measurement_points)
    MC.set_detector_function(detector)

    MC.run('Demo {}'.format(os.path.split(asm_filenames[0])[-1]))


def simulate_qasm_files(qasm_filenames, config_filename, qxc, MC):
    """
    Takes one or more asm_files as input and runs them on the hardware
    """

    counter_param = ManualParameter('name_ctr', initial_value=0)

    MC.soft_avg(len(qasm_filenames))

    qx_det = det.QX_Hard_Detector(qxc, qasm_filenames,
                                     p_error=0.006, num_avg=128)
    measurement_points = extract_msmt_pts_from_config(config_filename)

    MC.set_sweep_function(swf.None_Sweep())
    MC.set_detector_function(qx_det)
    MC.set_sweep_points(measurement_points)
    MC.run('Demo QX sim {}'.format(os.path.split(qasm_filenames[0])[-1]))
