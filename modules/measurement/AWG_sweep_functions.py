import numpy as np
import logging
from modules.measurement import sweep_functions as swf

from modules.measurement.pulse_sequences import standard_sequences as st_seqs
default_gauss_width = 10


class CBox_T1(swf.Hard_Sweep):
    def __init__(self, IF, meas_pulse_delay, RO_trigger_delay, mod_amp, AWG,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.meas_pulse_delay = meas_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.parameter_name = 'tau'
        self.unit = 's'
        self.name = 'T1'
        self.AWG = AWG
        self.mod_amp = mod_amp
        self.upload = upload

    def prepare(self, **kw):
        if self.upload:
            st_seqs.CBox_T1_marker_seq(IF=self.IF, times=self.sweep_points,
                                       meas_pulse_delay=self.meas_pulse_delay,
                                       RO_trigger_delay=self.RO_trigger_delay,
                                       verbose=False)
            self.AWG.set('ch3_amp', self.mod_amp)
            self.AWG.set('ch4_amp', self.mod_amp)


class CBox_OffOn(swf.Hard_Sweep):
    def __init__(self, IF, meas_pulse_delay, RO_trigger_delay, mod_amp,
                 AWG, CBox,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.meas_pulse_delay = meas_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.parameter_name = 'Tape elment'
        self.unit = ''
        self.name = 'Off-On'
        self.tape = [0, 1]
        self.sweep_points = np.array(self.tape)  # array for transpose in MC
        self.AWG = AWG
        self.CBox = CBox
        self.mod_amp = mod_amp
        # would actually like to check if file is already loaded
        # filename can be get using AWG.get('setup_filename')
        self.upload = upload

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Tape')
        self.CBox.AWG1_mode.set('Tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)

        if self.upload:
            st_seqs.CBox_single_pulse_seq(
                IF=self.IF,
                meas_pulse_delay=self.meas_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay, verbose=False)
            self.AWG.set('ch3_amp', self.mod_amp)
            self.AWG.set('ch4_amp', self.mod_amp)


# class AWG_Sweep(swf.Hard_Sweep):
#     def __init__(self, Duplexer=False, **kw):
#         self.sweep_control = 'hard'
#         self.TD_Meas = qt.instruments['TD_Meas']
#         self.AWG = qt.instruments['AWG']
#         self.AWG_model = self.TD_Meas.get_AWG_model()
#         self.Duplexer = Duplexer

#         # Optionally use different filename than default
#         self.filename = kw.pop('filename', self.filename)
#         if not hasattr(self, 'cal_points'):
#             default_cal_points = 10 if not self.TD_Meas.get_multiplex() else 12
#             self.cal_points = kw.pop('cal_points', default_cal_points)
#         if kw.pop('add_filename_tags', True):
#             if self.AWG_model == 'AWG5014':
#                 self.filename = self.filename + '_5014'
#             elif self.AWG_model == 'AWG520':
#                 self.filename = self.filename + '_520'
#             if self.Duplexer is True:
#                 self.filename = 'Duplex_' + self.filename

#     def prepare(self, **kw):
#         if not hasattr(self, 'sweep_points'):
#             self.TD_Meas.set_NoSegments(0)
#         else:
#             self.TD_Meas.set_NoSegments(len(self.sweep_points))

#         self.TD_Meas.set_AWG_seq_filename(self.filename)
#         # TD_Meas stuff should be removed from here to prevent convolution
#         self.AWG.set_setup_filename(self.filename,
#                                     force_load=False)

#         self.TD_Meas.set_cal_points(self.cal_points)
#         if not hasattr(self, 'sweep_points'):
#             self.sweep_points = np.linspace(1, self.TD_Meas.get_NoSegments(),
#                                             self.TD_Meas.get_NoSegments())


# class Rabi(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.name = 'Rabi'
#         self.parameter_name = 'amplitude'
#         self.unit = 'arb unit'
#         self.sweep_points = np.linspace(1, 70, 70)
#         self.filename = 'Rabi%s_%i' % (qubit_suffix, gauss_width)
#         super(self.__class__, self).__init__(**kw)


# class T1(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", stepsize=100,
#                  NoSegments=70, **kw):
#         self.name = 'T1'
#         self.parameter_name = 'time'
#         self.unit = 'ns'
#         self.available_stepsizes = [50, 100, 200, 500, 1000, 1500, 2000, 3000]
#         self.filename = 'T1%s_%i_%i' % (qubit_suffix, stepsize, gauss_width)
#         self.sweep_points = np.linspace(stepsize, NoSegments*stepsize, NoSegments)

#         if stepsize not in self.available_stepsizes:
#             logging.error('Stepsize not available')
#         super(self.__class__, self).__init__(**kw)


# class Ramsey(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", stepsize=100, **kw):
#         self.name = kw.pop('name', 'Ramsey')
#         self.parameter_name = 'time'
#         self.unit = 'ns'
#         self.available_stepsizes = [5, 10, 30, 100, 200, 500, 1000, 1500, 2000]
#         self.filename = 'Ramsey%s_%i_%i' % (qubit_suffix, stepsize, gauss_width)
#         self.sweep_points = (np.linspace(stepsize, 70*stepsize, 70) + 30)

#         if stepsize not in self.available_stepsizes:
#             logging.error('Stepsize not available')
#         super(self.__class__, self).__init__(**kw)


# class Echo(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", stepsize=100, **kw):
#         self.sweep_control = 'hard'
#         self.name = 'Echo'
#         self.parameter_name = 'time'
#         self.unit = 'ns'

#         self.filename = 'Echo%s_%i_%i' % (qubit_suffix, stepsize, gauss_width)
#         self.sweep_points = np.linspace(stepsize, 70*stepsize, 70)

#         self.available_stepsizes = [100, 200, 500, 1000, 1500, 2000]
#         if stepsize not in self.available_stepsizes:
#             logging.error('Stepsize not available')
#         super(self.__class__, self).__init__(**kw)


# class AllXY(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", subsequence_sufix='',
#                  sequence="AllXY", prepi=False, **kw):
#         self.sweep_control = 'hard'
#         self.name = sequence
#         self.parameter_name = sequence
#         self.unit = 'Arb.unit'
#         prepi_str=""
#         if prepi:
#             prepi_str="_pi"
#         else:
#             prepi_str=""

#         if sequence is "AllXY":
#             self.sweep_points = np.linspace(1, 42, 42)
#             self.cal_points = [list(range(2)), list(range(-8, -4))] # should be n analysis
#             self.ideal_data = np.concatenate((np.zeros(10), 0.5*np.ones(24),
#                                              np.ones(8)))
#         elif sequence is "AllXY21":
#             self.sweep_points = np.linspace(1, 21, 21)
#             self.cal_points = [list(range(1)), list(range(-4, -2))] # should be in analysis
#             self.ideal_data = np.concatenate((np.zeros(5), 0.5*np.ones(12),
#                                              np.ones(4)))
#         elif sequence is "AllXYNiels":
#             self.sweep_points = np.linspace(1, 11, 11)
#             self.cal_points = [list(range(1)), list(range(-4, -2))] # should be n analysis
#             self.ideal_data = np.concatenate((np.zeros(3), 0.5*np.ones(4),
#                                              np.ones(4)))
#         elif sequence is "AllXYReed":
#             self.sweep_points = np.linspace(1, 5, 5)
#             self.cal_points = [list(range(1)), list(range(-2, 0))] # should be n analysis
#             self.ideal_data = np.concatenate((np.zeros(1), 0.5*np.ones(2),
#                                              np.ones(2)))
#         elif sequence is "AllXYshort":
#             self.sweep_points = np.linspace(1, 5, 5)
#             self.cal_points = [list(range(1)), list(range(-2, 0))] # should be n analysis
#             self.ideal_data = np.concatenate((np.zeros(1), 0.5*np.ones(2),
#                                              np.ones(2)))

#         elif sequence is "AllXYAdriaan1":
#             self.sweep_points = np.linspace(1, 5, 5)
#             self.cal_points = [list(range(1)), list(range(-2, 0))] # should be n analysis
#             self.ideal_data = np.concatenate((np.zeros(3),  np.ones(2)))

#         elif sequence is "AllXYAdriaan2":
#             self.sweep_points = np.linspace(1, 5, 5)
#             self.cal_points = [list(range(1)), list(range(-2, 0))] # should be n analysis
#             self.ideal_data = np.concatenate((np.zeros(1),  np.ones(4)))
#         else:
#             raise Exception('Invalid sequence "%s" specified' % sequence)
#         self.filename = '%s%s%s%s_%i' % (sequence, prepi_str, qubit_suffix,
#                                        subsequence_sufix, gauss_width)
#         print("filename", self.filename)
#         super(self.__class__, self).__init__(**kw)


# class Off(AWG_Sweep):
#     def __init__(self, **kw):
#         self.sweep_control = 'hard'
#         self.name = 'Off'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.sweep_points = np.linspace(1, 100, 100)
#         self.filename = 'Off'
#         super(Off, self).__init__(**kw)


# class RB(AWG_Sweep):
#     def __init__(self, seed, num_Cliffords=700, mode="", gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'RB'
#         self.parameter_name = 'RB'
#         self.unit = 'Arb.unit'
#         self.filename = 'RB%s_%i_%i_%i' % (mode, num_Cliffords, seed, gauss_width)
#         super(self.__class__, self).__init__(**kw)

# class RBalternating(AWG_Sweep):
#     def __init__(self, seed, num_Cliffords=400, gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'RB_alternating'
#         self.parameter_name = 'RB_alternating'
#         self.unit = 'Arb.unit'
#         self.filename = 'alternating%s_%i_%i' % (qubit_suffix, gauss_width, seed)
#         super(self.__class__, self).__init__(**kw)


# class RBfixed(AWG_Sweep):
#     ''' Performs Randomized Benchmarking with a fixed number of Clifford
#         gates.
#     '''
#     def __init__(self, num_Cliffords, gauss_width=default_gauss_width, qubit_suffix="",
#                  **kw):
#         self.sweep_control = 'hard'
#         self.name = 'RBfixed'
#         self.parameter_name = 'RBfixed'
#         self.unit = 'Arb.unit'
#         self.filename = 'RBfixed%s%i_%i' % (qubit_suffix, num_Cliffords, gauss_width)
#         super(self.__class__, self).__init__(**kw)


# class RBinterleaved(AWG_Sweep):
#     ''' Performs Randomized Benchmarking with a fixed number of Clifford
#         gates.
#     '''
#     def __init__(self, seed, interleaved_Clifford, num_Cliffords,
#                  gauss_width=default_gauss_width, qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'RBinterleaved'
#         self.parameter_name = 'RBinterleaved'
#         self.unit = 'Arb.unit'
#         self.filename = 'RB%s%i_%s_%i_%i' % (qubit_suffix,
#                                              num_Cliffords,
#                                              interleaved_Clifford,
#                                              gauss_width,
#                                              seed)
#         super(self.__class__, self).__init__(**kw)


# class OnOff(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", nr_segments=100, nr_measurements=1,
#                  long_meas=False, **kw):
#         self.sweep_control = 'hard'
#         self.name = 'OnOff'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.sweep_points = np.linspace(1, nr_segments, nr_segments)
#         self.filename = 'OnOff%s_%i' % (qubit_suffix, gauss_width)
#         if long_meas:
#             self.filename = 'OnOff_longmeas%s_%i' % (qubit_suffix, gauss_width)

#         if nr_measurements == 2:
#             self.name = 'OnOff_2'
#             self.filename = 'OnOff_2%s_%i' % (qubit_suffix, gauss_width)
#         self.cal_points = [list(range(0, nr_segments, 2)), list(range(1, nr_segments, 2))]
#         super(OnOff, self).__init__(**kw)


# class FPGA_OnOff(AWG_Sweep):
#     '''
#     Same as regular on off sequence except it uses the CBox for triggering
#     '''
#     def __init__(self, **kw):
#         self.sweep_control = 'hard'
#         self.name = 'FPGA_OnOff'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.sweep_points = np.linspace(1, 2, 2)
#         self.filename = 'FPGA_OnOff_25'
#         super(FPGA_OnOff, self).__init__(**kw)


# class DragDetuning(AWG_Sweep):
#     ''' This pulse sends alternatingly XpY90 and YpX90 pulses. The difference
#         is an indication of how far off the Drag parameter is (assuming that
#         the mixer is properly calibrated
#     '''
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'DragDetuning'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.sweep_points = np.linspace(1, 50, 50)
#         self.cal_points = [list(range(0, 50, 2)), list(range(1, 50, 2))]
#         self.filename = 'XpY90_YpX90%s_%i' % (qubit_suffix, gauss_width)
#         super(DragDetuning, self).__init__(**kw)


# # Fine-tuning driving amplitude
# class PiHalfX180(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'PiHalfX180'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.filename = 'PiHalfX180%s_%i' % (qubit_suffix, gauss_width)
#         super(self.__class__, self).__init__(**kw)


# class PiHalfX360(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'PiHalfX360'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.filename = 'PiHalfX360%s_%i' % (qubit_suffix, gauss_width)
#         super(self.__class__, self).__init__(**kw)


# class PiX360(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'PiX360'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.sweep_points = np.linspace(1, 60, 60)
#         self.filename = 'PiX360%s_%i' % (qubit_suffix, gauss_width)
#         super(self.__class__, self).__init__(**kw)


# class PiHalfY180(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'PiHalfY180'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.filename = 'PiHalfY180%s_%i' % (qubit_suffix, gauss_width)
#         super(self.__class__, self).__init__(**kw)


# class PiHalfY360(AWG_Sweep):
#     def __init__(self, gauss_width=default_gauss_width,
#                  qubit_suffix="", **kw):
#         self.sweep_control = 'hard'
#         self.name = 'PiHalfY360'
#         self.parameter_name = 'repetitions'
#         self.unit = 'Arb.unit'
#         self.filename = 'PiHalfY360%s_%i' % (qubit_suffix, gauss_width)
#         super(self.__class__, self).__init__(**kw)


# class AWG_Sweep_File(AWG_Sweep):
#     def __init__(self, filename, NoSegments=None,
#                  gauss_width=default_gauss_width,
#                  add_filename_tags=True, **kw):
#         self.sweep_control = 'hard'
#         self.filename = filename
#         if add_filename_tags:
#             self.filename += '_%i' % gauss_width
#         self.name = filename
#         if NoSegments:
#             self.sweep_points = np.linspace(1, NoSegments, NoSegments)
#         self.parameter_name = 'Pulse Number'
#         self.unit = 'Arb.unit'
#         # self.TD_Meas.set_NoSegments(len(self.sweep_points))
#         super(AWG_Sweep_File, self).__init__(
#             add_filename_tags=add_filename_tags, **kw)


# class ILoveLeo(AWG_Sweep):
#     def __init__(self, **kw):
#         self.sweep_control = 'hard'
#         self.sweep_points = np.arange(0, 60, 1)*10
#         self.name = 'ILoveLeo'
#         self.parameter_name = 'time'
#         self.unit = 'ns'
#         self.filename = 'ILoveLeo'
#         super(self.__class__, self).__init__(**kw)
