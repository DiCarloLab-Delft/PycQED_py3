import numpy as np
import logging
from modules.measurement import sweep_functions as swf
from modules.measurement.randomized_benchmarking import randomized_benchmarking as rb
from modules.measurement.pulse_sequences import standard_sequences as st_seqs
from modules.measurement.pulse_sequences import single_qubit_tek_seq_elts as sqs
default_gauss_width = 10  # magic number should be removed,
# note magic number only used in old mathematica seqs


class Rabi(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars, n=1, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.n = n
        self.upload = upload

        self.name = 'Rabi'
        self.parameter_name = 'amplitude'
        self.unit = 'V'

    def prepare(self, **kw):
        if self.upload:
            sqs.Rabi_seq(amps=self.sweep_points,
                         pulse_pars=self.pulse_pars,
                         RO_pars=self.RO_pars,
                         n=self.n)


class T1(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload

        self.name = 'T1'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.T1_seq(times=self.sweep_points,
                       pulse_pars=self.pulse_pars,
                       RO_pars=self.RO_pars)


class AllXY(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars, double_points=False, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.double_points = double_points
        self.upload = upload

        self.parameter_name = 'AllXY element'
        self.unit = '#'
        self.name = 'AllXY'
        if not double_points:
            self.sweep_points = np.arange(21)
        else:
            self.sweep_points = np.arange(42)

    def prepare(self, **kw):
        if self.upload:
            sqs.AllXY_seq(pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars,
                          double_points=self.double_points)


class OffOn(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload

        self.parameter_name = 'OffOn element'
        self.unit = '#'
        self.name = 'OffOn'
        self.sweep_points = np.arange(2)

    def prepare(self, **kw):
        if self.upload:
            sqs.OffOn_seq(pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars)


class Randomized_Benchmarking(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars,
                 nr_seeds, nr_cliffords,
                 cal_points=True,
                 upload=True):
        # If nr_cliffords is None it still needs to be specfied when setting
        # the experiment
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.nr_seeds = nr_seeds
        self.cal_points = cal_points
        self.sweep_points = nr_cliffords

        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'
        self.name = 'Randomized_Benchmarking'
        self.sweep_points = nr_cliffords

        if self.cal_points:
            self.sweep_points = np.concatenate([nr_cliffords,
                                   [nr_cliffords[-1]+.2,
                                    nr_cliffords[-1]+.3,
                                    nr_cliffords[-1]+.7,
                                    nr_cliffords[-1]+.8]])

    def prepare(self, **kw):
        if self.upload:
            sqs.Randomized_Benchmarking_seq(
                self.pulse_pars, self.RO_pars,
                nr_cliffords=self.sweep_points,
                nr_seeds=self.nr_seeds,
                cal_points=self.cal_points)


class Ramsey(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars,
                 artificial_detuning=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning

        self.name = 'Ramsey'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.Ramsey_seq(times=self.sweep_points,
                           pulse_pars=self.pulse_pars,
                           RO_pars=self.RO_pars,
                           artificial_detuning=self.artificial_detuning,
                           cal_points=self.cal_points)


class CBox_T1(swf.Hard_Sweep):
    def __init__(self, IF, RO_pulse_delay, RO_trigger_delay, mod_amp, AWG,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.name = 'T1'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.AWG = AWG
        self.mod_amp = mod_amp
        self.upload = upload

    def prepare(self, **kw):
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_T1_marker_seq(IF=self.IF, times=self.sweep_points,
                                       RO_pulse_delay=self.RO_pulse_delay,
                                       RO_trigger_delay=self.RO_trigger_delay,
                                       verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class CBox_Ramsey(swf.Hard_Sweep):
    def __init__(self, IF, RO_pulse_length,
                 RO_pulse_delay, RO_trigger_delay, pulse_separation,
                 AWG, CBox, cal_points=True,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.pulse_separation = pulse_separation
        self.RO_pulse_length = RO_pulse_length
        self.name = 'T2*'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.AWG = AWG
        self.CBox = CBox
        self.upload = upload
        self.cal_points = cal_points

    def prepare(self, **kw):
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_Ramsey_marker_seq(
                IF=self.IF, times=self.sweep_points,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_pulse_length=self.RO_pulse_length,
                RO_trigger_delay=self.RO_trigger_delay,
                pulse_separation=self.pulse_separation,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)

        # gets assigned in MC.set sweep_points
        nr_elts = len(self.sweep_points)
        if self.cal_points:  # append the calibration points to the tape
            tape = [3, 3] * (nr_elts-4) + [0, 0, 0, 0, 0, 1, 0, 1]
        else:
            tape = [3, 3] * nr_elts

        self.AWG.stop()
        # TODO Change to segmented tape if we have the new timing tape
        self.CBox.AWG0_mode.set('segmented tape')
        self.CBox.AWG1_mode.set('segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', tape)
        self.CBox.set('AWG1_tape', tape)


class CBox_Echo(swf.Hard_Sweep):
    def __init__(self, IF,
                 RO_pulse_delay, RO_trigger_delay, pulse_separation,
                 AWG, CBox, cal_points=True,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.pulse_separation = pulse_separation
        self.name = 'T2-echo'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.AWG = AWG
        self.CBox = CBox
        self.upload = upload
        self.cal_points = cal_points
        logging.warning('Underlying sequence is not implemented')
        logging.warning('Replace it with the multi-pulse sequence')

    def prepare(self, **kw):
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_Echo_marker_seq(
                IF=self.IF, times=self.sweep_points,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)

        # gets assigned in MC.set sweep_points
        nr_elts = len(self.sweep_points)
        if self.cal_points:
            tape = [3, 3] * (nr_elts-4) + [0, 1]
        else:
            tape = [3, 3] * nr_elts

        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', tape)
        self.CBox.set('AWG1_tape', tape)


class CBox_OffOn(swf.Hard_Sweep):
    def __init__(self, IF, RO_pulse_delay, RO_trigger_delay,
                 RO_pulse_length,
                 AWG, CBox,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.parameter_name = 'Tape element'
        self.unit = ''
        self.name = 'Off-On'
        self.tape = [0, 1]
        self.sweep_points = np.array(self.tape)  # array for transpose in MC
        self.AWG = AWG
        self.CBox = CBox
        self.RO_pulse_length = RO_pulse_length
        # would actually like to check if file is already loaded
        # filename can be get using AWG.get('setup_filename')
        self.upload = upload

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)

        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_single_pulse_seq(
                IF=self.IF,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                RO_pulse_length=self.RO_pulse_length,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)
            # Prevents reloading, potentially bug prone as reusing the swf
            # does not rest the upload flag
            self.upload = False


class CBox_AllXY(swf.Hard_Sweep):
    def __init__(self, IF, pulse_separation,
                 RO_pulse_delay,
                 RO_trigger_delay,
                 RO_pulse_length,
                 AWG, CBox,
                 double_points=True,
                 upload=True):
        '''
        Generates a sequence for the AWG to trigger the CBox and sets the tape
        in the CBox to measure an AllXY.

        double_points: True will measure the tape twice per element, this
            should give insight wether the deviation is real.
        '''
        super().__init__()
        self.parameter_name = 'AllXY element'
        self.unit = '#'
        self.name = 'AllXY'
        # would actually like to check if file is already loaded
        # filename can be get using AWG.get('setup_filename')
        self.upload = upload

        # The AllXY tape
        self.tape = np.array([0, 0, 1, 1,  # 1, 2
                              2, 2, 1, 2,  # 3, 4
                              2, 1, 3, 0,  # 5, 6
                              4, 0, 3, 4,  # 7, 8
                              4, 3, 3, 2,  # 9, 10
                              4, 1, 1, 4,  # 11, 12
                              2, 3, 3, 1,  # 13, 14
                              1, 3, 4, 2,  # 15, 16
                              2, 4, 1, 0,  # 17, 18
                              2, 0, 3, 3,  # 19, 20
                              4, 4])       # 21
        if double_points:
            double_tape = []
            for i in range(len(self.tape)//2):
                for j in range(2):
                    double_tape.extend((self.tape[2*i:2*i+2]))
            self.tape = double_tape
        self.sweep_points = np.arange(int(len(self.tape)/2))  # 2 pulses per elt

        # Making input pars available to prepare
        # Required instruments
        self.AWG = AWG
        self.CBox = CBox
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.RO_pulse_length = RO_pulse_length
        self.pulse_separation = pulse_separation

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)

        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')

            st_seqs.CBox_two_pulse_seq(
                IF=self.IF,
                pulse_separation=self.pulse_separation,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_pulse_length=self.RO_pulse_length,
                RO_trigger_delay=self.RO_trigger_delay, verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class CBox_multi_element_tape(swf.Hard_Sweep):
    def __init__(self, n_pulses, tape,
                 pulse_separation,
                 IF, RO_pulse_delay, RO_trigger_delay,
                 RO_pulse_length,
                 AWG, CBox,
                 upload=True):
        '''
        Sets an arbitrary tape as a sequence
        n_pulses is the number of pulses per element in the sequence
        by default
        '''
        super().__init__()
        self.n_pulses = n_pulses
        self.parameter_name = 'Element'
        self.unit = '#'
        self.name = 'multi-element tape'
        self.tape = tape

        self.upload = upload

        self.sweep_points = np.arange(int(len(self.tape)/n_pulses))
        self.AWG = AWG
        self.CBox = CBox
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.RO_pulse_length = RO_pulse_length
        self.pulse_separation = pulse_separation

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_multi_pulse_seq(
                n_pulses=self.n_pulses, pulse_separation=self.pulse_separation,
                IF=self.IF,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                RO_pulse_length=self.RO_pulse_length,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class Resetless_tape(swf.Hard_Sweep):
    def __init__(self, n_pulses, tape,
                 pulse_separation, resetless_interval,
                 IF, RO_pulse_delay, RO_trigger_delay,
                 RO_pulse_length,
                 AWG, CBox,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.parameter_name = 'Tape element'
        self.unit = ''
        self.name = 'Resetless_tape'
        self.tape = tape
        # array for transpose in MC these values are bs
        self.sweep_points = np.array(self.tape)
        self.AWG = AWG
        self.CBox = CBox
        self.RO_pulse_length = RO_pulse_length
        # would actually like to check if file is already loaded
        # filename can be get using AWG.get('setup_filename')
        self.upload = upload

        self.n_pulses = n_pulses
        self.resetless_interval = resetless_interval
        self.pulse_separation = pulse_separation

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_resetless_multi_pulse_seq(
                n_pulses=self.n_pulses, pulse_separation=self.pulse_separation,
                resetless_interval=self.resetless_interval,
                IF=self.IF,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                RO_pulse_length=self.RO_pulse_length,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class CBox_RB_sweep(swf.Hard_Sweep):
    def __init__(self,
                 IF, RO_pulse_length,
                 RO_pulse_delay, RO_trigger_delay,
                 pulse_separation,
                 AWG, CBox, LutMan,
                 cal_points=True,
                 nr_cliffords=[1, 3, 5, 10, 20],
                 nr_seeds=3, max_seq_duration=15e-6,
                 safety_margin=500e-9,
                 upload=True):
        super().__init__()
        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'
        self.name = 'Randomized_Benchmarking'
        self.safety_margin = safety_margin
        # Making input pars available to prepare
        # Required instruments
        self.AWG = AWG
        self.CBox = CBox
        self.LutMan = LutMan
        self.nr_seeds = nr_seeds
        self.cal_points = [0, 0, 1, 1]
        self.nr_cliffords = np.array(nr_cliffords)
        self.max_seq_duration = max_seq_duration
        self.pulse_separation_ns = pulse_separation*1e9

        self.IF = IF
        self.RO_pulse_length = RO_pulse_length
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        # Funny last sweep point values are to make the cal points appear
        # in sensible (visible) places in the plot
        self.sweep_points = np.concatenate([nr_cliffords,
                                           [nr_cliffords[-1]+.2,
                                            nr_cliffords[-1]+.3,
                                            nr_cliffords[-1]+.7,
                                            nr_cliffords[-1]+.8]])

    def prepare(self, upload_tek_seq=True, **kw):
        self.AWG.stop()
        n_cls = self.nr_cliffords
        time_tape = []
        pulse_length = self.LutMan.gauss_width.get()*4
        for seed in range(self.nr_seeds):
            for n_cl in n_cls:
                cliffords = rb.randomized_benchmarking_sequence(n_cl)
                cl_tape = rb.convert_clifford_sequence_to_tape(
                    cliffords,
                    self.LutMan.lut_mapping.get())
                for i, tape_elt in enumerate(cl_tape):
                    if i == 0:
                        # wait_time is in ns
                        wait_time = (self.max_seq_duration*1e9 -
                                     (len(cl_tape)-1)*self.pulse_separation_ns -
                                     pulse_length)
                    else:
                        wait_time = self.pulse_separation_ns - pulse_length
                    end_of_marker = (i == (len(cl_tape)-1))
                    entry = self.CBox.create_timing_tape_entry(
                        wait_time, tape_elt, end_of_marker, prepend_elt=0)
                    time_tape.extend(entry)

            for cal_pt in self.cal_points:
                wait_time = self.max_seq_duration*1e9 - pulse_length
                time_tape.extend(self.CBox.create_timing_tape_entry(
                    wait_time, cal_pt, True, prepend_elt=0))
        # print('Total tape length', len(time_tape))
        for awg in range(3):
            self.CBox.set('AWG{}_mode'.format(awg), 'Segmented')
            self.CBox.set_segmented_tape(awg, time_tape)
            self.CBox.restart_awg_tape(awg)
        if upload_tek_seq:
            self.upload_tek_seq()

    def upload_tek_seq(self):
        st_seqs.CBox_single_pulse_seq(
            IF=self.IF,
            RO_pulse_delay=self.RO_pulse_delay +
            self.max_seq_duration+self.safety_margin,
            RO_trigger_delay=self.RO_trigger_delay,
            RO_pulse_length=self.RO_pulse_length)


class Two_d_CBox_RB_seq(swf.Soft_Sweep):
    def __init__(self, CBox_RB_sweepfunction):
        super().__init__()
        self.parameter_name = 'Idx'
        self.unit = '#'
        self.name = 'Randomized_Benchmarking_random_seeds'
        self.CBox_RB_sweepfunction = CBox_RB_sweepfunction

    def set_parameter(self, val):
        '''
        Uses the CBox RB sweepfunction to upload a new tape of random cliffords
        explicitly does not reupload the AWG sequence.
        '''
        self.CBox_RB_sweepfunction.prepare(upload_tek_seq=False)







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
