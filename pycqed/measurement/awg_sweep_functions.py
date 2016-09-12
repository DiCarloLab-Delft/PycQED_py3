import numpy as np
import logging
from measurement import sweep_functions as swf
from measurement.randomized_benchmarking import randomized_benchmarking as rb
from measurement.pulse_sequences import standard_sequences as st_seqs
from measurement.pulse_sequences import single_qubit_tek_seq_elts as sqs
from measurement.pulse_sequences import single_qubit_2nd_exc_seqs as sqs2
default_gauss_width = 10  # magic number should be removed,
# note magic number only used in old mathematica seqs


class Rabi(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars, n=1, upload=True, return_seq=False):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.n = n
        self.upload = upload
        self.name = 'Rabi'
        self.parameter_name = 'amplitude'
        self.unit = 'V'
        self.return_seq = return_seq

    def prepare(self, **kw):
        if self.upload:
            sqs.Rabi_seq(amps=self.sweep_points,
                         pulse_pars=self.pulse_pars,
                         RO_pars=self.RO_pars,
                         n=self.n, return_seq=self.return_seq)


class Rabi_amp90(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars, n=1, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.n = n
        self.upload = upload
        self.name = 'Rabi_amp90'
        self.parameter_name = 'ratio_amp90_amp180'
        self.unit = ''

    def prepare(self, **kw):
        if self.upload:
            sqs.Rabi_amp90_seq(scales=self.sweep_points,
                               pulse_pars=self.pulse_pars,
                               RO_pars=self.RO_pars,
                               n=self.n)


class Rabi_2nd_exc(swf.Hard_Sweep):
    def __init__(self, pulse_pars, pulse_pars_2nd,
                 RO_pars, amps=None, n=1, cal_points=True, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.n = n
        self.upload = upload
        self.name = 'Rabi 2nd excited state'
        self.parameter_name = 'amplitude'
        self.unit = 'V'
        if cal_points and amps is not None:
            self.sweep_points = np.concatenate([amps,
                                               [amps[-1]*1.05,
                                                amps[-1]*1.06,
                                                amps[-1]*1.07,
                                                amps[-1]*1.08,
                                                amps[-1]*1.09,
                                                amps[-1]*1.1]])

    def prepare(self, **kw):
        if self.upload:
            sqs2.Rabi_2nd_exc_seq(amps=self.sweep_points,
                                  pulse_pars=self.pulse_pars,
                                  pulse_pars_2nd=self.pulse_pars_2nd,
                                  RO_pars=self.RO_pars,
                                  n=self.n)


class Ramsey_2nd_exc(swf.Hard_Sweep):
    def __init__(self, pulse_pars, pulse_pars_2nd,
                 RO_pars, times=None, n=1, cal_points=True, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.n = n
        self.upload = upload
        self.name = 'Rabi 2nd excited state'
        self.parameter_name = 'amplitude'
        self.unit = 'V'
        if cal_points and times is not None:
            self.sweep_points = np.concatenate([times,
                                               [times[-1]*1.05,
                                                times[-1]*1.06,
                                                times[-1]*1.07,
                                                times[-1]*1.08,
                                                times[-1]*1.09,
                                                times[-1]*1.1]])
    def prepare(self, **kw):
        if self.upload:
            sqs2.Ramsey_2nd_exc_seq(times=self.sweep_points,
                                    pulse_pars=self.pulse_pars,
                                    pulse_pars_2nd=self.pulse_pars_2nd,
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
    def __init__(self, pulse_pars, RO_pars, upload=True,
                 pulse_comb='OffOn', nr_samples=2):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload

        self.parameter_name = 'sample'
        self.unit = '#'
        self.name = pulse_comb
        self.sweep_points = np.arange(nr_samples)

    def prepare(self, **kw):
        if self.upload:
            sqs.OffOn_seq(pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars, pulse_comb=self.name)


class Butterfly(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars, initialize=False, upload=True,
                 post_msmt_delay=2000e-9):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'Buttefly element'
        self.unit = '#'
        self.name = 'Butterfly'
        self.sweep_points = np.arange(2)
        self.initialize = initialize
        self.post_msmt_delay = post_msmt_delay

    def prepare(self, **kw):
        if self.upload:
            sqs.Butterfly_seq(pulse_pars=self.pulse_pars,
                              post_msmt_delay=self.post_msmt_delay,
                              RO_pars=self.RO_pars, initialize=self.initialize)


class Randomized_Benchmarking(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars,
                 nr_seeds, nr_cliffords,
                 cal_points=True,
                 double_curves=False, seq_name=None,
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
        self.double_curves = double_curves
        self.seq_name = seq_name

        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'
        self.name = 'Randomized_Benchmarking'
        self.sweep_points = nr_cliffords
        if double_curves:
            nr_cliffords = np.repeat(nr_cliffords, 2)

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
                cal_points=self.cal_points,
                double_curves=self.double_curves,
                seq_name=self.seq_name)


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


class Echo(swf.Hard_Sweep):
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
        self.name = 'Echo'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.Echo_seq(times=self.sweep_points,
                         pulse_pars=self.pulse_pars,
                         RO_pars=self.RO_pars,
                         artificial_detuning=self.artificial_detuning,
                         cal_points=self.cal_points)


class Motzoi_XY(swf.Hard_Sweep):
    def __init__(self, motzois, pulse_pars, RO_pars, upload=True):
        '''
        Measures 2 points per motzoi value specified in motzois and adds 4
        calibration points to it.
        '''
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.name = 'Motzoi_XY'
        self.parameter_name = 'motzoi'
        self.unit = ' '
        sweep_pts = np.repeat(motzois, 2)
        self.sweep_points = np.append(sweep_pts,
                                   [motzois[-1]+(motzois[-1]-motzois[-2])]*4)

    def prepare(self, **kw):
        if self.upload:
            sqs.Motzoi_XY(motzois=self.sweep_points,
                          pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars)


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
            ch4_amp = self.AWG.get('ch4_amp')
            st_seqs.CBox_T1_marker_seq(IF=self.IF, times=self.sweep_points,
                                       RO_pulse_delay=self.RO_pulse_delay,
                                       RO_trigger_delay=self.RO_trigger_delay,
                                       verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)

class CBox_v3_T1(swf.Hard_Sweep):
    def __init__(self, CBox, upload=True):
        super().__init__()
        self.name = 'T1'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.upload = upload
        self.CBox = CBox


    def prepare(self, **kw):
        if self.upload:
            self.CBox.AWG0_mode('Codeword-trigger mode')
            self.CBox.AWG1_mode('Codeword-trigger mode')
            self.CBox.AWG2_mode('Codeword-trigger mode')
            self.CBox.set_master_controller_working_state(0, 0, 0)
            self.CBox.load_instructions('CBox_v3_test_program\T1.asm')
            self.CBox.set_master_controller_working_state(1, 0, 0)

class CBox_v3_T1(swf.Hard_Sweep):
    def __init__(self, CBox, upload=True):
        super().__init__()
        self.name = 'T1'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.upload = upload
        self.CBox = CBox

    def prepare(self, **kw):
        if self.upload:
            self.CBox.AWG0_mode('Codeword-trigger mode')
            self.CBox.AWG1_mode('Codeword-trigger mode')
            self.CBox.AWG2_mode('Codeword-trigger mode')
            self.CBox.set_master_controller_working_state(0, 0, 0)
            self.CBox.load_instructions('CBox_v3_test_program\T1.asm')
            self.CBox.set_master_controller_working_state(1, 0, 0)


class CBox_Ramsey(swf.Hard_Sweep):
    def __init__(self, IF, RO_pulse_length,
                 RO_pulse_delay, RO_trigger_delay, pulse_delay,
                 AWG, CBox, cal_points=True,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.pulse_delay = pulse_delay
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
                pulse_delay=self.pulse_delay,
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
                 RO_pulse_delay, RO_trigger_delay, pulse_delay,
                 AWG, CBox, cal_points=True,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.pulse_delay = pulse_delay
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
    def __init__(self, IF, pulse_delay,
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
        self.pulse_delay = pulse_delay

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
                pulse_delay=self.pulse_delay,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_pulse_length=self.RO_pulse_length,
                RO_trigger_delay=self.RO_trigger_delay, verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class CBox_multi_element_tape(swf.Hard_Sweep):
    def __init__(self, n_pulses, tape,
                 pulse_delay,
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
        self.pulse_delay = pulse_delay

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
                n_pulses=self.n_pulses, pulse_delay=self.pulse_delay,
                IF=self.IF,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                RO_pulse_length=self.RO_pulse_length,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class Resetless_tape(swf.Hard_Sweep):
    def __init__(self, n_pulses, tape,
                 pulse_delay, resetless_interval,
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
        self.pulse_delay = pulse_delay

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
                n_pulses=self.n_pulses, pulse_delay=self.pulse_delay,
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
                 pulse_delay,
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
        self.pulse_delay_ns = pulse_delay*1e9

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
                                     (len(cl_tape)-1)*self.pulse_delay_ns -
                                     pulse_length)
                    else:
                        wait_time = self.pulse_delay_ns - pulse_length
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

