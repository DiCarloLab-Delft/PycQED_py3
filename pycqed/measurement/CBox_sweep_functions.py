import numpy as np
import logging
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.sweep_functions import Soft_Sweep
from pycqed.measurement.waveform_control_CC import waveform as wf

# Commented out as there is no module named Experiments.CLEAR.prepare_for_CLEAR.prepare_for_CLEAR
# from Experiments.CLEAR.prepare_for_CLEAR import prepare_for_CLEAR

import time
import imp
gauss_width = 10
imp.reload(wf)


class CBox_Sweep(swf.Hard_Sweep):

    def __init__(self, Duplexer=False, **kw):
        self.sweep_control = 'hard'
        if not hasattr(self, 'cal_points'):
            self.cal_points = kw.pop('cal_points', 10)

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass

######################
# Time domain sweeps #
######################


class T1(CBox_Sweep):
    '''
    Performs a T1 measurement using a tektronix and the CBox.
    The tektronix is used for timing the pulses, a Pi-pulse is loaded onto the
    CBox.
    '''

    def __init__(self, gauss_width=25,
                 qubit_suffix="", NoSegments=70, stepsize=2000,
                 amp180=4000, f_modulation=-0.02, **kw):
        self.name = 'T1'
        self.parameter_name = 'time'
        self.unit = 'ns'
        # Available stepsizes needs to be verified! this is copy from AWG_swf
        self.available_stepsizes = [50, 100, 200, 500, 1000, 1500, 2000, 3000]

        self.filename = 'FPGA_T1_%i_5014' % (stepsize)

        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)
        self.NoSegments = NoSegments
        self.amp180 = amp180
        self.gauss_width = gauss_width
        self.f_modulation = f_modulation

        if stepsize not in self.available_stepsizes:
            logging.error('Stepsize not available')
        super(self.__class__, self).__init__(**kw)

    def prepare(self):
        self.CBox.set_awg_mode(0, 1)
        self.CBox.set_awg_mode(1, 1)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)

        Wave_I, Wave_Q = wf.mod_gauss(self.amp180, self.gauss_width,
                                      self.f_modulation)
        self.CBox.set_awg_lookuptable(0, 0, 1, np.round(Wave_I))
        self.CBox.set_awg_lookuptable(0, 0, 0, np.round(Wave_Q))
        # additionally loading to AWG1 for scope
        self.CBox.set_awg_lookuptable(1, 0, 1, np.round(Wave_I))
        self.CBox.set_awg_lookuptable(1, 0, 0, np.round(Wave_Q))

        # These two lines should be combined to CBox.set_No_Samples but is
        # untested
        NoAvg = self.CBox.get_avg_size()
        self.CBox.set_averaging_parameters(self.NoSegments, NoAvg)


class Lutman_par_with_reload(Soft_Sweep):

    def __init__(self, LutMan, parameter):
        '''
        Generic sweep function that combines setting a LutMan parameter
        with reloading lookuptables.
        '''
        super().__init__()
        self.LutMan = LutMan
        self.parameter = parameter
        self.name = parameter.name
        self.parameter_name = parameter.label
        self.unit = parameter.unit

    def set_parameter(self, val):
        self.parameter.set(val)
        self.LutMan.load_pulses_onto_AWG_lookuptable()


class Lutman_par_with_reload_single_pulse(Soft_Sweep):

    def __init__(self, LutMan, parameter, pulse_names=['X180']):
        '''
        Generic sweep function that combines setting a LutMan parameter
        with reloading lookuptables.
        '''
        super().__init__()
        self.LutMan = LutMan
        self.parameter = parameter
        self.name = parameter.name
        self.parameter_name = parameter.label
        self.unit = parameter.unit
        self.pulse_names = pulse_names

    def set_parameter(self, val):
        self.parameter.set(val)
        for pulse_name in self.pulse_names:
            self.LutMan.load_pulse_onto_AWG_lookuptable(pulse_name)


class LutMan_amp180_90(Soft_Sweep):
    '''
    Sweeps both the amp180 parameter and the amp90 of the CBox_lut_man
    Automatically sets amp90 to half of amp180.
    The amp180 is the sweep parameter that is set and tracked.
    '''

    def __init__(self, LutMan, reload_pulses=True, awg_nr=0):
        super(self.__class__, self).__init__()
        self.awg_nr = awg_nr
        self.reload_pulses = reload_pulses
        self.name = 'lookuptable amp180'
        self.parameter_name = 'amp180'
        self.unit = 'mV'
        self.LutMan = LutMan

    def set_parameter(self, val):
        self.LutMan.set('Q_amp180', val)
        self.LutMan.set('Q_amp90', val/2.0)
        if self.reload_pulses:
            self.LutMan.load_pulses_onto_AWG_lookuptable()


class DAC_offset(CBox_Sweep):
    '''
    Varies DAC offsets in CBox AWG's. Additionally identity pulses are loaded
    in the lookuptable 0, of I and Q channels
    '''

    def __init__(self,  AWG_nr, dac_ch, CBox):
        super(self.__class__, self).__init__()
        self.sweep_control = 'soft'  # Overwrites 'hard sweep part'
        self.name = 'CBox DAC offset'
        self.parameter_name = 'Voltage'
        self.unit = 'mV'
        self.filename = 'FPGA_DAC_offset_sweep_5014'
        self.dac_channel = dac_ch
        self.AWG_nr = AWG_nr
        self.CBox = CBox
        # any arbitrary sequence that is not time dependent on the pulse
        # trigger will do

    def set_parameter(self, val):
        self.CBox.set_dac_offset(self.AWG_nr, self.dac_channel, val)


class Ramsey(CBox_Sweep):
    '''
    Performs a T2 Ramsey measurement using a tektronix and the CBox.
    The tektronix is used for timing the pulses.
    Codewords are used to determine what pulse will be used.

    WARNING:
    The artificial detuning is applied by delaying the pi/2 pulses as the
    sideband modulation is the same for every pulse.
    This creates an error in the x-values of the sweep points<50ns.
    This should be taken care of when interpreting data for shorter timescales.
    '''

    def __init__(self, gauss_width=25,
                 qubit_suffix="", NoSegments=70, stepsize=50, **kw):
        print('WARNING, this function is deprecated. use Ramsey_tape()')
    #     self.name = 'Ramsey'
    #     self.parameter_name = 'time'
    #     self.unit = 'ns'
    #     self.available_stepsizes = [50, 100, 200, 500, 1000, 1500]
    #     # NOTE: stepsizes below 50ns are not available because of SBmod freq
    #     # self.available_stepsizes = [5, 10, 30, 100, 200, 500, 1000, 1500]
    #     self.CBox_lut_man = qt.instruments['CBox_lut_man']
    #     self.filename = 'FPGA_Codeword_Ramsey_%i_5014' % (stepsize)

    #     base_pulse_delay = 200
    #     self.sweep_points = np.linspace(stepsize+base_pulse_delay,
    #                                     NoSegments*stepsize + base_pulse_delay,
    #                                     NoSegments)
    #     self.NoSegments = NoSegments

    #     if stepsize not in self.available_stepsizes:
    #         raise Exception('Stepsize not available')
    #     super(self.__class__, self).__init__(**kw)

    # def prepare(self):
    #     self.CBox.set_acquisition_mode(0)
    #     self.CBox.set_awg_mode(0, 0)
    #     self.CBox.set_awg_mode(1, 0)
    #     self.AWG.stop()
    #     self.AWG.set_setup_filename(self.filename,
    #                                 force_load=False)

    #     self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
    #     self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)
    #     self.CBox.set_nr_samples(self.NoSegments)


class Ramsey_tape(CBox_Sweep):
    '''
    Performs a T2 Ramsey measurement using a tektronix and the CBox.
    The tektronix is used for timing the pulses.
    Codewords are used to determine what pulse will be used.

    Artificial detuning is applied by delaying the triggers 5 ns
    '''

    def __init__(self, gauss_width=25,
                 qubit_suffix="", NoSegments=70, stepsize=50, **kw):
        self.name = 'Ramsey'
        print('Using tape mode Ramsey')
        self.parameter_name = 'time'
        self.unit = 'ns'
        self.available_stepsizes = [50, 100, 200, 500, 1000, 1500]
        # NOTE: stepsizes below 50ns are not available because of SBmod freq
        # self.available_stepsizes = [5, 10, 30, 100, 200, 500, 1000, 1500]
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.TD_Meas = qt.instruments['TD_Meas']
        self.filename = 'FPGA_Ramsey_%i_5014' % (stepsize)

        base_pulse_delay = 200
        self.sweep_points = np.arange(stepsize+base_pulse_delay,
                                      NoSegments*(stepsize+5) +
                                      base_pulse_delay,
                                      stepsize + 5, dtype=float)

        self.NoSegments = NoSegments
        self.NoCalpoints = 10

        if stepsize not in self.available_stepsizes:
            raise Exception('Stepsize not available')
        super(self.__class__, self).__init__(**kw)

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 2)
        self.CBox.set_awg_mode(1, 2)
        self.AWG.stop()
        self.TD_Meas.set_CBox_tape_mode(True)
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)

        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)
        ramsey_tape = [3, 3] * int((self.NoSegments - self.NoCalpoints))
        cal_zero_tape = [0] * int(self.NoCalpoints/2)
        cal_one_tape = [1] * int(self.NoCalpoints/2)
        tape = np.array(ramsey_tape+cal_zero_tape+cal_one_tape)

        self.CBox.set_awg_tape(0, len(tape), tape)
        self.CBox.set_awg_tape(1, len(tape), tape)
        self.CBox.set_nr_samples(self.NoSegments)


class Echo(CBox_Sweep):
    '''
    Performs a T2 Echo measurement using a tektronix and the CBox.
    The tektronix is used for timing the pulses, a Pi-pulse is loaded onto the
    CBox.
    '''

    def __init__(self, stepsize,
                 amp180, amp90, gauss_width=25,
                 qubit_suffix="", NoSegments=70, f_modulation=-0.02, **kw):
        print("amp180", amp180)
        self.name = 'Echo'
        self.parameter_name = 'time'
        self.unit = 'ns'
        # Available stepsizes needs to be verified! this is copy from AWG_swf
        self.available_stepsizes = [50, 100, 200, 500, 1000, 1500, 2000, 3000]

        self.filename = 'FPGA_Echo_%i_5014' % (stepsize)

        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)
        self.NoSegments = NoSegments
        self.amp180 = amp180
        self.amp90 = amp90
        self.gauss_width = gauss_width
        self.f_modulation = f_modulation

        if stepsize not in self.available_stepsizes:
            logging.error('Stepsize not available')
        super(self.__class__, self).__init__(**kw)

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 0)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)

        Wave_I_180, Wave_Q_180 = wf.mod_gauss(self.amp180, self.gauss_width,
                                              self.f_modulation)
        Wave_I_90, Wave_Q_90 = wf.mod_gauss(self.amp90, self.gauss_width,
                                            self.f_modulation)
        self.CBox.set_awg_lookuptable(0, 7, 1, np.round(Wave_I_180))
        self.CBox.set_awg_lookuptable(0, 7, 0, np.round(Wave_Q_180))
        self.CBox.set_awg_lookuptable(0, 0, 1, np.round(Wave_I_90))
        self.CBox.set_awg_lookuptable(0, 0, 0, np.round(Wave_Q_90))

        # These two lines should be combined to CBox.set_No_Samples but is
        # untested
        NoAvg = self.CBox.get_avg_size()
        self.CBox.set_averaging_parameters(self.NoSegments, NoAvg)


class T1_tape(CBox_Sweep):
    '''
    Performs a T1 measurement using a tektronix for metronome and the CBox to
    produce pulses in tape mode. The tektronix is used for timing the pulses, a
    Pi-pulse is loaded onto the CBox.
    '''

    def __init__(self, gauss_width=25,
                 qubit_suffix="", NoSegments=70, stepsize=4000,
                 amp180=4000, amp90=2000, f_modulation=-0.02, cal_points=10, **kw):
        self.name = 'T1_tape'
        self.parameter_name = 'time'
        self.unit = 'ns'
        # Available stepsizes needs to be verified! this is copy from AWG_swf
        self.available_stepsizes = [50, 100, 200,
                                    500, 1000, 1500, 2000, 3000, 4000]

        self.filename = 'FPGA_Tape_T1_%i_5014' % (stepsize)

        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)
        self.NoSegments = NoSegments
        self.amp180 = amp180
        self.amp90 = amp90
        self.gauss_width = gauss_width
        self.f_modulation = f_modulation

        if stepsize not in self.available_stepsizes:
            logging.error('Stepsize not available')
        super(self.__class__, self).__init__(**kw)

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        print("CBox set to mode 0")
        self.CBox.set_awg_mode(0, 0)
        self.CBox.set_awg_mode(1, 0)
        self.AWG.stop()
        print("AWG is stopped")
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)

        Wave_I_180, Wave_Q_180 = wf.mod_gauss(self.amp180, self.gauss_width,
                                              self.f_modulation)
        Wave_I_0 = Wave_I_180*0
        Wave_Q_0 = Wave_I_0

        self.CBox.set_awg_lookuptable(0, 0, 1, np.round(Wave_I_180))
        self.CBox.set_awg_lookuptable(0, 0, 0, np.round(Wave_Q_180))
        print("1")
        self.CBox.set_awg_lookuptable(0, 7, 1, np.round(Wave_I_0))
        self.CBox.set_awg_lookuptable(0, 7, 0, np.round(Wave_Q_0))
        # copying the tables to AWG2 for scope
        self.CBox.set_awg_lookuptable(1, 0, 1, np.round(Wave_I_180))
        self.CBox.set_awg_lookuptable(1, 0, 0, np.round(Wave_Q_180))
        print("2")
        self.CBox.set_awg_lookuptable(1, 7, 1, np.round(Wave_I_0))
        self.CBox.set_awg_lookuptable(1, 7, 0, np.round(Wave_Q_0))
        sequence_points = self.NoSegments-self.cal_points
        tape_length = (sequence_points)*self.NoSegments
        tape = 7*np.ones(tape_length)
        print("tape_length", tape_length)
        for i in range(sequence_points):
            tape[(i+1)*(sequence_points)-i-1] = 0
            print(tape[i*(sequence_points):i *
                       (sequence_points)+sequence_points])
        print("done first part")
        # adding calibration points
        for i in range(self.cal_points):
            first_cal_segment = (sequence_points)**2
            segment = first_cal_segment+(i+1)*(sequence_points)-1
            # print segment
            if i > (self.cal_points/2-1):
                tape[segment] = 0
            # print segment-(sequence_points)+1
            print(tape[segment-sequence_points+1:segment+1])
            print(i)
        print("3")
        self.CBox.set_awg_tape(0, len(tape), tape)
        print("tape length", len(tape))
        # copying the tables to AWG2 for scope
        self.CBox.set_awg_tape(1, len(tape), tape)
        # These two lines should be combined to CBox.set_No_Samples but is
        # untested
        NoAvg = self.CBox.get_avg_size()
        self.CBox.set_averaging_parameters(self.NoSegments, NoAvg)
        print("tape is loaded")
        # for i in range(len(tape)):
        #     if np.mod(i,20) == 0:
        #         print ("#\n")


class OnOff_touch_n_go(CBox_Sweep):
    '''
    Performs OnOff measurement using the CBox to produce pulses in codeword
    tape mode.
    '''

    def __init__(self,
                 NoSegments=2, stepsize=2000, pulses='OffOn',
                 NoShots=8000, **kw):
        self.name = 'FPGA_touch_n_go_calibration'
        self.parameter_name = 'time'
        self.unit = 'ns'
        self.NoSegments = NoSegments
        self.NoShots = NoShots
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.linspace(stepsize, stepsize*NoSegments,
                                        NoSegments)
        self.pulses = pulses

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_log_length(self.NoShots)
        self.CBox.set_awg_mode(0, 2)
        self.AWG.stop()
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        if self.pulses == 'OffOn':
            tape = np.array([0, 1])
        if self.pulses == 'OffOff':
            tape = np.array([0, 0])
        if self.pulses == 'OnOn':
            tape = np.array([1, 1])
        self.CBox.set_awg_tape(0, len(tape), tape)
        print("tape", tape)
        NoAvg = self.CBox.get_avg_size()
        self.CBox.set_averaging_parameters(self.NoSegments, NoAvg)


class custom_tape_touch_n_go(CBox_Sweep):

    def __init__(self,
                 NoSegments=2, stepsize=2000,
                 custom_tape=None, NoShots=8000, **kw):
        self.name = 'custom_tape_touch_n_go'
        self.parameter_name = 'msmt index'
        self.unit = ''
        self.NoSegments = NoSegments
        self.NoShots = NoShots
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.arange(NoSegments)
        self.custom_tape = custom_tape

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_run_mode(0)
        print('setting nr of shots to', self.NoShots)
        self.CBox.set_log_length(self.NoShots)
        self.CBox.set_awg_mode(0, 2)
        self.AWG.stop()
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        if self.custom_tape is None:
            tape = np.array([0, 0])
        else:
            tape = self.custom_tape
            print("using the custom tape ", tape)
        self.CBox.set_awg_tape(0, len(tape), tape)


class random_telegraph_tape_touch_n_go(CBox_Sweep):

    def __init__(self,
                 qubit_suffix="", NoSegments=2, stepsize=2000,
                 p_switch_us=0, NoShots=8000, pulse_a=0, pulse_b=1, **kw):
        self.name = 'random_telegraph_tape_touch_n_go'
        self.parameter_name = 'time'
        self.unit = 'ns'
        self.NoSegments = NoSegments
        self.NoShots = NoShots
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.linspace(stepsize, stepsize*NoSegments,
                                        NoSegments)
        self.p_switch_us = p_switch_us
        self.p_switch = 1-(1-self.p_switch_us)**(stepsize/1000)
        self.pulse_a = pulse_a
        self.pulse_b = pulse_b

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_log_length(self.NoShots)
        self.CBox.set_awg_mode(0, 2)
        self.AWG.stop()

        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        if self.NoShots > 4000:
            tape_elements = 4000
        else:
            tape_elements = self.NoShots
        tape = np.zeros(tape_elements)
        tape[0] = self.pulse_a
        for i in range(tape_elements-1):
            if np.random.rand(1) < self.p_switch:  # flipping with chance p_switch
                if tape[i] == self.pulse_a:
                    tape[i+1] = self.pulse_b
                else:
                    tape[i+1] = self.pulse_a
            else:  # no flipping event
                tape[i+1] = tape[i]
        self.CBox.set_awg_tape(0, len(tape), tape)


class AllXY(CBox_Sweep):
    '''
    Performs AllXY measurement using the CBox to produce pulses in codeword
    trigger mode. The tektronix is used for the coded trigges.
    '''

    def __init__(self, gauss_width=25,
                 qubit_suffix="", NoSegments=42, stepsize=1, **kw):
        print('Deprecated, recommend using AllXY_tape() instead')

        self.name = 'AllXY'
        self.parameter_name = 'time'
        self.unit = 'ns'
        # Available stepsizes needs to be verified! this is copy from AWG_swf
        self.filename = 'FPGA_AllXY_5014'
        self.cal_points = [list(range(10)), list(range(-8, 0))]
        self.NoSegments = NoSegments
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 0)
        self.CBox.set_awg_mode(1, 0)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)

        NoAvg = self.CBox.get_avg_size()
        self.CBox.set_averaging_parameters(self.NoSegments, NoAvg)


class AllXY_tape(CBox_Sweep):
    '''
    Performs AllXY measurement using the CBox toproduce pulses in tape mode.
    The tektronix is used to time the pulses.
    '''

    def __init__(self,
                 qubit_suffix="", NoSegments=42, stepsize=1, **kw):
        self.name = 'AllXY_tape'
        self.parameter_name = 'time'
        self.unit = 'ns'
        self.filename = 'FPGA_Tape_AllXY_5014'
        self.cal_points = [list(range(10)), list(range(-8, 0))]
        self.NoSegments = NoSegments
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.TD_Meas = qt.instruments['TD_Meas']

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 2)
        self.CBox.set_awg_mode(2, 2)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(2)
        self.TD_Meas.set_CBox_tape_mode(True)
        # print "AWG 1 luts loaded"
        tape = np.array([0, 0, 0, 0, 1, 1, 1, 1,  # 1, 3
                         2, 2, 2, 2, 1, 2, 1, 2,  # 5, 7
                         2, 1, 2, 1, 3, 0, 3, 0,  # 9, 11
                         4, 0, 4, 0, 3, 4, 3, 4,  # 13, 15
                         4, 3, 4, 3, 3, 2, 3, 2,  # 17, 19
                         4, 1, 4, 1, 1, 4, 1, 4,  # 21,23
                         2, 3, 2, 3, 3, 1, 3, 1,  # 25, 27
                         1, 3, 1, 3, 4, 2, 4, 2,  # 29, 31
                         2, 4, 2, 4, 1, 0, 1, 0,  # 33, 35
                         2, 0, 2, 0, 3, 3, 3, 3,  # 37, 39
                         4, 4, 4, 4])              # 41

        self.CBox.set_awg_tape(0, len(tape), tape)
        self.CBox.set_awg_tape(2, len(tape), tape)
        self.CBox.set_nr_samples(self.NoSegments)


class OnOff_tape(CBox_Sweep):
    '''
    Performs OnOff measurement using the CBox toproduce pulses in tape mode.
    The tektronix is used to time the pulses.
    '''

    def __init__(self,
                 qubit_suffix="", NoSegments=2, stepsize=1, pulses='OffOn', **kw):
        self.name = 'OnOff_tape'
        self.parameter_name = 'time'
        self.unit = 'ns'
        self.filename = 'FPGA_Tape_OnOff_5014'
        self.NoSegments = NoSegments
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.pulses = pulses
        print('New version tape')

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 2)
        self.CBox.set_awg_mode(1, 2)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)

        # print "AWG 1 luts loaded"
        if self.pulses == 'OffOn':
            tape = np.array([0, 1])
        if self.pulses == 'OffOff':
            tape = np.array([0, 0])
        if self.pulses == 'OnOn':
            tape = np.array([1, 1])
        self.CBox.set_awg_tape(0, len(tape), tape)
        self.CBox.set_awg_tape(1, len(tape), tape)
        self.CBox.set_nr_samples(self.NoSegments)


class OnOff_transients(CBox_Sweep):
    '''
    Performs OnOff measurement using the CBox toproduce pulses in tape mode.
    The tektronix is used to time the pulses.
    '''

    def __init__(self,
                 qubit_suffix="", NoSegments=2, stepsize=1, pulses='OffOn', **kw):
        self.name = 'OnOff_tape'
        self.parameter_name = 'time'
        self.unit = 'ns'
        self.filename = 'FPGA_Tape_OnOff_5014'
        self.NoSegments = NoSegments
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.pulses = pulses
        print('New version tape')

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 2)
        self.CBox.set_awg_mode(1, 2)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)

        # print "AWG 1 luts loaded"
        if self.pulses == 'OffOn':
            tape = np.array([0, 1])
        if self.pulses == 'OffOff':
            tape = np.array([0, 0])
        if self.pulses == 'OnOn':
            tape = np.array([1, 1])
        self.CBox.set_awg_tape(0, len(tape), tape)
        self.CBox.set_awg_tape(1, len(tape), tape)
        self.CBox.set_nr_samples(self.NoSegments)


class single_element_tape_test(CBox_Sweep):
    '''
    Performs a measurement similar to AllXY in the syndrome it produces
    but only uses a single pulse per segment.
    '''

    def __init__(self,
                 qubit_suffix="", NoSegments=42, stepsize=1, **kw):
        self.name = 'Single_element_test_tape'
        self.parameter_name = 'time'
        self.unit = 'ns'
        self.filename = 'FPGA_tape_single_test_5014'
        self.cal_points = [list(range(10)), list(range(-8, 0))]
        self.NoSegments = NoSegments
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        print('New version tape')

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 2)
        self.CBox.set_awg_mode(1, 2)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)

        tape = np.array([0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,  # 10 times identity

                         3, 4, 3, 4, 3,
                         4, 3, 4, 3, 4,
                         3, 4, 3, 4, 3,
                         4, 3, 4, 3, 4,
                         3, 4, 3, 4,  # 24 times pi/2 pulses
                         1, 2, 1, 2, 1,
                         2, 1, 2  # 8 times pi pulse
                         ])

        tape = np.array([0, 0, 1, 1, 2,
                         2, 1, 2, 2, 1,  # 10 times identity

                         3, 0, 4, 0, 3,
                         4, 4, 3, 3, 2,
                         4, 1, 1, 4, 2,
                         3, 3, 1, 1, 3,
                         4, 2, 2, 4,  # 24 times pi/2 pulses

                         # 1, 2, 1, 2, 1,
                         # 2, 1, 2
                         1, 0, 2, 0, 3,
                         3, 4, 4  # 8 times pi pulse
                         ])

        self.CBox.set_awg_tape(0, len(tape), tape)
        self.CBox.set_awg_tape(1, len(tape), tape)
        self.CBox.set_nr_samples(self.NoSegments)


class drag_detuning(CBox_Sweep):
    '''
    Performs drag_detuning measurement using the CBox to produce pulses in codeword
    trigger mode. The tektronix is used for the coded trigges.
    '''

    def __init__(self,
                 qubit_suffix="", NoSegments=2, stepsize=1, **kw):
        self.name = 'drag_detuning'
        self.parameter_name = 'time'
        self.unit = 'ns'
        # Available stepsizes needs to be verified! this is copy from AWG_swf
        self.filename = 'FPGA_DragDetuning_5014'
        self.NoSegments = NoSegments
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        super(self.__class__, self).__init__(**kw)
        self.sweep_points = np.linspace(stepsize, NoSegments*stepsize,
                                        NoSegments)

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 0)
        self.CBox.set_awg_mode(1, 0)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)

        NoAvg = self.CBox.get_avg_size()
        self.CBox.set_averaging_parameters(self.NoSegments, NoAvg)


class flipping_sequence(CBox_Sweep):
    '''
    Loads a codeword trigger sequence that consists of applying a X90 pulse
    follwed by N X180 pulses. With 1<N<50  followed by 10 calibration points.
    '''

    def __init__(self, gauss_width=25,
                 qubit_suffix="",  **kw):
        self.name = 'Flipping sequence'
        self.parameter_name = 'number of X180 pulses '
        self.unit = 'N'
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.filename = 'FPGA_X90_N_X180_5014'
        self.NoSegments = 60
        self.sweep_points = np.linspace(
            1, 2 * self.NoSegments, self.NoSegments)
        super(self.__class__, self).__init__(**kw)

    def prepare(self):
        self.CBox.set_acquisition_mode(0)
        self.CBox.set_awg_mode(0, 0)
        self.CBox.set_awg_mode(1, 0)
        self.AWG.stop()
        self.AWG.set_setup_filename(self.filename,
                                    force_load=False)

        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(1)
        self.CBox.set_nr_samples(self.NoSegments)


######################
# CLEAR sweeps       #
######################

# Rampdown sweepfunctions
class CBox_CLEAR_amplitude_1(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_amplitude_1, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse amplitude 1'
        self.parameter_name = 'CLEAR pulse amplitude 1'
        self.unit = 'mV'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_amp_CLEAR_1(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_amplitude_2(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_amplitude_2, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse amplitude 2'
        self.parameter_name = 'CLEAR pulse amplitude 2'
        self.unit = 'mV'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_amp_CLEAR_2(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_amplitude_a1(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_amplitude_a1, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse amplitude a1'
        self.parameter_name = 'CLEAR pulse amplitude a1'
        self.unit = 'mV'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_amp_CLEAR_a1(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_amplitude_a2(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_amplitude_a2, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse amplitude a2'
        self.parameter_name = 'CLEAR pulse amplitude a2'
        self.unit = 'mV'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_amp_CLEAR_a2(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_amplitude_b1(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_amplitude_b1, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse amplitude b1'
        self.parameter_name = 'CLEAR pulse amplitude b1'
        self.unit = 'mV'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_amp_CLEAR_b1(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_amplitude_b2(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_amplitude_b2, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse amplitude b2'
        self.parameter_name = 'CLEAR pulse amplitude b2'
        self.unit = 'mV'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_amp_CLEAR_b2(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_phase_1(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_phase_1, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse phase 1'
        self.parameter_name = 'CLEAR pulse phase 1'
        self.unit = 'deg'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_phase_CLEAR_1(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_phase_2(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_phase_2, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse phase 2'
        self.parameter_name = 'CLEAR pulse phase 2'
        self.unit = 'deg'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_phase_CLEAR_2(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_phase_a1(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_phase_a1, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse phase a1'
        self.parameter_name = 'CLEAR pulse phase a1'
        self.unit = 'deg'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_phase_CLEAR_a1(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_phase_a2(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_phase_a2, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse phase a2'
        self.parameter_name = 'CLEAR pulse phase a2'
        self.unit = 'deg'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_phase_CLEAR_a2(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_phase_b1(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_phase_b1, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse phase b1'
        self.parameter_name = 'CLEAR pulse phase b1'
        self.unit = 'deg'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_phase_CLEAR_b1(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_phase_b2(Soft_Sweep):
    '''
    Setting the amplitude of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_phase_b2, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse phase b2'
        self.parameter_name = 'CLEAR pulse phase b2'
        self.unit = 'deg'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_phase_CLEAR_b2(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_length_unc(Soft_Sweep):
    '''
    Setting the length of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_length_unc, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse length unconditional'
        self.parameter_name = 'CLEAR pulse length unconditional'
        self.unit = 'ns'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_length_CLEAR_unc(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_CLEAR_length_c(Soft_Sweep):
    '''
    Setting the length of the CBox CLEAR pulse
    '''

    def __init__(self, **kw):
        super(CBox_CLEAR_length_unc, self).__init__()
        self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
        self.name = 'CBox CLEAR pulse length conditional'
        self.parameter_name = 'CLEAR pulse length conditional'
        self.unit = 'ns'

    def set_parameter(self, val):
        self.CBox_lut_man_2.set_M_length_CLEAR_c(val)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(1)
        self.CBox_lut_man_2.load_pulses_onto_AWG_lookuptable(2)


class CBox_tng_RO_Pulse_length(Soft_Sweep):
    '''
    Setting the length of the tng Readout Pulse
    '''

    def __init__(self, **kw):
        super(CBox_tng_RO_Pulse_length, self).__init__()
        self.CBox = qt.instruments['CBox']
        self.name = 'CBox_tng_RO_Pulse_length'
        self.parameter_name = 'Readout pulse length'
        self.unit = 'ns'

    def set_parameter(self, val):
        self.CBox.set_tng_readout_pulse_length(val)


class CBox_integration_length(Soft_Sweep):
    '''
    Setting the length of the tng Readout Pulse
    '''

    def __init__(self, **kw):
        super(CBox_integration_length, self).__init__()
        self.CBox = qt.instruments['CBox']
        self.name = 'CBox_integration_length'
        self.parameter_name = 'Readout integration length'
        self.unit = 'ns'

    def set_parameter(self, val):
        self.CBox.set_integration_length(int(val/5))


class CBox_tng_heartbeat_interval(Soft_Sweep):
    '''
    Setting the length of the tng heartbeat interval
    '''

    def __init__(self, **kw):
        super(CBox_tng_heartbeat_interval, self).__init__()
        self.CBox = qt.instruments['CBox']
        self.name = 'CBox_tng_heartbeat_interval'
        self.parameter_name = 'heartbeat_interval'
        self.unit = 'ns'

    def set_parameter(self, val):
        self.CBox.set_tng_heartbeat_interval(val)


class CBox_tng_burst_heartbeat_and_heartbeat_interval(Soft_Sweep):
    '''
    Setting the length burst heartbeat interval
    Setting the heartbeat to: burst heartbeat interval * iterations
                                +200000 for relaxation to steady state
    '''

    def __init__(self, **kw):
        super(CBox_tng_burst_heartbeat_and_heartbeat_interval, self).__init__()
        self.CBox = qt.instruments['CBox']
        self.name = 'CBox_tng_burst_heartbeat_interval'
        self.parameter_name = 'burst_heartbeat_interval'
        self.unit = 'ns'

    def set_parameter(self, val):
        iterations = self.CBox.get_tng_burst_heartbeat_n()
        self.CBox.set_tng_heartbeat_interval(val*iterations+200000)
        self.CBox.set_tng_burst_heartbeat_interval(val)


class CBox_tng_Ramsey_idle_and_heartbeat(Soft_Sweep):
    '''
    Setting the length of the tng Readout Pulse
    '''

    def __init__(self, heartbeat_start, **kw):
        super(CBox_tng_Ramsey_idle_and_heartbeat, self).__init__()
        self.CBox = qt.instruments['CBox']
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.name = 'CBox_tng_Ramsey_idle_and_heartbeat'
        self.parameter_name = 'Ramsey_idle'
        self.unit = 'ns'
        self.heartbeat_start = heartbeat_start

    def set_parameter(self, val):
        self.CBox_lut_man.set_Ramsey_idling(val)
        self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
        self.CBox.set_tng_readout_delay(100+val)
        self.CBox.set_tng_heartbeat_interval(self.heartbeat_start+val)


class CBox_tng_Ramsey_idle_and_heartbeat_v2(Soft_Sweep):
    '''
    Setting the length of the tng Readout Pulse
    Differs from old version that it uses 2! pulses with a delay between them
    '''

    def __init__(self, burst_heartbeat_start, **kw):
        super(CBox_tng_Ramsey_idle_and_heartbeat_v2, self).__init__()
        self.CBox = qt.instruments['CBox']
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.name = 'CBox_tng_Ramsey_idle_and_heartbeat'
        self.parameter_name = 'Ramsey_idle'
        self.unit = 'ns'
        self.burst_heartbeat_start = burst_heartbeat_start

    def set_parameter(self, val):
        self.CBox.set_tng_readout_delay(100)
        self.CBox.set_tng_second_pre_rotation_delay(100+val)
        self.CBox.set_tng_burst_heartbeat_interval(self.burst_heartbeat_start
                                                   + val)


class None_Sweep_tape_restart(Soft_Sweep):

    def __init__(self, sweep_control='soft', **kw):
        super(None_Sweep_tape_restart, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep_tape_restart'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.CBox = qt.instruments['CBox']

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.restart_awg_tape(2)


class prepare_for_conditional_depletion(Soft_Sweep):

    def __init__(self, AllXY_trigger=200, sweep_control='soft', double_pulse_Ramsey_idling=100, RTF_qubit_pulses=False, **kw):
        super(prepare_for_conditional_depletion, self).__init__()
        import Experiments.CLEAR.prepare_for_CLEAR as pfC
        self.pfC = pfC
        self.sweep_control = sweep_control
        self.name = 'prepare_for_conditional_depletion'
        self.parameter_name = 'depletion_pulse_length'
        self.unit = 'ns'
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.AllXY_trigger = AllXY_trigger
        self.double_pulse_Ramsey_idling = double_pulse_Ramsey_idling
        self.CBox = qt.instruments['CBox']
        self.RTF_qubit_pulses = RTF_qubit_pulses

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''

        self.pfC.prepare_for_CLEAR(length=300, depletion=True,
                                   integration=400,
                                   conditional=True, CLEAR_length=val,
                                   CLEAR_double_segment=False,
                                   CLEAR_double_frequency=True,
                                   cost_function='AllXY',
                                   AllXY_trigger=self.AllXY_trigger)
        if self.RTF_qubit_pulses:
            self.CBox_lut_man.set_lut_mapping(['I', 'X180', 'X90_X180_mX90',
                                               'X90_X90', 'X90_X180_X90'])
            # This sets the idling in the X90_X90 element
            self.CBox_lut_man.set_Ramsey_idling(
                self.double_pulse_Ramsey_idling)
            self.CBox.set_tng_readout_delay(
                100 + self.double_pulse_Ramsey_idling)
            self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)


class prepare_for_unconditional_depletion(Soft_Sweep):

    def __init__(self, AllXY_trigger=200, sweep_control='soft', RTF_qubit_pulses=False, double_pulse_Ramsey_idling=100, **kw):
        super(prepare_for_unconditional_depletion, self).__init__()
        import Experiments.CLEAR.prepare_for_CLEAR as pfC
        self.pfC = pfC
        self.sweep_control = sweep_control
        self.name = 'prepare_for_unconditional_depletion'
        self.parameter_name = 'depletion_pulse_length'
        self.unit = 'ns'
        self.CBox_lut_man = qt.instruments['CBox_lut_man']
        self.CBox = qt.instruments['CBox']
        self.AllXY_trigger = AllXY_trigger
        self.RTF_qubit_pulses = RTF_qubit_pulses
        self.double_pulse_Ramsey_idling = double_pulse_Ramsey_idling

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        self.pfC.prepare_for_CLEAR(length=300, depletion=True,
                                   integration=460,
                                   conditional=False,
                                   CLEAR_length=val,
                                   CLEAR_double_segment=True,
                                   CLEAR_double_frequency=True,
                                   cost_function='AllXY',
                                   AllXY_trigger=self.AllXY_trigger)
        if self.RTF_qubit_pulses:
            self.CBox_lut_man.set_lut_mapping(['I', 'X180', 'X90_X180_mX90',
                                               'X90_X90', 'X90_X180_X90'])
            # This sets the idling in the X90_X90 element
            self.CBox_lut_man.set_Ramsey_idling(
                self.double_pulse_Ramsey_idling)
            self.CBox.set_tng_readout_delay(
                100 + self.double_pulse_Ramsey_idling)
            self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
