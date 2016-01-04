import qt
import numpy as np
from instrument import Instrument
import instruments
import time
import sys
import logging
import types
from matplotlib import pyplot as plt
from modules.analysis import analysis_toolbox as a_tools
from functools import reduce

class TimeDomainMeasurement(Instrument):
    '''
    This is the measurement control instrument for time domain measurements.
    It controls the instruments (AWG, self.ATS) required for TD experiments
    '''
    def __init__(self, name,
                 RF='RF', LO='LO',
                 IF=0,
                 multiplex=False, **kw):
        logging.info(__name__ +
                     ' : Initializing TimeDomain instrument')

        Instrument.__init__(self, name, tags=['Measurement_control'])

        self.add_parameter('LO_frequency', units='Hz', flag=Instrument.FLAG_GETSET,
                           type=float)
        self.add_parameter('f_readout', units='Hz',flag=Instrument.FLAG_GETSET,
                           type=float)
        self.add_parameter('f_readout_list', flag=Instrument.FLAG_GETSET,
                           type=list)
        self.add_parameter('IF', units='Hz', flag=Instrument.FLAG_GETSET,
                           type=float)
        self.add_parameter('NoSweeps', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('NoSegments', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('points_per_trace', flag=Instrument.FLAG_GETSET,
                           type=int)
        # self.add_parameter('NoCalPoints', flag=Instrument.FLAG_GETSET,
        #                    type=int)
        self.add_parameter('cal_mode',
                           flag=Instrument.FLAG_GETSET, type=str)
        self.add_parameter('variable_averaging',
                           flag=Instrument.FLAG_GETSET, type=bool)
        self.add_parameter('averaging_threshold',
                           flag=Instrument.FLAG_GETSET, type=float)
        self.add_parameter('Navg', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('int_start', units='ns', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('t_int', units='ns', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('AWG_seq_filename', flag=Instrument.FLAG_GETSET,
                           type=bytes)
        self.add_parameter('force_load_sequence', flag=Instrument.FLAG_GETSET,
                           type=bool)
        self.add_parameter(
            'LO_source', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'LO_power', type=float, units='dBm',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'RF_source', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'RF_power', type=float, units='dBm',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'RF_source_list', type=list,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'RF_power_list', type=list,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'multiplex', type=bool,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'cal_zero_points', type=list,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'cal_one_points', type=list,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('averaging_mode', type=str,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('IF_window', type=int, units='Hz',
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter(
            'shot_mode', type=bool,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'save_average_transients', type=bool,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('single_channel_IQ', type=int,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('filter_window', type=int, units='Hz',
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('use_plotmon', type=bool,
                           flags=Instrument.FLAG_GETSET)
        # self.add_parameter(
        #     'qubit_drive', type=str,
        #     flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)

        self.add_parameter('CBox_tape_mode', type=bool,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('CBox_touch_n_go_mode', type=bool,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('CBox_touch_n_go_save_shots', type=bool,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('CBox_touch_n_go_save_transients', type=bool,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter('CBox_touch_n_go_mode_trigger_fraction', type=float,
                           flags=Instrument.FLAG_GETSET)
        self.add_parameter(
            'MC', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)



        self.set_RF_source(RF)
        self.set_LO_source(LO)
        self.set_IF(IF)

        self.do_set_shot_mode(False)
        self.set_multiplex(False)
        self.set_force_load_sequence(False)

        self.set_variable_averaging(True)
        self.set_averaging_threshold(0.005)
        self.set_averaging_mode('normal')
        self.set_IF_window(0e6)
        self.set_cal_mode('ZERO')
        self.set_shot_mode(False)
        self.set_save_average_transients(False)
        self.single_channel_IQ = 0
        self.set_filter_window(21)
        self.set_use_plotmon(True)
        self.set_CBox_tape_mode(False)
        self.set_CBox_touch_n_go_mode(False)
        self.set_CBox_touch_n_go_save_shots(False)
        self.set_CBox_touch_n_go_save_transients(False)

        instrs = ['ATS_TD', 'AWG', 'Plotmon', 'MC']
        for instr in instrs:
            exec('self.%s= qt.instruments["%s"]' % (instr, instr))



        # self.set_NoCalPoints(10)
    def _do_set_MC(self, MC_name):
        # Needed for datasaving of single shots.
        # It needs to know which MC to ask for data
        self.MC = qt.instruments[MC_name]

    def _do_get_MC(self):
        return self.MC


    def set_AWG_source(self, AWG_name):
        self.AWG = qt.instruments[AWG_name]

    def get_AWG_source(self):
        return self.AWG

    def do_set_use_plotmon(self, val):
        '''
        Disables or enables sending data to the plotmon.
        used for example when taking single traces and plotting is undesired as
        it is done from the soft sweep in MC.
        '''
        self.use_plotmon = val

    def do_get_use_plotmon(self):
        return self.use_plotmon

    def do_set_CBox_tape_mode(self, val):
        '''
        Disables or enables the resetting of CBox tape every AVG round.
        CBox needs to be reset to first segment for each AVG round
        '''
        if val:
            self.CBox = qt.instruments["CBox"]
        self.CBox_tape_mode = val

    def do_get_CBox_tape_mode(self):
        return self.CBox_tape_mode

    def do_set_CBox_touch_n_go_mode(self, val):
        '''
        Disables or enables the trigger counter readout for touch n go
        measurements.
        '''
        if val:
            self.CBox = qt.instruments["CBox"]
        self.CBox_touch_n_go_mode = val

    def do_get_CBox_touch_n_go_mode(self):
        return self.CBox_touch_n_go_mode

    def do_set_CBox_touch_n_go_save_shots(self, val):
        '''
        Disables or enables the trigger counter readout for touch n go
        measurements.
        '''
        if val:
            self.CBox = qt.instruments["CBox"]
        self.CBox_touch_n_go_save_shots = val

    def do_get_CBox_touch_n_go_save_shots(self):
        return self.CBox_touch_n_go_save_shots

    def do_set_CBox_touch_n_go_save_transients(self, val):
        '''
        retreives the CBox transient measurement when in touch n go mode.
        touch n go logging mode should be set to 3.
        '''
        if val:
            self.CBox = qt.instruments["CBox"]
        self.CBox_touch_n_go_save_transients = val

    def do_get_CBox_touch_n_go_save_transients(self):
        return self.CBox_touch_n_go_save_transients

    def do_set_CBox_touch_n_go_mode_trigger_fraction(self, val):
        '''
        averages the trigger fraction over different averaging rounds in
        touch n go measurements.
        '''
        self.CBox_touch_n_go_mode_trigger_fraction = val

    def do_get_CBox_touch_n_go_mode_trigger_fraction(self):
        return self.CBox_touch_n_go_mode_trigger_fraction

    def do_set_cal_zero_points(self, cal_points):
        self.cal_zero_points = cal_points

    def do_get_cal_zero_points(self):
        return self.cal_zero_points

    def do_set_cal_one_points(self, cal_points):
        self.cal_one_points = cal_points

    def do_get_cal_one_points(self):
        return self.cal_one_points

    def set_cal_points(self, cal_points):
        self.cal_points = cal_points
        if isinstance(cal_points, int):
            if self.multiplex == False:
                self.set_cal_zero_points(list(range(-cal_points, -cal_points/2)))
                self.set_cal_one_points(list(range(-cal_points/2, 0)))
            else:
                cal_lst = [list(range(-cal_points, -cal_points/4*3)),
                           list(range(-cal_points/4*3, -cal_points/4*2)),
                           list(range(-cal_points/4*2, -cal_points/4*1)),
                           list(range(-cal_points/4, 0))]
                num_readouts = len(self.f_readout_list)
                self.set_cal_zero_points([cal_lst[0] + cal_lst[2],
                                          cal_lst[0] + cal_lst[1]] +
                                          [cal_lst[0]] * num_readouts)
                self.set_cal_one_points([cal_lst[1] + cal_lst[3],
                                         cal_lst[2] + cal_lst[3]]+
                                         [cal_lst[3]] * num_readouts)
        elif isinstance(cal_points, list):
            if isinstance(cal_points[0], int):
                total_points = sum(cal_points)
                self.set_cal_zero_points(list(range(-total_points, -cal_points[1])))
                self.set_cal_one_points(list(range(-cal_points[1], 0)))
            else:
                self.set_cal_zero_points(cal_points[0])
                self.set_cal_one_points(cal_points[1])
        else:
            raise NameError('Cal points must be integer or list of '
                            'zero points and cal one points')

    def get_cal_points(self):
        return self.cal_points

    def prepare(self):
        self.AWG.stop()
        self.initialize_instruments()
        self.initialize_data_arrays()

    def measure(self):
        if self.save_average_transients or self.shot_mode or self.CBox_touch_n_go_mode:
            data_object = self.MC.get_data_object()
            data_group = data_object['Experimental Data']

        t0 = time.time()
        t = [0, 0, 0, 0]  # used to time different parts of the experiment
        # could use better variable naming.
        if self.shot_mode:
            data_shot = np.zeros((2, self.NoSweeps * self.Navg,
                                  self.NoSegments))

        int_start = self.int_start
        int_end = self.int_start+self.t_int
        tbase = self.ATS_TD.get_t_base()
        if self.IF != 0:
            if self.multiplex is False:
                self.cosI = np.cos(2*np.pi*self.IF*tbase)[int_start:int_end]
                self.sinI = np.sin(2*np.pi*self.IF*tbase)[int_start:int_end]
                self.int_dat_ch1 = np.zeros(self.NoSegments)
                self.int_dat_ch2 = np.zeros(self.NoSegments)
            else:
                self.cosI = []
                self.sinI = []
                self.int_dat_ch1 = [np.zeros(self.NoSegments)] \
                    * len(self.f_readout_list)
                self.int_dat_ch2 = [np.zeros(self.NoSegments)] \
                    * len(self.f_readout_list)

                for IF in self.IF_list:
                    self.cosI.append(np.cos(2*np.pi*IF*tbase)[int_start:int_end])
                    self.sinI.append(np.sin(2*np.pi*IF*tbase)[int_start:int_end])
        else:
            # Required to exist for the shot mode, 0 is if self.IF == 0
            self.cosI = np.cos(0*tbase)[int_start:int_end]
            self.sinI = np.sin(0*tbase)[int_start:int_end]

        if self.save_average_transients is True:
            self.average_transients_I = np.zeros([len(tbase),
                                                 self.NoSegments])
            self.average_transients_Q = np.zeros([len(tbase),
                                                 self.NoSegments])

        t[3] = time.time() - t0
        self.trigger_fraction_cum = 0
        for kk in range(self.Navg):
            qt.msleep(0.1)
            awg_stop_cnt = 0
            while (self.AWG.get_state() != 'Idle'):
                if (awg_stop_cnt < 5e2):  # If longer than 10 secs c
                    print('Waiting for AWG to stop')
                    qt.msleep(0.2)
                    awg_stop_cnt += 1
                else:
                    raise Exception
            ta = time.time()

            if self.CBox_tape_mode is True:
                self.CBox.restart_awg_tape(0)
                self.CBox.restart_awg_tape(1)
                self.CBox.restart_awg_tape(2)
                #resettng the CBOx tape to remain in sync over different cycles
                print("tape mode has been restarted")
            if self.CBox_touch_n_go_mode is True:
                self.CBox.set_acquisition_mode(6)
                self.CBox.set_run_mode(1)
                print("touch 'n go is running")
            self.ATS_TD.start_acquisition()
            t[0] += time.time() - ta
            ta = time.time()
            qt.msleep(0.2)
            self.AWG.start()
            qt.msleep(0.1)
            awg_start_cnt = 0
            t[1] += time.time() - ta
            while self.AWG.get_state() == 'Idle':
                if (awg_start_cnt < 1e3):  # if longer than 100 secs
                    print('Waiting for AWG to start')
                    qt.msleep(1)
                    awg_start_cnt += 1
                else:
                    raise Exception

            ta = time.time()

            dat_ch1 = self.ATS_TD.average_data(1)
            dat_ch2 = self.ATS_TD.average_data(2)
            if self.CBox_touch_n_go_mode is True:

                counters = self.CBox.get_sequencer_counters()
                trigger_fraction = counters[1]/float(counters[0])
                self.trigger_fraction_cum = self.trigger_fraction_cum + trigger_fraction
                print("trigger_fraction ", trigger_fraction)

                self.CBox.set_run_mode(0)
                if self.CBox_touch_n_go_save_shots:
                    raw_data = self.CBox.get_integration_log_results()
                    self.touch_n_go_I_shots = raw_data[0]
                    self.touch_n_go_Q_shots = raw_data[1]
                elif self.CBox_touch_n_go_save_transients:
                    print("getting data from CBox")
                    raw_data = self.CBox.get_input_avg_results()
                    self.touch_n_go_transient_0 = raw_data[0]
                    self.touch_n_go_transient_1 = raw_data[1]

                self.CBox.set_acquisition_mode(0)
                self.CBox.set_acquisition_mode(6)
            # This is a quick hack to make single channel acquisition work.
            # It supports measuring on a single channel but does not yet support
            # Usint the other channel to measure another signal.
            if self.single_channel_IQ == 1:
                dat_ch2 = np.zeros(dat_ch1.shape)
            elif self.single_channel_IQ == 2:
                dat_ch1 = np.zeros(dat_ch1.shape)

            t[2] += time.time() - ta

            self.AWG.stop()

            if self.save_average_transients is True:
                self.average_transients_I += dat_ch1.T
                self.average_transients_Q += dat_ch2.T

            int_start = self.int_start
            int_end = self.int_start+self.t_int

            if self.multiplex is False:
                if self.averaging_mode == 'FFT':
                    # print 'Averaging using FFT mode'
                    t2 = time.time()
                    data = np.array([dat_ch1[:, int_start:int_end],
                                    dat_ch2[:, int_start:int_end]])

                    freqs = np.fft.fftfreq(len(data[0, 0]), d=1e-9)

                    filter_pos_idx = (np.where(
                        (freqs <= self.IF + self.IF_window) &
                        (freqs >= self.IF - self.IF_window))[0])
                    filter_neg_idx = (np.where(
                        (-freqs <= self.IF + self.IF_window) &
                        (-freqs >= self.IF - self.IF_window))[0])

                    filter_neg = np.zeros(len(data[0, 0]))
                    filter_pos = np.zeros(len(data[0, 0]))
                    filter_pos[filter_pos_idx] = 1
                    filter_neg[filter_neg_idx] = 1

                    fft_data = np.zeros(data[0].shape, dtype=complex)
                    for k, datasegm in  enumerate(zip(data[0], data[1])):
                        fft_data[k] = np.fft.fft(datasegm[0]+1.j*datasegm[1])

                    plt.figure('real')
                    plt.clf()
                    plt.figure('imag')
                    plt.clf()
                    for k in [0, 1, 40, 41]:
                        plt.figure('real')
                        plt.plot(freqs, fft_data[k].real, label='segm %d' % k)
                        # plt.plot(freqs, filter_pos * fft_data[k].real, 'o', label='segm %d' % k)
                        plt.figure('imag')
                        plt.plot(freqs, fft_data[k].imag, label='segm %d' % k)
                        # plt.plot(freqs, filter_pos * fft_data[k].imag, 'o', label='segm %d' % k)
                    plt.legend()

                    self.int_dat_ch1 = self.int_dat_ch1 + \
                        np.mean(filter_pos * fft_data.real, 1)# +filter_neg * fft_data.real, 1)
                    self.int_dat_ch2 = self.int_dat_ch2 + \
                        np.mean(filter_pos * fft_data.imag, 1)# -filter_neg * fft_data.imag, 1)
                else:
                    t1 = time.time()
                    # Non FFT Version
                    # print 'Averaging using normal mode'
                    if self.IF == 0:
                        self.int_dat_ch1 += np.average(
                            dat_ch1[:, int_start:int_end], 1)
                        self.int_dat_ch2 += np.average(
                            dat_ch2[:, int_start:int_end], 1)
                    else:
                        self.int_dat_ch1 += np.average(np.multiply(
                            self.cosI, dat_ch1[:, int_start:int_end]) +
                            np.multiply(
                            self.sinI, dat_ch2[:, int_start:int_end]),
                            1)

                        self.int_dat_ch2 += np.average(np.multiply(
                            self.sinI, dat_ch1[:, int_start:int_end]) -
                            np.multiply(
                            self.cosI, dat_ch2[:, int_start:int_end]),
                            1)

                outp = [self.int_dat_ch1, self.int_dat_ch2]
                if self.cal_mode == 'NONE':
                    pass
                elif self.cal_mode == 'ZERO':
                    I_zero = np.mean(self.int_dat_ch1[self.cal_zero_points])
                    Q_zero = np.mean(self.int_dat_ch2[self.cal_one_points])
                    M = a_tools.calculate_rotation_matrix(I_zero, Q_zero)

                    outp =  outp + \
                            [np.asarray(elem)[0] for elem in M * outp]
                elif self.cal_mode == 'ZERO_ONE':
                    I_zero = np.mean(self.int_dat_ch1[self.cal_zero_points])
                    Q_zero = np.mean(self.int_dat_ch2[self.cal_zero_points])
                    I_one = np.mean(self.int_dat_ch1[self.cal_one_points])
                    Q_one = np.mean(self.int_dat_ch2[self.cal_one_points])

                    M = a_tools.calculate_rotation_matrix(I_one-I_zero,
                                                          Q_one-Q_zero)
                    outp = [np.asarray(elem)[0] for elem in M * outp]
                    [rotated_data_ch1, rotated_data_ch2] = outp

                    cal_zero = np.mean(rotated_data_ch1[self.cal_zero_points])
                    cal_one = np.mean(rotated_data_ch1[self.cal_one_points])

                    normalized_data_ch1 = a_tools.normalize_TD_data(
                        rotated_data_ch1, cal_zero, cal_one)

                    if self.variable_averaging is True:
                        uncertainty = np.mean([
                            np.std(normalized_data_ch1[self.cal_zero_points]),
                            np.std(normalized_data_ch1[self.cal_one_points])])

                    outp = [self.int_dat_ch1, self.int_dat_ch2,
                            normalized_data_ch1 * (kk+1), rotated_data_ch2]

                for k, data in enumerate(outp):
                    if self.use_plotmon:
                        self.Plotmon.plot2D(k+1, [np.arange(len(data),
                                                            dtype=float),
                                            data / (kk+1)])
            elif self.multiplex is True:

                data_ch1 = dat_ch1[:, int_start:int_end]
                data_ch2 = dat_ch2[:, int_start:int_end]

                if self.averaging_mode == 'FFT':
                    # FFT
                    t0 = time.time()

                    freqs = np.fft.fftfreq(len(data_ch1[0]), d=1e-9)

                    fft_data = np.zeros(data_ch1.shape, dtype=complex)
                    for k, data in  enumerate(zip(data_ch1, data_ch2)):
                        fft_data[k] = np.fft.fft(data[0]+1.j*data[1])

                    for i, IF in enumerate(self.IF_list):
                        filter_pos_idx = (np.where(
                            (freqs <= IF + self.IF_window) &
                            (freqs >= IF - self.IF_window))[0])
                        filter_neg_idx = (np.where(
                            (-freqs <= IF + self.IF_window) &
                            (-freqs >= IF - self.IF_window))[0])

                        filter_neg = np.zeros(len(data[0]))
                        filter_pos = np.zeros(len(data[0]))
                        filter_pos[filter_pos_idx] = 1
                        filter_neg[filter_neg_idx] = 1

                        self.int_dat_ch1[i] = self.int_dat_ch1[i] + \
                            np.mean(filter_pos[i] * fft_data.real +
                                    filter_neg[i] * fft_data.real, 1)
                        self.int_dat_ch2[i] = self.int_dat_ch2[i] + \
                            np.mean(filter_pos[i] * fft_data.imag -
                                    filter_neg[i] * fft_data.imag, 1)

                        if (i < 3) and self.use_plotmon:
                                self.Plotmon.plot2D(1+2*i, [np.arange(
                                    len(self.int_dat_ch1[i]),
                                    dtype=float), self.int_dat_ch1[i] / (kk+1)])
                                self.Plotmon.plot2D(2+2*i, [np.arange(
                                    len(self.int_dat_ch2[i]),
                                    dtype=float), self.int_dat_ch2[i] / (kk+1)])
                else:
                    for i, IF in enumerate(self.IF_list):
                        # Smoothing
                        cosI = self.cosI[i]
                        sinI = self.sinI[i]

                        shifted_data_1 = cosI*data_ch1 + sinI*data_ch2
                        shifted_data_2 = sinI*data_ch1 - cosI*data_ch2

                        smoothed_data_1 = np.zeros(shifted_data_1.shape)
                        smoothed_data_2 = np.zeros(shifted_data_2.shape)

                        for jj in range(int(self.NoSegments)):
                            smoothed_data_1[jj] = a_tools.smooth(
                                shifted_data_1[jj],
                                window_len=self.filter_window,
                                window='flat')
                            smoothed_data_2[jj] = a_tools.smooth(
                                shifted_data_2[jj],
                                window_len=self.filter_window,
                                window='flat')

                        self.int_dat_ch1[i] = self.int_dat_ch1[i] + \
                            np.mean(smoothed_data_1, 1)
                        self.int_dat_ch2[i] = self.int_dat_ch2[i] + \
                            np.mean(smoothed_data_2, 1)

                        if (i < 3) and self.use_plotmon:
                            self.Plotmon.plot2D(1+2*i, [np.arange(
                                len(self.int_dat_ch1[i]),
                                dtype=float), self.int_dat_ch1[i] / (kk+1)])
                            self.Plotmon.plot2D(2+2*i, [np.arange(
                                len(self.int_dat_ch2[i]),
                                dtype=float), self.int_dat_ch2[i] / (kk+1)])
                outp = []
                for i in range(len(self.IF_list)):
                    outp.append([self.int_dat_ch1[i], self.int_dat_ch2[i]])

                if self.cal_mode == 'NONE':
                    pass
                elif self.cal_mode == 'ZERO':
                    for k in range(len(self.IF_list)):
                        I_zero = np.mean(outp[k][0][self.cal_zero_points[k]])
                        Q_zero = np.mean(outp[k][1][self.cal_zero_points[k]])

                        M = a_tools.calculate_rotation_matrix(I_zero, Q_zero)

                        outp[k] =  outp[k] + \
                            [np.asarray(elem)[0] for elem in M * outp[k]]

                elif self.cal_mode == 'ZERO_ONE':
                    for k in range(len(self.IF_list)):
                        I_zero = np.mean(outp[k][0][self.cal_zero_points[k]])
                        Q_zero = np.mean(outp[k][1][self.cal_zero_points[k]])
                        I_one = np.mean(outp[k][0][self.cal_one_points[k]])
                        Q_one = np.mean(outp[k][1][self.cal_one_points[k]])
                        M = a_tools.calculate_rotation_matrix(I_one-I_zero,
                                                              Q_one-Q_zero)
                        [rotated_data_ch1, rotated_data_ch2] = \
                            [np.asarray(elem)[0] for elem in M * outp[k]]

                        cal_zero = np.mean(rotated_data_ch1[self.cal_zero_points[k]])
                        cal_one = np.mean(rotated_data_ch1[self.cal_one_points[k]])

                        normalized_data_ch1 = a_tools.normalize_TD_data(
                            rotated_data_ch1, cal_zero, cal_one)

                        if self.variable_averaging is True:
                            uncertainty = np.mean([
                                np.std(normalized_data_ch1[self.cal_zero_points[k]]),
                                np.std(normalized_data_ch1[self.cal_one_points[k]])])

                        outp[k] = outp[k] + \
                            [normalized_data_ch1 * (kk+1), rotated_data_ch2]

                    outp = reduce(lambda x, y: x+y, outp)
            average_transient_I = np.average(1.*dat_ch1, 0)
            average_transient_Q = np.average(1.*dat_ch2, 0)

            if self.use_plotmon:

                if self.save_average_transients is True:
                    self.Plotmon.plot3D(1, self.average_transients_I*1.)
                    self.Plotmon.plot3D(2, self.average_transients_Q*1.)
                    tbase = self.ATS_TD.get_t_base()
                    self.Plotmon.plot2D(5, [tbase, np.average(
                                        1.*self.average_transients_I, 1)])
                    self.Plotmon.plot2D(6, [tbase, np.average(
                                        1.*self.average_transients_Q, 1)])
                else:
                    self.Plotmon.plot2D(5, [tbase, average_transient_I])
                    self.Plotmon.plot2D(6, [tbase, average_transient_Q])
                    tbase = self.ATS_TD.get_t_base()
                    self.Plotmon.plot3D(1, 1.*dat_ch1)
                    self.Plotmon.plot3D(2, 1.*dat_ch2)



            # np.savez('D:/raw_data_%d.npz'%k, [dat_ch1, dat_ch2])

            self.AWG.stop()

            if self.shot_mode is True:
                t_shot_zero = time.time()
                dat = self.ATS_TD.get_data(silent=True)
                t_get_shots = time.time() - t_shot_zero
                buffers, records, trace_length = dat[0].shape
                for buff in np.arange(buffers):
                    for record in np.arange(records):
                        data_sliced = dat[:, buff, record, int_start:int_end]
                        data_shot[:, kk*self.NoSweeps + buff, record] = \
                            [np.mean(data_sliced[0] * self.cosI +
                             data_sliced[1] * self.sinI),
                             np.mean(data_sliced[0] * self.sinI -
                             data_sliced[1] * self.cosI)]

                t_call_shots = time.time() - t_shot_zero

                print('Getting shots took %.3f s, processing took %.3f s' \
                    %(t_get_shots, t_call_shots))


            percdone = ((kk+1)*1.)/(self.Navg)*100
            elapsed_time = time.time()-t0
            scrmes = '%3d' % percdone + '%' + \
                ' completed, elapsed time: %5.2fs' % (elapsed_time)
            sys.stdout.write(43*'\b'+scrmes)

            if (self.cal_mode is 'ZERO_ONE') and \
                    (self.variable_averaging is True) and \
                    (uncertainty < self.averaging_threshold):

                print('\nNoise within threshold {} - stopping measurement'\
                    .format(self.averaging_threshold))
                break

        # Normalize data, also in the case where averaging is stopped early
        outp = [elem / (kk+1) for elem in outp]

        elapsed_time = time.time()-t0
        scrmes = '100%'+' completed, elapsed time: %5.2fs\n' % \
            (elapsed_time)
        if self.NoSegments !=1:
            sys.stdout.write(43*'\b'+scrmes)

        if self.save_average_transients is True:
            data_group['average_transients_I'] = self.average_transients_I
            data_group['average_transients_Q'] = self.average_transients_Q

        if self.CBox_touch_n_go_mode is True:
            print("saving touch n go")
            trigger_fraction_avg = self.trigger_fraction_cum/self.Navg
            print("trigger fraction average", trigger_fraction_avg)
            self.set_CBox_touch_n_go_mode_trigger_fraction(trigger_fraction_avg)
            if self.CBox_touch_n_go_save_shots:
                data_group['touch_n_go_I_shots'] = self.touch_n_go_I_shots
                data_group['touch_n_go_Q_shots'] = self.touch_n_go_Q_shots
            elif self.CBox_touch_n_go_save_transients:
                print("adding data group for transients")
                data_group['touch_n_go_transient_0'] = self.touch_n_go_transient_0
                data_group['touch_n_go_transient_1'] = self.touch_n_go_transient_1
        if self.shot_mode:
            data_group['single_shot_I'] = data_shot[0]
            data_group['single_shot_Q'] = data_shot[1]
        # print 'Pure Measurement Time = ', t
        # print '\nFinished Sequence'
        return(outp)

    def initialize_instruments(self):
        self.set_RF_params()
        self.set_LO_params()
        self.mmtmode = 'averaged'
        self.set_AWG_params()
        if self.NoSegments == 0:
            self.NoSegments = self.AWG.get_sequence_length()
        self.set_ATS_TD_params()

    def set_LO_params(self):
        # print 'Setting LO parameters'
        if self.multiplex is False:
            self.LO.set_frequency(self.f_readout-self.IF)
        else:
            self.LO.set_frequency(self.LO_frequency)
        self.LO.on()

    def set_RF_params(self):
        # print 'Setting RF parameters'
        if self.multiplex is False:
            self.RF.set_frequency(self.f_readout)
            self.RF.set_pulsemod_state('On')
            self.RF.set_power(self.RF_power)
            #self.RF.on()
        else:
            for i in range(len(self.IF_list)):
                self.RFs[i].set_frequency(self.f_readout_list[i])
                self.RFs[i].set_pulsemod_state('On')
                self.RFs[i].set_power(self.RF_power_list[i])
                self.RFs[i].on()

    def set_ATS_TD_params(self):
        # print 'Setting ATS_TD parameters'
        if self.IF != 0:
            self.ATS_TD.set_finite_IF(True)
        else:
            self.ATS_TD.set_finite_IF(False)
        self.ATS_TD.set_TD_mode(True)
        self.ATS_TD.set_NoSweeps(self.get_NoSweeps())
        self.ATS_TD.set_NoSegments(self.get_NoSegments())
        self.ATS_TD.set_points_per_trace(self.get_points_per_trace())
        self.ATS_TD.init()

    def set_AWG_params(self):
        # print 'Setting AWG parameters'
        self.AWG.stop()
        # Commented out as this is now in the AWG_swee function
        # self.AWG.set_setup_filename(self.get_AWG_seq_filename(),
        #                             force_load=self.force_load_sequence)
        # FIXME: parameter should also be removed.


    def initialize_data_arrays(self):
        self.int_dat_ch1 = np.zeros(self.NoSegments)
        self.int_dat_ch2 = np.zeros(self.NoSegments)

    def calculate_rotation_matrix(self, delta_I, delta_Q):
        '''
        Calculates a matrix that rotates the data to lie along the Q-axis.
        Input can be either the I and Q coordinates of the zero cal_point or
        the difference between the 1 and 0 cal points.

        NOTE: this function exists in analysis aswell. Do not use this version.
        '''

        angle = np.arctan2(delta_Q, delta_I)
        rotation_matrix = np.transpose(
            np.matrix([[np.cos(angle), -1*np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]))
        return rotation_matrix

    def _do_get_averaging_mode(self):
        return self.averaging_mode

    def _do_set_averaging_mode(self, averaging_mode):
        self.averaging_mode = averaging_mode

    def _do_get_IF_window(self):
        return self.IF_window

    def _do_set_IF_window(self, IF_window):
        self.IF_window = IF_window

    def _do_get_IF(self):
        return self.IF

    def _do_set_IF(self, IF):
        self.IF = IF

    def _do_get_f_readout(self):
        return self.f_readout

    def _do_set_f_readout(self, f_readout):
        self.f_readout = f_readout

    def _do_get_NoSweeps(self):
        return self.NoSweeps

    def _do_set_NoSweeps(self, NoSweeps):
        self.NoSweeps = NoSweeps

    def _do_get_NoSegments(self):
        return self.NoSegments

    def _do_set_NoSegments(self, NoSegments):
        self.NoSegments = NoSegments

    def _do_get_points_per_trace(self):
        return self.points_per_trace

    def _do_set_points_per_trace(self, points_per_trace):
        self.points_per_trace = points_per_trace

    def _do_get_AWG_seq_filename(self):
        return self.AWG_seq_filename

    def _do_set_AWG_seq_filename(self, AWG_seq_filename):
        self.AWG_seq_filename = AWG_seq_filename

    def _do_get_Navg(self):
        return self.Navg

    def _do_set_Navg(self, Navg):
        self.Navg = Navg

    def _do_get_int_start(self):
        return self.int_start

    def _do_set_int_start(self, int_start):
        self.int_start = int_start

    def _do_get_t_int(self):
        return self.t_int

    def _do_set_t_int(self, t_int):
        self.t_int = t_int

    def do_set_LO_source(self, val):
        self.LO = qt.instruments[val]

    def do_get_LO_source(self):
        return self.LO.get_name()

    def do_set_LO_power(self, val):
        self.LO_power = val
        self.LO.set_power(val)

    def do_get_LO_power(self):
        return self.LO_power

    def do_set_RF_source(self, val):
        self.RF = qt.instruments[val]

    def do_get_RF_source(self):
        return self.RF.get_name()

    def do_set_RF_power(self, val):
        self.RF_power = val
        self.RF.set_power(val)

    def do_get_RF_power(self):
        return self.RF_power

    def get_RF_status(self):
        return self.RF.get_status()

    def get_LO_status(self):
        return self.LO.get_status()

    def do_set_cal_mode(self, cal_mode):
        if (cal_mode.upper() == 'ZERO'):
            self.cal_mode = 'ZERO'
        elif (cal_mode.upper() == 'ZERO_ONE'):
            self.cal_mode = 'ZERO_ONE'
        elif (cal_mode.upper() == 'NONE'):
            self.cal_mode = 'NONE'
        else:
            logging.error(__name__ + ' : Unable to set cal mode to %s,\
                expected "ZERO", "ZERO_ONE" or "NONE"' % cal_mode)

    def do_get_cal_mode(self):
        return self.cal_mode

    def do_set_variable_averaging(self, variable_averaging):
        self.variable_averaging = variable_averaging

    def do_get_variable_averaging(self):
        return self.variable_averaging

    def do_set_averaging_threshold(self, averaging_threshold):
        self.averaging_threshold = averaging_threshold

    def do_get_averaging_threshold(self):
        return self.averaging_threshold

    def do_get_force_load_sequence(self):
        return self.force_load_sequence

    def do_set_force_load_sequence(self, force_load_sequence):
        self.force_load_sequence = force_load_sequence

    def get_AWG_model(self):
        return self.AWG.get_AWG_model()

    def do_set_multiplex(self, multiplex):
        self.multiplex = multiplex

    def do_get_multiplex(self):
        return self.multiplex

    def do_set_RF_source_list(self, sources):
        self.RFs = []
        for source in sources:
            self.RFs.append(qt.instruments[source])

    def do_get_RF_source_list(self):
        RF_names = []
        for source in self.RFs:
            RF_names.append(source.get_name())
        return RF_names

    def do_set_RF_power_list(self, val):
        self.RF_power_list = val
        for i, source in enumerate(self.RFs):
            source.set_power(val[i])

    def do_get_RF_power_list(self):
        return self.RF_power_list

    def _do_get_f_readout_list(self):
        return self.f_readout_list

    def _do_set_f_readout_list(self, f_readout_list):
        self.f_readout_list = f_readout_list
        if hasattr(self, 'LO_frequency'):
            self.IF_list = [f - self.LO_frequency for f in f_readout_list]

    def _do_get_LO_frequency(self):
        return self.LO_frequency

    def _do_set_LO_frequency(self, LO_frequency):
        self.LO_frequency = LO_frequency
        if hasattr(self, 'f_readout_list'):
            self.IF_list = [f - LO_frequency for f in self.f_readout_list]

    def get_IF_list(self):
        return self.IF_list

    def do_set_shot_mode(self, shot_mode):
        self.shot_mode = shot_mode

    def do_get_shot_mode(self):
        return self.shot_mode

    def do_set_save_average_transients(self, save_average_transients):
        self.save_average_transients = save_average_transients

    def do_get_save_average_transients(self):
        return self.save_average_transients

    def do_set_single_channel_IQ(self, single_channel_IQ):
        self.single_channel_IQ = single_channel_IQ

    def do_get_single_channel_IQ(self):
        return self.single_channel_IQ

    def do_set_filter_window(self, filter_window):
        self.filter_window = filter_window

    def do_get_filter_window(self):
        return self.filter_window
