import qt
import numpy as np
from instrument import Instrument
import instruments
import time
import sys
import logging
import types
from modules.analysis import analysis_toolbox as a_tools

class PulsedSpectroscopy(Instrument):
    '''
    This is the measurement control instrument for Pulsed Spectroscopy.
    It controls the instruments (AWG, self.ATS) required for Pulsed_Spec.

    It should get a HM style optimization on the number of segments and number
    of buffers.

    Currently it is quite slow. -> Time profiling?

    It works similar to the time domain measurement (TD_Meas) but never stops
    the AWG. Designed to be used with 1 element sequences for which a soft
    sweep of another parameter is varied.
    '''
    def __init__(self, name,
                 RF=qt.instruments['RF'], LO=qt.instruments['LO'],
                 IF=0, cal_mode='AMP_PHASE', **kw):
        logging.info(__name__ + ' : Initializing measurement control instrument')
        Instrument.__init__(self, name, tags=['Measurement_control'])

        self.add_parameter('f_readout', flag=Instrument.FLAG_GETSET, units='Hz',
                           type=float)
        self.add_parameter('IF', units='Hz', flag=Instrument.FLAG_GETSET,
                           type=float)
        # self.add_parameter('NoSweeps', flag=Instrument.FLAG_GETSET,
        #                    type=types.IntType)
        # self.add_parameter('NoSegments', flag=Instrument.FLAG_GETSET,
        #                    type=types.IntType)
        self.add_parameter('NoReps', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('points_per_trace', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('NoCalPoints', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('rotate_data_to_I_axis', flag=Instrument.FLAG_GETSET,
                           type=bool)
        self.add_parameter('Navg', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('int_start', units='ns', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('t_int', units='ns', flag=Instrument.FLAG_GETSET,
                           type=int)
        self.add_parameter('AWG_seq_filename', flag=Instrument.FLAG_GETSET,
                           type=bytes)
        self.add_parameter('cal_mode',
                           flag=Instrument.FLAG_GETSET, type=str)
        self.add_parameter(
            'LO_source', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'LO_power', units='dBm', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'RF_source', type=str,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter(
            'RF_power', units='dBm', type=float,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('single_channel_IQ', type=int,
                           flags=Instrument.FLAG_GETSET)

        self.RF = RF
        self.LO = LO
        self.IF = IF
        self.cal_mode = cal_mode
        self.single_channel_IQ = 0
        instrs = ['ATS_TD', 'AWG', 'Plotmon']
        for instr in instrs:
            exec('self.%s= qt.instruments["%s"]' % (instr, instr))

        self.set_NoReps(3000)

    def measure(self):
        '''
        Measures for NoReps and returns 2 values, either I and Q or
        amp and phase depending on the setting on this instrument.
        '''

        self.int_dat_ch1 = np.zeros(self.NoSegments)
        self.int_dat_ch2 = np.zeros(self.NoSegments)
        # because averaging over NoSweeps happens during acquisition

        int_start = self.int_start
        int_end = self.int_start+self.t_int

        for kk in range(self.Navg):
            self.ATS_TD.start_acquisition()
            dat_ch1 = self.ATS_TD.average_data(1)
            dat_ch2 = self.ATS_TD.average_data(2)

            if self.IF == 0:
                # By looking at this code I assume it is not implemented for
                # IF = 0 as it is just TD code...
                self.int_dat_ch1 += np.average(
                    dat_ch1[:, int_start:int_end], 1)
                self.int_dat_ch2 += np.average(
                    dat_ch2[:, int_start:int_end], 1)

            else:
                # Takes only the data in the integration window
                data_ch1 = dat_ch1[:, int_start:int_end]
                data_ch2 = dat_ch2[:, int_start:int_end]

                # Demodulats and smoothes the data
                cosI = self.cosI[int_start:int_end]
                sinI = self.sinI[int_start:int_end]
                shifted_data_1 = cosI*data_ch1 + sinI*data_ch2
                shifted_data_2 = sinI*data_ch1 - cosI*data_ch2
                smoothed_data_1 = np.zeros(shifted_data_1.shape)
                smoothed_data_2 = np.zeros(shifted_data_2.shape)

                # The inner loop which is repeated for NoSweeps
                for jj in range(int(self.NoSegments)):
                    smoothed_data_1[jj] = a_tools.smooth(
                        shifted_data_1[jj],
                        window_len=np.floor(2/(self.IF*1e-9)),
                        window='flat')
                    smoothed_data_2[jj] = a_tools.smooth(
                        shifted_data_2[jj],
                        window_len=np.floor(2/(self.IF*1e-9)),
                        window='flat')

                # Average
                self.int_dat_ch1 = self.int_dat_ch1 + \
                    np.average(smoothed_data_1, 1)
                self.int_dat_ch2 = self.int_dat_ch2 + \
                    np.average(smoothed_data_2, 1)

                # This is a quick hack to make single channel acquisition work.
                # It supports measuring on a single channel but does not yet support
                # Usint the other channel to measure another signal.
                if self.single_channel_IQ == 1:
                    dat_ch2 = np.zeros(dat_ch1.shape)
                elif self.single_channel_IQ == 2:
                    dat_ch1 = np.zeros(dat_ch1.shape)

            if (time.time() - self.tplot) > 0.5:
                self.tplot = time.time()
                average_transient_I = np.average(1.*dat_ch1, 0)
                average_transient_Q = np.average(1.*dat_ch2, 0)

                tbase = self.ATS_TD.get_t_base()
                # plots data after demodulation
                self.Plotmon.plot2D(5, [tbase, average_transient_I])
                self.Plotmon.plot2D(6, [tbase, average_transient_Q])

        if self.cal_mode == 'AMP_PHASE':
            amp_array = np.sqrt(self.int_dat_ch1**2 +
                                self.int_dat_ch2**2)
            phase_array = 180./(np.pi)*np.arctan2(
                self.int_dat_ch1, self.int_dat_ch2)

            amp = np.mean(amp_array)
            phase = np.mean(phase_array)
            outp = [amp, phase]

        elif self.cal_mode == 'IQ':
            I_mean = np.mean(self.int_dat_ch1)
            Q_mean = np.mean(self.int_dat_ch2)
            outp = [I_mean, Q_mean]

        return(outp)

    def initialize_instruments(self):
        self.set_RF_params()
        self.set_LO_params()
        self.mmtmode = 'averaged'
        self.set_ATS_TD_params()
        self.set_AWG_params()
        self.tplot = time.time()

    def set_LO_params(self):
        self.LO.set_frequency(self.f_readout+self.IF)
        self.LO.on()

    def set_RF_params(self):
        self.RF.set_frequency(self.f_readout)
        self.RF.set_pulsemod_state('On')
        self.RF.set_power(self.RF_power)
        self.RF.on()

    def set_ATS_TD_params(self):
        if self.IF != 0:
            self.ATS_TD.set_finite_IF(True)
        else:
            self.ATS_TD.set_finite_IF(False)
        self.ATS_TD.set_TD_mode(True)
        self.ATS_TD.set_NoSweeps(self.NoSweeps)
        self.ATS_TD.set_NoSegments(self.NoSegments)
        self.ATS_TD.set_points_per_trace(self.get_points_per_trace())
        self.ATS_TD.init()
        if self.IF != 0:
            tbase = self.ATS_TD.get_t_base()
            self.cosI = np.cos(2*np.pi*self.IF*tbase)
            self.sinI = np.sin(2*np.pi*self.IF*tbase)

    def set_AWG_params(self):
        self.AWG.stop()
        self.AWG.set_setup_filename(self.get_AWG_seq_filename())

    def calculate_rotation_matrix(self, delta_I, delta_Q):
        '''
        Calculates a matrix that rotates the data to lie along the Q-axis.
        Input can be either the I and Q coordinates of the zero cal_point or
        the difference between the 1 and 0 cal points.
        '''

        angle = np.arctan2(delta_Q, delta_I)
        rotation_matrix = np.transpose(
            np.matrix([[np.cos(angle), -1*np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]))
        return rotation_matrix

    def _do_get_IF(self):
        return self.IF

    def _do_set_IF(self, IF):
        self.IF = IF

    def _do_get_f_readout(self):
        return self.f_readout

    def _do_set_f_readout(self, frequency):
        self.f_readout = frequency
        self.RF.set_frequency(self.f_readout)
        self.LO.set_frequency(self.f_readout+self.IF)

    def _do_get_NoReps(self):
        return self.NoSweeps

    def _do_set_NoReps(self, NoReps):
        self.NoReps = NoReps
        # self._do_set_NoSweeps(np.ceil(np.sqrt(NoReps)))
        # self._do_set_NoSegments(np.ceil(np.sqrt(NoReps)))
        self.NoSweeps = np.ceil(np.sqrt(NoReps))
        self.NoSegments = self.NoSweeps



    # def _do_get_NoSweeps(self):
    #     return self.NoSweeps

    # def _do_set_NoSweeps(self, NoSweeps):
    #     self.NoSweeps = NoSweeps

    # def _do_get_NoSegments(self):
    #     return self.NoSegments

    # def _do_set_NoSegments(self, NoSegments):
    #     self.NoSegments = NoSegments

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

    def do_set_LO_source(self, name):
        self.LO = qt.instruments[name]

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

    def do_set_NoCalPoints(self, NoCalPoints):
        self.NoCalPoints = NoCalPoints

    def do_get_NoCalPoints(self):
        return self.NoCalPoints

    def do_set_rotate_data_to_I_axis(self, rotate_data_to_I_axis):
        self.rotate_data_to_I_axis = rotate_data_to_I_axis

    def do_get_rotate_data_to_I_axis(self):
        return self.rotate_data_to_I_axis

    def do_set_cal_mode(self, cal_mode):
        '''
        Options are 'IQ' and 'AMP_PHASE'
        '''
        if (cal_mode.upper() == 'IQ'):
            self.cal_mode = 'IQ'
        elif (cal_mode.upper() == 'AMP_PHASE'):
            self.cal_mode = 'AMP_PHASE'
        else:
            logging.error(__name__ + ' : Unable to set cal mode to %s,\
                expected "IQ", or "AMP_PHASE"' % cal_mode)

    def do_get_cal_mode(self):
        return self.cal_mode

    def do_set_single_channel_IQ(self, single_channel_IQ):
        self.single_channel_IQ = single_channel_IQ

    def do_get_single_channel_IQ(self):
        return self.single_channel_IQ
