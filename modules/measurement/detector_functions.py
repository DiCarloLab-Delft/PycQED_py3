'''
Module containing a collection of sweep functions used by the Measurement Control Instrument.
These are closely related to the sweep functions from modules/measurement_toolbox created by Gijs.
In the future these can possibly be merged
'''
import numpy as np
import cmath #only used to get phase from complex number.
import logging
import time
from modules.analysis import analysis_toolbox as a_tools
from modules.analysis.fit_toolbox import functions as fn
from modules.measurement.waveform_control import pulse
from modules.measurement.waveform_control import pulse_library as pl
from modules.measurement.waveform_control import pulsar
from modules.measurement.waveform_control import element
from modules.measurement.waveform_control import sequence

class Detector_Function(object):
    '''
    Detector_Function class for MeasurementControl(Instrument)
    '''
    def __init__(self, **kw):
        self.set_kw()
        self.name = 'Experiment_name'
        self.value_names = ['val A', 'val B']
        self.value_units = ['arb. units', 'arb. units']

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s'%(key,kw[key]))

    def get_values(self):
        pass

    def prepare(self, **kw):
        pass

    def initialize_data_arrays(self, sweep_points):
        pass

    def finish(self, **kw):
        pass


###############################################################################
###############################################################################
####################             None Detector             ####################
###############################################################################
###############################################################################

class None_Detector(Detector_Function):
    def __init__(self, **kw):
        super(None_Detector,self).__init__()
        self.detector_control = 'soft'
        self.set_kw()
        self.name = 'None_Detector'
        self.value_names = ['None']
        self.value_units = ['None']

    def acquire_data_point(self, **kw):
        '''
        Returns something random for testing
        '''
        return np.random.random()

class Hard_Detector(Detector_Function):
    def __init__(self, **kw):
        super(Hard_Detector, self).__init__()
        self.detector_control = 'hard'

    def prepare(self, sweep_points=None):
        pass

    def finish(self):
        pass


class Soft_Detector(Detector_Function):
    def __init__(self, **kw):
        super(Soft_Detector, self).__init__()
        self.detector_control = 'soft'

    def acquire_data_point(self, **kw):
        return np.random.random()

    def prepare(self):
        pass
###################################################################################
###################################################################################
####################     Hardware Controlled Detectors     ########################
###################################################################################
###################################################################################


class Dummy_Detector_Hard(Hard_Detector):
    def __init__(self, **kw):
        super(Dummy_Detector_Hard, self).__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.name = 'Dummy_Detector'
        self.value_names = ['distance', 'Power']
        self.value_units = ['m', 'nW']

    def initialize_data_arrays(self, sweep_points):
        self.data = np.zeros(len(sweep_points))

    def get_values(self):
        x = np.arange(0, 10, 60)
        self.data = np.sin(x / np.pi)

        return self.data


class TimeDomainDetector(Hard_Detector):

    def __init__(self, **kw):
        super(TimeDomainDetector, self).__init__()
        self.TD_Meas = qt.instruments['TD_Meas']
        self.name = 'TimeDomainMeasurement'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']

    def prepare(self, sweep_points):
        self.TD_Meas.set_NoSegments(len(sweep_points))
        self.TD_Meas.set_cal_mode('None')
        self.TD_Meas.prepare()

    def get_values(self):
        return self.TD_Meas.measure()


class timedomain_single_trace_detector(Soft_Detector):
    '''
    Used for playing acquring single segments with the ATS as a soft sweep.
    This is used for example while reloading lookuptables for each element in
    the CBox AWG's.
    '''

    def __init__(self, **kw):
        super(timedomain_single_trace_detector, self).__init__()
        self.TD_Meas = qt.instruments['TD_Meas']
        self.name = 'TimeDomainMeasurement'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']

    def prepare(self, **kw):
        self.TD_Meas.set_NoSegments(1)
        self.TD_Meas.set_cal_mode('None')
        self.TD_Meas.prepare()

    def acquire_data_point(self, **kw):
        data = self.TD_Meas.measure()
        I = data[0][0]
        Q = data[1][0]
        return [I, Q]


class TimeDomainDetector_cal(Hard_Detector):

    '''
    Takes Calibration points and uses them to redefine I and
    Q.
    '''
    def __init__(self, **kw):
        print('NOTE Time domain detector cal is obsolete')
        print('The rotation is now included in all analysis scripts')
        super(TimeDomainDetector_cal, self).__init__(**kw)
        self.TD_Meas = qt.instruments['TD_Meas']
        self.name = 'TimeDomainMeasurement'
        self.value_names = ['I_raw', 'Q_raw', 'I_cal', 'Q_cal']
        self.value_units = ['V', 'V', 'V', 'V']

    def prepare(self, sweep_points):
        self.TD_Meas.set_NoSegments(len(sweep_points))
        self.TD_Meas.set_cal_mode('ZERO_ONE')
        self.TD_Meas.prepare()

    def get_values(self):
        data = self.TD_Meas.measure()

        return data

class TimeDomainDetector_multiplexed(Hard_Detector):

    def __init__(self, rotate_to_zero=False, **kw):
        super(TimeDomainDetector_multiplexed, self).__init__()
        self.TD_Meas = qt.instruments['TD_Meas']
        self.name = 'TimeDomainMeasurement'
        self.NoIFs = len(self.TD_Meas.get_IF_list())
        self.rotate_to_zero = rotate_to_zero
        self.value_names = []
        for i in range(self.NoIFs):
            self.value_names.append('I_%s' % (i+1))
            self.value_names.append('Q_%s' % (i+1))
        self.value_units = ['V', 'V']*self.NoIFs

    def prepare(self, sweep_points):
        self.TD_Meas.set_NoSegments(len(sweep_points))
        if self.rotate_to_zero is True:
            self.TD_Meas.set_cal_mode('ZERO_ONE')

    def get_values(self):
        return self.TD_Meas.measure()


class TimeDomainDetector_multiplexed_cal(Hard_Detector):

    def __init__(self, **kw):
        super(TimeDomainDetector_multiplexed_cal, self).__init__()
        self.TD_Meas = qt.instruments['TD_Meas']
        self.name = 'TimeDomainMeasurement'
        self.NoIFs = len(self.TD_Meas.get_IF_list())
        self.value_names = []
        for i in range(1, self.NoIFs+1):
            self.value_names.append('I_raw_%s' % i)
            self.value_names.append('Q_raw_%s' % i)
            self.value_names.append('I_cal_%s' % i)
            self.value_names.append('Q_cal_%s' % i)
        self.value_units = ['V']*4*self.NoIFs

    def prepare(self, sweep_points):
        self.TD_Meas.set_NoSegments(len(sweep_points))
        self.TD_Meas.prepare()

    def get_values(self):
        return self.TD_Meas.measure()


class VNA_Detector(Hard_Detector):
    def __init__(self, bw=100, power=-50, averages=1, **kw):
        super(VNA_Detector, self).__init__()
        self.name = 'VNA_Detector'
        self.value_names = ['S21 (ampl)', 'S21 (comp)',
                            'S21 (real)', 'S21 (imag)']
        self.value_units = ['mV', 'mV', 'mV', 'mV']
        self.VNA = qt.instruments['VNA']
        self.bw = bw
        self.power = power
        self.averages = averages

    def prepare(self, sweep_points):
        self.VNA.set_format('COMP')

        fstart = sweep_points[0]
        df = sweep_points[1] - sweep_points[0]
        npoints = len(sweep_points)
        fstop = fstart + npoints * df

        self.VNA.prepare_sweep(fstart, fstop, npoints,
                                   self.bw, self.power, self.averages)

    def get_values(self):
        self.VNA.start_single_sweep()
        freq, S21 = self.VNA.download_trace()
        return (np.abs(S21), np.angle(S21, deg=True), S21.real, S21.imag)

    def finish(self, **kw):
        pass

class VNA_ATT_Detector(Hard_Detector):
    def __init__(self, bw=100, power=-50, averages=1, **kw):
        super(VNA_Detector, self).__init__()
        self.name = 'VNA_Detector'
        self.value_names = ['S21 (ampl)', 'S21 (comp)',
                            'S21 (real)', 'S21 (imag)']
        self.value_units = ['mV', 'mV', 'mV', 'mV']
        self.VNA_ATT = qt.instruments['VNA_ATT']
        self.bw = bw
        self.power = power
        self.averages = averages

    def prepare(self, sweep_points):
        self.VNA_ATT.set_format('COMP')

        fstart = sweep_points[0]
        df = sweep_points[1] - sweep_points[0]
        npoints = len(sweep_points)
        fstop = fstart + npoints * df

        self.VNA_ATT.prepare_sweep(fstart, fstop, npoints,
                                   self.bw, self.power, self.averages)

    def get_values(self):
        self.VNA_ATT.start_single_sweep()
        freq, S21 = self.VNA_ATT.download_trace()
        return (np.abs(S21), S21, S21.real, S21.imag)

    def finish(self, **kw):
        pass

class Signal_Hound_Spectrum_Track(Hard_Detector):

    def __init__(self, start_freq, end_freq, freq_step, source=None, Navg=1, **kw):
        super(Signal_Hound_Spectrum_Track, self).__init__()
        self.SH = qt.instruments['SH']
        if source is not None:
            self.source = qt.instruments[source]
        else:
            self.source = None
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.freq_step = freq_step
        self.name = 'SignalHound_fixed_frequency'
        self.value_names = ['Power']
        self.value_units = ['dBm']
        self.SH.set_frequency(start_freq)
        self.tplot = time.time()
        self.Plotmon = qt.instruments['Plotmon']
        self.Navg = Navg

    def get_values(self, **kw):
        data = np.zeros(len(self.sweep_points))
        for i, freq in enumerate(self.sweep_points):

            if self.source:
                self.source.set_frequency(freq*1e9)
            self.SH.set_frequency(freq)
            self.SH.prepare_for_measurement()
            data[i] = self.SH.get_power_at_freq(Navg=self.Navg)
            self.SH.abort()
            if (time.time() - self.tplot) > 0.5 or (i+1 == len(self.sweep_points)):
                    # only update plot every 0.5s
                    self.tplot = time.time()
                    if self.Plotmon is not None:
                        self.Plotmon.plot2D(1,
                                            [self.sweep_points[:i],
                                             data[:i]])
        return [data]

    def prepare(self, sweep_points, **kw):
        if self.source is not None:
            self.source.on()
        self.SH.set_acquisition_mode('average')
        self.SH.set_span(.25e-3)
        self.SH.set_rbw(25e4)
        self.SH.set_vbw(25e4)
        self.SH.set_device_mode('sweeping')
        self.SH.set_external_reference(True)
        self.SH.set_scale('log-scale')
        self.SH.prepare_for_measurement()
        MC = kw.pop('MC_obj', qt.instruments['MC'])
        self.sweep_points = np.arange(self.start_freq,
                                      self.end_freq, self.freq_step)
        MC.set_sweep_points(self.sweep_points)

    def finish(self, **kw):
        self.SH.abort()


# Detectors for QuTech Control box modes
class QuTechCBox_input_average_Detector(Hard_Detector):
    def __init__(self, AWG='AWG', **kw):
        super(QuTechCBox_input_average_Detector, self).__init__()
        self.CBox = qt.instruments['CBox']
        self.name = 'CBox_Streaming_data'
        self.value_names = ['Ch0', 'Ch1']
        self.value_units = ['a.u.', 'a.u.']
        if AWG is not None:
            self.AWG = qt.instruments[AWG]

    def get_values(self):
        if self.AWG is not None:
            self.AWG.start()
        data = self.CBox.get_input_avg_results()
        return data

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        self.CBox.set('acquisition_mode', 0)
        self.CBox.set('acquisition_mode', 3)

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.CBox.set('acquisition_mode', 0)


class CBox_integrated_average_detector(Hard_Detector):
    def __init__(self, CBox, AWG, seg_per_point=1, **kw):
        '''
        Integration average detector.
        Defaults to averaging data in a number of segments equal to the
        nr of sweep points specificed.

        seg_per_point allows you to use more than 1 segment per sweeppoint.
        this is for example useful when doing a MotzoiXY measurement in which
        there are 2 datapoints per sweep point.
        '''
        super().__init__(**kw)
        self.CBox = CBox
        self.name = 'CBox_integrated_average_detector'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        self.AWG = AWG
        self.seg_per_point = seg_per_point

    def get_values(self):
        succes = False
        i = 0
        while not succes:
            try:
                data = self.CBox.get_integrated_avg_results()
                succes = True
            except:
                logging.warning('Exception caught retrying')
                self.CBox.set('acquisition_mode', 0)
                self.AWG.stop()
                self.CBox.restart_awg_tape(0)
                self.CBox.restart_awg_tape(1)
                self.CBox.restart_awg_tape(2)

                self.CBox.set('acquisition_mode', 4)
                self.AWG.start()  # Is needed here to ensure data aligns with seq elt
            i += 1
            if i > 10:
                break

        return data

    def prepare(self, sweep_points):
        self.CBox.set('nr_samples', self.seg_per_point*len(sweep_points))
        self.AWG.stop()  # needed to align the samples
        self.CBox.set('acquisition_mode', 0)
        self.CBox.set('acquisition_mode', 4)
        self.AWG.start()  # Is needed here to ensure data aligns with seq elt

    def finish(self):
        self.CBox.set('acquisition_mode', 0)


class CBox_single_integration_average_det(Soft_Detector):
    '''
    Detector used for acquiring single points of the CBox while externally
    triggered by the AWG.
    Soft version of the regular integrated avg detector.
    '''
    def __init__(self, CBox, **kw):
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_single_integration_avg_det'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']

    def acquire_data_point(self, **kw):
        self.CBox.set('acquisition_mode', 4)
        data = self.CBox.get_integrated_avg_results()
        self.CBox.set('acquisition_mode', 0)
        return data

    def prepare(self):
        self.CBox.set('nr_samples', 1)
        self.CBox.set('acquisition_mode', 0)

    def finish(self):
        self.CBox.set('acquisition_mode', 0)


class CBox_single_int_avg_with_LutReload(CBox_single_integration_average_det):
    '''
    Detector used for acquiring single points of the CBox while externally
    triggered by the AWG.
    Very similar to the regular integrated avg detector.
    '''
    def __init__(self, CBox, LutMan, reload_pulses='all', awg_nrs=[0], **kw):
        super().__init__(CBox, **kw)
        self.LutMan = LutMan
        self.reload_pulses = reload_pulses
        self.awg_nrs = awg_nrs

    def acquire_data_point(self, **kw):
        #
        # self.LutMan.load_pulse_onto_AWG_lookuptable('X180', 1)
        if self.reload_pulses == 'all':
            for awg_nr in self.awg_nrs:
                self.LutMan.load_pulses_onto_AWG_lookuptable(awg_nr)
        else:
            for pulse_name in self.reload_pulses:
                for awg_nr in self.awg_nrs:
                    self.LutMan.load_pulse_onto_AWG_lookuptable(
                        pulse_name, awg_nr)
        return super().acquire_data_point(**kw)


class CBox_integration_logging_det(Hard_Detector):
    def __init__(self, CBox, AWG, LutMan=None, reload_pulses=False,
                 awg_nrs=None, **kw):
        '''
        If you want AWG reloading you should give a LutMan and specify
        on what AWG nr to reload default is no reloading of pulses.
        '''
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_integration_logging_detector'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        self.AWG = AWG

        self.LutMan = LutMan
        self.reload_pulses = reload_pulses
        self.awg_nrs = awg_nrs

    def get_values(self):
        exception_mode = True
        if exception_mode:
            success = False
            i = 0
            while not success and i < 10:
                try:
                    d = self._get_values()
                    success = True
                except Exception as e:
                    logging.warning('Exception {} caught, retaking data'.format(e))
                    i += 1
        else:
            d = self._get_values()
        return d

    def _get_values(self):
        self.AWG.stop()
        self.CBox.set('acquisition_mode', 0)
        if self.awg_nrs is not None:
            for awg_nr in self.awg_nrs:
                self.CBox.restart_awg_tape(awg_nr)
                if self.reload_pulses:
                    self.LutMan.load_pulses_onto_AWG_lookuptable(awg_nr)
        self.CBox.set('acquisition_mode', 'integration logging')
        self.AWG.start()

        data = self.CBox.get_integration_log_results()

        self.CBox.set('acquisition_mode', 0)
        return data

    def finish(self):
        self.CBox.set('acquisition_mode', 0)
        self.AWG.stop()


class CBox_state_counters_det(Hard_Detector):
    def __init__(self, CBox, AWG, **kw):
        super().__init__()
        self.CBox = CBox
        self.name = 'CBox_state_counters_detector'
        # A and B refer to the counts for the different weight functions
        self.value_names = ['no error A', 'single error A', 'double error A',
                            '|0> A', '|1> A',
                            'no error B', 'single error B', 'double error B',
                            '|0> B', '|1> B', ]
        self.value_units = ['#']*10
        self.AWG = AWG

    def get_values(self):
        success = False
        i = 0
        while not success and i < 10:
            try:
                d = self._get_values()
                success = True
            except Exception as e:
                logging.warning('Exception {} caught, retaking data'.format(e))
                i += 1
        return d

    def _get_values(self):
        self.AWG.stop()
        self.CBox.set('acquisition_mode', 0)
        self.CBox.set('acquisition_mode', 'integration logging')
        self.AWG.start()
        data = self.CBox.get_qubit_state_log_counters()
        self.CBox.set('acquisition_mode', 0)
        return np.concatenate(data)  # concatenates counters A and B

    def finish(self):
        self.CBox.set('acquisition_mode', 0)
        self.AWG.stop()


class CBox_digitizing_shots_det(CBox_integration_logging_det):
    """docstring for  CBox_digitizing_shots_det"""
    def __init__(self, CBox, AWG, threshold,
                 LutMan=None, reload_pulses=False, awg_nrs=None):
        super().__init__(CBox, AWG, LutMan, reload_pulses, awg_nrs)
        self.name = 'CBox_digitizing_shots_detector'
        self.value_names = ['Declared state', 'I', 'Q']
        self.value_units = ['', 'a.u.', 'a.u.']
        self.threshold = threshold

    def get_values(self):
        dat = super().get_values()
        d = dat[0]
        # comparing 8000 vals with threshold takes 3.8us
        # converting to int 10.8us and to float 13.8us, let's see later if we
        # can cut that.
        return (d > self.threshold).astype(int), dat[0], dat[1]



# class QuTechCBox_AlternatingShots_Logging_Detector_Touch_N_Go(Hard_Detector):
#     def __init__(self, NoSamples=10000, AWG='AWG', **kw):
#         super(QuTechCBox_AlternatingShots_Logging_Detector_Touch_N_Go, self).__init__()
#         self.CBox = qt.instruments['CBox']
#         self.name = 'CBox_Streaming_data'
#         self.value_names = ['I_0', 'Q_0', 'I_1', 'Q_1']
#         self.value_units = ['a.u.', 'a.u.', 'a.u.', 'a.u.']
#         self.NoSamples = NoSamples
#         if AWG is not None:
#             self.AWG = qt.instruments[AWG]

#     def get_values(self):
#         exception_mode = True
#         if exception_mode:
#             success = False
#             i = 0
#             while not success and i < 10:
#                 try:
#                     d = self._get_values()
#                     success = True
#                 except Exception as e:
#                     print()
#                     print('Timeout exception caught, retaking data points')
#                     print(str(e))
#                     i += 1
#                     self.CBox.set_run_mode(0)
#                     self.CBox.set('acquisition_mode', 0)
#                     self.CBox.restart_awg_tape(0)
#                     self.CBox.restart_awg_tape(1)
#                     self.CBox.restart_awg_tape(2)
#                     self.prepare()
#         else:
#             d = self._get_values()
#         return d

#     def _get_values(self):
#         raw_data = self.CBox.get_integration_log_results()
#         I_data_0, I_data_1 = a_tools.zigzag(raw_data[0])
#         Q_data_0, Q_data_1 = a_tools.zigzag(raw_data[1])
#         data = [I_data_0, Q_data_0, I_data_1, Q_data_1]
#         print(np.shape(data))
#         return data

#     def prepare(self, **kw):
#         if self.AWG is not None:
#             self.AWG.stop()
#         self.CBox.set('acquisition_mode', 6)
#         self.CBox.set_run_mode(1)

#     def finish(self):
#         counters = np.array(self.CBox.get_sequencer_counters())
#         triggerfraction = float(counters[1])/float(counters[0])
#         print("trigger fraction", triggerfraction)
#         self.CBox.set('acquisition_mode', 0)
#         self.CBox.set_run_mode(0)
#         if self.AWG is not None:
#             self.AWG.stop()


# class QuTechCBox_Shots_Logging_Detector_Touch_N_Go(Hard_Detector):
#     def __init__(self, digitize=True,  timeout=2, **kw):
#         super(QuTechCBox_Shots_Logging_Detector_Touch_N_Go,
#               self).__init__()
#         self.CBox = qt.instruments['CBox']
#         self.name = 'CBox_shots_data'
#         self.digitize = digitize
#         self.timeout = timeout
#         if self.digitize:
#             self.value_names = ['digitized values']
#             self.value_units = ['a.u.']
#             self.threshold_weight0 = self.CBox.get_signal_threshold_line0()
#         else:
#             self.value_names = ['integration result']
#             self.value_units = ['a.u.']

#     def prepare(self, **kw):
#         self.old_timeout = self.CBox.get_measurement_timeout()
#         self.CBox.set_measurement_timeout(self.timeout)
#         # ensures quick detection of the CBox crash

#     def get_values(self):
#         exception_mode = True
#         if exception_mode:
#             success = False
#             i = 0
#             while not success and i < 10:
#                 try:
#                     d = self._get_values()
#                     success = True
#                 except Exception as e:
#                     print()
#                     print('Timeout exception caught, retaking data points')
#                     print(str(e))
#                     i += 1
#                     time.sleep(.1)
#                     self.CBox.set_run_mode(0)
#                     self.CBox.set('acquisition_mode', 0)
#                     self.CBox.restart_awg_tape(0)
#                     self.CBox.restart_awg_tape(1)
#                     self.CBox.restart_awg_tape(2)
#             # mode = raw_input('Press any key to continue')
#         else:
#             d = self._get_values()
#         return d

#     def _get_values(self):
#         '''
#         private version of the acquisition command used for the workaround
#         '''
#         self.CBox.set('acquisition_mode', 6)
#         self.CBox.set_run_mode(1)

#         raw_data = self.CBox.get_integration_log_results()
#         weight0_data = raw_data[0]
#         if self.digitize:
#             data_0 = [1 if d < self.threshold_weight0 else -1
#                       for d in weight0_data]
#         else:
#             data_0 = weight0_data
#         self.CBox.set_run_mode(0)
#         self.CBox.set('acquisition_mode', 0)

#         return data_0

#     def finish(self, **kw):
#         self.CBox.set_measurement_timeout(self.old_timeout)

##############################################################################
##############################################################################
####################     Software Controlled Detectors     ###################
##############################################################################
##############################################################################


class Dummy_Detector_Soft(Soft_Detector):
    def __init__(self, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['mV', 'mV']
        self.i = 0

    def acquire_data_point(self, **kw):
        x = self.i/15.
        self.i += 1
        time.sleep(self.delay)
        return np.array([np.sin(x/np.pi), np.cos(x/np.pi)])


class Source_frequency_detector(Soft_Detector):
    def __init__(self, source, **kw):
        self.set_kw()
        self.S = source
        self.detector_control = 'soft'
        self.name = 'Source frequency detector'
        self.value_names = ['frequency']
        self.value_units = ['Hz']
        self.i = 0

    def acquire_data_point(self, **kw):
        return self.S.get('frequency')

class Function_Detector(Soft_Detector):
    def __init__(self, sweep_function, result_keys, value_names=None,
                 value_units=None, msmt_kw={}, **kw):
        super(Function_Detector, self).__init__()
        self.sweep_function = sweep_function
        self.result_keys = result_keys
        self.value_names = value_names
        self.value_units = value_units
        self.msmt_kw = msmt_kw
        if self.value_names is None:
            self.value_names = result_keys
        if self.value_units is None:
            self.value_units = [""] * len(result_keys)

    def acquire_data_points(self, **kw):
        result = self.sweep_function(**self.msmt_kw)
        return [result[key] for key in result_keys]


class Detect_simulated_hanger_Soft(Soft_Detector):
    def __init__(self, **kw):
        self.set_kw()

        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['mV', 'mV']
        self.source = qt.instruments['DS']

    def acquire_data_point(self, **kw):
        f = self.source.get_frequency()
        f0 = 5.e9
        Q = 10000.
        Qe = 12000.
        theta = 0.2
        A = 50.
        Inoise = np.random.randn()
        Qnoise = np.random.randn()

        IQ = fn.disp_hanger_S21_complex(*(f, f0, Q, Qe, A, theta))
        return IQ.real+Inoise, IQ.imag+Qnoise


class Heterodyne_probe(Soft_Detector):
    def __init__(self, HS, **kw):
        super().__init__(**kw)
        self.HS = HS
        self.name = 'Heterodyne probe'
        self.value_names = ['|S21|', 'S21 angle']  # , 'Re{S21}', 'Im{S21}']
        self.value_units = ['a.u.', 'deg']  # , 'a.u.', 'a.u.']

    def prepare(self):
        self.HS.prepare()

    def acquire_data_point(self, **kw):
        S21 = self.HS.probe()
        return abs(S21), np.angle(S21)/(2*np.pi)*360,# S21.real, S21.imag



class PulsedSpectroscopyDetector(Soft_Detector):

    def __init__(self, AWG_filename='Spec_5014', **kw):
        # AWG_filename='Off_5014'
        super(PulsedSpectroscopyDetector, self).__init__()
        self.Pulsed_Spec = qt.instruments['Pulsed_Spec']
        self.AWG = qt.instruments['AWG']
        self.ATS = qt.instruments['ATS']
        self.name = 'Pulsed_Spec'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']
        self.filename = AWG_filename

    def prepare(self, **kw):
        self.Pulsed_Spec.set_AWG_seq_filename(self.filename)
        self.Pulsed_Spec.initialize_instruments()
        self.AWG.start()
        self.ATS.abort()
        self.ATS.configure_board()

    def acquire_data_point(self, **kw):
        integrated_data = self.Pulsed_Spec.measure()
        return integrated_data

    def finish(self):
        self.AWG.stop()


class Signal_Hound_fixed_frequency(Soft_Detector):
    def __init__(self, signal_hound, frequency=None, Navg=1, delay=0.1,
                 prepare_for_each_point=False, **kw):
        super().__init__()
        self.frequency = frequency
        self.name = 'SignalHound_fixed_frequency'
        self.value_names = ['Power']
        self.value_units = ['dBm']
        self.delay = delay
        self.SH = signal_hound
        if frequency is not None:
            self.SH.set('frequency', frequency)
        self.Navg = Navg
        self.prepare_for_each_point = prepare_for_each_point

    def acquire_data_point(self, **kw):
        if self.prepare_for_each_point:
            self.SH.prepare_for_measurement()
        time.sleep(self.delay)
        return self.SH.get_power_at_freq(Navg=self.Navg)

    def prepare(self, **kw):
        self.SH.prepare_for_measurement()

    def finish(self, **kw):
        self.SH.abort()


class RS_FSV_fixed_frequency(Soft_Detector):

    def __init__(self, frequency=None, delay=.1, bw=300,
                 span=1e-5, npoints=101, **kw):
        super(RS_FSV_fixed_frequency, self).__init__()
        self.FSV = qt.instruments['FSV']
        if frequency is None:
            frequency = self.FSV.get_marker_frequency()
        self.frequency = frequency
        self.name = 'FSV_fixed_frequency'
        self.value_names = ['Power at %.3f Hz' % self.frequency]
        self.value_units = ['dBm']

        self.bw = bw
        self.delay = delay
        self.span = span
        self.npoints = npoints

    def prepare(self, **kw):
        self.FSV.prepare_sweep(self.frequency-self.span/2,
                               self.frequency+self.span/2,
                               self.npoints, self.bw, 1, 1, 'ON')
        self.FSV.set_marker_frequency(self.frequency)
        self.FSV.set_reference_level(-20)

    def acquire_data_point(self, navg=1, **kw):
        time.sleep(.1)
        return np.array([self.FSV.get_marker_power()])

    # def finish(self, **kw):
    #     self.FSV.stop_streaming()


class SH_mixer_skewness_det(Soft_Detector):
    '''
    Based on the "Signal_Hound_fixed_frequency" detector.
    generates an AWG seq to measure sideband transmission

    Inputs:
        frequency       (Hz)
        QI_amp_ratio    (parameter)
        IQ_phase        (parameter)
        SH              (instrument)
        f_mod           (Hz)

    '''
    def __init__(self, frequency, QI_amp_ratio, IQ_phase, SH,
                 I_ch, Q_ch,
                 station,
                 Navg=1, delay=0.1, f_mod=10e6, verbose=False, **kw):
        super(SH_mixer_skewness_det, self).__init__()
        self.SH = SH
        self.frequency = frequency
        self.name = 'SignalHound_mixer_skewness_det'
        self.value_names = ['Power']
        self.value_units = ['dBm']
        self.delay = delay
        self.SH.frequency.set(frequency*1e-9) # Accepts input in Hz
        self.Navg = Navg
        self.QI_amp_ratio = QI_amp_ratio
        self.IQ_phase = IQ_phase
        self.pulsar = station.pulsar
        self.f_mod = f_mod
        self.I_ch = I_ch
        self.Q_ch = Q_ch
        self.verbose = verbose

    def acquire_data_point(self, **kw):
        QI_ratio = self.QI_amp_ratio.get()
        skewness = self.IQ_phase.get()
        if self.verbose:
            print('QI ratio: %.3f' % QI_ratio)
            print('skewness: %.3f' % skewness)
        self.generate_awg_seq(QI_ratio, skewness, self.f_mod)
        self.pulsar.AWG.start()
        time.sleep(self.delay)
        return self.SH.get_power_at_freq(Navg=self.Navg)

    def generate_awg_seq(self, QI_ratio, skewness, f_mod):
        SSB_modulation_el = element.Element('SSB_modulation_el',
                                            pulsar=self.pulsar)
        cos_pulse = pulse.CosPulse(channel=self.I_ch, name='cos_pulse')
        sin_pulse = pulse.CosPulse(channel=self.Q_ch, name='sin_pulse')

        SSB_modulation_el.add(pulse.cp(cos_pulse, name='cos_pulse',
                              frequency=f_mod, amplitude=0.3,
                              length=1e-6, phase=0))
        SSB_modulation_el.add(pulse.cp(sin_pulse, name='sin_pulse',
                              frequency=f_mod, amplitude=0.3*QI_ratio,
                              length=1e-6, phase=90+skewness))

        seq = sequence.Sequence('Sideband_modulation_seq')
        seq.append(name='SSB_modulation_el', wfname='SSB_modulation_el',
                   trigger_wait=False)
        self.pulsar.AWG.stop()
        self.pulsar.program_awg(seq, SSB_modulation_el)

    def prepare(self, **kw):
        self.SH.prepare_for_measurement()

    def finish(self, **kw):
        self.SH.abort()
