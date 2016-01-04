from modules.analysis import analysis_toolbox as a_tools
from instrument import Instrument
import logging
import numpy as np
import h5py
import types
import qt
import time
from modules.measurement import sweep_functions as swf
from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.analysis import measurement_analysis as MA
from modules.measurement import AWG_sweep_functions as awg_swf
import imp


class JPA_object(Instrument):
    '''
    Instrument for handling a JPA, putting all functions for calibration etc.
    '''

    def __init__(self, name, reset=False, **kw):

        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['Container', 'JPA-Object'])
        # Qubit parameters

        self.add_parameter('pump_source',
                           tags='pump',
                           type=str, flags=Instrument.FLAG_GETSET)
        self.add_parameter('pump_frequency', units='GHz', tags='pump',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('pump_power', units='dBm',
                           tags='pump',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_channel_coil', units='',
                           tags='coil',
                           type=int, flags=Instrument.FLAG_GETSET)
        self.add_parameter('dac_channel_coil_voltage', units='mV',
                           tags='coil',
                           type=float, flags=Instrument.FLAG_GETSET)
        self.add_parameter('coil_setting_sleeptime', units='s',
                           tags='coil',
                           type=float, flags=Instrument.FLAG_GETSET)

        self.IVVI = qt.instruments['IVVI']
        self.Flux_Control = qt.instruments['Flux_Control']

    def reload_modules(self):
        imp.reload(MA)
        imp.reload(det)
        imp.reload(cdet)
        imp.reload(swf)
        imp.reload(awg_swf)

    def dac_frequency_scan(self, dac_start, dac_stop, dac_step,
                            freq_step,
                            freq_start, freq_stop):
        HM = qt.instruments['HM']
        MC = qt.instruments['MC']
        dac_channel = self.get_dac_channel_coil()
        frequencies = np.arange(freq_start, freq_stop, freq_step) #GHz
        dac_values = np.arange(dac_start, dac_stop, dac_step) #mV
        HM.set_RF_source(self.do_get_pump_source)
        HM.set_RF_power(self.do_get_pump_power)

        MC.set_sweep_function(swf.Bias_Dac_mV_pump_phase_cal(dac_channel,
                            sleeptime=self.do_get_coil_setting_sleeptime,
                            dac_init=True, dac_ref_value=dac_values[0],
                            phase_init=True,
                       pump_source=self.do_get_pump_source))
        MC.set_sweep_points(dac_values)#[::-1])
        MC.set_sweep_function_2D(swf.HM_frequency_GHz_JPA())
        MC.set_sweep_points_2D(frequencies)
        MC.set_detector_function(det.HomodyneDetector())
        MC.run_2D(debug_mode=True,
                  name='JPA_dac_freq_arches_pump_power_%s'%self.do_get_pump_power)
        MA.TwoD_Analysis(auto=True, plot_all=True, label='JPA')

    def dac_power_scan(self, dac_start, dac_stop, dac_step,
                            power_step,
                            power_start, power_stop):
        HM = qt.instruments['HM']
        MC = qt.instruments['MC']
        dac_channel = self.get_dac_channel_coil()
        pump_powers = np.arange(power_start, power_stop, power_step) #dBm
        dac_values = np.arange(dac_start, dac_stop, dac_step) #mV
        HM.set_RF_source(self.do_get_pump_source)
        HM.set_RF_frequency(self.do_get_pump_frequency)

        MC.set_sweep_function(swf.Bias_Dac_mV_pump_phase_cal(dac_channel,
                            sleeptime=self.do_get_coil_setting_sleeptime,
                            dac_init=True, dac_ref_value=dac_values[0],
                            phase_init=True,
                       pump_source=self.do_get_pump_source))
        MC.set_sweep_points(dac_values)#[::-1])
        MC.set_sweep_function_2D(swf.HM_frequency_GHz_JPA())
        MC.set_sweep_points_2D(pump_powers)
        MC.set_detector_function(det.HomodyneDetector())
        MC.run_2D(debug_mode=True,
                  name='JPA_dac_power_arches_pump_frequency_%s'%self.do_get_pump_frequency)
        MA.TwoD_Analysis(auto=True, plot_all=True, label='JPA')

    def calibrate_phase_and_lock_dac(self, required_phase_to_lock_dac, dac_start, dac_stop, dac_step):
        HM = qt.instruments['HM']
        MC = qt.instruments['MC']
        dac_values = np.arange(dac_start, dac_stop, dac_step) #mV
        HM.set_RF_source(self.do_get_pump_source)
        HM.set_RF_frequency(self.do_get_pump_frequency)
        MC.set_sweep_function(swf.Bias_Dac_mV_pump_phase_cal(5, sleeptime=0.2,
                    dac_init=True, dac_ref_value=dac_values[0],  phase_init=True,
                    pump_source='S3', find_dac=True, required_phase_to_lock_dac= required_phase_to_lock_dac))
        MC.set_sweep_points(dac_values)#[::-1])
        MC.set_detector_function(det.HomodyneDetector())
        MC.run(debug_mode=True, name='JPA_find_dac_for_phase_%s_degrees'% required_phase_to_lock_dac)
        MA.MeasurementAnalysis(auto=True, label='JPA')

    def calibrate_gain_SH(self, dac_start, dac_stop, dac_step, gain):
        HM = qt.instruments['HM']
        MC = qt.instruments['MC']
        SH = qt.instruments['SH']
        dac_values = np.arange(dac_start, dac_stop, dac_step) #mV
        HM.set_RF_source(self.do_get_pump_source)
        HM.set_RF_frequency(self.do_get_pump_frequency)
        MC.set_sweep_function(swf.Bias_Dac_mV_pump_phase_cal(5, sleeptime=0.2,
                    dac_init=True, dac_ref_value=dac_values[0],  phase_init=True,
                    pump_source='S3', find_dac=True, required_phase_to_lock_dac= required_phase_to_lock_dac))
        MC.set_sweep_points(dac_values)#[::-1])
        MC.set_detector_function(det.HomodyneDetector())
        name = 'JPA_tune_gain_SH'
        MC.run(debug_mode=True, name=name)
        MA.MeasurementAnalysis(auto=True, label=name)


    def do_get_pump_source(self):
        return self.pump_source

    def do_set_pump_source(self, pump_source):
        self.pump_source = pump_source

    def do_get_pump_frequency(self):
        return self.pump_frequency

    def do_set_pump_frequency(self, pump_frequency):
        self.pump_frequency = pump_frequency

    def do_get_pump_power(self):
        return self.pump_power

    def do_set_pump_power(self, pump_power):
        self.pump_power = pump_power

    def do_get_dac_channel_coil(self):
        return self.dac_channel_coil

    def do_set_dac_channel_coil(self, dac_channel_coil):
        self.dac_channel_coil = dac_channel_coil

    def do_get_dac_channel_coil_voltage(self):
        return self.dac_channel_coil_voltage

    def do_set_dac_channel_coil_voltage(self, dac_channel_coil_voltage):
        self.dac_channel_coil_voltage = dac_channel_coil_voltage

    def do_get_coil_setting_sleeptime(self):
        return self.coil_setting_sleeptime

    def do_set_coil_setting_sleeptime(self, coil_setting_sleeptime):
        self.coil_setting_sleeptime = coil_setting_sleeptime




