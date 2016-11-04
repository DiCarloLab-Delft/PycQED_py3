import numpy as np
import logging
import time
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as MA


class Sweep_function(object):
    '''
    sweep_functions class for MeasurementControl(Instrument)
    '''
    def __init__(self, **kw):
        self.set_kw()

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass


class Soft_Sweep(Sweep_function):
    def __init__(self, **kw):
        self.set_kw()
        self.sweep_control = 'soft'

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        pass
##############################################################################


class None_Sweep(Soft_Sweep):
    def __init__(self, sweep_control='soft', sweep_points=None,
                 **kw):
        super(None_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.sweep_points = sweep_points

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        pass


class QX_Sweep(Soft_Sweep):
    """
       QX Input Test
    """
    def __init__(self, qxc, sweep_control='soft', sweep_points=None,**kw):
        super(QX_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = 'QX_Sweep'
        self.parameter_name = 'Error Rate'
        self.unit = 'P'
        self.sweep_points = sweep_points
        self.__qxc = qxc
        self.__qxc.create_qubits(2)
        self.__cnt = 0
	
    # def set_qx_interface(self, qxc):
    	# self.__qxc = qxc

    def set_parameter(self, val):
        #
        #Set the parameter(s) to be sweeped. Differs per sweep function
        #
        print("[+] qx_sweep : parameter :",val)
        circuit_name = ("circuit%i" % self.__cnt)
        print("[+] creating circuit ", circuit_name)
        # self.__qxc.create_circuit(circuit_name,"prepz q0; prepz q1; h q0; h q1; h q1; cnot q0,q1; h q0; h q0; measure q0; measure q1")
        self.__qxc.create_circuit(circuit_name,"prepz q0; h q0; x q0; z q0; y q0; y q0; z q0; x q0; h q0; measure q0;")
	# self.__qxc.create_circuit(circuit_name,"y q0,1.57079632679; rx q0,3.14159265359; rx q0,3.14159265359; ry q0,3.14159265359; rx q0,3.14159265359; ry q0,3.14159265359; ry q0,3.14159265359; ry q0,3.14159265359; rx q0,3.14159265359; rx q0,3.14159265359; ry q0,1.57079632679; rx q0,3.14159265359")
        self.__cnt = self.__cnt+1
        # pass


class Delayed_None_Sweep(Soft_Sweep):
    def __init__(self, sweep_control='soft', delay=0, **kw):
        super().__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.delay = delay
        self.time_last_set = 0
        if delay > 60:
            logging.warning(
                'setting a delay of {:.g}s are you sure?'.format(delay))

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        while (time.time() - self.time_last_set) < self.delay:
            pass  # wait
        self.time_last_set = time.time()


class Dummy_Set_DS_frequency_GHz(Soft_Sweep):
    def __init__(self, **kw):
        super(Dummy_Set_DS_frequency_GHz, self).__init__()
        self.source = qt.instruments['DS']  # a dummy microwave source
        self.sweep_control = 'soft'
        self.name = 'Dummy_src_frequency'
        self.parameter_name = 'frequency'
        self.unit = 'GHz'

    def set_parameter(self, val):
        self.source.set_frequency(val*1e9)


class Dummy_Source_Frequency_GHz(Soft_Sweep):
    def __init__(self, **kw):
        super(Dummy_Source_Frequency_GHz, self).__init__()
        self.source = qt.instruments['DS']
        self.sweep_control = 'soft'
        self.name = 'Dummy_sweep'
        self.parameter_name = 'frequency'
        self.unit = 'GHz'

    def set_parameter(self, val):
        # self.source.set_frequency(val*1e9)
        pass


###################################


class Source_frequency_GHz_Resonator_Scan(Soft_Sweep):
    def __init__(self, start_freq_res, end_freq_res, **kw):
        super(Source_frequency_GHz_Resonator_Scan, self).__init__(**kw)
        self.name = 'Source frequency'
        self.parameter_name = '%s-frequency' % self.S.get_name()
        self.unit = 'GHz'

        self.HM = kw.pop('HM', qt.instruments['HM'])
        self.AWG = kw.pop('AWG', qt.instruments['AWG'])

        self.start_freq_res = start_freq_res
        self.end_freq_res = end_freq_res

    def prepare(self, **kw):
        # Measure resonator
        if 'Nested_MC' not in qt.instruments.get_instrument_names():
            qt.instruments.create('Nested_MC', 'MeasurementControl')
        from pycqed.measurement import calibration_toolbox as cal_tools
        resonator_res = cal_tools.find_resonator_frequency(
            start_freq=self.start_freq_res,
            end_freq=self.end_freq_res,
            MC_name='Nested_MC')
        self.HM.set_frequency(resonator_res['f_resonator']*1e9)
        # super(Source_frequency_GHz_Resonator_Scan, self).prepare(**kw)
        self.S.on()
        self.AWG.start()

    def set_parameter(self, val):
        self.S.set_frequency(val*1e9)


class Bias_Hyst_mV(Soft_Sweep):
    def __init__(self, dac_channel, sleeptime=0, return_point=-1000, **kw):
        super(Bias_Hyst_mV, self).__init__()
        self.dac_channel = dac_channel
        self.name = 'Dac_'+str(dac_channel)+' Voltage'
        self.parameter_name = 'Dac_'+str(dac_channel)
        self.unit = 'mV'
        self.return_point = return_point
        self.sleeptime = sleeptime

    def set_parameter(self, val):
        qt.msleep(self.sleeptime)
        eval("qt.instruments['IVVI'].set_dac%d(self.return_point)" % self.dac_channel)
        eval("qt.instruments['IVVI'].set_dac%d(val)" % self.dac_channel)



class AWG_amp(Soft_Sweep):
    def __init__(self, channel, AWG):
        super().__init__()
        self.name = 'AWG Channel Amplitude'
        self.channel = channel
        self.parameter_name = 'AWG_ch{}_amp'.format(channel)
        self.AWG = AWG
        self.unit = 'V'

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.AWG.stop()
        exec('self.AWG.ch{}_amp({})'.format(self.channel, val))
        self.AWG.start()



class AWG_multi_channel_amplitude(Soft_Sweep):
    '''
    Sweep function to sweep multiple AWG channels simultaneously
    '''
    def __init__(self, AWG, channels, delay=0, **kw):
        super().__init__()
        self.name = 'AWG channel amplitude chs %s' % channels
        self.parameter_name = 'AWG chs %s' % channels
        self.unit = 'V'
        self.AWG = AWG
        self.channels = channels
        self.delay = delay

    def set_parameter(self, val):
        for ch in self.channels:
            self.AWG.set('ch{}_amp'.format(ch), val)
        time.sleep(self.delay)


class AWG_sequence(Soft_Sweep):
    '''
    Sweep AWG sequences
    Note this only works in continuous mode
    '''
    def __init__(self, **kw):
        super(AWG_sequence, self).__init__()
        self.AWG = qt.instruments['AWG']

        self.name = 'AWG sequence'
        self.parameter_name = 'Sequence'
        self.unit = 'name'

    def set_parameter(self, filename):
        self.AWG.set_setup_filename(filename, force_load=True)
        for channel in range(1,4):
            eval('self.AWG.set_ch%d_status("on")' % channel)
        self.AWG.start()
        qt.msleep(0.5)

class AWG_phase_sequence(Soft_Sweep):
    '''
    Sweep AWG sequences
    Note this only works in continuous mode
    '''
    def __init__(self, **kw):
        super(AWG_phase_sequence, self).__init__()
        self.AWG = qt.instruments['AWG']

        self.AWG.set_run_mode('CONT')

        self.name = 'AWG phase sequence'
        self.parameter_name = 'Sequence'
        self.unit = 'name'

    def set_parameter(self, phase):
        #Unfortunately, string formatting is required
        phase_str = str(float(phase)).replace('.05', 'p05')
        phase_str = phase_str.replace('.0', 'p')
        phase_str = phase_str.replace('.', 'p')
        filename = 'MixerCalPhase_{}_5014'.format(phase_str)

        self.AWG.set_setup_filename(filename, force_load=True)
        for channel in range(1,4):
            eval('self.AWG.set_ch%d_status("on")' % channel)
        self.AWG.start()
        qt.msleep(0.5)



class Duplexer_Phase(Soft_Sweep):
    '''
    Sweep Duplexer Phase for Calibration
    '''
    def __init__(self, channel_in, channel_out, delay=.1, **kw):
        super(Duplexer_Phase, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.Duplexer = qt.instruments['Duplexer']
        self.name = 'Duplexer Phase Channel in %s; Channel Out %s ' \
            % (channel_in, channel_out)
        self.parameter_name = 'Phase'
        self.unit = 'a.u.'

        self.delay = delay

    def set_parameter(self, val):
        qt.msleep(self.delay)
        self.Duplexer.set_phase(self.channel_in, self.channel_out, val)


class Duplexer_attenuation(Soft_Sweep):
    '''
    Sweep Duplexer attenuation for Calibration
    '''
    def __init__(self, channel_in, channel_out, **kw):
        super(Duplexer_attenuation, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.Duplexer = qt.instruments['Duplexer']
        self.name = 'Duplexer attenuation Channel in %s; Channel Out %s ' \
            % (channel_in, channel_out)
        self.parameter_name = 'attenuation'
        self.unit = 'DAC value'

    def set_parameter(self, val):
        self.Duplexer.set_attenuation(self.channel_in, self.channel_out, val)


class Duplexer_all_attenuations(Soft_Sweep):
    '''
    Sweep Duplexer Phase for Calibration
    '''
    def __init__(self, **kw):
        super(Duplexer_all_attenuations, self).__init__()
        self.Duplexer = qt.instruments['Duplexer']
        self.name = 'Duplexer attenuations'
        self.parameter_name = 'attenuation'
        self.unit = 'DAC value'

    def set_parameter(self, val):
        self.Duplexer.set_all_attenuations_to(val)


###############################################################################
####################          Hardware Sweeps      ############################
###############################################################################

class Hard_Sweep(Sweep_function):
    def __init__(self, **kw):
        super(Hard_Sweep, self).__init__()
        self.sweep_control = 'hard'
        self.parameter_name = 'None'
        self.unit = 'a.u.'

    def start_acquistion(self):
        pass

# NOTE: AWG_sweeps are located in AWG_sweep_functions


class VNA_sweep(Hard_Sweep):
    def __init__(self):
        super(VNA_sweep,self).__init__()
        self.name = 'VNA_sweep'
        self.parameter_name = 'frequency'
        self.unit = 'GHz'
        self.filename = 'VNA_sweep'

###############################################################################
####################        for JPA calibrations  ############################
###############################################################################


class Bias_Dac_mV_pump_phase_cal(Soft_Sweep):
    def __init__(self, dac_channel, dac_init=True, dac_ref_value=-200,  phase_init=True,
                 pump_source='S3',  sleeptime=0, find_dac=False, required_phase_to_lock_dac=10, **kw):
        super(Bias_Dac_mV_pump_phase_cal, self).__init__()
        self.dac_channel = dac_channel
        self.name = 'Dac_'+str(dac_channel)+' Voltage'
        self.parameter_name = 'Dac_'+str(dac_channel)
        self.unit = 'mV'
        self.sleeptime = sleeptime
        self.phase_init = phase_init
        self.dac_ref_value = dac_ref_value
        self.dac_init = dac_init
        self.pump_source = pump_source
        self.dac_locked = False
        self.required_phase_to_lock_dac = required_phase_to_lock_dac
        self.find_dac = find_dac

    def set_parameter(self, val):
        print("dac_ref_value",self.dac_ref_value)
        if self.find_dac:
            #this first routine locks the IVVI value when a certain phase is measured
            if not val == self.dac_ref_value:
                #skipping the phase calibration value because this could lock it before the sweep starts
                if self.dac_locked:
                    val = self.locked_dac_value
                else:
                    HM = qt.instruments['HM']
                    self.MC_phase = qt.instruments.create('MC_phase',
                                                  'MeasurementControl')
                    self.pump_source = qt.instruments[self.pump_source]
                    #calibrating phase
                    self.MC_phase.set_sweep_function(None_Sweep())
                    self.MC_phase.set_sweep_points(np.arange(0, 30, 20))#[::-1])
                    self.MC_phase.set_detector_function(det.HomodyneDetector())
                    self.MC_phase.run(debug_mode=True, name='pump_phase_cal')
                    MA.MeasurementAnalysis(auto=True)
                    ma = MA.MeasurementAnalysis()
                    ma.get_naming_and_values()
                    #ma.data_file.close()
                    phase_measured = ma.measured_values[1, 0]
                    if phase_measured>self.required_phase_to_lock_dac:
                            self.dac_locked=True
                            self.locked_dac_value = val
                            print("The dac was locked at %s mV measuring a phase of %s degrees" %(val, phase_measured))
                    self.MC_phase.remove()
                    AWG = qt.instruments['AWG']
                    AWG.start()

        print("val", val)

        eval("qt.instruments['IVVI'].set_dac%d(val)" % self.dac_channel)

        if val == self.dac_ref_value:
            #this routine is used to calibrate the generator phase at the first dac value
            if self.phase_init:
                time.sleep(5)
                HM = qt.instruments['HM']
                self.MC_phase = qt.instruments.create('MC_phase',
                                              'MeasurementControl')
                self.pump_source = qt.instruments[self.pump_source]
                #calibrating phase
                self.MC_phase.set_sweep_function(Source_phase_deg(Source=self.pump_source))
                self.MC_phase.set_sweep_points(np.arange(0, 30, 20))#[::-1])
                self.MC_phase.set_detector_function(det.HomodyneDetector())
                self.MC_phase.run(debug_mode=True, name='pump_phase_cal')
                MA.MeasurementAnalysis(auto=True)
                ma = MA.MeasurementAnalysis()
                ma.get_naming_and_values()
                #ma.data_file.close()
                phase_measured = ma.measured_values[1, 0]
                phase_setting = (-170.0 - phase_measured)+360
                self.pump_source.set_phase(phase_setting)
                print("setting phase offset to%s " %phase_setting)
                self.MC_phase.remove()
                AWG = qt.instruments['AWG']
                AWG.start()
        qt.msleep(self.sleeptime)





class HM_frequency_GHz_JPA(Soft_Sweep):
    def __init__(self, **kw):
        super(HM_frequency_GHz_JPA, self).__init__()
        self.HM = qt.instruments['HM']
        self.name = 'HM frequency'
        self.parameter_name = 'frequency'
        self.unit = 'GHz'

    def set_parameter(self, val):
        print(self.HM.get_frequency())
        if abs(self.HM.get_frequency()-val*1e9)>1e3:
            self.HM.set_frequency(val*1e9)
        else:
            print("not changing the freq")
