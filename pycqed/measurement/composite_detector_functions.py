import numpy as np
import time
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import CBox_sweep_functions as CB_swf
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from qcodes.instrument.parameter import ManualParameter
import imp
import matplotlib.pyplot as plt
imp.reload(awg_swf)


class Qubit_Characterization_Detector(det.Soft_Detector):
    '''
    Performs a set of measurements that finds f_resonator, f_qubit,
    T1, T2*, and T2-echo.
    '''

    def __init__(self, qubit,
                 pulse_amp_guess=0.7,
                 spec_start=None,
                 spec_stop=None,
                 AWG_name='AWG',
                 spec_sweep_range=0.04,
                 HM_sweep_range=0.01,
                 resonator_use_min=False,
                 freq_calc='dac',
                 pulsed=False,
                 res_fitting_model='hanger',
                 Rabi_after_Ramsey=False,
                 **kw):
        # import placed here to prevent circular import statement
        #   as some cal_tools use composite detectors.
        from pycqed.measurement import calibration_toolbox as cal_tools
        imp.reload(cal_tools)
        self.cal_tools = cal_tools
        self.detector_control = 'soft'
        self.name = 'Qubit_Characterization'
        self.value_names = ['f_resonator', 'f_resonator_stderr',
                            'f_qubit', 'f_qubit_stderr',
                            'pulse_amp_ch1', 'pulse_amp_ch2',
                            'T1', 'T1_stderr',
                            'T2_star', 'T2_star_stderr',
                            'T2_echo', 'T2_echo_stderr']
        self.value_units = ['GHz', 'GHz', 'GHz', 'GHz', 'V', 'V',
                            'us', 'us', 'us', 'us', 'us', 'us']

        # Add all the relevant instruments
        self.cal_tools = cal_tools
        self.qubit = qubit
        self.freq_calc = freq_calc
        self.pulsed = pulsed

        self.resonator_use_min = resonator_use_min
        self.nested_MC_name = 'MC_Qubit_Characterization_detector'
        self.AWG = qt.instruments[AWG_name]
        self.nested_MC = qt.instruments[self.nested_MC_name]
        self.qubit_drive_ins = qt.instruments[self.qubit.get_qubit_drive()]
        self.HM = qt.instruments['HM']
        self.Pulsed_Spec = qt.instruments['Pulsed_Spec']
        self.TD_Meas = qt.instruments['TD_Meas']

        if self.qubit.get_pulse_amp_control() == 'Duplexer':
            self.Duplexer = qt.instruments['Duplexer']

        self.spec_sweep_range = spec_sweep_range
        self.HM_sweep_range = HM_sweep_range
        # Setting the constants
        self.pulse_amp_guess = pulse_amp_guess
        self.spec_start = spec_start
        self.spec_stop = spec_stop
        self.calreadoutevery = 1
        self.loopcnt = 0

        self.Rabi_after_Ramsey = Rabi_after_Ramsey

        self.T1_stepsize = 1500
        self.Echo_stepsize = 500

        self.res_fitting_model = res_fitting_model

    def prepare(self, **kw):

        self.nested_MC = qt.instruments.create(
            self.nested_MC_name,
            'MeasurementControl')

    def acquire_data_point(self, *args, **kw):

        t_zero = time.time()
        self.loopcnt += 1
        self.switch_to_freq_sweep()

        cur_f_RO = self.qubit.get_current_RO_frequency()

        if self.qubit.get_RF_source() is not None:
            self.HM.set_RF_source(self.qubit.get_RF_source())

        self.HM.set_RF_power(self.qubit.get_RF_CW_power())

        if np.mod(self.loopcnt, self.calreadoutevery) == 0:  # Added by Leo
            resonator_scan = self.cal_tools.find_resonator_frequency(
                MC_name=self.nested_MC_name,
                start_freq=cur_f_RO - self.HM_sweep_range / 2,
                end_freq=cur_f_RO + self.HM_sweep_range / 2,
                use_min=self.use_resonator_min,
                suppress_print_statements=True,
                fitting_model=self.res_fitting_model)

            f_resonator = resonator_scan['f_resonator']
            f_resonator_stderr = resonator_scan['f_resonator_stderr']
            print('Readout frequency: ', f_resonator)

        self.qubit.set_current_RO_frequency(f_resonator)

        if self.pulsed is True:
            if self.qubit.get_RF_source() is not None:
                self.Pulsed_Spec.set_RF_source(self.qubit.get_RF_source())
            self.Pulsed_Spec.set_RF_power(self.qubit.get_RF_CW_power())
            self.Pulsed_Spec.set_f_readout(
                self.qubit.get_current_RO_frequency()*1e9)
        else:
            self.HM.set_RF_power(self.qubit.get_RF_CW_power())
            self.HM.set_frequency(self.qubit.get_current_RO_frequency()*1e9)
        qubit_scan = self.cal_tools.find_qubit_frequency_spec(
                MC_name=self.nested_MC_name,
                qubit=self.qubit,
                start_freq=self.spec_start,
                end_freq=self.spec_stop,
                spec_sweep_range=self.spec_sweep_range,
                source_power='qubit',
                freq_calc=self.freq_calc,
                suppress_print_statements=True, pulsed=self.pulsed)
        f_qubit = qubit_scan['f_qubit']

        print('Estimated qubit frequency: ', f_qubit)
        self.qubit.set_current_frequency(f_qubit)
        self.HM.set_sources('Off')

        #############################
        # Start of Time Domain part #
        #############################

        self.TD_Meas.set_f_readout(self.qubit.get_current_RO_frequency()*1e9)
        self.qubit_drive_ins.set_frequency(
            (self.qubit.get_current_frequency() +
             self.qubit.get_sideband_modulation_frequency()) * 1e9)
        self.switch_to_time_domain_measurement()

        # Calibrate pulse Amplitude
        if self.pulse_amp_guess is not None:
            self.qubit.set_pulse_amplitude_I(self.pulse_amp_guess)
            self.qubit.set_pulse_amplitude_Q(self.pulse_amp_guess)

        amp_ch1, amp_ch2 = self.cal_tools.calibrate_pulse_amplitude(
            MC_name=self.nested_MC_name,
            qubit=self.qubit,
            # max nr iterations is kept lower than when full tuneup is required
            max_nr_iterations=5, desired_accuracy=.1, Navg=2,
            # desired accuracy is not very high as it only needs to be good
            # enough for a Ramsey, T1 and T2-echo
            suppress_print_statements=False)

        self.qubit.set_pulse_amplitude_I(amp_ch1)
        self.qubit.set_pulse_amplitude_Q(amp_ch2)

        [f_qubit, f_qubit_stderr, T2_star, T2_star_stderr, stepsize] = \
            self.cal_tools.find_qubit_frequency_ramsey(
                MC_name=self.nested_MC_name,
                qubit=self.qubit,
                suppress_print_statements=False)
        print('Measured T2_star: %.4f +/- %.4f ' % (T2_star, T2_star_stderr))

        if self.Rabi_after_Ramsey is True:
            amp_ch1, amp_ch2 = self.cal_tools.calibrate_pulse_amplitude(
                MC_name=self.nested_MC_name,
                qubit=self.qubit,
                # max nr iterations is kept lower than when full tuneup is required
                max_nr_iterations=5, desired_accuracy=.05, Navg=5,
                # desired accuracy is not very high as it only needs to be good
                # enough for a Ramsey, T1 and T2-echo
                suppress_print_statements=False)

            self.qubit.set_pulse_amplitude_I(amp_ch1)
            self.qubit.set_pulse_amplitude_Q(amp_ch2)

        #  T1
        self.nested_MC.set_detector_function(det.TimeDomainDetector_cal())

        if self.qubit.get_pulse_amp_control() == 'AWG':
            self.nested_MC.set_sweep_function(awg_swf.T1(
                stepsize=self.T1_stepsize,
                gauss_width=self.qubit.get_gauss_width()))
        elif self.qubit.get_pulse_amp_control() == 'Duplexer':
            self.nested_MC.set_sweep_function(awg_swf.T1(
                stepsize=self.T1_stepsize,
                gauss_width=self.qubit.get_gauss_width(),
                Duplexer=True))

        self.nested_MC.run()
        T1_a = ma.T1_Analysis(auto=True, close_file=False)
        T1, T1_stderr = T1_a.get_measured_T1()
        T1_a.finish()
        print('Measured T1: %.4f +/- %.4f ' % (T1, T1_stderr))

        # T2-Echo
        if self.qubit.get_pulse_amp_control() == 'AWG':
            self.nested_MC.set_sweep_function(awg_swf.Echo(
                stepsize=self.Echo_stepsize,
                gauss_width=self.qubit.get_gauss_width()))
        elif self.qubit.get_pulse_amp_control() == 'Duplexer':
            self.nested_MC.set_sweep_function(awg_swf.Echo(
                stepsize=self.Echo_stepsize,
                gauss_width=self.qubit.get_gauss_width(),
                Duplexer=True))

        self.nested_MC.run()
        qt.msleep(1)
        T2_a = ma.Ramsey_Analysis(auto=True, close_file=False, label='Echo')
        T2_echo, T2_echo_stderr = T2_a.get_measured_T2_star()
        T2_a.finish()
        print('Measured T2_echo: %.4f +/- %.4f ' % (T2_echo, T2_echo_stderr))

        print('Iteration took %s' % (time.time()-t_zero))
        self.loopcnt += 1

        return_vals = [f_resonator, f_resonator_stderr,
                       f_qubit, f_qubit_stderr,
                       amp_ch1, amp_ch2,
                       T1, T1_stderr,
                       T2_star, T2_star_stderr,
                       T2_echo, T2_echo_stderr]
        print(return_vals)
        return return_vals



    def switch_to_freq_sweep(self):
        self.qubit_drive_ins.off()
        if self.qubit.get_pulse_amp_control == 'Duplexer':
            self.Duplexer.set_all_switches_to('OFF')
            self.Duplexer.set_switch(3, self.qubit.get_duplexer_output_channel(),
                                  'ON')
        self.AWG.start()
        self.HM.init()
        self.AWG.stop()
        self.HM.set_sources('On')

    def switch_to_time_domain_measurement(self):
        self.HM.set_sources('Off')
        self.TD_Meas.set_RF_source(self.qubit.get_RF_source())
        self.TD_Meas.set_RF_power(self.qubit.get_RF_CW_power())
        self.qubit_drive_ins.set_power(self.qubit.get_drive_power())
        self.qubit_drive_ins.on()

        if self.qubit.get_pulse_amp_control() == 'Duplexer':
            print('Duplexer Time Domain prep')
            self.Duplexer.set_all_switches_to('OFF')
            self.Duplexer.set_switch(3,
                                  self.qubit.get_duplexer_output_channel(),
                                  'OFF')
            self.Duplexer.set_switch(1,
                                  self.qubit.get_duplexer_output_channel(),
                                  'ON')
            self.Duplexer.set_attenuation(1,
                                       self.qubit.get_duplexer_output_channel(),
                                       self.pulse_amp_guess)
            self.Duplexer.set_switch(2, self.qubit.get_duplexer_output_channel(),
                                  'ON')
            self.Duplexer.set_attenuation(2,
                                       self.qubit.get_duplexer_output_channel(),
                                       0)


class TimeDomainDetector_integrated(det.Soft_Detector):
    '''
    This is a detector that turns the hard time Domain Detector into
    a soft detector by integrating over it and returning a number instead
    of a sweep.
    Technically it is averaging and not integrating
    currently the sweepfunction is hard coded.
    '''

    def __init__(self, **kw):
        # super(TimeDomainDetector_integrated, self).__init__()
        self.detector_control = 'soft'
        self.name = 'Integrated_TimeDomain'
        self.value_names = ['I_int', 'Q_int']
        self.value_units = ['V', 'V']

    def acquire_data_point(self, *args, **kw):
        data = self.MC_timedomain.run(debug_mode=True)
        integrated_data = np.average(data, 1)

        return integrated_data

    def prepare(self, **kw):
        self.MC_timedomain = qt.instruments.create('MC_timedomain',
                                                   'MeasurementControl')
        self.MC_timedomain.set_sweep_function(awg_swf.Off())
        self.MC_timedomain.set_detector_function(
            det.TimeDomainDetector())

    def finish(self, **kw):
        self.MC_timedomain.remove()


class SSRO_Fidelity_Detector_CBox(det.Soft_Detector):
    '''
    Currently only for CBox.
    '''
    def __init__(self, measurement_name, MC, AWG, CBox,
                 RO_pulse_length, RO_pulse_delay, RO_trigger_delay,
                 raw=True, analyze=True, **kw):
        self.detector_control = 'soft'
        self.name = 'SSRO_Fidelity'
        # For an explanation of the difference between the different
        # Fidelities look in the analysis script
        if raw:
            self.value_names = ['F-raw']
            self.value_units = [' ']
        else:
            self.value_names = ['F', 'F corrected']
            self.value_units = [' ', ' ']
        self.measurement_name = measurement_name
        self.NoSamples = kw.get('NoSamples', 8000)  # current max of log mode
        self.MC = MC
        self.CBox = CBox
        self.AWG = AWG

        #self.IF = kw.pop('IF', -20e6)
        self.RO_trigger_delay = RO_trigger_delay
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_pulse_length = RO_pulse_length

        self.i = 0

        self.raw = raw  # Performs no fits if True
        self.analyze = analyze

        self.upload = True

    def prepare(self, **kw):
        self.CBox.set('log_length', self.NoSamples)

        self.MC.set_sweep_function(awg_swf.CBox_OffOn(
            IF=self.IF,
            RO_pulse_delay=self.RO_pulse_delay,
            RO_trigger_delay=self.RO_trigger_delay,
            RO_pulse_length=self.RO_pulse_length,
            AWG=self.AWG, CBox=self.CBox,
            upload=self.upload))

        self.MC.set_detector_function(
            det.CBox_alternating_shots_det(self.CBox, self.AWG))

    def acquire_data_point(self, *args, **kw):
        self.i += 1
        self.MC.run(name=self.measurement_name+'_'+str(self.i))
        if self.analyze:
            ana = ma.SSRO_Analysis(label=self.measurement_name,
                                   no_fits=self.raw, close_file=True)
            # Arbitrary choice, does not think about the deffinition
            if self.raw:
                return ana.F_raw
            else:
                return ana.F_raw, ana.F_corrected


class SSRO_Fidelity_Detector_Tek(det.Soft_Detector):
    '''
    For Qcodes. Readout with CBox, pulse generation with 5014
    '''
    def __init__(self, measurement_name,  MC, AWG, acquisition_instr,
                 pulse_pars, RO_pars, raw=True, analyze=True, upload=True,
                 IF=None, weight_function_I=0, weight_function_Q=1,
                 optimized_weights=False, one_weight_function_UHFQC=False,
                 wait=0.0, close_fig=True, SSB=False,
                 nr_averages=1024, integration_length=1e-6,
                 nr_shots=4094, **kw):
        self.detector_control = 'soft'
        self.name = 'SSRO_Fidelity'
        # For an explanation of the difference between the different
        # Fidelities look in the analysis script
        if raw:
            self.value_names = ['F_a', 'theta']
            self.value_units = [' ', 'rad']
        else:
            self.value_names = ['F_a', 'F_d', 'SNR']
            self.value_units = [' ', ' ', ' ']
        self.measurement_name = measurement_name
        self.MC = MC
        self.acquisition_instr = acquisition_instr
        self.AWG = AWG
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.optimized_weights = optimized_weights
        self.i = 0
        self.raw = raw  # Performs no fits if True
        self.analyze = analyze
        self.upload = upload
        self.wait = wait
        self.close_fig = close_fig
        self.SSB = SSB
        self.IF = IF
        self.nr_shots = nr_shots
        if 'CBox' in str(self.acquisition_instr):
            self.CBox = self.acquisition_instr
        elif 'UHFQC' in str(self.acquisition_instr):
            self.UHFQC = self.acquisition_instr
        self.nr_averages = nr_averages
        self.integration_length = integration_length
        self.weight_function_I = weight_function_I
        self.weight_function_Q = weight_function_Q
        self.one_weight_function_UHFQC = one_weight_function_UHFQC


    def prepare(self, **kw):
        if not self.optimized_weights:
            self.soft_rotate=True
            self.MC.set_sweep_function(awg_swf.OffOn(
                                       pulse_pars=self.pulse_pars,
                                       RO_pars=self.RO_pars,
                                       upload=self.upload))
            self.MC.set_sweep_points(np.arange(self.nr_shots))
            if 'CBox' in str(self.acquisition_instr):
                self.MC.set_detector_function(
                    det.CBox_integration_logging_det(self.acquisition_instr,
                                                     self.AWG,
                                 integration_length=self.integration_length))
                self.CBox = self.acquisition_instr
                if self.SSB:
                    raise ValueError('SSB is only possible in CBox with optimized weights')
                else:
                    self.CBox.lin_trans_coeffs([1,0,0,1])
                    self.CBox.demodulation_mode('double')
                    if self.IF==None:
                        raise ValueError('IF has to be provided when not using optimized weights')
                    else:
                        self.CBox.upload_standard_weights(IF=self.IF)


            elif 'UHFQC' in str(self.acquisition_instr):
                self.MC.set_detector_function(
                    det.UHFQC_integration_logging_det(self.acquisition_instr,
                                                          self.AWG, channels=[self.weight_function_I,self.weight_function_Q],
                                                          integration_length=self.integration_length, nr_shots=min(self.nr_shots, 4094)))
                if self.SSB:
                    self.UHFQC.prepare_SSB_weight_and_rotation(IF=self.IF, weight_function_I=self.weight_function_I, weight_function_Q=self.weight_function_Q)
                else:
                    if self.IF==None:
                        raise ValueError('IF has to be provided when not using optimized weights')
                    else:
                        self.UHFQC.prepare_DSB_weight_and_rotation(IF=self.IF, weight_function_I=self.weight_function_I, weight_function_Q=self.weight_function_Q)

    def acquire_data_point(self, *args, **kw):
        self.time_start = time.time()
        if self.optimized_weights:
            self.soft_rotate = False
            if 'CBox' in str(self.acquisition_instr):
                self.CBox.nr_averages(int(self.nr_averages))
                if self.SSB:
                    self.CBox.lin_trans_coeffs([1,1,-1,1])
                    # self.CBox.demodulation_mode(1)
                    self.CBox.demodulation_mode('single')
                else:
                    self.CBox.lin_trans_coeffs([1,0,0,1])
                    # self.CBox.demodulation_mode(0)
                    self.CBox.demodulation_mode('double')
                nr_samples = 512
                self.CBox.nr_samples.set(nr_samples)
                SWF = awg_swf.OffOn(
                                    pulse_pars=self.pulse_pars,
                                    RO_pars=self.RO_pars,
                                    pulse_comb='OffOff',
                                    nr_samples=nr_samples)
                SWF.prepare()
                self.CBox.acquisition_mode('idle')
                self.AWG.start()
                self.CBox.acquisition_mode('input averaging')
                inp_avg_res = self.CBox.get_input_avg_results()

                transient0_I = inp_avg_res[0]
                transient0_Q = inp_avg_res[1]

                SWF = awg_swf.OffOn(
                                    pulse_pars=self.pulse_pars,
                                    RO_pars=self.RO_pars,
                                    pulse_comb='OnOn',
                                    nr_samples=nr_samples)
                SWF.prepare()
                self.CBox.acquisition_mode('idle')
                self.CBox.acquisition_mode('input averaging')
                self.AWG.start()
                inp_avg_res = self.CBox.get_input_avg_results()
                self.CBox.acquisition_mode('idle')
                transient1_I = inp_avg_res[0]
                transient1_Q = inp_avg_res[1]

                optimized_weights_I = (transient1_I-transient0_I)
                optimized_weights_I = optimized_weights_I-np.mean(optimized_weights_I)
                weight_scale_factor = 127./np.max(np.abs(optimized_weights_I))
                optimized_weights_I = np.floor(weight_scale_factor*optimized_weights_I).astype(int)

                optimized_weights_Q = (transient1_Q-transient0_Q)
                optimized_weights_Q = optimized_weights_Q-np.mean(optimized_weights_Q)
                weight_scale_factor = 127./np.max(np.abs(optimized_weights_Q))
                optimized_weights_Q = np.floor(weight_scale_factor*optimized_weights_Q).astype(int)


                self.CBox.sig0_integration_weights.set(optimized_weights_I)
                if self.SSB:
                    self.CBox.sig1_integration_weights.set(optimized_weights_Q)  # disabling the Q quadrature
                else:
                    self.CBox.sig1_integration_weights.set(np.multiply(optimized_weights_Q,0))  # disabling the Q quadrature
                self.MC.set_sweep_function(awg_swf.OffOn(
                                           pulse_pars=self.pulse_pars,
                                           RO_pars=self.RO_pars))
                self.MC.set_sweep_points(np.arange(self.nr_shots))
                self.MC.set_detector_function(
                    det.CBox_integration_logging_det(self.CBox, self.AWG, integration_length=self.integration_length))

            elif 'UHFQC' in str(self.acquisition_instr):
                nr_samples = 4096
                self.AWG.stop()
                self.UHFQC.awgs_0_userregs_0(int(self.nr_averages))#0 for rl, 1 for iavg
                self.UHFQC.awgs_0_userregs_1(1)#0 for rl, 1 for iavg
                self.UHFQC.quex_iavg_length(nr_samples)
                self.UHFQC.quex_iavg_avgcnt(int(np.log2(self.nr_averages)))
                self.UHFQC.awgs_0_single(1)
                SWF = awg_swf.OffOn(
                                    pulse_pars=self.pulse_pars,
                                    RO_pars=self.RO_pars,
                                    pulse_comb='OffOff',
                                    nr_samples=nr_samples)
                SWF.prepare()
                self.UHFQC.awgs_0_enable(1)
                try:
                    temp = self.UHFQC.awgs_0_enable()
                except:
                    temp = self.UHFQC.awgs_0_enable()
                del temp
                self.AWG.start()
                while self.UHFQC.awgs_0_enable() == 1:
                    time.sleep(0.01)

                self.channels=[0,1]
                data = ['']*len(self.channels)
                for i, channel in enumerate(self.channels):
                    dataset = eval("self.UHFQC.quex_iavg_data_{}()".format(channel))
                    data[i] = dataset[0]['vector']
                # data = self.UHFQC.single_acquisition(nr_samples,
                #                              self.poll_time, timeout=0,
                #                              channels=set([0,1]),
                #                              mode='iavg')
                # data = np.array([data[key] for key in data.keys()])
                transient0_I = data[0]
                transient0_Q = data[1]
                self.AWG.stop()
                self.UHFQC.quex_iavg_length(nr_samples)

                SWF = awg_swf.OffOn(
                                    pulse_pars=self.pulse_pars,
                                    RO_pars=self.RO_pars,
                                    pulse_comb='OnOn',
                                    nr_samples=nr_samples)
                SWF.prepare()
                self.UHFQC.awgs_0_enable(1)
                try:
                    temp = self.UHFQC.awgs_0_enable()
                except:
                    temp = self.UHFQC.awgs_0_enable()
                del temp

                self.AWG.start()

                while self.UHFQC.awgs_0_enable() == 1:
                    time.sleep(0.01)
                data = ['']*len(self.channels)
                for i, channel in enumerate(self.channels):
                    dataset = eval("self.UHFQC.quex_iavg_data_{}()".format(channel))
                    data[i] = dataset[0]['vector']
                # data = self.UHFQC.single_acquisition(nr_samples,
                #                              self.poll_time, timeout=0,
                #                              channels=set([0,1]),
                #                              mode='iavg')
                # data = np.array([data[key] for key in data.keys()])
                transient1_I = data[0]
                transient1_Q = data[1]

                optimized_weights_I = (transient1_I-transient0_I)
                optimized_weights_I = optimized_weights_I-np.mean(optimized_weights_I)
                weight_scale_factor = 1./np.max(np.abs(optimized_weights_I))
                optimized_weights_I = np.array(weight_scale_factor*optimized_weights_I)


                optimized_weights_Q = (transient1_Q-transient0_Q)
                optimized_weights_Q = optimized_weights_Q-np.mean(optimized_weights_Q)
                weight_scale_factor = 1./np.max(np.abs(optimized_weights_Q))
                optimized_weights_Q = np.array(weight_scale_factor*optimized_weights_Q)

                eval('self.UHFQC.quex_wint_weights_{}_real(np.array(optimized_weights_I))'.format(self.weight_function_I))
                if self.SSB:
                    eval('self.UHFQC.quex_wint_weights_{}_imag(np.array(optimized_weights_Q))'.format(self.weight_function_I))
                    if not self.one_weight_function_UHFQC:
                        eval('self.UHFQC.quex_wint_weights_{}_real(np.array(optimized_weights_I))'.format(self.weight_function_Q))
                        eval('self.UHFQC.quex_wint_weights_{}_imag(np.array(optimized_weights_Q))'.format(self.weight_function_Q))
                    eval('self.UHFQC.quex_rot_{}_real(1.0)'.format(self.weight_function_I))
                    eval('self.UHFQC.quex_rot_{}_imag(-1.0)'.format(self.weight_function_I))
                    if not self.one_weight_function_UHFQC:
                        eval('self.UHFQC.quex_rot_{}_real(1.0)'.format(self.weight_function_Q))
                        eval('self.UHFQC.quex_rot_{}_imag(1.0)'.format(self.weight_function_Q))
                else:
                    eval('self.UHFQC.quex_wint_weights_{}_imag(0*np.array(optimized_weights_Q))'.format(self.weight_function_I)) #disabling the other weight fucntions
                    if not self.one_weight_function_UHFQC:
                        eval('self.UHFQC.quex_wint_weights_{}_real(0*np.array(optimized_weights_I))'.format(self.weight_function_Q))
                        eval('self.UHFQC.quex_wint_weights_{}_imag(0*np.array(optimized_weights_Q))'.format(self.weight_function_Q))
                    eval('self.UHFQC.quex_rot_{}_real(1.0)'.format(self.weight_function_I))
                    eval('self.UHFQC.quex_rot_{}_imag(0.0)'.format(self.weight_function_I))
                    if not self.one_weight_function_UHFQC:

                        eval('self.UHFQC.quex_rot_{}_real(0.0)'.format(self.weight_function_Q))
                        eval('self.UHFQC.quex_rot_{}_imag(0.0)'.format(self.weight_function_Q))
                eval('self.UHFQC.quex_wint_weights_{}_real()'.format(self.weight_function_I)) #reading out weights as check
                eval('self.UHFQC.quex_wint_weights_{}_imag()'.format(self.weight_function_I)) #reading out weights as check
                eval('self.UHFQC.quex_wint_weights_{}_real()'.format(self.weight_function_Q)) #reading out weights as check
                eval('self.UHFQC.quex_wint_weights_{}_imag()'.format(self.weight_function_Q)) #reading out weights as check


                self.MC.set_sweep_function(awg_swf.OffOn(
                                           pulse_pars=self.pulse_pars,
                                           RO_pars=self.RO_pars))
                self.MC.set_sweep_points(np.arange(self.nr_shots))
                self.MC.set_detector_function(
                    det.UHFQC_integration_logging_det(self.UHFQC, self.AWG,
                                                      channels=[self.weight_function_I,self.weight_function_Q],
                                                      integration_length=self.integration_length, nr_shots=min(self.nr_shots, 4094)))
        self.i += 1
        self.MC.run(name=self.measurement_name+'_'+str(self.i))

        if self.analyze:
            ana = ma.SSRO_Analysis(rotate=self.soft_rotate,
                                   label=self.measurement_name,
                                   no_fits=self.raw, close_file=False,
                                   close_fig=True, auto=True)
            if self.optimized_weights:
                # data_group = self.MC.data_object.create_group('Transients Data')
                dset = ana.g.create_dataset('Transients', (nr_samples, 4),
                                            maxshape=(nr_samples, 4))
                dset[:, 0] = transient0_I
                dset[:, 1] = transient0_Q
                dset[:, 2] = transient1_I
                dset[:, 3] = transient1_Q
            ana.data_file.close()

            # Arbitrary choice, does not think about the deffinition
            time_end=time.time()
            nett_wait = self.wait-time_end+self.time_start
            print(self.time_start)
            if nett_wait>0:
                time.sleep(nett_wait)
            if self.raw:
                return ana.F_a, ana.theta
            else:
                return ana.F_a, ana.F_d, ana.SNR
'''
    def acquire_data_point(self, *args, **kw):
        self.time_start = time.time()
        if self.set_integration_weights:
            nr_samples = 512
            self.CBox.nr_samples.set(nr_samples)
            self.MC.set_sweep_function(awg_swf.OffOn(
                                       pulse_pars=self.pulse_pars,
                                       RO_pars=self.RO_pars,
                                       pulse_comb='OffOff',
                                       nr_samples=nr_samples))
            self.MC.set_detector_function(det.CBox_input_average_detector(
                                          self.CBox, self.AWG))
            self.MC.run('Measure_transients_0')
            a0 = ma.MeasurementAnalysis(auto=True, close_fig=self.close_fig)
            self.MC.set_sweep_function(awg_swf.OffOn(
                                       pulse_pars=self.pulse_pars,
                                       RO_pars=self.RO_pars,
                                       pulse_comb='OnOn',
                                       nr_samples=nr_samples))
            self.MC.set_detector_function(det.CBox_input_average_detector(
                                          self.CBox, self.AWG))
            self.MC.run('Measure_transients_1')
            a1 = ma.MeasurementAnalysis(auto=True, close_fig=self.close_fig)
            transient0 = a0.data[1, :]
            transient1 = a1.data[1, :]
            optimized_weights = transient1-transient0
            optimized_weights = optimized_weights+np.mean(optimized_weights)
            self.CBox.sig0_integration_weights.set(optimized_weights)
            self.CBox.sig1_integration_weights.set(
                np.multiply(optimized_weights, self.use_Q))  # disabling the Q quadrature

            self.MC.set_sweep_function(awg_swf.OffOn(
                                       pulse_pars=self.pulse_pars,
                                       RO_pars=self.RO_pars))

            self.MC.set_detector_function(
                det.CBox_integration_logging_det(self.CBox, self.AWG))
        self.i += 1
        self.MC.run(name=self.measurement_name+'_'+str(self.i))
        if self.analyze:
            ana = ma.SSRO_Analysis(label=self.measurement_name,
                                   no_fits=self.raw, close_file=True,
                                   close_fig=self.close_fig)
            # Arbitrary choice, does not think about the deffinition
            time_end=time.time()
            nett_wait = self.wait-time_end+self.time_start
            print(self.time_start)
            if nett_wait>0:
                time.sleep(nett_wait)
            if self.raw:
                return ana.F_raw, ana.theta
            else:
                return ana.F, ana.F_corrected
'''

class CBox_trace_error_fraction_detector(det.Soft_Detector):
    def __init__(self, measurement_name, MC, AWG, CBox,
                 sequence_swf=None,
                 threshold=None,
                 calibrate_threshold='self-consistent',
                 save_raw_trace=False,
                 **kw):
        super().__init__(**kw)
        self.name = measurement_name
        self.threshold = threshold
        self.value_names = ['no err',
                            'single err',
                            'double err']
        self.value_units = ['%', '%', '%']

        self.AWG = AWG
        self.MC = MC
        self.CBox = CBox
        # after testing equivalence this is to be removed
        self.save_raw_trace = save_raw_trace
        self.calibrate_threshold = calibrate_threshold

        self.sequence_swf = sequence_swf

    def calibrate_threshold_conventional(self):
        self.CBox.lin_trans_coeffs.set([1, 0, 0, 1])
        ssro_d = SSRO_Fidelity_Detector_CBox(
            'SSRO_det', self.MC, self.AWG, self.CBox,
            RO_pulse_length=self.sequence_swf.RO_pulse_length,
            RO_pulse_delay=self.sequence_swf.RO_pulse_delay,
            RO_trigger_delay=self.sequence_swf.RO_trigger_delay)
        ssro_d.prepare()
        ssro_d.acquire_data_point()
        a = ma.SSRO_Analysis(auto=True, close_fig=True,
                             label='SSRO', no_fits=True,
                             close_file=True)
        # SSRO analysis returns the angle to rotate by
        theta = a.theta  # analysis returns theta in rad

        rot_mat = [np.cos(theta), -np.sin(theta),
                   np.sin(theta), np.cos(theta)]
        self.CBox.lin_trans_coeffs.set(rot_mat)
        self.threshold = a.V_th_a  # allows
        self.CBox.sig0_threshold_line.set(int(a.V_th_a))
        self.sequence_swf.upload = True
        # make sure the sequence gets uploaded
        return int(self.threshold)

    def calibrate_threshold_self_consistent(self):
        self.CBox.lin_trans_coeffs.set([1, 0, 0, 1])
        ssro_d = CBox_SSRO_discrimination_detector(
            'SSRO-disc-det',
            MC=self.MC, AWG=self.AWG, CBox=self.CBox,
            sequence_swf=self.sequence_swf)
        ssro_d.prepare()
        discr_vals = ssro_d.acquire_data_point()
        # hardcoded indices correspond to values in CBox SSRO discr det
        theta = discr_vals[2] * 2 * np.pi/360

        # Discr returns the current angle, rotation is - that angle
        rot_mat = [np.cos(-1*theta), -np.sin(-1*theta),
                   np.sin(-1*theta), np.cos(-1*theta)]
        self.CBox.lin_trans_coeffs.set(rot_mat)

        # Measure it again to determine the threshold after rotating
        discr_vals = ssro_d.acquire_data_point()
        # hardcoded indices correspond to values in CBox SSRO discr det
        theta = discr_vals[2]
        self.threshold = int(discr_vals[3])

        self.CBox.sig0_threshold_line.set(int(self.threshold))
        return int(self.threshold)

    def prepare(self, **kw):
        self.i = 0
        if self.threshold is None:  # calibrate threshold
            if self.calibrate_threshold is 'conventional':
                self.calibrate_threshold_conventional()
            elif self.calibrate_threshold == 'self-consistent':
                self.calibrate_threshold_self_consistent()
            else:
                raise Exception(
                    'calibrate_threshold "{}"'.format(self.calibrate_threshold)
                    + 'not recognized')
        else:
            self.CBox.sig0_threshold_line.set(int(self.threshold))
        self.MC.set_sweep_function(self.sequence_swf)

        # if self.counters:
        # self.counters_d = det.CBox_state_counters_det(self.CBox, self.AWG)

        self.dig_shots_det = det.CBox_digitizing_shots_det(
            self.CBox, self.AWG,
            threshold=self.CBox.sig0_threshold_line.get())
        self.MC.set_detector_function(self.dig_shots_det)

    def acquire_data_point(self, **kw):
        if self.i > 0:
            # overwrites the upload arg if the sequence swf has it to
            # prevent reloading
            self.sequence_swf.upload = False
        self.i += 1
        if self.save_raw_trace:
            self.MC.run(self.name+'_{}'.format(self.i))
            a = ma.MeasurementAnalysis(auto=False)
            a.get_naming_and_values()
            trace = a.measured_values[0]
            a.finish()  # close the datafile
            return self.count_error_fractions(trace, len(trace))
        else:
            self.sequence_swf.prepare()
            counters = self.counters_d.get_values()
            # no err, single and double for weight A
            return counters[0:3]/self.CBox.get('log_length')*100

    def count_error_fractions(self, trace, trace_length):
        no_err_counter = 0
        single_err_counter = 0
        double_err_counter = 0
        for i in range(len(trace)-2):
            if trace[i] == trace[i+1]:
                # A single error is associated with a qubit error
                single_err_counter += 1
                if trace[i] == trace[i+2]:
                    # If there are two errors in a row this is associated with
                    # a RO error, this counter must be substracted from the
                    # single counter
                    double_err_counter += 1
            else:
                no_err_counter += 1
        return (no_err_counter/len(trace)*100,
                single_err_counter/len(trace)*100,
                double_err_counter/len(trace)*100)


class CBox_SSRO_discrimination_detector(det.Soft_Detector):
    def __init__(self, measurement_name, MC, AWG, CBox,
                 sequence_swf,
                 threshold=None,
                 calibrate_threshold=False,
                 save_raw_trace=False,
                 counters=True,
                 analyze=True,
                 **kw):
        super().__init__(**kw)

        self.name = measurement_name
        if threshold is None:
            self.threshold = CBox.sig0_threshold_line.get()
        else:
            self.threshold = threshold

        self.value_names = ['F-discr. cur. th.',
                            'F-discr. optimal',
                            'theta',
                            'optimal I-threshold',
                            'rel. separation',
                            'rel. separation I']  # projected along I axis
        self.value_units = ['%', '%', 'deg', 'a.u', '1/sigma', '1/sigma']

        self.AWG = AWG
        self.MC = MC
        self.CBox = CBox
        # Required to set some kind of sequence that does a pulse
        self.sequence_swf = sequence_swf

        # If analyze is False it cannot be used as a detector anymore
        self.analyze = analyze

    def prepare(self, **kw):
        self.i = 0
        self.MC.set_sweep_function(self.sequence_swf)
        self.MC.set_detector_function(det.CBox_integration_logging_det(
            self.CBox, self.AWG))

    def acquire_data_point(self, **kw):
        if self.i > 0:
            # overwrites the upload arg if the sequence swf has it to
            # prevent reloading
            self.sequence_swf.upload = False
        self.i += 1

        self.MC.run(self.name+'_{}'.format(self.i))
        if self.analyze:
            a = ma.SSRO_discrimination_analysis(
                label=self.name+'_{}'.format(self.i),
                current_threshold=self.threshold)
            return (a.F_discr_curr_t*100, a.F_discr*100,
                    a.theta, a.opt_I_threshold,
                    a.relative_separation, a.relative_separation_I)


class CBox_RB_detector(det.Soft_Detector):
    def __init__(self, measurement_name, MC, AWG, CBox, LutMan,
                 nr_cliffords, desired_nr_seeds,
                 IF,
                 RO_pulse_length, RO_pulse_delay, RO_trigger_delay,
                 pulse_delay,
                 T1=None, **kw):
        super().__init__(**kw)
        self.name = measurement_name
        self.nr_cliffords = nr_cliffords
        self.desired_nr_seeds = desired_nr_seeds
        self.AWG = AWG
        self.MC = MC
        self.CBox = CBox
        self.LutMan = LutMan
        self.IF = IF
        self.RO_pulse_length = RO_pulse_length
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.pulse_delay = pulse_delay
        self.T1 = T1
        self.value_names = ['F_cl']
        self.value_units = ['']

    def calculate_seq_duration_and_max_nr_seeds(self, nr_cliffords,
                                                pulse_delay):
        max_nr_cliffords = max(nr_cliffords)
        # For few cliffords the number of gates is not the average number of
        # gates so pick the max, rounded to ns
        max_seq_duration = np.round(max(max_nr_cliffords*pulse_delay *
                                        (1.875+.5), 10e-6), 9)
        max_idling_waveforms_per_seed = max_seq_duration/(1200e-9)
        max_nr_waveforms = 29184  # hard limit from the CBox
        max_nr_seeds = int(max_nr_waveforms/((max_idling_waveforms_per_seed +
                           np.mean(nr_cliffords)*1.875)*(len(nr_cliffords)+4)))
        return max_seq_duration, max_nr_seeds

    def prepare(self, **kw):
        max_seq_duration, max_nr_seeds = \
            self.calculate_seq_duration_and_max_nr_seeds(self.nr_cliffords,
                                                         self.pulse_delay)
        nr_repetitions = int(np.ceil(self.desired_nr_seeds/max_nr_seeds))
        self.total_nr_seeds = nr_repetitions*max_nr_seeds

        averages_per_tape = self.desired_nr_seeds//nr_repetitions
        self.CBox.nr_averages.set(int(2**np.ceil(np.log2(averages_per_tape))))

        rb_swf = awg_swf.CBox_RB_sweep(nr_cliffords=self.nr_cliffords,
                                       nr_seeds=max_nr_seeds,
                                       max_seq_duration=max_seq_duration,
                                       safety_margin=0,
                                       IF=self.IF,
                                       RO_pulse_length=self.RO_pulse_length,
                                       RO_pulse_delay=self.RO_pulse_delay,
                                       RO_trigger_delay=self.RO_trigger_delay,
                                       pulse_delay=self.pulse_delay,
                                       AWG=self.AWG,
                                       CBox=self.CBox,
                                       LutMan=self.LutMan)

        self.i = 0
        self.MC.set_sweep_function(rb_swf)
        self.MC.set_sweep_function_2D(awg_swf.Two_d_CBox_RB_seq(rb_swf))
        self.MC.set_sweep_points_2D(np.arange(nr_repetitions))
        self.MC.set_detector_function(det.CBox_integrated_average_detector(
                                      self.CBox, self.AWG))

    def acquire_data_point(self, **kw):
            self.i += 1
            self.MC.run(self.name+'_{}_{}seeds'.format(
                        self.i, self.total_nr_seeds), mode='2D')
            a = ma.RandomizedBench_2D_flat_Analysis(
                auto=True, close_main_fig=True, T1=self.T1,
                pulse_delay=self.pulse_delay)
            F_cl = a.fit_res.params['fidelity_per_Clifford'].value
            return F_cl



class Chevron_optimization_v1(det.Soft_Detector):
    '''
    Chevron optimization.
    '''
    def __init__(self, flux_channel, dist_dict, AWG, MC_nested, qubit,
                 kernel_obj, cost_function_opt=0, **kw):
        super().__init__()
        kernel_dir_path = 'kernels/'
        self.name = 'chevron_optimization_v1'
        self.value_names = ['Cost function','SWAP Time']
        self.value_units = ['a.u.','ns']
        self.kernel_obj = kernel_obj
        self.AWG = AWG
        self.MC_nested = MC_nested
        self.qubit = qubit
        self.dist_dict = dist_dict
        self.flux_channel = flux_channel
        self.cost_function_opt = cost_function_opt
        self.dist_dict['ch%d'%self.flux_channel].append('')
        self.nr_averages = kw.get('nr_averages', 1024)

        self.awg_amp_par = ManualParameter(name='AWG_amp', units='Vpp', label='AWG Amplitude')
        self.awg_amp_par.get = lambda: self.AWG.get('ch{}_amp'.format(self.flux_channel))
        self.awg_amp_par.set = lambda val: self.AWG.set('ch{}_amp'.format(self.flux_channel),val)
        self.awg_value = 2.0

        kernel_before_list = self.dist_dict['ch%d'%self.flux_channel]
        kernel_before_loaded = []
        for k in kernel_before_list:
            if k is not '':
                kernel_before_loaded.append(np.loadtxt(kernel_dir_path+k))
        self.kernel_before = kernel_obj.convolve_kernel(kernel_before_loaded,
                                                        30000)

    def acquire_data_point(self, **kw):
        # # Before writing it
        # # Summarize what to do:

        # # Update kernel from kernel object

        kernel_file = 'optimizing_kernel_%s'%a_tools.current_timestamp()
        self.kernel_obj.save_corrections_kernel(kernel_file, self.kernel_before,)
        self.dist_dict['ch%d'%self.flux_channel][-1] = kernel_file+'.txt'


        self.qubit.dist_dict = self.dist_dict
        self.qubit.RO_acq_averages(self.nr_averages)
        self.qubit.measure_chevron(amps=[self.awg_amp_par()],
                                   length=np.arange(0, 81e-9, 1e-9),
                                   MC=self.MC_nested)

        # # fit it
        ma_obj = ma.chevron_optimization_v2(auto=True, label='Chevron_slice')
        cost_val = ma_obj.cost_value[self.cost_function_opt]

        # # Return the cost function sum(min)+sum(1-max)
        return cost_val, 0.5*ma_obj.period



    def prepare(self):
        pass

    def finish(self):
        pass




class SWAPN_optimization(det.Soft_Detector):
    '''
    SWAPN optimization.
    Wrapper around a SWAPN sequence to create a cost function.

    The kernel object is used to determine the (pre)distortion kernel.
    It is common to do a sweep over one of the kernel parameters as a sweep
    function.
    '''
    def __init__(self, nr_pulses_list, AWG, MC_nested, qubit,
                 kernel_obj,  cache, cost_choice='sum',**kw):

        super().__init__()
        self.name = 'swapn_optimization'
        self.value_names = ['Cost function', 'Single SWAP Fid']
        self.value_units = ['a.u.', 'ns']
        self.kernel_obj = kernel_obj
        self.cache_obj = cache
        self.AWG = AWG
        self.MC_nested = MC_nested
        self.cost_choice = cost_choice
        self.nr_pulses_list = nr_pulses_list
        self.qubit = qubit

    def acquire_data_point(self, **kw):
        # # Update kernel from kernel object

        # # Measure the swapn
        times_vec = self.nr_pulses_list
        cal_points = 4
        lengths_cal = times_vec[-1] + np.arange(1, 1+cal_points)*(times_vec[1]-times_vec[0])
        lengths_vec = np.concatenate((times_vec, lengths_cal))

        flux_pulse_pars = self.qubit.get_flux_pars()
        mw_pulse_pars, RO_pars = self.qubit.get_pulse_pars()

        repSWAP = awg_swf.SwapN(mw_pulse_pars,
                                RO_pars,
                                flux_pulse_pars, AWG=self.AWG,
                                dist_dict=self.kernel_obj.kernel(),
                                upload=True)
        # self.AWG.set('ch%d_amp'%self.qubit.fluxing_channel(), 2.)
        # seq = repSWAP.pre_upload()

        self.MC_nested.set_sweep_function(repSWAP)
        self.MC_nested.set_sweep_points(lengths_vec)

        self.MC_nested.set_detector_function(self.qubit.int_avg_det_rot)
        self.AWG.set('ch%d_amp'%self.qubit.fluxing_channel(),
                     self.qubit.swap_amp())
        self.MC_nested.run('SWAPN_%s'%self.qubit.name)

        # # fit it
        ma_obj = ma.SWAPN_cost(auto=True, cost_func=self.cost_choice)
        return ma_obj.cost_val, ma_obj.single_swap_fid

    def prepare(self):
        pass

    def finish(self):
        pass







class AllXY_devition_detector_CBox(det.Soft_Detector):
    '''
    Currently only for CBox.
    Todo: remove the predefined values for the sequence
    '''
    def __init__(self, measurement_name, MC, AWG, CBox,
                 IF, RO_trigger_delay, RO_pulse_delay, RO_pulse_length,
                 pulse_delay,
                 LutMan=None,
                 reload_pulses=False, **kw):
        '''
        If reloading of pulses is desired the LutMan is a required instrument
        '''
        self.detector_control = 'soft'
        self.name = 'AllXY_dev_i'
        # For an explanation of the difference between the different
        # Fidelities look in the analysis script
        self.value_names = ['Total_deviation', 'Avg deviation']
        # Should only return one instead of two but for now just for
        # convenience as I understand the scale of total deviation
        self.value_units = ['', '']
        self.measurement_name = measurement_name
        self.MC = MC
        self.CBox = CBox
        self.AWG = AWG

        self.IF = IF
        self.RO_trigger_delay = RO_trigger_delay
        self.RO_pulse_delay = RO_pulse_delay
        self.pulse_delay = pulse_delay
        self.RO_pulse_length = RO_pulse_length

        self.LutMan = LutMan
        self.reload_pulses = reload_pulses

    def prepare(self, **kw):
        self.i = 0
        self.MC.set_sweep_function(awg_swf.CBox_AllXY(
                                   IF=self.IF,
                                   pulse_delay=self.pulse_delay,
                                   RO_pulse_delay=self.RO_pulse_delay,
                                   RO_trigger_delay=self.RO_trigger_delay,
                                   RO_pulse_length=self.RO_pulse_length,
                                   AWG=self.AWG, CBox=self.CBox))
        self.MC.set_detector_function(
            det.CBox_integrated_average_detector(self.CBox, self.AWG))

    def acquire_data_point(self, *args, **kw):
        if self.i > 0:
            self.MC.sweep_functions[0].upload = False
        self.i += 1
        if self.reload_pulses:
            self.LutMan.load_pulses_onto_AWG_lookuptable(0)
            self.LutMan.load_pulses_onto_AWG_lookuptable(1)
            self.LutMan.load_pulses_onto_AWG_lookuptable(2)

        self.MC.run(name=self.measurement_name+'_'+str(self.i))

        ana = ma.AllXY_Analysis(label=self.measurement_name)
        tot_dev = ana.deviation_total
        avg_dev = tot_dev/21

        return tot_dev, avg_dev


# class SSRO_Fidelity_Detector_ATS(det.Soft_Detector):
#     '''
#     SSRO fidelity measurement with ATS
#     '''
#     def __init__(self, measurement_name=None, no_fits=False, nr_measurements=1,
#                  **kw):
#         self.detector_control = 'soft'
#         self.name = 'SSRO_Fidelity'
#         if nr_measurements==2:
#             self.name = 'SSRO_Fidelity_2'
#         # For an explanation of the difference between the different
#         # Fidelities look in the analysis script
#         self.no_fits = no_fits
#         if self.no_fits:
#             print("data for nofits")
#             self.value_names = ['F_raw']
#             self.value_units = [' ']
#         else:
#             self.value_names = ['F', 'F corrected']
#             self.value_units = [' ', ' ']
#         self.measurement_name = measurement_name
#         self.gauss_width = kw.get('gauss_width',10)
#         self.qubit_suffix = kw.get('qubit_suffix','')
#         self.nr_measurements = nr_measurements
#         self.TD_Meas = qt.instruments['TD_Meas']

#     def prepare(self, **kw):
#         imp.reload(ma)
#         self.MC_SSRO = qt.instruments.create('MC_SSRO', 'MeasurementControl')
#         self.MC_SSRO.set_sweep_function(awg_swf.OnOff(
#                                     gauss_width=self.gauss_width,
#                                     qubit_suffix=self.qubit_suffix,
#                                     nr_segments=2,
#                                     nr_measurements=self.nr_measurements))
#         #self.MC_SSRO.set_sweep_points([np.linspace(1,2,2)])
#         self.TD_Meas.set_shot_mode(True)
#         self.TD_Meas.set_MC('MC_SSRO')
#         self.MC_SSRO.set_detector_function(det.TimeDomainDetector())

#     def acquire_data_point(self, *args, **kw):
#         if self.measurement_name is not None:
#             measurement_name = self.measurement_name
#         else:
#             measurement_name = 'SSRO_Fid_{:.9}'.format(
#                 kw.pop('sweep_point', None))
#         print('measurement name is %s' % measurement_name)
#         self.MC_SSRO.run(name=measurement_name)

#         t0 = time.time()
#         ana = ma.SSRO_Analysis(auto=True, close_file=True,
#                                label=measurement_name,
#                                no_fits=self.no_fits)
#         print('analyzing took %.2f' % ((time.time() - t0)))

#         # Arbitrary choice, does not think about the deffinition
#         if self.no_fits:
#             return ana.F_raw
#         else:
#             return ana.F, ana.F_corrected

#     def finish(self, **kw):
#         self.TD_Meas.set_MC('MC')
#         self.MC_SSRO.remove()
#         self.TD_Meas.set_shot_mode(False)


# class SSRO_Fidelity_Detector_CBox_optimum_weights(det.Soft_Detector):
#     '''
#     SSRO fidelity measurement with CBox when driving from
#     '''
#     def __init__(self, measurement_name=None, no_fits=False, NoShots=8000,
#                  set_weights=True, substract_weight_offsets=True,
#                  nr_iterations=1, create_plots=True, **kw):
#         self.detector_control = 'soft'
#         self.name = 'SSRO_Fidelity'
#         # For an explanation of the difference between the different
#         # Fidelities look in the analysis script
#         self.no_fits = no_fits
#         if self.no_fits:
#             print("data for nofits")
#             self.value_names = ['F_raw', 'threshold']
#             self.value_units = [' ', ' ']
#             self.value_names = ['F_raw', 'threshold', 'weight']
#             self.value_units = [' ', 'a.u', '', 'a.u']
#         else:
#             self.value_names = ['F', 'F corrected', 'threshold',
#                                 'weight', 'signal']
#             self.value_units = [' ', ' ', 'a.u', ' ', 'a.u']
#         self.measurement_name = measurement_name
#         self.TD_Meas = qt.instruments['TD_Meas']
#         self.CBox = qt.instruments['CBox']
#         self.MC = qt.instruments['MC']
#         self.NoShots = NoShots
#         self.set_weights = set_weights
#         self.substract_weight_offsets = substract_weight_offsets
#         self.rotate = not(set_weights)
#         self.nr_iterations = nr_iterations
#         self.create_plots = create_plots

#     def prepare(self, **kw):
#         self.MC_SSRO = qt.instruments.create('MC_SSRO', 'MeasurementControl')
#         self.TD_Meas.set_MC('MC_SSRO')

#     def acquire_data_point(self, *args, **kw):
#         if self.measurement_name is not None:
#             measurement_name = self.measurement_name
#         else:
#             # FIXME this statement always has none in the name
#             measurement_name = 'SSRO_Fid_inner_{:.9}_outer_{:.9}'.format(
#                 kw.pop('sweep_point', None),
#                 kw.get('sweep_point_outer', None))
#         self.TD_Meas.prepare()
#         # setting the RF and LO settings from Time domain
#         # Here the transient setting sould be done
#         self.LoggingMode = 3  # 2 for shots, 3 for transients
#         self.CBox.set_tng_logging_mode(self.LoggingMode)
#         self.CBox.set_lin_trans_coeffs(1, 0, 0, 0)
#         self.CBox.set_nr_averages(2**12)

#         if self.set_weights:
#             print("setting the weights")
#             # Acquire |0> transient
#             nr_iterations = self.nr_iterations  # Fixme find proper name
#             for i in range(nr_iterations):
#                 self.MC_SSRO.set_sweep_function(CB_swf.OnOff_touch_n_go(
#                                                 pulses='OffOff',
#                                                 NoSegments=512, stepsize=5,
#                                                 NoShots=self.NoShots))
#                 self.MC_SSRO.set_detector_function(
#                     det.QuTechCBox_input_average_Detector_Touch_N_Go())
#                 self.MC_SSRO.run('transient_Off')
#                 if self.create_plots:
#                     ana0 = ma.MeasurementAnalysis(auto=True, close_file=False)
#                 else:
#                     ana0 = ma.MeasurementAnalysis(auto=False, close_file=False)
#                     ana0.get_naming_and_values()
#                 # Acquire |1> transient
#                 self.MC_SSRO.set_sweep_function(CB_swf.OnOff_touch_n_go(
#                                                 pulses='OnOn',
#                                                 NoSegments=512, stepsize=5,
#                                                 NoShots=self.NoShots))
#                 self.MC_SSRO.run('transient_On')
#                 if self.create_plots:
#                     ana1 = ma.MeasurementAnalysis(auto=True, close_file=False)
#                 else:
#                     ana1 = ma.MeasurementAnalysis(auto=False, close_file=False)
#                     ana1.get_naming_and_values()
#                 try:
#                     trace0 += ana0.get_values(key='Ch0')
#                     trace1 += ana1.get_values(key='Ch0')
#                 except NameError:
#                     trace0 = ana0.get_values(key='Ch0')
#                     trace1 = ana1.get_values(key='Ch0')

#                 ana0.finish()
#                 ana1.finish()

#             if self.substract_weight_offsets:
#                 # Factor 2 is to prevent running into the dac range of the CBox
#                 weight = (trace1 - trace0 - np.mean(trace1-trace0))/2.0
#                 max_weight = np.abs(np.max(weight))
#                 min_weight = np.abs(np.min(weight))
#                 scale_factor = 100 / np.max([min_weight, max_weight])
#                 weight = np.round(scale_factor * weight)
#             else:
#                 weight = (trace1 - trace0)/2.0
#             weight = np.round(weight)
#             average_weight = np.mean(weight)

#             # Set the optimal weight function based on the trace difference
#             self.CBox.set_integration_weights(line=0, weights=weight)
#             self.CBox.set_integration_weights(line=1, weights=np.multiply(weight,0))
#             #setting line 1 t zero attampting to solve the feedback problem
#         else:
#             "not setting the weights"
#             average_weight = 0  # only here to not break the detector

#         # returning to the standard SSRO
#         self.LoggingMode = 2
#         self.CBox.set_tng_logging_mode(self.LoggingMode)
#         self.MC_SSRO.set_sweep_function(CB_swf.OnOff_touch_n_go(NoShots=self.NoShots))
#         self.MC_SSRO.set_detector_function(
#             det.QuTechCBox_AlternatingShots_Logging_Detector_Touch_N_Go())

#         self.MC_SSRO.run(name=measurement_name)
#         t0 = time.time()
#         ana = ma.SSRO_Analysis(auto=True, close_file=True,
#                                label=measurement_name,
#                                no_fits=self.no_fits,
#                                plot_histograms=self.create_plots,
#                                rotate=self.rotate)

#         # print 'analyzing took %.2f' % ((time.time() - t0))
#         self.CBox.set_signal_threshold_line0(ana.V_opt_raw)
#         print("Fidelity %s, threshold set to %s" % (ana.F_raw, ana.V_opt_raw))
#         # Arbitrary choice, does not think about the deffinition
#         if self.no_fits:
#             return ana.F_raw, ana.V_opt_raw, average_weight
#         else:
#             signal = np.abs(ana.mu1_1 - ana.mu0_0)
#             return ana.F, ana.F_corrected, ana.V_opt_raw, average_weight, signal

#     def finish(self, **kw):
#         self.TD_Meas.set_MC('MC')
#         self.MC_SSRO.remove()


# class touch_n_go_butterfly_detector(det.Soft_Detector):
#     '''
#     First calibrates the single shot readout to determine the threshold
#     After that performs a butterfly experiment which consists of multiple
#     back to back measurements.

#     epsij_k
#     i =

#     This detector is written for the CBox operating in touch n go mode.
#     '''
#     def __init__(self, measurement_name=None,
#                  n_iter=5, postselection=False, no_fits=True,**kw):
#         self.detector_control = 'soft'
#         self.name = 'touch_n_go_butterfly_detector'
#         # For an explanation of the difference between the different
#         # Fidelities look in the analysis script
#         self.measurement_name = measurement_name
#         self.TD_Meas = qt.instruments['TD_Meas']
#         self.CBox = qt.instruments['CBox']
#         self.value_names = ['eps00_0', 'eps01_0', 'eps10_0', 'eps11_0',
#                             'eps00_1', 'eps01_1', 'eps10_1', 'eps11_1',
#                             'F_raw', 'F_bf', 'mmt_ind_rel', 'mmt_ind_exc']
#         self.value_units = ['', '', '', '',
#                             '', '', '', '',
#                             '', '', '', '']
#         self.n_iter = n_iter
#         self.postselection = postselection
#         self.no_fits = no_fits

#     def prepare(self, **kw):
#         self.MC = qt.instruments['MC_nest1']
#         if self.MC is None:
#             self.MC = qt.instruments.create('MC_nest1', 'MeasurementControl')

#     def acquire_data_point(self, *args, **kw):
#         self.calibrate_weight_functions(no_fits=self.no_fits)
#         # self.measure_butterfly_coefficients_interleaved()
#         coeffs = self.measure_butterfly_coefficients()
#         return [coeffs[x] for x in self.value_names]

#     def calibrate_weight_functions(self, create_plots=True,
#                                    prepare_TD_Meas=True, no_fits=True):
#         if prepare_TD_Meas:
#             # Note by Adriaan, should this be here? this is a CBox only msmt
#             # Would be cleaner if we don't invoke TD_Meas here.
#             # Made it a function argument so I can not use it without breaking
#             # other peoples work.
#             self.TD_Meas.prepare()
#         # setting the RF and LO settings from Time domain instruments
#         # Here the transient setting sould be done

#         cycle_heartbeat = self.CBox.get_tng_heartbeat_interval()
#         self.CBox.set_tng_burst_heartbeat_n(1)
#         readout_delay = self.CBox.get_tng_readout_delay()
#         second_pre_rotation_delay = self.CBox.get_tng_second_pre_rotation_delay()
#         self.CBox.set_tng_second_pre_rotation_delay(100)
#         self.CBox.set_tng_readout_delay(0)

#         self.CBox.set_tng_heartbeat_interval(100000)  # setting it to 100 us
#         M = SSRO_Fidelity_Detector_CBox_optimum_weights(
#             no_fits=no_fits, measurement_name='SSRO_Fidelity',
#             create_plots=create_plots)
#         M.prepare()
#         ana_SSRO = M.acquire_data_point()
#         M.finish()
#         self.F_raw = ana_SSRO[0]
#         if not no_fits:
#             self.F_corrected = ana_SSRO[1]
#         # setting the heartbeat interval to the old value
#         self.CBox.set_tng_heartbeat_interval(cycle_heartbeat)
#         self.CBox.set_tng_readout_delay(readout_delay)
#         self.CBox.set_tng_second_pre_rotation_delay(second_pre_rotation_delay)


#     def measure_butterfly_coefficients(self):
#         Shots_p_iteration = 8000
#         if self.postselection:
#             n_cycles = 4
#             tape = np.zeros(n_cycles)
#         else:
#             n_cycles = 2
#             tape = np.zeros(n_cycles)
#         self.CBox.set_tng_burst_heartbeat_n(n_cycles)
#         self.MC.set_sweep_function(CB_swf.custom_tape_touch_n_go(
#                                    custom_tape=tape,
#                                    NoShots=Shots_p_iteration,
#                                    NoSegments=n_cycles))
#         self.MC.set_sweep_function_2D(CB_swf.None_Sweep_tape_restart())
#         self.MC.set_sweep_points_2D(np.arange(
#                                     Shots_p_iteration/n_cycles*self.n_iter))
#         self.MC.set_detector_function(
#             det.QuTechCBox_Shots_Logging_Detector_Touch_N_Go(digitize=False,
#                                                              timeout=1))
#         self.MC.run(name='mmt_ind_exc', mode='2D')
#         if self.postselection:
#             tape[1] = 1
#         else:
#             tape[0] = 1

#         self.MC.set_sweep_function(CB_swf.custom_tape_touch_n_go(
#                                    custom_tape=tape,
#                                    NoShots=Shots_p_iteration,
#                                    NoSegments=n_cycles))
#         self.MC.set_sweep_function_2D(CB_swf.None_Sweep_tape_restart())
#         self.MC.set_sweep_points_2D(np.arange(
#                                     Shots_p_iteration/n_cycles*self.n_iter))
#         self.MC.run(name='mmt_ind_rel', mode='2D')
#         bf_an = ma.butterfly_analysis(auto=True, label_exc='mmt_ind_exc',
#                                       label_rel='mmt_ind_rel',
#                                       postselection=self.postselection)

#         bf_an.butterfly_coeffs['F_raw'] = self.F_raw

#         return bf_an.butterfly_coeffs

#     def measure_butterfly_coefficients_interleaved(self):
        # This function is tested but does not have analysis yet.
        msmts_p_butterfly = 5
        nr_sweeps = 10000
        total_nr_shots = msmts_p_butterfly*2 * nr_sweeps
        shots_p_iteration = 8000 - 8000 % (msmts_p_butterfly*2.)
        if shots_p_iteration != 8000:
            raise ValueError('See CBox issue 42')
        n_iter = np.ceil(total_nr_shots/shots_p_iteration)
        print('Shots per iteration', shots_p_iteration)
        print('n_iter', n_iter)
        # ensures it takes at least nr_sweeps iterations
        tape_rel = np.zeros(msmts_p_butterfly)
        tape_rel[0] = 1
        tape_exc = np.zeros(msmts_p_butterfly)
        tape = np.concatenate([tape_exc, tape_rel])

        self.CBox.set_tng_burst_heartbeat_n(len(tape)/2)
        self.MC.set_sweep_function(CB_swf.custom_tape_touch_n_go(
                                   custom_tape=tape,
                                   NoShots=shots_p_iteration,
                                   NoSegments=len(tape)))
        self.MC.set_sweep_function_2D(CB_swf.None_Sweep_tape_restart())
        self.MC.set_sweep_points_2D(np.arange(shots_p_iteration/len(tape)
                                              * n_iter))
        self.MC.set_detector_function(
            det.QuTechCBox_Shots_Logging_Detector_Touch_N_Go(digitize=False,
                                                             timeout=1))
        self.MC.run(name='butterfly_interleaved', mode='2D')
        return


# class Average_cycles_to_failure_detector(touch_n_go_butterfly_detector):
#     '''
#     Average cycles to failure detector for conditional CLEAR cycles.
#     This detecor first calls the SSRO with optimum weights detector to set the
#     measurement threshold. Then it measures the average cycles to failure.
#     initialized: the experiment takes short traces and only counts the
#         number of traces to an error

#     It is a child of the touch_n_go_butterfly_detector in order to inherit
#     the calibrate_weight_functions() function.
#     '''
#     def __init__(self, measurement_name=None, trace_length=100,
#                  trace_repetitions=100, calibrate_weight_function=True,
#                  flipping_sequence=False, tape=[0, 0], create_SSRO_plots=True,
#                  no_fits=True,  **kw):

#         self.detector_control = 'soft'
#         self.name = 'Average_cycles_to_failure_detector'
#         # For an explanation of the difference between the different
#         # Fidelities look in the analysis script
#         self.measurement_name = measurement_name
#         self.TD_Meas = qt.instruments['TD_Meas']
#         self.CBox = qt.instruments['CBox']
#         self.tape = tape
#         self.trace_length = trace_length
#         self.trace_repetitions = trace_repetitions
#         self.flipping_sequence = flipping_sequence
#         self.calibrate_weight_function = calibrate_weight_function
#         self.no_fits = no_fits
#         if self.no_fits:
#             self.value_names = ['mean_rounds_to_failure', 'standard error',
#                                 'RO error fraction', 'Flip error fraction',
#                                 'F_raw']
#             self.value_units = [' ', ' ', '%', '%', ' ']
#         else:
#             self.value_names = ['mean_rounds_to_failure', 'standard error',
#                                 'RO error fraction', 'Flip error fraction',
#                                 'F_raw', 'F_corrected']
#             self.value_units = [' ', ' ', '%', '%', ' ', ' ']
#         self.create_SSRO_plots = create_SSRO_plots


#     def acquire_data_point(self, iteration, *args, **kw):
#         if self.measurement_name is None:
#             self.measurement_name = \
#                 'Average_cycles_to_failure_tape_%s' % (
#                     self.tape[0:3])
#         # print 'iteration : %s' %iteration
#         if self.calibrate_weight_function and ((iteration-1) % 1 == 0):
#             print('Calibrating weight function ')
#             self.calibrate_weight_functions(
#                 create_plots=self.create_SSRO_plots, no_fits=self.no_fits)

#         data = self.measure_cycles_to_failure()
#         return data

#     def measure_cycles_to_failure(self):
#         # Determine number of iterations
#         self.CBox.set_run_mode(0)
#         self.CBox.set_acquisition_mode(0)
#         trace_length = self.trace_length
#         trace_repetitions = self.trace_repetitions
#         total_nr_shots = trace_length * trace_repetitions
#         shots_p_iteration = 8000. - (8000 % (trace_length))
#         n_iter = np.ceil(total_nr_shots/shots_p_iteration)
#         if shots_p_iteration != 8000:
#             raise ValueError('See CBox issue 42')
#         print(('Experiment needs to iterate' +
#                ' %s times to get %s iterations per trace' %
#                (n_iter, trace_repetitions)))
#         # Set the heartbeat settings
#         HB_interval = self.CBox.get_tng_heartbeat_interval()
#         Brst_interval = self.CBox.get_tng_burst_heartbeat_interval()
#         self.CBox.set_tng_heartbeat_interval(HB_interval +
#                                              trace_length*Brst_interval)
#         self.CBox.set_tng_burst_heartbeat_n(trace_length)
#         # Start the measurement
#         self.MC.set_sweep_function(CB_swf.custom_tape_touch_n_go(
#                                    custom_tape=self.tape,
#                                    NoShots=shots_p_iteration,
#                                    NoSegments=trace_length))
#         self.MC.set_sweep_function_2D(CB_swf.None_Sweep_tape_restart())
#         self.MC.set_sweep_points_2D(np.arange(shots_p_iteration/trace_length
#                                               * n_iter))
#         self.MC.set_detector_function(
#             det.QuTechCBox_Shots_Logging_Detector_Touch_N_Go(digitize=True,
#                                                              timeout=1))

#         self.MC.run(name=self.measurement_name, mode='2D')

#         a = ma.rounds_to_failure_analysis(
#             auto=True, label=self.measurement_name,
#             flipping_sequence=self.flipping_sequence)
#         a.finish()
#         self.CBox.set_tng_burst_heartbeat_n(1)
#         self.CBox.set_tng_heartbeat_interval(HB_interval)
#         if self.no_fits:
#             return a.mean_rtf, a.std_err_rtf, a.RO_err_frac, a.flip_err_frac, self.F_raw
#         else:
#             return a.mean_rtf, a.std_err_rtf, a.RO_err_frac, a.flip_err_frac, self.F_raw, self.F_corrected



# class Photon_number_detector(det.Soft_Detector):
#     '''
#     Photon number detector for the CLEAR pulse experiment.
#     Sequences are based on the AllXY sequences
#     Only works for AWG driving

#     Requires defining a sweep function that is used for the photon number detector
#     ('defalt is awg_swf.AllXY())
#     '''
#     def __init__(self, measurement_name=None, data_acquisition='ATS',
#                  driving='CBox', CLEAR_double=False, optimize_on_ground=True,
#                  sequence="AllXY", prepi=False, add_X90_prerotation=False,
#                  flip_trigger_states=False,**kw):
#         self.detector_control = 'soft'
#         self.name = '%s_deviation' % sequence

#         self.data_acquisition = data_acquisition
#         self.measurement_name = measurement_name
#         self.TD_Meas = qt.instruments['TD_Meas']
#         self.driving = driving
#         self.optimize_on_ground = optimize_on_ground
#         self.CLEAR_double = CLEAR_double
#         print("clear double",self.CLEAR_double)
#         if self.CLEAR_double:
#             self.value_names = ['%s error sum' % sequence, "error ground",
#                                 "error excited", 'amp1', 'amp2', 'phi1', 'phi2', 'ampa1', 'ampb1', 'phia1', 'phib1']
#             self.value_units = ['a.u.', 'a.u.', 'a.u.', 'mV', 'mV', 'deg', 'deg', 'mV','mV', 'deg', 'deg']
#         else:
#             if optimize_on_ground:
#                 self.state = "ground"
#             else:
#                 self.state = "excited"
#             self.value_names = ['%s error (%s)' % (sequence, self.state)]
#             self.value_units = ['a.u.']
#         print("measurement_name", measurement_name)
#         self.CBox = qt.instruments['CBox']
#         self.CBox_lut_man = qt.instruments['CBox_lut_man']
#         self.CBox_lut_man_2 = qt.instruments['CBox_lut_man_2']
#         self.prepi = prepi
#         self.sweep_function = awg_swf.AllXY(sequence=sequence, prepi=self.prepi)
#         self.add_X90_prerotation = add_X90_prerotation
#         self.flip_trigger_states = flip_trigger_states

#     def prepare(self, **kw):
#         self.MC_AllXY = qt.instruments.create('MC_AllXY',
#                                               'MeasurementControl')
#         self.TD_Meas.set_MC(self.MC_AllXY.get_name())
#         self.TD_Meas.prepare()
#         if self.driving is "CBox":
#             self.MC_AllXY.set_sweep_function(CB_swf.AllXY())
#         elif self.driving is "AWG":
#             self.MC_AllXY.set_sweep_function(self.sweep_function)
#         if self.data_acquisition == 'ATS':
#             self.MC_AllXY.set_detector_function(det.TimeDomainDetector())
#         elif self.data_acquisition == 'CBox':
#             self.MC_AllXY.set_detector_function(
#                 det.QuTechCBox_integrated_average_Detector())

#     def acquire_data_point(self, *args, **kw):
#         if self.measurement_name is not None:
#             measurement_name = self.measurement_name
#         else:
#             iteration = kw.get('iteration', None)
#             print("iteration", iteration)
#             if iteration is None:
#                 sweep_point = kw.get('sweep_point', None)
#                 sweep_point_outer = kw.get('sweep_point_outer', None)
#                 print("sweep outer", sweep_point_outer)
#                 measurement_name = 'AllXY_dev_inner_%s_outer_%s' % (
#                     sweep_point, sweep_point_outer)
#             else:
#                 measurement_name = 'AllXY_iteration_%s' % iteration
#         if self.CLEAR_double is False:
#             if self.optimize_on_ground:
#                 self.CBox_lut_man.set_lut_mapping(['I', 'X180', 'Y180', 'X90',
#                                                    'Y90', 'Block',
#                                                    'X180_delayed'])
#                 self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
#             else:
#                 self.CBox_lut_man.set_lut_mapping(['X180', 'X180', 'Y180',
#                                                    'X90', 'Y90', 'Block',
#                                                    'X180_delayed'])
#                 self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
#             self.MC_AllXY.run(name=measurement_name)
#             t0 = time.time()
#             flip_ax = self.prepi != self.optimize_on_ground
#             ana = ma.AllXY_Analysis(auto=True, flip_axis=flip_ax,
#                                     cal_points=self.sweep_function.cal_points,
#                                     ideal_data=self.sweep_function.ideal_data)
#             print('analyzing AllXY took %.2f' % ((time.time() - t0)))
#             if self.TD_Meas.get_CBox_touch_n_go_save_transients():
#                 ma.TransientAnalysis(auto=True, label='AllXY')
#             deviation = ana.deviation_total
#             return deviation

#         elif self.CLEAR_double:
#             measurement_name = 'AllXY_iteration_'
#             self.CBox_lut_man.set_lut_mapping(['I', 'X180', 'Y180', 'X90', 'Y90', 'Block',
#                              'X180_delayed'])
#             print("identity is loaded")
#             self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
#             if self.flip_trigger_states:
#                 self.CBox.set_tng_trigger_state(0) #ground state pulses will be played in conditional mode
#             self.MC_AllXY.run(name=measurement_name+"_ground")
#             ana_g = ma.AllXY_Analysis(auto=True,
#                                       cal_points=self.sweep_function.cal_points,
#                                       ideal_data=self.sweep_function.ideal_data)
#             error_g = ana_g.deviation_total
#             if self.TD_Meas.get_CBox_touch_n_go_save_transients():
#                 ma.TransientAnalysis(auto=True, label='AllXY')
#             print("error ground", error_g)
#             self.CBox_lut_man.set_lut_mapping(['X180', 'X180', 'Y180', 'X90', 'Y90', 'Block',
#                              'X180_delayed'])
#             self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
#             print("pi pulse is loaded")
#             if self.flip_trigger_states:
#                 self.CBox.set_tng_trigger_state(1) #excited state pulses will be played in conditional mode
#             self.MC_AllXY.run(name=measurement_name+"_excited")
#             ana_e = ma.AllXY_Analysis(auto=True, flip_axis=True,
#                                       cal_points=self.sweep_function.cal_points,
#                                       ideal_data=self.sweep_function.ideal_data)
#             error_e = ana_e.deviation_total
#             if self.TD_Meas.get_CBox_touch_n_go_save_transients():
#                 ma.TransientAnalysis(auto=True, label='AllXY')
#             print("error excited", error_e)
#             if self.add_X90_prerotation:
#                 self.CBox_lut_man.set_lut_mapping(['X90', 'X180', 'Y180', 'X90', 'Y90', 'Block',
#                              'X180_delayed'])
#                 self.CBox_lut_man.load_pulses_onto_AWG_lookuptable(0)
#                 self.MC_AllXY.run(name=measurement_name+"_90")
#                 ana_90 = ma.AllXY_Analysis(auto=True,
#                                           cal_points=self.sweep_function.cal_points,
#                                           ideal_data=self.sweep_function.ideal_data)
#                 error_90 = ana_90.deviation_total
#                 if self.TD_Meas.get_CBox_touch_n_go_save_transients():
#                     ma.TransientAnalysis(auto=True, label='AllXY')
#                 print("error superposition", error_90)

#             deviation = error_g + error_e
#             #extracting pulse parameters
#             amp1 = self.CBox_lut_man_2.get_M_amp_CLEAR_1()
#             amp2 = self.CBox_lut_man_2.get_M_amp_CLEAR_2()
#             phi1 = self.CBox_lut_man_2.get_M_phase_CLEAR_1()
#             phi2 = self.CBox_lut_man_2.get_M_phase_CLEAR_2()
#             ampa1 = self.CBox_lut_man_2.get_M_amp_CLEAR_a1()
#             ampb1 = self.CBox_lut_man_2.get_M_amp_CLEAR_b1()
#             phia1 = self.CBox_lut_man_2.get_M_phase_CLEAR_a1()
#             phib1 = self.CBox_lut_man_2.get_M_phase_CLEAR_b1()
#             print("total deviation", deviation)
#             return deviation, error_g, error_e, amp1, amp2, phi1, phi2, ampa1, ampb1, phia1, phib1

#     def finish(self, **kw):
        self.MC_AllXY.remove()



# class AllXY_deviation_detector(det.Soft_Detector):
#     '''
#     Performs an All XY measurement and returns the deviation to the ideal
#     expected shape.

#     '''
#     def __init__(self, measurement_name=None, data_acquisition='ATS',
#                  driving='CBox', CLEAR_double=False, optimize_on_ground=True,
#                  source_name='S2',
#                  sequence="AllXY", prepi=False, **kw):
#         self.detector_control = 'soft'
#         self.name = '%s_deviation' % sequence

#         self.data_acquisition = data_acquisition
#         self.measurement_name = measurement_name
#         self.TD_Meas = qt.instruments['TD_Meas']
#         self.driving = driving
#         self.optimize_on_ground = optimize_on_ground
#         self.CLEAR_double = CLEAR_double
#         if self.CLEAR_double:
#             self.value_names = ['%s error sum' % sequence, "error ground",
#                                 "error excited"]
#             self.value_units = ['a.u.', 'a.u.', 'a.u.']
#         else:
#             if optimize_on_ground:
#                 self.state = "ground"
#             else:
#                 self.state = "excited"
#             self.value_names = ['%s error (%s)' % (sequence, self.state)]
#             self.value_units = ['a.u.']
#         print("measurement_name", measurement_name)
#         self.source = qt.instruments[source_name]
#         self.prepi = prepi
#         self.sweep_function = awg_swf.AllXY(sequence=sequence, prepi=self.prepi)

#     def prepare(self, **kw):
#         self.TD_Meas.prepare()
#         # setting the RF and LO settings from Time domain instruments
#         # Here the transient setting sould be done

#         self.MC_AllXY = qt.instruments.create('MC_AllXY',
#                                               'MeasurementControl')
#         self.TD_Meas.set_MC(self.MC_AllXY.get_name())

#         if self.driving is "CBox":
#             self.MC_AllXY.set_sweep_function(CB_swf.AllXY())
#         elif self.driving is "AWG":
#             self.MC_AllXY.set_sweep_function(self.sweep_function)
#         if self.data_acquisition == 'ATS':
#             self.MC_AllXY.set_detector_function(det.TimeDomainDetector())
#         elif self.data_acquisition == 'CBox':
#             self.MC_AllXY.set_detector_function(
#                 det.QuTechCBox_integrated_average_Detector())

#     def acquire_data_point(self, *args, **kw):
#         if self.measurement_name is not None:
#             measurement_name = self.measurement_name
#         else:
#             iteration = kw.get('iteration', None)
#             print("iteration", iteration)
#             if iteration is None:
#                 sweep_point = kw.get('sweep_point', None)
#                 sweep_point_outer = kw.get('sweep_point_outer', None)
#                 print("sweep outer", sweep_point_outer)
#                 measurement_name = 'AllXY_dev_inner_%s_outer_%s' % (
#                     sweep_point, sweep_point_outer)
#             else:
#                 measurement_name = 'AllXY_iteration_%s' % iteration
#         if self.CLEAR_double is False:
#             if self.optimize_on_ground:
#                 self.source.off()
#             else:
#                 self.source.on()
#             self.MC_AllXY.run(name=measurement_name)
#             t0 = time.time()
#             flip_ax = self.prepi != self.optimize_on_ground
#             ana = ma.AllXY_Analysis(auto=True, flip_axis=flip_ax,
#                                     cal_points=self.sweep_function.cal_points,
#                                     ideal_data=self.sweep_function.ideal_data)
#             print('analyzing AllXY took %.2f' % ((time.time() - t0)))
#             if self.TD_Meas.get_CBox_touch_n_go_save_transients():
#                 ma.TransientAnalysis(auto=True, label='AllXY')
#             deviation = ana.deviation_total
#             return deviation

#         elif self.CLEAR_double:
#             measurement_name = 'AllXY_iteration_'
#             self.source.off()
#             self.MC_AllXY.run(name=measurement_name+"_ground")
#             ana_g = ma.AllXY_Analysis(auto=True,
#                                       cal_points=self.sweep_function.cal_points,
#                                       ideal_data=self.sweep_function.ideal_data)
#             error_g = ana_g.deviation_total
#             if self.TD_Meas.get_CBox_touch_n_go_save_transients():
#                 ma.TransientAnalysis(auto=True, label='AllXY')
#             print("error ground", error_g)
#             self.source.on()
#             self.MC_AllXY.run(name=measurement_name+"_excited")
#             ana_e = ma.AllXY_Analysis(auto=True, flip_axis=True,
#                                       cal_points=self.sweep_function.cal_points,
#                                       ideal_data=self.sweep_function.ideal_data)
#             error_e = ana_e.deviation_total
#             if self.TD_Meas.get_CBox_touch_n_go_save_transients():
#                 ma.TransientAnalysis(auto=True, label='AllXY')
#             print("error excited", error_e)
#             deviation = error_g + error_e
#             print("total deviation", deviation)
#             return deviation, error_g, error_e

#     def finish(self, **kw):
#         self.MC_AllXY.remove()


# class drag_detuning_detector(det.Soft_Detector):
#     '''
#     Performs a drag_detuning measurement and returns the deviation.

#     '''
#     def __init__(self, measurement_name=None, data_acquisition='ATS', **kw):
#         self.detector_control = 'soft'
#         self.name = 'drag_detuning'
#         self.value_names = ['drag detuning']
#         self.value_units = ['a.u.']
#         self.data_acquisition = data_acquisition
#         self.measurement_name = measurement_name

#     def prepare(self, **kw):
#         self.MC_drag = qt.instruments.create('MC_drag', 'MeasurementControl')
#         self.MC_drag.set_sweep_function(CB_swf.drag_detuning())

#         if self.data_acquisition == 'ATS':
#             self.MC_drag.set_detector_function(det.TimeDomainDetector())
#         elif self.data_acquisition == 'CBox':
#             self.MC_drag.set_detector_function(
#                 det.QuTechCBox_integrated_average_Detector())

#     def acquire_data_point(self, *args, **kw):
#         data = self.MC_drag.run(name=self.measurement_name)
#         distance = (data[0][1] - data[0][0])#**2 -
#         #        (data[1][1] - data[1][0])**2)
#         print('Drag detuning = ', distance)
#         return distance

#     def finish(self, **kw):
#         self.MC_drag.remove()


# class drag_detuning_detector_v2(det.Soft_Detector):
#     '''
#     Performs a drag_detuning measurement and returns both the deviation and
#     the measured values.

#     '''
#     def __init__(self, measurement_name=None, data_acquisition='ATS', **kw):
#         self.detector_control = 'soft'
#         self.name = 'drag_detuning'
#         self.value_names = ['drag detuning', 'I_0', 'Q_0', 'I_1', 'Q_1']
#         self.value_units = ['a.u.', 'dac_val', 'dac_val', 'dac_val', 'dac_val']
#         self.data_acquisition = data_acquisition
#         self.measurement_name = measurement_name

#     def prepare(self, **kw):
#         self.MC_drag = qt.instruments.create('MC_drag', 'MeasurementControl')
#         self.MC_drag.set_sweep_function(CB_swf.drag_detuning())

#         if self.data_acquisition == 'ATS':
#             self.MC_drag.set_detector_function(det.TimeDomainDetector())
#         elif self.data_acquisition == 'CBox':
#             self.MC_drag.set_detector_function(
#                 det.QuTechCBox_integrated_average_Detector())

#     def acquire_data_point(self, *args, **kw):
#         data = self.MC_drag.run(name=self.measurement_name)
#         distance = (data[0][1] - data[0][0])#**2 -
#         #        (data[1][1] - data[1][0])**2)
#         print('Drag detuning = ', distance)
#         returned_data = [distance,
#                          data[0][0], data[1][0],  # I_0, Q_0
#                          data[0][1], data[1][1]]  # I_1, Q_1
#         return distance

#     def finish(self, **kw):
#         self.MC_drag.remove()


class TwoTone_Spectroscopy(det.Soft_Detector):
    '''
    Performs a two tone spectroscopy.
    First the resonator is scanned and its resonance is found
    The HM frequency is set to the resonator's frequency, after
    which the qubit is scanned.

    optional args:
        homodynefvals, spectroscopyfvals (GHZ)
        optional kwargs:
        detuning = 0 (in Hz or GHz)
        peak = True (whether the resonator is a peak or not)
    '''

    def __init__(self, *args, **kw):
        # super(Tracked_Spectroscopy, self).__init__()
        self.detector_control = 'soft'
        self.S = kw.pop('Source', qt.instruments['S1'])

        self.name = 'TrackedSpectroscopyMeasurement'
        self.value_names = ['fres', 'fqub']
        self.value_units = ['GHz', 'GHz']
        if len(args) == 2:
            self.homodynefvals = args[0]
            self.spectroscopyfvals = args[1]

        self.resonatorispeak = kw.pop('resonatorispeak', True)
        self.detuning = kw.pop('detuning', 0)
        if self.detuning > 1:
            self.detuning /= 1.e9

    def prepare(self, **kw):
        self.MC = qt.instruments.create('MC_spec_track', 'MeasurementControl')

    def acquire_data_point(self, *args, **kw):
        if len(args) == 2:
            self.homodynefvals = args[0]
            self.spectroscopyfvals = args[1]
        elif (not hasattr(self, 'homodynefvals')
              or (not hasattr(self, 'spectroscopyfvals'))):
            raise NameError('No homodynefvals and/or spectroscopyfvals set')

        HM = qt.instruments['HM']

        # Find resonator resonance frequency
        HM.on()
        homodynedata = self.homodyne(self.homodynefvals, **kw)
        print(np.shape(homodynedata))
        print('peak', self.resonatorispeak)
        self.fres = self.findresonance(homodynedata,
                                       peak=self.resonatorispeak)
        print('Resonator resonance frequency:', self.fres)

        # Find qubit resonance frequency
        HM.set_frequency((self.fres + self.detuning) * 1.e9)
        spectroscopydata = self.spectroscopy(self.spectroscopyfvals, **kw)
        self.fqub = self.findresonance(spectroscopydata,
                                       peak=not self.resonatorispeak)

        print('Qubit resonance frequency:', self.fqub)

        HM.off()
        self.MC.remove()
        return np.array([self.fres, self.fqub])

    def homodyne(self, fvals, **kw):
        self.MC.set_sweep_function(swf.HM_frequency_GHz())
        self.MC.set_sweep_points(fvals)
        self.MC.set_detector_function(det.HomodyneDetector())
        data = self.MC.run(subfolder='resonator', **kw)
        return data

    def spectroscopy(self, fvals, **kw):
        self.MC.set_sweep_function(swf.Source_frequency_GHz(Source=self.S))
        self.MC.set_sweep_points(fvals)
        self.MC.set_detector_function(det.HomodyneDetector())
        data = self.MC.run(subfolder='qubit', **kw)
        return data

    def findresonance(self, data, peak=True):
        if peak:
            resonanceind = np.argmax(data[:, 0])
            resonancefreq = self.MC.get_sweep_points()[resonanceind]
        else:
            resonanceind = np.argmin(data[:, 0])
            resonancefreq = self.MC.get_sweep_points()[resonanceind]
        return resonancefreq

    def finish(self, **kw):
        self.MC_timedomain.remove()


class Qubit_Spectroscopy(det.Soft_Detector):
    '''
    Performs a set of measurements that finds f_resonator, f_qubit,
    '''

    def __init__(self,
                 qubit,
                 res_start=None,
                 res_stop=None,
                 res_freq_step=None,
                 res_use_min=False,
                 res_use_max=False,
                 use_FWHM=False,
                 res_t_int=None,
                 spec_start=None,
                 spec_stop=None,
                 spec_freq_step=0.0001,
                 spec_t_int=None,
                 use_offset=None,
                 spec_sweep_range=0.04,
                 fitting_model='hanger',
                 pulsed=False,
                 **kw):

        # # import placed here to prevent circular import statement
        # #   as some cal_tools use composite detectors.
        from pycqed.measurement import calibration_toolbox as cal_tools
        imp.reload(cal_tools)
        self.cal_tools = cal_tools
        self.qubit = qubit
        self.nested_MC_name = 'Qubit_Spectroscopy_MC'
        self.cal_tools = cal_tools
        self.use_FWHM = use_FWHM
        # Instruments
        self.HM = qt.instruments['HM']
        self.Pulsed_Spec = qt.instruments['Pulsed_Spec']
        self.RF_power = kw.pop('RF_power', qubit.get_RF_CW_power())

        self.qubit_source = qt.instruments[qubit.get_qubit_source()]
        self.qubit_source_power = kw.pop('qubit_source_power',
                                         qubit.get_source_power())

        self.Plotmon = qt.instruments['Plotmon']

        self.res_start = res_start
        self.res_stop = res_stop
        self.res_freq_step = res_freq_step
        self.res_use_min = res_use_min
        self.res_use_max = res_use_max

        self.spec_start = spec_start
        self.spec_stop = spec_stop
        self.spec_freq_step = spec_freq_step
        self.spec_sweep_range = spec_sweep_range

        if (res_t_int is not None) and (spec_t_int is not None):
            self.res_t_int = res_t_int
            self.spec_t_int = spec_t_int
            self.alternate_t_int = True
        else:
            self.alternate_t_int = False

        self.resonator_data = None
        self.qubit_data = None

        self.fitting_model = fitting_model
        self.pulsed = pulsed

        self.detector_control = 'soft'
        self.name = 'Qubit_Spectroscopy_detector'
        self.value_names = ['f_resonator', 'f_resonator_stderr',
                            'f_qubit', 'f_qubit_stderr']
        self.value_units = ['GHz', 'GHz', 'GHz', 'GHz']
        self.msmt_kw = {key.split('msmt_')[1]: val
                        for key, val in list(kw.items()) if 'msmt_' in key}
        # Setting the constants

    def prepare(self, **kw):
        if self.qubit.get_RF_source() is not None:
            self.HM.set_RF_source(self.qubit.get_RF_source())
        self.HM.set_RF_power(self.RF_power)
        self.qubit_source.set_power(self.qubit_source_power)
        self.HM.init()
        self.HM.set_sources('On')

        self.nested_MC = qt.instruments.create(
            self.nested_MC_name,
            'MeasurementControl')
        self.loopcnt = 0

    def acquire_data_point(self, *args, **kw):
        def plot_resonator_data(data):
            if self.resonator_data is None:
                self.resonator_data = data[0]
            else:
                self.resonator_data = np.vstack((self.resonator_data, data[0]))
            self.Plotmon.plot3D(1, np.transpose(self.resonator_data*1.))

        def plot_qubit_data(data):
            if self.qubit_data is None:
                self.qubit_data = data[0]
            else:
                self.qubit_data = np.vstack((self.qubit_data, data[0]))
            self.Plotmon.plot3D(2, np.transpose(self.qubit_data*1.))

        if self.alternate_t_int:
            self.HM.set_t_int(self.res_t_int)
            self.HM.init(optimize=True)

        self.loopcnt += 1
        self.HM.set_sources('On')

        if self.res_start is None:
            cur_f_RO = self.qubit.get_current_RO_frequency()
            self.res_start = cur_f_RO-0.003
            self.res_stop = cur_f_RO+0.003

        print('Scanning for resonator starting at ' + str(self.res_start))
        resonator_scan = \
            self.qubit.find_resonator_frequency(
                use_FWHM=self.use_FWHM,
                MC_name=self.nested_MC_name,
                f_start=self.res_start,
                f_stop=self.res_stop,
                f_step=self.res_freq_step,
                suppress_print_statements=False,
                fitting_model=self.fitting_model,
                use_min=self.res_use_min,
                use_max=self.res_use_max)

        # plot_resonator_data(resonator_scan['data'])
        # print 'BLUUUUURB'
        f_resonator = resonator_scan['f_resonator']+0.00001
        f_resonator_stderr = resonator_scan['f_resonator_stderr']
        self.qubit.set_current_RO_frequency(f_resonator)
        print('Finished resonator scan. Readout frequency: ', f_resonator)

        if self.pulsed is True:
            if self.qubit.get_RF_source() is not None:
                self.Pulsed_Spec.set_RF_source(self.qubit.get_RF_source())
            self.Pulsed_Spec.set_RF_power(self.qubit.get_RF_TD_power())
            self.Pulsed_Spec.set_f_readout(
                self.qubit.get_current_RO_frequency()*1e9)
        else:
            self.HM.set_RF_power(self.qubit.get_RF_CW_power())
            self.HM.set_frequency(self.qubit.get_current_RO_frequency()*1e9)

        if self.alternate_t_int:
            self.HM.set_t_int(self.spec_t_int)
            self.HM.init(optimize=True)

        print('Scanning for qubit')
        qubit_scan = self.qubit.find_frequency_spec(
            MC_name=self.nested_MC_name,
            f_step=self.spec_freq_step,
            f_start=self.spec_start,
            f_stop=self.spec_stop,
            update_qubit=False,  # We do not want a failed track to update
            suppress_print_statements=True,
            source_power=self.qubit_source_power,
            pulsed=self.pulsed)

        # plot_qubit_data(qubit_scan['data'])

        f_qubit = qubit_scan['f_qubit']
        f_qubit_stderr = qubit_scan['f_qubit_stderr']

        print('Estimated qubit frequency: ', f_qubit)
        self.qubit.set_current_frequency(f_qubit)
        self.HM.set_sources('Off')

        return_vals = [f_resonator, f_resonator_stderr,
                       f_qubit, f_qubit_stderr]
        return return_vals

    def finish(self, **kw):
        self.HM.set_sources('Off')
        self.nested_MC.remove()


class Tracked_Qubit_Spectroscopy(det.Soft_Detector):
    '''
    Performs a set of measurements that finds f_resonator, f_qubit, and
    tracks them.

    Uses functions on the qubit object.

    If the sweep points are handed it uses those in predicting frequencies,
    if no sweep points are given it assumes linear spacing between sweep points.
    '''

    def __init__(self, qubit,
                 qubit_initial_frequency=None,
                 qubit_span=0.04,
                 qubit_init_factor=5,
                 qubit_stepsize=0.0005,
                 qubit_t_int=None,
                 resonator_initial_frequency=None,
                 resonator_span=0.010,
                 resonator_stepsize=0.0001,
                 resonator_use_min=False,
                 resonator_use_max=False,
                 resonator_t_int=None,
                 No_of_fitpoints=10,
                 sweep_points=None,
                 fitting_model='hanger',
                 pulsed=False,
                 **kw):

        # import placed here to prevent circular import statement
        #   as some cal_tools use composite detectors.
        from pycqed.measurement import calibration_toolbox \
            as cal_tools
        self.cal_tools = cal_tools
        self.qubit = qubit
        self.nested_MC_name = 'Qubit_Spectroscopy_MC'
        self.cal_tools = cal_tools
        self.resonator_frequency = resonator_initial_frequency
        self.resonator_span = resonator_span
        self.resonator_stepsize = resonator_stepsize
        self.qubit_frequency = qubit_initial_frequency
        self.qubit_span = qubit_span
        self.qubit_init_factor = qubit_init_factor
        self.qubit_stepsize = qubit_stepsize
        self.No_of_fitpoints = No_of_fitpoints
        self.pulsed = pulsed
        self.resonator_use_min = resonator_use_min
        self.resonator_use_max = resonator_use_max
        self.sweep_points = sweep_points

        if (resonator_t_int is not None) and (qubit_t_int is not None):
            self.resonator_t_int = resonator_t_int
            self.qubit_t_int = qubit_t_int
            self.alternate_t_int = True
        else:
            self.alternate_t_int = False

        # Instruments
        if self.pulsed is True:
            self.Pulsed_Spec = qt.instruments['Pulsed_Spec']
        self.fitting_model = fitting_model
        self.AWG = qt.instruments['AWG']  # only for triggering
        self.HM = qt.instruments['HM']

        self.qubit_source = qt.instruments[self.qubit.get_qubit_source()]
        self.RF_power = kw.pop('RF_power', self.qubit.get_RF_CW_power())
        self.qubit_source_power = kw.pop('qubit_source_power',
                                         self.qubit.get_source_power())

        self.detector_control = 'soft'
        self.name = 'Qubit_Spectroscopy'
        self.value_names = ['f_resonator', 'f_resonator_stderr',
                            'f_qubit', 'f_qubit_stderr']
        self.value_units = ['GHz', 'GHz', 'GHz', 'GHz']

    def prepare(self, **kw):
        self.nested_MC = qt.instruments.create(
            self.nested_MC_name, 'MeasurementControl')

        if self.pulsed is True:
            if self.qubit.get_RF_source() is not None:
                self.Pulsed_Spec.set_RF_source(self.qubit.get_RF_source())
                self.HM.set_RF_source(self.qubit.get_RF_source())
            self.Pulsed_Spec.set_RF_power(self.RF_power)
            self.HM.set_RF_power(self.RF_power)
        else:
            if self.qubit.get_RF_source() is not None:
                self.HM.set_RF_source(self.qubit.get_RF_source())
                print('Setting RF source of HM to'+self.qubit.get_RF_source())
            self.HM.set_RF_power(self.RF_power)
        self.qubit_source.set_power(self.qubit_source_power)

        self.AWG.start()
        self.HM.init()
        self.AWG.stop()

        self.HM.set_sources('On')
        self.resonator_frequencies = np.zeros(len(self.sweep_points))
        self.qubit_frequencies = np.zeros(len(self.sweep_points))
        self.loopcnt = 0

        print('\nStarting Tracked Spectroscopy')

    def determine_frequencies(self, loopcnt):
        if self.loopcnt == 0:
            '''
            Uses the inital frequencies to determine where to look
            '''
            f_resonator = self.resonator_frequency

            f_qubit = self.qubit_frequency
            f_qubit_start = self.qubit_frequency - self.qubit_span/2
            f_qubit_end = self.qubit_frequency + self.qubit_span/2

        elif self.loopcnt == 1:
            '''
            Expects the qubit at self.qubit_frequency.
            '''
            f_resonator = self.resonator_frequency

            f_qubit = self.qubit_frequency
            f_qubit_start = self.qubit_frequency \
                - self.qubit_span * self.qubit_init_factor/2
            f_qubit_end = self.qubit_frequency\
                + self.qubit_span * self.qubit_init_factor/2

        elif self.loopcnt == 2:
            '''
            Predicts the qubit and resonator frequency for the third point
            by using the last two points in linear extrapolation.
            uses the np.polyfit and np.poly2d functions.
            '''
            res_fit_coeff = np.polyfit(
                self.sweep_points[:self.loopcnt],
                self.resonator_frequencies[:self.loopcnt], 1)
            res_fit = np.poly1d(res_fit_coeff)
            qub_fit_coeff = np.polyfit(
                self.sweep_points[:self.loopcnt],
                self.qubit_frequencies[:self.loopcnt], 1)
            qub_fit = np.poly1d(qub_fit_coeff)

            f_resonator = res_fit(self.sweep_points[loopcnt])
            f_qubit = qub_fit(self.sweep_points[loopcnt])

            f_qubit_start = f_qubit - self.qubit_span / 2
            f_qubit_end = f_qubit + self.qubit_span / 2
        else:
            '''
            After measuring 3 points quadratic extrapolation is used based
            on all the previous measured points to predict the frequencies.
            uses the np.polyfit and np.poly1d functions.
            '''
            res_fit_coeff = np.polyfit(
                self.sweep_points[:self.loopcnt],
                self.resonator_frequencies[:self.loopcnt], 2)
            res_fit = np.poly1d(res_fit_coeff)
            qub_fit_coeff = np.polyfit(
                self.sweep_points[:self.loopcnt],
                self.qubit_frequencies[:self.loopcnt], 2)
            qub_fit = np.poly1d(qub_fit_coeff)

            f_resonator = res_fit(self.sweep_points[loopcnt])
            f_qubit = qub_fit(self.sweep_points[loopcnt])

            f_qubit_start = f_qubit - self.qubit_span / 2
            f_qubit_end = f_qubit + self.qubit_span / 2

        f_resonator_start = f_resonator - self.resonator_span/2
        f_resonator_end = f_resonator + self.resonator_span/2
        print('Expected qubit frequency: %s' % f_qubit)
        print('Expected resonator frequency: %s' % f_resonator)

        return {'f_resonator_start': f_resonator_start,
                'f_resonator_end': f_resonator_end,
                'f_resonator': f_resonator,
                'f_qubit_start': f_qubit_start,
                'f_qubit_end': f_qubit_end,
                'f_qubit': f_qubit}

    def acquire_data_point(self, *args, **kw):
        self.HM.set_sources('On')

        frequencies = self.determine_frequencies(self.loopcnt)

        # Resonator
        print('\nScanning for resonator.' + \
            'range: {fmin} - {fmax} GHz   span {span} MHz'.format(
                fmin=frequencies['f_resonator_start'],
                fmax=frequencies['f_resonator_end'],
                span=(frequencies['f_resonator_end'] -
                      frequencies['f_resonator_start']) * 1e3))

        if self.alternate_t_int:
            self.HM.set_t_int(self.resonator_t_int)
            self.HM.init(optimize=True)

        resonator_scan = self.qubit.find_resonator_frequency(
            MC_name='Qubit_Spectroscopy_MC',
            f_start=frequencies['f_resonator_start'],
            f_stop=frequencies['f_resonator_end'],
            f_step=self.resonator_stepsize,
            suppress_print_statements=True,
            fitting_model=self.fitting_model,
            use_min=self.resonator_use_min,
            use_max=self.resonator_use_max)

        f_resonator = resonator_scan['f_resonator']
        self.resonator_frequency = f_resonator  # in 2nd loop value is updated
        f_resonator_stderr = resonator_scan['f_resonator_stderr']
        Q_resonator = resonator_scan['quality_factor']
        # Q_resonator_stderr = resonator_scan['quality_factor_stderr']
        print('Finished resonator scan. Readout frequency: ', f_resonator)

        # Qubit
        print('Scanning for qubit.' + \
            'range: {fmin} - {fmax} GHz   span {span} MHz'.format(
                fmin=frequencies['f_qubit_start'],
                fmax=frequencies['f_qubit_end'],
                span=(frequencies['f_qubit_end'] -
                      frequencies['f_qubit_start']) * 1e3))

        self.qubit.set_current_RO_frequency(f_resonator)
        if self.pulsed is False:
            self.HM.set_frequency(self.qubit.get_current_RO_frequency()*1e9)
        elif self.pulsed is True:
            self.Pulsed_Spec.set_f_readout(
                self.qubit.get_current_RO_frequency()*1e9)

        if self.alternate_t_int:
            self.HM.set_t_int(self.qubit_t_int)
            self.HM.init(optimize=True)

        qubit_scan = self.qubit.find_frequency_spec(
            MC_name='Qubit_Spectroscopy_MC',
            f_step=self.qubit_stepsize,
            f_start=frequencies['f_qubit_start'],
            f_stop=frequencies['f_qubit_end'],
            update_qubit=False,  # We do not want a failed track to update
            suppress_print_statements=True,
            source_power=self.qubit_source_power,
            pulsed=self.pulsed)

        f_qubit = qubit_scan['f_qubit']
        f_qubit_stderr = qubit_scan['f_qubit_stderr']
        qubit_linewidth = qubit_scan['qubit_linewidth']
        self.qubit_frequency = f_qubit
        print('Measured Qubit frequency: ', f_qubit)

        self.qubit.set_current_frequency(f_qubit)
        self.resonator_linewidth = f_resonator / Q_resonator
        self.qubit_linewidth = qubit_linewidth

        if self.loopcnt == 1:
            self.resonator_span = max(min(5*self.resonator_linewidth, 0.005),
                                      self.resonator_span)

        print('Resonator width = {linewidth}, span = {span}'.format(
            linewidth=self.resonator_linewidth, span=self.resonator_span))
        print('Qubit width = {}'.format(self.qubit_linewidth))

        self.resonator_frequencies[self.loopcnt] = f_resonator
        self.qubit_frequencies[self.loopcnt] = f_qubit

        self.HM.set_sources('Off')
        self.loopcnt += 1

        return_vals = [f_resonator, f_resonator_stderr,
                       f_qubit, f_qubit_stderr]
        return return_vals

    def finish(self, **kw):
        self.HM.set_sources('Off')
        self.nested_MC.remove()


# class T1_Detector(Qubit_Characterization_Detector):
#     def __init__(self, qubit,
#                  spec_start=None,
#                  spec_stop=None,
#                  pulse_amp_guess=0.7,
#                  AWG_name='AWG',
#                  **kw):
#         from pycqed.measurement import calibration_toolbox as cal_tools
#         imp.reload(cal_tools)
#         self.detector_control = 'soft'
#         self.name = 'T1_Detector'
#         self.value_names = ['f_resonator', 'f_resonator_stderr',
#                             'f_qubit', 'f_qubit_stderr', 'T1', 'T1_stderr']
#         self.value_units = ['GHz', 'GHz', 'GHz', 'GHz', 'us', 'us']

#         self.AWG = qt.instruments[AWG_name]
#         self.pulse_amp_guess = pulse_amp_guess
#         self.cal_tools = cal_tools
#         self.qubit = qubit
#         self.nested_MC_name = 'MC_T1_detector'
#         self.nested_MC = qt.instruments[self.nested_MC_name]
#         self.cal_tools = cal_tools
#         self.spec_start = spec_start
#         self.spec_stop = spec_stop
#         self.qubit_drive_ins = qt.instruments[self.qubit.get_qubit_drive()]
#         self.HM = qt.instruments['HM']
#         self.TD_Meas = qt.instruments['TD_Meas']
#         # Setting the constants
#         self.calreadoutevery = 1
#         self.loopcnt = 0
#         self.T1_stepsize = 500

#     def prepare(self, **kw):

#         self.nested_MC = qt.instruments.create(
#             self.nested_MC_name,
#             'MeasurementControl')

#     def acquire_data_point(self, *args, **kw):
#         self.loopcnt += 1

#         self.switch_to_freq_sweep()

#         cur_f_RO = self.qubit.get_current_RO_frequency()
#         resonator_scan = self.cal_tools.find_resonator_frequency(
#             MC_name=self.nested_MC_name,
#             start_freq=cur_f_RO-0.002,
#             end_freq=cur_f_RO+0.002,
#             suppress_print_statements=True)
#         f_resonator = resonator_scan['f_resonator']
#         f_resonator_stderr = resonator_scan['f_resonator_stderr']
#         print('Readout frequency: ', f_resonator)

#         self.qubit.set_current_RO_frequency(f_resonator)
#         self.HM.set_frequency(self.qubit.get_current_RO_frequency()*1e9)

#         qubit_scan = self.cal_tools.find_qubit_frequency_spec(
#             MC_name=self.nested_MC_name,
#             qubit=self.qubit,
#             start_freq=None,
#             end_freq=None,
#             suppress_print_statements=True)
#         f_qubit = qubit_scan['f_qubit']
#         f_qubit_stderr = qubit_scan['f_qubit_stderr']

#         print('Estimated qubit frequency: ', f_qubit)
#         self.qubit.set_current_frequency(f_qubit)

#         #############################
#         # Start of Time Domain part #
#         #############################

#         self.TD_Meas.set_f_readout(self.qubit.get_current_RO_frequency()*1e9)
#         self.qubit_drive_ins.set_frequency(
#             (self.qubit.get_current_frequency() +
#              self.qubit.get_sideband_modulation_frequency()) * 1e9)
#         self.switch_to_time_domain_measurement()

#         self.qubit.set_pulse_amplitude_I(self.pulse_amp_guess)
#         self.qubit.set_pulse_amplitude_Q(self.pulse_amp_guess)
#         amp_ch1, amp_ch2 = self.cal_tools.calibrate_pulse_amplitude(
#             MC_name=self.nested_MC_name,
#             qubit=self.qubit,
#             max_nr_iterations=3, desired_accuracy=.1, Navg=4,
#             suppress_print_statements=False)

#         self.qubit.set_pulse_amplitude_I(amp_ch1)
#         self.qubit.set_pulse_amplitude_Q(amp_ch2)

#         self.nested_MC.set_detector_function(det.TimeDomainDetector_cal())

#         if self.qubit.pulse_amp_control == 'AWG':
#             self.nested_MC.set_sweep_function(awg_swf.T1(
#                 stepsize=self.T1_stepsize,
#                 gauss_width=self.qubit.get_gauss_width()))
#         elif self.qubit.pulse_amp_control == 'Duplexer':
#             self.nested_MC.set_sweep_function(awg_swf.T1(
#                 stepsize=self.T1_stepsize,
#                 gauss_width=self.qubit.get_gauss_width(),
#                 Duplexer=True))

#         self.nested_MC.run()
#         T1_a = ma.T1_Analysis(auto=True, close_file=False)
#         T1, T1_stderr = T1_a.get_measured_T1()
#         T1_a.finish()
#         self.qubit_drive_ins.off()

#         return_vals = [f_resonator, f_resonator_stderr,
#                        f_qubit, f_qubit_stderr, T1, T1_stderr]
#         return return_vals

#     def finish(self, **kw):
#         self.HM.set_sources('Off')
#         self.nested_MC.remove()

# class HM_SH(det.Soft_Detector):
#     '''
#     Combining a Homodyne measurment and the Signal Hound power at a fixed
#     frequency
#     '''

#     def __init__(self, frequency, **kw):
#         # super(TimeDomainDetector_integrated, self).__init__()

#         self.detector_control = 'soft'
#         self.name = 'HM_SH'
#         self.value_names = ['S21_magn', 'S21_phase', 'Power']
#         self.value_units = ['V', 'deg', 'dBm']
#         self.HomodyneDetector = det.HomodyneDetector()
#         self.Signal_Hound_fixed_frequency = det.Signal_Hound_fixed_frequency(
#                                                     frequency=frequency)

#     def acquire_data_point(self, *args, **kw):
#         HM_data = self.HomodyneDetector.acquire_data_point()
#         SH_data = self.Signal_Hound_fixed_frequency.acquire_data_point()

#         return [HM_data[0], HM_data[1], SH_data]

#     def prepare(self, **kw):
#         self.HomodyneDetector.prepare()
#         self.Signal_Hound_fixed_frequency.prepare()
#         print('prepare worked')

#     def finish(self, **kw):
#         self.HomodyneDetector.finish()
#         self.Signal_Hound_fixed_frequency.finish()
