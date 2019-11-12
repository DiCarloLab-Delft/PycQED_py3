import numpy as np
import time
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as ma
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.analysis import analysis_toolbox as a_tools
from qcodes.instrument.parameter import ManualParameter


class SSRO_Fidelity_Detector_Tek(det.Soft_Detector):

    '''
    For Qcodes. Readout with CBox, UHFLI, DDM, pulse generation with 5014
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
        elif 'DDM' in str(self.acquisition_instr):
            self.DDM = self.acquisition_instr

        self.nr_averages = nr_averages
        self.integration_length = integration_length
        self.weight_function_I = weight_function_I
        self.weight_function_Q = weight_function_Q
        self.one_weight_function_UHFQC = one_weight_function_UHFQC

    def prepare(self, **kw):
        if not self.optimized_weights:
            self.soft_rotate = True
            self.MC.set_sweep_function(awg_swf.SingleLevel(
                                       pulse_pars=self.pulse_pars,
                                       RO_pars=self.RO_pars,
                                       upload=self.upload))
            self.MC.set_sweep_points(np.arange(self.nr_shots))
            if 'CBox' in str(self.acquisition_instr):
                self.MC.set_detector_function(
                    det.CBox_integration_logging_det(
                        self.acquisition_instr,
                        self.AWG,
                        integration_length=self.integration_length))
                self.CBox = self.acquisition_instr
                if self.SSB:
                    raise ValueError(
                        'SSB is only possible in CBox with optimized weights')
                else:
                    self.CBox.lin_trans_coeffs([1, 0, 0, 1])
                    self.CBox.demodulation_mode('double')
                    if self.IF == None:
                        raise ValueError(
                            'IF has to be provided when not using optimized weights')
                    else:
                        self.CBox.upload_standard_weights(IF=self.IF)

            elif 'UHFQC' in str(self.acquisition_instr):
                self.MC.set_detector_function(
                    det.UHFQC_integration_logging_det(
                        self.acquisition_instr, self.AWG,
                        channels=[
                            self.weight_function_I, self.weight_function_Q],
                        integration_length=self.integration_length,
                        nr_shots=min(self.nr_shots, 4094)))
                if self.SSB:
                    self.UHFQC.prepare_SSB_weight_and_rotation(
                        IF=self.IF, weight_function_I=self.weight_function_I,
                        weight_function_Q=self.weight_function_Q)
                else:
                    if self.IF == None:
                        raise ValueError(
                            'IF has to be provided when not using optimized weights')
                    else:
                        self.UHFQC.prepare_DSB_weight_and_rotation(
                            IF=self.IF,
                            weight_function_I=self.weight_function_I,
                            weight_function_Q=self.weight_function_Q)
            elif 'DDM' in str(self.acquisition_instr):
                self.MC.set_detector_function(
                    det.DDM_integration_logging_det(
                        self.acquisition_instr, self.AWG,
                        channels=[
                            self.weight_function_I, self.weight_function_Q],
                        integration_length=self.integration_length,
                        nr_shots=min(self.nr_shots, 8000)))
                if self.SSB:
                    self.DDM.prepare_SSB_weight_and_rotation(
                        IF=self.IF, weight_function_I=self.weight_function_I,
                        weight_function_Q=self.weight_function_Q)
                #not yet implemented
                # else:
                #     if self.IF == None:
                #         raise ValueError(
                #             'IF has to be provided when not using optimized weights')
                #     else:
                #         self.UHFQC.prepare_DSB_weight_and_rotation(
                #             IF=self.IF,
                #             weight_function_I=self.weight_function_I,
                #             weight_function_Q=self.weight_function_Q)

    def acquire_data_point(self, *args, **kw):
        self.time_start = time.time()
        if self.optimized_weights:
            self.soft_rotate = False
            if 'CBox' in str(self.acquisition_instr):
                self.CBox.nr_averages(int(self.nr_averages))
                if self.SSB:
                    self.CBox.lin_trans_coeffs([1, 1, -1, 1])
                    # self.CBox.demodulation_mode(1)
                    self.CBox.demodulation_mode('single')
                else:
                    self.CBox.lin_trans_coeffs([1, 0, 0, 1])
                    # self.CBox.demodulation_mode(0)
                    self.CBox.demodulation_mode('double')
                self.nr_samples = 512
                self.CBox.nr_samples.set(self.nr_samples)
                SWF = awg_swf.SingleLevel(
                    pulse_pars=self.pulse_pars,
                    RO_pars=self.RO_pars,
                    level='g',
                    nr_samples=self.nr_samples)
                SWF.prepare()
                self.CBox.acquisition_mode('idle')
                self.AWG.start()
                self.CBox.acquisition_mode('input averaging')
                inp_avg_res = self.CBox.get_input_avg_results()

                transient0_I = inp_avg_res[0]
                transient0_Q = inp_avg_res[1]

                SWF = awg_swf.SingleLevel(
                    pulse_pars=self.pulse_pars,
                    RO_pars=self.RO_pars,
                    level='e',
                    nr_samples=self.nr_samples)
                SWF.prepare()
                self.CBox.acquisition_mode('idle')
                self.CBox.acquisition_mode('input averaging')
                self.AWG.start()
                inp_avg_res = self.CBox.get_input_avg_results()
                self.CBox.acquisition_mode('idle')
                transient1_I = inp_avg_res[0]
                transient1_Q = inp_avg_res[1]

                optimized_weights_I = (transient1_I-transient0_I)
                optimized_weights_I = optimized_weights_I - \
                    np.mean(optimized_weights_I)
                weight_scale_factor = 127./np.max(np.abs(optimized_weights_I))
                optimized_weights_I = np.floor(
                    weight_scale_factor*optimized_weights_I).astype(int)

                optimized_weights_Q = (transient1_Q-transient0_Q)
                optimized_weights_Q = optimized_weights_Q - \
                    np.mean(optimized_weights_Q)
                weight_scale_factor = 127./np.max(np.abs(optimized_weights_Q))
                optimized_weights_Q = np.floor(
                    weight_scale_factor*optimized_weights_Q).astype(int)

                self.CBox.sig0_integration_weights.set(optimized_weights_I)
                if self.SSB:
                    self.CBox.sig1_integration_weights.set(
                        optimized_weights_Q)  # disabling the Q quadrature
                else:
                    self.CBox.sig1_integration_weights.set(
                        np.multiply(optimized_weights_Q, 0))  # disabling the Q quadrature
                self.MC.set_sweep_function(awg_swf.SingleLevel(
                                           pulse_pars=self.pulse_pars,
                                           RO_pars=self.RO_pars))
                self.MC.set_sweep_points(np.arange(self.nr_shots))
                self.MC.set_detector_function(
                    det.CBox_integration_logging_det(self.CBox, self.AWG, integration_length=self.integration_length))

            elif 'UHFQC' in str(self.acquisition_instr):
                self.nr_samples = 4096
                self.channels=[
                            self.weight_function_I, self.weight_function_Q]
                #copy pasted from input average prepare
                self.AWG.stop()
                self.UHFQC.qas_0_monitor_length(self.nr_samples)
                self.UHFQC.qas_0_monitor_averages(self.nr_averages)
                self.UHFQC.awgs_0_userregs_1(1)  # 0 for rl, 1 for iavg
                self.UHFQC.awgs_0_userregs_0(
                    int(self.nr_averages))  # 0 for rl, 1 for iavg
                self.nr_sweep_points = self.nr_samples
                self.UHFQC.acquisition_initialize(channels=self.channels, mode='iavg')

                #prepare sweep
                SWF = awg_swf.SingleLevel(
                    pulse_pars=self.pulse_pars,
                    RO_pars=self.RO_pars,
                    level='g',
                    nr_samples=self.nr_samples)
                SWF.prepare()

                #get values detector
                self.UHFQC.qas_0_result_enable(0) # resets UHFQC internal readout counters
                self.UHFQC.acquisition_arm()
                # starting AWG
                if self.AWG is not None:
                    self.AWG.start()

                data_raw=self.UHFQC.acquisition_poll(samples=self.nr_sweep_points,
                                                     arm=False, acquisition_time=0.01)
                data = np.array([data_raw[key] for key in data_raw.keys()])

                #calculating transients
                transient0_I = data[0]
                transient0_Q = data[1]

                self.AWG.stop()
                SWF = awg_swf.SingleLevel(
                    pulse_pars=self.pulse_pars,
                    RO_pars=self.RO_pars,
                    level='e',
                    nr_samples=self.nr_samples)
                SWF.prepare()

                 #get values detector
                self.UHFQC.qas_0_result_enable(0) # resets UHFQC internal readout counters
                self.UHFQC.acquisition_arm()
                # starting AWG
                if self.AWG is not None:
                    self.AWG.start()

                data_raw=self.UHFQC.acquisition_poll(samples=self.nr_sweep_points,
                                                     arm=False, acquisition_time=0.01,
                                                     timeout=100)
                data = np.array([data_raw[key] for key in data_raw.keys()])

                #calculating transients
                transient1_I = data[0]
                transient1_Q = data[1]

                optimized_weights_I = (transient1_I-transient0_I)
                optimized_weights_I = optimized_weights_I - \
                    np.mean(optimized_weights_I)
                weight_scale_factor = 1./np.max(np.abs(optimized_weights_I))
                optimized_weights_I = np.array(
                    weight_scale_factor*optimized_weights_I)

                optimized_weights_Q = (transient1_Q-transient0_Q)
                optimized_weights_Q = optimized_weights_Q - \
                    np.mean(optimized_weights_Q)
                weight_scale_factor = 1./np.max(np.abs(optimized_weights_Q))
                optimized_weights_Q = np.array(
                    weight_scale_factor*optimized_weights_Q)

                eval('self.UHFQC.qas_0_integration_weights_{}_real(np.array(optimized_weights_I))'.format(
                    self.weight_function_I))
                if self.SSB:
                    eval('self.UHFQC.qas_0_integration_weights_{}_imag(np.array(optimized_weights_Q))'.format(
                        self.weight_function_I))
                    if not self.one_weight_function_UHFQC:
                        eval('self.UHFQC.qas_0_integration_weights_{}_real(np.array(optimized_weights_I))'.format(
                            self.weight_function_Q))
                        eval('self.UHFQC.qas_0_integration_weights_{}_imag(np.array(optimized_weights_Q))'.format(
                            self.weight_function_Q))
                    eval(
                        'self.UHFQC.qas_0_rotations_{}(1.0-1.0j)'.format(self.weight_function_I))
                    if not self.one_weight_function_UHFQC:
                        eval(
                            'self.UHFQC.qas_0_rotations_{}(1.0+1.0j)'.format(self.weight_function_Q))
                else:
                    # disabling the other weight fucntions
                    eval(
                        'self.UHFQC.qas_0_integration_weights_{}_imag(0*np.array(optimized_weights_Q))'.format(self.weight_function_I))
                    if not self.one_weight_function_UHFQC:
                        eval(
                            'self.UHFQC.qas_0_integration_weights_{}_real(0*np.array(optimized_weights_I))'.format(self.weight_function_Q))
                        eval(
                            'self.UHFQC.qas_0_integration_weights_{}_imag(0*np.array(optimized_weights_Q))'.format(self.weight_function_Q))
                    eval(
                        'self.UHFQC.qas_0_rotations_{}(1.0+0j)'.format(self.weight_function_I))
                    if not self.one_weight_function_UHFQC:
                        eval(
                            'self.UHFQC.qas_0_rotations_{}(0+0j)'.format(self.weight_function_Q))
                # reading out weights as check
                eval('self.UHFQC.qas_0_integration_weights_{}_real()'.format(
                    self.weight_function_I))
                # reading out weights as check
                eval('self.UHFQC.qas_0_integration_weights_{}_imag()'.format(
                    self.weight_function_I))
                # reading out weights as check
                eval('self.UHFQC.qas_0_integration_weights_{}_real()'.format(
                    self.weight_function_Q))
                # reading out weights as check
                eval('self.UHFQC.qas_0_integration_weights_{}_imag()'.format(
                    self.weight_function_Q))

                self.MC.set_sweep_function(awg_swf.SingleLevel(
                                           pulse_pars=self.pulse_pars,
                                           RO_pars=self.RO_pars))
                self.MC.set_sweep_points(np.arange(self.nr_shots))
                self.MC.set_detector_function(
                    det.UHFQC_integration_logging_det(self.UHFQC, self.AWG,
                                                      channels=[
                                                          self.weight_function_I, self.weight_function_Q],
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
                dset = ana.g.create_dataset('Transients', (self.nr_samples, 4),
                                            maxshape=(self.nr_samples, 4))
                dset[:, 0] = transient0_I
                dset[:, 1] = transient0_Q
                dset[:, 2] = transient1_I
                dset[:, 3] = transient1_Q
            ana.data_file.close()

            # Arbitrary choice, does not think about the deffinition
            time_end = time.time()
            nett_wait = self.wait-time_end+self.time_start
            print(self.time_start)
            if nett_wait > 0:
                time.sleep(nett_wait)
            if self.raw:
                return ana.F_a, ana.theta
            else:
                return ana.F_a, ana.F_d, ana.SNR

class Chevron_optimization_v1(det.Soft_Detector):

    '''
    Chevron optimization.
    '''

    def __init__(self, flux_channel, dist_dict, AWG, MC_nested, qubit,
                 kernel_obj, cost_function_opt=0, **kw):
        super().__init__()
        kernel_dir = 'kernels/'
        self.name = 'chevron_optimization_v1'
        self.value_names = ['Cost function', 'SWAP Time']
        self.value_units = ['a.u.', 'ns']
        self.kernel_obj = kernel_obj
        self.AWG = AWG
        self.MC_nested = MC_nested
        self.qubit = qubit
        self.dist_dict = dist_dict
        self.flux_channel = flux_channel
        self.cost_function_opt = cost_function_opt
        self.dist_dict['ch%d' % self.flux_channel].append('')
        self.nr_averages = kw.get('nr_averages', 1024)

        self.awg_amp_par = ManualParameter(
            name='AWG_amp', unit='Vpp', label='AWG Amplitude')
        self.awg_amp_par.get = lambda: self.AWG.get(
            'ch{}_amp'.format(self.flux_channel))
        self.awg_amp_par.set = lambda val: self.AWG.set(
            'ch{}_amp'.format(self.flux_channel), val)
        self.awg_value = 2.0

        kernel_before_list = self.dist_dict['ch%d' % self.flux_channel]
        kernel_before_loaded = []
        for k in kernel_before_list:
            if k is not '':
                kernel_before_loaded.append(np.loadtxt(kernel_dir+k))
        self.kernel_before = kernel_obj.convolve_kernel(kernel_before_loaded,
                                                        30000)

    def acquire_data_point(self, **kw):
        # # Before writing it
        # # Summarize what to do:

        # # Update kernel from kernel object

        kernel_file = 'optimizing_kernel_%s' % a_tools.current_timestamp()
        self.kernel_obj.save_corrections_kernel(
            kernel_file, self.kernel_before,)
        self.dist_dict['ch%d' % self.flux_channel][-1] = kernel_file+'.txt'

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
                 kernel_obj,  cache, cost_choice='sum', **kw):

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
        lengths_cal = times_vec[-1] + \
            np.arange(1, 1+cal_points)*(times_vec[1]-times_vec[0])
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
        self.AWG.set('ch%d_amp' % self.qubit.fluxing_channel(),
                     self.qubit.SWAP_amp())
        self.MC_nested.run('SWAPN_%s' % self.qubit.name)

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
            # update_qubit=False,  # We do not want a failed track to update
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
                 nested_MC,
                 qubit_initial_frequency=None,
                 qubit_span=0.04e9,
                 qubit_init_factor=5,
                 qubit_stepsize=0.0005e9,
                 resonator_initial_frequency=None,
                 resonator_span=0.010e9,
                 resonator_stepsize=0.0001e9,
                 resonator_use_min=False,
                 resonator_use_max=False,
                 No_of_fitpoints=10,
                 sweep_points=None,
                 fitting_model='hanger',
                 pulsed=False,
                 **kw):
        self.nested_MC = nested_MC
        self.qubit = qubit
        if resonator_initial_frequency != None:

            self.resonator_frequency = resonator_initial_frequency
        else:
            self.resonator_frequency = self.qubit.f_res()
        self.resonator_span = resonator_span
        self.resonator_stepsize = resonator_stepsize
        if qubit_initial_frequency != None:
            self.qubit_frequency = qubit_initial_frequency
        else:
            self.qubit_frequency = self.qubit.f_qubit()
        self.qubit_span = qubit_span
        self.qubit_init_factor = qubit_init_factor
        self.qubit_stepsize = qubit_stepsize
        self.No_of_fitpoints = No_of_fitpoints
        self.pulsed = pulsed
        self.resonator_use_min = resonator_use_min
        self.resonator_use_max = resonator_use_max
        self.sweep_points = sweep_points

        # Instruments
        self.fitting_model = fitting_model

        # self.qubit_source = qubit.cw_source
        # self.RF_power = kw.pop('RF_power', self.qubit.RO_power_cw())
        # if pulsed:
        #     self.qubit_source_power = kw.pop('qubit_source_power',
        #                                      self.qubit.get_source_power())
        # else:

        self.detector_control = 'soft'
        self.name = 'Qubit_Spectroscopy'
        self.value_names = ['f_resonator',  # 'f_resonator_stderr',
                            'f_qubit']  # , 'f_qubit_stderr']
        self.value_units = ['Hz', 'Hz']  # , 'Hz', 'Hz']

    def prepare(self, **kw):

        # if self.pulsed is True:
        #     if self.qubit.get_RF_source() is not None:
        #         self.Pulsed_Spec.set_RF_source(self.qubit.get_RF_source())
        #         self.HM.set_RF_source(self.qubit.get_RF_source())
        #     self.Pulsed_Spec.set_RF_power(self.RF_power)
        #     self.HM.set_RF_power(self.RF_power)
        # else:
        #     if self.qubit.get_RF_source() is not None:
        #         self.HM.set_RF_source(self.qubit.get_RF_source())
        #         print('Setting RF source of HM to'+self.qubit.get_RF_source())
        #     self.HM.set_RF_power(self.RF_power)
        # self.qubit_source.set_power(self.qubit_source_power)

        # self.AWG.start()
        # self.HM.init()
        # self.AWG.stop()

        # self.HM.set_sources('On')
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
        # self.HM.set_sources('On')

        frequencies = self.determine_frequencies(self.loopcnt)

        # Resonator
        print('\nScanning for resonator.' +
              'range: {fmin} - {fmax} GHz   span {span} MHz'.format(
                  fmin=frequencies['f_resonator_start'],
                  fmax=frequencies['f_resonator_end'],
                  span=(frequencies['f_resonator_end'] -
                        frequencies['f_resonator_start'])))

        # if self.alternate_t_int:
        #     self.HM.set_t_int(self.resonator_t_int)
        #     self.HM.init(optimize=True)

        freqs_res = np.arange(frequencies['f_resonator_start'],
                              frequencies['f_resonator_end'],
                              self.resonator_stepsize)
        f_resonator = self.qubit.find_resonator_frequency(
            MC=self.nested_MC,
            freqs=freqs_res, update=False,
            use_min=self.resonator_use_min)*1e9  # to correct for fit
        # FIXME: remove the 1e9 after reloading the qubit object

        # FIXME not returned in newest version
        # self.resonator_frequency = f_resonator  # in 2nd loop value is updated
        # f_resonator_stderr = resonator_scan['f_resonator_stderr']
        # Q_resonator = resonator_scan['quality_factor']
        # Q_resonator_stderr = resonator_scan['quality_factor_stderr']
        print('Finished resonator scan. Readout frequency: ', f_resonator)

        # Qubit
        print('Scanning for qubit.' +
              'range: {fmin} - {fmax} GHz   span {span} MHz'.format(
                  fmin=frequencies['f_qubit_start'],
                  fmax=frequencies['f_qubit_end'],
                  span=(frequencies['f_qubit_end'] -
                        frequencies['f_qubit_start'])))

        self.qubit.f_RO(f_resonator)
        freqs_qub = np.arange(frequencies['f_qubit_start'],
                              frequencies['f_qubit_end'],
                              self.qubit_stepsize)

        self.qubit.measure_spectroscopy(
            freqs=freqs_qub,
            MC=self.nested_MC,
            pulsed=self.pulsed)
        a = ma.Qubit_Spectroscopy_Analysis(label=self.qubit.msmt_suffix)
        f_qubit, std_err_f_qubit = a.get_frequency_estimate()
        # f_qubit = qubit_scan['f_qubit']
        # f_qubit_stderr = qubit_scan['f_qubit_stderr']
        # qubit_linewidth = qubit_scan['qubit_linewidth']
        # self.qubit_frequency = f_qubit
        self.qubit_frequency = f_qubit
        print('Measured Qubit frequency: ', f_qubit)

        self.qubit.set_current_frequency(f_qubit)
        # self.resonator_linewidth = f_resonator / Q_resonator
        # self.qubit_linewidth = qubit_linewidth

        if self.loopcnt == 1:
            self.resonator_span = max(min(5*self.resonator_linewidth, 0.005),
                                      self.resonator_span)

        # print('Resonator width = {linewidth}, span = {span}'.format(
        #     linewidth=self.resonator_linewidth, span=self.resonator_span))
        # print('Qubit width = {}'.format(self.qubit_linewidth))

        self.resonator_frequencies[self.loopcnt] = f_resonator
        self.qubit_frequencies[self.loopcnt] = f_qubit

        # self.HM.set_sources('Off')
        self.loopcnt += 1
        return_vals = [f_resonator, f_qubit]
        # return_vals = [f_resonator, f_resonator_stderr,
        #                f_qubit, f_qubit_stderr]
        return return_vals

    def finish(self, **kw):
        # self.HM.set_sources('Off')
        pass


class FluxTrack(det.Soft_Detector):
    '''
    '''

    def __init__(self, qubit, device, MC, AWG, cal_points=False, **kw):
        self.detector_control = 'soft'
        self.name = 'FluxTrack'
        self.cal_points = cal_points
        self.value_names = [r' +/- $F |1\rangle$',
                            r' + $F |1\rangle$', r' - $F |1\rangle$']
        self.value_units = ['', '', '']
        self.qubit = qubit
        self.AWG = AWG
        self.MC = MC
        self.operations_dict = device.get_operation_dict()
        self.dist_dict = qubit.dist_dict()
        self.nested_MC = MC

        self.FluxTrack_swf = awg_swf.awg_seq_swf(
            fsqs.FluxTrack,
            # parameter_name='Amplitude',
            unit='V',
            AWG=self.AWG,
            fluxing_channels=[self.qubit.fluxing_channel()],
            awg_seq_func_kwargs={'operation_dict': self.operations_dict,
                                 'q0': self.qubit.name,
                                 'cal_points': self.cal_points,
                                 'distortion_dict': self.dist_dict,
                                 'upload': True})

    def prepare(self, **kw):
        self.FluxTrack_swf.prepare()
        self.FluxTrack_swf.upload = False

    def acquire_data_point(self, *args, **kw):
            # acquire with MC_nested
        self.MC.set_sweep_function(self.FluxTrack_swf)
        self.MC.set_sweep_points(np.arange(2+4*self.cal_points))
        if self.cal_points:
            d = self.qubit.int_avg_det_rot
        else:
            d = self.qubit.int_avg_det

        self.MC.set_detector_function(d)
        self.MC.run('FluxTrack_point_%s' % self.qubit.name)

        ma_obj = ma.MeasurementAnalysis(auto=True, label='FluxTrack_point')
        y_p = ma_obj.measured_values[0, 0]
        y_m = ma_obj.measured_values[0, 1]
        y_mean = np.mean([y_p, y_m])
        return (y_mean, y_p, y_m)


class purity_CZ_detector(det.Soft_Detector):

    def __init__(self, measurement_name: str, MC, device, q0, q1,
                 return_purity_only: bool=True):
        self.measurement_name = measurement_name
        self.MC = MC
        self.name = 'purity_CZ_detector'
        self.detector_control = 'soft'
        self.device = device
        self.q0 = q0
        self.q1 = q1
        self.return_purity_only = return_purity_only

        if self.return_purity_only:
            self.value_names = ['Purity sum', 'Purity {}'.format(q0.name),
                                'Purity {}'.format(q1.name)]
            self.value_units = ['']*3
        else:
            self.value_names = ['Ps', 'P_{}'.format(q0.name),
                                'p_{}'.format(q1.name), 'IX', 'IY',
                                'IZ', 'XI', 'YI', 'ZI']
            self.value_units = ['']*3 + ['frac']*6

    def prepare(self):
        self.i = 0
        purity_CZ_seq = qwfs.purity_CZ_seq(self.q0.name, self.q1.name)
        self.s = swf.QASM_Sweep_v2(qasm_fn=purity_CZ_seq.name,
                                   config=self.device.qasm_config(),
                                   CBox=self.device.central_controller.get_instr(),
                                   verbosity_level=0,
                                   parameter_name='Segment',
                                   unit='#', disable_compile_and_upload=True)

        self.d = self.device.get_correlation_detector()

        # the sequence only get's compiled and uploaded in the prepare
        self.s.compile_and_upload(self.s.qasm_fn, self.s.config)

    def acquire_data_point(self, **kw):

        self.MC.set_sweep_function(self.s)
        self.MC.set_sweep_points(np.arange(3))
        self.MC.set_detector_function(self.d)
        dat = self.MC.run(name=self.measurement_name+'_'+str(self.i))
        dset = dat["dset"]

        q0_states = dset[:, 1]
        q1_states = dset[:, 2]

        # P_q0 = <sigma_x>^2 + <sigma_y>^2 + <sigma_z>^2
        purity_q0 = (self.frac_to_pauli_exp(q0_states[0]) +
                     self.frac_to_pauli_exp(q0_states[1]) +
                     self.frac_to_pauli_exp(q0_states[2]))
        purity_q1 = (self.frac_to_pauli_exp(q1_states[0]) +
                     self.frac_to_pauli_exp(q1_states[1]) +
                     self.frac_to_pauli_exp(q1_states[2]))
        ps = purity_q0 + purity_q1

        self.i += 1
        if self.return_purity_only:
            return ps, purity_q0, purity_q1
        else:
            return ps, purity_q0, purity_q1, q0_states, q1_states

    def frac_to_pauli_exp(self, frac):
        """
        converts a measured fraction to a pauli expectation value
        <sigma_i>^2 = (2 * (frac - 0.5))**2
        """

        sigma_i2 = (2*(frac - 0.5))**2
        return sigma_i2


class purityN_CZ_detector(purity_CZ_detector):

    def __init__(self, measurement_name: str, N: int,
                 MC, device, q0, q1,
                 return_purity_only: bool=True):
        super().__init__(measurement_name=measurement_name, MC=MC,
                         device=device, q0=q0, q1=q1,
                         return_purity_only=return_purity_only)
        self.N = N

    def prepare(self):
        self.i = 0
        purity_CZ_seq = qwfs.purity_N_CZ_seq(self.q0.name, self.q1.name,
                                             N=self.N)
        QWG_flux_lutmans = [self.q0.flux_LutMan.get_instr(),
                            self.q1.flux_LutMan.get_instr()]

        self.s = swf.QWG_flux_QASM_Sweep(
            qasm_fn=purity_CZ_seq.name,
            config=self.device.qasm_config(),
            CBox=self.device.central_controller.get_instr(),
            QWG_flux_lutmans=QWG_flux_lutmans,
            parameter_name='Segment',
            unit='#', disable_compile_and_upload=False,
            verbosity_level=0)

        self.d = self.device.get_correlation_detector()

    def acquire_data_point(self, **kw):

        self.MC.set_sweep_function(self.s)
        self.MC.set_sweep_points(np.arange(3))
        self.MC.set_detector_function(self.d)
        dat = self.MC.run(name=self.measurement_name+'_'+str(self.i))
        dset = dat["dset"]

        q0_states = dset[:, 1]
        q1_states = dset[:, 2]

        # P_q0 = <sigma_x>^2 + <sigma_y>^2 + <sigma_z>^2
        purity_q0 = (self.frac_to_pauli_exp(q0_states[0]) +
                     self.frac_to_pauli_exp(q0_states[1]) +
                     self.frac_to_pauli_exp(q0_states[2]))
        purity_q1 = (self.frac_to_pauli_exp(q1_states[0]) +
                     self.frac_to_pauli_exp(q1_states[1]) +
                     self.frac_to_pauli_exp(q1_states[2]))
        ps = purity_q0 + purity_q1

        self.i += 1
        # self.s.disable_compile_and_upload = True
        if self.return_purity_only:
            return ps, purity_q0, purity_q1
        else:
            return ps, purity_q0, purity_q1, q0_states, q1_states

    def frac_to_pauli_exp(self, frac):
        """
        converts a measured fraction to a pauli expectation value
        <sigma_i>^2 = (2 * (frac - 0.5))**2
        """

        sigma_i2 = (2*(frac - 0.5))**2
        return sigma_i2



class CPhase_optimization(det.Soft_Detector):

    def __init__(self, qb_control, qb_target, MC,
                 ramsey_phases=None,
                 spacing=10e-9,
                 **kw):
        '''
        soft detector function used in the CZ gate tuneup optimization

        Args:
            qb_control (QuDev_Transmon): control qubit (with flux pulses)
            qb_target (QuDev_Transmon): target qubit
            MC(MeasurementControl): measurement control used in the detector
                                     function to run the actual experiment
            maxiter (int): maximum optimization steps passed to the nelder mead function
            name (str): measurement name
            spacing (float): safety spacing between drive pulses and flux pulse

        '''
        super().__init__()

        self.flux_pulse_length = ManualParameter(name='flux_pulse_length',
                                                 unit='s',
                                                 label='Flux pulse length')
        self.flux_pulse_amp = ManualParameter(name='flux_pulse_amp',
                                              unit='V',
                                              label='Flux pulse amplitude')

        self.name = 'CZ_cond_phase_optimization'
        self.value_names = ['Cost function']
        self.value_units = ['a.u.']
        self.MC = MC
        self.qb_control = qb_control
        self.qb_target = qb_target
        self.spacing = spacing

        self.nr_averages = kw.get('nr_averages', 1024)

        if ramsey_phases is None:
            self.ramsey_phases = np.linspace(0, 2*np.pi, 16, endpoint=False)
            self.ramsey_phases = np.concatenate((self.ramsey_phases,
                                                 self.ramsey_phases))
        else:
            self.ramsey_phases = ramsey_phases

    def acquire_data_point(self, **kw):

        print(self.flux_pulse_amp.name, ' set to ',
              self.flux_pulse_amp(), self.flux_pulse_amp.unit)
        print(self.flux_pulse_length.name, ' set to ',
              self.flux_pulse_length(), self.flux_pulse_length.unit)

        cphases, population_losses = self.qb_control.measure_cphase(
                    MC=self.MC,
                    qb_target=self.qb_target,
                    amps=[self.flux_pulse_amp()],
                    lengths=[self.flux_pulse_length()],
                    phases=self.ramsey_phases,
                    return_population_loss=True,
                    upload_channels=[self.qb_control.flux_pulse_channel()],
                    prepare_for_timedomain=False,
                    spacing=self.spacing
                    )
        print('measured conditional phase: ', cphases[0]/np.pi*180)

        cost_val = np.abs(cphases[0] % (2*np.pi) - np.pi)/np.pi \
                    + population_losses[0]

        # # Return the cost function
        return cost_val

    def prepare(self):
        pass

    def finish(self):
        pass

