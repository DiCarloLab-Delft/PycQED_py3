'''
Contains old qtlab versions of the calibration toolbox.
Files that have been moved to the regular
'''


def mixer_carrier_cancellation(frequency, AWG_nr=0, AWG_channel1=1, AWG_channel2=2,
                               AWG_name='AWG', pulse_amp_control='AWG',
                               analyzer='Signal Hound',
                               voltage_grid=[.1, 0.05, 0.02],
                               xtol=0.001):
    '''
    Varies the mixer offsets to minimize leakage at the carrier frequency.

    pulse_amp_control, the name of the instrument at which the offsets are set.
        options for pulse_amp_control are 'AWG' and 'CBox'

    Set the Duplexer such that the desired mixer is adressed, pay attention
    to the attenuation settings. Also, set the RF source to the right
    frequency and turn it on.

    voltage_grid defines the ranges for the preliminary coarse sweeps.
    If the range is too small, add another number infront of -0.12
    '''
    AWG = qt.instruments[AWG_name]
    AWG_type = AWG.get_type()
    MC = qt.instruments['MC']
    CBox = qt.instruments['CBox']

    ch_1_min = 0  # Initializing variables used later on
    ch_2_min = 0
    last_ch_1_min = 1
    last_ch_2_min = 1
    ii = 0
    min_power = 0
    '''
    Make coarse sweeps to approximate the minimum
    '''
    for voltage_span in voltage_grid:
        # Channel 1
        if pulse_amp_control == 'CBox':
            MC.set_sweep_function(CB_swf.DAC_offset(AWG_channel=AWG_channel1,
                                                    AWG_nr=AWG_nr))
        else:
            MC.set_sweep_function(swf.AWG_channel_offset(channel=AWG_channel1,
                                  AWG_name=AWG_name))

        if analyzer == 'Signal Hound':
            MC.set_detector_function(
                det.Signal_Hound_fixed_frequency(SH=SH, frequency=frequency))
        elif analyzer is 'FSV':
            MC.set_detector_function(
                det.RS_FSV_fixed_frequency(frequency=frequency))
        else:
            raise NameError('Analyzer type not found')

        MC.set_sweep_points(np.linspace(ch_1_min + voltage_span,
                                        ch_1_min - voltage_span, 11))
        print('-'*10)
        print(MC.get_sweep_points())
        print('-'*10)
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel1,
               sweep_delay=.1, debug_mode=True)

        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)

        ch_1_min = Mixer_Calibration_Analysis.fit_results[0]

        if pulse_amp_control == 'CBox':
            CBox.set_dac_offset(AWG_nr, AWG_channel1, ch_1_min)
        else:
            exec('AWG.set_ch%s_offset(%s)' % (AWG_channel1, ch_1_min))

        # Channel 2
        if pulse_amp_control == 'CBox':
            MC.set_sweep_function(CB_swf.DAC_offset(AWG_channel=AWG_channel2,
                                                    AWG_nr=AWG_nr))
        else:
            MC.set_sweep_function(swf.AWG_channel_offset(channel=AWG_channel2,
                                  AWG_name=AWG_name))

        MC.set_sweep_points(np.linspace(ch_2_min + voltage_span,
                                        ch_2_min - voltage_span, 11))

        print('-'*10)
        print(MC.get_sweep_points())
        print('-'*10)
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel2,
               sweep_delay=.1, debug_mode=True)

        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)

        ch_2_min = Mixer_Calibration_Analysis.fit_results[0]

        if pulse_amp_control == 'CBox':
            CBox.set_dac_offset(AWG_nr, AWG_channel2, ch_2_min)
        else:
            exec('AWG.set_ch%s_offset(%s)' % (AWG_channel2, ch_2_min))

        '''
        Refine and repeat the sweeps to find the minimum
        '''

    while(abs(last_ch_1_min - ch_1_min) > xtol
          and abs(last_ch_2_min - ch_2_min) > xtol):

        ii += 1

        if pulse_amp_control == 'CBox':
            MC.set_sweep_function(CB_swf.DAC_offset(AWG_channel=AWG_channel1,
                                  AWG_nr=AWG_nr))
            dac_resolution = 1.0
        else:
            MC.set_sweep_function(swf.AWG_channel_offset(channel=AWG_channel1,
                                  AWG_name=AWG_name))
            dac_resolution = 0.001
        if analyzer is 'Signal Hound':
            MC.set_detector_function(
                det.Signal_Hound_fixed_frequency(SH=SH, frequency=frequency))
        elif analyzer is 'FSV':
            MC.set_detector_function(
                det.RS_FSV_fixed_frequency(frequency=frequency))

        MC.set_sweep_points(np.linspace(ch_1_min - dac_resolution*6,
                            ch_1_min + dac_resolution*6, 13))
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel1,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)

        last_ch_1_min = ch_1_min
        ch_1_min = Mixer_Calibration_Analysis.fit_results[0]

        if pulse_amp_control == 'CBox':
            CBox.set_dac_offset(AWG_nr, AWG_channel1, ch_1_min)

        else:
            exec('AWG.set_ch%s_offset(%s)' % (AWG_channel1, ch_1_min))

        if pulse_amp_control == 'CBox':
            MC.set_sweep_function(CB_swf.DAC_offset(AWG_channel=AWG_channel2,
                                  AWG_nr=AWG_nr))
        else:
            MC.set_sweep_function(swf.AWG_channel_offset(channel=AWG_channel2,
                                  AWG_name=AWG_name))
        MC.set_sweep_points(np.linspace(ch_2_min - dac_resolution*6,
                                        ch_2_min + dac_resolution*6, 13))
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel2,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)

        last_ch_2_min = ch_2_min
        min_power = min(Mixer_Calibration_Analysis.measured_powers)
        ch_2_min = Mixer_Calibration_Analysis.fit_results[0]

        if pulse_amp_control == 'CBox':
            CBox.set_dac_offset(AWG_nr, AWG_channel2, ch_2_min)

        else:
            exec('AWG.set_ch%s_offset(%s)' % (AWG_channel2, ch_2_min))

        if ii > 10:
            logging.error('Mixer calibration did not converge')
            break
    return ch_1_min, ch_2_min

def mixer_carrier_cancellation_duplexer(qubit, **kw):
    Duplexer = qt.instruments['Duplexer']
    AWG = qt.instruments['AWG']
    AWG.stop()
    output_channel = qubit.get_duplexer_output_channel()
    frequency = (qubit.get_current_frequency()
                 + qubit.get_sideband_modulation_frequency())
    drive = qt.instruments[qubit.get_qubit_drive()]
    drive.set_frequency(frequency * 1e9)
    drive.set_power(qubit.get_drive_power())
    drive.on()

    Duplexer.set_all_switches_to('Off')

    print('Cancelling carrier leakage mixer 1')
    Duplexer.set_switch(1, output_channel, 'ON')
    Duplexer.set_attenuation(1, output_channel, 49000)

    mixer_carrier_cancellation(frequency=frequency,
                               AWG_channel1=1,
                               AWG_channel2=2, **kw)

    Duplexer.set_switch(1, output_channel, 'OFF')

    print('Cancelling carrier leakage mixer 2')
    Duplexer.set_switch(2, output_channel, 'ON')
    Duplexer.set_attenuation(2, output_channel, 49000)

    mixer_carrier_cancellation(frequency=frequency,
                               AWG_channel1=3,
                               AWG_channel2=4, **kw)

    Duplexer.set_switch(2, output_channel, 'OFF')
    drive.off()



def measure_E_c(qubit, E_c_estimate, span_x=0.05, span_y=0.2,
                stepsize_x=0.002, nr_sweeps=30,
                res_span=0.005, res_step=50e-6, MC_name='MC',
                qubit_source12='S2', find_resonator_frequency=True,
                second_source_extra_power=6):
    '''
    Uses three sources, The HM LO and RF sources are set at the resonator.
    The qubit drive instrument (needs to be rewired) and the qubit source are
    used as the other sources.
    This function does not return anything as analysis needs to be done by hand
    Corresponding analysis file is MA.Three_Tone_Spectroscopy_Analysis()
    '''

    if qubit.get_freq_calc() == 'flux':
        dac_channel = qubit.get_dac_channel()
        Flux_Control = qt.instruments['Flux_Control']
        flux = Flux_Control.get_flux(dac_channel)
        f01 = qubit.calculate_frequency_flux(flux)
        f12 = f01 - E_c_estimate
    else:
        f01 = qubit.get_current_frequency()
        f12 = f01 - E_c_estimate
    HM = qt.instruments['HM']
    MC = qt.instruments[MC_name]
    AWG = qt.instruments['AWG']

    second_source_power = qubit.get_source_power()+second_source_extra_power
    # power of the source sweeping vertically set to quadruple (+3dB)
    # qubit source power
    spec_range_x = np.arange(f01-span_x/2,
                             f01+span_x/2, stepsize_x)
    spec_range_y = np.linspace(f12 - span_y/2.,
                               f12 + span_y/2., nr_sweeps)
    qubit_source = qt.instruments[qubit.get_qubit_source()]
    qubit_source12 = qt.instruments[qubit_source12]

    HM.init()
    HM.set_sources('On')

    if find_resonator_frequency:
        cur_f_RO = qubit.get_current_RO_frequency()
        qubit.find_resonator_frequency(
            MC_name=MC_name,
            f_start=cur_f_RO-res_span/2,
            f_stop=cur_f_RO+res_span/2,
            suppress_print_statements=True,
            f_step=res_step,
            use_min=True)
        print('Readout frequency: ', qubit.get_current_RO_frequency())

    HM.set_frequency(qubit.get_current_RO_frequency()*1e9)
    qubit_source12.set_power(second_source_power)
    qubit_source.set_power(qubit.get_source_power())

    qubit_source12.on()
    qubit_source.on()

    AWG.start()
    MC.set_sweep_function(swf.Source_frequency_GHz(Source=qubit_source))
    MC.set_sweep_function_2D(swf.Source_frequency_GHz(Source=qubit_source12))
    MC.set_sweep_points(spec_range_x)
    MC.set_sweep_points_2D(spec_range_y)
    MC.set_detector_function(det.HomodyneDetector())

    MC.run_2D('Three_tone_spec_E_c')
    AWG.stop()

    HM.set_sources('Off')
    qubit_source12.off()
    qubit_source.off()



def mixer_skewness_calibration_adaptive(source,
                                        generator_frequency,
                                        mixer=None,
                                        sideband_frequency=.1,
                                        MC_name='MC',
                                        pulse_amp_control='AWG'):
    '''
    Calibrates the mixer skewnness
    Using:
      * The AWG 5014, in this case the detector function generates sequences
        for the AWG based on parameters set in the mixer object/instrument.
      * The CBox, in this case a fixed sequence is played in the tektronix
        to ensure the CBox is continously triggered and the paramters are
        reloaded between each measued point.
    Note: mixer cannot be None when driving with the AWG
    '''

    MC = qt.instruments['MC']
    optimization_method = 'Powell'
    # Preparation of experiment
    source.on()
    source.set_frequency(generator_frequency*1e9)
    ampl_min_lst = np.empty(2)
    phase_min_lst = np.empty(2)

    # Initialization of numerical optimization
    if pulse_amp_control is 'AWG':
        sweepfunctions = [swf.IQ_mixer_QI_ratio(mixer),
                          swf.IQ_mixer_skewness(mixer)]
    elif pulse_amp_control is 'CBox':
        AWG.set_setup_filename('FPGA_cont_drive_5014')
        sweepfunctions = [CB_swf.Lut_man_QI_amp_ratio(reload_pulses=True),
                          CB_swf.Lut_man_IQ_phase_skewness(reload_pulses=True)]
    else:
        raise ValueError('pulse_amp_control "%s" not recognized'
                         % pulse_amp_control)

    for i, name in enumerate(
            ['Numerical mixer calibration spurious sideband',
             'Numerical mixer calibration desired sideband']):
        # phase center determines where you signal ends up,
        # sign determines where you look for the minimal signal with the
        # signal hound
        if i == 0:
            phase_center = 0
            sign = -1
        else:
            phase_center = 180
            sign = 1
        # Set the detector
        if pulse_amp_control is 'AWG':
            detector = det.SH_mixer_skewness_det(generator_frequency +
                                                 sign*sideband_frequency,
                                                 mixer=mixer,
                                                 f_mod=sideband_frequency,
                                                 Navg=5, delay=0.5)
        else:
            detector = det.Signal_Hound_fixed_frequency(SH,
                generator_frequency + sign*sideband_frequency,
                Navg=5, delay=0.3)

        start_val = np.array([0.8, phase_center-10])  # units:(ratio, degree)
        initial_stepsize = np.array([.1, 10.])
        x0 = start_val/initial_stepsize  # Initial guess
        x_scale = 1/initial_stepsize  # Scaling parameters

        ad_func_pars = {'x0': x0,
                        'x_scale': x_scale,
                        'bounds': None,
                        'minimize': True,
                        'ftol': 1e-3,
                        'xtol':  5e-2,
                        'maxiter': 500,  # Maximum No. iterations,
                        'maxfun': 500,  # Maximum No. function evaluations,
                        'factr': 1e7,  # 1e7,
                        'pgtol': 1e-2,      # Gradient tolerance,
                        'epsilon': 5e-1,     # Tolerance in measured val,
                        'epsilon_COBYLA': 1,  # Initial step length,
                        'accuracy_COBYLA': 1e-2,  # Convergence tolerance,
                        'constraints': np.array([2, 90])}

        # Experiment:
        MC.set_sweep_functions(sweepfunctions)  # sets swf1 and swf2
        MC.set_detector_function(detector)  # sets test_detector
        MC.set_measurement_mode('adaptive')
        MC.set_measurement_mode_temp(adaptive_function='optimization',
                                     method=optimization_method)

        MC.set_adaptive_function_parameters(ad_func_pars)
        MC.run(name=name)
        a = MA.OptimizationAnalysis(auto=True, label='Numerical')
        ampl_min_lst[i] = a.optimization_result[0][0]
        phase_min_lst[i] = a.optimization_result[0][1]

    print()
    print('Finished calibration')
    print('*'*80)
    print('Phase at minimum w-: {} deg, w+: {} deg'.format(
        phase_min_lst[0], phase_min_lst[1]))
    print('QI_amp_ratio at minimum w-: {},  w+: {}'.format(
        ampl_min_lst[0], ampl_min_lst[1]))
    # print 'Power at minimum: {} dBm'.format(power_min)
    print('*'*80)

    phi = -1*(np.mod((phase_min_lst[0] - (phase_min_lst[1]-180)), 360))/2.0
    alpha = (1/ampl_min_lst[0] + 1/ampl_min_lst[1])/2.
    print('Phi = {} deg'.format(phi))
    print('alpha = {}'.format(alpha))

    # Wrap up experiment
    source.off()
    return phi, alpha

def mixer_skewness_calibration(generator_frequency, sideband_frequency=.1,
                               phase_ranges=[40, 8],
                               QI_amp_ratio_ranges=[.6, .1],
                               estimated_IQ_phase_skewness=0,
                               estimated_QI_amp_ratio=1.,
                               phase_step=None, QI_amp_ratio_step=None,
                               AWG_channel=1,
                               mixer=None,
                               AWG_name='AWG', MC_name='MC', drive_name='IQ',
                               pulse_amp_control='AWG',
                               analyzer='Signal Hound'):
    '''
    Currently supports calibrating mixer skewness both with the Duplexer
    and with the CBox.

    The value AWG_channel is not used when calibrating with the CBox.
    '''
    def set_default_AWG_settings():
        AWG.stop()
        AWG.set_run_mode('SEQ')
        for channel in range(1, 4):
            eval('AWG.set_ch{}_QI_amp_ratio({})'.format(channel,
                 estimated_QI_amp_ratio))

    def measure(name, phase_center, phase_range, QI_amp_ratio_center,
                QI_amp_ratio_range,
                phase_step=None, QI_amp_ratio_step=None):
        phase_range = float(phase_range)
        QI_amp_ratio_range = float(QI_amp_ratio_range)
        if phase_step is None:
            phase_step = phase_range / 20.
        if QI_amp_ratio_step is None:
            QI_amp_ratio_step = max(np.round(QI_amp_ratio_range/30, 3), .001)
        sweep_points_phase = np.arange(phase_center - np.round(phase_range / 2,
                                                               1),
                                       phase_center + np.round(phase_range / 2,
                                                               1),
                                       phase_step)
        sweep_points_ampl = np.arange(QI_amp_ratio_center - QI_amp_ratio_range / 2,
                                      QI_amp_ratio_center + QI_amp_ratio_range / 2,
                                      QI_amp_ratio_step)
        MC.set_sweep_points_2D(sweep_points_phase)
        MC.set_sweep_points(sweep_points_ampl)
        MC.run_2D(name=name)
        ma = MA.Mixer_Skewness_Analysis(auto=True)

        phase_min = ma.phase_min
        QI_min = ma.QI_min

        return phase_min, QI_min

    print('Measuring mixer skewness at frequency', generator_frequency+sideband_frequency)
    MC = qt.instruments[MC_name]
    AWG = qt.instruments[AWG_name]
    drive = qt.instruments[drive_name]
    drive.set_frequency(generator_frequency * 1e9)

    if pulse_amp_control == 'AWG':
        MC.set_sweep_function(swf.IQ_mixer_QI_ratio(mixer))
        MC.set_sweep_function_2D(swf.IQ_mixer_skewness(mixer))

    elif pulse_amp_control != 'CBox':
        CBox = qt.instruments['CBox']
        CBox.set_awg_mode(0, 0)
        CBox.set_awg_mode(1, 0)
        set_default_AWG_settings()
        MC.set_sweep_function_2D(swf.AWG_phase_sequence())
        MC.set_sweep_function(swf.AWG_channel_amplitude(AWG_channel))
    else:
        AWG.set_setup_filename('FPGA_cont_drive_5014')
        MC.set_sweep_function(CB_swf.Lut_man_QI_amp_ratio(reload_pulses=True))
        # only reload on inner loop for speedup
        MC.set_sweep_function_2D(
            CB_swf.Lut_man_IQ_phase_skewness(reload_pulses=False))

    if QI_amp_ratio_ranges is None:
        # Generate QI_amp_ratio range depending on range of phases
        QI_amp_ratio_ranges = np.sqrt(np.array(phase_ranges)) / 20

    AWG.start()
    drive.on()

    ampl_min_lst = [0, 0]
    phase_min_lst = [0, 0]

    for i, name in enumerate(
            ['Mixer calibration spurious sideband',
             'Mixer calibration desired sideband']):
        QI_amp_ratio_center = estimated_QI_amp_ratio
        if i == 0:
            phase_center = estimated_IQ_phase_skewness + 180
            sign = -1
        else:
            phase_center = estimated_IQ_phase_skewness
            sign = 1

        if analyzer == 'Signal Hound' and pulse_amp_control == 'AWG':
            MC.set_detector_function(
                det.SH_mixer_skewness_det(generator_frequency +
                                          sign*sideband_frequency,
                                          mixer=mixer,
                                          f_mod=sideband_frequency))
        elif analyzer == 'Signal Hound':
            MC.set_detector_function(
                det.Signal_Hound_fixed_frequency(generator_frequency +
                                                 sign*sideband_frequency,
                                                 delay=.1, Navg=3))
        else:
            MC.set_detector_function(
                det.RS_FSV_fixed_frequency(frequency=generator_frequency +
                                           sign*sideband_frequency))

        for k, phase_range in enumerate(phase_ranges):
            print('*' * 20)
            print('at loop {} with phase: min={}, max={}, step={}'.format(
                k+1, phase_center - phase_range / 2,
                phase_center + phase_range / 2,
                float(phase_range) / 5))
            QI_amp_ratio_range = QI_amp_ratio_ranges[min(
                k, len(QI_amp_ratio_ranges)-1)]

            phase_min, ampl_min = measure(
                name,
                phase_center, phase_range,
                QI_amp_ratio_center, QI_amp_ratio_range,
                phase_step=phase_step,
                QI_amp_ratio_step=QI_amp_ratio_step)
            phase_center = phase_min
            QI_amp_ratio_center = ampl_min

            print('Minimum phase: {} degree'.format(phase_min))
            print('Minimum QI_amp_ratio: {}'.format(ampl_min))
            # print 'Power at minimum: {} dBm'.format(power_min)
            print()

        ampl_min_lst[i] = ampl_min

        phase_min_lst[i] = phase_min

    if pulse_amp_control != 'CBox':
        set_default_AWG_settings()
    print('Finished calibration')
    print('*'*80)
    print('Phase at minimum w-: {} deg, w+: {} deg'.format(
        phase_min_lst[0], phase_min_lst[1]))
    print('QI_amp_ratio at minimum w-: {},  w+: {}'.format(
        ampl_min_lst[0], ampl_min_lst[1]))
    # print 'Power at minimum: {} dBm'.format(power_min)
    print('*'*80)

    phi = -1*(np.mod((phase_min_lst[0] - phase_min_lst[1]), 360) - 180)/2.0
    alpha = (1/ampl_min_lst[0] + 1/ampl_min_lst[1])/2.
    print('Phi = {} deg'.format(phi))
    print('alpha = {}'.format(alpha))

    return phi, alpha



def fine_reso_freq_range(start_freq,stop_freq,target_freq=None,precise_range=5e6,verbose = False):
    '''
    Create a finer frequency range around the resonator based on the previous resonator position.
    Please use start_freq < stop_freq.
    start_freq and stop_freq are both in Hertz
    '''
    if (target_freq == None):
        previous_timestamp = a_tools.latest_data(contains='Resonator_scan', return_timestamp=True)[0]
        reso_dict = {'f_res_fit':'Fitted Params HM.f0.value'}
        numeric_params = ['f_res_fit']
        data = (a_tools.get_data_from_timestamp_list([previous_timestamp], reso_dict,
                                        numeric_params=numeric_params, filter_no_analysis=False))
        precise_range = precise_range
        reso_freq = data['f_res_fit'][0]*1e9
    else:
        reso_freq = target_freq
    if verbose:
        print('Making a fine list around '+str(reso_freq/1e9)+' GHz')
    if reso_freq == None:
        freq_list_res = np.arange(start_freq,stop_freq,2e5) # Straight part fast, because reso = None
    elif reso_freq < start_freq or reso_freq > stop_freq:
        freq_list_res = np.arange(start_freq,stop_freq,2e5) # Straight part fast, because reso out of range
    elif reso_freq <= start_freq + precise_range/2.:
        freq_list_res = np.hstack([np.arange(start_freq,reso_freq+precise_range/2.,2.5e4), # Reso part precise
              np.arange(reso_freq+precise_range/2.,stop_freq,2e5)]) # Straight part fast
    elif reso_freq >= stop_freq - precise_range/2.:
        freq_list_res = np.hstack([np.arange(start_freq,reso_freq-precise_range/2.,2e5), # Straight part fast
              np.arange(reso_freq-precise_range/2.,stop_freq,2.5e4)]) # Reso part precise
    else:
        freq_list_res = np.hstack([np.arange(start_freq,reso_freq-precise_range/2.,2e5), # Straight part fast
              np.arange(reso_freq-precise_range/2.,reso_freq+precise_range/2.,2.5e4), # Reso part precise
              np.arange(reso_freq+precise_range/2.,stop_freq,2e5)]) # Straight part fast

    return freq_list_res