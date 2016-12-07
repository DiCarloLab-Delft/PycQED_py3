import numpy as np
import logging
from qcodes.instrument.parameter import ManualParameter
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import CBox_sweep_functions as CB_swf
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as MA
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
from pycqed.measurement.optimization import nelder_mead
import imp
imp.reload(MA)
imp.reload(CB_swf)

'''
Contains general calibration routines, most notably for calculating mixer
offsets and skewness. Not all has been transferred to QCodes.

Those things that have not been transferred have a placeholder function and
raise a NotImplementedError.
'''


def measure_E_c(**kw):
    raise NotImplementedError('see archived calibration toolbox')


def mixer_carrier_cancellation_duplexer(**kw):
    raise NotImplementedError('see archived calibration toolbox')


def mixer_carrier_cancellation(SH, source, MC,
                               chI_par, chQ_par,
                               frequency=None,
                               voltage_grid=[0.050, 0.020, 0.010, 0.005, 0.002],
                               range_min = None,
                               range_max = None):

    ch_min = [0, 0]  # Initializing variables used later on
    source.on()
    if frequency is None:
        frequency = source.get('frequency')
    else:
        source.set('frequency', frequency)


    for voltage_span in voltage_grid:
        # Channel 0
        for i, ch_par in enumerate([chI_par, chQ_par]):
            MC.set_sweep_function(ch_par)
            MC.set_detector_function(
                det.Signal_Hound_fixed_frequency(signal_hound=SH, frequency=frequency))

            swp_start = ch_min[i] + voltage_span
            swp_end = ch_min[i] - voltage_span
            swp_pts = np.linspace(swp_start, swp_end,  11)
            print(swp_pts)
            MC.set_sweep_points(swp_pts)
            MC.run(name='Mixer_cal_{}'.format(ch_par.name),
                   sweep_delay=.1)
            Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
                label='Mixer_cal')
            ch_min[i] = Mixer_Calibration_Analysis.fit_results[0]
            ch_par(ch_min[i])

    chI_par(ch_min[0])
    chQ_par(ch_min[1])
    return ch_min



def mixer_skewness_calibration_QWG(SH, source, QWG,
                                    alpha, phi,
                                    MC,
                                    ch_pair=1,
                                    frequency=None, f_mod=None,
                                    name='mixer_skewness_calibration_QWG'):
    '''
    Inputs:
        SH              (instrument)
        Source          (instrument)     MW-source used for driving
        alpha           (parameter)
        phi             (parameter)
        frequency       (float Hz)       Spurious SB freq: f_source - f_mod
        f_mod           (float Hz)       Modulation frequency
        I_ch/Q_ch       (int or str)     Specifies the AWG channels

    returns:
        alpha, phi     the coefficients that go in the predistortion matrix
    For the spurious sideband:
        alpha = 1/QI_amp_optimal
        phi = -IQ_phase_optimal
    For details, see Leo's notes on mixer skewness calibration in the docs
    '''

    QWG.ch1_default_waveform('zero')
    QWG.ch2_default_waveform('zero')
    QWG.ch3_default_waveform('zero')
    QWG.ch4_default_waveform('zero')

    QWG.run_mode('CONt')
    QWG.stop()
    QWG.start()

    if f_mod is None:
        f_mod = QWG.get('ch_pair{}_sideband_frequency'.format(ch_pair))
    else:
        QWG.set('ch_pair{}_sideband_frequency'.format(ch_pair), f_mod)
    if frequency is None:
        # Corresponds to the frequency where to minimize with the SH
        frequency = source.frequency.get() - f_mod


    d = det.Signal_Hound_fixed_frequency(SH, frequency)

    ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': [1.0, 0.0],
                    'initial_step': [.15, 10],
                    'no_improv_break': 10,
                    'minimize': True,
                    'maxiter': 500}
    MC.set_sweep_functions([alpha, phi])
    MC.set_detector_function(d)
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.run(name=name, mode='adaptive')
    a = MA.OptimizationAnalysis()
    # phi and alpha are the coefficients that go in the predistortion matrix
    alpha = a.optimization_result[0][0]
    phi = a.optimization_result[0][1]

    return phi, alpha



def mixer_skewness_calibration_5014(SH, source, station,
                                    MC=None,
                                    QI_amp_ratio=None, IQ_phase=None,
                                    frequency=None, f_mod=10e6,
                                    I_ch=1, Q_ch=2,
                                    name='mixer_skewness_calibration_5014'):
    '''
    Loads a cos and sin waveform in the specified I and Q channels of the
    tektronix 5014 AWG (taken from station.pulsar.AWG).
    By looking at the frequency corresponding with the spurious sideband the
    phase_skewness and amplitude skewness that minimize the signal correspond
    to the mixer skewness.

    Inputs:
        SH              (instrument)
        Source          (instrument)     MW-source used for driving
        station         (qcodes station) Contains the AWG and pulasr sequencer
        QI_amp_ratio    (parameter)      qcodes parameter
        IQ_phase        (parameter)
        frequency       (float Hz)       Spurious SB freq: f_source - f_mod
        f_mod           (float Hz)       Modulation frequency
        I_ch/Q_ch       (int or str)     Specifies the AWG channels

    returns:
        alpha, phi     the coefficients that go in the predistortion matrix
    For the spurious sideband:
        alpha = 1/QI_amp_optimal
        phi = -IQ_phase_optimal
    For details, see Leo's notes on mixer skewness calibration in the docs
    '''
    if frequency is None:
        # Corresponds to the frequency where to minimize with the SH
        frequency = source.frequency.get() - f_mod
    if QI_amp_ratio is None:
        QI_amp_ratio = ManualParameter('QI_amp', initial_value=1)
    if IQ_phase is None:
        IQ_phase = ManualParameter('IQ_phase', units='deg', initial_value=0)
    if MC is None:
        MC = station.MC
    if type(I_ch) is int:
        I_ch = 'ch{}'.format(I_ch)
    if type(Q_ch) is int:
        Q_ch = 'ch{}'.format(Q_ch)

    d = det.SH_mixer_skewness_det(frequency, QI_amp_ratio, IQ_phase, SH,
                                  f_mod=f_mod,
                                  I_ch=I_ch, Q_ch=Q_ch, station=station)
    S1 = pw.wrap_par_to_swf(QI_amp_ratio)
    S2 = pw.wrap_par_to_swf(IQ_phase)

    ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': [1.0, 0.0],
                    'initial_step': [.15, 10],
                    'no_improv_break': 10,
                    'minimize': True,
                    'maxiter': 500}
    MC.set_sweep_functions([S1, S2])
    MC.set_detector_function(d)
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.run(name=name, mode='adaptive')
    a = MA.OptimizationAnalysis()
    # phi and alpha are the coefficients that go in the predistortion matrix
    alpha = 1/a.optimization_result[0][0]
    phi = -1*a.optimization_result[0][1]

    return phi, alpha


def mixer_skewness_calibration_adaptive(**kw):
    raise NotImplementedError('see archived calibration toolbox')


def mixer_carrier_cancellation_5014(AWG, SH, source, MC,
                                    frequency=None,
                                    AWG_channel1=1,
                                    AWG_channel2=2,
                                    voltage_grid=[.1, 0.05, 0.02],
                                    xtol=0.001, **kw):
    '''
    Varies the mixer offsets to minimize leakage at the carrier frequency.
    this is the version for a tektronix AWG.

    station:    QCodes station object that contains the instruments
    source:     the source for which carrier leakage must be minimized
    frequency:  frequency in Hz on which to minimize leakage, if None uses the
                current frequency of the source

    returns:
         ch_1_min, ch_2_min

    voltage_grid defines the ranges for the preliminary coarse sweeps.
    If the range is too small, add another number infront of -0.12

    Note: Updated for QCodes
    '''
    ch_1_min = 0  # Initializing variables used later on
    ch_2_min = 0
    last_ch_1_min = 1
    last_ch_2_min = 1
    ii = 0
    min_power = 0

    source.on()
    if frequency is None:
        frequency = source.get('frequency')
    else:
        source.set('frequency', frequency)

    '''
    Make coarse sweeps to approximate the minimum
    '''
    if type(AWG_channel1) is int:
        ch1_offset = AWG['ch{}_offset'.format(AWG_channel1)]
    else:
        ch1_offset = AWG[AWG_channel1+'_offset']
    if type(AWG_channel2) is int:
        ch2_offset = AWG['ch{}_offset'.format(AWG_channel2)]
    else:
        ch2_offset = AWG[AWG_channel2+'_offset']

    ch1_swf = pw.wrap_par_to_swf(ch1_offset)
    ch2_swf = pw.wrap_par_to_swf(ch2_offset)
    for voltage_span in voltage_grid:
        # Channel 1
        MC.set_sweep_function(ch1_swf)
        MC.set_detector_function(
            det.Signal_Hound_fixed_frequency(signal_hound=SH,
                                             frequency=frequency))
        MC.set_sweep_points(np.linspace(ch_1_min + voltage_span,
                                        ch_1_min - voltage_span, 11))
        MC.run(name='Mixer_cal_Offset_%s' % AWG_channel1,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        ch_1_min = Mixer_Calibration_Analysis.fit_results[0]
        ch1_offset.set(ch_1_min)

        # Channel 2
        MC.set_sweep_function(ch2_swf)
        MC.set_sweep_points(np.linspace(ch_2_min + voltage_span,
                                        ch_2_min - voltage_span, 11))
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel2,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        ch_2_min = Mixer_Calibration_Analysis.fit_results[0]
        ch2_offset.set(ch_2_min)

    # Refine and repeat the sweeps to find the minimum
    while(abs(last_ch_1_min - ch_1_min) > xtol
          and abs(last_ch_2_min - ch_2_min) > xtol):
        ii += 1
        dac_resolution = 0.001
        # channel 1 finer sweep
        MC.set_sweep_function(ch1_swf)
        MC.set_sweep_points(np.linspace(ch_1_min - dac_resolution*6,
                                        ch_1_min + dac_resolution*6, 13))
        MC.run(name='Mixer_cal_Offset_%s' % AWG_channel1,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        last_ch_1_min = ch_1_min
        ch_1_min = Mixer_Calibration_Analysis.fit_results[0]
        ch1_offset.set(ch_1_min)
        # Channel 2 finer sweep
        MC.set_sweep_function(ch2_swf)
        MC.set_sweep_points(np.linspace(ch_2_min - dac_resolution*6,
                                        ch_2_min + dac_resolution*6, 13))
        MC.run(name='Mixer_cal_Offset_%s' % AWG_channel2,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        last_ch_2_min = ch_2_min
        min_power = min(Mixer_Calibration_Analysis.measured_powers)
        ch_2_min = Mixer_Calibration_Analysis.fit_results[0]
        ch2_offset.set(ch_2_min)

        if ii > 10:
            logging.error('Mixer calibration did not converge')
            break
    print(ch_1_min, ch_2_min)
    return ch_1_min, ch_2_min


def mixer_carrier_cancellation_UHFQC(UHFQC, SH, source, MC,
                                     frequency=None,
                                     AWG_channel1=0,
                                     AWG_channel2=1,
                                     voltage_grid=[.1, 0.05, 0.02],
                                     xtol=0.001, **kw):
    '''
    Varies the mixer offsets to minimize leakage at the carrier frequency.
    this is the version for a UHFQC.

    station:    QCodes station object that contains the instruments
    source:     the source for which carrier leakage must be minimized
    frequency:  frequency in Hz on which to minimize leakage, if None uses the
                current frequency of the source

    returns:
         ch_1_min, ch_2_min

    voltage_grid defines the ranges for the preliminary coarse sweeps.
    If the range is too small, add another number infront of -0.12

    Note: Updated for QCodes
    '''
    ch_1_min = 0  # Initializing variables used later on
    ch_2_min = 0
    last_ch_1_min = 1
    last_ch_2_min = 1
    ii = 0
    min_power = 0

    source.on()
    if frequency is None:
        frequency = source.get('frequency')
    else:
        source.set('frequency', frequency)

    '''
    Make coarse sweeps to approximate the minimum
    '''
    ch1_offset = UHFQC.sigouts_0_offset
    ch2_offset = UHFQC.sigouts_1_offset

    ch1_swf = UHFQC.sigouts_0_offset
    ch2_swf = UHFQC.sigouts_1_offset
    for voltage_span in voltage_grid:
        # Channel 1
        MC.set_sweep_function(ch1_swf)
        MC.set_detector_function(
            det.Signal_Hound_fixed_frequency(signal_hound=SH,
                                             frequency=frequency))
        MC.set_sweep_points(np.linspace(ch_1_min + voltage_span,
                                        ch_1_min - voltage_span, 11))
        MC.run(name='Mixer_cal_Offset_%s' % AWG_channel1,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        ch_1_min = Mixer_Calibration_Analysis.fit_results[0]
        ch1_offset(ch_1_min)

        # Channel 2
        MC.set_sweep_function(ch2_swf)
        MC.set_sweep_points(np.linspace(ch_2_min + voltage_span,
                                        ch_2_min - voltage_span, 11))
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel2,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        ch_2_min = Mixer_Calibration_Analysis.fit_results[0]
        ch2_offset(ch_2_min)

    # Refine and repeat the sweeps to find the minimum
    while(abs(last_ch_1_min - ch_1_min) > xtol
          and abs(last_ch_2_min - ch_2_min) > xtol):
        ii += 1
        dac_resolution = 0.001
        # channel 1 finer sweep
        MC.set_sweep_function(ch1_swf)
        MC.set_sweep_points(np.linspace(ch_1_min - dac_resolution*6,
                                        ch_1_min + dac_resolution*6, 13))
        MC.run(name='Mixer_cal_Offset_%s' % AWG_channel1,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        last_ch_1_min = ch_1_min
        ch_1_min = Mixer_Calibration_Analysis.fit_results[0]
        ch1_offset.set(ch_1_min)
        # Channel 2 finer sweep
        MC.set_sweep_function(ch2_swf)
        MC.set_sweep_points(np.linspace(ch_2_min - dac_resolution*6,
                                        ch_2_min + dac_resolution*6, 13))
        MC.run(name='Mixer_cal_Offset_%s' % AWG_channel2,
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        last_ch_2_min = ch_2_min
        min_power = min(Mixer_Calibration_Analysis.measured_powers)
        ch_2_min = Mixer_Calibration_Analysis.fit_results[0]
        ch2_offset.set(ch_2_min)

        if ii > 10:
            logging.error('Mixer calibration did not converge')
            break
    print(ch_1_min, ch_2_min)
    return ch_1_min, ch_2_min


def mixer_carrier_cancellation_CBox(CBox, SH, source, MC,
                                    frequency=None,
                                    awg_nr=0,
                                    voltage_grid=[50, 20, 10, 5, 2],
                                    xtol=1):
    '''
    Varies the mixer offsets to minimize leakage at the carrier frequency.
    this is the version for the QuTech ControlBox

    voltage_grid defines the ranges for the preliminary coarse sweeps.
    If the range is too small, add another number infront of -0.12
    input arguments:
        frequency:  in GHz, if None uses the frequency the source is set to

    Note: Updated for QCodes
    '''
    logging.DeprecationWarning('CBox carrier cancelation is deprecated. \n' +
                    'Replace it with mixer carrier cancelation and pass'
                    ' the channel parameters directly.')
    ch0_swf = CB_swf.DAC_offset(awg_nr, dac_ch=0, CBox=CBox)
    ch1_swf = CB_swf.DAC_offset(awg_nr, dac_ch=1, CBox=CBox)

    return mixer_carrier_cancellation(SH, source, MC,
                               chI_par=ch0_swf, chQ_par=ch1_swf,
                               frequency=frequency, voltage_grid=voltage_grid,
                               xtol=xtol)



def mixer_skewness_cal_CBox_adaptive(CBox, SH, source,
                                     LutMan,
                                     AWG,
                                     MC,
                                     awg_nrs=[0],
                                     calibrate_both_sidebands=False,
                                     verbose=True):
    '''
    Input args
        CBox
        SH:     Signal Hound
        source: MW-source connected to the mixer
        LutMan: Used for changing the pars and loading the pulses
        AWG:    Used for supplying triggers to the CBox
        MC:
        awg_nrs: The awgs used in the CBox to which the pulses are uploaded.
                 (list to allow setting a copy on e.g. awg_nr = 1)


    Calibrates the mixer skewnness
    The CBox, in this case a fixed sequence is played in the tektronix
    to ensure the CBox is continously triggered and the parameters are
    reloaded between each measured point.

    If calibrate_both_sidebands is True the optimization runs two calibrations,
    first it tries to minimize the power in the spurious sideband by varying
    the phase and amplitude skewness. After that it flips the phase 180 degrees
    and repeates the same experiment for the desired sideband. Both should
    give the same result.

    For a description on how to translate these coefficients to a rotation
    matrix see the notes in docs/notes/MixerSkewnessCalibration_LDC_150629.pdf

    If calibrate_both_sidebands is False it will only minimize the signal in
    the spurious sideband. and return those values.

    '''
    # Loads a train of pulses to the AWG to trigger the CBox continuously
    AWG.stop()

    #  Ensure that the block is 4 periods of the modulation freq
    total_time = 200e-6  # Set by the triggerbox
    time_per_pulse = abs(round(1/LutMan.f_modulation.get())*4)
    LutMan.block_length.set(time_per_pulse)  # in ns
    LutMan.ampCW.set(200)
    n_pulses = int(total_time//(time_per_pulse*1e-9))

    # Timing tape that constructs the CW-tone
    timing = [0]*(n_pulses)
    pulse_ids = [LutMan.lut_mapping.get().index('ModBlock')]*n_pulses
    end_of_marker = [False]*(n_pulses-1)+[True]
    tape0 = []
    for i in range(n_pulses):
        tape0.extend(CBox.create_timing_tape_entry(timing[i], pulse_ids[i],
                                                   end_of_marker[i]))
    for awg_nr in awg_nrs:
        LutMan.load_pulses_onto_AWG_lookuptable(awg_nr)
        CBox.set_segmented_tape(awg_nr, tape0)
        CBox.set('AWG{:g}_mode'.format(awg_nr), 'segmented')

    # divide instead of multiply by 1e-9 because of rounding errs
    st_seqs.single_marker_seq()

    AWG.start()
    sweepfunctions = [CB_swf.Lutman_par_with_reload(LutMan,
                                                    LutMan.QI_amp_ratio,
                                                    awg_nrs=awg_nrs),
                      CB_swf.Lutman_par_with_reload(LutMan,
                                                    LutMan.IQ_phase_skewness,
                                                    awg_nrs=awg_nrs)]
    ampl_min_lst = np.empty(2)
    phase_min_lst = np.empty(2)
    if calibrate_both_sidebands:
        sidebands = ['Numerical mixer calibration spurious sideband',
                     'Numerical mixer calibration desired sideband']
    else:
        sidebands = ['Numerical mixer calibration spurious sideband']

    for i, name in enumerate(sidebands):

        sign = -1 if i is 0 else 1  # Flips freq to minimize signal
        # Note Signal hound has frequency in GHz
        detector = det.Signal_Hound_fixed_frequency(
            SH, frequency=(source.frequency.get()/1e9 +
                           sign*LutMan.f_modulation.get()),
            Navg=5, delay=.3)
        # Timing is not finetuned and can probably be sped up

        xtol = 5e-3
        ftol = 1e-3
        start_ratio = 0.8
        phase_center = i * 180  # i=0 is spurious sideband, i=1 is desired
        r_step = .1
        sk_step = 10.
        start_skewness = phase_center-10
        ad_func_pars = {'adaptive_function': 'Powell',
                        'x0': [start_ratio, start_skewness],
                        'direc': [[r_step, 0],
                                  [0, sk_step],
                                  [0, 0]],  # direc is a tuple of vectors
                        'ftol': ftol,
                        'xtol': xtol, 'minimize': True}

        MC.set_sweep_functions(sweepfunctions)  # sets swf1 and swf2
        MC.set_detector_function(detector)  # sets test_detector
        MC.set_adaptive_function_parameters(ad_func_pars)
        MC.run(name=name, mode='adaptive')
        a = MA.OptimizationAnalysis(auto=True, label='Numerical')
        ampl_min_lst[i] = a.optimization_result[0][0]
        phase_min_lst[i] = a.optimization_result[0][1]

    if calibrate_both_sidebands:
        phi = -1*(np.mod((phase_min_lst[0] - (phase_min_lst[1]-180)), 360))/2.0
        alpha = (1/ampl_min_lst[0] + 1/ampl_min_lst[1])/2.
        if verbose:
            print('Finished calibration')
            print('*'*80)
            print('Phase at minimum w-: {} deg, w+: {} deg'.format(
                phase_min_lst[0], phase_min_lst[1]))
            print('QI_amp_ratio at minimum w-: {},  w+: {}'.format(
                ampl_min_lst[0], ampl_min_lst[1]))
            print('*'*80)
            print('Phi = {} deg'.format(phi))
            print('alpha = {}'.format(alpha))
        return phi, alpha
    else:
        return phase_min_lst[0], ampl_min_lst[0]
