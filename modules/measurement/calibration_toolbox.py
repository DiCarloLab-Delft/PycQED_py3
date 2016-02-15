import numpy as np
import logging
from modules.measurement import sweep_functions as swf
from modules.measurement import CBox_sweep_functions as CB_swf
from modules.measurement import detector_functions as det
from modules.analysis import measurement_analysis as MA
from modules.measurement import mc_parameter_wrapper as pw
from modules.measurement.pulse_sequences import standard_sequences as st_seqs
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


def mixer_carrier_cancellation(**kw):
    raise NotImplementedError('Use either mixer'
                              + 'mixer_carrier_cancellation_CBox or' +
                              'mixer_carrier_cancellation_5014' +
                              'See the archived folder for the old function')


def mixer_skewness_calibration(**kw):
    raise NotImplementedError('see archived calibration toolbox')


def mixer_skewness_calibration_adaptive(**kw):
    raise NotImplementedError('see archived calibration toolbox')


def mixer_carrier_cancellation_5014(station,
                                    frequency=None,
                                    AWG_channel1=1,
                                    AWG_channel2=2,
                                    voltage_grid=[.1, 0.05, 0.02],
                                    xtol=0.001, **kw):
    '''
    Varies the mixer offsets to minimize leakage at the carrier frequency.
    this is the version for a tektronix AWG.

    voltage_grid defines the ranges for the preliminary coarse sweeps.
    If the range is too small, add another number infront of -0.12

    Note: Updated for QCodes
    '''
    AWG = station['AWG']
    MC = kw.pop('MC', station.MC)
    SH = station['Signal hound']
    ch_1_min = 0  # Initializing variables used later on
    ch_2_min = 0
    last_ch_1_min = 1
    last_ch_2_min = 1
    ii = 0
    min_power = 0
    '''
    Make coarse sweeps to approximate the minimum
    '''
    ch1_offset = AWG['ch{}_offset'.format(AWG_channel1)]
    ch2_offset = AWG['ch{}_offset'.format(AWG_channel2)]

    ch1_swf = pw.wrap_par_to_swf(ch1_offset)
    ch2_swf = pw.wrap_par_to_swf(ch2_offset)
    for voltage_span in voltage_grid:
        # Channel 1
        MC.set_sweep_function(ch1_swf)
        MC.set_detector_function(
            det.Signal_Hound_fixed_frequency(signal_hound=SH, frequency=frequency))
        MC.set_sweep_points(np.linspace(ch_1_min + voltage_span,
                                        ch_1_min - voltage_span, 11))
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel1,
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
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel1,
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
        MC.run(name='Mixer_cal_Offset_ch%s' % AWG_channel2,
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
                                    voltage_grid=[50, 20, 10, 5],
                                    xtol=1, **kw):
    '''
    Varies the mixer offsets to minimize leakage at the carrier frequency.
    this is the version for the QuTech ControlBox

    voltage_grid defines the ranges for the preliminary coarse sweeps.
    If the range is too small, add another number infront of -0.12
    input arguments:
        frequency:  in GHz, if None uses the frequency the source is set to

    Note: Updated for QCodes
    '''
    ch_0_min = 0  # Initializing variables used later on
    ch_1_min = 0
    last_ch_0_min = 1
    last_ch_1_min = 1
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

    ch0_swf = CB_swf.DAC_offset(awg_nr, dac_ch=0, CBox=CBox)
    ch1_swf = CB_swf.DAC_offset(awg_nr, dac_ch=1, CBox=CBox)
    for voltage_span in voltage_grid:
        # Channel 0
        MC.set_sweep_function(ch0_swf)
        MC.set_detector_function(
            det.Signal_Hound_fixed_frequency(signal_hound=SH, frequency=frequency*1e-9))
        MC.set_sweep_points(np.linspace(ch_0_min + voltage_span,
                                        ch_0_min - voltage_span, 11))
        MC.run(name='Mixer_cal_Offset_awg{}_dac{}'.format(awg_nr, 0),
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        ch_0_min = Mixer_Calibration_Analysis.fit_results[0]
        CBox.set_dac_offset(awg_nr, 0, ch_0_min)

        # Channel 2
        MC.set_sweep_function(ch1_swf)
        MC.set_sweep_points(np.linspace(ch_1_min + voltage_span,
                                        ch_1_min - voltage_span, 11))
        MC.run(name='Mixer_cal_Offset_awg{}_dac{}'.format(awg_nr, 1),
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        ch_1_min = Mixer_Calibration_Analysis.fit_results[0]
        CBox.set_dac_offset(awg_nr, 1, ch_1_min)

    # Refine and repeat the sweeps to find the minimum
    while(abs(last_ch_0_min - ch_0_min) > xtol
          and abs(last_ch_1_min - ch_1_min) > xtol):
        ii += 1
        dac_resolution = 1000/2**14  # quantization of the dacs
        # channel 1 finer sweep
        MC.set_sweep_function(ch0_swf)
        MC.set_sweep_points(np.linspace(ch_0_min - dac_resolution*6,
                            ch_0_min + dac_resolution*6, 13))
        MC.run(name='Mixer_cal_Offset_awg{}_dac{}'.format(awg_nr, 0),
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        last_ch_0_min = ch_0_min
        ch_0_min = Mixer_Calibration_Analysis.fit_results[0]
        CBox.set_dac_offset(awg_nr, 0, ch_0_min)
        # Channel 2 finer sweep
        MC.set_sweep_function(ch1_swf)
        MC.set_sweep_points(np.linspace(ch_1_min - dac_resolution*6,
                                        ch_1_min + dac_resolution*6, 13))
        MC.run(name='Mixer_cal_Offset_awg{}_dac{}'.format(awg_nr, 1),
               sweep_delay=.1, debug_mode=True)
        Mixer_Calibration_Analysis = MA.Mixer_Calibration_Analysis(
            label='Mixer_cal', auto=True)
        last_ch_1_min = ch_1_min
        min_power = min(Mixer_Calibration_Analysis.measured_powers)
        ch_1_min = Mixer_Calibration_Analysis.fit_results[0]
        CBox.set_dac_offset(awg_nr, 1, ch_1_min)
        if ii > 10:
            logging.error('Mixer calibration did not converge')
            break

    CBox.set_dac_offset(awg_nr, 0, ch_0_min)
    CBox.set_dac_offset(awg_nr, 1, ch_1_min)
    return ch_0_min, ch_1_min


def mixer_skewness_cal_CBox_adaptive(CBox, SH, source,
                                     LutMan,
                                     AWG,
                                     MC,
                                     awg_nr=0):
    '''
    Input args
        CBox
        SH:     Signal Hound
        source: MW-source connected to the mixer
        LutMan: Used for changing the pars and loading the pulses
        AWG:    Used for supplying triggers to the CBox
        MC:
        awg_nr: The awg used in the CBox to which the pulses are uploaded.


    Calibrates the mixer skewnness
    The CBox, in this case a fixed sequence is played in the tektronix
    to ensure the CBox is continously triggered and the paramters are
    reloaded between each measued point.

    The optimization runs two calibrations, first it tries to minimize the
    power in the spurious sideband by varying the phase and amplitude skewness.
    After that it flips the phase 180 degrees and repeates the same experiment
    for the desired sideband. Both should give the same result.

    For a description on how to translate these coefficients to a rotation
    matrix see the notes in docs/notes/MixerSkewnessCalibration_LDC_150629.pdf

    '''

    # Loads a train of pulses to the AWG to trigger the CBox continuously

    AWG.stop()
    CBox.AWG0_mode.set('Tape')
    CBox.AWG1_mode.set('Tape')
    # Note if the mod block is not in the lutmapping this will raise an error
    tape = [LutMan.lut_mapping.get().index('ModBlock')]
    CBox.set('AWG0_tape', tape)
    CBox.set('AWG1_tape', tape)
    marker_sep = LutMan.block_length.get()/1e9
    # divide instead of multiply by 1e-9 because of rounding errs
    st_seqs.CBox_marker_train_seq(
        marker_separation=marker_sep)  # Lutman is in ns
    AWG.start()

    sweepfunctions = [pw.wrap_par_to_swf(LutMan.QI_amp_ratio),
                      pw.wrap_par_to_swf(LutMan.IQ_phase_skewness)]
    logging.warning('Check that the AWG-seq is correct')
    logging.warning('Check that the playing the right pulses')
    logging.warning('Check that pulses reload')

    ampl_min_lst = np.empty(2)
    phase_min_lst = np.empty(2)

    for i, name in enumerate(
            ['Numerical mixer calibration spurious sideband',
             'Numerical mixer calibration desired sideband']):

        sign = -1 if i is 0 else 1  # Flips freq to minimize signal
        # Note Signal hound has frequency in GHz
        detector = det.Signal_Hound_fixed_frequency(
            SH, frequency=(source.frequency.get()/1e9 +
                           sign*LutMan.f_modulation.get()),
            Navg=5, delay=.3)

        xtol = 5e-2
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
        return phi, alpha
