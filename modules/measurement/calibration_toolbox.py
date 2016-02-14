import numpy as np
import logging
from modules.measurement import sweep_functions as swf
from modules.measurement import CBox_sweep_functions as CB_swf
from modules.measurement import detector_functions as det
from modules.analysis import measurement_analysis as MA
from modules.measurement import mc_parameter_wrapper as pw
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



