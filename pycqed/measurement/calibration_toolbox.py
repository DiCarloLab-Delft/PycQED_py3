import cma
from deprecated import deprecated

from qcodes.instrument.parameter import ManualParameter

from pycqed.measurement import detector_functions as det
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement.optimization import nelder_mead

from pycqed.analysis import measurement_analysis as ma

# Imported for type annotations
from pycqed.measurement.measurement_control import MeasurementControl
from pycqed.instrument_drivers.physical_instruments.USB_SA124B import SignalHound_USB_SA124B


'''
Contains general calibration routines, most notably for calculating mixer
offsets and skewness. Not all has been transferred to QCodes.

Those things that have not been transferred have a placeholder function and
raise a NotImplementedError.
'''


@deprecated(version='0.4', reason='not used within pyqed')
def measure_E_c(**kw):
    raise NotImplementedError('see archived calibration toolbox')


@deprecated(version='0.4', reason='not used within pyqed')
def mixer_carrier_cancellation_duplexer(**kw):
    raise NotImplementedError('see archived calibration toolbox')


def mixer_carrier_cancellation(
        SH: SignalHound_USB_SA124B,
        source,
        MC: MeasurementControl,
        chI_par, chQ_par,
        frequency: float = None,
        SH_ref_level: float = -40,
        init_stepsize: float = 0.1,
        x0=(0.0, 0.0),
        label: str = 'Offset_calibration',
        ftarget=-110,
        maxiter=300
):
    """
    Varies the mixer offsets to minimize leakage at the carrier frequency.
    this is a generic version.

    Args:
        SH           (instr) : Signal hound used to measure power
        source       (instr) : mw_source that provides the leakage tone
        MC           (instr) :
        chI_par       (par)  :
        chQ_par       (par)  :
        frequency    (float) : the frequency in Hz at which to minimize leakage
        SH_ref_level (float) : Signal hound reference level
        init_stepsize (float): initial stepsize for Nelder mead algorithm
        x0           (tuple) : starting point for optimization
        ftarget      (float) : termination value
    """

    source.on()
    if frequency is None:
        frequency = source.frequency()
    else:
        source.frequency(frequency)

    '''
    Make coarse sweeps to approximate the minimum
    '''
    SH.ref_lvl(SH_ref_level)
    detector = det.Signal_Hound_fixed_frequency(
        SH,
        frequency=(source.frequency()),
        Navg=5,
        delay=0.0,
        prepare_for_each_point=False
    )

    ad_func_pars = {'adaptive_function': cma.fmin,
                    'x0': x0,
                    'sigma0':1,
                    'options': {'maxiter': maxiter,    # maximum function cals
                                # Scaling for individual sigma's
                                'cma_stds': [init_stepsize]*2,
                                'ftarget': ftarget
                                },
                    'minimize': True}
    MC.set_sweep_functions([chI_par, chQ_par])
    MC.set_detector_function(detector)  # sets test_detector
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.run(name=label, mode='adaptive')
    a = ma.OptimizationAnalysis(label=label)
    # v2 creates a pretty picture of the optimizations
    ma.OptimizationAnalysis_v2(label=label)

    ch_1_min = a.optimization_result[0][0]
    ch_2_min = a.optimization_result[0][1]
    return ch_1_min, ch_2_min


@deprecated(version='0.4', reason='not used within pyqed')
def multi_channel_mixer_carrier_cancellation(SH, source, MC,
                               channel_pars,
                               frequency: float=None,
                               SH_ref_level: float=-40,
                               init_stepsize: float=0.1,
                               x0: tuple=None):
    """
    Varies the mixer offsets to minimize leakage at the carrier frequency.
    this is a generic version compatible with multiple channels.

    Args:
        SH           (instr) : Signal hound used to measure power
        source       (instr) : mw_source that provides the leakage tone
        MC           (instr) :
        channel_pars  (par)  : list of offset parameters
        frequency    (float) : the frequency in Hz at which to minimize leakage
        SH_ref_level (float) : Signal hound reference level
        init_stepsize (float): initial stepsize for Nelder mead algorithm
        x0           (tuple) : starting point for optimization
    returns:
        optimization_result (tuple): a tuple containing the final value for
                                     each of the varied parameters.
    """

    source.on()
    if frequency is None:
        frequency = source.frequency()
    else:
        source.frequency(frequency)

    SH.ref_lvl(SH_ref_level)
    detector = det.Signal_Hound_fixed_frequency(
        SH, frequency=(source.frequency()),
        Navg=5, delay=0.0, prepare_each_point=False)

    if x0 is None:
        x0 = [0.0]*len(channel_pars)

    ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': x0,
                    'initial_step': [init_stepsize]*len(channel_pars),
                    'no_improv_break': 15,
                    'minimize': True,
                    'maxiter': 500}
    MC.set_sweep_functions(channel_pars)
    MC.set_detector_function(detector)  # sets test_detector
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.run(name='Offset_calibration', mode='adaptive')
    a = ma.OptimizationAnalysis(label='Offset_calibration')

    return a.optimization_result[0]


@deprecated(version='0.4', reason='not used within pyqed')
def mixer_skewness_calibration_QWG(SH, source, QWG,
                                   alpha, phi,
                                   MC,
                                   ch_pair=1,
                                   frequency=None, f_mod=None,
                                   SH_ref_level: float=-40,
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
    SH.ref_lvl(SH_ref_level)
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
    a = ma.OptimizationAnalysis()
    # phi and alpha are the coefficients that go in the predistortion matrix
    alpha = a.optimization_result[0][0]
    phi = a.optimization_result[0][1]

    return phi, alpha


@deprecated(version='0.4', reason='not used within pyqed')
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
        IQ_phase = ManualParameter('IQ_phase', unit='deg', initial_value=0)
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
                    'no_improv_break': 12,
                    'minimize': True,
                    'maxiter': 500}
    MC.set_sweep_functions([S1, S2])
    MC.set_detector_function(d)
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.run(name=name, mode='adaptive')
    a = ma.OptimizationAnalysis()
    # phi and alpha are the coefficients that go in the predistortion matrix
    alpha = 1/a.optimization_result[0][0]
    phi = -1*a.optimization_result[0][1]

    return phi, alpha


@deprecated(version='0.4', reason='not used within pyqed')
def mixer_skewness_calibration_adaptive(**kw):
    raise NotImplementedError('see archived calibration toolbox')


@deprecated(version='0.4', reason='not used within pyqed')
def mixer_carrier_cancellation_5014(AWG, SH, source, MC,
                                    frequency=None,
                                    AWG_channel1=1,
                                    AWG_channel2=2,
                                    SH_ref_level: float=-40,
                                    **kw):
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
    source.on()
    if frequency is None:
        frequency = source.get('frequency')
    else:
        source.set('frequency', frequency)

    '''
    Make coarse sweeps to approximate the minimum
    '''
    S1 = AWG.ch1_offset  # to be dedicated to actual channel
    S2 = AWG.ch2_offset

    SH.ref_lvl(SH_ref_level)
    detector = det.Signal_Hound_fixed_frequency(
        SH, frequency=(source.frequency.get()),
        Navg=5, delay=0.0, prepare_for_each_point=False)

    ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': [0.0, 0.0],
                    'initial_step': [0.01, 0.01],
                    'no_improv_break': 15,
                    'minimize': True,
                    'maxiter': 500}
    MC.set_sweep_functions([S1, S2])
    MC.set_detector_function(detector)  # sets test_detector
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.run(name='Offset_calibration', mode='adaptive')
    a = ma.OptimizationAnalysis(auto=True, label='Offset_calibration')
    ch_1_min = a.optimization_result[0][0]
    ch_2_min = a.optimization_result[0][1]
    return ch_1_min, ch_2_min


@deprecated(version='0.4', reason='not used within pyqed')
def mixer_carrier_cancellation_UHFQC(UHFQC, SH, source, MC,
                                     frequency=None,
                                     SH_ref_level: float=-40,
                                     **kw):
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
    source.on()
    if frequency is None:
        frequency = source.get('frequency')
    else:
        source.set('frequency', frequency)

    '''
    Make coarse sweeps to approximate the minimum
    '''
    S1 = UHFQC.sigouts_0_offset
    S2 = UHFQC.sigouts_1_offset

    SH.ref_lvl(SH_ref_level)
    detector = det.Signal_Hound_fixed_frequency(
        SH, frequency=(source.frequency.get()),
        Navg=5, delay=0.0, prepare_each_point=False)

    ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': [0.0, 0.0],
                    'initial_step': [0.01, 0.01],
                    'no_improv_break': 15,
                    'minimize': True,
                    'maxiter': 500}
    MC.set_sweep_functions([S1, S2])
    MC.set_detector_function(detector)  # sets test_detector
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.run(name='Offset_calibration', mode='adaptive')
    a = ma.OptimizationAnalysis(auto=True, label='Offset_calibration')
    ch_1_min = a.optimization_result[0][0]
    ch_2_min = a.optimization_result[0][1]
    return ch_1_min, ch_2_min
