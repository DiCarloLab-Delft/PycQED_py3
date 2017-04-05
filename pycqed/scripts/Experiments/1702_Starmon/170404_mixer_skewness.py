def convert_to_clocks(duration, f_sampling=200e6, rounding_period=None):
    """
    #FIXME: move this to a qasm helper or ins_lib file
    convert a duration in seconds to an integer number of clocks

        f_sampling: 200e6 is the CBox sampling frequency
    """
    if rounding_period is not None:
        duration = max(duration//rounding_period, 1)*rounding_period
    clock_duration = int(duration*f_sampling)
    return clock_duration


from pycqed.measurement.waveform_control_CC import instruction_lib as ins_lib
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta
from pycqed.measurement import CBox_sweep_functions as cbs
from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sqqs




def mixer_skewness_calibration_CBoxV3(SH, source, LutMan, MC, CBox,
                                      f_mod, frequency=None,
                                      name='mixer_skewness_calibration_CBox'):
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

    # phi and alpha are the coefficients that go in the predistortion matrix

    # Load the pulses required for a conintuous tone
    LutMan.lut_mapping()[0] = 'ModBlock'
    Mod_Block_len = 500e-9
    Mod_Block_len_clk = convert_to_clocks(Mod_Block_len)
    LutMan.Q_modulation(f_mod)
    LutMan.Q_ampCW(.5)  # not 1 as we want some margin for the alpha correction
    LutMan.load_pulses_onto_AWG_lookuptable()

    # load the QASM/QuMis sequence
    operation_dict = {}
    operation_dict['Pulse'] = {
        'duration': Mod_Block_len_clk,
        'instruction': ins_lib.cbox_awg_pulse(
            codeword=0, awg_channels=[LutMan.awg_nr()],
            duration=Mod_Block_len_clk)}

    # this generates a SSB coninuous wave sequence
    cw_tone_elt = sqqs.CW_tone()
    cw_tone_asm = qta.qasm_to_asm(cw_tone_elt.name, operation_dict)
    CBox.load_instructions(cw_tone_asm.name)
    CBox.start()


    frequency = source.frequency() - f_mod
    alpha_swf = cbs.Lutman_par_with_reload_single_pulse(
        LutMan=LutMan,
        parameter=LutMan.mixer_alpha,
        pulse_names=['ModBlock'])

    phi_swf = cbs.Lutman_par_with_reload_single_pulse(
        LutMan=LutMan,
        parameter=LutMan.mixer_phi,
        pulse_names=['ModBlock'])
    d = det.Signal_Hound_fixed_frequency(SH, frequency)

    ad_func_pars = {'adaptive_function': nelder_mead,
                    'x0': [1.0, 0.0],
                    'initial_step': [.15, 10],
                    'no_improv_break': 10,
                    'minimize': True,
                    'maxiter': 500}
    MC.set_sweep_functions([alpha_swf, phi_swf])
    MC.set_detector_function(d)
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.run(name=name, mode='adaptive')
    a = ma.OptimizationAnalysis()
    ma.OptimizationAnalysis_v2()

    alpha = a.optimization_result[0][0]
    phi = a.optimization_result[0][1]

    return phi, alpha


mixer_skewness_calibration_CBoxV3(SH=SH, source=S87,
                                  MC=MC,  CBox=CBox, LutMan=CBox_LutMan,
                                  f_mod=-50e6)
