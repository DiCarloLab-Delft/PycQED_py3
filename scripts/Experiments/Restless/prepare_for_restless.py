import time
import numpy as np
from copy import deepcopy
t0 = time.time()
station = station
VIP_mon_4_tek = VIP_mon_4_tek
VIP_mon_2_tek =VIP_mon_2_tek
AWG = AWG
AWG520 = AWG520
CBox = CBox
MC = MC
IVVI = IVVI

from scipy.optimize import minimize_scalar

print('Defining functions')


def set_CBox_cos_sine_weigths(IF):
    '''
    Maybe I should add this to the CBox driver
    '''
    t_base = np.arange(512)*5e-9

    cosI = np.cos(2*np.pi * t_base*IF)
    sinI = np.sin(2*np.pi * t_base*IF)
    w0 = np.round(cosI*120)
    w1 = np.round(sinI*120)

    CBox.set('sig0_integration_weights', w0)
    CBox.set('sig1_integration_weights', w1)


def set_trigger_slow():
    AWG520.ch1_m1_high.set(2)
    AWG520.ch1_m2_high.set(0)


def set_trigger_fast():
    AWG520.ch1_m1_high.set(0)
    AWG520.ch1_m2_high.set(2)


def calibrate_RO_threshold_no_rotation():
    d = det.CBox_integration_logging_det(CBox, AWG)
    MC.set_sweep_function(swf.None_Sweep(sweep_control='hard'))
    MC.set_sweep_points(np.arange(8000))
    MC.set_detector_function(d)
    MC.run('threshold_determination')
    a = ma.SSRO_single_quadrature_discriminiation_analysis()
    CBox.sig0_threshold_line.set(int(a.opt_threshold))


def measure_allXY(pulse_pars, RO_pars):
    set_trigger_slow()
    MC.set_sweep_function(awg_swf.AllXY(
        pulse_pars=pulse_pars_duplex, RO_pars=RO_pars,
        upload=True, double_points=True))
    MC.set_detector_function(det.CBox_integrated_average_detector(CBox, AWG))
    MC.run('AllXY')
    ma.AllXY_Analysis()


def measure_RB(pulse_pars, RO_pars, upload=True, T1=25e-6, close_fig=True, pulse_delay=20e-9, label_suffix=''):
    set_trigger_slow()
    nr_seeds = 50
    nr_cliffords = [2, 4, 8, 16, 30, 60, 100, 200, 300, 400, 600, 800, 1200]
    MC.set_sweep_function(awg_swf.Randomized_Benchmarking(
        pulse_pars=pulse_pars, RO_pars=RO_pars, double_curves=True,
        nr_cliffords=nr_cliffords, nr_seeds=nr_seeds, upload=upload))

    MC.set_detector_function(det.CBox_integrated_average_detector(CBox, AWG))
    label ='RB_{}seeds'.format(nr_seeds)+label_suffix
    MC.run(label)
    ma.RB_double_curve_Analysis(
        close_main_fig=close_fig, T1=T1,
        pulse_delay=pulse_delay)

def calibrate_JPA_dac(pulse_pars, RO_pars, upload=True):
    set_trigger_slow()
    set_CBox_cos_sine_weigths(RO_pars['mod_frequency'])
    ad_func_pars = {'adaptive_function': minimize_scalar,
                    'bracket': [-310, -300, -290],
                    'minimize': False,
                    'par_idx': 4}
    MC.set_adaptive_function_parameters(ad_func_pars)
    MC.set_sweep_function(IVVI.dac5)
    if upload:
        sqs.OffOn_seq(pulse_pars=pulse_pars, RO_pars=RO_pars)
    d = cdet.CBox_SSRO_discrimination_detector(
        'SSRO-disc',
        analyze=True,
        MC=nested_MC,
        AWG=AWG,
        CBox=CBox,
        sequence_swf=swf.None_Sweep(sweep_control='hard',
                                    sweep_points=np.arange(1))) #is arbitrare
    MC.set_detector_function(d)
    MC.run(name='JPA_dac_tuning', mode='adaptive')
    ma.MeasurementAnalysis(label='JPA_dac_tuning')



print('setting params of qubit objects')

VIP_mon_4_tek.pulse_I_channel.set('ch1')
VIP_mon_4_tek.pulse_Q_channel.set('ch2')
VIP_mon_4_tek.RO_I_channel.set('ch3')
VIP_mon_4_tek.RO_Q_channel.set('ch4')
VIP_mon_4_tek.RO_acq_marker_channel.set('ch4_marker1')
VIP_mon_4_tek.RO_pulse_marker_channel.set('ch2_marker1')
VIP_mon_4_tek.RO_pulse_power.set(-25)
#VIP_mon_4_tek.RO_pulse_type.set('MW_IQmod_pulse')
VIP_mon_4_tek.RO_pulse_type.set('Gated_MW_RO_pulse')
VIP_mon_4_tek.RO_acq_marker_delay.set(150e-9)
VIP_mon_4_tek.RO_pulse_length.set(8e-7)

VIP_mon_4_tek.RO_pulse_delay.set(50e-9)
VIP_mon_4_tek.f_RO_mod.set(-20e6)
VIP_mon_4_tek.RO_amp.set(0.12)
VIP_mon_4_tek.f_pulse_mod.set(-50e6)
VIP_mon_4_tek.f_RO.set(7.1332*1e9)


#readout pulses and acquisition vipmon_2
VIP_mon_2_tek.RO_I_channel.set('ch3')
VIP_mon_2_tek.RO_Q_channel.set('ch4')
VIP_mon_2_tek.RO_acq_marker_channel.set('ch4_marker1')
VIP_mon_2_tek.RO_pulse_marker_channel.set('ch2_marker1')
VIP_mon_2_tek.RO_pulse_power.set(-35)
VIP_mon_2_tek.f_RO(6.8488e9)

VIP_mon_2_tek.RO_pulse_type.set('Gated_MW_RO_pulse')
VIP_mon_2_tek.RO_acq_marker_delay.set(150e-9)
VIP_mon_2_tek.RO_pulse_length.set(700e-9)
VIP_mon_2_tek.RO_pulse_delay.set(50e-9)
VIP_mon_2_tek.RO_amp.set(0.12)


Qubit_LO.pulsemod_state('Off') #this is the qubit LO
RF.pulsemod_state('On') # this is for pulsed readout RF


#qubit pulses vipmon 2
VIP_mon_2_tek.pulse_I_channel.set('ch1')
VIP_mon_2_tek.pulse_Q_channel.set('ch2')
VIP_mon_2_tek.gauss_sigma.set(4e-9)
VIP_mon_2_tek.pulse_delay.set(20e-9)
VIP_mon_2_tek.f_pulse_mod(-50e6)

print('setting duplxer qubit params')
gen.load_settings_onto_instrument(VIP_mon_2_dux)
VIP_mon_2_dux.RO_pulse_power.set(-35)
VIP_mon_2_dux.f_RO(6.8488e9)

VIP_mon_2_dux.RO_acq_marker_channel.set('ch4_marker1')
VIP_mon_2_dux.RO_pulse_marker_channel.set('ch2_marker1')

VIP_mon_2_dux.RO_pulse_type.set('Gated_MW_RO_pulse')
VIP_mon_2_dux.RO_acq_marker_delay.set(150e-9)
VIP_mon_2_dux.RO_pulse_length.set(700e-9)
VIP_mon_2_dux.RO_pulse_delay.set(50e-9)
VIP_mon_2_dux.RO_amp.set(0.12)

VIP_mon_2_dux.gauss_sigma.set(4e-9)
VIP_mon_2_dux.pulse_delay.set(20e-9)
VIP_mon_2_dux.f_pulse_mod(-50e6)


VIP_mon_2_dux.pulse_GI_offset(0.01)
VIP_mon_2_dux.pulse_GQ_offset(.029)
VIP_mon_2_dux.pulse_DI_offset(0.02)
VIP_mon_2_dux.pulse_DQ_offset(0.030)

VIP_mon_2_dux.D_alpha(0.8565)
VIP_mon_2_dux.D_phi_skew(-9.101)
VIP_mon_2_dux.G_alpha(0.8244)
VIP_mon_2_dux.G_phi_skew(-10.645)

VIP_mon_2_dux.Mux_G_att(0.3)
VIP_mon_2_dux.Mux_D_att(0.7)

VIP_mon_2_dux.Mux_G_phase(10693)
VIP_mon_2_dux.Mux_D_phase(30000)


AWG.timeout(180)


# readout timing trigger voltages
AWG520.ch1_amp(2.0)
AWG520.ch1_offset(1.0)
set_trigger_slow()


#duplexer attenuations
print('setting duplexer settings')
DUX_1_default = 0.3
DUX_2_default = 0.7

Dux.in1_out1_phase(10693)
Dux.in2_out1_phase(30000)



#JPA pump settings
Pump.on()
Pump.power(-3)
Pump.frequency(VIP_mon_2_tek.f_RO()+10e6)
Pump.on()

VIP_mon_2_tek.f_JPA_pump_mod(10e6)


print('setting IVVI parameters')
IVVI.dac1.set(-40)
IVVI.dac2.set(0)  # was 70 for sweetspot VIP_mon_4
# IVVI.dac5(-299.962)  # JPA pump dac

print('setting AWG parameters')
AWG.ch1_offset.set(0.010)
AWG.ch2_offset.set(0.029)
AWG.ch3_offset.set(0.002)
AWG.ch4_offset.set(0.030)
AWG.clock_freq.set(1e9)
AWG.trigger_level.set(0.2)

print('setting CBox parameters')
# Calibrated at 6.6GHz (22-2-2016)
CBox.set_dac_offset(0, 0, -38.8779296875)  # Q channel
CBox.set_dac_offset(0, 1,  16.1220703125)  # I channel qubit drive AWG

CBox.set_dac_offset(1, 1, 0)  # I channel
CBox.set_dac_offset(1, 0, 0)  # Q channel readout AWG
set_CBox_cos_sine_weigths(VIP_mon_2_tek.f_RO_mod())
CBox.set('nr_averages', 2048)
# this is the max nr of averages that does not slow down the heterodyning
CBox.set('nr_samples', 75)  # Shorter because of min marker spacing
CBox.set('integration_length', 140)
CBox.set('acquisition_mode', 0)
CBox.set('lin_trans_coeffs', [1, 0, 0, 1])
CBox.set('log_length', 8000)

CBox.set('AWG0_mode', 'Tape')
CBox.set('AWG1_mode', 'Tape')
CBox.set('AWG0_tape', [1, 1])
CBox.set('AWG1_tape', [1, 1])
CBox.integration_length.set(200)
set_CBox_cos_sine_weigths(VIP_mon_2_tek.f_RO_mod())
CBox.lin_trans_coeffs.set([1, 0, 0, 1])


print('setting pulse_pars_duplex')

pulse_pars_duplex, RO_pars = VIP_mon_2_dux.get_pulse_pars()
t1 = time.time()
print('Ran prepare for restless in {:.2g}s'.format(t1-t0))
