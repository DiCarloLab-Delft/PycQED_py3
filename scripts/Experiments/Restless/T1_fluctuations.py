# #traditinal
# MC.live_plot_enabled=False
# CBox.nr_averages(512)
# times=np.linspace(0,120e-6,21)
# pulse_pars, RO_pars = VIP_mon_2_tek.get_pulse_pars()
# MC.set_sweep_function(awg_swf.T1(pulse_pars=pulse_pars, RO_pars=RO_pars, upload=True))
# MC.set_sweep_points(times)
# MC.set_detector_function(det.CBox_integrated_average_detector(CBox, AWG))
# MC.run('T1')
# MC.sweep_functions[0].upload=False
# for i in range(2):
#     MC.run('T1')
# ma.T1_Analysis(label='Measurement')

#resetless
CBox.nr_averages(2048*4)
CBox.nr_averages()

sweep_pts_rstls = np.arange(600)#[opt_att]*100
sweep_pts_trad = 20
total_repetitions = 10000

pulse_pars, RO_pars = VIP_mon_2_tek.get_pulse_pars()
VIP_mon_2_tek.prepare_for_timedomain()

Dux.in1_out1_attenuation.set(DUX_1_default)
Dux.in1_out1_switch('ON')
Dux.in2_out1_switch('OFF')


set_trigger_slow()
CBox.lin_trans_coeffs.set([1, 0, 0, 1])
VIP_mon_2_tek.measure_ssro(set_integration_weights=True)
for j in range(total_repetitions): #makes it repeat all night
    set_trigger_fast()
    dead_time = 4e-6
    sq.Rabi_seq([pulse_pars['amplitude']], pulse_pars, RO_pars, post_msmt_delay=dead_time)
    calibrate_RO_threshold_no_rotation()

    AWG.start()
    MC.set_sweep_function(swf.None_Sweep('soft'))
    MC.set_sweep_points(sweep_pts_rstls)
    MC.set_detector_function(det.CBox_single_qubit_event_s_fraction(CBox))
    #MC.set_detector_function(det.CBox_digitizing_shots_det(CBox, AWG, CBox.sig0_threshold_line()))
    data = MC.run('SNR_alternating_PIrstl')
    ma.MeasurementAnalysis()

    #traditinal
    set_trigger_slow()
    MC.live_plot_enabled=False
    CBox.nr_averages(512)
    times=np.linspace(0,120e-6,21)
    pulse_pars, RO_pars = VIP_mon_2_tek.get_pulse_pars()
    MC.set_sweep_function(awg_swf.T1(pulse_pars=pulse_pars, RO_pars=RO_pars, upload=True))
    MC.set_sweep_points(times)
    MC.set_detector_function(det.CBox_integrated_average_detector(CBox, AWG))
    MC.run('T1')
    MC.sweep_functions[0].upload=False
    for i in range(sweep_pts_trad):
        MC.run('T1')

# Restore settings from before the sweep
Dux.in1_out1_attenuation.set(DUX_1_default)
#Dux.in2_out1_attenuation.set(DUX_2_default)