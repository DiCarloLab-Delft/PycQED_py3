fq = VIP_mon_2_tek.f_qubit()
fq2 = 6.16e9
f_span= 20e6*2
f_step= .5e6

f_span2 = 20e6
f_step2 = .5e6

d = det.CBox_single_integration_average_det(CBox, acq_mode='AmpPhase')

sps.Pulsed_spec_seq(RO_pars, spec_marker_channels=['ch1_marker1'],
                    marker_interval=10e-6, spec_pulse_length=3e-6)
VIP_mon_2_tek.prepare_for_timedomain()
Qubit_LO.pulsemod_state('ON')
Qubit_LO.power(-27)
Pump.power(-13)

MC.set_sweep_function(Qubit_LO.frequency)
MC.set_sweep_function_2D(Pump.frequency)
MC.set_sweep_points(np.arange(fq-f_span/2, fq+f_span/2, f_step))
# MC.set_sweep_points_2D(np.arange(fq2-f_span2/2, fq2+f_span2/2, f_step2))
MC.set_sweep_points_2D(np.arange(6.135e9, 6.17e9, f_step2))
MC.set_detector_function(d)
MC.run('Three tone scan pulsed', mode='2D')
ma.MeasurementAnalysis(TwoD=True)