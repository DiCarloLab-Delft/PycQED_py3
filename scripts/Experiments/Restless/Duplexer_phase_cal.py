from modules.measurement.pulse_sequences import calibration_elements as cal_elts
reload(cal_elts)
from scipy.optimize import minimize_scalar

cal_elts.station = station

mod_freq = -50e6

cal_elts.cos_seq(.1, mod_freq, ['ch1', 'ch2', 'ch3', 'ch4'],
                             phases = [0, 90, 180, 270],
                             marker_channels=['ch4_marker1', 'ch4_marker2'])


AWG.start()
f = Qubit_LO.frequency()+mod_freq
MC.set_sweep_function(Dux.in1_out1_phase)
MC.set_detector_function(det.Signal_Hound_fixed_frequency(SH,
                         frequency=f))
MC.set_sweep_points(np.arange(8000, 20000, 100))
MC.run('Duplexer_phase_sweep')
ma.MeasurementAnalysis()

# ad_func_pars = {'adaptive_function': minimize_scalar,
#                 'bracket': [5000, 12000, 15000]}
# MC.set_adaptive_function_parameters(ad_func_pars)
# MC.run(name='adaptive_duplexer_phase_cal', mode='adaptive')

# ma.MeasurementAnalysis()