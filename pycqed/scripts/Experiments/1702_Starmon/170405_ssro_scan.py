

from pycqed.measurement import detector_functions as det
from pycqed.scripts.Experiments.intel_demo import qasm_helpers as qh
from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sqqs



detected_function = QR.measure_ssro

d = det.Function_Detector_list(
    detected_function, msmt_kw={'nr_shots': 4000, 'MC': nested_MC},
    result_keys=['F_a', 'F_d'], value_units=['', ''])
# MC.set_sweep_function(QR.RO_pulse_length)

MC.set_sweep_function(QR.f_RO)
MC.set_sweep_function_2D(QR.RO_amp)
f_RO = 7.19156e9
freqs = np.linspace(f_RO-5e6, f_RO+5e6, 21)
MC.set_sweep_points(freqs)
MC.set_sweep_points_2D(np.linspace(1, 0.4, 11))
MC.set_detector_function(d)
MC.run('RO_freq_amp', mode='2D')

