#

freqs = np.arange(4.46e9, 4.72e9, 1e6)
anharm = 300e6

QL_LO.on()
QR_LO.on()
QR_LO.power(-10)
QL_LO.power(-10)

freqs2 = np.linspace(freqs[0]-anharm, freqs[-1]-anharm, 11)

MC.set_sweep_function(QL_LO.frequency)
MC.set_sweep_function_2D(QR_LO.frequency)
MC.set_sweep_points(freqs)
MC.set_sweep_points_2D(freqs2)
MC.set_detector_function(
    det.Heterodyne_probe(QL.heterodyne_instr, trigger_separation=2.8e-6))
MC.run('three-tone-spec'+QL.msmt_suffix, mode='2D')
