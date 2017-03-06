
QR.find_resonator_frequency()
QR.find_frequency(pulsed=True)

QR_LO.frequency(4.683e9)
QR_LO.power(-10)
QR_LO.on()
QR_LO.pulsemod_state('off')


QR.cw_source.on()
freqs = np.arange(4.9e9, 5.e9, 1e6)
dac_voltages= np.arange(520, 540, .5)

MC.set_sweep_function(QR.cw_source.frequency)
MC.set_sweep_points(freqs)
MC.set_sweep_function_2D(FC.flux0)
MC.set_sweep_points_2D(dac_voltages)
MC.set_detector_function(
    det.Heterodyne_probe(QR.heterodyne_instr, trigger_separation=2.8e-6))
MC.run('QR-avoided crossing', mode='2D')
QR.cw_source.off()
