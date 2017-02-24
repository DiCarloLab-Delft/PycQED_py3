

## This comes from the qubit object
HS_sweep = swf.Heterodyne_Frequency_Sweep(
                                    RO_pulse_type=QR.RO_pulse_type(),
                                    RF_source=QR.find_instrument(QR.RF_RO_source()),
                                    LO_source=QR.LO,IF=QR.f_RO_mod())
QR.f_res(7.1912e9)
IVVI.set_dacs_zero()
for dac_ch in [IVVI.dac1]:
    MC.set_sweep_function(HS_sweep)
    MC.set_sweep_function_2D(dac_ch)
    MC.set_detector_function(QR.int_avg_det_single)
    MC.set_sweep_points(np.arange(QR.f_res()-3e6, QR.f_res()+2e6, 50e3))
    MC.set_sweep_points_2D(np.linspace(-1000, 1000, 41))
    MC.run('Resonator_dac_{}'.format(QR.name),  mode='2D')
    ma.TwoD_Analysis()




LO.frequency(
IVVI.set_dacs_zero()
QL_LO.power()
QL_LO.on()
MC.set_sweep_function(QL_LO.frequency)
MC.set_detector_function(QL.int_avg_det_single)
MC.set_sweep_points(np.arange(4.696e9, 4.71e9, 0.05e6)
MC.run('CW_spec{}'.format(QL.name),  mode='2D')
ma.Me
