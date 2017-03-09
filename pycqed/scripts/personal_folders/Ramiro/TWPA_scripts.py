# try to see a transmission experiment
HS.RF = RF
CBox.nr_averages(2**15)
TWPA_Pump.frequency(8.19e9)
TWPA_Pump.power(4)
TWPA_Pump.off()
RF.power(-20)
HS.frequency(7.3e9)
RF.on()
MC.set_sweep_function(pw.wrap_par_to_swf(HS.frequency))
MC.set_sweep_points(np.arange(7e9,8e9,5e6))
MC.set_detector_function(det.Heterodyne_probe(HS))
MC.run(name='TWPA_Pump_line_off')



freqs = np.linspace(7.078e9, 7.093e9, 21)
powers = np.arange(-20, -60.1, -6)
print('Expected duration {}s'.format(len(freqs)*len(powers)*(120+21)/3600))



TWPA_Pump.frequency(8.19e9)
TWPA_Pump.power(4)
for i in range(4):
    TWPA_Pump.on()
    MC.set_sweep_function(HS.frequency)
    MC.set_sweep_function_2D(RF.power)
    MC.set_sweep_points(freqs)
    MC.set_sweep_points_2D(powers)
    MC.set_detector_function(ssrodet)
    MC.run('2D-RO_TWPA_ON', mode='2D')

    TWPA_Pump.off()
    ssrodet = AncB.measure_ssro(return_detector=True, MC=nested_MC)
    MC.set_sweep_function(HS.frequency)
    MC.set_sweep_function_2D(RF.power)
    MC.set_sweep_points(freqs)
    MC.set_sweep_points_2D(powers)
    MC.set_detector_function(ssrodet)
    MC.run('2D-RO_TWPA_OFF', mode='2D')

# dispersive feature scan
TWPA_Pump.on()
TWPA_Pump.power(-30)
HS.RF = TWPA_Pump
MC.set_sweep_function(pw.wrap_par_to_swf(HS.frequency))
MC.set_sweep_points(np.arange(8.1e9, 8.4e9, 1.e6))
MC.set_detector_function(det.Heterodyne_probe(HS))
MC.run(name='TWPA_Pump_line')

# 2D scan
TWPA_Pump.on()
TWPA_Pump.power(-0)
HS.RF = TWPA_Pump
MC.set_sweep_function(pw.wrap_par_to_swf(HS.frequency))
MC.set_sweep_function_2D(pw.wrap_par_to_swf(TWPA_Pump.power))
MC.set_sweep_points(np.arange(8.1e9, 8.4e9, 1.e6))
MC.set_sweep_points_2D(np.arange(-10, 0.5, 1.))
MC.set_detector_function(det.Heterodyne_probe(HS))
MC.run(name='TWPA_Pump_line', mode='2D')