from pycqed.instrument_drivers.meta_instrument import device_object as do

Starmon = do.DeviceObject('Starmon')
station.add_component(Starmon)
Starmon.add_qubits([QL, QR])

Starmon.Buffer_Flux_Flux(10e-9)
Starmon.Buffer_Flux_MW(40e-9)
Starmon.Buffer_MW_MW(20e-9)
Starmon.Buffer_MW_Flux(10e-9)
station.sequencer_config = Starmon.get_operation_dict()['sequencer_config']



# This should not be needed
QR.find_resonator_frequency()

# this makes sure the qubit is there and the pulsed spec is prepared
QR.find_frequency(pulsed=True)



QR.cw_source.off()

# we have calibrated the pi-pulse for QL

mqts.avoided_crossing_spec_seq(Starmon.get_operation_dict(), 'QL', 'QR',
                               RO_target='QR')

freqs = np.arange(4.8e9, 5.1e9, .3e6)
dac_voltages= np.arange(505, 540, .5)

QL.td_source.on()

MC.set_sweep_function(QR.cw_source.frequency)
MC.set_sweep_points(freqs)
MC.set_sweep_function_2D(FC.flux0)
MC.set_sweep_points_2D(dac_voltages)
MC.set_detector_function(
    det.Heterodyne_probe(QR.heterodyne_instr, trigger_separation=2.8e-6))
MC.run('QR-avoided crossing', mode='2D')
