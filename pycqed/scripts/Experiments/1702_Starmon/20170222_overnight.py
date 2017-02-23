power_vec = np.arange(-60,-9,-5)

for p in power_vec:
    qEast.RO_power_cw(p)
    qWest.RO_power_cw(p)

    freqs = np.arange(7.18e9,7.195e9,.1e6)
    qEast.find_resonator_frequency(freqs=freqs)

    freqs = np.arange(7.075e9,7.095e9,.1e6)
    qWest.find_resonator_frequency(freqs=freqs)

dacs = np.arange(-1000,1001,25)

freqs = np.arange(7.075e9,7.095e9,.1e6)
qWest.measure_resonator_dac(freqs, dacs)
IVVI.dac1(0.)
IVVI.dac2(0.)

freqs = np.arange(7.18e9,7.195e9,.1e6)
qEast.measure_resonator_dac(freqs, dacs)
IVVI.dac1(0.)
IVVI.dac2(0.)