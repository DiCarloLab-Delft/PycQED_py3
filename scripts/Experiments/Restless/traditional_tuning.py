qubit = VIP_mon_2_dux
nr_cliffords = [2, 4, 8, 16, 30, 60, 100, 200, 300, 400, 600, 800, 1200]
# set_trigger_slow()

qubit.measure_ssro()
qubit.measure_T1(np.linspace(0, 80e-6, 41))
qubit.find_pulse_amplitude(amps=np.linspace(-.5, .5, 31),
                           max_n=1, update=True,
                           take_fit_I=False)
qubit.find_frequency(method='ramsey', steps=[3, 10, 30, 100, 300], update=True)
qubit.find_frequency(method='ramsey', steps=[100, 300], update=True)

qubit.measure_motoi_XY(motzois=np.linspace(-.3, -.1, 21))
qubit.find_pulse_amplitude(amps=np.linspace(-.5, .5, 31),
                           N_steps=[3, 7, 19], max_n=100, take_fit_I=False)
qubit.measure_allxy()
qubit.measure_randomized_benchmarking(nr_cliffords=nr_cliffords)
ma.RandomizedBenchmarking_Analysis(close_main_fig=False, T1=22e-6,
                                   pulse_delay=qubit.pulse_delay.get(),
                                   label='RB')
