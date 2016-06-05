exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())
qubit = VIP_mon_2_dux
pulse_pars, RO_pars=VIP_mon_2_dux.get_pulse_pars()
calibrate_duplexer_phase_2D(pulse_pars)
set_trigger_slow()
qubit = VIP_mon_2_dux
nr_cliffords = [2, 4, 8, 16, 30, 60, 100, 200, 300, 400, 600, 800, 1200]
# qubit.measure_ssro()
CBox.nr_averages(2048)
qubit.measure_T1(np.linspace(0, 120e-6, 41))
a = ma.T1_Analysis(label='T1', auto=True)
T1 = a.T1
qubit.find_pulse_amplitude(amps=np.linspace(-.5, .5, 31),
                             max_n=1, update=True,
                             take_fit_I=True)
qubit.find_frequency(method='ramsey', steps=[30,100, 300], update=True)
qubit.measure_motoi_XY(motzois=np.linspace(-0.4, -0.2, 21))
qubit.find_pulse_amplitude(amps=np.linspace(-.5, .5, 31),
                             N_steps=[3, 7, 19], max_n=100, take_fit_I=True)
CBox.nr_averages(4096)
qubit.measure_allxy()
qubit.measure_randomized_benchmarking(nr_cliffords=nr_cliffords)
ma.RandomizedBenchmarking_Analysis(close_main_fig=False, T1=T1,
                                    pulse_delay=qubit.pulse_delay.get(),
                                    label='RB')