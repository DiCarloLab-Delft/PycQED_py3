exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())
CBox.nr_averages(2048)
set_trigger_slow()
qubit = VIP_mon_2_dux
calibrate_duplexer_phase_2D(pulse_pars)
qubit.find_pulse_amplitude(amps=np.linspace(-.3, .3, 31),
                             max_n=1, update=True,
                             take_fit_I=False)
qubit.find_frequency(method='ramsey', steps=[5,30,100, 300], update=True)
qubit.measure_motoi_XY(motzois=np.linspace(-0.25, -0.15, 21))
qubit.find_pulse_amplitude(amps=np.linspace(-.3, .3, 31),
                               N_steps=[3, 7, 19], max_n=100, take_fit_I=False)
qubit.find_amp90_scaling(N_steps=[5,9],max_n=100,
                              take_fit_I=False)
qubit.measure_T1(np.linspace(0, 120e-6, 41))
a = ma.T1_Analysis(label='T1', auto=True)
T1 = a.T1
CBox.nr_averages(4096)
pulse_pars, RO_pars=VIP_mon_2_dux.get_pulse_pars()
measure_allXY(pulse_pars=pulse_pars, RO_pars=RO_pars)
#measure_RB(pulse_pars=pulse_pars, RO_pars=RO_pars, T1=T1)