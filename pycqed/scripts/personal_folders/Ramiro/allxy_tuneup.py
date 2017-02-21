qubit_list = [q0, q1]

for qubit in qubit_list:
    qubit.calibrate_mixer_offsets(SH)
    qubit.RO_acq_averages(2**9)
    qubit.find_frequency(method='ramsey', update=True)
    qubit.RO_acq_averages(2**10)
    qubit.find_pulse_amplitude(amps=np.linspace(-1.0, 1.0, 21),
                                    N_steps=[3,7], max_n=100, take_fit_I=True)
    qubit.measure_motzoi_XY(motzois=np.linspace(-0.2, 0.2, 21))
    qubit.find_pulse_amplitude(amps=np.linspace(-1.0, 1.0, 21),
                               N_steps=[3, 7, 19], max_n=100, take_fit_I=True)
    qubit.find_amp90_scaling(scales=0.5,N_steps=[5, 9], max_n=100,
                             take_fit_I=True)