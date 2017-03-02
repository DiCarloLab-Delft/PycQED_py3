
# this is absed on the f_dac_QL function defined in 170301_5014_dac arc.
# this should be in the qubit object

QL.f_qubit(f_dac_QL(IVVI.dac2()))
QR.f_qubit(f_dac_QR(IVVI.dac1()))

fluxes = np.linspace(-1000, 1000, 11)
for flux in fluxes:
    MC.soft_avg(4)
    IVVI.dac2(flux)

    QR.f_qubit(f_dac_QR(IVVI.dac1()))
    print('Estimated qubit freq to {:.4g}'.format(QR.f_qubit()))
    times = np.arange(0, .1e-6, 1e-9)
    artificial_detuning = 2/times[-1]

    for i in range(2):
        QR.measure_ramsey(times, artificial_detuning=artificial_detuning)
        a = ma.Ramsey_Analysis(auto=True, close_fig=True)
        fitted_freq = a.fit_res.params['frequency'].value
        detuning = fitted_freq-artificial_detuning
        QR.f_qubit(QR.f_qubit()-detuning)

    QR.find_frequency(method='Ramsey')
    MC.soft_avg(8)
    # times = np.arange(0, 60e-6, .8e-6)
    times = np.concatenate([np.arange(0, 5.01e-6, 200e-9),
                        np.arange(5e-6, 60e-6, .8e-6)])
    QR.measure_echo(times=times, artificial_detuning=4/times[-1])
    QR.measure_ramsey(times=times, artificial_detuning=4/times[-1])
    # times = np.arange(0, 60e-6, 1e-6)
    QR.measure_T1(times)

