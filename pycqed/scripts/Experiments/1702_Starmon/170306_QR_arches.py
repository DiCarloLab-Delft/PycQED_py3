
fluxes_a = np.concatenate([np.linspace(0, 400, 5), np.linspace(0, -400, 5)])
fluxes_b = np.linspace(-600, 600, 61)
fluxes_c = np.linspace(-600, 600, 261)

for fluxes in [fluxes_a, fluxes_b, fluxes_c]:
    for flux in fluxes:
        MC.soft_avg(4)
        FC.flux0(flux)
        QR.f_qubit(QR.calculate_frequency())
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
        MC.soft_avg(5)
        # times = np.arange(0, 60e-6, .8e-6)
        times = np.concatenate([np.arange(0, 5.01e-6, 200e-9),
                            np.arange(5e-6, 60e-6, .8e-6)])
        QR.measure_echo(times=times, artificial_detuning=4/times[-1])
        QR.measure_ramsey(times=times, artificial_detuning=4/times[-1])
        # times = np.arange(0, 60e-6, 1e-6)
        QR.measure_T1(times)

