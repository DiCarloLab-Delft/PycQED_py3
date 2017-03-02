params_QR = {'E_c': 0.28,
 'asymmetry': 0,
 'dac_flux_coefficient': 0.0015562384838092452,
 'dac_sweet_spot': -16.592195531988864,
 'f_max': 6.0892992727170974}

# Arches parameters for all the qubits
from pycqed.analysis import fitting_models as fit_mods
f_dac_QR = lambda d: fit_mods.Qubit_dac_to_freq(dac_voltage=d, **params_QR)*1e9


params_QL = {'E_c': 0.28,
 'asymmetry': 0,
 'dac_flux_coefficient': 0.00044891020060318243,
 'dac_sweet_spot': 75,
 'f_max': 4.683295911524479}


# Arches parameters for all the qubits
from pycqed.analysis import fitting_models as fit_mods
f_dac_QL = lambda d: fit_mods.Qubit_dac_to_freq(dac_voltage=d, **params_QL)*1e9



integration_length=2e-6
chi=2e6
pulse_length=500e-9

q = QL
q.RO_pulse_length()
q.RO_amp(0.2)

LutMan0
switch_to_IQ_mod_RO_UHFQC(q)
q.RO_acq_weight_function_I(0)
q.RO_acq_weight_function_Q(1)
q.RO_acq_integration_length(integration_length)

LutMan0.M_modulation(q.f_RO_mod())
LutMan0.M0_modulation(LutMan0.M_modulation()+chi)
LutMan0.M1_modulation(LutMan0.M_modulation()-chi)
LutMan0.Q_modulation(0)
LutMan0.Q_amp180(0.8/2)
LutMan0.M_up_amp(0.2)
LutMan0.Q_gauss_width(100e-9)
LutMan0.Q_motzoi_parameter(0)
LutMan0.M_amp(0.2)
LutMan0.M_down_amp0(0.0/2)
LutMan0.M_down_amp1(0.0/2)
LutMan0.M_length(pulse_length-100e-9)
LutMan0.M_up_length(100.0e-9)
LutMan0.M_down_length(1e-9)
LutMan0.Q_gauss_nr_sigma(5)
LutMan0.acquisition_delay(270e-9)

multiplexed_wave = [['LutMan0', 'M_up_mid_double_dep'],
                    ]

LutManMan.generate_multiplexed_pulse(multiplexed_wave)
# LutManMan.render_wave('Multiplexed_pulse', time_unit='s')
# LutManMan.render_wave_PSD(
#     'Multiplexed_pulse', f_bounds=[00e6, 1000e6], y_bounds=[1e-18, 1e-6])
LutManMan.load_pulse_onto_AWG_lookuptable('Multiplexed_pulse')

# Enabling the AWG channel on the UHFQC
UHFQC_1.awgs_0_enable(1)

RF.off()


fluxes_a = np.linspace(-400, 400, 11)
fluxes_b = np.linspace(-600, 600, 61)
fluxes_c = np.linspace(-600, 600, 261)

for fluxes in [fluxes_a, fluxes_b, fluxes_c]:
    for flux in fluxes:
        MC.soft_avg(4)
        IVVI.dac2(flux)
        QL.f_qubit(f_dac_QL(IVVI.dac2()))
        print('Estimated qubit freq to {:.4g}'.format(QL.f_qubit()))
        times = np.arange(0, .1e-6, 1e-9)
        artificial_detuning = 2/times[-1]

        for i in range(2):
            QL.measure_ramsey(times, artificial_detuning=artificial_detuning)
            a = ma.Ramsey_Analysis(auto=True, close_fig=True)
            fitted_freq = a.fit_res.params['frequency'].value
            detuning = fitted_freq-artificial_detuning
            QL.f_qubit(QL.f_qubit()-detuning)

        QL.find_frequency(method='Ramsey')
        MC.soft_avg(8)
        # times = np.arange(0, 60e-6, .8e-6)
        times = np.concatenate([np.arange(0, 5.01e-6, 200e-9),
                            np.arange(5e-6, 60e-6, .8e-6)])
        QL.measure_echo(times=times, artificial_detuning=4/times[-1])
        QL.measure_ramsey(times=times, artificial_detuning=4/times[-1])
        # times = np.arange(0, 60e-6, 1e-6)
        QL.measure_T1(times)


