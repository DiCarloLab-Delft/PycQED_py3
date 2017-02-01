# decays
for ii in range(100):
    AncT.measure_T1(times=np.linspace(0, 40e-6, 60))
    DataT.measure_T1(times=np.linspace(0, 40e-6, 60))
    AncT.measure_ramsey(times=np.linspace(0, 20e-6, 60), artificial_detuning=4./20e-6)
    DataT.measure_ramsey(times=np.linspace(0, 20e-6, 60), artificial_detuning=4./20e-6)
    AncT.measure_echo(times=np.linspace(0, 40e-6, 60), artificial_detuning=4./40e-6)
    DataT.measure_echo(times=np.linspace(0, 40e-6, 60), artificial_detuning=4./40e-6)
# still missing the bus T1


# fixed the min looking function, want to try it
AncT.CZ_channel_amp(1.056)
DataT.SWAP_amp(1.055)
AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())
AncT.CZ_corr_amp(0.)
DataT.SWAP_corr_amp(0.)
AncT.CZ_channel_amp(1.056)
DataT.SWAP_amp(1.055)
AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())
AncT.CZ_corr_amp(0.)
DataT.SWAP_corr_amp(0.)

def print_CZ_pars(AncT, DataT):
    print('********************************')
    print('CZ Amp = {:.3f} Vpp'.format(AncT.CZ_channel_amp()))
    print('SWAP Amp = {:.3f} Vpp'.format(DataT.SWAP_amp()))
    print('qS phase corr = {:.3f}'.format(DataT.SWAP_corr_amp()))
    print('qZ phase corr = {:.3f}'.format(AncT.CZ_corr_amp()))
    print('********************************')

for i in range(1000):
    # print state before new round of cals
    print_CZ_pars(AncT, DataT)
    # starts a 2Q calibration
    CZ_amps = np.linspace(-0.01, .01, 11) + AncT.CZ_channel_amp()
    AWG.ch4_amp(DataT.SWAP_amp())
    corr_amps = np.arange(.0, .3, 0.01)
    MC.set_sweep_function(AWG.ch3_amp)
    MC.set_sweep_points(CZ_amps)
    d = czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
    MC.set_detector_function(d)
    MC.run('CZ_cost_function')
    ma.MeasurementAnalysis(label='CZ_cost_function')
    AncT.CZ_channel_amp(fix_phase_2Q()[0])
    AWG.ch3_amp(AncT.CZ_channel_amp())

    for jj in range(2):
        # prints state after 2Q_cal
        mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                                    np.linspace(0.0, .14, 60),
                                    # need 64 to be same as tomo seq,
                                    sweep_qubit='DataT', excitations=0)
        DataT.SWAP_corr_amp(fix_phase_qs()[0])
        print_CZ_pars(AncT, DataT)
        mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                                    np.linspace(0.1, .3, 60),
                                    # need 64 to be same as tomo seq,
                                    sweep_qubit='AncT', excitations=0)
        AncT.CZ_corr_amp(fix_phase_qcp()[0])
        print_CZ_pars(AncT, DataT)

    for kk in range(2):
        MC.soft_avg(1)  # To be removed later, should not be needed
        for target_bell in [0, 10, 20]:
            mq_mod.tomo2Q_bell(bell_state=target_bell, device=S5,
                               qS_name='DataT', qCZ_name='AncT',
                               nr_shots=512, nr_rep=1)
            tomo.analyse_tomo(MLE=False, target_bell=target_bell % 10)

# nice functions
def fix_phase_qcp():
    label = 'SWAP_CP_SWAP'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[:-4]
    cp_acq_weight = 0
    y = a.measured_values[cp_acq_weight, :-4]
    return a_tools.find_min(x, y, )


def fix_phase_qs():
    label = 'SWAP_CP_SWAP'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[:-4]
    qs_acq_weigth = 1
    y = a.measured_values[qs_acq_weigth, :-4]
    return a_tools.find_min(x, y, )


def fix_phase_2Q():
    label = 'CZ_cost_function'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[:]
    cp_acq_weight = 0
    y = a.measured_values[cp_acq_weight, :]
    return x[np.argmax(y)], y[np.argmax(y)]





################################
from pycqed.measurement.optimization import nelder_mead

MC.set_sweep_function(AWG.ch3_amp)
MC.set_sweep_function_2D(AWG.ch4_amp)
d = czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
MC.set_detector_function(d)

ad_func_pars = {'adaptive_function': nelder_mead,
                'x0': [AncT.CZ_channel_amp(), DataT.SWAP_amp()],
                'initial_step': 0.1, 'minimize': False}
MC.set_adaptive_function_parameters(ad_func_pars)
MC.run(name='CZ_cost_function_optimization', mode='adaptive')


ma.OptimizationAnalysis(label='CZ_cost_function_optimization')
ma.OptimizationAnalysis_v2(label='CZ_cost_function_optimization')
