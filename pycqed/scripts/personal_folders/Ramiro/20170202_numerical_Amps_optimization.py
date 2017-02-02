reload_mod_stuff()
# cphases
from pycqed.measurement.optimization import nelder_mead
from pycqed.scripts.Experiments.Five_Qubits import CZ_tuneup as czt

opt_init_CZ_amp = 1.04
opt_init_SWAP_amp = 1.045
corr_amps = np.linspace(.05, .25, 30)

MC.set_sweep_function(AWG.ch3_amp)
MC.set_sweep_function_2D(AWG.ch4_amp)
d = czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
MC.set_detector_function(d)

ad_func_pars = {'adaptive_function': nelder_mead,
                'x0': [opt_init_CZ_amp, opt_init_SWAP_amp],
                'initial_step': [0.1, 0.03], 'minimize': False}
MC.set_adaptive_function_parameters(ad_func_pars)
MC.run(name='CZ_cost_function_optimization', mode='adaptive')

ma_obj = ma.OptimizationAnalysis(label='CZ_cost_function_optimization')
ma.OptimizationAnalysis_v2(label='CZ_cost_function_optimization')
opt_CZ_amp, opt_SWAP_amp = ma_obj.optimization_result[0]

# setting optimized values
AncT.CZ_channel_amp(opt_CZ_amp)
DataT.SWAP_amp(opt_SWAP_amp)
AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())

# 1Q calibrations
for jj in range(2):
    mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                                np.linspace(0., -0.25, 60),
                                # need 64 to be same as tomo seq,
                                sweep_qubit='DataT', excitations=0)
    DataT.SWAP_corr_amp(fix_phase_qs()[0])
    print_CZ_pars(AncT, DataT)
    mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                                np.linspace(0., -0.25, 60),
                                # need 64 to be same as tomo seq,
                                sweep_qubit='AncT', excitations=0)
    AncT.CZ_corr_amp(fix_phase_qcp()[0])
    print_CZ_pars(AncT, DataT)

MC.soft_avg(1)  # To be removed later, should not be needed
for target_bell in [0, 10, 20, 1, 2, 3]:
    mq_mod.tomo2Q_bell(bell_state=target_bell, device=S5,
                       qS_name='DataT', qCZ_name='AncT',
                       nr_shots=512, nr_rep=1)
    tomo.analyse_tomo(MLE=False, target_bell=target_bell % 10)