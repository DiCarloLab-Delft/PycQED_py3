
AncT.CZ_channel_amp(1.056)
DataT.SWAP_amp(1.055)
AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())
AncT.CZ_corr_amp(0.)
DataT.SWAP_corr_amp(0.)


def print_CZ_pars(AncT, DataT):
    print('********************************')
    print('CPhase Amp = {:.3f} Vpp'.format(AncT.CZ_channel_amp()))
    print('qS phase corr = {:.3f}'.format(DataT.SWAP_corr_amp()))
    print('qCP phase corr = {:.3f}'.format(AncT.CZ_corr_amp()))
    print('********************************')

for i in range(1000):
    # print state before new round of cals
    print_CZ_pars(AncT, DataT)
    # starts a 2Q calibration
    CZ_amps = np.linspace(-0.005, .005, 11) + AncT.CZ_channel_amp()
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
        print_CZ_pars(AncT, DataT)

        mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                                    np.linspace(0, .13, 60),
                                    sweep_qubit='DataT', excitations=0)
        DataT.SWAP_corr_amp(fix_phase_qs()[0])
        print_CZ_pars(AncT, DataT)
        mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                                    np.linspace(0, .18, 60),
                                    sweep_qubit='AncT', excitations=0)
        AncT.CZ_corr_amp(fix_phase_qcp()[0])
        print_CZ_pars(AncT, DataT)

    for kk in range(2):
        for target_bell in [0, 10, 20]:
            MC.soft_avg(1)  # To be removed later, should not be needed
            mq_mod.tomo2Q_bell(bell_state=target_bell, device=S5,
                               qS_name='DataT', qCZ_name='AncT',
                               nr_shots=1024, nr_rep=10)
            tomo.analyse_tomo(MLE=False, target_bell=target_bell % 10)
