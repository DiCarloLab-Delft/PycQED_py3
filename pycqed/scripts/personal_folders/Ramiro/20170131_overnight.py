# CZ_amps = np.linspace(1.03, 1.07, 41)
# S_amps = np.linspace(1.04, 1.06, 21)

# AWG.ch4_amp(DataT.SWAP_amp())
# corr_amps = np.arange(.0, .3, 0.01)
# MC.set_sweep_function(AWG.ch3_amp)
# MC.set_sweep_points(CZ_amps)
# MC.set_sweep_function_2D(AWG.ch4_amp)
# MC.set_sweep_points_2D(S_amps)
# d=czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
# MC.set_detector_function(d)
# MC.run('CZ_cost_function', mode='2D')
# ma.MeasurementAnalysis(label='CZ_cost_function', TwoD=True)
# ma.MeasurementAnalysis(label='CZ_cost_function')

AncT.CZ_channel_amp(1.056)
AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())
AncT.CZ_corr_amp(0.)
DataT.SWAP_corr_amp(0.)

for i in range(1000):
    for jj in range(3):
        qs_corr = AncT.CZ_corr_amp()
        qcp_corr = DataT.SWAP_corr_amp()
        amp_2Q = AncT.CZ_channel_amp()
        print('********************************')
        print('CPhase Amp = %.3f Vpp'%amp_2Q)
        print('qS phase corr = %.3f'%qs_corr)
        print('qCP phase corr = %.3f'%qcp_corr)
        print('********************************')

        CZ_amps = np.linspace(-0.005, .005, 11) + amp_2Q
        AWG.ch4_amp(DataT.SWAP_amp())
        corr_amps = np.arange(.0, .3, 0.01)
        MC.set_sweep_function(AWG.ch3_amp)
        MC.set_sweep_points(CZ_amps)
        d=czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
        MC.set_detector_function(d)
        MC.run('CZ_cost_function')
        ma.MeasurementAnalysis(label='CZ_cost_function')
        amp_2Q = fix_phase_2Q()[0]
        AncT.CZ_channel_amp(amp_2Q)
        AWG.ch3_amp(AncT.CZ_channel_amp())
        print('********************************')
        print('CPhase Amp = %.3f Vpp'%amp_2Q)
        print('qS phase corr = %.3f'%qs_corr)z
        print('qCP phase corr = %.3f'%qcp_corr)
        print('********************************')

        mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                            np.linspace(0, .13, 60), sweep_qubit='DataT', excitations=0)
        qs_corr = fix_phase_qs()[0]
        DataT.SWAP_corr_amp(qs_corr)
        print('********************************')
        print('CPhase Amp = %.3f Vpp'%amp_2Q)
        print('qS phase corr = %.3f'%qs_corr)
        print('qCP phase corr = %.3f'%qcp_corr)
        print('********************************')
        mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                            np.linspace(0, .18, 60), sweep_qubit='AncT', excitations=0)
        qcp_corr = fix_phase_qcp()[0]
        AncT.CZ_corr_amp(qcp_corr)
        print('********************************')
        print('CPhase Amp = %.3f Vpp'%amp_2Q)
        print('qS phase corr = %.3f'%qs_corr)
        print('qCP phase corr = %.3f'%qcp_corr)
        print('********************************')


    for kk in range(5):
        mq_mod.tomo2Q_bell(bell_state=0, device=S5, qS_name='DataT', qCZ_name='AncT',
                           nr_shots=1024, nr_rep=10)
        tomo.analyse_tomo(MLE=False, target_bell=0)