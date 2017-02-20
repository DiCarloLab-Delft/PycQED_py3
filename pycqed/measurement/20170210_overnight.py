#######################
# prepares to start
#######################
AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())
AncT.CZ_corr_amp(0.)
DataT.SWAP_corr_amp(0.)

CZ_amp_vec = []
SWAP_amp_vec = []
rSWAP_amp_vec = []

for i in range(3):
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
    CZ_amp_vec.append(AncT.CZ_channel_amp())
    print_CZ_pars(AncT, DataT)

    # starts a SWAP calibration
    SWAP_amps = np.linspace(-0.01, .01, 11) + DataT.SWAP_amp()
    corr_amps = np.arange(.0, .3, 0.01)
    MC.set_sweep_function(AWG.ch4_amp)
    MC.set_sweep_points(SWAP_amps)
    d = czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
    MC.set_detector_function(d)
    MC.run('SWAP_cost_function')
    ma.MeasurementAnalysis(label='SWAP_cost_function')
    DataT.SWAP_amp(fix_swap_amp()[0])
    AWG.ch4_amp(DataT.SWAP_amp())
    SWAP_amp_vec.append(DataT.SWAP_amp())
    print_CZ_pars(AncT, DataT)

    # starts a rSWAP calibration
    mq_mod.rSWAP_scan(S5, 'DataT', 'AncT', recovery_swap_amps=np.linspace(0.435, 0.465, 60))
    ma.MeasurementAnalysis(label='rSWAP', auto=True)
    DataT.rSWAP_pulse_amp(get_rSWAP_amp()[0])
    rSWAP_amp_vec.append(DataT.rSWAP_pulse_amp())
    print_CZ_pars(AncT, DataT)

AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())
AncT.CZ_corr_amp(0.)
DataT.SWAP_corr_amp(0.)

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
for iii in range(1):
    for target_bell in [0]:
        mq_mod.tomo2Q_bell(bell_state=target_bell, device=S5,
                           qS_name='DataT', qCZ_name='AncT',
                           nr_shots=1024, nr_rep=1)
        tomo.analyse_tomo(MLE=False, target_bell=target_bell % 10)

reload_mod_stuff()
for cardinal_state in range(36):
    mq_mod.tomo2Q_cphase_cardinal(cardinal_state=cardinal_state, device=S5,
                       qS_name='DataT', qCZ_name='AncT',
                       nr_shots=1024, nr_rep=10)
    ma.Tomo_Multiplexed(auto=True, MLE=True, target_bell=0)
for i in range(100):
    #######################
    # prepares to start
    #######################
    AWG.ch3_amp(AncT.CZ_channel_amp())
    AWG.ch4_amp(DataT.SWAP_amp())
    AncT.CZ_corr_amp(0.)
    DataT.SWAP_corr_amp(0.)

    CZ_amp_vec = []
    SWAP_amp_vec = []
    rSWAP_amp_vec = []

    for i in range(3):
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
        CZ_amp_vec.append(AncT.CZ_channel_amp())
        print_CZ_pars(AncT, DataT)

        # starts a SWAP calibration
        SWAP_amps = np.linspace(-0.01, .01, 11) + DataT.SWAP_amp()
        corr_amps = np.arange(.0, .3, 0.01)
        MC.set_sweep_function(AWG.ch4_amp)
        MC.set_sweep_points(SWAP_amps)
        d = czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
        MC.set_detector_function(d)
        MC.run('SWAP_cost_function')
        ma.MeasurementAnalysis(label='SWAP_cost_function')
        DataT.SWAP_amp(fix_swap_amp()[0])
        AWG.ch4_amp(DataT.SWAP_amp())
        SWAP_amp_vec.append(DataT.SWAP_amp())
        print_CZ_pars(AncT, DataT)

        # starts a rSWAP calibration
        mq_mod.rSWAP_scan(S5, 'DataT', 'AncT', recovery_swap_amps=np.linspace(0.435, 0.465, 60))
        ma.MeasurementAnalysis(label='rSWAP', auto=True)
        DataT.rSWAP_pulse_amp(get_rSWAP_amp()[0])
        rSWAP_amp_vec.append(DataT.rSWAP_pulse_amp())
        print_CZ_pars(AncT, DataT)

    AWG.ch3_amp(AncT.CZ_channel_amp())
    AWG.ch4_amp(DataT.SWAP_amp())
    AncT.CZ_corr_amp(0.)
    DataT.SWAP_corr_amp(0.)

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
    for iii in range(1):
        for target_bell in [0]:
            mq_mod.tomo2Q_bell(bell_state=target_bell, device=S5,
                               qS_name='DataT', qCZ_name='AncT',
                               nr_shots=1024, nr_rep=1)
            tomo.analyse_tomo(MLE=False, target_bell=target_bell % 10)

    reload_mod_stuff()
    for cardinal_state in range(36):
        mq_mod.tomo2Q_cphase_cardinal(cardinal_state=cardinal_state, device=S5,
                           qS_name='DataT', qCZ_name='AncT',
                           nr_shots=1024, nr_rep=1)
        ma.Tomo_Multiplexed(auto=True, MLE=True, target_bell=0)