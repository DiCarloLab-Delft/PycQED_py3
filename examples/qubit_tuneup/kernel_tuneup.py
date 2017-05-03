import qcodes as qc
station = qc.station
k0 = qc.components['k0']
k1 = qc.components['k1']
MC = station.MC


DataT.fluxing_channel(4)
AncT.fluxing_channel(3)


k0.channel(4)
k0.kernel_dir_path(
    r'D:\GitHubRepos\iPython-Notebooks\Experiments\1607_Qcodes_5qubit\kernels')
k0.kernel_list(['precompiled_RT_20161206.txt'])

k1.channel(3)
k1.kernel_dir_path(
    r'D:\GitHubRepos\iPython-Notebooks\Experiments\1607_Qcodes_5qubit\kernels')
k1.kernel_list(['precompiled_AncT_RT_20161203.txt',
                'kernel_fridge_lowpass_20161024_1.00.txt',
                'kernel_skineffect_0.7.txt',
                'kernel_fridge_slow1_20161203_15_-0.013.txt'])

k1.bounce_tau_1(16)
k1.bounce_amp_1(-0.03)

k1.bounce_tau_2(1)
k1.bounce_amp_2(-0.04)

dist_dict = {'ch_list': ['ch4', 'ch3'],
             'ch4': k0.kernel(),
             'ch3': k1.kernel()}

DataT.dist_dict(dist_dict)
AncT.dist_dict(dist_dict)

###########################################
# Simple chevron DataT
# Conventional SWAP chevron
###########################################
# S5 is the device type object
mq_mod.measure_chevron(S5, 'DataT', amps=np.arange(1.0,1.1,0.005),
                           length=np.arange(0, 120e-9, 2e-9),
                           MC=station.MC)

###########################################
#   SWAP-N
###########################################
mq_mod.measure_SWAPN(S5, 'DataT', swap_amps=np.arange(1.05, 1.06, 0.001))

###########################################
# 2 exciation chevron to test if initial kernels worked
###########################################

AncT.SWAP_amp(1.48)
# 2-excitation chevron
MC.live_plot_enabled(True)

# DataT.dist_dict(dist_dict)
AncT.RO_acq_averages(256)

DataT.fluxing_channel(4)
AncT.fluxing_channel(3)
int_avg_det = det.UHFQC_integrated_average_detector(
    UHFQC=qubit._acquisition_instr, AWG=qubit.AWG,
    channels=[DataT.RO_acq_weight_function_I(),
              AncT.RO_acq_weight_function_I()], nr_averages=AncT.RO_acq_averages(),
    integration_length=AncT.RO_acq_integration_length(), cross_talk_suppression=True)
# reload(fsqs)
# reload(awg_swf)
# fsqs.station=station
mw_pulse_pars_DataT, RO_pars_DataT = DataT.get_pulse_pars()
mw_pulse_pars_AncT, RO_pars_AncT = AncT.get_pulse_pars()
flux_pulse_pars_DataT = DataT.get_flux_pars()
flux_pulse_pars_AncT = AncT.get_flux_pars()

chevron_pulse_lengths = np.arange(0, 120e-9, 2e-9)
AWG.ch3_amp(AncT.SWAP_amp())
AWG.ch4_amp(DataT.SWAP_amp())

s = awg_swf.chevron_with_excited_bus_2Qubits(mw_pulse_pars_AncT, mw_pulse_pars_DataT,
                                             flux_pulse_pars_AncT, flux_pulse_pars_DataT, RO_pars_AncT,
                                             dist_dict=dist_dict, AWG=AWG, excitations=1)
s.upload = True

MC.set_detector_function(int_avg_det)
MC.set_sweep_function(s)
MC.set_sweep_points(chevron_pulse_lengths)
MC.set_sweep_function_2D(AWG.ch3_amp)
MC.set_sweep_points_2D(np.linspace(1.1, 1.6, 61))
label = 'chevron_with_excited_bus_2D'
MC.run(label, mode='2D')
ma.MeasurementAnalysis(label=label, transpose=True, TwoD=True)


###################################
## Ram-Z timing calibration
###################################
AncT.RO_acq_averages(1024)
int_avg_det = det.UHFQC_integrated_average_detector(
            UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
            channels=[DataT.RO_acq_weight_function_I(),
                      AncT.RO_acq_weight_function_I()],
            nr_averages=AncT.RO_acq_averages(),
            integration_length=AncT.RO_acq_integration_length(),
            cross_talk_suppression=True)

pulse_dict = AncT.get_operation_dict()

ram_Z_sweep = awg_swf.awg_seq_swf(fsqs.Ram_Z_delay_seq,
    awg_seq_func_kwargs={'pulse_dict':pulse_dict, 'q0':'AncT',
                         'inter_pulse_delay':40e-9,
                         'distortion_dict': dist_dict},
                         parameter_name='times')


MC.set_sweep_function(ram_Z_sweep)
MC.set_sweep_points(np.arange(-100e-9, 200e-9, 5e-9))
MC.set_detector_function(int_avg_det)
MC.run('Ram_Z_AncT')
ma.MeasurementAnalysis()