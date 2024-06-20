###################################################################
# Spectroscopy experiments
####################################################################
MC.live_plot_enabled(True)
nested_MC.live_plot_enabled(True)
# Apply spectroscopy settings
file_cfg = gc.generate_config(in_filename=input_file,
                          out_filename=config_fn,
                          mw_pulse_duration=20,
                          ro_duration=3000,
                          flux_pulse_duration=40,
                          init_duration=200000)
qubit = X3
qubit.ro_acq_averages(2**8)
qubit.ro_pulse_length(2e-6)
qubit.ro_acq_integration_length(2e-6)
qubit.instr_spec_source(qubit.instr_LO_mw())
qubit.instr_FluxCtrl('AWG8_8320')
qubit.fl_dc_ch('sigouts_3_offset')


# Resonator spectroscocpy
qubit.find_resonator_frequency(update=True,use_min=True)

# Perform spectroscopy experimentsq
qubit.measure_qubit_frequency_dac_scan(freqs=np.arange(-30e6,30e6,0.25e6)+qubit.freq_qubit(),
    dac_values=np.linspace(-10e-3, 10e-3, 11)+qubit.fl_dc_I0(), mode='CW', update=False)
qubit.find_frequency(disable_metadata=True, spec_mode='CW',
                  freqs = np.arange(-50e6, 50e6, 0.25e6)+qubit.freq_qubit())

# # Select frequency detuning based on measured flux arc
# for detuning in np.arange(0, 200e6, 20e6):
#   freq_qubit = 6.055e9-detuning
#   current = np.max((a.dac_fit_res['fit_polynomial']-freq_qubit).roots)
#   fluxcurrent.FBL_X4(np.real(current))
#   qubit.find_frequency(disable_metadata=True, spec_mode='CW', f_step=.1e6)

# Calibrate single qubit gate
# Z3.calibrate_mw_pulse_amplitude_coarse()
S17GBT.Flipping_wrapper(qubit=Z3.name, station=station)
Z3.calibrate_frequency_ramsey(disable_metadata=True)
S17GBT.Motzoi_wrapper(qubit=Z3.name, station=station)
S17GBT.Flipping_wrapper(qubit=Z3.name, station=station)
S17GBT.AllXY_wrapper(qubit=Z3.name, station=station)
S17GBT.SSRO_wrapper(qubit=Z3.name, station=station)
S17GBT.T1_wrapper(qubit=Z3.name, station=station)
S17GBT.T2_wrapper(qubit=Z3.name, station=station)
S17GBT.Randomized_benchmarking_wrapper(qubit=Z3.name, station=station)


# Recover timedomain settings
Z3.ro_pulse_length(250e-9)
Z3.ro_acq_integration_length(500e-9)
Z3.ro_freq(7.599e9)
file_cfg = gc.generate_config(in_filename=input_file,
                          out_filename=config_fn,
                          mw_pulse_duration=20,
                          ro_duration=600,
                          flux_pulse_duration=80,
                          init_duration=200000)
MW_LO_1.frequency(6.200e9)
MW_LO_2.frequency(6.140e9)
MW_LO_3.frequency(5.090e9)
MW_LO_4.frequency(5070000000.0-30e6)
MW_LO_5.frequency(6.950e9)
Z3.prepare_for_timedomain()


# Reset filters
lin_dist_kern_Z3.reset_kernels()
device.prepare_for_timedomain(qubits=['Z3'])
# Measure cryoscope
S17GBT.Flux_arc_wrapper(Qubit='Z3', station=station)
S17GBT.Cryoscope_wrapper(Qubit='Z3', station=station, max_duration=500e-9)

# Set filters manually
reload(ma2)
a = ma2.cv2.multi_qubit_cryoscope_analysis(
    label='Cryoscope',
    update_FIRs=False,
    update_IIRs=True,
    extract_only=False)

for qubit, fltr in a.proc_data_dict['exponential_filter'].items():
    lin_dist_kern = device.find_instrument(f'lin_dist_kern_{qubit}')
    filter_dict = {'params': fltr,
                   'model': 'exponential', 'real-time': True }
    # Check wich is the first empty exponential filter
    for i in range(4):
        _fltr = lin_dist_kern.get(f'filter_model_0{i}')
        if _fltr == {}:
            lin_dist_kern.set(f'filter_model_0{i}', filter_dict)
            # return True
            break
        else:
            print(f'filter_model_0{i} used.')
# print('All exponential filter tabs are full. Filter not updated.')



flux_lm_Z3.vcz_use_net_zero_pulse_NE(False)
flux_lm_Z3.vcz_use_asymmetric_amp_NE(False)
flux_lm_Z3.vcz_asymmetry_NE(0)
S17GBT.Calibrate_CZ_gate(
    qH='Z3', qL='D7', station=station,
    park_distance = 700e6,
    apply_parking_settings = True,
    tmid_offset_samples = 5,
    calibration_type = 'full',
    benchmark = True,
    recompile=True,
    live_plot_enabled=True,
    asymmetry_compensation = False,
    calibrate_asymmetry = False,
    )

flux_lm_Z3.vcz_use_net_zero_pulse_NE(False)
S17GBT.Single_qubit_phase_calibration_wrapper(qH='Z3', qL='D7',
            station=station, fine_cphase_calibration=False)

flux_lm_Z3.vcz_use_net_zero_pulse_NE(True)
S17GBT.Single_qubit_phase_calibration_wrapper(qH='Z3', qL='D7',
            station=station, fine_cphase_calibration=False)

S17GBT.TwoQ_Randomized_benchmarking_wrapper(qH='Z3', qL='D7', 
            station=station, recompile=False)

##############################################
# Add 6 dB attn
##############################################
gen.load_settings_onto_instrument_v2(lin_dist_kern_Z3, timestamp='20230830_104942')
bias = 0.1601765900850296
Z3.instr_FluxCtrl('AWG8_8279')
Z3.fl_dc_ch('sigouts_4_offset')
Z3.instr_FluxCtrl.get_instr().set(Z3.fl_dc_ch(), bias)
# Measure cryoscope
flux_lm_Z3.q_polycoeffs_freq_01_det(np.array([ 8.07411150e+09, -3.21463324e+06, -5.55337208e+06]))
S17GBT.Flux_arc_wrapper(Qubit='Z3', station=station)
# S17GBT.Cryoscope_wrapper(Qubit='Z3', station=station, max_duration=500e-9)

S17GBT.Single_qubit_phase_calibration_wrapper(qH='Z3', qL='D7',
            station=station, fine_cphase_calibration=False)
S17GBT.TwoQ_Randomized_benchmarking_wrapper(qH='Z3', qL='D7', 
            station=station, recompile=True)




for i in range(5):
    S17GBT.Cryoscope_wrapper(Qubit='Z3', station=station,
        max_duration=60e-9,
        ro_acq_averages = 2**12)
    a = ma2.cv2.multi_qubit_cryoscope_analysis(
        update_FIRs=False,
        update_IIRs=False,
        extract_only=False,
        derivative_window_length=2e-9)
    a = ma2.cv2.multi_qubit_cryoscope_analysis(
        update_FIRs=True,
        update_IIRs=False,
        extract_only=False,
        derivative_window_length=2e-9)
    filter_dict = {'params': {'weights': a.proc_data_dict['conv_filters']['Z3']},
                   'model': 'FIR', 'real-time': True }
    lin_dist_kern_Z3.filter_model_04(filter_dict)

S17GBT.Calibrate_CZ_gate(
    qH='Z3', qL='D7', station=station,
    park_distance = 700e6,
    apply_parking_settings = True,
    tmid_offset_samples = 5,
    calibration_type = 'AB fine',
    benchmark = True,
    recompile=True,
    live_plot_enabled=True,
    asymmetry_compensation = False,
    calibrate_asymmetry = False,
    )



Qubits = [
    #'D1', 'D2', 'D3',
    #'D4', 'D5', 'D6',
    #'D7',
    #'D8',
    #'D9',
    #'Z1', 'Z2', 'Z3', 'Z4',
    #'X1', 
    #'X2', 
    'X3', 'X4',
          ]
# Run single-qubit calibration graph
t_SQG = time.time()
for q in Qubits:
    device.find_instrument(q).calibrate_optimal_weights(disable_metadata=True)
    device.find_instrument(q).calibrate_frequency_ramsey(disable_metadata=True)
    S17GBT.Flipping_wrapper(qubit=q, station=station)
    S17GBT.Motzoi_wrapper(qubit=q, station=station)
    S17GBT.Flipping_wrapper(qubit=q, station=station)
    S17GBT.AllXY_wrapper(qubit=q, station=station)
    S17GBT.SSRO_wrapper(qubit=q, station=station)
    S17GBT.T1_wrapper(qubit=q, station=station)
    S17GBT.T2_wrapper(qubit=q, station=station)
    S17GBT.Randomized_benchmarking_wrapper(qubit = q, station=station)
S17GBT.save_snapshot_metadata(station=station)
t_SQG = time.time()-t_SQG