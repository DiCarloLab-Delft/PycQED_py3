###########################################
# VCZ calibration (coarse landscape) FLUX dance 1
###########################################
file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# set CZ parameters
flux_lm_X3.cfg_awg_channel_amplitude(0.28500000000000003)
flux_lm_X3.vcz_amp_dac_at_11_02_NE(.5)
flux_lm_D8.vcz_amp_dac_at_11_02_SW(0)

flux_lm_D6.cfg_awg_channel_amplitude(0.19302332066356387)
flux_lm_D6.vcz_amp_dac_at_11_02_SW(.5)
flux_lm_X2.vcz_amp_dac_at_11_02_NE(0)

flux_lm_X1.cfg_awg_channel_amplitude(0.25166666666666665)
flux_lm_X1.vcz_amp_dac_at_11_02_NE(.5)
flux_lm_D2.vcz_amp_dac_at_11_02_SW(0)

# Set park parameters
flux_lm_D7.cfg_awg_channel_amplitude(.21)
flux_lm_Z4.cfg_awg_channel_amplitude(.19)
flux_lm_Z1.cfg_awg_channel_amplitude(.21)
flux_lm_D1.cfg_awg_channel_amplitude(.235)
flux_lm_D7.park_amp(.5)
flux_lm_Z4.park_amp(.5)
flux_lm_Z1.park_amp(.5)
flux_lm_D1.park_amp(.5)
flux_lm_D7.park_double_sided(True)
flux_lm_Z4.park_double_sided(True)
flux_lm_Z1.park_double_sided(True)
flux_lm_D1.park_double_sided(True)

device.ro_acq_averages(1024)
device.ro_acq_digitized(False)
device.ro_acq_weight_type('optimal')
device.prepare_fluxing(qubits=['D7', 'Z4', 'Z1', 'D1'])
device.prepare_for_timedomain(qubits=['X3', 'D8', 'D6', 'X2', 'X1', 'D2'])

pairs = [['X3', 'D8'], ['D6', 'X2'], ['X1', 'D2']]
parked_qubits = ['D7', 'Z1', 'Z4', 'D1']
from pycqed.measurement import cz_cost_functions as cf
conv_cost_det = det.Function_Detector(
      get_function=cf.conventional_CZ_cost_func2,
      msmt_kw={'device': device,
               'MC': MC,
               'pairs' : pairs,
               'parked_qbs': parked_qubits,
               'prepare_for_timedomain': False,
               'disable_metadata': True,
               'extract_only': True,
               'disable_metadata': True,
               'flux_codeword': 'flux-dance-1',
               'parked_qubit_seq': 'ground',
               'include_single_qubit_phase_in_cost': False,
               'target_single_qubit_phase': 360,
               'include_leakage_in_cost': True,
               'target_phase': 180,
               'cond_phase_weight_factor': 2},
      value_names=[f'cost_function_val_{pair}' for pair in pairs ] +
                  [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      result_keys=[f'cost_function_val_{pair}' for pair in pairs ] +
                  [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      value_units=['a.u.' for pair in pairs ] + 
                  ['deg' for pair in pairs ] + 
                  ['%' for pair in pairs ])

Sw_functions = [ swf.FLsweep(flux_lm_X3, flux_lm_X3.vcz_amp_sq_NE, 'cz_NE'),
                 swf.FLsweep(flux_lm_D6, flux_lm_D6.vcz_amp_sq_SW, 'cz_SW'),
                 swf.FLsweep(flux_lm_X1, flux_lm_X1.vcz_amp_sq_NE, 'cz_NE') ]
swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [1.2/3, 1, 1.2/3])
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_X3, flux_lm_D8,
                                           flux_lm_D6, flux_lm_X2,
                                           flux_lm_X1, flux_lm_D2], 
                               which_gate= ['NE', 'SW', 
                                            'SW', 'NE', 
                                            'NE', 'SW'],
                               fl_lm_park = [flux_lm_Z1, flux_lm_D7, flux_lm_Z4, flux_lm_D1],
                               speed_limit = [2.9583333333333334e-08, 2.75e-08, 2.75e-08])

# swf2.set_parameter(5)
# plt.plot(flux_lm_D5._wave_dict['cz_SE'], label='D5')
# plt.plot(flux_lm_X3._wave_dict['cz_NW'], label='X3')
# plt.plot(flux_lm_X2._wave_dict['cz_NW'], label='X2')
# plt.plot(flux_lm_D7._wave_dict['cz_SE'], label='D7')
# plt.plot(flux_lm_Z1._wave_dict['park'], label='Z1')
# plt.plot(flux_lm_Z1._wave_dict['park'], label='Z4')
# plt.plot(flux_lm_Z1._wave_dict['park'], label='D8')
# plt.axhline(.5, color='k', ls='--', alpha=.25)
# plt.legend()
# plt.show()

nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.95, 1.05, 21))
nested_MC.set_sweep_points_2D(np.linspace(0, 10, 11)[::1])

nested_MC.cfg_clipping_mode(True)
label = 'VCZ_2D_{}_tm{}'.format(pairs, ' sweep')
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')



###########################################
# VCZ calibration (coarse landscape) FLUX dance 2
###########################################
file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# set CZ parameters
flux_lm_X3.cfg_awg_channel_amplitude(0.3242724012703858)
flux_lm_X3.vcz_amp_dac_at_11_02_NW(.5)
flux_lm_D7.vcz_amp_dac_at_11_02_SE(0)

flux_lm_D5.cfg_awg_channel_amplitude(0.16687470158591108)
flux_lm_D5.vcz_amp_dac_at_11_02_SE(.5)
flux_lm_X2.vcz_amp_dac_at_11_02_NW(0)

flux_lm_X1.cfg_awg_channel_amplitude(0.27975182997855896)
flux_lm_X1.vcz_amp_dac_at_11_02_NW(.5)
flux_lm_D1.vcz_amp_dac_at_11_02_SE(0)

# Set park parameters
flux_lm_D8.cfg_awg_channel_amplitude(.22)
flux_lm_Z4.cfg_awg_channel_amplitude(.19)
flux_lm_Z1.cfg_awg_channel_amplitude(.21)
flux_lm_D2.cfg_awg_channel_amplitude(.225)
flux_lm_D8.park_amp(.5)
flux_lm_Z4.park_amp(.5)
flux_lm_Z1.park_amp(.5)
flux_lm_D2.park_amp(.5)
flux_lm_D8.park_double_sided(True)
flux_lm_Z4.park_double_sided(True)
flux_lm_Z1.park_double_sided(True)
flux_lm_D2.park_double_sided(True)

device.ro_acq_averages(1024)
device.ro_acq_digitized(False)
device.ro_acq_weight_type('optimal')
device.prepare_fluxing(qubits=['D8', 'Z4', 'Z1', 'D2'])
device.prepare_for_timedomain(qubits=['X3', 'D7', 'D5', 'X2', 'X1', 'D1'])

pairs = [['X3', 'D7'], ['D5', 'X2'], ['X1', 'D1']]
parked_qubits = ['D8', 'Z1', 'Z4', 'D2']
from pycqed.measurement import cz_cost_functions as cf
conv_cost_det = det.Function_Detector(
      get_function=cf.conventional_CZ_cost_func2,
      msmt_kw={'device': device,
               'MC': MC,
               'pairs' : pairs,
               'parked_qbs': parked_qubits,
               'prepare_for_timedomain': False,
               'disable_metadata': True,
               'extract_only': True,
               'disable_metadata': True,
               'flux_codeword': 'flux-dance-2',
               'parked_qubit_seq': 'ground',
               'include_single_qubit_phase_in_cost': False,
               'target_single_qubit_phase': 360,
               'include_leakage_in_cost': True,
               'target_phase': 180,
               'cond_phase_weight_factor': 2},
      value_names=[f'cost_function_val_{pair}' for pair in pairs ] +
                  [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      result_keys=[f'cost_function_val_{pair}' for pair in pairs ] +
                  [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      value_units=['a.u.' for pair in pairs ] + 
                  ['deg' for pair in pairs ] + 
                  ['%' for pair in pairs ])

Sw_functions = [ swf.FLsweep(flux_lm_X3, flux_lm_X3.vcz_amp_sq_NW, 'cz_NW'),
                 swf.FLsweep(flux_lm_D5, flux_lm_D5.vcz_amp_sq_SE, 'cz_SE'),
                 swf.FLsweep(flux_lm_X1, flux_lm_X1.vcz_amp_sq_NW, 'cz_NW') ]
swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [1.2/3, 1, 1.2/3])
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_X3, flux_lm_D7,
                                           flux_lm_D5, flux_lm_X2,
                                           flux_lm_X1, flux_lm_D1], 
                               which_gate= ['NW', 'SE', 
                                            'SE', 'NW', 
                                            'NW', 'SE'],
      			                   fl_lm_park = [flux_lm_Z1, flux_lm_D8, flux_lm_Z4, flux_lm_D2],
      			                   speed_limit = [2.9583333333333334e-08, 2.4166666666666668e-08, 2.5416666666666666e-08])

# swf2.set_parameter(5)
# plt.plot(flux_lm_X4._wave_dict['cz_SE'], label='X4')
# plt.plot(flux_lm_D9._wave_dict['cz_NW'], label='D9')
# plt.plot(flux_lm_D5._wave_dict['cz_NW'], label='D5')
# plt.plot(flux_lm_X3._wave_dict['cz_SE'], label='X3')
# plt.plot(flux_lm_X2._wave_dict['cz_NW'], label='X2')
# plt.plot(flux_lm_D3._wave_dict['cz_SE'], label='D3')
# plt.plot(flux_lm_Z1._wave_dict['park'], label='Z1')
# plt.plot(flux_lm_Z1._wave_dict['park'], label='Z4')
# plt.plot(flux_lm_Z1._wave_dict['park'], label='D8')
# plt.axhline(.5, color='k', ls='--', alpha=.25)
# plt.legend()
# plt.show()

nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.95, 1.05, 21))
nested_MC.set_sweep_points_2D(np.linspace(0, 10, 11)[::1])

nested_MC.cfg_clipping_mode(True)
label = 'VCZ_2D_{}_tm{}'.format(pairs, ' sweep')
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')



coha = ma2.Conditional_Oscillation_Heatmap_Analysis(
            label="223142_VCZ_2D_[['X3', 'D7'], ['D5', 'X2'], ['X1', 'D1']]_fine_sweep",
            for_multi_CZ = True,
            pair = {'pair_name':['X3','D7'],'sweep_ratio':[1.2/3,1],'pair_num':0},
            close_figs=True,
            extract_only=False,
            plt_orig_pnts=True,
            plt_contour_L1=False,
            plt_contour_phase=True,
            plt_optimal_values=True,
            plt_optimal_values_max=1,
            find_local_optimals=True,
            plt_clusters=False,
            cluster_from_interp=False,
            clims={
                "Cost func": [0., 300],
                "missing fraction": [0, 30],
                "offset difference": [0, 30]
            },
            target_cond_phase=180,
            phase_thr=15,
            L1_thr=5,
            clustering_thr=0.15,
            gen_optima_hulls=True,
            hull_L1_thr=4,
            hull_phase_thr=20,
            plt_optimal_hulls=True,
            save_cond_phase_contours=[180],
          )

###########################################
# VCZ calibration (coarse landscape) FLUX dance 3
###########################################
file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# set CZ parameters
flux_lm_X4.cfg_awg_channel_amplitude(0.2658333333333333)
flux_lm_X4.vcz_amp_dac_at_11_02_SE(.5)
flux_lm_D9.vcz_amp_dac_at_11_02_NW(0)

flux_lm_D5.cfg_awg_channel_amplitude(0.2)
flux_lm_D5.vcz_amp_dac_at_11_02_NW(.5)
flux_lm_X3.vcz_amp_dac_at_11_02_SE(0)

flux_lm_X2.cfg_awg_channel_amplitude(0.316)
flux_lm_X2.vcz_amp_dac_at_11_02_SE(.5)
flux_lm_D3.vcz_amp_dac_at_11_02_NW(0)

# Set park parameters
flux_lm_D8.cfg_awg_channel_amplitude(.22)
flux_lm_Z4.cfg_awg_channel_amplitude(.19)
flux_lm_Z1.cfg_awg_channel_amplitude(.21)
flux_lm_D2.cfg_awg_channel_amplitude(.225)
flux_lm_D8.park_amp(.5)
flux_lm_Z4.park_amp(.5)
flux_lm_Z1.park_amp(.5)
flux_lm_D2.park_amp(.5)
flux_lm_D8.park_double_sided(True)
flux_lm_Z4.park_double_sided(True)
flux_lm_Z1.park_double_sided(True)
flux_lm_D2.park_double_sided(True)

# flux-dance 3 
## input from user besides cfg amps & speedlimt & flux-danace code
pairs = [['X4', 'D9'], ['D5', 'X3'], ['X2', 'D3']]
which_gate= [['SE', 'NW'],['NW', 'SE'], ['SE', 'NW']]
parked_qubits = ['D8', 'Z1', 'Z4', 'D2']
## processed 
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]

list_qubits_used = np.asarray(pairs).flatten().tolist()
which_gates = np.asarray(which_gate).flatten().tolist()
device.ro_acq_averages(1024)
device.ro_acq_digitized(False)
device.ro_acq_weight_type('optimal')
device.prepare_fluxing(qubits=parked_qubits)
device.prepare_for_timedomain(qubits=list_qubits_used)

from pycqed.measurement import cz_cost_functions as cf
conv_cost_det = det.Function_Detector(
      get_function=cf.conventional_CZ_cost_func2,
      msmt_kw={'device': device,
               'MC': MC,
               'pairs' : pairs,
               'parked_qbs': parked_qubits,
               'prepare_for_timedomain': False,
               'disable_metadata': True,
               'extract_only': True,
               'disable_metadata': True,
               'flux_codeword': 'flux-dance-3',
               'parked_qubit_seq': 'ground',
               'include_single_qubit_phase_in_cost': False,
               'target_single_qubit_phase': 360,
               'include_leakage_in_cost': True,
               'target_phase': 180,
               'cond_phase_weight_factor': 2},
      value_names=[f'cost_function_val_{pair}' for pair in pairs ] +
                   [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      result_keys=[f'cost_function_val_{pair}' for pair in pairs ] +
                  [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      value_units=['a.u.' for pair in pairs ] + 
                  ['deg' for pair in pairs ] + 
                  ['%' for pair in pairs ])
Sw_functions = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_sq_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]

swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [.5, 1, .2])
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [device.find_instrument("flux_lm_{}".format(qubit))\
                                          for qubit in list_qubits_used], 
                               which_gate= which_gates,
                               fl_lm_park = flux_lms_park,
                               speed_limit = [2.75e-08, 2.75e-08, 2.75e-8]) # input
nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.95, 1.05, 31))
nested_MC.set_sweep_points_2D(np.linspace(0, 10, 11)[::1])

nested_MC.cfg_clipping_mode(True)
label = 'VCZ_2D_{}_tm{}'.format(pairs, ' sweep')
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')

###########################################
# VCZ calibration (coarse landscape) FLUX dance 4
###########################################
file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# set CZ parameters
flux_lm_X4.cfg_awg_channel_amplitude(0.261)
flux_lm_X4.vcz_amp_dac_at_11_02_SW(.5)
flux_lm_D8.vcz_amp_dac_at_11_02_NE(0)

flux_lm_D4.cfg_awg_channel_amplitude(0.201)
flux_lm_D4.vcz_amp_dac_at_11_02_NE(.5)
flux_lm_X3.vcz_amp_dac_at_11_02_SW(0)

flux_lm_X2.cfg_awg_channel_amplitude(0.31174999999999997)
flux_lm_X2.vcz_amp_dac_at_11_02_SW(.5)
flux_lm_D2.vcz_amp_dac_at_11_02_NE(0)

# Set park parameters
flux_lm_D9.cfg_awg_channel_amplitude(.206)
flux_lm_Z3.cfg_awg_channel_amplitude(.214)
flux_lm_Z1.cfg_awg_channel_amplitude(.21)
flux_lm_D3.cfg_awg_channel_amplitude(.223)
flux_lm_D9.park_amp(.5)
flux_lm_Z3.park_amp(.5)
flux_lm_Z1.park_amp(.5)
flux_lm_D3.park_amp(.5)
flux_lm_D9.park_double_sided(True)
flux_lm_Z3.park_double_sided(True)
flux_lm_Z1.park_double_sided(True)
flux_lm_D3.park_double_sided(True)

# flux-dance 4 
## input from user besides cfg amps & speedlimt & flux-danace code word
pairs = [['X4', 'D8'], ['D4', 'X3'], ['X2', 'D2']]
which_gate= [['SW', 'NE'],['NE', 'SW'], ['SW', 'NE']]
parked_qubits = ['D9', 'Z1', 'Z3', 'D3']
## processed 
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]

list_qubits_used = np.asarray(pairs).flatten().tolist()
which_gates = np.asarray(which_gate).flatten().tolist()
device.ro_acq_averages(1024)
device.ro_acq_digitized(False)
device.ro_acq_weight_type('optimal')
device.prepare_fluxing(qubits=parked_qubits)
device.prepare_for_timedomain(qubits=list_qubits_used)

from pycqed.measurement import cz_cost_functions as cf
conv_cost_det = det.Function_Detector(
      get_function=cf.conventional_CZ_cost_func2,
      msmt_kw={'device': device,
               'MC': MC,
               'pairs' : pairs,
               'parked_qbs': parked_qubits,
               'prepare_for_timedomain': False,
               'disable_metadata': True,
               'extract_only': True,
               'disable_metadata': True,
               'flux_codeword': 'flux-dance-4',
               'parked_qubit_seq': 'ground',
               'include_single_qubit_phase_in_cost': False,
               'target_single_qubit_phase': 360,
               'include_leakage_in_cost': True,
               'target_phase': 180,
               'cond_phase_weight_factor': 2},
      value_names=[f'cost_function_val_{pair}' for pair in pairs ] +
                   [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      result_keys=[f'cost_function_val_{pair}' for pair in pairs ] +
                  [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      value_units=['a.u.' for pair in pairs ] + 
                  ['deg' for pair in pairs ] + 
                  ['%' for pair in pairs ])

Sw_functions = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_sq_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]
swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [.6, 1.8, 1.2/3])
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [device.find_instrument("flux_lm_{}".format(qubit))\
                                          for qubit in list_qubits_used], 
                               which_gate= which_gates,
                               fl_lm_park = flux_lms_park,
                               speed_limit = [2.75e-08, 2.78e-8,2.75e-08]) # input
nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.985, 1.005, 31))
nested_MC.set_sweep_points_2D(np.linspace(0, 10, 11)[::-1])

nested_MC.cfg_clipping_mode(True)
label = 'VCZ_2D_{}_tm{}'.format(pairs, ' sweep')
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')




###########################################
# VCZ calibration (coarse landscape) FLUX dance 4
###########################################
file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# set CZ parameters
flux_lm_D4.cfg_awg_channel_amplitude(0.201)
flux_lm_D4.vcz_amp_dac_at_11_02_NE(.5)
flux_lm_X3.vcz_amp_dac_at_11_02_SW(0)

# Set park parameters
flux_lm_Z3.cfg_awg_channel_amplitude(.3)#(.214)
flux_lm_Z1.cfg_awg_channel_amplitude(.3)#(.21)
flux_lm_Z3.park_amp(.5)
flux_lm_Z1.park_amp(.5)
flux_lm_Z3.park_double_sided(False)
flux_lm_Z1.park_double_sided(False)

plt.plot(flux_lm_D4._wave_dict['cz_NE'], label='D4')
plt.plot(flux_lm_X3._wave_dict['cz_SW'], label='X3')
plt.plot(flux_lm_Z1._wave_dict['park'], label='Z1')
plt.plot(flux_lm_Z3._wave_dict['park'], label='Z3')
plt.axhline(.5, color='k', ls='--', alpha=.25)
plt.legend()
plt.show()

# flux-dance 4 
## input from user besides cfg amps & speedlimt & flux-danace code word
pairs = [['D4', 'X3']]
which_gate= [['NE', 'SW']]
parked_qubits = ['Z1', 'Z3']
## processed 
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]

list_qubits_used = np.asarray(pairs).flatten().tolist()
which_gates = np.asarray(which_gate).flatten().tolist()
device.ro_acq_averages(1024)
device.ro_acq_digitized(False)
device.ro_acq_weight_type('optimal')
device.prepare_fluxing(qubits=parked_qubits)
device.prepare_for_timedomain(qubits=list_qubits_used)

from pycqed.measurement import cz_cost_functions as cf
conv_cost_det = det.Function_Detector(
      get_function=cf.conventional_CZ_cost_func2,
      msmt_kw={'device': device,
               'MC': MC,
               'pairs' : pairs,
               'parked_qbs': parked_qubits,
               'wait_time_before_flux_ns': 60,
               'wait_time_after_flux_ns': 60,
               'prepare_for_timedomain': False,
               'disable_metadata': True,
               'extract_only': True,
               'disable_metadata': True,
               'flux_codeword': 'cz',
               'parked_qubit_seq': 'ground',
               'include_single_qubit_phase_in_cost': False,
               'target_single_qubit_phase': 360,
               'include_leakage_in_cost': True,
               'target_phase': 180,
               'cond_phase_weight_factor': 2},
      value_names=[f'cost_function_val_{pair}' for pair in pairs ] +
                   [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      result_keys=[f'cost_function_val_{pair}' for pair in pairs ] +
                  [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      value_units=['a.u.' for pair in pairs ] + 
                  ['deg' for pair in pairs ] + 
                  ['%' for pair in pairs ])

Sw_functions = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_sq_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]
swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [1])
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [device.find_instrument("flux_lm_{}".format(qubit))\
                                          for qubit in list_qubits_used], 
                               which_gate= which_gates,
                               fl_lm_park = flux_lms_park,
                               speed_limit = [2.78e-8]) # input
nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.95, 1.05, 11))
nested_MC.set_sweep_points_2D([0,1,2,3,4,5,6,7,8,9,10])

nested_MC.cfg_clipping_mode(True)
label = 'VCZ_2D_{}_tm{}'.format(pairs, ' sweep')
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')


###########################################
# VCZ calibration (coarse landscape) FLUX dance 4 (olddd)
###########################################
file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# set CZ parameters
flux_lm_X4.cfg_awg_channel_amplitude(0.261)
flux_lm_X4.vcz_amp_dac_at_11_02_SW(.5)
flux_lm_D8.vcz_amp_dac_at_11_02_NE(0)

flux_lm_D4.cfg_awg_channel_amplitude(0.25999999046325684)
flux_lm_D4.vcz_amp_dac_at_11_02_NE(.5)
flux_lm_X3.vcz_amp_dac_at_11_02_SW(0)

flux_lm_X2.cfg_awg_channel_amplitude(0.31174999999999997)
flux_lm_X2.vcz_amp_dac_at_11_02_SW(.5)
flux_lm_D2.vcz_amp_dac_at_11_02_NE(0)

# Set park parameters
flux_lm_D9.cfg_awg_channel_amplitude(.206)
flux_lm_Z3.cfg_awg_channel_amplitude(.214)
flux_lm_Z1.cfg_awg_channel_amplitude(.21)
flux_lm_D3.cfg_awg_channel_amplitude(.223)
flux_lm_D9.park_amp(.5)
flux_lm_Z3.park_amp(.5)
flux_lm_Z1.park_amp(.5)
flux_lm_D3.park_amp(.5)
flux_lm_D9.park_double_sided(True)
flux_lm_Z3.park_double_sided(True)
flux_lm_Z1.park_double_sided(True)
flux_lm_D3.park_double_sided(True)

# flux-dance 4 
## input from user besides cfg amps & speedlimt & flux-danace code word
pairs = [['X4', 'D8'], ['D4', 'X3'], ['X2', 'D2']]
which_gate= [['SW', 'NE'],['NE', 'SW'], ['SW', 'NE']]
parked_qubits = ['D9', 'Z1', 'Z3', 'D3']
## processed 
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]

list_qubits_used = np.asarray(pairs).flatten().tolist()
which_gates = np.asarray(which_gate).flatten().tolist()
device.ro_acq_averages(1024)
device.ro_acq_digitized(False)
device.ro_acq_weight_type('optimal')
device.prepare_fluxing(qubits=parked_qubits)
device.prepare_for_timedomain(qubits=list_qubits_used)

from pycqed.measurement import cz_cost_functions as cf
conv_cost_det = det.Function_Detector(
      get_function=cf.conventional_CZ_cost_func2,
      msmt_kw={'device': device,
               'MC': MC,
               'pairs' : pairs,
               'parked_qbs': parked_qubits,
               'prepare_for_timedomain': False,
               'disable_metadata': True,
               'extract_only': True,
               'disable_metadata': True,
               'flux_codeword': 'flux-dance-4',
               'parked_qubit_seq': 'ground',
               'include_single_qubit_phase_in_cost': False,
               'target_single_qubit_phase': 360,
               'include_leakage_in_cost': True,
               'target_phase': 180,
               'cond_phase_weight_factor': 2},
      value_names=[f'cost_function_val_{pair}' for pair in pairs ] +
                   [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      result_keys=[f'cost_function_val_{pair}' for pair in pairs ] +
                  [f'delta_phi_{pair}' for pair in pairs ] + 
                  [f'missing_fraction_{pair}' for pair in pairs ],
      value_units=['a.u.' for pair in pairs ] + 
                  ['deg' for pair in pairs ] + 
                  ['%' for pair in pairs ])

Sw_functions = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_sq_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]
swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [.6, 1.8, 1.2/3])
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [device.find_instrument("flux_lm_{}".format(qubit))\
                                          for qubit in list_qubits_used], 
                               which_gate= which_gates,
                               fl_lm_park = flux_lms_park,
                               speed_limit = [2.75e-08, 2.78e-8,2.75e-08]) # input
nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.985, 1.005, 31))
nested_MC.set_sweep_points_2D(np.linspace(0, 10, 11)[::-1])

nested_MC.cfg_clipping_mode(True)
label = 'VCZ_2D_{}_tm{}'.format(pairs, ' sweep')
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')
