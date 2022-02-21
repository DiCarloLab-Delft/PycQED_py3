###########################################
# VCZ calibration (fine landscape) FLUX dance 1
###########################################
# Align flux pulses
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_X3, flux_lm_D8,
                                           flux_lm_D6, flux_lm_X2], 
                               which_gate= ['NE', 'SW', 
                                            'SW', 'NE'],
                               fl_lm_park = [flux_lm_Z1, flux_lm_D7, flux_lm_Z4],
                               speed_limit = [2.9583333333333334e-08, 
                               				  2.75e-08])
swf2.set_parameter(4)
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_X1, flux_lm_D2], 
                               which_gate= ['NE', 'SW'],
                               fl_lm_park = [flux_lm_D1],
                               speed_limit = [2.75e-08])
swf2.set_parameter(6)

file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# flux-dance 2 
## input from user 
pairs = [['X3', 'D8'], ['D6', 'X2'], ['X1', 'D2']]
which_gate= [['NE', 'SW'],['SW', 'NE'], ['NE', 'SW']]
parked_qubits = ['D7', 'Z1', 'Z4', 'D1']
cfg_amps = [0.28500000000000003,0.19302332066356387,0.25166666666666665]
## processed 
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]

# set CZ parameters
for i,flux_lm_target in enumerate(flux_lms_target): 
  flux_lm_target.cfg_awg_channel_amplitude(cfg_amps[i])
  flux_lm_target.set("vcz_amp_dac_at_11_02_{}".format(which_gate[i][0]), 0.5)
  flux_lms_control[i].set("vcz_amp_dac_at_11_02_{}".format(which_gate[i][1]), 0)

# Set park parameters
for i,flux_lm_park in enumerate(flux_lms_park): 
  flux_lm_park.cfg_awg_channel_amplitude(.3)
  flux_lm_park.park_amp(.5)
  flux_lm_park.park_double_sided(True)

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


Sw_functions = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_sq_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]

swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [1.2/3, 1, 1.2/3])
Sw_functions_2 = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_fine_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]

swf2 = swf.multi_sweep_function(Sw_functions_2, sweep_point_ratios= [1, 1, 1])
MC.live_plot_enabled(True)
nested_MC.live_plot_enabled(True)
nested_MC.cfg_clipping_mode(True)
nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.97, 1.03, 21))
nested_MC.set_sweep_points_2D(np.linspace(0, 1, 11))
label = 'VCZ_2D_{}_fine_sweep'.format(pairs)
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')
###########################################
# VCZ calibration (fine landscape) FLUX dance 2
###########################################	
# Align flux pulses
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_X3, flux_lm_D7,
                                           flux_lm_D5, flux_lm_X2,
                                           flux_lm_X1, flux_lm_D1], 
                               which_gate= ['NW', 'SE', 
                                            'SE', 'NW', 
                                            'NW', 'SE'],
                               fl_lm_park = [flux_lm_Z1, flux_lm_D8, flux_lm_Z4, flux_lm_D2],
                               speed_limit = [2.9583333333333334e-08, 
                           					  2.4166666666666668e-08,
                           					  2.5416666666666666e-08])
swf2.set_parameter(5)
file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# flux-dance 2 
## input from user 
pairs = [['X3', 'D7'], ['D5', 'X2'], ['X1', 'D1']]
which_gate= [['NW', 'SE'],['SE', 'NW'], ['NW', 'SE']]
parked_qubits = ['D8', 'Z1', 'Z4', 'D2']
cfg_amps = [0.3242724012703858,0.16687470158591108,0.27975182997855896]
## processed 
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]

# set CZ parameters
for i,flux_lm_target in enumerate(flux_lms_target): 
  flux_lm_target.cfg_awg_channel_amplitude(cfg_amps[i])
  flux_lm_target.set("vcz_amp_dac_at_11_02_{}".format(which_gate[i][0]), 0.5)
  flux_lms_control[i].set("vcz_amp_dac_at_11_02_{}".format(which_gate[i][1]), 0)

# Set park parameters
for i,flux_lm_park in enumerate(flux_lms_park): 
  flux_lm_park.cfg_awg_channel_amplitude(.3)
  flux_lm_park.park_amp(.5)
  flux_lm_park.park_double_sided(True)

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


Sw_functions = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_sq_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]

swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [1.2/3, 1, 1.2/3])
Sw_functions_2 = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_fine_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]

swf2 = swf.multi_sweep_function(Sw_functions_2, sweep_point_ratios= [1, 1, 1])
MC.live_plot_enabled(True)
nested_MC.live_plot_enabled(True)
nested_MC.cfg_clipping_mode(True)
nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.95, 1.05, 41))
nested_MC.set_sweep_points_2D(np.linspace(0, 1, 21))
label = 'VCZ_2D_{}_fine_sweep'.format(pairs)
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')

###########################################
# VCZ calibration (fine landscape) FLUX dance 3
###########################################	
# Align flux pulses
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_D5, flux_lm_X3,
                                           flux_lm_X2, flux_lm_D3], 
                               which_gate= ['NW', 'SE', 
                                            'SE', 'NW'],
                               fl_lm_park = [flux_lm_Z1, flux_lm_Z4, flux_lm_D2],
                               speed_limit = [2.75e-08, 2.75e-8])
swf2.set_parameter(8)
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_X4, flux_lm_D9], 
                               which_gate= ['SE', 'NW'],
                               fl_lm_park = [flux_lm_D8],
                               speed_limit = [2.75e-8])
swf2.set_parameter(5)

file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# flux-dance 3 
pairs = [['X4', 'D9'], ['D5', 'X3'], ['X2', 'D3']]
which_gate= [['SE', 'NW'],['NW', 'SE'], ['SE', 'NW']]
parked_qubits = ['D8', 'Z1', 'Z4', 'D2']
cfg_amps = [] # input 
## processed 
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]

# set CZ parameters
for i,flux_lm_target in enumerate(flux_lms_target): 
  flux_lm_target.cfg_awg_channel_amplitude(cfg_amps[i])
  flux_lm_target.set("vcz_amp_dac_at_11_02_{}".format(which_gate[i][0]), 0.5)
  flux_lms_control[i].set("vcz_amp_dac_at_11_02_{}".format(which_gate[i][1]), 0)

# Set park parameters
for i,flux_lm_park in enumerate(flux_lms_park): 
  flux_lm_park.cfg_awg_channel_amplitude(.3)
  flux_lm_park.park_amp(.5)
  flux_lm_park.park_double_sided(True)

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

swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [1.2/3, 1, 1.2/3])
Sw_functions_2 = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_fine_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]

swf2 = swf.multi_sweep_function(Sw_functions_2, sweep_point_ratios= [1, 1, 1])
MC.live_plot_enabled(True)
nested_MC.live_plot_enabled(True)
nested_MC.cfg_clipping_mode(True)
nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.95, 1.05, 41))
nested_MC.set_sweep_points_2D(np.linspace(0, 1, 21))
label = 'VCZ_2D_{}_fine_sweep'.format(pairs)
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')

###########################################
# VCZ calibration (fine landscape) FLUX dance 4
###########################################	
# Align flux pulses
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_X4, flux_lm_D8,
                                           flux_lm_D4, flux_lm_X3],
                               which_gate= ['SW', 'NE',
                                            'NE', 'SW'],
                               fl_lm_park = [flux_lm_D9, flux_lm_Z1, flux_lm_Z3],
                               speed_limit = [2.75e-08,
                                              2.9583333333333334e-08]) # input
swf2.set_parameter(7) # input
swf2 = swf.flux_t_middle_sweep(fl_lm_tm = [flux_lm_X2, flux_lm_D2],
                               which_gate= ['SW', 'NE'],
                               fl_lm_park = [flux_lm_D3],
                               speed_limit = [2.75e-08]) # input
swf2.set_parameter(3) # input
file_cfg = gc.generate_config(in_filename=input_file,
                              out_filename=config_fn,
                              mw_pulse_duration=20,
                              ro_duration=2200,
                              flux_pulse_duration=60,
                              init_duration=200000)

# flux-dance 4 
## input from user besides cfg amps & speedlimt & flux-danace code word
pairs = [['X4', 'D8'], ['D4', 'X3'], ['X2', 'D2']]
which_gate= [['SW', 'NE'],['NE', 'SW'], ['SW', 'NE']]
parked_qubits = ['D9', 'Z1', 'Z3', 'D3']
cfg_amps = [] # input 
## processed 
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]
flux_lms_target = [device.find_instrument("flux_lm_{}".format(pair[0]))\
                     for pair in pairs]
flux_lms_control = [device.find_instrument("flux_lm_{}".format(pair[1]))\
                     for pair in pairs]
flux_lms_park = [device.find_instrument("flux_lm_{}".format(qb))\
                     for qb in parked_qubits]

# set CZ parameters
for i,flux_lm_target in enumerate(flux_lms_target): 
  flux_lm_target.cfg_awg_channel_amplitude(cfg_amps[i])
  flux_lm_target.set("vcz_amp_dac_at_11_02_{}".format(which_gate[i][0]), 0.5)
  flux_lms_control[i].set("vcz_amp_dac_at_11_02_{}".format(which_gate[i][1]), 0)

# Set park parameters
for i,flux_lm_park in enumerate(flux_lms_park): 
  flux_lm_park.cfg_awg_channel_amplitude(.3)
  flux_lm_park.park_amp(.5)
  flux_lm_park.park_double_sided(True)

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

swf1 = swf.multi_sweep_function(Sw_functions, sweep_point_ratios= [1.2/3, 1, 1.2/3])
Sw_functions_2 = [swf.FLsweep(flux_lm_target, flux_lm_target.parameters['vcz_amp_fine_{}'.format(gate[0])],
                'cz_{}'.format(gate[0])) for flux_lm_target, gate in \
                zip(flux_lms_target,which_gate)]

swf2 = swf.multi_sweep_function(Sw_functions_2, sweep_point_ratios= [1, 1, 1])
MC.live_plot_enabled(True)
nested_MC.live_plot_enabled(True)
nested_MC.cfg_clipping_mode(True)
nested_MC.set_sweep_function(swf1)
nested_MC.set_sweep_function_2D(swf2)
nested_MC.set_sweep_points(np.linspace(.95, 1.05, 41))
nested_MC.set_sweep_points_2D(np.linspace(0, 1, 21))
label = 'VCZ_2D_{}_fine_sweep'.format(pairs)
nested_MC.set_detector_function(conv_cost_det)
result = nested_MC.run(label, mode='2D')
try:
    ma2.Conditional_Oscillation_Heatmap_Analysis(label=label)
except Exception:
    print('Failed Analysis')