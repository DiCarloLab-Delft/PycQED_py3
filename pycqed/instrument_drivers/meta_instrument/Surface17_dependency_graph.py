from importlib import reload
import autodepgraph
reload(autodepgraph)
from autodepgraph import AutoDepGraph_DAG
from pycqed.measurement import hdf5_data as h5d
import numpy as np
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.utilities.general import get_gate_directions
from pycqed.measurement import sweep_functions as swf
import matplotlib.pyplot as plt

###############################################################################
# Single- and Two- qubit gate calibration graph
###############################################################################
import os
import pycqed as pq
from pycqed.measurement.openql_experiments import generate_CC_cfg as gc
input_file = os.path.join(pq.__path__[0], 'measurement',
                          'openql_experiments', 'config_cc_s17_direct_iq.json.in')
config_fn = os.path.join(pq.__path__[0], 'measurement',
                       'openql_experiments', 'output_cc_s17','config_cc_s17_direct_iq.json')

###############################################################################
# Single qubit gate calibration graph
###############################################################################
class Single_qubit_gate_calibration(AutoDepGraph_DAG):
	def __init__(self, 
		         name: str,
		         station,
		         **kwargs):
		super().__init__(name, **kwargs)
		self.station = station
		self.create_dep_graph()

	def create_dep_graph(self):
		'''
		Dependency graph for the calibration of 
		single-qubit gates.
		'''
		print(f'Creating dependency graph for single-qubit gate calibration')
		##############################
		# Grah nodes
		##############################
		module_name = 'pycqed.instrument_drivers.meta_instrument.Surface17_dependency_graph'
		
		Qubits = [
			'D1', 'D2', 'D3',
			'D4', 'D5', 'D6',
			'D7', 'D8', 'D9',
			# 'X1', 'X3', 'X4',
			'Z1', 'Z2', 'Z3', 'Z4',
			]

		for qubit in Qubits:
			self.add_node(f'{qubit} Prepare for gate calibration',
			    calibrate_function=module_name+'.prepare_for_single_qubit_gate_calibration',
		        calibrate_function_args={
		          	'qubit' : qubit,
		          	'station': self.station,
		          	})

			self.add_node(f'{qubit} Frequency',
			              calibrate_function=qubit+'.calibrate_frequency_ramsey',
			              calibrate_function_args={
			              	'steps':[3, 10, 30],
			              	'disable_metadata': True})

			self.add_node(f'{qubit} Flipping',
			              calibrate_function=module_name+'.Flipping_wrapper',
			              calibrate_function_args={
			              	'qubit' : qubit,
			              	'station': self.station,
			              	})

			self.add_node(f'{qubit} Motzoi',
			              calibrate_function=module_name+'.Motzoi_wrapper',
			              calibrate_function_args={
			              	'qubit': qubit,
			              	'station': self.station,
			              	})

			self.add_node(f'{qubit} AllXY',
			              calibrate_function=module_name+'.AllXY_wrapper',
			              calibrate_function_args={
			              	'qubit': qubit,
			              	'station': self.station,
			              	})

			self.add_node(f'{qubit} Readout',
			              calibrate_function=module_name+'.SSRO_wrapper',
			              calibrate_function_args={
			              	'qubit': qubit,
			              	'station': self.station,
			              	})

			self.add_node(f'{qubit} T1',
			              calibrate_function=module_name+'.T1_wrapper',
			              calibrate_function_args={
			              	'qubit': qubit,
			              	'station': self.station,
			              	})

			self.add_node(f'{qubit} T2',
			              calibrate_function=module_name+'.T2_wrapper',
			              calibrate_function_args={
			              	'qubit': qubit,
			              	'station': self.station,
			              	})

			self.add_node(f'{qubit} Randomized Benchmarking',
			              calibrate_function=module_name+'.Randomized_benchmarking_wrapper',
			              calibrate_function_args={
			              	'qubit': qubit,
			              	'station': self.station,
			              	})

			# self.add_node(f'{qubit} drive mixer calibration',
			#               calibrate_function=module_name+'.drive_mixer_wrapper',
			#               calibrate_function_args={
			#               	'qubit': qubit,
			#               	'station': self.station,
			#               	})

			##############################
			# Node depdendencies
			##############################
			self.add_edge(f'{qubit} Frequency',
			              f'{qubit} Prepare for gate calibration')

			self.add_edge(f'{qubit} Flipping',
			              f'{qubit} Frequency')

			self.add_edge(f'{qubit} Motzoi',
			              f'{qubit} Frequency')

			self.add_edge(f'{qubit} AllXY',
			              f'{qubit} Flipping')

			self.add_edge(f'{qubit} AllXY',
			              f'{qubit} Motzoi')

			self.add_edge(f'{qubit} Randomized Benchmarking',
			              f'{qubit} AllXY')

			self.add_edge(f'{qubit} Randomized Benchmarking',
			              f'{qubit} Readout')

			self.add_edge(f'{qubit} Randomized Benchmarking',
			              f'{qubit} T1')

			self.add_edge(f'{qubit} Randomized Benchmarking',
			              f'{qubit} T2')

		self.add_node(f'Save snapshot',
              calibrate_function=module_name+'.save_snapshot_metadata',
              calibrate_function_args={
              	'station': self.station,
              	})
		for qubit in Qubits:
			self.add_edge(f'Save snapshot',
			              f'{qubit} Randomized Benchmarking')

		##############################
		# Create graph
		##############################
		self.cfg_plot_mode = 'svg'
		self.update_monitor()
		self.cfg_svg_filename
		url = self.open_html_viewer()
		print('Dependency graph created at ' + url)


def prepare_for_single_qubit_gate_calibration(qubit:str, station):
	'''
	Initial function to prepare qubit for calibration.
	We will set all relevant parameters for mw and readout.
	This is such that we only perform full preparation of
	the qubit once in the graph and all additional calibrated
	parameters are uploaded individually making the whole
	procedure time efficient.
	'''
	Q_inst = station.components[qubit]
	# Set initial parameters for calibration
	Q_inst.mw_gauss_width(5e-9)
	Q_inst.mw_motzoi(0)
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal')
	Q_inst.ro_acq_averages(2**10)
	Q_inst.ro_acq_digitized(False)
	# Set microwave lutman
	Q_lm = Q_inst.instr_LutMan_MW.get_instr()
	Q_lm.set_default_lutmap()
	# Prepare for timedomain
	Q_inst.prepare_for_timedomain()
	return True


def Flipping_wrapper(qubit:str, station):
	'''
	Wrapper function around flipping measurement.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal')
	Q_inst.ro_acq_averages(2**11)
	# # Check if RO pulse has been uploaded onto UHF
	# # (We do this by checking if the resonator 
	# # combinations of the RO lutman contain
	# # exclusively this qubit).
	# RO_lm = Q_inst.instr_LutMan_RO.get_instr()
	# _res_combs = RO_lm.resonator_combinations()
	# if _res_combs != [[Q_inst.cfg_qubit_nr()]]:
	# 	Q_inst.prepare_readout()
	# else:
	# 	# Just update detector functions (for avg and IQ)
	# 	Q_inst._prep_ro_integration_weights()
	# 	Q_inst._prep_ro_instantiate_detectors()
	Q_inst.prepare_for_timedomain()
	# Run loop of experiments
	nr_repetitions = 4
	for i in range(nr_repetitions):
		# Prepare for timedomain
		# (disable upload of waveforms on
		# awg sincethese will always be the
		# same if using real-time modulation.)
		Q_inst.cfg_prepare_mw_awg(False)
		Q_inst._prep_mw_pulses()
		Q_inst.cfg_prepare_mw_awg(True)

		# perform measurement
		a = Q_inst.measure_flipping(
			update=True,
			disable_metadata=True,
			prepare_for_timedomain=False)
		# if amplitude is lower than threshold
		if a == True:
			return True
	return False


def Motzoi_wrapper(qubit:str, station):
	'''
	Wrapper function around Motzoi measurement.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal')
	Q_inst.ro_acq_averages(2**11)
	# Check if RO pulse has been uploaded onto UHF
	# (We do this by checking if the resonator 
	# combinations of the RO lutman contain
	# exclusively this qubit).
	RO_lm = Q_inst.instr_LutMan_RO.get_instr()
	_res_combs = RO_lm.resonator_combinations()
	if _res_combs != [[Q_inst.cfg_qubit_nr()]]:
		Q_inst.prepare_readout()
	else:
		# Just update detector functions (for avg and IQ)
		Q_inst._prep_ro_integration_weights()
		Q_inst._prep_ro_instantiate_detectors()
	# Prepare for timedomain
	Q_inst._prep_mw_pulses()
	# perform measurement
	_range = .3
	for i in range(4):
		outcome = Q_inst.calibrate_motzoi(
			update=True,
			motzois=np.linspace(-_range/2, _range/2, 5),
			disable_metadata=True,
			prepare_for_timedomain=False)
		# If successfull calibration
		if outcome != False:
			return True
		# if not increase range and try again
		else:
			_range += .1
	# If not successful after 4 attempts fail node
	return False


def AllXY_wrapper(qubit:str, station):
	'''
	Wrapper function around AllXY measurement.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal')
	Q_inst.ro_acq_averages(2**13)
	# Check if RO pulse has been uploaded onto UHF
	# (We do this by checking if the resonator 
	# combinations of the RO lutman contain
	# exclusively this qubit).
	RO_lm = Q_inst.instr_LutMan_RO.get_instr()
	_res_combs = RO_lm.resonator_combinations()
	if _res_combs != [[Q_inst.cfg_qubit_nr()]]:
		Q_inst.prepare_readout()
	else:
		# Just update detector functions (for avg and IQ)
		Q_inst._prep_ro_integration_weights()
		Q_inst._prep_ro_instantiate_detectors()
	# Set microwave lutman
	Q_lm = Q_inst.instr_LutMan_MW.get_instr()
	Q_lm.set_default_lutmap()
	# Prepare for timedomain
	Q_inst._prep_mw_pulses()
	out = Q_inst.measure_allxy(
		disable_metadata=True,
		prepare_for_timedomain=False)
	if out > .02:
		return False
	else:
		return True


def SSRO_wrapper(qubit:str, station):
	'''
	Wrapper function around AllXY measurement.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal IQ')
	Q_inst.ro_acq_digitized(False)
	Q_inst.ro_acq_averages(2**10) # Not used in this experiment
	# Check if RO pulse has been uploaded onto UHF
	# (We do this by checking if the resonator 
	# combinations of the RO lutman contain
	# exclusively this qubit).
	RO_lm = Q_inst.instr_LutMan_RO.get_instr()
	_res_combs = RO_lm.resonator_combinations()
	if _res_combs != [[Q_inst.cfg_qubit_nr()]]:
		Q_inst.prepare_readout()
	else:
		# Just update detector functions (for avg and IQ)
		Q_inst._prep_ro_integration_weights()
		Q_inst._prep_ro_instantiate_detectors()
	# Set microwave lutman
	Q_lm = Q_inst.instr_LutMan_MW.get_instr()
	Q_lm.set_default_lutmap()
	# Prepare for timedomain
	Q_inst._prep_td_sources()
	Q_inst._prep_mw_pulses()
	Q_inst.measure_ssro(
		f_state=True,
		nr_shots_per_case=2**15,
		disable_metadata=True,
		prepare=False)
	return True


def T1_wrapper(qubit:str, station):
	'''
	Wrapper function around AllXY measurement.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal IQ')
	Q_inst.ro_acq_digitized(False)
	Q_inst.ro_acq_averages(2**9)
	# Check if RO pulse has been uploaded onto UHF
	# (We do this by checking if the resonator 
	# combinations of the RO lutman contain
	# exclusively this qubit).
	RO_lm = Q_inst.instr_LutMan_RO.get_instr()
	_res_combs = RO_lm.resonator_combinations()
	if _res_combs != [[Q_inst.cfg_qubit_nr()]]:
		Q_inst.prepare_readout()
	else:
		# Just update detector functions (for avg and IQ)
		Q_inst._prep_ro_integration_weights()
		Q_inst._prep_ro_instantiate_detectors()
	# measure
	Q_inst.measure_T1(
		disable_metadata=True,
		prepare_for_timedomain=True)
	return True


def T2_wrapper(qubit:str, station):
	'''
	Wrapper function around AllXY measurement.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal IQ')
	Q_inst.ro_acq_digitized(False)
	Q_inst.ro_acq_averages(2**9)
	# Check if RO pulse has been uploaded onto UHF
	# (We do this by checking if the resonator 
	# combinations of the RO lutman contain
	# exclusively this qubit).
	RO_lm = Q_inst.instr_LutMan_RO.get_instr()
	_res_combs = RO_lm.resonator_combinations()
	if _res_combs != [[Q_inst.cfg_qubit_nr()]]:
		Q_inst.prepare_readout()
	else:
		# Just update detector functions (for avg and IQ)
		Q_inst._prep_ro_integration_weights()
		Q_inst._prep_ro_instantiate_detectors()
	# measure
	Q_inst.measure_echo(
		disable_metadata=True,
		prepare_for_timedomain=False)
	return True


def Randomized_benchmarking_wrapper(qubit:str, station):
	'''
	Wrapper function around Randomized benchmarking.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal IQ')
	Q_inst.ro_acq_averages(2**10) # Not used in RB
	# Check if RO pulse has been uploaded onto UHF
	# (We do this by checking if the resonator 
	# combinations of the RO lutman contain
	# exclusively this qubit).
	RO_lm = Q_inst.instr_LutMan_RO.get_instr()
	_res_combs = RO_lm.resonator_combinations()
	if _res_combs != [[Q_inst.cfg_qubit_nr()]]:
		Q_inst.prepare_readout()
	else:
		# Just update detector functions (for avg and IQ)
		Q_inst._prep_ro_integration_weights()
		Q_inst._prep_ro_instantiate_detectors()
	# Set microwave lutman
	Q_lm = Q_inst.instr_LutMan_MW.get_instr()
	Q_lm.set_default_lutmap()
	# Prepare for timedomain
	Q_inst._prep_td_sources()
	Q_inst._prep_mw_pulses()
	# measurement
	Q_inst.measure_single_qubit_randomized_benchmarking(
		nr_cliffords=2**np.arange(12), # should change to 11
	    nr_seeds=15,
	    recompile=False, 
	    prepare_for_timedomain=False,
	    disable_metadata=True)
	return True


def drive_mixer_wrapper(qubit:str, station):
	'''
	Wrapper function for drive mixer calibration.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	SH = Q_inst.instr_SH.get_instr()
	connect(qubit)
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	# Set default microwave lutman
	Q_lm = Q_inst.instr_LutMan_MW.get_instr()
	Q_lm.set_default_lutmap()
	# Setup Signal hound for leakage
	SH.ref_lvl(-40)
	SH.rbw(1e3)
	SH.vbw(1e3)
	# Measure leakage
	Q_inst.calibrate_mixer_offsets_drive(
					update=True,
					ftarget=-105)
	# Setup Signal hound for skewness
	SH.ref_lvl(-60)
	SH.rbw(1e3)
	SH.vbw(1e3)
	# Measure skewness
	Q_inst.calibrate_mixer_skewness_drive(
					update=True,
					maxfevals=120)
	return True


def save_snapshot_metadata(station, Qubits=None, Qubit_pairs = None):
	'''
	Save snapshot of system.
	'''
	MC = station.components['QInspire_MC']
	name = 'System_snapshot'
	MC.set_measurement_name(name)
	with h5d.Data(
		name=MC.get_measurement_name(), datadir=MC.datadir()
	) as MC.data_object:
		MC.get_measurement_begintime()
		MC.save_instrument_settings(MC.data_object)
	if Qubits == None:
		Qubits = [
			'D1', 'D2', 'D3',
			'D4', 'D5', 'D6',
			'D7', 'D8', 'D9',
			'Z1', 'Z2', 'Z3', 'Z4',
			# 'X1', 'X2', 'X3', 'X4',
			]

	if Qubit_pairs == None:
		Qubit_pairs = [
			['Z3', 'D7'], 
			['D5', 'Z1'], 
			['Z4', 'D9'],
			['Z1', 'D2'], 
			['D4', 'Z3'], 
			['D6', 'Z4'],
			['Z4', 'D8'], 
			['D4', 'Z1'], 
			['D6', 'Z2'],
			['Z2', 'D3'], 
			['Z1', 'D1'], 
			['D5', 'Z4'],
			# ['X1', 'D2'], 
			# ['D6', 'X2'], 
			# ['X3', 'D8'],
			# ['X1', 'D1'], 
			# ['D5', 'X2'], 
			# ['X3', 'D7'],
			# ['X4', 'D9'], 
			# ['D5', 'X3'], 
			# ['X2', 'D3'],
			# ['X4', 'D8'], 
			# ['D4', 'X3'], 
			# ['X2', 'D2'],
						]
	ma2.gbta.SingleQubitGBT_analysis(Qubits=Qubits)
	ma2.gbta.TwoQubitGBT_analysis(Qubit_pairs=Qubit_pairs)

	return True


###############################################################################
# Two qubit gate calibration graph
###############################################################################
import os
import pycqed as pq
from pycqed.measurement.openql_experiments import generate_CC_cfg as gc
input_file = os.path.join(pq.__path__[0], 'measurement',
                          'openql_experiments', 'config_cc_s5_direct_iq.json.in')
config_fn = os.path.join(pq.__path__[0], 'measurement',
                         'openql_experiments', 'output_cc_s5_direct_iq',
                         'cc_s5_direct_iq.json')

class Two_qubit_gate_calibration(AutoDepGraph_DAG):
	def __init__(self, 
		         name: str,
		         station,
		         Qubit_pairs: list = None,
		         **kwargs):
		super().__init__(name, **kwargs)
		if Qubit_pairs == None:
			Qubit_pairs = [
				['Z3', 'D7'], 
				['D5', 'Z1'], 
				['Z4', 'D9'],
				['Z1', 'D2'], 
				['D4', 'Z3'], 
				['D6', 'Z4'],
				['Z4', 'D8'], 
				['D4', 'Z1'], 
				['D6', 'Z2'],
				['Z2', 'D3'], 
				['Z1', 'D1'], 
				['D5', 'Z4'],
				# ['X1', 'D2'], 
				# ['D6', 'X2'], 
				# ['X3', 'D8'],
				# ['X1', 'D1'], 
				# ['D5', 'X2'], 
				# ['X3', 'D7'],
				# ['X4', 'D9'], 
				# ['D5', 'X3'], 
				# ['X2', 'D3'],
				# ['X4', 'D8'], 
				# ['D4', 'X3'], 
				# ['X2', 'D2'],
							]
		self.station = station
		self.create_dep_graph(Qubit_pairs=Qubit_pairs)

	def create_dep_graph(self, Qubit_pairs:list):
		'''
		Dependency graph for the calibration of 
		single-qubit gates.
		'''
		print(f'Creating dependency graph for two-qubit gate calibration')
		##############################
		# Grah nodes
		##############################
		module_name = 'pycqed.instrument_drivers.meta_instrument.Surface17_dependency_graph'
		

		# Single-qubit nodes
		Qubits = np.unique(np.array(Qubit_pairs).flatten())
		for q in Qubits:
			self.add_node(f'{q} Flux arc',
			    calibrate_function=module_name+'.Flux_arc_wrapper',
		        calibrate_function_args={
		          	'Qubit' : q,
		          	'station': self.station,
		          	})
		# Two-qubit nodes
		QL_detunings = {
			('Z1', 'D2') : 400e6,
			('Z1', 'D1') : 400e6,
			('Z4', 'D8') : 100e6,
			('Z4', 'D9') : 100e6,
			('X3', 'D7') : 100e6,
			('X3', 'D8') : 100e6,
		}
		for pair in Qubit_pairs:
			self.add_node(f'{pair[0]}, {pair[1]} Chevron',
			    calibrate_function=module_name+'.Chevron_wrapper',
		        calibrate_function_args={
		          	'qH' : pair[0],
		          	'qL' : pair[1],
		          	'station': self.station,
		          	'qL_det': QL_detunings[tuple(pair)] \
		          			  if tuple(pair) in QL_detunings.keys() else 0
		          			  })

			self.add_node(f'{pair[0]}, {pair[1]} SNZ tmid',
			    calibrate_function=module_name+'.SNZ_tmid_wrapper',
		        calibrate_function_args={
		          	'qH' : pair[0],
		          	'qL' : pair[1],
		          	'station': self.station
		          	})

			self.add_node(f'{pair[0]}, {pair[1]} SNZ AB',
			    calibrate_function=module_name+'.SNZ_AB_wrapper',
		        calibrate_function_args={
		          	'qH' : pair[0],
		          	'qL' : pair[1],
		          	'station': self.station
		          	})

			self.add_node(f'{pair[0]}, {pair[1]} Asymmetry',
			    calibrate_function=module_name+'.Asymmetry_wrapper',
		        calibrate_function_args={
		          	'qH' : pair[0],
		          	'qL' : pair[1],
		          	'station': self.station
		          	})

			self.add_node(f'{pair[0]}, {pair[1]} 1Q phase',
			    calibrate_function=module_name+'.Single_qubit_phase_calibration_wrapper',
		        calibrate_function_args={
		          	'qH' : pair[0],
		          	'qL' : pair[1],
		          	'station': self.station
		          	})
			
			self.add_node(f'{pair[0]}, {pair[1]} 2Q IRB',
			    calibrate_function=module_name+'.TwoQ_Randomized_benchmarking_wrapper',
		        calibrate_function_args={
		          	'qH' : pair[0],
		          	'qL' : pair[1],
		          	'station': self.station
		          	})

		# Save snpashot
		self.add_node('Save snapshot',
              calibrate_function=module_name+'.save_snapshot_metadata',
              calibrate_function_args={
              	'station': self.station,
              	})

		##############################
		# Node depdendencies
		##############################
		for Q_pair in Qubit_pairs:
			self.add_edge('Save snapshot',
						  f'{Q_pair[0]}, {Q_pair[1]} 2Q IRB')

			self.add_edge(f'{Q_pair[0]}, {Q_pair[1]} 2Q IRB',
			              f'{Q_pair[0]}, {Q_pair[1]} 1Q phase')
			
			self.add_edge(f'{Q_pair[0]}, {Q_pair[1]} 1Q phase',
			              f'{Q_pair[0]}, {Q_pair[1]} Asymmetry')

			self.add_edge(f'{Q_pair[0]}, {Q_pair[1]} Asymmetry',
			              f'{Q_pair[0]}, {Q_pair[1]} SNZ AB')

			self.add_edge(f'{Q_pair[0]}, {Q_pair[1]} SNZ AB',
			              f'{Q_pair[0]}, {Q_pair[1]} SNZ tmid')

			self.add_edge(f'{Q_pair[0]}, {Q_pair[1]} SNZ tmid',
			              f'{Q_pair[0]}, {Q_pair[1]} Chevron')

			self.add_edge(f'{Q_pair[0]}, {Q_pair[1]} Chevron',
			              f'{Q_pair[0]} Flux arc')
			self.add_edge(f'{Q_pair[0]}, {Q_pair[1]} Chevron',
			              f'{Q_pair[1]} Flux arc')

		##############################
		# Create graph
		##############################
		self.cfg_plot_mode = 'svg'
		self.update_monitor()
		self.cfg_svg_filename
		url = self.open_html_viewer()
		print('Dependency graph created at ' + url)


def Cryoscope_wrapper(Qubit, station, detuning=None, update_FIRs=False):
	'''
	Wrapper function for measurement of Cryoscope.
	This will update the required polynomial coeficients
	for detuning to voltage conversion.
	'''
	# Set gate duration
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=140,
	                              init_duration=200000)
	# Setup measurement
	Q_inst = station.components[Qubit]
	Q_inst.ro_acq_averages(2**10)
	Q_inst.ro_acq_weight_type('optimal')
	station.components['QInspire_MC'].live_plot_enabled(False)
	station.components['QInspire_nMC'].live_plot_enabled(False)
	# Q_inst.prepare_readout()
	# Set microwave lutman
	Q_mlm = Q_inst.instr_LutMan_MW.get_instr()
	Q_mlm.set_inspire_lutmap()
	# Q_mlm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
	# Set flux lutman
	Q_flm = Q_inst.instr_LutMan_Flux.get_instr()
	Q_flm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
	# Q_inst.prepare_for_timedomain()
	# Find amplitudes corresponding to specified frequency detunings
	# if there are existing polycoefs, try points at specified detunings
	if detuning == None:
		# if Qubit in ['D4', 'D5', 'D6']:
		# 	detuning = 600e6
		# else:
		detuning = 400e6 # QNW
	if all(Q_flm.q_polycoeffs_freq_01_det() != None):
		sq_amp = get_DAC_amp_frequency(detuning, Q_flm)
	# else:
	# 	sq_amp = .5
	Q_flm.sq_amp(sq_amp)

	device = station.components['device']
	device.ro_acq_weight_type('optimal')
	device.measure_cryoscope(
		qubits=[Qubit],
		times = np.arange(0e-9, 100e-9, 1/2.4e9), # np.arange(0e-9, 5e-9, 1/2.4e9)
		wait_time_flux = 20,
		update_FIRs = update_FIRs)
	# If not successful after 3 attempts fail node
	return True


def Flux_arc_wrapper(Qubit, station):
	'''
	Wrapper function for measurement of flux arcs.
	This will update the required polynomial coeficients
	for detuning to voltage conversion.
	'''
	# Set gate duration
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=60,
	                              init_duration=200000)
	# Setup measurement
	Q_inst = station.components[Qubit]
	Q_inst.ro_acq_averages(2**7)
	Q_inst.ro_acq_weight_type('optimal')
	station.components['QInspire_MC'].live_plot_enabled(False)
	station.components['QInspire_nMC'].live_plot_enabled(False)
	# Q_inst.prepare_readout()
	# Set microwave lutman
	Q_mlm = Q_inst.instr_LutMan_MW.get_instr()
	Q_mlm.set_inspire_lutmap()
	# Q_mlm.set_default_lutmap()
	# Q_mlm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
	# Set flux lutman
	Q_flm = Q_inst.instr_LutMan_Flux.get_instr()
	Q_flm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
	Q_inst.prepare_for_timedomain()
	# Find amplitudes corresponding to specified frequency detunings
	# if there are existing polycoefs, try points at specified detunings
	if Qubit in ['QNW', 'QNE']:
		Detunings = [600e6, 200e6]
		# Detunings = [600e6, 400e6, 200e6]
	else:
		Detunings = [600e6, 400e6]
		# Detunings = [900e6, 700e6, 500e6]
	if all(Q_flm.q_polycoeffs_freq_01_det() != None):
		# Amps = [-0.28, -0.18, 0.18, 0.28] # QC
		# Amps = [-0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4, 0.5] # QSW, QSE
		Amps = [-0.4, -0.35, -0.3, 0.3, 0.35, 0.4] # QNW, QNE
		# Amps = [ 0, 0, 0, 0, 0, 0]
		# for j, det in enumerate(Detunings):
		# 	sq_amp = get_DAC_amp_frequency(det, Q_flm)
		# 	Amps[j] = -sq_amp
		# 	Amps[-(j+1)] = sq_amp
	# If not, try some random amplitudes
	else:
		Amps = [-0.4, -0.2, 0.2, 0.4] # [-0.18, -0.1, 0.1, 0.18]
	Amps = [-0.4, -0.30, -0.25, -0.2, 0.2, 0.25, 0.30, 0.4] # QSE
	# Amps = [-0.4, -0.35, -0.3, -0.25, 0.25, 0.3, 0.35, 0.4] # QNW
	print(Amps)
	# Measure flux arc
	for i in range(2):
		a = Q_inst.calibrate_flux_arc(
			Amplitudes=Amps,
			Times = np.arange(20e-9, 40e-9, 1/2.4e9),
			update=True,
			disable_metadata=True,
			prepare_for_timedomain=False)
		max_freq = np.max(a.proc_data_dict['Freqs'])
		# If flux arc spans 750 MHz
		if max_freq>np.max(Detunings)-150e6:
			return True
		# Expand scan range to include higher frequency
		else:
			for j, det in enumerate(Detunings):
				sq_amp = get_DAC_amp_frequency(det, Q_flm)
				Amps[j] = -sq_amp
				Amps[-(j+1)] = sq_amp
	# If not successful after 3 attempts fail node
	return False


def Chevron_wrapper(qH, qL, station,
					avoided_crossing:str = '11-02',
					qL_det: float = 0,
					park_distance: float = 700e6):
	'''
	Wrapper function for measurement of Chevrons.
	Using voltage to detuning information, we predict the 
	amplitude of the interaction for the desired avoided 
	crossing and measure a chevron within frequency range.
	Args:
		qH: High frequency qubit.
		qL: Low frequency qubit.
		avoided crossing: "11-02" or "11-20" 
						  (in ascending detuning order)
		qL_det: Detuning of low frequency qubit. This 
				feature is used to avoid spurious TLSs.
		park_distance: Minimum (frequency) distance of
					   parked qubits to low-frequency 
					   qubit. 
	'''
	# Set gate duration
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=2000,
	                              flux_pulse_duration=60,
	                              init_duration=200000)
	# Setup for measurement
	# station.components['QInspire_MC'].live_plot_enabled(True)
	station.components['QInspire_MC'].live_plot_enabled(False)
	station.components['QInspire_nMC'].live_plot_enabled(False)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	# Change waveform durations
	flux_lm_H.cfg_max_wf_length(60e-9)
	flux_lm_L.cfg_max_wf_length(60e-9)
	flux_lm_H.AWG.get_instr().reset_waveforms_zeros()
	flux_lm_L.AWG.get_instr().reset_waveforms_zeros()
	# Set amplitude
	flux_lm_H.sq_amp(.5)
	# Set frequency of low frequency qubit
	if qL_det < 10e6:
		sq_amp_L = 0 # avoids error near 0 in the flux arc.
	else:
		dircts = get_gate_directions(qH, qL)
		flux_lm_L.set(f'q_freq_10_{dircts[1]}', qL_det)
		sq_amp_L = get_DAC_amp_frequency(qL_det, flux_lm_L)
	flux_lm_L.sq_amp(sq_amp_L)
	flux_lm_L.sq_length(60e-9)
	for lm in [flux_lm_H, flux_lm_L]:
		load_single_waveform_on_HDAWG(lm, wave_id='square')
	# Set frequency of parked qubits
	park_freq = Q_L.freq_qubit()-qL_det-park_distance
	for q in Park_dict[(qH, qL)]:
		Q_inst = station.components[q]
		flux_lm_p = Q_inst.instr_LutMan_Flux.get_instr()
		park_det = Q_inst.freq_qubit()-park_freq
		# Only park if the qubit is closer than then 350 MHz
		if park_det>20e6:
			sq_amp_park = get_DAC_amp_frequency(park_det, flux_lm_p)
			flux_lm_p.sq_amp(sq_amp_park)
		else:
			flux_lm_p.sq_amp(0)
		flux_lm_p.sq_length(60e-9)
		load_single_waveform_on_HDAWG(flux_lm_p, wave_id='square')
	# Estimate avoided crossing amplitudes
	f_H, a_H = Q_H.freq_qubit(), Q_H.anharmonicity()
	f_L, a_L = Q_L.freq_qubit(), Q_L.anharmonicity()
	detuning_11_02, detuning_11_20 = \
		calculate_avoided_crossing_detuning(f_H, f_L, a_H, a_L)
	# Estimating scan ranges based on frequency range
	scan_range = 200e6
	if avoided_crossing == '11-02':
		_det = detuning_11_02
	elif avoided_crossing == '11-20':
		_det = detuning_11_20
	A_range = []
	for r in [-scan_range/2, scan_range/2]: # [scan_range/2, scan_range]:
		_ch_amp = get_Ch_amp_frequency(_det+r+qL_det, flux_lm_H)
		A_range.append(_ch_amp)
	# Perform measurement of 11_02 avoided crossing
	device = station.components['device']
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**9)
	# !PROBLEM! prepare for readout is not enough 
	# for wtv reason, need to look into this!
	# device.prepare_readout(qubits=[qH, qL])
	device.prepare_for_timedomain(qubits=[qH, qL], bypass_flux=False)
	park_qubits = Park_dict[(qH, qL)]+[qL]
	device.measure_chevron(
	    q0=qH,
	    q_spec=qL,
	    amps=np.linspace(A_range[0], A_range[1], 21),
	    q_parks=park_qubits,
	    lengths=np.linspace(10, 60, 21) * 1e-9,
	    target_qubit_sequence='excited',
	    waveform_name="square",
	    prepare_for_timedomain=False,
	    disable_metadata=True,
	    recover_q_spec = True,
	)
	# Change waveform durations
	flux_lm_H.cfg_max_wf_length(40e-9)
	flux_lm_L.cfg_max_wf_length(40e-9)
	flux_lm_H.AWG.get_instr().reset_waveforms_zeros()
	flux_lm_L.AWG.get_instr().reset_waveforms_zeros()
	# Run analysis
	a = ma2.tqg.Chevron_Analysis(
				 QH_freq=Q_H.freq_qubit(),
				 QL_det=qL_det,
                 avoided_crossing=avoided_crossing,
                 Out_range=flux_lm_H.cfg_awg_channel_range(),
                 DAC_amp=flux_lm_H.sq_amp(),
                 Poly_coefs=flux_lm_H.q_polycoeffs_freq_01_det())
	# Update flux lutman parameters
	dircts = get_gate_directions(qH, qL)
	# tp of SNZ
	tp = a.qoi['Tp']
	tp_dig = np.ceil((tp/2)*2.4e9)*2/2.4e9
	flux_lm_H.set(f'vcz_time_single_sq_{dircts[0]}', tp_dig/2)
	flux_lm_L.set(f'vcz_time_single_sq_{dircts[1]}', tp_dig/2)
	# detuning frequency of interaction
	flux_lm_H.set(f'q_freq_10_{dircts[0]}', a.qoi['detuning_freq'])
	flux_lm_L.set(f'q_freq_10_{dircts[1]}', qL_det)
	return True


def SNZ_tmid_wrapper(qH, qL, station,
					 park_distance: float = 700e6):
	'''
	Wrapper function for measurement of of SNZ landscape.
	Using voltage to detuning information, we set the 
	amplitude of the interaction based on previous updated
	values of qubit detunings (q_freq_10_<direction>) from
	Chevron measurement.
	Args:
		qH: High frequency qubit.
		qL: Low frequency qubit.
		park_distance: Minimum (frequency) distance of
					   parked qubits to low-frequency
	'''
	# Set gate duration
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=2000,
	                              flux_pulse_duration=40,
	                              init_duration=200000)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	station.components['QInspire_MC'].live_plot_enabled(False)
	station.components['QInspire_nMC'].live_plot_enabled(False)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	flux_lm_H.set(f'vcz_amp_sq_{dircts[0]}', 1)
	flux_lm_H.set(f'vcz_amp_fine_{dircts[0]}', 0.5)
	flux_lm_H.set(f'vcz_amp_dac_at_11_02_{dircts[0]}', 0.5)
	# Set frequency of low frequency qubit
	qL_det = flux_lm_L.get(f'q_freq_10_{dircts[1]}') # detuning at gate
	if qL_det < 10e6:
		sq_amp_L = 0 # avoids error near 0 in the flux arc.
	else:
		sq_amp_L = get_DAC_amp_frequency(qL_det, flux_lm_L)
	flux_lm_L.set(f'vcz_amp_sq_{dircts[1]}', 1)
	flux_lm_L.set(f'vcz_amp_fine_{dircts[1]}', 0)
	flux_lm_L.set(f'vcz_amp_dac_at_11_02_{dircts[1]}', sq_amp_L)
	# Set frequency of parked qubits
	park_freq = Q_L.freq_qubit()-qL_det-park_distance
	for q in Park_dict[(qH, qL)]:
		Q_inst = station.components[q]
		flux_lm_p = Q_inst.instr_LutMan_Flux.get_instr()
		park_det = Q_inst.freq_qubit()-park_freq
		# Only park if the qubit is closer than then 350 MHz
		if park_det>20e6:
			amp_park = get_DAC_amp_frequency(park_det, flux_lm_p)
			flux_lm_p.park_amp(amp_park)
		else:
			flux_lm_p.park_amp(0)
	# Estimating scan ranges based on frequency range
	scan_range = 40e6
	_det = flux_lm_H.get(f'q_freq_10_{dircts[0]}') # detuning at gate
	A_range = []
	for r in [-scan_range/2, scan_range/2]:
		_ch_amp = get_Ch_amp_frequency(_det+r, flux_lm_H,
						   DAC_param=f'vcz_amp_dac_at_11_02_{dircts[0]}')
		A_range.append(_ch_amp)
	# Perform measurement of 11_02 avoided crossing
	device = station['device']
	device.ro_acq_averages(2**8)
	device.ro_acq_weight_type('optimal')
	device.prepare_for_timedomain(qubits=[qH, qL], bypass_flux=True)
	device.prepare_fluxing(qubits=[qH, qL]+Park_dict[(qH, qL)])
	device.measure_vcz_A_tmid_landscape(
		Q0 = [qH],
		Q1 = [qL],
		T_mids = np.arange(10), # change from 20 to 10 (RDC, 03-11-2023) 
		A_ranges = [A_range],
		A_points = 11,
		Q_parks = Park_dict[(qH, qL)],
		flux_codeword = 'cz',
		prepare_for_timedomain=False,
		flux_pulse_duration = 40e-9,
		disable_metadata=False)
	a = ma2.tqg.VCZ_tmid_Analysis(Q0=[qH], Q1=[qL],
          A_ranges=[A_range],
          Poly_coefs = [flux_lm_H.q_polycoeffs_freq_01_det()],
          DAC_amp = flux_lm_H.get(f'vcz_amp_dac_at_11_02_{dircts[0]}'),
          Out_range = flux_lm_H.cfg_awg_channel_range(),
          Q0_freq = Q_H.freq_qubit(),
          label=f'VCZ_Amp_vs_Tmid_{[qH]}_{[qL]}_{Park_dict[(qH, qL)]}')
	opt_det, opt_tmid = a.qoi['opt_params_0']
	# Set new interaction frequency
	flux_lm_H.set(f'q_freq_10_{dircts[0]}', opt_det)
	# round tmid to th sampling point
	opt_tmid = np.round(opt_tmid) # RDC added / 2 (3-11-2023)
	# Set optimal timing SNZ parameters
	Flux_lm_ps = [ device.find_instrument(q).instr_LutMan_Flux.get_instr()\
				   for q in Park_dict[(qH, qL)] ]
	tmid_swf = swf.flux_t_middle_sweep(
		fl_lm_tm =  [flux_lm_H, flux_lm_L], 
		fl_lm_park = Flux_lm_ps,
		which_gate = list(dircts),
		duration = 40e-9,
		t_pulse = [flux_lm_H.get(f'vcz_time_single_sq_{dircts[0]}')*2])
	tmid_swf.set_parameter(opt_tmid)
	return True


def SNZ_AB_wrapper(qH, qL, station,
				   park_distance: float = 700e6):
	'''
	Wrapper function for measurement of of SNZ landscape.
	Using voltage to detuning information, we set the 
	amplitude of the interaction based on previous updated
	values of qubit detunings (q_freq_10_<direction>) from
	Chevron measurement.
	Args:
		qH: High frequency qubit.
		qL: Low frequency qubit.
		park_distance: Minimum (frequency) distance of
					   parked qubits to low-frequency
	'''
	# Set gate duration
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=2000,
	                              flux_pulse_duration=40,
	                              init_duration=200000)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	station.components['QInspire_MC'].live_plot_enabled(False)
	station.components['QInspire_nMC'].live_plot_enabled(False)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	flux_lm_H.set(f'vcz_amp_sq_{dircts[0]}', 1)
	flux_lm_H.set(f'vcz_amp_fine_{dircts[0]}', 0.5)
	flux_lm_H.set(f'vcz_amp_dac_at_11_02_{dircts[0]}', 0.5)
	# Set frequency of low frequency qubit
	qL_det = flux_lm_L.get(f'q_freq_10_{dircts[1]}') # detuning at gate
	if qL_det < 10e6:
		sq_amp_L = 0 # avoids error near 0 in the flux arc.
	else:
		sq_amp_L = get_DAC_amp_frequency(qL_det, flux_lm_L)
	flux_lm_L.set(f'vcz_amp_sq_{dircts[1]}', 1)
	flux_lm_L.set(f'vcz_amp_fine_{dircts[1]}', 0)
	flux_lm_L.set(f'vcz_amp_dac_at_11_02_{dircts[1]}', sq_amp_L)
	# Set frequency of parked qubits
	park_freq = Q_L.freq_qubit()-qL_det-park_distance
	for q in Park_dict[(qH, qL)]:
		Q_inst = station.components[q]
		flux_lm_p = Q_inst.instr_LutMan_Flux.get_instr()
		park_det = Q_inst.freq_qubit()-park_freq
		# Only park if the qubit is closer than then 350 MHz
		if park_det>20e6:
			amp_park = get_DAC_amp_frequency(park_det, flux_lm_p)
			flux_lm_p.park_amp(amp_park)
		else:
			flux_lm_p.park_amp(0)
	# Estimating scan ranges based on frequency range
	scan_range = 30e6
	_det = flux_lm_H.get(f'q_freq_10_{dircts[0]}') # detuning at gate
	A_range = []
	for r in [-scan_range/2, scan_range/2]:
		_ch_amp = get_Ch_amp_frequency(_det+r, flux_lm_H,
						   DAC_param=f'vcz_amp_dac_at_11_02_{dircts[0]}')
		A_range.append(_ch_amp)
	# Perform measurement of 11_02 avoided crossing
	device = station['device']
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**8)
	device.prepare_for_timedomain(qubits=[qH, qL], bypass_flux=True)
	device.prepare_fluxing(qubits=[qH, qL]+Park_dict[(qH, qL)])
	device.measure_vcz_A_B_landscape(
		Q0 = [qH],
		Q1 = [qL],
		B_amps = np.linspace(0, 1, 15),
		A_ranges = [A_range],
		A_points = 15,
		Q_parks = Park_dict[(qH, qL)],
		flux_codeword = 'cz',
		update_flux_params = False,
		prepare_for_timedomain=False,
		disable_metadata=False)
	# Run frequency based analysis
	a = ma2.tqg.VCZ_B_Analysis(Q0=[qH], Q1=[qL],
			A_ranges=[A_range],
			directions=[dircts],
			Poly_coefs = [flux_lm_H.q_polycoeffs_freq_01_det()],
			DAC_amp = flux_lm_H.get(f'vcz_amp_dac_at_11_02_{dircts[0]}'),
			Out_range = flux_lm_H.cfg_awg_channel_range(),
			Q0_freq = Q_H.freq_qubit(),
			tmid = flux_lm_H.get(f'vcz_time_middle_{dircts[0]}'),
			label=f'VCZ_Amp_vs_B_{[qH]}_{[qL]}_{Park_dict[(qH, qL)]}')
	tp_factor = a.qoi['tp_factor_0']
	tmid_H = flux_lm_H.get(f'vcz_time_middle_{dircts[0]}')*2.4e9
	Flux_lm_ps = [ device.find_instrument(q).instr_LutMan_Flux.get_instr()\
				   for q in Park_dict[(qH, qL)] ]
	if tp_factor<0.98:
		tp = flux_lm_H.get(f'vcz_time_single_sq_{dircts[0]}')
		tp_dig = (np.ceil((tp)*2.4e9)+2)/2.4e9
		flux_lm_H.set(f'vcz_time_single_sq_{dircts[0]}', tp_dig)
		flux_lm_L.set(f'vcz_time_single_sq_{dircts[1]}', tp_dig)
		return False
	elif tp_factor>1.2:
		tp = flux_lm_H.get(f'vcz_time_single_sq_{dircts[0]}')
		tp_dig = (np.ceil((tp)*2.4e9)-1)/2.4e9
		flux_lm_H.set(f'vcz_time_single_sq_{dircts[0]}', tp_dig)
		flux_lm_L.set(f'vcz_time_single_sq_{dircts[1]}', tp_dig)
		return False
	else:
		flux_lm_H.set(f'q_freq_10_{dircts[0]}', a.qoi[f'Optimal_det_{qH}'])
		flux_lm_H.set(f'vcz_amp_fine_{dircts[0]}', a.qoi[f'Optimal_amps_{qH}'][1])
		return True


def Asymmetry_wrapper(qH, qL, station):
	'''
	Wrapper function for fine-tuning SS using asymr of the SNZ pulse. 
	returns True.
	'''
	# Set gate duration
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=40,
	                              init_duration=200000)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	mw_lutman_H = Q_H.instr_LutMan_MW.get_instr()
	mw_lutman_L = Q_L.instr_LutMan_MW.get_instr()
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	# Set DAC amplitude for 2Q gate 
	det_qH = flux_lm_H.get(f'q_freq_10_{dircts[0]}')
	det_qL = flux_lm_L.get(f'q_freq_10_{dircts[1]}')
	amp_qH = get_DAC_amp_frequency(det_qH, flux_lm_H)
	amp_qL = get_DAC_amp_frequency(det_qL, flux_lm_L)
	for i, det, amp, flux_lm in zip([        0,			1],
								 	[	det_qH,    det_qL],
								 	[	amp_qH,    amp_qL],
								 	[flux_lm_H, flux_lm_L]):
		if det < 20e6:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', 0)
		else:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', amp)
	# Set preparation params
	device = station['device']
	flux_cw = 'cz'
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**10)
	# Prepare readout
	device.prepare_readout(qubits=[qH, qL])
	# Load flux waveforms
	load_single_waveform_on_HDAWG(flux_lm_H, f'cz_{dircts[0]}')
	load_single_waveform_on_HDAWG(flux_lm_L, f'cz_{dircts[1]}')
	for mw1 in [mw_lutman_H, mw_lutman_L]:
	    mw1.load_phase_pulses_to_AWG_lookuptable()
	flux_lm_H.set(f'vcz_use_asymmetric_amp_{dircts[0]}',True)
	# Choose asymetry range
	if 'D' in qH:
		asymmetries = np.linspace(-.005, .005, 7)
	else:

		asymmetries = np.linspace(-.002, .002, 7)
	# Measure
	device.calibrate_vcz_asymmetry( 
	    Q0 = qH, 
	    Q1 = qL,
	    prepare_for_timedomain=False,
	    Asymmetries = asymmetries,
	    Q_parks = Park_dict[(qH,qL)],
	    update_params = True,
	    flux_codeword = 'cz',
	    disable_metadata = True)
	device.prepare_fluxing(qubits=[qH])
	return True


def Single_qubit_phase_calibration_wrapper(qH, qL, station):
	'''
	Wrapper function for fine-tunig CP 180 phase, SQ phase updates of 360, and verification. 
	Returns True if successful calibration otherwise
	returns False.
	'''
	# Set gate duration
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=2000,
	                              flux_pulse_duration=40,
	                              init_duration=200000)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	station.components['QInspire_MC'].live_plot_enabled(False)
	station.components['QInspire_nMC'].live_plot_enabled(False)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	mw_lutman_H = Q_H.instr_LutMan_MW.get_instr()
	mw_lutman_L = Q_L.instr_LutMan_MW.get_instr()
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	# Set DAC amplitude for 2Q gate 
	det_qH = flux_lm_H.get(f'q_freq_10_{dircts[0]}')
	det_qL = flux_lm_L.get(f'q_freq_10_{dircts[1]}')
	amp_qH = get_DAC_amp_frequency(det_qH, flux_lm_H)
	amp_qL = get_DAC_amp_frequency(det_qL, flux_lm_L)
	for i, det, amp, flux_lm in zip([        0,			1],
								 	[	det_qH,    det_qL],
								 	[	amp_qH,    amp_qL],
								 	[flux_lm_H, flux_lm_L]):
		if det < 20e6:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', 0)
		else:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', amp)
	# Set preparation params
	device = station['device']
	flux_cw = 'cz'
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**10)
	# Prepare readout
	device.prepare_for_timedomain(qubits=[qH, qL])
	# Load flux waveforms
	# device.prepare_fluxing(qubits=[qH, qL])
	# load_single_waveform_on_HDAWG(flux_lm_H, f'cz_{dircts[0]}')
	# load_single_waveform_on_HDAWG(flux_lm_L, f'cz_{dircts[1]}')
	# Check if mw phase pulses are uploaded
	for lutman in [mw_lutman_H, mw_lutman_L]:
		lutmap =  lutman.LutMap()
		if lutmap[32]['name'] != 'rPhi90':
			lutman.load_phase_pulses_to_AWG_lookuptable()
	###################################
	# SQ phase update
	###################################
	device.measure_parity_check_ramsey(
	    Q_target = [qH],
	    Q_control = [qL],
	    flux_cw_list = [flux_cw],
	    prepare_for_timedomain = False,
	    downsample_angle_points = 3,
	    update_mw_phase=True,
	    mw_phase_param=f'vcz_virtual_q_ph_corr_{dircts[0]}',
	    disable_metadata=True)
	device.measure_parity_check_ramsey(
	    Q_target = [qL],
	    Q_control = [qH],
	    flux_cw_list = [flux_cw],
	    prepare_for_timedomain = False,
	    downsample_angle_points = 3,
	    update_mw_phase=True,
	    mw_phase_param=f'vcz_virtual_q_ph_corr_{dircts[1]}',
	    disable_metadata=True)
	mw_lutman_H.upload_single_qubit_phase_corrections()
	mw_lutman_L.upload_single_qubit_phase_corrections()
	###################################
	# Verification 
	###################################
	# device.measure_conditional_oscillation(q0 = qH, q1=qL,
	#                                         disable_metadata=True)
	# device.measure_conditional_oscillation(q0 = qL, q1=qH,
	#                                         disable_metadata=True)
	return True


def TwoQ_Randomized_benchmarking_wrapper(qH, qL, station):
	'''
	Wrapper function around Randomized benchmarking.
	Returns True if successful calibration otherwise
	returns False.
	'''
	# Set gate duration
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=800,
	                              flux_pulse_duration=40,
	                              init_duration=200000)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	mw_lutman_H = Q_H.instr_LutMan_MW.get_instr()
	mw_lutman_L = Q_L.instr_LutMan_MW.get_instr()
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	# Set DAC amplitude for 2Q gate 
	det_qH = flux_lm_H.get(f'q_freq_10_{dircts[0]}')
	det_qL = flux_lm_L.get(f'q_freq_10_{dircts[1]}')
	amp_qH = get_DAC_amp_frequency(det_qH, flux_lm_H)
	amp_qL = get_DAC_amp_frequency(det_qL, flux_lm_L)
	for i, det, amp, flux_lm in zip([       0,			1],
								 	[	det_qH,    det_qL],
								 	[	amp_qH,    amp_qL],
								 	[flux_lm_H, flux_lm_L]):
		if det < 20e6:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', 0)
		else:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', amp)
	# Prepare device 
	device = station['device']
	flux_cw = 'cz'
	device.ro_acq_weight_type('optimal IQ')
	device.ro_acq_averages(2**10)
	# Set preparation params 
	mw_lutman_H.set_default_lutmap()
	mw_lutman_L.set_default_lutmap()
	device.prepare_for_timedomain(qubits=[qH, qL], bypass_flux=True)
	# Load flux waveforms
	load_single_waveform_on_HDAWG(flux_lm_H, f'cz_{dircts[0]}')
	load_single_waveform_on_HDAWG(flux_lm_L, f'cz_{dircts[1]}')
	# measurement
	device.measure_two_qubit_interleaved_randomized_benchmarking(
	    qubits = [qH, qL],
	    nr_seeds = 20,
	    measure_idle_flux = False,
	    prepare_for_timedomain=False,
	    recompile=False,
	    nr_cliffords = np.array([1., 3., 5., 7., 9., 11., 15.,
	                             20., 30., 50.]),
	    flux_codeword = flux_cw)
	return True


def TLS_density_wrapper(qubit,
						station,
						qubit_parks = None,
		                detuning = None,
						two_qubit_gate_duration = 40e-9,
		                max_duration = 60e-9):
	'''
	Wrapper function for measurement of TLS density.
	Using a dynamical square pulse to flux the qubit
	away while parking park_qubits.
	Args:
	    qubit: fluxed qubit.
	    park_qubits: list of parked qubits.
	'''
	if qubit_parks == None:
		qubit_parks = {
			'QNW': ['QC'], # There was QC
			'QNE': ['QC'],
			'QC':  ['QSW', 'QSE'],
			'QSW': [],
			'QSE': [],
		}
	# Setup for measurement
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	device = station.components['device']
	Flux_lm_q = station.components[qubit].instr_LutMan_Flux.get_instr()
	det_0 = Flux_lm_q.q_polycoeffs_freq_01_det()[-1]+20e6
	if np.any(detuning) == None:
		detuning = np.arange(det_0+20e6, 1500e6, 5e6)
	# Convert detuning to list of amplitudes
	Flux_lm_q.sq_amp(0.5)
	Amps = np.real([ get_Ch_amp_frequency(det, Flux_lm_q, DAC_param='sq_amp')\
			 for  det in detuning ])
	# Check parking qubits if needed and set the right parking distance.  
	Parked_qubits = qubit_parks[qubit]
	# set parking amps for parked qubits. 
	if not Parked_qubits:
		print('no parking qubits are defined')
	else:
		# Handle frequency of parked qubits
		for i, q_park in enumerate(Parked_qubits):
			Q_park = station.components[q_park]
			# minimum allowed detuning
			minimum_detuning = 600e6
			f_q = station.components[qubit].freq_qubit()
			f_q_min = f_q-detuning[-1]
			# required parked qubit frequency
			f_q_park = f_q_min-minimum_detuning
			det_q_park = Q_park.freq_qubit() - f_q_park
			fl_lm_park = Q_park.instr_LutMan_Flux.get_instr()
			if det_q_park > 10e6:
				park_amp = get_DAC_amp_frequency(det_q_park, fl_lm_park)
			else:
				park_amp = 0
			fl_lm_park.sq_amp(park_amp)
			fl_lm_park.sq_length(max_duration)
			if max_duration > two_qubit_gate_duration:
				fl_lm_park.cfg_max_wf_length(max_duration)
				fl_lm_park.AWG.get_instr().reset_waveforms_zeros()
	# prepare for timedomains
	if max_duration > two_qubit_gate_duration:
		Flux_lm_q.cfg_max_wf_length(max_duration)
		Flux_lm_q.AWG.get_instr().reset_waveforms_zeros()
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**8)
	device.ro_acq_digitized(True)
	# device.prepare_readout(qubits=[qubit, 'QC'])
	# device.ro_acq_digitized(False)
	if not Parked_qubits:
		Parked_qubits = []
	if qubit == 'C':
		spectator_qubit = 'NW'
	else:
		spectator_qubit = 'C'
	device.prepare_for_timedomain(qubits=[qubit, spectator_qubit], bypass_flux=True)
	device.prepare_fluxing(qubits=[qubit, spectator_qubit]+Parked_qubits)
	device.measure_chevron(
	    q0=qubit,
	    q_spec=spectator_qubit,
	    amps=Amps,
	    q_parks=Parked_qubits,
	    lengths= np.linspace(10e-9, max_duration, 6),
	    target_qubit_sequence='ground',
	    waveform_name="square",
	    # buffer_time=40e-9,
	    prepare_for_timedomain=False,
	    disable_metadata=True,
	)
	# Reset waveform durations
	if max_duration > two_qubit_gate_duration:
		Flux_lm_q.cfg_max_wf_length(two_qubit_gate_duration)
		Flux_lm_q.AWG.get_instr().reset_waveforms_zeros()
		if not Parked_qubits:
			print('no parking qubits are defined')
		else:
			for q_park in Parked_qubits:
				fl_lm_park = Q_park.instr_LutMan_Flux.get_instr()
				fl_lm_park.cfg_max_wf_length(two_qubit_gate_duration)
				fl_lm_park.AWG.get_instr().reset_waveforms_zeros()
	# Run landscape analysis
	interaction_freqs = { 
		d : Flux_lm_q.get(f'q_freq_10_{d}')\
		for d in ['NW', 'NE', 'SW', 'SE']\
		if 2e9 > Flux_lm_q.get(f'q_freq_10_{d}') > 10e6
		}
	isparked = False
	flux_lm_qpark = None
	q0 = 'SW'
	q1 = 'SE'
	print(qubit)
	if qubit == q0 or qubit == q1:
		isparked = True
		flux_lm_qpark = station.components[qubit].instr_LutMan_Flux.get_instr()
	a = ma2.tqg.TLS_landscape_Analysis(
				Q_freq = station.components[qubit].freq_qubit(),
				Out_range=Flux_lm_q.cfg_awg_channel_range(),
				DAC_amp=Flux_lm_q.sq_amp(),
				Poly_coefs=Flux_lm_q.q_polycoeffs_freq_01_det(),
				interaction_freqs=interaction_freqs,
				flux_lm_qpark = flux_lm_qpark,
				isparked = isparked)
	return True

# Dictionary for necessary parking for each interaction
Park_dict = {
			 ('QNW', 'QC'): [],
			 ('QNE', 'QC'): [],
			 ('QC', 'QSW'): ['QSE'],
			 ('QC', 'QSE'): ['QSW'],
			 }
###########################################
# Helper functions for theory predictions #
###########################################
def transmon_hamiltonian(n, Ec, Ej, phi=0, ng=0):
    Ej_f = Ej*np.abs(np.cos(np.pi*phi))
    I = np.diag((np.arange(-n-ng,n+1-ng)-0)**2,k=0)
    D = np.diag(np.ones(2*n),k=1) + np.diag(np.ones(2*n),k=-1)
    return 4*Ec*I-Ej_f/2*D

def solve_hamiltonian(EC, EJ, phi=0, ng=0, n_level=1):
    n = 10
    H = transmon_hamiltonian(n, EC, EJ, phi=phi, ng=ng)
    eigvals, eigvec = np.linalg.eigh(H)
    eigvals -= eigvals[0]
    freq_1 = eigvals[n_level]
    freq_2 = eigvals[n_level+1]
    return freq_1, freq_2

from scipy.optimize import minimize
def find_transmon_params(f0, a0):
    # Define cost function to minimize
    def cost_func(param):
        EC, EJ = param
        EC *= 1e6 # Needed for optimizer to converge
        EJ *= 1e9 #
        n = 10
        H = transmon_hamiltonian(n, EC, EJ, phi=0)
        eigvals, eigvec = np.linalg.eigh(H)
        eigvals -= eigvals[0]
        freq = eigvals[1]
        anha = eigvals[2]-2*eigvals[1]
        return (freq-f0)**2 + (anha-a0)**2
    # Run minimizer and record values
    Ec, Ej = minimize(cost_func, x0=[300, 15], options={'disp':True}).x
    Ec *= 1e6
    Ej *= 1e9
    return Ec, Ej

def calculate_avoided_crossing_detuning(f_H, f_L, a_H, a_L):
    Ec_H, Ej_H = find_transmon_params(f_H, a_H)
    Phi = np.linspace(0, .4, 21)
    E02 = np.ones(21)
    E11 = np.ones(21)
    for i, p in enumerate(Phi):
        E1, E2 = solve_hamiltonian(Ec_H, Ej_H, phi=p, ng=0, n_level=1)
        E02[i] = E2
        E11[i] = E1+f_L
    p_02 = np.poly1d(np.polyfit(Phi, E02, deg=2))
    p_11 = np.poly1d(np.polyfit(Phi, E11, deg=2))
    # detuning of 11-02
    phi_int_1 = np.max((p_02-p_11).roots)
    detuning_1 = p_11(0)-p_11(phi_int_1)
    # detuning of 11-20
    f_20 = 2*f_L+a_L
    phi_int_2 = np.max((p_11-f_20).roots)
    detuning_2 = p_11(0)-p_11(phi_int_2)
    return detuning_1, detuning_2

def get_frequency_waveform(wave_par, flux_lutman):
	'''
	Calculate detuning of waveform.
	'''
	poly_coefs = flux_lutman.q_polycoeffs_freq_01_det()
	out_range = flux_lutman.cfg_awg_channel_range()
	ch_amp = flux_lutman.cfg_awg_channel_amplitude()
	dac_amp = flux_lutman.get(wave_par)
	out_volt = dac_amp*ch_amp*out_range/2
	poly_func = np.poly1d(poly_coefs)
	freq = poly_func(out_volt)
	return np.real(freq)

def get_DAC_amp_frequency(freq, flux_lutman):
	'''
	Function to calculate DAC amp corresponding 
	to frequency detuning.
	'''
	poly_coefs = flux_lutman.q_polycoeffs_freq_01_det()
	out_range = flux_lutman.cfg_awg_channel_range()
	ch_amp = flux_lutman.cfg_awg_channel_amplitude()
	poly_func = np.poly1d(poly_coefs)
	out_volt = max((poly_func-freq).roots)
	sq_amp = out_volt/(ch_amp*out_range/2)
	# Safe check in case amplitude exceeds maximum
	if sq_amp>1:
		print(f'WARNING had to increase gain of {flux_lutman.name} to {ch_amp}!')
		flux_lutman.cfg_awg_channel_amplitude(ch_amp*1.5)
		# Can't believe Im actually using recursion!!!
		sq_amp = get_DAC_amp_frequency(freq, flux_lutman)
	return np.real(sq_amp)

def get_Ch_amp_frequency(freq, flux_lutman, DAC_param='sq_amp'):
	'''
	Function to calculate channel gain corresponding 
	to frequency detuning.
	'''
	poly_coefs = flux_lutman.q_polycoeffs_freq_01_det()
	out_range = flux_lutman.cfg_awg_channel_range()
	dac_amp = flux_lutman.get(DAC_param)
	poly_func = np.poly1d(poly_coefs)
	out_volt = max((poly_func-freq).roots)
	ch_amp = out_volt/(dac_amp*out_range/2)
	return np.real(ch_amp)

def load_single_waveform_on_HDAWG(lutman, wave_id):
    """
    Load a single waveform on HDAWG
    Args:
        regenerate_waveforms (bool): if True calls
            generate_standard_waveforms before uploading.
        stop_start           (bool): if True stops and starts the AWG.
    """
    AWG = lutman.AWG.get_instr()
    AWG.stop()
    for idx, waveform in lutman.LutMap().items():
        lutman.load_waveform_onto_AWG_lookuptable(
            wave_id=wave_id, regenerate_waveforms=True)
    lutman.cfg_awg_channel_amplitude()
    lutman.cfg_awg_channel_range()
    AWG.start()

def plot_wave_dicts(qH: list, 
					qL: list,
					station, 
					label =''):

	
	plt.close('all')
	Q_Hs = [station.components[Q] for Q in qH]
	Q_Ls = [station.components[Q] for Q in qL]
	flux_lm_Hs = [Q_inst.instr_LutMan_Flux.get_instr() for Q_inst in Q_Hs]
	flux_lm_Ls = [Q_inst.instr_LutMan_Flux.get_instr() for Q_inst in Q_Ls]
	n_colors = 2*len(flux_lm_Hs)+6
	cmap = plt.get_cmap("tab10", n_colors)
	
	fig, ax = plt.subplots(figsize=(9,5), dpi=120)
	ax2 = ax.twiny()
	ax.set_title(f"Plot waveforms {qH}_{qL}", y=1.1, fontsize=14)
	for i,Q in enumerate(Q_Hs):
		dircts = get_gate_directions(Q.name, Q_Ls[i].name)
		ax.plot(flux_lm_Hs[i]._wave_dict_dist[f'cz_{dircts[0]}'],
	            linestyle='-', linewidth=1.5,marker = '.',
	            markersize=5, color=cmap(i), label=f'{Q.name}-{dircts[0]}')
		ax.plot(flux_lm_Ls[i]._wave_dict_dist[f'cz_{dircts[1]}'],
	            linestyle='--', linewidth=1.5,
	            markersize=8, color=cmap(i+len(flux_lm_Hs)), label=f'{Q_Ls[i].name}_{dircts[1]}')
		for j,q in enumerate(Park_dict[Q.name, Q_Ls[i].name]):
			if q not in qH+qL: 
				ax.plot(station.components[q].instr_LutMan_Flux.get_instr()._wave_dict_dist[f'park'],
					linestyle='-', linewidth=1,markersize=3,alpha = 0.6,
					color=cmap(j+i+1+len(flux_lm_Hs)), label=f'{q}_Park')	

		ax.axhline(0.5, color='k', ls=':', alpha=0.8)
		ax.axhline(-0.5, color='k', ls=':', alpha=0.8)
		ax.axhline(0, color='k', ls=':', alpha=0.8)
		max_len = len(flux_lm_Hs[i]._wave_dict_dist[f'cz_{dircts[0]}'])
		ax.set_xticks(np.arange(0, max_len+1, 8))
		ax.set_xlabel("Duration (sampling points)", fontsize=12)
		ax.set_yticks(np.arange(-0.5,0.51,0.1))
		ax.set_ylabel("Amplitude (a.u.)", fontsize=12)
	    # set ticks of top axis according to tick positions of bottom axis,
	    # but with units of ns
		ax2.set_xlim(ax.get_xlim())
		ax2.set_xticks(np.arange(0, max_len+1, 8))
		ax2.set_xticklabels([f"{t:.1f}" for t in 1/2.4 * np.arange(0, max_len+1, 8)],
	                        fontsize=8)
		ax2.set_xlabel("Duration (ns)", fontsize=12)

		ax.grid(True)
		ax.legend(loc='upper right', fontsize=12)
	
	plt.tight_layout()
	# plt.savefig(r"D:\Experiments\202208_Uran\Figures" + fr"\Flux_Pulses_{label}_{qH}_{qL}.png", format='png')
	plt.show()
	plt.close('all')
