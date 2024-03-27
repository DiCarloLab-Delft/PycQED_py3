import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import autodepgraph
reload(autodepgraph)
from autodepgraph import AutoDepGraph_DAG
from pycqed.measurement import hdf5_data as h5d
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.utilities.general import get_gate_directions, get_nearest_neighbors,\
									 get_parking_qubits
from pycqed.measurement import sweep_functions as swf
import pycqed.instrument_drivers.library.DIO as DIO
from pycqed.utilities.general import check_keyboard_interrupt, print_exception
from pycqed.instrument_drivers.meta_instrument.device_object_CCL import DeviceCCL as Device
from pycqed.instrument_drivers.meta_instrument.qubit_objects.CCL_Transmon import CCLight_Transmon as Transmon
from pycqed.instrument_drivers.meta_instrument.LutMans.flux_lutman_vcz import HDAWG_Flux_LutMan as FluxLutMan
from pycqed.qce_utils.control_interfaces.connectivity_surface_code import Repetition9Layer, QubitIDObj
from pycqed.analysis.tools import cryoscope_tools as ct
from pycqed.measurement import detector_functions
###############################################################################
# Single- and Two- qubit gate calibration graph
###############################################################################
import os
import logging
import pycqed as pq
from pycqed.measurement.openql_experiments import generate_CC_cfg as gc
input_file = os.path.join(pq.__path__[0], 'measurement',
                          'openql_experiments', 'config_cc_s17_direct_iq.json.in')
config_fn = os.path.join(pq.__path__[0], 'measurement',
                       'openql_experiments', 'output_cc_s17','config_cc_s17_direct_iq.json')
logging.basicConfig(level=logging.INFO)

class Full_calibration(AutoDepGraph_DAG):
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
		print(f'Creating dependency graph for full gate calibration')
		##############################
		# Grah nodes
		##############################
		module_name = 'pycqed.instrument_drivers.meta_instrument.Surface17_dependency_graph'
		
		##########################################################################
		# Single qubit Graph
		##########################################################################
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

		self.add_node(f'Save snapshot single-qubit',
              calibrate_function=module_name+'.save_snapshot_metadata',
              calibrate_function_args={
              	'station': self.station,
              	})
		for qubit in Qubits:
			self.add_edge(f'Save snapshot single-qubit',
			              f'{qubit} Randomized Benchmarking')


		##########################################################################
		# Two qubit Graph
		##########################################################################
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
			['X1', 'D2'], 
			['D6', 'X2'], 
			['X3', 'D8'],
			['X1', 'D1'], 
			['D5', 'X2'], 
			['X3', 'D7'],
			['X4', 'D9'], 
			['D5', 'X3'], 
			['X2', 'D3'],
			['X4', 'D8'], 
			['D4', 'X3'], 
			['X2', 'D2'],
			]
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
			# After detuning search (Yuejie&Sean)
			('D5', 'X2') : 60e6,
			('D5', 'X3') : 110e6,
			('Z1', 'D1') : 380e6,
			('Z1', 'D2') : 310e6,
			('X1', 'D1') : 0e6,
			('X3', 'D8') : 60e6,
			('Z4', 'D8') : 100e6,
			('X1', 'D2') : 165e6,	# After second round search
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
		self.add_node('Save snapshot two-qubit',
              calibrate_function=module_name+'.save_snapshot_metadata',
              calibrate_function_args={
              	'station': self.station,
              	'Two_qubit_freq_trajectories': True
              	})

		##############################
		# Node depdendencies
		##############################
		for Q_pair in Qubit_pairs:
			self.add_edge('Save snapshot two-qubit',
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

		#############################
		# Final dependency
		#############################
		for q in Qubits:
			self.add_edge(f'{q} Flux arc',
				          f'Save snapshot single-qubit')

		##########################################################################
		# Parity checks Graph
		##########################################################################
		stabilizers = ['Z1', 'Z2', 'Z3', 'Z4',
					   # 'X1', 'X2', 'X3', 'X4',
					  ]

		for stab in stabilizers:
			self.add_node(f'{stab} Horizontal calibration',
				calibrate_function=module_name+'.Horizontal_calibration_wrapper',
				calibrate_function_args={
					'stabilizer_qubit': stab,
					'station': self.station
					})

			self.add_node(f'{stab} Ancilla phase verification',
				calibrate_function=module_name+'.Measure_parity_check_phase_wrapper',
				calibrate_function_args={
					'stabilizer_qubit': stab,
					'station': self.station
					})

			self.add_node(f'{stab} Data-qubit phase correction',
				calibrate_function=module_name+'.Data_qubit_phase_calibration_wrapper',
				calibrate_function_args={
					'stabilizer_qubit': stab,
					'station': self.station
					})

			self.add_node(f'{stab} Parity assignment fidelity',
				calibrate_function=module_name+'.Parity_check_fidelity_wrapper',
				calibrate_function_args={
					'stabilizer_qubit': stab,
					'station': self.station
					})

			self.add_node(f'{stab} Parity repeatability',
				calibrate_function=module_name+'.Parity_check_repeatability_wrapper',
				calibrate_function_args={
					'stabilizer_qubit': stab,
					'station': self.station
					})

		# Save snpashot
		self.add_node('Save snapshot parity-checks',
				calibrate_function=module_name+'.save_snapshot_metadata',
				calibrate_function_args={
					'station': self.station,
					'parity_check': True,
					})
		##############################
		# Node depdendencies
		##############################
		for stab in stabilizers:
			self.add_edge('Save snapshot parity-checks',
						  f'{stab} Parity repeatability')
			
			self.add_edge(f'{stab} Parity repeatability',
						  f'{stab} Parity assignment fidelity')
			
			self.add_edge(f'{stab} Parity assignment fidelity',
						  f'{stab} Data-qubit phase correction')

			self.add_edge(f'{stab} Data-qubit phase correction',
						  f'{stab} Ancilla phase verification')

			self.add_edge(f'{stab} Ancilla phase verification',
						  f'{stab} Horizontal calibration')

			self.add_edge(f'{stab} Horizontal calibration',
						  f'Save snapshot two-qubit')

		##############################
		# Create graph
		##############################
		self.cfg_plot_mode = 'svg'
		self.update_monitor()
		self.cfg_svg_filename
		url = self.open_html_viewer()
		print('Dependency graph created at ' + url)

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
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst = station.components[qubit]
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal')
	Q_inst.ro_acq_averages(2**10)
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
	# Q_inst.prepare_for_timedomain()
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
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst = station.components[qubit]
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
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst = station.components[qubit]
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal')
	Q_inst.ro_acq_averages(2**12)
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
	file_cfg = gc.generate_config(in_filename=input_file,
                                  out_filename=config_fn,
                                  mw_pulse_duration=20,
                                  ro_duration=600,
                                  flux_pulse_duration=40,
                                  init_duration=200000)
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst = station.components[qubit]
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
		post_select=True,
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
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst = station.components[qubit]
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
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst = station.components[qubit]
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
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst = station.components[qubit]
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
		nr_cliffords=2**np.arange(11),
	    nr_seeds=15,
	    recompile=False, 
	    prepare_for_timedomain=False,
	    disable_metadata=False)
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


###############################################################################
# Two qubit gate calibration graph
###############################################################################
import os
import pycqed as pq
from pycqed.measurement.openql_experiments import generate_CC_cfg as gc
input_file = os.path.join(pq.__path__[0], 'measurement',
                          'openql_experiments', 'config_cc_s17_direct_iq.json.in')
config_fn = os.path.join(pq.__path__[0], 'measurement',
                       'openql_experiments', 'output_cc_s17','config_cc_s17_direct_iq.json')

TWOQ_GATE_DURATION = 60e-9
TWOQ_GATE_DURATION_NS = 60

OFFSET_QUBITS = []  # ['X2', 'X3', 'X4', 'D7', 'D9']

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
			('Z1', 'D2') : 250e6,#400e6,
			('Z1', 'D1') : 400e6,
			('Z4', 'D8') : 100e6,
			# ('Z4', 'D9') : 100e6,
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


def Cryoscope_wrapper(Qubit, station, detuning=None, 
					  update_IIRs=False,
					  update_FIRs=False,
					  max_duration: float = 100e-9, **kw):
	'''
	Wrapper function for measurement of Cryoscope.
	This will update the required polynomial coeficients
	for detuning to voltage conversion.
	'''
	# Set gate duration
	flux_duration_ns = int(max_duration*1e9) + 100
	# flux_duration_ns = int(max_duration*1e9) 
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=flux_duration_ns,
	                              init_duration=200000)
	if 'live_plot_enabled' in kw.keys():
		_live_plot = kw['live_plot_enabled']
	else:
		_live_plot = False
	station.components['MC'].live_plot_enabled(_live_plot)
	station.components['nested_MC'].live_plot_enabled(_live_plot)
	# Setup measurement
	Q_inst = station.components[Qubit]
	# Q_inst.prepare_readout()
	# Set microwave lutman
	Q_mlm = Q_inst.instr_LutMan_MW.get_instr()
	Q_mlm.set_default_lutmap()
	# Q_mlm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
	# Set flux lutman
	Q_flm = Q_inst.instr_LutMan_Flux.get_instr()
	if max_duration > TWOQ_GATE_DURATION:
		Q_flm.cfg_max_wf_length(max_duration)
		Q_flm.AWG.get_instr().reset_waveforms_zeros()
	Q_flm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
	Q_inst.prepare_for_timedomain()
	# Find amplitudes corresponding to specified frequency detunings
	# if there are existing polycoefs, try points at specified detunings
	if detuning == None:
		if Qubit in ['D4', 'D5', 'D6']:
			detuning = 600e6
		else:
			detuning = 900e6
	# TODO: Commented out because we want to start from default condition
	if all(Q_flm.q_polycoeffs_freq_01_det() != None):
		sq_amp = get_DAC_amp_frequency(detuning, Q_flm, 
							negative_amp=True if Qubit in OFFSET_QUBITS else False)
	else:
		sq_amp = .5
	# sq_amp = .5
	if 'sq_amp' in kw:
		sq_amp = kw['sq_amp']
	
	Q_flm.sq_amp(sq_amp)
	if sq_amp < 0:
		print('Using negative amp')
	device = station.components['device']
	if 'ro_acq_averages' in kw.keys():
		avg = kw['ro_acq_averages']
	else:
		avg = 2**10
	device.ro_acq_averages(avg)
	device.ro_acq_weight_type('optimal')
	device.measure_cryoscope(
		qubits=[Qubit],
		times = np.arange(0e-9, max_duration, 1/2.4e9),
		wait_time_flux = 40,
		update_FIRs = update_FIRs,
		update_IIRs = update_IIRs)
	# Reset wavform duration
	if max_duration > TWOQ_GATE_DURATION:
		Q_flm.cfg_max_wf_length(TWOQ_GATE_DURATION)
		Q_flm.AWG.get_instr().reset_waveforms_zeros()
	return True


def Flux_arc_wrapper(Qubit, station,
                     Detunings: list = None,
                     fix_zero_detuning:bool=True, 
                     Amps = None, 
                     repetitions = 2**10):
	'''
	Wrapper function for measurement of flux arcs.
	This will update the required polynomial coeficients
	for detuning to voltage conversion.
	'''
	# Set gate duration
	TQG_duration_ns = TWOQ_GATE_DURATION_NS 
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=TQG_duration_ns,
	                              init_duration=200000)
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Setup measurement
	Q_inst = station.components[Qubit]
	Q_inst.ro_acq_averages(repetitions)
	Q_inst.ro_acq_weight_type('optimal')
	# Q_inst.prepare_readout()
	# Set microwave lutman
	Q_mlm = Q_inst.instr_LutMan_MW.get_instr()
	Q_mlm.set_default_lutmap()
	# Q_mlm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
	# Set flux lutman
	Q_flm = Q_inst.instr_LutMan_Flux.get_instr()
	check_flux_wf_duration(Q_flm) # (legacy)
	Q_flm.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)
	Q_inst.prepare_for_timedomain()
	# Find amplitudes corresponding to specified frequency detunings
	# if there are existing polycoefs, try points at specified detunings
	if Detunings is None:
		if Qubit in ['D4', 'D5', 'D6']:
			Detunings = [600e6, 400e6, 200e6]
		else:
			Detunings = [900e6, 700e6, 500e6]
	
   	# TODO: Commented out because we want to start from default condition
	if Amps is None:
		if all(Q_flm.q_polycoeffs_freq_01_det() != None):
			# Amps = [ 0, 0, 0, 0, ]
			Amps = [ 0, 0, 0, 0, 0, 0]
			# To avoid updates to the channel gain during this step
			# we calculate all the amplitudes before setting them
			for det in Detunings:
				get_DAC_amp_frequency(det, Q_flm, negative_amp=True)
				get_DAC_amp_frequency(det, Q_flm)
			for j, det in enumerate(Detunings):
				Amps[j] = get_DAC_amp_frequency(det, Q_flm, negative_amp=True)
				Amps[-(j+1)] = get_DAC_amp_frequency(det, Q_flm)
		# If not, try some random amplitudes
	# else:
	# 	Amps = [-0.4, -0.35, -0.3, 0.3, 0.35, 0.4]


 
	# Measure flux arc
	for i in range(2):
		print(Amps)
		a = Q_inst.calibrate_flux_arc(
			Amplitudes=Amps,
			Times = np.arange(40e-9, 60e-9, 1/2.4e9),
			update=True,
			disable_metadata=True,
			prepare_for_timedomain=False,
			fix_zero_detuning=fix_zero_detuning)
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
					avoided_crossing: str = '11-02',
					qL_det: float = 0,
					park_distance: float = 700e6,
					negative_amp: bool = False,
					use_premeasured_values: bool = True,
					**kw):
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
	TQG_duration_ns = TWOQ_GATE_DURATION_NS
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=TQG_duration_ns,
	                              init_duration=200000)
	if 'live_plot_enabled' in kw.keys():
		_live_plot = kw['live_plot_enabled']
	else:
		_live_plot = False
	station.components['MC'].live_plot_enabled(_live_plot)
	station.components['nested_MC'].live_plot_enabled(_live_plot)
	# Setup for measurement
	device = station.components['device']
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**9)
	# Perform measurement of 11_02 avoided crossing
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	# For qubits off the sweet-spot, the amplitude should be negative
	if qH in OFFSET_QUBITS:
		negative_amp = True
	if negative_amp:
		flux_lm_H.sq_amp(-.5)
	else:
		flux_lm_H.sq_amp(.5)
	flux_lm_H.sq_delay(0)
	# Set frequency of low frequency qubit
	if abs(qL_det) < 10e6:
		sq_amp_L = 0 # avoids error near 0 in the flux arc.
	else:
		dircts = get_gate_directions(qH, qL)
		flux_lm_L.set(f'q_freq_10_{dircts[1]}', qL_det)
		sq_amp_L = get_DAC_amp_frequency(qL_det, flux_lm_L,
						 negative_amp=True if qL in OFFSET_QUBITS else False)
	flux_lm_L.sq_amp(sq_amp_L)
	flux_lm_L.sq_length(60e-9)
	flux_lm_L.sq_delay(0)
	# for lm in [flux_lm_H, flux_lm_L]:
		# load_single_waveform_on_HDAWG(lm, wave_id='square')
	device.prepare_fluxing(qubits = [qH, qL])
	# Set frequency of parked qubits
	park_freq = Q_L.freq_qubit()-qL_det-park_distance
	for q in get_parking_qubits(qH, qL):
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
		flux_lm_p.sq_delay(0)
		load_single_waveform_on_HDAWG(flux_lm_p, wave_id='square')
	# Estimate avoided crossing amplitudes
	f_H, a_H = Q_H.freq_qubit(), Q_H.anharmonicity()
	f_L, a_L = Q_L.freq_qubit(), Q_L.anharmonicity()
	detuning_11_02, detuning_11_20 = \
		calculate_avoided_crossing_detuning(f_H, f_L, a_H, a_L)
	# Estimating scan ranges based on frequency range
	if 'scan_range' not in kw:
		scan_range = 200e6
	else:
		scan_range = kw['scan_range']
	if avoided_crossing == '11-02':
		_det = detuning_11_02
	elif avoided_crossing == '11-20':
		_det = detuning_11_20
	A_range = []
	for r in [-scan_range/2, scan_range/2]:
		_ch_amp = get_Ch_amp_frequency(_det+r+qL_det, flux_lm_H)
		A_range.append(_ch_amp)
	# Known values (if these are given,
	# this measurement will be skipped)
	Qubit_pair_Tp = {
		('Z3', 'D7'): 2.125e-08,
		('D5', 'Z1'): 1.875e-08,
		('Z4', 'D9'): 2.2083333333333333e-08,
		('D4', 'Z3'): 1.75e-08,
		('D6', 'Z4'): 1.875e-08,
		('Z1', 'D2'): 2.4166666666666668e-08,
		('D4', 'Z1'): 1.75e-08,
		('D6', 'Z2'): 1.75e-08,
		('Z4', 'D8'): 2.0833333333333335e-08,
		('Z2', 'D3'): 2.1666666666666665e-08+1/2.4e9,
		('Z1', 'D1'): 54 / 2.4e9,
		('D5', 'Z4'): 1.875e-08,
		('X1', 'D1'): 2.0833333333333335e-08,  # (48 sampling points) -> Go to 50 sampling points (2.0833333333333335e-08)
		('X1', 'D2'): 2.2083333333333333e-08+2/2.4e9,
		('D5', 'X2'): 1.875e-08-2/2.4e9,
		('D6', 'X2'): 1.9583333333333333e-08-1/2.4e9,
		('D4', 'X3'): 2.2083333333333333e-08,
		('D5', 'X3'): 2.0416666666666668e-08,
		('X2', 'D2'): 2.0833333333333335e-08-1/2.4e9,  # Increased Tp time, because of undershoot
		('X2', 'D3'): 1.9583333333333333e-08-1/2.4e9,
		('X3', 'D7'): 2.0416666666666668e-08-1/2.4e9,
		('X3', 'D8'): 2.1666666666666665e-08-1/2.4e9,
		('X4', 'D8'): 2.0833333333333335e-08,
		('X4', 'D9'): 1.9583333333333333e-08,
	}
	# Run measurement
	# !PROBLEM! prepare for readout is not enough 
	# for wtv reason, need to look into this!
	# device.prepare_readout(qubits=[qH, qL])
	device.prepare_for_timedomain(qubits=[qH, qL], bypass_flux=False)
	park_qubits = get_parking_qubits(qH, qL)+[qL]
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
	)
	# run analysis
	a = ma2.tqg.Chevron_Analysis(
				 QH_freq=Q_H.freq_qubit(),
				 QL_det=qL_det,
				 avoided_crossing=avoided_crossing,
				 Out_range=flux_lm_H.cfg_awg_channel_range(),
				 DAC_amp=flux_lm_H.sq_amp(),
				 Poly_coefs=flux_lm_H.q_polycoeffs_freq_01_det())
	if ((qH, qL) in Qubit_pair_Tp.keys()) and use_premeasured_values:
		# Hardcoding optimal TPs
		print('Using pre-measured optimal values')
		# Update flux lutman parameters
		dircts = get_gate_directions(qH, qL)
		flux_lm_H.set(f'vcz_time_single_sq_{dircts[0]}', Qubit_pair_Tp[(qH, qL)])
		flux_lm_L.set(f'vcz_time_single_sq_{dircts[1]}', Qubit_pair_Tp[(qH, qL)])
	else:
		# Update flux lutman parameters
		dircts = get_gate_directions(qH, qL)
		# tp of SNZ
		tp = a.qoi['Tp']
		tp_dig = np.ceil((tp/2)*2.4e9)*2/2.4e9
		print('Find fitting value:', tp_dig/2, 's')
		# tp_dig += 2*2/2.4e9 # To prevent too short SNZ cases
		if [qH, qL] in [['Z1', 'D1'], ['Z2', 'D3'], ['D4', 'X3']]:
			tp_dig += 2*2/2.4e9	 # this should be removed later
		if [qH, qL] in [['Z3', 'D7']]:
			tp_dig += 3*2/2.4e9	 # this should be removed later
		if [qH, qL] in [['Z1', 'D2']]:
			tp_dig += 1*2/2.4e9	 # this should be removed later
		if qL_det > 200e6:
			tp_dig += 8/2.4e9
		flux_lm_H.set(f'vcz_time_single_sq_{dircts[0]}', tp_dig/2)
		flux_lm_L.set(f'vcz_time_single_sq_{dircts[1]}', tp_dig/2)
		print('Setting tp/2 to', flux_lm_H.get(f'vcz_time_single_sq_{dircts[0]}'), 's')
	# detuning frequency of interaction
	flux_lm_H.set(f'q_freq_10_{dircts[0]}', a.qoi['detuning_freq'])
	flux_lm_L.set(f'q_freq_10_{dircts[1]}', qL_det)
	return True


def SNZ_tmid_wrapper(qH, qL, station,
					 park_distance: float = 700e6,
					 apply_parking_settings: bool = True,
					 asymmetry_compensation: bool = False,
					 tmid_offset_samples: int = 0,
					 **kw):
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
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=TQG_duration_ns,
	                              init_duration=200000)
	if 'live_plot_enabled' in kw.keys():
		_live_plot = kw['live_plot_enabled']
	else:
		_live_plot = False
	station.components['MC'].live_plot_enabled(_live_plot)
	station.components['nested_MC'].live_plot_enabled(_live_plot)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	flux_lm_H.set(f'vcz_amp_sq_{dircts[0]}', 1)
	flux_lm_H.set(f'vcz_amp_fine_{dircts[0]}', 0.5)
	flux_lm_H.set(f'vcz_amp_dac_at_11_02_{dircts[0]}', 0.5)
	# For qubits off the sweet-spot, the amplitude should be negative
	if qH in OFFSET_QUBITS:
		flux_lm_H.set(f'vcz_amp_dac_at_11_02_{dircts[0]}', -0.5)
	# Set frequency of low frequency qubit
	qL_det = flux_lm_L.get(f'q_freq_10_{dircts[1]}') # detuning at gate
	if abs(qL_det) < 10e6:
		sq_amp_L = 0 # avoids error near 0 in the flux arc.
	else:
		sq_amp_L = get_DAC_amp_frequency(qL_det, flux_lm_L,
						 negative_amp=True if qL in OFFSET_QUBITS else False)
	flux_lm_L.set(f'vcz_amp_sq_{dircts[1]}', 1)
	flux_lm_L.set(f'vcz_amp_fine_{dircts[1]}', 0)
	flux_lm_L.set(f'vcz_amp_dac_at_11_02_{dircts[1]}', sq_amp_L)
	# Check waveform durations
	check_flux_wf_duration(flux_lm_H)
	check_flux_wf_duration(flux_lm_L)
	# Set frequency of parked qubits
	Parked_qubits = get_parking_qubits(qH, qL)
	if apply_parking_settings:
		park_freq = Q_L.freq_qubit()-qL_det-park_distance
		for q in Parked_qubits:
			Q_inst = station.components[q]
			flux_lm_p = Q_inst.instr_LutMan_Flux.get_instr()
			park_det = Q_inst.freq_qubit()-park_freq
			# Only park if the qubit is closer than <park_distance>
			if park_det>10e6:
				amp_park_pos = get_DAC_amp_frequency(park_det, flux_lm_p)
				amp_park_neg = get_DAC_amp_frequency(park_det, flux_lm_p, 
													 negative_amp=True)
				_Amps = [amp_park_pos, amp_park_neg]
				amp_park_idx = np.argmax(np.abs(_Amps))
				# Update parking amplitude in lookup table
				flux_lm_p.park_amp(_Amps[amp_park_idx])
			else:
				flux_lm_p.park_amp(0)
			# Check wf duration of park qubits
			check_flux_wf_duration(flux_lm_p)
	# Estimating scan ranges based on frequency range
	if 'scan_range' not in kw:
		scan_range = 40e6
	else:
		scan_range = kw['scan_range']
    
	if 'A_points' not in kw:
		A_points = 11
	else:
		A_points = kw['A_points']
	_det = flux_lm_H.get(f'q_freq_10_{dircts[0]}') # detuning at gate
	# Predict required gate asymetry
	if asymmetry_compensation:
		# We use the sq_amp to calculate positive and negative amps for the pulse.
		# (vcz_amp_dac_at_11_02 does not allow negative values).
		_amp = flux_lm_H.get(f'vcz_amp_dac_at_11_02_{dircts[0]}')
		flux_lm_H.sq_amp(+_amp)
		gain_high = get_Ch_amp_frequency(_det, flux_lm_H, DAC_param='sq_amp')
		flux_lm_H.sq_amp(-_amp)
		gain_low  = get_Ch_amp_frequency(_det, flux_lm_H, DAC_param='sq_amp')
		gain = (gain_high+gain_low)/2
		asymmetry = (gain_high-gain_low)/(gain_high+gain_low)
		flux_lm_H.set(f'vcz_use_asymmetric_amp_{dircts[0]}', True)
		flux_lm_H.set(f'vcz_asymmetry_{dircts[0]}', asymmetry)
		# flux_lm_H.set(f'cfg_awg_channel_amplitude', gain)
		# Set new detunning corresponding to average gain
		ch_amp_0 = get_Ch_amp_frequency(_det, flux_lm_H, DAC_param=f'sq_amp')
		delta_ch_amp_p = get_Ch_amp_frequency(_det+scan_range/2, flux_lm_H, DAC_param=f'sq_amp') - ch_amp_0
		delta_ch_amp_m = get_Ch_amp_frequency(_det-scan_range/2, flux_lm_H, DAC_param=f'sq_amp') - ch_amp_0
		A_range = [gain+delta_ch_amp_m, gain+delta_ch_amp_p]
	# Predict range without asymmetry
	else:
		A_range = []
		for r in [-scan_range/2, scan_range/2]:
			_ch_amp = get_Ch_amp_frequency(_det+r, flux_lm_H,
							   DAC_param=f'vcz_amp_dac_at_11_02_{dircts[0]}')
			A_range.append(_ch_amp)
	# Assess if unipolar pulse is required
	# if qH in OFFSET_QUBITS:
	# 	flux_lm_H.set(f'vcz_use_net_zero_pulse_{dircts[0]}', False)
	# 	# if working with asymmetric pulses
	# 	if asymmetry_compensation:
	# 		flux_lm_H.set(f'vcz_use_net_zero_pulse_{dircts[0]}', True)
	# 	else:
	# 		flux_lm_H.set(f'vcz_use_net_zero_pulse_{dircts[0]}', False)
	# 	# Setting pading amplitude to ensure net-zero waveform
	# 	make_unipolar_pulse_net_zero(flux_lm_H, f'cz_{dircts[0]}')
	# 	if tmid_offset_samples == 0:
	# 		tmid_offset_samples = 1
	# Perform measurement of 11_02 avoided crossing
	device = station['device']
	device.ro_acq_averages(2**8)
	device.ro_acq_weight_type('optimal')
	device.prepare_for_timedomain(qubits=[qH, qL], bypass_flux=True)
	device.prepare_fluxing(qubits=[qH, qL]+Parked_qubits)
	device.measure_vcz_A_tmid_landscape(
		Q0 = [qH],
		Q1 = [qL],
		T_mids = np.arange(10) + tmid_offset_samples,
		A_ranges = [A_range],
		A_points = A_points,
		Q_parks = Parked_qubits,
		flux_codeword = 'cz',
		flux_pulse_duration = TWOQ_GATE_DURATION,
		prepare_for_timedomain=False,
		disable_metadata=True)
	a = ma2.tqg.VCZ_tmid_Analysis(Q0=[qH], Q1=[qL],
		A_ranges=[A_range],
		Poly_coefs = [flux_lm_H.q_polycoeffs_freq_01_det()],
		DAC_amp = flux_lm_H.get(f'vcz_amp_dac_at_11_02_{dircts[0]}'),
		Out_range = flux_lm_H.cfg_awg_channel_range(),
		Q0_freq = Q_H.freq_qubit(),
		asymmetry = flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}')\
					if asymmetry_compensation else 0,
		label=f'VCZ_Amp_vs_Tmid_{[qH]}_{[qL]}_{Parked_qubits}')
	opt_det, opt_tmid = a.qoi['opt_params_0']
	# Set new interaction frequency
	flux_lm_H.set(f'q_freq_10_{dircts[0]}', opt_det)
	# round tmid to th sampling point
	opt_tmid = np.round(opt_tmid)
	# Set optimal timing SNZ parameters
	Flux_lm_ps = [ device.find_instrument(q).instr_LutMan_Flux.get_instr()\
				   for q in Parked_qubits ]
	tmid_swf = swf.flux_t_middle_sweep(
		fl_lm_tm =  [flux_lm_H, flux_lm_L], 
		fl_lm_park = Flux_lm_ps,
		which_gate = list(dircts),
		duration=TWOQ_GATE_DURATION,
		time_park=TWOQ_GATE_DURATION-(6/2.4e9),
		t_pulse = [flux_lm_H.get(f'vcz_time_single_sq_{dircts[0]}')*2])
	tmid_swf.set_parameter(opt_tmid)
	return True


def SNZ_AB_wrapper(qH, qL, station,
				   park_distance: float = 700e6,
				   apply_parking_settings: bool = True,
				   asymmetry_compensation: bool = False,
       			   flux_cw: str = 'cz',
				   **kw):
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
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=TQG_duration_ns,
	                              init_duration=200000)
	if 'live_plot_enabled' in kw.keys():
		_live_plot = kw['live_plot_enabled']
	else:
		_live_plot = False
	station.components['MC'].live_plot_enabled(_live_plot)
	station.components['nested_MC'].live_plot_enabled(_live_plot)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	flux_lm_H.set(f'vcz_amp_sq_{dircts[0]}', 1)
	flux_lm_H.set(f'vcz_amp_fine_{dircts[0]}', 0.5)
	flux_lm_H.set(f'vcz_amp_dac_at_11_02_{dircts[0]}', 0.5)
	# For qubits off the sweet-spot, the amplitude should be negative
	if qH in OFFSET_QUBITS:
		flux_lm_H.set(f'vcz_amp_dac_at_11_02_{dircts[0]}', -0.5)

	# Assess if unipolar pulse is required
	# if qH in OFFSET_QUBITS:
	# 	# For qubits off the sweet-spot, the amplitude should be negative
	# 	flux_lm_H.set(f'vcz_amp_dac_at_11_02_{dircts[0]}', -0.5)
	# 	flux_lm_H.set(f'vcz_use_net_zero_pulse_{dircts[0]}', False)
	# 	# # if working with asymmetric pulses
	# 	if asymmetry_compensation:
	# 		flux_lm_H.set(f'vcz_use_net_zero_pulse_{dircts[0]}', True)
	# 	else:
	# 		flux_lm_H.set(f'vcz_use_net_zero_pulse_{dircts[0]}', False)
	# 	# Setting pading amplitude to ensure net-zero waveform
	# 	make_unipolar_pulse_net_zero(flux_lm_H, f'cz_{dircts[0]}')
	# Set frequency of low frequency qubit
 
	qL_det = flux_lm_L.get(f'q_freq_10_{dircts[1]}') # detuning at gate
	if abs(qL_det) < 10e6:
		sq_amp_L = 0 # avoids error near 0 in the flux arc.
	else:
		sq_amp_L = get_DAC_amp_frequency(qL_det, flux_lm_L,
						 negative_amp=True if qL in OFFSET_QUBITS else False)
	flux_lm_L.set(f'vcz_amp_sq_{dircts[1]}', 1)
	flux_lm_L.set(f'vcz_amp_fine_{dircts[1]}', 0)
	flux_lm_L.set(f'vcz_amp_dac_at_11_02_{dircts[1]}', sq_amp_L)
	# Check waveform durations
	check_flux_wf_duration(flux_lm_H)
	check_flux_wf_duration(flux_lm_L)
	# Set frequency of parked qubits
	Parked_qubits = get_parking_qubits(qH, qL)
	if apply_parking_settings:
		park_freq = Q_L.freq_qubit()-qL_det-park_distance
		for q in Parked_qubits:
			Q_inst = station.components[q]
			flux_lm_p = Q_inst.instr_LutMan_Flux.get_instr()
			park_det = Q_inst.freq_qubit()-park_freq
			# Only park if the qubit is closer than then 350 MHz
			if park_det>10e6:
				amp_park_pos = get_DAC_amp_frequency(park_det, flux_lm_p)
				amp_park_neg = get_DAC_amp_frequency(park_det, flux_lm_p,
													 negative_amp=True)
				_Amps = [amp_park_pos, amp_park_neg]
				amp_park_idx = np.argmax(np.abs(_Amps))
				# Update parking amplitude in lookup table
				flux_lm_p.park_amp(_Amps[amp_park_idx])
			else:
				flux_lm_p.park_amp(0)
			# Check wf duration of park qubits
			check_flux_wf_duration(flux_lm_p)
	# Estimating scan ranges based on frequency range
	scan_range = 30e6
	_det = flux_lm_H.get(f'q_freq_10_{dircts[0]}') # detuning at gate
	# Predict required range taking into account pulse asymmetry
	if asymmetry_compensation:
		# We use the sq_amp to calculate positive and negative amps for the pulse.
		# (vcz_amp_dac_at_11_02 does not allow negative values).
		_amp = flux_lm_H.get(f'vcz_amp_dac_at_11_02_{dircts[0]}')
		flux_lm_H.sq_amp(+_amp)
		gain_high = get_Ch_amp_frequency(_det, flux_lm_H, DAC_param='sq_amp')
		flux_lm_H.sq_amp(-_amp)
		gain_low  = get_Ch_amp_frequency(_det, flux_lm_H, DAC_param='sq_amp')
		gain = (gain_high+gain_low)/2
		# Set new detunning corresponding to average gain
		ch_amp_0 = get_Ch_amp_frequency(_det, flux_lm_H, DAC_param=f'sq_amp')
		delta_ch_amp_p = get_Ch_amp_frequency(_det+scan_range/2, flux_lm_H,
											  DAC_param=f'sq_amp') - ch_amp_0
		delta_ch_amp_m = get_Ch_amp_frequency(_det-scan_range/2, flux_lm_H,
											  DAC_param=f'sq_amp') - ch_amp_0
		A_range = [gain+delta_ch_amp_m, gain+delta_ch_amp_p]
	# Predict range without asymmetry
	else:
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
	device.prepare_fluxing(qubits=[qH, qL]+Parked_qubits)
	device.measure_vcz_A_B_landscape(
		Q0 = [qH],
		Q1 = [qL],
		B_amps = np.linspace(0, 1, 15),
		A_ranges = [A_range],
		A_points = 15,
		Q_parks = Parked_qubits,
		flux_codeword = flux_cw,
		update_flux_params = False,
		prepare_for_timedomain=False,
		disable_metadata=True)
	# Run frequency based analysis
	a = ma2.tqg.VCZ_B_Analysis(Q0=[qH], Q1=[qL],
			A_ranges=[A_range],
			directions=[dircts],
			Poly_coefs = [flux_lm_H.q_polycoeffs_freq_01_det()],
			DAC_amp = flux_lm_H.get(f'vcz_amp_dac_at_11_02_{dircts[0]}'),
			Out_range = flux_lm_H.cfg_awg_channel_range(),
			Q0_freq = Q_H.freq_qubit(),
			asymmetry = flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}')\
						if asymmetry_compensation else 0,
			tmid = flux_lm_H.get(f'vcz_time_middle_{dircts[0]}'),
			label=f'VCZ_Amp_vs_B_{[qH]}_{[qL]}_{Parked_qubits}')
	# Set optimal gate params
	flux_lm_H.set(f'q_freq_10_{dircts[0]}', a.qoi[f'Optimal_det_{qH}'])
	flux_lm_H.set(f'vcz_amp_fine_{dircts[0]}', a.qoi[f'Optimal_amps_{qH}'][1])
	return True


def Unipolar_wrapper(qH, qL, station,
				   	 park_distance: float = 700e6,
				   	 apply_parking_settings: bool = True,
				   	 **kw):
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
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=TQG_duration_ns,
	                              init_duration=200000)
	if 'live_plot_enabled' in kw.keys():
		_live_plot = kw['live_plot_enabled']
	else:
		_live_plot = False
	station.components['MC'].live_plot_enabled(_live_plot)
	station.components['nested_MC'].live_plot_enabled(_live_plot)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	flux_lm_H.set('sq_amp', .5)
	flux_lm_H.sq_delay(6e-9)
	# Assess if unipolar pulse is required
	if qH in OFFSET_QUBITS:
		# For qubits off the sweet-spot, the amplitude should be negative
		flux_lm_H.set('sq_amp', -0.5)
	# Set frequency of low frequency qubit
	qL_det = flux_lm_L.get(f'q_freq_10_{dircts[1]}') # detuning at gate
	if abs(qL_det) < 10e6:
		sq_amp_L = 0 # avoids error near 0 in the flux arc.
	else:
		sq_amp_L = get_DAC_amp_frequency(qL_det, flux_lm_L)
	flux_lm_L.set('sq_amp', sq_amp_L)
	# Check waveform durations
	check_flux_wf_duration(flux_lm_H)
	check_flux_wf_duration(flux_lm_L)
	# Set frequency of parked qubits
	Parked_qubits = get_parking_qubits(qH, qL)
	if apply_parking_settings:
		park_freq = Q_L.freq_qubit()-qL_det-park_distance
		for q in Parked_qubits:
			Q_inst = station.components[q]
			flux_lm_p = Q_inst.instr_LutMan_Flux.get_instr()
			park_det = Q_inst.freq_qubit()-park_freq
			# Only park if the qubit is closer than then 350 MHz
			if park_det>20e6:
				amp_park_pos = get_DAC_amp_frequency(park_det, flux_lm_p)
				amp_park_neg = get_DAC_amp_frequency(park_det, flux_lm_p,
													 negative_amp=True)
				_Amps = [amp_park_pos, amp_park_neg]
				amp_park_idx = np.argmax(np.abs(_Amps))
				# Update parking amplitude in lookup table
				flux_lm_p.park_amp(_Amps[amp_park_idx])
			else:
				flux_lm_p.park_amp(0)
			# Check wf duration of park qubits
			check_flux_wf_duration(flux_lm_p)
	# Estimating scan ranges based on frequency range
	scan_range = 20e6
	_det = flux_lm_H.get(f'q_freq_10_{dircts[0]}') # detuning at gate
	# Predict range 
	A_range = []
	for r in [-scan_range/2, scan_range/2]:
		_ch_amp = get_Ch_amp_frequency(_det+r, flux_lm_H,
						   DAC_param='sq_amp')
		A_range.append(_ch_amp)
	# Perform measurement of 11_02 avoided crossing
	device = station['device']
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**8)
	device.prepare_for_timedomain(qubits=[qH, qL], bypass_flux=True)
	device.prepare_fluxing(qubits=[qH, qL]+Parked_qubits)
	device.measure_unipolar_A_t_landscape(
		Q0 = [qH],
		Q1 = [qL],
		times = np.linspace(35e-9, 50e-9, 15),
		A_ranges = [A_range],
		A_points = 15,
		Q_parks = Parked_qubits,
		flux_codeword = 'sf_square',
		update_flux_params = False,
		prepare_for_timedomain=False,
		disable_metadata=True)
	# Run frequency based analysis
	a = ma2.tqg.VCZ_B_Analysis(Q0=[qH], Q1=[qL],
			A_ranges=[A_range],
			directions=[dircts],
			Poly_coefs = [flux_lm_H.q_polycoeffs_freq_01_det()],
			DAC_amp = flux_lm_H.get('sq_amp'),
			Out_range = flux_lm_H.cfg_awg_channel_range(),
			Q0_freq = Q_H.freq_qubit(),
			l1_coef = .5,
			label=f'Unipolar_Amp_vs_t_{[qH]}_{[qL]}_{Parked_qubits}')
	# Set optimal gate params
	flux_lm_H.set(f'q_freq_10_{dircts[0]}', a.qoi[f'Optimal_det_{qH}'])
	flux_lm_H.set('sq_length', a.qoi[f'Optimal_amps_{qH}'][1])
	return True


def Asymmetry_wrapper(qH, qL, station, flux_cw: str = 'cz'):
	'''
	Wrapper function for fine-tuning SS using asymr of the SNZ pulse. 
	returns True.
	'''
	# Set gate duration
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=1000,
	                              flux_pulse_duration=TQG_duration_ns,
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
	amp_qH = get_DAC_amp_frequency(det_qH, flux_lm_H,
						 negative_amp=True if qH in OFFSET_QUBITS else False)
	amp_qL = get_DAC_amp_frequency(det_qL, flux_lm_L,
						 negative_amp=True if qL in OFFSET_QUBITS else False)
	# Compensate for asymmetry of cz pulse
	_asymmetry = flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}')
	if abs(_asymmetry)> .075:
		amp_qH = amp_qH/(1+flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}'))
	for i, det, amp, flux_lm in zip([        0,			1],
								 	[	det_qH,    det_qL],
								 	[	amp_qH,    amp_qL],
								 	[flux_lm_H, flux_lm_L]):
		if abs(det) < 10e6:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', 0)
		else:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', amp)
	# Set preparation params
	device = station['device']
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**11)  # 2**10
	# Prepare readout
	device.prepare_readout(qubits=[qH, qL])
	# Load flux waveforms
	load_single_waveform_on_HDAWG(flux_lm_H, f'cz_{dircts[0]}')
	load_single_waveform_on_HDAWG(flux_lm_L, f'cz_{dircts[1]}')
	for mw1 in [mw_lutman_H, mw_lutman_L]:
	    mw1.load_phase_pulses_to_AWG_lookuptable()
	flux_lm_H.set(f'vcz_use_asymmetric_amp_{dircts[0]}',True)
	# Estimating asymmetry ranges based on frequency range
	# if qH in ['X2', 'X3', 'X4']:
	if qH in ['D4', 'D5', 'D6']:
		_asym = flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}')
		asymmetries = np.linspace(-.5e-2, .5e-2, 7)+_asym
	else: 
		_asym = flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}')
		asymmetries = np.linspace(-.25e-2, .25e-2, 7)+_asym
	# 	scan_range = 10e6
	# 	_det = flux_lm_H.get(f'q_freq_10_{dircts[0]}') # detuning at gate
	# 	# Get DAC amplitudes for each detuning
	# 	sq_amp_0 = get_DAC_amp_frequency(_det, flux_lm_H)
	# 	sq_amp_1 = get_DAC_amp_frequency(_det+scan_range/2, flux_lm_H)
	# 	# Estimate asymmetry based required DAC amps
	# 	asymetry_r = 1 - sq_amp_1/sq_amp_0
	# 	asymmetries = np.linspace(-asymetry_r, asymetry_r, 7)
	# Measure

	device.calibrate_vcz_asymmetry( 
	    Q0 = qH, 
	    Q1 = qL,
	    prepare_for_timedomain=False,
	    Asymmetries = asymmetries,
	    Q_parks = get_parking_qubits(qH,qL),
	    update_params = True,
	    flux_codeword = flux_cw,
	    disable_metadata = True)
	device.prepare_fluxing(qubits=[qH])
	return True


def Single_qubit_phase_calibration_wrapper(qH, qL, station,
										   park_distance=700e6,
										   apply_parking_settings: bool = True,
										   fine_cphase_calibration: bool = False,
										   qSpectator: list = None,
										   pc_repetitions = 1):
	'''
	Wrapper function for fine-tunig CP 180 phase, SQ phase updates of 360, and verification. 
	Returns True if successful calibration otherwise
	returns False.
	'''
	# Set gate duration
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	# file_cfg = gc.generate_config(in_filename=input_file,
	#                               out_filename=config_fn,
	#                               mw_pulse_duration=20,
	#                               ro_duration=1000,
	#                               flux_pulse_duration=TQG_duration_ns,
	#                               init_duration=200000)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	mw_lutman_H = Q_H.instr_LutMan_MW.get_instr()
	mw_lutman_L = Q_L.instr_LutMan_MW.get_instr()
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	# Set DAC amplitude for 2Q gate 
	det_qH = flux_lm_H.get(f'q_freq_10_{dircts[0]}')
	det_qL = flux_lm_L.get(f'q_freq_10_{dircts[1]}')
	amp_qH = get_DAC_amp_frequency(det_qH, flux_lm_H, 
						negative_amp=True if qH in OFFSET_QUBITS else False)
	amp_qL = get_DAC_amp_frequency(det_qL, flux_lm_L,
						negative_amp=True if qL in OFFSET_QUBITS else False)
	# Compensate for asymmetry of cz pulse
	_asymmetry = flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}')
	if abs(_asymmetry)> .075:
		amp_qH = amp_qH/(1+flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}'))
	for i, det, amp, flux_lm in zip([        0,			1],
								 	[	det_qH,    det_qL],
								 	[	amp_qH,    amp_qL],
								 	[flux_lm_H, flux_lm_L]):
		if abs(det) < 10e6:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', 0)
		else:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', amp)
	# Assess if unipolar pulse is required
	# if qH in OFFSET_QUBITS:
	# 	# Setting pading amplitude to ensure net-zero waveform
	# 	make_unipolar_pulse_net_zero(flux_lm_H, f'cz_{dircts[0]}')
	# Set frequency of parked qubits
	qL_det = flux_lm_L.get(f'q_freq_10_{dircts[1]}') # detuning at gate
	Parked_qubits = get_parking_qubits(qH, qL)
	for q in Parked_qubits:
		park_freq = Q_L.freq_qubit()-qL_det-park_distance
		Q_inst = station.components[q]
		flux_lm_p = Q_inst.instr_LutMan_Flux.get_instr()
		park_det = Q_inst.freq_qubit()-park_freq
		print('park_det', park_det)
		if apply_parking_settings:
			# Only park if the qubit is closer than then 350 MHz
			if park_det>20e6:
				# Choose sign of parking waveform (necessary for off-sweetspot qubits)
				amp_park_pos = get_DAC_amp_frequency(park_det, flux_lm_p)
				amp_park_neg = get_DAC_amp_frequency(park_det, flux_lm_p, 
													 negative_amp=True)
				_Amps = [amp_park_pos, amp_park_neg]
				amp_park_idx = np.argmax(np.abs(_Amps))
				# Update parking amplitude in lookup table
				flux_lm_p.park_amp(_Amps[amp_park_idx])
				print('park_amp', flux_lm_p.park_amp())
			else:
				flux_lm_p.park_amp(0)
		load_single_waveform_on_HDAWG(flux_lm_p, 'park')
	# Set preparation params
	device = station['device']
	flux_cw = 'cz'
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**10)
	# Prepare readout
	if qSpectator is None:
		qSpectator = []
	qubits_awaiting_prepare = [qH, qL] + qSpectator
	if check_prepare_readout(qubits=qubits_awaiting_prepare, station=station):
		device.prepare_readout(qubits=qubits_awaiting_prepare)
	else:
		# if preparation is not deemed necessary try just updating detectors
		try:
			# acq_ch_map = device._acq_ch_map
			acq_ch_map = device._prep_ro_assign_weights(qubits=qubits_awaiting_prepare)
			# device._prep_ro_integration_weights(qubits=qubits)
			device._prep_ro_instantiate_detectors(qubits=qubits_awaiting_prepare, acq_ch_map=acq_ch_map)
		except:
			device.prepare_readout(qubits=qubits_awaiting_prepare)
	# device.prepare_readout(qubits=qubits_awaiting_prepare)
	# Load flux waveforms
	load_single_waveform_on_HDAWG(flux_lm_H, f'cz_{dircts[0]}')
	load_single_waveform_on_HDAWG(flux_lm_L, f'cz_{dircts[1]}')
	# Check waveform durations
	check_flux_wf_duration(flux_lm_H)
	check_flux_wf_duration(flux_lm_L)
	if apply_parking_settings:
		for q in Parked_qubits:
			flux_lm_p = Q_inst.instr_LutMan_Flux.get_instr()
			check_flux_wf_duration(flux_lm_p)


	# device.prepare_for_timedomain(qubits=[qH, qL])
	# Check if mw phase pulses are uploaded


	# prepare_for_parity_check('X4', station, Data_qubits=['D8', 'D9'])
	for q in [qH, qL]:
		Q = station.components[q]
		Q._prep_td_sources()
		if check_prepare_mw(q, station):
			Q.cfg_prepare_mw_awg(False)
			Q._prep_mw_pulses()
			Q.cfg_prepare_mw_awg(True)
			mw_lm = Q.instr_LutMan_MW.get_instr()
			mw_lm.set_default_lutmap()
			mw_lm.load_phase_pulses_to_AWG_lookuptable()
	# Calibrate conditional phase
	if fine_cphase_calibration:
		device.calibrate_parity_check_phase(
		    Q_ancilla = [qH],
		    Q_control = [qL],
		    Q_pair_target = [qH, qL],
		    # flux_cw_list = ['repetition_code_3', 'repetition_code_4'],
		    flux_cw_list = [flux_cw],
		    downsample_angle_points = 3,
		    prepare_for_timedomain = False,
		    update_mw_phase=False,
		    mw_phase_param = f'vcz_virtual_q_ph_corr_{dircts[0]}',
		    disable_metadata=True)
		load_single_waveform_on_HDAWG(flux_lm_H, f'cz_{dircts[0]}')
	# single-qubit phase of high frequency qubit
	device.measure_parity_check_ramsey(
	    Q_target = [qH],
	    Q_control = [qL],
     	Q_spectator=qSpectator,
	    # flux_cw_list = ['repetition_code_3', 'repetition_code_4'],
	    flux_cw_list = [flux_cw],
	    prepare_for_timedomain = False,
	    downsample_angle_points = 3,
	    update_mw_phase=True,
	    mw_phase_param=f'vcz_virtual_q_ph_corr_{dircts[0]}',
	    disable_metadata=True,
	    pc_repetitions=pc_repetitions)
	# Calibrate low frequency qubit phase
	device.measure_parity_check_ramsey(
	    Q_target = [qL],
	    Q_control = [qH],
     	Q_spectator=qSpectator,
	    # flux_cw_list = ['repetition_code_3', 'repetition_code_4'],
	    flux_cw_list = [flux_cw],
	    prepare_for_timedomain = False,
	    downsample_angle_points = 3,
	    update_mw_phase=True,
	    mw_phase_param=f'vcz_virtual_q_ph_corr_{dircts[1]}',
	    disable_metadata=True,
	    pc_repetitions=pc_repetitions)

	mw_lutman_H.upload_single_qubit_phase_corrections()
	mw_lutman_L.upload_single_qubit_phase_corrections()
	return True


def TwoQ_Randomized_benchmarking_wrapper(qH, qL, station, **kw):
	'''
	Wrapper function around Randomized benchmarking.
	Returns True if successful calibration otherwise
	returns False.
	'''
	# Set gate duration
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	# Buffer time after gate
	if 'buffer_time_ns' in kw.keys():
		buffer_time_ns = kw['buffer_time_ns']
	else:
		buffer_time_ns = 0
	TQG_duration_ns += buffer_time_ns
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=800,
	                              flux_pulse_duration=TQG_duration_ns,
	                              init_duration=200000)
	# Setup for measurement
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
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
	amp_qH = get_DAC_amp_frequency(det_qH, flux_lm_H, 
						negative_amp=True if qH in OFFSET_QUBITS else False)
	amp_qL = get_DAC_amp_frequency(det_qL, flux_lm_L,
						negative_amp=True if qL in OFFSET_QUBITS else False)
	# Compensate for asymmetry of cz pulse
	_asymmetry = flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}')
	if abs(_asymmetry)> .075:
		amp_qH = amp_qH/(1+flux_lm_H.get(f'vcz_asymmetry_{dircts[0]}'))
	for i, det, amp, flux_lm in zip([       0,			1],
								 	[	det_qH,    det_qL],
								 	[	amp_qH,    amp_qL],
								 	[flux_lm_H, flux_lm_L]):
		if abs(det) < 10e6:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', 0)
		else:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', amp)
		# Check waveform duration
		check_flux_wf_duration(flux_lm_L)
	# Prepare device 
	device = station['device']
	flux_cw = 'cz'
	device.ro_acq_weight_type('optimal IQ')
	# device.ro_acq_averages(2**8)
	device.ro_acq_averages(2**10)
	device.ro_acq_digitized(False)
	# Set preparation params 
	mw_lutman_H.set_default_lutmap()
	mw_lutman_L.set_default_lutmap()
	# Check waveform durations
	check_flux_wf_duration(flux_lm_H)
	check_flux_wf_duration(flux_lm_L)
	device.prepare_for_timedomain(qubits=[qH, qL], bypass_flux=True)
	# Load flux waveforms
	load_single_waveform_on_HDAWG(flux_lm_H, f'cz_{dircts[0]}')
	load_single_waveform_on_HDAWG(flux_lm_L, f'cz_{dircts[1]}')
	# Assess recompilation
	if 'recompile' in kw.keys():
		_recompile = kw['recompile']
	else:
		_recompile = False
	# measurement
	device.measure_two_qubit_interleaved_randomized_benchmarking(
		qubits = [qH, qL],
		nr_seeds = 20,
		measure_idle_flux = False,
		prepare_for_timedomain = False,
		recompile = _recompile,
		nr_cliffords = np.array([0, 1., 3., 5., 7., 9., 11., 15.,
								 20., 30., 50.]),
		flux_codeword = flux_cw)
	return True


def TLS_density_wrapper(Qubit, station,
		                detuning = None,
		                max_duration = 120e-9,
						use_second_excited_state: bool = False):
	'''
	Wrapper function for measurement of TLS density.
	Using a dynamical square pulse to flux the qubit
	away while parking park_qubits.
	Args:
	    Qubit: fluxed qubit.
	    park_qubits: list of parked qubits.
	'''
	Qubit_parks = {
		'D1': [],
		'D2': [],
		'D3': [],
		'D4': ['Z1', 'Z3', 'X3'],
		'D5': ['Z1', 'Z4', 'X2', 'X3'],
		'D6': ['Z2', 'Z4', 'X2'],
		'D7': [],
		'D8': [],
		'D9': [],
		'Z1': ['D1', 'D2'],
		'Z2': ['D3'],
		'Z3': ['D7'],
		'Z4': ['D8', 'D9'],
		'X1': ['D1', 'D2'],
		'X2': ['D2', 'D3'],
		'X3': ['D7', 'D8'],
		'X4': ['D8', 'D9'],
	}
	# Set gate duration
	if max_duration>TWOQ_GATE_DURATION:
		delta = int(np.round((max_duration-TWOQ_GATE_DURATION)*1e9/20)*20)
	else:
		delta = 0
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	file_cfg = gc.generate_config(in_filename=input_file,
								out_filename=config_fn,
								mw_pulse_duration=20,
								ro_duration=1000,
								flux_pulse_duration=TQG_duration_ns+delta+20,
								init_duration=200000)
	# Setup for measurement
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	device = station.components['device']
	Flux_lm_q = station.components[Qubit].instr_LutMan_Flux.get_instr()
	# Determine minimum detuning
	p_coefs = Flux_lm_q.q_polycoeffs_freq_01_det()
	freq_func = np.poly1d(p_coefs)
	amp_0 = -p_coefs[1]/(2*p_coefs[0])
	det_0 = freq_func(amp_0)
	# det_0 = Flux_lm_q.q_polycoeffs_freq_01_det()[-1]
	if detuning is None:
		detuning = np.arange(det_0, 1500e6, 5e6)
	# Convert detuning to list of amplitudes
	Flux_lm_q.sq_amp(0.5)
	Amps = [ get_Ch_amp_frequency(det, Flux_lm_q, DAC_param='sq_amp')\
			 for  det in detuning ]
	# Check parking qubits if needed and set the right parking distance.  
	Parked_qubits = Qubit_parks[Qubit]
	# set parking amps for parked qubits. 
	if not Parked_qubits:
		print('no parking qubits are defined')
	else:
		# Handle frequency of parked qubits
		for q_park in Parked_qubits:
			Q_park = station.components[q_park]
			# minimum allowed detuning
			minimum_detuning = 600e6
			f_q = station.components[Qubit].freq_qubit()
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
			if max_duration > TWOQ_GATE_DURATION:
				fl_lm_park.cfg_max_wf_length(max_duration)
				fl_lm_park.AWG.get_instr().reset_waveforms_zeros()
	# prepare for timedomains
	if max_duration > TWOQ_GATE_DURATION:
		Flux_lm_q.cfg_max_wf_length(max_duration)
		Flux_lm_q.AWG.get_instr().reset_waveforms_zeros()
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**8)
	device.ro_acq_digitized(True)
	device.prepare_for_timedomain(qubits=[Qubit, 'D1'], bypass_flux=True)
	device.prepare_fluxing(qubits=[Qubit, 'D1']+Parked_qubits)
	# device.prepare_readout(qubits=[Qubit, 'D1'])
	# device.ro_acq_digitized(False)
	if not Parked_qubits:
		Parked_qubits = None
	device.measure_chevron(
	    q0=Qubit,
	    q_spec='D1',
	    amps=Amps,
	    q_parks=Parked_qubits,
	    lengths= np.linspace(10e-9, max_duration, 12),
	    target_qubit_sequence='ground',
	    waveform_name="square",
	    buffer_time=40e-9,
	    prepare_for_timedomain=False,
	    disable_metadata=True,
		second_excited_state=use_second_excited_state,
	)
	# Reset waveform durations
	if max_duration > TWOQ_GATE_DURATION:
		Flux_lm_q.cfg_max_wf_length(TWOQ_GATE_DURATION)
		Flux_lm_q.AWG.get_instr().reset_waveforms_zeros()
		if not Parked_qubits:
			print('no parking qubits are defined')
		else:
			for q_park in Parked_qubits:
				fl_lm_park = Q_park.instr_LutMan_Flux.get_instr()
				fl_lm_park.cfg_max_wf_length(TWOQ_GATE_DURATION)
				fl_lm_park.AWG.get_instr().reset_waveforms_zeros()
	# Run landscape analysis
	interaction_freqs = { 
		d : Flux_lm_q.get(f'q_freq_10_{d}')\
		for d in ['NW', 'NE', 'SW', 'SE']\
		if 2e9 > Flux_lm_q.get(f'q_freq_10_{d}') > 10e6
		}
	a = ma2.tqg.TLS_landscape_Analysis(
				Q_freq = station.components[Qubit].freq_qubit(),
				Out_range=Flux_lm_q.cfg_awg_channel_range(),
				DAC_amp=Flux_lm_q.sq_amp(),
				Poly_coefs=Flux_lm_q.q_polycoeffs_freq_01_det(),
				interaction_freqs=interaction_freqs)
	return True


def Parking_experiment_wrapper(qH, qL, qP, station, relative_to_qH: bool = True, park_stepsize: float = 5e6):
	'''
	Wrapper function for fine-tunig CP 180 phase, SQ phase updates of 360, and verification. 
	Returns True if successful calibration otherwise
	returns False.
	'''
	# Set gate duration
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	file_cfg = gc.generate_config(in_filename=input_file,
								out_filename=config_fn,
								mw_pulse_duration=20,
								ro_duration=1000,
								flux_pulse_duration=TQG_duration_ns,
								init_duration=200000)
	# Setup for measurement
	dircts = get_gate_directions(qH, qL)
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	Q_P = station.components[qP]
	mw_lutman_H = Q_H.instr_LutMan_MW.get_instr()
	mw_lutman_L = Q_L.instr_LutMan_MW.get_instr()
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	# Set preparation params
	device = station['device']
	device.ro_acq_weight_type('optimal')
	device.ro_acq_averages(2**9)
	# # Prepare readout
	if check_prepare_readout(qubits=[qH, qL, qP], station=station):
		device.prepare_readout(qubits=[qH, qL, qP])
	else:
		# if preparation is not deemed necessary try just updating detectors
		try:
			acq_ch_map = device._acq_ch_map
			device._prep_ro_instantiate_detectors(qubits=[qH, qL, qP], acq_ch_map=acq_ch_map)
		except:
			device.prepare_readout(qubits=[qH, qL, qP])
	# device.prepare_readout(qubits=[qH, qL, qP])
	# Prepare MW pulses
	for q in [qH, qL, qP]:
		Q = station.components[q]
		Q._prep_td_sources()
		if check_prepare_mw(q, station):
			Q.cfg_prepare_mw_awg(False)
			Q._prep_mw_pulses()
			Q.cfg_prepare_mw_awg(True)
			mw_lm = Q.instr_LutMan_MW.get_instr()
			mw_lm.set_default_lutmap()
			mw_lm.load_phase_pulses_to_AWG_lookuptable()
	# Set DAC amplitude for 2Q gate 
	det_qH = flux_lm_H.get(f'q_freq_10_{dircts[0]}')
	det_qL = flux_lm_L.get(f'q_freq_10_{dircts[1]}')
	amp_qH = get_DAC_amp_frequency(det_qH, flux_lm_H)
	amp_qL = get_DAC_amp_frequency(det_qL, flux_lm_L)
	for i, det, amp, flux_lm in zip([		 0,			1],
								 	[	det_qH,    det_qL],
								 	[	amp_qH,    amp_qL],
								 	[flux_lm_H, flux_lm_L]):
		if det < 20e6:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', 0)
		else:
			flux_lm.set(f'vcz_amp_dac_at_11_02_{dircts[i]}', amp)
	# Load flux waveforms
	load_single_waveform_on_HDAWG(flux_lm_H, f'cz_{dircts[0]}')
	load_single_waveform_on_HDAWG(flux_lm_L, f'cz_{dircts[1]}')
	# Park remainder neighboring qubits away (at least 1GHz from qH)
	Qspectators = get_parking_qubits(qH, qL)
	for q in Qspectators:
		if q != qP:
			Q = station.components[q]
			flux_lm = Q.instr_LutMan_Flux.get_instr()
			park_dist = 1000e6
			park_freq = Q_H.freq_qubit()-det_qH-park_dist
			park_det = Q.freq_qubit()-park_freq
			amp_park = get_DAC_amp_frequency(park_det, flux_lm)
			flux_lm.set('park_amp', amp_park)
			load_single_waveform_on_HDAWG(flux_lm, 'park')
	# Select parking distances
	# (we start the sweep at 20 MHz qP detuning)
	park_det_init = 20e6
	park_dist_init = park_det_init
	if relative_to_qH:
		park_freq = Q_P.freq_qubit() - park_det_init
		park_dist_init = Q_H.freq_qubit()-det_qH-park_freq
	Park_distances = np.arange(park_dist_init, 1000e6, park_stepsize)
	# Measure
	# device.calibrate_park_frequency(
	# 	qH=qH, qL=qL, qP=qP,
	# 	Park_distances = Park_distances,
	# 	prepare_for_timedomain=False,
	# 	disable_metadata=False)	
	return True


def Calibrate_CZ_gate(qH, qL, station,
					  qL_det: float = 0,
					  park_distance: float = 700e6,
					  apply_parking_settings: bool = True,
					  asymmetry_compensation: bool = False,
					  calibrate_asymmetry: bool = True,
					  benchmark: bool = True,
					  tmid_offset_samples: int = 0,
					  calibration_type: str = 'full',
					  **kw):
	'''
	Calibrates and benchmarks two-qubit gate.
	Calibration types:
		Full : Performs full cz calibration.
		Express: Only single qubit-phase calibration.
		Fine: Fine cphase calibration using Bamp sweep.
				 
	'''
	# Parse calibration steps
	calibration_steps = []
	if 'full' in calibration_type.lower():
		calibration_steps.append('Chevron')
		calibration_steps.append('Tmid')
		calibration_steps.append('AB')
		if 'no chevron' in calibration_type.lower():
			if 'Chevron' in calibration_steps:
				calibration_steps.remove('Chevron')
	else:
		if 'tmid' in calibration_type.lower():
			calibration_steps.append('Tmid')
		if 'ab' in calibration_type.lower():
			calibration_steps.append('AB')

	# Calibrate for asymmetryc cz pulse?
	use_negative_amp=False
	if asymmetry_compensation:
		use_negative_amp=True
	if 'fine' in calibration_type.lower():
		fine_calibration = True
	else:
		fine_calibration = False
	# Perform calibration steps
	if 'Chevron' in calibration_steps:
		# Measure interaction Chevron
		Chevron_wrapper(qH=qH, qL=qL, station=station, 
						qL_det=qL_det,
						park_distance=park_distance,
						negative_amp=use_negative_amp,
						**kw)
	if 'Tmid' in calibration_steps:
		# SNZ A vs Tmid landscape
		SNZ_tmid_wrapper(qH=qH, qL=qL, station=station,
		                 apply_parking_settings=apply_parking_settings,
						 park_distance=park_distance,
						 asymmetry_compensation=asymmetry_compensation,
						 tmid_offset_samples=tmid_offset_samples,
						 **kw)
	if 'AB' in calibration_steps:
		# SNZ A vs B landscape
		SNZ_AB_wrapper(qH=qH, qL=qL, station=station,
		               apply_parking_settings=apply_parking_settings,
					   park_distance=park_distance,
					   asymmetry_compensation=asymmetry_compensation,
					   **kw)
	# Pulse asymmetry calibration
	if calibrate_asymmetry:
	    Asymmetry_wrapper(qH=qH, qL=qL, station=station)
	# Single qubit phase calibration
	Single_qubit_phase_calibration_wrapper(
		qH=qH, qL=qL, station=station,
		apply_parking_settings=apply_parking_settings,
		park_distance=park_distance,
		fine_cphase_calibration=fine_calibration)
	# Interleaved randomized benchmarking
	if benchmark:
	    TwoQ_Randomized_benchmarking_wrapper(qH=qH, qL=qL, station=station, **kw)

	
###############################################################################
# Parity check calibration graph
###############################################################################
import os
import pycqed as pq
from pycqed.measurement.openql_experiments import generate_CC_cfg as gc
input_file = os.path.join(pq.__path__[0], 'measurement',
                          'openql_experiments', 'config_cc_s17_direct_iq.json.in')
config_fn = os.path.join(pq.__path__[0], 'measurement',
                       'openql_experiments', 'output_cc_s17','config_cc_s17_direct_iq.json')


class Parity_check_calibration(AutoDepGraph_DAG):
	def __init__(self, 
		         name: str,
		         station,
		         stabilizers: list = None,
		         **kwargs):
		super().__init__(name, **kwargs)
		if stabilizers == None:
			stabilizers = [
						'Z1',
						'Z2',
						'Z3',
						'Z4',
						# 'X1',
						# 'X2',
						# 'X3',
						# 'X4',
							]
		self.station = station
		self.create_dep_graph(stabilizers=stabilizers)

	def create_dep_graph(self, stabilizers:list):
		'''
		Dependency graph for the calibration of 
		single-qubit gates.
		'''
		print(f'Creating dependency graph for Parity check calibration')
		##############################
		# Grah nodes
		##############################
		module_name = 'pycqed.instrument_drivers.meta_instrument.Surface17_dependency_graph'


		for stab in stabilizers:

			self.add_node(f'{stab} Horizontal calibration',
			    calibrate_function=module_name+'.Horizontal_calibration_wrapper',
		        calibrate_function_args={
		        	'stabilizer_qubit': stab,
		          	'station': self.station
		          	})

			self.add_node(f'{stab} Ancilla phase verification',
			    calibrate_function=module_name+'.Measure_parity_check_phase_wrapper',
		        calibrate_function_args={
		        	'stabilizer_qubit': stab,
		          	'station': self.station
		          	})

			self.add_node(f'{stab} Data-qubit phase correction',
			    calibrate_function=module_name+'.Data_qubit_phase_calibration_wrapper',
		        calibrate_function_args={
		        	'stabilizer_qubit': stab,
		          	'station': self.station
		          	})

			self.add_node(f'{stab} Parity assignment fidelity',
			    calibrate_function=module_name+'.Parity_check_fidelity_wrapper',
		        calibrate_function_args={
		        	'stabilizer_qubit': stab,
		          	'station': self.station
		          	})

			self.add_node(f'{stab} Parity repeatability',
			    calibrate_function=module_name+'.Parity_check_repeatability_wrapper',
		        calibrate_function_args={
		        	'stabilizer_qubit': stab,
		          	'station': self.station
		          	})

			# self.add_node('Spectator_data_qubits',
			# 	calibrate_function=module_name+'.Spectator_data_qubits_wrapper',
			# 	calibrate_function_args={
			# 		'stabilizer': stab,
			# 		'station': self.station
			# 		})

			# self.add_node('DIO_calibration',
			#     calibrate_function=module_name+'.DIO_calibration',
			#     calibrate_function_args={
			#     	'stabilizer': stab,
			#       	'station': self.station
			#       	})

		# Save snpashot
		self.add_node('Save snapshot',
              calibrate_function=module_name+'.save_snapshot_metadata',
              calibrate_function_args={
              	'station': self.station,
              	})

		##############################
		# Node depdendencies
		##############################
		for stab in stabilizers:
			self.add_edge('Save snapshot',
						  f'{stab} Parity repeatability')
			
			self.add_edge(f'{stab} Parity repeatability',
						  f'{stab} Parity assignment fidelity')
			
			self.add_edge(f'{stab} Parity assignment fidelity',
						  f'{stab} Data-qubit phase correction')

			self.add_edge(f'{stab} Data-qubit phase correction',
						  f'{stab} Ancilla phase verification')

			self.add_edge(f'{stab} Ancilla phase verification',
						  f'{stab} Horizontal calibration')

		##############################
		# Create graph
		##############################
		self.cfg_plot_mode = 'svg'
		self.update_monitor()
		self.cfg_svg_filename
		url = self.open_html_viewer()
		print('Dependency graph created at ' + url)


def Prepare_for_parity_check_wrapper(station):
	'''
	Wrapper function to prepare for timedomain of all parity checks of
	a stabilizer.
	'''
	# Prepare for timedomain of parity check
	for q in ['Z1', 'Z2', 'Z3', 'Z4',
			  'X1', 'X2', 'X3', 'X4']:
		prepare_for_parity_check(q, station)
	return True


def Horizontal_calibration_wrapper(stabilizer_qubit, station,
								   Q_control: list = None,
								   flux_cw_list = None,
								   mw_phase_param = None):
	'''
	Wrapper function to calibrate parity check CZ phases
	returns True
	'''
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Prepare for timedomain of parity check
	device = station.components['device']
	device.ro_acq_averages(2**8)
	device.ro_acq_digitized(False)
	prepare_for_parity_check(stabilizer_qubit, station,
							 Data_qubits = Q_control)
	Q_ancilla = [stabilizer_qubit]
	if not Q_control:
		Q_control = list(get_nearest_neighbors(stabilizer_qubit).keys())
	# Parity check settings
	if flux_cw_list == None:
		if 'X' in stabilizer_qubit:
			flux_cw_list = [f'flux_dance_{i}' for i in [1, 2, 3, 4]]
		else:
			flux_cw_list = [f'flux_dance_{i}' for i in [5, 6, 7, 8]]
		flux_cw_list = ['cz' for q in Q_control]
	if mw_phase_param == None:
		# if 'X' in stabilizer_qubit:
		# 	mw_phase_param = 'vcz_virtual_q_ph_corr_step_4'
		# else:
		# 	mw_phase_param = 'vcz_virtual_q_ph_corr_step_8'
		dircts = get_gate_directions(q0=stabilizer_qubit, q1=Q_control[0])
		mw_phase_param = f'vcz_virtual_q_ph_corr_{dircts[0]}'
	# Calibrate CZ with each data qubit
	for q in Q_control:
		# If high frequency qubit
		if q in ['D4','D5','D6']:
			# Order of qubits requires high freq. qubit first
			Q_pair_target = [q, stabilizer_qubit]
		else: 
			Q_pair_target = [stabilizer_qubit, q]
		device.calibrate_parity_check_phase(
			Q_ancilla = Q_ancilla,
			Q_control = Q_control,
			Q_pair_target = Q_pair_target,
			flux_cw_list = flux_cw_list,
			downsample_angle_points = 3,
			mw_phase_param = mw_phase_param,
			update_flux_param=True,
			update_mw_phase=True,
			prepare_for_timedomain=False,
			disable_metadata = True)
		# upload new waveform
		Q_H = station.components[Q_pair_target[0]]
		fl_lm_q = Q_H.instr_LutMan_Flux.get_instr()
		dircts = get_gate_directions(*Q_pair_target)
		load_single_waveform_on_HDAWG(fl_lm_q, f'cz_{dircts[0]}')
		# upload phase corrections
		Q_S = station.components[stabilizer_qubit]
		mw_lm_q = Q_S.instr_LutMan_MW.get_instr()
		mw_lm_q.upload_single_qubit_phase_corrections()
	return True


def Measure_parity_check_phase_wrapper(stabilizer_qubit, station, 
									   Q_control: list = None,
									   flux_cw_list = None, 
									   mw_phase_param = None,
									   pc_repetitions: int = 1,
									   ):
	'''
	Wrapper function to measure pairty checks 
	returns True
	'''
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Prepare for timedomain of parity check
	device = station.components['device']
	device.ro_acq_averages(2**8)
	device.ro_acq_digitized(False)
	prepare_for_parity_check(stabilizer_qubit, station,
							 Data_qubits = Q_control)
	# Parity check settings
	Q_ancilla = [stabilizer_qubit]
	if not Q_control:
		Q_control = list(get_nearest_neighbors(stabilizer_qubit).keys())
	if flux_cw_list == None:
		if 'X' in stabilizer_qubit:
			flux_cw_list = [f'flux_dance_{i}' for i in [1, 2, 3, 4]]
		else:
			flux_cw_list = [f'flux_dance_{i}' for i in [5, 6, 7, 8]]
		flux_cw_list = ['cz' for q in Q_control]
	if mw_phase_param == None:
		# if 'X' in stabilizer_qubit:
		# 	mw_phase_param = 'vcz_virtual_q_ph_corr_step_4'
		# else:
		# 	mw_phase_param = 'vcz_virtual_q_ph_corr_step_8'
		dircts = get_gate_directions(q0 = stabilizer_qubit, q1 = Q_control[0])
		mw_phase_param = f'vcz_virtual_q_ph_corr_{dircts[0]}'
	# Measure
	qoi = device.measure_parity_check_ramsey(
		Q_target = Q_ancilla,
		Q_control = Q_control,
		flux_cw_list = flux_cw_list,
		mw_phase_param=mw_phase_param,
		pc_repetitions=pc_repetitions,
		update_mw_phase=True,
		prepare_for_timedomain = False,
		disable_metadata=True)
	# upload phase corrections
	Q_S = station.components[stabilizer_qubit]
	mw_lm_q = Q_S.instr_LutMan_MW.get_instr()
	mw_lm_q.upload_single_qubit_phase_corrections()
	return qoi


def Data_qubit_phase_calibration_wrapper(stabilizer_qubit, station, 
										 Q_data: list = None,
										 flux_cw_list = None, 
										 mw_phase_param = None,
									   	 pc_repetitions: int = 1,
										 ):
	'''
	Wrapper function to calibrate single qubit phases of data-qubits in pairty checks 
	returns True
	'''
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Prepare for timedomain of parity check
	device = station.components['device']
	device.ro_acq_averages(2**8)
	device.ro_acq_digitized(False)
	prepare_for_parity_check(stabilizer_qubit, station,
							 Data_qubits = Q_data)
	# Parity check settings
	Q_ancilla = [stabilizer_qubit]
	if not Q_data:
		Q_data = list(get_nearest_neighbors(stabilizer_qubit).keys())
	if flux_cw_list == None:
		if 'X' in stabilizer_qubit:
			flux_cw_list = [f'flux_dance_{i}' for i in [1, 2, 3, 4]]
		else:
			flux_cw_list = [f'flux_dance_{i}' for i in [5, 6, 7, 8]]
		flux_cw_list = ['cz' for q in Q_data]
	if mw_phase_param == None:
		# if 'X' in stabilizer_qubit:
		# 	mw_phase_param = 'vcz_virtual_q_ph_corr_step_4'
		# else:
		# 	mw_phase_param = 'vcz_virtual_q_ph_corr_step_8'
		mw_phase_param = []
		for q in Q_data:
			dircts = get_gate_directions(q0 = q, q1 = stabilizer_qubit)
			mw_phase_param.append(f'vcz_virtual_q_ph_corr_{dircts[0]}')
	# Measure
	qoi = device.measure_parity_check_ramsey(
		Q_target = Q_data,
		Q_control = Q_ancilla,
		flux_cw_list = flux_cw_list,
		downsample_angle_points = 1,
		update_mw_phase=True,
		mw_phase_param=mw_phase_param,
		pc_repetitions=pc_repetitions,
		prepare_for_timedomain = False,
		disable_metadata=True)
	# upload phase corrections
	for q in Q_data:
		Q_c = station.components[q]
		mw_lm_q = Q_c.instr_LutMan_MW.get_instr()
		mw_lm_q.upload_single_qubit_phase_corrections()
	return qoi


def Parity_check_fidelity_wrapper(stabilizer_qubit, station, 
								  Q_data: list = None,
								  heralded_init = False,
								  flux_cw_list = None):
	'''
	Wrapper function to measure pairty checks 
	returns True
	'''
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Prepare for timedomain of parity check
	device = station.components['device']
	device.ro_acq_averages(2**10)
	device.ro_acq_digitized(False)
	device.ro_acq_integration_length(500e-9)
	prepare_for_parity_check(stabilizer_qubit, station, Data_qubits = Q_data)
	# Parity check settings
	Q_ancilla = [stabilizer_qubit]
	if not Q_data:
		Q_data = list(get_nearest_neighbors(stabilizer_qubit).keys())
	if flux_cw_list == None:
		if 'X' in stabilizer_qubit:
			flux_cw_list = [f'flux_dance_{i}' for i in [1, 2, 3, 4]]
		else:
			flux_cw_list = [f'flux_dance_{i}' for i in [5, 6, 7, 8]]
		flux_cw_list = ['cz' for q in Q_data]
	if not heralded_init:
		device.prepare_readout(qubits=Q_ancilla)
	# Measure
	device.measure_parity_check_fidelity(
		Q_ancilla = Q_ancilla,
		Q_control = Q_data,
		flux_cw_list = flux_cw_list,
		initialization_msmt=heralded_init,
		prepare_for_timedomain = False,
		disable_metadata=True)
	return True


def Parity_check_repeatability_wrapper(stabilizer_qubit, station, 
									   Q_data: list = None,
									   sim_measurement = True,
									   readout_duration_ns = 420,
									   repetitions = 5,
									   heralded_init = False,
									   flux_cw_list = None, 
									   n_rounds = None,):
	'''
	Wrapper function to measure pairty check repeatability 
	n_rounds : list 
	returns True
	'''
	# Set Parity check duration
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Prepare for timedomain of parity check
	device = station.components['device']
	device.ro_acq_averages(2**10)
	device.ro_acq_digitized(False)
	device.ro_acq_integration_length(500e-9)
	prepare_for_parity_check(stabilizer_qubit, station,
							 Data_qubits = Q_data)
	if not Q_data:
		Q_data = list(get_nearest_neighbors(stabilizer_qubit).keys())
	if flux_cw_list == None:
		if 'X' in stabilizer_qubit:
			flux_cw_list = [f'flux_dance_{i}' for i in [1, 2, 3, 4]]
		else:
			flux_cw_list = [f'flux_dance_{i}' for i in [5, 6, 7, 8]]
		flux_cw_list = ['cz' for q in Q_data]
	# can't avoid preparaing for timedomain here as it orders the qubits
	if n_rounds == None:
		# n_rounds = [1, 2]
		n_rounds = [2]
	# Measure
	for n in n_rounds:
		device.measure_weight_n_parity_tomography(
	    		ancilla_qubit = stabilizer_qubit,
	    		data_qubits = Q_data,
	    		flux_cw_list = flux_cw_list,
	    		sim_measurement=sim_measurement,
	    		readout_duration_ns = readout_duration_ns,
	    		n_rounds = n,
	    		repetitions = 3,
	    		prepare_for_timedomain = True,
	    		disable_metadata=True)
	return True


def Surface_13_wrapper(station, log_zero = None,
					   measurement_time_ns: int = 500,
					   prepare_LRU_pulses: bool = True):
	'''
	Wrapper routine to measure the surface-13 experiment. 
	'''
	#######################################################
	# Preparation
	#######################################################
	assert (measurement_time_ns-20)%40 == 0, 'Not a valid measurement time!'
	# Set configuration
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=measurement_time_ns,
	                              flux_pulse_duration=TQG_duration_ns,
	                              init_duration=200000)
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	device = station.components['device']
	integration_time = measurement_time_ns-20
	device.ro_acq_integration_length(integration_time*1e-9)
	# Get qubits directly involved in parity check
	Data_qubits = ['D1', 'D2', 'D3',
				   'D4', 'D5', 'D6',
				   'D7', 'D8', 'D9']
	Ancilla_qubits = ['Z1', 'Z2', 'Z3', 'Z4']
	All_qubits = Data_qubits + Ancilla_qubits
	# Prepare qubits
	import time
	# prepare readout
	# t_ro = time.time()
	# # prepapare readout detectors
	# device.ro_acq_weight_type('custom')
	# # ordered_ro_list = ['Z3', 'D4', 'D5', 'D1', 'D2', 'D3', 'D7', 'Z1', 'D6',
	# # 				   'D8', 'D9', 'Z2', 'Z4']
	# ordered_ro_list = ['Z3', 'D4', 'D5', 'X2', 'D1', 'X1', 'D2', 'D3', 'D7',
	# 				   'X3', 'Z1', 'D6', 'D8', 'D9', 'Z2', 'Z4']
	# ordered_ro_dict = {q: 'optimal IQ' for q in ordered_ro_list}
	# acq_ch_map = device._prep_ro_assign_weights(qubits=ordered_ro_list,
	# 	qubit_int_weight_type_dict = ordered_ro_dict)
	# device._prep_ro_integration_weights(qubits=ordered_ro_list,
	# 	qubit_int_weight_type_dict = ordered_ro_dict)
	# device._prep_ro_instantiate_detectors(qubits=ordered_ro_list,
	# 									  acq_ch_map=acq_ch_map)
	# # Prepare readout pulses with custom channel map
	# RO_lutman_1 = station.components['RO_lutman_1']
	# RO_lutman_2 = station.components['RO_lutman_2']
	# RO_lutman_3 = station.components['RO_lutman_3']
	# RO_lutman_4 = station.components['RO_lutman_4']
	# if [11] not in RO_lutman_1.resonator_combinations():
	# 	RO_lutman_1.resonator_combinations([[11], 
	# 		RO_lutman_1.resonator_combinations()[0]])
	# RO_lutman_1.load_waveforms_onto_AWG_lookuptable()
	# if [3, 7] not in RO_lutman_2.resonator_combinations():
	# 	RO_lutman_2.resonator_combinations([[3, 7],
	# 		RO_lutman_2.resonator_combinations()[0]])
	# RO_lutman_2.load_waveforms_onto_AWG_lookuptable()
	# if [8, 12] not in RO_lutman_4.resonator_combinations():
	# 	RO_lutman_4.resonator_combinations([[8, 12],
	# 		RO_lutman_4.resonator_combinations()[0]])
	# RO_lutman_4.load_waveforms_onto_AWG_lookuptable()
	# if [14, 10] not in RO_lutman_3.resonator_combinations():
	# 	RO_lutman_3.resonator_combinations([[14, 10], 
	# 		RO_lutman_3.resonator_combinations()[0]])
	# RO_lutman_3.load_waveforms_onto_AWG_lookuptable()
	# t_ro = time.time()-t_ro
	# prepare flux pulses
	t_fl = time.time()
	# Set Flux pulse amplitudes
	for q in All_qubits:
		fl_lm_q = station.components[q].instr_LutMan_Flux.get_instr()
		set_combined_waveform_amplitudes(fl_lm_q)
	# upload flux pulses of data qubits
	for stabilizer_qubit in Ancilla_qubits:
		Neighbors_dict = get_nearest_neighbors(stabilizer_qubit)
		for q, dirct in Neighbors_dict.items():
			fl_lm_q = station.components[q].instr_LutMan_Flux.get_instr()
			waveforms_to_upload = ['park', f'cz_{dirct}']
			load_single_waveform_on_HDAWG(fl_lm_q, waveforms_to_upload)
		# upload flux pulses of ancilla qubit
		Q_A = station.components[stabilizer_qubit]
		fl_lm_q = Q_A.instr_LutMan_Flux.get_instr()
		waveforms_to_upload = []
		for dirct in Neighbors_dict.values():
			if dirct == 'NW':
				waveforms_to_upload.append('cz_SE')
			elif dirct == 'NE':
				waveforms_to_upload.append('cz_SW')
			elif dirct == 'SW':
				waveforms_to_upload.append('cz_NE')
			else: # SE
				waveforms_to_upload.append('cz_NW')
		load_single_waveform_on_HDAWG(fl_lm_q, waveforms_to_upload)
	t_fl = time.time()-t_fl
	# prepare timings
	t_tim = time.time()
	device.prepare_timing()
	t_tim = time.time()-t_tim
	# Upload mw pulses
	t_mw = time.time()
	for q in All_qubits:
		Q = station.components[q]
		Q._prep_td_sources()
		mw_lm = Q.instr_LutMan_MW.get_instr()
		mw_lm.upload_single_qubit_phase_corrections()
		if check_prepare_mw(q, station, lutmap='default_lutmap'):
			mw_lm = Q.instr_LutMan_MW.get_instr()
			mw_lm.set_default_lutmap()
			Q._prep_mw_pulses()
	t_mw = time.time()-t_mw

	# Prepare LRUs if desired
	if prepare_LRU_pulses:
		LRU_duration = measurement_time_ns-20
		Z_LRU_duration = measurement_time_ns//3-20
		# Set LRU LO triggering
		MW_LO_6 = station.components['MW_LO_6']
		MW_LO_6.pulsemod_state(True)
		MW_LO_6.pulsemod_source('INT')
		MW_LO_6.visa_handle.write('pulm:delay 160 ns')
		MW_LO_6.visa_handle.write(f'pulm:width {measurement_time_ns+Z_LRU_duration} ns')
		MW_LO_6.visa_handle.write('pulm:trig:mode EXT')
		MW_LO_6.visa_handle.write('conn:trig:omod PETR')
		MW_LO_7 = station.components['MW_LO_7']
		MW_LO_7.pulsemod_state(True)
		MW_LO_7.pulsemod_source('INT')
		MW_LO_7.visa_handle.write('pulm:delay 390 ns')
		MW_LO_7.visa_handle.write('pulm:width 140 ns')
		MW_LO_7.visa_handle.write('pulm:trig:mode EXT')
		MW_LO_7.visa_handle.write('conn:trig:omod PETR')
		# for D4, D5, D6
		station.components['D4'].LRU_duration(LRU_duration*1e-9+Z_LRU_duration*1e-9)
		station.components['D5'].LRU_duration(LRU_duration*1e-9+Z_LRU_duration*1e-9)
		station.components['D6'].LRU_duration(LRU_duration*1e-9+Z_LRU_duration*1e-9)
		station.components['D6'].LRU_duration(Z_LRU_duration*1e-9)
		station.components['Z1'].LRU_duration(Z_LRU_duration*1e-9)
		station.components['Z2'].LRU_duration(Z_LRU_duration*1e-9)
		station.components['Z3'].LRU_duration(Z_LRU_duration*1e-9)
		station.components['Z4'].LRU_duration(Z_LRU_duration*1e-9)
		# upload pulses
		station.components['D4']._prep_LRU_pulses()
		station.components['D5']._prep_LRU_pulses()
		station.components['D6']._prep_LRU_pulses()
		station.components['Z1']._prep_LRU_pulses()
		station.components['Z2']._prep_LRU_pulses()
		station.components['Z2']._prep_LRU_pulses()
		station.components['Z4']._prep_LRU_pulses()

	# print(f'Preparation time RO:\t{t_ro}')
	print(f'Preparation time FL:\t{t_fl}')
	print(f'Preparation time TIM:\t{t_tim}')
	print(f'Preparation time MW:\t{t_mw}')
	# Run experiment
	device.measure_defect_rate(
			ancilla_qubit='Z3',
			data_qubits=['D4', 'D7'],
			experiments=['surface_13', 'surface_13_LRU',],
			Rounds= [15],
			repetitions = 2,
			lru_qubits = ['D4', 'D5', 'D6', 'Z1', 'Z2', 'Z3', 'Z4'],
			prepare_for_timedomain = False,
			prepare_readout = True,
			heralded_init = True,
			stabilizer_type = 'X',
			initial_state_qubits = None,
			measurement_time_ns = measurement_time_ns,
			analyze = False)
	# Run Pij matrix analysis
	a = ma2.pba.Repeated_stabilizer_measurements(
		ancilla_qubit = Ancilla_qubits,
		data_qubits = Data_qubits,
		Rounds = [15],
		heralded_init = True,
		number_of_kernels =  2,
		Pij_matrix = True,
		experiments = ['surface_13', 'surface_13_LRU'],
		extract_only = False)
	# # Surface_13 experiment
	if log_zero: 
		for state in [
			 [],							  # I
			 ['D1','D2'], 					  # X1
			 ['D8','D9'], 					  # X4
			 ['D2','D3','D5','D6'], 		  # X2
			 ['D4','D5','D7','D8'], 		  # X3
			 ['D1','D3','D5','D6'], 		  # X1 X2
			 ['D4','D5','D7','D9'], 		  # X3 X4
			 ['D1','D2','D8','D9'], 		  # X1 X4
			 ['D2','D3','D4','D6','D7','D8'], # X2 X3 
			 ['D1','D2','D4','D5','D7','D8'], # X1 X3
			 ['D2','D3','D5','D6','D8','D9'], # X2 X4
			 ['D1','D3','D4','D6','D7','D8'], # X1 X2 X3
			 ['D2','D3','D4','D6','D7','D9'], # X2 X3 X4
			 ['D1','D3','D5','D6','D8','D9'], # X1 X2 X4
			 ['D1','D2','D4','D5','D7','D9'], # X1 X3 X4
			 ['D1','D3','D4','D6','D7','D9']  # X1 X2 X3 X4
					 ]:
			device.measure_defect_rate(
					ancilla_qubit='Z3',
					data_qubits=['D4', 'D7'],
					experiments=['surface_13', 'surface_13_LRU'],
					Rounds=[1, 2, 4, 8, 16],
					lru_qubits = ['D4', 'D5', 'D6', 'Z1', 'Z2', 'Z3', 'Z4'],
					repetitions = 20,
					prepare_for_timedomain = False,
					prepare_readout = False,
					heralded_init = True,
					stabilizer_type = 'Z',
					initial_state_qubits = state,
					measurement_time_ns = measurement_time_ns,
					analyze = False)
			# Run Pij matrix analysis
			a = ma2.pba.Repeated_stabilizer_measurements(
				ancilla_qubit = Ancilla_qubits,
				# experiments=['surface_13'],
				experiments=['surface_13', 'surface_13_LRU'],
				data_qubits = Data_qubits,
				Rounds = [1, 2, 4, 8, 16],
				heralded_init = True,
				number_of_kernels =  2,
				Pij_matrix = True,
				extract_only = False)


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

############################################
# Helper functions for waveform parameters #
############################################
def get_parking_frequency(qubit_name: str) -> float:
    """:return: Qubit frequency when parking [Hz]."""
    qubit: Transmon = Device.find_instrument(qubit_name)
    flux_lutman: FluxLutMan = qubit.instr_LutMan_Flux.get_instr()
    park_detuning: float = get_frequency_waveform(
		wave_par='park_amp',
		flux_lutman=flux_lutman,
	)
    qubit_frequency: float = qubit.freq_qubit()
    return qubit_frequency - park_detuning

def set_parking_frequency(qubit_name: str, park_frequency: float) -> float:
    """:return: Qubit frequency when parking [Hz]."""
    qubit: Transmon = Device.find_instrument(qubit_name)
    flux_lutman: FluxLutMan = qubit.instr_LutMan_Flux.get_instr()
    qubit_frequency: float = qubit.freq_qubit()
    park_detuning: float = qubit_frequency - park_frequency
    park_amplitude: float = get_DAC_amp_frequency(
		freq=park_detuning,
		flux_lutman=flux_lutman,
		negative_amp=False,
	)
    # Logging info
    old_park_frequency: float = get_parking_frequency(qubit_name)
    old_park_amplitude: float = flux_lutman.park_amp()
    # Update parking amplitude
    flux_lutman.park_amp(park_amplitude)
    logging.info(f"Parking amplitude of {qubit_name} is updated from {old_park_amplitude} ({(old_park_frequency * 1e-9):0.2f} GHz) to {park_amplitude} ({(park_frequency * 1e-9):0.2f} GHz)")
    return get_parking_frequency(qubit_name)

def set_parking_detuning(qubit_name: str, park_detuning: float) -> float:
    """:return: Qubit frequency when parking [Hz]."""
    qubit: Transmon = Device.find_instrument(qubit_name)
    qubit_frequency: float = qubit.freq_qubit()
    park_frequency: float = qubit_frequency - park_detuning
    return set_parking_frequency(
		qubit_name=qubit_name,
		park_frequency=park_frequency,
	)

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
	return freq

def get_DAC_amp_frequency(freq, flux_lutman, negative_amp:bool=False):
	'''
	Function to calculate DAC amp corresponding 
	to frequency detuning.
	'''
	poly_coefs = flux_lutman.q_polycoeffs_freq_01_det()
	out_range = flux_lutman.cfg_awg_channel_range()
	ch_amp = flux_lutman.cfg_awg_channel_amplitude()
	poly_func = np.poly1d(poly_coefs)
	if negative_amp:
		out_volt = min((poly_func-freq).roots)
	else:
		out_volt = max((poly_func-freq).roots)
	sq_amp = out_volt/(ch_amp*out_range/2)
	# Safe check in case amplitude exceeds maximum
	if abs(sq_amp)>1:
		print(f'WARNING had to increase gain of {flux_lutman.name} to {ch_amp}!')
		flux_lutman.cfg_awg_channel_amplitude(ch_amp*1.5)
		# Can't believe Im actually using recursion!!!
		sq_amp = get_DAC_amp_frequency(freq, flux_lutman)
	return sq_amp

def get_Ch_amp_frequency(freq, flux_lutman, DAC_param='sq_amp'):
	'''
	Function to calculate channel gain corresponding 
	to frequency detuning.
	'''
	poly_coefs = flux_lutman.q_polycoeffs_freq_01_det()
	out_range = flux_lutman.cfg_awg_channel_range()
	dac_amp = flux_lutman.get(DAC_param)
	poly_func = np.poly1d(poly_coefs)
	if dac_amp < 0: # if negative amplitude
		out_volt = min((poly_func-freq).roots)
	else: # if positive amplitude
		out_volt = max((poly_func-freq).roots)
	ch_amp = out_volt/(dac_amp*out_range/2)
	if isinstance(ch_amp, complex):
		print('Warning: Complex amplitude estimated, setting it to zero.')
		ch_amp = 0
	return ch_amp

def load_single_waveform_on_HDAWG(lutman, wave_id):
	"""
	Load a single waveform on HDAWG.
	"""
	AWG = lutman.AWG.get_instr()
	AWG.stop()
	# Allow wave_id to be a list of waveforms
	if isinstance(wave_id, str):
		wave_id = [wave_id]
	for wf in wave_id:
		if check_flux_wf_upload(lutman, wf):
			print(f'Uploading {wf} in {lutman.name}.')
			lutman.load_waveform_onto_AWG_lookuptable(
				wave_id=wf, regenerate_waveforms=True)
	lutman.cfg_awg_channel_amplitude()
	lutman.cfg_awg_channel_range()
	AWG.start()

def set_combined_waveform_amplitudes(flux_lutman):
	'''
	Set waveform amplitudes for all gate directions and
	parking for a flux lutman.
	'''
	# print(f'Setting common amps in {flux_lutman.name}.')
	qubit = flux_lutman.name.split('_')[-1]
	# calculate park detuning
	park_det = get_frequency_waveform('park_amp', flux_lutman)
	# Remove default values (need to fix this in pycqed instead)
	for drct in ['NW', 'NE', 'SW', 'SE']:
		det = flux_lutman.get(f'q_freq_10_{drct}')
		if det == 6e9:
			flux_lutman.set(f'q_freq_10_{drct}', 0)
	# Get CZ detunings
	cz_NW_det = flux_lutman.q_freq_10_NW()
	cz_NE_det = flux_lutman.q_freq_10_NE()
	cz_SW_det = flux_lutman.q_freq_10_SW()
	cz_SE_det = flux_lutman.q_freq_10_SE()
	# Required detunings dictionary
	Detunings = {'park_amp' : park_det,
				 'vcz_amp_dac_at_11_02_NW' : cz_NW_det,
				 'vcz_amp_dac_at_11_02_NE' : cz_NE_det,
				 'vcz_amp_dac_at_11_02_SW' : cz_SW_det,
				 'vcz_amp_dac_at_11_02_SE' : cz_SE_det}
	# Find waveform with maximum detuning
	max_wf = max(Detunings, key=Detunings.get)
	# Set amplitude of DAC to 0.5 and scale gain accordingly
	if qubit in OFFSET_QUBITS:
		flux_lutman.set(max_wf, -0.3)
	else:
		flux_lutman.set(max_wf, 0.3)
	max_wf_gain = get_Ch_amp_frequency(Detunings[max_wf], flux_lutman,
									   DAC_param = max_wf)
	flux_lutman.cfg_awg_channel_amplitude(max_wf_gain)
	Detunings.pop(max_wf) # remove waveform from detuning dict
	# Set remaining waveform amplitudes
	for wf, det in Detunings.items():
		if det > 20e6:
			wf_amp = get_DAC_amp_frequency(det, flux_lutman,
						negative_amp=True if qubit in OFFSET_QUBITS else False)
		else:
			wf_amp = 0
		flux_lutman.set(wf, wf_amp)

def prepare_for_parity_check(stabilizer_qubit, station,
							 Data_qubits: list = None):
	'''
	Wrapper function to prepare for timedomain of parity check of
	a stabilizer.
	'''
	# Set configuration
	TQG_duration_ns = TWOQ_GATE_DURATION_NS - 20
	file_cfg = gc.generate_config(in_filename=input_file,
	                              out_filename=config_fn,
	                              mw_pulse_duration=20,
	                              ro_duration=500,
	                              flux_pulse_duration=TQG_duration_ns,
	                              init_duration=200000)
	device = station.components['device']
	device.ro_acq_weight_type('optimal')
	# Get qubits directly involved in parity check
	if not Data_qubits:
		Data_qubits = list(get_nearest_neighbors(stabilizer_qubit).keys())
	PC_qubits = [stabilizer_qubit]+Data_qubits
	# Get spectator qubits of parity check
	Spec_qubits = []
	for q in Data_qubits:
		_qubits = list(get_nearest_neighbors(q).keys())
		_qubits.remove(stabilizer_qubit)
		Spec_qubits = Spec_qubits + _qubits
	Spec_qubits = np.unique(Spec_qubits)
	# Set Flux pulse amplitudes
	for q in list(PC_qubits)+list(Spec_qubits):
		fl_lm_q = station.components[q].instr_LutMan_Flux.get_instr()
		set_combined_waveform_amplitudes(fl_lm_q)
	print(f'Parity check qubits: {" ".join(PC_qubits)}')
	print(f'Spectator qubits: {" ".join(Spec_qubits)}')
	# Prepare parity check qubits
	# device.prepare_for_timedomain(qubits=PC_qubits)
	import time
	# prepare readout
	t_ro = time.time()
	if check_prepare_readout(qubits=PC_qubits, station=station):
		device.prepare_readout(qubits=PC_qubits)
	else:
		# if preparation is not deemed necessary try just updating detectors
		try:
			acq_ch_map = device._acq_ch_map
			device._prep_ro_instantiate_detectors(qubits=PC_qubits, acq_ch_map=acq_ch_map)
		except:
			device.prepare_readout(qubits=PC_qubits)
	t_ro = time.time()-t_ro
	# prepare flux pulses
	t_fl = time.time()
	# upload flux pulses of data qubits
	Neighbors_dict = get_nearest_neighbors(stabilizer_qubit)
	for q, dirct in Neighbors_dict.items():
		fl_lm_q = station.components[q].instr_LutMan_Flux.get_instr()
		waveforms_to_upload = ['park', f'cz_{dirct}']
		load_single_waveform_on_HDAWG(fl_lm_q, waveforms_to_upload)
	# upload flux pulses of ancilla qubit
	Q_A = station.components[stabilizer_qubit]
	fl_lm_q = Q_A.instr_LutMan_Flux.get_instr()
	waveforms_to_upload = []
	for dirct in Neighbors_dict.values():
		if dirct == 'NW':
			waveforms_to_upload.append('cz_SE')
		elif dirct == 'NE':
			waveforms_to_upload.append('cz_SW')
		elif dirct == 'SW':
			waveforms_to_upload.append('cz_NE')
		else: # SE
			waveforms_to_upload.append('cz_NW')
	load_single_waveform_on_HDAWG(fl_lm_q, waveforms_to_upload)
	# Prepare parking of spectator qubits
	for q in Spec_qubits:
		fl_lm_q = station.components[q].instr_LutMan_Flux.get_instr()
		load_single_waveform_on_HDAWG(fl_lm_q, 'park')
	t_fl = time.time()-t_fl
	# prepare timings
	t_tim = time.time()
	device.prepare_timing()
	t_tim = time.time()-t_tim
	# Upload mw pulses
	t_mw = time.time()
	for q in PC_qubits:
		Q = station.components[q]
		Q._prep_td_sources()
		if check_prepare_mw(q, station):
			Q.cfg_prepare_mw_awg(False)
			Q._prep_mw_pulses()
			Q.cfg_prepare_mw_awg(True)
			mw_lm = Q.instr_LutMan_MW.get_instr()
			mw_lm.set_default_lutmap()
			mw_lm.load_phase_pulses_to_AWG_lookuptable()
			mw_lm.upload_single_qubit_phase_corrections()
	t_mw = time.time()-t_mw
	print(f'Preparation time RO:\t{t_ro}')
	print(f'Preparation time FL:\t{t_fl}')
	print(f'Preparation time TIM:\t{t_tim}')
	print(f'Preparation time MW:\t{t_mw}')
	return True

def check_prepare_readout(qubits, station):
	'''
	Function to assess weather readout pulses have to be 
	reuploaded. This is done by looking at the resonator
	combinations present in the RO lutmans.
	Returns True if preparation is necessary (otherwise
	returns False).
	'''
	# Assess required readout combinations
	ro_lms = []
	resonators_in_lm = {}
	for qb_name in qubits:
		qb = station.components[qb_name]
		# qubit and resonator number are identical
		res_nr = qb.cfg_qubit_nr()
		ro_lm = qb.instr_LutMan_RO.get_instr()
		# Add resonator to list of resonators in lm
		if ro_lm not in ro_lms:
			ro_lms.append(ro_lm)
			resonators_in_lm[ro_lm.name] = []
		resonators_in_lm[ro_lm.name].append(res_nr)
	# Check if required resonator combinations are
	# present in RO lutmans.
	check_list = []
	for ro_lm in ro_lms:
		res_combs = ro_lm.resonator_combinations()
		check_list.append(resonators_in_lm[ro_lm.name] in res_combs)
	return not all(check_list)

def check_prepare_mw(qubit, station,
					 lutmap='phase_lutmap'):
	'''
	Function to assess weather mw pulses have to be 
	reuploaded. This is done by looking at each uploaded
	waveform in the HDAWG and comparing it to the new 
	generated.
	Returns True if preparation is necessary (otherwise
	returns False).
	'''
	# Required lutmap for parity check experiments
	if lutmap == 'phase_lutmap':
		required_lutmap = {0: {'name': 'I', 'theta': 0, 'phi': 0, 'type': 'ge'},
						   1: {'name': 'rX180', 'theta': 180, 'phi': 0, 'type': 'ge'},
						   2: {'name': 'rY180', 'theta': 180, 'phi': 90, 'type': 'ge'},
						   3: {'name': 'rX90', 'theta': 90, 'phi': 0, 'type': 'ge'},
						   4: {'name': 'rY90', 'theta': 90, 'phi': 90, 'type': 'ge'},
						   5: {'name': 'rXm90', 'theta': -90, 'phi': 0, 'type': 'ge'},
						   6: {'name': 'rYm90', 'theta': -90, 'phi': 90, 'type': 'ge'},
						   7: {'name': 'rPhi90', 'theta': 90, 'phi': 0, 'type': 'ge'},
						   8: {'name': 'spec', 'type': 'spec'},
						   9: {'name': 'rPhi90', 'theta': 90, 'phi': 0, 'type': 'ge'},
						   10: {'name': 'rPhi90', 'theta': 90, 'phi': 20, 'type': 'ge'},
						   11: {'name': 'rPhi90', 'theta': 90, 'phi': 40, 'type': 'ge'},
						   12: {'name': 'rPhi90', 'theta': 90, 'phi': 60, 'type': 'ge'},
						   13: {'name': 'rPhi90', 'theta': 90, 'phi': 80, 'type': 'ge'},
						   14: {'name': 'rPhi90', 'theta': 90, 'phi': 100, 'type': 'ge'},
						   15: {'name': 'rPhi90', 'theta': 90, 'phi': 120, 'type': 'ge'},
						   16: {'name': 'rPhi90', 'theta': 90, 'phi': 140, 'type': 'ge'},
						   27: {'name': 'rXm180', 'phi': 0, 'theta': -180, 'type': 'ge'},
						   30: {'name': 'rX23', 'theta': 180, 'phi': 0, 'type': 'fh'},
						   51: {'name': 'phaseCorrLRU', 'type': 'phase'},
						   52: {'name': 'phaseCorrStep1', 'type': 'phase'},
						   53: {'name': 'phaseCorrStep2', 'type': 'phase'},
						   54: {'name': 'phaseCorrStep3', 'type': 'phase'},
						   55: {'name': 'phaseCorrStep4', 'type': 'phase'},
						   56: {'name': 'phaseCorrStep5', 'type': 'phase'},
						   57: {'name': 'phaseCorrStep6', 'type': 'phase'},
						   58: {'name': 'phaseCorrStep7', 'type': 'phase'},
						   59: {'name': 'phaseCorrStep8', 'type': 'phase'},
						   60: {'name': 'phaseCorrNW', 'type': 'phase'},
						   61: {'name': 'phaseCorrNE', 'type': 'phase'},
						   62: {'name': 'phaseCorrSW', 'type': 'phase'},
						   63: {'name': 'phaseCorrSE', 'type': 'phase'},
						   17: {'name': 'rPhi90', 'theta': 90, 'phi': 160, 'type': 'ge'},
						   18: {'name': 'rPhi90', 'theta': 90, 'phi': 180, 'type': 'ge'},
						   19: {'name': 'rPhi90', 'theta': 90, 'phi': 200, 'type': 'ge'},
						   20: {'name': 'rPhi90', 'theta': 90, 'phi': 220, 'type': 'ge'},
						   21: {'name': 'rPhi90', 'theta': 90, 'phi': 240, 'type': 'ge'},
						   22: {'name': 'rPhi90', 'theta': 90, 'phi': 260, 'type': 'ge'},
						   23: {'name': 'rPhi90', 'theta': 90, 'phi': 280, 'type': 'ge'},
						   24: {'name': 'rPhi90', 'theta': 90, 'phi': 300, 'type': 'ge'},
						   25: {'name': 'rPhi90', 'theta': 90, 'phi': 320, 'type': 'ge'},
						   26: {'name': 'rPhi90', 'theta': 90, 'phi': 340, 'type': 'ge'}}
	elif lutmap == 'default_lutmap':
		required_lutmap = {0: {'name': 'I', 'theta': 0, 'phi': 0, 'type': 'ge'},
						   1: {'name': 'rX180', 'theta': 180, 'phi': 0, 'type': 'ge'},
						   2: {'name': 'rY180', 'theta': 180, 'phi': 90, 'type': 'ge'},
						   3: {'name': 'rX90', 'theta': 90, 'phi': 0, 'type': 'ge'},
						   4: {'name': 'rY90', 'theta': 90, 'phi': 90, 'type': 'ge'},
						   5: {'name': 'rXm90', 'theta': -90, 'phi': 0, 'type': 'ge'},
						   6: {'name': 'rYm90', 'theta': -90, 'phi': 90, 'type': 'ge'},
						   7: {'name': 'rPhi90', 'theta': 90, 'phi': 0, 'type': 'ge'},
						   8: {'name': 'spec', 'type': 'spec'},
						   9: {'name': 'rX12', 'theta': 180, 'phi': 0, 'type': 'ef'},
						   10: {'name': 'square', 'type': 'square'},
						   11: {'name': 'rY45', 'theta': 45, 'phi': 90, 'type': 'ge'},
						   12: {'name': 'rYm45', 'theta': -45, 'phi': 90, 'type': 'ge'},
						   13: {'name': 'rX45', 'theta': 45, 'phi': 0, 'type': 'ge'},
						   14: {'name': 'rXm45', 'theta': -45, 'phi': 0, 'type': 'ge'},
						   15: {'name': 'rX12_90', 'theta': 90, 'phi': 0, 'type': 'ef'},
						   16: {'name': 'rX23_90', 'theta': 90, 'phi': 0, 'type': 'fh'},
						   27: {'name': 'rXm180', 'phi': 0, 'theta': -180, 'type': 'ge'},
						   30: {'name': 'rX23', 'theta': 180, 'phi': 0, 'type': 'fh'},
						   51: {'name': 'phaseCorrLRU', 'type': 'phase'},
						   52: {'name': 'phaseCorrStep1', 'type': 'phase'},
						   53: {'name': 'phaseCorrStep2', 'type': 'phase'},
						   54: {'name': 'phaseCorrStep3', 'type': 'phase'},
						   55: {'name': 'phaseCorrStep4', 'type': 'phase'},
						   56: {'name': 'phaseCorrStep5', 'type': 'phase'},
						   57: {'name': 'phaseCorrStep6', 'type': 'phase'},
						   58: {'name': 'phaseCorrStep7', 'type': 'phase'},
						   59: {'name': 'phaseCorrStep8', 'type': 'phase'},
						   60: {'name': 'phaseCorrNW', 'type': 'phase'},
						   61: {'name': 'phaseCorrNE', 'type': 'phase'},
						   62: {'name': 'phaseCorrSW', 'type': 'phase'},
						   63: {'name': 'phaseCorrSE', 'type': 'phase'}}
	else:
		raise ValueError("Accepted lutmaps are 'phase_lutmap' and 'default_lutmap'.")
	# Assess uploaded lutmap
	Q = station.components[qubit]
	mw_lm = Q.instr_LutMan_MW.get_instr()
	uploaded_lutmap = mw_lm.LutMap()
	# compare lutmaps 
	check_lutmaps = (required_lutmap == uploaded_lutmap)
	# Check if all waveforms match
	if check_lutmaps:
		wf_check_list = {}
		mw_lm.generate_standard_waveforms()
		for wf_idx, wf in mw_lm._wave_dict.items():
			# Get uploaded waveforms
			AWG = mw_lm.AWG.get_instr()
			wf_name_I = 'wave_ch{}_cw{:03}'.format(mw_lm.channel_I(), wf_idx)
			wf_name_Q = 'wave_ch{}_cw{:03}'.format(mw_lm.channel_Q(), wf_idx)
			uploaded_wf_I = AWG.get(wf_name_I)
			uploaded_wf_Q = AWG.get(wf_name_Q)
			# Check if uploaded wf match new wf
			_check_wf_I = all(wf[0]==uploaded_wf_I)
			_check_wf_Q = all(wf[1]==uploaded_wf_Q)
			wf_check_list[wf_idx] = _check_wf_I and _check_wf_Q
		check_waveforms = all(wf_check_list.values())
		return not check_waveforms
	else:
		return True

def check_flux_wf_upload(flux_lutman, wave_id):
	'''
	Assess if flux waveform needs re-uploading.
	This is done by looking at current waveform
	in the _wave_dict of the flux lutman.
	'''
	# Get present waveform
	present_wf = flux_lutman._wave_dict[wave_id] + 0
	# Check new waveform
	if "i" == wave_id:
		new_wf = flux_lutman._gen_i()
	elif "square" == wave_id:
		new_wf = flux_lutman._gen_square()
	elif "park" == wave_id:
		new_wf = flux_lutman._gen_park()
	elif "cz" in wave_id:
		which_gate = wave_id.split('_')[-1]
		new_wf = flux_lutman._gen_cz(which_gate=which_gate)
	# Check if waveform lengths are the same
	if len(new_wf) == len(present_wf):
		# If so, check if all points in waveforms match
		return not all(new_wf == present_wf)
	else:
		return False

def check_flux_wf_duration(flux_lutman):
	'''
	Checks whether waveform duration of lutman has changed.
	If so it resets the HDAWG to ensure changes will take effect 
	'''
	# If current duration is shorter, update duration
	if flux_lutman.cfg_max_wf_length() < TWOQ_GATE_DURATION:
		flux_lutman.cfg_max_wf_length(TWOQ_GATE_DURATION)
	# If duration is higher, update and reset waveforms
	# (this is necessary for changes to take effect).
	elif flux_lutman.cfg_max_wf_length() > TWOQ_GATE_DURATION:
		flux_lutman.cfg_max_wf_length(TWOQ_GATE_DURATION)
		awg = flux_lutman.AWG.get_instr()
		awg.reset_waveforms_zeros()
		print(f'Loading waveforms to match {TWOQ_GATE_DURATION_NS:.0f} '+\
			   'ns gate duration')
		flux_lutman.load_waveforms_onto_AWG_lookuptable()

def make_unipolar_pulse_net_zero(flux_lutman, wave_id):
	'''
	Adds appropritate padding amplitude to pulse in 
	order to achieve net zero area of cz waveform.
	'''
	assert 'cz' in wave_id, 'Only meant for cz waveforms'
	# Look for waveform
	dirct = wave_id.split('_')[-1]
	flux_lutman.set(f'vcz_amp_pad_{dirct}', 0)
	flux_lutman.generate_standard_waveforms()
	wf = flux_lutman._wave_dict[wave_id]
	n_samples = flux_lutman.get(f'vcz_amp_pad_samples_{dirct}')
	# Set amplitude of padding to achieve net-zeroness
	net_area = np.trapz(wf)*1/2.4e9
	time_pad = (flux_lutman.get(f'vcz_time_pad_{dirct}') - n_samples/2.4e9)*2
	amp_pad = -(net_area)/time_pad
	# # Ensure amplitude is lower than avoided crossing amp
	# assert amp_pad < 0.5
	flux_lutman.set(f'vcz_amp_pad_{dirct}', amp_pad)

def align_CZ_gate_pulses(qH, qL, station):
	'''
	Aligns CZ gate pulses gate qubits and parking pulses.
	'''
	# Setup qubits and lutmans
	dircts = get_gate_directions(qH, qL)
	Parked_qubits = get_parking_qubits(qH, qL)
	Q_H = station.components[qH]
	Q_L = station.components[qL]
	flux_lm_H = Q_H.instr_LutMan_Flux.get_instr()
	flux_lm_L = Q_L.instr_LutMan_Flux.get_instr()
	Flux_lm_ps = [ station.components[q].instr_LutMan_Flux.get_instr()\
				   for q in Parked_qubits ]
	# Get gate parameters
	tp = flux_lm_H.get(f'vcz_time_single_sq_{dircts[0]}')*2
	n_tmid = int(flux_lm_H.get(f'vcz_time_middle_{dircts[0]}')*2.4e9)
	# Align pulses
	tmid_swf = swf.flux_t_middle_sweep(
		fl_lm_tm =  [flux_lm_H, flux_lm_L], 
		fl_lm_park = Flux_lm_ps,
		which_gate = list(dircts),
		duration=TWOQ_GATE_DURATION,
		time_park=TWOQ_GATE_DURATION-(6/2.4e9),
		t_pulse = [tp])
	tmid_swf.set_parameter(n_tmid)

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
		for j,q in enumerate(get_parking_qubits(Q.name, Q_Ls[i].name)):
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

def save_snapshot_metadata(station, Qubits=None, Qubit_pairs = None,
						   analyze=False, parity_check=False, 
						   Two_qubit_freq_trajectories=False,
						   label=None):
	'''
	Save snapshot of system and run compile analysis with
	summary of performances for single- and two-qubit gates,
	parity checks and two-qubit frequency trajectories.
	'''
	MC = station.components['MC']
	if not label:
		label = 'System_snapshot'
	MC.set_measurement_name(label)
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
			'X1', 'X2', 'X3', 'X4',
			]

	if Qubit_pairs == None:
		Qubit_pairs = [
			['D4', 'Z1'],
			['D5', 'Z1'],
			['Z1', 'D1'],
			['Z1', 'D2'],
			['D6', 'Z2'],
			['Z2', 'D3'],
			['D4', 'Z3'],
			['Z3', 'D7'],
			['D5', 'Z4'],
			['D6', 'Z4'],
			['Z4', 'D8'],
			['Z4', 'D9'],
			['X1', 'D1'],
			['X1', 'D2'],
			['D6', 'X2'],
			['D5', 'X2'],
			['X2', 'D3'],
			['X2', 'D2'],
			['D5', 'X3'],
			['D4', 'X3'],
			['X3', 'D8'],
			['X3', 'D7'],
			['X4', 'D9'],
			['X4', 'D8'],
						]
	# Plot single- and two-qubit gate benchmarks
	if analyze:
		ma2.gbta.SingleQubitGBT_analysis(Qubits=Qubits)
		ma2.gbta.TwoQubitGBT_analysis(Qubit_pairs=Qubit_pairs)
	# Plot two-qubit gate frequency trajectories
	if Two_qubit_freq_trajectories:
		ma2.tqg.TwoQubitGate_frequency_trajectory_analysis(Qubit_pairs=Qubit_pairs)
	# Plot parity-check benchmarks
	if parity_check:
		ma2.gbta.ParityCheckGBT_analysis(Stabilizers=['Z1', 'Z2', 'Z3', 'Z4', 'X1', 'X2', 'X3', 'X4'])
	return True


def DIO_calibration(station, force: bool = False):
	'''
	Checks for DIO errors in all instruments and calibrates
	them if error is found.
	'''
	# Get all intruments
	no_error = True
	awgs_with_errors = []
	cc = station.components['cc']
	UHFQC_1 = station.components['UHFQC_1']
	UHFQC_2 = station.components['UHFQC_2']
	UHFQC_3 = station.components['UHFQC_3']
	UHFQC_4 = station.components['UHFQC_4']
	AWG8_8481 = station.components['AWG8_8481']
	AWG8_8068 = station.components['AWG8_8068']
	AWG8_8074 = station.components['AWG8_8074']
	AWG8_8076 = station.components['AWG8_8076']
	AWG8_8499 = station.components['AWG8_8499']
	AWG8_8320 = station.components['AWG8_8320']
	AWG8_8279 = station.components['AWG8_8279']
	AWG8_8071 = station.components['AWG8_8071']
	device = station.components['device']

	# Helper function
	def _prep_awg(awg):
		'''
		Helper function to prepare AWG.
		This needs to be performed after DIO calibration.
		'''
		for k in station.components.keys():
			if ('MW_lutman' in k) or ('flux_lm' in k):
				lutman = station.components[k]
				if lutman.AWG() == awg:
					qubit_name = lutman.name.split('_')[-1]
					# qubit = station.components[qubit_name]
					# lutman.load_waveforms_onto_AWG_lookuptable()
					device.prepare_for_timedomain(qubits=[qubit_name],
												  prepare_for_readout=False)

	############################################
	# UHFQC DIO calibration
	############################################
	# UHFQC_1
	UHFQC_1.check_errors()
	_errors = UHFQC_1._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('UHFQC_1')
		# no_error = False
		UHFQC_1._errors = {}
		print(f'Calibrating DIO on UHFQC_1.')
		try:
			DIO.calibrate(sender=cc, receiver=UHFQC_1, sender_dio_mode='uhfqa')
			print(UHFQC_1.name, UHFQC_1._get_dio_calibration_delay(), 8)
		except:
			print(f'Failed DIO calibration on {UHFQC_1.name}!')
		UHFQC_1._set_dio_calibration_delay(8)
		UHFQC_1.clear_errors()
		UHFQC_1.sigins_0_range(0.2)
		UHFQC_1.sigins_1_range(0.2)
		station.components['RO_lutman_1'].load_DIO_triggered_sequence_onto_UHFQC()
	# UHFQC_2
	UHFQC_2.check_errors()
	_errors = UHFQC_2._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		# no_error = False
		if _errors != {}:
			awgs_with_errors.append('UHFQC_2')
		UHFQC_2._errors = {}
		print(f'Calibrating DIO on UHFQC_2.')
		try:
			DIO.calibrate(sender=cc, receiver=UHFQC_2, sender_dio_mode='uhfqa')
			print(UHFQC_2.name, UHFQC_2._get_dio_calibration_delay(), 2)
		except:
			print(f'Failed DIO calibration on {UHFQC_2.name}!')
		UHFQC_2._set_dio_calibration_delay(2)
		UHFQC_2.clear_errors()
		UHFQC_2.sigins_0_range(0.3)
		UHFQC_2.sigins_1_range(0.3)
		station.components['RO_lutman_2'].load_DIO_triggered_sequence_onto_UHFQC()
	# UHFQC_3
	UHFQC_3.check_errors()
	_errors = UHFQC_3._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('UHFQC_3')
		# no_error = False	# this is commented because some holdoff error cannot be fixed
		UHFQC_3._errors = {}
		print(f'Calibrating DIO on UHFQC_3.')
		try:
			DIO.calibrate(sender=cc, receiver=UHFQC_3, sender_dio_mode='uhfqa')
			print(UHFQC_3.name, UHFQC_3._get_dio_calibration_delay(), 2)
		except:
			print(f'Failed DIO calibration on {UHFQC_3.name}!')
		UHFQC_3._set_dio_calibration_delay(2)
		UHFQC_3.clear_errors()
		UHFQC_3.sigins_0_range(0.5)
		UHFQC_3.sigins_1_range(0.5)
		station.components['RO_lutman_3'].load_DIO_triggered_sequence_onto_UHFQC()
	# UHFQC_4
	UHFQC_4.check_errors()
	_errors = UHFQC_4._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		# no_error = False
		if _errors != {}:
			awgs_with_errors.append('UHFQC_4')
		UHFQC_4._errors = {}
		print(f'Calibrating DIO on UHFQC_4.')
		try:
			DIO.calibrate(sender=cc, receiver=UHFQC_4, sender_dio_mode='uhfqa')
			print(UHFQC_4.name, UHFQC_4._get_dio_calibration_delay(), 7)
		except:
			print(f'Failed DIO calibration on {UHFQC_4.name}!')
		UHFQC_4._set_dio_calibration_delay(7)
		UHFQC_4.clear_errors()
		UHFQC_4.sigins_0_range(0.4)
		UHFQC_4.sigins_1_range(0.4)
		station.components['RO_lutman_4'].load_DIO_triggered_sequence_onto_UHFQC()

	############################################
	# MW HDAWG DIO calibration
	############################################
	# AWG8_8481
	AWG8_8481.check_errors()
	_errors = AWG8_8481._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('AWG8_8481')
		no_error = False
		AWG8_8481._errors = {}
		print(f'Calibrating DIO on AWG8_8481.')
		AWG8_8481.set('dios_0_interface', 0)
		AWG8_8481.set('dios_0_interface', 1)
		AWG8_8481.clear_errors()
		try:
			DIO.calibrate(sender=cc, receiver=AWG8_8481, sender_dio_mode='awg8-mw-direct-iq')
			print(AWG8_8481.name, AWG8_8481._get_dio_calibration_delay(), 6)
		except:
			print(f'Failed DIO calibration on {AWG8_8481.name}!')
		AWG8_8481._set_dio_calibration_delay(6)
		AWG8_8481.clear_errors()
		_prep_awg('AWG8_8481')
	# AWG8_8068
	AWG8_8068.check_errors()
	_errors = AWG8_8068._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('AWG8_8068')
		no_error = False
		AWG8_8068._errors = {}
		print(f'Calibrating DIO on AWG8_8068.')
		AWG8_8068.set('dios_0_interface', 0)
		AWG8_8068.set('dios_0_interface', 1)
		AWG8_8068.clear_errors()
		try:
			DIO.calibrate(sender=cc, receiver=AWG8_8068, sender_dio_mode='awg8-mw-direct-iq')
			print(AWG8_8068.name, AWG8_8068._get_dio_calibration_delay(), 4)
		except:
			print(f'Failed DIO calibration on {AWG8_8068.name}!')
		AWG8_8068._set_dio_calibration_delay(4)
		AWG8_8068.clear_errors()
		_prep_awg('AWG8_8068')
	# AWG8_8074
	AWG8_8074.check_errors()
	_errors = AWG8_8074._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('AWG8_8074')
		no_error = False
		AWG8_8074._errors = {}
		print(f'Calibrating DIO on AWG8_8074.')
		AWG8_8074.set('dios_0_interface', 0)
		AWG8_8074.set('dios_0_interface', 1)
		AWG8_8074.clear_errors()
		try:
			DIO.calibrate(sender=cc, receiver=AWG8_8074, sender_dio_mode='awg8-mw-direct-iq')
			print(AWG8_8074.name, AWG8_8074._get_dio_calibration_delay(), 6)
		except:
			print(f'Failed DIO calibration on {AWG8_8074.name}!')
		AWG8_8074._set_dio_calibration_delay(6)
		AWG8_8074.clear_errors()
		_prep_awg('AWG8_8074')
	# AWG8_8076
	AWG8_8076.check_errors()
	_errors = AWG8_8076._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('AWG8_8076')
		no_error = False
		AWG8_8076._errors = {}
		print(f'Calibrating DIO on AWG8_8076.')
		AWG8_8076.set('dios_0_interface', 0)
		AWG8_8076.set('dios_0_interface', 1)
		AWG8_8076.clear_errors()
		try:
			DIO.calibrate(sender=cc, receiver=AWG8_8076, sender_dio_mode='awg8-mw-direct-iq')
			print(AWG8_8076.name, AWG8_8076._get_dio_calibration_delay(), 4)
		except:
			print(f'Failed DIO calibration on {AWG8_8076.name}!')
		AWG8_8076._set_dio_calibration_delay(4)
		AWG8_8076.clear_errors()
		_prep_awg('AWG8_8076')
	# AWG8_8499
	AWG8_8499.check_errors()
	_errors = AWG8_8499._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('AWG8_8499')
		no_error = False
		AWG8_8499._errors = {}
		print(f'Calibrating DIO on AWG8_8499.')
		AWG8_8499.set('dios_0_interface', 0)
		AWG8_8499.set('dios_0_interface', 1)
		AWG8_8499.clear_errors()
		try:
			DIO.calibrate(sender=cc, receiver=AWG8_8499, sender_dio_mode='awg8-mw-direct-iq')
			print(AWG8_8499.name, AWG8_8499._get_dio_calibration_delay(), 6)
		except:
			print(f'Failed DIO calibration on {AWG8_8499.name}!')
		AWG8_8499._set_dio_calibration_delay(6)
		AWG8_8499.clear_errors()
		_prep_awg('AWG8_8499')

	############################################
	# Flux HDAWG DIO calibration
	############################################
	# AWG8_8279
	AWG8_8279.check_errors()
	_errors = AWG8_8279._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('AWG8_8279')
		no_error = False
		AWG8_8279._errors = {}
		print(f'Calibrating DIO on AWG8_8279.')
		AWG8_8279.set('dios_0_interface', 0)
		AWG8_8279.set('dios_0_interface', 1)
		AWG8_8279.clear_errors()
		try:
			DIO.calibrate(sender=cc, receiver=AWG8_8279, sender_dio_mode='awg8-flux')
			print(AWG8_8279.name, AWG8_8279._get_dio_calibration_delay(), 6)
		except:
			print(f'Failed DIO calibration on {AWG8_8279.name}!')
		AWG8_8279._set_dio_calibration_delay(6)
		AWG8_8279_channels = [0, 1, 2, 3, 4, 5, 6, 7]
		for this_ch in AWG8_8279_channels:
			AWG8_8279.setd('sigouts/%d/precompensation/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/exponentials/0/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/exponentials/1/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/exponentials/2/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/exponentials/3/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/exponentials/4/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/exponentials/5/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/exponentials/6/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/exponentials/7/enable' % (int(this_ch)), True)
			AWG8_8279.setd('sigouts/%d/precompensation/fir/enable' % (int(this_ch)), True)
			AWG8_8279.set('sigouts_{}_delay'.format(int(this_ch)), 0e-9 + 4 * 10 / 3 * 1e-9 - 2 * 3.33e-9)
		AWG8_8279.clear_errors()
		_prep_awg('AWG8_8279')
	# AWG8_8320
	AWG8_8320.check_errors()
	_errors = AWG8_8320._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('AWG8_8320')
		no_error = False
		AWG8_8320._errors = {}
		print(f'Calibrating DIO on AWG8_8320.')
		AWG8_8320.set('dios_0_interface', 0)
		AWG8_8320.set('dios_0_interface', 1)
		AWG8_8320.clear_errors()
		try:
			DIO.calibrate(sender=cc, receiver=AWG8_8320, sender_dio_mode='awg8-flux')
			print(AWG8_8320.name, AWG8_8320._get_dio_calibration_delay(), 1)
		except:
			print(f'Failed DIO calibration on {AWG8_8320.name}!')
		AWG8_8320._set_dio_calibration_delay(1)
		AWG8_8320_channels = [0, 1, 2, 3, 4, 5, 6, 7]
		for this_ch in AWG8_8320_channels:
			AWG8_8320.setd('sigouts/%d/precompensation/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/exponentials/0/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/exponentials/1/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/exponentials/2/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/exponentials/3/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/exponentials/4/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/exponentials/5/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/exponentials/6/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/exponentials/7/enable' % (int(this_ch)), True)
			AWG8_8320.setd('sigouts/%d/precompensation/fir/enable' % (int(this_ch)), True)
			AWG8_8320.set('sigouts_{}_delay'.format(int(this_ch)), 18e-9 + 2 * 3.33e-9)
		AWG8_8320.clear_errors()
		_prep_awg('AWG8_8320')
	# AWG8_8071
	AWG8_8071.check_errors()
	_errors = AWG8_8071._errors
	# if 'AWGDIOTIMING' in _errors.keys():
	if _errors != {} or force:
		if _errors != {}:
			awgs_with_errors.append('AWG8_8071')
		no_error = False
		AWG8_8071._errors = {}
		print(f'Calibrating DIO on AWG8_8071.')
		AWG8_8071.set('dios_0_interface', 0)
		AWG8_8071.set('dios_0_interface', 1)
		AWG8_8071.clear_errors()
		try:
			DIO.calibrate(sender=cc, receiver=AWG8_8071, sender_dio_mode='awg8-flux')
			print(AWG8_8071.name, AWG8_8071._get_dio_calibration_delay(), 6)
		except:
			print(f'Failed DIO calibration on {AWG8_8071.name}!')
		AWG8_8071._set_dio_calibration_delay(6)
		AWG8_8071_channels = [0, 1, 2, 3, 4, 5, 6, 7]
		for this_ch in AWG8_8071_channels:
			AWG8_8071.setd('sigouts/%d/precompensation/enable' % (int(this_ch)), True)
			AWG8_8071.setd('sigouts/%d/precompensation/exponentials/0/enable' % (int(this_ch)), True)
			AWG8_8071.setd('sigouts/%d/precompensation/exponentials/1/enable' % (int(this_ch)), True)
			AWG8_8071.setd('sigouts/%d/precompensation/exponentials/2/enable' % (int(this_ch)), True)
			AWG8_8071.setd('sigouts/%d/precompensation/exponentials/3/enable' % (int(this_ch)), True)
			AWG8_8071.setd('sigouts/%d/precompensation/fir/enable' % (int(this_ch)), True)
			AWG8_8071.set('sigouts_{}_delay'.format(int(this_ch)), 7e-9 - 2 * 3.33e-9)
		AWG8_8071.clear_errors()
		_prep_awg('AWG8_8071')
	# apply the right delays
	device.tim_flux_latency_0(-240e-9 - 4 * 36.67e-9)  # 8320
	device.tim_flux_latency_1(-240e-9 - 4 * 36.67e-9)  # 8279
	device.tim_flux_latency_2(-240e-9)  # 8071
	device.tim_mw_latency_0(0)  # 8076
	device.tim_mw_latency_1(-10e-9)  # 8074
	device.tim_mw_latency_2(-15e-9)  # 8499
	device.tim_mw_latency_3(0)  # 8068
	device.tim_mw_latency_4(-10e-9)  # 8481
	device.prepare_timing()
	return no_error, awgs_with_errors

###############################################################################
# LRU calibration graph
###############################################################################
class LRU_gate_calibration(AutoDepGraph_DAG):
	def __init__(self, 
		         name: str,
		         Qubits: str,
		         station,
		         **kwargs):
		super().__init__(name, **kwargs)
		self.station = station
		self.create_dep_graph(Qubits=Qubits)

	def create_dep_graph(self, Qubits:str):
		'''
		Dependency graph for the calibration of 
		single-qubit gates.
		'''
		print(f'Creating dependency graph for LRU gate calibration')
		##############################
		# Grah nodes
		##############################
		module_name = 'pycqed.instrument_drivers.meta_instrument.Surface17_dependency_graph'
		for qubit in Qubits:
			self.add_node(f'{qubit} Prepare for LRU calibration',
			    calibrate_function=module_name+'.prepare_for_LRU_calibration',
			    calibrate_function_args={
			      	'qubit' : qubit,
			      	'station': self.station,
			      	})

			self.add_node(f'{qubit} Sweep LRU Frequency',
			              calibrate_function=module_name+'.LRU_frequency_wrapper',
			              calibrate_function_args={
			      	'qubit' : qubit,
			      	'station': self.station,
			      	})

			self.add_node(f'{qubit} LRU drive mixer calibration',
			              calibrate_function=module_name+'.LRU_mixer_offset_wrapper',
			              calibrate_function_args={
			      	'qubit' : qubit,
			      	'station': self.station,
			      	})
			##############################
			# Node depdendencies
			##############################
			self.add_edge(f'{qubit} Sweep LRU Frequency',
			              f'{qubit} Prepare for LRU calibration')

			self.add_edge(f'{qubit} LRU drive mixer calibration',
			              f'{qubit} Sweep LRU Frequency')
		# Add master node that saves snapshot
		self.add_node(f'Save snapshot',
              calibrate_function=module_name+'.save_snapshot_metadata',
              calibrate_function_args={
              	'station': self.station,
              	})
		for qubit in Qubits:
			self.add_edge(f'Save snapshot',
			              f'{qubit} LRU drive mixer calibration')
		# Add dependencies between qubits
		# D1 and D3 share the AWG channel from D4
		# self.add_edge(f'D1 Prepare for LRU calibration',
		# 			  f'D4 LRU drive mixer calibration')
		# self.add_edge(f'D3 Prepare for LRU calibration',
		# 			  f'D4 LRU drive mixer calibration')
		# # D8 and D9 share the AWG channel from D4
		# self.add_edge(f'D8 Prepare for LRU calibration',
		# 			  f'D5 LRU drive mixer calibration')
		# self.add_edge(f'D9 Prepare for LRU calibration',
		# 			  f'D5 LRU drive mixer calibration')
		# # D2 and D7 share the AWG channel from D6
		# self.add_edge(f'D2 Prepare for LRU calibration',
		# 			  f'D6 LRU drive mixer calibration')
		# self.add_edge(f'D7 Prepare for LRU calibration',
		# 			  f'D6 LRU drive mixer calibration')

		##############################
		# Create graph
		##############################
		self.cfg_plot_mode = 'svg'
		self.update_monitor()
		self.cfg_svg_filename
		url = self.open_html_viewer()
		print('Dependency graph created at ' + url)


def prepare_for_LRU_calibration(qubit:str, station):
	'''
	Initial function to prepare qubit for calibration.
	We will set all relevant parameters for mw and readout.
	This is such that we only perform full preparation of
	the qubit once in the graph and all additional calibrated
	parameters are uploaded individually making the whole
	procedure time efficient.
	'''
	Q_inst = station.components[qubit]
	# Dictionary with LRU amplitude parameters
	LRU_param_dict = {
		# High frequency qubits
		'D4': {'ch_range': 5, 'ch_amp': 0.95},
		'D5': {'ch_range': 3, 'ch_amp': 1.00},
		'D6': {'ch_range': 5, 'ch_amp': 0.80},
		# 'D4': {'ch_range': 3, 'ch_amp': 1.00},
		# 'D5': {'ch_range': 3, 'ch_amp': 1.00},
		# 'D6': {'ch_range': 3, 'ch_amp': 1.00},
		# Low frequency qubits (these parameters are not used)
		'D1': {'ch_range': .8, 'ch_amp': 1}, 
		'D2': {'ch_range': .8, 'ch_amp': 1},
		'D3': {'ch_range': .8, 'ch_amp': 1},
		'D7': {'ch_range': .8, 'ch_amp': 1},
		'D8': {'ch_range': .8, 'ch_amp': 1},
		'D9': {'ch_range': .8, 'ch_amp': 1},
		# Mid frequency qubits
		'Z1': {'ch_range': 5, 'ch_amp': 0.85},
		'Z2': {'ch_range': 5, 'ch_amp': 0.60},
		'Z3': {'ch_range': 5, 'ch_amp': 0.60},
		'Z4': {'ch_range': 5, 'ch_amp': 0.80},
		'X1': {'ch_range': 5, 'ch_amp': 0.50},
		'X2': {'ch_range': 5, 'ch_amp': 0.50},
		'X3': {'ch_range': 5, 'ch_amp': 0.50},
		'X4': {'ch_range': 5, 'ch_amp': 0.50},
	}
	############################################
	# Set initial parameters for calibration
	############################################
	Q_inst.ro_acq_averages(2**10)
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal IQ')
	Q_inst.ro_acq_digitized(False)
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
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
		Q_inst._prep_ro_instantiate_detectors()
	# Set microwave lutman
	Q_lm = Q_inst.instr_LutMan_MW.get_instr()
	Q_lm.set_default_lutmap()
	############################################
	# Set LRU parameters
	############################################
	Q_inst.LRU_duration(220e-9)
	Q_inst.LRU_duration_rise(30e-9)
	Q_inst.LRU_amplitude(1)
	Q_inst.LRU_channel_amp(LRU_param_dict[qubit]['ch_amp'])
	Q_inst.LRU_channel_range(LRU_param_dict[qubit]['ch_range'])
	# Set LRU LO powers
	# High frequency
	station.components['MW_LO_6'].power(25) 		 # [D4, D5, D6]
	station.components['MW_LO_6'].frequency(5.192e9) # 
	# Mid frequency
	station.components['MW_LO_7'].power(25) 		 # [Z2, Z3, Z4, X3]
	station.components['MW_LO_7'].frequency(3.888e9) # 
	station.components['MW_LO_10'].power(25) 		  # [Z1, X4]
	station.components['MW_LO_10'].frequency(4.095e9) # 
	# station.components['MW_LO_15'].power(15) # X2
	# station.components['MW_LO_15'].power(15) # X1
	# # Low frequency
	# station.components['MW_LO_11'].power(6) # D1
	# station.components['MW_LO_9'].power(8) # D2
	# station.components['MW_LO_12'].power(10) # D3
	# station.components['MW_LO_8'].power(10) # D7
	# station.components['MW_LO_14'].power(10) # D8
	# station.components['MW_LO_13'].power(9) # D9
	############################################
	# Prepare for timedomain
	############################################
	# For low frequency qubits that share AWG channels
	if Q_inst.name in ['D1', 'D3']:
		station.components['D4']._prep_td_sources()
		station.components['D4']._prep_LRU_pulses()
	if Q_inst.name in ['D8', 'D9']:
		station.components['D5']._prep_td_sources()
		station.components['D5']._prep_LRU_pulses()
	if Q_inst.name in ['D2', 'D7']:
		station.components['D6']._prep_td_sources()
		station.components['D6']._prep_LRU_pulses()
	# Prepare qubit
	Q_inst._prep_td_sources()
	Q_inst._prep_mw_pulses()
	if Q_inst.instr_LutMan_LRU():
		Q_inst._prep_LRU_pulses()
	return True


def LRU_frequency_wrapper(qubit:str, station):
	'''
	Wrapper function around LRU frequency sweep calibration.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_weight_type('optimal IQ')
	Q_inst._prep_ro_instantiate_detectors()
	# Run experiment
	outcome = Q_inst.calibrate_LRU_frequency(
		frequencies = np.linspace(-30e6, 30e6, 121)+Q_inst.LRU_freq(),
		nr_shots_per_point=2**10,
		update=True,
		prepare_for_timedomain=False,
		disable_metadata=True)
	return outcome


def LRU_mixer_offset_wrapper(qubit:str, station):
	'''
	Wrapper function around LRU mixer offset calibration.
	Returns True if successful calibration otherwise
	returns False.
	'''
	Q_inst = station.components[qubit]
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set initial parameters for calibration
	Q_inst.ro_soft_avg(1)
	Q_inst._prep_td_sources()
	if Q_inst.instr_LutMan_LRU():
		Q_inst._prep_LRU_pulses()
	# Run experiment
	if Q_inst.name in ['D4', 'D5', 'D6']: 
		connect(f'{Q_inst.name}_LRU')
		outcome = Q_inst.calibrate_mixer_offsets_LRU(
							update=True,
							ftarget=-110,
							disable_metadata=True)
	# If low frequency qubits perform only single channel mixer calibration
	elif Q_inst.name in ['D1', 'D2', 'D3', 'D7', 'D8', 'D9']:
		outcome = Q_inst.calibrate_mixer_offset_LRU_single_channel(prepare = False,
                                        currents = np.linspace(-10e-3, 5e-3, 21),
                                        disable_metadata=True,
                                        adaptive_sampling = True,
                                        ch_par = station.components['LRUcurrent'].parameters[f'LRU_{Q_inst.name}'])
	# This was not implemented yet
	elif Q_inst.name in ['Z1', 'Z2', 'Z3', 'Z4', 'X1', 'X2', 'X3', 'X4']:
		outcome = True
	return outcome


def measure_LRU_wrapper(qubit:str, station):
	'''
	Wrapper function around LRU measurement.
	'''
	Q_inst = station.components[qubit]
	station.components['MC'].live_plot_enabled(False)
	station.components['nested_MC'].live_plot_enabled(False)
	# Set parameters for measurement
	Q_inst.ro_soft_avg(1)
	Q_inst.ro_acq_averages(2**12)
	Q_inst.ro_acq_weight_type('optimal IQ')
	Q_inst.ro_acq_digitized(False)
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
		Q_inst._prep_ro_instantiate_detectors()
	# Set microwave lutman
	Q_lm = Q_inst.instr_LutMan_MW.get_instr()
	Q_lm.set_default_lutmap()
	# upload lru offset params 
	Q_inst._prep_td_sources()
	Q_inst._prep_mw_pulses()
	if Q_inst.instr_LutMan_LRU():
		Q_inst._prep_LRU_pulses()
	outcome = Q_inst.measure_LRU_experiment(prepare_for_timedomain = False,
											disable_metadata = True)
	return outcome


###############################################################################
# Connect Switch box #
###############################################################################
import clr
import sys
sys.path.append('C:\\Windows\\SysWOW64')
clr.AddReference('mcl_RF_Switch_Controller_NET45')
from mcl_RF_Switch_Controller_NET45 import USB_RF_SwitchBox
switchbox = USB_RF_SwitchBox()
status = switchbox.Connect()
# Test if connected
if status > 0:
  [_, _, serialnumber] = switchbox.Send_SCPI('SN?', '')
  print(f'Successfully connected {serialnumber}')
#################
# Connect qubit #
#################
def switch_box_disconnect_all(LRU=False):
  side = 0
  if LRU:
    side = 1
  flag = 1
  [status, _, _] = switchbox.Send_SCPI(f'SETA={side}', '')
  flag *= status
  [status, _, _] = switchbox.Send_SCPI(f'SETB={side}', '')
  flag *= status
  [status, _, _] = switchbox.Send_SCPI(f'SETC={side}', '')
  flag *= status
  [status, _, _] = switchbox.Send_SCPI(f'SETD={side}', '')
  flag *= status
  if flag > 0:
    print(f'Successfully disconnected all qubits')

def switch_box_connect(qubit):
  splitter_qubits_map = {('A', 1): ['X1', 'X2', 'D9', 'D1'],
                         ('B', 1): ['D5', 'X4', 'X3', 'D2', 'D8'],
                         ('C', 1): ['D4', 'Z2', 'Z4', 'D3'],
                         ('D', 1): ['D6', 'Z1', 'Z3', 'D7'],
                         ('A', 0): ['D5_LRU'],
                         ('B', 0): ['D4_LRU'],
                         ('C', 0): ['D6_LRU']}
  # Find splitter belonging to qubit=
  for channel, _side in splitter_qubits_map.keys():
    if qubit in splitter_qubits_map[(channel, _side)]:
      switch = channel
      side = _side

  # is this an LRU channel?
  if 'LRU' in qubit:
    switch_box_disconnect_all(LRU=True)
  else:
    switch_box_disconnect_all()
  # Set that switch to on
  [status, _, _] = switchbox.Send_SCPI(f'SET{switch}={side}', '')
  if status > 0:
    print(f'Successfully connected {qubit}')
  else:
    print(f'Failed to connect {qubit}')