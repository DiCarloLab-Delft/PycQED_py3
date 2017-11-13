import numpy as np
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
# import dataprep for tomography module
# import tomography module
from pycqed.analysis_v2 import tomography_dataprep as dataprep
from pycqed.analysis_v2 import tomography_V2 as tomography_V2
from pycqed.analysis import measurement_analysis as ma 
try:
	import qutip as qt
except ImportError as e:
	logging.warning('Could not import qutip, tomo code will not work')

class TomographyExecute(object):

	"""
	Gets data from timestamps of HDF5 file;
	accepts only voltage data
	Calls data prep module to threshold the data, convert it to counts,
	get callibrated bin operators(Measurement operators).
	This output is then used to call various tomographies, of user's choosing.
	Based on the result a matrix histogram with fidelity to a target state is computed(optional);
	Also a pauli label plot is printed.
	"""

	def __init__(self, label='', timestamp=None,
                 target_cardinal=None, target_bell=None,
                 plot_matrix_histogram = False,
                 start_shot=0, end_shot=-1,
                 verbose=0,
                 tomography_type="SDPA",
                 fig_format='png',
                 q0_label='q0',
                 q1_label='q1', close_fig=True):
		self.label = label
		self.timestamp = timestamp
		self.target_cardinal = target_cardinal
		self.target_bell = target_bell
		self.start_shot = start_shot
		self.end_shot = end_shot
		self.verbose = verbose
		self.q0_label = q0_label
		self.q1_label = q1_label
		self.close_fig = close_fig
		self.tomography_type = tomography_type
		self.plot_matrix_histogram = plot_matrix_histogram

		

		
		a = ma.MeasurementAnalysis(auto=False, label=self.label,
		                           timestamp=self.timestamp)
		a.get_naming_and_values()
		self.t_stamp = a.timestamp_string
		self.savefolder = a.folder
		# hard coded number of segments for a 2 qubit state tomography
		# constraint imposed by UHFLI
		self.nr_segments = 64

		self.shots_q0 = np.zeros(
		    (self.nr_segments, int(len(a.measured_values[0])/self.nr_segments)))
		self.shots_q1 = np.zeros(
		    (self.nr_segments, int(len(a.measured_values[1])/self.nr_segments)))
		for i in range(self.nr_segments):
		    self.shots_q0[i, :] = a.measured_values[0][i::self.nr_segments]
		    self.shots_q1[i, :] = a.measured_values[1][i::self.nr_segments]

		# Get correlations between shots
		# Calculating correlations is not required

		##########################################
		# Giving data to data_prep module
		##########################################
		instantiated_dataprep = dataprep.TomoPrep(self.shots_q0,
												  self.shots_q1,
												  start_calliration_index =36, 
												  no_of_tomographic_rotations=36, 
												  no_of_repetitions_callibration =7)
		self.measurement_operators, self.counts_from_data_prep = instantiated_dataprep.assemble_input_for_tomography()

		##########################################
		# Calling Various tomographies depending on user choice
		##########################################
		tomos = tomography_V2.TomoAnalysis(2)

		if self.tomography_type == "SDPA":
			self.rhos = tomos.execute_SDPA_2qubit_tomo(self.measurement_operators,self.counts_from_data_prep, used_bins= [0], 
												   correct_measurement_operators=False, N_total=512)
		if self.tomography_type == "MLE":
			self.rhos =  tomos.execute_mle_T_matrix_tomo(self.measurement_operators[0], 
														 self.counts_from_data_prep[:,0]/512.0,
			 											 weights_tomo =False,
		                                   				 show_time=True, ftol=0.01, xtol=0.001, full_output=0, max_iter=100,
		                                   				 TE_correction_matrix = None)
		if self.tomography_type == "LI":
			(self.basis_decomposition, self.rhos) = tomos.execute_pseudo_inverse_tomo(self.meas_operators[0], self.counts_from_data_prep[:,0]/512.0, use_pauli_basis=False,
		                                   											  verbose=False)
		if self.plot_matrix_histogram == True:
			qt.matrix_histogram_complex(self.rhos)

	def get_density_matrix(self):

		#returns rho as a quantum object 
		return self.rhos