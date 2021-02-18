###########################################################################
# AutoDepGraph for Quantum Inspire
###########################################################################
"""
Third version of Graph Based Tuneup designed specifically for the Quantum
Inspire project. Includes only routines relevant for tuneup (readout, 
single-qubit and two-qubit fine-calibration), all characterization routines
were stripped. Additions include framework for two-qubit calibration.
"""

from autodepgraph import AutoDepGraph_DAG

class inspire_dep_graph(AutoDepGraph_DAG):
    def __init__(self, name: str, device, **kwargs):
        super().__init__(name, **kwargs)
        self.device = device

        qubits = []
        for qubit in self.device.qubits():
            if qubit != 'fakequbit':
                qubits.append(self.device.find_instrument(qubit))
        self.create_dep_graph(Qubit_list=qubits)

    def create_dep_graph(self, Qubit_list):
        print('Creating Graph ...')

        ########################################################
        # GRAPH NODES
        ########################################################
        for Qubit in Qubit_list:
        	################################
            # Qubit Readout Calibration
            ################################
            self.add_node(Qubit.name + ' SSRO Fidelity',
                        calibrate_function=self.device.name + '.calibrate_optimal_weights_mux',
                        calibrate_function_args={'qubits': list([qubit for qubit in self.device.qubits() \
                        	if self.device.find_instrument(qubit).instr_LutMan_RO()==Qubit.instr_LutMan_RO()]), 'q_target': Qubit.name, \
                          'return_analysis': False})

            ################################
            # Single Qubit Gate Assessment
            ################################
            self.add_node(Qubit.name + ' T1',
                           calibrate_function = Qubit.name + '.measure_T1')
            self.add_node(Qubit.name + ' T2_Star',
                           calibrate_function = Qubit.name + '.measure_ramsey')
            self.add_node(Qubit.name + ' T2_Echo',
                           calibrate_function = Qubit.name + '.measure_echo')
            self.add_node(Qubit.name + ' ALLXY',
                          calibrate_function = Qubit.name + '.allxy_GBT')

            ################################
            # Single Qubit Gate Calibration
            ################################
            self.add_node(Qubit.name + ' Frequency Fine',
                          calibrate_function=Qubit.name + '.calibrate_frequency_ramsey')
                          #check_function=Qubit.name + '.check_ramsey', tolerance=0.1e-3)
            self.add_node(Qubit.name + ' Flipping',
                          calibrate_function=Qubit.name + '.flipping_GBT')
            self.add_node(Qubit.name + ' MOTZOI Calibration',
                          calibrate_function=Qubit.name + '.calibrate_motzoi')
            self.add_node(Qubit.name + ' Second Flipping',
                          calibrate_function=Qubit.name + '.flipping_GBT')
            self.add_node(Qubit.name + ' ALLXY',
                          calibrate_function=Qubit.name + '.allxy_GBT')
            self.add_node(Qubit.name + ' RB Fidelity',
                          calibrate_function=Qubit.name + '.measure_single_qubit_randomized_benchmarking')

            ###################################################################
            # Qubit Dependencies
            ###################################################################
            # First depends on second being done
            self.add_edge(Qubit.name + ' T1',
                          Qubit.name + ' SSRO Fidelity')
            self.add_edge(Qubit.name + ' T2_Star',
                          Qubit.name + ' T1')
            self.add_edge(Qubit.name + ' T2_Echo',
                          Qubit.name + ' T2_Star')
            self.add_edge(Qubit.name + ' Frequency Fine',
                          Qubit.name + ' T2_Echo')

            self.add_edge(Qubit.name + ' Flipping',
                          Qubit.name + ' Frequency Fine')
            self.add_edge(Qubit.name + ' MOTZOI Calibration',
                          Qubit.name + ' Flipping')
            self.add_edge(Qubit.name + ' Second Flipping',
                          Qubit.name + ' MOTZOI Calibration')
            self.add_edge(Qubit.name + ' ALLXY',
                          Qubit.name + ' Second Flipping')
            self.add_edge(Qubit.name + ' RB Fidelity',
                          Qubit.name + ' ALLXY')

        ################################
        # Multiplexed Readout Assessment
        ################################
        self.add_node('Device SSRO Fidelity',
                        calibrate_function=self.device.name + '.calibrate_optimal_weights_mux',
                        calibrate_function_args={'qubits': list([qubit for qubit in self.device.qubits()]), 'q_target': Qubit.name})

        ################################
        # Two-qubit Calibration
        ################################
        cardinal = {str(['QNW','QC']):'SE', str(['QNE','QC']):'SW', str(['QC','QSW']):'SW', str(['QC','QSE']):'SE', \
        			str(['QC','QSW','QSE']):'SW', str(['QC','QSE','QSW']):'SE'}

        for pair in [['QNW','QC'], ['QNE','QC'], ['QC','QSW','QSE'], ['QC','QSE','QSW']]:
	        self.add_node('{}-{} Theta Calibration'.format(pair[0], pair[1]),
	                        calibrate_function=self.device.name + '.calibrate_cz_thetas',
	                        calibrate_function_args={ 'operation_pairs': list([(pair,cardinal[str(pair)])]), 'phase_offset': 1 })

	        self.add_node('{}-{} Phases Calibration'.format(pair[0], pair[1]),
	                        calibrate_function=self.device.name + '.calibrate_phases',
	                        calibrate_function_args={ 'operation_pairs': list([(pair,cardinal[str(pair)])]) })

	    ################################
        # Two-qubit Assessment
        ################################
        for pair in [['QNW','QC'], ['QNE','QC'], ['QC','QSW','QSE'], ['QC','QSE','QSW']]:
        	self.add_node('{}-{} Conditional Oscillation'.format(pair[0], pair[1]),
	                        calibrate_function=self.device.name + '.measure_conditional_oscillation',
	                        calibrate_function_args={'q0':pair[0], 'q1':pair[1], 'q2':pair[2] if len(pair)==3 else None,
	                        						   'parked_qubit_seq':'ramsey' if len(pair)==3 else None})

        for pair in [['QNW','QC'], ['QNE','QC'], ['QC','QSW','QSE'], ['QC','QSE','QSW']]:
	        self.add_node('{}-{} Randomized Benchmarking'.format(pair[0], pair[1]),
	                        calibrate_function=self.device.name + '.measure_two_qubit_interleaved_randomized_benchmarking',
	                        calibrate_function_args={'qubits':pair, 'MC':self.device.instr_MC()})

	    ################################
        # Device Dependencies
        ################################
        self.add_node('Device Prepare Inspire',
                      calibrate_function=self.device.name + '.prepare_for_inspire')
        self.add_node('Upload Calibration Results',
                      calibrate_function=self.device.name + '.prepare_for_inspire')

        for Qubit in Qubit_list:
        	self.add_edge('Device SSRO Fidelity',
                          Qubit.name + ' RB Fidelity')

        for pair in [['QNW','QC'], ['QNE','QC'], ['QC','QSW','QSE'], ['QC','QSE','QSW']]:
        	self.add_edge('{}-{} Theta Calibration'.format(pair[0], pair[1]),
                          'Device SSRO Fidelity')
        	self.add_edge('{}-{} Phases Calibration'.format(pair[0], pair[1]),
                          '{}-{} Theta Calibration'.format(pair[0], pair[1]))
        	self.add_edge('{}-{} Conditional Oscillation'.format(pair[0], pair[1]),
                          '{}-{} Phases Calibration'.format(pair[0], pair[1]))
        	self.add_edge('{}-{} Randomized Benchmarking'.format(pair[0], pair[1]),
                          '{}-{} Conditional Oscillation'.format(pair[0], pair[1]))
        	self.add_edge('Device Prepare Inspire',
                          '{}-{} Randomized Benchmarking'.format(pair[0], pair[1]))
        self.add_edge('Upload Calibration Results', 'Device Prepare Inspire')

        self.cfg_plot_mode = 'svg'
        self.update_monitor()
        self.cfg_svg_filename

        url = self.open_html_viewer()
        print('Dependancy Graph Created. URL = ' + url)
