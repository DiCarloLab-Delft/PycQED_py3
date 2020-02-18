from autodepgraph import AutoDepGraph_DAG
###########################################################################
# AutoDepGraph
###########################################################################
"""
Dependancy graphs for different devices can be made here. Usually, a new class
is created for each new device, with nameing convention 'device_dep_graph' with
device the name of the device (e.g. Octobox).

Graphs usually consist of two parts:
- A global part which normally is just charactarizing resonators
- A qubit specific part, which is made with a loop over all qubits

For now, the general part is performed by calling functions over a 'fakequbit',
which is a hacky way of doing the resonator measurements without setting any
parameters in the real qubits.
The calibration methods should already be compatible with this way of
characterizing, and they should already update the relevant qubit parameters
(but it won't hurt to check)

Args:
    name    (str)   : Name of the dependency graph (not sure what this does).
    device  (device): The device object that you would like to make a graph of.

This should be called as 'dag = DDG.device_dep_graph(name, device)'. Then you
can work with the dag object that was just created, and do any operation on
there.
"""


class octobox_dep_graph(AutoDepGraph_DAG):
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
        Qubit = self.device.find_instrument('fakequbit')

        cal_True_delayed = 'autodepgraph.node_functions.calibration_functions.test_calibration_True_delayed'

        ########################################################
        # GRAPH NODES
        ########################################################
        self.add_node('All Qubits at Sweetspot')

        for Qubit in Qubit_list:
            ################################
            # Qubit Characterization
            ################################
            self.add_node(Qubit.name + ' Rabi',
                          calibrate_function=Qubit.name +
                            '.calibrate_mw_pulse_amplitude_coarse',
                          check_function=Qubit.name + '.check_rabi',
                          tolerance=0.01)
            self.add_node(Qubit.name + ' Frequency Fine',
                          calibrate_function=Qubit.name +
                            '.calibrate_frequency_ramsey',
                          check_function=Qubit.name + '.check_ramsey',
                          tolerance=0.1e-3)
            self.add_node(Qubit.name + ' f_12 estimate',
                          calibrate_function=Qubit.name +
                            '.find_anharmonicity_estimate')
            self.add_node(Qubit.name + ' Anharmonicity',
                           calibrate_function = Qubit.name +
                            '.measure_anharmonicity_test')

            ################################
            # Single Qubit Gate Calibration
            ################################
            self.add_node(Qubit.name + ' ALLXY',
                          calibrate_function=Qubit.name +
                            '.calibrate_mw_gates_allxy2')

            ################################
            # Readout Calibration
            ################################
            self.add_node(Qubit.name + ' Acquisition Delay Calibration',
                          calibrate_function=Qubit.name +
                            '.calibrate_ro_acq_delay')
            self.add_node(Qubit.name + ' Dispersive Shift',
                          calibrate_function=Qubit.name +
                            '.measure_dispersive_shift_pulsed')
            self.add_node(Qubit.name + ' SSRO Coarse tune-up',
                          calibrate_function=Qubit.name +
                            '.calibrate_ssro_coarse')
            self.add_node(Qubit.name + ' SSRO Pulse Duration',
                          calibrate_function=Qubit.name +
                            '.calibrate_ssro_pulse_duration')
            self.add_node(Qubit.name + ' SSRO Optimization',
                          calibrate_function=Qubit.name +
                            '.calibrate_ssro_fine')
            self.add_node(Qubit.name + ' RO mixer calibration',
                          calibrate_function=Qubit.name +
                            '.calibrate_mixer_offsets_RO')
            self.add_node(Qubit.name + ' SSRO Fidelity',
                          calibrate_function=Qubit.name + '.measure_ssro',
                          calibrate_function_args={'post_select': True})

            ##############################
            # Coherence Measurments
            ##############################
            self.add_node(Qubit.name + ' T1',
                           calibrate_function = Qubit.name + '.measure_T1')
            self.add_node(Qubit.name + ' T2_Echo',
                           calibrate_function = Qubit.name + '.measure_echo')
            self.add_node(Qubit.name + ' T2_Star',
                           calibrate_function = Qubit.name + '.measure_ramsey')

            ###################################################################
            # DEPENDENCIES
            ###################################################################
            self.add_edge(Qubit.name + ' Rabi',
                          'All Qubits at Sweetspot')
            self.add_edge(Qubit.name + ' Frequency Fine',
                          Qubit.name + ' Rabi')

            self.add_edge(Qubit.name + ' ALLXY',
                          Qubit.name + ' Rabi')
            self.add_edge(Qubit.name + ' ALLXY',
                          Qubit.name + ' Frequency Fine')
            self.add_edge(Qubit.name + ' Acquisition Delay Calibration',
                          Qubit.name + ' Rabi')
            self.add_edge(Qubit.name + ' Dispersive Shift',
                          Qubit.name + ' Rabi')
            self.add_edge(Qubit.name + ' SSRO Coarse tune-up',
                          Qubit.name + ' Dispersive Shift')
            self.add_edge(Qubit.name + ' SSRO Coarse tune-up',
                          Qubit.name + ' Acquisition Delay Calibration')
            self.add_edge(Qubit.name + ' SSRO Pulse Duration',
                          Qubit.name + ' SSRO Coarse tune-up')
            self.add_edge(Qubit.name + ' SSRO Optimization',
                          Qubit.name + ' SSRO Pulse Duration')
            self.add_edge(Qubit.name + ' SSRO Fidelity',
                          Qubit.name + ' SSRO Optimization')
            self.add_edge(Qubit.name + ' SSRO Fidelity',
                          Qubit.name + ' RO mixer calibration')

            self.add_edge(Qubit.name + ' T1',
                          Qubit.name + ' Frequency Fine')
            self.add_edge(Qubit.name + ' T2_Echo',
                          Qubit.name + ' Frequency Fine')
            self.add_edge(Qubit.name + ' T2_Star',
                          Qubit.name + ' Frequency Fine')

            self.add_edge(Qubit.name + ' f_12 estimate',
                          'All Qubits at Sweetspot')
            self.add_edge(Qubit.name + ' Anharmonicity',
                          Qubit.name + ' f_12 estimate')

        self.cfg_plot_mode = 'svg'
        self.update_monitor()
        self.cfg_svg_filename

        url = self.open_html_viewer()
        print('Dependancy Graph Created. URL = '+url)
        # self.open_html_viewer()
