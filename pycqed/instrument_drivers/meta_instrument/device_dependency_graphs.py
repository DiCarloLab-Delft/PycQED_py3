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

        # Initial Resonator things:
        self.add_node('Resonators Wide Search',
                      calibrate_function=Qubit.name + '.find_resonators')
        self.add_node('Zoom on resonators',
                      calibrate_function=Qubit.name + '.find_resonator_frequency_initial')
        self.add_node('Resonators Power Scan',
                      calibrate_function=Qubit.name + '.find_test_resonators')
        self.add_node('Resonators Flux Sweep',
                      calibrate_function=Qubit.name + '.find_qubit_resonator_fluxline')

        self.add_edge('Zoom on resonators', 'Resonators Wide Search')
        self.add_edge('Resonators Power Scan',
                      'Zoom on resonators')
        self.add_edge('Resonators Flux Sweep',
                      'Zoom on resonators')
        self.add_edge('Resonators Flux Sweep',
                      'Resonators Power Scan')

        # Multi-qubit measurements:
        self.add_node('{} - {} avoided crossing'.format(Qubit_list[0].name,
                                                        Qubit_list[1].name),
                      calibrate_function= Qubit_list[0].name+'.measure_avoided_crossing')
        self.add_node('Two Qubit ALLXY',
                      calibrate_function=self.device.name + '.measure_two_qubit_allxy')

        # Qubit specific methods
        for Qubit in Qubit_list:
            # Resonators
            self.add_node(Qubit.name + ' Resonator Frequency',
                          calibrate_function=Qubit.name + '.find_resonator_frequency')
            self.add_node(Qubit.name + ' Resonator Power Scan',
                          calibrate_function=Qubit.name + '.calibrate_ro_pulse_amp_CW')

            # Calibration of instruments and ro
            self.add_node(Qubit.name + ' Calibrations',
                          calibrate_function=cal_True_delayed)
            self.add_node(Qubit.name + ' Mixer Skewness Drive',
                          calibrate_function=cal_True_delayed)
                          # calibrate_function=Qubit.name + '.calibrate_mixer_skewness_drive')
            self.add_node(Qubit.name + ' Mixer Skewness Readout',
                          calibrate_function=cal_True_delayed)
                          # calibrate_function=Qubit.name + '.calibrate_mixer_skewness_RO')
            self.add_node(Qubit.name + ' Mixer Offset Drive',
                          calibrate_function=Qubit.name + '.calibrate_mixer_offsets_drive')
            self.add_node(Qubit.name + ' Mixer Offset Readout',
                          calibrate_function=Qubit.name + '.calibrate_mixer_offsets_RO')

            # Qubits calibration
            self.add_node(Qubit.name + ' Prepare Characterizing',
                          calibrate_function=Qubit.name + '.prepare_characterizing')
            self.add_node(Qubit.name + ' Frequency Coarse',
                          calibrate_function=Qubit.name + '.find_frequency_adaptive',
                          check_function=Qubit.name + '.check_qubit_spectroscopy',
                          tolerance=0.2e-3)
            self.add_node(Qubit.name + ' Frequency at Sweetspot',
                          calibrate_function=Qubit.name + '.find_frequency')
            self.add_node(Qubit.name + ' Spectroscopy Power',
                          calibrate_function=Qubit.name + '.calibrate_spec_pow')
            self.add_node(Qubit.name + ' Sweetspot',
                          calibrate_function=Qubit.name + '.find_qubit_sweetspot')
            self.add_node(Qubit.name + ' Rabi',
                          calibrate_function=Qubit.name + '.calibrate_mw_pulse_amplitude_coarse',
                          check_function=Qubit.name + '.check_rabi',
                          tolerance=0.01)
            self.add_node(Qubit.name + ' Frequency Fine',
                          calibrate_function=Qubit.name + '.calibrate_frequency_ramsey',
                          check_function=Qubit.name + '.check_ramsey',
                          tolerance=0.1e-3)
            self.add_node(Qubit.name + ' f_12 estimate',
                          calibrate_function=Qubit.name + '.find_anharmonicity_estimate')
            self.add_node(Qubit.name + ' DAC Arc Polynomial',
                          calibrate_function=Qubit.name + '.measure_flux_arc_tracked_spectroscopy')

            # Validate qubit calibration
            self.add_node(Qubit.name + ' ALLXY',
                          calibrate_function=Qubit.name + '.calibrate_mw_gates_allxy')
            self.add_node(Qubit.name + ' MOTZOI Calibration',
                          calibrate_function=Qubit.name + '.calibrate_motzoi')

            # If all goes well, the qubit is fully 'calibrated' and can be controlled

            # Qubits measurements
            self.add_node(Qubit.name + ' Anharmonicity')
            # self.add_node(Qubit.name + ' Avoided Crossing')
            self.add_node(Qubit.name + ' T1')
            self.add_node(Qubit.name + ' T1(time)')
            self.add_node(Qubit.name + ' T1(frequency)')
            self.add_node(Qubit.name + ' T2_Echo')
            self.add_node(Qubit.name + ' T2_Echo(time)')
            self.add_node(Qubit.name + ' T2_Echo(frequency)')
            self.add_node(Qubit.name + ' T2_Star')
            self.add_node(Qubit.name + ' T2_Star(time)')
            self.add_node(Qubit.name + ' T2_Star(frequency)')
            ###################################################################
            # EDGES
            ###################################################################

            # Resonators
            self.add_edge(Qubit.name + ' Resonator Frequency',
                          Qubit.name + ' Prepare Characterizing')
            self.add_edge(Qubit.name + ' Resonator Frequency',
                          Qubit.name + ' Resonator Power Scan',)
            self.add_edge(Qubit.name + ' Resonator Power Scan',
                          Qubit.name + ' Prepare Characterizing')
            self.add_edge(Qubit.name + ' Frequency Coarse',
                          Qubit.name + ' Resonator Power Scan')
            self.add_edge(Qubit.name + ' Frequency Coarse',
                          Qubit.name + ' Resonator Frequency')

            # Calibrations
            self.add_edge(Qubit.name + ' Calibrations',
                          Qubit.name + ' Mixer Skewness Readout')
            self.add_edge(Qubit.name + ' Calibrations',
                          Qubit.name + ' Mixer Skewness Drive')
            self.add_edge(Qubit.name + ' Calibrations',
                          Qubit.name + ' Mixer Offset Drive')
            self.add_edge(Qubit.name + ' Calibrations',
                          Qubit.name + ' Mixer Offset Readout')
            self.add_edge(Qubit.name + ' Resonator Power Scan',
                          Qubit.name + ' Calibrations')

            # Qubit
            self.add_edge(Qubit.name + ' Prepare Characterizing',
                          'Resonators Flux Sweep')
            self.add_edge(Qubit.name + ' Spectroscopy Power',
                          Qubit.name + ' Frequency Coarse')
            self.add_edge(Qubit.name + ' Sweetspot',
                          Qubit.name + ' Frequency Coarse')
            self.add_edge(Qubit.name + ' Sweetspot',
                          Qubit.name + ' Spectroscopy Power')
            self.add_edge(Qubit.name + ' Rabi',
                          Qubit.name + ' Frequency at Sweetspot')
            self.add_edge(Qubit.name + ' Frequency Fine',
                          Qubit.name + ' Frequency at Sweetspot')
            self.add_edge(Qubit.name + ' Frequency Fine',
                          Qubit.name + ' Rabi')

            self.add_edge(Qubit.name + ' Frequency at Sweetspot',
                          Qubit.name + ' Sweetspot')

            self.add_edge(Qubit.name + ' ALLXY',
                          Qubit.name + ' Rabi')
            self.add_edge(Qubit.name + ' ALLXY',
                          Qubit.name + ' Frequency Fine')
            self.add_edge(Qubit.name + ' ALLXY',
                          Qubit.name + ' MOTZOI Calibration')

            # Perform initial measurements to see if they make sense
            self.add_edge(Qubit.name + ' T1',
                          Qubit.name + ' ALLXY')
            self.add_edge(Qubit.name + ' T2_Echo',
                          Qubit.name + ' ALLXY')
            self.add_edge(Qubit.name + ' T2_Star',
                          Qubit.name + ' ALLXY')

            # Measure as function of frequency and time
            self.add_edge(Qubit.name + ' T1(frequency)',
                          Qubit.name + ' T1')
            self.add_edge(Qubit.name + ' T1(time)',
                          Qubit.name + ' T1')

            self.add_edge(Qubit.name + ' T2_Echo(frequency)',
                          Qubit.name + ' T2_Echo')
            self.add_edge(Qubit.name + ' T2_Echo(time)',
                          Qubit.name + ' T2_Echo')

            self.add_edge(Qubit.name + ' T2_Star(frequency)',
                          Qubit.name + ' T2_Star')
            self.add_edge(Qubit.name + ' T2_Star(time)',
                          Qubit.name + ' T2_Star')

            self.add_edge(Qubit.name + ' DAC Arc Polynomial',
                          Qubit.name + ' Frequency at Sweetspot')

            # Measurements of anharmonicity and avoided crossing
            self.add_edge(Qubit.name + ' f_12 estimate',
                          Qubit.name + ' Frequency at Sweetspot')
            self.add_edge(Qubit.name + ' Anharmonicity',
                          Qubit.name + ' f_12 estimate')
            # self.add_edge(Qubit.name + ' Avoided Crossing',
            #               Qubit.name + ' DAC Arc Polynomial')



        self.add_edge('Two Qubit ALLXY',
                      self.device.qubits()[0] + ' ALLXY')
        self.add_edge('Two Qubit ALLXY',
                      self.device.qubits()[1] + ' ALLXY')
        self.add_edge('{} - {} avoided crossing'.format(Qubit_list[0].name,
                                                        Qubit_list[1].name),
                      Qubit_list[0].name + ' DAC Arc Polynomial')
        self.add_edge('{} - {} avoided crossing'.format(Qubit_list[0].name,
                                                        Qubit_list[1].name),
                      Qubit_list[1].name + ' DAC Arc Polynomial')
        self.cfg_plot_mode = 'svg'
        self.update_monitor()
        self.cfg_svg_filename

        url = self.open_html_viewer()
        print('Dependancy Graph Created. URL = '+url)
        # self.open_html_viewer()
