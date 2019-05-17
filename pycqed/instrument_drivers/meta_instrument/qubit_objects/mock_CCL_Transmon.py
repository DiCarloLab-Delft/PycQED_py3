from .CCL_Transmon import CCLight_Transmon

import numpy as np
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as ma
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object import Qubit
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

from autodepgraph import AutoDepGraph_DAG


class Mock_CCLight_Transmon(CCLight_Transmon):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_mock_params()

    def add_mock_params(self):
        """
        Add qubit parameters that are used to mock the system.

        These parameters are
            - prefixed with `mock_`
            - describe "hidden" parameters to mock real experiments.
        """

        self.add_parameter(
            'mock_freq_qubit', label='qubit frequency', unit='Hz',
            docstring='A fixed value, can be made fancier by making it depend on Flux through E_c, E_j and flux',
            parameter_class=ManualParameter)

        self.add_parameter('mock_freq_res', label='qubit frequency', unit='Hz',
                           parameter_class=ManualParameter)

        self.add_parameter('mock_mw_amp180', label='Pi-pulse amplitude', unit='V',
                           initial_value=0.5, parameter_class=ManualParameter)

    def measure_rabi(self, MC=None, amps=None,
                     analyze=True, close_fig=True, real_imag=True,
                     prepare_for_timedomain=True, all_modules=False):
        """
        Measurement is the same with and without vsm; therefore there is only 
        one measurement method rather than two. In the calibrate_mw_pulse_amp_coarse,
        the required parameter is updated.
        """
        if MC is None:
            MC = self.instr_MC.get_instr()
        if amps is None:
            amps = np.linspace(0.1, 1, 31)

        s = swf.None_Sweep()
        f_rabi = 1/self.mock_mw_amp180()
        mocked_values = np.cos(np.pi*f_rabi*amps)
        # Add (0,0.5) normal distributed noise
        mocked_values += np.random.normal(0, 0.1, np.size(mocked_values))
        d = det.Mock_Detector(value_names=['Rabi Amplitude'], value_units=['-'],
                              detector_control='soft', mock_values=mocked_values)  # Does not exist yet

        MC.set_sweep_function(s)
        MC.set_sweep_points(amps)
        MC.set_detector_function(d)
        MC.run('mock_rabi_')
        a = ma.Rabi_Analysis(label='rabi_')
        return a.rabi_amplitudes['piPulse']

###############################################################################
# AutoDepGraph
###############################################################################

    def dep_graph(self):
        cal_True_delayed = 'autodepgraph.node_functions.calibration_functions.test_calibration_True_delayed'
        self.dag = AutoDepGraph_DAG(name=self.name+' DAG')
        self.dag.add_node('VNA Analysis', calibrate_function=cal_True_delayed)

        self.dag.add_node('VNA Wide Search',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node('VNA Zoom on resonators',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node('Test Resonator Power Sweep',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node('Qubit Resonators Power Sweep',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node('Qubit Resonators Flux Sweep: All Qubits, All Flux Lines',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node('Qubit Resonators Flux Sweep: Dedicated Lines',
                          calibrate_function=cal_True_delayed)

        self.dag.add_node(' Resonator Frequency Coarse',
                          calibrate_function=cal_True_delayed)
        # Resonators
        self.dag.add_node(self.name + ' Resonator Frequency Fine',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' Resonator Power Scan',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' Resonator Flux Sweep',
                          calibrate_function=cal_True_delayed)

        # Calibration of instruments and ro
        self.dag.add_node(self.name + ' Calibrations',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' Mixer Skewness',
                          calibrate_function=self.name + '.calibrate_mixer_skewness_drive')
        self.dag.add_node(self.name + ' Mixer Offset Drive',
                          calibrate_function=self.name + '.calibrate_mixer_offsets_drive')
        self.dag.add_node(self.name + ' Mixer Offset Readout',
                          calibrate_function=self.name + '.calibrate_mixer_offsets_RO')
        self.dag.add_node(self.name + ' Ro/MW pulse timing',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' Ro Pulse Amplitude',
                          calibrate_function=self.name + '.ro_pulse_amp_CW')

        # Qubits calibration
        self.dag.add_node(self.name + ' Frequency Coarse',
                          calibrate_function=self.name + '.find_frequency')
        self.dag.add_node(self.name + ' Sweetspot',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' Rabi',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' Frequency Fine',
                          calibrate_function=self.name + '.calibrate_frequency_ramsey')
        self.dag.add_node(self.name + ' RO power',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' f_12 estimate',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' DAC Arc Polynomial',
                          calibrate_function=cal_True_delayed)

        # Validate qubit calibration
        self.dag.add_node(self.name + ' ALLXY',
                          calibrate_function=self.name + '.measure_allxy')
        self.dag.add_node(self.name + ' MOTZOI Calibration',
                          calibrate_function=self.name + '.calibrate_motzoi')
        # If all goes well, the qubit is fully 'calibrated' and can be controlled
        self.dag.add_node(self.name + ' Ready for measurement')

        # Qubits measurements
        self.dag.add_node(self.name + ' High Power Spectroscopy')
        self.dag.add_node(self.name + ' Anharmonicity')
        self.dag.add_node(self.name + ' Avoided Crossing')
        self.dag.add_node(self.name + ' T1')
        self.dag.add_node(self.name + ' T1(time)')
        self.dag.add_node(self.name + ' T1(frequency)')
        self.dag.add_node(self.name + ' T2_Echo')
        self.dag.add_node(self.name + ' T2_Echo(time)')
        self.dag.add_node(self.name + ' T2_Echo(frequency)')
        self.dag.add_node(self.name + ' T2_Star')
        self.dag.add_node(self.name + ' T2_Star(time)')
        self.dag.add_node(self.name + ' T2_Star(frequency)')

        # Resonators
        self.dag.add_edge(self.name + ' Resonator Frequency Fine',
                          ' Resonator Frequency Coarse')
        self.dag.add_edge(self.name + ' Resonator Power Scan',
                          self.name + ' Resonator Frequency Fine')
        self.dag.add_edge(self.name + ' Resonator Flux Sweep',
                          self.name + ' Resonator Power Scan')
        self.dag.add_edge(self.name + ' Resonator Flux Sweep',
                          self.name + ' Resonator Frequency Fine')

        # Qubit Calibrations
        self.dag.add_edge(self.name + ' Frequency Coarse',
                          self.name + ' Resonator Power Scan')
        self.dag.add_edge(self.name + ' Frequency Coarse',
                          self.name + ' Resonator Flux Sweep')
        # self.dag.add_edge(self.name + ' Frequency Coarse',
                          # self.name + ' Calibrations')

        # Calibrations
        self.dag.add_edge(self.name + ' Calibrations',
                          self.name + ' Mixer Skewness')
        self.dag.add_edge(self.name + ' Calibrations',
                          self.name + ' Mixer Offset Drive')
        self.dag.add_edge(self.name + ' Calibrations',
                          self.name + ' Mixer Offset Readout')
        self.dag.add_edge(self.name + ' Calibrations',
                          self.name + ' Ro/MW pulse timing')
        self.dag.add_edge(self.name + ' Calibrations',
                          self.name + ' Ro Pulse Amplitude')
        # Qubit
        self.dag.add_edge(self.name + ' Sweetspot',
                          self.name + ' Frequency Coarse')
        self.dag.add_edge(self.name + ' Rabi',
                          self.name + ' Sweetspot')
        self.dag.add_edge(self.name + ' Frequency Fine',
                          self.name + ' Sweetspot')
        self.dag.add_edge(self.name + ' Frequency Fine',
                          self.name + ' Rabi')

        self.dag.add_edge(self.name + ' ALLXY',
                          self.name + ' Rabi')
        self.dag.add_edge(self.name + ' ALLXY',
                          self.name + ' Frequency Fine')
        self.dag.add_edge(self.name + ' ALLXY',
                          self.name + ' RO power')
        self.dag.add_edge(self.name + ' Ready for measurement',
                          self.name + ' ALLXY')
        self.dag.add_edge(self.name + ' MOTZOI Calibration',
                          self.name + ' ALLXY')

        # Perform initial measurements to see if they make sense
        self.dag.add_edge(self.name + ' T1',
                          self.name + ' Ready for measurement')
        self.dag.add_edge(self.name + ' T2_Echo',
                          self.name + ' Ready for measurement')
        self.dag.add_edge(self.name + ' T2_Star',
                          self.name + ' Ready for measurement')

        # Measure as function of frequency and time
        self.dag.add_edge(self.name + ' T1(frequency)',
                          self.name + ' T1')
        self.dag.add_edge(self.name + ' T1(time)',
                          self.name + ' T1')

        self.dag.add_edge(self.name + ' T2_Echo(frequency)',
                          self.name + ' T2_Echo')
        self.dag.add_edge(self.name + ' T2_Echo(time)',
                          self.name + ' T2_Echo')

        self.dag.add_edge(self.name + ' T2_Star(frequency)',
                          self.name + ' T2_Star')
        self.dag.add_edge(self.name + ' T2_Star(time)',
                          self.name + ' T2_Star')

        self.dag.add_edge(self.name + ' DAC Arc Polynomial',
                          self.name + ' Sweetspot')

        # Measurements of anharmonicity and avoided crossing
        self.dag.add_edge(self.name + ' f_12 estimate',
                          self.name + ' Frequency Fine')
        self.dag.add_edge(self.name + ' High Power Spectroscopy',
                          self.name + ' Sweetspot')
        self.dag.add_edge(self.name + ' Anharmonicity',
                          self.name + ' High Power Spectroscopy')
        self.dag.add_edge(self.name + ' Avoided Crossing',
                          self.name + ' DAC Arc Polynomial')

        self.dag.cfg_plot_mode = 'svg'
        self.dag.update_monitor()
        self.dag.cfg_svg_filename
        url = self.dag.open_html_viewer()
        print(url)

    # def show_dep_graph(self, dag):
    #     self.dag.cfg_plot_mode = 'svg'
    #     self.dag.update_monitor()
    #     self.dag.cfg_svg_filename
    #     url = self.dag.open_html_viewer()
    #     print(url)
