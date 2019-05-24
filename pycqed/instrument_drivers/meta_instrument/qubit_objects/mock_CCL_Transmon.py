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

        self.add_parameter('mock_Ec', label='charging energy', unit='Hz',
                           parameter_class=ManualParameter, initial_value=266.3e6)

        self.add_parameter('mock_Ej', label='josephson energy', unit='Hz',
                           parameter_class=ManualParameter, initial_value=17.76e9)

        self.add_parameter('mock_freq_qubit', label='qubit frequency', unit='Hz',
                           docstring='A fixed value, can be made fancier by making it depend on Flux through E_c, E_j and flux',
                           parameter_class=ManualParameter,
                           initial_value=np.sqrt(8*self.mock_Ec()*self.mock_Ej())-self.mock_Ec())

        self.add_parameter('mock_freq_res', label='resonator frequency', unit='Hz',
                           parameter_class=ManualParameter, initial_value=7.487628e9)

        self.add_parameter('mock_ro_pulse_amp_CW', label='Readout pulse amplitude',
                           unit='Hz', parameter_class=ManualParameter,
                           initial_value=0.048739)

        self.add_parameter('mock_res_power_shift', label='resonator power shift',
                           unit='Hz', parameter_class=ManualParameter,
                           initial_value=2.6e6)

        self.add_parameter('mock_residual_flux_current', label='magnitude of sweetspot current',
                           unit='A', parameter_class=ManualParameter,
                           initial_value=0.5e-3)

        self.add_parameter('mock_mw_amp180', label='Pi-pulse amplitude', unit='V',
                           initial_value=0.5, parameter_class=ManualParameter)

        self.add_parameter('mock_T1', label='relaxation time', unit='s',
                           initial_value=29e-6, parameter_class=ManualParameter)

        self.add_parameter('mock_T2_star', label='Ramsey T2', unit='s',
                           initial_value=1e-6, parameter_class=ManualParameter)

    def find_resonator_power(self, freqs=None, powers=None):
        if freqs is None:
            freq_center = self.freq_res()
            freq_range = 10e6
            freqs = np.arange(freq_center-1/2*freq_range, freq_center+1/2*freq_range,
                0.1e6)

        if powers is None:
            powers = np.arange(-40, 0, 4)

        self.measure_resonator_power(freqs=freqs, powers=powers)

        return True

    def find_resonator_sweetspot(self, freqs=None, dac_values=None):
        if freqs is None:
            freq_center = self.freq_res()
            freq_range = 10e6
            freqs = np.arange(freq_center-1/2*freq_range, freq_center+1/2*freq_range,
                0.5e6)

        if dac_values is None:
            dac_values = np.arange(-1e-3, -1e3, )

        return True

    def measure_spectroscopy(self, freqs, pulsed=True, MC=None, analyze=True,
                             close_fig=True, label='',
                             prepare_for_continuous_wave=True):
        '''
        Can be made fancier by implementing different types of spectroscopy 
        (e.g. pulsed/CW) and by imp using cfg_spec_mode.

        Uses a Lorentzian as a result for now
        '''
        if MC is None:
            MC = self.instr_MC.get_instr()

        s = swf.None_Sweep()
        h = 10e-3  # Lorentian baseline [V]
        A = 9e-3   # Height of peak [V]
        w = 4e6    # Full width half maximum of peak
        mocked_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                            ((freqs - self.mock_freq_qubit()))**2)

        mocked_values += np.random.normal(0, 1e-3, np.size(mocked_values))

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft', mock_values=mocked_values)
        MC.set_sweep_function(s)
        MC.set_sweep_points(freqs)

        MC.set_detector_function(d)
        MC.run('mock_spectroscopy_'+self.msmt_suffix)

        a = ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)
        return True  # a.fit_params['f0']

    def measure_resonator_power(self, freqs, powers, MC=None,
                                analyze: bool = True, close_fig: bool = True,
                                fluxChan=None):
        if MC is None:
            MC = self.instr_MC.get_instr()

        s1 = swf.None_Sweep(name='Heterodyne Frequency', parameter_name='Frequency',
                            unit='Hz')
        s2 = swf.None_Sweep(name='Readout Power', parameter_name='Power',
                            unit='dBm')

        res_power = 20*np.log10(self.mock_ro_pulse_amp_CW())
        pow_shift = 20
        mocked_values = []
        h = 10e-3
        for power in powers:
            if power <= res_power:
                # Good signal
                pulse_amp = 10**(-power/20)
                A = 9e-3
                w = 0.5e6
                f0 = self.mock_freq_res()
                new_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                                 ((freqs - f0))**2)
            elif (power > res_power) and (power < res_power + pow_shift):
                # no signal at all -> width increases, peak decreases
                pulse_amp = 10**(-power/20)

                # A = 9e-3-8.99e-3*1/2*(1+np.sin(np.pi*(power - res_power)/pow_shift))
                A0 = 9e-3
                n = 6
                b = A0/10
                a = (A0-b)/(-1/2*pow_shift)**n
                A = a*(power - (res_power+1/2*pow_shift))**n + b

                w = 0.5e6
                w = 0.5e6 + 0.5e6*np.sin(np.pi*(power - res_power)/pow_shift)

                b = res_power + self.mock_freq_res()*(pow_shift/self.mock_res_power_shift())
                f0 = (b - power)/pow_shift*self.mock_res_power_shift()

                new_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                                 ((freqs - f0))**2)

                for i, value in enumerate(new_values):
                    d = np.abs(value - A0)

                    value += np.random.normal(0, d/5, 1)
                    new_values[i] = value
                # new_values += np.random.normal(0, 1e-3, np.size(new_values))
            else:
                # High power regime
                A = 9e-3
                w = 0.5e6
                f0 = self.mock_freq_res() - self.mock_res_power_shift()
                new_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                                 ((freqs - f0))**2)
            mocked_values = np.concatenate([mocked_values, new_values])

        mocked_values += np.random.normal(0, 1e-4, np.size(mocked_values))
        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft', mock_values=mocked_values)

        MC.set_sweep_function(s1)
        MC.set_sweep_function_2D(s2)

        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(powers)

        MC.set_detector_function(d)
        MC.run('Resonator_power_scan'+self.msmt_suffix, mode='2D')

        if analyze:
            ma.TwoD_Analysis(label='Resonator_power_scan',
                             close_fig=close_fig, normalize=True)

    def measure_heterodyne_spectroscopy(self, freqs, MC=None, analyze=True, close_fig=True):
        '''
        For finding resonator frequencies. Uses a lorentzian fit for now, might
        be extended in the future

        Dependent on readout power
        '''
        if MC is None:
            MC = self.instr_MC.get_instr()

        s = swf.None_Sweep()

        if self.ro_pulse_amp_CW() > self.mock_ro_pulse_amp_CW():
            # High power regime, shift by some value
            shift = self.mock_res_power_shift()
            self.mock_freq_res(self.mock_freq_res()-shift)

        h = 10e-3  # Lorentian baseline [V]
        A = 9e-3   # Height of peak [V]
        w = 0.5e6    # Full width half maximum of peak
        mocked_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                            ((freqs - self.mock_freq_res()))**2)
        mocked_values += np.random.normal(0, 0.5e-3, np.size(mocked_values))

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft', mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(freqs)
        MC.set_detector_function(d)
        MC.run('Resonator_scan'+self.msmt_suffix)

    def measure_resonator_frequency_dac_scan(self, freqs, dac_values, pulsed=True,
                                             MC=None, analyze=True, fluxChan=None,
                                             close_fig=True,
                                             nested_resonator_calibration=False,
                                             resonator_freqs=None):
        '''
        Measures resonator frequency versus flux. Simple model which just shifts
        the lorentzian peak with a simple cosine by 2 MHz.
        '''

        if MC is None:
            MC = self.instr_MC.get_instr()
        s1 = swf.None_Sweep(name='Heterodyne Frequency', parameter_name='Frequency',
                            unit='Hz')
        s2 = swf.None_Sweep(name='Flux', parameter_name='FBL',
                            unit='A')

        freq_res = self.mock_freq_res()
        Iref = self.mock_residual_flux_current()
        I0 = 10e-3
        df = 2e6

        mocked_values = []
        for dac_value in dac_values:
            h = 10e-3  # Lorentian baseline [V]
            A = 9e-3   # Height of peak [V]
            w = 0.5e6    # Full width half maximum of peak
            f0 = freq_res - df*(np.sin(1/2*np.pi*(dac_value-Iref)/I0))**2
            new_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                             ((freqs - f0))**2)
            new_values += np.random.normal(0, 0.1e-3, np.size(new_values))

            mocked_values = np.concatenate([mocked_values, new_values])

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft', mock_values=mocked_values)

        MC.set_sweep_function(s1)
        MC.set_sweep_function_2D(s2)

        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(dac_values)

        MC.set_detector_function(d)
        MC.run('Resonator_dac_scan'+self.msmt_suffix, mode='2D')

        if analyze:
            ma.TwoD_Analysis(label='Resonator_dac_scan', close_fig=close_fig)

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

    def measure_ramsey(self, times=None, MC=None,
                       artificial_detuning: float = None,
                       freq_qubit: float = None, label: str = '',
                       prepare_for_timedomain=True, analyze=True,
                       close_fig=True, update=True, detector=False, double_fit=False):
        if MC is None:
            MC = self.instr_MC.get_instr()

        if times is None:
            stepsize = (self.mock_T2_star()*4/61)//(abs(self.cfg_cycle_time())) \
                * abs(self.cfg_cycle_time())
            times = np.arange(0, self.mock_T2_star()*4, stepsize)

        if artificial_detuning is None:
            artificial_detuning = 5/times[-1]

        # Calibration points:
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1]+1*dt,
                                 times[-1]+2*dt,
                                 times[-1]+3*dt,
                                 times[-1]+4*dt)])
        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        if freq_qubit is None:
            freq_qubit = self.freq_qubit()

        # self.instr_Lo_mw.get_instr().set('frequency', freq_qubit -
            # self.mw_freq_mod.get() + artificial_detuning)

        s = swf.None_Sweep()
        phase = 0
        oscillation_offset = 0
        exponential_offset = 0.5
        frequency = self.mock_freq_qubit() - freq_qubit + artificial_detuning
        # Mock values without calibration points
        mocked_values = 0.5 * np.exp(-(times[0:-4] / self.mock_T2_star())) * (np.cos(
            2 * np.pi * frequency * times[0:-4] + phase) + oscillation_offset) + exponential_offset
        mocked_values = np.concatenate(
            [mocked_values, (0, 0, 1, 1)])  # Calibration points
        # Add noise:
        mocked_values += np.random.normal(0, 0.01, np.size(mocked_values))

        d = det.Mock_Detector(value_names=['F|1>'], value_units=['V'],
                              detector_control='soft', mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('mock_ramsey_')

        if analyze:
            a = ma.Ramsey_Analysis(auto=True, closefig=True,
                                   freq_qubit=freq_qubit,
                                   artificial_detuning=artificial_detuning)
            self.T2_star(a.T2_star['T2_star'])
            res = {'T2star': a.T2_star['T2_star'],
                   'frequency': a.qubit_frequency}
            return res

    def measure_T1(self, times=None, MC=None, analyze=True, close_fig=True,
                   update=True, prepare_for_timedomain=True):
        '''
        Very simple version that just returns a exponential decay based on mock_T1.
        Might be improved by making it depend on how close your pulse amp is.
        '''
        if MC is None:
            MC = self.instr_MC.get_instr()

        if times is None:
            times = np.linspace(0, self.T1()*4, 31)
        # Calibration points
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1]+1*dt,
                                 times[-1]+2*dt,
                                    times[-1]+3*dt,
                                    times[-1]+4*dt)])

        s = swf.None_Sweep()

        amplitude = 1
        offset = 0
        mocked_values = amplitude * \
            np.exp(-(times[0:-4]/self.mock_T1())) + offset
        mocked_values = np.concatenate([mocked_values, (0, 0, 1, 1)])

        mocked_values += np.random.normal(0, 0.05, np.size(mocked_values))

        d = det.Mock_Detector(value_names=['F|1>'], value_units=[''],
                              detector_control='soft', mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('mock_T1_')

        if analyze:
            a = ma.T1_Analysis(auto=True, close_fig=True)
            if update:
                self.T1(a.T1)
            return a.T1

    ###########################################################################
    # AutoDepGraph
    ###########################################################################

    def dep_graph(self):
        cal_True_delayed = 'autodepgraph.node_functions.calibration_functions.test_calibration_True_delayed'
        self.dag = AutoDepGraph_DAG(name=self.name+' DAG')

        self.dag.add_node('VNA Wide Search',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node('VNA Zoom on resonators',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node('VNA Resonators Power Scan',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node('VNA Resonators Flux Sweep',
                          calibrate_function=cal_True_delayed)

        # Resonators
        self.dag.add_node(self.name + ' Resonator Frequency',
                          calibrate_function=self.name + '.find_resonator_frequency')
        self.dag.add_node(self.name + ' Resonator Power Scan',
                          calibrate_function=self.name + '.find_resonator_power')
        self.dag.add_node(self.name + ' Resonator Sweetspot',
                          calibrate_function=self.name + '.find_resonator_sweetspot')

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
        #######################################################################
        # EDGES
        #######################################################################

        # VNA
        self.dag.add_edge('VNA Zoom on resonators', 'VNA Wide Search')
        self.dag.add_edge('VNA Resonators Power Scan',
                          'VNA Zoom on resonators')
        self.dag.add_edge('VNA Resonators Flux Sweep',
                          'VNA Zoom on resonators')
        self.dag.add_edge('VNA Resonators Flux Sweep',
                          'VNA Resonators Power Scan')
        # Resonators
        self.dag.add_edge(self.name + ' Resonator Frequency',
                          'VNA Resonators Power Scan')
        self.dag.add_edge(self.name + ' Resonator Frequency',
                          'VNA Resonators Flux Sweep')
        self.dag.add_edge(self.name + ' Resonator Power Scan',
                          self.name + ' Resonator Frequency')
        self.dag.add_edge(self.name + ' Resonator Sweetspot',
                          self.name + ' Resonator Frequency')
        self.dag.add_edge(self.name + ' Frequency Coarse',
                          self.name + ' Resonator Power Scan')
        self.dag.add_edge(self.name + ' Frequency Coarse',
                          self.name + ' Resonator Sweetspot')

        # Qubit Calibrations
        self.dag.add_edge(self.name + ' Frequency Coarse',
                          self.name + ' Resonator Frequency')
        # self.dag.add_edge(self.name + ' Frequency Coarse',
        #                   self.name + ' Calibrations')

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
