from .CCL_Transmon import CCLight_Transmon
import os
import time
import numpy as np
import logging
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.analysis_v2 import spectroscopy_analysis as sa2
import pycqed.analysis.analysis_toolbox as a_tools
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from autodepgraph import AutoDepGraph_DAG
import matplotlib.pyplot as plt
from pycqed.analysis import fitting_models as fm


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
        # Qubit resonator
        self.add_parameter('mock_freq_res_bare', label='bare resonator freq',
                           unit='Hz', parameter_class=ManualParameter,
                           initial_value=7.487628e9)

        self.add_parameter('mock_Qe', parameter_class=ManualParameter,
                           initial_value=13945)

        self.add_parameter('mock_Q', parameter_class=ManualParameter,
                           initial_value=8459)

        self.add_parameter('mock_theta', parameter_class=ManualParameter,
                           initial_value=-0.1)

        self.add_parameter('mock_slope', parameter_class=ManualParameter,
                           initial_value=0)

        self.add_parameter('mock_phi_I', parameter_class=ManualParameter,
                           initial_value=0)

        self.add_parameter('mock_phi_0', parameter_class=ManualParameter,
                           initial_value=0)

        self.add_parameter('mock_pow_shift',
                           label='power needed for low to high regime',
                           unit='dBm', initial_value=20,
                           parameter_class=ManualParameter)

        # Test resonator
        self.add_parameter('mock_freq_test_res',
                           label='test resonator frequency',
                           unit='Hz', parameter_class=ManualParameter,
                           initial_value=7.76459e9)

        self.add_parameter('mock_test_Qe', parameter_class=ManualParameter,
                           initial_value=1.8e6)

        self.add_parameter('mock_test_Q', parameter_class=ManualParameter,
                           initial_value=1e6)

        self.add_parameter('mock_test_theta', parameter_class=ManualParameter,
                           initial_value=-0.1)

        self.add_parameter('mock_test_slope', parameter_class=ManualParameter,
                           initial_value=0)

        self.add_parameter('mock_test_phi_I', parameter_class=ManualParameter,
                           initial_value=0)

        self.add_parameter('mock_test_phi_0', parameter_class=ManualParameter,
                           initial_value=0)

        # Qubit
        self.add_parameter('mock_Ec', label='charging energy', unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=266.3e6)

        self.add_parameter('mock_Ej1', label='josephson energy', unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=8.647e9)

        self.add_parameter('mock_Ej2', label='josephson energy', unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=8.943e9)

        self.add_parameter('mock_freq_qubit_bare', label='qubit frequency',
                           unit='Hz',
                           initial_value=(np.sqrt(8*self.mock_Ec() *
                                                  (self.mock_Ej1()+self.mock_Ej2())) -
                                          self.mock_Ec()),
                           parameter_class=ManualParameter)

        self.add_parameter('mock_anharmonicity', label='anharmonicity',
                           unit='Hz', initial_value=self.mock_Ec(),
                           parameter_class=ManualParameter)

        # Qubit flux
        self.add_parameter('mock_fl_dc_I_per_phi0', unit='A/Wb',
                           initial_value={'FBL_Q1': 20e-3,
                                          'FBL_Q2': 2},
                           parameter_class=ManualParameter)

        self.add_parameter('mock_sweetspot_phi_over_phi0',
                           label='magnitude of sweetspot flux',
                           unit='-', parameter_class=ManualParameter,
                           initial_value=0.02)

        self.add_parameter('mock_fl_dc_ch',
                           label='most closely coupled fluxline',
                           unit='', initial_value='FBL_Q1',
                           parameter_class=ManualParameter)

        # Qubit-resonator interaction
        self.add_parameter('mock_coupling01',
                           label='coupling qubit to resonator', unit='Hz',
                           initial_value=60e6, parameter_class=ManualParameter)

        self.add_parameter('mock_coupling12',
                           label='coupling 12 transition to resonator',
                           unit='Hz', initial_value=60e6,
                           parameter_class=ManualParameter)

        self.add_parameter('mock_chi01', label='coupling 01 transition',
                           unit='Hz',
                           initial_value=(self.mock_coupling01())**2 /
                                         (self.mock_freq_qubit_bare() -
                                          self.mock_freq_res_bare()),
                           parameter_class=ManualParameter)

        self.add_parameter('mock_chi12', label='coupling 12 transition',
                           unit='Hz',
                           initial_value=(self.mock_coupling12())**2 /
                                         (self.mock_freq_qubit_bare() +
                                          self.mock_anharmonicity() -
                                          self.mock_freq_res_bare()),
                           parameter_class=ManualParameter)

        self.add_parameter('mock_chi', label='dispersive shift', unit='Hz',
                           initial_value=self.mock_chi01()-self.mock_chi12()/2,
                           parameter_class=ManualParameter)

        # Readout parameters
        self.add_parameter('mock_ro_pulse_amp_CW',
                           label='Readout pulse amplitude',
                           unit='Hz', parameter_class=ManualParameter,
                           initial_value=0.048739)

        self.add_parameter('mock_spec_pow', label='optimal spec power',
                           unit='dBm', initial_value=-35,
                           parameter_class=ManualParameter)

        self.add_parameter('mock_12_spec_amp',
                           label='amplitude for 12 transition', unit='-',
                           initial_value=0.5, parameter_class=ManualParameter)

        self.add_parameter('mock_mw_amp180', label='Pi-pulse amplitude',
                           unit='V', initial_value=0.41235468,
                           parameter_class=ManualParameter)

        self.add_parameter('noise', label='nominal noise level', unit='V',
                           initial_value=0.16e-3,
                           parameter_class=ManualParameter)

        # Qubit characteristics
        self.add_parameter('mock_T1', label='relaxation time', unit='s',
                           initial_value=29e-6,
                           parameter_class=ManualParameter)

        self.add_parameter('mock_T2_star', label='Ramsey T2', unit='s',
                           initial_value=23.478921e-6,
                           parameter_class=ManualParameter)

        self.add_parameter('mock_T2_echo', label='Echo T2', unit='s',
                           initial_value=46.2892e-6,
                           parameter_class=ManualParameter)

    def find_spec_pow(self, freqs=None, powers=None, update=True):
        '''
        Should find the optimal spectroscopy power where the A/w ratio is
        at a maximum
        '''

        if freqs is None:
            freq_center = self.freq_qubit()
            freq_range = 200e6
            freqs = np.arange(freq_center-freq_range/2,
                              freq_center+freq_range/2, 0.5e6)

        if powers is None:
            powers = np.arange(-40, -9, 2)

        w = []
        A = []

        t_start = time.strftime('%Y%m%d_%H%M%S')

        for i, power in enumerate(powers):
            self.spec_pow(power)
            self.measure_spectroscopy(freqs=freqs, analyze=False)

            a = ma.Homodyne_Analysis(label=self.msmt_suffix,
                                     fitting_model='lorentzian',
                                     qb_name=self.name)
            w.append(a.params['kappa'].value)
            A.append(a.params['A'].value)

        t_stop = time.strftime('%Y%m%d_%H%M%S')
        Awratio = np.divide(A, w)

        ind = np.argmax(Awratio)
        # Should be some analysis and iterative method to
        best_spec_pow = powers[ind]
        # find the optimum
        print('from find_spec_pow:' + str(best_spec_pow))
        if update:
            import pycqed.analysis_v2.spectroscopy_analysis as sa

            a = sa.SpecPowAnalysis(t_start=t_start, t_stop=t_stop,
                                   label='spectroscopy_'+self.msmt_suffix,
                                   pow_key='Instrument settings.'+self.name +
                                           '.spec_pow.value')
            best_spec_pow = a.fit_res['spec_pow']

            self.spec_pow(best_spec_pow)
            print(self.spec_pow())
            return True

    ###########################################################################
    # Mock measurement methods
    ###########################################################################
    def measure_with_VNA(self, start_freq, stop_freq, npts,
                         MC=None, name='Mock_VNA', analyze=True):
        """
        Returns a SlopedHangerFuncAmplitude shaped function
        """
        VNA = self.instr_VNA.get_instr()

        VNA.start_frequency(start_freq)
        VNA.stop_frequency(stop_freq)
        VNA.npts(npts)

        if MC is None:
            MC = self.instr_MC.get_instr()

        s = swf.None_Sweep(name='Frequency', parameter_name='Frequency',
                           unit='Hz')
        freqs = np.linspace(VNA.start_frequency(), VNA.stop_frequency(),
                            VNA.npts())
        f0_res = self.calculate_mock_resonator_frequency()/1e9
        f0_test = self.mock_freq_test_res()/1e9

        Q_test = 346215
        Qe_test = 368762
        Q = 290000
        Qe = 255000
        A = 0.37
        theta = -0.1
        slope = 0
        f = freqs

        A_res = np.abs((slope * (f / 1.e9 - f0_res) / f0_res) *
                       fm.HangerFuncAmplitude(f, f0_res, Q, Qe, A, theta))
        A_test = np.abs((slope * (f / 1.e9 - f0_test) / f0_test) *
                        fm.HangerFuncAmplitude(f, f0_res, Q_test,
                                               Qe_test, A, theta))

        A_res = np.abs(A * (1. - Q / Qe * np.exp(1.j * theta) /
                            (1. + 2.j * Q * (f / 1.e9 - f0_res) / f0_res)))
        A_test = np.abs(A * (1. - Q_test / Qe_test * np.exp(1.j * theta) /
                             (1. + 2.j * Q_test * (f / 1.e9 - f0_test) / f0_test)))

        baseline = 0.40
        A_res -= A
        A_test -= A
        mocked_values = A + A_res + A_test

        mocked_values += np.random.normal(0, A/500, np.size(mocked_values))

        d = det.Mock_Detector(value_names=['ampl'], value_units=['V'],
                              detector_control='soft',
                              mock_values=mocked_values)
        MC.set_sweep_function(s)
        MC.set_sweep_points(freqs)
        MC.set_detector_function(d)

        MC.run(name + self.msmt_suffix)

        if analyze:
            ma2.Basic1DAnalysis()

    def measure_spectroscopy(self, freqs, mode='pulsed_marked', MC=None,
                             analyze=True, close_fig=True, label='',
                             prepare_for_continuous_wave=True):
        '''
        Can be made fancier by implementing different types of spectroscopy
        (e.g. pulsed/CW) and by imp using cfg_spec_mode.

        Uses a Lorentzian as a result for now
        '''
        if MC is None:
            MC = self.instr_MC.get_instr()

        s = swf.None_Sweep(name='Homodyne Frequency',
                           parameter_name='Frequency',
                           unit='Hz')

        h = self.measurement_signal(excited=False)  # Lorentian baseline [V]
        A0 = self.measurement_signal(excited=True) - h  # Peak height

        # Height of peak [V]
        K_power = 1/np.sqrt(1+15**(-(self.spec_pow()-self.mock_spec_pow())/7))
        # K_current = np.sqrt(np.abs(np.cos(2*np.pi*total_flux)))
        A = K_power*A0  # K_current*

        # Width of peak
        wbase = 4e6
        w = (wbase /
             np.sqrt(0.1+10**(-(self.spec_pow()-self.mock_spec_pow()/2)/7)) +
             wbase)

        f0 = self.calculate_mock_qubit_frequency()

        peak_01 = A*(w/2.0)**2 / ((w/2.0)**2 + ((freqs - f0))**2)
        # 1-2 transition:
        if self.spec_amp() > self.mock_12_spec_amp() and self.spec_pow() >= -10:
            A12 = A*0.5
            w12 = 1e6
            f02over2 = f0 - self.mock_anharmonicity()/2
            peak_02 = A12*(w12/2.0)**2 / ((w12/2.0)**2 + ((freqs-f02over2))**2)
        else:
            peak_02 = 0

        mocked_values = h + peak_01 + peak_02

        mocked_values += np.random.normal(0,
                                          self.noise()/np.sqrt(self.ro_acq_averages()), np.size(mocked_values))

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft',
                              mock_values=mocked_values)
        MC.set_sweep_function(s)
        MC.set_sweep_points(freqs)

        MC.set_detector_function(d)
        MC.run('mock_spectroscopy_'+self.msmt_suffix+label)

        if analyze:
            a = ma.Homodyne_Analysis(
                label=self.msmt_suffix, close_fig=close_fig)
            return a.params['f0'].value

    def measure_resonator_power(self, freqs, powers, MC=None,
                                analyze: bool = True, close_fig: bool = True,
                                fluxChan=None, label: str = ''):
        if MC is None:
            MC = self.instr_MC.get_instr()

        s1 = swf.None_Sweep(name='Heterodyne Frequency',
                            parameter_name='Frequency',
                            unit='Hz')
        s2 = swf.None_Sweep(name='Readout Power', parameter_name='Power',
                            unit='dBm')

        mocked_values = []
        try:
            device = self.instr_device.get_instr()
            qubit_names = device.qubits()
            # Create linecuts:
            for power in powers:
                h = 10**(power/20)*10e-3
                new_values = h
                for name in qubit_names:
                    if name != 'fakequbit':
                        qubit = device.find_instrument(name)
                        f0, response = qubit.calculate_mock_resonator_response(
                            power, freqs)
                        new_values += response

                mocked_values = np.concatenate([mocked_values, new_values])
        except AttributeError:
            logging.warning('No device found! Using this mock only for for '
                            'resonator frequencies')
            for power in powers:
                h = 10**(power/20)*10e-3
                new_values = h + self.calculate_mock_resonator_response(power,
                                                                        freqs)
                mocked_values = np.concatenate([mocked_values, new_values])

        mocked_values += np.random.normal(0,
                                          self.noise()/np.sqrt(self.ro_acq_averages()), np.size(mocked_values))
        mocked_values = np.abs(mocked_values)
        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft',
                              mock_values=mocked_values)

        MC.set_sweep_function(s1)
        MC.set_sweep_function_2D(s2)

        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(powers)

        MC.set_detector_function(d)
        MC.run('Resonator_power_scan'+self.msmt_suffix + label, mode='2D')

        if analyze:
            # ma.TwoD_Analysis(label='Resonator_power_scan',
            #                  close_fig=close_fig, normalize=True)
            a = ma.Resonator_Powerscan_Analysis(label='Resonator_power_scan',
                                                close_figig=True)
            return a

    def measure_heterodyne_spectroscopy(self, freqs, MC=None, analyze=True,
                                        close_fig=True, label=''):
        '''
        For finding resonator frequencies. Uses a lorentzian fit for now, might
        be extended in the future

        Dependent on readout power: too high power will yield wrong frequency
        '''
        if MC is None:
            MC = self.instr_MC.get_instr()

        s = swf.None_Sweep(name='Heterodyne Frequency',
                           parameter_name='Frequency',
                           unit='Hz')

        # Mock Values, matches resonator power scan:
        # Find all resonator frequencies:
        power = 20*np.log10(self.ro_pulse_amp_CW())
        dips = []
        try:
            device = self.instr_device.get_instr()
            qubit_names = device.qubits()
            for name in qubit_names:
                if name != 'fakequbit':
                    qubit = device.find_instrument(name)
                    freq, dip = qubit.calculate_mock_resonator_response(power,
                                                                        freqs)
                    dips.append(dip)
        except AttributeError:
            logging.warning('No device found! Using this mock only for for '
                            'resonator frequencies')
            freq, dips = self.calculate_mock_resonator_response(power,
                                                                freqs)

        h = 10**(power/20)*10e-3

        mocked_values = h
        for dip in dips:
            mocked_values += dip
        mocked_values += np.random.normal(0,
                                          self.noise()/np.sqrt(self.ro_acq_averages()),
                                          np.size(mocked_values))

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft',
                              mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(freqs)
        MC.set_detector_function(d)
        MC.run('Resonator_scan'+self.msmt_suffix+label)

        if analyze:
            ma.Homodyne_Analysis(label='Resonator_scan', close_fig=close_fig)

    def measure_qubit_frequency_dac_scan(self, freqs, dac_values,
                                         MC=None, analyze=True, fluxChan=None,
                                         close_fig=True, mode='pulsed_marked',
                                         nested_resonator_calibration=False,
                                         resonator_freqs=None):

        if MC is None:
            MC = self.instr_MC.get_instr()

        # Assume flux is controlled by SPI rack
        fluxcurrent = self.instr_FluxCtrl.get_instr()
        if fluxChan is None:
            fluxChan = self.fl_dc_ch()

        s1 = swf.None_Sweep(name='Frequency', parameter_name='Frequency',
                            unit='Hz')
        s2 = swf.None_Sweep(name='Flux', parameter_name=fluxChan, unit='A')

        mocked_values = []
        for dac_value in dac_values:
            fluxcurrent[fluxChan](dac_value)

            total_flux = self.calculate_mock_flux()

            h = self.measurement_signal(excited=False)
            A0 = self.measurement_signal(excited=True) - h

            K_current = np.sqrt(np.abs(np.cos(2*np.pi*total_flux)))
            K_power = 1 / \
                np.sqrt(1+15**(-(self.spec_pow()-self.mock_spec_pow())/7))
            A = K_current*K_power*A0   # Height of peak [V]

            wbase = 4e6
            w = wbase/np.sqrt(0.1+10**(-(self.spec_pow() -
                                         self.mock_spec_pow()/2)/7)) + wbase

            f0 = self.calculate_mock_qubit_frequency()

            new_values = h + A*(w/2.0)**2 / ((w/2.0)**2 +
                                             ((freqs - f0))**2)

            new_values += np.random.normal(0, self.noise()/np.sqrt(self.ro_acq_averages()),
                                           np.size(new_values))

            mocked_values = np.concatenate([mocked_values, new_values])

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft',
                              mock_values=mocked_values)

        MC.set_sweep_function(s1)
        MC.set_sweep_function_2D(s2)

        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(dac_values)

        MC.set_detector_function(d)
        t_start = time.strftime('%Y%m%d_%H%M%S')
        MC.run(name='Qubit_dac_scan'+self.msmt_suffix, mode='2D')

        if analyze:
            ma.TwoD_Analysis(label='Qubit_dac_scan', close_fig=close_fig)
            timestamp = a_tools.get_timestamps_in_range(t_start,
                                                        label='Qubit_dac_scan' +
                                                              self.msmt_suffix)
            timestamp = timestamp[0]
            ma2.da.DAC_analysis(timestamp=timestamp)

    def measure_resonator_frequency_dac_scan(self, freqs, dac_values, fluxChan,
                                             pulsed=True, MC=None,
                                             analyze=True, close_fig=True,
                                             nested_resonator_calibration=False,
                                             resonator_freqs=None, label=''):
        '''
        Measures resonator frequency versus flux. Simple model which just
        shifts the Lorentzian peak with a simple cosine by 2 MHz.
        '''

        if MC is None:
            MC = self.instr_MC.get_instr()
        s1 = swf.None_Sweep(name='Heterodyne Frequency',
                            parameter_name='Frequency',
                            unit='Hz')
        s2 = swf.None_Sweep(name='Flux', parameter_name=fluxChan,
                            unit='A')

        fluxcurrent = self.instr_FluxCtrl.get_instr()
        power = 20*np.log10(self.ro_pulse_amp_CW())

        mocked_values = []

        try:
            device = self.instr_device.get_instr()
            qubit_names = device.qubits()
            # Create linecuts:
            for dac_value in dac_values:
                fluxcurrent[fluxChan](dac_value)
                h = 10**(power/20)*10e-3
                new_values = h
                for name in qubit_names:
                    if name != 'fakequbit':
                        qubit = device.find_instrument(name)
                        f0, response = qubit.calculate_mock_resonator_response(
                            power, freqs)
                        new_values += response

                mocked_values = np.concatenate([mocked_values, new_values])

        except AttributeError:
            logging.warning('No device found! Using this mock only for for '
                            'resonator frequencies')
            for dac_value in dac_values:
                fluxcurrent[fluxChan](dac_value)
                h = 10**(power/20)*10e-3
                new_values = h + self.calculate_mock_resonator_response(power,
                                                                        freqs)
                mocked_values = np.concatenate([mocked_values, new_values])

        mocked_values += np.random.normal(0,
                                          self.noise()/np.sqrt(self.ro_acq_averages()),
                                          np.size(mocked_values))
        mocked_values = np.abs(mocked_values)

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft',
                              mock_values=mocked_values)

        MC.set_sweep_function(s1)
        MC.set_sweep_function_2D(s2)

        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(dac_values)

        MC.set_detector_function(d)
        MC.run('Resonator_dac_scan'+self.msmt_suffix+label, mode='2D')

        if analyze:
            ma.TwoD_Analysis(label='Resonator_dac_scan', close_fig=close_fig,
                             normalize=False)
            import pycqed.analysis_v2.dac_scan_analysis as dsa
            dsa.Susceptibility_to_Flux_Bias(label='Resonator_dac_scan')

    def measure_rabi(self, MC=None, amps=None,
                     analyze=True, close_fig=True, real_imag=True,
                     prepare_for_timedomain=True, all_modules=False):
        """
        Measurement is the same with and without vsm; therefore there is only
        one measurement method rather than two. In
        calibrate_mw_pulse_amp_coarse, the required parameter is updated.
        """
        if MC is None:
            MC = self.instr_MC.get_instr()
        if amps is None:
            amps = np.linspace(0.1, 1, 31)

        s = swf.None_Sweep(name='Channel Amplitude',
                           parameter_name='mw channel amp',
                           unit='a.u.')
        f_rabi = 1/self.mock_mw_amp180()

        low_lvl = self.measurement_signal(excited=False)
        high_lvl = self.measurement_signal(excited=True)

        freq_qubit = self.calculate_mock_qubit_frequency()

        detuning = np.abs(self.freq_qubit() - freq_qubit)/1e6
        highlow = (high_lvl-low_lvl)*np.exp(-detuning)

        high_lvl = low_lvl + highlow

        signal_amp = (high_lvl - low_lvl)/2
        offset = (high_lvl + low_lvl)/2
        mocked_values = offset + signal_amp*np.cos(np.pi*f_rabi*amps)

        mocked_values = self.values_to_IQ(mocked_values)  # This adds noise too

        d = det.Mock_Detector(value_names=['raw w0', 'raw w1'],
                              value_units=['V', 'V'], detector_control='soft',
                              mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(amps)
        MC.set_detector_function(d)
        MC.run('mock_rabi'+self.msmt_suffix)
        a = ma.Rabi_Analysis(label='rabi_', close_fig=close_fig)

        return a.rabi_amplitudes['piPulse']

    def measure_ramsey(self, times=None, MC=None,
                       artificial_detuning: float = None,
                       freq_qubit: float = None, label: str = '',
                       prepare_for_timedomain=True, analyze=True,
                       close_fig=True, update=True, detector=False,
                       double_fit=False):
        if MC is None:
            MC = self.instr_MC.get_instr()

        if times is None:
            stepsize = (self.T2_star()*4/61)//(abs(self.cfg_cycle_time())) \
                * abs(self.cfg_cycle_time())
            times = np.arange(0, self.T2_star()*4, stepsize)

        if artificial_detuning is None:
            artificial_detuning = 3/times[-1]

        # Calibration points:
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1]+1*dt,
                                 times[-1]+2*dt,
                                 times[-1]+3*dt,
                                 times[-1]+4*dt)])
        # if prepare_for_timedomain:
        #     self.prepare_for_timedomain()

        if freq_qubit is None:
            freq_qubit = self.freq_qubit()

        self.instr_LO_mw.get_instr().set('frequency', freq_qubit -
                                         self.mw_freq_mod.get() +
                                         artificial_detuning)

        s = swf.None_Sweep(name='T2_star', parameter_name='Time',
                           unit='s')

        low_lvl = self.measurement_signal(excited=False)
        high_lvl = self.measurement_signal(excited=True)

        mock_freq_qubit = self.calculate_mock_qubit_frequency()

        detuning = np.abs(self.freq_qubit() - freq_qubit)/1e6
        highlow = (high_lvl-low_lvl)*np.exp(-detuning)
        high_lvl = low_lvl + highlow

        signal_amp = (high_lvl - low_lvl)/2
        offset = (high_lvl + low_lvl)/2

        phase = 0
        oscillation_offset = 0
        exponential_offset = offset
        frequency = freq_qubit - mock_freq_qubit + artificial_detuning

        # Mock values without calibration points
        mocked_values = (signal_amp *
                         np.exp(-(times[0:-4] / self.mock_T2_star())) *
                         (np.cos(2*np.pi*frequency*times[0:-4] + phase) +
                          oscillation_offset) + exponential_offset)

        # Calibration points
        mocked_values = np.concatenate([mocked_values,
                                        low_lvl, low_lvl, high_lvl, high_lvl])

        # Add noise:
        mocked_values += np.random.normal(0,
                                          self.noise()/np.sqrt(self.ro_acq_averages()),
                                          np.size(mocked_values))

        mocked_values = self.values_to_IQ(mocked_values)

        d = det.Mock_Detector(value_names=['raw w1', 'raw w0'],
                              value_units=['V', 'V'],
                              detector_control='soft',
                              mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('mock_Ramsey' + self.msmt_suffix)

        if analyze:
            a = ma.Ramsey_Analysis(auto=True, closefig=True,
                                   freq_qubit=freq_qubit,
                                   artificial_detuning=artificial_detuning)
            if update:
                self.T2_star(a.T2_star['T2_star'])
            if double_fit:
                b = ma.DoubleFrequency()
                res = {
                    'T2star1': b.tau1,
                    'T2star2': b.tau2,
                    'frequency1': b.f1,
                    'frequency2': b.f2
                }
                return res

            else:
                res = {
                    'T2star': a.T2_star['T2_star'],
                    'frequency': a.qubit_frequency,
                }
                return res

    def measure_echo(self, times=None, MC=None, analyze=True, close_fig=True,
                     update=True, label: str = ''):

        if MC is None:
            MC = self.instr_MC.get_instr()

        if times is None:
            stepsize = (self.T2_echo()*2/61)//(abs(self.cfg_cycle_time())) \
                * abs(self.cfg_cycle_time())
            times = np.arange(0, self.T2_echo()*4, stepsize*2)

        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1]+1*dt,
                                 times[-1]+2*dt,
                                 times[-1]+3*dt,
                                 times[-1]+4*dt)])

        s = swf.None_Sweep(parameter_name='Time', unit='s')

        low_lvl = self.measurement_signal(excited=False)
        high_lvl = self.measurement_signal(excited=True)

        freq_qubit = self.calculate_mock_qubit_frequency()

        detuning = np.abs(self.freq_qubit() - freq_qubit)/1e6
        highlow = (high_lvl-low_lvl)*np.exp(-detuning)

        high_lvl = low_lvl + highlow

        signal_amp = (high_lvl - low_lvl)/2
        offset = (high_lvl + low_lvl)/2

        phase = np.pi
        oscillation_offset = 0
        exponential_offset = offset
        frequency = 4/times[-1]  # 4 oscillations

        # Mock values without calibration points
        mocked_values = (signal_amp *
                         np.exp(-(times[0:-4] / self.mock_T2_echo())) *
                         (np.cos(2*np.pi*frequency*times[0:-4] + phase) +
                          oscillation_offset) + exponential_offset)

        mocked_values = self.values_to_IQ(mocked_values)

        d = det.Mock_Detector(value_names=['raw w1', 'raw w0'],
                              value_units=['V', 'V'],
                              detector_control='soft',
                              mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('mock_echo' + self.msmt_suffix)

        if analyze:
            # N.B. v1.5 analysis
            a = ma.Echo_analysis_V15(label='echo', auto=True, close_fig=True)
            if update:
                self.T2_echo(a.fit_res.params['tau'].value)
            return a.fit_res.params['tau'].value

    def measure_T1(self, times=None, MC=None, analyze=True, close_fig=True,
                   update=True, prepare_for_timedomain=True):
        '''
        Very simple version that just returns a exponential decay based on
        mock_T1. Might be improved by making it depend on how close your
        pulse amp is.
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

        s = swf.None_Sweep(parameter_name='Time', unit='s')

        low_lvl = self.measurement_signal(excited=False)
        high_lvl = self.measurement_signal(excited=True)

        freq_qubit = self.calculate_mock_qubit_frequency()

        detuning = np.abs(self.freq_qubit() - freq_qubit)/1e6
        highlow = (high_lvl-low_lvl)*np.exp(-detuning)

        high_lvl = low_lvl + highlow

        amplitude = high_lvl - low_lvl

        mocked_values = amplitude*np.exp(-(times[0:-4]/self.mock_T1()))+low_lvl
        mocked_values = np.concatenate(
            [mocked_values, low_lvl, low_lvl, high_lvl, high_lvl])

        mocked_values = self.values_to_IQ(mocked_values)

        d = det.Mock_Detector(value_names=['raw w0', 'raw w1'],
                              value_units=['V', 'V'], detector_control='soft',
                              mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('mock_T1'+self.msmt_suffix)

        if analyze:
            a = ma.T1_Analysis(auto=True, close_fig=True)
            if update:
                self.T1(a.T1)
            return a.T1

    def measure_ALLXY(self, MC=None, label: str = '', analyze=True,
                      close_fig=True):
        """
        NOT IMPLEMENTED YET
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        if analyze:
            a = ma.ALLXY_Analysis(close_main_fig=close_fig)
            return a.deviation_total

    def measurement_signal(self, excited=False):
        '''
        Sets the readout signal level, depending on the readout frequency and
        resonator frequency.
        The 'excited' parameter indicates whether the qubit is in the excited
        state, which results in a 2 chi shift of the resonator
        '''
        power = 20*np.log10(self.ro_pulse_amp_CW())
        f_ro = self.ro_freq()
        h = 10**(power/20)*10e-3  # Lorentian baseline [V]

        f0, dip = self.calculate_mock_resonator_response(power, np.array([f_ro]),
                                                         excited=excited)
        signal = h + dip

        if type(signal) is list:
            signal = signal[0]

        return signal

    def values_to_IQ(self, mocked_values, theta=15):
        theta = theta * np.pi/180
        MockI = 1/np.sqrt(2)*np.real(mocked_values*np.exp(1j*theta))
        MockQ = 1/np.sqrt(2)*np.real(mocked_values*np.exp(1j*(theta-np.pi/2)))

        IQ_values = []
        for I, Q in zip(MockI, MockQ):
            I += np.random.normal(0, self.noise() /
                                  np.sqrt(self.ro_acq_averages()), 1)
            Q += np.random.normal(0, self.noise() /
                                  np.sqrt(self.ro_acq_averages()), 1)
            IQ_values.append([I, Q])
        return IQ_values

    def calculate_f_qubit_from_power_scan(self, f_bare, f_shifted,
                                          g_coupling=65e6, RWA=False):
        '''
        Inputs are in Hz
        f_bare: the resonator frequency without a coupled qubit
        f_shifted: the reso freq shifted due to coupling of a qwubit
        g_coupling: the coupling strengs
        Output:
        f_q: in Hz
        '''
        w_r = f_bare * 2 * np.pi
        w_shift = f_shifted * 2*np.pi
        g = 2*np.pi * g_coupling
        shift = (w_shift - w_r)/g**2
        # f_shift > 0 when f_qubit<f_res
        # For the non-RWA result (only dispersive approximation)
        if (RWA is False):
            w_q = -1/(shift) + np.sqrt(1/(shift**2)+w_r**2)
        # For the RWA approximation
        else:
            w_q = -1/shift + w_r
        return w_q/(2.*np.pi)

    def calculate_g_coupling_from_frequency_shift(self, f_bare, f_shifted,
                                                  f_qubit):
        w_r = 2*np.pi * f_bare
        w_shift = 2*np.pi * f_shifted
        w_q = 2*np.pi*f_qubit
        shift = w_shift-w_r
        rhs = 1./(w_q-w_r) + 1./(w_q+w_r)
        # rhs_RWA = 1./(w_q-w_r)
        return np.sqrt(np.abs(shift/rhs))/(2*np.pi)

    def calculate_mock_flux(self):
        """
        Calculates total flux through SQUID loop by a weighted sum of all
        contributions from all FBLs, and subtracting the sweetspot flux.
        """

        fluxcurrent = self.instr_FluxCtrl.get_instr()
        flux = 0
        for FBL in fluxcurrent.channel_map:
            current = fluxcurrent[FBL]()

            flux += current/self.mock_fl_dc_I_per_phi0()[FBL]

        flux -= self.mock_sweetspot_phi_over_phi0()

        return flux

    def calculate_mock_qubit_frequency(self):
        '''
        Cleaner way of calculating the qubit frequency, depending on:
        - Flux (current)
        - Ec, EJ
        - Chi01
        '''
        phi_over_phi0 = self.calculate_mock_flux()
        if self.mock_Ej1() > self.mock_Ej2():
            alpha = self.mock_Ej1()/self.mock_Ej2()
        else:
            alpha = self.mock_Ej2()/self.mock_Ej1()
        d = (alpha-1)/(alpha+1)
        Ej_sum = self.mock_Ej1() + self.mock_Ej2()

        Ej_eff = np.abs(Ej_sum*np.cos(np.pi*phi_over_phi0) *
                        np.sqrt(1 + d**2 * (np.tan(np.pi*phi_over_phi0))**2))

        f_qubit = (np.sqrt(8*self.mock_Ec()*Ej_eff) -
                   self.mock_Ec() + self.mock_chi01())

        return f_qubit

    def calculate_mock_resonator_response(self, power, freqs, excited=False):
        """
        Cleaner way of calculating resonator frequency, depending on power etc.
        Makes it easier to use a mock device with multiple resonators
        Returns resonant frequency and Lorentzian as a combination of both the
        test and qubit resonators

        TODO: Make hanger instead of Lorentzian
        """
        res_power = 20*np.log10(self.mock_ro_pulse_amp_CW())
        pow_shift = self.mock_pow_shift()

        h = 10**(power/20)*10e-3   # Lorentzian baseline [V]

        Q = self.mock_Q()
        Qe = self.mock_Qe()
        theta = self.mock_theta()
        slope = self.mock_slope()
        phi_I = self.mock_phi_I()
        phi_0 = self.mock_phi_0()

        Q_test = self.mock_test_Q()
        Qe_test = self.mock_test_Qe()
        theta_test = self.mock_test_theta()
        slope_test = self.mock_test_slope()
        phi_I_test = self.mock_test_phi_I()
        phi_0_test = self.mock_test_phi_0()

        if power <= res_power:
            # Good signal
            f0 = self.calculate_mock_resonator_frequency(excited=excited)
            res_qubit_dip = fm.hanger_func_complex_SI(freqs, f0, Q, Qe, h,
                                                      theta, phi_I, phi_0,
                                                      slope=slope)
        elif (power > res_power) and (power < res_power + pow_shift):
            # Noisy regime -> width increases, peak decreases
            f0 = self.calculate_mock_resonator_frequency(excited=excited)
            f0_high = self.mock_freq_res_bare()

            f_shift = f0 - f0_high
            f0 = f0 - ((power - res_power)/pow_shift)*f_shift

            Q_decrease = (1+(power-res_power)/pow_shift*10)

            Q_nonlinear = Q  # /Q_decrease
            Qe_nonlinear = Qe  # /Q_decrease
            res_qubit_dip = fm.hanger_func_complex_SI(freqs, f0, Q_nonlinear,
                                                      Qe_nonlinear, h,
                                                      theta, phi_I, phi_0,
                                                      slope=slope) - h
            res_qubit_dip = res_qubit_dip/Q_decrease

            res_qubit_dip += h
            # Add some extra noise
            for i, value in enumerate(res_qubit_dip):
                d = np.abs(value - h)

                value += np.random.normal(0,
                                          self.noise() /
                                          self.ro_acq_averages()*10*Q_decrease,
                                          1)
                res_qubit_dip[i] = value
        else:
            # High power regime
            f0 = self.mock_freq_res_bare()
            res_qubit_dip = fm.hanger_func_complex_SI(freqs, f0, Q, Qe, h,
                                                      theta, phi_I, phi_0,
                                                      slope=slope)

        if self.mock_freq_test_res() is not None:
            f0_test_res = self.mock_freq_test_res()
            test_res_response = fm.hanger_func_complex_SI(freqs, f0_test_res,
                                                          Q_test, Qe_test, h,
                                                          theta_test,
                                                          phi_I_test,
                                                          phi_0_test,
                                                          slope=slope_test)
        else:
            test_res_response = np.ones(np.size(freqs))*h

        response = np.real(res_qubit_dip - h + test_res_response - h)
        return f0, response

    def calculate_mock_resonator_frequency(self, excited=False):
        """
        Calculates the resonator frequency depeding on flux etc.
        """
        freq_qubit_12 = self.calculate_mock_qubit_frequency() + self.mock_Ec()
        freq_qubit = self.calculate_mock_qubit_frequency()
        chi01 = self.mock_coupling01()**2 / (freq_qubit -
                                             self.mock_freq_res_bare())

        chi12 = self.mock_coupling12()**2 / (freq_qubit_12 -
                                             self.mock_freq_res_bare())
        if excited:
            chi = chi01 - chi12/2
        else:
            chi = 0

        f0 = self.mock_freq_res_bare() - chi12/2 - 2*chi

        return f0
    ###########################################################################
    # AutoDepGraph
    ###########################################################################

    def dep_graph(self):
        dag = AutoDepGraph_DAG('DAG')
        cal_True_delayed = 'autodepgraph.node_functions.calibration_functions.test_calibration_True_delayed'
        dag.add_node('Resonators Wide Search',
                     calibrate_function=self.name + '.find_resonators')
        dag.add_node('Zoom on resonators',
                     calibrate_function=self.name + '.find_resonator_frequency_initial')
        dag.add_node('Resonators Power Scan',
                     calibrate_function=self.name + '.find_test_resonators')
        dag.add_node('Resonators Flux Sweep',
                     calibrate_function=self.name + '.find_qubit_resonator_fluxline')

        dag.add_node(self.name + ' Resonator Frequency',
                     calibrate_function=self.name + '.find_resonator_frequency')
        dag.add_node(self.name + ' Resonator Power Scan',
                     calibrate_function=self.name + '.calibrate_ro_pulse_amp_CW')

        # Calibration of instruments and ro
        # dag.add_node(self.name + ' Calibrations',
        #                   calibrate_function=cal_True_delayed)
        # dag.add_node(self.name + ' Mixer Skewness',
        #                   calibrate_function=self.name + '.calibrate_mixer_skewness_drive')
        # dag.add_node(self.name + ' Mixer Offset Drive',
        #                   calibrate_function=self.name + '.calibrate_mixer_offsets_drive')
        # dag.add_node(self.name + ' Mixer Offset Readout',
        #                   calibrate_function=self.name + '.calibrate_mixer_offsets_RO')
        # dag.add_node(self.name + ' Ro/MW pulse timing',
        #                   calibrate_function=cal_True_delayed)
        # dag.add_node(self.name + ' Ro Pulse Amplitude',
        #                   calibrate_function=self.name + '.ro_pulse_amp_CW')

        # Qubits calibration
        dag.add_node(self.name + ' Frequency Coarse',
                     calibrate_function=self.name + '.find_frequency',
                     check_function=self.name + '.check_qubit_spectroscopy',
                     tolerance=0.2e-3)
        dag.add_node(self.name + ' Frequency at Sweetspot',
                     calibrate_function=self.name + '.find_frequency')
        dag.add_node(self.name + ' Spectroscopy Power',
                     calibrate_function=self.name + '.calibrate_spec_pow')
        dag.add_node(self.name + ' Sweetspot',
                     calibrate_function=self.name + '.find_qubit_sweetspot')
        dag.add_node(self.name + ' Rabi',
                     calibrate_function=self.name + '.calibrate_mw_pulse_amplitude_coarse',
                     check_function=self.name + '.check_rabi',
                     tolerance=0.01)
        dag.add_node(self.name + ' Frequency Fine',
                     calibrate_function=self.name + '.calibrate_frequency_ramsey',
                     check_function=self.name + '.check_ramsey',
                     tolerance=0.1e-3)

        # Validate qubit calibration
        dag.add_node(self.name + ' ALLXY',
                     calibrate_function=self.name + '.measure_allxy')
        dag.add_node(self.name + ' MOTZOI Calibration',
                     calibrate_function=self.name + '.calibrate_motzoi')
        # If all goes well, the qubit is fully 'calibrated' and can be controlled

        # Qubits measurements
        dag.add_node(self.name + ' Anharmonicity')
        dag.add_node(self.name + ' Avoided Crossing')
        dag.add_node(self.name + ' T1')
        dag.add_node(self.name + ' T1(time)')
        dag.add_node(self.name + ' T1(frequency)')
        dag.add_node(self.name + ' T2_Echo')
        dag.add_node(self.name + ' T2_Echo(time)')
        dag.add_node(self.name + ' T2_Echo(frequency)')
        dag.add_node(self.name + ' T2_Star')
        dag.add_node(self.name + ' T2_Star(time)')
        dag.add_node(self.name + ' T2_Star(frequency)')
        #######################################################################
        # EDGES
        #######################################################################

        # VNA
        dag.add_edge('Zoom on resonators', 'Resonators Wide Search')
        dag.add_edge('Resonators Power Scan',
                     'Zoom on resonators')
        dag.add_edge('Resonators Flux Sweep',
                     'Zoom on resonators')
        dag.add_edge('Resonators Flux Sweep',
                     'Resonators Power Scan')
        # Resonators
        dag.add_edge(self.name + ' Resonator Frequency',
                     'Resonators Power Scan')
        dag.add_edge(self.name + ' Resonator Frequency',
                     'Resonators Flux Sweep')
        dag.add_edge(self.name + ' Resonator Power Scan',
                     self.name + ' Resonator Frequency')
        dag.add_edge(self.name + ' Frequency Coarse',
                     self.name + ' Resonator Power Scan')
        # Qubit Calibrations
        dag.add_edge(self.name + ' Frequency Coarse',
                     self.name + ' Resonator Frequency')
        # dag.add_edge(self.name + ' Frequency Coarse',
        #                   self.name + ' Calibrations')

        # Calibrations
        # dag.add_edge(self.name + ' Calibrations',
        #                   self.name + ' Mixer Skewness')
        # dag.add_edge(self.name + ' Calibrations',
        #                   self.name + ' Mixer Offset Drive')
        # dag.add_edge(self.name + ' Calibrations',
        #                   self.name + ' Mixer Offset Readout')
        # dag.add_edge(self.name + ' Calibrations',
        #                   self.name + ' Ro/MW pulse timing')
        # dag.add_edge(self.name + ' Calibrations',
        #                   self.name + ' Ro Pulse Amplitude')
        # Qubit
        dag.add_edge(self.name + ' Spectroscopy Power',
                     self.name + ' Frequency Coarse')
        dag.add_edge(self.name + ' Sweetspot',
                     self.name + ' Frequency Coarse')
        dag.add_edge(self.name + ' Sweetspot',
                     self.name + ' Spectroscopy Power')
        dag.add_edge(self.name + ' Rabi',
                     self.name + ' Frequency at Sweetspot')
        dag.add_edge(self.name + ' Frequency Fine',
                     self.name + ' Frequency at Sweetspot')
        dag.add_edge(self.name + ' Frequency Fine',
                     self.name + ' Rabi')

        dag.add_edge(self.name + ' Frequency at Sweetspot',
                     self.name + ' Sweetspot')

        dag.add_edge(self.name + ' ALLXY',
                     self.name + ' Rabi')
        dag.add_edge(self.name + ' ALLXY',
                     self.name + ' Frequency Fine')
        dag.add_edge(self.name + ' ALLXY',
                     self.name + ' MOTZOI Calibration')

        # Perform initial measurements to see if they make sense
        dag.add_edge(self.name + ' T1',
                     self.name + ' ALLXY')
        dag.add_edge(self.name + ' T2_Echo',
                     self.name + ' ALLXY')
        dag.add_edge(self.name + ' T2_Star',
                     self.name + ' ALLXY')

        # Measure as function of frequency and time
        dag.add_edge(self.name + ' T1(frequency)',
                     self.name + ' T1')
        dag.add_edge(self.name + ' T1(time)',
                     self.name + ' T1')

        dag.add_edge(self.name + ' T2_Echo(frequency)',
                     self.name + ' T2_Echo')
        dag.add_edge(self.name + ' T2_Echo(time)',
                     self.name + ' T2_Echo')

        dag.add_edge(self.name + ' T2_Star(frequency)',
                     self.name + ' T2_Star')
        dag.add_edge(self.name + ' T2_Star(time)',
                     self.name + ' T2_Star')

        dag.add_edge(self.name + ' DAC Arc Polynomial',
                     self.name + ' Frequency at Sweetspot')

        # Measurements of anharmonicity and avoided crossing
        dag.add_edge(self.name + ' f_12 estimate',
                     self.name + ' Frequency at Sweetspot')
        dag.add_edge(self.name + ' Anharmonicity',
                     self.name + ' f_12 estimate')
        dag.add_edge(self.name + ' Avoided Crossing',
                     self.name + ' DAC Arc Polynomial')

        dag.cfg_plot_mode = 'svg'
        dag.update_monitor()
        dag.cfg_svg_filename

        url = dag.open_html_viewer()
        print('Dependancy Graph Created. URL = '+url)
        self._dag = dag
        return dag
        # Create folder
        # datadir = r'D:\GitHubRepos\PycQED_py3\pycqed\tests\test_output'
        # datestr = time.strftime('%Y%m%d')
        # timestr = time.strftime('%H%M%S')
        # folder_name = '_Dep_Graph_Maintenance'

        # try:
        #     os.mkdir(datadir + '\\' + datestr + '\\' + timestr + folder_name)
        # except FileExistsError:
        #    pass
    # def show_dep_graph(self, dag):
    #     dag.cfg_plot_mode = 'svg'
    #     dag.update_monitor()
    #     dag.cfg_svg_filename
    #     url = dag.open_html_viewer()
    #     print(url)
