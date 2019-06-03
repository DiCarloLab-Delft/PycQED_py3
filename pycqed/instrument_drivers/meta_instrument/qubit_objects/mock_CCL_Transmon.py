from .CCL_Transmon import CCLight_Transmon
import time
import numpy as np
import logging
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2
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
                           initial_value=np.sqrt(8*self.mock_Ec()*self.mock_Ej()) - self.mock_Ec())

        self.add_parameter('mock_freq_res', label='resonator frequency', unit='Hz',
                           parameter_class=ManualParameter, initial_value=7.487628e9)

        self.add_parameter('mock_ro_pulse_amp_CW', label='Readout pulse amplitude',
                           unit='Hz', parameter_class=ManualParameter,
                           initial_value=0.048739)

        self.add_parameter('mock_residual_flux_current', label='magnitude of sweetspot current',
                           unit='A', parameter_class=ManualParameter,
                           initial_value=0.25e-3)

        self.add_parameter('mock_mw_amp180', label='Pi-pulse amplitude', unit='V',
                           initial_value=0.5, parameter_class=ManualParameter)

        self.add_parameter('mock_T1', label='relaxation time', unit='s',
                           initial_value=29e-6, parameter_class=ManualParameter)

        self.add_parameter('mock_T2_star', label='Ramsey T2', unit='s',
                           initial_value=1e-6, parameter_class=ManualParameter)

        self.add_parameter('mock_anharmonicity', label='anharmonicity', unit='Hz',
                           initial_value=274e6, parameter_class=ManualParameter)

        self.add_parameter('mock_spec_pow', label='optimal spec power', unit='dBm',
                           initial_value=-35, parameter_class=ManualParameter)

        self.add_parameter('mock_12_spec_amp', label='amplitude for 12 transition',
                           unit='-', initial_value=0.5, parameter_class=ManualParameter)

        self.add_parameter('mock_current', label='current through FBL', unit='A',
                           initial_value=0, parameter_class=ManualParameter)

        self.add_parameter('mock_baseline', label='resonator signal',
                           unit='V', parameter_class=ManualParameter)

        self.add_parameter('mock_spec_maximum', label='resonator baseline',
                           unit='V', parameter_class=ManualParameter)

        self.add_parameter('mock_pow_shift', label='power needed for low to high regime',
                           unit='dBm', initial_value=20, parameter_class=ManualParameter)

        self.add_parameter('mock_coupling01', label='coupling qubit to resonator',
                           unit='Hz', initial_value=60e6, parameter_class=ManualParameter)

        self.add_parameter('mock_coupling12', label='coupling 12 transition to resonator',
                           unit='Hz', initial_value=5e6, parameter_class=ManualParameter)

        self.add_parameter('mock_chi01', label='coupling 01 transition', unit='Hz',
                           initial_value=(self.mock_coupling01(
                           ))**2/(self.mock_freq_qubit() - self.mock_freq_res()),
                           parameter_class=ManualParameter)

        self.add_parameter('mock_chi12', label='coupling 12 transition', unit='Hz',
                           initial_value=(self.mock_coupling12(
                           ))**2/(self.mock_freq_qubit()+self.mock_anharmonicity() - self.mock_freq_res()),
                           parameter_class=ManualParameter)

        self.add_parameter('mock_chi', label='dispersive shift', units='Hz',
                           initial_value=self.mock_chi01()-self.mock_chi12()/2,
                           parameter_class=ManualParameter)

        self.add_parameter('noise', label='noise level', unit='V',
                           initial_value=0.01e-3, parameter_class=ManualParameter)

        self.add_parameter('mock_res_width', label='resonator peak width', unit='Hz',
                           initial_value=1e6, parameter_class=ManualParameter)

    def find_resonator_power(self, freqs=None, powers=None, update=True):
        if freqs is None:
            freq_center = self.freq_res()
            freq_range = 10e6
            freqs = np.arange(freq_center-1/2*freq_range, freq_center+1/2*freq_range,
                              0.1e6)

        if powers is None:
            powers = np.arange(-40, 0.1, 4)

        self.measure_resonator_power(freqs=freqs, powers=powers, analyze=False)

        if update:
            fit_res = ma.Resonator_Powerscan_Analysis(label='Resonator_power_scan',
                                                      close_figig=True)
            shift = fit_res.results[0]
            power = fit_res.results[1]

            ro_pow = 10**(power/20)
            self.ro_pulse_amp_CW(ro_pow)

            f_qubit_estimate = self.freq_res() + (60e6)**2/shift
            self.freq_qubit(f_qubit_estimate)

        return True

    def find_resonator_frequency_VNA(self, freqs=None, use_min=False, MC=None,
                                     update=True):
        '''
        quick script that uses measure_heterodyne_spectroscopy on a wide range
        to act as a sort of mock of a VNA resonator scan'.

        If it is a first scan (no freq_res yet in qubit object) it will perform
        a wide range scan. Otherwise it will zoom in on a resonator
        '''

        if freqs is None:
            if self.freq_res() is None:
                freqs = np.arange(7e9, 8.5e9, 2e6)
            else:
                freq_center = self.freq_res()
                freq_span = 100e6
                freqs = np.arange(freq_center-freq_span/2,
                                  freq_center+freq_span/2, 1e6)

        self.measure_heterodyne_spectroscopy(freqs=freqs, MC=MC, analyze=True)
        a = ma.Homodyne_Analysis(label=self.msmt_suffix)

        if use_min:
            f_res = a.min_frequency
        else:
            f_res = a.fit_results.params['f0'].value*1e9

        if f_res > max(freqs) or f_res < max(freqs):
            logging.warning(
                'extracted frequency outside range of scan. Not updating')
        if update:
            print('Updating ' + self.msmt_suffix + '.freq_res')
            self.freq_res(f_res)
            self.ro_freq(f_res)
        return f_res

    def find_resonator_sweetspot(self, freqs=None, dac_values=None, fluxChan=None,
                                 update=True):
        '''
        Finds the resonator sweetspot current.
        TODO: - measure all FBL-resonator combinations
        TODO: - implement way of distinguishing which fluxline is most coupled
        TODO: - create method that moves qubits away from sweetspot when they are
                not being measured (should not move them to some other qubit 
                frequency of course)
        '''
        if freqs is None:
            freq_center = self.freq_res()
            freq_range = 20e6
            freqs = np.arange(freq_center-1/2*freq_range, freq_center+1/2*freq_range,
                              0.5e6)

        if dac_values is None:
            dac_values = np.linspace(-10e-3, 10e-3, 101)

        if fluxChan is None:
            fluxChan = 'FBL_1'

        self.measure_resonator_frequency_dac_scan(freqs=freqs, dac_values=dac_values,
                                                  fluxChan=fluxChan, analyze=False)
        if update:
            import pycqed.analysis_v2.dac_scan_analysis as dsa
            fit_res = dsa.Susceptibility_to_Flux_Bias(label='Resonator_dac_scan')

        return True

    def find_qubit_sweetspot(self, freqs=None, dac_values=None, update=True, 
                             set_to_sweetspot=True):
        '''
        Should be edited such that it contains reference to different measurement
        methods (tracking / 2D scan / broad spectroscopy)

        TODO: If spectroscopy does not yield a peak, it should discard it
        '''
        if freqs is None:
            freq_center = self.freq_qubit()
            freq_range = 200e6
            freqs = np.arange(freq_center-1/2*freq_range, freq_center+1/2*freq_range,
                              0.5e6)

        if dac_values is None:
            dac_values = np.linspace(-1e-3, 1e-3, 6)

        t_start = time.time()
        t_start = time.strftime('%Y%m%d_%H%M%S')

        for dac_value in dac_values:
            self.mock_current(dac_value)
            self.find_frequency(freqs=freqs)

        t_end = time.time()
        t_end = time.strftime('%Y%m%d_%H%M%S')

        a = ma2.DACarcPolyFit(t_start=t_start, t_stop=t_end,
                              label='spectroscopy__' + self.name,
                              dac_key='Instrument settings.'+self.name+'.mock_current',
                              degree=2)

        pc = a.fit_res['fit_polycoeffs']

        self.flux_polycoeff(pc)
        sweetspot_current = -pc[1]/(2*pc[0])

        if update:
            self.fl_dc_V0(sweetspot_current)
        if set_to_sweetspot:
            self.mock_current(sweetspot_current)
        
        return True

    def find_anharmonicity_estimate(self, freqs=None, anharmonicity=None,
                                    update=True):
        '''
        Finds an estimate of the anharmonicity by doing a spectroscopy around 
        150 MHz below the qubit frequency.

        TODO: if spec_pow is too low/high, it should adjust it to approx the 
              ideal spec_pow + 25 dBm
        '''

        if anharmonicity is None:
            # Standard estimate, negative by convention
            anharmonicity = self.anharmonicity()

        f12_estimate = self.freq_qubit()*2 + anharmonicity

        if freqs is None:
            freq_center = f12_estimate/2
            freq_range = 100e6
            freqs = np.arange(freq_center-1/2*freq_range, freq_center+1/2*freq_range,
                              0.5e6)

        self.spec_pow(self.spec_pow()+25)
        self.measure_spectroscopy(freqs=freqs, pulsed=False, analyze=False)

        a = ma.Homodyne_Analysis(label=self.msmt_suffix)
        f02 = 2*a.params['f0'].value*1e9
        if update:
            self.anharmonicity(f02-2*self.freq_qubit())
            return True

    def find_spec_pow(self, freqs=None, powers=None, update=True):
        '''
        Should find the optimal spectroscopy power where the A/w ratio is 
        at a maximum
        '''

        if freqs is None:
            freq_center = self.freq_qubit()
            freq_range = 100e6
            freqs = np.arange(freq_center-1/2*freq_range, freq_center+1/2*freq_range,
                              0.5e6)
        if powers is None:
            powers = [-30, -20, -10]

        w = np.zeros(np.size(powers))
        A = np.zeros(np.size(powers))

        for i, power in enumerate(powers):
            self.spec_pow(power)
            self.measure_spectroscopy(freqs=freqs, analyze=False)

            a = ma.Homodyne_Analysis(label=self.msmt_suffix,
                                     fitting_model='lorentzian')
            w[i] = a.params['kappa'].value
            A[i] = a.params['A'].value

        Awratio = np.divide(A, w)

        ind = Awratio.index(max(Awratio))
        best_spec_pow = powers[ind]  # Should be some analysis and iterative method to
        # find the optimum

        if update:
            self.spec_pow(best_spec_pow)
            return True

    def measure_spectroscopy(self, freqs, pulsed=True, MC=None, analyze=True,
                             close_fig=True, label='',
                             prepare_for_continuous_wave=True):
        '''
        Can be made fancier by implementing different types of spectroscopy 
        (e.g. pulsed/CW) and by imp using cfg_spec_mode.

        Uses a Lorentzian as a result for now

        TODO: make height of peak depend on resonator shift
        '''
        if MC is None:
            MC = self.instr_MC.get_instr()

        s = swf.None_Sweep(name='Homodyne Frequency', parameter_name='Frequency',
                           unit='Hz')
        h = self.measurement_signal(excited=False)  # Lorentian baseline [V]
        A0 = self.measurement_signal(excited=True) - h  # Peak height

        current = self.mock_current()
        Iref = self.mock_residual_flux_current()
        I0 = 20e-3

        # Height of peak [V]
        K = 1/np.sqrt(1+15**(-(self.spec_pow()-self.mock_spec_pow())/7))
        current_correction = np.sqrt(np.abs(np.cos(2*np.pi*(current-Iref)/I0)))
        A = K*current_correction*A0

        # Width of peak
        wbase = 4e6
        w = wbase/np.sqrt(0.1+10**(-(self.spec_pow()-self.mock_spec_pow()/2)/7))+wbase

        # f0 = self.mock_freq_qubit() - df*(np.sin(1/2*np.pi*(current-Iref)/I0))**2
        f0 = np.sqrt(8*self.mock_Ec()*self.mock_Ej() *
            np.abs(np.cos(np.pi*(current-Iref)/I0))) - self.mock_Ec()
        if self.spec_amp() > self.mock_12_spec_amp():  # 1-2 transition
            A12 = A*0.5
            w12 = 1e6
            f02over2 = self.mock_freq_qubit()-self.mock_anharmonicity()/2
            mocked_values = h + A*(w/2.0)**2 / ((w/2.0)**2 + ((freqs - f0))**2) + \
                A12*(w12/2.0)**2 / ((w12/2.0)**2 + ((freqs - f02over2))**2)
        else:
            mocked_values = h + A*(w/2.0)**2 / ((w/2.0)**2 +
                                                ((freqs - f0))**2)

        mocked_values += np.random.normal(0,
                                          self.noise(), np.size(mocked_values))

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft', mock_values=mocked_values)
        MC.set_sweep_function(s)
        MC.set_sweep_points(freqs)

        MC.set_detector_function(d)
        MC.run('mock_spectroscopy_'+self.msmt_suffix)

        if analyze:
            a = ma.Homodyne_Analysis(
                label=self.msmt_suffix, close_fig=close_fig)
            return a.params['f0'].value

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

        for power in powers:
            h = 10**(power/20)*45e-3  # Lorentian baseline [V]
            A = 0.9*h   # Height of peak [V]
            w = self.mock_res_width()
            if power <= res_power:
                # Good signal
                f0 = self.mock_freq_res()
                new_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                                 ((freqs - f0))**2)
            elif (power > res_power) and (power < res_power + pow_shift):
                # no signal at all -> width increases, peak decreases
                A0 = A
                n = 6
                b = A0/10
                a = (A0-b)/(-1/2*pow_shift)**n
                A = a*(power - (res_power+1/2*pow_shift))**n + b

                w = 0.5e6
                w = 0.5e6 + 0.5e6*np.sin(np.pi*(power - res_power)/pow_shift)

                b = res_power + self.mock_freq_res()*(pow_shift/self.mock_chi())
                f0 = (power-b)/pow_shift*self.mock_chi()

                new_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                                 ((freqs - f0))**2)
                # Add some extra noise
                for i, value in enumerate(new_values):
                    d = np.abs(value - A0)

                    value += np.random.normal(0, d/5, 1)
                    new_values[i] = value
            else:
                # High power regime
                f0 = self.mock_freq_res() + self.mock_chi()
                new_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                                 ((freqs - f0))**2)
            mocked_values = np.concatenate([mocked_values, new_values])

        mocked_values += np.random.normal(0,
                                          self.noise(), np.size(mocked_values))
        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft', mock_values=mocked_values)

        MC.set_sweep_function(s1)
        MC.set_sweep_function_2D(s2)

        MC.set_sweep_points(freqs)
        MC.set_sweep_points_2D(powers)

        MC.set_detector_function(d)
        MC.run('Resonator_power_scan'+self.msmt_suffix, mode='2D')

        if analyze:
            # ma.TwoD_Analysis(label='Resonator_power_scan',
            #                  close_fig=close_fig, normalize=True)
            a = ma.Resonator_Powerscan_Analysis(label='Resonator_power_scan',
                                                close_figig=True)
            print(a)
            return a

    def measure_heterodyne_spectroscopy(self, freqs, MC=None, analyze=True,
                                        close_fig=True):
        '''
        For finding resonator frequencies. Uses a lorentzian fit for now, might
        be extended in the future

        Dependent on readout power: too high power will yield wrong frequency
        '''
        if MC is None:
            MC = self.instr_MC.get_instr()

        s = swf.None_Sweep(name='Heterodyne Frequency', parameter_name='Frequency',
                           unit='Hz')

        # Mock Values, matches resonator power scan:
        res_power = 20*np.log10(self.mock_ro_pulse_amp_CW())
        pow_shift = self.mock_pow_shift()  # At which power the high power regime occurs

        power = 20*np.log10(self.ro_pulse_amp_CW())
        h = 10**(power/20)*45e-3  # Lorentian baseline [V]
        A = 0.9*h   # Height of peak [V]
        w = self.mock_res_width()

        if power <= res_power:
            # Good signal
            f0 = self.mock_freq_res()
            mocked_values = h - A*(w/2.0)**2 / ((w/2.0)**2 + ((freqs - f0))**2)

        elif (power > res_power) and (power < res_power + pow_shift):
            # no signal at all -> width increases, peak decreases
            A0 = A
            n = 6
            b = A0/10
            a = (A0-b)/(-1/2*pow_shift)**n
            A = a*(power - (res_power+1/2*pow_shift))**n + b

            w = 0.5e6 + 0.5e6*np.sin(np.pi*(power - res_power)/pow_shift)

            b = res_power + self.mock_freq_res()*(pow_shift/self.mock_chi())
            f0 = (b - power)/pow_shift*self.mock_chi()

            mocked_values = h - A*(w/2.0)**2 / ((w/2.0)**2 + ((freqs - f0))**2)
            # Add some extra noise
            for i, value in enumerate(mocked_values):
                d = np.abs(value - A0)

                value += np.random.normal(0, d/5, 1)
                mocked_values[i] = value
        else:
            # High power regime
            f0 = self.mock_freq_res() + self.mock_chi()
            mocked_values = h - A*(w/2.0)**2 / ((w/2.0)**2 + ((freqs - f0))**2)

        mocked_values += np.random.normal(0,
                                          self.noise(), np.size(mocked_values))

        d = det.Mock_Detector(value_names=['Magnitude'], value_units=['V'],
                              detector_control='soft', mock_values=mocked_values)

        MC.set_sweep_function(s)
        MC.set_sweep_points(freqs)
        MC.set_detector_function(d)
        MC.run('Resonator_scan'+self.msmt_suffix)

        if analyze:
            ma.Homodyne_Analysis(label='Resonator_scan', close_fig=close_fig)

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
        I0 = 20e-3
        df = 2e6

        mocked_values = []
        for dac_value in dac_values:
            h = self.ro_pulse_amp_CW()*45e-3  # Lorentian baseline [V]
            A = 0.9*h   # Height of peak [V]
            w = self.mock_res_width()    # Full width half maximum of peak
            f0 = freq_res - df*(np.sin(1/2*np.pi*(dac_value-Iref)/I0))**2
            new_values = h - A*(w/2.0)**2 / ((w/2.0)**2 +
                                             ((freqs - f0))**2)
            new_values += np.random.normal(0,
                                           self.noise(), np.size(new_values))

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
            ma.TwoD_Analysis(label='Resonator_dac_scan', close_fig=close_fig,
                             normalize=False)
            import pycqed.analysis_v2.dac_scan_analysis as dsa
            dsa.Susceptibility_to_Flux_Bias(label='Resonator_dac_scan')

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

        s = swf.None_Sweep(name='Channel Amplitude', parameter_name='mw channel amp',
                           unit='a.u.')
        f_rabi = 1/self.mock_mw_amp180()

        low_lvl = self.measurement_signal(excited=False)
        high_lvl = self.measurement_signal(excited=True)

        detuning = np.abs(self.freq_qubit() - self.mock_freq_qubit())/1e6
        highlow = (high_lvl-low_lvl)*np.exp(-detuning)

        high_lvl = low_lvl + highlow

        signal_amp = (high_lvl - low_lvl)/2
        offset = (high_lvl + low_lvl)/2
        mocked_values = offset + signal_amp*np.cos(np.pi*f_rabi*amps)

        # Add (0,0.5) normal distributed noise
        mocked_values += np.random.normal(0,
                                          self.noise(), np.size(mocked_values))
        d = det.Mock_Detector(value_names=['raw w0'], value_units=['V'],
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

        s = swf.None_Sweep(parameter_name='Time',
                           unit='s')
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
                                   artificial_detuning=artificial_detuning,
                                   textbox=False)  # Textbox gives error on std of 'tau'
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

        s = swf.None_Sweep(parameter_name='Time', unit='s')

        low_lvl = self.measurement_signal(excited=False)
        high_lvl = self.measurement_signal(excited=True)

        detuning = np.abs(self.freq_qubit() - self.mock_freq_qubit())/1e6
        highlow = (high_lvl-low_lvl)*np.exp(-detuning)

        high_lvl = low_lvl + highlow

        amplitude = high_lvl - low_lvl
        calibration = low_lvl + amplitude

        mocked_values = amplitude * \
            np.exp(-(times[0:-4]/self.mock_T1())) + low_lvl
        mocked_values = np.concatenate(
            [mocked_values, (0, 0, calibration, calibration)])

        mocked_values += np.random.normal(0,
                                          self.noise(), np.size(mocked_values))

        d = det.Mock_Detector(value_names=['raw w0'], value_units=['V'],
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

    def measurement_signal(self, excited=False):
        '''
        Sets the readout signal level, depending on the readout frequency and 
        resonator frequency.
        The 'excited' parameter indicates whether the qubit is in the excited 
        state, which results in a 2 chi shift of the resonator
        '''
        power = 20*np.log10(self.ro_pulse_amp_CW())
        # Low power regime power
        res_power = 20*np.log10(self.mock_ro_pulse_amp_CW())
        pow_shift = self.mock_pow_shift()  # At which power the high power regime occurs

        f_ro = self.ro_freq()

        h = 10**(power/20)*45e-3  # Lorentian baseline [V]
        A = 0.9*h   # Height of peak [V]
        w = self.mock_res_width()

        # Determine if in high / low power regime:
        if power <= res_power:
            f0 = self.mock_freq_res()
            if excited:
                f0 = self.mock_freq_res() + 2*self.mock_chi()

            signal = h - A*(w/2.0)**2 / ((w/2.0)**2 + ((f_ro - f0))**2)
        elif (power > res_power) and (power < res_power + pow_shift):
            # Messy regime between low and high power
            A0 = A
            n = 6
            b = A0/10
            a = (A0-b)/(-1/2*pow_shift)**n
            A = a*(power - (res_power+1/2*pow_shift))**n + b

            w = 0.5e6 + 0.5e6*np.sin(np.pi*(power - res_power)/pow_shift)

            b = res_power + self.mock_freq_res()*(pow_shift/self.mock_chi())
            f0 = (b - power)/pow_shift*self.mock_chi()

            signal = h - A*(w/2.0)**2 / ((w/2.0)**2 + ((f_ro - f0))**2)
            signal += np.random.normal(0, A0/10, 1)
        else:
            f0 = self.mock_freq_res() + self.mock_chi()
            signal = h - A*(w/2.0)**2 / ((w/2.0)**2 + ((f_ro - f0))**2)

        return signal

    ###########################################################################
    # AutoDepGraph
    ###########################################################################

    def dep_graph(self):
        cal_True_delayed = 'autodepgraph.node_functions.calibration_functions.test_calibration_True_delayed'
        self.dag = AutoDepGraph_DAG(name=self.name+' DAG')

        self.dag.add_node('VNA Wide Search',
                          calibrate_function=self.name + '.find_resonator_frequency_VNA')
        self.dag.add_node('VNA Zoom on resonators',
                          calibrate_function=self.name + '.find_resonator_frequency_VNA')
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
        self.dag.add_node(self.name + ' Frequency at Sweetspot',
                          calibrate_function=self.name + '.find_frequency')
        self.dag.add_node(self.name + ' Spectroscopy Power',
                          calibrate_function=self.name + '.find_spec_pow')
        self.dag.add_node(self.name + ' Sweetspot',
                          calibrate_function=self.name + '.find_qubit_sweetspot')
        self.dag.add_node(self.name + ' Rabi',
                          calibrate_function=cal_True_delayed)
        self.dag.add_node(self.name + ' Frequency Fine',
                          calibrate_function=self.name + '.calibrate_frequency_ramsey')
        self.dag.add_node(self.name + ' f_12 estimate',
                          calibrate_function=self.name + '.find_anharmonicity_estimate')
        self.dag.add_node(self.name + ' DAC Arc Polynomial',
                          calibrate_function=cal_True_delayed)

        # Validate qubit calibration
        self.dag.add_node(self.name + ' ALLXY',
                          calibrate_function=self.name + '.measure_allxy')
        self.dag.add_node(self.name + ' MOTZOI Calibration',
                          calibrate_function=self.name + '.calibrate_motzoi')
        # If all goes well, the qubit is fully 'calibrated' and can be controlled

        # Qubits measurements
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
        self.dag.add_edge(self.name + ' Spectroscopy Power',
                          self.name + ' Frequency Coarse')
        self.dag.add_edge(self.name + ' Sweetspot',
                          self.name + ' Frequency Coarse')
        self.dag.add_edge(self.name + ' Sweetspot',
                          self.name + ' Spectroscopy Power')
        self.dag.add_edge(self.name + ' Rabi',
                          self.name + ' Frequency at Sweetspot')
        self.dag.add_edge(self.name + ' Frequency Fine',
                          self.name + ' Frequency at Sweetspot')
        self.dag.add_edge(self.name + ' Frequency Fine',
                          self.name + ' Rabi')

        self.dag.add_edge(self.name + ' Frequency at Sweetspot',
                          self.name + ' Sweetspot')

        self.dag.add_edge(self.name + ' ALLXY',
                          self.name + ' Rabi')
        self.dag.add_edge(self.name + ' ALLXY',
                          self.name + ' Frequency Fine')
        self.dag.add_edge(self.name + ' ALLXY',
                          self.name + ' MOTZOI Calibration')

        # Perform initial measurements to see if they make sense
        self.dag.add_edge(self.name + ' T1',
                          self.name + ' ALLXY')
        self.dag.add_edge(self.name + ' T2_Echo',
                          self.name + ' ALLXY')
        self.dag.add_edge(self.name + ' T2_Star',
                          self.name + ' ALLXY')

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
                          self.name + ' Frequency at Sweetspot')

        # Measurements of anharmonicity and avoided crossing
        self.dag.add_edge(self.name + ' f_12 estimate',
                          self.name + ' Frequency at Sweetspot')
        self.dag.add_edge(self.name + ' Anharmonicity',
                          self.name + ' f_12 estimate')
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
