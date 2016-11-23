import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.analysis.analysis_toolbox import calculate_transmon_transitions
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.measurement import mc_parameter_wrapper as pw

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
from analysis import fitting_models as fit_mods


class Qubit(Instrument):
    '''
    Abstract base class for the qubit object.
    Contains a template for all the functions a qubit should have.
    N.B. This is not intended to be initialized.

    Specific types of qubits should inherit from this class, different setups
    can inherit from those to further specify the functionality.

    Possible inheritance tree
    - Qubit (general template class)
        - GateMon
        - Transmon (contains qubit specific functions)
            - transmon in setup a (contains setup specific functions)


    Naming conventions for functions
        The qubit object is a combination of a parameter holder and a
        convenient way of performing measurements. As a convention the qubit
        object contains the following types of functions designated by a prefix
        - measure_
            common measurements such as "spectroscopy" and "ramsey"
        - find_
            used to extract a specific quantity such as the qubit frequency
            these functions should always call "measure_" functions off the
            qubit objects (merge with calibrate?)
        - calibrate_
            used to find the optimal parameters of some quantity (merge with
            find?).
        - calculate_
            calculates a quantity based on parameters in the qubit object and
            those specified
            e.g. calculate_frequency


    Open for discussion:
        - is a split at the level below qubit really required?
        - is the name "find_" a good name or should it be merged with measure
            or calibrate?
        - Should the pulse-parameters be grouped here in some convenient way?
            (e.g. parameter prefixes)


    Future music (too complicated for now):
        - Instead of having a parameter resonator, attach a resonator object
          that has it's own frequency parameter, attach a mixer object that has
          it's own calibration routines.
    '''
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.msmt_suffix = '_' + name  # used to append to measuremnet labels

    def measure_T1(self):
        # Note: I made all functions lowercase but for T1 it just looks too
        # ridiculous
        raise NotImplementedError()

    def measure_rabi(self):
        raise NotImplementedError()

    def measure_ramsey(self):
        raise NotImplementedError()

    def measure_echo(self):
        raise NotImplementedError()

    def measure_allxy(self):
        raise NotImplementedError()

    def measure_ssro(self):
        raise NotImplementedError()

    def measure_spectroscopy(self):
        raise NotImplementedError()

    def measure_heterodyne_spectroscopy(self):
        raise NotImplementedError()


class Transmon(Qubit):
    '''
    circuit-QED Transmon as used in DiCarlo Lab.
    Adds transmon specific parameters as well
    '''
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_parameter('EC', units='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        self.add_parameter('EJ', units='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('assymetry',
                           parameter_class=ManualParameter)

        self.add_parameter('dac_voltage', units='mV',
                           parameter_class=ManualParameter)
        self.add_parameter('dac_sweet_spot', units='mV',
                           parameter_class=ManualParameter)
        self.add_parameter('E_c', units='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('dac_flux_coefficient', units='',
                           parameter_class=ManualParameter)
        self.add_parameter('asymmetry', units='',
                           parameter_class=ManualParameter)
        self.add_parameter('dac_channel', vals=vals.Ints(),
                           parameter_class=ManualParameter)
        self.add_parameter('flux',
                           parameter_class=ManualParameter)

        self.add_parameter('f_qubit', label='qubit frequency', units='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_max', label='qubit frequency', units='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_res', label='resonator frequency', units='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_RO', label='readout frequency', units='Hz',
                           parameter_class=ManualParameter)

        # Sequence/pulse parameters
        self.add_parameter('RO_pulse_delay', units='s',
                           parameter_class=ManualParameter)
        self.add_parameter('RO_pulse_length', units='s',
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_marker_delay', units='s',
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_marker_channel',
                           parameter_class=ManualParameter,
                           vals=vals.Strings())
        self.add_parameter('RO_amp', units='V',
                           parameter_class=ManualParameter)
        # Time between start of pulses
        self.add_parameter('pulse_delay', units='s',
                           parameter_class=ManualParameter)

        self.add_parameter('f_qubit_calc', vals=vals.Enum(None, 'dac', 'flux'),
                           # in the future add 'tracked_dac', 'tracked_flux',
                           parameter_class=ManualParameter)



    def calculate_frequency(self, EC=None, EJ=None, assymetry=None,
                            dac_voltage=None, flux=None,
                            no_transitions=1):
        '''
        Calculates transmon energy levels from the full transmon qubit
        Hamiltonian.

        Parameters of the qubit object are used unless specified.
        Flux can be specified both in terms of dac voltage or flux but not
        both.

        If specified in terms of a dac voltage it will convert the dac voltage
        to a flux using convert dac to flux (NOT IMPLEMENTED)
        '''
        if EC is None:
            EC = self.EC.get()
        if EJ is None:
            EJ = self.EJ.get()
        if assymetry is None:
            assymetry = self.assymetry.get()

        if dac_voltage is not None and flux is not None:
            raise ValueError('Specify either dac voltage or flux but not both')
        if flux is None:
            flux = self.flux.get()
        # Calculating with dac-voltage not implemented yet

        freq = calculate_transmon_transitions(
            EC, EJ, asym=assymetry, reduced_flux=flux, no_transitions=1)
        return freq

    def calculate_flux(self, frequency):
        raise NotImplementedError()

    def prepare_for_timedomain(self):
        raise NotImplementedError()

    def prepare_for_continuous_wave(self):
        raise NotImplementedError()

    def find_frequency(self, method='spectroscopy', pulsed=False,
                       steps=[1, 3, 10, 30, 100, 300, 1000],
                       freqs=None,
                       f_span=100e6, use_max=False, f_step=1e6,
                       verbose=True, update=True,
                       close_fig=True):

        if method.lower() == 'spectroscopy':
            if freqs is None:
                # If not specified it should specify wether to use the last
                # known one or wether to calculate and how
                if self.f_qubit_calc() is None:
                    freqs = np.arange(self.f_qubit.get()-f_span/2,
                                      self.f_qubit.get()+f_span/2,
                                      f_step)
                elif self.f_qubit_calc() is 'dac':
                    f_pred = fit_mods.QubitFreqDac(dac_voltage=self.IVVI.get('dac%d'%self.dac_channel()),
                                                   f_max=self.f_max()*1e-9,
                                                   E_c=self.E_c()*1e-9,
                                                   dac_sweet_spot=self.dac_sweet_spot(),
                                                   dac_flux_coefficient=self.dac_flux_coefficient(),
                                                   asymmetry=self.asymmetry())*1e9
                    freqs = np.arange(f_pred-f_span/2,
                                      f_pred+f_span/2,
                                      f_step)
                elif self.f_qubit_calc() is 'flux':
                    fluxes = self.FluxCtrl.flux_vector()
                    mappings = np.array(self.FluxCtrl.dac_mapping())
                    my_flux = np.sum(np.where(mappings == self.dac_channel(),
                                              mappings,
                                              0))-1
                    print(mappings, my_flux, self.dac_channel())
                    omega = lambda flux, f_max, EC, asym: (f_max + EC) * (asym**2 + (1-asym**2)*np.cos(np.pi*flux)**2)**0.25 - EC
                    f_pred_calc = lambda flux: omega(flux=flux,
                                                     f_max=self.f_max()*1e-9,
                                                     EC=self.E_c()*1e-9,
                                                     asym=self.asymmetry())*1e9
                    f_pred = f_pred_calc(fluxes[my_flux])
                    freqs = np.arange(f_pred-f_span/2,
                                      f_pred+f_span/2,
                                      f_step)
                    print(freqs.min(),freqs.max())
            # args here should be handed down from the top.
            self.measure_spectroscopy(freqs, pulsed=pulsed, MC=None,
                                      analyze=True, close_fig=close_fig, use_max=use_max, update=update)
            if pulsed:
                label = 'pulsed-spec'
            else:
                label = 'spectroscopy'
            analysis_spec = ma.Qubit_Spectroscopy_Analysis(label=label,close_fig=True)
            if use_max:
                self.f_qubit(analysis_spec.peaks['peak'])
            else:
                self.f_qubit(analysis_spec.fitted_freq)
            # TODO: add updating and fitting
        elif method.lower() == 'ramsey':

            stepsize = abs(1/self.f_pulse_mod.get())
            cur_freq = self.f_qubit.get()
            # Steps don't double to be more robust against aliasing
            for n in steps:
                times = np.arange(self.pulse_delay.get(),
                                  50*n*stepsize, n*stepsize)
                artificial_detuning = 4/times[-1]
                self.measure_ramsey(times,
                                    artificial_detuning=artificial_detuning,
                                    f_qubit=cur_freq,
                                    label='_{}pulse_sep'.format(n),
                                    analyze=False)
                a = ma.Ramsey_Analysis(auto=True, close_fig=close_fig,
                                       close_file=False)
                fitted_freq = a.fit_res.params['frequency'].value
                measured_detuning = fitted_freq-artificial_detuning
                cur_freq -= measured_detuning

                qubit_ana_grp = a.analysis_group.create_group(self.msmt_suffix)
                qubit_ana_grp.attrs['artificial_detuning'] = \
                    str(artificial_detuning)
                qubit_ana_grp.attrs['measured_detuning'] = \
                    str(measured_detuning)
                qubit_ana_grp.attrs['estimated_qubit_freq'] = str(cur_freq)
                a.finish()  # make sure I close the file
                if verbose:
                    print('Measured detuning:{:.2e}'.format(measured_detuning))
                    print('Setting freq to: {:.9e}, \n'.format(cur_freq))
                if times[-1] > 1.5*a.T2_star:
                    # If the last step is > T2* then the next will be for sure
                    if verbose:
                        print('Breaking of measurement because of T2*')
                    break
            if verbose:
                print('Converged to: {:.9e}'.format(cur_freq))
            if update:
                self.f_qubit.set(cur_freq)
                print("update",update)
            return cur_freq

    def find_resonator_frequency(self, **kw):
        raise NotImplementedError()

    def find_pulse_amplitude(self, amps,
                             N_steps=[3, 7, 13, 17], max_n=18,
                             close_fig=True, verbose=False,
                             MC=None, update=True, take_fit_I=False):
        '''
        Finds the pulse-amplitude using a rabi experiment.

        If amps is an array it starts by fitting a cos to a Rabi experiment
        to get an initial guess for the amplitude.
        If amps is a float it uses that as the initial amplitude and starts
        doing rabi flipping experiments around that optimum directly.
        '''
        if MC is None:
            MC = self.MC
        if np.size(amps) != 1:
            self.measure_rabi(amps, n=1, MC=MC, analyze=False)
            a = ma.Rabi_Analysis(close_fig=close_fig)
            if take_fit_I:
                    ampl = abs(a.fit_res[0].params['period'].value)/2
                    print("taking I")
            else:
                if (a.fit_res[0].params['period'].stderr <=
                        a.fit_res[1].params['period'].stderr):
                    ampl = abs(a.fit_res[0].params['period'].value)/2
                else:
                    ampl = abs(a.fit_res[1].params['period'].value)/2
        else:
            ampl = amps
        if verbose:
            print('Initial Amplitude:', ampl, '\n')

        for n in N_steps:
            if n > max_n:
                break
            else:
                ampl_span = 0.3*ampl/n
                amps = np.linspace(ampl-ampl_span, ampl+ampl_span, 15)
                self.measure_rabi(amps, n=n, MC=MC, analyze=False)
                a = ma.Rabi_parabola_analysis(close_fig=close_fig)
                if take_fit_I:
                    ampl = a.fit_res[0].params['x0'].value
                else:
                    if (a.fit_res[0].params['x0'].stderr <=
                            a.fit_res[1].params['x0'].stderr):
                        ampl = a.fit_res[0].params['x0'].value
                    else:
                        ampl = a.fit_res[1].params['x0'].value
                if verbose:
                    print('Found amplitude', ampl, '\n')
        if update:
            self.amp180.set(ampl)
            print(ampl)

        # After this it should enter a loop where it fine tunes the amplitude
        # based on fine scanes around the optimum with higher sensitivity.

    def find_amp90_scaling(self, scales=0.5,
                           N_steps=[5, 9], max_n=100,
                           close_fig=True, verbose=False,
                           MC=None, update=True, take_fit_I=False):
        '''
        Finds the scaling factor of pi/2 pulses w.r.t pi pulses using a rabi
        type with each pi pulse replaced by 2 pi/2 pulses.

        If scales is an array it starts by fitting a cos to a Rabi experiment
        to get an initial guess for the amplitude.

        This experiment is only useful after carefully calibrating the pi pulse
        using flipping sequences.
        '''
        if MC is None:
            MC = self.MC
        if np.size(scales) != 1:
            self.measure_rabi_amp90(scales=scales, n=1, MC=MC, analyze=False)
            a = ma.Rabi_Analysis(close_fig=close_fig)
            if take_fit_I:
                    scale = abs(a.fit_res[0].params['period'].value)/2
            else:
                if (a.fit_res[0].params['period'].stderr <=
                        a.fit_res[1].params['period'].stderr):
                    scale = abs(a.fit_res[0].params['period'].value)/2
                else:
                    scale = abs(a.fit_res[1].params['period'].value)/2
        else:
            scale = scales
        if verbose:
            print('Initial scaling factor:', scale, '\n')

        for n in N_steps:
            if n > max_n:
                break
            else:
                scale_span = 0.3*scale/n
                scales = np.linspace(scale-scale_span, scale+scale_span, 15)
                self.measure_rabi_amp90(scales, n=n, MC=MC, analyze=False)
                a = ma.Rabi_parabola_analysis(close_fig=close_fig)
                if take_fit_I:
                    scale = a.fit_res[0].params['x0'].value
                else:
                    if (a.fit_res[0].params['x0'].stderr <=
                            a.fit_res[1].params['x0'].stderr):
                        scale = a.fit_res[0].params['x0'].value
                    else:
                        scale = a.fit_res[1].params['x0'].value
                if verbose:
                    print('Founcaleitude', scale, '\n')
        if update:
            self.amp90_scale(scale)
            print("should be updated")
            print(scale)
