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
from pycqed.analysis import fitting_models as fit_mods


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

    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.msmt_suffix = '_' + name  # used to append to measuremnet labels
        self._operations = {}
        self.add_parameter('operations',
                           docstring='a list of all operations available on the qubit',
                           get_cmd=self._get_operations)

    def get_idn(self):
        return {'driver': self.__class__, 'name': self.name}

    def _get_operations(self):
        return self._operations

    def measure_T1(self):
        # Note: I made all functions lowercase but for T1 it just looks too
        # ridiculous
        raise NotImplementedError()

    def measure_rabi(self):
        raise NotImplementedError()


    def measure_flipping_seq(self,  N=np.arange(31)*2,
                     MC=None, analyze=True, close_fig=True,
                     verbose=False, upload=True):
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

    def measure_transients(self):
        raise NotImplementedError()

    def calibrate_optimal_weights(self):
        raise NotImplementedError()

    def measure_heterodyne_spectroscopy(self):
        raise NotImplementedError()

    def add_operation(self, operation_name):
        self._operations[operation_name] = {}

    def link_param_to_operation(self, operation_name, parameter_name,
                                argument_name):
        """
        Links an existing param to an operation for use in the operation dict.

        An example of where to use this would be the flux_channel.
        Only one parameter is specified but it is relevant for multiple flux
        pulses. You don't want a different parameter that specifies the channel
        for the iSWAP and the CZ gate. This can be solved by linking them to
        your operation.

        Args:
            operation_name (str): The operation of which this parameter is an
                argument. e.g. mw_control or CZ
            parameter_name (str): Name of the parameter
            argument_name  (str): Name of the arugment as used in the sequencer
            **kwargs get passed to the add_parameter function
        """
        if parameter_name not in self.parameters:
            raise KeyError('Parameter {} needs to be added first'.format(
                parameter_name))

        if operation_name in self.operations().keys():
            self._operations[operation_name][argument_name] = parameter_name
        else:
            raise KeyError('Unknown operation {}, add '.format(operation_name) +
                           'first using add operation')

    def add_pulse_parameter(self,
                            operation_name,
                            parameter_name,
                            argument_name,
                            initial_value=None,
                            vals=vals.Numbers(),
                            **kwargs):
        """
        Add a pulse parameter to the qubit.

        Args:
            operation_name (str): The operation of which this parameter is an
                argument. e.g. mw_control or CZ
            parameter_name (str): Name of the parameter
            argument_name  (str): Name of the arugment as used in the sequencer
            **kwargs get passed to the add_parameter function
        Raises:
            KeyError: if this instrument already has a parameter with this
                name.
        """
        if parameter_name in self.parameters:
            raise KeyError(
                'Duplicate parameter name {}'.format(parameter_name))

        if operation_name in self.operations().keys():
            self._operations[operation_name][argument_name] = parameter_name
        else:
            raise KeyError('Unknown operation {}, add '.format(operation_name) +
                           'first using add operation')

        self.add_parameter(parameter_name,
                           initial_value=initial_value,
                           vals=vals,
                           parameter_class=ManualParameter, **kwargs)

        # for use in RemoteInstruments to add parameters to the server
        # we return the info they need to construct their proxy
        return

    def get_operation_dict(self, operation_dict={}):
        for op_name, op in self.operations().items():
            operation_dict[op_name + ' ' + self.name] = {'target_qubit':
                                                         self.name}
            for argument_name, parameter_name in op.items():
                operation_dict[op_name + ' ' + self.name][argument_name] = \
                    self.get(parameter_name)
        return operation_dict


class Transmon(Qubit):

    '''
    circuit-QED Transmon as used in DiCarlo Lab.
    Adds transmon specific parameters as well
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_parameter('E_c', unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        self.add_parameter('E_j', unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        self.add_parameter('dac_voltage', unit='mV',
                           parameter_class=ManualParameter)
        self.add_parameter('dac_sweet_spot', unit='mV',
                           parameter_class=ManualParameter)
        self.add_parameter('dac_flux_coefficient', unit='',
                           parameter_class=ManualParameter)
        self.add_parameter('asymmetry', unit='',
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('dac_channel', vals=vals.Ints(),
                           parameter_class=ManualParameter)

        self.add_parameter('f_qubit', label='qubit frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_max', label='qubit frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_res', label='resonator frequency', unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('f_RO', label='readout frequency', unit='Hz',
                           parameter_class=ManualParameter)

        # Sequence/pulse parameters
        self.add_parameter('RO_pulse_delay', unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('RO_pulse_length', unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_marker_delay', unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('RO_acq_marker_channel',
                           parameter_class=ManualParameter,
                           vals=vals.Strings())
        self.add_parameter('RO_amp', unit='V',
                           parameter_class=ManualParameter)
        # Time between start of pulses
        self.add_parameter('pulse_delay', unit='s',
                           initial_value=0,
                           vals=vals.Numbers(0, 1e-6),
                           parameter_class=ManualParameter)

        self.add_parameter('f_qubit_calc_method',
                           vals=vals.Enum('latest', 'dac', 'flux'),
                           # in the future add 'tracked_dac', 'tracked_flux',
                           initial_value='latest',
                           parameter_class=ManualParameter)

    def calculate_frequency(self,
                            dac_voltage=None,
                            flux=None):
        '''
        Calculates the f01 transition frequency from the cosine arc model.
        (function available in fit_mods. Qubit_dac_to_freq)

        Parameters of the qubit object are used unless specified.
        Flux can be specified both in terms of dac voltage or flux but not
        both.
        '''

        if dac_voltage is not None and flux is not None:
            raise ValueError('Specify either dac voltage or flux but not both')

        if self.f_qubit_calc_method() is 'latest':
            f_qubit_estimate = self.f_qubit()

        elif self.f_qubit_calc_method() == 'dac':
            if dac_voltage is None:
                dac_voltage = self.IVVI.get_instr().get(
                        'dac{}'.format(self.dac_channel()))

            f_qubit_estimate = fit_mods.Qubit_dac_to_freq(
                dac_voltage=dac_voltage,
                f_max=self.f_max(),
                E_c=self.E_c(),
                dac_sweet_spot=self.dac_sweet_spot(),
                dac_flux_coefficient=self.dac_flux_coefficient(),
                asymmetry=self.asymmetry())

        elif self.f_qubit_calc_method() == 'flux':
            if flux is None:
                flux = self.FluxCtrl.get_instr().get(
                    'flux{}'.format(self.dac_channel()))
            f_qubit_estimate = fit_mods.Qubit_dac_to_freq(
                dac_voltage=flux,
                f_max=self.f_max(),
                E_c=self.E_c(),
                dac_sweet_spot=0,
                dac_flux_coefficient=1,
                asymmetry=self.asymmetry())
        return f_qubit_estimate

    def calculate_flux(self, frequency):
        raise NotImplementedError()

    def prepare_for_timedomain(self):
        raise NotImplementedError()

    def prepare_for_continuous_wave(self):
        raise NotImplementedError()

    def find_frequency(self, method='spectroscopy', pulsed=False,
                       steps=[1, 3, 10, 30, 100, 300, 1000],
                       freqs=None,
                       f_span=100e6,
                       use_max=False,
                       f_step=1e6,
                       verbose=True,
                       update=True,
                       close_fig=True):
        """
        Finds the qubit frequency using either the spectroscopy or the Ramsey
        method.
        Frequency prediction is done using
        """

        if method.lower() == 'spectroscopy':
            if freqs is None:
                f_qubit_estimate = self.calculate_frequency()
                freqs = np.arange(f_qubit_estimate - f_span/2,
                                  f_qubit_estimate + f_span/2,
                                  f_step)
            # args here should be handed down from the top.
            self.measure_spectroscopy(freqs, pulsed=pulsed, MC=None,
                                      analyze=True, close_fig=close_fig)
            if pulsed:
                label = 'pulsed-spec'
            else:
                label = 'spectroscopy'
            analysis_spec = ma.Qubit_Spectroscopy_Analysis(
                label=label, close_fig=True)

            if update:
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
            return cur_freq

    def find_resonator_frequency(self, use_min=False,
                                 update=True,
                                 freqs=None,
                                 MC=None, close_fig=True):
        '''
        Finds the resonator frequency by performing a heterodyne experiment
        if freqs == None it will determine a default range dependent on the
        last known frequency of the resonator.
        '''
        if freqs is None:
            f_center = self.f_res.get()
            f_span = 10e6
            f_step = 100e3
            freqs = np.arange(f_center-f_span/2, f_center+f_span/2, f_step)
        self.measure_heterodyne_spectroscopy(freqs, MC, analyze=False)
        a = ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)
        if use_min:
            f_res = a.min_frequency
        else:
            f_res = a.fit_results.params['f0'].value*1e9  # fit converts to Hz
        if f_res > max(freqs) or f_res < min(freqs):
            logging.warning('exracted frequency outside of range of scan')
        elif update:  # don't update if the value is out of the scan range
            self.f_res.set(f_res)
        self.f_RO(self.f_res())
        return f_res

    def find_pulse_amplitude(self, amps=np.linspace(-.5, .5, 31),
                             N_steps=[3, 7, 13, 17], max_n=18,
                             close_fig=True, verbose=False,
                             MC=None, update=True, take_fit_I=False):
        '''
        Finds the pulse-amplitude using a Rabi experiment.
        Fine tunes by doing a Rabi around the optimum with an odd
        multiple of pulses.

        Args:
            amps: (array or float) amplitudes of the first Rabi if an array,
                if a float is specified it will be treated as an estimate
                for the amplitude to be found.
            N_steps: (list of int) number of pulses used in the fine tuning
            max_n: (int) break of if N> max_n
        '''
        if MC is None:
            MC = self.MC.get_instr()
        if np.size(amps) != 1:
            self.measure_rabi(amps, n=1, MC=MC, analyze=False)
            a = ma.Rabi_Analysis(close_fig=close_fig)
            # Decide which quadrature to take by comparing the contrast
            if take_fit_I:
                ampl = abs(a.fit_res[0].params['period'].value)/2.
            elif (np.abs(max(a.measured_values[0]) -
                         min(a.measured_values[0]))) > (
                    np.abs(max(a.measured_values[1]) -
                           min(a.measured_values[1]))):
                ampl = a.fit_res[0].params['period'].value/2.
            else:
                ampl = a.fit_res[1].params['period'].value/2.
        else:
            ampl = amps
        if verbose:
            print('Initial Amplitude:', ampl, '\n')

        for n in N_steps:
            if n > max_n:
                break
            else:
                old_amp = ampl
                ampl_span = 0.35*ampl/n
                amps = np.linspace(ampl-ampl_span, ampl+ampl_span, 15)
                self.measure_rabi(amps, n=n, MC=MC, analyze=False)
                a = ma.Rabi_parabola_analysis(close_fig=close_fig)
                # Decide which quadrature to take by comparing the contrast
                if take_fit_I:
                    ampl = a.fit_res[0].params['x0'].value
                elif min(amps)<a.fit_res[0].params['x0'].value<max(amps)\
                        and min(amps)<a.fit_res[1].params['x0'].value<max(amps):
                    if (np.abs(max(a.fit_res[0].data)-min(a.fit_res[0].data)))>\
                        (np.abs(max(a.fit_res[1].data)-min(a.fit_res[1].data))):
                        ampl = a.fit_res[0].params['x0'].value
                    else:
                        ampl = a.fit_res[1].params['x0'].value
                elif min(amps)<a.fit_res[0].params['x0'].value<max(amps):
                    ampl = a.fit_res[0].params['x0'].value
                elif min(amps)<a.fit_res[1].params['x0'].value<max(amps):
                    ampl = a.fit_res[1].params['x0'].value
                else:
                    ampl_span*=1.5
                    amps = np.linspace(old_amp-ampl_span, old_amp+ampl_span, 15)
                    self.measure_rabi(amps, n=n, MC=MC, analyze=False)
                    a = ma.Rabi_parabola_analysis(close_fig=close_fig)
                    # Decide which quadrature to take by comparing the contrast
                    if take_fit_I:
                        ampl = a.fit_res[0].params['x0'].value
                    elif (np.abs(max(a.measured_values[0]) -
                                 min(a.measured_values[0]))) > (
                          np.abs(max(a.measured_values[1]) -
                                 min(a.measured_values[1]))):
                        ampl = a.fit_res[0].params['x0'].value
                    else:
                        ampl = a.fit_res[1].params['x0'].value
                if verbose:
                    print('Found amplitude', ampl, '\n')
        if update:
            self.amp180.set(ampl)


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
