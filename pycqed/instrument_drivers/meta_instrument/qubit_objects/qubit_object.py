import logging
import numpy as np
import time

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.utilities.general import gen_sweep_pts
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import fitting_models as fit_mods


class Qubit(Instrument):

    '''
    Abstract base class for the qubit object.
    Contains a template for all methods a qubit (should) has.
    N.B. This is not intended to be initialized.

    Specific types of qubits should inherit from this class, different
    hardware configurations can inherit from those to further specify
    the functionality.

    Possible inheritance tree
    - Qubit (general template class)
        - GateMon
        - Transmon (contains qubit specific methods)
            - transmon in setup a (contains setup specific methods)


    Naming conventions for methods
        The qubit object is a combination of a parameter holder and a
        convenient way of performing measurements. As a convention the qubit
        object contains the following types of methods designated by a prefix

        - measure_xx() -> bool
            A measure_xx method performs a specific experiment such as
                a "spectroscopy" or "ramsey".
            A measure_xx method typically has a hardware dependent
            implementation

        - calibrate_xx() -> bool
            A calibrate_xx method defines a standard protocol to perform a
                specific calibration.
            A calibrate_xx method should be blind callable (callable without
                specifying any arguments).
            A calibrate_xx method should return a boolean indicating the
                success of the calibration.
            A calibrate_xx method should update the internal parameter it is
                related to.
            A calibrate_xx method should be defined in the abstract base class
                whenever possible and rely on implementations of corresponding
                measure_xx methods in the hardware dependent child classes.

        - find_xx
            similar to calibrate_xx() naming difference is historical

        - calculate_
            calculates a quantity based on parameters specified in the qubit
            object e.g. calculate_frequency

    Naming conventions for parameters:
        (only for qubit objects after Sept 2017)
        Parameters are grouped based on their functionality. This grouping
        is achieved through the parameter name.

        Prefixes are listed here:
            instr_  : references to other instruments
            ro_     : parameters relating to RO both CW and TD readout.
            mw_     : parameters of single qubit MW control
            spec_   : parameters relating to spectroscopy (single qubit CW)
            fl_     : parameters relating to flux control, this includes both
                      flux pulsing as well as flux offset (DC).
            cfg_    : configuration, this can be info relevant for compilers
                      or configurations that determine how the qubit operates.
                      examples are cfg_qasm and cfg_f_qubit_calc_method.

            ""      : properties of the qubit do not have a prefix, examples
                      are T1, T2, etc., F_ssro, F_RB, etc., f_qubit, E_C, etc.

    Open for discussion:
        - is a split at the level below qubit really required?
        - is the name "find_" a good name or should it be merged with measure
            or calibrate?
        - Should the pulse-parameters be grouped here in some convenient way?
            (e.g. parameter prefixes)
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.msmt_suffix = '_' + name  # used to append to measurement labels
        self._operations = {}
        self.add_parameter('operations',
                           docstring='a list of all operations available on the qubit',
                           get_cmd=self._get_operations)

    def connect_message(self, begin_time=None):
        t = time.time() - (begin_time or self._t0)

        con_msg = ('Connected to: {repr} '
                   'in {t:.2f} s'.format(repr=self.__repr__(), t=t))
        print(con_msg)

    def add_parameters(self):
        """
        Add parameters to the qubit object grouped according to the
        naming conventions described above

        Prefixes are listed here:
            instr_  : references to other instruments
            ro_     : parameters relating to RO both CW and TD readout.
            mw_     : parameters of single qubit MW control
            spec_   : parameters relating to spectroscopy (single qubit CW)
            fl_     : parameters relating to flux control, this includes both
                      flux pulsing as well as flux offset (DC).
            cfg_    : configuration, this can be info relevant for compilers
                      or configurations that determine how the qubit operates.
                      examples are cfg_qasm and cfg_f_qubit_calc_method.

            ""      : properties of the qubit do not have a prefix, examples
                      are T1, T2, etc., F_ssro, F_RB, etc., f_qubit, E_C, etc.
        """
        self.add_instrument_ref_parameters()
        self.add_ro_parameters()
        self.add_mw_parameters()
        self.add_spec_parameters()
        self.add_flux_parameters()
        self.add_config_parameters()
        self.add_generic_qubit_parameters()

    def add_instrument_ref_parameters(self):
        pass

    def add_ro_parameters(self):
        pass

    def add_mw_parameters(self):
        pass

    def add_spec_parameters(self):
        pass

    def add_flux_parameters(self):
        pass

    def add_config_parameters(self):
        pass

    def add_generic_qubit_parameters(self):
        pass

    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name}

    def _get_operations(self):
        return self._operations

    def measure_T1(self, times=None, MC=None,
                   close_fig: bool=True, update: bool=True)->float:
        """
        Performs a T1 experiment.
        Args:
            times:      array of times to measure at, if None will define a
                        suitable range based on the last known T1
            MC:         instance of the MeasurementControl
            close_fig:  close the figure in plotting
            update :    update self.T1 with the measured value

        returns:
            T1 (float) the measured value
        """

        # Note: I made all functions lowercase but for T1 it just looks too
        # ridiculous
        raise NotImplementedError()

    def measure_rabi(self):
        raise NotImplementedError()

    def measure_flipping(self,  number_of_flips=2*np.arange(60),
                         MC=None, label='',
                         equator=True,
                         analyze=True, close_fig=True, verbose=True):
        raise NotImplementedError

    def measure_ramsey(self):
        raise NotImplementedError()

    def measure_echo(self):
        raise NotImplementedError()

    def measure_allxy(self, MC=None, analyze: bool=True,
                      close_fig: bool=True):
        """
        Performs an AllXY experiment.
        Args:
            MC        : instance of the MeasurementControl
            analyze   : perform analysis
            close_fig : close the figure in plotting

        returns:
            T1 (float) the measured value
        """
        raise NotImplementedError()

    def measure_ssro(self):
        raise NotImplementedError()

    def measure_spectroscopy(self):
        raise NotImplementedError()

    def measure_transients(self, MC=None, analyze: bool=True,
                           cases=('off', 'on'),
                           prepare: bool=True):
        '''
        Measure transients for the cases specified.
        Args:
            MC      (instr): measurement control
            analyze (bool) : run analysis and create figure
            cases   (list) : list of strings specifying cases to perform
                transients for, valid cases are "off" and "on" corresponding
                to preparing the qubit in the 0 or 1 state respectively.
            prepare (bool) : if True runs prepare for timedomain before
                measuring the transients
        Returns:
            list of numpy arrays containing the transients for the cases
            specified.
        '''
        if prepare:
            self.prepare_for_timedomain()
        raise NotImplementedError()

    def measure_motzoi(self, motzois=np.linspace(-.3, .3, 31),
                       MC=None, analyze=True, close_fig=True):
        raise NotImplementedError()

    def find_resonator_frequency(self, use_min=False,
                                 update=True,
                                 freqs=None,
                                 MC=None, close_fig=True):
        '''
        Finds the resonator frequency by performing a heterodyne experiment
        if freqs == None it will determine a default range dependent on the
        last known frequency of the resonator.
        '''
        # This snippet exists to be backwards compatible 9/2017.
        try:
            freq_res_par = self.freq_res
            freq_RO_par = self.ro_freq
        except AttributeError():
            logging.warning('Rename the f_res parameter to freq_res')
            freq_res_par = self.f_res
            freq_RO_par = self.f_RO

        if freqs is None:
            f_center = freq_res_par()
            if f_center is None:
                raise ValueError('Specify "freq_res" to generate a freq span')
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
            freq_res_par(f_res)
            freq_RO_par(f_res)
        return f_res

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

            label = 'spec'
            analysis_spec = ma.Qubit_Spectroscopy_Analysis(
                label=label, close_fig=True)

            if update:
                if use_max:
                    self.freq_qubit(analysis_spec.peaks['peak'])
                else:
                    self.freq_qubit(analysis_spec.fitted_freq)
                # TODO: add updating and fitting
        elif method.lower() == 'ramsey':
            return self.calibrate_frequency_ramsey(
                steps=steps, verbose=verbose, update=update,
                close_fig=close_fig)
        return self.freq_qubit()

    def calibrate_motzoi(self, MC=None, verbose=True, update=True):
        motzois = gen_sweep_pts(center=0, span=1, num=31)

        # large range
        a = self.measure_motzoi(MC=MC, motzois=motzois, analyze=True)
        opt_motzoi = a.optimal_motzoi
        if opt_motzoi > max(motzois) or opt_motzoi < min(motzois):
            if verbose:
                print('optimal motzoi {:.3f} '.format(opt_motzoi) +
                      'outside of measured span, aborting')
            return False

        # fine range around optimum
        motzois = gen_sweep_pts(center=a.optimal_motzoi, span=.4, num=31)
        a = self.measure_motzoi(motzois)
        opt_motzoi = a.optimal_motzoi
        if opt_motzoi > max(motzois) or opt_motzoi < min(motzois):
            if verbose:
                print('optimal motzoi {:.3f} '.format(opt_motzoi) +
                      'outside of measured span, aborting')
        if update:
            if verbose:
                print('Setting motzoi to {:.3f}'.format(opt_motzoi))
            self.motzoi(opt_motzoi)
        return opt_motzoi

    def calibrate_optimal_weights(self, MC=None, verify=True,
                                  analyze=False, update=True)->bool:
        raise NotImplementedError()

    def calibrate_MW_RO_latency(self, MC=None, update: bool=True)-> bool:
        """
        Calibrates parameters:
            "latency_MW"
            "RO_acq_delay"


        Used to calibrate the delay of the MW pulse with respect to the
        RO pulse and the RO acquisition delay.


        The MW_pulse_latency is calibrated by setting the frequency of
        the LO to the qubit frequency such that both the MW and the RO pulse
        will show up in the RO.
        Measuring the transients will  show what the optimal latency is.

        Note that a lot of averages may be required when using dedicated drive
        lines.

        This function does NOT overwrite the values that were set in the qubit
        object and as such can be used to verify the succes of the calibration.

        Currently (28/6/2017) the experiment has to be analysed by hand.

        """
        raise NotImplementedError()

    def calibrate_Flux_pulse_latency(self, MC=None, update=True)-> bool:
        """
        Calibrates parameter: "latency_Flux"

        Used to calibrate the timing between the MW and Flux pulses.

        Flux pulse latency is calibrated using a Ram-Z experiment.
        The experiment works as follows:
        - x90 | square_flux  # defines t = 0
        - wait (should be slightly longer than the pulse duration)
        - x90
        - wait
        - RO

        The position of the square flux pulse is varied to find the
        optimal latency.
        """
        raise NotImplementedError
        return True

    def calibrate_MW_RO_latency(self, MC=None, update: bool=True)-> bool:
        """
        Calibrates parameters:
            "latency_MW"
            "RO_acq_delay"


        Used to calibrate the delay of the MW pulse with respect to the
        RO pulse and the RO acquisition delay.


        The MW_pulse_latency is calibrated by setting the frequency of
        the LO to the qubit frequency such that both the MW and the RO pulse
        will show up in the RO.
        Measuring the transients will  show what the optimal latency is.

        Note that a lot of averages may be required when using dedicated drive
        lines.

        This function does NOT overwrite the values that were set in the qubit
        object and as such can be used to verify the succes of the calibration.

        Currently (28/6/2017) the experiment has to be analysed by hand.

        """
        raise NotImplementedError()
        return True

    def calibrate_Flux_pulse_latency(self, MC=None, update=True)-> bool:
        """
        Calibrates parameter: "latency_Flux"

        Used to calibrate the timing between the MW and Flux pulses.

        Flux pulse latency is calibrated using a Ram-Z experiment.
        The experiment works as follows:
        - x90 | square_flux  # defines t = 0
        - wait (should be slightly longer than the pulse duration)
        - x90
        - wait
        - RO

        The position of the square flux pulse is varied to find the
        optimal latency.
        """
        raise NotImplementedError
        return True

    def calculate_frequency(self, calc_method=None, V_per_phi0=None, V=None):
        '''
        Calculates an estimate for the qubit frequency.
        Arguments are optional and parameters of the object are used if not
        specified.
        Args:
            calc_method : can be "latest" or "flux" uses last known frequency
                    or calculates using the cosine arc model as specified
                    in fit_mods.Qubit_dac_to_freq
                corresponding par. : cfg_qubit_freq_calc_method

            V_per_phi0 : dac flux coefficient, converts volts to Flux.
                    Set to 1 to reduce the model to pure flux.
                corresponding par. : fl_dc_V_per_phi
            V  : dac value used when calculating frequency
                corresponding par. : fl_dc_V

        Calculates the f01 transition frequency using the cosine arc model.
        (function available in fit_mods. Qubit_dac_to_freq)

        The parameter cfg_qubit_freq_calc_method determines how it is
        calculated.
        Parameters of the qubit object are used unless specified.
        Flux can be specified both in terms of dac voltage or flux but not
        both.
        '''
        if self.cfg_qubit_freq_calc_method() == 'latest':
            qubit_freq_est = self.freq_qubit()

        elif self.cfg_qubit_freq_calc_method() == 'flux':
            if V is None:
                V = self.fl_dc_V()
            if V_per_phi0 is None:
                V_per_phi0 = self.fl_dc_V_per_phi0()

            qubit_freq_est = fit_mods.Qubit_dac_to_freq(
                dac_voltage=V,
                f_max=self.freq_max(),
                E_c=self.E_c(),
                dac_sweet_spot=self.fl_dc_V0(),
                V_per_phi0=V_per_phi0,
                asymmetry=self.asymmetry())

        return qubit_freq_est

    def calibrate_mixer_offsets_drive(self, update: bool=True)-> bool:
        '''
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        '''
        raise NotImplementedError()

        return True

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
        self.add_parameter('T1', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 200e-6))
        self.add_parameter('T2_echo', unit='s',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('T2_star', unit='s',
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
        self.add_parameter('F_ssro',
                           initial_value=0,
                           label='RO assignment fidelity',
                           vals=vals.Numbers(0.0, 1.0),
                           parameter_class=ManualParameter)
        self.add_parameter('F_discr',
                           initial_value=0,
                           label='RO discrimination fidelity',
                           vals=vals.Numbers(0.0, 1.0),
                           parameter_class=ManualParameter)
        self.add_parameter('F_RB',
                           initial_value=0,
                           label='RB single qubit Clifford fidelity',
                           vals=vals.Numbers(0, 1.0),
                           parameter_class=ManualParameter)
        self.add_parameter('V_per_phi0',
                           initial_value=1,
                           label='V per phi0',
                           vals=vals.Numbers(),
                           docstring='Conversion between flux and voltage. '
                                     'How many volts need to be applied to '
                                     'have a flux of 1 phi0 (pulsed).',
                           parameter_class=ManualParameter)
        self.add_parameter('V_offset',
                           initial_value=0,
                           label='V offset',
                           vals=vals.Numbers(),
                           docstring='AWG voltage at which the sweet spot is '
                                     'found (pulsed).',
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

    def prepare_readout(self):
        """
        Configures the readout. Consists of the following steps
        - instantiate the relevant detector functions
        - set the microwave frequencies and sources
        - generate the RO pulse
        - set the integration weights
        """
        raise NotImplementedError()

    def calibrate_frequency_ramsey(self, steps=[1, 1, 3, 10, 30, 100, 300, 1000],
                                   stepsize=None, verbose=True, update=True,
                                   close_fig=True):
        if stepsize is None:
            stepsize = abs(1/self.f_pulse_mod.get())
        cur_freq = self.f_qubit.get()
        # Steps don't double to be more robust against aliasing
        for n in steps:
            times = np.arange(self.pulse_delay.get(),
                              50*n*stepsize, n*stepsize)
            artificial_detuning = 2.5/times[-1]
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
            if times[-1] > 2.*a.T2_star:
                # If the last step is > T2* then the next will be for sure
                if verbose:
                    print('Breaking of measurement because of T2*')
                break
        if verbose:
            print('Converged to: {:.9e}'.format(cur_freq))
        if update:
            self.f_qubit.set(cur_freq)
        return cur_freq

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
            return self.calibrate_frequency_ramsey(
                steps=steps, verbose=verbose, update=update,
                close_fig=close_fig)
        return self.f_qubit()

    def find_frequency_pulsed(self):
        raise NotImplementedError()

    def find_frequency_cw_spec(self):
        raise NotImplementedError()

    def calibrate_pulse_amplitude_coarse(self,
                                         amps=np.linspace(-.5, .5, 31),
                                         close_fig=True, verbose=False,
                                         MC=None, update=True,
                                         take_fit_I=False):
        """
        Calibrates the pulse amplitude using a single rabi oscillation
        """

        self.measure_rabi(amps, n=1, MC=MC, analyze=False)
        a = ma.Rabi_Analysis(close_fig=close_fig)
        # Decide which quadrature to take by comparing the contrast
        if take_fit_I or len(a.measured_values) == 1:
            ampl = abs(a.fit_res[0].params['period'].value)/2.
        elif (np.abs(max(a.measured_values[0]) -
                     min(a.measured_values[0]))) > (
                np.abs(max(a.measured_values[1]) -
                       min(a.measured_values[1]))):
            ampl = a.fit_res[0].params['period'].value/2.
        else:
            ampl = a.fit_res[1].params['period'].value/2.

        if update:
            self.Q_amp180.set(ampl)
        return ampl

    def calibrate_pulse_amplitude_flipping(self,
                                           MC=None, update: bool=True,
                                           fine_accuracy: float=0.005,
                                           desired_accuracy: float=0.00005,
                                           max_iterations: int=10,
                                           verbose: bool=True):
        """
        Calibrates the pulse amplitude using a flipping sequence.
        The flipping sequence itself should be implemented using the
        "measure_flipping" method.
        It converges to the optimal amplitude using first a coarse and then
        a finer scan with more pulses.

        Args:
            MC                    : The measurement control used, if None
                                   uses the one specified in the qubit object.
            updates       (bool)  : if True updates the Q_amp180 parameter
            fine_accuracy (float) : the accuracy to switch to the fine scan
            desired_accuracy (float): the accuracy after which to terminate
                                      the optimization
            max_iterations  (int) : always terminate after this number of
                                     optimizations.
            verbose         (bool): if true adds additional print statements.
        returns:
            success         (bool): True if optimization converged.
        """
        success = False
        fine = False
        for k in range(max_iterations):
            old_Q_amp180 = self.Q_amp180()
            if not fine:
                number_of_flips = 2*np.arange(60)
            if fine:
                number_of_flips = 8*np.arange(60)
            a = self.measure_flipping(MC=MC, number_of_flips=number_of_flips)
            Q_amp180_scale_factor = a.get_scale_factor()

            # Check if Q_amp180_scale_factor is within boundaries
            if Q_amp180_scale_factor > 1.1:
                Q_amp180_scale_factor = 1.1
                if verbose:
                    print('Qubit drive scaling %.3f ' % Q_amp180_scale_factor
                          + 'is too high, capping at 1.1')
            elif Q_amp180_scale_factor < 0.9:
                Q_amp180_scale_factor = 0.9
                if verbose:
                    print('Qubit drive scaling %.3f ' % Q_amp180_scale_factor
                          + 'is too low, capping at 0.9')

            self.Q_amp180(np.round(Q_amp180_scale_factor * self.Q_amp180(), 7))

            if verbose:
                print('Q_amp180_scale_factor: {:.4f}, new Q_amp180: {}'.format(
                      Q_amp180_scale_factor, self.Q_amp180()))

            if (abs(Q_amp180_scale_factor-1) < fine_accuracy) and (not fine):
                if verbose:
                    print('Getting close to optimum, increasing sensitivity')
                fine = True

            if abs(Q_amp180_scale_factor-1) < desired_accuracy:
                if verbose:
                    print('within threshold')
                success = True
                break

        # If converged?
        if success and verbose:
            print('Drive calibration set to {}'.format(self.Q_amp180()))
        if not update or not success:
            self.Q_amp180(old_Q_amp180)
        return success

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
            ampl = self.calibrate_pulse_amplitude_coarse(
                amps=amps, close_fig=close_fig, verbose=verbose,
                MC=MC, update=update,
                take_fit_I=take_fit_I)
        else:
            ampl = amps
        if verbose:
            print('Initial Amplitude:', ampl, '\n')

        for n in N_steps:
            if n > max_n:
                break
            else:
                old_amp = ampl
                ampl_span = 0.5*ampl/n
                amps = np.linspace(ampl-ampl_span, ampl+ampl_span, 15)
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
                if not min(amps) < ampl < max(amps):
                    ampl_span *= 2
                    amps = np.linspace(old_amp-ampl_span,
                                       old_amp+ampl_span, 15)
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
            self.Q_amp180.set(ampl)
        return ampl

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
