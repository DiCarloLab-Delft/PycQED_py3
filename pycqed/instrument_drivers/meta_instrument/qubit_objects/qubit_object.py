import logging
import numpy as np
import time
import warnings

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from pycqed.measurement import detector_functions as det
from qcodes.instrument.parameter import ManualParameter

from pycqed.utilities.general import gen_sweep_pts
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import measurement_analysis as ma2
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.tools import plotting as plt_tools
from pycqed.instrument_drivers.meta_instrument.Resonator import resonator

class Qubit(Instrument):

    """
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

        - tune_xx_to_
            Similar to calibrate but actively tries to set a parameter xx to a
            specific target value. An example is tune_to_frequency where
            several routines are used to set the qubit frequency to a desired
            value.

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
            tim_    : parameters related to timing, used to set latencies,
                        these are generally part of a device object (rather
                        than the qubit objects) but are listed here for
                        completeness.
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
    """

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
                   close_fig: bool=True, update: bool=True,
                   prepare_for_timedomain: bool=True)->float:
        """
        Performs a T1 experiment.
        Args:
            times (array):
                array of times to measure at, if None will define a
                suitable range based on the last known T1

            MC (MeasurementControl):
                instance of the MeasurementControl

            close_fig (bool):
                close the figure in plotting

            update (bool):
                update self.T1 with the measured value

        returns:
            T1 (float):
                the measured value
        """

        # Note: I made all functions lowercase but for T1 it just looks too
        # ridiculous
        raise NotImplementedError()

    def measure_rabi(self):
        raise NotImplementedError()


    def measure_flipping(self, number_of_flips=np.arange(20), equator=True,
                         MC=None, analyze=True, close_fig=True, update=True,
                         ax='x', angle='180'):
        raise NotImplementedError()

    def measure_ramsey(self):
        """
        Ramsey measurement used to measure the inhomogenuous dephasing time T2* as well as
        the qubit frequency. The measurement consists of the pi/2 pulses with a variable delay
        time between. The MW LO can be intentionally detuned from the qubit frequency.
        Consequently the measurement yields decaying oscillations which is easier to fit
        accurately than the monotonuous decay.

        Args:
            times (array):
                array of delay times between the two pi/2 pulses

            artificial_detuning (float):
                intentional detuing from the known qubit frequency
        """
        raise NotImplementedError()

    def measure_echo(self, times=None, MC=None,
                     analyze=True, close_fig=True, update=True):
        """
        Performs the Hahn echo measurement to estimate dephasing time of the qubit decouplied
        from the majority of the low frequency noise. The sequence of the experiment is
        pi/2 - wait/2 - pi - wait/2 - pi/2
        with variable (identical) delay times between pulses. The final pi/2 pulse is performed
        around variable axis. Consequently the measurement yields decaying oscillatioins instead
        of monotunous decay, which enables to more easily spot potential problems with the applied
        microwave pulses.

        Args:
            times (array):
                list of total waiting time between two pi/2 pulses. Half of the delay
                is inserted before, and half after the central pi pule.
        """
        raise NotImplementedError()

    def measure_allxy(self, MC=None, analyze: bool=True,
                      close_fig: bool=True,
                      prepare_for_timedomain: bool=True):
        """
        Performs an AllXY experiment. AllXY experiment consists of 21 pairs of
        MW control pulses folowed by the qubit measurement (in this routine
        each pair is repeated twice). In the ideal case the result of
        this measurement should be a staircase, and specific errors in the MW gate tuenup
        result in characteristic deviations from the ideal shape.

        For detailed description of the AllXY measurement and symptomes of different errors
        see PhD thesis by Matthed Reed (2013, Schoelkopf lab), pp. 124.
        https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf

        Args:
            MC (MeasurementControl):
                instance of the MeasurementControl

            analyze (bool):
                perform analysis

            close_fig (bool):
                close the figure in plotting
        """
        raise NotImplementedError()

    def measure_ssro(self, MC=None, analyze: bool=True, nr_shots: int=1024*8,
                     cases=('off', 'on'), update_threshold: bool=True,
                     prepare: bool=True, no_figs: bool=False,
                     update: bool=True,
                     verbose: bool=True):
        raise NotImplementedError()


    def measure_spectroscopy(self, freqs, pulsed=True, MC=None,
                             analyze=True, close_fig=True):
        raise NotImplementedError()


    def measure_resonator_power(self, freqs, powers,
                                MC=None, analyze: bool=True,
                                close_fig: bool=True):
        raise NotImplementedError()

    def measure_transients(self, MC=None, analyze: bool=True,
                           cases=('off', 'on'),
                           prepare: bool=True, depletion_analysis: bool=True,
                           depletion_analysis_plot: bool=True,
                           depletion_optimization_window=None):
        """
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
        """
        if prepare:
            self.prepare_for_timedomain()
        raise NotImplementedError()

    def measure_motzoi(self, motzois=np.linspace(-.3, .3, 31),
                       MC=None, analyze=True, close_fig=True):
        raise NotImplementedError()

    def find_resonators(self, start_freq=6.9e9, stop_freq=7.9e9, VNA_power=-40,
                        bandwidth=200, timeout=200, f_step=250e3, with_VNA=None,
                        verbose=True):
        """
        Performs a wide range scan to find all resonator dips. Will use VNA if
        one is connected and linked to the qubit object, or if specified via
        'with_VNA'.

        Will not do any checks, but rather saves the resonators in the device.
        In the next step (find_resonator_frequency_initial), we will take a look
        whether we have found all resonators and give a warning if not.

        TODO: Add measure_with_VNA to CCL Transmon object
        """
        if with_VNA is None:
            try:
                if self.instr_VNA.get_instr() == '':
                    with_VNA = False
                else:
                    with_VNA = True

            except:
                with_VNA = False

        if with_VNA:
            raise NotImplementedError
        else:
            self.ro_pulse_amp(0.08)
            self.ro_pulse_amp_CW(0.06)
            self.ro_acq_averages(2**10)
            self.ro_soft_avg(1)
            freqs = np.arange(start_freq, stop_freq + f_step, f_step)
            self.measure_heterodyne_spectroscopy(freqs=freqs, analyze=False)
            result = ma2.sa.Initial_Resonator_Scan_Analysis()

        # Create resonator list
        found_resonators = []
        for i, freq in enumerate(result.peaks):
            found_resonators.append(resonator(identifier=i, freq=freq))

        if verbose:
            print('Found resonators:')
            for res in found_resonators:
                freq, unit = plt_tools.SI_val_to_msg_str(res.freq, 'Hz', float)
                print('{}:\t{:.3f} {}'.format(res.identifier, freq, unit))

        try:
            device = self.instr_device.get_instr()
        except AttributeError:
            logging.warning('Could not update device resonators: No device '
                            'found for {}. Returning list of resonators.'
                            .format(self.name))
            return found_resonators

        # Try to find a resonator list:
        if not hasattr(device, 'expected_resonators'):
            device.found_resonators = found_resonators
            logging.warning('No resonators specified for this device')
            device.expected_resonators = []
            return True
        else:
            if device.expected_resonators:
                print('Expected resonators:')
                for res in device.expected_resonators:
                    freq, unit = plt_tools.SI_val_to_msg_str(res.freq, 'Hz',
                                                             float)
                    print('{}:\t{:.3f} {}'.format(res.identifier, freq, unit))
            else:
                logging.warning('No resonators specified for this device')
                return True

        # device.resonators = list(np.repeat(device.resonators,2))
        if len(found_resonators) > len(device.resonators):
            logging.warning('More resonators found than expected. Checking for '
                            'duplicates in next node')

        elif len(found_resonators) < len(device.resonators):
            num_missing = len(device.resonators) - len(found_resonators)
            logging.warning('Missing {} resonator(s). Checking which are '
                            'missing ...'.format(num_missing))
            deltas = []
            for i, found_res in enumerate(found_resonators):
                deltas.append(found_res.freq - device.resonators[i].freq)

            expected_freqs = []
            for res in device.resonators:
                expected_freqs.append(res.freq)

            expected_spacing = np.diff(expected_freqs)
            device.expected_spacing = expected_spacing
            found_spacing = np.diff(result.peaks)

            missing_idx = []
            res_idx = 0
            for i in range(len(found_spacing)):
                if np.abs(found_spacing[i] - expected_spacing[res_idx]) > 25e6:
                    missing_idx.append(i+1)
                    res_idx += 1
                res_idx += 1

            missing_resonators = [device.resonators[ind] for ind in missing_idx]

            print('Missing resonators:')
            for missing_res in missing_resonators:
                print(missing_res.identifier)
                missing_res.type = 'missing'
                device.missing_resonators = missing_resonators
            print('Will look for missing resonators in next node')
        else:
            print('Found all expected resonators.')
            for found_res, res in zip(found_resonators, device.resonators):
                res.freq = found_res.freq

        return True

    def find_resonator_frequency_initial(self, start_freq=6.9e9, stop_freq=7.9e9,
                                         npts=50001, use_min=False, MC=None,
                                         update=True, with_VNA=None,
                                         resonators=None, look_for_missing=True):
        """
        DISCLAIMER: designed for automation routines, seperate usage not
        adviced.

        First checks whether the number of found resonators from a wide scan
        matches the number of expected resonators as specified in the device
        object.
        If it matches, will skip this step no usefull information will be
        obtained besides the frequency, which is already known.

        If there are too many, it will do a resonator scan for each one and
        check whether they are resonators or not.

        If there aree too few, it will try to find the missing ones by looking
        at the spacing and expected spacing of the resonators, predict the
        frequency of the missing resonator and perform a high resolution scan
        to try and find it.
        """
        if with_VNA is None:
            try:
                if self.instr_VNA.get_instr() == '':
                    with_VNA = False
                else:
                    with_VNA = True
            except:
                with_VNA = False

        if resonators is None:
            try:
                device = self.instr_device.get_instr()

            except AttributeError:
                logging.warning('Could not find device resonator dictionary: '
                                'No device found for {}.'.format(self.name))
                return False

        # expected_resonators = list(np.repeat(device.expected_resonators,2))
        expected_resonators = device.expected_resonators
        found_resonators = device.found_resonators

        # Check if any resonators are expected:
        if expected_resonators:
            if len(found_resonators) == len(expected_resonators):
                print('Found all expected resonators.')
                for found_res, res in zip(found_resonators, expected_resonators):
                    res.freq = found_res.freq
                return True

            elif len(found_resonators) > len(expected_resonators):
                logging.warning('More resonators found than expected. '
                                'Checking each candidate at high resolution.')
                new_res = self.measure_individual_resonators(with_VNA=with_VNA)

                if len(new_res) == len(expected_resonators):
                    return True
                elif len(new_res) > len(expected_resonators):
                    logging.warning('Not all false positives removed. '
                                    'Retrying ...')
                    return False

            elif len(found_resonators) < len(expected_resonators):
                num_missing = len(device.resonators) - len(found_resonators)
                logging.warning('Missing {} resonator(s). Checking which are '
                                'missing ...'.format(num_missing))

                # Find missing resonators
                if look_for_missing:
                    raise NotImplementedError
                else:
                    return True
        else:
            print('Scanning all found resonators')
            new_res = self.measure_individual_resonators(with_VNA=with_VNA)
            device.resonators = new_res
            return True


        # # First check if number of resonators matches prediction, else try to
        # # find and remove duplicates
        # if len(device.found_resonators) == len(device.resonators):
        #     return True

        # elif len(device.found_resonators) > len(device.resonators):
        #     result = self.find_additional_resonators(device.resonators,
        #                                              found_resonators,
        #                                              with_VNA=with_VNA)
        #     return result

        # else:
        #     if not look_for_missing:
        #         for res in resonators:
        #             if res.type == 'missing':
        #                 res.type = 'broken'
        #     else:
        #         for i, res in enumerate(device.resonators):
        #             if res.type == 'missing':
        #                 f_step = 50e3
        #                 f_span = 100e6
        #                 f_center = (device.resonators[i+1].freq -
        #                             device.expected_spacing[i])
        #                 freqs = np.arange(f_center - f_span/2,
        #                                   f_center + f_span/2,
        #                                   f_step)

        #                 self.measure_heterodyne_spectroscopy(freqs=freqs,
        #                                                      analyze=False)
        #                 name = 'Resonator'
        #                 a = ma.Homodyne_Analysis(label=name, qb_name=self.name)
        #                 dip = np.amin(a.data_y)
        #                 offset = a.fit_results.params['A'].value

        #                 if (np.abs(dip/offset) > 0.6 or
        #                     np.isnan(a.fit_results.params['Qc'].stderr)):
        #                     freq, unit = plt_tools.SI_val_to_msg_str(f_center,
        #                                                              'Hz',
        #                                                              float)
        #                     print('No resonator found where {} ({:.3f} {}}) is '
        #                           'expected'.format(res.identifier, freq, unit))
        #                     res.type = 'broken'
        #                 else:
        #                     res.type = 'unknown'
        #                     if use_min:
        #                         res.freq = a.min_frequency
        #                     else:
        #                         res.freq = a.fit_results.params['f0'].value*1e9
        # return True

    def measure_individual_resonators(self, with_VNA=False, use_min=False):
        """
        Specifically designed for use in automation, not recommended to use by
        hand!
        Finds which peaks were wrongly assigend as a resonator in the resonator
        wide search
        """
        device = self.instr_device.get_instr()
        found_resonators = device.found_resonators

        new_resonators = []
        for i, res in enumerate(found_resonators):
            freq = res.freq
            str_freq, unit = plt_tools.SI_val_to_msg_str(freq, 'Hz', float)
            if with_VNA:
                raise NotImplementedError
            else:
                old_avger=self.ro_acq_averages()
                self.ro_acq_averages(2**14)
                self.ro_pulse_amp(0.08)
                self.ro_pulse_amp_CW(0.06)
                freqs = np.arange(freq - 5e6, freq + 5e6, 50e3)
                label = '_{:.3f}_{}'.format(str_freq, unit)
                name = 'Resonator_scan' + self.msmt_suffix + label
                self.measure_heterodyne_spectroscopy(freqs=freqs,
                                                     analyze=False,
                                                     label=label)

            a = ma.Homodyne_Analysis(label=name, qb_name=self.name)
            self.ro_acq_averages(old_avger)

            dip = np.amin(a.data_y)
            offset = a.fit_results.params['A'].value

            if ((np.abs(dip/offset) > 0.7) or getattr(a.fit_results.params['f0'],'stderr', None) is None):

                print('Removed candidate {} ({:.3f} {}): Not a resonator'
                      .format(res.identifier, str_freq, unit))

            else:
                if use_min:
                    f_res = a.min_frequency
                else:
                    f_res = a.fit_results.params['f0'].value*1e9

                # Check if not a duplicate
                if i > 0:
                    prev_freq = found_resonators[i-1].freq
                    if np.abs(prev_freq - f_res) < 10e6:
                        print('Removed candidate: {} ({:.3f} {}): Duplicate'
                              .format(res.identifier, str_freq, unit))
                    else:
                        found_resonators[i].freq = f_res
                        print("Added resonator {} ({:.3f} {})"
                              .format(res.identifier, str_freq, unit))
                        new_resonators.append(res)

                else:
                    found_resonators[i].freq = f_res
                    print("Added resonator {} ({:.3f} {})"
                          .format(res.identifier, str_freq, unit))
                    new_resonators.append(res)
        return new_resonators

    def find_test_resonators(self, with_VNA=None, resonators=None):
        """
        Does a power sweep over the resonators to see if they have a qubit
        attached or not, and changes the state in the resonator object
        """
        if with_VNA is None:
            try:
                if self.instr_VNA.get_instr() == '':
                    with_VNA = False
                else:
                    with_VNA = True
            except:
                with_VNA = False

        if resonators is None:
            try:
                device = self.instr_device.get_instr()
            except AttributeError:
                logging.warning('Could not find device resonators: '
                                'No device found for {}'.format(self.name))
                return False
            resonators = self.instr_device.get_instr().resonators

        for res in device.resonators:

            freq = res.freq
            label = '_resonator_{}'.format(res.identifier)
            if res.type == 'test_resonator':
                powers = np.linspace(-20, 0.1, 3)
                f_step = 25e3
            else:
                powers = np.arange(-40, 0.1, 10)
                f_step = 25e3

            if with_VNA:
                VNA = self.instr_VNA.get_instr()
                VNA.start_frequency(freq - 20e6)
                VNA.stop_frequency(freq + 20e6)
                self.measure_VNA_power_sweep()  # not implemented yet
            else:
                if res.type == 'test_resonator':
                    logging.warning('Heterodyne spectroscopy insufficient for '
                                    'test resonators. Skipping')
                    res.freq_low = res.freq
                    continue
                freqs = np.arange(freq - 8e6, freq + 4e6, f_step)
                self.measure_resonator_power(freqs=freqs, powers=powers,
                                             analyze=False, label=label)

            fit_res = ma.Resonator_Powerscan_Analysis_test(label='Resonator_power_scan',
                                                      close_fig=True,
                                                      use_min=True)
            # Update resonator types
            if np.abs(fit_res.shift) > 200e3:
                if res.type == 'unknown':
                    res.type = 'qubit_resonator'
                elif res.type == 'qubit_resonator':
                    print('Resonator {}: confirmed resonator shift.'
                          .format(res.identifier))
                else:
                    logging.warning('No resonator power shift found for '
                                    'resonator {}. Consider adding/removing '
                                    'attenuation.'.format(res.identifier))
            else:
                if res.type == 'unknown':
                    res.type = 'test_resonator'
                elif res.type == 'test_resonator':
                    print('Resonator {}: confirmed test resonator'
                          .format(res.identifier))
                    res.freq_low = res.freq
                else:
                    logging.warning('Resonator shift found for test resonator '
                                    '{}. Apperently not a test resonator.'
                                    .format(res.identifier))

            # Update resonator attributes
            res.freq_low = fit_res.f_low
            res.freq_high = fit_res.f_high
            res.shift = fit_res.shift
            res.ro_amp = 10**(fit_res.power/20)

        return True

    def find_test_resonators_test(self, with_VNA=None, resonators=None):
        """
        Does a power sweep over the resonators to see if they have a qubit
        attached or not, and changes the state in the resonator object
        """
        if with_VNA is None:
            try:
                if self.instr_VNA.get_instr() == '':
                    with_VNA = False
                else:
                    with_VNA = True
            except:
                with_VNA = False

        if resonators is None:
            try:
                device = self.instr_device.get_instr()
            except AttributeError:
                logging.warning('Could not find device resonators: '
                                'No device found for {}'.format(self.name))
                return False
            resonators = self.instr_device.get_instr().resonators

        for res in device.resonators:

            freq = res.freq
            label = '_resonator_{}'.format(res.identifier)
            if res.type == 'test_resonator':
                powers = np.linspace(-20, 0.1, 3)
                f_step = 25e3
            else:
                powers = np.arange(-40, 0.1, 10)
                f_step = 25e3

            if with_VNA:
                VNA = self.instr_VNA.get_instr()
                VNA.start_frequency(freq - 20e6)
                VNA.stop_frequency(freq + 20e6)
                self.measure_VNA_power_sweep()  # not implemented yet
            else:
                if res.type == 'test_resonator':
                    logging.warning('Heterodyne spectroscopy insufficient for '
                                    'test resonators. Skipping')
                    res.freq_low = res.freq
                    continue
                freqs = np.arange(freq - 6e6, freq + 3e6, f_step)
                self.measure_resonator_power(freqs=freqs, powers=powers,
                                             analyze=False, label=label)

            fit_res = ma.Resonator_Powerscan_Analysis_test(label='Resonator_power_scan',
                                                      close_fig=True,
                                                      use_min=True)

            # Update resonator types
            shift = np.max(np.array([fit_res.shift1, fit_res.shift2]))

            if np.abs(shift) > 200e3:
                if res.type == 'unknown':
                    res.type = 'qubit_resonator'
                elif res.type == 'qubit_resonator':
                    print('Resonator {}: confirmed resonator shift.'
                          .format(res.identifier))
                else:
                    logging.warning('No resonator power shift found for '
                                    'resonator {}. Consider adding/removing '
                                    'attenuation.'.format(res.identifier))
            else:
                if res.type == 'unknown':
                    res.type = 'test_resonator'
                elif res.type == 'test_resonator':
                    print('Resonator {}: confirmed test resonator'
                          .format(res.identifier))
                    res.freq_low = res.freq
                else:
                    logging.warning('Resonator shift found for test resonator '
                                    '{}. Apperently not a test resonator.'
                                    .format(res.identifier))

            # Update resonator attributes
            index_f = np.argmax(np.array([fit_res.shift1, fit_res.shift2]))
            fit_res.f_low = np.array([fit_res.f_low1, fit_res.f_low2])
            res.freq_low = fit_res.f_low[index_f]
            fit_res.f_high = np.array([fit_res.f_high1, fit_res.f_high2])
            res.freq_high = fit_res.f_high[index_f]
            # res.freq_low = fit_res.f_low
            # res.freq_high = fit_res.f_high
            res.shift = shift
            res.ro_amp = 10**(fit_res.power/20)

        return True

    def find_qubit_resonator_fluxline(self, with_VNA=None, dac_values=None,
                                      verbose=True, resonators=None):
        """
        --- WARNING: UPDATING PARAMETERS ONLY WORKS WITH DEVICE OBJECT! ---

        Does a resonator DAC scan with all qubit resonators and all fluxlines.
        """
        if with_VNA is None:
            try:
                if self.instr_VNA.get_instr() == '':
                    with_VNA = False
                else:
                    with_VNA = True
            except:
                with_VNA = False

        if resonators is None:
            try:
                device = self.instr_device.get_instr()
            except AttributeuxError:
                logging.warning('Could not find device resonators: '
                                'No device found for {}.'.format(self.name))
                return False
            resonators = device.resonators

        if dac_values is None:
            dac_values = np.arange(-10e-3, 10e-3, 1e-3)

        fluxcurrent = self.instr_FluxCtrl.get_instr()
        for FBL in fluxcurrent.channel_map:
            fluxcurrent[FBL](0)

        for res in resonators:
            if res.type == 'qubit_resonator':
                self.ro_pulse_amp(res.ro_amp)
                self.ro_pulse_amp_CW(res.ro_amp)
                best_amplitude = 0  # For comparing which one is coupled closest

                if with_VNA:
                    VNA = self.instr_VNA.get_instr()
                    VNA.start_frequency(res.freq_low - 10e6)
                    VNA.stop_frequency(res.freq_low + 10e6)

                freqs = np.arange(res.freq_low - np.abs(res.shift) - 15e6,
                                  res.freq_low + 4e6,
                                  0.2e6)
                for fluxline in fluxcurrent.channel_map:
                    label = '_resonator_{}_{}'.format(res.identifier, fluxline)
                    t_start = time.strftime('%Y%m%d_%H%M%S')

                    self.measure_resonator_frequency_dac_scan(freqs=freqs,
                                                              dac_values=dac_values,
                                                              fluxChan=fluxline,
                                                              analyze=False,
                                                              label=label)
                    fluxcurrent[fluxline](0)
                    str_freq, unit = plt_tools.SI_val_to_msg_str(res.freq, 'Hz',
                                                                 float)
                    print('Finished flux sweep resonator {} ({:.3f} {}) with {}'
                          .format(res.identifier, str_freq, unit, fluxline))
                    timestamp = a_tools.get_timestamps_in_range(t_start,
                                                                label=self.msmt_suffix)[0]

                    fit_res = ma2.VNA_DAC_Analysis(timestamp)

                    amplitude = fit_res.dac_fit_res.params['amplitude'].value
                    if amplitude > best_amplitude:
                        best_amplitude = amplitude
                        res.qubit = fluxline.split('_', 1)[-1]
                        res.sweetspot = fit_res.sweet_spot_value
                        res.fl_dc_I_per_phi0 = fit_res.current_to_flux

        if verbose:
            for res in self.instr_device.get_instr().resonators:
                if res.type == 'qubit_resonator':
                    freq, unit = plt_tools.SI_val_to_msg_str(res.freq_low,
                                                             'Hz',
                                                             float)
                    print('{}, f = {:.3f} {}, linked to {},'
                          ' sweetspot current = {:.3f} mA'
                          .format(res.type, freq, unit, res.qubit, res.sweetspot*1e3))
                else:
                    freq, unit = plt_tools.SI_val_to_msg_str(res.freq,
                                                             'Hz',
                                                             float)
                    print('{}, f = {:.3f} {}'.format(res.type, freq, unit))

        # Set properties for all qubits in device if device exists
        device = self.instr_device.get_instr()
        assigned_qubits = []
        for q in device.qubits():
            if q == 'fakequbit':
                pass
            qubit = device.find_instrument(q)

            for res in device.resonators:
                if qubit.name == res.qubit:
                    if qubit.name in assigned_qubits:
                        logging.warning('Multiple resonators found for {}. '
                                        'Aborting'.format(qubit.name))
                        return False
                    assigned_qubits.append(qubit.name)
                    qubit.freq_res(res.freq_low)
                    qubit.ro_freq(res.freq_low)
                    qubit.fl_dc_I0(res.sweetspot)
                    qubit.fl_dc_I_per_phi0(res.fl_dc_I_per_phi0)
                    qubit.fl_dc_ch('FBL_' + res.qubit)
                    if qubit.freq_qubit() is None:
                        qubit.freq_qubit(res.freq_low -
                                         np.abs((70e6)**2/(res.shift)))
        return True

    def find_resonator_sweetspot(self, freqs=None, dac_values=None,
                                 fluxChan=None, update=True):
        """
        Finds the resonator sweetspot current.
        TODO: - measure all FBL-resonator combinations
        TODO: - implement way of distinguishing which fluxline is most coupled
        TODO: - create method that moves qubits away from sweetspot when they
                are not being measured (should not move them to some other
                qubit frequency of course)
        """
        if freqs is None:
            freq_center = self.freq_res()
            freq_range = 20e6
            freqs = np.arange(freq_center - freq_range/2,
                              freq_center + freq_range/2, 0.5e6)

        if dac_values is None:
            dac_values = np.linspace(-10e-3, 10e-3, 101)

        if fluxChan is None:
            if self.fl_dc_ch() == 1:  # Initial value
                fluxChan = 'FBL_1'
            else:
                fluxChan = self.fl_dc_ch()

        t_start = time.strftime('%Y%m%d_%H%M%S')
        self.measure_resonator_frequency_dac_scan(freqs=freqs,
                                                  dac_values=dac_values,
                                                  fluxChan=fluxChan,
                                                  analyze=False)
        if update:

            import pycqed.analysis_v2.spectroscopy_analysis as sa
            timestamp = ma.a_tools.get_timestamps_in_range(t_start,label = 'Resonator')[0]
            fit_res = sa.VNA_DAC_Analysis(timestamp=timestamp)
            sweetspot_current = fit_res.sweet_spot_value
            self.fl_dc_I0(sweetspot_current)
            fluxcurrent = self.instr_FluxCtrl.get_instr()
            fluxcurrent[self.fl_dc_ch()](sweetspot_current)

        return True

    def find_resonator_frequency(self, use_min=True,
                                 update=True,
                                 freqs=None,
                                 MC=None, close_fig=True):
        """
        Performs heterodyne spectroscopy to identify the frequecy of the (readout)
        resonator frequency.

        Args:
            use_min (bool):
                'True' uses the frequency at minimum amplitude. 'False' uses
                the fit result

            update (bool):
                update the internal parameters with this fit
                Finds the resonator frequency by performing a heterodyne experiment
                if freqs == None it will determine a default range dependent on the
                last known frequency of the resonator.

            freqs (array):
                list of frequencies to sweep. By default set to +-5 MHz around
                the last recorded frequency, with 100 kHz step
        """

        # This snippet exists to be backwards compatible 9/2017.
        try:
            freq_res_par = self.freq_res
            freq_RO_par = self.ro_freq
        except:
            warnings.warn("Deprecation warning: rename f_res to freq_res")
            freq_res_par = self.f_res
            freq_RO_par = self.f_RO

        old_avg = self.ro_acq_averages()
        self.ro_acq_averages(2**14)

        if freqs is None:
            f_center = freq_res_par()
            if f_center is None:
                raise ValueError('Specify "freq_res" to generate a freq span')
            f_span = 10e6
            f_step = 100e3
            freqs = np.arange(f_center-f_span/2, f_center+f_span/2, f_step)
        self.measure_heterodyne_spectroscopy(freqs, MC, analyze=False)
        a = ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)

        self.ro_acq_averages(old_avg)

        if use_min:
            f_res = a.min_frequency
        else:
            f_res = a.fit_results.params['f0'].value*1e9  # fit converts to Hz
        if f_res > max(freqs) or f_res < min(freqs):
            logging.warning('extracted frequency outside of range of scan')
        elif update:  # don't update if the value is out of the scan range
            freq_res_par(f_res)
            freq_RO_par(f_res)
        return f_res

    def find_frequency(self, method='spectroscopy', spec_mode='pulsed_marked',
                       steps=[1, 3, 10, 30, 100, 300, 1000],
                       artificial_periods=4,
                       freqs=None,
                       f_span=100e6,
                       use_max=False,
                       f_step=1e6,
                       verbose=True,
                       update=True,
                       close_fig=True,
                       MC=None,
                       label = ''):
        """
        Finds the qubit frequency using either the spectroscopy or the Ramsey
        method.

        In case method=='spectroscopy' this routine runs measure_spectroscopy and performs
        analysis looking for peaks in the spectrum.

        In case metgod=='ramsey' this routine performs series ofamsey ramsey measurements
        for increasing range of the delay times. Using short ramsey sequence with relatively
        large artificial detuning yields robust measurement of the qubit frequency, and increasing
        the relay times allows for more precise frequency measurement.

        Args:
            method (str {'spectroscopy', 'ramsey'}):
                specifies whether to perform spectroscopy ('spectroscopy') or series of
                ramsey measurements ('ramsey') to find the qubit frequency.

            spec_mode (str {'CW', 'pulsed_marked', 'pulsed_mixer'}):
                specifies the mode of the spectroscopy measurements (currently only implemented
                by Timo for CCL_Transmon). Possivle values: 'CW', 'pulsed_marked', 'pulsed_mixer'

            steps (array):
                maximum delay between pi/2 pulses (in microseconds) in a subsequent ramsey measurements.
                The find_frequency routine is terminated when all steps are performed or if
                the fitted T2* significantly exceeds the maximum delay

            artificial_periods (float):
                specifies the automatic choice of the artificial detuning in the ramsey
                measurements, in such a way that ramsey measurement should show 4 full oscillations.

            freqs (array):
                list of sweeped frequencies in case of spectroscopy measurement

            f_span (float):
                span of sweeped frequencies around the currently recorded qubit frequency in
                the spectroscopy measurement

            f_step (flaot):
                increment of frequency between data points in spectroscopy measurement

            update (bool):
                boolean indicating whether to update the qubit frequency in the qubit object
                according to the result of the measurement
        """
        if method.lower() == 'spectroscopy':
            if freqs is None:
                f_qubit_estimate = self.calculate_frequency()
                freqs = np.arange(f_qubit_estimate - f_span/2,
                                  f_qubit_estimate + f_span/2,
                                  f_step)
            # args here should be handed down from the top.
            self.measure_spectroscopy(freqs, mode=spec_mode, MC=MC,
                                      analyze=False, label = label,
                                      close_fig=close_fig)

            label = 'spec'
            analysis_spec = ma.Qubit_Spectroscopy_Analysis(
                label=label, close_fig=True, qb_name=self.name)

            # Checks to see if there is a peak:
            freq_peak = analysis_spec.peaks['peak']
            offset = analysis_spec.fit_res.params['offset'].value
            peak_height = np.amax(analysis_spec.data_dist)

            if freq_peak is None:
                success = False
            elif peak_height < 3*offset:
                success = False
            elif peak_height < 3*np.mean(analysis_spec.data_dist):
                success = False
            else:
                success = True

            if success:
                if update:
                    if use_max:
                        self.freq_qubit(analysis_spec.peaks['peak'])
                    else:
                        self.freq_qubit(analysis_spec.fitted_freq)
                    return True
                    # TODO: add updating and fitting
            else:
                logging.warning('No peak found! Not updating.')
                return False

        elif method.lower() == 'ramsey':
            return self.calibrate_frequency_ramsey(
                steps=steps, artificial_periods=artificial_periods,
                verbose=verbose, update=update,
                close_fig=close_fig)
        return analysis_spec.fitted_freq

    def calibrate_spec_pow(self, freqs=None, start_power=-35, power_step = 5,
                           threshold=0.5, verbose=True):
        """
        Finds the optimal spectroscopy power for qubit spectroscopy (not pulsed)
        by varying it in steps of 5 dBm, and ending when the peak has power
        broadened by 1+threshold (default: broadening of 10%)
        """
        old_avg = self.ro_acq_averages()
        self.ro_acq_averages(2**15)
        if freqs is None:
            freqs = np.arange(self.freq_qubit() - 20e6,
                              self.freq_qubit() + 20e6, 0.2e6)
        power = start_power

        w0, w = 1e6,0.7e6

        while w < (1 + threshold) * w0:
            self.spec_pow(power)
            self.measure_spectroscopy(freqs=freqs, analyze=False,
                                      label='spec_pow_' + str(power) + '_dBm')

            a = ma.Qubit_Spectroscopy_Analysis(label=self.msmt_suffix,
                                               qb_name=self.name)

            freq_peak = a.peaks['peak']
            if np.abs(freq_peak - self.freq_qubit()) > 10e6:
                logging.warning('Peak has shifted for some reason. Aborting.')
                return False
            w = a.fit_res.params['kappa'].value
            power += power_step


            if w < w0:
                w0 = w
        self.ro_acq_averages(old_avg)
        if verbose:
            print('setting spectroscopy power to {}'.format(power-5))
        self.spec_pow(power-power_step)
        return True


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

    def calibrate_optimal_weights(self, MC=None, verify: bool=True,
                                  analyze: bool=True, update: bool=True,
                                  no_figs: bool=False)->bool:
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

    def calibrate_frequency_ramsey(self,
                                   steps=[1, 1, 3, 10, 30, 100, 300, 1000],
                                   artificial_periods = 2.5,
                                   stepsize:float =20e-9,
                                   verbose: bool=True, update: bool=True,
                                   close_fig: bool=True,
                                   test_beating: bool=True):
        """
        Runs an iterative procudere of ramsey experiments to estimate
        frequency detuning to converge to the qubit frequency up to the limit
        set by T2*.

        Args:
            steps (array):
                multiples of the initial stepsize on which to run the

            artificial_periods (float):
                intended number of periods in theramsey measurement, used to adjust
                the artificial detuning

            stepsize (float):
                smalles stepsize in ns for which to run ramsey experiments.
        """

        self.ro_acq_averages(2**10)
        cur_freq = self.freq_qubit()
        # Steps don't double to be more robust against aliasing
        for n in steps:
            times = np.arange(self.mw_gauss_width()*4,
                              50*n*stepsize, n*stepsize)
            artificial_detuning = artificial_periods/times[-1]
            self.measure_ramsey(times,
                                artificial_detuning=artificial_detuning,
                                freq_qubit=cur_freq,
                                label='_{}pulse_sep'.format(n),
                                analyze=False)
            a = ma.Ramsey_Analysis(auto=True, close_fig=close_fig,
                                   freq_qubit=cur_freq,
                                   artificial_detuning=artificial_detuning,
                                   close_file=False)
            if test_beating and a.fit_res.chisqr > 0.4:
                logging.warning('Found double frequency in Ramsey: large '
                                'deviation found in single frequency fit.'
                                'Returning True to continue automation. Retry '
                                'with test_beating=False to ignore.')

                return True
            fitted_freq = a.fit_res.params['frequency'].value
            measured_detuning = fitted_freq-artificial_detuning
            cur_freq = a.qubit_frequency

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
            if times[-1] > 2.*a.T2_star['T2_star']:
                # If the last step is > T2* then the next will be for sure
                if verbose:
                    print('Breaking of measurement because of T2*')
                break
        if verbose:
            print('Converged to: {:.9e}'.format(cur_freq))
        if update:
            self.freq_qubit(cur_freq)
        return cur_freq

    def calculate_frequency(self, calc_method=None, I_per_phi0=None, I=None):
        """
        Calculates an estimate for the qubit frequency.
        Arguments are optional and parameters of the object are used if not
        specified.

        Args:
            calc_method (str {'latest', 'flux'}):
                can be "latest" or "flux" uses last known frequency
                or calculates using the cosine arc model as specified
                in fit_mods.Qubit_dac_to_freq
                corresponding par. : cfg_qubit_freq_calc_method

            I_per_phi0 (float):
                dac flux coefficient, converts volts to Flux.
                Set to 1 to reduce the model to pure flux.
                corresponding par. : fl_dc_I_per_phi0)

            I (float):
                dac value used when calculating frequency
                corresponding par. : fl_dc_I

        Calculates the f01 transition frequency using the cosine arc model.
        (function available in fit_mods. Qubit_dac_to_freq)

        The parameter cfg_qubit_freq_calc_method determines how it is
        calculated.
        Parameters of the qubit object are used unless specified.
        Flux can be specified both in terms of dac voltage or flux but not
        both.
        """
        if self.cfg_qubit_freq_calc_method() == 'latest':
            qubit_freq_est = self.freq_qubit()

        elif self.cfg_qubit_freq_calc_method() == 'flux':

            if I is None:
                I = self.fl_dc_I()
            if I_per_phi0 is None:
                I_per_phi0 = self.fl_dc_I_per_phi0()

            qubit_freq_est = fit_mods.Qubit_dac_to_freq(
                dac_voltage=I,
                f_max=self.freq_max(),
                E_c=self.E_c(),
                dac_sweet_spot=self.fl_dc_I0(),
                V_per_phi0=I_per_phi0, # legacy naming in fit_mods function
                asymmetry=self.asymmetry())

        return qubit_freq_est

    def calibrate_mixer_offsets_drive(self, update: bool=True)-> bool:
        """
        Calibrates the mixer skewness and updates the I and Q offsets in
        the qubit object.
        """
        raise NotImplementedError()

        return True



    def tune_freq_to_sweetspot(self, freqs=None, dac_values=None, verbose=True,
                               fit_phase=False, use_dips=False):
        """
        Tunes the qubit to the sweetspot
        """

        within_50MHz_of_sweetspot = True

        # if within 50 MHz of sweetspot, we can start the iterative procedure
        if within_50MHz_of_sweetspot:
            pass

        # Requires an estimate of I_per_phi0 (which should be a current)
        if freqs is None:
            freqs = self.freq_max() + np.arange(-80e6, +20e6, .5e6)

        # Should be replaced by self.fl_dc_I() # which gets this automatically
        # self.fl_dc_I()
        fluxcontrol = self.instr_FluxCtrl.get_instr()
        current_dac_val = fluxcontrol.parameters[(self.fl_dc_ch())].get()

        # Should correspond to approx 50MHz around sweetspot.
        dac_range  = 0.1 * self.fl_dc_I_per_phi0()
        if dac_values is None:
            dac_values = current_dac_val + np.linspace(-dac_range/2, dac_range/2, 6)

        self.measure_qubit_frequency_dac_scan(freqs=freqs, dac_values=dac_values)

        analysis_obj = ma.TwoD_Analysis(label='Qubit_dac_scan', close_fig=True)
        freqs = analysis_obj.sweep_points
        dac_vals = analysis_obj.sweep_points_2D
        if fit_phase:
            signal_magn = analysis_obj.measured_values[1]
        else:
            signal_magn = analysis_obj.measured_values[0]
            if use_dips:
                signal_magn = -signal_magn


        # FIXME: This function should be moved out of the qubit object upon cleanup.
        def quick_analyze_dac_scan(x_vals, y_vals, Z_vals):
            def find_peaks(x_vals, y_vals, Z_vals):
                peaks = np.zeros(len(y_vals))
                for i in range(len(y_vals)):
                    p_dict = a_tools.peak_finder(x_vals, Z_vals[:, i],
                        optimize=False, num_sigma_threshold=15)
                        # FIXME hardcoded num_sigma_threshold
                    try:
                        peaks[i] = p_dict['peak']
                    except Exception as e:
                        logging.warning(e)
                        peaks[i] = np.NaN

                return peaks

            peaks = find_peaks(x_vals, y_vals, Z_vals)

            dac_masked=  y_vals[~np.isnan(peaks)]
            peaks_masked= peaks[~np.isnan(peaks)]
            pv = np.polyfit(x=dac_masked, y=peaks_masked, deg=2)
            sweetspot_current = -0.5*pv[1]/pv[0]
            sweetspot_freq = np.polyval(pv,sweetspot_current)
            return sweetspot_current, sweetspot_freq


        dac_sweetspot, freq_sweetspot = quick_analyze_dac_scan(
            x_vals=freqs, y_vals=dac_vals, Z_vals=signal_magn)

        if dac_sweetspot>np.max(dac_values) or dac_sweetspot<np.min(dac_values):
            warnings.warn("Fit returns something weird. Not updating flux bias")
            procedure_success = False
        elif freq_sweetspot > self.freq_max()+50e6:
            warnings.warn("Fit returns something weird. Not updating flux bias")
            procedure_success = False
        elif freq_sweetspot < self.freq_max()-50e6:
            warnings.warn("Fit returns something weird. Not updating flux bias")
            procedure_success = False
        else:
            procedure_success = True
        if not procedure_success:
            # reset the current to the last known value.
            fluxcontrol.parameters[(self.fl_dc_ch())].set(current_dac_val)

        if verbose:
            # FIXME replace by unit aware printing
            print("Setting flux bias to {:.3f} mA".format(dac_sweetspot*1e3))
            print("Setting qubit frequency to {:.4f} GHz".format(freq_sweetspot*1e-9))

        # self.fl_dc_I(dac_sweetspot)
        # FIXME, this should be included in the set of fl_dc_I
        fluxcontrol.parameters[(self.fl_dc_ch())].set(dac_sweetspot)
        self.freq_qubit(freq_sweetspot)
        self.fl_dc_I(dac_sweetspot)
        self.fl_dc_I0(dac_sweetspot)

    def tune_freq_to(self,
                     target_frequency,
                     MC=None,
                     nested_MC=None,
                     calculate_initial_step: bool = False,
                     initial_flux_step: float = None,
                     max_repetitions=15,
                     resonator_use_min=True,
                     find_res=None):
        """
        Iteratively tune the qubit frequency to a specific target frequency
        """
        if MC is None:
            MC = self.instr_MC.get_instr()
        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        # Check if target frequency is within range
        if target_frequency > self.freq_max():
            raise ValueError('Attempting to tune to a frequency ({:.2f} GHz)'
                             'larger than the sweetspot frequency ({:.2f} GHz)'
                             .format(target_frequency, self.freq_max()))

        # Current frequency
        f_q = self.freq_qubit()
        delta_freq = target_frequency - f_q

        # User may overwrite need to find resonator
        if abs(delta_freq) > 10e6 and find_res is None:
            find_res = True

        fluxcontrol = self.instr_FluxCtrl.get_instr()
        fluxpar = fluxcontrol.parameters[(self.fl_dc_ch())]

        current_dac_val = fluxpar.get()

        # set up ranges and parameters
        if calculate_initial_step:
            raise NotImplementedError()
        #    construct predicted arch from I_per_phi0, E_c, E_j BALLPARK SHOULD
        #    SUFFICE.
        #    predict first jump
        #    next_dac_value =
        else:
            if initial_flux_step is None:
                # If we do not calculate the initial step, we take small steps
                # from # our starting point
                initial_flux_step = self.fl_dc_I_per_phi0()/30

            next_dac_value = current_dac_val + initial_flux_step

        def measure_qubit_freq_nested(target_frequency, steps=0.2e6,
                                      spans=[100e6, 400e6, 800e6, 1200e6, 1500e6],
                                      **kw):
            # measure freq
            if find_res:
                freq_res = self.find_resonator_frequency(
                                    MC=nested_MC,
                                    use_min=resonator_use_min)
            else:
                freq_res = self.freq_res()

            spec_succes = False
            for span in spans:
                spec_succes = self.find_frequency(f_span=span,  MC=nested_MC)
                if spec_succes:
                    break

            if not spec_succes:
                raise ValueError("Could not find the qubit. Aborting.")
            freq_qubit = self.freq_qubit()  # as updated in this function call

            abs_freq_diff = abs(target_frequency-freq_qubit)

            return {'abs_freq_diff': abs_freq_diff, 'freq_qubit': freq_qubit,
                    'freq_resonator': freq_res}

        qubit_freq_det = det.Function_Detector(measure_qubit_freq_nested,
            msmt_kw={'target_frequency': target_frequency},
            result_keys=['abs_freq_diff', 'freq_qubit', 'freq_resonator'],
            value_units=['Hz']*3)

        from scipy.optimize import minimize_scalar
        ad_func_pars = {'adaptive_function': minimize_scalar,
                        'method': 'brent',
                        'bracket': [current_dac_val, next_dac_value],
                        # 'x0': x0,
                        'tol': 1e-6,  # Relative tolerance in brent
                        'minimize': True,
                        'options': {'maxiter': max_repetitions}}

        MC.set_sweep_function(fluxpar)
        MC.set_detector_function(qubit_freq_det)
        MC.set_adaptive_function_parameters(ad_func_pars)
        MC.run('Tune_to_freq', mode='adaptive')

    def measure_heterodyne_spectroscopy(self, freqs, MC=None,
                                        analyze=True, close_fig=True):
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

    """
    circuit-QED Transmon as used in DiCarlo Lab.
    Adds transmon specific parameters as well
    """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_parameter('E_c', unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('E_j', unit='Hz',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('anharmonicity', unit='Hz',
                           label='Anharmonicity',
                           docstring='Anharmonicity, negative by convention',
                           parameter_class=ManualParameter,
                           # typical target value
                           initial_value=-300e6,
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
        self.add_parameter('I_per_phi0',
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
        """
        Calculates the f01 transition frequency from the cosine arc model.
        (function available in fit_mods. Qubit_dac_to_freq)

        Parameters of the qubit object are used unless specified.
        Flux can be specified both in terms of dac voltage or flux but not
        both.
        """

        if dac_voltage is not None and flux is not None:
            raise ValueError('Specify either dac voltage or flux but not both')

        if self.f_qubit_calc_method() == 'latest':
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
                                   artificial_periods=2.5,
                                   stepsize=None, verbose=True, update=True,
                                   close_fig=True):
        if stepsize is None:
            stepsize = abs(1/self.f_pulse_mod.get())
        cur_freq = self.f_qubit.get()
        # Steps don't double to be more robust against aliasing
        for n in steps:
            times = np.arange(self.pulse_delay.get(),
                              50*n*stepsize, n*stepsize)
            artificial_detuning = artificial_periods/times[-1]
            self.measure_ramsey(times,
                                artificial_detuning=artificial_detuning,
                                f_qubit=cur_freq,
                                label='_{}pulse_sep'.format(n),
                                analyze=False)
            a = ma.Ramsey_Analysis(auto=True, close_fig=close_fig,
                                   qb_name=self.name,
                                   artificial_detuning=artificial_detuning,
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

            if times[-1] > 2.*a.T2_star['T2_star']:
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
                       artificial_periods = 2.5,
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
                label=label, close_fig=True, qb_name=self.name)

            if update:
                if use_max:
                    self.f_qubit(analysis_spec.peaks['peak'])
                else:
                    self.f_qubit(analysis_spec.fitted_freq)
                # TODO: add updating and fitting
        elif method.lower() == 'ramsey':
            return self.calibrate_frequency_ramsey(
                steps=steps, artificial_periods=artificial_periods,
                verbose=verbose, update=update, close_fig=close_fig)
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
            self.amp180.set(ampl)
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
        """
        Finds the pulse-amplitude using a Rabi experiment.
        Fine tunes by doing a Rabi around the optimum with an odd
        multiple of pulses.

        Args:
            amps (array or float):
                amplitudes of the first Rabi if an array,
                if a float is specified it will be treated as an estimate
                for the amplitude to be found.

            N_steps (list of int):
                number of pulses used in the fine tuning

            max_n (int):
                break of if N> max_n
        """
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
            self.amp180.set(np.abs(ampl))



    def find_amp90_scaling(self, scales=0.5,
                           N_steps=[5, 9], max_n=100,
                           close_fig=True, verbose=False,
                           MC=None, update=True, take_fit_I=False):
        """
        Finds the scaling factor of pi/2 pulses w.r.t pi pulses using a rabi
        type with each pi pulse replaced by 2 pi/2 pulses.

        If scales is an array it starts by fitting a cos to a Rabi experiment
        to get an initial guess for the amplitude.

        This experiment is only useful after carefully calibrating the pi pulse
        using flipping sequences.
        """
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
