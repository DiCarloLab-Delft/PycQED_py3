'''
Module containing a collection of detector functions used by the
Measurement Control.
'''
import numpy as np
from copy import deepcopy
import logging
import time
from string import ascii_uppercase
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.fit_toolbox import functions as fn
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control import sequence
from qcodes.instrument.parameter import _BaseParameter
import pycqed.measurement.pulse_sequences.calibration_elements as cal_elts
import logging
log = logging.getLogger(__name__)

class Detector_Function(object):

    '''
    Detector_Function class for MeasurementControl
    '''

    def __init__(self, **kw):
        self.name = self.__class__.__name__
        self.set_kw()
        self.value_names = ['val A', 'val B']
        self.value_units = ['arb. units', 'arb. units']

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def get_values(self):
        pass

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass


class Multi_Detector(Detector_Function):
    """
    Combines several detectors of the same type (hard/soft) into a single
    detector.
    """

    def __init__(self, detectors: list,
                 det_idx_suffix: bool=True, **kw):
        """
        detectors     (list): a list of detectors to combine.
        det_idx_suffix(bool): if True suffixes the value names with
                "_det{idx}" where idx refers to the relevant detector.
        """
        self.detectors = detectors
        self.name = 'Multi_detector'
        self.value_names = []
        self.value_units = []
        for i, detector in enumerate(detectors):
            for detector_value_name in detector.value_names:
                if det_idx_suffix:
                    detector_value_name += '_det{}'.format(i)
                self.value_names.append(detector_value_name)
            for detector_value_unit in detector.value_units:
                self.value_units.append(detector_value_unit)

        self.detector_control = self.detectors[0].detector_control
        for d in self.detectors:
            if d.detector_control != self.detector_control:
                raise ValueError('All detectors should be of the same type')

    def prepare(self, **kw):
        for detector in self.detectors:
            detector.prepare(**kw)

    def get_values(self):
        values_list = []
        for detector in self.detectors:
            new_values = detector.get_values()
            values_list.append(new_values)
        values = np.concatenate(values_list)
        return values

    def acquire_data_point(self):
        # N.B. get_values and acquire_data point are virtually identical.
        # the only reason for their existence is a historical distinction
        # between hard and soft detectors that leads to some confusing data
        # shape related problems, hence the append vs concatenate
        values = []
        for detector in self.detectors:
            new_values = detector.acquire_data_point()
            values = np.append(values, new_values)
        return values

    def finish(self):
        for detector in self.detectors:
            detector.finish()


class IndexDetector(Detector_Function):
    def __init__(self, detector, index):
        super().__init__()
        self.detector = detector
        self.index = index
        self.name = detector.name + '[{}]'.format(index)
        self.value_names = [detector.value_names[index]]
        self.value_units = [detector.value_units[index]]
        self.detector_control = detector.detector_control

    def prepare(self, **kw):
        self.detector.prepare(**kw)

    def get_values(self):
        return self.detector.get_values()[self.index]

    def acquire_data_point(self):
        return self.detector.acquire_data_point()[self.index]

    def finish(self):
        self.detector.finish()

###############################################################################
###############################################################################
####################             None Detector             ####################
###############################################################################
###############################################################################


class None_Detector(Detector_Function):

    def __init__(self, **kw):
        super(None_Detector, self).__init__()
        self.detector_control = 'soft'
        self.set_kw()
        self.name = 'None_Detector'
        self.value_names = ['None']
        self.value_units = ['None']

    def acquire_data_point(self, **kw):
        '''
        Returns something random for testing
        '''
        return np.random.random()


class Hard_Detector(Detector_Function):

    def __init__(self, **kw):
        super().__init__()
        self.detector_control = 'hard'

    def prepare(self, sweep_points=None):
        pass

    def finish(self):
        pass


class Soft_Detector(Detector_Function):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.detector_control = 'soft'

    def acquire_data_point(self, **kw):
        return np.random.random()

    def prepare(self, sweep_points=None):
        pass


##########################################################################
##########################################################################
####################     Hardware Controlled Detectors     ###############
##########################################################################
##########################################################################


class Dummy_Detector_Hard(Hard_Detector):

    def __init__(self, delay=0, noise=0, **kw):
        super(Dummy_Detector_Hard, self).__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = ['distance', 'Power']
        self.value_units = ['m', 'W']
        self.delay = delay
        self.noise = noise
        self.times_called = 0

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points

    def get_values(self):
        x = self.sweep_points
        noise = self.noise * (np.random.rand(2, len(x)) - .5)
        data = np.array([np.sin(x / np.pi),
                         np.cos(x/np.pi)])
        data += noise
        time.sleep(self.delay)
        # Counter used in test suite to test how many times data was acquired.
        self.times_called += 1

        return data

class Dummy_Shots_Detector(Hard_Detector):

    def __init__(self, max_shots=10, **kw):
        super().__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = ['shots']
        self.value_units = ['m']
        self.max_shots = max_shots
        self.times_called = 0

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points

    def get_values(self):
        x = self.sweep_points

        start_idx = self.times_called*self.max_shots % len(x)

        dat = x[start_idx:start_idx+self.max_shots]
        self.times_called += 1
        return dat


class Sweep_pts_detector(Detector_Function):

    """
    Returns the sweep points, used for testing purposes
    """

    def __init__(self, params, chunk_size=80):
        self.detector_control = 'hard'
        self.value_names = []
        self.value_units = []
        self.chunk_size = chunk_size
        self.i = 0
        for par in params:
            self.value_names += [par.name]
            self.value_units += [par.units]

    def prepare(self, sweep_points):
        self.i = 0
        self.sweep_points = sweep_points

    def get_values(self):
        return self.get()

    def acquire_data_point(self):
        return self.get()

    def get(self):
        print('passing chunk {}'.format(self.i))
        start_idx = self.i*self.chunk_size
        end_idx = start_idx + self.chunk_size
        self.i += 1
        time.sleep(.2)
        if len(np.shape(self.sweep_points)) == 2:
            return self.sweep_points[start_idx:end_idx, :].T
        else:
            return self.sweep_points[start_idx:end_idx]

##############################################################################
##############################################################################
####################     Software Controlled Detectors     ###################
##############################################################################
##############################################################################


class Dummy_Detector_Soft(Soft_Detector):

    def __init__(self, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']
        self.i = 0
        # self.x can be used to set x value externally
        self.x = None

    def acquire_data_point(self, **kw):
        if self.x is None:
            x = self.i/15.
        self.i += 1
        time.sleep(self.delay)
        return np.array([np.sin(x/np.pi), np.cos(x/np.pi)])


class Dummy_Detector_Soft_diff_shape(Soft_Detector):
    # For testing purpose, returns data in a slightly different shape

    def __init__(self, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']
        self.i = 0
        # self.x can be used to set x value externally
        self.x = None

    def acquire_data_point(self, **kw):
        if self.x is None:
            x = self.i/15.
        self.i += 1
        time.sleep(self.delay)
        # This is the format an N-D detector returns data in.
        return np.array([[np.sin(x/np.pi), np.cos(x/np.pi)]]).reshape(2, -1)


class Function_Detector(Soft_Detector):
    """
    Defines a detector function that wraps around an user-defined function.
    Inputs are:
        get_function (callable) : function used for acquiring values
        value_names (list) : names of the elements returned by the function
        value_units (list) : units of the elements returned by the function
        result_keys (list) : keys of the dictionary returned by the function
                             if not None
        msmt_kw   (dict)   : kwargs for the get_function, dict items can be
            values or parameters. If they are parameters the output of the
            get method will be used for each get_function evaluation.

        prepare_function (callable): function used as the prepare method
        prepare_kw (dict)   : kwargs for the prepare function
        always_prepare (bool) : if True calls prepare every time data is
            acquried

    The input function get_function must return a dictionary.
    The contents(keys) of this dictionary are going to be the measured
    values to be plotted and stored by PycQED
    """

    def __init__(self, get_function, value_names=None,
                 detector_control: str='soft',
                 value_units: list=None, msmt_kw: dict ={},
                 result_keys: list=None,
                 prepare_function=None, prepare_function_kw: dict={},
                 always_prepare: bool=False, **kw):
        super().__init__()
        self.get_function = get_function
        self.result_keys = result_keys
        self.value_names = value_names
        self.value_units = value_units
        self.msmt_kw = msmt_kw
        self.detector_control = detector_control
        if self.value_names is None:
            self.value_names = result_keys
        if self.value_units is None:
            self.value_units = ['a.u.'] * len(self.value_names)

        self.prepare_function = prepare_function
        self.prepare_function_kw = prepare_function_kw
        self.always_prepare = always_prepare

    def prepare(self, **kw):
        if self.prepare_function is not None:
            self.prepare_function(**self.prepare_function_kwargs)

    def acquire_data_point(self, **kw):
        measurement_kwargs = {}
        # If an entry has a get method that will be used to set the value.
        # This makes parameters work in this context.
        for key, item in self.msmt_kw.items():
            if isinstance(item, _BaseParameter):
                value = item.get()
            else:
                value = item
            measurement_kwargs[key] = value

        # Call the function
        result = self.get_function(**measurement_kwargs)
        if self.result_keys is None:
            return result
        else:
            results = [result[key] for key in self.result_keys]
            if len(results) == 1:
                return results[0]  # for a single entry we don't want a list
            return results

    def get_values(self):
        return self.acquire_data_point()



# --------------------------------------------
# Zurich Instruments UHFQC detector functions
# --------------------------------------------

class UHFQC_Base(Hard_Detector):
    """
    Base Class for all UHF detectors
    """
    def __init__(self, UHFQC=None, detectors=None):
        super().__init__()
        if detectors is None:
            # if no detector is provided then itself is the only detector
            self.detectors = [self]
            assert UHFQC is not None, "UHFQC required when using single detector"
            self.UHFQC = UHFQC
        else:
            # in multi uhf mode several detectors are passed.
            self.detectors = [p[1] for p in sorted(
                [(d.UHFQC.devname, d) for d in detectors], reverse=True)]
        self.AWG = None

        self.UHFs = [d.UHFQC for d in self.detectors]
        self.UHF_map = {UHF.name: i
                   for UHF, i in zip(self.UHFs, range(len(self.detectors)))}

    def poll_data(self):
        if self.AWG is not None:
            self.AWG.stop()

        for UHF in self.UHFs:
            UHF.set('qas_0_result_enable', 1)

        if self.AWG is not None:
            self.AWG.start()

        acq_paths = {UHF.name: UHF._acquisition_nodes for UHF in self.UHFs}

        data = {UHF.name: {k: [] for k, dummy in enumerate(UHF._acquisition_nodes)}
                for UHF in self.UHFs}

        # Acquire data
        gotem = {UHF.name: [False] * len(UHF._acquisition_nodes) for UHF in
                 self.UHFs}
        accumulated_time = 0

        while accumulated_time < self.UHFs[0].timeout() and \
                not all(np.concatenate(list(gotem.values()))):
            dataset = {}
            for UHF in self.UHFs:
                if not all(gotem[UHF.name]):
                    time.sleep(0.01)
                    dataset[UHF.name] = UHF.poll(0.01)
            for UHFname in dataset.keys():
                for n, p in enumerate(acq_paths[UHFname]):
                    if p in dataset[UHFname]:
                        for v in dataset[UHFname][p]:
                            data[UHFname][n] = np.concatenate(
                                (data[UHFname][n], v['vector']))

                            if len(data[UHFname][n]) >= self.detectors[
                                self.UHF_map[UHFname]].nr_sweep_points:
                                gotem[UHFname][n] = True
            accumulated_time += 0.01 * len(self.UHFs)

        if not all(np.concatenate(list(gotem.values()))):
            for UHF in self.UHFs:
                UHF.acquisition_finalize()
                for n, c in enumerate(UHF._acquisition_nodes):
                    if n in data[UHF.name]:
                        n_swp = len(data[UHF.name][n])
                        tot_swp = self.detectors[
                            self.UHF_map[UHF.name]].nr_sweep_points
                        log.info(f"\t: Channel {n}: Got {n_swp} of {tot_swp} "
                                 f"samples")
                raise TimeoutError("Error: Didn't get all results!")

        data_raw = {UHF.name: np.array([data[UHF.name][key]
                    for key in sorted(data[UHF.name].keys())]) for UHF in
                    self.UHFs}

        return data_raw


class UHFQC_multi_detector(UHFQC_Base):
    """
    Combines several UHF detectors into a single detector
    """
    def __init__(self, detectors, **kw):
        super().__init__(detectors=detectors)
        self.AWG = None
        self.value_names = []
        self.value_units = []

        for d in self.detectors:
            self.value_names += [vn + ' ' + d.UHFQC.name for vn in d.value_names]
            self.value_units += d.value_units
            if d.AWG is not None:
                if self.AWG is None:
                    self.AWG = d.AWG
                elif self.AWG != d.AWG:
                    raise Exception('Not all AWG instances in UHFQC_multi_detector'
                                    ' are the same')
                d.AWG = None

        # currently only has support for classifier detector data
        self.correlated = kw.get('correlated', False)
        self.averaged = kw.get('averaged', True)
        if 'classifier' in self.detectors[0].name:
            print(self.detectors[0].name)
            self.correlated = self.detectors[0].get_values_function_kwargs.get(
                'correlated', True)
            self.averaged = self.detectors[0].get_values_function_kwargs.get(
                'averaged', True)
            print(self.correlated)
        if self.correlated:
            self.value_names += ['correlation']
            self.value_units += ['']

    def prepare(self, sweep_points):
        for d in self.detectors:
            d.prepare(sweep_points)

    def get_values(self):
        raw_data = self.poll_data()

        processed_data = [self.detectors[self.UHF_map[UHF]].process_data(d)
                          for UHF, d in raw_data.items()]
        processed_data = np.concatenate(processed_data)
        if self.correlated:
            corr_data = self.get_correlations_classif_det(np.concatenate([
                d for d in raw_data.values()]))
            processed_data = np.concatenate([processed_data, corr_data], axis=0)

        print('final ', processed_data.shape)
        return processed_data

    def get_correlations_classif_det(self, data):
        classifier_params_list = []
        state_prob_mtx_list = []
        for d in self.detectors:
            classifier_params_list += d.classifier_params_list
            state_prob_mtx_list += d.state_prob_mtx_list
        print('data ', data.shape)

        nr_states = len(self.detectors[0].state_labels)
        len_ch_pairs = sum([len(d.channel_str_pairs) for d in self.detectors])
        clf_data_all = np.zeros(
            (self.detectors[0].nr_sweep_points * self.detectors[0].nr_shots,
             nr_states * len_ch_pairs))
        for i in range(len_ch_pairs):
            clf_data = a_tools.predict_gm_proba_from_clf(
                data[2 * i: 2 * i + 2, :].T, classifier_params_list[i])
            if self.detectors[0].get_values_function_kwargs.get(
                    'thresholded', True):
                # clf_data must be 2 dimensional, rows are shots*sweep_points,
                # columns are nr_states
                clf_data = np.isclose(np.repeat([np.arange(nr_states)],
                                                clf_data.shape[0], axis=0).T,
                                      np.argmax(clf_data, axis=1)).T
            if state_prob_mtx_list is not None:
                clf_data = (np.linalg.inv(
                    state_prob_mtx_list[i]).T @ clf_data.T).T
                log.info('Correlated data corrected based on state_prob_mtx.')
            else:
                log.info('not correcting correlated data')
            clf_data_all[:, nr_states * i: nr_states * i + nr_states] = clf_data

        print('clf_data_all1 ', clf_data_all.shape)
        # can only correlate corresponding probabilities on all channels;
        # it cannot correlate selected channels
        nr_states = len(self.detectors[0].state_labels)
        q = clf_data_all.shape[1] // nr_states
        qb_states_list = [np.argmax(
            clf_data_all[:, i * nr_states: i * nr_states + nr_states],
            axis=1) for i in range(q)]
        corr_data = np.sum(np.array(qb_states_list) % 2, axis=0) % 2
        if self.averaged:
            corr_data = np.reshape(
                corr_data, (self.detectors[0].nr_shots,
                            self.detectors[0].nr_sweep_points))
            corr_data = np.mean(corr_data, axis=0)
        corr_data = np.reshape(corr_data, (1, corr_data.size))
        print('corr_data ', corr_data.shape)

        return corr_data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()

        for d in self.detectors:
            d.finish()


class UHFQC_input_average_detector(UHFQC_Base):

    '''
    Detector used for acquiring averaged input traces withe the UHFQC

    '''

    def __init__(self, UHFQC, AWG=None, channels=(0, 1),
                 nr_averages=1024, nr_samples=4096, **kw):
        super(UHFQC_input_average_detector, self).__init__(UHFQC)
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = f'{UHFQC.name}_ch{channel}'
            self.value_units[i] = 'V'
            # UHFQC returned data is in Volts
            # January 2018 verified by Xavi
        self.AWG = AWG
        self.nr_samples = nr_samples - nr_samples % 4
        self.nr_averages = nr_averages

    def get_values(self):
        raw_data = self.poll_data()
        processed_data = self.process_data(raw_data[self.UHFQC.name])
        return processed_data

    def process_data(self, raw_data):
        # IMPORTANT: No re-scaling factor needed for input average mode as
        # the UHFQC returns volts
        # Verified January 2018 by Xavi
        return raw_data


    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        self.nr_sweep_points = self.nr_samples
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels, 
                                          samples=self.nr_samples,
                                          averages=self.nr_averages,
                                          loop_cnt=int(self.nr_averages),
                                          mode='iavg')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_integrated_average_detector(UHFQC_Base):

    '''
    Detector used for integrated average results with the UHFQC

    '''

    def __init__(self, UHFQC, AWG=None,
                 integration_length: float=1e-6, nr_averages: int=1024,
                 channels: list=(0, 1, 2, 3), result_logging_mode: str='raw',
                 real_imag: bool=True,
                 seg_per_point: int =1, single_int_avg: bool =False,
                 chunk_size: int=None,
                 values_per_point: int=1, values_per_point_suffex: list=None,
                 always_prepare: bool=False,
                 prepare_function=None, prepare_function_kwargs: dict=None,
                 **kw):
        """
        Args:
        UHFQC (instrument) : data acquisition device
        AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller

        integration_length (float): integration length in seconds
        nr_averages (int)         : nr of averages per data point
            IMPORTANT: this must be a power of 2

        result_logging_mode (str) :  options are
            - raw        -> returns raw data in V
            - lin_trans  -> applies the linear transformation matrix and
                            subtracts the offsets defined in the UFHQC.
                            This is typically used for crosstalk suppression
                            and normalization. Requires optimal weights.
            - digitized  -> returns fraction of shots based on the threshold
                            defined in the UFHQC. Requires optimal weights.

        real_imag (bool)     : if False returns data in polar coordinates
                                useful for e.g., spectroscopy
                                #FIXME -> should be named "polar"
        single_int_avg (bool): if True makes this a soft detector

        Args relating to changing the amoung of points being detected:

        seg_per_point (int)  : number of segments per sweep point,
                does not do any renaming or reshaping.
                Here for deprecation reasons.
        chunk_size    (int)  : used in single shot readout experiments.
        values_per_point (int): number of values to measure per sweep point.
                creates extra column/value_names in the dataset for each channel.
        values_per_point_suffex (list): suffex to add to channel names for
                each value. should be a list of strings with lenght equal to
                values per point.
        always_prepare (bool) : when True the acquire/get_values method will
            first call the prepare statement. This is particularly important
            when it is both a single_int_avg detector and acquires multiple
            segments per point.
        """
        super().__init__(UHFQC)

        self.name = '{}_UHFQC_integrated_average'.format(result_logging_mode)
        self.channels = deepcopy(channels)
        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = \
                '{}_{} w{}'.format(UHFQC.name, result_logging_mode, channel)
        if result_logging_mode == 'raw':
            # Units are only valid when using SSB or DSB demodulation.
            # value corrsponds to the peak voltage of a cosine with the
            # demodulation frequency.
            self.value_units = ['Vpeak']*len(self.channels)
            self.scaling_factor = 1/(1.8e9*integration_length*nr_averages)
        elif result_logging_mode == 'lin_trans':
            self.value_units = ['a.u.']*len(self.channels)
            self.scaling_factor = 1/nr_averages

        elif result_logging_mode == 'digitized':
            self.value_units = ['frac']*len(self.channels)
            self.scaling_factor = 1#/0.00146484375

        self.value_names, self.value_units = self._add_value_name_suffex(
            value_names=self.value_names, value_units=self.value_units,
            values_per_point=values_per_point,
            values_per_point_suffex=values_per_point_suffex)
        self.single_int_avg = single_int_avg
        if self.single_int_avg:
            self.detector_control = 'soft'
        # useful in combination with single int_avg
        self.always_prepare = always_prepare
        # Directly specifying seg_per_point is deprecated. values_per_point
        # replaces this functionality -MAR Dec 2017
        self.seg_per_point = max(seg_per_point, values_per_point)

        self.AWG = AWG
        self.nr_averages = nr_averages
        self.integration_length = integration_length
        # 0/1/2 crosstalk supressed /digitized/raw
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        self.result_logging_mode_idx = res_logging_indices[result_logging_mode]
        self.result_logging_mode = result_logging_mode
        self.chunk_size = chunk_size

        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs
        self._set_real_imag(real_imag)

    def _add_value_name_suffex(self, value_names: list, value_units: list,
                               values_per_point: int,
                               values_per_point_suffex: list):
        """
        For use with multiple values_per_point. Adds
        """
        if values_per_point == 1:
            return value_names, value_units
        else:
            new_value_names = []
            new_value_units = []
            if values_per_point_suffex is None:
                values_per_point_suffex = ascii_uppercase[:len(value_names)]

            for vn, vu in zip(value_names, value_units):
                for val_suffix in values_per_point_suffex:
                    new_value_names.append('{} {}'.format(vn, val_suffix))
                    new_value_units.append(vu)
            return new_value_names, new_value_units

    def _set_real_imag(self, real_imag=False):
        """
        Function so that real imag can be changed after initialization
        """

        self.real_imag = real_imag
        # Commented this out as it is already done in the init -MAR Dec 2017
        # if self.result_logging_mode == 'raw':
        #     self.value_units = ['V']*len(self.channels)
        # else:
        #     self.value_units = ['']*len(self.channels)

        if not self.real_imag:
            if len(self.channels) != 2:
                raise ValueError('Length of "{}" is not 2'.format(
                                 self.channels))
            self.value_names[0] = 'Magn'
            self.value_names[1] = 'Phase'
            self.value_units[1] = 'deg'

    def get_values(self):
        if self.always_prepare:
            self.prepare()
        data_raw = self.poll_data()
        data_processed = self.process_data(data_raw[self.UHFQC.name])

        # if self.AWG is not None:
        #     self.AWG.stop()
        #
        #
        # # starting AWG
        # if self.AWG is not None:
        #     # self.AWG.start(exclude=['UHF1'])
        #     self.AWG.start()

        # time.sleep(0.1)
        #
        # data_raw = self.UHFQC.acquisition_poll(
        #     samples=self.nr_sweep_points, arm=False, acquisition_time=0.1)
        # the self.channels should be the same as data_raw.keys().
        # this is to be tested (MAR 26-9-2017)
        return data_processed

    def process_data(self, data_raw):
        data = data_raw * self.scaling_factor
        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'qas_0_trans_offset_weightfunction_{}'.format(channel))
        if not self.real_imag:
            data = self.convert_to_polar(data)

        no_virtual_channels = len(self.value_names)//len(self.channels)

        data = np.reshape(data.T,
                          (-1, no_virtual_channels, len(self.channels))).T
        data = data.reshape((len(self.value_names), -1))

        return data

    def convert_to_polar(self, data):
        if len(data) != 2:
            raise ValueError('Expect 2 channels for rotation. Got {}'.format(
                             len(data)))
        I = data[0]
        Q = data[1]
        S21 = I + 1j*Q
        data[0] = np.abs(S21)
        data[1] = np.angle(S21)/(2*np.pi)*360
        return data

    def acquire_data_point(self):
        return self.get_values()

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()

        # Determine the number of sweep points and set them
        if sweep_points is None or self.single_int_avg:
            # this case will be used when it is a soft detector
            # Note: single_int_avg overrides chunk_size
            # single_int_avg = True is equivalent to chunk_size = 1
            self.nr_sweep_points = self.seg_per_point
        else:
            self.nr_sweep_points = len(sweep_points)*self.seg_per_point

        # this sets the result to integration and rotation outcome
            if (self.chunk_size is not None and
                    self.chunk_size < self.nr_sweep_points):
                # Chunk size is defined and smaller than total number of sweep
                # points -> only acquire one chunk
                self.nr_sweep_points = self.chunk_size * self.seg_per_point

            if (self.chunk_size is not None and
                    self.chunk_size < self.nr_sweep_points):
                # Chunk size is defined and smaller than total number of sweep
                # points -> only acquire one chunk
                self.nr_sweep_points = self.chunk_size * self.seg_per_point

        # Optionally perform extra actions on prepare
        # This snippet is placed here so that it has a chance to modify the
        # nr_sweep_points in a UHFQC detector
        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

        # Do not enable the rerun button; the AWG program uses userregs/0 to
        # define the number of iterations in the loop
        self.UHFQC.awgs_0_single(1)
        self.UHFQC.qas_0_integration_length(int(self.integration_length*1.8e9))
        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels, 
                                          samples=self.nr_sweep_points,
                                          averages=self.nr_averages,
                                          loop_cnt=int(self.nr_averages),
                                          mode='rl')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.acquisition_finalize()


class UHFQC_correlation_detector(UHFQC_integrated_average_detector):
    '''
    Detector used for correlation mode with the UHFQC.
    The argument 'correlations' is a list of tuples specifying which channels
    are correlated, and on which channel the correlated signal is output.
    For instance, 'correlations=[(0, 1, 3)]' will put the correlation of
    channels 0 and 1 on channel 3.
    '''

    def __init__(self, UHFQC, AWG=None, integration_length=1e-6,
                 nr_averages=1024,  real_imag=True,
                 channels: list = [0, 1], correlations: list=[(0, 1)],
                 result_logging_mode: str='raw',
                 used_channels=None, value_names=None,
                 seg_per_point=1, single_int_avg=False,
                 **kw):
        super().__init__(
            UHFQC, AWG=AWG, integration_length=integration_length,
            nr_averages=nr_averages, real_imag=real_imag,
            channels=channels,
            seg_per_point=seg_per_point, single_int_avg=single_int_avg,
            result_logging_mode=result_logging_mode,
            **kw)

        self.result_logging_mode = result_logging_mode
        self.correlations = correlations
        self.thresholding = self.result_logging_mode == 'digitized'
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        self.result_logging_mode_idx = res_logging_indices[
            self.result_logging_mode]

        self.used_channels = used_channels
        if self.used_channels is None:
            self.used_channels = self.channels

        if value_names is None:
            self.value_names = []
            for ch in channels:
                self.value_names += ['{}_{} w{}'.format(
                    UHFQC.name, self.result_logging_mode, ch)]
        else:
            self.value_names = value_names

        # Note that V^2 is in brackets to prevent confusion with unit prefixes
        if self.result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.value_names) + \
                               ['(V^2)']*len(self.correlations)
            self.scaling_factor = 1/(1.8e9*integration_length)
        elif self.result_logging_mode == 'lin_trans':
            self.value_units = ['a.u']*len(self.value_names) + \
                               ['a.u.']*len(self.correlations)
            self.scaling_factor = 1
        elif self.result_logging_mode == 'digitized':
            self.value_units = ['frac']*len(self.value_names) + \
                               ['normalized']*len(self.correlations)
            self.scaling_factor = 1

        for corr in correlations:
            self.value_names += ['corr ({},{})'.format(corr[0], corr[1])]

        self.define_correlation_channels()

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()
        if sweep_points is None or self.single_int_avg:
            self.nr_sweep_points = self.seg_per_point
        else:
            self.nr_sweep_points = len(sweep_points) * self.seg_per_point

        self.UHFQC.qas_0_integration_length(int(self.integration_length*(1.8e9)))

        self.set_up_correlation_weights()

        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels, 
                                          samples=self.nr_sweep_points,
                                          averages=self.nr_averages,
                                          loop_cnt=int(self.nr_averages*self.nr_sweep_points),
                                          mode='rl')

    def define_correlation_channels(self):
        self.correlation_channels = []
        used_channels = deepcopy(self.used_channels)
        for corr in self.correlations:
            # Start by assigning channels
            if corr[0] not in used_channels or corr[1] not in used_channels:
                raise ValueError('Correlations should be in used channels')

            correlation_channel = -1

            # # 9 is the (current) max number of weights in the UHFQA
            for ch in range(9):
                # Find the first unused channel to set up as correlation
                if ch not in used_channels:
                    # selects the lowest available free channel
                    correlation_channel = ch
                    self.channels += [ch]
                    self.correlation_channels += [correlation_channel]

                    print('Using channel {} for correlation ({}, {}).'
                          .format(ch, corr[0], corr[1]))
                    # correlation mode is turned on in the
                    # set_up_correlation_weights method
                    break
                    # FIXME, can currently only use one correlation

            if correlation_channel < 0:
                raise ValueError('No free channel available for correlation.')
            else:
                used_channels += [ch]

    def set_up_correlation_weights(self):
        if self.thresholding:
            # correlations mode after threshold
            # NOTE: thresholds need to be set outside the detctor object.
            self.UHFQC.qas_0_result_source(5)
        else:
            # correlations mode before threshold
            self.UHFQC.qas_0_result_source(4)
        # Configure correlation mode
        for ch in self.channels:
            if ch not in self.correlation_channels:
                # Disable correlation mode as this is used for normal
                # acquisition
                self.UHFQC.set('qas_0_correlations_{}_enable'.format(ch), 0)

        for correlation_channel, corr in zip(self.correlation_channels,
                                             self.correlations):
            # Duplicate source channel to the correlation channel and select
            # second channel as channel to correlate with.
            copy_int_weights_real = \
                self.UHFQC.get('qas_0_integration_weights_{}_real'.format(corr[0]))[
                    0]['vector']
            copy_int_weights_imag = \
                self.UHFQC.get('qas_0_integration_weights_{}_imag'.format(corr[0]))[
                    0]['vector']

            copy_rot_matrix = self.UHFQC.get('qas_0_rotations_{}'.format(corr[0]))

            self.UHFQC.set(
                'qas_0_integration_weights_{}_real'.format(correlation_channel),
                copy_int_weights_real)
            self.UHFQC.set(
                'qas_0_integration_weights_{}_imag'.format(correlation_channel),
                copy_int_weights_imag)

            self.UHFQC.set(
                'qas_0_rotations_{}'.format(correlation_channel),
                copy_rot_matrix)

            # Enable correlation mode one the correlation output channel and
            # set the source to the second source channel
            self.UHFQC.set('qas_0_correlations_{}_mode'.format(correlation_channel), 1)
            self.UHFQC.set('qas_0_correlations_{}_source'.format(correlation_channel),
                           corr[1])

            # If thresholding is enabled, set the threshold for the correlation
            # channel.
            if self.thresholding:
                thresh_level = \
                    self.UHFQC.get('qas_0_thresholds_{}_level'.format(corr[0]))
                self.UHFQC.set(
                    'qas_0_thresholds_{}_level'.format(correlation_channel),
                    thresh_level)

    def get_values(self):
        data_raw = self.poll_data()
        data_processed = self.process_data(data_raw[self.UHFQC.name])
        return data_processed

    def process_data(self, data_raw):
        if self.thresholding:
            data = data_raw
        else:
            data = []
            for n, ch in enumerate(self.used_channels):
                if ch in self.correlation_channels:
                    data.append(3 * np.array(data_raw[n]) *
                                (self.scaling_factor**2 / self.nr_averages))
                else:
                    data.append(np.array(data_raw[n]) *
                                (self.scaling_factor / self.nr_averages))
        return data


class UHFQC_integration_logging_det(UHFQC_Base):

    '''
    Detector used for integrated average results with the UHFQC

    '''

    def __init__(self, UHFQC, AWG=None,
                 integration_length: float=1e-6,
                 nr_shots: int=4094,
                 channels: list=(0, 1),
                 result_logging_mode: str='raw',
                 always_prepare: bool=False,
                 prepare_function=None,
                 prepare_function_kwargs: dict=None,
                 **kw):
        """
        Args:
        UHFQC (instrument) : data acquisition device
        AWG   (instrument) : device responsible for starting and stopping
                             the experiment, can also be a central controller.
        integration_length (float): integration length in seconds
        nr_shots (int)     : nr of shots (max is 4095)
        channels (list)    : index (channel) of UHFQC weight functions to use

        result_logging_mode (str):  options are
            - raw        -> returns raw data in V
            - lin_trans  -> applies the linear transformation matrix and
                            subtracts the offsets defined in the UFHQC.
                            This is typically used for crosstalk suppression
                            and normalization. Requires optimal weights.
            - digitized  -> returns fraction of shots based on the threshold
                            defined in the UFHQC. Requires optimal weights.
        always_prepare (bool) : when True the acquire/get_values method will
            first call the prepare statement. This is particularly important
            when it is both a single_int_avg detector and acquires multiple
            segments per point.
        """
        super().__init__(UHFQC)

        self.name = '{}_UHFQC_integration_logging_det'.format(
            result_logging_mode)
        self.channels = channels

        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = \
                '{}_{} w{}'.format(UHFQC.name, result_logging_mode, channel)
        if result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.channels)
            self.scaling_factor = 1  # /(1.8e9*integration_length)
        else:
            self.value_units = ['']*len(self.channels)
            self.scaling_factor = 1

        self.AWG = AWG
        self.integration_length = integration_length
        self.nr_shots = nr_shots

        # 0/1/2 crosstalk supressed /digitized/raw
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        # mode 3 is statistics logging, this is implemented in a
        # different detector
        self.result_logging_mode_idx = res_logging_indices[result_logging_mode]
        self.result_logging_mode = result_logging_mode

        self.always_prepare = always_prepare
        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()

        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

        # The averaging-count is used to specify how many times the AWG program
        # should run
        self.UHFQC.awgs_0_single(1)

        self.nr_sweep_points = self.nr_shots*len(sweep_points)
        self.UHFQC.qas_0_integration_length(int(self.integration_length*(1.8e9)))

        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels, 
                                          samples=self.nr_sweep_points,
                                          averages=1, #for single shot readout
                                          loop_cnt=int(self.nr_shots),
                                          mode='rl')

    def get_values(self):
        if self.always_prepare:
            self.prepare()
        # if self.AWG is not None:
        #     self.AWG.stop()
        #
        # # resets UHFQC internal readout counters
        # self.UHFQC._daq.setInt('/' + self.UHFQC._device + '/quex/rl/readout',
        #                        self._get_readout())
        #
        # # starting AWG
        # if self.AWG is not None:
        #     self.AWG.start()
        #
        # data_raw = self.UHFQC.acquisition_poll(
        #     samples=self.nr_shots, arm=False, acquisition_time=0.01)
        data_raw = self.poll_data()
        data_processed = self.process_data(data_raw[self.UHFQC.name])
        return data_processed

    def process_data(self, data_raw):
        data = data_raw * self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'qas_0_trans_offset_weightfunction_{}'.format(channel))
        return data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_classifier_detector(UHFQC_Base):

    '''
    Detector used for integrated average results with the UHFQC

    '''

    def __init__(self, UHFQC, AWG=None,
                 integration_length: float=1e-6,
                 nr_shots: int=4094,
                 channels: list=(0, 1),
                 result_logging_mode: str='raw',
                 always_prepare: bool=False,
                 prepare_function=None,
                 prepare_function_kwargs: dict=None,
                 get_values_function_kwargs: dict=None, **kw):
        """
        Args:
        UHFQC (instrument) : data acquisition device
        AWG   (instrument) : device responsible for starting and stopping
                             the experiment, can also be a central controller.
        integration_length (float): integration length in seconds
        nr_shots (int)     : nr of shots (max is 4095)
        channels (list)    : index (channel) of UHFQC weight functions to use
        result_logging_mode (str):  options are
            - raw        -> returns raw data in V
            - lin_trans  -> applies the linear transformation matrix and
                            subtracts the offsets defined in the UFHQC.
                            This is typically used for crosstalk suppression
                            and normalization. Requires optimal weights.
            - digitized  -> returns fraction of shots based on the threshold
                            defined in the UFHQC. Requires optimal weights.
        always_prepare (bool) : when True the acquire/get_values method will
            first call the prepare statement. This is particularly important
            when it is both a single_int_avg detector and acquires multiple
            segments per point.
        get_values_function_kwargs (dict): looks for the following keys:
            classified (default: True): whether to classify the shots or not
            averaged (default: True): whether to average over the shots
            thresholded (default:True): whether to threshold the shots
            correlated (default: False): whether to calculate <ZZ...Z>,
                assuming |f> states are |g> states. This creates and additional
                data column called "correlation"
            classifier_params (default: None): this must be provided if
                classified == True. Parameters for the classifier (or list of
                parameters for the classifiers if multiple qubits) used to
                classify the shots.
            sate_prob_mtx (default: None): state assignment probability matrix
                (or list of matrices if multiple qubits) used to correct for
                RO errors. No correction is applied if None.
        """
        super().__init__(UHFQC)

        self.name = '{}_UHFQC_classifier_det'.format(
            result_logging_mode)
        self.state_labels = ['pg', 'pe', 'pf']
        self.channels = channels
        self.correlated = get_values_function_kwargs.get('correlated', True)

        # Currently doesn't work with single readout channel;
        # assumes 2 channels per data point
        channel_strings = [str(ch) for ch in self.channels]
        self.channel_str_pairs = [''.join(channel_strings[2*j: 2*j+2]) for
                                  j in range(len(self.channels)//2)]
        self.value_names = ['']*(
                len(self.state_labels) * len(self.channel_str_pairs))
        idx = 0
        for ch_pair in self.channel_str_pairs:
            for state in self.state_labels:
                self.value_names[idx] = '{}_{} w{}'.format(UHFQC.name,
                    state, ch_pair)
                idx += 1

        # if self.correlated:
        #     self.value_names += ['correlation']

        if result_logging_mode == 'raw':
            self.value_units = ['']*len(self.value_names)
            self.scaling_factor = 1  # /(1.8e9*integration_length)
        else:
            self.value_units = ['']*len(self.value_names)
            self.scaling_factor = 1

        self.AWG = AWG
        self.integration_length = integration_length
        self.nr_shots = nr_shots

        # 0/1/2 crosstalk supressed /digitized/raw
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        # mode 3 is statistics logging, this is implemented in a
        # different detector
        self.result_logging_mode_idx = res_logging_indices[result_logging_mode]
        self.result_logging_mode = result_logging_mode

        self.always_prepare = always_prepare
        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs
        self.get_values_function_kwargs = get_values_function_kwargs

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()

        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

        self.nr_sweep_points = len(sweep_points)
        # The averaging-count is used to specify how many times the AWG program
        # should run
        self.UHFQC.awgs_0_single(1)
        self.UHFQC.qas_0_integration_length(int(self.integration_length*(1.8e9)))

        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels, 
                                          samples=self.nr_shots*self.nr_sweep_points,
                                          averages=1, #for single shot readout
                                          loop_cnt=int(self.nr_shots),
                                          mode='rl')

    def get_values(self):
        if self.always_prepare:
            self.prepare()
        data_raw = self.poll_data()
        processed_data = self.process_data(data_raw[self.UHFQC.name])
        return processed_data

    def process_data(self, data_raw):
        data = data_raw * self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'qas_0_trans_offset_weightfunction_{}'.format(channel))

        classified_data = data
        if self.get_values_function_kwargs.get('classified', True):
            # Classify data into qutrit states
            classifier_params_list = self.get_values_function_kwargs.get(
                'classifier_params', None)
            if not isinstance(classifier_params_list, list):
                classifier_params_list = [classifier_params_list]
            state_prob_mtx_list = self.get_values_function_kwargs.get(
                'state_prob_mtx', None)
            if state_prob_mtx_list is not None and \
                    not isinstance(state_prob_mtx_list, list):
                state_prob_mtx_list = [state_prob_mtx_list]

            self.classifier_params_list = classifier_params_list
            self.state_prob_mtx_list = state_prob_mtx_list

            classified_data = self.classify_shots(
                data, self.classifier_params_list, self.state_prob_mtx_list,
                self.get_values_function_kwargs.get('averaged', True),
                self.get_values_function_kwargs.get('thresholded', True))

        return classified_data.T

    def classify_shots(self, data, classifier_params_list,
                       state_prob_mtx_list=None, averaged=False,
                       thresholded=True):

        if classifier_params_list is None:
            raise ValueError('Please specify the classifier parameters list.')

        nr_states = len(self.state_labels)
        classified_data = np.zeros(
            (nr_states*len(self.channel_str_pairs),
             self.nr_sweep_points if averaged else
             self.nr_sweep_points*self.nr_shots))

        clf_data_all = np.zeros((self.nr_sweep_points*self.nr_shots,
                                nr_states*len(self.channel_str_pairs)))

        for i in range(len(self.channel_str_pairs)):
            clf_data = a_tools.predict_gm_proba_from_clf(
                data[2*i: 2*i+2, :].T, classifier_params_list[i])
            if thresholded:
                # clf_data must be 2 dimensional, rows are shots*sweep_points,
                # columns are nr_states
                clf_data = np.isclose(np.repeat([np.arange(nr_states)],
                                                clf_data.shape[0], axis=0).T,
                                      np.argmax(clf_data, axis=1)).T
            clf_data_all[:, nr_states*i: nr_states*i+nr_states] = clf_data

            if averaged:
                # reshape into (nr_shots, nr_sweep_points, nr_data_columns)
                clf_data = np.reshape(
                    clf_data, (self.nr_shots, self.nr_sweep_points,
                               clf_data.shape[-1]))
                clf_data = np.mean(clf_data, axis=0)
            if state_prob_mtx_list is not None:
                clf_data = np.linalg.inv(
                    state_prob_mtx_list[i]).T @ clf_data.T
                log.info('Data corrected based on state_prob_mtx.')
            else:
                log.info('not correcting data')
                clf_data = clf_data.T

            classified_data[nr_states * i: nr_states * i + nr_states, :] = \
                clf_data
        # if True:
        #     # can only correlate corresponding probabilities on all channels;
        #     # it cannot correlate selected channels
        #     q = clf_data_all.shape[1] // nr_states
        #     print(clf_data_all.shape[1])
        #     print(nr_states, q)
        #     qb_states_list = [np.argmax(
        #         clf_data_all[:, i*nr_states: i*nr_states + nr_states],
        #         axis=1) for i in range(q)]
        #     corr_data = np.sum(np.array(qb_states_list) % 2, axis=0) % 2
        #     if averaged:
        #         corr_data = np.reshape(
        #             corr_data, (self.nr_shots, self.nr_sweep_points))
        #         corr_data = np.mean(corr_data, axis=0)
        #     corr_data = np.reshape(corr_data, (1, corr_data.size))
        #     classified_data = np.concatenate([classified_data, corr_data],
        #                                      axis=0)
        return classified_data.T

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
