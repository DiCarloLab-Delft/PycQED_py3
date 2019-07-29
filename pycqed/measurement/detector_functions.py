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


class Detect_simulated_hanger_Soft(Soft_Detector):

    def __init__(self, **kw):
        self.set_kw()

        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['mV', 'mV']

    def acquire_data_point(self, **kw):
        f = self.source.get_frequency()
        f0 = 5.e9
        Q = 10000.
        Qe = 12000.
        theta = 0.2
        A = 50.
        Inoise = np.random.randn()
        Qnoise = np.random.randn()

        IQ = fn.disp_hanger_S21_complex(*(f, f0, Q, Qe, A, theta))
        return IQ.real+Inoise, IQ.imag+Qnoise

class Heterodyne_probe(Soft_Detector):

    def __init__(self, HS, threshold=1.75, trigger_separation=10e-6,
                 demod_mode='double', RO_length=2000e-9, **kw):
        super().__init__(**kw)
        self.HS = HS
        self.name = 'Heterodyne probe'
        self.value_names = ['|S21|', 'S21 angle']  # , 'Re{S21}', 'Im{S21}']
        self.value_units = ['V', 'deg']
        self.first = True
        self.last_frequency = 0.
        self.threshold = threshold
        self.last = 1.
        self.HS.trigger_separation(trigger_separation)
        if 'double' in demod_mode:
            HS.single_sideband_demod(False)
        else:
            HS.single_sideband_demod(True)

        self.trigger_separation = trigger_separation
        self.demod_mode = demod_mode
        self.RO_length = RO_length

    def prepare(self):
        self.HS.RO_length(self.RO_length)
        self.HS.trigger_separation(self.trigger_separation)

        self.HS.prepare()

    def acquire_data_point(self, **kw):
        S21 = self.HS.probe()
        return abs(S21), np.angle(S21)/(2*np.pi)*360,  # S21.real, S21.imag

    def finish(self):
        self.HS.finish()


class Heterodyne_probe_soft_avg(Soft_Detector):

    def __init__(self, HS, threshold=1.75, Navg=10, **kw):
        super().__init__(**kw)
        self.HS = HS
        self.name = 'Heterodyne probe'
        self.value_names = ['|S21|', 'S21 angle']  # , 'Re{S21}', 'Im{S21}']
        self.value_units = ['mV', 'deg']  # , 'a.u.', 'a.u.']
        self.first = True
        self.last_frequency = 0.
        self.threshold = threshold
        self.last = 1.
        self.Navg = Navg

    def prepare(self):
        self.HS.prepare()

    def acquire_data_point(self, **kw):
        accum_real = 0.
        accum_imag = 0.
        for i in range(self.Navg):
            measure = self.acquire_single_data_point(**kw)
            accum_real += measure[0]
            accum_imag += measure[1]
        S21 = (accum_real+1j*accum_imag)/float(self.Navg)

        return abs(S21), np.angle(S21)/(2*np.pi)*360

    def acquire_single_data_point(self, **kw):
        passed = False
        c = 0
        while(not passed):
            S21 = self.HS.probe()
            cond_a = (
                abs(S21)/self.last > self.threshold) or (self.last/abs(S21) > self.threshold)
            cond_b = self.HS.frequency() > self.last_frequency
            if cond_a and cond_b:
                passed = False
            else:
                passed = True
            if self.first or c > 3:
                passed = True
            if not passed:
                print('retrying HS probe')
            c += 1
        self.last_frequency = self.HS.frequency()
        self.first = False
        self.last = abs(S21)
        return S21.real, S21.imag

    def finish(self):
        self.HS.finish()


# --------------------------------------------
# Zurich Instruments UHFQC detector functions
# --------------------------------------------


class UHFQC_multi_detector(Hard_Detector):
    """
    Combines several UHF detectors into a single detector
    """
    def __init__(self, detectors):
        super().__init__()
        self.detectors = [p[1] for p in sorted( \
            [(d.UHFQC._device, d) for d in detectors], reverse=True)]
        self.AWG = None
        self.value_names = []
        self.value_units = []
        self.AWG = None
        for d in self.detectors:
            self.value_names += [vn + ' ' + d.UHFQC.name for vn in d.value_names]
            self.value_units += d.value_units
            if d.AWG is not None:
                if self.AWG is None:
                    self.AWG = d.AWG
                elif self.AWG != d.AWG:
                    raise Exception('Not all AWG instances in UHFQC_multi_...  '
                                    'are the same')
                d.AWG = None

    def prepare(self, sweep_points):
        # _daq = self.detectors[0].UHFQC._daq
        # for d in self.detectors[1:]:
        #     d.UHFQC._daq = _daq

        for d in self.detectors:
            d.prepare(sweep_points)

        # for d in self.detectors[1:]:
        #     self.detectors[0].UHFQC.acquisition_paths += \
        #         d.UHFQC.acquisition_paths
        #     d.UHFQC.acquisition_paths = []
        #
        #     self.detectors[0].value_names += d.value_names
        #     d.value_names = []
        #     self.detectors[0].channels += d.channels
        #     d.channels = []

    def get_values(self):
        if self.AWG is not None:
            self.AWG.stop()

        UHFs = [d.UHFQC for d in self.detectors]

        for UHF in UHFs:
            UHF._daq.setInt('/' + UHF._device + '/quex/rl/readout', 1)


        if self.AWG is not None:
            self.AWG.start()

        acq_paths = {UHF.name: UHF.acquisition_paths for UHF in UHFs}

        data = {UHF.name: {k: [] for k, dummy in enumerate(UHF.acquisition_paths)}
                for UHF in UHFs}

        # Acquire data
        gotem = {UHF.name: [False] * len(UHF.acquisition_paths) for UHF in UHFs}
        accumulated_time = 0

        while accumulated_time < UHFs[0].timeout() and \
                not all(np.concatenate(list(gotem.values()))):
            dataset = {}
            for UHF in UHFs:
                if not all(gotem[UHF.name]):
                    time.sleep(0.01)
                    dataset[UHF.name] = UHF._daq.poll(0.001, 1, 4, True)

            # print(dataset)

            for UHFname in dataset.keys():
                for n, p in enumerate(acq_paths[UHFname]):
                    if p in dataset[UHFname]:
                        for v in dataset[UHFname][p]:
                            data[UHFname][n] = np.concatenate(
                                (data[UHFname][n], v['vector']))
                            if len(data[UHFname][n]) >= self.detectors[
                                0].nr_sweep_points:
                                gotem[UHFname][n] = True
            accumulated_time += 0.01*len(UHFs)

        if  not all(np.concatenate(list(gotem.values()))):
            for UHF in UHFs:
                UHF.acquisition_finalize()
                for n, c in enumerate(UHF.acquisition_paths):
                    if n in data[UHF.name]:
                        print("\t: Channel {}: Got {} of {} samples".format(
                            n, len(data[UHF.name][n]), self.detectors[0].nr_sweep_points))
                raise TimeoutError("Error: Didn't get all results!")


        data_flat = np.concatenate([np.array([data[UHF.name][key]
                         for key in data[UHF.name].keys()]) for UHF in UHFs])



        data = data_flat * self.detectors[0].scaling_factor

        no_virtual_channels = 1

        # data = np.reshape(data.T,
        #                   (-1, no_virtual_channels, len(self.channels))).T
        # data = data.reshape((len(self.value_names), -1))

        return data


    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()

        for d in self.detectors:
            d.finish()

class UHFQC_input_average_detector(Hard_Detector):

    '''
    Detector used for acquiring averaged input traces withe the UHFQC

    '''

    def __init__(self, UHFQC, AWG=None, channels=(0, 1),
                 nr_averages=1024, nr_samples=4096, **kw):
        super(UHFQC_input_average_detector, self).__init__()
        self.UHFQC = UHFQC
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = 'ch{}'.format(channel)
            self.value_units[i] = 'V'
            # UHFQC returned data is in Volts
            # January 2018 verified by Xavi
        self.AWG = AWG
        self.nr_samples = nr_samples - nr_samples % 4
        self.nr_averages = nr_averages

    def get_values(self):
        self.UHFQC.quex_rl_readout(0)  # resets UHFQC internal readout counters

        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(samples=self.nr_sweep_points,
                                               arm=False, acquisition_time=0.01)
        data = np.array([data_raw[key]
                         for key in sorted(data_raw.keys())])
        # IMPORTANT: No re-scaling factor needed for input average mode as the UHFQC returns volts
        # Verified January 2018 by Xavi

        return data

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.quex_iavg_length(self.nr_samples)
        self.UHFQC.quex_iavg_avgcnt(int(np.log2(self.nr_averages)))
        self.UHFQC.awgs_0_userregs_1(1)  # 0 for rl, 1 for iavg
        self.UHFQC.awgs_0_userregs_0(
            int(self.nr_averages)) 
        self.nr_sweep_points = self.nr_samples
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='iavg')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()

class UHFQC_demodulated_input_avg_det(UHFQC_input_average_detector):
    '''
    Detector used for acquiring averaged input traces withe the UHFQC.
    Additionally trace are demoulated.
    '''

    def __init__(self, f_RO_mod, UHFQC,
                 real_imag=True, AWG=None, channels=(0, 1),
                 nr_averages=1024, nr_samples=4096, **kw):
        super().__init__(UHFQC=UHFQC, AWG=AWG, channels=channels,
                         nr_averages=nr_averages, nr_samples=nr_samples, **kw)
        self.f_mod = f_RO_mod
        self.real_imag = real_imag

    def get_values(self):
        data = super().get_values()
        timePts = np.arange(len(data[0])) / 1.8e9
        data_I_demod = (np.array(data[0]) * np.cos(2*np.pi*timePts*self.f_mod) -
                        np.array(data[1]) * np.sin(2*np.pi*timePts*self.f_mod))
        data_Q_demod = (np.array(data[0]) * np.sin(2*np.pi*timePts*self.f_mod) +
                        np.array(data[1]) * np.cos(2*np.pi*timePts*self.f_mod))
        ret_data = np.array([data_I_demod, data_Q_demod])

        if not self.real_imag:
            S21 = data[0] + 1j*data[1]
            amp = np.abs(S21)
            phase = np.angle(S21) / (2*np.pi) * 360
            ret_data = np.array([amp, phase])

        return ret_data


class UHFQC_integrated_average_detector(Hard_Detector):

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
        super().__init__()
        self.UHFQC = UHFQC
        # if nr_averages # is not a powe of 2:

        #    raise ValueError('Some descriptive message {}'.format(nr_averages))
        # if integration_length > some value:
        #     raise ValueError

        self.name = '{}_UHFQC_integrated_average'.format(result_logging_mode)
        self.channels = deepcopy(channels)
        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = '{} w{}'.format(result_logging_mode,
                                                  channel)
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

    def _get_readout(self):
        return sum([(1 << c) for c in self.channels])

    def get_values(self):
        if self.always_prepare:
            self.prepare()
        if self.AWG is not None:
            self.AWG.stop()

        # resets UHFQC internal readout counters
        self.UHFQC._daq.setInt('/' + self.UHFQC._device + '/quex/rl/readout', 
                                self._get_readout())

        # starting AWG
        if self.AWG is not None:
            # self.AWG.start(exclude=['UHF1'])
            self.AWG.start()


        # print(self.nr_sweep_points)
        time.sleep(0.1)
        # self.UHFQC._daq.sync()
        # self.UHFQC.acquisition_arm()
        data_raw = self.UHFQC.acquisition_poll(
            samples=self.nr_sweep_points, arm=False, acquisition_time=0.1)
        # the self.channels should be the same as data_raw.keys().
        # this is to be tested (MAR 26-9-2017)

        data = np.array([data_raw[key]
                         for key in sorted(data_raw.keys())])*self.scaling_factor
        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'quex_trans_offset_weightfunction_{}'.format(channel))
        if not self.real_imag:
            data = self.convert_to_polar(data)

        no_virtual_channels = len(self.value_names)//len(self.channels)

        data = np.reshape(data.T,
                          (-1, no_virtual_channels, len(self.channels))).T
        data = data.reshape((len(self.value_names), -1))
        # Fixme: For debugging purposes
        #self.UHFQC.acquisition_finalize()
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

        self.UHFQC.awgs_0_userregs_0(
            # int(self.nr_averages*self.nr_sweep_points))
            int(self.nr_averages))
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg (input avg)
        # The AWG program uses userregs/0 to define the number of iterations in
        # the loop

        self.UHFQC.quex_rl_length(self.nr_sweep_points)
        self.UHFQC.quex_rl_avgcnt(int(np.log2(self.nr_averages)))
        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))

        self.UHFQC.quex_rl_source(self.result_logging_mode_idx)
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.acquisition_finalize()

class UHFQC_SSRO_detector(Hard_Detector):

    '''
    Detector used for single-shot acquisition with the UHFQC.
    '''

    def __init__(self, UHFQC, AWG=None,
                 integration_length: float=1e-6,
                 nr_shots: int=4094,
                 channels: list=(0, 1),
                 result_logging_mode: str='raw',
                 operation='average',**kw):

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
        """

        super().__init__()
        self.UHFQC = UHFQC
        self.name = '{}_UHFQC_SSRO_detector'.format(result_logging_mode)
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = '{} w{}'.format(result_logging_mode,
                                                  channel)
        if result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.channels)
            self.scaling_factor = 1/(1.8e9*integration_length*nr_shots)
        elif result_logging_mode == 'lin_trans':
            self.value_units = ['a.u.']*len(self.channels)
            self.scaling_factor = 1/nr_shots

        elif result_logging_mode == 'digitized':
            self.value_units = ['frac']*len(self.channels)
            self.scaling_factor = 1

        self.AWG = AWG
        self.nr_shots = nr_shots
        self._set_operation(operation)
        self.integration_length = integration_length

        # 0/1/2 crosstalk supressed /digitized/raw
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        self.result_logging_mode_idx = res_logging_indices[result_logging_mode]
        self.result_logging_mode = result_logging_mode

    def _set_operation(self, operation='average'):
        """
        Function so that value names and units can be changed
        after initialization
        """

        self.operation = operation
        if self.result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.channels)
        else:
            self.value_units = ['']*len(self.channels)

        # if not self.operation:
        #     if len(self.channels) != 2:
        #         raise ValueError()
        #     self.value_names[0] = 'Magn'
        #     self.value_names[1] = 'Phase'
        #     self.value_units[1] = 'deg'

    def apply_operation(self, data):

        if self.operation is 'average':
            return np.mean(data)
        elif self.operation is '<zz>':
            pass
        else:
            raise ValueError('Unknown operation.')

    def get_values(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.quex_rl_readout(1)  # resets UHFQC internal readout counters
        # self.UHFQC.acquisition_arm()
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(
            samples=self.nr_shots, arm=False, acquisition_time=0.01)
        data = np.array([data_raw[key]
                         for key in data_raw.keys()])*self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'quex_trans_offset_weightfunction_{}'.format(channel))

        #data = self.apply_operation(data)

        return data

    def acquire_data_point(self):
        return self.get_values()

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()

        # Determine the number of sweep points and set them
        if sweep_points is None:
            logging.warning('sweep_points is None. Setting nr_sweep_points=1.')
            self.nr_sweep_points = 1
        else:
            self.nr_sweep_points = len(sweep_points)

        self.UHFQC.awgs_0_single(1)
        self.UHFQC.awgs_0_userregs_0(int(self.nr_shots))
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg (input avg)
        # The AWG program uses userregs/0 to define the number of iterations in
        # the loop

        self.UHFQC.quex_rl_length(self.nr_sweep_points)
        self.UHFQC.quex_rl_avgcnt(int(np.log2(self.nr_shots)))
        # self.UHFQC.quex_rl_avgcnt(0) #for SS
        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))

        self.UHFQC.quex_rl_source(self.result_logging_mode_idx)
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


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
                 used_channels=None,
                 value_names=None,
                 seg_per_point=1, single_int_avg=False, thresholding=False,
                 **kw):
        super().__init__(
            UHFQC, AWG=AWG, integration_length=integration_length,
            nr_averages=nr_averages, real_imag=real_imag,
            channels=channels,
            seg_per_point=seg_per_point, single_int_avg=single_int_avg,
            result_logging_mode='raw',  # FIXME -> do the proper thing (MAR)
            **kw)

        self.correlations = correlations
        self.thresholding = thresholding

        self.used_channels = used_channels
        if self.used_channels is None:
            self.used_channels = self.channels

        if value_names is None:
            self.value_names = []
            for ch in channels:
                self.value_names += ['w{}'.format(ch)]
        else:
            self.value_names = value_names

        # Note that V^2 is in brackets to prevent confusion with unit prefixes
        if not thresholding:
            self.value_units = ['V']*len(self.value_names) + \
                               ['(V^2)']*len(self.correlations)
        else:
            self.value_units = ['fraction']*len(self.value_names) + \
                               ['normalized']*len(self.correlations)
        for corr in correlations:
            self.value_names += ['corr ({},{})'.format(corr[0], corr[1])]

        self.define_correlation_channels()

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()
        if sweep_points is None or self.single_int_avg:
            self.nr_sweep_points = self.seg_per_point
        else:
            self.nr_sweep_points = len(sweep_points)*self.seg_per_point

        self.UHFQC.quex_rl_length(self.nr_sweep_points)
        self.UHFQC.quex_rl_avgcnt(int(np.log2(self.nr_averages)))
        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))

        self.set_up_correlation_weights()

        # Configure the result logger to not do any averaging
        # The AWG program uses userregs/0 to define the number o iterations in
        # the loop
        self.UHFQC.awgs_0_userregs_0(
            int(self.nr_averages*self.nr_sweep_points))
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg

        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def define_correlation_channels(self):
        self.correlation_channels = []
        used_channels = deepcopy(self.used_channels)
        for corr in self.correlations:
            print(corr)
            print(self.channels)
            print(used_channels)
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
            self.UHFQC.quex_rl_source(5)
        else:
            # correlations mode before threshold
            self.UHFQC.quex_rl_source(4)
        # Configure correlation mode
        for ch in self.channels:
            if ch not in self.correlation_channels:
                # Disable correlation mode as this is used for normal
                # acquisition
                self.UHFQC.set('quex_corr_{}_mode'.format(ch), 0)

        for correlation_channel, corr in zip(self.correlation_channels,
                                             self.correlations):
            # Duplicate source channel to the correlation channel and select
            # second channel as channel to correlate with.
            copy_int_weights_real = \
                self.UHFQC.get('quex_wint_weights_{}_real'.format(corr[0]))[
                    0]['vector']
            copy_int_weights_imag = \
                self.UHFQC.get('quex_wint_weights_{}_imag'.format(corr[0]))[
                    0]['vector']

            copy_rot_matrix_real = \
                self.UHFQC.get('quex_rot_{}_real'.format(corr[0]))
            copy_rot_matrix_imag = \
                self.UHFQC.get('quex_rot_{}_imag'.format(corr[0]))

            self.UHFQC.set(
                'quex_wint_weights_{}_real'.format(correlation_channel),
                copy_int_weights_real)
            self.UHFQC.set(
                'quex_wint_weights_{}_imag'.format(correlation_channel),
                copy_int_weights_imag)

            self.UHFQC.set(
                'quex_rot_{}_real'.format(correlation_channel),
                copy_rot_matrix_real)
            self.UHFQC.set(
                'quex_rot_{}_imag'.format(correlation_channel),
                copy_rot_matrix_imag)
            # Enable correlation mode one the correlation output channel and
            # set the source to the second source channel
            self.UHFQC.set('quex_corr_{}_mode'.format(correlation_channel), 1)
            self.UHFQC.set('quex_corr_{}_source'.format(correlation_channel),
                           corr[1])

            # If thresholding is enabled, set the threshold for the correlation
            # channel.
            if self.thresholding:
                thresh_level = \
                    self.UHFQC.get('quex_thres_{}_level'.format(corr[0]))
                self.UHFQC.set(
                    'quex_thres_{}_level'.format(correlation_channel),
                    thresh_level)

    def get_values(self):
        # Slightly different way to deal with scaling factor
        self.scaling_factor = 1 / (1.8e9*self.integration_length)

        if self.AWG is not None:
            self.AWG.stop()
        # self.UHFQC.quex_rl_readout(1)  # resets UHFQC internal readout counters
        self.UHFQC._daq.setInt('/' + self.UHFQC._device + '/quex/rl/readout', 
                                self._get_readout())
        # self.UHFQC.acquisition_arm()
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(samples=self.nr_sweep_points,
                                               arm=False,
                                               acquisition_time=0.01)
        data = []
        if self.thresholding:
            for key in sorted(data_raw.keys()):
                # data.append(np.array(data_raw[key])/0.00146484375)
                data.append(np.array(data_raw[key]))
        else:
            for key in sorted(data_raw.keys()):
                if key in self.correlation_channels:
                    data.append(3*np.array(data_raw[key]) *
                                (self.scaling_factor**2 / self.nr_averages))
                else:
                    data.append(np.array(data_raw[key]) *
                                (self.scaling_factor / self.nr_averages))

        return data


class UHFQC_integration_logging_det(Hard_Detector):

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
        super().__init__()

        self.UHFQC = UHFQC
        self.name = '{}_UHFQC_integration_logging_det'.format(
            result_logging_mode)
        self.channels = channels

        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = '{} w{}'.format(result_logging_mode,
                                                  channel)
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
        self.UHFQC.awgs_0_userregs_0(self.nr_shots) # The AWG program uses 
        # userregs/0 to define the number of iterations
        # in the loop
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg (input avg)


        self.UHFQC.quex_rl_length(self.nr_shots*len(sweep_points))
        self.UHFQC.quex_rl_avgcnt(0)  # log2(1) for single shot readout
        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))

        self.UHFQC.quex_rl_source(self.result_logging_mode_idx)
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def _get_readout(self):
        return sum([(1 << c) for c in self.channels])

    def get_values(self):
        if self.always_prepare:
            self.prepare()
        if self.AWG is not None:
            self.AWG.stop()

        # resets UHFQC internal readout counters
        self.UHFQC._daq.setInt('/' + self.UHFQC._device + '/quex/rl/readout',
                               self._get_readout())

        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(
            samples=self.nr_shots, arm=False, acquisition_time=0.01)
        data = np.array([data_raw[key]
                         for key in sorted(data_raw.keys())])*self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'quex_trans_offset_weightfunction_{}'.format(channel))
        return data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_integration_average_classifier_det(Hard_Detector):

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
                 get_values_function_kwargs: dict=None,
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
        super().__init__()

        self.UHFQC = UHFQC
        self.name = '{}_UHFQC_integration_logging_det'.format(
            result_logging_mode)
        self.channels = channels
        self.state_labels = ['pg', 'pe', 'pf']

        channel_strings = [str(ch) for ch in self.channels]
        self.channel_str_pairs = [''.join(channel_strings[2*j: 2*j+2]) for
                                  j in range(len(self.channels)//2)]
        self.value_names = ['']*(len(
            self.state_labels)*len(self.channel_str_pairs))
        idx = 0
        for ch_pair in self.channel_str_pairs:
            for state in self.state_labels:
                self.value_names[idx] = '{} w{}'.format(
                    state, ch_pair)
                idx += 1

        if result_logging_mode == 'raw':
            self.value_units = ['']*(len(
                self.state_labels)*len(self.channel_str_pairs))
            self.scaling_factor = 1  # /(1.8e9*integration_length)
        else:
            self.value_units = ['']*(len(
                self.state_labels)*len(self.channel_str_pairs))
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
        self.UHFQC.awgs_0_userregs_0(int(self.nr_shots))
        # The AWG program uses userregs/0 to define the number of iterations
        # in the loop
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg (input avg)


        self.UHFQC.quex_rl_length(self.nr_shots*self.nr_sweep_points)
        self.UHFQC.quex_rl_avgcnt(0)  # log2(1) for single shot readout
        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))

        self.UHFQC.quex_rl_source(self.result_logging_mode_idx)
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def get_values(self):
        if self.always_prepare:
            self.prepare()
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.quex_rl_readout(1)  # resets UHFQC internal readout counters

        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(
            samples=self.nr_shots*self.nr_sweep_points,
            arm=False, acquisition_time=0.01)
        data = np.array([data_raw[key] for key in
                         sorted(data_raw.keys())])*self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'quex_trans_offset_weightfunction_{}'.format(channel))

        # Classify data into qutrit states
        # TODO: make this work for qubits as well!
        classifier_params_list = self.get_values_function_kwargs.get(
                    'classifier_params', None)
        if not isinstance(classifier_params_list, list):
            classifier_params_list = [classifier_params_list]
        state_prob_mtx_list = self.get_values_function_kwargs.get(
                    'state_prob_mtx', None)
        if state_prob_mtx_list is None:
            state_prob_mtx_list = [None]*len(classifier_params_list)
        elif not isinstance(state_prob_mtx_list, list):
            state_prob_mtx_list = [state_prob_mtx_list]

        nr_states = len(self.state_labels)
        classified_data = np.zeros(
            (nr_states*len(self.channel_str_pairs),self.nr_sweep_points))
        if self.get_values_function_kwargs.get('classify', True):
            for i in range(len(self.channel_str_pairs)):
                classified_data[nr_states*i: nr_states*i+nr_states, :] = \
                    self.classify_shots(data[2*i: 2*i+2, :],
                                        classifier_params_list[i],
                                        state_prob_mtx_list[i],
                                        self.get_values_function_kwargs.get(
                                           'average', True))
        return classified_data

    def classify_shots(self, data, classifier_params_dict,
                       state_prob_mtx=None, average=False):
        if classifier_params_dict is None:
            raise ValueError('Please specify the classifier parameters dict.')

        classified_data = a_tools.predict_gm_proba_from_clf(
            data.T, classifier_params_dict)

        if average:
            classified_data = np.mean(np.reshape(
                classified_data, (self.nr_shots, self.nr_sweep_points,
                                  classified_data.shape[1])), axis=0)
        if state_prob_mtx is not None:
            classified_data = np.linalg.inv(state_prob_mtx) @ classified_data.T
            log.info('Data corrected based on state_prob_mtx.')
        else:
            classified_data = classified_data.T

        return classified_data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_statistics_logging_det(Soft_Detector):
    """
    Detector used for the statistics logging mode of the UHFQC
    """

    def __init__(self, UHFQC, AWG, nr_shots: int,
                 integration_length: float,
                 channels: list,
                 statemap: dict,
                 channel_names: list=None,
                 normalize_counts: bool=True):
        """
        Detector for the statistics logger mode in the UHFQC.

            UHFQC (instrument) : data acquisition device
            AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller.
            integration_length (float): integration length in seconds
            nr_shots (int)     : nr of shots, if a number larger than the max
                of the UFHQC (4095) is chosen it will take chunks of 4095.
                The total number of chunks is rounded up.
            channels (list)    : index (channel) of UHFQC weight functions
                to use
            statemap (dict) : dictionary specifying the expected output state
                for each 2 bit input state.
                e.g.:
                    statemap ={'00': '00', '01':'10', '10':'10', '11':'00'}
            normalize_counts (bool) : if False, returns total counts if True
                return fraction of counts
        """
        super().__init__()
        self.UHFQC = UHFQC
        self.AWG = AWG
        self.nr_shots = nr_shots

        self.integration_length = integration_length
        self.normalize_counts = normalize_counts

        self.channels = channels
        if channel_names is None:
            channel_names = ['ch{}'.format(ch) for ch in channels]
        self.value_names = []
        for ch in channel_names:
            ch_value_names = ['{} counts'.format(ch), '{} flips'.format(ch),
                              '{} state errors'.format(ch)]
            self.value_names.extend(ch_value_names)
        self.value_names.append('Total state errors')
        if not self.normalize_counts:
            self.value_units = '#'*len(self.value_names)
        else:
            self.value_units = ['frac']*len(self.value_names)
        self.statemap = statemap

        self.max_shots = 4095  # hardware limit of UHFQC

    @staticmethod
    def statemap_to_array(statemap: dict):
        """
        Converts a statemap dictionary to a numpy array that the
        UHFQC can accept as input.

        Args:
            statemap (dict) : dictionary specifying the expected output state
                for each 2 bit input state.
                e.g.:
                    statemap ={'00': '00', '01':'10', '10':'10', '11':'00'}

        """

        if not statemap.keys() == {'00', '01', '10', '11'}:
            raise ValueError('Invalid statemap: {}'.format(statemap))
        fm = {'00': 0x0,
              '01': 0x1,
              '10': 0x2,
              '11': 0x3}
        arr = np.array([fm[statemap['00']], fm[statemap['01']],
                        fm[statemap['10']], fm[statemap['11']]],
                       dtype=np.uint32)
        return arr

    def prepare(self):
        if self.AWG is not None:
            self.AWG.stop()

        max_shots = 4095  # hardware limit of UHFQC
        self.nr_chunks = self.nr_shots//max_shots + 1
        self.shots_per_chunk = np.min([self.nr_shots, max_shots])

        # The averaging-count is used to specify how many times the AWG program
        # should run
        self.UHFQC.awgs_0_single(1)
        self.UHFQC.awgs_0_userregs_0(self.shots_per_chunk)
        self.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg (input avg)
        # The AWG program uses userregs/0 to define the number of iterations
        # in the loop

        # Configure the results logger not to do any averaging
        # self.UHFQC.quex_rl_length(self.nr_shots)
        self.UHFQC.quex_rl_avgcnt(0)  # log2(1) for single shot readout
        # self.UHFQC.quex_rl_source(0)

        # configure the results logger (rl)
        self.UHFQC.quex_rl_length(4)  # 4 values per channel for stat mode
        self.UHFQC.quex_rl_source(3)  # statistics logging is rl mode "3"

        # Configure the statistics logger (sl)
        self.UHFQC.quex_sl_length(self.shots_per_chunk)
        self.UHFQC.quex_sl_statemap(self.statemap_to_array(self.statemap))

        # above should be all

        self.UHFQC.quex_wint_length(int(self.integration_length*(1.8e9)))
        # I think this should not be there but let's leave it
        #
        self.UHFQC.acquisition_initialize(channels=self.channels, mode='rl')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.acquisition_finalize()

    def acquire_data_point(self, **kw):
        if self.AWG is not None:
            self.AWG.stop()
        data_concat = np.zeros(7)
        for i in range(self.nr_chunks):
            # resets UHFQC internal readout counters
            self.UHFQC.quex_rl_readout(1)
            # resets UHFQC internal sl counters ?
            self.UHFQC.quex_sl_readout(0)

            # self.UHFQC.acquisition_arm()
            # starting AWG
            if self.AWG is not None and i == 0:
                self.AWG.start()

            data = self.UHFQC.acquisition_poll(samples=4,  # double check this
                                               arm=False,
                                               acquisition_time=0.01)

            data_concat += np.concatenate((data[0][:-1], data[1]))

        if self.normalize_counts:
            return data_concat/(self.nr_chunks*self.shots_per_chunk)
        return data_concat


class UHFQC_single_qubit_statistics_logging_det(UHFQC_statistics_logging_det):

    def __init__(self, UHFQC, AWG, nr_shots: int,
                 integration_length: float,
                 channel: int,
                 statemap: dict,
                 channel_name: str=None,
                 normalize_counts: bool=True):
        """
        Detector for the statistics logger mode in the UHFQC.

            UHFQC (instrument) : data acquisition device
            AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller.
            integration_length (float): integration length in seconds
            nr_shots (int)     : nr of shots (max is 4095)
            channel  (int)    : index (channel) of UHFQC weight function
            statemap (dict) : dictionary specifying the expected output state
                for each 1 bit input state.
                e.g.:
                    statemap ={'0': '0', '1':'1'}
            channel_name (str) : optional name of the channel


        """
        super(UHFQC_statistics_logging_det, self).__init__()
        self.UHFQC = UHFQC
        self.AWG = AWG
        self.nr_shots = nr_shots
        self.integration_length = integration_length
        self.normalize_counts = normalize_counts

        self.channels = [channel, int((channel+1) % 5)]

        if channel_name is None:
            channel_name = ['ch{}'.format(channel)]

        self.value_names = ['ch{} flips'.format(channel),
                            'ch{} 1-counts'.format(channel)]
        if not self.normalize_counts:
            self.value_units = '#'*len(self.value_names)
        else:
            self.value_units = ['frac']*len(self.value_names)
        self.statemap = statemap

        self.max_shots = 4095  # hardware limit of UHFQC

        self.statemap = self.statemap_one2two_bit(statemap)

    @staticmethod
    def statemap_one2two_bit(one_bit_sm: dict):
        """
        Converts a one bit statemap to an appropriate dummy 2-bit statemap
        """
        sm = one_bit_sm
        two_bit_sm = {'00': '{}{}'.format(sm['0'], sm['0']),
                      '10': '{}{}'.format(sm['0'], sm['0']),
                      '01': '{}{}'.format(sm['1'], sm['1']),
                      '11': '{}{}'.format(sm['1'], sm['1'])}
        return two_bit_sm

    def acquire_data_point(self, **kw):
        # Returns only the data for the relevant channel and then
        # reverts the order to start with the number of flips
        return super().acquire_data_point()[:2][::-1]

class UHFQC_mixer_calibration_det(UHFQC_integrated_average_detector):
    """
    Based on the "UHFQC_integrated_average" detector.
    generates an AWG seq to measure sideband transmission.
    """

    def __init__(self, UHFQC, station, UHFQC_channels, pulseIch,
                 pulseQch, alpha, phi_skew, f_mod, RO_trigger_channel, RO_pars,
                 amplitude=0.1, nr_averages=2**10, RO_trigger_separation=5e-6,
                 verbose=False,  data_points=1):
        super().__init__(UHFQC, AWG=station.pulsar, integration_length=2.2e-6,
                         nr_averages=nr_averages, channels=UHFQC_channels,
                         real_imag=False, single_int_avg=True,
                         always_prepare=True,
                         seg_per_point=data_points)
        self.name = 'UHFQC_mixer_skewness_det'
        self.station = station
        self.pulseIch = pulseIch
        self.pulseQch = pulseQch
        self.alpha = alpha
        self.phi_skew = phi_skew
        self.f_mod = f_mod
        self.amplitude = amplitude
        self.verbose = verbose
        self.RO_trigger_separation = RO_trigger_separation
        self.RO_trigger_channel = RO_trigger_channel
        self.RO_pars = RO_pars
        self.data_points = data_points
        self.n_measured = 0
        self.seq = None
        self.elts = []
        self.finished = False
        self.channels = ['AWG1_ch3_m2', 'AWG1_ch2_m2', 'AWG1_ch1_m1', 'AWG1_ch1_m2', 'AWG1_ch2_m1', 'AWG1_ch3', 'AWG1_ch4']
        print(self.channels)

    def acquire_data_point(self):
        print(self.n_measured,'::',self.data_points)
        if self.n_measured < self.data_points:
               new_seq, new_elt = cal_elts.mixer_calibration_sequence(
                                        self.RO_trigger_separation,
                                        self.amplitude,
                                        None,
                                        self.RO_pars,
                                        self.pulseIch, self.pulseQch,
                                        f_pulse_mod=self.f_mod,
                                        phi_skew=self.phi_skew(),
                                        alpha=self.alpha(),
                                        upload=False)
               print(new_seq,'::',new_elt)
               new_elt[0].name = '{}-pulse-elt_{}'.\
                                 format(len(new_elt[0].pulses),self.n_measured)
               if self.seq is None:
                   self.seq = sequence.Sequence('Sideband_modulation_seq')
                   self.seq.append_element(*new_elt,trigger_wait=True)
               else:
                   self.seq.append_element(*new_elt,trigger_wait=True)
               self.elts.append(*new_elt)
               self.n_measured += 1
        else:
            logging.warning('The expected number of measurement points is '
                            'exceeded. returning [0] and not adding new sequence'
                            'element. <UHFQC_mixer_calibration_det>')
        if self.n_measured >= self.data_points:
            self.finished = True
            self.station.pulsar.program_awgs(self.seq, *self.elts,
                                             channels=self.channels,
                                             verbose=self.verbose)
            return super().acquire_data_point()
        else:
            return [0]

    def prepare(self, **kw):
        super().prepare(**kw)

    def finish(self):
        super().finish()




class UHFQC_mixer_skewness_det(UHFQC_integrated_average_detector):
    """
    Based on the "UHFQC_integrated_average" detector.
    generates an AWG seq to measure sideband transmission.
    """

    def __init__(self, UHFQC, station, UHFQC_channels, pulseIch,
                 pulseQch, alpha, phi_skew, f_mod, RO_trigger_channel,
                 RO_pars,
                 amplitude=0.1, nr_averages=2**10, RO_trigger_separation=5e-6,
                 verbose=False):
        super().__init__(UHFQC, AWG=station.pulsar, integration_length=2.2e-6,
                         nr_averages=nr_averages, channels=UHFQC_channels,
                         real_imag=False, single_int_avg=True)
        self.name = 'UHFQC_mixer_skewness_det'
        self.station = station
        self.pulseIch = pulseIch
        self.pulseQch = pulseQch
        self.alpha = alpha
        self.phi_skew = phi_skew
        self.f_mod = f_mod
        self.amplitude = amplitude
        self.verbose = verbose
        self.RO_trigger_separation = RO_trigger_separation
        self.RO_trigger_channel = RO_trigger_channel
        self.RO_pars = RO_pars

    def acquire_data_point(self):
        if self.verbose:
            print('alpha: {:.3f}'.format(self.alpha()))
            print('phi_skew: {:.3f}'.format(self.phi_skew()))
        cal_elts.mixer_calibration_sequence(
                self.RO_trigger_separation, self.amplitude, None,
                self.RO_pars,
                self.pulseIch, self.pulseQch, f_pulse_mod=self.f_mod,
                phi_skew=self.phi_skew(), alpha=self.alpha()
            )

        return super().acquire_data_point()

    def prepare(self, **kw):
        super().prepare(**kw)

    def finish(self):
        super().finish()

class UHFQC_readout_mixer_skewness_det(UHFQC_integrated_average_detector):
    def __init__(self, UHFQC, AWG, channels, alpha, phi_skew, f_RO_mod, RO_amp,
                 RO_pulse_length, nr_averages, integration_length=2.2e-6,
                 verbose=False):
        super().__init__(UHFQC, AWG=AWG, integration_length=integration_length,
                         nr_averages=nr_averages, channels=channels,
                         real_imag=False, single_int_avg=True, verbose=verbose)
        self.name = 'UHFQC_readout_mixer_skewness_det'
        self.alpha = alpha
        self.phi_skew = phi_skew
        self.f_RO_mod = f_RO_mod
        self.RO_amp = RO_amp
        self.RO_pulse_length = RO_pulse_length
        self.verbose = verbose

    def acquire_data_point(self):
        if self.verbose:
            print('alpha: {:.3f}'.format(self.alpha()))
            print('phi_skew: {:.3f}'.format(self.phi_skew()))
        self.UHFQC.awg_sequence_acquisition_and_pulse_SSB(
            self.f_RO_mod, self.RO_amp, self.RO_pulse_length,
            alpha=self.alpha(), phi_skew=self.phi_skew())
        return super().acquire_data_point()

    def prepare(self, **kw):
        super().prepare(**kw)

    def finish(self):
        super().finish()

class UHFQC_single_qubit_statistics_logging_det(UHFQC_statistics_logging_det):

    def __init__(self, UHFQC, AWG, nr_shots: int,
                 integration_length: float,
                 channel: int,
                 statemap: dict,
                 channel_name: str=None,
                 normalize_counts: bool=True):
        """
        Detector for the statistics logger mode in the UHFQC.

            UHFQC (instrument) : data acquisition device
            AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller.
            integration_length (float): integration length in seconds
            nr_shots (int)     : nr of shots (max is 4095)
            channel  (int)    : index (channel) of UHFQC weight function
            statemap (dict) : dictionary specifying the expected output state
                for each 1 bit input state.
                e.g.:
                    statemap ={'0': '0', '1':'1'}
            channel_name (str) : optional name of the channel


        """
        super(UHFQC_statistics_logging_det, self).__init__()
        self.UHFQC = UHFQC
        self.AWG = AWG
        self.nr_shots = nr_shots
        self.integration_length = integration_length
        self.normalize_counts = normalize_counts

        self.channels = [channel, int((channel+1) % 5)]

        if channel_name is None:
            channel_name = ['ch{}'.format(channel)]

        self.value_names = ['{} counts'.format(channel),
                            '{} flips'.format(channel),
                            '{} state errors'.format(channel)]
        if not self.normalize_counts:
            self.value_units = '#'*len(self.value_names)
        else:
            self.value_units = ['frac']*len(self.value_names)
        self.statemap = statemap

        self.max_shots = 4095  # hardware limit of UHFQC

        self.statemap = self.statemap_one2two_bit(statemap)

    @staticmethod
    def statemap_one2two_bit(one_bit_sm: dict):
        """
        Converts a one bit statemap to an appropriate dummy 2-bit statemap
        """
        sm = one_bit_sm
        two_bit_sm = {'00': '{}{}'.format(sm['0'], sm['0']),
                      '10': '{}{}'.format(sm['0'], sm['0']),
                      '01': '{}{}'.format(sm['1'], sm['1']),
                      '11': '{}{}'.format(sm['1'], sm['1'])}
        return two_bit_sm

    def acquire_data_point(self, **kw):
        # Returns only the data for the relevant channel
        return super().acquire_data_point()[0:3]

# --------------------------------------------
# Fake detectors
# --------------------------------------------


class Chevron_sim(Hard_Detector):

    """
    Returns a simulated chevron as if it was measured.
    """

    def __init__(self, simulation_dict, **kw):
        super(Chevron_sim, self).__init__(**kw)
        self.simulation_dict = simulation_dict
        self.name = 'Simulated Chevron'
        self.value_names = []
        self.value_units = []

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points
        self.t_start = self.sweep_points[0]
        self.dt = self.sweep_points[1] - self.sweep_points[0]

    def get_values(self):
        return chev_lib.chevron_slice(self.simulation_dict['detuning'],
                                      self.simulation_dict['dist_step'],
                                      self.simulation_dict['g'],
                                      self.t_start,
                                      self.dt,
                                      self.simulation_dict['dist_step'])


class ATS_integrated_average_continuous_detector(Hard_Detector):
    # deprecated

    def __init__(self, ATS, ATS_acq, AWG, seg_per_point=1, normalize=False, rotate=False,
                 nr_averages=1024, integration_length=1e-6, **kw):
        '''
        Integration average detector.
        '''
        super().__init__(**kw)
        self.ATS_acq = ATS_acq
        self.ATS = ATS
        self.name = 'ATS_integrated_average_detector'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        self.AWG = AWG
        self.seg_per_point = seg_per_point
        self.rotate = rotate
        self.normalize = normalize
        self.cal_points = kw.get('cal_points', None)
        self.nr_averages = nr_averages
        self.integration_length = integration_length

    def get_values(self):
        self.AWG.stop()
        self.AWG.start()
        data = self.ATS_acq.acquisition()
        return data

    def rotate_and_normalize(self, data):
        """
        Rotates and normalizes
        """
        if self.cal_points is None:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=data,
                    cal_zero_points=list(range(-4, -2)),
                    cal_one_points=list(range(-2, 0)))
        else:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.measured_values[0:2],
                    cal_zero_points=self.cal_points[0],
                    cal_one_points=self.cal_points[1])
        return self.corr_data, self.corr_data

    def prepare(self, sweep_points):
        self.ATS.config(clock_source='INTERNAL_CLOCK',
                        sample_rate=100000000,
                        clock_edge='CLOCK_EDGE_RISING',
                        decimation=0,
                        coupling=['AC', 'AC'],
                        channel_range=[2., 2.],
                        impedance=[50, 50],
                        bwlimit=['DISABLED', 'DISABLED'],
                        trigger_operation='TRIG_ENGINE_OP_J',
                        trigger_engine1='TRIG_ENGINE_J',
                        trigger_source1='EXTERNAL',
                        trigger_slope1='TRIG_SLOPE_POSITIVE',
                        trigger_level1=128,
                        trigger_engine2='TRIG_ENGINE_K',
                        trigger_source2='DISABLE',
                        trigger_slope2='TRIG_SLOPE_POSITIVE',
                        trigger_level2=128,
                        external_trigger_coupling='AC',
                        external_trigger_range='ETR_5V',
                        trigger_delay=0,
                        timeout_ticks=0
                        )
        self.ATS.update_acquisitionkwargs(samples_per_record=1024,
                                          records_per_buffer=70,
                                          buffers_per_acquisition=self.nr_averages,
                                          channel_selection='AB',
                                          transfer_offset=0,
                                          external_startcapture='ENABLED',
                                          enable_record_headers='DISABLED',
                                          alloc_buffers='DISABLED',
                                          fifo_only_streaming='DISABLED',
                                          interleave_samples='DISABLED',
                                          get_processed_data='DISABLED',
                                          allocated_buffers=self.nr_averages,
                                          buffer_timeout=1000
                                          )

    def finish(self):
        pass


# DDM detector functions
class DDM_input_average_detector(Hard_Detector):

    '''
    Detector used for acquiring averaged input traces withe the DDM

    '''

    '''
    Detector used for acquiring single points of the DDM while externally
    triggered by the AWG.
    Soft version of the regular integrated avg detector.
    # not yet pair specific
    '''

    def __init__(self, DDM, AWG, channels=[1, 2], nr_averages=1024, nr_samples=1024, **kw):
        super(DDM_input_average_detector, self).__init__()

        self.DDM = DDM
        self.name = 'DDM_input_averaging_data'
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = 'ch{}'.format(channel)
            self.value_units[i] = 'V'
        self.AWG = AWG
        self.nr_samples = nr_samples
        self.nr_averages = nr_averages

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        self.DDM.ch_pair1_inavg_scansize.set(self.nr_samples)
        self.DDM.ch_pair1_inavg_Navg(self.nr_averages)
        self.nr_sweep_points = self.nr_samples

    def get_values(self):
        # arming DDM trigger
        self.DDM.ch_pair1_inavg_enable.set(1)
        self.DDM.ch_pair1_run.set(1)
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()
        # polling the data, function checks that measurement is finished
        data = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            data[i] = eval("self.DDM.ch{}_inavg_data()".format(channel))/127
        return data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class DDM_integrated_average_detector(Hard_Detector):

    '''
    Detector used for integrated average results with the DDM

    '''

    def __init__(self, DDM, AWG, integration_length=1e-6, nr_averages=1024, rotate=False,
                 channels=[1, 2, 3, 4, 5], crosstalk_suppression=False,
                 **kw):
        super(DDM_integrated_average_detector, self).__init__()
        self.DDM = DDM
        self.name = 'DDM_integrated_average'
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        self.cal_points = kw.get('cal_points', None)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = 'w{}'.format(channel)
            self.value_units[i] = 'V'
        self.rotate = rotate
        self.AWG = AWG
        self.nr_averages = nr_averages
        self.integration_length = integration_length
        self.rotate = rotate
        self.crosstalk_suppression = crosstalk_suppression
        self.scaling_factor = 1/(500e6*integration_length)/127

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()
        if sweep_points is None:
            self.nr_sweep_points = 1
        else:
            self.nr_sweep_points = len(sweep_points)
        # this sets the result to integration and rotation outcome
        for i, channel in enumerate(self.channels):
            eval("self.DDM.ch_pair1_weight{}_wint_intlength({})".format(
                channel, self.integration_length*500e6))
        self.DDM.ch_pair1_tvmode_naverages(self.nr_averages)
        self.DDM.ch_pair1_tvmode_nsegments(self.nr_sweep_points)

    def get_values(self):
        # arming DDM trigger
        self.DDM.ch_pair1_tvmode_enable.set(1)
        self.DDM.ch_pair1_run.set(1)
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()
        # polling the data, function checks that measurement is finished
        data = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            data[i] = eval("self.DDM.ch_pair1_weight{}_tvmode_data()".format(
                channel))*self.scaling_factor
        if self.rotate:
            return self.rotate_and_normalize(data)
        else:
            return data

    def acquire_data_point(self):
        return self.get_values()

    def rotate_and_normalize(self, data):
        """
        Rotates and normalizes
        """
        if self.cal_points is None:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=data,
                    cal_zero_points=list(range(-4, -2)),
                    cal_one_points=list(range(-2, 0)))
        else:
            self.corr_data, self.zero_coord, self.one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.measured_values[0:2],
                    cal_zero_points=self.cal_points[0],
                    cal_one_points=self.cal_points[1])
        return self.corr_data, self.corr_data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class DDM_integration_logging_det(Hard_Detector):

    '''
    Detector used for integrated average results with the UHFQC

    '''

    def __init__(self, DDM, AWG, integration_length=1e-6,
                 channels=[1, 2], nr_shots=4096, **kw):
        super(DDM_integration_logging_det, self).__init__()
        self.DDM = DDM
        self.name = 'DDM_integration_logging_det'
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = 'w{}'.format(channel)
            self.value_units[i] = 'V'
        if len(self.channels) == 2:
            self.value_names = ['I', 'Q']
            self.value_units = ['V', 'V']
        self.AWG = AWG
        self.integration_length = integration_length
        self.nr_shots = nr_shots
        self.scaling_factor = 1/(500e6*integration_length)/127

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        if sweep_points is None:
            self.nr_sweep_points = 1
        else:
            self.nr_sweep_points = len(sweep_points)
        # this sets the result to integration and rotation outcome
        for i, channel in enumerate(self.channels):
            eval("self.DDM.ch_pair1_weight{}_wint_intlength({})".format(
                channel, self.integration_length*500e6))
        self.DDM.ch_pair1_logging_nshots(self.nr_shots)

    def get_values(self):
        # arming DDM trigger
        self.DDM.ch_pair1_logging_enable.set(1)
        self.DDM.ch_pair1_run.set(1)
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()
        # polling the data, function checks that measurement is finished
        data = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            data[i] = eval("self.DDM.ch_pair1_weight{}_logging_int()".format(
                channel))*self.scaling_factor
        return data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
