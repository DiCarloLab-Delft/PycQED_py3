"""
UHFQC related detector functions
extracted from pycqed/measurement/detector_functions.py commit 0da380ad2adf2dc998f5effef362cdf264b87948
"""

import logging
import time
import numpy as np
import numpy.fft as fft
from string import ascii_uppercase
from deprecated import deprecated

from pycqed.measurement.det_fncs.Base import Soft_Detector, Hard_Detector, Multi_Detector

# import instruments for type annotations
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController import UHFQC

log = logging.getLogger(__name__)


class Multi_Detector_UHF(Multi_Detector):
    """
    Special multi detector
    """

    def get_values(self):
        # Since master (holding cc object) is first in self.detectors,
        self.detectors[0].AWG.stop()

        # Prepare and arm
        for detector in self.detectors:
            # Ramiro pointed out that prepare gets called by MC
            # detector.prepare()
            detector.arm()
            detector.UHFQC.sync()

        # Run (both in parallel and implicitly)
        self.detectors[0].AWG.start()

        # Define the timeout as the timeout defined for the first UHF
        timeout = self.detectors[0].UHFQC.timeout()

        # Get data
        # Initialize the dictionaries to store the data from the detector
        data_raw = []
        gotem = []
        for detector in self.detectors:
            data_raw.append({k: [] for k, _ in enumerate(detector.UHFQC._acquisition_nodes)})
            gotem.append([False]*len(detector.UHFQC._acquisition_nodes))

        start_time = time.time()
        # Outer loop: repeat until all results are acquired or timeout is reached
        while (time.time() - start_time) < timeout and not all(all(g) for g in gotem):

            # Inner loop over detectors
            for m, detector in enumerate(self.detectors):

                # Poll the data with a short interval
                poll_interval_seconds = 0.010
                dataset = detector.UHFQC.poll(poll_interval_seconds)

                # Loop over the nodes (channels) of the detector
                for n, p in enumerate(detector.UHFQC._acquisition_nodes):

                    # check if the node is in the dataset returned by the poll() function
                    if p in dataset:

                        # Note: we only expect one vector per node (m: detector, n: channel)
                        data_raw[m][n] = dataset[p][0]['vector']

                        # check if the vector has the right length
                        if len(data_raw[m][n]) == detector.get_num_samples():
                            gotem[m][n] = True

        # Error handling
        if not all(all(g) for g in gotem):
            for m, detector in enumerate(self.detectors):
                detector.UHFQC.acquisition_finalize()
                for n, _c in enumerate(detector.UHFQC._acquisition_nodes):
                    if n in data_raw[m]:
                        print("\t{}: Channel {}: Got {} of {} samples".format(
                            detector.UHFQC.devname, n, len(data_raw[m][n]), detector.get_num_samples()))
                raise TimeoutError("Error: Didn't get all results!")

        # Post-process the data
        # Note: the detector must feature the get_values_postprocess() function
        # to be used within the multi-detector
        values_list = []
        for m, detector in enumerate(self.detectors):
            values_list.append(detector.get_values_postprocess(data_raw[m]))

        # Pad all result vectors for them to have equal length.
        maximum = 0
        minimum = len(values_list[0][0])   #left index of values_list: detector; right index: channel
        for feedline in values_list:
            for result in feedline:
                if len(result)>maximum: maximum=len(result)
                if len(result)<minimum: minimum=len(result)
        if maximum != minimum:
            padded_values_list = []
            for index, feedline in enumerate(values_list):
                padded_values_list.append([])
                for result in feedline:
                    padded_values_list[index].append(np.pad(result, (0, maximum-len(result))))
            values_list = [np.array(values) for values in padded_values_list]

        values = np.concatenate(values_list)
        return values

    def acquire_data_point(self):
        # N.B. get_values and acquire_data point are virtually identical.
        # the only reason for their existence is a historical distinction
        # between hard and soft detectors that leads to some confusing data
        # shape related problems, hence the append vs concatenate

        # FIXME: It is not clear if this construction works with multiple
        # segments
        return self.get_values().flatten()


class UHFQC_input_average_detector(Hard_Detector):

    '''
    Detector used for acquiring averaged input traces withe the UHFQC
    '''

    def __init__(
            self,
            UHFQC: UHFQC,
            AWG: CC = None,
            channels=(0, 1),
            nr_averages=1024,
            nr_samples=4096,
            **kw
    ):
        super().__init__()
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
        self.nr_samples = nr_samples
        self.nr_averages = nr_averages

    def _get_readout(self):
        return sum([(1 << c) for c in self.channels])

    def get_values(self):
        self.UHFQC.acquisition_arm()

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
        self.nr_sweep_points = self.nr_samples
        self.UHFQC.acquisition_initialize(
            samples=self.nr_samples, averages=self.nr_averages,
            channels=self.channels, mode='iavg')

    def finish(self):
        self.UHFQC.acquisition_finalize()
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_demodulated_input_avg_det(UHFQC_input_average_detector):
    '''
    Detector used for acquiring averaged input traces withe the UHFQC.
    Additionally trace are demodulated.
    '''

    def __init__(
            self,
            f_RO_mod,
            UHFQC: UHFQC,
            real_imag=True,
            AWG: CC=None,
            channels=(0, 1),
            nr_averages=1024,
            nr_samples=4096,
            **kw
    ):
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


@deprecated(version='0.4', reason="broken code")
class UHFQC_spectroscopy_detector(Soft_Detector):
    '''
    Detector used for the spectroscopy mode
    '''

    def __init__(
            self,
            UHFQC: UHFQC,
            ro_freq_mod,
            AWG=None,
            channels=(0, 1),
            nr_averages=1024,
            integration_length=4096,
            **kw
    ):
        super().__init__()
        # FIXME: code commented out, some __init__ parameters no longer used
        #UHFQC=UHFQC, AWG=AWG, channels=channels,
        # nr_averages=nr_averages, nr_samples=nr_samples, **kw
        self.UHFQC = UHFQC
        self.ro_freq_mod = ro_freq_mod

    def acquire_data_point(self):
        RESULT_LENGTH = 1600  # FIXME: hardcoded
        vals = self.UHFQC.acquisition(
            samples=RESULT_LENGTH, acquisition_time=0.010, timeout=10)
        a = max(np.abs(fft.fft(vals[0][1:int(RESULT_LENGTH/2)])))
        b = max(np.abs(fft.fft(vals[1][1:int(RESULT_LENGTH/2)])))
        return a+b


class UHFQC_integrated_average_detector(Hard_Detector):

    '''
    Detector used for integrated average results with the UHFQC
    '''

    def __init__(
            self,
            UHFQC: UHFQC,
            AWG: CC = None,
            integration_length: float = 1e-6,
            nr_averages: int = 1024,
            channels: list = (0, 1, 2, 3),
            result_logging_mode: str = 'raw',
            real_imag: bool = True,
            value_names: list = None,
            seg_per_point: int = 1,
            single_int_avg: bool = False,
            chunk_size: int = None,
            values_per_point: int = 1,
            values_per_point_suffex: list = None,
            always_prepare: bool = False,
            prepare_function=None,
            prepare_function_kwargs: dict = None,
            **kw
    ):
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
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            if value_names is None:
                self.value_names[i] = '{} w{}'.format(result_logging_mode,
                                                      channel)
            else:
                self.value_names[i] = 'w{} {}'.format(channel,
                                                      value_names[i])
        if result_logging_mode == 'raw':
            # Units are only valid when using SSB or DSB demodulation.
            # value corrsponds to the peak voltage of a cosine with the
            # demodulation frequency.
            self.value_units = ['Vpeak']*len(self.channels)
            self.scaling_factor = 1/(1.8e9*integration_length)
        elif result_logging_mode == 'lin_trans':
            self.value_units = ['a.u.']*len(self.channels)
            self.scaling_factor = 1

        elif result_logging_mode == 'digitized':
            self.value_units = ['frac']*len(self.channels)
            self.scaling_factor = 1

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

        rounded_nr_averages = 2**int(np.log2(nr_averages))
        if rounded_nr_averages != nr_averages:
            log.warning("nr_averages must be a power of 2, rounded to {} (from {}) ".format(
                rounded_nr_averages, nr_averages))

        self.nr_averages = rounded_nr_averages
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

        if not self.real_imag:
            if len(self.channels) % 2 != 0:
                raise ValueError('Length of "{}" is not even'.format(
                                 self.channels))
            for i in range(len(self.channels)//2):
                self.value_names[2*i] = 'Magn'
                self.value_names[2*i+1] = 'Phase'
                self.value_units[2*i+1] = 'deg'

    def _get_readout(self):
        return sum([(1 << c) for c in self.channels])

    def arm(self):
        # resets UHFQC internal readout counters
        self.UHFQC.acquisition_arm()
        self.UHFQC.sync()

    def get_num_samples(self):
        return self.nr_sweep_points

    def get_values(self, arm=True, is_single_detector=True):
        if is_single_detector:
            if self.always_prepare:
                self.prepare()

            if self.AWG is not None:
                self.AWG.stop()

            if arm:
                self.arm()
                self.UHFQC.sync()

            # starting AWG
            if self.AWG is not None:
                self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(samples=self.get_num_samples(), arm=False, acquisition_time=0.01)

        return self.get_values_postprocess(data_raw)

    def get_values_postprocess(self, data_raw):
        # if len(data_raw[next(iter(data_raw))])>1:
        #     print('[DEBUG UHF SWF] SHOULD HAVE HAD AN ERROR')
        # data = np.array([data_raw[key]
        data = np.array([data_raw[key]
                         for key in sorted(data_raw.keys())])*self.scaling_factor
        # log.debug('[UHF detector] RAW shape',[data_raw[key]
        #                  for key in sorted(data_raw.keys())])
        # log.debug('[UHF detector] shape 1',data.shape)

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
        """
        Convert data to polar coordinates,
        assuming that the channels in ascencing are used as pairs of I, Q
        """
        if len(data) % 2 != 0:
            raise ValueError('Expect even number of channels for rotation. Got {}'.format(
                             len(data)))
        for i in range(len(data)//2):
            I, Q = data[2*i], data[2*i+1]
            S21 = I + 1j*Q
            data[2*i] = np.abs(S21)
            data[2*i+1] = np.angle(S21)/(2*np.pi)*360
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

        self.UHFQC.qas_0_integration_length(int(self.integration_length*self.UHFQC.clock_freq()))
        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.acquisition_initialize(
            samples=self.nr_sweep_points,
            averages=self.nr_averages,
            channels=self.channels,
            mode='rl'
        )

    def finish(self):
        self.UHFQC.acquisition_finalize()

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

    def __init__(
            self,
            UHFQC: UHFQC,
            AWG: CC = None,
            integration_length=1e-6,
            nr_averages=1024,
            rotate=False,
            real_imag=True,
            channels: list = [0, 1],
            correlations: list = [(0, 1)],
            value_names=None,
            seg_per_point=1,
            single_int_avg=False,
            thresholding=False,
            **kw
    ):
        super().__init__(
            UHFQC, AWG=AWG, integration_length=integration_length,
            nr_averages=nr_averages, real_imag=real_imag,
            channels=channels,
            seg_per_point=seg_per_point, single_int_avg=single_int_avg,
            result_logging_mode='lin_trans',
            **kw)
        self.correlations = correlations
        self.thresholding = thresholding

        if value_names is None:
            self.value_names = []
            for ch in channels:
                self.value_names += ['w{}'.format(ch)]
        else:
            self.value_names = value_names

        if not thresholding:
            # N.B. units set to a.u. as the lin_trans matrix and optimal weights
            # are always on for the correlation mode to work.
            self.value_units = ['a.u.']*len(self.value_names) + \
                               ['a.u.']*len(self.correlations)
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

        self.UHFQC.qas_0_integration_length(int(self.integration_length*(self.UHFQC.clock_freq())))
        self.set_up_correlation_weights()
        self.UHFQC.acquisition_initialize(
            samples=self.nr_sweep_points,
            averages=self.nr_averages,
            channels=self.channels,
            mode='rl'
        )

    def define_correlation_channels(self):
        self.correlation_channels = []
        for corr in self.correlations:
            # Start by assigning channels
            if corr[0] not in self.channels or corr[1] not in self.channels:
                raise ValueError('Correlations should be in channels')

            correlation_channel = -1

            # 10 is the (current) max number of weights in the UHFQC (release 19.05)
            for ch in range(10):
                if ch in self.channels:
                    # Disable correlation mode as this is used for normal acquisition
                    self.UHFQC.set('qas_0_correlations_{}_enable'.format(ch), 0)

                # Find the first unused channel to set up as correlation
                if ch not in self.channels:
                    # selects the lowest available free channel
                    self.channels += [ch]
                    correlation_channel = ch

                    # correlation mode is turned on in the
                    # set_up_correlation_weights method
                    break
                    # FIXME, can currently only use one correlation

            if correlation_channel < 0:
                raise ValueError('No free channel available for correlation.')

            self.correlation_channels += [correlation_channel]

    def set_up_correlation_weights(self):
        if self.thresholding:
            # correlations mode after threshold
            # NOTE: thresholds need to be set outside the detector object.
            self.UHFQC.qas_0_result_source(5)
            log.info('Setting {} result source to 5 (corr threshold)'.format(self.UHFQC.name))
        else:
            # correlations mode before threshold
            self.UHFQC.qas_0_result_source(4)
            log.info('Setting {} result source to 4 (corr no threshold)'.format(self.UHFQC.name))
        # Configure correlation mode
        for correlation_channel, corr in zip(self.correlation_channels, self.correlations):
            # Duplicate source channel to the correlation channel and select
            # second channel as channel to correlate with.
            log.info('Setting w{} on {} as correlation weight for w{}'.format(correlation_channel, self.UHFQC.name, corr))

            log.debug('Copying weights of w{} to w{}'.format(corr[0], correlation_channel))

            copy_int_weights_real = self.UHFQC.get('qas_0_integration_weights_{}_real'.format(corr[0]))
            copy_int_weights_imag = self.UHFQC.get('qas_0_integration_weights_{}_imag'.format(corr[0]))
            self.UHFQC.set('qas_0_integration_weights_{}_real'.format(correlation_channel), copy_int_weights_real)
            self.UHFQC.set('qas_0_integration_weights_{}_imag'.format(correlation_channel), copy_int_weights_imag)

            copy_rot = self.UHFQC.get('qas_0_rotations_{}'.format(corr[0]))
            self.UHFQC.set('qas_0_rotations_{}'.format(correlation_channel), copy_rot)

            log.debug('Setting correlation source of w{} to w{}'.format(correlation_channel, corr[1]))
            # Enable correlation mode one the correlation output channel and
            # set the source to the second source channel
            self.UHFQC.set('qas_0_correlations_{}_enable'.format(correlation_channel), 1)
            self.UHFQC.set('qas_0_correlations_{}_source'.format(correlation_channel), corr[1])

            # If thresholding is enabled, set the threshold for the correlation channel.
            if self.thresholding:
                # Because correlation happens after threshold, the threshold
                # has to be set to whatever the weight function is set to.
                log.debug('Copying threshold for w{} to w{}'.format(corr[0],  correlation_channel))
                thresh_level = self.UHFQC.get('qas_0_thresholds_{}_level'.format(corr[0]))
                self.UHFQC.set('qas_0_thresholds_{}_level'.format(correlation_channel), thresh_level)

    def get_values(self):

        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.acquisition_arm()
        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        data_raw = self.UHFQC.acquisition_poll(samples=self.nr_sweep_points, arm=False, acquisition_time=0.01)

        data = []

        for key in sorted(data_raw.keys()):
            data.append(np.array(data_raw[key]))
        # Scale factor for correlation mode is always 1.
        # The correlation mode only supports use in optimal weights,
        # in which case we do not scale by the integration length.

        return data


class UHFQC_integration_logging_det(Hard_Detector):

    '''
    Detector used for integrated average results with the UHFQC

    '''

    def __init__(
            self,
            UHFQC: UHFQC,
            AWG: CC = None,
            integration_length: float = 1e-6,
            nr_shots: int = 4094,
            channels: list = (0, 1),
            result_logging_mode: str = 'raw',
            value_names: list = None,
            always_prepare: bool = False,
            prepare_function=None,
            prepare_function_kwargs: dict = None,
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
        self.name = '{}_UHFQC_integration_logging_det'.format(result_logging_mode)
        self.channels = channels

        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            if value_names is None:
                self.value_names[i] = '{} w{}'.format(result_logging_mode, channel)
            else:
                self.value_names[i] = 'w{} {}'.format(channel, value_names[i])

        if result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.channels)
            self.scaling_factor = 1 / (1.8e9*integration_length)
            # N.B. this ensures correct units of Volt but beware that
            # to set the acq threshold one needs to correct for this
            # scaling factor. Note that this should never be needed as
            # digitized mode is supposed to work with optimal weights.
            log.debug('Setting scale factor for int log to {}'.format(self.scaling_factor))
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

    def _get_readout(self):
        return sum([(1 << c) for c in self.channels])

    def arm(self):
        # UHFQC internal readout counters reset as part of the call to acquisition_initialize
        self.UHFQC.acquisition_arm()
        self.UHFQC.sync()

    def get_num_samples(self):
        return self.nr_shots

    def get_values(self, arm=True, is_single_detector=True):
        if is_single_detector:
            if self.always_prepare:
                self.prepare()

            if self.AWG is not None:
                self.AWG.stop()

            if arm:
                self.arm()
                self.UHFQC.sync()

            # starting AWG
            if self.AWG is not None:
                self.AWG.start()

        # Get the data
        data_raw = self.UHFQC.acquisition_poll(samples=self.get_num_samples(), arm=False, acquisition_time=0.01)
        return self.get_values_postprocess(data_raw)

    def get_values_postprocess(self, data_raw):
        data = np.array([data_raw[key]
        # data = np.array([data_raw[key][-1]
                         for key in sorted(data_raw.keys())])*self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UHFQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i] - self.UHFQC.get('qas_0_trans_offset_weightfunction_{}'.format(channel))
        return data

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()

        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

        self.UHFQC.qas_0_integration_length(int(self.integration_length*(1.8e9)))
        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.acquisition_initialize(samples=self.nr_shots, averages=1, channels=self.channels, mode='rl')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()


class UHFQC_statistics_logging_det(Soft_Detector):
    """
    Detector used for the statistics logging mode of the UHFQC
    """

    def __init__(
            self,
            UHFQC: UHFQC,
            AWG: CC,
            nr_shots: int,
            integration_length: float,
            channels: list,
            statemap: dict,
            channel_names: list = None,
            normalize_counts: bool = True
    ):
        """
        Detector for the statistics logger mode in the UHFQC.

            UHFQC (instrument) : data acquisition device
            AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller.
            integration_length (float): integration length in seconds
            nr_shots (int)     : nr of shots
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

        self.max_shots = 4095  # hardware limit of UHFQC. FIXME: no longer true, but apparently unused

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

        # Configure the statistics logger (sl)
        self.UHFQC.qas_0_result_statistics_enable(1)
        self.UHFQC.qas_0_result_statistics_length(self.nr_shots)
        self.UHFQC.qas_0_result_statistics_statemap(self.statemap_to_array(self.statemap))

        # above should be all
        self.UHFQC.qas_0_integration_length(int(self.integration_length*(1.8e9)))
        self.UHFQC.acquisition_initialize(samples=self.nr_shots, averages=1, channels=self.channels, mode='rl')

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
        self.UHFQC.acquisition_finalize()

    def _get_readout(self):
        return sum([(1 << c) for c in self.channels])

    def acquire_data_point(self, **kw):
        if self.AWG is not None:
            self.AWG.stop()

        self.UHFQC.acquisition_arm()

        # starting AWG
        if self.AWG is not None:
            self.AWG.start()

        # We don't actually care about this result
        self.UHFQC.acquisition_poll(
            samples=self.nr_shots,  # double check this
            arm=False,
            acquisition_time=0.01
        )

        # Get statistics of the individual channels
        data = np.array([])
        for c in self.channels:
            ones = self.UHFQC.get('qas_0_result_statistics_data_{}_ones'.format(c))
            flips = self.UHFQC.get('qas_0_result_statistics_data_{}_flips'.format(c))
            errors = self.UHFQC.get('qas_0_result_statistics_data_{}_errors'.format(c))
            data = np.concatenate((data, np.array([ones, flips, errors])))

        # Add total number of state errors at the end
        stateerrors = self.UHFQC.get('qas_0_result_statistics_stateerrors')
        data = np.concatenate((data, np.array([stateerrors])))

        if self.normalize_counts:
            return data/(self.nr_shots)

        return data


class UHFQC_single_qubit_statistics_logging_det(UHFQC_statistics_logging_det):

    def __init__(
            self,
            UHFQC: UHFQC,
            AWG: CC,
            nr_shots: int,
            integration_length: float,
            channel: int,
            statemap: dict,
            channel_name: str = None,
            normalize_counts: bool = True
    ):
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
        super().__init__(
            UHFQC=UHFQC,
            AWG=AWG,
            nr_shots=nr_shots,
            integration_length=integration_length,
            channels=[channel],
            statemap=UHFQC_single_qubit_statistics_logging_det.statemap_one2two_bit(statemap),
            channel_names=[channel_name if channel_name is not None else 'ch{}'.format(channel)],
            normalize_counts=normalize_counts
        )

        self.value_names = ['ch{} flips'.format(channel),
                            'ch{} 1-counts'.format(channel)]

        if not self.normalize_counts:
            self.value_units = '#'*len(self.value_names)
        else:
            self.value_units = ['frac']*len(self.value_names)

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
