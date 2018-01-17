"""
This file contains an instrument for correcting distortions
using linear filtering (scipy.signal.lfilter).

It is based on the kernel_object.DistortionsKernel
"""
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.measurement import kernel_functions_ZI as kf


class LinDistortionKernel(Instrument):

    def __init__(self, name, num_models=10, **kw):
        super().__init__(name, **kw)
        self._num_models = num_models

        self.add_parameter('cfg_hardware_friendly',
                           initial_value=False,
                           parameter_class=ManualParameter,
                           vals=vals.Bool())
        self.add_parameter('cfg_sampling_rate',
                           parameter_class=ManualParameter,
                           initial_value=1e9,
                           vals=vals.Numbers())

        self.add_parameter('cfg_gain_correction',
                           parameter_class=ManualParameter,
                           initial_value=1,
                           vals=vals.Numbers())

        for i in range(self._num_models):
            self.add_parameter('filter_model_{:02}'.format(i),
                               parameter_class=ManualParameter,
                               initial_value={},
                               vals=vals.Dict())

    def reset_kernels(self):
        """
        Resets all kernels to an empty dict so no distortion is applied.
        """
        for filt_id in range(self._num_models):
            self.set('filter_model_{:02}'.format(filt_id), {})

    def get_first_empty_filter(self):
        """
        Resets all kernels to an empty dict so no distortion is applied.
        """
        for filt_id in range(self._num_models):
            if self.get('filter_model_{:02}'.format(filt_id)) == {}:
                return filt_id
        raise ValueError('No empty filter')


    def distort_waveform(self, waveform, length_samples: int=None):
        filter_models = []
        if length_samples!= None:
            extra_samples = length_samples - len(waveform)
            y_sig = np.concatenate([waveform, np.zeros(extra_samples)])
        else:
            y_sig = waveform
        for filt_id in range(self._num_models):
            filt = self.get('filter_model_{:02}'.format(filt_id))

            if not filt:
                pass  # dict is empty
            else:
                model = filt['model']
                if not self.cfg_hardware_friendly():
                    if model == 'high-pass':
                        y_sig = kf.bias_tee_correction(
                            y_sig, sampling_rate=self.cfg_sampling_rate(),
                            **filt['params'])
                    elif model == 'exponential':
                        y_sig = kf.exponential_decay_correction(
                            y_sig, sampling_rate=self.cfg_sampling_rate(),
                            **filt['params'])
                    else:
                        raise KeyError('Model {} not recognized'.format(model))
                else:
                    raise NotImplementedError()
        y_sig *= self.cfg_gain_correction()
        return y_sig


    def print_overview(self):
        print("*"*80)
        print("Overview of {}".format(self.name))
        for filt_id in range(self._num_models):

            filt = self.get('filter_model_{:02}'.format(filt_id))
            if filt != {}:
                model = filt['model']
                params= filt['params']

                print('Model {} {}: \n {}'.format(filt_id, model, params))

        print("*"*80)