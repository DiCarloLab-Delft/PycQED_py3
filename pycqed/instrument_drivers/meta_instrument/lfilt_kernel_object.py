"""
This file contains an instrument for correcting distortions
using linear filtering (scipy.signal.lfilter).

It is based on the kernel_object.DistortionsKernel
"""
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

        for i in range(self._num_models):
            self.add_parameter('filter_model_{:02}'.format(i),
                               parameter_class=ManualParameter,
                               initial_value={},
                               vals=vals.Dict())

    def distort_waveform(self, waveform):
        filter_models = []
        y_sig = waveform
        for filt_id in range(self._num_models):
            filt = self.get('filter_model_{:02}'.format(filt_id))

            if not filt:
                pass  # dict is empty
            else:
                model = filt['model']
                if not self.cfg_hardware_friendly():
                    if model == 'bias_tee':
                        y_sig = kf.bias_tee_correction(
                            y_sig, sampling_rate=self.cfg_sampling_rate(),
                            **filt['params'])
                    elif model == 'exponential_decay':
                        y_sig = kf.exponential_decay_correction(
                            y_sig, sampling_rate=self.cfg_sampling_rate(),
                            **filt['params'])
                    else:
                        raise KeyError()
                else:
                    raise NotImplementedError()

        return y_sig
