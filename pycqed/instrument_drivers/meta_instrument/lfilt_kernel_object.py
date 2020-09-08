"""
This file contains an instrument for correcting distortions
using linear filtering (scipy.signal.lfilter) and/or setting
the real-time distortion corrections in the HDAWG instrument.

It is based on the kernel_object.DistortionsKernel
"""
import numpy as np
import logging
from scipy import signal
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter

from pycqed.measurement import kernel_functions_ZI as kf


class LinDistortionKernel(Instrument):
    def __init__(self, name, num_models=10, **kw):
        super().__init__(name, **kw)
        self._num_models = num_models

        self.add_parameter(
            "cfg_sampling_rate",
            parameter_class=ManualParameter,
            initial_value=1e9,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "cfg_gain_correction",
            parameter_class=ManualParameter,
            initial_value=1,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "instr_AWG",
            parameter_class=InstrumentRefParameter,
            docstring="Used in combination with the real-time "
            "predistortion filters of the ZI HDAWG",
        )
        self.add_parameter(
            "cfg_awg_channel",
            parameter_class=ManualParameter,
            vals=vals.Ints(),
            docstring="Used in combination with the real-time "
            "predistortion filters of the ZI HDAWG",
        )

        for i in range(self._num_models):
            self.add_parameter(
                "filter_model_{:02}".format(i),
                parameter_class=ManualParameter,
                initial_value={},
                vals=vals.Dict(),
            )

    def reset_kernels(self):
        """
        Resets all kernels to an empty dict so no distortion is applied.
        """
        for filt_id in range(self._num_models):
            self.set("filter_model_{:02}".format(filt_id), {})

    def get_first_empty_filter(self):
        """
        Resets all kernels to an empty dict so no distortion is applied.
        """
        for filt_id in range(self._num_models):
            if self.get("filter_model_{:02}".format(filt_id)) == {}:
                return filt_id
        raise ValueError("No empty filter")

    def get_number_of_realtime_filters(self):
        rt_exp_models = 0
        rt_fir_models = 0
        rt_bounce_models = 0
        for filt_id in range(self._num_models):

            filt = self.get("filter_model_{:02}".format(filt_id))
            if filt != {}:
                model = filt["model"]
                params = filt["params"]
                if (filt["model"] == "FIR") and (filt["real-time"] is True):
                    rt_fir_models += 1
                elif (filt["model"] == "exponential") and (filt["real-time"] is True):
                    rt_exp_models += 1
                elif (filt["model"] == "bounce") and (filt["real-time"] is True):
                    rt_bounce_models += 1
        return {
            "rt_exp_models": rt_exp_models,
            "rt_fir_models": rt_fir_models,
            "rt_bounce_models": rt_bounce_models,
        }

    def set_unused_realtime_distortions_zero(self):
        """
        Turns off all unused real-time distortion filters by setting their
        amplitude to zero. This method of disabling is used so as not to
        change the latency that is introduced.
        """
        max_exp_filters = 5
        try:
            AWG = self.instr_AWG.get_instr()
        except Exception as e:
            logging.warning(e)
            logging.warning("Could not set realtime distortions to 0, AWG not found")
            return

        # Returns a dict with filter type and number of that type
        nr_filts = self.get_number_of_realtime_filters()

        # set exp_filters to 0
        for i in range(max_exp_filters):
            if i >= nr_filts["rt_exp_models"]:
                AWG.set(
                    "sigouts_{}_precompensation_exponentials_{}_amplitude".format(
                        self.cfg_awg_channel() - 1, i
                    ),
                    0,
                )

        # set bounce filters to 0
        if nr_filts["rt_bounce_models"] == 0:
            AWG.set(
                "sigouts_{}_precompensation_bounces_{}_enable".format(
                    self.cfg_awg_channel() - 1, 0
                ),
                0,
            )

        # Reset

        # 'FIXME: FIR filter reset is disabled because of #148'
        if nr_filts["rt_fir_models"] == 0:
            impulse_resp = np.zeros(40)
            impulse_resp[0] = 1
            AWG.set(
                "sigouts_{}_precompensation_fir_coefficients".format(
                    self.cfg_awg_channel() - 1
                ),
                impulse_resp,
            )

        # set bias-tee filters to 0
        pass  # Currently broken

    def distort_waveform(
        self, waveform, length_samples: int = None, inverse: bool = False
    ):
        """
        Distorts a waveform using the models specified in the Kernel Object.

        Args:
            waveform (array)    : waveform to be distorted
            lenght_samples (int): number of samples after which to cut of wf
            inverse (bool)      : if True apply the inverse of the waveform.

        Return:
            y_sig (array)       : waveform with distortion filters applied

        N.B. The bounce correction does not have an inverse implemented
            (June 2018) MAR
        N.B.2 The real-time FIR also does not have an inverse implemented.
            (May 2019) MAR
        N.B.3 the real-time distortions are reset and set on the HDAWG every
            time a waveform is distorted. This is a suboptimal workflow.

        """
        if length_samples is not None:
            extra_samples = length_samples - len(waveform)
            if extra_samples >= 0:
                y_sig = np.concatenate([waveform, np.zeros(extra_samples)])
            else:
                y_sig = waveform[:extra_samples]
        else:
            y_sig = waveform

        # Specific real-time filters are turned on below
        self.set_unused_realtime_distortions_zero()
        nr_real_time_exp_models = 0
        nr_real_time_hp_models = 0
        nr_real_time_bounce_models = 0
        for filt_id in range(self._num_models):
            filt = self.get("filter_model_{:02}".format(filt_id))

            if not filt:
                pass  # dict is empty
            else:
                model = filt["model"]
                AWG = self.instr_AWG.get_instr()
                if model == "high-pass":
                    if "real-time" in filt.keys() and filt["real-time"]:
                        # Implementation tested and found not working -MAR
                        raise NotImplementedError()
                        nr_real_time_hp_models += 1
                        if nr_real_time_hp_models > 1:
                            raise ValueError()
                    else:
                        y_sig = kf.bias_tee_correction(
                            y_sig,
                            sampling_rate=self.cfg_sampling_rate(),
                            inverse=inverse,
                            **filt["params"]
                        )
                elif model == "exponential":
                    if "real-time" in filt.keys() and filt["real-time"]:

                        AWG.set(
                            "sigouts_{}_precompensation_exponentials"
                            "_{}_timeconstant".format(
                                self.cfg_awg_channel() - 1, nr_real_time_exp_models
                            ),
                            filt["params"]["tau"],
                        )
                        AWG.set(
                            "sigouts_{}_precompensation_exponentials"
                            "_{}_amplitude".format(
                                self.cfg_awg_channel() - 1, nr_real_time_exp_models
                            ),
                            filt["params"]["amp"],
                        )
                        AWG.set(
                            "sigouts_{}_precompensation_exponentials"
                            "_{}_enable".format(
                                self.cfg_awg_channel() - 1, nr_real_time_exp_models
                            ),
                            1,
                        )

                        nr_real_time_exp_models += 1
                        if nr_real_time_exp_models > 5:
                            raise ValueError()
                    else:
                        y_sig = kf.exponential_decay_correction(
                            y_sig,
                            sampling_rate=self.cfg_sampling_rate(),
                            inverse=inverse,
                            **filt["params"]
                        )
                elif model == "bounce":
                    if "real-time" in filt.keys() and filt["real-time"]:

                        AWG.set(
                            "sigouts_{}_precompensation_bounces"
                            "_{}_delay".format(
                                self.cfg_awg_channel() - 1, nr_real_time_bounce_models
                            ),
                            filt["params"]["tau"],
                        )
                        AWG.set(
                            "sigouts_{}_precompensation_bounces"
                            "_{}_amplitude".format(
                                self.cfg_awg_channel() - 1, nr_real_time_bounce_models
                            ),
                            filt["params"]["amp"],
                        )
                        AWG.set(
                            "sigouts_{}_precompensation_bounces"
                            "_{}_enable".format(
                                self.cfg_awg_channel() - 1, nr_real_time_bounce_models
                            ),
                            1,
                        )

                        nr_real_time_bounce_models += 1
                        if nr_real_time_bounce_models > 1:
                            raise ValueError()
                    else:
                        y_sig = kf.first_order_bounce_corr(
                            sig=y_sig,
                            delay=filt["params"]["tau"],
                            amp=filt["params"]["amp"],
                            awg_sample_rate=2.4e9,
                        )

                elif model == "FIR":
                    fir_filter_coeffs = filt["params"]["weights"]
                    if "real-time" in filt.keys() and filt["real-time"]:
                        if len(fir_filter_coeffs) != 40:
                            raise ValueError(
                                "Realtime FIR filter must contain 40 weights"
                            )
                        else:
                            AWG.set(
                                "sigouts_{}_precompensation_fir_coefficients".format(
                                    self.cfg_awg_channel() - 1
                                ),
                                fir_filter_coeffs,
                            )
                            AWG.set(
                                "sigouts_{}_precompensation_fir_enable".format(
                                    self.cfg_awg_channel() - 1
                                ),
                                1,
                            )
                    else:
                        if not inverse:
                            y_sig = signal.lfilter(fir_filter_coeffs, 1, y_sig)
                        elif inverse:
                            y_sig = signal.lfilter(np.ones(1), fir_filter_coeffs, y_sig)

                else:
                    raise KeyError("Model {} not recognized".format(model))

        if inverse:
            y_sig /= self.cfg_gain_correction()
        else:
            y_sig *= self.cfg_gain_correction()
        return y_sig

    def print_overview(self):
        print("*" * 80)
        print("Overview of {}".format(self.name))
        for filt_id in range(self._num_models):

            filt = self.get("filter_model_{:02}".format(filt_id))
            if filt != {}:
                model = filt["model"]
                params = filt["params"]

                print("Model {} {}: \n \t{}".format(filt_id, model, params))
                if "real-time" in filt.keys() and filt["real-time"]:
                    print("\treal-time : True")
                else:
                    print("\treal-time : False")

        print("*" * 80)
