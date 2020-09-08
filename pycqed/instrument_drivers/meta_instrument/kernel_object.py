import os
import logging
import numpy as np
import json

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.measurement.kernel_functions import (
    kernel_generic,
    htilde_bounce,
    htilde_skineffect,
    save_kernel,
    step_bounce,
    step_skineffect,
    heaviside,
)

import pycqed.measurement.kernel_functions as kf
from pycqed.instrument_drivers.pq_parameters import ConfigParameter


class DistortionKernel(Instrument):

    """
    Implements a distortion kernel for a flux channel.
    It contains the parameters and functions needed to produce a kernel file
    according to the models shown in the functions.
    """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        self.add_parameter(
            "channel",
            initial_value=1,
            vals=vals.Ints(),
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "kernel_list",
            initial_value=[],
            vals=vals.Lists(vals.Strings()),
            parameter_class=ConfigParameter,
            docstring="List of filenames of external kernels to be loaded",
        )

        self.add_parameter(
            "kernel_dir",
            initial_value="kernels/",
            vals=vals.Strings(),
            parameter_class=ManualParameter,
            docstring="Path for loading external kernels,"
            + "such as room temperature correction kernels.",
        )

        self.add_parameter(
            "config_changed", vals=vals.Bool(), get_cmd=self._get_config_changed
        )
        self.add_parameter(
            "kernel",
            vals=vals.Arrays(),
            get_cmd=self._get_kernel,
            docstring=(
                "Returns the predistortion kernel. \n"
                + "Recalculates if the parameters changed,\n"
                + "otherwise returns a precalculated kernel.\n"
                + "Kernel is based on parameters in kernel object \n"
                + "and files specified in the kernel list."
            ),
        )

        self.add_parameter(
            "skineffect_alpha",
            unit="",
            parameter_class=ConfigParameter,
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "skineffect_length",
            unit="s",
            parameter_class=ConfigParameter,
            initial_value=600e-9,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "decay_amp_1",
            unit="",
            initial_value=0,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "decay_tau_1",
            unit="s",
            initial_value=1e-9,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "decay_length_1",
            unit="s",
            initial_value=100e-9,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "decay_amp_2",
            unit="",
            initial_value=0,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "decay_tau_2",
            unit="s",
            initial_value=1e-9,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "decay_length_2",
            unit="s",
            initial_value=100e-9,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "bounce_amp_1",
            unit="",
            initial_value=0,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "bounce_tau_1",
            unit="s",
            initial_value=0,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "bounce_length_1",
            unit="s",
            initial_value=1e-9,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "bounce_amp_2",
            unit="",
            initial_value=0,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "bounce_tau_2",
            unit="s",
            initial_value=0,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "bounce_length_2",
            unit="s",
            initial_value=1e-9,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "poly_a",
            unit="",
            initial_value=0,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "poly_b",
            unit="",
            initial_value=0,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "poly_c",
            unit="",
            initial_value=1,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )
        self.add_parameter(
            "poly_length",
            unit="s",
            initial_value=600e-9,
            parameter_class=ConfigParameter,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "corrections_length",
            unit="s",
            parameter_class=ConfigParameter,
            initial_value=10e-6,
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "sampling_rate",
            parameter_class=ManualParameter,
            initial_value=1e9,
            vals=vals.Numbers(),
        )

    def add_kernel_to_kernel_list(self, kernel_name):
        v = vals.Strings()
        v.validate(kernel_name)
        kernel_list = self.kernel_list()
        if kernel_name in kernel_list:
            raise ValueError('Kernel "{}" already in kernel list'.format(kernel_name))
        kernel_list.append(kernel_name)
        self._config_changed = True  # has to be done by hand as appending to
        # the list does not correctlyupdate the changed flag
        self.kernel_list(kernel_list)

    def remove_kernel_from_kernel_list(self, kernel_name):
        ker_list = self.kernel_list()
        ker_list.remove(kernel_name)
        self._config_changed = True
        self.kernel_list(ker_list)

    def _get_config_changed(self):
        return self._config_changed

    def get_idn(self):
        return self.name

    def get_bounce_kernel_1(self):
        return kf.bounce_kernel(
            amp=self.bounce_amp_1(),
            time=self.bounce_tau_1() * self.sampling_rate(),
            length=self.bounce_length_1() * self.sampling_rate(),
        )

    def get_bounce_kernel_2(self):
        return kf.bounce_kernel(
            amp=self.bounce_amp_2(),
            time=self.bounce_tau_2() * self.sampling_rate(),
            length=self.bounce_length_2() * self.sampling_rate(),
        )

    def get_skin_kernel(self):
        return kf.skin_kernel(
            alpha=self.skineffect_alpha(),
            length=self.skineffect_length() * self.sampling_rate(),
        )

    def get_decay_kernel_1(self):
        return kf.decay_kernel(
            amp=self.decay_amp_1(),
            tau=self.decay_tau_1(),
            length=self.decay_length_1(),
            sampling_rate=self.sampling_rate(),
        )

    def get_decay_kernel_2(self):
        return kf.decay_kernel(
            amp=self.decay_amp_2(),
            tau=self.decay_tau_2(),
            length=self.decay_length_2(),
            sampling_rate=self.sampling_rate(),
        )

    def get_poly_kernel(self):
        """
        Returns polynomial kernel, see kernel_functions.poly_kernel for details
        """
        return kf.poly_kernel(
            coeffs=[self.poly_a(), self.poly_b(), self.poly_c()],
            length=self.poly_length() * self.sampling_rate(),
        )

    def convolve_kernel(self, kernel_list, length_samples=None):
        """
        kernel_list : (list of arrays)
        length_samples      : (int) maximum for convolution
        Performs a convolution of different kernels
        """
        kernels = kernel_list[0]
        for k in kernel_list[1:]:
            kernels = np.convolve(k, kernels)[: max(len(k), int(length_samples))]
        if length_samples is not None:
            return kernels[: int(length_samples)]
        return kernels

    def kernel_to_cache(self, cache):
        """
        cache (dict): a dictionary containing predistortion kernels

        This will add 'OPT_Chevron.tmp' to the cache dictionary that
        contains an array with the distortions based on the kernel object.
        """
        logging.warning("deprecated, do not use!")
        kernel_list = [
            self.get_bounce_kernel_1(),
            self.get_bounce_kernel_2(),
            self.get_skin_kernel(),
            self.get_decay_kernel_1(),
            self.get_decay_kernel_2(),
        ]
        cache.update({"OPT_chevron.tmp": self.convolve_kernel(kernel_list)})

    def get_corrections_kernel(self):

        kernel_list_before = self.kernel_list()

        external_kernels = []
        for k_name in kernel_list_before:
            f_name = os.path.join(self.kernel_dir(), k_name)
            print("Loading {}".format(f_name))

            suffix = f_name.split(".")[-1]
            if suffix == "txt":
                kernel_vec = np.loadtxt(f_name)
                external_kernels.append(kernel_vec)
            elif suffix == "json":
                # Load from json file containing also metadata about fit model
                with open(f_name) as infile:
                    kernel_dict = json.load(infile)
                external_kernels.append(kernel_dict["kernel"])
            else:
                raise ValueError('File format "{}" not recognized.'.format(suffix))

        kernel_object_kernels = [
            self.get_bounce_kernel_1(),
            self.get_bounce_kernel_2(),
            self.get_skin_kernel(),
            self.get_decay_kernel_1(),
            self.get_decay_kernel_2(),
            self.get_poly_kernel(),
        ]

        kernel_list = external_kernels + kernel_object_kernels
        return self.convolve_kernel(
            kernel_list,
            length_samples=int(self.corrections_length() * self.sampling_rate()),
        )

    def save_corrections_kernel(self, filename):

        # if type(kernel_list_before) is not list:
        #     kernel_list_before = [kernel_list_before]
        save_kernel(self.get_corrections_kernel(), save_file=filename)
        return filename

    def _get_kernel(self):
        """
        Returns the kernel.
        """
        if self.config_changed():
            print("{} configuration changed, recalculating kernels".format(self.name))
            self._precalculated_kernel = self.get_corrections_kernel()
            self._config_changed = False

        return self._precalculated_kernel
