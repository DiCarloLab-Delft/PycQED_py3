import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.measurement.kernel_functions import kernel_generic, htilde_bounce, \
    htilde_skineffect, save_kernel, step_bounce, step_skineffect, \
    heaviside, kernel_generic2

import pycqed.measurement.kernel_functions as kf


class Distortion(Instrument):

    '''
    Implements a distortion kernel for a flux channel.
    It contains the parameters and functions needed to produce a kernel file
    according to the models shown in the functions.
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_parameter('skineffect_alpha', units='',
                           parameter_class=ManualParameter,
                           initial_value=0,
                           vals=vals.Numbers())
        self.add_parameter('skineffect_length', units='ns',
                           parameter_class=ManualParameter,
                           initial_value=600,
                           vals=vals.Numbers())

        self.add_parameter('decay_amp_1', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_tau_1', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_length_1', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_amp_2', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_tau_2', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_length_2', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        self.add_parameter('bounce_amp', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('bounce_tau', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('bounce_length', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        self.add_parameter('poly_a', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('poly_b', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('poly_c', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('poly_length', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

        self.add_parameter('corrections_length', units='ns',
                           parameter_class=ManualParameter,
                           initial_value=1000,
                           vals=vals.Numbers())

    def get_idn(self):
        return self.name

    def get_bounce_kernel(self):
        return kf.bounce_kernel(amp=self.bounce_amp(),
                             time=self.bounce_tau(),
                             length=self.bounce_length())

    def get_skin_kernel(self):
        return kf.skin_kernel(alpha=self.skineffect_alpha(),
                           length=self.skineffect_length())

    def get_decay_kernel_1(self):
        return kf.decay_kernel(amp=self.decay_amp_1(), tau=self.decay_tau_1(),
                            length=self.decay_length_1())

    def get_decay_kernel_2(self):
        return kf.decay_kernel(amp=self.decay_amp_2(), tau=self.decay_tau_2(),
                            length=self.decay_length_2())

    # def get_poly_kernel(self):
    #     return poly_kernel(a=self.poly_a(),
    #                        b=self.poly_b(),
    #                        c=self.poly_c(),
    #                        length=self.poly_length())

    def convolve_kernel(self, kernel_list, length=None):
        """
        kernel_list : (list of arrays)
        length      : (int) maximum length for convolution
        Performs a convolution of different kernels
        """
        kernels = kernel_list[0]
        for k in kernel_list[1:]:
            kernels = np.convolve(k, kernels)[:max(len(k), len(kernels))]
        return kernels

    def kernel_to_cache(self, cache):
        """
        cache (dict): a dictionary containing predistortion kernels

        This will add 'OPT_Chevron.tmp' to the cache dictionary that
        contains an array with the distortions based on the kernel object.
        """
        kernel_list = [self.get_bounce_kernel(),
                       self.get_skin_kernel(),
                       self.get_decay_kernel_1(),
                       self.get_decay_kernel_2()]
        cache.update({'OPT_chevron.tmp': self.convolve_kernel(kernel_list)})

    def get_corrections_kernel(self, kernel_list_before=None):
        kernel_list = [self.get_bounce_kernel(), self.get_skin_kernel(),
                       self.get_decay_kernel_1(),
                       self.get_decay_kernel_2()]
        if kernel_list_before is not None:
            kernel_list_before.extend(kernel_list)
            return self.convolve_kernel(kernel_list_before,
                                        length=self.corrections_length())
        else:
            return self.convolve_kernel(kernel_list,
                                        length=self.corrections_length())

    def save_corrections_kernel(self, filename, kernel_list_before=None):

        # if type(kernel_list_before) is not list:
        #     kernel_list_before = [kernel_list_before]
        save_kernel(self.get_corrections_kernel(kernel_list_before),
                    save_file=filename)
        return filename


