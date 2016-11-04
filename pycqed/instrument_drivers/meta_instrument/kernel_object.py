import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from measurement.kernel_functions import kernel_generic, htilde_bounce, \
    htilde_skineffect, save_kernel, step_bounce, step_skineffect


def bounce_kernel(amp=0.02, time=4, length=601):
    """
    Generates a bounce kernel, with the specified parameters.

    kernel_step_function:
        heaviside(t) + amp*heaviside(t-time)
    """
    t_kernel = np.arange(length)
    bounce_pairs = [[amp, time]]
    # kernel_bounce = kernel_generic(htilde_bounce, t_kernel, bounce_pairs)
    # kernel_bounce /= np.sum(kernel_bounce)
    bounce_kernel_step = step_bounce(t_kernel, bounce_pairs)
    kernel_bounce = np.zeros(bounce_kernel_step.shape)
    kernel_bounce[0] = bounce_kernel_step[0]
    kernel_bounce[1:] = bounce_kernel_step[1:]-bounce_kernel_step[:-1]
    kernel_bounce /= np.sum(kernel_bounce)

    return kernel_bounce
# save_kernel(kernel_bounce, save_file='kernel_bounce_%.3f_%d'%tuple(bounce_pairs[0]))


def decay_kernel(amp=1., tau=11000, length=20000):
    """
    Generates a decay kernel, with the specified parameters

    kernel_step_function
        1 + amp*np.exp(-t_kernel/tau)
    """
    t_kernel = np.arange(length)
    decay_kernel_step = 1 + amp*np.exp(-t_kernel/tau)
    decay_kernel = np.zeros(decay_kernel_step.shape)
    decay_kernel[0] = decay_kernel_step[0]
    decay_kernel[1:] = decay_kernel_step[1:]-decay_kernel_step[:-1]
    return decay_kernel
# save_kernel(decay_kernel, save_file='kernel_fridge_slow1_160212')


def skin_kernel(alpha=0., length=601):
    """
    Generates a skin effect kernel, with the specified parameters

    kernel_step_function
        heaviside(t+1)*(1-errf(alpha/21./np.sqrt(t+1)))
    """
    t_kernel = np.arange(length)
    # kernel_skineffect = kernel_generic(htilde_skineffect,t_kernel,alpha)
    # kernel_skineffect /= np.sum(kernel_skineffect)
    skineffect_kernel_step = step_skineffect(t_kernel, alpha)
    kernel_skineffect = np.zeros(skineffect_kernel_step.shape)
    kernel_skineffect[0] = skineffect_kernel_step[0]
    kernel_skineffect[1:] = skineffect_kernel_step[
        1:]-skineffect_kernel_step[:-1]
    kernel_skineffect /= np.sum(kernel_skineffect)
    return kernel_skineffect
# save_kernel(kernel_skineffect, save_file='kernel_skineffect_%.1f'%(alpha))


def poly_kernel(a=0, b=11000, c=0, length=30000):
    """
    Generates a polynomial kernel(like the one used for bias-tee), with the specified parameters

    """
    t_kernel = np.arange(length)
    poly_kernel_step = a*t_kernel**2+b*t_kernel+c
    kernel_poly = np.zeros(poly_kernel_step.shape)
    kernel_poly[0] = poly_kernel_step[0]
    kernel_poly[1:] = poly_kernel_step[1:]-poly_kernel_step[:-1]
    return kernel_poly
# save_kernel(kernel_skineffect, save_file='kernel_skineffect_%.1f'%(alpha))


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
                           vals=vals.Numbers())
        self.add_parameter('skineffect_length', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_amp', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_tau', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_length', units='ns',
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
                           vals=vals.Numbers())

    def get_bounce_kernel(self):
        return bounce_kernel(amp=self.bounce_amp(), time=self.bounce_tau(),
                             length=self.bounce_length())

    def get_skin_kernel(self):
        return skin_kernel(alpha=self.skineffect_alpha(),
                           length=self.skineffect_length())

    def get_decay_kernel(self):
        return decay_kernel(amp=self.decay_amp(), tau=self.decay_tau(),
                            length=self.decay_length())

    def get_poly_kernel(self):
        return poly_kernel(a=self.poly_a(),
                           b=self.poly_b(),
                           c=self.poly_c(),
                           length=self.poly_length())

    def convolve_kernel(self, kernel_list, length=None):
        if length is None:
            length = max([len(k) for k in kernel_list])
        total_kernel = kernel_list[0]
        for k in kernel_list[1:]:
            total_kernel = np.convolve(total_kernel, k)[:length]
        return total_kernel

    def get_corrections_kernel(self, kernel_list_before=None):
        kernel_list = [self.get_bounce_kernel(), self.get_skin_kernel(),
                       self.get_decay_kernel(), self.get_poly_kernel()]
        if kernel_list_before is not None:
            kernel_list_before.extend(kernel_list)
            return self.convolve_kernel(kernel_list_before,
                                        length=self.corrections_length())
        else:
            return self.convolve_kernel(kernel_list,
                                        length=self.corrections_length())

    def save_corrections_kernel(self, filename, kernel_list_before=None):
        if type(kernel_list_before) is not list:
            kernel_list_before = [kernel_list_before]
        save_kernel(self.get_corrections_kernel(kernel_list_before),
                    save_file=filename)
        return filename
