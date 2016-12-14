import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from measurement.kernel_functions import kernel_generic, htilde_bounce, \
    htilde_skineffect, save_kernel, step_bounce, step_skineffect, \
    heaviside, kernel_generic2


def bounce_kernel(amp=0.02, time=4, length=601):
    """
    Generates a bounce kernel, with the specified parameters.

    kernel_step_function:
        heaviside(t) + amp*heaviside(t-time)
    """
    bounce = lambda t, amp, time: heaviside(t) - amp*np.double((t+1)>time)*heaviside(t)
    htilde_bounce = lambda t, time: bounce(t, amp, time) - bounce(t-1, amp, time)
    t_kernel = np.arange(length)
    if abs(amp)>0.:
        kernel_bounce = kernel_generic2(htilde_bounce, t_kernel, time)
    else:
        kernel_bounce = np.zeros(length)
        kernel_bounce[0] = 1.
    return kernel_bounce
# save_kernel(kernel_bounce, save_file='kernel_bounce_%.3f_%d'%tuple(bounce_pairs[0]))


def decay_kernel(amp=1., tau=11000, length=2000):
    """
    Generates a decay kernel, with the specified parameters

    kernel_step_function
        1 + amp*np.exp(-t_kernel/tau)
    """
    tau_k = (1.-amp)*tau
    amp_k = amp/(amp-1)
    t_kernel = np.arange(length)
    if abs(amp)>0.:
        kernel_decay_step = 1 + amp_k*np.exp(-t_kernel/tau_k)
        kernel_decay = np.zeros(kernel_decay_step.shape)
        kernel_decay[0] = kernel_decay_step[0]
        kernel_decay[1:] = kernel_decay_step[1:]-kernel_decay_step[:-1]
    else:
        kernel_decay = np.zeros(length)
        kernel_decay[0] = 1.
    return kernel_decay

# save_kernel(decay_kernel, save_file='kernel_fridge_slow1_160212')


def skin_kernel(alpha=0., length=601):
    """
    Generates a skin effect kernel, with the specified parameters

    kernel_step_function
        heaviside(t+1)*(1-errf(alpha/21./np.sqrt(t+1)))
    """
    t_kernel = np.arange(length)
    if abs(alpha)>0.:
        kernel_skineffect = kernel_generic2(htilde_skineffect, t_kernel, alpha)
    else:
        kernel_skineffect = np.zeros(length)
        kernel_skineffect[0] = 1.
    return kernel_skineffect
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
                           vals=vals.Numbers())

    def get_bounce_kernel(self):
        return bounce_kernel(amp=self.bounce_amp(), time=self.bounce_tau(),
                             length=self.bounce_length())

    def get_skin_kernel(self):
        return skin_kernel(alpha=self.skineffect_alpha(),
                           length=self.skineffect_length())

    def get_decay_kernel_1(self):
        return decay_kernel(amp=self.decay_amp_1(), tau=self.decay_tau_1(),
                            length=self.decay_length_1())

    def get_decay_kernel_2(self):
        return decay_kernel(amp=self.decay_amp_2(), tau=self.decay_tau_2(),
                            length=self.decay_length_2())

    # def get_poly_kernel(self):
    #     return poly_kernel(a=self.poly_a(),
    #                        b=self.poly_b(),
    #                        c=self.poly_c(),
    #                        length=self.poly_length())

    def convolve_kernel(self, kernel_list, length=None):
        kernels = kernel_list[0]
        for k in kernel_list[1:]:
            kernels = np.convolve(k, kernels)[:max(len(k), len(kernels))]
        return kernels

    def kernel_to_cache(self, cache):
        kernel_list = [self.get_bounce_kernel(), self.get_skin_kernel(),
                       self.get_decay_kernel_1(),
                       self.get_decay_kernel_2()]
        cache.update({'OPT_chevron.tmp':self.convolve_kernel(kernel_list)})

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
        if type(kernel_list_before) is not list:
            kernel_list_before = [kernel_list_before]
        save_kernel(self.get_corrections_kernel(kernel_list_before),
                    save_file=filename)
        return filename
