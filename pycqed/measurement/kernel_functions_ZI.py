"""
Kernel functions designed to be compatible with real-time digital filters
(IIR and FIR)

This implements three basic filters and a rounding function
- multipath_bias_tee
- multipath_filter
- multipath_filter2

"""
import logging
import textwrap
import numpy as np
from scipy import signal

# Uses an "old-style" kernel to correct the bounce. This should be replaced
# by a signal.lfilter provided by Yves (MAR May 2018)
from pycqed.measurement.kernel_functions import bounce_kernel


def bias_tee_correction(ysig, tau: float, sampling_rate: float=1,
                        inverse: bool=False):
    """
    Corrects for a bias tee correction using a linear IIR filter with time
    constant tau.
    """
    # factor 2 comes from bilinear transform
    k = 2*tau*sampling_rate
    b = [1, -1]
    a = [(k+1)/k, -(k-1)/k]

    if inverse:
        filtered_signal = signal.lfilter(b, a, ysig)
    else:
        filtered_signal = signal.lfilter(a, b, ysig)
    return filtered_signal


def exponential_decay_correction(ysig, tau: float, amp: float,
                                 sampling_rate: float=1,
                                 inverse: bool=False):
    """
    Corrects for an exponential decay using a linear IIR filter.

    Fitting should be done to the following function:
        y = gc*(1 + amp *exp(-t/tau))
    where gc is a gain correction factor that is ignored in the corrections.
    """

    # alpha ~1/8 is like averaging 8 samples, sets the timescale for averaging
    # larger alphas break the approximation of the low pass filter
    # numerical instability occurs if alpha > .03
    alpha = 1 - np.exp(-1/(sampling_rate*tau*(1+amp)))
    # the coefficient k (and the filter transfer function)
    # depend on the sign of amp

    if amp >= 0.0:
        k = amp/(1+amp-alpha)
        # compensating overshoot by a low-pass filter
        # correction filter y[n] = (1-k)*x[n] + k*u[n] = x[n] + k*(u[n]-x[n])
        # where u[n] = u[n-1] + alpha*(x[n] - u[n-1])
        a = [(1-k + k*alpha), -(1-k)*(1-alpha)]
    else:
        k = -amp/(1+amp)/(1-alpha)
        # compensating low-pass by an overshoot
        # correction filter y[n] = (1+k)*x[n] - k*u[n] = x[n] - k*(u[n]-x[n])
        # where u[n] = u[n-1] + alpha*(x[n] - u[n-1])
        a = [(1 + k - k*alpha), -(1+k)*(1-alpha)]

    # while the denominator stays the same
    b = [1, -(1-alpha)]
    # if alpha > 0.03 the filter can be unstable.

    if inverse:
        filtered_signal = signal.lfilter(b, a, ysig)
    else:
        filtered_signal = signal.lfilter(a, b, ysig)
    return filtered_signal


def bounce_correction(ysig, tau: float, amp: float,
                      sampling_rate: float = 1,
                      inverse: bool=False):
    """
    Corrects for a bounce

    Args:
        ysig: the signal to be predistorted
        tau: the time at which the bounce occurs
        amp: the amplitude of the bounce correction
        sampling_rate: the sampling rate of the signal
    returns:
        filtered_signal : the signal corrected for the bounce
    """

    # kernel is cut of after 8*tau, this menas that it will only correct
    # bounces up to 8th order, this is good for coefficients << 1
    kern = bounce_kernel(amp, time=tau, length=8*tau,
                         sampling_rate=sampling_rate)
    if inverse:
        raise NotImplemented
    else:
        filter_signal = np.convolve(ysig, kern, mode='full')
    return filter_signal[:len(ysig)]


#################################################################
#              hardware friendly functions                      #
#################################################################


def exponential_decay_correction_hw_friendly(ysig, tau: float, amp: float,
                                             sampling_rate: float=1,
                                             inverse: bool=False):
    """
    Corrects for an exponential decay using "multipath_filter2".

    Fitting should be done to the following function:
        y = gc*(1 + amp *exp(-t/tau))
    where gc is a gain correction factor that is ignored in the corrections.
    """

    alpha = 1 - np.exp(-1/(sampling_rate*tau*(1+amp)))
    # N.B. The if statement from the conventional "linear" filter is
    # completely gone. This black magic needs to be understood. MAR July 2018
    # k = amp/(1+amp-alpha)
    if amp >= 0.0:
        # compensating overshoot by a low-pass filter
        # correction filter y[n] = (1-k)*x[n] + k*u[n] = x[n] + k*(u[n]-x[n])
        # where u[n] = u[n-1] + alpha*(x[n] - u[n-1])
        k = amp/(1+amp-alpha)
    else:
        # compensating low-pass by an overshoot
        # correction filter y[n] = (1+k)*x[n] - k*u[n] = x[n] - k*(u[n]-x[n])
        # where u[n] = u[n-1] + alpha*(x[n] - u[n-1])
        # the correction filter differs just in the sign of k, so allow the k to be negative here
        k = amp/(1+amp)/(1-alpha)


    filtered_signal = multipath_filter2(sig=ysig, alpha=alpha, k=k, paths=8)

    if inverse:
        raise NotImplementedError()
    # ensure that the lenght of the returned signal is the same as the input
    return filtered_signal[:len(ysig)]


# Delay a signal by d clock cycles.
# This is done by adding d zero entries to the front of the signal array, and by removing the last d entries.
def sigdelay(sig, d):
    # delays the signal sig by d clock cycles. The argument d must be an integer.
    s = np.zeros(sig.shape)
    s[d:] = sig[:-d]
    return s


def coef_round(value, force_bshift=None):
    """
    hardware friendly
    rounds coefficients to hardware-compatible pseudo-float
    """
    if type(value) == dict:
        new_value = {}
        for key in value:
            new_value[key] = coef_round(value[key])
    elif type(value) == list:
        new_value = []
        for val in value:
            new_value.append(coef_round(val))
    else:
        # scalar value
        new_value = 0.
        # to simplify the cores we allow 4 discrete shifts by 4 bits: 0, -4, -8
        # and -12
        if force_bshift is None:
            bshift = np.max(
                [np.min([np.floor(-np.log2(np.abs(value))/4.)*4., 12.]), 0.])
        else:
            bshift = force_bshift
        # 18 bits within the multiplication itself
        scaling = np.power(2, 18+bshift)
        new_value = np.round(value*scaling) / scaling

    return new_value


def multipath_bias_tee(sig, k, paths):
    """
    hardware friendly
    hardware friendly bias-tee (or any other AC coupling) compensation filter
    """
    tpl = np.ones((paths, ))
    cs = 0
    acc = []
    for i in np.arange(0, sig.size, paths):
        cs = cs + np.sum(sig[i:(i+paths)])
        acc = np.append(acc, tpl*cs)
    return sig + 1./k * (2*acc - sig)


def multipath_filter(sig, alpha, k, paths):
    """
    hardware friendly
    exponential moving average correction filter
    """
    tpl = np.ones((paths, ))
    hw_alpha = alpha*float(paths)

    duf = tpl * 0.
    acc = 0

    for i in np.arange(0, sig.size, paths):
        acc = acc + hw_alpha*(np.mean(sig[i:(i+paths)]) - acc)
        duf = np.append(duf, tpl * acc)
    duf = duf[0:sig.size]
    return sig + k * (duf - sig)



def multipath_filter2(sig, alpha, k, paths,
                      hw_rounding: bool=True,
                      ppl = 2):
    """
    hardware friendly
    exponential moving average correction filter with pipeline simulation
    """

    tpl = np.ones((paths, ))
    hw_alpha = alpha*float(paths*ppl)
    hw_k = k

    if hw_rounding:
        hw_alpha = coef_round(hw_alpha)
        hw_k = coef_round(hw_k)

    duf = tpl * 0.
    # due to the pipelining, there are actually ppl interleaved filters
    acc = np.zeros((ppl, ))

    # make sure our vector has a length that is a multiple of ppl*paths
    extra = int(ppl*paths*np.ceil(sig.size/ppl/paths)-sig.size)
    if extra > 0:
        sig = np.append(sig, extra*[sig[-1]])

    # first create an array of averaged path values
    du = []
    for i in np.arange(0, sig.size, paths):
        du = np.append(du, np.mean(sig[i:(i+paths)]))

    # loop through the average bases
    for i in np.arange(0, du.size, ppl):
        # loop through the individual sub-filters
        for j in np.arange(0, ppl):
            # first calculate the filter input as an average of the previous
            # path averages
            ss = 0
            for l in np.arange(0, ppl):
                if i+j-l >= 0:
                    ss = ss + du[i+j-l]

            ss = ss / float(ppl)
            # then process it by the filter
            acc[j] = acc[j] + hw_alpha*(ss - acc[j])
            # and add the filter output to the signal correction vector
            duf = np.append(duf, tpl * acc[j])
    duf = duf[0:sig.size]
    return sig + hw_k * (duf - sig)


def first_order_bounce_corr(sig, delay, amp, awg_sample_rate,
                            scope_sample_rate=None, bufsize=256,
                            sim_hw_delay=False):
    """
    This function simulates the real-time bounce correction.

    Args:
        sig:           The signal to be filtered as a numpy array.
        delay:         The delay of the bounce in seconds.
        amp:           The amplitude of the bounce.
        sampling_rate: The sampling rate in Hz.

    Returns:
        sigout: Numpy array representing the output signal of the filter
    """
    delay_n_samples = int(round(awg_sample_rate*delay))
    if not 1 <= delay_n_samples < bufsize - 8:
        raise ValueError(textwrap.dedent("""
            The maximum delay ("{}"/ {:.2f}ns)needs to be less than {:d} (bufsize-8) AWG samples to save hardware resources.
            The delay needs to be at least 1 AWG sample.")
            """.format(delay_n_samples, delay*1e9, bufsize - 8)))
    if not -1 < amp < 1:
        raise ValueError(
            "The amplitude ({}) needs to be between -1 and 1.".format(amp))

    # The scope sampling rate is equal to the AWG sampling rate by default.
    if scope_sample_rate is None:
        scope_sample_rate = awg_sample_rate

    # Reserve buffer space for bounce compensation
    shift_reg = np.zeros(delay_n_samples)

    awg_sample_incr = awg_sample_rate/scope_sample_rate

    previous_awg_sample_cnt = 0
    present_awg_sample_cnt = 0

    amp_hw = coef_round(amp, force_bshift=0)

    sigout = np.zeros(len(sig))

    for i, s in enumerate(sig):
        # Compute output signal
        sigout[i] = s + amp_hw*shift_reg[-1]

        present_awg_sample_cnt += awg_sample_incr
        awg_sample_diff = int(present_awg_sample_cnt) - previous_awg_sample_cnt
        if awg_sample_diff >= 1:
            # Update shift register with present scope sample
            shift_reg[awg_sample_diff:] = shift_reg[:-awg_sample_diff]
            shift_reg[:awg_sample_diff] = s*np.ones(awg_sample_diff)
            previous_awg_sample_cnt = int(present_awg_sample_cnt)

    if sim_hw_delay:
        sigout = sigdelay(sigout, int(round(8*(4+5)/awg_sample_incr)))

    return sigout


# def first_order_bounce_corr_with_interpolation(sig, delay, amp, awg_sample_rate, scope_sample_rate = None, bufsize=256):
#     """ This function simulates the real-time bounce correction.

#     Args:
#         sig:           The signal to be filtered as a numpy array.
#         delay:         The delay of the bounce in seconds.
#         amp:           The amplitude of the bounce.
#         sampling_rate: The sampling rate in Hz.

#     Returns:
#         sigout: Numpy array representing the output signal of the filter
#     """
#     delay_n_samples = int(np.floor(awg_sample_rate*delay))
#     interpolation_fraction = awg_sample_rate*delay - delay_n_samples
#     print(interpolation_fraction)
#     if not 1 <= delay_n_samples < bufsize - 8:
#         raise ValueError(textwrap.dedent("""
#             The maximum delay needs to be less than {:d} (bufsize-8) AWG samples to save hardware resources.
#             The delay needs to be at least 1 AWG sample.")
#             """.format(bufsize - 8)))
#     if not -1 < amp < 1:
#         raise ValueError("The amplitude needs to be between -1 and 1.")

#     # The scope sampling rate is equal to the AWG sampling rate by default.
#     if scope_sample_rate is None:
#         scope_sample_rate = awg_sample_rate

#     # Reserve buffer space for bounce compensation.
#     shift_reg = np.zeros(delay_n_samples + 1)
#     delay_reg = 0.0

#     awg_sample_incr = awg_sample_rate/scope_sample_rate

#     previous_awg_sample_cnt = 0
#     present_awg_sample_cnt = 0

#     amp_hw = coef_round(amp, force_bshift=0)

#     sigout = np.zeros(len(sig))

#     for i, s in enumerate(sig):
#         # Compute output signal
#         interpolation = (1-interpolation_fraction)*shift_reg[-1] + interpolation_fraction*delay_reg
#         sigout[i] = s + amp_hw*interpolation

#         present_awg_sample_cnt += awg_sample_incr
#         awg_sample_diff = int(present_awg_sample_cnt) - previous_awg_sample_cnt
#         if awg_sample_diff >= 1:
#             # Update the delay register used in the interpolation.
#             delay_reg = shift_reg[-1]

#             # Update shift register with present scope sample.
#             shift_reg[awg_sample_diff:] = shift_reg[:-awg_sample_diff]
#             shift_reg[:awg_sample_diff] = s*np.ones(awg_sample_diff)
#             previous_awg_sample_cnt = int(present_awg_sample_cnt)

#     return sigout

def first_order_bounce_kern(delay, amp, sampling_rate):
    """ This function computes the filter kernel for first-order bounce (only one reflection considered).

    Args:
        delay:          The delay in seconds
        amp:            The amplitude of the bounce
        sampling_rate:  The sampling rate in Hz

    Returns:
        kern: Numpy array representing the filter kernel
    """
    delay_n_samples = round(sampling_rate*delay)
    if not delay_n_samples >= 1:
        raise ValueError("The delay needs to be at least one sample.")
    kern = np.zeros(delay_n_samples+1)
    kern[0] = 1.0
    kern[-1] = amp
    return kern


def ideal_inverted_fir_kernel(impulse_response, zero_ind=0, zero_padding=0):
    """
    This function computes the ideal inverted FIR filter kernel for a given impulse_response.

    Args:
        impulse_response: Array representing the impulse response of the distortion to be corrected.
        zero_ind:         Index of the time 0
        zero_padding:     Number of zeros to append to the the impulse_response

    Returns:
        kern_re_trunc:    The inverted kernel as a real-valued numpy array.
    """
    resp = np.concatenate([impulse_response, np.zeros(zero_padding)])

    f = np.fft.fft(resp)
    kern = np.fft.ifft(1/f)
    kern_re_trunc = np.real(kern)[zero_ind:]
    return kern_re_trunc
