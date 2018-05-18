"""
Kernel functions designed to be compatible with real-time digital filters
(IIR and FIR)

This implements three basic filters and a rounding function
- multipath_bias_tee
- multipath_filter
- multipath_filter2

"""
import logging
import numpy as np
from scipy import signal


def bias_tee_correction(ysig, tau: float, sampling_rate: float=1):
    """
    Corrects for a bias tee correction using a linear IIR filter with time
    constant tau.
    """
    # factor 2 comes from bilinear transform
    k = 2*tau*sampling_rate
    b = [1, -1]
    a = [(k+1)/k, -(k-1)/k]

    # hint: to get the inverse just use the filter with (b, a, ysig)
    filtered_signal = signal.lfilter(a, b, ysig)
    return filtered_signal


def exponential_decay_correction(ysig, tau: float, amp: float,
                                 sampling_rate: float=1):
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

    # hint: to get the inverse just use the filter with (a, b, ysig)
    filtered_signal = signal.lfilter(a, b, ysig)
    return filtered_signal




#################################################################
#              hardware friendly functions                      #
#################################################################


def coef_round(value):
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
        bshift = np.max(
            [np.min([np.floor(-np.log2(np.abs(value))/4.)*4., 12.]), 0.])
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


def multipath_filter2(sig, alpha, k, paths):
    """
    hardware friendly
    exponential moving average correction filter with pipeline simulation
    """
    ppl = 4
    tpl = np.ones((paths, ))
    hw_alpha = alpha*float(paths*ppl)

    duf = tpl * 0.
    # due to the pipelining, there are actually ppl interleaved filters
    acc = np.zeros((ppl, ))

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
    return sig + k * (duf - sig)

def multipath_bounce_correction(sig, delay, amp, paths = 8, bufsize = 128):
    """
    This function simulates a possible FPGA implementation of a first-order bounce correction filter.
    The signal (sig) is assumed to be a numpy array representing a wavefomr with sampling rate 2.4 GSa/s.
    The delay is specified in number of samples. It needs to be an interger.
    The amplitude (amp) of the bounce is specified relative to the amplitude of the input signal.
    It is constrained to be smaller than 1. The amplitude is represented as a 18-bit fixed point number on the FPGA.
    """
    assert 0 <= delay < bufsize-8, "The maximulm delay is limitted to 120 (bufsize-8) samples to save hardware resources."
    assert -1 <= amp < 1, "The amplitude needs to be between -1 and 1."

    sigout = np.zeros(len(sig))
    buffer = np.zeros(bufsize)

    amp_hw = coef_round(amp)

    # iterate in steps of eight samples through the input signal to simulate the implementation with parallel paths on the FPGA
    for i in range(0, len(sig), paths):
        buffer[paths:] = buffer[:-paths]
        buffer[:paths] = sig[i:i+8]
        sigout[i:i+8] = sig[i:i+8] + amp_hw*buffer[delay:delay+8]
    return sigout


def ideal_inverted_fir_kernel(impulse_response, zero_ind=0):
    """
    This function computes the ideal inverted FIR filter kernel for a given impulse_response.
    The argument zero_ind provides the index corresponding to time t=0 within the impulse_response array.
    """
    f = np.fft.fft(impulse_response)
    impulse_response_inv = np.fft.ifft(1/f)
    impulse_response_inv_re_trunc= np.real(impulse_response_inv)[zero_ind:]
    return impulse_response_inv_re_trunc
