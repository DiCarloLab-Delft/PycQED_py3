"""
Tools to for cryoscope analysis
Brian Tarasinski
Dec 2017
Edited by Adriaan Rol
"""
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from pycqed.analysis.tools.plotting import (set_xlabel, set_ylabel,
                                            flex_colormesh_plot_vs_xy)

import scipy.signal as ss
import scipy.optimize as so
import scipy.interpolate as si


def normalize_sincos(
        data,
        window_size_frac=500,
        window_size=None,
        do_envelope=True):

    if window_size is None:
        window_size = len(data) // window_size_frac

        # window size for savgol filter must be odd
        window_size -= (window_size + 1) % 2

    mean_data_r = ss.savgol_filter(data.real, window_size, 0, 0)
    mean_data_i = ss.savgol_filter(data.imag, window_size, 0, 0)

    mean_data = mean_data_r + 1j * mean_data_i

    if do_envelope:
        envelope = np.sqrt(
            ss.savgol_filter(
                (np.abs(
                    data -
                    mean_data))**2,
                window_size,
                0,
                0))
    else:
        envelope = 1
    norm_data = ((data - mean_data) / envelope)
    return norm_data


def fft_based_freq_guess_complex(y):
    """
    guess the shape of a sinusoidal complex signal y (in multiples of
        sampling rate), by selecting the peak in the fft.
    return guess (f, ph, off, amp) for the model
        y = amp*exp(2pi i f t + ph) + off.
    """
    fft = np.fft.fft(y)[1:len(y)]
    freq_guess_idx = np.argmax(np.abs(fft))
    if freq_guess_idx >= len(y) // 2:
        freq_guess_idx -= len(y)
    freq_guess = 1 / len(y) * (freq_guess_idx + 1)

    phase_guess = np.angle(fft[freq_guess_idx]) + np.pi / 2
    amp_guess = np.absolute(fft[freq_guess_idx]) / len(y)
    offset_guess = np.mean(y)

    return freq_guess, phase_guess, offset_guess, amp_guess


class CryoscopeAnalyzer:
    def __init__(
            self,
            time,
            complex_data,
            norm_window_size=61,
            demod_freq=None,
            derivative_window_length=None,
            derivative_order=2,
            nyquist_order=0,
            demod_smooth=None):
        """
        analyse a cryoscope measurement.

        time: array of times (lengths of Z pulse)
        complex_data: measured data, combine x- and y- results in a
            complex number

        norm_window_size: window size used for normalizing sine and cosine

        demod_freq: frequency for demodulation. Is guessed if None.

        derivative_window_length, derivative_order: parameters of the sovgol
            filter used for extracting frequency.
        Needs some playing around sometimes.

        demod_smooth: when the demodulated signal should be smoothed before
        taking derivative, set this to a tuple (window_length, order),
        again parametrizing a sovgol filter.

        """
        self.time = time
        self.data = complex_data
        self.norm_data = normalize_sincos(self.data, window_size=61)
        self.demod_freq = demod_freq
        self.derivative_window_length = derivative_window_length
        self.demod_smooth = demod_smooth
        self.nyquist_order = nyquist_order

        self.sampling_rate = 1 / (self.time[1] - self.time[0])

        if self.derivative_window_length is None:
            self.derivative_window_length = 7 / self.sampling_rate

        self.derivative_window_size = max(
            3, int(self.derivative_window_length * self.sampling_rate))
        self.derivative_window_size += (self.derivative_window_size + 1) % 2

        if self.demod_freq is None:
            self.demod_freq = - \
                fft_based_freq_guess_complex(self.norm_data)[
                    0] * self.sampling_rate

        self.demod_data = np.exp(
            2 * np.pi * 1j * self.time * self.demod_freq) * self.norm_data

        if self.demod_smooth:
            n, o = self.demod_smooth
            r, i = self.demod_data.real, self.demod_data.imag
            r = ss.savgol_filter(r, n, o, 0)
            i = ss.savgol_filter(i, n, o, 0)
            self.demod_data = r + 1j * i

        # extract the phase. unwrapping only works well if demodulation is
        # good!
        self.phase = np.unwrap(np.angle(self.demod_data))

        # extract frequency by a lowpass-derivative filter.

        # use a savitzky golay filter: it take sliding window of length
        # `window_length`, fits a polynomial, returns derivative at
        # middle point
        self.detuning = ss.savgol_filter(
            self.phase / (
                2 * np.pi),
            window_length=self.derivative_window_size,
            polyorder=derivative_order,
            deriv=1) * self.sampling_rate
        self.real_detuning = self.get_real_detuning(self.nyquist_order)

    def get_real_detuning(self, nyquist_order=None):
        if nyquist_order is None:
            nyquist_order = self.nyquist_order

        real_detuning = self.detuning-self.demod_freq+self.sampling_rate*nyquist_order
        return real_detuning

    def get_amplitudes(self):
        """
        Converts the real detuning to amplitude
        """
        real_detuning = self.get_real_detuning()
        if hasattr(self, 'freq_to_amp'):

            amplitudes = self.freq_to_amp(real_detuning)
            return amplitudes
        else:
            raise NotImplementedError('Add a "freq_to_amp" method.')

    def plot_short_time_fft(self, ax=None,
                            title='Short time Fourier Transform',
                            window_size=100, **kw):

        if ax is None:
            ax = plt.gca()
        ax.set_title(title)

        f, t, Zxx = ss.stft(self.norm_data, fs=self.sampling_rate,
                            nperseg=window_size,
                            noverlap=0.95 * window_size, return_onesided=False)
        m = np.argsort(f)
        ax.pcolormesh(self.time[0] + t, f[m], np.abs(Zxx)[m, :])
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')

        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)
        formatter = matplotlib.ticker.EngFormatter(unit='Hz')
        ax.yaxis.set_major_formatter(formatter)

    def plot_raw_data(self, ax=None, title="Raw cryoscope data",
                      style=".-", **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)
        ax.plot(self.time, self.data.real, style, label="Re", color="C0")
        ax.plot(self.time, self.data.imag, style, label="Im", color="C1")
        ax.legend()
        set_xlabel(ax, 'Time', 's')
        set_ylabel(ax, "Amplitude", 'a.u.')

    def plot_normalized_data(self, ax=None, title='Normalized cryoscope data',
                             style=".-", **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)

        ax.plot(
            self.time,
            self.norm_data.real,
            style,
            label="Re",
            color="C0")
        ax.plot(self.time, self.norm_data.imag, style, label="Im", color="C1")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)

    def plot_demodulated_data(self, ax=None,
                              title='Demodulated cryoscope data',
                              style=".-", **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)
        ax.plot(
            self.time,
            self.demod_data.real,
            style,
            label="Re",
            color="C0")
        ax.plot(
            self.time,
            self.demod_data.imag,
            style,
            label="Im",
            color="C1")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)

    def plot_normalized_data_circle(self, ax=None,
                                    title='Normalized cryoscope data', **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.plot(self.norm_data.real, self.norm_data.imag, ".")

    def plot_phase(self, ax=None, title="Cryoscope demodulated phase",
                   wrap=False, **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)
        if wrap:
            ax.plot(self.time, self.phase % (2 * np.pi), ".", color="C0")
        else:
            ax.plot(self.time, self.phase, ".", label="Im", color="C0")
        set_xlabel(ax, 'Time', 's')
        set_ylabel(ax, 'Phase', 'deg')

    def plot_detuning(self, ax=None,
                      title="Detuning from demodulation frequency", **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)
        ax.plot(self.time, self.detuning, ".-", color="C0")
        set_xlabel(ax, 'Time', 's')
        set_ylabel(ax, 'Frequency', 'Hz')

    def plot_frequency(self, ax=None, title='Detuning frequency',
                       nyquists=None, style=".-", show_demod_freq=True, **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)

        if nyquists is None:
            nyquists = [self.nyquist_order]
        for n in nyquists:
            if show_demod_freq:
                ax.axhline(-self.demod_freq + self.sampling_rate *
                           n, linestyle='--', c='grey')
            real_detuning = self.get_real_detuning(n)
            ax.plot(self.time, real_detuning, style)
        set_xlabel(ax, 'Time', 's')
        set_ylabel(ax, 'Frequency', 'Hz')

    def plot_amplitude(self, ax=None, title='Cryoscope amplitude',
                       nyquists=None, style=".-", **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)
        amp = self.get_amplitudes()
        ax.plot(self.time, amp, style)
        set_xlabel(ax, 'Time', 's')
        set_ylabel(ax, 'Amplitude', 'V')


def sincos_model_real_imag(times, freq, phase):
    r, i = np.cos(2 *
                  np.pi *
                  times *
                  freq +
                  phase), np.sin(2 *
                                 np.pi *
                                 times *
                                 freq +
                                 phase)
    return np.hstack((r, i))


class DacArchAnalysis:
    """
    Given cryscope time series from a square region in time and amplitude,
    fit complex oscillations to obtain a dac arch. Tries to be smart about
    supersampled signals, after constructing the arc, fits a polynomial
    in order to facilitate interpolation.
    """

    def __init__(
            self,
            times,
            amps,
            data,
            exclusion_indices=[],
            poly_fit_order=2,
            nyquist_calc='auto',
            invert_frequency_sign=False,
            plot_fits=False):
        """
        Extract a dac arch from a set of cryoscope-style measurements.

        times: array of pulse lengths
        amps: array of pulse amplitudes

        data: 2D array of measurement results (size of len(times x amps)).
            Complex numbers containing x- and y- values

        poly_fit_order: order of model used for fitting the dac-arch

        invert_frequency_sign: boolean. might be useful if x and y are
            interchanged in measurement

        plot_fits: plots how the fit is going, for display.
        """
        self.data = data
        self.times = times
        self.amps = np.array(amps)

        self.poly_fit_order = poly_fit_order

        self.sampling_rate = 1 / (self.times[1] - self.times[0])

        self.freqs = []
        self.excl_amps = []
        self.excl_freqs = []
        self.exclusion_indices = exclusion_indices

        self.norm_data = []

        for d in self.data:
            self.norm_data.append(normalize_sincos(d, window_size=11))

        self.norm_data = np.array(self.norm_data)

        for nd in self.norm_data:
            guess_f, guess_ph, *_ = fft_based_freq_guess_complex(nd)
            guess_f *= self.sampling_rate

            nd_real_imag = np.hstack([nd.real, nd.imag])

            fit, err = so.curve_fit(sincos_model_real_imag,
                                    self.times, nd_real_imag,
                                    p0=[guess_f, guess_ph])

            if plot_fits:
                plt.figure()
                plt.plot(self.times, nd.real, "-.b")
                plt.plot(self.times, nd.imag, ".-r")

                tt = np.linspace(self.times[0], self.times[-1], 300)
                plt.plot(tt, sincos_model_real_imag(tt, *fit)[:len(tt)], "--b")
                plt.plot(tt, sincos_model_real_imag(tt, *fit)[len(tt):], "--r")

            self.freqs.append(fit[0])

        self.freqs = np.array(self.freqs)
        if nyquist_calc == 'auto':
            self.nyquist = np.cumsum(self.freqs[1:] < self.freqs[:-1])
            self.nyquist = np.hstack(([0], self.nyquist))
        elif nyquist_calc == 'disabled':
            self.nyquist = np.zeros(len(self.freqs))
        else:
            raise NotImplementedError()
            # FIXME: proper support for auto nyquist with
            # a proper nyquist should be extracte

        self.freqs = self.freqs + self.nyquist * self.sampling_rate

        if invert_frequency_sign:
            self.freqs = -self.freqs

        # Exclude data from excludion indices
        self.filt_freqs = np.delete(self.freqs, self.exclusion_indices)
        self.filt_amps = np.delete(self.amps, self.exclusion_indices)
        self.excl_freqs = self.freqs[self.exclusion_indices]
        self.excl_amps = self.amps[self.exclusion_indices]

        self.poly_fit = np.polyfit(self.filt_amps, self.filt_freqs,
                                   self.poly_fit_order)

        self._inv_interpolation = None

    def amp_to_freq(self, amp):
        """
        Find the frequency that corresponds to a given amplitude by
        evaluating the fit to the extracted data.
        """
        return np.polyval(self.poly_fit, amp)

    def freq_to_amp(self, freq, kind='root_parabola', **kw):
        """
        Find the amplitude that corresponds to a given frequency, by
        numerically inverting the fit.

        freq: The frequency or set of frequencies.
        kind: Which technique to use:
            "interpolate": Uses numerical interpolation to find the inverse.
            Only works if freq is in the range of measured dac values.
            "root": Finds the inverse of the model numerical. Slow, but can
            extrapolate.

        **kw : get passed on to methods that implement the different "kind"
            of calculations.
        """

        if kind == 'interpolate':
            if self._inv_interpolation is None:
                no_samples = 50
                self.sampled_amps = np.linspace(
                    np.min(
                        self.amps), np.max(
                        self.amps), no_samples)

                self.sampled_freqs = self.amp_to_freq(self.sampled_amps)

                self._inv_interpolation = si.interp1d(
                    self.sampled_freqs, self.sampled_amps, kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate')
            return self._inv_interpolation(freq)
        if kind == 'root':
            return np.vectorize(self._freq_to_amp_root)(freq)

        if kind == 'root_parabola':
            # return self._freq_to_amp_root_parabola(freq, **kw)
            return freq_to_amp_root_parabola(freq=freq,
                                             poly_coeffs=self.poly_fit, **kw)

        raise ValueError("`kind` not understood")

    # def _freq_to_amp_root_parabola(self, freq, positive_branch=True):
    #     """
    #     Converts freq in Hz to amplitude.

    #     Requires "poly_fit" to be set to the polynomial values
    #     extracted from the cryoscope flux arc.

    #     Assumes a parabola to find the roots but should also work for a higher
    #     order polynomial, except that it will pick the wrong branch.

    #     N.B. this method assumes that the polycoeffs are with respect to the
    #         amplitude in units of V.
    #     """

    #     # recursive allows dealing with an array of freqs
    #     if isinstance(freq, (list, np.ndarray)):
    #         return np.array([self._freq_to_amp_root_parabola(
    #             f, positive_branch=positive_branch) for f in freq])

    #     p = np.poly1d(self.poly_fit)
    #     sols = (p-freq).roots

    #     # sols returns 2 solutions (for a 2nd order polynomial)
    #     if positive_branch:
    #         sol = np.max(sols)
    #     else:
    #         sol = np.min(sols)

    #     # imaginary part is ignored, instead sticking to closest real value
    #     return np.real(sol)

    def _freq_to_amp_root(self, freq):
        """
        Find the amplitude corresponding to a given frequency by numerically
        inverting the fit.
        """

        poly = np.array(self.poly_fit)
        poly[-1] -= freq

        roots = np.roots(poly)

        # return the solution that is real and closest to the givenamp range

        real_mask = np.abs(roots.imag) < 1e-8

        if not any(real_mask):
            return None

        dist_from_range = np.abs(roots[real_mask] - np.mean(self.amps))

        return roots[real_mask][np.argmin(dist_from_range)].real

    def plot_freqs(self, ax=None, title='', **kw):
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)

        amps_sorted = [x for x, _ in sorted(
            zip(self.filt_amps, self.filt_freqs))]
        freqs_sorted = [y for _, y in sorted(
            zip(self.filt_amps, self.filt_freqs))]

        ax.plot(amps_sorted, freqs_sorted, ".-")
        ax.scatter(self.excl_amps, self.excl_freqs, marker='x', color='C3')
        aa = np.linspace(min(self.amps), max(self.amps), 50)
        ax.plot(aa, np.polyval(self.poly_fit, aa), label='fit')
        set_xlabel(ax, "Amplitude", 'V')
        set_ylabel(ax, 'Detuning', 'Hz')

    def plot_ffts(self, ax=None, title='', nyquist_unwrap=False, **kw):
        if ax is None:
            ax = plt.gca()
        if nyquist_unwrap:
            raise NotImplementedError
        ax.set_title(title)
        ffts = np.fft.fft(self.norm_data)

        freqs = np.arange(len(ffts[0])) * self.sampling_rate / len(ffts[0])

        flex_colormesh_plot_vs_xy(xvals=np.array(self.amps), yvals=freqs,
                                  zvals=np.abs(ffts).T,
                                  ax=ax)

        ax.scatter(self.filt_amps, self.filt_freqs % self.sampling_rate, color="C1",
                   facecolors='none', label='Dominant freqs.')

        ax.scatter(self.excl_amps, self.excl_freqs % self.sampling_rate,
                   color="C3",
                   marker='x')
        aa = np.linspace(min(self.amps), max(self.amps), 300)

        ax.plot(aa, np.polyval(self.poly_fit, aa) % self.sampling_rate, "r",
                label='fit')
        set_xlabel(ax, "Amplitude", 'V')  # a.u.
        set_ylabel(ax, 'Detuning', 'Hz')
        ax.legend()


def freq_to_amp_root_parabola(freq, poly_coeffs, positive_branch=True):
    """
    Converts freq in Hz to amplitude in V.

    Requires "poly_coeffs" to be set to the polynomial values
    extracted from the cryoscope flux arc.

    Assumes a parabola to find the roots but should also work for a higher
    order polynomial, except that it will pick the wrong branch.

    N.B. this method assumes that the polycoeffs are with respect to the
        amplitude in units of V.
    """
    # recursive allows dealing with an array of freqs
    if isinstance(freq, (list, np.ndarray)):
        return np.array([freq_to_amp_root_parabola(
            freq=f, poly_coeffs=poly_coeffs,
            positive_branch=positive_branch) for f in freq])

    p = np.poly1d(poly_coeffs)
    sols = (p-freq).roots

    # sols returns 2 solutions (for a 2nd order polynomial)
    if positive_branch:
        sol = np.max(sols)
    else:
        sol = np.min(sols)

    # imaginary part is ignored, instead sticking to closest real value
    return np.real(sol)
