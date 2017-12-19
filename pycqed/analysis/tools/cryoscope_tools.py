"""
Tools to for cryoscope analysis
Brian Tarasinski
Dec 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

import scipy.signal as ss
import scipy.optimize as so

def normalize_sincos(data, window_size_frac=500, window_size=None, do_envelope=True):

    if window_size is None:
        window_size = len(data)//window_size_frac

        #window size for savgol filter must be odd
        window_size -= (window_size+1)%2

    mean_data_r = ss.savgol_filter(data.real, window_size, 0, 0)
    mean_data_i = ss.savgol_filter(data.imag, window_size, 0, 0)

    mean_data = mean_data_r + 1j*mean_data_i

    if do_envelope:
        envelope = np.sqrt(ss.savgol_filter((np.abs(data-mean_data))**2, window_size, 0, 0))
    else:
        envelope = 1
    norm_data = ((data-mean_data)/envelope)
    return norm_data

def fft_based_freq_guess_complex(y):
    """
    guess the shape of a sinusoidal complex signal y (in multiples of sampling rate),
    by selecting the peak in the fft.
    return guess (f, ph, off, amp) for the model y = amp*exp(2pi i f t + ph) + off.
    """
    fft = np.fft.fft(y)[1:len(y)]
    freq_guess_idx = np.argmax(np.abs(fft))
    if freq_guess_idx >= len(y)//2:
        freq_guess_idx -= len(y)
    freq_guess = 1/len(y)*(freq_guess_idx+1)
    #freq_guess = 2.4e9/window_size*np.mean(np.abs(fft)*np.arange(1, window_size//2))

    phase_guess = np.angle(fft[freq_guess_idx])+np.pi/2
    amp_guess = np.absolute(fft[freq_guess_idx])/len(y)
    offset_guess = np.mean(y)

    return freq_guess, phase_guess, offset_guess, amp_guess



class CryoscopeAnalyzer:
    def __init__(self, time, complex_data, norm_window_size=61, demod_freq=None,
                 derivative_window_length=None, derivative_order=2,
                 demod_smooth=None):
        """
        analyse a cryoscope measurement.

        time: array of times (lengths of Z pulse)
        complex_data: measured data, combine x- and y- results in a complex number

        norm_window_size: window size used for normalizing sine and cosine

        demod_freq: frequency for demodulation. Is guessed if None.

        derivative_window_length, derivative_order: parameters of the sovgol filter used for extracting frequency.
        Needs some playing around sometimes.

        demod_smooth: when the demodulated signal should be smoothed before taking derivative, set this to
        a tuple (window_length, order), again parametrizing a sovgol filter.

        """
        self.time = time
        self.data = complex_data
        self.norm_data = normalize_sincos(self.data, window_size=61)
        self.demod_freq = demod_freq
        self.derivative_window_length = derivative_window_length
        self.demod_smooth = demod_smooth

        self.sampling_rate = 1/(self.time[1] - self.time[0])

        if self.derivative_window_length is None:
            self.derivative_window_length = 7/self.sampling_rate

        self.derivative_window_size = max(3, int(self.derivative_window_length * self.sampling_rate))
        self.derivative_window_size += (self.derivative_window_size+1)%2

        if self.demod_freq is None:
            self.demod_freq = -fft_based_freq_guess_complex(self.norm_data)[0]*self.sampling_rate

        self.demod_data = np.exp(2*np.pi*1j*self.time*self.demod_freq)*self.norm_data

        if self.demod_smooth:
            n, o = self.demod_smooth
            r, i = self.demod_data.real, self.demod_data.imag
            r = ss.savgol_filter(r, n, o, 0)
            i = ss.savgol_filter(i, n, o, 0)
            self.demod_data = r + 1j*i


        # extract the phase. unwrapping only works well if demodulation is good!
        self.phase = np.unwrap(np.angle(self.demod_data))

        # extract frequency by a lowpass-derivative filter.
        # savitzky golay filter: take sliding window of length `window_length`
        # fit polynomial, return derivative at middle point
        self.detuning = ss.savgol_filter(self.phase/(2*np.pi),
                                         window_length=self.derivative_window_size,
                                         polyorder=derivative_order, deriv=1)*self.sampling_rate

    def plot_short_time_fft(self, window_size=100):

        f, t, Zxx = ss.stft(self.norm_data, fs=self.sampling_rate, nperseg=window_size,
                            noverlap=0.95*window_size, return_onesided=False)
        m = np.argsort(f)

        ax = plt.gca()

        ax.pcolormesh(self.time[0] + t, f[m], np.abs(Zxx)[m, :])
        ax.set_title('Short time Fourier Transform')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)
        formatter = matplotlib.ticker.EngFormatter(unit='Hz')
        ax.yaxis.set_major_formatter(formatter)

    def plot_raw_data(self, style=".-"):
        ax = plt.gca()
        ax.set_title("Raw cryoscope data")
        ax.plot(self.time, self.data.real, style, label="Re", color="blue")
        ax.plot(self.time, self.data.imag, style, label="Im", color="red")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)

    def plot_normalized_data(self, style=".-"):
        ax = plt.gca()
        ax.set_title("Normalized cryoscope data")
        ax.plot(self.time, self.norm_data.real, style, label="Re", color="blue")
        ax.plot(self.time, self.norm_data.imag, style, label="Im", color="red")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)

    def plot_demodulated_data(self, style=".-"):
        ax = plt.gca()
        ax.set_title("Demodulated cryoscope data")
        ax.plot(self.time, self.demod_data.real, style, label="Re", color="blue")
        ax.plot(self.time, self.demod_data.imag, style, label="Im", color="red")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)

    def plot_normalized_data_circle(self):
        plt.title("Normalized cryoscope data")
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.plot(self.norm_data.real, self.norm_data.imag, ".")

    def plot_phase(self, wrap=False):
        ax = plt.gca()
        plt.title("Cryoscope demodulated phase")
        if wrap:
            plt.plot(self.time, self.phase%(2*np.pi),".", color="blue")
        else:
            plt.plot(self.time, self.phase,".", label="Im", color="blue")
        plt.xlabel("Time")
        plt.ylabel("Phase")
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)

    def plot_detuning(self):
        ax = plt.gca()
        plt.title("Detuning from demodulation frequency")
        plt.plot(self.time, self.detuning, ".-", color="blue")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)
        formatter = matplotlib.ticker.EngFormatter(unit='Hz')
        ax.yaxis.set_major_formatter(formatter)

    def plot_frequency(self, nyquists=[0], style=".-", show_demod_freq=True):
        ax = plt.gca()
        plt.title("Detuning frequency")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        formatter = matplotlib.ticker.EngFormatter(unit='s')
        ax.xaxis.set_major_formatter(formatter)
        formatter = matplotlib.ticker.EngFormatter(unit='Hz')
        ax.yaxis.set_major_formatter(formatter)
        for n in nyquists:
            if show_demod_freq:
                plt.axhline(-self.demod_freq + self.sampling_rate*n)
            real_detuning = self.detuning-self.demod_freq+self.sampling_rate*n
            plt.plot(self.time, real_detuning, style)


def sincos_model_real_imag(times, freq, phase):
    r, i = np.cos(2*np.pi*times*freq+phase), np.sin(2*np.pi*times*freq+phase)
    return np.hstack((r, i))



class DacArchAnalysis:
    """
    Given cryscope time series from a square region in time and amplitude,
    fit complex oscillations to obtain a dac arch. Tries to be smart about
    supersampled signals, after constructing the arc, fits a polynomial
    in order to facilitate interpolation.
    """
    def __init__(self, times, amps, data, poly_fit_order=2, invert_frequency_sign=False, plot_fits=False):
        """
        Extract a dac arch from a set of cryoscope-style measurements.

        times: array of pulse lengths
        amps: array of pulse amplitudes

        data: 2D array of measurement results (size of len(times x amps)). Complex numbers containing x- and y- values

        poly_fit_order: order of model used for fitting the dac-arch

        invert_frequency_sign: boolean. might be useful if x and y are interchanged in measurement

        plot_fits: plots how the fit is going, for display.
        """
        self.data = data
        self.times = times
        self.amps = amps

        self.poly_fit_order = poly_fit_order

        self.sampling_rate = 1/(self.times[1] - self.times[0])

        self.freqs = []

        self.norm_data = []

        for d in self.data:
            self.norm_data.append(normalize_sincos(d, window_size=11))

        self.norm_data = np.array(self.norm_data)



        for nd in self.norm_data:
            guess_f, guess_ph, *_ = fft_based_freq_guess_complex(nd)
            guess_f *= self.sampling_rate


            nd_real_imag = np.hstack([nd.real, nd.imag])

            fit, err = so.curve_fit(sincos_model_real_imag, self.times, nd_real_imag, p0=[guess_f, guess_ph])

            if plot_fits:
                plt.figure()
                plt.plot(self.times, nd.real, "-.b")
                plt.plot(self.times, nd.imag, ".-r")

                tt = np.linspace(self.times[0], self.times[-1], 300)
                plt.plot(tt, sincos_model_real_imag(tt, *fit)[:len(tt)], "--b")
                plt.plot(tt, sincos_model_real_imag(tt, *fit)[len(tt):], "--r")

            self.freqs.append(fit[0])


        self.freqs = np.array(self.freqs)

        self.nyquist = np.cumsum(self.freqs[1:] < self.freqs[:-1])
        self.nyquist = np.hstack(([0], self.nyquist))

        self.freqs = self.freqs + self.nyquist*self.sampling_rate

        if invert_frequency_sign:
            self.freqs = -self.freqs


        self.poly_fit = np.polyfit(self.amps, self.freqs, self.poly_fit_order)




    def amp_to_freq(self, amp):
        return np.polyval(self.poly_fit, amp)

    def freq_to_amp(self, freq):

        poly = np.array(self.poly_fit)
        poly[-1] -= freq

        roots = np.roots(poly)

        # return the solution that is real and closest to the givenamp range

        real_mask = np.abs(roots.imag) < 1e-8

        if not any(real_mask):
            return None

        dist_from_range = np.abs(roots[real_mask] - np.mean(self.amps))

        return roots[real_mask][np.argmin(dist_from_range)].real


    def plot_freqs(self):
        plt.plot(self.amps, self.freqs, ".-")
        ax = plt.gca()
        formatter = matplotlib.ticker.EngFormatter(unit='Hz')
        ax.yaxis.set_major_formatter(formatter)
        plt.xlabel("Amplitude")
        plt.ylabel("Detuning")

        aa = np.linspace(min(self.amps), max(self.amps), 50)

        plt.plot(aa, np.polyval(self.poly_fit, aa))

    def plot_ffts(self, nyquist_unwrap=False):

        if nyquist_unwrap:
            raise NotImplementedError

        ffts = np.fft.fft(self.norm_data)


        freqs = np.arange(len(ffts[0]))*self.sampling_rate/len(ffts[0])

        print("shape freqs", freqs.shape)

        def shift_helper(x):
            diff = np.diff(x)/2
            diff = np.hstack((diff[0], diff, -diff[-1]))
            xshift = np.hstack((x, x[-1])) - diff
            return xshift

        print(np.diff(shift_helper(self.amps)))

        aa, ff = np.meshgrid(shift_helper(self.amps), shift_helper(freqs))

        plt.pcolormesh(aa, ff, np.abs(ffts).T)
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        ax = plt.gca()
        formatter = matplotlib.ticker.EngFormatter(unit='Hz')
        ax.yaxis.set_major_formatter(formatter)

        plt.scatter(self.amps, self.freqs%self.sampling_rate, color="red")

        aa = np.linspace(min(self.amps), max(self.amps), 300)

        plt.plot(aa, np.polyval(self.poly_fit, aa)%self.sampling_rate, ".r")

