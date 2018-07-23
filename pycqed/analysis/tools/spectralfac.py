import logging
import numpy as np


class SpectralFactorization:

    eps = 1e-12

    def __init__(self, power_spectrum, delta_f, zero_padding=0,
                 suppress_small_elements_warning: bool=False):
        """
        Calculate the minimum-phase causal wavelet of a given power spectrum.

        In other words: Given the power transmission spectrum S(f) = |h(f)|² of
        a system with transfer function h(f), reconstruct the phase of h(f)
        so that h(t) = 0 for t < 0.

        The answer is only unique up to an all-pass component; the returned
        solution h(t) is the minimum-phase solution, characterized for instance
        by:

        - h(f) has no zeros with Im(f) > 0.
        - Among all solutions, h(t) is the most energy-concentrated around zero:
            For any $t_0$, $\sum_{t = t_0}^\infty |h(t)|²$ is as small as possible.

        `power_spectrum`: A sampled representation of the one-sided power spectrum.
        `delta_f`: The sampling interval.

        So that |h(n*delta_f)|² = power_spectrum[n].

        `zero_padding`: How many extra zeros to append to the power spectrum
        before using FFT. Experimental.
        """

        self.power_spectrum = power_spectrum
        self.delta_f = delta_f
        self.zero_padding = zero_padding

        if any(self.power_spectrum < self.eps):
            if not suppress_small_elements_warning:
                logging.warning(
                    "Some elements of power spectrum are too small, setting to zero"
                )
            self.power_spectrum[self.power_spectrum<self.eps]=self.eps
            # self.power_spectrum = np.max(self.power_spectrum, self.eps)

        # make two-sided version of the power spectrum, amenable for discrete FFT
        self.two_sided_power_spectrum = np.concatenate([
            power_spectrum,
            np.zeros(2 * self.zero_padding) + self.eps, power_spectrum[-1:0:-1]
        ])

        # size and sampling interval
        self.n_samples = len(self.two_sided_power_spectrum)
        self.delta_t = 1 / (self.delta_f * self.n_samples)

        # The fourier transform of the power spectrum is the autocorrelation of the wavelet
        self.autocorrelation = np.fft.ifft(
            self.two_sided_power_spectrum, norm="ortho")

        # make sure the two-sided version is shifted correctly
        if not np.allclose(self.autocorrelation.imag, 0):
            logging.warning("Autocorrelation not real, something is fishy")

        #self.autocorrelation = self.autocorrelation.real

        self._do_spectral_factorization()

    def _do_spectral_factorization(self):
        """
        Perform the spectral factorization.

        See comments inside function for method.
        """

        # calculate the cepstrum of the magnitude |h|
        # (The cepstrum is the inverse fourier transform of the logarithm of the function living in fourier space)
        # It's independent variable is thus time-like, but usually called quefrency
        self.magnitude_cepstrum = np.fft.ifft(
            np.log(self.two_sided_power_spectrum) / 2, norm='ortho')

        # make the cepstrum causal by folding;
        # remove negative quefrency components, double positive components
        self.magnitude_cepstrum[1:self.n_samples // 2 + 1] *= 2
        self.magnitude_cepstrum[self.n_samples // 2 + 1:] = 0

        # transform back to fourier space
        self.transfer_function = np.exp(
            np.fft.fft(self.magnitude_cepstrum, norm='ortho'))

        # fourier transform again to time to get impulse response
        self.impulse_response = np.fft.ifft(
            self.transfer_function,
            norm='ortho') / (self.delta_t / self.delta_f)**.5

        # integrate to get step response
        self.step_response = np.cumsum(self.impulse_response) * self.delta_t

