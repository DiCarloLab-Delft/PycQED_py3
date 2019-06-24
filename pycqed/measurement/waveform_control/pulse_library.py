"""
Library containing pulse shapes.
"""

import numpy as np
import scipy as sp
from pycqed.measurement.waveform_control.pulse import Pulse, apply_modulation
from pycqed.utilities.general import int_to_bin


class SSB_DRAG_pulse(Pulse):
    '''
    Gauss pulse on the I channel, derivative of Gauss on the Q channel.
    modulated with Single Sideband (SSB)  modulation.

    Required arguments:
        name (str) : base name of the pulse
        I_channel (str) : name of the channel on which to act (as defined in pular)
        Q_channel (str) : " "

    kwargs:
        amplitude (V)
        sigma (s)
        nr_sigma (int) (default=4)
        motzoi ( ) (default=0)

        mod_frequency (Hz)
        phase (deg)
        phaselock (bool)

        alpha (arb. units): QI amplitude
        phi_skew (deg) :    phase skewness

    I_env is a gaussian
    Q_env is the derivative of a gaussian
    The envelope is transformation:
    Signal = predistortion * modulation * envelope

    See Leo's notes on mixer predistortion in the docs for details

    [I_mod] = [1        tan(phi-skew)] [cos(wt+phi)   sin(wt+phi)] [I_env]
    [Q_mod]   [0  sec(phi-skew)/alpha] [-sin(wt+phi)  cos(wt+phi)] [Q_env]


    The predistortion * modulation matrix is implemented in a single step using
    the following matrix

    M*mod = [cos(x)-tan(phi-skew)sin(x)      sin(x)+tan(phi-skew)cos(x) ]
            [-sin(x)sec(phi-skew)/alpha  cos(x)sec(phi-skew)/alpha]

    where: x = wt+phi

    Reduces to a Gaussian pulse if motzoi == 0
    Reduces to an unmodulated pulse if mod_frequency == 0
    '''

    def __init__(self, name, element_name, I_channel, Q_channel, **kw):
        super().__init__(name, element_name)
        self.I_channel = I_channel
        self.Q_channel = Q_channel
        self.channels = [I_channel, Q_channel]

        self.amplitude = kw.pop('amplitude', 0.1)
        self.sigma = kw.pop('sigma', 0.25e-6)
        self.nr_sigma = kw.pop('nr_sigma', 4)
        self.motzoi = kw.pop('motzoi', 0)

        self.mod_frequency = kw.pop('mod_frequency', 1e6)
        self.phase = kw.pop('phase', 0.)
        self.phaselock = kw.pop('phaselock', True)

        self.alpha = kw.pop('alpha', 1)  # QI amp ratio
        self.phi_skew = kw.pop('phi_skew', 0)  # IQ phase skewness

        self.length = self.sigma * self.nr_sigma
        self.codeword = kw.pop('codeword', 'no_codeword')

    def __call__(self, **kw):
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.sigma = kw.pop('sigma', self.sigma)
        self.nr_sigma = kw.pop('nr_sigma', self.nr_sigma)
        self.motzoi = kw.pop('motzoi', self.motzoi)
        self.mod_frequency = kw.pop('mod_frequency', self.mod_frequency)
        self.phase = kw.pop('phase', self.phase)
        self.phaselock = kw.pop('phaselock', self.phaselock)

        self.length = self.sigma * self.nr_sigma
        return self

    def chan_wf(self, chan, tvals):
        idx0 = np.where(tvals >= tvals[0])[0][0]
        idx1 = np.where(tvals <= tvals[0] + self.length)[0][-1] + 1
        wf = np.zeros(len(tvals))
        t = tvals - tvals[0]  # Gauss envelope should not be displaced
        mu = self.length / 2.0
        if not self.phaselock:
            tvals = tvals.copy() - tvals[idx0]

        gauss_env = self.amplitude * np.exp(-(0.5 * (
            (t - mu)**2) / self.sigma**2))
        deriv_gauss_env = self.motzoi * -1 * (t - mu) / (self.sigma **
                                                         1) * gauss_env
        # substract offsets
        gauss_env -= (gauss_env[0] + gauss_env[-1]) / 2.
        deriv_gauss_env -= (deriv_gauss_env[0] + deriv_gauss_env[-1]) / 2.

        # Note prefactor is multiplied by self.sigma to normalize
        if chan == self.I_channel:
            I_mod, Q_mod = apply_modulation(
                gauss_env,
                deriv_gauss_env,
                tvals[idx0:idx1],
                mod_frequency=self.mod_frequency,
                phase=self.phase,
                phi_skew=self.phi_skew,
                alpha=self.alpha)
            wf[idx0:idx1] += I_mod

        if chan == self.Q_channel:
            I_mod, Q_mod = apply_modulation(
                gauss_env,
                deriv_gauss_env,
                tvals[idx0:idx1],
                mod_frequency=self.mod_frequency,
                phase=self.phase,
                phi_skew=self.phi_skew,
                alpha=self.alpha)
            wf[idx0:idx1] += Q_mod

        return wf


class BufferedSquarePulse(Pulse):
    def __init__(self,
                 element_name,
                 channel=None,
                 channels=None,
                 name='buffered square pulse',
                 **kw):
        super().__init__(name, element_name)
        if channel is None and channels is None:
            raise ValueError('Must specify either channel or channels')
        elif channels is None:
            self.channels.append(channel)
        else:
            self.channels = channels
        self.amplitude = kw.pop('amplitude', 0)

        self.pulse_length = kw.pop('pulse_length', 0)
        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.codeword = kw.pop('codeword', 'no_codeword')

    def __call__(self, **kw):
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.pulse_length = kw.pop('pulse_length', self.pulse_length)
        self.buffer_length_start = kw.pop('buffer_length_start',
                                          self.buffer_length_start)
        self.buffer_length_end = kw.pop('buffer_length_end',
                                        self.buffer_length_end)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma',
                                            self.gaussian_filter_sigma)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.channels = kw.pop('channels', self.channels)
        self.channels.append(self.channel)
        return self

    def chan_wf(self, chan, tvals):
        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * self.amplitude
            wave *= (tvals >= tvals[0] + self.buffer_length_start)
            wave *= (tvals <
                     tvals[0] + self.buffer_length_start + self.pulse_length)
            return wave
        else:
            tstart = tvals[0] + self.buffer_length_start
            tend = tvals[0] + self.buffer_length_start + self.pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                    (tvals - tend) * scaling)) * self.amplitude
            return wave


class BufferedCZPulse(Pulse):
    def __init__(self,
                 channel,
                 element_name,
                 aux_channels_dict=None,
                 name='buffered CZ pulse',
                 **kw):
        super().__init__(name, element_name)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.amplitude = kw.pop('amplitude', 0)
        self.frequency = kw.pop('frequency', 0)
        self.phase = kw.pop('phase', 0.)

        self.pulse_length = kw.pop('pulse_length', 0)
        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse', 5e-9)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.codeword = kw.pop('codeword', 'no_codeword')

    def __call__(self, **kw):
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.pulse_length = kw.pop('pulse_length', self.pulse_length)
        self.buffer_length_start = kw.pop('buffer_length_start',
                                          self.buffer_length_start)
        self.buffer_length_end = kw.pop('buffer_length_end',
                                        self.buffer_length_end)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse',
                                             self.extra_buffer_aux_pulse)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma',
                                            self.gaussian_filter_sigma)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.channels = kw.pop('channels', self.channels)
        self.channels.append(self.channel)
        return self

    def chan_wf(self, chan, tvals):
        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if chan != self.channel:
            amp = self.aux_channels_dict[chan]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * amp
            wave *= (tvals >= tvals[0] + buffer_start)
            wave *= (tvals < tvals[0] + buffer_start + pulse_length)
        else:
            tstart = tvals[0] + buffer_start
            tend = tvals[0] + buffer_start + pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                    (tvals - tend) * scaling)) * amp
        t_rel = tvals - tvals[0]
        wave *= np.cos(
            2 * np.pi * (self.frequency * t_rel + self.phase / 360.))
        return wave


class NZBufferedCZPulse(Pulse):
    def __init__(self, channel, aux_channels_dict=None,
                 name='NZ buffered CZ pulse', **kw):
        super().__init__(name)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.amplitude = kw.pop('amplitude', 0) #of first half
        self.alpha = kw.pop('alpha', 1) #this will be applied to 2nd half
        self.pulse_length = kw.pop('pulse_length', 0)
        self.length1 = self.alpha*self.pulse_length/(self.alpha + 1)

        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse', 5e-9)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

        # these are here so that we can use the CZ pulse dictionary that is
        # created by add_CZ_pulse in QuDev_transmon.py
        self.frequency = kw.pop('frequency', 0)
        self.phase = kw.pop('phase', 0.)

    def __call__(self, **kw):
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.alpha = kw.pop('alpha', self.alpha)
        self.pulse_length = kw.pop('pulse_length', self.pulse_length)
        self.length1 = self.alpha*self.pulse_length/(self.alpha + 1)
        self.buffer_length_start = kw.pop('buffer_length_start',
                                          self.buffer_length_start)
        self.buffer_length_end = kw.pop('buffer_length_end',
                                        self.buffer_length_end)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse',
                                             self.extra_buffer_aux_pulse)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma',
                                            self.gaussian_filter_sigma)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.channels = kw.pop('channels', self.channels)
        self.channels.append(self.channel)
        return self

    def chan_wf(self, chan, tvals):
        amp1 = self.amplitude
        amp2 = -self.amplitude*self.alpha
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        l1 = self.length1
        if chan != self.channel:
            amp1 = self.aux_channels_dict[chan]
            amp2 = -amp1*self.alpha
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2*self.extra_buffer_aux_pulse
            l1 = self.alpha*pulse_length/(self.alpha + 1)

        if self.gaussian_filter_sigma == 0:
            wave1 = np.ones_like(tvals)*amp1
            wave1 *= (tvals >= tvals[0] + buffer_start)
            wave1 *= (tvals < tvals[0] + buffer_start + l1)

            wave2 = np.ones_like(tvals)*amp2
            wave2 *= (tvals >= tvals[0] + buffer_start + l1)
            wave2 *= (tvals < tvals[0] + buffer_start + pulse_length)

            wave = wave1 + wave2
        else:
            tstart = tvals[0] + buffer_start
            tend = tvals[0] + buffer_start + l1
            tend2 = tvals[0] + buffer_start + pulse_length
            scaling = 1/np.sqrt(2)/self.gaussian_filter_sigma
            wave = 0.5*(amp1*sp.special.erf((tvals - tstart)*scaling) -
                        amp1*sp.special.erf((tvals - tend)*scaling) +
                        amp2*sp.special.erf((tvals - tend)*scaling) -
                        amp2*sp.special.erf((tvals - tend2)*scaling))
        return wave


class NZMartinisGellarPulse(Pulse):
    def __init__(self, channel, wave_generation_func,
                 aux_channels_dict=None,
                 name='NZMartinisGellarPulse', **kw):
        super().__init__(name)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.theta_f = kw.pop('theta_f', np.pi/2)
        self.alpha = kw.pop('alpha', 1) # this will be applied to 2nd half
        self.pulse_length = kw.pop('pulse_length', 0)

        self.buffer_length_start = kw.pop('buffer_length_start', 0)
        self.buffer_length_end = kw.pop('buffer_length_end', 0)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse', 5e-9)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

        self.wave_generation_func = wave_generation_func
        self.qbc_freq = kw.pop('qbc_freq', 0)
        self.qbt_freq = kw.pop('qbt_freq', 0)
        self.anharmonicity = kw.pop('anharmonicity', 0)
        self.J = kw.pop('J', 0)
        self.loop_asym = kw.pop('loop_asym', 0)
        self.dphi_dV= kw.pop('dphi_dV', 0)
        self.lambda_2 = kw.pop('lambda_2', 0)

    def __call__(self, **kw):
        self.theta_f = kw.pop('theta_f', self.theta_f)
        self.alpha = kw.pop('alpha', self.alpha)
        self.pulse_length = kw.pop('pulse_length', self.pulse_length)
        self.buffer_length_start = kw.pop('buffer_length_start',
                                          self.buffer_length_start)
        self.buffer_length_end = kw.pop('buffer_length_end',
                                        self.buffer_length_end)
        self.extra_buffer_aux_pulse = kw.pop('extra_buffer_aux_pulse',
                                             self.extra_buffer_aux_pulse)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end
        self.channels = kw.pop('channels', self.channels)
        self.channels.append(self.channel)

        self.wave_generation_func = kw.pop('wave_generation_func',
                                           self.wave_generation_func)
        self.qbc_freq = kw.pop('qbc_freq', self.qbc_freq)
        self.qbt_freq = kw.pop('qbt_freq', self.qbt_freq)
        self.J = kw.pop('J', self.J)
        self.loop_asym = kw.pop('loop_asym', self.loop_asym)
        self.dphi_dV = kw.pop('dphi_dV', self.dphi_dV)
        self.lambda_2 = kw.pop('lambda_2', self.lambda_2)
        return self

    def chan_wf(self, chan, tvals):
        params_dict = {
            'pulse_length': self.pulse_length,
            'theta_f': self.theta_f,
            'qbc_freq': self.qbc_freq,
            'qbt_freq': self.qbt_freq,
            'anharmonicity': self.anharmonicity,
            'J': self.J,
            'dphi_dV': self.dphi_dV,
            'loop_asym': self.loop_asym,
            'lambda_2': self.lambda_2,
            'alpha': self.alpha,
            'buffer_length_start': self.buffer_length_start
        }
        return self.wave_generation_func(tvals, params_dict)


class GaussFilteredCosIQPulse(Pulse):
    def __init__(self,
                 I_channel,
                 Q_channel,
                 element_name,
                 name='gauss filtered cos IQ pulse',
                 **kw):
        super().__init__(name, element_name)

        self.I_channel = I_channel
        self.Q_channel = Q_channel
        self.channels = [self.I_channel, self.Q_channel]

        self.amplitude = kw.pop('amplitude', 0)
        self.mod_frequency = kw.pop('mod_frequency', 0)
        self.phase = kw.pop('phase', 0.)
        self.phi_skew = kw.pop('phi_skew', 0.)
        self.alpha = kw.pop('alpha', 1.)

        self.pulse_length = kw.pop('pulse_length', 0)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma', 0)
        self.nr_sigma = kw.pop('nr_sigma', 5)
        self.phase_lock = kw.pop('phase_lock', False)
        self.length = self.pulse_length + \
                      self.gaussian_filter_sigma*self.nr_sigma
        self.codeword = kw.pop('codeword', 'no_codeword')

    def __call__(self, **kw):
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.mod_frequency = kw.pop('mod_frequency', self.mod_frequency)
        self.pulse_length = kw.pop('pulse_length', self.pulse_length)
        self.gaussian_filter_sigma = kw.pop('gaussian_filter_sigma',
                                            self.gaussian_filter_sigma)
        self.nr_sigma = kw.pop('nr_sigma', self.nr_sigma)
        self.length = self.pulse_length + \
                      self.gaussian_filter_sigma*self.nr_sigma
        self.channels = kw.pop('channels', self.channels)
        self.channels.append(self.I_channel)
        self.channels.append(self.Q_channel)
        return self

    def chan_wf(self, chan, tvals, **kw):
        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * self.amplitude
            wave *= (tvals >= tvals[0])
            wave *= (tvals < tvals[0] + self.pulse_length)
        else:
            tstart = tvals[0] + 0.5 * self.gaussian_filter_sigma * self.nr_sigma
            tend = tstart + self.pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                    (tvals - tend) * scaling)) * self.amplitude
        I_mod, Q_mod = apply_modulation(
            wave,
            np.zeros_like(wave),
            tvals,
            mod_frequency=self.mod_frequency,
            phase=self.phase,
            phi_skew=self.phi_skew,
            alpha=self.alpha,
            phase_lock=self.phase_lock)
        if chan == self.I_channel:
            return I_mod
        if chan == self.Q_channel:
            return Q_mod
