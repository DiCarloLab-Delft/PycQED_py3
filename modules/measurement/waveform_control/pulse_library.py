import numpy as np
'''
Library containing pulse shapes.
'''


from modules.measurement.waveform_control import pulse


class MW_IQmod_pulse(pulse.Pulse):
    '''
    Block pulse on the I channel modulated with IQ modulation.

    kwargs:
        amplitude (V)
        length (s)
        mod_frequency (Hz)
        phase (deg)
        phaselock (bool)

    I_env is a block pulse
    transformation:
    [I_mod] = [cos(wt+phi)   0] [I_env]
    [Q_mod]   [-sin(wt+phi)  0] [0]
    '''
    def __init__(self, name, I_channel, Q_channel, **kw):
        pulse.Pulse.__init__(self, name)
        self.I_channel = I_channel
        self.Q_channel = Q_channel
        self.channels = [I_channel, Q_channel]

        self.mod_frequency = kw.pop('mod_frequency', 1e6)
        self.amplitude = kw.pop('amplitude', 0.1)
        self.length = kw.pop('length', 1e-6)
        self.phase = kw.pop('phase', 0.)
        self.phaselock = kw.pop('phaselock', True)

    def __call__(self, **kw):
        self.mod_frequency = kw.pop('mod_frequency', self.mod_frequency)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.length = kw.pop('length', self.length)
        self.phase = kw.pop('phase', self.phase)
        self.phaselock = kw.pop('phaselock', self.phaselock)
        return self

    def chan_wf(self, chan, tvals):
        idx0 = np.where(tvals >= tvals[0])[0][0]
        idx1 = np.where(tvals <= tvals[0] + self.length)[0][-1] + 1
        wf = np.zeros(len(tvals))

        # in this case we start the wave with zero phase at the effective start
        # time (up to the specified phase)
        if not self.phaselock:
            tvals = tvals.copy() - tvals[idx0]

        if chan == self.I_channel:
            wf[idx0:idx1] += self.amplitude * np.cos(2 * np.pi * (
                self.mod_frequency * tvals[idx0:idx1] + self.phase/360.))

        if chan == self.Q_channel:
            wf[idx0:idx1] += self.amplitude * np.sin(2 * np.pi * (
                self.mod_frequency * tvals[idx0:idx1] + self.phase/360.))
        return wf


class SSB_DRAG_pulse(pulse.Pulse):
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

    I_env is a gaussian
    transformation:
    [I_mod] = [cos(wt+phi)   sin(wt+phi)] [I_env]
    [Q_mod]   [-sin(wt+phi)  cos(wt+phi)] [Q_env]

    Reduces to a Gaussian pulse if motzoi == 0
    Reduces to an unmodulated pulse if mod_frequency == 0
    '''
    def __init__(self, name, I_channel, Q_channel, **kw):
        pulse.Pulse.__init__(self, name)
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

        self.length = self.sigma * self.nr_sigma

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
        mu = self.length/2.0
        if not self.phaselock:
            tvals = tvals.copy() - tvals[idx0]

        gauss_env = self.amplitude*np.exp(-(0.5 * ((t-mu)**2) / self.sigma**2))
        deriv_gauss_env = self.motzoi * -1 * (t-mu)/(self.sigma**1) * gauss_env
        # substract offsets
        gauss_env -= (gauss_env[0]+gauss_env[-1])/2.
        deriv_gauss_env -= (deriv_gauss_env[0]+deriv_gauss_env[-1])/2.

        # Note prefactor is multiplied by self.sigma to normalize
        if chan == self.I_channel:
            wf[idx0:idx1] += gauss_env * np.cos(2 * np.pi * (
                self.mod_frequency * tvals[idx0:idx1] + self.phase/360.)) \
                + deriv_gauss_env * np.sin(2 * np.pi * (
                    self.mod_frequency * tvals[idx0:idx1] + self.phase/360.))

        if chan == self.Q_channel:
            wf[idx0:idx1] += -gauss_env * np.sin(2 * np.pi * (
                self.mod_frequency * tvals[idx0:idx1] + self.phase/360.))\
                + deriv_gauss_env * np.cos(2 * np.pi * (
                    self.mod_frequency * tvals[idx0:idx1] + self.phase/360.))

        return wf

