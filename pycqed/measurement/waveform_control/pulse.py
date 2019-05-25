# This module implements the basic class for pulses as well as some very
# generic pulse types.
#
# author: Wolfgang Pfaff

import numpy as np
import scipy as scipy
from copy import deepcopy


def cp(pulse, *arg, **kw):
    """
    create a copy of the pulse, configure it by given arguments (using the
        call method of the pulse class), and return the copy
    """
    pulse_copy = deepcopy(pulse)

    return pulse_copy(*arg, **kw)


class Pulse:
    """
    A generic pulse. The idea is that a certain implementation of a pulse
    is able to return a 'waveform', which we define as an array of time values
    and an array of amplitude values for each channel of the pulse.

    There are three stages of configuring a pulse:
    1) Implementation of the specific class
    2) when adding to a sequence element (when __call__ is implemented
       in that way)
    3) when the sequence element object calls wf() (this 'finalizes' the
       numerical values).

    A pulse does not yet know about any discretization using a clock.
    This is all done in the sequence element.

    See the examples for more information.
    """

    def __init__(self, name, element_name):
        self.length = None
        self.name = name
        self.element_name = element_name
        self.channels = []
        self.start_offset = 0
        # the time within (or outside) the pulse that is the 'logical' start
        # of the pulse (for referencing)
        self.stop_offset = 0
        # the time within (or outside) the pulse that is the 'logical' stop
        # of the pulse (for referencing)

        self._t0 = None
        self._clock = None

    def __call__(self):
        return self

    def get_wfs(self, tvals, **kw):
        """
        The time values in tvals can always be given as one array of time
        values, or as a separate array for each channel of the pulse.
        """
        wfs = {}
        for c in self.channels:
            if type(tvals) == dict:
                if c not in tvals:
                    continue
                wfs[c] = self.chan_wf(c, tvals[c],**kw)
            else:
                if hasattr(self, 'chan_wf'):
                    wfs[c] = self.chan_wf(c, tvals, **kw)
                elif hasattr(self, 'wf'):
                    wfs = self.wf(tvals)
                else:
                    raise Exception('Could not find a waveform-generator function!')

        return wfs
    
    def pulse_area(self, c, tvals):
        """
        Returns the area of a pulse on the channel c in the time interval 
        tvals.
        """
        if isinstance(tvals, dict):
            wfs = self.chan_wf(c, tvals[c])
            dt = tvals[c][1] - tvals[c][0]
        else:
            if hasattr(self, 'chan_wf'):
                wfs = self.chan_wf(c, tvals)
            elif hasattr(self, 'wf'):
                wfs = self.wf(tvals)
            else:
                raise Exception('Could not find a waveform-generator function!')
            dt = tvals[1] - tvals[0]
        
        return sum(wfs)*dt

    def algorithm_time(self, val=None):
        """
        Getter/Setter for the start time of the pulse.
        """
        if val is None:
            return self._t0
        else:
            self._t0 = val

    def element_time(self, element_start_time):
        """
        Returns the pulse time in the element frame.
        """
        return self.algorithm_time() - element_start_time

    def effective_start(self):
        return self._t0 + self.start_offset

    def end(self):
        """
        returns the end time of the pulse.
        """
        return self._t0 + self.length

    def effective_stop(self):
        return self.end() - self.stop_offset

    def effective_length(self):
        return self.length - self.start_offset - self.stop_offset

# Z virtual pulse
class Z_Pulse(Pulse):

    def __init__(self, element_name, name='Z pulse', **kw):
        super().__init__(name, element_name)
        self.length = 0
        self.codeword = kw.pop('codeword', 'no_codeword')


# Some simple pulse definitions.
class SquarePulse(Pulse):

    def __init__(self, element_name, channel=None, channels=None, name='square pulse', **kw):
        super().__init__(name, element_name)
        if channel is None and channels is None:
            raise ValueError('Must specify either channel or channels')
        elif channels is None:
            self.channel = channel  # this is just for convenience, internally
            # this is the part the sequencer element wants to communicate with
            self.channels.append(channel)
        else:
            self.channels = channels
        self.amplitude = kw.pop('amplitude', 0)
        self.length = kw.pop('length', 0)
        self.codeword = kw.pop('codeword', 'no_codeword')

    def __call__(self, **kw):
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.length = kw.pop('length', self.length)
        self.channel = kw.pop('channel', self.channel)
        self.channels = kw.pop('channels', self.channels)
        self.channels.append(self.channel)
        return self

    def chan_wf(self, chan, tvals):
        return np.ones(len(tvals)) * self.amplitude
        


class CosPulse(Pulse):

    def __init__(self, channel, element_name, name='cos pulse', **kw):
        super().__init__(name, element_name)

        self.channel = channel  # this is just for convenience, internally
        self.channels.append(channel)
        # this is the part the sequencer element wants to communicate with
        self.frequency = kw.pop('frequency', 1e6)
        self.amplitude = kw.pop('amplitude', 0.)
        self.length = kw.pop('length', 0.)
        self.phase = kw.pop('phase', 0.)
        self.codeword = kw.pop('codeword', 'no_codeword')

    def __call__(self, **kw):
        self.frequency = kw.pop('frequency', self.frequency)
        self.amplitude = kw.pop('amplitude', self.amplitude)
        self.length = kw.pop('length', self.length)
        self.phase = kw.pop('phase', self.phase)

        return self

    def chan_wf(self, chan, tvals):
        return self.amplitude * np.cos(2 * np.pi *
                                       (self.frequency * tvals +
                                        self.phase / 360.))

def apply_modulation(I_env, Q_env, tvals, mod_frequency,
                     phase=0, phi_skew=0, alpha=1, phase_lock=False):
    '''
    Applies single sideband modulation, requires timevals to make sure the
    phases are correct.

    Input args:
        I_env (array)
        Q_env (array)
        tvals (array):              in seconds
        mod_frequency(float):       in Hz
        phase (float):              in degree
        phi_skew (float):           in degree
        alpha (float):              ratio
    returns:
        [I_mod, Q_mod] = M*mod*[I_env, Q_env]

    Signal = predistortion * modulation * envelope
    See Leo's notes on mixer predistortion in the docs for details

    [I_mod] = [1        tan(phi-skew)] [cos(wt+phi)   sin(wt+phi)] [I_env]
    [Q_mod]   [0  sec(phi-skew)/alpha] [-sin(wt+phi)  cos(wt+phi)] [Q_env]

    The predistortion * modulation matrix is implemented in a single step
    using the following matrix

    M*mod = [cos(x)-tan(phi-skew)sin(x)      sin(x)+tan(phi-skew)cos(x) ]
            [-sin(x)sec(phi-skew)/alpha  cos(x)sec(phi-skew)/alpha]
    '''
    if phase_lock:
        tvals_wave = tvals - tvals[0]
    else:
        tvals_wave = tvals
    tan_phi_skew = np.tan(2 * np.pi * phi_skew / 360)
    sec_phi_alpha = 1 / (np.cos(2 * np.pi * phi_skew / 360) * alpha)

    I_mod = (I_env * (np.cos(2 * np.pi * (mod_frequency * tvals_wave +
                                          phase / 360)) - tan_phi_skew *
                      np.sin(2 * np.pi * (mod_frequency * tvals_wave +
                                          phase / 360))) +
             Q_env * (np.sin(2 * np.pi * (mod_frequency * tvals_wave +
                                          phase / 360)) + tan_phi_skew *
                      np.cos(2 * np.pi * (mod_frequency * tvals_wave + phase / 360))))

    Q_mod = (-1 * I_env * sec_phi_alpha * np.sin(2 * np.pi * (mod_frequency *
                                                              tvals_wave + phase / 360.)) +
             + Q_env * sec_phi_alpha * np.cos(2 * np.pi * (
                 mod_frequency * tvals_wave + phase / 360.)))
    return [I_mod, Q_mod]
