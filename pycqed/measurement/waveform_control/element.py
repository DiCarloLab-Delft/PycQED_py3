# Implementation of sequence elements that are composed out of pulses
#
# author: Wolfgang Pfaff
# modified by: Adriaan Rol

import numpy as np
import pprint
from copy import deepcopy
import logging
import pycqed.measurement.waveform_control.fluxpulse_predistortion as flux_dist


class Element:
    """
    Implementation of a sequence element.
    Basic idea: add different pulses, and compose the actual numeric
    arrays that form the amplitudes for the hardware (typically an AWG).
    """

    def __init__(self, name, pulsar, **kw):
        self.name = name
        self.pulsar = pulsar

        self.ignore_offset_correction = kw.pop('ignore_offset_correction',
                                               False)

        self.global_time = kw.pop('global_time', True)
        self.time_offset = kw.pop('time_offset', 0)

        self.ignore_delays = kw.pop('ignore_delays', False)
        # Default fixed point, used for aligning RO elements. Aligns first RO
        self.readout_fixed_point = kw.pop('readout_fixed_point', 1e-6)
        # used to track if a correction has been applied
        self.fixed_point_applied = False

        self.pulses = {}
        self._last_added_pulse = None

    # tools for time calculations

    # returns number of samples that can be carried out in time t for channel c
    def _time2sample(self, c, t):
        return int(t * self._clock(c) + 0.5)

    def _time2sampletime(self, c, t):
        return self._sample2time(c, self._time2sample(c, t))

    # calculates passed time given s samples
    def _sample2time(self, c, s):
        return s / self._clock(c)

    # returns number of samples per second for channel c
    def _clock(self, c):
        return self.pulsar.clock(c)

    def offset(self):
        """
        Returns the smallest t0 of all pulses/channels after correcting for
        delay.
        """
        if self.ignore_offset_correction:
            return 0
        else:
            t0s = []
            for p in self.pulses:
                for c in self.pulses[p].channels:
                    if self.pulsar.get('{}_active'.format(c)):
                        t0s.append(self.pulses[p].t0() - self.channel_delay(c))
            offset = min(t0s)
            if self.fixed_point_applied:
                offset += calculate_time_correction(
                              offset, self.readout_fixed_point) - \
                          2*self.readout_fixed_point
            return offset

    def ideal_length(self):
        """
        Returns the nominal length of the element before taking into account
        the discretization using the clock.
        """
        ts = []
        for p in self.pulses:
            for c in self.pulses[p].channels:
                ts.append(self.pulse_end_time(p, c))

        return (max(ts) if len(ts) > 0 else 0.) + \
               self.pulsar.inter_element_spacing()

    def length(self, c):
        """
        Returns the actual length of the sequence, including all corrections.
        """
        return self.samples(c) / self._clock(c)

    def samples(self, c):
        """
        Returns the number of samples the elements occupies.
        """

        ends = []
        for p in self.pulses:
            if c in self.pulses[p].channels:
                ends.append(self.pulse_end_sample(p, c))
        if len(ends) == 0:
            return 0
        samples = max(ends) + 1
        add_spacing = self.pulsar.inter_element_spacing() - \
                      self.pulsar.get('{}_inter_element_deadtime'.format(c))
        samples += int(np.ceil(add_spacing / self._clock(c)))
        samples = max(
            samples,
            int(self.pulsar.get('{}_min_length'.format(c)) * self._clock(c)))
        while samples % self.pulsar.get('{}_granularity'.format(c)) != 0:
            samples += 1
        return samples

    def real_time(self, t, c):
        """
        Returns an actual time, i.e., the correction for delay
        is reversed. It is, however, correctly discretized.
        """
        return int((t + self.channel_delay(c)) * self._clock(c) +
                   0.5) / self._clock(c)

    def real_times(self, tvals, c):
        return ((tvals + self.channel_delay(c)) * self._clock(c) +
                0.5).astype(int) / self._clock(c)

    def shift_all_pulses(self, dt):
        """
        Shifts all pulses by a time dt, this is used for correcting the phase
        of a fixed reference.
        """
        #self.ignore_offset_correction = True
        for name, pulse in self.pulses.items():
            pulse._t0 += dt

    ######################
    # channel management #
    ######################

    def channel_delay(self, c):
        if not self.ignore_delays:
            return self.pulsar.get('{}_delay'.format(c))
        else:
            return 0

    ####################
    # pulse management #
    ####################

    def _auto_pulse_name(self, base='pulse'):
        i = 0
        while base + '-' + str(i) in self.pulses:
            i += 1
        return base + '-' + str(i)

    def add(self,
            pulse,
            name=None,
            start=0,
            refpulse=None,
            refpoint='end',
            refpoint_new='start',
            operation_type='other'):
        """
        Function adds a pulse to the element, there are several options to set
        where in the element the pulse is added.

        name (str)          : name used for referencing the pulse in the
                              element, if not specified generates one based on
                              the default pulse name
        start (float)       : time between refpoint and refpoint_new used to
                              define the start of the pulse
        refpulse (str)      : name of pulse used as reference for timing

        refpoint ('start'|'end'|'center') : reference point in reference
                                            pulse used
        refpoint_new ('start'|'end'|'center'): reference point in added
                                               pulse used
        """
        pulse = deepcopy(pulse)
        pulse.operation_type = operation_type
        if name is None:
            name = self._auto_pulse_name(pulse.name)

        #print('adding pulse ' + name)

        t0 = start - pulse.start_offset
        if refpoint not in ['start', 'center', 'end']:
            raise ValueError('refpoint not recognized')

        if refpoint_new not in ['start', 'center', 'end']:
            raise ValueError('refpoint not recognized')

        if refpulse is not None:
            if refpoint is None:
                refpoint = 'end'

            if refpoint_new == 'start':
                t0 += self.pulses[refpulse].effective_stop()

                if refpoint == 'start':
                    t0 -= self.pulses[refpulse].effective_length()
                elif refpoint == 'center':
                    t0 -= self.pulses[refpulse].effective_length() / 2.

            elif refpoint_new == 'end':
                t0 += (self.pulses[refpulse].effective_stop() -
                       pulse.effective_length())
                if refpoint == 'start':
                    t0 -= self.pulses[refpulse].effective_length()
                elif refpoint == 'center':
                    t0 -= self.pulses[refpulse].effective_length() / 2.

            elif refpoint_new == 'center':
                t0 += (self.pulses[refpulse].effective_stop() -
                       pulse.effective_length() / 2.)
                if refpoint == 'start':
                    t0 -= self.pulses[refpulse].effective_length()
                elif refpoint == 'center':
                    t0 -= self.pulses[refpulse].effective_length() / 2.

        pulse._t0 = t0
        self.pulses[name] = pulse
        self._last_added_pulse = name
        # Shift all pulses to the fixed point for the first RO pulse encountered
        if operation_type == 'RO' and self.fixed_point_applied is False:
            time_corr = calculate_time_correction(t0, self.readout_fixed_point)
            self.shift_all_pulses(time_corr)
            self.fixed_point_applied = True
            #print('adding time correction of {}'.format(time_corr))

        #print('added pulse t0 is {}'.format(pulse.t0()))

        #print('offset is {}'.format(self.offset()))

        return name

    def append(self, *pulses):
        n = None
        for i, p in enumerate(pulses):
            if i == 0:
                n = self.add(
                    p, refpulse=self._last_added_pulse, refpoint='end')
            else:
                n = self.add(p, refpulse=n, refpoint='end')
        return n

    def next_pulse_time(self, cname, t0=0):
        refpulse = self._last_added_pulse

        if refpulse is None:
            return 0

        t0 += self.pulses[refpulse].effective_stop()
        return t0 - self.channel_delay(cname) - self.offset()

    def next_pulse_global_time(self, cname, t0=0):
        refpulse = self._last_added_pulse

        if refpulse is None:
            return self.time_offset - self.channel_delay(cname)

        t0 += self.pulses[refpulse].effective_stop()
        return t0 + self.time_offset - self.offset()

    def pulse_start_time(self, pname, cname):
        return self.pulses[pname].t0() - self.channel_delay(cname) - \
            self.offset()

    def pulse_end_time(self, pname, cname):
        return self.pulses[pname].end() - self.channel_delay(cname) - \
            self.offset()

    def pulse_global_end_time(self, pname, cname):
        return self.pulse_end_time(pname, cname) + self.time_offset - \
            self.offset()

    def pulse_length(self, pname):
        return self.pulses[pname].length

    def pulse_start_sample(self, pname, cname):
        return self._time2sample(cname, self.pulse_start_time(pname, cname))

    def pulse_samples(self, pname, cname):
        return self._time2sample(cname, self.pulses[pname].length)

    def pulse_end_sample(self, pname, cname):
        return self.pulse_start_sample(pname, cname) + \
            self.pulse_samples(pname, cname) - 1

    def effective_pulse_start_time(self, pname, cname):
        return self.pulse_start_time(pname, cname) + \
            self.pulses[pname].start_offset

    def effective_pulse_end_time(self, pname, cname):
        return self.pulse_end_time(pname, cname) - \
            self.pulses[pname].stop_offset

    # computing the numerical waveform
    def ideal_waveforms(self, channels=None):
        wfs = {}
        tvals = {}

        for c in self.pulsar.channels:
            if channels is not None and c not in channels:
                continue
            nsamples = self.samples(c)
            wfs[c] = np.zeros(nsamples)
            tvals[c] = np.arange(nsamples) / self._clock(c) + \
                       self.time_offset + self.channel_delay(c)
        # we first compute the ideal function values
        for p in self.pulses:
            if channels is not None and \
               len(set(self.pulses[p].channels) & set(channels)) == 0:
                continue
            chan_tvals = {}
            for c in self.pulses[p].channels:
                if not self.global_time:
                    psamples = self.pulse_samples(p, c)
                    chan_tvals[c] = tvals[c].copy()[:psamples]
                else:
                    idx0 = self.pulse_start_sample(p, c)
                    idx1 = self.pulse_end_sample(p, c) + 1
                    if idx0 < 0 or idx1 < 0:
                        raise Exception(
                            'Pulse {} on channel {} in element {} starts at a '
                            'negative time.'.format(p, c, self.name))
                    chan_tvals[c] = tvals[c].copy()[idx0:idx1]
            pulsewfs = self.pulses[p].get_wfs(chan_tvals)

            for c in self.pulses[p].channels:
                if channels is not None and c not in channels:
                    continue
                idx0 = self.pulse_start_sample(p, c)
                idx1 = self.pulse_end_sample(p, c) + 1
                wfs[c][idx0:idx1] += pulsewfs[c]

        return tvals, wfs

    def waveforms(self, channels=None):
        """
        return:
            tvals, wfs

        Returns the waveforms for all used channels.
        Trunctates/clips (channel-imposed) all values
        that are out of bounds
        """
        tvals, wfs = self.ideal_waveforms(channels)

        # add charge buildup compensation pulse
        for c in wfs:
            if self.pulsar.get('{}_type'.format(c)) == 'analog':
                if self.pulsar.get('{}_charge_buildup_compensation'.format(c)):
                    tau = self.pulsar.get('{}_discharge_timescale'.format(c))
                    comp_delay = self.pulsar.get(
                        '{}_compensation_pulse_delay'.format(c))
                    amp = self.pulsar.get('{}_amp'.format(c))
                    amp *= self.pulsar.get(
                        '{}_compensation_pulse_scale'.format(c))
                    t = tvals[c]
                    dt = t[1] - t[0]
                    tend = t[-1] + dt
                    wf = wfs[c]
                    if tau is None:
                        integral = wf.sum() * dt
                        if integral > 0:
                            amp = -amp
                        tcomp = -integral / amp
                    else:
                        integral = (wf * np.exp((t - tend) / tau)).sum() * dt
                        if integral > 0:
                            amp = -amp
                        tcomp = tau * np.log(1 - integral / (amp * tau))
                    textra = np.arange(tend, tend + tcomp + 2 * comp_delay, dt)
                    t = np.append(t, textra)
                    wf = np.append(
                        wf,
                        amp * ((textra < tend + tcomp + comp_delay) *
                               (textra >= tend + comp_delay)))
                    tvals[c] = t
                    wfs[c] = wf

        # do predistortion
        for c in wfs:
            if len(wfs[c]) == 0:
                continue
            if self.pulsar.get('{}_type'.format(c)) == 'analog':
                if self.pulsar.get(
                        '{}_distortion'.format(c)) == 'precalculate':
                    distortion_dictionary = self.pulsar.get(
                        '{}_distortion_dict'.format(c))
                    fir_kernels = distortion_dictionary.get('FIR', None)
                    if fir_kernels is not None:
                        if hasattr(fir_kernels, '__iter__') and not \
                           hasattr(fir_kernels[0], '__iter__'): # 1 kernel only
                            wfs[c] = flux_dist.filter_fir(fir_kernels, wfs[c])
                        else:
                            for kernel in fir_kernels:
                                wfs[c] = flux_dist.filter_fir(kernel, wfs[c])
                    iir_filters = distortion_dictionary.get('IIR', None)
                    if iir_filters is not None:
                        wfs[c] = flux_dist.filter_iir(iir_filters[0],
                                                      iir_filters[1], wfs[c])

            # truncate all values that are out of bounds
            amp = self.pulsar.get('{}_amp'.format(c))
            if self.pulsar.get('{}_type'.format(c)) == 'analog':
                if np.max(wfs[c]) > amp:
                    logging.warning('Clipping waveform {} > {}'.format(
                        np.max(wfs[c]), amp))
                if np.min(wfs[c]) < -amp:
                    logging.warning('Clipping waveform {} < {}'.format(
                        np.min(wfs[c]), -amp))
                np.clip(wfs[c], -amp, amp, out=wfs[c])
            elif self.pulsar.get('{}_type'.format(c)) == 'marker':
                wfs[c][wfs[c] > 0] = amp
                wfs[c][wfs[c] <= 0] = 0
        return tvals, wfs
        #returns a dictionary wfs containing
        #channel names as keys and the respective amplitudes as list in values

    def normalized_waveforms(self, channels=None):
        """
        Returns the final numeric arrays, in which channel-imposed
        restrictions are obeyed (bounds, TTL)
        """
        tvals, wfs = self.waveforms(channels)

        for wf in wfs:
            amp = self.pulsar.get('{}_amp'.format(wf))
            # #Temporarily fix to AWG8v1
            # if 'AWG8' in wf:
            #     amp = 1

            if self.pulsar.get('{}_type'.format(wf)) == 'analog':
                wfs[wf] = wfs[wf] / amp
            if self.pulsar.get('{}_type'.format(wf)) == 'marker':
                wfs[wf] = (wfs[wf] > 0).astype(np.int)
        # returns a dictionary wfs containing
        # channel names as keys and the normalized (-1 to 1)
        # amplitudes as list in values
        return tvals, wfs

    # testing and inspection
    def print_overview(self, pulses=True):
        overview = {}
        overview['name'] = self.name
        overview['offset'] = self.offset()
        overview['ideal length'] = self.ideal_length()
        overview['channels'] = {}
        channels = overview['channels']
        for c in self.pulsar.channels:
            if not self.pulsar.get('{}_active'.format(c)):
                continue
            channels[c] = {}
            channels[c]['length'] = self.length(c)
            channels[c]['samples'] = self.samples(c)
        overview['pulses'] = {}
        if pulses:
            pulses = overview['pulses']
            for p in self.pulses:
                pulses[p] = {}
                pulses[p]['length'] = self.pulse_length(p)
                for c in self.pulses[p].channels:
                    if not self.pulsar.get('{}_active'.format(c)):
                        continue
                    pulses[p][c] = {}
                    pulses[p][c]['start time'] = self.pulse_start_time(p, c)
                    pulses[p][c]['end time'] = self.pulse_end_time(p, c)
                    pulses[p][c]['start sample'] = self.pulse_start_sample(
                        p, c)
                    pulses[p][c]['end sample'] = self.pulse_end_sample(p, c)
                    pulses[p][c]['samples'] = self.pulse_samples(p, c)
        pprint.pprint(overview)


# Helper functions, previously part of the element object but moved outside
# to be able to use them in other modules (eg higher level parts of the
# sequencer)


def calculate_time_correction(t0, fixed_point=1e-6):
    return (fixed_point - t0) % fixed_point


def is_divisible_by_clock(value, clock=1e9):
    """
    checks if "value" is divisible by the clock period.
    This funciton is needed because of floating point errors

    It performs this by multiplying everything by 1e11 (looking at 0.01ns
    resolution for divisibility)
    """
    if np.round(value * 1e11) % (1 / clock * 1e11) == 0:
        return True
    else:
        return False


def combine_elements(element_list):
    name = tuple(el.name for el in element_list)
    # create new element with a tuple of the contained elements names as name
    element = Element(
        name,
        element_list[0].pulsar,
        ignore_offset_correction=True,
        global_time=True,
        time_offset=0,
        ignore_delays=element_list[0].ignore_delays)
    element.fixed_point_applied = True
    for i, originalel in enumerate(element_list):
        pulsar = originalel.pulsar
        originalel.pulsar = None
        el = deepcopy(originalel)
        originalel.pulsar = pulsar
        if i != 0:
            # shifts all pulses of the element to be added by one element length
            # - the offset of the element
            el.shift_all_pulses(-originalel.offset() + element.ideal_length())
        # creates pulses dictionary to be added to the combined element
        pulses = {
            el.name + '_' + p + '_' + str(i): el.pulses[p]
            for p in el.pulses
        }
        for p in pulses:
            # takes the pulse with key p and updates the name of the
            # pulse to match the key
            pulses[p].name = p
        # fills the originally empty pulse dict with the shifted pulses
        element.pulses.update(pulses)
    return element
