import numpy as np
from copy import deepcopy
import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulse as bpl  # base pulse lib
import pycqed.measurement.waveform_control.pulsar as ps
from collections import OrderedDict as odict


class Sequence:
    """
    A Sequence consists of several segments, which can be played back on the 
    AWGs sequentially.
    """

    def __init__(self, name, pulsar):
        self.name = name
        self.pulsar = pulsar
        self.segments = []

    def gen_waveforms(self):
        """
        Returns a dictionary containing for each AWG, for each element on that
        AWG, for each channel the waveforms of all segments, as well as a 
        list providing the order in which the elements should be played on the
        AWGs.
        """

        return None


class Segment:
    """
    Consists of a list of UnresolvedPulses, each of which contains information about in
    which element the pulse is played and when it is played (reference point + delay)
    as well as an instance of class Pulse.
    """

    def __init__(self, name, pulsar):
        self.name = name
        self.pulsar = pulsar
        self.unresolved_pulses = []
        self.previous_pulse = None
        self.elements = odict()
        self.element_start_end = {}
        self.elements_on_AWG = {}
        self.trigger_pars = {'length': 20e-9, 'amplitude': 1}
        self._pulse_names = set()

    def add(self, pulse_pars):

        pars_copy = deepcopy(pulse_pars)

        if pars_copy.get('name') in self._pulse_names:
            raise ValueError('Name of added pulse already exists!')
        if pars_copy.get('name', None) is None:
            pars_copy['name'] = pulse_pars['pulse_type'] + '_' + str(
                len(self.unresolved_pulses))
        self._pulse_names.add(pars_copy['name'])

        new_pulse = UnresolvedPulse(pars_copy)

        if new_pulse.ref_pulse == 'previous_pulse':
            if self.previous_pulse != None:
                new_pulse.ref_pulse = self.previous_pulse.pulse_obj.name
            else:
                raise ValueError('No previous pulse has been added!')

        self.unresolved_pulses.append(new_pulse)

        self.previous_pulse = new_pulse
        # if self.elements is odict(), the resolve_timing function has to be
        # called prior to generating the waveforms
        self.elements = odict()

    def resolve_timing(self):
        """
        For each pulse in the unresolved_pulses list, this method:
            * updates the _t0 of the pulse by using the timing description of
              the UnresolvedPulse
            * saves the resolved pulse in the elements ordered dictionary by 
              ascending element start time and the pulses in each element by 
              ascenging _t0
            * orderes the UnresolvedPulses list by ascending _t0
        """

        visited_pulses = []
        ref_points = []
        i = 0

        pulses = self.gen_refpoint_dict()

        # add pulses that refer to segment start
        for pulse in pulses['segment_start']:

            if pulse.pulse_obj.name in pulses:
                ref_points.append((pulse.pulse_obj.name, pulse))

            t0 = pulse.delay - pulse.ref_point_new * pulse.pulse_obj.length
            pulse.pulse_obj.algorithm_time(t0)
            visited_pulses.append((t0, i, pulse))
            i += 1

        if len(visited_pulses) == 0:
            raise ValueError('No pulse references to the segment start!')

        while len(ref_points) > 0:
            new_ref_points = []
            for (name, pulse) in ref_points:
                for p in pulses[name]:

                    # add p.name to reference list if it is used as a key
                    # in pulses
                    if p.pulse_obj.name in pulses:
                        new_ref_points.append((p.pulse_obj.name, p))

                    t0 = pulse.pulse_obj.algorithm_time() + p.delay - \
                        p.ref_point_new * p.pulse_obj.length + \
                        p.ref_point * pulse.pulse_obj.length
                    p.pulse_obj.algorithm_time(t0)

                    visited_pulses.append((t0, i, p))
                    i += 1

            ref_points = new_ref_points

        if len(visited_pulses) != len(self.unresolved_pulses):
            raise Exception('Not all pulses have been added!')

        # adds the resolved pulses to the elements OrderedDictionary
        for (t0, i, p) in sorted(visited_pulses):
            if p.pulse_obj.element_name not in self.elements:
                self.elements[p.pulse_obj.element_name] = [p.pulse_obj]
            elif p.pulse_obj.element_name in self.elements:
                self.elements[p.pulse_obj.element_name].append(p.pulse_obj)

        # sort unresolved_pulses by ascending pulse middle. Used for Z_gate
        # resolution
        for i in range(len(visited_pulses)):
            t0 = visited_pulses[i][0]
            p = visited_pulses[i][2]
            visited_pulses[i] = (t0 + p.pulse_obj.length / 2,
                                 visited_pulses[i][1], p)

        ordered_unres_pulses = []
        for (t0, i, p) in sorted(visited_pulses):
            ordered_unres_pulses.append(p)

        self.unresolved_pulses = ordered_unres_pulses

        self.find_element_start_end()
        self.resolve_Z_gates()

    def gen_refpoint_dict(self):
        """
        Returns a dictionary of UnresolvedPulses with their reference_points as 
        keys.
        """

        pulses = {}

        for pulse in self.unresolved_pulses:
            if pulse.ref_pulse not in pulses:
                pulses[pulse.ref_pulse] = [pulse]
            elif pulse.ref_pulse in pulses:
                pulses[pulse.ref_pulse].append(pulse)

        return pulses

    def gen_AWG_dict(self):
        """
        Returns a dictionary with element names as keys and a set of used
        AWGs for each element as value.
        """

        if self.elements == odict():
            self.resolve_timing()

        AWG_dict = {}

        for element in self.elements:
            AWG_dict[element] = set()

            for pulse in self.elements[element]:
                for channel in pulse.channels:
                    AWG_dict[element].add(self.pulsar.get(channel + '_awg'))

        return AWG_dict

    def gen_elements_on_AWG(self):
        """
        Updates the self.elements_on_AWG dictionary
        """

        if self.elements == odict():
            self.resolve_timing()

        self.elements_on_AWG = {}

        for element in self.elements:
            for pulse in self.elements[element]:
                for channel in pulse.channels:
                    AWG = self.pulsar.get(channel + '_awg')
                    if AWG in self.elements_on_AWG and \
                        element not in self.elements_on_AWG[AWG]:
                        self.elements_on_AWG[AWG].append(element)
                    elif AWG not in self.elements_on_AWG:
                        self.elements_on_AWG[AWG] = [element]

    def gen_trigger_el(self):
        """
        For each element:
            For each AWG the element is played on, this method:
                * adds the element to the elements_on_AWG dictionary
                * instatiates a trigger pulse on the triggering channel of the 
                  AWG, placed in a suitable element on the triggering AWG,
                  taking AWG delay into account.
                * adds the trigger pulse to the elements list 
        """

        if self.elements == odict():
            self.resolve_timing()

        AWG_dict = self.gen_AWG_dict()

        for element in AWG_dict:
            for AWG in AWG_dict[element]:

                if self.pulsar.get('{}_trigger_channels'.format(AWG)) == None:
                    continue

                trigger_pulse_time = self.element_start_end[element][
                    0] - self.pulsar.get(AWG + '_delay')

                # Find the trigger_AWGs that trigger the AWG
                trigger_AWGs = set()
                for channel in self.pulsar.get(
                        '{}_trigger_channels'.format(AWG)):
                    trigger_AWGs.add(self.pulsar.get('{}_awg'.format(channel)))

                # For each trigger_AWG, find the elements to play the trigger
                # pulse in
                trigger_elements = {}
                for trigger_AWG in trigger_AWGs:
                    # if there is no element on that AWG create a new element
                    if self.elements_on_AWG.get(trigger_AWG, None) == None:
                        trigger_elements[trigger_AWG] = 'trigger_element'
                    # else find the element that is closest to the
                    # trigger pulse
                    else:
                        trigger_elements[
                            trigger_AWG] = self.find_trigger_element(
                                trigger_AWG, trigger_pulse_time)

                # add the trigger pulse to all triggering channels
                for channel_name in self.pulsar.get(
                        '{}_trigger_channels'.format(AWG)):
                    trig_pulse = bpl.SquarePulse(
                        trigger_elements[self.pulsar.get(
                            '{}_awg'.format(channel_name))],
                        channel=channel_name,
                        **self.trigger_pars)

                    trig_pulse.algorithm_time(trigger_pulse_time)

                    if trig_pulse.element_name in self.elements:
                        self.elements[trig_pulse.element_name].append(
                            trig_pulse)
                    else:
                        self.elements[trig_pulse.element_name] = [trig_pulse]

        # tests if any of the elements overlap. True indicates that element
        # dictionary will be time sorted prior to testing.
        self.test_overlap(True)

    def find_trigger_element(self, trigger_AWG, trigger_pulse_time):
        """
        For a trigger_AWG that is used for generating triggers as well as 
        normal pulses, this method returns the name of the element to which the 
        trigger pulse is closest.
        """

        time_distance = []

        for element in self.elements_on_AWG[trigger_AWG]:
            for i in [0, 1]:
                time_distance.append([
                    abs(trigger_pulse_time + self.trigger_pars['length'] -
                        self.element_start_end[element][i]), element
                ])

        trigger_element = min(time_distance)[1]

        return trigger_element

    def test_overlap(self, sort=False):
        """
        Tests for all AWGs if any of their elements overlap. sort=True 
        indicates that element dictionary will be time sorted prior to testing.
        """

        # find new start and end times of all elements and sort them by
        # accending t0

        if sort:
            self.find_element_start_end(True)

        self._test_trigger_AWG()
        self.gen_elements_on_AWG

        for AWG in self.elements_on_AWG:
            for i in range(len(self.elements_on_AWG[AWG]) - 1):
                next_el = self.elements_on_AWG[AWG][i + 1]
                prev_el = self.elements_on_AWG[AWG][i]
                if self.element_start_end[next_el][0] < \
                        self.element_start_end[prev_el][1]:
                    raise ValueError('{} and {} on {} overlap!'.format(
                        self.elements_on_AWG[AWG][i],
                        self.elements_on_AWG[AWG][i + 1], AWG))

    def _test_trigger_AWG(self):
        """
        Checks if there is more than one element on the AWGs that are not 
        triggered by another AWG.
        """
        self.gen_elements_on_AWG()

        for AWG in self.elements_on_AWG:
            if self.pulsar.get('{}_trigger_channels'.format(AWG)) == None:
                if len(self.elements_on_AWG[AWG]) > 1:
                    raise ValueError(
                        'There is more than one element on {}'.format(AWG))

    def resolve_Z_gates(self):
        qubit_phases = {}

        for pulse in self.unresolved_pulses:
            for qubit in pulse.basis_rotation:
                if qubit in qubit_phases:
                    qubit_phases[qubit] += pulse.basis_rotation[qubit]
                else:
                    qubit_phases[qubit] = pulse.basis_rotation[qubit]

            if pulse.operation_type[0] == 'MW':
                try:
                    pulse.pulse_obj.phase -= qubit_phases[
                        pulse.operation_type[1]]
                except KeyError:
                    qubit_phases[pulse.operation_type[1]] = 0

    def find_element_start_end(self, sort=False):
        """
        Given a segment, this method:
            * finds the start and end times for each element 
            * saves them in element_start_end, ordered by accending start time
            * changes the order of self.elements dictionary by order of start 
              times 
        """

        self.element_start_end = {}

        if self.elements == odict():
            self.resolve_timing()

        if sort:
            # sorts the pulses in the elements with respect to increasing _t0
            self.time_sort()
            unordered_start = []
            new_elements = odict()

            for element in self.elements:
                unordered_start.append(
                    (self.elements[element][0].algorithm_time(), element))

            for (t_start, element) in sorted(unordered_start):
                new_elements[element] = self.elements[element]

            self.elements = new_elements

        for element in self.elements:
            t_start_list = []
            t_end_list = []
            for pulse in self.elements[element]:
                t_start_list.append(pulse.algorithm_time())
                t_end_list.append(pulse.algorithm_time() + pulse.length)

            t_start = min(t_start_list)
            t_end = max(t_end_list)
            self.element_start_end[element] = [t_start, t_end]

    def time_sort(self):
        """
        Given a segment, this method sorts the entries of the elements 
        dictionary (which are lists of UnresolvedPulses) by accending _t0.
        """

        if self.elements == odict():
            self.resolve_timing()

        for element in self.elements:
            # i takes care of pulses happening at the same time, to sort
            # by order in which they were added
            i = 0
            old_list = []
            for pulse in self.elements[element]:
                old_list.append([pulse.algorithm_time(), i, pulse])
                i += 1

            new_list = sorted(old_list)
            self.elements[element] = [pulse for (t0, i, pulse) in new_list]

    def reduce_to_segment_start(self):

        # not used at the moment, change to self.elements

        segment_t0 = float('inf')
        for pulse in self.unresolved_pulses:
            segment_t0 = min(segment_t0, pulse.pulse_obj.algorithm_time())

        for pulse in self.unresolved_pulses:
            pulse.delay -= segment_t0

    def waveforms(self):
        """
        After all the pulses have been added, the timing resolved and the 
        trigger pulses added, the waveforms of the segment can be compiled.
        This method returns a dictionary:
        AWG_wfs = 
          = {AWG_name: 
                {(position_of_element, element_name): 
                    {channel_id: channel_waveforms}
                ...
                }
            ...
            }
        """

        AWG_wfs = {}

        for AWG in self.elements_on_AWG:
            AWG_wfs[AWG] = {}
            channel_list = self.find_AWG_channels(AWG)
            for (i, element) in enumerate(self.elements_on_AWG[AWG]):
                AWG_wfs[AWG][(i, element)] = {}
                tvals = self.tvals(channel_list, element)

                wfs = {}
                for channel in channel_list:
                    wfs[channel] = np.zeros(len(tvals[channel]))

                element_start_time = self.element_start_end[element][0]
                for pulse in self.elements[element]:
                    # checks whether pulse is played on AWG
                    if set(pulse.channels) & set(channel_list) == set():
                        continue
                    chan_tvals = {}
                    for channel in pulse.channels:
                        # checks if pulse is played on AWG
                        if channel not in channel_list:
                            continue

                        pulse_start = self.time2sample(
                            pulse.element_time(element_start_time), channel)
                        pulse_end = self.time2sample(
                            pulse.element_time(element_start_time) +
                            pulse.length, channel)
                        chan_tvals[channel] = tvals[channel].copy(
                        )[pulse_start:pulse_end]
                        print(channel)
                        print(pulse_start, pulse_end)
                        print(len(tvals[channel]))

                    pulse_wfs = pulse.get_wfs(chan_tvals)

                    for channel in pulse.channels:
                        # checks if pulse is played on AWG
                        if channel not in channel_list:
                            continue
                        pulse_start = self.time2sample(
                            pulse.element_time(element_start_time), channel)
                        pulse_end = self.time2sample(
                            pulse.element_time(element_start_time) +
                            pulse.length, channel)
                        wfs[channel][pulse_start:
                                     pulse_end] += pulse_wfs[channel]

                for channel in channel_list:
                    AWG_wfs[AWG][(i, element)][self.pulsar.get(
                        '{}_id'.format(channel))] = (wfs[channel])

        return AWG_wfs

    def tvals(self, channel_list, element):
        """
        Returns a dictionary with channel names of the used channels in the
        element as keys and the tvals array for the channel as values.
        """

        tvals = {}

        for channel in channel_list:
            samples = self.element_samples(element, channel)
            tvals[channel] = np.arange(samples) / self.pulsar.clock(
                channel) + self.element_start_end[element][0]  # + delay?

        return tvals

    def find_AWG_channels(self, AWG):
        channel_list = []
        for channel in self.pulsar.channels:
            if self.pulsar.get('{}_awg'.format(channel)) == AWG:
                channel_list.append(channel)

        return channel_list

    def element_samples(self, element, channel):
        """
        Returns the number of samples the element occupies for the channel.
        """
        el_time = self.element_start_end[element][1] - self.element_start_end[
            element][0]
        return self.time2sample(el_time, channel)

    def time2sample(self, t, channel):
        """
        Converts time to a number of samples for a channel.
        """
        return int(t * self.pulsar.clock(channel) + 0.5)


class UnresolvedPulse:
    """
    pulse_pars: dictionary containing pulse parameters
    reference_point: 'segment_start', 'previous_pulse', pulse.name
    ref_point: 'start', 'end' -- reference point of the reference pulse
    ref_point_new: 'start', 'end' -- reference point of the new pulse
    """

    def __init__(self, pulse_pars):
        self.ref_pulse = pulse_pars.get('reference_pulse', 'previous_pulse')

        if pulse_pars.get('ref_point', 'end') == 'end':
            self.ref_point = 1
        elif pulse_pars.get('ref_point', 'end') == 'middle':
            self.ref_point = 0.5
        elif pulse_pars.get('ref_point', 'end') == 'start':
            self.ref_point = 0

        if pulse_pars.get('ref_point_new', 'start') == 'start':
            self.ref_point_new = 0
        elif pulse_pars.get('ref_point_new', 'start') == 'middle':
            self.ref_point_new = 0.5
        elif pulse_pars.get('ref_point_new', 'start') == 'end':
            self.ref_point_new = 1

        self.delay = pulse_pars['pulse_delay']
        self.original_phase = pulse_pars.get('phase', 0)
        self.operation_type = pulse_pars.pop('operation_type', ("Other", ))
        self.basis_rotation = pulse_pars.pop('basis_rotation', {})

        if self.operation_type[0] == 'MW':
            try:
                self.operation_type[1]
            except:
                raise Exception('For MW pulses a target qubit has to be \
                    specified!')

        try:
            # Look for the function in pl = pulse_lib
            pulse_func = getattr(pl, pulse_pars['pulse_type'])
        except AttributeError:
            try:
                # Look for the function in bpl = pulse
                pulse_func = getattr(bpl, pulse_pars['pulse_type'])
            except AttributeError:
                raise KeyError('pulse_type {} not recognized'.format(
                    pulse_pars['pulse_type']))

        self.pulse_obj = \
            pulse_func(**pulse_pars)
