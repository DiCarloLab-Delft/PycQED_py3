import numpy as np
from copy import deepcopy
import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulse as bpl  # base pulse lib
import pycqed.measurement.waveform_control.pulsar as ps


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
        self.elements = {}
        self.element_start_end = {}
        self.elements_on_AWG = {}
        self.trigger_pars = {'length': 20e-9, 'amplitude': 1}

    def add(self, pulse_pars):

        pars_copy = deepcopy(pulse_pars)

        if pars_copy.get('name') in [
                pulse.pulse_obj.name for pulse in self.unresolved_pulses
        ]:
            raise ValueError('Name of added pulse already exists!')

        elif pars_copy.get('name', None) == None:
            pars_copy['name'] = pulse_pars['pulse_type'] + '_' + str(
                len(self.unresolved_pulses))

        new_pulse = UnresolvedPulse(pars_copy)

        if new_pulse.ref_pulse == 'previous_pulse':
            if self.previous_pulse != None:
                new_pulse.ref_pulse = self.previous_pulse.pulse_obj.name
            else:
                raise ValueError('No previous pulse has been added!')

        self.unresolved_pulses.append(new_pulse)

        self.previous_pulse = new_pulse
        # if self.elements is None, the resolve_timing function has to be
        # called prior to generating the waveforms
        self.elements = {}

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

        if self.elements == {}:
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

        if self.elements == {}:
            self.resolve_timing()

        self.elements_on_AWG = {}

        AWG_dict = self.gen_AWG_dict()

        for element in AWG_dict:
            for AWG in AWG_dict[element]:
                if AWG in self.elements_on_AWG:
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

        if self.elements == {}:
            self.resolve_timing()

        self.gen_elements_on_AWG()

        AWG_dict = self.gen_AWG_dict()

        for element in AWG_dict:
            for AWG in AWG_dict[element]:
                if self.pulsar.get('{}_trigger_channels'.format(AWG)) == None:
                    continue

                trigger_pulse_time = self.element_start_end[element][
                    0] - self.pulsar.get(AWG + '_delay')

                # Find the AWGs that trigger the AWG
                trigger_AWGs = set()
                for channel in self.pulsar.get(
                        '{}_trigger_channels'.format(AWG)):
                    trigger_AWGs.add(self.pulsar.get('{}_awg'.format(channel)))

                # For each trigger_AWG, ind the elements to play the trigger
                # pulse in
                trigger_elements = {}
                for trigger_AWG in trigger_AWGs:
                    if self.elements_on_AWG.get(trigger_AWG, None) == None:
                        trigger_elements[trigger_AWG] = 'trigger_element'
                    else:
                        trigger_elements[
                            trigger_AWG] = self.find_trigger_element(
                                trigger_AWG, trigger_pulse_time)

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

        self.test_overlap()

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

    def test_overlap(self):
        """
        Tests for all AWGs if any of their elements overlap.
        """

        self.find_element_start_end()
        self.gen_elements_on_AWG()

        for AWG in self.elements_on_AWG:
            for i in range(len(self.elements_on_AWG[AWG]) - 1):
                next_el = self.elements_on_AWG[AWG][i + 1]
                prev_el = self.elements_on_AWG[AWG][i]
                if self.element_start_end[next_el][0] < \
                        self.element_start_end[prev_el][1]:
                    raise ValueError('{} and {} on {} overlap!'.format(
                        self.elements_on_AWG[AWG][i],
                        self.elements_on_AWG[AWG][i + 1], AWG))

    def resolve_timing(self):
        """
        For each pulse in the unresolved_pulses list, this method:
            * updates the _t0 of the pulse by using the timing description of
              the UnresolvedPulse
            * saves the resolved pulse in the elements dictionary
        """

        visited_pulses = set()
        ref_points = []

        pulses = self.gen_refpoint_dict()

        # add pulses that refer to segment start
        for pulse in pulses['segment_start']:
            visited_pulses.add(pulse)

            if pulse.pulse_obj.name in pulses:
                ref_points.append((pulse.pulse_obj.name, pulse))

            pulse.pulse_obj.algorithm_time(
                pulse.delay - pulse.ref_point_new * pulse.pulse_obj.length)

            # create an entry in the elements dictionary
            if pulse.pulse_obj.element_name not in self.elements:
                self.elements[pulse.pulse_obj.element_name] = [pulse.pulse_obj]
            elif pulse.pulse_obj.element_name in self.elements:
                self.elements[pulse.pulse_obj.element_name].append(
                    pulse.pulse_obj)

        if len(visited_pulses) == 0:
            raise ValueError('No pulse references to the segment start!')

        while len(ref_points) > 0:
            new_ref_points = []
            for (name, pulse) in ref_points:
                for p in pulses[name]:

                    if p in visited_pulses:
                        raise ValueError('Pulse chain shows cyclicity!')

                    visited_pulses.add(p)

                    # add p.name to reference list if it is used as a key
                    # in pulses
                    if p.pulse_obj.name in pulses:
                        new_ref_points.append((p.pulse_obj.name, p))

                    p.pulse_obj.algorithm_time(
                        pulse.pulse_obj.algorithm_time() + p.delay -
                        p.ref_point_new * p.pulse_obj.length +
                        p.ref_point * pulse.pulse_obj.length)

                    # create an entry in the elements dictionary
                    if p.pulse_obj.element_name not in self.elements:
                        self.elements[p.pulse_obj.element_name] = [p.pulse_obj]
                    elif p.pulse_obj.element_name in self.elements:
                        self.elements[p.pulse_obj.element_name].append(
                            p.pulse_obj)

            ref_points = new_ref_points

        if len(visited_pulses) != len(self.unresolved_pulses):
            raise ValueError('Not all pulses have been added!')

        self.find_element_start_end()

    def time_sort(self):
        """
        Given a segment, this method sorts the entries of the elements 
        dictionary (which are list of UnresolvedPulses) by accending _t0.
        """

        if self.elements == {}:
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

    def find_element_start_end(self):
        """
        Given a segment, this method:
            * finds the start and end times for each element 
            * saves them in element_start_end, ordered by accending start time
            * changes the order of self.elements dictionary by order of start 
              times 
        """

        # sorts the pulses in the elements in accending order
        self.time_sort()
        unordered_start_end = []
        new_elements = {}

        for element in self.elements:
            unordered_start_end.append(
                (self.elements[element][0].algorithm_time(),
                 self.elements[element][-1].algorithm_time() +
                 self.elements[element][-1].length, element))

        for (t_start, t_end, element) in sorted(unordered_start_end):
            self.element_start_end[element] = \
                [t_start, t_end]
            new_elements[element] = self.elements[element]

        self.elements = new_elements

    def reduce_to_segment_start(self):

        segment_t0 = float('inf')
        for pulse in self.unresolved_pulses:
            segment_t0 = min(segment_t0, pulse.pulse_obj.algorithm_time())

        for pulse in self.unresolved_pulses:
            pulse.delay -= segment_t0


class UnresolvedPulse:
    """
    pulse_pars: dictionary containing pulse parameters
    reference_point: 'segment_start', 'previous_pulse', pulse.name
    ref_point: 'start', 'end' -- reference point of the reference pulse
    ref_point_new: 'start', 'end' -- reference point of the new pulse
    """

    def __init__(self, pulse_pars):
        #self.element = pulse_pars['element_name']
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
