import numpy as np
from copy import deepcopy
import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulse as bpl  # base pulse lib
from pycqed.measurement.waveform_control import pulsar as ps


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
        self.resolved_pulses = None
        self.previous_pulse = None
        self.elements = {}
        self.element_start_end = {}
        self.elements_on_AWG = {}

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

        # create an entry in the elements dictionary
        if new_pulse.pulse_obj.element_name not in self.elements:
            self.elements[new_pulse.pulse_obj.element_name] = [
                new_pulse.pulse_obj
            ]
        elif new_pulse.pulse_obj.element_name in self.elements:
            self.elements[new_pulse.pulse_obj.element_name].append(
                new_pulse.pulse_obj)

        self.previous_pulse = new_pulse
        # if resolved_pulses is None, the resolve_timing function has to be
        # called prior to generating the waveforms
        self.resolved_pulses = None

    def gen_refpoint_dict(self):
        """
        Returns a dictionary of unresolved pulses with reference points as 
        keys to easily refer to pulses
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
        Generates a dictionary with element names as keys and a set of used
        AWGs for each element as value.
        """

        AWG_dict = {}

        for element in self.elements:
            AWG_dict[element] = set()

            for pulse in self.elements[element]:
                AWG_dict[element].add(self.pulsar.get(pulse.channel + '_AWG'))

        return AWG_dict

    def gen_trigger_el(self):
        """
        Adds the trigger pulses for each element to the resolved_pulses list 
        and updates the elements_on_AWG dictionary.
        """

        self.find_element_start_end()

        trigger_pulses = []
        AWG_dict = self.gen_AWG_dict()
        trigger_pars = {'length': 20e-9, 'amplitude': 1}

        for element in AWG_dict:
            for AWG in AWG_dict[element]:
                # add element to elements_on_AWG
                if AWG in self.elements_on_AWG:
                    self.elements_on_AWG[AWG].append(element)
                elif AWG not in self.elements_on_AWG:
                    self.elements_on_AWG[AWG] = [element]

                trig_pulse = bpl.SquarePulse(
                    'trigger_element', channel=AWG + "_Master",
                    **trigger_pars)  # think about Master channel name!

                trig_pulse.algorithm_time(self.element_start_end[element] -
                                          self.pulsar.get(AWG + '_delay'))

                trigger_pulses.append(trig_pulse)

        self.resolved_pulses += trigger_pulses

    def test_overlap(self):
        """
        Tests for all AWGs if any of their elements overlap.
        """

        for AWG in self.elements_on_AWG:
            for i in range(len(self.elements_on_AWG[AWG]) - 1):
                next_el = self.elements_on_AWG[AWG][i + 1]
                prev_el = self.elements_on_AWG[AWG][i]
                if self.element_start_end[next_el][0] < \
                        self.element_start_end[prev_el][1]:
                    raise ValueError('Elements on {} overlap!'.format(AWG))

    def resolve_timing(self):
        """
        Resolves the timing for all UnresolvedPulses by updating pulse_obj._t0
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

            ref_points = new_ref_points

        if len(visited_pulses) != len(self.unresolved_pulses):
            raise ValueError('Not all pulses have been added!')

        self.resolved_pulses = \
            [pulse.pulse_obj for pulse in self.unresolved_pulses]

    def time_sort(self):
        """
        Given a segment, this method sorts the entries of the elements 
        dictionary (which are list of UnresolvedPulses) by accending _t0.
        """

        for element in self.elements:
            old_list = [(pulse.algorithm_time(),pulse) \
                for pulse in self.elements[element]]

            new_list = sorted(old_list)
            self.elements[element] = [pulse for (t0, pulse) in new_list]

    def find_element_start_end(self):
        """
        Given a segment, this method finds the start and end times for each 
        element and saves them in a dictionary, ordered by accending start time.
        """

        # sorts the pulses in the elements in accending order
        self.time_sort()
        unordered_start_times = []
        unordered_end_times = {}

        for element in self.elements:
            unordered_start_times.append(
                (self.elements[element][0].algorithm_time(), element))
            unordered_end_times[element] = self.elements[element][
                -1].algorithm_time() + self.elements[element][-1].length

        for (t0, element) in sorted(unordered_start_times):
            self.element_start_end[element] = \
                [t0, unordered_end_times[element]]

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
