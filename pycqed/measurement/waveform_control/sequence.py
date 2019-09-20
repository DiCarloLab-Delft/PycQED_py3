# A Sequence contains segments which then contain the pulses. The Sequence
# provides the information for the AWGs, in which order to play the segments.
#
# author: Michael Kerschbaum
# created: 04/2019

import numpy as np
import pycqed.measurement.waveform_control.pulsar as ps
from collections import OrderedDict as odict


class Sequence:
    """
    A Sequence consists of several segments, which can be played back on the 
    AWGs sequentially.
    """

    def __init__(self, name):
        self.name = name
        self.pulsar = ps.Pulsar.get_instance()
        self.segments = odict()
        self.awg_sequence = {}

    def add(self, segment):
        if segment.name in self.segments:
            raise Exception('Name {} already exisits in the sequence!'.format(
                segment.name))
        self.segments[segment.name] = segment

    def extend(self, segments):
        """
        Extends the sequence given a list of segments
        Args:
            segments (list): segments to add to the sequence
        """
        for seg in segments:
            self.add(seg)


    def generate_waveforms_sequences(self, awgs=None):
        """
        Calculates and returns 
            * a dictionary of waveforms used in the sequence, indexed
                by their hash value
            * For each awg, a list of elements, each element consisting of
                a waveform-hash for each codeword and each channel
        """
        waveforms = {}
        sequences = {}
        for seg in self.segments.values():
            seg.resolve_segment()
            seg.gen_elements_on_awg()

        if awgs is None:
            awgs = set()
            for seg in self.segments.values():
                awgs |= set(seg.elements_on_awg)

        for awg in awgs:
            sequences[awg] = odict()
            for segname, seg in self.segments.items():
                sequences[awg][segname] = None
                for elname in seg.elements_on_awg.get(awg, []):
                    sequences[awg][elname] = {'metadata': {}}
                    for cw in seg.get_element_codewords(elname, awg=awg):
                        sequences[awg][elname][cw] = {}
                        for ch in seg.get_element_channels(elname, awg=awg):
                            h = seg.calculate_hash(elname, cw, ch)
                            chid = self.pulsar.get(f'{ch}_id')
                            sequences[awg][elname][cw][chid] = h
                            if h not in waveforms:
                                wf = seg.waveforms(awgs={awg}, 
                                    elements={elname}, channels={ch}, 
                                    codewords={cw})
                                waveforms[h] = wf.popitem()[1].popitem()[1]\
                                                 .popitem()[1].popitem()[1]
                    if elname in seg.acquisition_elements:
                        sequences[awg][elname]['metadata']['acq'] = True
                    else:
                        sequences[awg][elname]['metadata']['acq'] = False
        return waveforms, sequences
                
    def n_acq_elements(self, per_segment=False):
        """
        Gets the number of acquisition elements in the sequence.
        Args:
            per_segment (bool): Whether or not to return the number of
                acquisition elements per segment. Defaults to False.

        Returns:
            number of acquisition elements (list (if per_segment) or int)

        """
        n_readouts = [len(seg.acquisition_elements)
                      for seg in self.segments.values()]
        if not per_segment:
            n_readouts = np.sum(n_readouts)
        return n_readouts


    def __repr__(self):
        string_repr = f"####### {self.name} #######\n"
        for seg_name, seg in self.segments.items():
            string_repr += str(seg) + "\n"
        return string_repr

