import numpy as np
from copy import deepcopy
import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulse as bpl  # base pulse lib
import pycqed.measurement.waveform_control.pulsar as ps
import pycqed.measurement.waveform_control.segment as sg
from collections import OrderedDict as odict


class Sequence:
    """
    A Sequence consists of several segments, which can be played back on the 
    AWGs sequentially.
    """

    def __init__(self, name, pulsar):
        self.pulsar = pulsar
        self.name = name
        self.segments = odict()
        self.awg_sequence = {}

    def add(self, segment):
        if segment.name in self.segments:
            raise Exception('Name {} already exisits in the sequence!'.format(
                segment.name))
        self.segments[segment.name] = segment

    def sequence_for_awg(self):
        """
        Returns for an AWG a sequence with the ordered lists containing
        element name, segment name and 'RO' flag for readout elements and 
        'codeword' flag for codeword elements
        """

        self.awg_sequence = {}
        for awg in self.pulsar.awgs:
            self.awg_sequence[awg] = []
            for segment in self.segments:
                seg = self.segments[segment]
                seg.gen_elements_on_awg()

                if awg not in seg.elements_on_awg:
                    continue

                for element in seg.elements_on_awg[awg]:
                    self.awg_sequence[awg].append([element, segment])
                    if element in seg.acquisition_elements:
                        self.awg_sequence[awg][-1].append('RO')
