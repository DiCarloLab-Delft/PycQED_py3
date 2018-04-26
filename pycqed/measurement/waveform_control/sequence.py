from copy import copy

class Sequence:
    """
    Class that contains a sequence.
    A sequence is defined as a list of elements that are given
    certain properties, such as repetition, trigger wait, playback type.

    We keep this independent of element generation here.
    Elements are simply referred to by name, the task to ensure
    they are available lies with the Pulsar. N.B. this means that
    Sequence.elements does not contain instances of the Element class, only
    names and meta-data.

    Properties of an element:
        name:         The name that is used to refer to the element within the
                      sequence. Must be unique in the sequence.
        wfname:       The name of the Element instance that defines the
                      waveform. This name is used to look up the waveform
                      later. Can be the same for multiple members of the
                      sequence. The special value 'codeword' is used to
                      indicate an element that plays a waveform depending on
                      the value of the DIO input of the AWG. The mapping is
                      specified by the Sequence.codewords dictionary, which
                      maps the DIO states to Element names (wfname).
        repetitions:  Number of times the element will be repeated.
        trigger_wait: Boolean flag whether the playback of the element should
                      wait for a trigger.
        goto_target:  If not None, specifies the name of the element that will
                      be played next. If None, the next element in the elements
                      list will be played.
        flags:        A set of options (as strings) for this element. For
                      example {'readout'} for the
    """

    def __init__(self, name):
        self.name = name
        self.elements = []
        self.codewords = {}

    def _make_element_spec(self, name, wfname, repetitions, trigger_wait,
                           goto_target=None, flags=None):
        if flags is None:
            flags = set()
        elt = {
            'name': name,
            'wfname': wfname,
            'repetitions': repetitions,
            'trigger_wait': trigger_wait,
            'goto_target': goto_target,
            'flags': flags,
        }
        return elt

    def insert(self, name, wfname, pos=None, repetitions=1, trigger_wait=False,
               goto_target=None, flags=None):
        """Creates a new element and adds it to the sequence."""
        for elt in self.elements:
            if elt['name'] == name:
                raise KeyError("Dyplicate element {}. Element names in sequence"
                               " must be unique.".format(name))
        elt = self._make_element_spec(name, wfname, repetitions,
                                      trigger_wait, goto_target, flags)
        if pos is None:
            pos = len(self.elements)

        self.elements.insert(pos, elt)

    def append(self, name, wfname, **kwargs):
        if 'pos' in kwargs:
            raise KeyError("Invalid parameter 'pos'")
        self.insert(name, wfname, **kwargs)

    def element_count(self):
        return len(self.elements)

    def element_index(self, name, start_idx=1):
        names = [elt['name'] for elt in self.elements]
        return names.index(name) + start_idx

    def join(self, sequence_b):
        # todo Implement checking of identical codewords and element names
        sequence_new = copy(self)
        sequence_new.elements = self.elements + sequence_b.elements
        # join dictionaries
        sequence_new.codewords = {**self.codewords, **sequence_b.codewords}
        sequence_new.name = self.name + "+" + sequence_b.name
        return sequence_new

    def __add__(self, sequence_b):
        return self.join(sequence_b)

    def set_codeword(self, codeword, wfname):
        """
        Sets the waveform corresponding to a codeword.
        Args:
            codeword: An integer corresponding to the DIO value to set the
                      output waveform for.
            wfname:   The name of the Element to play for the codeword.
        """
        self.codewords[codeword] = wfname

    def append_element(self, element, **kw):
        """
        Differs from normal append that it takes an element as input and not
        the arguments to make an element
        """
        self.append(element.name, element.name, **kw)

    def insert_element(self, element, **kw):
        """
        Differs from normal append that it takes an element as input and not
        the arguments to make an element
        """
        self.insert(element.name, element.name, **kw)

    def precompiled_sequence(self):
        """
        Creates a new sequence where element repetitions have been
        unwrapped and consecutive elements that do not wait for a trigger have
        been merged into one. The 'wfname' field of the compiled elements is 
        a tuple, containing names of all the elements that have been compiled
        together.
        """

        precompiled_sequence = Sequence(self.name + '_precompiled')
        precompiled_sequence.codewords = self.codewords.copy()
        visited = set()
        i_elt = 0

        while True:
            # add first element of the precompiled element
            visited |= {i_elt}
            elt = self.elements[i_elt]
            if elt['trigger_wait']:
                for i in range(elt['repetitions']):
                    precompiled_sequence.elements.append(
                        {'name': [elt['name'] + '_' + str(i)],
                         'wfname': [elt['wfname']],
                         'repetitions': 1,
                         'trigger_wait': elt['trigger_wait'],
                         'goto_target': None,
                         'flags': elt['flags'].copy()}
                    )
            else:
                precompiled_sequence.elements.append(
                    {'name': [elt['name'] + 'x' + str(elt['repetitions'])],
                     'wfname': [elt['wfname']]*elt['repetitions'],
                     'repetitions': 1,
                     'trigger_wait': False,
                     'goto_target': None,
                     'flags': elt['flags'].copy()}
                )
                if 'readout' in elt['flags'] and elt['repetitions'] > 1:
                    precompiled_sequence.elements[-1]['flags'] |= \
                        {'readout_in_middle'}
            if elt['goto_target'] is None:
                i_elt += 1
            else:
                i_elt = elt['goto_target']
            if i_elt >= len(self.elements) or i_elt in visited:
                for elt in precompiled_sequence.elements:
                    elt['name'] = ','.join(elt['name'])
                    elt['wfname'] = tuple(elt['wfname'])
                return precompiled_sequence

            # add all following elements that should not wait for a trigger
            while not self.elements[i_elt]['trigger_wait']:
                visited |= {i_elt}
                elt = self.elements[i_elt]
                precompiled_sequence.elements[-1]['name'] += \
                    [elt['name'] + 'x' + str(elt['repetitions'])]
                precompiled_sequence.elements[-1]['wfname'] += \
                    [elt['wfname']] * elt['repetitions']
                if ('readout' in precompiled_sequence.elements[-1]['flags']) or\
                   ('readout' in elt['flags'] and elt['repetitions'] > 1):
                    precompiled_sequence.elements[-1]['flags'] |= \
                        {'readout_in_middle'}
                if elt['goto_target'] is None:
                    i_elt += 1
                else:
                    i_elt = elt['goto_target']
                if i_elt >= len(self.elements) or i_elt in visited:
                    for elt in precompiled_sequence.elements:
                        elt['name'] = ','.join(elt['name'])
                        elt['wfname'] = tuple(elt['wfname'])
                    return precompiled_sequence
