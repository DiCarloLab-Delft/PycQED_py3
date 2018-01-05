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