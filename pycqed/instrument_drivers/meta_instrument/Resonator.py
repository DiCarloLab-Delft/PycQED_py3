class resonator():
    """
    Quick class to describe the resonators to save them in a list.

    Input arguments when initializing resonators:
        identifier (str or float): Way of identifying the resonator (often the
                                   last character(s) of qubit names)
        freq (float): predicted frequency of resonator
        type (str): 'qubit_resonator' - a qubit resonator
                    'test_resonator' - a test resonator
                    'missing' - possibly broken, did not show up in scan
                    'broken' - resonator is not found and assumed broken

    For optimal use, it is adviced to input a list of all resonators to the
    device object of your setup.
    """

    def __init__(self, identifier, freq: float, **kw):
        self.identifier = identifier
        self.freq = freq
        self.freq_low = None
        self.type = kw.pop('type', 'unknown')
        self.qubit = kw.pop('qubit', 'None')
        self.sweetspot = 0

    def print_readable_snapshot(self):
        print(self.number)
        print(self.freq)
        print(self.type)
