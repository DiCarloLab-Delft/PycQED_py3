from abc import ABC, abstractmethod


class CodewordManager(ABC):
    """
    Abstract base class used to specify mappings between codewords and generator+integrator
    slots on the SHFQA.
    """

    def __init__(self):
        self._active_codewords = self._default_active_codewords()

    @property
    def active_codewords(self) -> list:
        """
        Property specifying which codewords will actually be considered in an experimental sequence.
        """
        return self._active_codewords

    @active_codewords.setter
    def active_codewords(self, codewords) -> None:
        """
        Setter of the 'active_codewords' property.
        """
        self._active_codewords = list(codewords)

    def active_slots(self) -> dict:
        """
        Returns a dictionary specifying which generators slots of the device
        are currently active. Specifically, this method accumulates the results from
        'codeword_slots()' over all stored codewords.
        """
        result = {}
        for codeword in self.active_codewords:
            for ch, slots in self.codeword_slots(codeword).items():
                try:
                    for slot in slots:
                        result[ch].append(slot)
                except KeyError:
                    result[ch] = slots
        return result

    def codeword_slots(self, codeword: int) -> dict:
        """
        Returns a dictionary specifying which generator slots a specified codeword
        will trigger in an experiment.
        """
        self._check_codeword(codeword)
        return self._codeword_slots(codeword)

    @abstractmethod
    def _codeword_slots(self, codeword: int) -> dict:
        """
        Customization point of public 'codeword_slots' method. Returns a dictionary specifying which generator
        slots a specified codeword will trigger in an experiment.
        """
        pass

    @abstractmethod
    def _default_active_codewords(self) -> list:
        """
        Returns a list of default codewords for the specified mapping.
        """
        pass

    @abstractmethod
    def _check_num_results(self, num_results: int) -> None:
        """
        Validator checking whether the provided number of results is compatible with the specified codeword mapping.
        """
        pass

    @abstractmethod
    def _check_codeword(self, codeword: int) -> None:
        """
        Validator checking whether the provided codeword is compatible with the specified codeword mapping.
        """
        pass

    class Error(Exception):
        """
        Exception raised in a context of wrong handling of codewords.
        """

        pass
