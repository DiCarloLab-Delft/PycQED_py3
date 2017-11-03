from .base_lutman import Base_LutMan
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from pycqed.measurement.waveform_control_CC import waveform as wf


class Base_Flux_LutMan(Base_LutMan):
    pass


class AWG_Flux_LutMan(Base_Flux_LutMan):
    def __init__(self, name, **kw):
        self.add_parameter('operating_mode',
                           initial_value='LongSeq',
                           vals=vals.Enum('Codeword', 'LongSeq'),
                           parameter_class=ManualParameter)
        self.add_parameter('awg_channel',
                           initial_value='LongSeq',
                           vals=vals.Ints(0, 8),
                           parameter_class=ManualParameter)
        self.add_parameter('distort',
                           initial_value=True,
                           vals=vals.Bools(),
                           parameter_class=ManualParameter)
        super().__init__(name, **kw)

    def set_default_lutmap(self):
        """
        Sets the "LutMap" parameter to
        """
        raise NotImplementedError()

    def _add_waveform_parameters(self):
        """
        Adds the parameters required to generate the standard waveforms
        """
        raise NotImplementedError()

    def generate_standard_waveforms(self):
        """
        Generates all the standard waveforms and populates self._wave_dict

        """
        raise NotImplementedError()

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
        """
        Loads a specific waveform to the AWG
        """
        raise NotImplementedError()
