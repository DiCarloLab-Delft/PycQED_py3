from .base_lutman import Base_LutMan
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from pycqed.measurement.waveform_control_CC import waveform as wf


class Base_Flux_LutMan(Base_LutMan):
    pass