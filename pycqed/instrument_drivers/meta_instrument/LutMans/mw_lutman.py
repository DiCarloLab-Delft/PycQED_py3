from .base_lutman import Base_LutMan
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import InstrumentRefParameter
from qcodes.utils import validators as vals


class Base_MW_LutMan(Base_LutMan):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

    def _add_waveform_parameters(self):
        self.add_parameter('I_channel',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, 8))

        self.add_parameter('Q_channel',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, 8))

        self.add_parameter('Q_amp180', unit='V', vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.1)
        self.add_parameter('Q_amp90_scale',
                           vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.5)
        self.add_parameter('Q_motzoi', vals=vals.Numbers(-2, 2),
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('Q_gauss_width',
                           vals=vals.Numbers(min_value=1e-9), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=4e-9)


    def set_default_lutmap(self):
        self.LutMap({})


    def generate_standard_waveforms(self):
        pass

class CBOX_MW_LutMan(Base_MW_LutMan):
    pass


class QWG_MW_LutMan(Base_MW_LutMan):
    pass


class AWG8_MW_LutMan(Base_MW_LutMan):
    pass
