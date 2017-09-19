from .base_lutman import Base_LutMan

class Base_RO_LutMan(Base_LutMan):
    def __init__(self, name, num_res: int=1, **kw):
        super().__init__(name, **kw)
        self._num_res = num_res



    def _add_waveform_parameters(self):


        for q in range(self._num_res):
            self.add_parameter('M_length_R{}'.format(q))
            self.add_parameter('M_amp1_R{}'.format(q))
            self.add_parameter('M_amp2_R{}'.format(q))


    def generate_standard_waveforms(self):
        """
        """
        # Only generate the combinations required/specified in the LutMap


class CBOX_MW_LutMan(Base_MW_LutMan):
    pass