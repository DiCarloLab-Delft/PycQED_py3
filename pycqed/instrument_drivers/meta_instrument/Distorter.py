import logging
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

class Distortions(Instrument):
    '''
    '''
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_parameter('dist_dict', unit='',
                           parameter_class=ManualParameter,
                           vals=vals.Anything()
                           set_cmd=self._do_set_dist_dict,
                           get_cmd=self._do_get_dist_dict)
    def _do_set_dist_dict(self,dist):
        self._dist_dict = dist
    def _do_get_dist_dict(self):
        return self._dist_dict

