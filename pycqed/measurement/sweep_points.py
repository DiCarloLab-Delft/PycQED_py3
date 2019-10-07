import logging
log = logging.getLogger(__name__)
from collections import OrderedDict


class SweepPoints(list):
    def __init__(self, param_name=None, values=None, unit='', param_label=None):
        super().__init__()
        if param_name is not None and values is not None:
            if param_label is None:
                param_label = param_name
            self.append({param_name: (values, unit, param_label)})

    def add_sweep_parameter(self, param_name, values, unit='',
                            param_label=None):
        if param_label is None:
            param_label = param_name
        if len(self) == 0:
            self.append({param_name: (values, unit, param_label)})
        else:
            self[-1].update({param_name: (values, unit, param_label)})

    def add_sweep_dimension(self):
        self.append(dict())

    def get_sweep_points_map(self, keys_list):
        sweep_points_map = OrderedDict()
        if len(keys_list) != len(self[0]):
            raise ValueError('The number of keys and number of sweep parameters'
                             'do not match.')

        for i, key in enumerate(keys_list):
            sweep_points_map[key] = [list(d)[i] for d in self]

        return sweep_points_map


