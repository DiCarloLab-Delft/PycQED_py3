import time

from qcodes import Instrument, validators as vals
from functools import partial
from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor


class virtual_SPI_S4g_FluxCurrent(Instrument):

    '''
    Virtual fluxcurrent parameter for usage with mock CCL.
    Can be improved upon.

    TODO: make binary output
    '''

    def __init__(self, name: str, channel_map: dict, **kwargs):
        t0 = time.time()
        super().__init__(name, **kwargs)

        self.channel_map = channel_map
        self.spi_rack = None
        module_ids = set([ch_map[0] for ch_map in channel_map.values()])

        self.current_sources = {}
        for mod_id in module_ids:
            self.current_sources[mod_id] = S4g_mock_module(mod_id)

        for parname, (mod_id, dac) in self.channel_map.items():
            self.add_parameter(
                parname,
                get_cmd=partial(self._get_current, parname),
                set_cmd=partial(self._set_current, parname),
                unit="A",
                vals=vals.Numbers(min_value=-50e-3, max_value=50e-3),
                step=0.00001,
                inter_delay=1e-3)

    # self.connect_message(t0)  # Does not work (yet)

    def _get_current(self, parname):
        mod_id, dac = self.channel_map[parname]
        current = self.current_sources[mod_id].get_settings(dac)
        return current

    def _set_current(self, parname, value):
        mod_id, dac = self.channel_map[parname]
        self.current_sources[mod_id].set_current(dac, value)

    def print_overview(self):
        msg = '{0:16}{1:4}\t{2:4}\t   {3:.4} \n'.format(
            'Name', 'Module', 'Channel', 'I')
        for ch_name, ch_map in self.channel_map.items():
            I = self.get(ch_name)
            scale_fac, unit = SI_prefix_and_scale_factor(I, 'A')
            msg += '{0:16}{1:4}\t{2:4}\t{3:.4} {4:4}\n'.format(
                ch_name, ch_map[0], ch_map[1], scale_fac*I, unit)
        print(msg)

    def close(self):
        super().close()


class S4g_mock_module(object):
    def __init__(self, module, reset_currents=True):
        self.module = module

        self.currents = [0.0, 0.0, 0.0, 0.0]

        for i in range(4):
            self.currents[i] = self.get_settings(i)

        if reset_currents:
            for i in range(4):
                self.set_current(i, 0.0)

    def get_settings(self, dac):
        return self.currents[dac]

    def set_current(self, DAC, current):
        self.currents[DAC] = float(current)
