import time

from qcodes.instrument.base import Instrument
from qcodes import validators
from functools import partial
from spirack.spi_rack import SPI_rack
from spirack.S4g_module import S4g_module
from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor


class QuTech_SPI_S4g_FluxCurrent(Instrument):
    def __init__(self, name: str, address: str,
                 channel_map: dict):
        """
        Creates an instance of the SPI S4g FluxCurrent instrument.

        Arguments:
            name:
            address: used to connect to the SPI rack e.g., "COM10"
            channel map: {"parameter_name": (module_nr, dac_nr)}

        For an example of how to use this instrument see
            examples/SPI_rack_examples/SPI_rack_example.py

        """
        t0 = time.time()
        super().__init__(name)
        self.channel_map = channel_map
        self.spi_rack = SPI_rack(address, 9600, 1)
        self.spi_rack.unlock()

        # Determine the set of modules required from the channel map
        module_ids = set([ch_map[0] for ch_map in channel_map.values()])
        # instantiate the controllers for the individual modules
        self.current_sources = {}
        for mod_id in module_ids:
            self.current_sources[mod_id] = S4g_module(
                self.spi_rack, module=mod_id, reset_currents=True)

        for parname, (mod_id, dac) in self.channel_map.items():
            self.add_parameter(
                parname,
                get_cmd=partial(self._get_current, parname),
                set_cmd=partial(self._set_current, parname),
                unit="A",
                vals=validators.Numbers(min_value=-50e-3, max_value=50e-3),
                step=0.00001,
                inter_delay=1e-3)

        self.connect_message(begin_time=t0)

    def _get_current(self, parname):
        mod_id, dac = self.channel_map[parname]
        current, span = self.current_sources[mod_id].get_settings(dac)
        # just to make sure
        assert span == 2
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

    def set_dacs_zero(self):
        """
        Sets the current for all modules to zero, also includes dacs that
        are not controlled by this instrument (this is intentional).
        """
        # First set all "parameters" to zero.
        # this ensures that the safe slow rampdown is used and that the
        # correct values are known to the instrument.
        for ch in self.channel_map:
            self.set(ch, 0)

        # "brute-set" all sources in known modules to zero, this is because
        # this is also a safety method that should ensure we are in an all
        # zero state.
        for s in self.current_sources.values():
            for dac in range(4):
                s.set_current(dac, 0.0)

    def close(self):
        self.spi_rack.close()
        super().close()



