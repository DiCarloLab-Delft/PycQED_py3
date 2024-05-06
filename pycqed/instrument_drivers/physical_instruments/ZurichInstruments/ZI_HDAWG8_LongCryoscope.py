# -------------------------------------------
# Module containing subclass of ZI_HDAWG8.
# Subclass overwrites _get_waveform_table and _codeword_table_preamble
# -------------------------------------------
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG8 import ZI_HDAWG8


class ZI_HDAWG8_LongCryoscope(ZI_HDAWG8):
    """
    Behaviour class, driver for ZurichInstruments HDAWG8 instrument.
    Intended for Long-Cryoscope measurement where a single flux-pulse waveform is uploaded.
    NOTE: Only intended to use for Flux-HDAWG implementations.
    """

    # region Class Properties
    @property
    def long_cryoscope_channel(self) -> int:
        return self._long_cryocopse_channel

    @long_cryoscope_channel.setter
    def long_cryoscope_channel(self, value: int) -> None:
        assert 0 <= value <= 7, f"Assumes channel value between 0 and 7, instead: {value}."
        self._long_cryocopse_channel = value

    @property
    def is_odd_channel(self) -> bool:
        """:return: Whether (focus) long-cryoscope channel is odd."""
        return self.long_cryoscope_channel % 2
    # endregion

    # region Class Constructor
    def __init__(self, name: str, device: str, interface: str = '1GbE', server: str = 'localhost', port=8004, num_codewords: int = 64, **kw):
        super().__init__(name=name, device=device, interface=interface, server=server, port=port, num_codewords=num_codewords, **kw)
        self._long_cryoscope_channel: int  # 0-indexed, 0-7
    # endregion

    # region Class Methods
    def _get_waveform_table(self, awg_nr: int) -> list:
        """
        Returns the waveform table.

        The waveform table determines the mapping of waveforms to DIO codewords.
        The index of the table corresponds to the DIO codeword.
        The entry is a tuple of waveform names.

        Example:
            ["wave_ch7_cw000", "wave_ch8_cw000",
            "wave_ch7_cw001", "wave_ch8_cw001",
            "wave_ch7_cw002", "wave_ch8_cw002"]

        The waveform table generated depends on the awg_nr and the codeword
        protocol.
        """
        assert self.cfg_codeword_protocol() == 'flux', f"Assumes this HDAWG is used as flux instrument, instead: {self.cfg_codeword_protocol()}."
        ch: int = awg_nr * 2
        wf_table = []
        cw_r: int = 1
        cw_l: int = 0

        is_odd_channel: bool = self.is_odd_channel
        if is_odd_channel:
            cw_r = 0
            cw_l = 1
        wf_table.append(
            (zibase.gen_waveform_name(ch, cw_l),
             zibase.gen_waveform_name(ch + 1, cw_r))
        )
        return wf_table

    def _codeword_table_preamble(self, awg_nr: int):
        """
        Defines a snippet of code to use in the beginning of an AWG program in order to define the waveforms.
        The generated code depends on the instrument type. For the HDAWG instruments, we use the seWaveDIO
        function.
        """
        program = ''

        wf_table = self._get_waveform_table(awg_nr=awg_nr)
        is_odd_channel: bool = self.is_odd_channel
        if is_odd_channel:
            dio_cws = [0, 1]
        else:
            dio_cws = [0, 8]

        # Assuming wf_table looks like this: [('wave_ch7_cw000', 'wave_ch8_cw000'), ('wave_ch7_cw000', 'wave_ch8_cw001')]
        for dio_cw, (wf_l, wf_r) in zip(dio_cws, wf_table):
            csvname_l = self.devname + '_' + wf_l
            csvname_r = self.devname + '_' + wf_r

            if self.cfg_sideband_mode() == 'static' or self.cfg_codeword_protocol() == 'flux':
                program += f'setWaveDIO({dio_cw}, \"{csvname_l}\", \"{csvname_r}\");\n'
            else:
                raise Exception(f"Unknown modulation type '{self.cfg_sideband_mode()}' and codeword protocol '{self.cfg_codeword_protocol()}'")
        return program

    @classmethod
    def from_other_instance(cls, instance: ZI_HDAWG8) -> 'ZI_HDAWG8_LongCryoscope':
        """:return: Class-method constructor based on (other) instrument instance."""
        name: str = instance.name
        device: str = instance.devname
        codeword_protocol: str = instance.cfg_codeword_protocol()
        dios_0_interface: int = instance.get('dios_0_interface')
        # Close current instance
        instance.stop()
        instance.close()
        # Connect new instance
        result_instance = ZI_HDAWG8_LongCryoscope(name=name, device=device)
        result_instance.cfg_codeword_protocol(codeword_protocol)
        result_instance.set('dios_0_interface', dios_0_interface)
        result_instance.clear_errors()
        return result_instance
    # endregion
