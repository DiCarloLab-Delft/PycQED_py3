import numpy as np
import matplotlib.pyplot as plt
import logging
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import InstrumentRefParameter
from qcodes.utils import validators as vals
from pycqed.analysis.fit_toolbox.functions import PSD
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel


class Base_LutMan(Instrument):
    """
    The base LutMan is an abstract base class for the individual LutMans.
    The idea of the Lookuptable Manager (LutMan) is to provide a convenient
    interface to manage the waveforms loaded on specific lookuptables in
    an AWG.

    The LutMan provides
        - A set of basic waveforms that are generated based on
            parameters specified in the LutMan
        - A LutMap, relating waveform names to specific lookuptable indices
        - Methods to upload and regenerate these waveforms.
        - Methods to render waves.

    The Base LutMan does not provide a set of FIXME: comment ends


    """

    def __init__(self, name, **kw):
        logging.info(__name__ + " : Initializing instrument")
        super().__init__(name, **kw)
        # FIXME: rename to instr_AWG to be consistent with other instr refs
        self.add_parameter(
            "AWG",
            parameter_class=InstrumentRefParameter,
            docstring=(
                "Name of the AWG instrument used, note that this can also be "
                "a UHFQC or a CBox as these also contain AWG's"
            ),
            vals=vals.Strings(),
        )
        self._add_cfg_parameters()
        self._add_waveform_parameters()
        self.add_parameter(
            "LutMap",
            docstring=(
                "Dictionary containing the mapping between waveform"
                " names and parameter names (codewords)."
            ),
            initial_value={},
            vals=vals.Dict(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "sampling_rate",
            unit="Hz",
            vals=vals.Numbers(1, 100e10),
            initial_value=1e9,
            parameter_class=ManualParameter,
        )

        # Used to determine bounds in plotting.
        # overwrite in child classes if used.
        self._voltage_min = None
        self._voltage_max = None

        # initialize the _wave_dict to an empty dictionary
        self._wave_dict = {}
        self.set_default_lutmap()

    def time_to_sample(self, time):
        """
        Takes a time in seconds and returns the corresponding sample
        """
        return int(time * self.sampling_rate())

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

    def _add_cfg_parameters(self):
        pass

    def generate_standard_waveforms(self):
        """
        Generates all the standard waveforms and populates self._wave_dict

        """
        raise NotImplementedError()

    def load_waveform_onto_AWG_lookuptable(
        self, waveform_name: str, regenerate_waveforms: bool = False
    ):
        """
        Loads a specific waveform to the AWG
        """
        raise NotImplementedError()

    def load_waveforms_onto_AWG_lookuptable(
        self, regenerate_waveforms: bool = True, stop_start: bool = True
    ):
        """
        Loads all waveforms specified in the LutMap to an AWG.

        Args:
            regenerate_waveforms (bool): if True calls
                generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.
        """
        AWG = self.AWG.get_instr()

        if stop_start:
            AWG.stop()
        if regenerate_waveforms:
            self.generate_standard_waveforms()

        for waveform_name, lookuptable in self.LutMap().items():
            self.load_waveform_onto_AWG_lookuptable(waveform_name)

        if stop_start:
            AWG.start()

    def render_wave(
        self, wave_id, show=True, time_units="lut_index", reload_pulses=True
    ):
        """
        Render a waveform.

        Args:
            wave_id: can be either the "name" of a waveform or
                the integer key in self._wave_dict.
        """
        if wave_id not in self.LutMap().keys():
            wave_id = get_wf_idx_from_name(wave_id, self.LutMap())

        if reload_pulses:
            self.generate_standard_waveforms()
        fig, ax = plt.subplots(1, 1)
        if time_units == "lut_index":
            x = np.arange(len(self._wave_dict[wave_id][0]))
            ax.set_xlabel("Lookuptable index (i)")
            if self._voltage_min is not None:
                ax.vlines(2048, self._voltage_min, self._voltage_max, linestyle="--")
        elif time_units == "s":
            x = np.arange(len(self._wave_dict[wave_id][0])) / self.sampling_rate.get()

            if self._voltage_min is not None:
                ax.vlines(
                    2048 / self.sampling_rate.get(),
                    self._voltage_min,
                    self._voltage_max,
                    linestyle="--",
                )

        if len(self._wave_dict[wave_id]) == 2:
            ax.plot(x, self._wave_dict[wave_id][0], marker=".", label="chI")
            ax.plot(x, self._wave_dict[wave_id][1], marker=".", label="chQ")
        elif len(self._wave_dict[wave_id]) == 4:
            ax.plot(x, self._wave_dict[wave_id][0], marker=".", label="chGI")
            ax.plot(x, self._wave_dict[wave_id][1], marker=".", label="chGQ")
            ax.plot(x, self._wave_dict[wave_id][2], marker=".", label="chDI")
            ax.plot(x, self._wave_dict[wave_id][3], marker=".", label="chDQ")
        else:
            raise ValueError("waveform shape not understood")
        ax.legend()
        if self._voltage_min is not None:
            ax.set_facecolor("gray")
            ax.axhspan(self._voltage_min, self._voltage_max, facecolor="w", linewidth=0)
            ax.set_ylim(self._voltage_min * 1.1, self._voltage_max * 1.1)

        ax.set_xlim(0, x[-1])
        if time_units == "s":
            set_xlabel(ax, "time", "s")
        set_ylabel(ax, "Amplitude", "V")
        if show:
            plt.show()
        return fig, ax

    def render_wave_PSD(
        self, wave_id, show=True, reload_pulses=True, f_bounds=None, y_bounds=None
    ):
        if wave_id not in self.LutMap().keys():
            wave_id = get_wf_idx_from_name(wave_id, self.LutMap())
        if reload_pulses:
            self.generate_standard_waveforms()
        fig, ax = plt.subplots(1, 1)
        f_axis, PSD_I = PSD(self._wave_dict[wave_id][0], 1 / self.sampling_rate())
        f_axis, PSD_Q = PSD(self._wave_dict[wave_id][1], 1 / self.sampling_rate())

        ax.plot(f_axis, PSD_I, marker=",", label="chI")
        ax.plot(f_axis, PSD_Q, marker=",", label="chQ")
        ax.legend()

        ax.set_yscale("log", nonposy="clip")
        if y_bounds is not None:
            ax.set_ylim(y_bounds[0], y_bounds[1])
        if f_bounds is not None:
            ax.set_xlim(f_bounds[0], f_bounds[1])
        set_xlabel(ax, "Frequency", "Hz")
        set_ylabel(ax, "Spectral density", "V^2/Hz")
        if show:
            plt.show()
        return fig, ax


def get_redundant_codewords(codeword: int, bit_width: int = 4, bit_shift: int = 0):
    """
    Takes in a desired codeword and generates the redundant codewords.

    Example A:
        Codeword = 5   -> '101'
        bit_width = 4  -> '0101'
        bit_shift = 0  -> xxxx0101
    The function should return all combinations for all
        xxxx0101

    Example B:
        Codeword = 5   -> '101'
        bit_width = 4  -> '0101'
        bit_shift = 4  -> 0101xxxx
    The function should return all combinations for all
        0101xxxx

    Args:
        codeword (int) : the desired codeword
        bit_width (int): the number of bits in the codeword, determines
            how many redundant combinations are generated.
        bit_shift (int): determines how many bits the codeword is shifted.

    returns:
        redundant_codewords (list): all redundant combinations of the codeword
            see example above.
    """
    codeword_shifted = codeword << bit_shift
    redundant_codewords = []
    for i in range(2 ** bit_width):
        if bit_shift == 0:  # assumes the higher bits are used
            redundant_codewords.append(codeword_shifted + (i << bit_width))
        else:  # assumes the lower bits are used
            redundant_codewords.append(codeword_shifted + i)
    return redundant_codewords


def get_wf_idx_from_name(name, lutmap):
    """Find first match to a name in a lutmap."""
    for idx_key, waveform in lutmap.items():
        if waveform["name"] == name:
            return idx_key
    else:
        return False
