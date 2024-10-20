import logging
import weakref
import copy
import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import Dict, List

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import InstrumentRefParameter
from qcodes.utils import validators as vals

from pycqed.analysis.fit_toolbox.functions import PSD
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel


log = logging.getLogger(__name__)


class Base_LutMan(Instrument):
    """
    The base LutMan is a (partly) abstract base class for the individual LutMans.
    The idea of the Lookuptable Manager (LutMan) is to provide a convenient
    interface to manage the waveforms loaded on specific lookuptables in
    an AWG.

    The LutMan provides
        - LutMap
            a dict that relates specific lookuptable indices (or 'codewords') to a 'waveform', which is a dict
            containing a name ("name") and properties ("type", and depending on that, other properties). Note that the
            allowe range of inices/codewords depends on the DIO mode of the connected instrument.

        - _wave_dict
            a dict with generated LutMap waveforms based on parameters specified in the LutMan, and the LutMap
            properties. The key is identical to the LutMap key (codeword), except for Base_RO_LutMan, which uses a
            string

        - Methods to upload and regenerate these waveforms.
        - Methods to render waves.
    """

    _all_lutmans: Dict[str, weakref.ref] = {}

    def __init__(self, name, **kw):
        log.info(f"Initializing '{name}'")
        super().__init__(name, **kw)

        # FIXME: rename to instr_AWG to be consistent with other instr refs
        self.add_parameter(
            "AWG",
            parameter_class=InstrumentRefParameter,
            docstring=(
                "Name of the AWG instrument used, note that this can also be "
                "a UHFQC as it also contain AWGs"
            ),
            vals=vals.Strings(),
        )

        # FIXME: allowing direct access requires that user maintains consistency
        #  between LutMap and _wave_dict (and instrument), see all the handling
        #  (e.g. *lutman.load_*) in CCL_Transmon/device_object_CCL/sweep_functions,
        #  and issue #626
        #  Also, some verification when setting would be nice
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

        # FIXME: get sampling_rate from AWG
        self.add_parameter(
            "sampling_rate",
            unit="Hz",
            vals=vals.Numbers(1, 100e10),
            initial_value=1e9,
            parameter_class=ManualParameter,
        )

        self._add_cfg_parameters()
        self._add_waveform_parameters()

        # Used to determine bounds in plotting.
        # overwrite in child classes if used.
        self._voltage_min = None
        self._voltage_max = None

        # initialize the _wave_dict to an empty dictionary
        self._wave_dict = {}
        self.set_default_lutmap()

        # support for make()
        self._record_instance(self)
        self._current_parameters = {}
        self._current_wave_dict = {}

    def __del__(self) -> None:
        self._remove_instance(self)
        super().__del__()

    ##########################################################################
    # Abstract functions
    ##########################################################################

    def _add_waveform_parameters(self):
        """
        Adds the parameters required to generate the standard waveforms
        """
        raise NotImplementedError()

    def _add_cfg_parameters(self):
        pass

    def set_default_lutmap(self):
        """
        Sets the "LutMap" parameter to its default

        """
        raise NotImplementedError()

    def generate_standard_waveforms(self):
        """
        Generates all the standard waveforms and populates self._wave_dict

        """
        raise NotImplementedError()

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str, regenerate_waveforms: bool = False):
        """
        Loads a specific waveform to the AWG
        """
        raise NotImplementedError()

    ##########################################################################
    # Overridden functions
    ##########################################################################

    def load_waveforms_onto_AWG_lookuptable(
            self,
            regenerate_waveforms: bool = True,
            stop_start: bool = True
    ):
        """
        Loads all waveforms specified in the LutMap to an AWG.

        Args:
            regenerate_waveforms (bool): if True calls generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.
            FIXME: inefficient if multiple LutMans (qubits) per instrument are updated
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
            self,
            wave_id,
            show=True,
            time_units="lut_index",
            reload_pulses=True
    ):
        """
        Render a waveform to a figure.

        Args:
            wave_id: can be either the "name" of a waveform or the integer key in self._wave_dict.
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

    ##########################################################################
    # Functions
    ##########################################################################

    def render_wave_PSD(
            self,
            wave_id,
            show=True,
            reload_pulses=True,
            f_bounds=None,
            y_bounds=None
    ):
        """
        Create a Power Spectral Density plot
        """
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

    def time_to_sample(self, time):
        """
        Takes a time in seconds and returns the corresponding sample
        """
        return int(time * self.sampling_rate())

    ##########################################################################
    # Functions: smart update
    ##########################################################################

    # FIXME: also handle codeword/waveform relation (HDAWG:upload_codeword_program/QWG:'wave_ch{}_cw{:03}') smartly?
    #  on HDAWG, its an expensive operation
    def _wave_dict_dirty(self):
        # exit on empty _wave_dict (initially the case)
        if not self._wave_dict:
            return True

        # exit on empty _current_wave_dict
        if not self._current_wave_dict:
            return True

        for key,val in self._wave_dict.items():
            # exit on new key (not present in _current_wave_dict)
            if self._current_wave_dict[key] is None:
                return True

            # exit on differences in the waveforms (note that the values are tuples of arrays)
            for x, y in zip(val, self._current_wave_dict[key]):
                if np.any(x != y):
                    return True

        return False

    def _parameters_dirty(self, verbose: bool=False):
        """ Check whether parameters have changed. NB: LutMap is also a parameter """

        # exit on empty dict
        if not self._current_parameters:
            return True

        # exit on differences in parameter values
        for key in self.parameters:
            if self.parameters[key].get() != self._current_parameters[key]:
                if verbose:
                    log.info(f"{self.name}: parameter {key} changed: new='{self.parameters[key].get()}'', old='{self._current_parameters[key]}'")
                return True

        return False

    def _is_dirty(self) -> bool:
        return self._wave_dict_dirty() or self._parameters_dirty()

    def _make(self) -> None:
        if self._parameters_dirty(verbose=True):  # only here we're verbose, or we'll get double log entries
            self.generate_standard_waveforms()  # NB: updates self._wave_dict

            # make a copy of (only the values of) the parameters
            self._current_parameters = {}
            for key in self.parameters:
                self._current_parameters[key] = self.parameters[key].get()

        if self._wave_dict_dirty():
            self.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=False)
            self._current_wave_dict = copy.deepcopy(self._wave_dict)

    ##########################################################################
    # Class methods
    ##########################################################################

    @classmethod
    def _record_instance(cls, instance: 'Base_LutMan') -> None:
        wr = weakref.ref(instance)
        name = instance.name
        cls._all_lutmans[name] = wr  # NB: we don't check uniqueness of name since Instrument already does

    @classmethod
    def _remove_instance(cls, instance: 'Base_LutMan') -> None:
        wr = weakref.ref(instance)

        # remove from _all_lutmans, but don't depend on the name to do it, in case name has changed or been deleted
        all_lm = cls._all_lutmans
        for name, ref in list(all_lm.items()):
            if ref is wr:
                del all_lm[name]

    @classmethod
    def make(cls) -> int:
        t1 = time()

        # collect work per AWG
        work: Dict[str, List] = {}
        num_dirty = 0
        for name, ref in list(cls._all_lutmans.items()):
            if ref()._is_dirty():
                num_dirty += 1
                awg = ref().AWG.get_instr()  # NB: can be none if not set
                if not work.get(awg.name):
                    work[awg.name] = []
                work[awg.name].append(ref)

        # for each AWG requiring work, update dirty LutMans
        for awg_name, refs in work.items():
            # stop AWG
            awg = Instrument.find_instrument(awg_name)
            awg.stop()

            # update dirty lutmans
            for ref in refs:
               ref()._make()

            # start AWG
            awg.start()

        t2 = time()
        log.info(f"updated {num_dirty} dirty LutMans in {t2-t1:.3g} s")
        return num_dirty


# FIXME: this is specific for very early versions of the QWG that only supported a single codeword triggering all 4
#  channels. Should be removed.
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
