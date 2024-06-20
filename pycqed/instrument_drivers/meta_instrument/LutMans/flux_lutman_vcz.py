from .base_lutman import Base_LutMan, get_wf_idx_from_name
import numpy as np
from copy import copy
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from qcodes.utils import validators as vals
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.measurement.waveform_control_CC import waveforms_flux as wfl
from pycqed.measurement.waveform_control_CC import waveforms_vcz as wf_vcz

import PyQt5
from qcodes_loop.plots.pyqtgraph import QtPlot
import matplotlib.pyplot as plt
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
import time

import logging

log = logging.getLogger(__name__)

"""
The default schema of this LutMap allows for 4 different 2Q gates.

NW     NE
  \   /
    Q
  /   \
SW     SE

First codeword is assigned to idling.
Codewords 2-5 are assigned to the two-qubit gates in clockwise order
(NE - SE - SW - NW)
Then we assign single qubit fluxing operations (parking and square)
Last codeword is reserved for custom waveforms defined by the user.

Args:
    lutmap
Return:
    valid (bool)

The schema for a lutmap is a dictionary with integer keys.
Every item in the dictionary must have the following keys:
    "name" : str
    "type" : one of valid_types
        {'idle', 'cz', 'idle_z', 'square', 'custom'}
    "which": str, optional used for two qubit flux pulses and one of
        {"NE", "SE", "SW", "NW"}
"""

_def_lm = {
    0: {"name": "i", "type": "idle"},
    1: {"name": "cz_NE", "type": "idle_z", "which": "NE"},
    2: {"name": "cz_SE", "type": "cz", "which": "SE"},
    3: {"name": "cz_SW", "type": "cz", "which": "SW"},
    4: {"name": "cz_NW", "type": "idle_z", "which": "NW"},
    5: {"name": "park", "type": "square"},
    6: {"name": "square", "type": "square"},
    7: {"name": "cz_aux", "type": "cz", "which": "aux"},
}


class Base_Flux_LutMan(Base_LutMan):
    """
    The default scheme of this LutMap allows for 4 different 2Q gates.

    NW     NE
      \   /
        Q
      /   \
    SW     SE
    """

    def render_wave(
        self,
        wave_name,
        time_units="s",
        reload_pulses: bool = True,
        render_distorted_wave: bool = True,
        QtPlot_win=None,
    ):
        """
        Renders a waveform
        """
        if reload_pulses:
            self.generate_standard_waveforms()

        x = np.arange(len(self._wave_dict[wave_name]))
        y = self._wave_dict[wave_name]

        if time_units == "lut_index":
            xlab = ("Lookuptable index", "i")
        elif time_units == "s":
            x = x / self.sampling_rate()
            xlab = ("Time", "s")

        if QtPlot_win is None:
            QtPlot_win = QtPlot(window_title=wave_name, figsize=(600, 400))

        if render_distorted_wave:
            if wave_name in self._wave_dict_dist.keys():
                x2 = np.arange(len(self._wave_dict_dist[wave_name]))
                if time_units == "s":
                    x2 = x2 / self.sampling_rate()

                y2 = self._wave_dict_dist[wave_name]
                QtPlot_win.add(
                    x=x2,
                    y=y2,
                    name=wave_name + " distorted",
                    symbol="o",
                    symbolSize=5,
                    xlabel=xlab[0],
                    xunit=xlab[1],
                    ylabel="Amplitude",
                    yunit="dac val.",
                )
            else:
                log.warning("Wave not in distorted wave dict")
        # Plotting the normal one second ensures it is on top.
        QtPlot_win.add(
            x=x,
            y=y,
            name=wave_name,
            symbol="o",
            symbolSize=5,
            xlabel=xlab[0],
            xunit=xlab[1],
            ylabel="Amplitude",
            yunit="V",
        )

        return QtPlot_win


class HDAWG_Flux_LutMan(Base_Flux_LutMan):
    
    # region Class Properties
    @property
    def total_length(self) -> int:
        """:return: Total number of sample points dedicated to waveform."""
        return self.total_park_length + self.total_pad_length

    @property
    def total_park_length(self) -> int:
        """:return: Total number of sample points dedicated to parking."""
        return int(np.round(self.sampling_rate() * self.park_length()))

    @property
    def total_pad_length(self) -> int:
        """:return: Total number of sample points dedicated to waveform padding."""
        return int(np.round(self.sampling_rate() * self.park_pad_length() * 2))

    @property
    def first_pad_length(self) -> int:
        """:return: Number of sample points in first pad-'arm' of two-sided parking."""
        equal_split: int = int(np.round(self.sampling_rate() * self.park_pad_length()))
        minimum_padding: int = 1
        return max(min(equal_split + self.park_pad_symmetry_offset(), self.total_pad_length - minimum_padding), minimum_padding)

    @property
    def second_pad_length(self) -> int:
        """:return: Number of sample points in second pad-'arm' of two-sided parking."""
        return self.total_pad_length - self.first_pad_length
    # endregion
    
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._wave_dict_dist = dict()
        self.sampling_rate(2.4e9)
        self._add_qubit_parameters()
        self._add_cz_sim_parameters()

    def set_default_lutmap(self):
        """Set the default lutmap for standard microwave drive pulses."""
        self.LutMap(_def_lm.copy())

    def generate_standard_waveforms(self):
        """
        Generate all the standard waveforms and populates self._wave_dict
        """
        self._wave_dict = {}
        # N.B. the  naming convention ._gen_{waveform_name} must be preserved
        # as it is used in the load_waveform_onto_AWG_lookuptable method.
        self._wave_dict["i"] = self._gen_i()
        self._wave_dict["square"] = self._gen_square()
        self._wave_dict["park"] = self._gen_park()
        self._wave_dict["custom_wf"] = self._gen_custom_wf()

        for _, waveform in self.LutMap().items():
            wave_name = waveform["name"]
            if waveform["type"] == "cz" or waveform["type"] == "idle_z":
                which_gate = waveform["which"]
            if waveform["type"] == "cz":
                self._wave_dict[wave_name] = self._gen_cz(which_gate=which_gate)
            elif waveform["type"] == "idle_z":
                # The vcz pulse itself has all parameters necessary for the correction
                self._wave_dict[wave_name] = self._gen_cz(which_gate=which_gate)

    def generate_cz_waveforms(self):
        """
        Generate CZ waveforms and populates self._wave_dict
        """
        self._wave_dict = {}

        for _, waveform in self.LutMap().items():
            wave_name = waveform["name"]
            if waveform["type"] == "cz" or waveform["type"] == "idle_z":
                which_gate = waveform["which"]
            if waveform["type"] == "cz":
                self._wave_dict[wave_name] = self._gen_cz(which_gate=which_gate)
            elif waveform["type"] == "idle_z":
                # The vcz pulse itself has all parameters necessary for the correction
                self._wave_dict[wave_name] = self._gen_cz(which_gate=which_gate)

    def _gen_i(self):
        return np.zeros(int(self.idle_pulse_length() * self.sampling_rate()))

    def _gen_square(self):
        return wf.single_channel_block(
            amp=self.sq_amp(),
            length=self.sq_length(),
            sampling_rate=self.sampling_rate(),
            delay=self.sq_delay(),
            gauss_sigma=self.sq_gauss_sigma(),
        )

    def _gen_park(self):
        zeros = np.zeros(int(self.park_pad_length() * self.sampling_rate()))
        # Padding
        first_zeros: np.ndarray = np.zeros(shape=self.first_pad_length)
        second_zeros: np.ndarray = np.zeros(shape=self.second_pad_length)
        
        if self.park_double_sided():
            ones = np.ones(int(self.park_length() * self.sampling_rate() / 2))
            pulse_pos = self.park_amp() * ones
            return np.concatenate((
                first_zeros, 
                +1 * pulse_pos,
                -1 * pulse_pos,
                second_zeros,
            ))
        else:
            pulse_pos = self.park_amp() * np.ones(
                int(self.park_length() * self.sampling_rate())
            )
            return np.concatenate((
                first_zeros,
                pulse_pos,
                second_zeros,
            ))

    def _add_qubit_parameters(self):
        """
        Adds parameters responsible for keeping track of qubit frequencies,
        coupling strengths etc.
        """
        self.add_parameter(
            "q_polycoeffs_freq_01_det",
            docstring="Coefficients of the polynomial used to convert "
            "amplitude in V to detuning in Hz. \nN.B. it is important to "
            "include both the AWG range and channel amplitude in the params.\n"
            "N.B.2 Sign convention: positive detuning means frequency is "
            "higher than current  frequency, negative detuning means its "
            "smaller.\n"
            "In order to convert a set of cryoscope flux arc coefficients to "
            " units of Volts they can be rescaled using [c0*sc**2, c1*sc, c2]"
            " where sc is the desired scaling factor that includes the sq_amp "
            "used and the range of the AWG (5 in amp mode).",
            vals=vals.Arrays(),
            # initial value is chosen to not raise errors
            initial_value=np.array([-2e9, 0, 0]),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "q_polycoeffs_anharm",
            docstring="coefficients of the polynomial used to calculate "
            "the anharmonicity (Hz) as a function of amplitude in V. "
            "N.B. it is important to "
            "include both the AWG range and channel amplitude in the params.\n",
            vals=vals.Arrays(),
            # initial value sets a flux independent anharmonicity of 300MHz
            initial_value=np.array([0, 0, -300e6]),
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "q_freq_01",
            vals=vals.Numbers(),
            docstring="Current operating frequency of qubit",
            # initial value is chosen to not raise errors
            initial_value=6e9,
            unit="Hz",
            parameter_class=ManualParameter,
        )

        for this_cz in ["NE", "NW", "SW", "SE"]:
            self.add_parameter(
                "q_freq_10_%s" % this_cz,
                vals=vals.Numbers(),
                docstring="Current operating frequency of qubit"
                " with which a CZ gate can be performed.",
                # initial value is chosen to not raise errors
                initial_value=6e9,
                unit="Hz",
                parameter_class=ManualParameter,
            )
            self.add_parameter(
                "q_J2_%s" % this_cz,
                vals=vals.Numbers(1e3, 500e6),
                unit="Hz",
                docstring="effective coupling between the 11 and 02 states.",
                # initial value is chosen to not raise errors
                initial_value=15e6,
                parameter_class=ManualParameter,
            )

    def _add_waveform_parameters(self):
        # CODEWORD 1: Idling
        self.add_parameter(
            "idle_pulse_length",
            unit="s",
            label="Idling pulse length",
            initial_value=40e-9,
            vals=vals.Numbers(0, 100e-6),
            parameter_class=ManualParameter,
        )

        # CODEWORDS 1-4: CZ
        # [2020-06-23] This dictionary is added here to be extended if a new or
        # different flux waveform for cz is to be tested
        # The cz waveform generators receive the `fluxlutman` and `which_gate`
        # as arguments
        self._cz_wf_generators_dict = {
            "vcz_waveform": wf_vcz.vcz_waveform
        }

        for this_cz in ["NE", "NW", "SW", "SE", "aux"]:
            self.add_parameter(
                "cz_wf_generator_%s" % this_cz,
                initial_value="vcz_waveform",
                # initial_value=None,
                vals=vals.Strings(),
                parameter_class=ManualParameter,
            )

            wf_vcz.add_vcz_parameters(self, which_gate=this_cz)

        # CODEWORD 5: Parking
        self.add_parameter(
            "park_length",
            unit="s",
            label="Parking pulse duration (total)",
            initial_value=40e-9,
            vals=vals.Numbers(0, 100e-6),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "park_pad_length",
            unit="s",
            label="Parking pulse padding duration (single-sided)",
            initial_value=0,
            vals=vals.Numbers(0, 100e-6),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "park_pad_symmetry_offset",
            unit="samples",
            label="Parking pulse padding samling point offset.\
                Applies offset to sampling points in initial padding (additive) and final padding (subtractive).\
                The offset is bounded such that the padding is minimal 1# and maximal total_padding - 1#.",
            initial_value=0,
            vals=vals.Numbers(-100, 100),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "park_amp",
            initial_value=0,
            label="Parking pulse amp. pos.",
            docstring="Parking pulse amplitude if `park_double_sided` is `False`, "
            "or positive amplitude for Net-Zero",
            unit="dac value",
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "park_double_sided",
            initial_value=False,
            vals=vals.Bool(),
            parameter_class=ManualParameter,
        )

        # CODEWORD 6: SQUARE
        self.add_parameter(
            "sq_amp",
            initial_value=0.5,
            # units is part of the total range of AWG8
            label="Square pulse amplitude",
            unit="dac value",
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "sq_length",
            unit="s",
            label="Square pulse duration",
            initial_value=40e-9,
            vals=vals.Numbers(0, 100e-6),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "sq_delay",
            unit="s",
            label="Square pulse delay",
            initial_value=0e-9,
            vals=vals.Numbers(0, 100e-6),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "sq_gauss_sigma",
            unit="s",
            label="Sigma for gaussian filter",
            initial_value=0e-9,
            vals=vals.Numbers(0, 100e-6),
            parameter_class=ManualParameter,
        )

        # CODEWORD 7: CUSTOM

        self.add_parameter(
            "custom_wf",
            initial_value=np.array([]),
            label="Custom waveform",
            docstring=(
                "Specifies a custom waveform, note that "
                "`custom_wf_length` is used to cut of the waveform if"
                "it is set."
            ),
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
        )
        self.add_parameter(
            "custom_wf_length",
            unit="s",
            label="Custom waveform length",
            initial_value=np.inf,
            docstring=(
                "Used to determine at what sample the custom waveform "
                "is forced to zero. This is used to facilitate easy "
                "cryoscope measurements of custom waveforms."
            ),
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=0),
        )

    def _gen_cz(self, which_gate, regenerate_cz=True):

        gate_str = "cz_%s" % which_gate

        wf_generator_name = self.get("cz_wf_generator_{}".format(which_gate))
        wf_generator = self._cz_wf_generators_dict[wf_generator_name]

        if regenerate_cz:
            self._wave_dict[gate_str] = wf_generator(self, which_gate=which_gate)
        cz_pulse = self._wave_dict[gate_str]

        return cz_pulse

    def calc_amp_to_eps(
        self,
        amp: float,
        state_A: str = "01",
        state_B: str = "02",
        which_gate: str = "NE",
        ):
        """
        Calculates detuning between two levels as a function of pulse
        amplitude in Volt.

            ε(V) = f_B (V) - f_A (V)

        Args:
            amp (float) : amplitude in Volt
            state_A (str) : string of 2 numbers denoting the state. The numbers
                correspond to the number of excitations in each qubits.
                The LSQ (right) corresponds to the qubit being fluxed and
                under control of this flux lutman.
            state_B (str) :

        N.B. this method assumes that the polycoeffs are with respect to the
            amplitude in units of V, including rescaling due to the channel
            amplitude and range settings of the AWG8.
            See also `self.get_dac_val_to_amp_scalefactor`.

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """
        polycoeffs_A = self.get_polycoeffs_state(state=state_A, which_gate=which_gate)
        polycoeffs_B = self.get_polycoeffs_state(state=state_B, which_gate=which_gate)
        polycoeffs = polycoeffs_B - polycoeffs_A
        return np.polyval(polycoeffs, amp)

    def calc_eps_to_dac(
        self,
        eps,
        state_A: str = "01",
        state_B: str = "02",
        which_gate: str = "NE",
        positive_branch=True,
        ):
        """
        See `calc_eps_to_amp`
        """
        return (
            self.calc_eps_to_amp(eps, state_A, state_B, which_gate, positive_branch)
            * self.get_amp_to_dac_val_scalefactor()
        )

    def calc_eps_to_amp(
        self,
        eps,
        state_A: str = "01",
        state_B: str = "02",
        which_gate: str = "NE",
        positive_branch=True,
        ):
        """
        Calculates amplitude in Volt corresponding to an energy difference
        between two states in Hz.
            V(ε) = V(f_b - f_a)

        N.B. this method assumes that the polycoeffs are with respect to the
            amplitude in units of V, including rescaling due to the channel
            amplitude and range settings of the AWG8.
            See also `self.get_dac_val_to_amp_scalefactor`.

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """
        # recursive allows dealing with an array of freqs
        if isinstance(eps, (list, np.ndarray)):
            return np.array(
                [
                    self.calc_eps_to_amp(
                        eps=e,
                        state_A=state_A,
                        state_B=state_B,
                        which_gate=which_gate,
                        positive_branch=positive_branch,
                    )
                    for e in eps
                ]
            )

        polycoeffs_A = self.get_polycoeffs_state(state=state_A, which_gate=which_gate)
        if state_B is not None:
            polycoeffs_B = self.get_polycoeffs_state(
                state=state_B, which_gate=which_gate
            )
            polycoeffs = polycoeffs_B - polycoeffs_A
        else:
            polycoeffs = copy(polycoeffs_A)
            polycoeffs[-1] = 0

        p = np.poly1d(polycoeffs)
        sols = (p - eps).roots

        # sols returns 2 solutions (for a 2nd order polynomial)
        if positive_branch:
            sol = np.max(sols)
        else:
            sol = np.min(sols)

        # imaginary part is ignored, instead sticking to closest real value
        # float is because of a typecasting bug in np 1.12 (solved in 1.14)
        return float(np.real(sol))

    def calc_net_zero_length_ratio(self, which_gate: str = "NE"):
        """
        Determine the lenght ratio of the net-zero pulses based on the
        parameter "czd_length_ratio".

        If czd_length_ratio is set to auto, uses the interaction amplitudes
        to determine the scaling of lengths. Note that this is a coarse
        approximation.
        """
        czd_length_ratio = self.get("czd_length_ratio_%s" % which_gate)
        if czd_length_ratio != "auto":
            return czd_length_ratio
        else:
            amp_J2_pos = self.calc_eps_to_amp(
                0,
                state_A="11",
                state_B="02",
                which_gate=which_gate,
                positive_branch=True,
            )
            amp_J2_neg = self.calc_eps_to_amp(
                0,
                state_A="11",
                state_B="02",
                which_gate=which_gate,
                positive_branch=False,
            )

            # lr chosen to satisfy (amp_pos*lr + amp_neg*(1-lr) = 0 )
            lr = -amp_J2_neg / (amp_J2_pos - amp_J2_neg)
            return lr

    def get_polycoeffs_state(self, state: str, which_gate: str = "NE"):
        """
        Args:
            state (str) : string of 2 numbers denoting the state. The numbers
                correspond to the number of excitations in each qubits.
                The LSQ (right) corresponds to the qubit being fluxed and
                under control of this flux lutman.

        Get's the polynomial coefficients that are used to calculate the
        energy levels of specific states.
        Note that avoided crossings are not taken into account here.
        N.B. The value of which_gate (and its default) only affect the
        other qubits (here noted as MSQ)


        """
        # Depending on the interaction (North or South) this qubit fluxes or not.
        # depending or whether it fluxes, it is LSQ or MSQ
        # depending on that, we use q_polycoeffs_freq_01_det or q_polycoeffs_freq_NE_det

        polycoeffs = np.zeros(3)
        freq_10 = self.get("q_freq_10_%s" % which_gate)
        if state == "00":
            pass
        elif state == "01":
            polycoeffs += self.q_polycoeffs_freq_01_det()
            polycoeffs[2] += self.q_freq_01()
        elif state == "02":
            polycoeffs += 2 * self.q_polycoeffs_freq_01_det()
            polycoeffs += self.q_polycoeffs_anharm()
            polycoeffs[2] += 2 * self.q_freq_01()
        elif state == "10":
            polycoeffs[2] += freq_10
        elif state == "11":
            polycoeffs += self.q_polycoeffs_freq_01_det()
            polycoeffs[2] += self.q_freq_01() + freq_10
        else:
            raise ValueError("State {} not recognized".format(state))
        return polycoeffs

    def _get_awg_channel_amplitude(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel() - 1  # -1 is to account for starting at 1
        awg_nr = awg_ch // 2
        ch_pair = awg_ch % 2

        channel_amp = AWG.get("awgs_{}_outputs_{}_amplitude".format(awg_nr, ch_pair))
        return channel_amp

    def _set_awg_channel_amplitude(self, val):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel() - 1  # -1 is to account for starting at 1
        awg_nr = awg_ch // 2
        ch_pair = awg_ch % 2
        AWG.set("awgs_{}_outputs_{}_amplitude".format(awg_nr, ch_pair), val)

    def _get_awg_channel_range(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel() - 1  # -1 is to account for starting at 1
        # channel range of 5 corresponds to -2.5V to +2.5V
        for i in range(5):
            channel_range_pp = AWG.get("sigouts_{}_range".format(awg_ch))
            if channel_range_pp is not None:
                break
            time.sleep(0.5)
        return channel_range_pp

    def _get_wf_name_from_cw(self, codeword: int):
        for idx, waveform in self.LutMap().items():
            if int(idx) == codeword:
                return waveform["name"]
        raise ValueError("Codeword {} not specified" " in LutMap".format(codeword))

    def _get_cw_from_wf_name(self, wf_name: str):
        for idx, waveform in self.LutMap().items():
            if wf_name == waveform["name"]:
                return int(idx)
        raise ValueError("Waveform {} not specified" " in LutMap".format(wf_name))

    def _gen_custom_wf(self):
        base_wf = copy(self.custom_wf())

        if self.custom_wf_length() != np.inf:
            # cuts of the waveform at a certain length by setting
            # all subsequent samples to 0.
            max_sample = int(self.custom_wf_length() * self.sampling_rate())
            base_wf[max_sample:] = 0
        return base_wf

    def calc_freq_to_amp(
        self,
        freq: float,
        state: str = "01",
        which_gate: str = "NE",
        positive_branch=True,
        ):
        """
        Calculates amplitude in Volt corresponding to the energy of a state
        in Hz.

        N.B. this method assumes that the polycoeffs are with respect to the
            amplitude in units of V, including rescaling due to the channel
            amplitude and range settings of the AWG8.
            See also `self.get_dac_val_to_amp_scalefactor`.

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """

        return self.calc_eps_to_amp(
            eps=freq,
            state_B=state,
            state_A="00",
            positive_branch=positive_branch,
            which_gate=which_gate,
        )

    def _add_cfg_parameters(self):

        self.add_parameter(
            "cfg_awg_channel",
            initial_value=1,
            vals=vals.Ints(1, 8),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "cfg_distort",
            initial_value=True,
            vals=vals.Bool(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "cfg_append_compensation",
            docstring=(
                "If True compensation pulses will be added to individual "
                " waveforms creating very long waveforms for each codeword"
            ),
            initial_value=True,
            vals=vals.Bool(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "cfg_compensation_delay",
            initial_value=3e-6,
            unit="s",
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "cfg_pre_pulse_delay",
            unit="s",
            label="Pre pulse delay",
            docstring="This parameter is used for fine timing corrections, the"
            " correction is applied in distort_waveform.",
            initial_value=0e-9,
            vals=vals.Numbers(0, 1e-6),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "instr_distortion_kernel", parameter_class=InstrumentRefParameter
        )
        self.add_parameter(
            "instr_partner_lutman",  # FIXME: unused?
            docstring="LutMan responsible for the corresponding"
            "channel in the AWG8 channel pair. "
            "Reference is used when uploading waveforms",
            parameter_class=InstrumentRefParameter,
        )
        self.add_parameter(
            "_awgs_fl_sequencer_program_expected_hash",  # FIXME: un used?
            docstring="crc32 hash of the awg8 sequencer program. "
            "This parameter is used to dynamically determine "
            "if the program needs to be uploaded. The initial_value is"
            " None, indicating that the program needs to be uploaded."
            " After the first program is uploaded, the value is set.",
            initial_value=None,
            vals=vals.Ints(),
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "cfg_max_wf_length",
            parameter_class=ManualParameter,
            initial_value=10e-6,
            unit="s",
            vals=vals.Numbers(0, 100e-6),
        )
        self.add_parameter(
            "cfg_awg_channel_range",
            docstring="peak peak value, channel range of 5 corresponds to -2.5V to +2.5V",
            get_cmd=self._get_awg_channel_range,
            unit="V_pp",
        )
        self.add_parameter(
            "cfg_awg_channel_amplitude",
            docstring="digital scale factor between 0 and 1",
            get_cmd=self._get_awg_channel_amplitude,
            set_cmd=self._set_awg_channel_amplitude,
            unit="a.u.",
            vals=vals.Numbers(0, 1),
        )

    def get_dac_val_to_amp_scalefactor(self):
        """
        Returns the scale factor to transform an amplitude in 'dac value' to an
        amplitude in 'V'.

        "dac_value" refers to the value between -1 and +1 that is set in a
        waveform.

        N.B. the implementation is specific to this type of AWG
        """
        if self.AWG() is None:
            log.warning("No AWG present, returning unity scale factor.")
            return 1
        channel_amp = self.cfg_awg_channel_amplitude()
        channel_range_pp = self.cfg_awg_channel_range()
        # channel range of 5 corresponds to -2.5V to +2.5V
        scalefactor = channel_amp * (channel_range_pp / 2)
        return scalefactor

    def get_amp_to_dac_val_scalefactor(self):
        if self.get_dac_val_to_amp_scalefactor() == 0:
            # Give a warning and don't raise an error as things should not
            # break because of this.
            log.warning(
                'AWG amp to dac scale factor is 0, check "{}" '
                "output amplitudes".format(self.AWG())
            )
            return 1
        return 1 / self.get_dac_val_to_amp_scalefactor()

    def calc_amp_to_freq(self, amp: float, state: str = "01", which_gate: str = "NE"):
        """
        Converts pulse amplitude in Volt to energy in Hz for a particular state
        Args:
            amp (float) : amplitude in Volt
            state (str) : string of 2 numbers denoting the state. The numbers
                correspond to the number of excitations in each qubits.
                The LSQ (right) corresponds to the qubit being fluxed and
                under control of this flux lutman.

        N.B. this method assumes that the polycoeffs are with respect to the
            amplitude in units of V, including rescaling due to the channel
            amplitude and range settings of the AWG8.
            See also `self.get_dac_val_to_amp_scalefactor`.
        N.B. The value of which_gate (and its default) only affect the
            other qubit frequencies (here noted as MSQ 10)

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """
        polycoeffs = self.get_polycoeffs_state(state=state, which_gate=which_gate)

        return np.polyval(polycoeffs, amp)

    def calc_parking_freq(self):
        _gain = self.cfg_awg_channel_amplitude()
        _rang = self.cfg_awg_channel_range()
        _dac  = self.park_amp()
        _out_amp = _gain*_dac*_rang/2

        _coefs = self.q_polycoeffs_freq_01_det()
        return np.polyval(_coefs, _out_amp)

    def calc_gate_freq(self, direction:str):
        _gain = self.cfg_awg_channel_amplitude()
        _rang = self.cfg_awg_channel_range()
        _dac  = self.get(f'vcz_amp_dac_at_11_02_{direction}')
        _fac  = self.get(f'vcz_amp_sq_{direction}')
        _out_amp = _gain*_dac*_fac*_rang/2

        _coefs = self.q_polycoeffs_freq_01_det()
        return np.polyval(_coefs, _out_amp)

    #################################
    #  Waveform loading methods     #
    #################################
    def load_waveform_onto_AWG_lookuptable(
        self, wave_id: str, regenerate_waveforms: bool = False
        ):
        """
        Loads a specific waveform to the AWG
        """

        # Here we are ductyping to determine if the waveform name or the
        # codeword was specified.
        if type(wave_id) == str:
            waveform_name = wave_id
            codeword = get_wf_idx_from_name(wave_id, self.LutMap())
        else:
            waveform_name = self.LutMap()[wave_id]["name"]
            codeword = wave_id

        if regenerate_waveforms:
            # only regenerate the one waveform that is desired
            if "cz" in waveform_name:
                # CZ gates contain information on which pair (NE, SE, SW, NW)
                # the gate is performed with this is specified in which_gate.
                gen_wf_func = getattr(self, "_gen_cz")
                self._wave_dict[waveform_name] = gen_wf_func(
                    which_gate=waveform_name[3:]
                )
            else:
                gen_wf_func = getattr(self, "_gen_{}".format(waveform_name))
                self._wave_dict[waveform_name] = gen_wf_func()

        waveform = self._wave_dict[waveform_name]
        codeword_str = "wave_ch{}_cw{:03}".format(self.cfg_awg_channel(), codeword)

        if self.cfg_append_compensation():
            waveform = self.add_compensation_pulses(waveform)

        if self.cfg_distort():
            # This is where the fixed length waveform is
            # set to cfg_max_wf_length
            waveform = self.distort_waveform(waveform)
            self._wave_dict_dist[waveform_name] = waveform
        else:
            # This is where the fixed length waveform is
            # set to cfg_max_wf_length
            waveform = self._append_zero_samples(waveform)
            self._wave_dict_dist[waveform_name] = waveform

        self.AWG.get_instr().set(codeword_str, waveform)

    def load_waveforms_onto_AWG_lookuptable(
        self, regenerate_waveforms: bool = True, stop_start: bool = True
        ):
        """
        Loads all waveforms specified in the LutMap to an AWG for both this
        LutMap and the partner LutMap.

        Args:
            regenerate_waveforms (bool): if True calls
                generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.

        """

        AWG = self.AWG.get_instr()

        if stop_start:
            AWG.stop()

        for idx, waveform in self.LutMap().items():
            self.load_waveform_onto_AWG_lookuptable(
                wave_id=idx, regenerate_waveforms=regenerate_waveforms
            )

        self.cfg_awg_channel_amplitude()
        self.cfg_awg_channel_range()

        if stop_start:
            AWG.start()

    def _append_zero_samples(self, waveform):
        """
        Helper method to ensure waveforms have the desired length
        """
        length_samples = roundup1024(
            int(self.sampling_rate() * self.cfg_max_wf_length()),
            self.cfg_max_wf_length()
            )
        extra_samples = length_samples - len(waveform)
        if extra_samples >= 0:
            y_sig = np.concatenate([waveform, np.zeros(extra_samples)])
        else:
            y_sig = waveform[:extra_samples]
        return y_sig

    def add_compensation_pulses(self, waveform):
        """
        Adds the inverse of the pulses at the end of a waveform to
        ensure flux discharging.
        """
        wf = np.array(waveform)  # catches a rare bug when wf is a list
        delay_samples = np.zeros(
            int(self.sampling_rate() * self.cfg_compensation_delay())
        )
        comp_wf = np.concatenate([wf, delay_samples, -1 * wf])
        return comp_wf

    def distort_waveform(self, waveform, inverse=False):
        """
        Modifies the ideal waveform to correct for distortions and correct
        fine delays.
        Distortions are corrected using the kernel object.
        """
        k = self.instr_distortion_kernel.get_instr()

        # Prepend zeros to delay waveform to correct for fine timing
        delay_samples = int(self.cfg_pre_pulse_delay() * self.sampling_rate())
        waveform = np.pad(waveform, (delay_samples, 0), "constant")

        # duck typing the distort waveform method
        if hasattr(k, "distort_waveform"):
            distorted_waveform = k.distort_waveform(
                waveform,
                length_samples=int(
                    roundup1024(self.cfg_max_wf_length() * self.sampling_rate(),
                    self.cfg_max_wf_length())
                ),
                inverse=inverse,
            )
        else:  # old kernel object does not have this method
            if inverse:
                raise NotImplementedError()
            distorted_waveform = k.convolve_kernel(
                [k.kernel(), waveform],
                length_samples=int(self.cfg_max_wf_length() * self.sampling_rate()),
            )
        return distorted_waveform

    #################################
    #  Plotting methods            #
    #################################
    def plot_cz_trajectory(self, axs=None, show=True, which_gate="NE"):
        """
        Plots the cz trajectory in frequency space.
        """
        q_J2 = self.get("q_J2_%s" % which_gate)

        if axs is None:
            f, axs = plt.subplots(figsize=(5, 7), nrows=3, sharex=True)

        dac_amps = self._wave_dict["cz_%s" % which_gate]
        t = np.arange(0, len(dac_amps)) * 1 / self.sampling_rate()

        CZ_amp = dac_amps * self.get_dac_val_to_amp_scalefactor()
        CZ_eps = self.calc_amp_to_eps(CZ_amp, "11", "02", which_gate=which_gate)
        CZ_theta = wfl.eps_to_theta(CZ_eps, q_J2)

        axs[0].plot(t, np.rad2deg(CZ_theta), marker=".")
        axs[0].fill_between(t, np.rad2deg(CZ_theta), color="C0", alpha=0.5)
        set_ylabel(axs[0], r"$\theta$", "deg")

        axs[1].plot(t, CZ_eps, marker=".")
        axs[1].fill_between(t, CZ_eps, color="C0", alpha=0.5)
        set_ylabel(axs[1], r"$\epsilon_{11-02}$", "Hz")

        axs[2].plot(t, CZ_amp, marker=".")
        axs[2].fill_between(t, CZ_amp, color="C0", alpha=0.1)
        set_xlabel(axs[2], "Time", "s")
        set_ylabel(axs[2], r"Amp.", "V")
        # axs[2].set_ylim(-1, 1)
        axs[2].axhline(0, lw=0.2, color="grey")
        CZ_amp_pred = self.distort_waveform(CZ_amp)[: len(CZ_amp)]
        axs[2].plot(t, CZ_amp_pred, marker=".")
        axs[2].fill_between(t, CZ_amp_pred, color="C1", alpha=0.3)
        if show:
            plt.show()
        return axs

    def plot_level_diagram(self, ax=None, show=True, which_gate="NE"):
        """
        Plots the level diagram as specified by the q_ parameters.
            1. Plotting levels
            2. Annotating feature of interest
            3. Adding legend etc.
            4. Add a twin x-axis to denote scale in dac amplitude

        """

        if ax is None:
            f, ax = plt.subplots()
        # 1. Plotting levels
        # maximum voltage of AWG in amp mode
        amps = np.linspace(-2.5, 2.5, 101)
        freqs = self.calc_amp_to_freq(amps, state="01", which_gate=which_gate)
        ax.plot(amps, freqs, label="$f_{01}$")
        ax.text(
            0,
            self.calc_amp_to_freq(0, state="01", which_gate=which_gate),
            "01",
            color="C0",
            ha="left",
            va="bottom",
            clip_on=True,
        )

        freqs = self.calc_amp_to_freq(amps, state="02", which_gate=which_gate)
        ax.plot(amps, freqs, label="$f_{02}$")
        ax.text(
            0,
            self.calc_amp_to_freq(0, state="02", which_gate=which_gate),
            "02",
            color="C1",
            ha="left",
            va="bottom",
            clip_on=True,
        )

        freqs = self.calc_amp_to_freq(amps, state="10", which_gate=which_gate)
        ax.plot(amps, freqs, label="$f_{10}$")
        ax.text(
            0,
            self.calc_amp_to_freq(0, state="10", which_gate=which_gate),
            "10",
            color="C2",
            ha="left",
            va="bottom",
            clip_on=True,
        )

        freqs = self.calc_amp_to_freq(amps, state="11", which_gate=which_gate)
        ax.plot(amps, freqs, label="$f_{11}$")
        ax.text(
            0,
            self.calc_amp_to_freq(0, state="11", which_gate=which_gate),
            "11",
            color="C3",
            ha="left",
            va="bottom",
            clip_on=True,
        )

        # 2. Annotating feature of interest
        ax.axvline(0, 0, 1e10, linestyle="dotted", c="grey")

        amp_J2 = self.calc_eps_to_amp(
            0, state_A="11", state_B="02", which_gate=which_gate
        )
        amp_J1 = self.calc_eps_to_amp(
            0, state_A="10", state_B="01", which_gate=which_gate
        )

        ax.axvline(amp_J2, ls="--", lw=1, c="C4")
        ax.axvline(amp_J1, ls="--", lw=1, c="C6")

        f_11_02 = self.calc_amp_to_freq(amp_J2, state="11", which_gate=which_gate)
        ax.plot([amp_J2], [f_11_02], color="C4", marker="o", label="11-02")
        ax.text(
            amp_J2,
            f_11_02,
            "({:.4f},{:.2f})".format(amp_J2, f_11_02 * 1e-9),
            color="C4",
            ha="left",
            va="bottom",
            clip_on=True,
        )

        f_10_01 = self.calc_amp_to_freq(amp_J1, state="01", which_gate=which_gate)

        ax.plot([amp_J1], [f_10_01], color="C5", marker="o", label="10-01")
        ax.text(
            amp_J1,
            f_10_01,
            "({:.4f},{:.2f})".format(amp_J1, f_10_01 * 1e-9),
            color="C5",
            ha="left",
            va="bottom",
            clip_on=True,
        )

        # 3. Adding legend etc.
        title = "Calibration visualization\n{}\nchannel {}".format(
            self.AWG(), self.cfg_awg_channel()
        )
        leg = ax.legend(title=title, loc=(1.05, 0.3))
        leg._legend_box.align = "center"
        set_xlabel(ax, "AWG amplitude", "V")
        set_ylabel(ax, "Frequency", "Hz")
        ax.set_xlim(-2.5, 2.5)

        ax.set_ylim(
            0, self.calc_amp_to_freq(0, state="02", which_gate=which_gate) * 1.1
        )

        # 4. Add a twin x-axis to denote scale in dac amplitude
        dac_val_axis = ax.twiny()
        dac_ax_lims = np.array(ax.get_xlim()) * self.get_amp_to_dac_val_scalefactor()
        dac_val_axis.set_xlim(dac_ax_lims)
        set_xlabel(dac_val_axis, "AWG amplitude", "dac")

        dac_val_axis.axvspan(1, 1000, facecolor=".5", alpha=0.5)
        dac_val_axis.axvspan(-1000, -1, facecolor=".5", alpha=0.5)
        # get figure is here in case an axis object was passed as input
        f = ax.get_figure()
        f.subplots_adjust(right=0.7)
        if show:
            plt.show()
        return ax

    def plot_cz_waveforms(
        self, qubits: list, which_gate_list: list, ax=None, show: bool = True
        ):
        """
        Plots the cz waveforms from several flux lutamns, mainly for
        verification, time alignment and debugging
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        flux_lm_list = [
            self.find_instrument("flux_lm_{}".format(qubit)) for qubit in qubits
        ]

        for flux_lm, which_gate, qubit in zip(flux_lm_list, which_gate_list, qubits):
            flux_lm.generate_standard_waveforms()
            waveform_name = "cz_{}".format(which_gate)
            ax.plot(
                flux_lm._wave_dict[waveform_name],
                ".-",
                label=waveform_name + " " + qubit,
            )
        ax.legend()
        fig = ax.get_figure()

        if show:
            fig.show()

        return fig

    #################################
    #  Simulation methods           #
    #################################

    def _add_cz_sim_parameters(self):
        for this_cz in ["NE", "NW", "SW", "SE"]:
            self.add_parameter(
                "bus_freq_%s" % this_cz,
                docstring="[CZ simulation] Bus frequency.",
                vals=vals.Numbers(0.1e9, 1000e9),
                initial_value=7.77e9,
                parameter_class=ManualParameter,
            )
            self.add_parameter(
                "instr_sim_control_CZ_%s" % this_cz,
                docstring="Noise and other parameters for CZ simulation.",
                parameter_class=InstrumentRefParameter,
            )

        self.add_parameter(
            "step_response",
            initial_value=np.array([]),
            label="Step response",
            docstring=(
                "Stores the normalized flux line step response. "
                "Intended for use in cz simulations with noise."
            ),
            parameter_class=ManualParameter,
            vals=vals.Arrays(),
        )


class QWG_Flux_LutMan(HDAWG_Flux_LutMan):
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._wave_dict_dist = dict()
        self.sampling_rate(1e9)

    def get_dac_val_to_amp_scalefactor(self):
        """
        Returns the scale factor to transform an amplitude in 'dac value' to an
        amplitude in 'V'.
         N.B. the implementation is specific to this type of AWG (QWG)
        """
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel()

        channel_amp = AWG.get("ch{}_amp".format(awg_ch))
        scale_factor = channel_amp
        return scale_factor

    def load_waveforms_onto_AWG_lookuptable(
        self, regenerate_waveforms: bool = True, stop_start: bool = True
    ):
        # We inherit from the HDAWG LutMan but do not require the fancy
        # loading because the QWG is a simple device!
        return Base_Flux_LutMan.load_waveforms_onto_AWG_lookuptable(
            self, regenerate_waveforms=regenerate_waveforms, stop_start=stop_start
        )

    def _get_awg_channel_amplitude(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel()

        channel_amp = AWG.get("ch{}_amp".format(awg_ch))
        return channel_amp

    def _set_awg_channel_amplitude(self, val):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel()

        channel_amp = AWG.set("ch{}_amp".format(awg_ch), val)
        return channel_amp

    def _add_cfg_parameters(self):

        self.add_parameter(
            "cfg_awg_channel",
            initial_value=1,
            vals=vals.Ints(1, 4),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "cfg_distort",
            initial_value=True,
            vals=vals.Bool(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "cfg_append_compensation",
            docstring=(
                "If True compensation pulses will be added to individual "
                " waveforms creating very long waveforms for each codeword"
            ),
            initial_value=True,
            vals=vals.Bool(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "cfg_compensation_delay",
            parameter_class=ManualParameter,
            initial_value=3e-6,
            unit="s",
            vals=vals.Numbers(),
        )

        self.add_parameter(
            "cfg_pre_pulse_delay",
            unit="s",
            label="Pre pulse delay",
            docstring="This parameter is used for fine timing corrections, the"
            " correction is applied in distort_waveform.",
            initial_value=0e-9,
            vals=vals.Numbers(0, 1e-6),
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "instr_distortion_kernel", parameter_class=InstrumentRefParameter
        )

        self.add_parameter(
            "cfg_max_wf_length",
            parameter_class=ManualParameter,
            initial_value=10e-6,
            unit="s",
            vals=vals.Numbers(0, 100e-6),
        )

        self.add_parameter(
            "cfg_awg_channel_amplitude",
            docstring="Output amplitude from 0 to 1.6 V",
            get_cmd=self._get_awg_channel_amplitude,
            set_cmd=self._set_awg_channel_amplitude,
            unit="V",
            vals=vals.Numbers(0, 1.6),
        )


class LRU_Flux_LutMan(Base_Flux_LutMan):
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._wave_dict_dist = dict()
        self.sampling_rate(2.4e9)

    def _add_waveform_parameters(self):
        # CODEWORD 1: Idling
        self.add_parameter(
            "idle_pulse_length",
            unit="s",
            label="Idling pulse length",
            initial_value=40e-9,
            vals=vals.Numbers(0, 100e-6),
            parameter_class=ManualParameter,
        )
        # Parameters for leakage reduction unit pulse.
        self.add_parameter('mw_lru_modulation', unit='Hz',
                           docstring=('Modulation frequency for LRU pulse.'),
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter, initial_value=0.0e6)
        self.add_parameter('mw_lru_amplitude', unit='frac',
                           docstring=('amplitude for LRU pulse.'),
                           vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter, initial_value=.8)
        self.add_parameter('mw_lru_duration',  unit='s',
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=300e-9)
        self.add_parameter('mw_lru_rise_duration',  unit='s',
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=30e-9)

    def _add_cfg_parameters(self):
        self.add_parameter(
            "cfg_awg_channel",
            initial_value=1,
            vals=vals.Ints(1, 8),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "_awgs_fl_sequencer_program_expected_hash",  # FIXME: un used?
            docstring="crc32 hash of the awg8 sequencer program. "
            "This parameter is used to dynamically determine "
            "if the program needs to be uploaded. The initial_value is"
            " None, indicating that the program needs to be uploaded."
            " After the first program is uploaded, the value is set.",
            initial_value=None,
            vals=vals.Ints(),
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "cfg_max_wf_length",
            parameter_class=ManualParameter,
            initial_value=10e-6,
            unit="s",
            vals=vals.Numbers(0, 100e-6),
        )
        self.add_parameter(
            "cfg_awg_channel_range",
            docstring="peak peak value, channel range of 5 corresponds to -2.5V to +2.5V",
            get_cmd=self._get_awg_channel_range,
            unit="V_pp",
        )
        self.add_parameter(
            "cfg_awg_channel_amplitude",
            docstring="digital scale factor between 0 and 1",
            get_cmd=self._get_awg_channel_amplitude,
            set_cmd=self._set_awg_channel_amplitude,
            unit="a.u.",
            vals=vals.Numbers(0, 1),
        )

    def set_default_lutmap(self):
        """Set the default lutmap for LRU drive pulses."""
        lm = {
            0: {"name": "i", "type": "idle"},
            1: {"name": "lru", "type": "lru"},
        }
        self.LutMap(lm)

    def generate_standard_waveforms(self):

        """
        Generate all the standard waveforms and populates self._wave_dict
        """

        self._wave_dict = {}
        # N.B. the  naming convention ._gen_{waveform_name} must be preserved
        # as it is used in the load_waveform_onto_AWG_lookuptable method.
        self._wave_dict["i"] = self._gen_i()
        self._wave_dict["lru"] = self._gen_lru()

    def _gen_i(self):
        return np.zeros(int(self.idle_pulse_length() * self.sampling_rate()))

    def _gen_lru(self):
        self.lru_func = wf.mod_lru_pulse
        _wf = self.lru_func(
            t_total = self.mw_lru_duration(),
            t_rise = self.mw_lru_rise_duration(),
            f_modulation = self.mw_lru_modulation(),
            amplitude = self.mw_lru_amplitude(),
            sampling_rate = self.sampling_rate())[0]
        return _wf

    def _get_awg_channel_amplitude(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel() - 1  # -1 is to account for starting at 1
        awg_nr = awg_ch // 2
        ch_pair = awg_ch % 2

        channel_amp = AWG.get("awgs_{}_outputs_{}_amplitude".format(awg_nr, ch_pair))
        return channel_amp

    def _set_awg_channel_amplitude(self, val):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel() - 1  # -1 is to account for starting at 1
        awg_nr = awg_ch // 2
        ch_pair = awg_ch % 2
        AWG.set("awgs_{}_outputs_{}_amplitude".format(awg_nr, ch_pair), val)

    def _get_awg_channel_range(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel() - 1  # -1 is to account for starting at 1
        # channel range of 5 corresponds to -2.5V to +2.5V
        for i in range(5):
            channel_range_pp = AWG.get("sigouts_{}_range".format(awg_ch))
            if channel_range_pp is not None:
                break
            time.sleep(0.5)
        return channel_range_pp

    def _append_zero_samples(self, waveform):
        """
        Helper method to ensure waveforms have the desired length
        """
        length_samples = roundup1024(
            int(self.sampling_rate() * self.cfg_max_wf_length()),
            self.cfg_max_wf_length()
        )
        extra_samples = length_samples - len(waveform)
        if extra_samples >= 0:
            y_sig = np.concatenate([waveform, np.zeros(extra_samples)])
        else:
            y_sig = waveform[:extra_samples]
        return y_sig

    def load_waveform_onto_AWG_lookuptable(
        self, wave_id: str, regenerate_waveforms: bool = False):
        """
        Loads a specific waveform to the AWG
        """
        # Here we are ductyping to determine if the waveform name or the
        # codeword was specified.
        if type(wave_id) == str:
            waveform_name = wave_id
            codeword = get_wf_idx_from_name(wave_id, self.LutMap())
        else:
            waveform_name = self.LutMap()[wave_id]["name"]
            codeword = wave_id

        if regenerate_waveforms:
            gen_wf_func = getattr(self, "_gen_{}".format(waveform_name))
            self._wave_dict[waveform_name] = gen_wf_func()

        waveform = self._wave_dict[waveform_name]
        codeword_str = "wave_ch{}_cw{:03}".format(self.cfg_awg_channel(), codeword)

        # This is where the fixed length waveform is
        # set to cfg_max_wf_length
        waveform = self._append_zero_samples(waveform)
        self._wave_dict_dist[waveform_name] = waveform

        self.AWG.get_instr().set(codeword_str, waveform)

    def load_waveforms_onto_AWG_lookuptable(
        self, regenerate_waveforms: bool = True, stop_start: bool = True):
        """
        Loads all waveforms specified in the LutMap to an AWG for both this
        LutMap and the partner LutMap.

        Args:
            regenerate_waveforms (bool): if True calls
                generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.

        """

        AWG = self.AWG.get_instr()

        if stop_start:
            AWG.stop()

        for idx, waveform in self.LutMap().items():
            self.load_waveform_onto_AWG_lookuptable(
                wave_id=idx, regenerate_waveforms=regenerate_waveforms
            )

        self.cfg_awg_channel_amplitude()
        self.cfg_awg_channel_range()

        if stop_start:
            AWG.start()


#########################################################################
# Convenience functions below
#########################################################################


def roundup1024(n, waveform_length):
    n_samples = int(waveform_length*2.4e9)
    return int(np.ceil(n / n_samples) * n_samples)
