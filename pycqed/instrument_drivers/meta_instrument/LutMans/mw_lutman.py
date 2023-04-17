from .base_lutman import Base_LutMan, get_redundant_codewords, get_wf_idx_from_name

import numpy as np
from collections.abc import Iterable
from collections import OrderedDict

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

from pycqed.measurement.waveform_control_CC import waveform as wf

def __get_nearest_rotation(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

default_mw_lutmap = {
    0   : {"name" : "i"     , "theta" :     0, "phi" : 0 , "type" : "ge"},
    1   : {"name" : "rx12"  , "theta" :   180, "phi" : 0 , "type" : "ef"},
    2   : {"name" : "rx45"  , "theta" :    45, "phi" : 0 , "type" : "ge"},
    3   : {"name" : "ry45"  , "theta" :    45, "phi" : 90, "type" : "ge"},
    4   : {"name" : "rx90"  , "theta" :    90, "phi" : 0 , "type" : "ge"},
    5   : {"name" : "ry90"  , "theta" :    90, "phi" : 90, "type" : "ge"},
    8   : {"name" : "rx180" , "theta" :   180, "phi" : 0 , "type" : "ge"},
    9   : {"name" : "ry180" , "theta" :   180, "phi" : 90, "type" : "ge"},
    12  : {"name" : "rxm90" , "theta" :   -90, "phi" : 0 , "type" : "ge"},
    13  : {"name" : "rym90" , "theta" :   -90, "phi" : 90, "type" : "ge"},
    14  : {"name" : "rxm45" , "theta" :   -45, "phi" : 0 , "type" : "ge"},
    15  : {"name" : "rym45" , "theta" :   -45, "phi" : 90, "type" : "ge"},
    16  : {"name" : "spec"  , "type"  : "spec"},
    17  : {"name" : "square", "type"  : "square"},
    18  : {"name" : "rPhi90", "theta" :    90, "phi" : 0 , "type" : "ge"},
    19  : {"name" : "rPhi180" , "theta" : 180, "phi" : 0 , "type" : "ge"},
    105 : {"name" : "phaseCorrPark1" , "type" : "phase"},
    106 : {"name" : "phaseCorrPark2" , "type" : "phase"},
    107 : {"name" : "phaseCorrPark3" , "type" : "phase"},
    108 : {"name" : "phaseCorrPark4" , "type" : "phase"},
    109 : {"name" : "phaseCorrPark5" , "type" : "phase"},
    110 : {"name" : "phaseCorrPark6" , "type" : "phase"},
    111 : {"name" : "phaseCorrPark7" , "type" : "phase"},
    112 : {"name" : "phaseCorrPark8" , "type" : "phase"},
    113 : {"name" : "phaseCorrNW" , "type" : "phase"},
    114 : {"name" : "phaseCorrNE" , "type" : "phase"},
    115 : {"name" : "phaseCorrSW" , "type" : "phase"},
    116 : {"name" : "phaseCorrSE" , "type" : "phase"},
}

inspire_mw_lutmap = {
    0  : {"name": "i"     , "theta":    0     , "phi" :  0 , "type" : "ge"},
    1  : {"name": "rx12"  , "theta":  180     , "phi" :  0 , "type" : "ef"},
    2  : {"name": "rx45"  , "theta":   45     , "phi" :  0 , "type" : "ge"},  
    3  : {"name": "ry45"  , "theta":   45     , "phi" : 90 , "type" : "ge"},
    4  : {"name": "rx90"  , "theta":   90     , "phi" :  0 , "type" : "ge"},
    5  : {"name": "ry90"  , "theta":   90     , "phi" : 90 , "type" : "ge"},
    6  : {"name": "rx135" , "theta":  135     , "phi" :  0 , "type" : "ge"},
    7  : {"name": "ry135" , "theta":  135     , "phi" : 90 , "type" : "ge"},
    8  : {"name": "rx180" , "theta":  180     , "phi" :  0 , "type" : "ge"},
    9  : {"name": "ry180" , "theta":  180     , "phi" : 90 , "type" : "ge"},
    10 : {"name": "rx225" , "theta": -135     , "phi" :  0 , "type" : "ge"},
    11 : {"name": "ry225" , "theta": -135     , "phi" : 90 , "type" : "ge"},
    12 : {"name": "rx270" , "theta":  -90     , "phi" :  0 , "type" : "ge"},
    13 : {"name": "ry270" , "theta":  -90     , "phi" : 90 , "type" : "ge"},
    14 : {"name": "rx315" , "theta":  -45     , "phi" :  0 , "type" : "ge"},
    15 : {"name": "ry315" , "theta":  -45     , "phi" : 90 , "type" : "ge"},
    16 : {"name": "rx6"   , "theta":    6.429 , "phi" :  0 , "type" : "ge"},
    17 : {"name": "rx13"  , "theta":   12.857 , "phi" :  0 , "type" : "ge"},
    18 : {"name": "rx19"  , "theta":   19.286 , "phi" :  0 , "type" : "ge"},
    19 : {"name": "rx26"  , "theta":   25.714 , "phi" :  0 , "type" : "ge"},
    20 : {"name": "rx32"  , "theta":   32.143 , "phi" :  0 , "type" : "ge"},
    21 : {"name": "rx39"  , "theta":   38.571 , "phi" :  0 , "type" : "ge"},
    22 : {"name": "rx51"  , "theta":   51.429 , "phi" :  0 , "type" : "ge"},
    23 : {"name": "rx58"  , "theta":   57.857 , "phi" :  0 , "type" : "ge"},
    24 : {"name": "rx64"  , "theta":   64.286 , "phi" :  0 , "type" : "ge"},
    25 : {"name": "rx71"  , "theta":   70.714 , "phi" :  0 , "type" : "ge"},
    26 : {"name": "rx77"  , "theta":   77.143 , "phi" :  0 , "type" : "ge"},
    27 : {"name": "rx84"  , "theta":   83.571 , "phi" :  0 , "type" : "ge"},
    28 : {"name": "rx96"  , "theta":   96.429 , "phi" :  0 , "type" : "ge"},
    29 : {"name": "rx103" , "theta":  102.857 , "phi" :  0 , "type" : "ge"},
    30 : {"name": "rx109" , "theta":  109.286 , "phi" :  0 , "type" : "ge"},
    31 : {"name": "rx116" , "theta":  115.714 , "phi" :  0 , "type" : "ge"},
    32 : {"name": "rx122" , "theta":  122.143 , "phi" :  0 , "type" : "ge"},
    33 : {"name": "rx129" , "theta":  128.571 , "phi" :  0 , "type" : "ge"},
    34 : {"name": "rx141" , "theta":  141.429 , "phi" :  0 , "type" : "ge"},
    35 : {"name": "rx148" , "theta":  147.857 , "phi" :  0 , "type" : "ge"},
    36 : {"name": "rx154" , "theta":  154.286 , "phi" :  0 , "type" : "ge"},
    37 : {"name": "rx161" , "theta":  160.714 , "phi" :  0 , "type" : "ge"},
    38 : {"name": "rx167" , "theta":  167.143 , "phi" :  0 , "type" : "ge"},
    39 : {"name": "rx174" , "theta":  173.571 , "phi" :  0 , "type" : "ge"},
    40 : {"name": "rx186" , "theta": -173.571 , "phi" :  0 , "type" : "ge"},
    41 : {"name": "rx193" , "theta": -167.143 , "phi" :  0 , "type" : "ge"},
    42 : {"name": "rx199" , "theta": -160.714 , "phi" :  0 , "type" : "ge"},
    43 : {"name": "rx206" , "theta": -154.286 , "phi" :  0 , "type" : "ge"},
    44 : {"name": "rx212" , "theta": -147.857 , "phi" :  0 , "type" : "ge"},
    45 : {"name": "rx219" , "theta": -141.429 , "phi" :  0 , "type" : "ge"},
    46 : {"name": "rx231" , "theta": -128.571 , "phi" :  0 , "type" : "ge"},
    47 : {"name": "rx238" , "theta": -122.143 , "phi" :  0 , "type" : "ge"},
    48 : {"name": "rx244" , "theta": -115.714 , "phi" :  0 , "type" : "ge"},
    49 : {"name": "rx251" , "theta": -109.286 , "phi" :  0 , "type" : "ge"},
    50 : {"name": "rx257" , "theta": -102.857 , "phi" :  0 , "type" : "ge"},
    51 : {"name": "rx264" , "theta":  -96.429 , "phi" :  0 , "type" : "ge"},
    52 : {"name": "rx276" , "theta":  -83.571 , "phi" :  0 , "type" : "ge"},
    53 : {"name": "rx283" , "theta":  -77.143 , "phi" :  0 , "type" : "ge"},
    54 : {"name": "rx289" , "theta":  -70.714 , "phi" :  0 , "type" : "ge"},
    55 : {"name": "rx296" , "theta":  -64.286 , "phi" :  0 , "type" : "ge"},
    56 : {"name": "rx302" , "theta":  -57.857 , "phi" :  0 , "type" : "ge"},
    57 : {"name": "rx309" , "theta":  -51.429 , "phi" :  0 , "type" : "ge"},
    58 : {"name": "rx321" , "theta":  -38.571 , "phi" :  0 , "type" : "ge"},
    59 : {"name": "rx328" , "theta":  -32.143 , "phi" :  0 , "type" : "ge"},
    60 : {"name": "rx334" , "theta":  -25.714 , "phi" :  0 , "type" : "ge"},
    61 : {"name": "rx341" , "theta":  -19.286 , "phi" :  0 , "type" : "ge"},
    62 : {"name": "rx347" , "theta":  -12.857 , "phi" :  0 , "type" : "ge"},
    63 : {"name": "rx354" , "theta":   -6.429 , "phi" :  0 , "type" : "ge"},
    64 : {"name": "ry6"   , "theta":    6.429 , "phi" : 90 , "type" : "ge"},
    65 : {"name": "ry13"  , "theta":   12.857 , "phi" : 90 , "type" : "ge"},
    66 : {"name": "ry19"  , "theta":   19.286 , "phi" : 90 , "type" : "ge"},
    67 : {"name": "ry26"  , "theta":   25.714 , "phi" : 90 , "type" : "ge"},
    68 : {"name": "ry32"  , "theta":   32.143 , "phi" : 90 , "type" : "ge"},
    69 : {"name": "ry39"  , "theta":   38.571 , "phi" : 90 , "type" : "ge"},
    70 : {"name": "ry51"  , "theta":   51.429 , "phi" : 90 , "type" : "ge"},
    71 : {"name": "ry58"  , "theta":   57.857 , "phi" : 90 , "type" : "ge"},
    72 : {"name": "ry64"  , "theta":   64.286 , "phi" : 90 , "type" : "ge"},
    73 : {"name": "ry71"  , "theta":   70.714 , "phi" : 90 , "type" : "ge"},
    74 : {"name": "ry77"  , "theta":   77.143 , "phi" : 90 , "type" : "ge"},
    75 : {"name": "ry84"  , "theta":   83.571 , "phi" : 90 , "type" : "ge"},
    76 : {"name": "ry96"  , "theta":   96.429 , "phi" : 90 , "type" : "ge"},
    77 : {"name": "ry103" , "theta":  102.857 , "phi" : 90 , "type" : "ge"},
    78 : {"name": "ry109" , "theta":  109.286 , "phi" : 90 , "type" : "ge"},
    79 : {"name": "ry116" , "theta":  115.714 , "phi" : 90 , "type" : "ge"},
    80 : {"name": "ry122" , "theta":  122.143 , "phi" : 90 , "type" : "ge"},
    81 : {"name": "ry129" , "theta":  128.571 , "phi" : 90 , "type" : "ge"},
    82 : {"name": "ry141" , "theta":  141.429 , "phi" : 90 , "type" : "ge"},
    83 : {"name": "ry148" , "theta":  147.857 , "phi" : 90 , "type" : "ge"},
    84 : {"name": "ry154" , "theta":  154.286 , "phi" : 90 , "type" : "ge"},
    85 : {"name": "ry161" , "theta":  160.714 , "phi" : 90 , "type" : "ge"},
    86 : {"name": "ry167" , "theta":  167.143 , "phi" : 90 , "type" : "ge"},
    87 : {"name": "ry174" , "theta":  173.571 , "phi" : 90 , "type" : "ge"},
    88 : {"name": "ry186" , "theta": -173.571 , "phi" : 90 , "type" : "ge"},
    89 : {"name": "ry193" , "theta": -167.143 , "phi" : 90 , "type" : "ge"},
    90 : {"name": "ry199" , "theta": -160.714 , "phi" : 90 , "type" : "ge"},
    91 : {"name": "ry206" , "theta": -154.286 , "phi" : 90 , "type" : "ge"},
    92 : {"name": "ry212" , "theta": -147.857 , "phi" : 90 , "type" : "ge"},
    93 : {"name": "ry219" , "theta": -141.429 , "phi" : 90 , "type" : "ge"},
    94 : {"name": "ry231" , "theta": -128.571 , "phi" : 90 , "type" : "ge"},
    95 : {"name": "ry238" , "theta": -122.143 , "phi" : 90 , "type" : "ge"},
    96 : {"name": "ry244" , "theta": -115.714 , "phi" : 90 , "type" : "ge"},
    97 : {"name": "ry251" , "theta": -109.286 , "phi" : 90 , "type" : "ge"},
    98 : {"name": "ry257" , "theta": -102.857 , "phi" : 90 , "type" : "ge"},
    99 : {"name": "ry264" , "theta":  -96.429 , "phi" : 90 , "type" : "ge"},
    100: {"name": "ry276" , "theta":  -83.571 , "phi" : 90 , "type" : "ge"},
    101: {"name": "ry283" , "theta":  -77.143 , "phi" : 90 , "type" : "ge"},
    102: {"name": "ry289" , "theta":  -70.714 , "phi" : 90 , "type" : "ge"},
    103: {"name": "ry296" , "theta":  -64.286 , "phi" : 90 , "type" : "ge"},
    104: {"name": "ry302" , "theta":  -57.857 , "phi" : 90 , "type" : "ge"},
    105: {"name": "ry309" , "theta":  -51.429 , "phi" : 90 , "type" : "ge"},
    106: {"name": "ry321" , "theta":  -38.571 , "phi" : 90 , "type" : "ge"},
    107: {"name": "ry328" , "theta":  -32.143 , "phi" : 90 , "type" : "ge"},
    108: {"name": "ry334" , "theta":  -25.714 , "phi" : 90 , "type" : "ge"},
    109: {"name": "ry341" , "theta":  -19.286 , "phi" : 90 , "type" : "ge"},
    110: {"name": "ry347" , "theta":  -12.857 , "phi" : 90 , "type" : "ge"},
    111: {"name": "ry354" , "theta":   -6.429 , "phi" : 90 , "type" : "ge"},
    112: {"name" : "phaseCorrPark" ,    "type" : "phase"},
    113: {"name" : "phaseCorrNW" ,      "type" : "phase"},
    114: {"name" : "phaseCorrNE" ,      "type" : "phase"},
    115: {"name" : "phaseCorrSW" ,      "type" : "phase"},
    116: {"name" : "phaseCorrSE" ,      "type" : "phase"},
    117: {"name" : "t" ,                "type" : "phase"},
    118: {"name" : "s" ,                "type" : "phase"},
    119: {"name" : "z" ,                "type" : "phase"},
    120: {"name" : "sdag" ,             "type" : "phase"},
    121: {"name" : "tdag" ,             "type" : "phase"},
}


valid_types = {'ge', 'ef', 'spec', 'raw-drag', 'ef-raw', 'square', 'phase'}

# for key, value in enumerate(range(15, 59)):
#     print(r'{} : {{"name": "rx{:.0f}", "theta": {:.3f} , "phi" : 0 , "type" : "ge"}},'.format(value, __get_nearest_rotation(np.round(np.arange(0,360,6.666)), (key+4)*6.666), (key+4)*6.666))

# for key, value in enumerate(range(70, 128)):
#     print(r'{} : {{"name": "ry{:.0f}", "theta": {:.3f} , "phi" : 90 , "type" : "ge"}},'.format(value, __get_nearest_rotation(np.round(np.arange(0,360,6.666)), key*6.666), key*6.666))


def mw_lutmap_is_valid(lutmap: dict) -> bool:
    """
    Test if lutmap obeys schema.

    Args:
        lutmap
    Return:
        valid (bool):
    """
    # FIXME: make this part of the validator for the LutMap parameter.
    for key, value in lutmap.items():
        if not isinstance(key, int):
            raise TypeError
        if value['type'] not in valid_types:
            raise ValueError("{} not in {}".format(value['type'],
                                                   valid_types))

    return True


def theta_to_amp(theta: float, amp180: float):
    """
    Convert θ in deg to pulse amplitude based on a reference amp180.

    Note that all angles are mapped onto the domain [-180, 180) so that
    the minimum possible angle for each rotation is used.
    """
    # phase wrapped to [-180, 180)
    theta_wrap = ((-theta+180) % 360-180)*-1
    amp = theta_wrap/180*amp180
    return amp


class Base_MW_LutMan(Base_LutMan):
    """
    The base class for the microwave lutman.

    Standard microwave pulses are generated based on a lutmap.

    - Schema of lutmap.
    - important attributes
        self._wave_dict

    Typical usage flow of the mw-lutmans

        1. specify a lutmap that determines what waveforms are used.
        2. set some parameters such as mw_amp180
        3. generate waveforms -> stored in self._wave_dict
        4. upload waveforms

    """

    ##########################################################################
    # Base_LutMan overrides
    ##########################################################################

    def set_default_lutmap(self):
        """Set the default lutmap for standard microwave drive pulses."""
        self.LutMap(default_mw_lutmap.copy())

    def set_inspire_lutmap(self):
        """Set the default lutmap for expanded microwave drive pulses."""
        self.LutMap(inspire_mw_lutmap.copy())

    def _add_waveform_parameters(self):
        # defined here so that the VSM based LutMan can overwrite this
        self.wf_func = wf.mod_gauss
        self.spec_func = wf.block_pulse

        self._add_channel_params()
        self._add_mixer_corr_pars()

        self.add_parameter(
            'cfg_sideband_mode',
            vals=vals.Enum('real-time', 'static'),
            initial_value='static',
            parameter_class=ManualParameter
        )

        # pulse parameters
        self.add_parameter(
            'mw_amp180',
            unit='frac',
            vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=1.0
        )
        self.add_parameter(
            'mw_amp90_scale',
            vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=0.5
        )
        self.add_parameter(
            'mw_motzoi',
            vals=vals.Numbers(-2, 2),
            parameter_class=ManualParameter,
            initial_value=0.0
        )
        self.add_parameter(
            'mw_gauss_width',
            vals=vals.Numbers(min_value=1e-9),
            unit='s',
            parameter_class=ManualParameter,
            initial_value=4e-9
        )
        self.add_parameter(
            'mw_phi',
            label='Phase of Rphi pulse',
            vals=vals.Numbers(),
            unit='deg',
            parameter_class=ManualParameter,
            initial_value=0
        )

        # spec parameters
        self.add_parameter(
            'spec_length',
            vals=vals.Numbers(),
            unit='s',
            parameter_class=ManualParameter,
            initial_value=20e-9
        )
        self.add_parameter(
            'spec_amp',
            vals=vals.Numbers(),
            unit='frac',
            parameter_class=ManualParameter,
            initial_value=1
        )

        # parameters related to timings
        self.add_parameter(
            'pulse_delay',
            unit='s',
            vals=vals.Numbers(0, 1e-6),
            parameter_class=ManualParameter,
            initial_value=0
        )
        # square pulse duration for larger pulses
        self.add_parameter(
            'sq_pulse_duration',
            unit='s',
            vals=vals.Numbers(0, 1e-6),
            parameter_class=ManualParameter,
            initial_value=40e-9
        )

        self.add_parameter(
            'mw_modulation',
            vals=vals.Numbers(),
            unit='Hz',
            docstring=('Modulation frequency for qubit driving pulses. Note'
                       ' that when using an AWG with builtin modulation this'
                       ' should be set to 0.'),
            parameter_class=ManualParameter,
            initial_value=50.0e6
        )
        self.add_parameter(
            'mw_ef_modulation',
            vals=vals.Numbers(),
            unit='Hz',
            docstring=('Modulation frequency for driving pulses to the second excited-state.'),
            parameter_class=ManualParameter,
            initial_value=50.0e6
        )
        self.add_parameter(
            'mw_ef_amp180',
            unit='frac',
            docstring=('Pulse amplitude for pulsing the ef/12 transition'),
            vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=.2
        )

    def generate_standard_waveforms(self, apply_predistortion_matrix: bool=True):
        self._wave_dict = OrderedDict()

        if self.cfg_sideband_mode() == 'static':
            f_modulation = self.mw_modulation()
        else:
            f_modulation = 0

        # lutmap is expected to obey lutmap mw schema
        for idx, waveform in self.LutMap().items():
            if waveform['type'] == 'ge':
                if waveform['theta'] == 90:
                    amp = self.mw_amp180()*self.mw_amp90_scale()
                elif waveform['theta'] == -90:
                    amp = - self.mw_amp180() * self.mw_amp90_scale()
                else:
                    amp = theta_to_amp(theta=waveform['theta'],
                                       amp180=self.mw_amp180())
                self._wave_dict[idx] = self.wf_func(
                    amp=amp,
                    phase=waveform['phi'],
                    sigma_length=self.mw_gauss_width(),
                    f_modulation=f_modulation,
                    sampling_rate=self.sampling_rate(),
                    motzoi=self.mw_motzoi(),
                    delay=self.pulse_delay())

            elif waveform['type'] == 'ef':
                amp = theta_to_amp(theta=waveform['theta'],
                                   amp180=self.mw_ef_amp180())
                self._wave_dict[idx] = self.wf_func(
                    amp=amp,
                    phase=waveform['phi'],
                    sigma_length=self.mw_gauss_width(),
                    f_modulation=self.mw_ef_modulation(),
                    sampling_rate=self.sampling_rate(),
                    motzoi=0,
                    delay=self.pulse_delay())

            elif waveform['type'] == 'raw-drag':
                self._wave_dict[idx] = self.wf_func(
                    **waveform["drag_pars"])

            elif waveform['type'] == 'spec':
                self._wave_dict[idx] = self.spec_func(
                    amp=self.spec_amp(),
                    length=self.spec_length(),
                    sampling_rate=self.sampling_rate(),
                    delay=0,
                    phase=0)

            elif waveform['type'] == 'square':
                # Using a slightly different construction as above
                # as the call signatures of these functions is different.
                # Apperently the VSM LutMan has both parameters, so make sure
                # we detect on the one only available in the VSM. Otherwise, we
                # won't get the needed four waveforms.
                if 'duration' in waveform.keys():
                    sq_pulse_duration = waveform['duration']
                else:
                    sq_pulse_duration = self.sq_pulse_duration()

                if 'sq_G_amp' in self.parameters:
                    self._wave_dict[idx] = wf.mod_square_VSM(
                        amp_G=self.sq_G_amp(),
                        amp_D=self.sq_D_amp(),
                        length=sq_pulse_duration,#self.mw_gauss_width()*4,
                        f_modulation=self.mw_modulation() if self.cfg_sideband_mode()!='real-time' else 0,
                        sampling_rate=self.sampling_rate()
                    )
                elif 'sq_amp' in self.parameters:
                    self._wave_dict[idx] = wf.mod_square(
                        amp=self.sq_amp(),
                        length=sq_pulse_duration,
                        f_modulation=self.mw_modulation() if self.cfg_sideband_mode()!='real-time' else 0,
                        phase=0,
                        motzoi=0,
                        sampling_rate=self.sampling_rate()
                    )
                else:
                    raise KeyError('Expected parameter "sq_amp" to exist')
            elif waveform['type'] == 'phase':
                #self._wave_dict[idx] = self.spec_func(
                    #amp=0,
                    ##length=self.mw_gauss_width()*4,
                    ## LDC Kludge for Inspire 2022/07/19
                    #length=20e-9,     
                    #sampling_rate=self.sampling_rate(),
                    #delay=0,
                    #phase=0)
                self._wave_dict[idx] = self.wf_func(
                    amp=0,
                    phase=0,
                    sigma_length=self.mw_gauss_width(),
                    f_modulation=f_modulation,
                    sampling_rate=self.sampling_rate(),
                    motzoi=self.mw_motzoi(),
                    delay=self.pulse_delay())
            else:
                raise ValueError

        # Add predistortions + test
        if (self.mixer_apply_predistortion_matrix()
                and apply_predistortion_matrix and self.cfg_sideband_mode != 'real-time'):
            self._wave_dict = self.apply_mixer_predistortion_corrections(
                self._wave_dict)
        return self._wave_dict

    # FIXME: seems to be overridden in all derived classes: remove
    def load_waveform_onto_AWG_lookuptable(
            self,
            waveform_idx: int,
            regenerate_waveforms: bool=False
            ):
        if not isinstance(waveform_idx, int) \
                or waveform_idx < 0 \
                or waveform_idx > 127:
            raise ValueError("`waveform_idx` must be a valid LutMap index!")

        if regenerate_waveforms:
            self.generate_standard_waveforms()

        waveforms = self._wave_dict[waveform_idx]
        codewords = self.codeword_idx_to_parnames(waveform_idx)

        for waveform, cw in zip(waveforms, codewords):
            self.AWG.get_instr().set(cw, waveform)

    ##########################################################################
    # Base_MW_LutMan functions (may be overridden in subclass)
    ##########################################################################

    def apply_mixer_predistortion_corrections(self, wave_dict):
        M = wf.mixer_predistortion_matrix(
            self.mixer_alpha(),
            self.mixer_phi()
        )
        for key, val in wave_dict.items():
            wave_dict[key] = np.dot(M, val)
        return wave_dict

    def _add_mixer_corr_pars(self):
        self.add_parameter(
            'mixer_alpha',
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
            initial_value=1.0
        )
        self.add_parameter(
            'mixer_phi',
            vals=vals.Numbers(),
            unit='deg',
            parameter_class=ManualParameter,
            initial_value=0.0
        )
        self.add_parameter(
            'mixer_apply_predistortion_matrix',
            vals=vals.Bool(),
            docstring=(
                'If True applies a mixer correction using mixer_phi and '
                'mixer_alpha to all microwave pulses.'),
            parameter_class=ManualParameter,
            initial_value=True
        )

    def _add_channel_params(self):
        """
        add parameters that define connectivity of logical channels to
        hardware channel numbers of the instrument involved (i.e. self.AWG)
        """
        self.add_parameter(
            'channel_I',
            parameter_class=ManualParameter,
            vals=vals.Numbers(1, self._num_channels)
        )
        self.add_parameter(
            'channel_Q',
            parameter_class=ManualParameter,
            vals=vals.Numbers(1, self._num_channels)
        )

    ############################################################################
    # Functions
    # FIXME: the load_* functions provide an undesired backdoor, also see issue #626
    ############################################################################

    def load_phase_pulses_to_AWG_lookuptable(self, phases=np.arange(0, 360, 20)):
        """
        Loads rPhi90 pulses onto the AWG lookuptable.
        """
        startIndex=32  # changed from 9, LDC, 22/10/23
        if len(phases) > 18:
            raise ValueError('max 18 phase values can be provided')
        for i, (phase) in enumerate(phases):
            self.LutMap()[startIndex+i] = {"name": "rPhi90", "theta": 90, "phi": phase, "type": "ge"}
        self.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)

    # FIXME: function is almost identical to load_phase_pulses_to_AWG_lookuptable, except for phi vs. theta
    def load_x_pulses_to_AWG_lookuptable(self, phases=np.arange(0, 360, 20)):
        """
        Loads rPhi90 pulses onto the AWG lookuptable.
        """

        if (len(phases) > 18):
            raise ValueError('max 18 amplitude values can be provided')
        lm = self.LutMap()
        startIndex=32 # changed from 9, LDC, 2022/10/23
        for i, (phase) in enumerate(phases):
            lm[startIndex+i] = {"name": "rPhi90",    "theta": phase,
                       "phi": 0, "type": "ge"}
        self.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)

    def load_square_waves_to_AWG_lookuptable(self):
        """
        Loads square pulses onto the AWG lookuptable.
        """

        self.set_default_lutmap()
        lm = self.LutMap()
        lm[10] = {"name": "square",
                    "type": "square",
                    "duration": 1e-6}
        lm[11] = {"name": "cw_11",
                    "type": "square"}
        for i in range(12,21):
            div = i-12
            lm[i] = {"name": "cw_{}".format(i),
                        "type": "square",
                        "duration": 40e-9*(i-11)/10}
        self.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)

    def load_ef_rabi_pulses_to_AWG_lookuptable(self, amps: list=None,
                                               mod_freqs: list=None):
        """
        Special loading method that loads (up to) 18 pulses in
        order to do a rabi on the ef (1-2) transition.

        This method also generates the waveforms.

        This method contains several steps
            1. determine what ef-pulses to generate
            2. generate a LutMap to use and upload the waveforms
            3. generate and upload waveforms.
        """

        # 1. Determine what ef-pulses to generate
        if not isinstance(amps, Iterable) and (mod_freqs is None):
            amps = [self.mw_ef_amp180()]
        elif len(amps) == 1:
            amps = [amps]*len(mod_freqs)

        if (len(amps) > 18):
            raise ValueError('max 18 amplitude values can be provided')

        if mod_freqs is None:
            mod_freqs = [self.mw_ef_modulation()]*len(amps)
        elif len(mod_freqs) == 1:
            mod_freqs = [mod_freqs]*len(amps)

        # 2. Generate a LutMap for the ef-pulses
        # FIXME: hardcoded indices must match OpenQL definitions
        lm = self.LutMap()
        startIndex=32 # changed from 9, LDC, 2022/10/23
        for i, (amp, mod_freq) in enumerate(zip(amps, mod_freqs)):
            lm[startIndex+i] = {"name": "", "type": "raw-drag",
                       "drag_pars": {
                           "amp": amp, "f_modulation": mod_freq,
                           "sigma_length": self.mw_gauss_width(),
                           "sampling_rate": self.sampling_rate(),
                           "motzoi": 0}
                       }

        # 3. generate and upload waveforms
        self.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)

    ##########################################################################
    # Private functions
    ##########################################################################

    def _codeword_idx_to_parnames(self, cw_idx: int):
        """Convert a codeword_idx to a list of par names for the waveform."""
        # the possible channels way of doing this is to make it work both for
        # VSM style lutmans and no VSM style lutmans.
        possible_channels = ('channel_GI', 'channel_GQ',
                             'channel_DI', 'channel_DQ',
                             'channel_I', 'channel_Q')
        codewords = ['wave_ch{}_cw{:03}'.format(self[ch](), cw_idx)
                     for ch in possible_channels if hasattr(self, ch)]
        return codewords


class QWG_MW_LutMan(Base_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 4
        super().__init__(name, **kw)

    ##########################################################################
    # Base_LutMan overrides
    ##########################################################################

    # FIXME: the parameter set depends on the subclass, which is awkward to handle in HAL_Transmon
    def _add_channel_params(self):
        super()._add_channel_params()
        self.add_parameter(
            'channel_amp',
            unit='a.u.',
            vals=vals.Numbers(-1.8, 1.8),
            set_cmd=self._set_channel_amp,
            get_cmd=self._get_channel_amp,
            docstring=('using the channel amp as additional'
                       'parameter to allow rabi-type experiments without'
                       'wave reloading. Should not be using VSM')
        )
        # parameters related to codeword bits
        self.add_parameter(
            'bit_shift',
            unit='',
            vals=vals.Ints(0, 8),
            parameter_class=ManualParameter,
            initial_value=0
        )
        self.add_parameter(
            'bit_width', unit='', vals=vals.Ints(0, 8),
            parameter_class=ManualParameter,
            initial_value=0
        )

    def _add_waveform_parameters(self):
        super()._add_waveform_parameters()
        # Parameters for a square pulse
        self.add_parameter(
            'sq_amp',
            unit='frac', vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=0.5
        )

    def load_waveform_onto_AWG_lookuptable(
            self, wave_id: str, regenerate_waveforms: bool=False):
        """
        Load a waveform into the AWG.

        Args:
            wave_id: can be either the "name" of a waveform or
                the integer key in self._wave_dict.
            regenerate_waveforms (bool) : if True regenerates all waveforms
        """
        if regenerate_waveforms:
            self.generate_standard_waveforms()

        if wave_id not in self.LutMap().keys():
            wave_id = get_wf_idx_from_name(wave_id, self.LutMap())

        wf_I, wf_Q = self._wave_dict[wave_id]

        wf_name_I = 'wave_ch{}_cw{:03}'.format(self.channel_I(), wave_id)
        wf_name_Q = 'wave_ch{}_cw{:03}'.format(self.channel_Q(), wave_id)

        self.AWG.get_instr().set(wf_name_I, wf_I)
        self.AWG.get_instr().set(wf_name_Q, wf_Q)

    ##########################################################################
    # Base_MW_LutMan overrides
    ##########################################################################

    def apply_mixer_predistortion_corrections(self, wave_dict):
        M = wf.mixer_predistortion_matrix(self.mixer_alpha(),
                                          self.mixer_phi())

        self.AWG.get_instr().set(
            'ch_pair{}_transform_matrix'.format(self.channel_I()), M)
        # in the QWG the predistortion matrix is implemented in hardware
        # and does not modify the actual wave dict.
        # the wave dict is still returned unmodified to preserve the
        # call signature as required for HDAWG compatibility.

        return wave_dict

    ##########################################################################
    # Private functions
    ##########################################################################

    def _set_channel_amp(self, val):
        AWG = self.AWG.get_instr()
        AWG.set('ch{}_amp'.format(self.channel_I()), val)
        AWG.set('ch{}_amp'.format(self.channel_Q()), val)

    def _get_channel_amp(self):
        AWG = self.AWG.get_instr()
        val_I = AWG.get('ch{}_amp'.format(self.channel_I()))
        val_Q = AWG.get('ch{}_amp'.format(self.channel_Q()))
        assert val_Q == val_I
        return val_I


class AWG8_MW_LutMan(Base_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 8
        super().__init__(name, **kw)
        self.sampling_rate(2.4e9)
        self._add_phase_correction_parameters()

    ##########################################################################
    # Base_LutMan overrides
    ##########################################################################

    # FIXME: the parameter set depends on the subclass, which is awkward to handle in HAL_Transmon
    def _add_channel_params(self):
        super()._add_channel_params()
        self.add_parameter(
            'channel_amp',
            unit='a.u.', vals=vals.Numbers(0, 1),
            set_cmd=self._set_channel_amp,
            get_cmd=self._get_channel_amp,
            docstring=('using the channel amp as additional'
                       'parameter to allow rabi-type experiments without'
                       'wave reloading. Should not be using VSM')
        )
        self.add_parameter(
            'channel_range',
            unit='V', vals=vals.Enum(0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5),
            set_cmd=self._set_channel_range,
            get_cmd=self._get_channel_range,
            docstring=('defines the channel range for the AWG sequencer output')
        )

        # Setting variable to track channel amplitude since it cannot be directly extracted from
        # HDAWG while using real-time modulation (because of mixer amplitude imbalance corrections)
        self.channel_amp_value = 0

    def _add_waveform_parameters(self):
        super()._add_waveform_parameters()
        # Parameters for a square pulse
        self.add_parameter('sq_amp', unit='frac', vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.5)

    def _add_phase_correction_parameters(self):
        self.add_parameter(
            name=f'vcz_virtual_q_ph_corr_park',
            parameter_class=ManualParameter,
            unit='deg',
            vals=vals.Numbers(-360, 360),
            initial_value=0.0,
            docstring=f"Virtual phase correction for parking."
                        "Will be applied as increment to sine generator phases via command table."
        )

        # corrections for phases that the qubit can acquire during one of its CZ gates
        for gate in ['NW','NE','SW','SE']:
            self.add_parameter(
                name=f'vcz_virtual_q_ph_corr_{gate}',
                parameter_class=ManualParameter,
                unit='deg',
                vals=vals.Numbers(-360, 360),
                initial_value=0.0,
                docstring=f"Virtual phase correction for two-qubit gate in {gate}-direction."
                            "Will be applied as increment to sine generator phases via command table."
            )

        # corrections for phases that the qubit can acquire during parking as spectator of a CZ gate.
        # this can happen in general for each of its neighbouring qubits (below: 'direction'),
        # while it is doing a gate in each possible direction (below: 'gate')
        # for direction in ['NW','NE','SW','SE']:
        #     for gate in ['NW','NE','SW','SE']:
        #         self.add_parameter(
        #             name=f'vcz_virtual_q_ph_corr_spec_{direction}_gate_{gate}',
        #             parameter_class=ManualParameter,
        #             unit='deg',
        #             vals=vals.Numbers(0, 360),
        #             initial_value=0.0,
        #             docstring=f"Virtual phase correction for parking as spectator of a qubit in direction {direction}, "
        #                       f"that is doing a gate in direction {gate}."
        #                         "Will be applied as increment to sine generator phases via command table."
        #         )

        # corrections for phases that the qubit can acquire during parking as part of a flux-dance step
        # there are 8 flux-dance steps for the S17 scheme.
        # NOTE: this correction must not be the same as the above one for the case of a spectator
        #       for a single CZ, because in a flux-dance the qubit can be parked because of multiple adjacent CZ gates
        # for step in np.arange(1,9):
        #     self.add_parameter(
        #         name=f'vcz_virtual_q_ph_corr_park_step_{step}',
        #         parameter_class=ManualParameter,
        #         unit='deg',
        #         vals=vals.Numbers(-360, 360),
        #         initial_value=0.0,
        #         docstring=f"Virtual phase correction for parking in flux-dance step {step}."
        #                     "Will be applied as increment to sine generator phases via command table."
        #     )

    def _reset_phase_correction_parameters(self):
        for gate in ['NW','NE','SW','SE']:
            self.parameters[f'vcz_virtual_q_ph_corr_{gate}'](0)
        for step in np.arange(1,9):
            self.parameters[f'vcz_virtual_q_ph_corr_park_step_{step}'](0)


    def _set_channel_range(self, val):
        awg_nr = (self.channel_I()-1)//2
        assert awg_nr == (self.channel_Q()-1)//2
        assert self.channel_I() < self.channel_Q()
        AWG = self.AWG.get_instr()
        if val == 0.8:
            AWG.set('sigouts_{}_range'.format(self.channel_I()-1), .8)
            AWG.set('sigouts_{}_direct'.format(self.channel_I()-1), 1)
            AWG.set('sigouts_{}_range'.format(self.channel_Q()-1), .8)
            AWG.set('sigouts_{}_direct'.format(self.channel_Q()-1), 1)
        else:
            AWG.set('sigouts_{}_direct'.format(self.channel_I()-1), 0)
            AWG.set('sigouts_{}_range'.format(self.channel_I()-1), val)
            AWG.set('sigouts_{}_direct'.format(self.channel_Q()-1), 0)
            AWG.set('sigouts_{}_range'.format(self.channel_Q()-1), val)


    def _get_channel_range(self):
        awg_nr = (self.channel_I()-1)//2
        assert awg_nr == (self.channel_Q()-1)//2
        assert self.channel_I() < self.channel_Q()

        AWG = self.AWG.get_instr()
        val = AWG.get('sigouts_{}_range'.format(self.channel_I()-1))
        assert val == AWG.get('sigouts_{}_range'.format(self.channel_Q()-1))
        return val

    def _set_channel_amp(self, val):
        AWG = self.AWG.get_instr()
        awg_nr = (self.channel_I()-1)//2
        # Enforce assumption that channel I preceeds channel Q and share AWG
        assert awg_nr == (self.channel_Q()-1)//2
        assert self.channel_I() < self.channel_Q()
        self.channel_amp_value = val

        if self.cfg_sideband_mode() == 'static':
            AWG.set('awgs_{}_outputs_{}_gains_0'.format(awg_nr, 0), val)
            AWG.set('awgs_{}_outputs_{}_gains_0'.format(awg_nr, 1), 0)
            AWG.set('awgs_{}_outputs_{}_gains_1'.format(awg_nr, 0), 0)
            AWG.set('awgs_{}_outputs_{}_gains_1'.format(awg_nr, 1), val)

        # In case of sideband modulation mode 'real-time', amplitudes have to be set
        # according to modulation matrix
        elif self.cfg_sideband_mode() == 'real-time':
            g0 = np.tan(np.radians(self.mixer_phi()))
            g1 = self.mixer_alpha()*1/np.cos(np.radians(self.mixer_phi()))

            if np.abs(val*g0) > 1.0 or np.abs(val*g1) > 1.0:
                raise Exception('Resulting amplitude from mixer parameters '+\
                                'exceed the maximum channel amplitude')
                # print('Resulting amplitude from mixer parameters '+\
                #       'exceed the maximum channel amplitude')
                # if np.abs(val*g0):
                #     g0 = 1/val
                # if np.abs(val*g1):
                #     g1 = 1/val

            AWG.set('awgs_{}_outputs_0_gains_0'.format(awg_nr), val)
            AWG.set('awgs_{}_outputs_1_gains_0'.format(awg_nr), 0)
            AWG.set('awgs_{}_outputs_0_gains_1'.format(awg_nr), val*g0)
            AWG.set('awgs_{}_outputs_1_gains_1'.format(awg_nr), val*g1)
        else:
            raise KeyError('Unexpected value for parameter sideband mode.')

    def _get_channel_amp(self):
        AWG = self.AWG.get_instr()
        awg_nr = (self.channel_I()-1)//2
        # Enforce assumption that channel I precedes channel Q and share AWG
        assert awg_nr == (self.channel_Q()-1)//2
        assert self.channel_I() < self.channel_Q()

        vals = []
        if self.cfg_sideband_mode() == 'static':
            vals.append(AWG.get('awgs_{}_outputs_{}_gains_0'.format(awg_nr, 0)))
            vals.append(AWG.get('awgs_{}_outputs_{}_gains_1'.format(awg_nr, 0)))
            vals.append(AWG.get('awgs_{}_outputs_{}_gains_0'.format(awg_nr, 1)))
            vals.append(AWG.get('awgs_{}_outputs_{}_gains_1'.format(awg_nr, 1)))
            assert vals[0]==vals[3]
            assert vals[1]==vals[2]==0

        # In case of sideband modulation mode 'real-time', amplitudes have to be set
        # according to modulation matrix
        elif self.cfg_sideband_mode() == 'real-time':
            vals.append(self.channel_amp_value)

        return vals[0]

    def load_waveform_onto_AWG_lookuptable(
        self, wave_id: str, regenerate_waveforms: bool=False):
        """
        Load a waveform into the AWG.

        Args:
            wave_id: can be either the "name" of a waveform or
                the integer key in self._wave_dict.
            regenerate_waveforms (bool) : if True regenerates all waveforms
        """
        if regenerate_waveforms:
            self.generate_standard_waveforms()

        if wave_id not in self.LutMap().keys():
            wave_id = get_wf_idx_from_name(wave_id, self.LutMap())

        wf_I, wf_Q = self._wave_dict[wave_id]

        wf_name_I = 'wave_ch{}_cw{:03}'.format(self.channel_I(), wave_id)
        wf_name_Q = 'wave_ch{}_cw{:03}'.format(self.channel_Q(), wave_id)

        self.AWG.get_instr().set(wf_name_I, wf_I)
        self.AWG.get_instr().set(wf_name_Q, wf_Q)

    def load_waveforms_onto_AWG_lookuptable(
            self,
            regenerate_waveforms: bool=True,
            stop_start: bool = True,
            force_load_sequencer_program: bool=False
    ):
        """
        Loads all waveforms specified in the LutMap to an AWG.

        Args:
            regenerate_waveforms (bool): if True calls generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.
            force_load_sequencer_program (bool): if True forces a new compilation
                and upload of the program on the sequencer. FIXME: parameter pack incompatible with base class
        """
        # Uploading the codeword program (again) is needed to link the new
        # waveforms in case the user has changed the codeword mode.
        if force_load_sequencer_program:
            # This ensures only the channels that are relevant get reconfigured
            if 'channel_GI' in self.parameters:
                awgs = [self.channel_GI()//2, self.channel_DI()//2]
            else:
                awgs = [self.channel_I()//2]
                # Enforce assumption that channel I precedes channel Q
                assert self.channel_I() < self.channel_Q()
                assert (self.channel_I())//2 < (self.channel_Q())//2

            self.AWG.get_instr().upload_codeword_program(awgs=awgs)

        # This ensures that settings other than the sequencer program are updated
        # for different sideband modulation modes
        if self.cfg_sideband_mode() == 'static':
            self.AWG.get_instr().cfg_sideband_mode('static')
            # Turn off modulation modes
            self.AWG.get_instr().set('awgs_{}_outputs_0_modulation_mode'.format((self.channel_I()-1)//2), 0)
            self.AWG.get_instr().set('awgs_{}_outputs_1_modulation_mode'.format((self.channel_Q()-1)//2), 0)

        elif self.cfg_sideband_mode() == 'real-time':
            if (self.channel_I()-1)//2 != (self.channel_Q()-1)//2:
                raise KeyError('In real-time sideband mode, channel I/Q should share same awg nr.')
            self.AWG.get_instr().cfg_sideband_mode('real-time')

            # Set same oscillator for I/Q pair and same harmonic
            self.AWG.get_instr().set('sines_{}_oscselect'.format(self.channel_I()-1), (self.channel_I()-1)//2)
            self.AWG.get_instr().set('sines_{}_oscselect'.format(self.channel_Q()-1), (self.channel_I()-1)//2)
            self.AWG.get_instr().set('sines_{}_harmonic'.format(self.channel_I()-1), 1)
            self.AWG.get_instr().set('sines_{}_harmonic'.format(self.channel_Q()-1), 1)
            # Create respective cossine/sin signals for modulation through phase-shift
            self.AWG.get_instr().set('sines_{}_phaseshift'.format(self.channel_I()-1), 90)
            self.AWG.get_instr().set('sines_{}_phaseshift'.format(self.channel_Q()-1), 0)
            # Create correct modulation modeI
            self.AWG.get_instr().set('awgs_{}_outputs_0_modulation_mode'.format((self.channel_I()-1)//2), 6)
            self.AWG.get_instr().set('awgs_{}_outputs_1_modulation_mode'.format((self.channel_Q()-1)//2), 6)
        else:
            raise ValueError('Unexpected value for parameter cfg_sideband_mode.')

        super().load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=regenerate_waveforms,
            stop_start=stop_start)

    ##########################################################################
    # Base_MW_LutMan overrides
    ##########################################################################

    def apply_mixer_predistortion_corrections(self, wave_dict):
            M = wf.mixer_predistortion_matrix(self.mixer_alpha(), self.mixer_phi())
            for key, val in wave_dict.items():
                wave_dict[key] = np.dot(M, val)
            return wave_dict

    def generate_standard_waveforms(
            self, apply_predistortion_matrix: bool=True):
        # FIXME: looks very similar to overridden function in Base_MW_LutMan
        self._wave_dict = OrderedDict()

        if self.cfg_sideband_mode() == 'static':
            f_modulation = self.mw_modulation()
        elif self.cfg_sideband_mode() == 'real-time':
            f_modulation = 0
            if ((self.channel_I()-1)//2 != (self.channel_Q()-1)//2):
                raise KeyError('In real-time sideband mode, channel I/Q should share same awg group.')

            self.AWG.get_instr().set('oscs_{}_freq'.format((self.channel_I()-1)//2),
                                     self.mw_modulation())
        else:
            raise KeyError('Unexpected argument for cfg_sideband_mode')

        # lutmap is expected to obey lutmap mw schema
        for idx, waveform in self.LutMap().items():
            if waveform['type'] == 'ge':
                if waveform['theta'] == 90:
                    amp = self.mw_amp180()*self.mw_amp90_scale()
                elif waveform['theta'] == -90:
                    amp = - self.mw_amp180() * self.mw_amp90_scale()
                else:
                    amp = theta_to_amp(theta=waveform['theta'],
                                       amp180=self.mw_amp180())
                self._wave_dict[idx] = self.wf_func(
                    amp=amp,
                    phase=waveform['phi'],
                    sigma_length=self.mw_gauss_width(),
                    f_modulation=f_modulation,
                    sampling_rate=self.sampling_rate(),
                    motzoi=self.mw_motzoi(),
                    delay=self.pulse_delay())

            elif waveform['type'] == 'ef':
                amp = theta_to_amp(theta=waveform['theta'],
                                   amp180=self.mw_ef_amp180())
                self._wave_dict[idx] = self.wf_func(
                    amp=amp,
                    phase=waveform['phi'],
                    sigma_length=self.mw_gauss_width(),
                    f_modulation=self.mw_ef_modulation(),
                    sampling_rate=self.sampling_rate(),
                    motzoi=0,
                    delay=self.pulse_delay())

            elif waveform['type'] == 'raw-drag':
                self._wave_dict[idx] = self.wf_func(
                    **waveform["drag_pars"])

            elif waveform['type'] == 'spec':
                self._wave_dict[idx] = self.spec_func(
                    amp=self.spec_amp(),
                    length=self.spec_length(),
                    sampling_rate=self.sampling_rate(),
                    delay=0,
                    phase=0)

            elif waveform['type'] == 'square':
                # Using a slightly different construction as above
                # as the call signatures of these functions is different.
                # Apperently the VSM LutMan has both parameters, so make sure
                # we detect on the one only available in the VSM. Otherwise, we
                # won't get the needed four waveforms.
                if 'sq_G_amp' in self.parameters:
                    self._wave_dict[idx] = wf.mod_square_VSM(
                        amp_G=self.sq_G_amp(),
                        amp_D=self.sq_D_amp(),
                        length=self.mw_gauss_width()*4,
                        f_modulation=self.mw_modulation() if self.cfg_sideband_mode()!='real-time' else 0,
                        sampling_rate=self.sampling_rate()
                    )
                elif 'sq_amp' in self.parameters:
                    self._wave_dict[idx] = wf.mod_square(
                        amp=self.sq_amp(),
                        length=self.mw_gauss_width()*4,
                        f_modulation=self.mw_modulation() if self.cfg_sideband_mode()!='real-time' else 0,
                        phase=0,
                        sampling_rate=self.sampling_rate()
                    )
                else:
                    raise KeyError('Expected parameter "sq_amp" to exist')

            elif waveform['type'] == 'phase':
                #fill codewords that are used for phase correction instructions
                # with a zero waveform
                self._wave_dict[idx] = wf.block_pulse(
                    amp=0,
                    sampling_rate=self.sampling_rate(),
                    #length=self.mw_gauss_width()*4,
                    length=20e-9,
                    )
                #self._wave_dict[idx] = self.wf_func(
                #    amp=0,
                #    phase=0,
                #    sigma_length=self.mw_gauss_width(),
                #    f_modulation=f_modulation,
                #    sampling_rate=self.sampling_rate(),
                #    motzoi=self.mw_motzoi(),
                #    delay=self.pulse_delay())
            else:
                raise ValueError

        # Add predistortions + test
        if (self.mixer_apply_predistortion_matrix() and apply_predistortion_matrix and
          self.cfg_sideband_mode() == 'static'):
            self._wave_dict = self.apply_mixer_predistortion_corrections(
                self._wave_dict)
        return self._wave_dict

    ##########################################################################
    # Functions
    # FIXME: these provide an undesired backdoor
    ##########################################################################

    def upload_single_qubit_phase_corrections(self):
        commandtable_dict = {
            "$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
            "header": {"version": "0.2"},
            "table": []
        }

        # manual waveform index 1-to-1 mapping
        for ind in np.arange(0, 112, 1):
            commandtable_dict['table'] += [{"index": int(ind),
                                            "waveform": {"index": int(ind)}
                                            }]

        # add phase corrections to the end of the codeword space
        # the first position is for parking-relatedrelated phase correction,
        # the last 4 are for phase corrections due to gate in corresponding direction
        
        # changed by LDC, 23/01/31
        # this change also requires changing the arange statements above and below
        phase_corr_inds = np.arange(112,117,1)

        phase = self.parameters[f"vcz_virtual_q_ph_corr_park"]()
        commandtable_dict['table'] += [{"index": int(phase_corr_inds[0]),
                                        "phase0": {"value": float(phase), "increment": True},
                                        "phase1": {"value": float(phase), "increment": True}
                                        }]

        for i,d in enumerate(['NW','NE','SW','SE']):
            phase = self.parameters[f"vcz_virtual_q_ph_corr_{d}"]()
            commandtable_dict['table'] += [{"index": int(phase_corr_inds[i+1]),
                                            "phase0": {"value": float(phase), "increment": True},
                                            "phase1": {"value": float(phase), "increment": True}
                                            }]

        # adding virtual gates for some specific Z rotations
        # LDC, 23/01/31
        # T gate
        commandtable_dict['table'] += [{"index": 117,
                                        "phase0": {"value": float(45), "increment": True},
                                        "phase1": {"value": float(45), "increment": True}
                                        }]
        # S gate
        commandtable_dict['table'] += [{"index": 118,
                                        "phase0": {"value": float(90), "increment": True},
                                        "phase1": {"value": float(90), "increment": True}
                                        }]
        # Z gate
        commandtable_dict['table'] += [{"index": 119,
                                        "phase0": {"value": float(180), "increment": True},
                                        "phase1": {"value": float(180), "increment": True}
                                        }]
        # Sdag gate
        commandtable_dict['table'] += [{"index": 120,
                                        "phase0": {"value": float(270), "increment": True},
                                        "phase1": {"value": float(270), "increment": True}
                                        }]
        # Tdag gate
        commandtable_dict['table'] += [{"index": 121,
                                        "phase0": {"value": float(315), "increment": True},
                                        "phase1": {"value": float(315), "increment": True}
                                        }]

        # currently there are 6 unused codewords
        # LDC, 23/01/31
        for ind in np.arange(122, 128, 1):
            commandtable_dict['table'] += [{"index": int(ind),
                                            "waveform": {"index": int(ind)}  
                                            }]  

        # NOTE: Whenever the command table is used, the phase offset between I and Q channels on
        # the HDAWG for real-time modulation has to be initialized from the table itself.
        # Index 1023 will be reserved for this (it should no be used for codeword triggering)
        commandtable_dict['table'] += [{"index": 1023,
                                        "phase0": {"value": 90.0, "increment": False},
                                        "phase1": {"value":  0.0, "increment": False}
                                        }]

        # get internal awg sequencer number (indexed 0,1,2,3)
        awg_nr = (self.channel_I() - 1) // 2
        commandtable_returned, status = self.AWG.get_instr().upload_commandtable(commandtable_dict, awg_nr)

        return commandtable_returned, status

    ##########################################################################
    # Private functions
    ##########################################################################

    def _set_channel_range(self, val):
        awg_nr = (self.channel_I()-1)//2
        assert awg_nr == (self.channel_Q()-1)//2
        assert self.channel_I() < self.channel_Q()
        AWG = self.AWG.get_instr()
        if val == 0.8:
            AWG.set('sigouts_{}_direct'.format(self.channel_I()-1), 1)
            AWG.set('sigouts_{}_range'.format(self.channel_I()-1), .8)
            AWG.set('sigouts_{}_direct'.format(self.channel_Q()-1), 1)
            AWG.set('sigouts_{}_range'.format(self.channel_Q()-1), .8)
            # FIXME: according to the ZI node documentation for SIGOUTS/*/DIRECT, offset control is not avaiable in this mode
        else:
            AWG.set('sigouts_{}_direct'.format(self.channel_I()-1), 0)
            AWG.set('sigouts_{}_range'.format(self.channel_I()-1), val)
            AWG.set('sigouts_{}_direct'.format(self.channel_Q()-1), 0)
            AWG.set('sigouts_{}_range'.format(self.channel_Q()-1), val)

    def _get_channel_range(self):
        awg_nr = (self.channel_I()-1)//2
        assert awg_nr == (self.channel_Q()-1)//2
        assert self.channel_I() < self.channel_Q()

        AWG = self.AWG.get_instr()  # FIXME: this line was missing, so the code below couldn't execute and is probably untested
        val = AWG.get('sigouts_{}_range'.format(self.channel_I()-1))
        assert val == AWG.get('sigouts_{}_range'.format(self.channel_Q()-1))
        return val

    def _set_channel_amp(self, val):
        AWG = self.AWG.get_instr()
        awg_nr = (self.channel_I()-1)//2
        # Enforce assumption that channel I preceeds channel Q and share AWG
        assert awg_nr == (self.channel_Q()-1)//2
        assert self.channel_I() < self.channel_Q()
        self.channel_amp_value = val

        if self.cfg_sideband_mode() == 'static':
            AWG.set('awgs_{}_outputs_{}_gains_0'.format(awg_nr, 0), val)
            AWG.set('awgs_{}_outputs_{}_gains_0'.format(awg_nr, 1), 0)
            AWG.set('awgs_{}_outputs_{}_gains_1'.format(awg_nr, 0), 0)
            AWG.set('awgs_{}_outputs_{}_gains_1'.format(awg_nr, 1), val)

        # In case of sideband modulation mode 'real-time', amplitudes have to be set
        # according to modulation matrix
        elif self.cfg_sideband_mode() == 'real-time':
            g0 = np.tan(np.radians(self.mixer_phi()))
            g1 = self.mixer_alpha()*1/np.cos(np.radians(self.mixer_phi()))

            if np.abs(val*g0) > 1.0 or np.abs(val*g1) > 1.0:
                raise Exception('Resulting amplitude from mixer parameters '+\
                                'exceed the maximum channel amplitude')
                # print('Resulting amplitude from mixer parameters '+\
                #       'exceed the maximum channel amplitude')
                # if np.abs(val*g0):
                #     g0 = 1/val
                # if np.abs(val*g1):
                #     g1 = 1/val

            AWG.set('awgs_{}_outputs_0_gains_0'.format(awg_nr), val)
            AWG.set('awgs_{}_outputs_1_gains_0'.format(awg_nr), 0)
            AWG.set('awgs_{}_outputs_0_gains_1'.format(awg_nr), val*g0)
            AWG.set('awgs_{}_outputs_1_gains_1'.format(awg_nr), val*g1)
        else:
            raise KeyError('Unexpected value for parameter sideband mode.')

    def _get_channel_amp(self):
        AWG = self.AWG.get_instr()
        awg_nr = (self.channel_I()-1)//2

        # Enforce assumption that channel I precedes channel Q and share AWG
        assert awg_nr == (self.channel_Q()-1)//2
        assert self.channel_I() < self.channel_Q()

        vals = []
        if self.cfg_sideband_mode() == 'static':
            vals.append(AWG.get('awgs_{}_outputs_{}_gains_0'.format(awg_nr, 0)))
            vals.append(AWG.get('awgs_{}_outputs_{}_gains_1'.format(awg_nr, 0)))
            vals.append(AWG.get('awgs_{}_outputs_{}_gains_0'.format(awg_nr, 1)))
            vals.append(AWG.get('awgs_{}_outputs_{}_gains_1'.format(awg_nr, 1)))
            assert vals[0]==vals[3]
            assert vals[1]==vals[2]==0

        # In case of sideband modulation mode 'real-time', amplitudes have to be set
        # according to modulation matrix
        elif self.cfg_sideband_mode() == 'real-time':
            vals.append(self.channel_amp_value)

        return vals[0]

class AWG8_VSM_MW_LutMan(AWG8_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 8
        super().__init__(name, **kw)
        self.wf_func = wf.mod_gauss_VSM
        self.spec_func = wf.block_pulse_vsm

    ##########################################################################
    # Base_LutMan overrides
    ##########################################################################

    # FIXME: the parameter set depends on the subclass, which is awkward to handle in HAL_Transmon
    def _add_waveform_parameters(self):
        super()._add_waveform_parameters()
        # Base_MW_LutMan._add_waveform_parameters(self)
        # Parameters for a square pulse
        self.add_parameter(
            'sq_G_amp',
            unit='frac',
            vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=0.5
        )
        self.add_parameter(
            'sq_D_amp',
            unit='frac',
            vals=vals.Numbers(-1, 1),
            parameter_class=ManualParameter,
            initial_value=0
        )

    def _add_channel_params(self):
        self.add_parameter(
            'channel_amp',
            unit='a.u.',
            vals=vals.Numbers(0, 1),
            set_cmd=self._set_channel_amp,
            get_cmd=self._get_channel_amp,
            docstring=('using the channel amp as additional'
                       'parameter to allow rabi-type experiments without'
                       'wave reloading. Should not be using VSM')
        )

        for ch in ['GI', 'GQ', 'DI', 'DQ']:
            self.add_parameter(
                'channel_{}'.format(ch),
                parameter_class=ManualParameter,
                vals=vals.Numbers(1, self._num_channels)
            )

    def load_waveform_onto_AWG_lookuptable(
        self, wave_id: str, regenerate_waveforms: bool=False):
        """
        Load a waveform into the AWG.

        Args:
            wave_id: can be either the "name" of a waveform or
                the integer key in self._wave_dict.
            regenerate_waveforms (bool) : if True regenerates all waveforms
        """
        if regenerate_waveforms:
            self.generate_standard_waveforms()

        if wave_id not in self.LutMap().keys():
            wave_id = get_wf_idx_from_name(wave_id, self.LutMap())

        GI, GQ, DI, DQ = self._wave_dict[wave_id]

        wf_name_GI = 'wave_ch{}_cw{:03}'.format(self.channel_GI(), wave_id)
        wf_name_GQ = 'wave_ch{}_cw{:03}'.format(self.channel_GQ(), wave_id)
        wf_name_DI = 'wave_ch{}_cw{:03}'.format(self.channel_DI(), wave_id)
        wf_name_DQ = 'wave_ch{}_cw{:03}'.format(self.channel_DQ(), wave_id)

        self.AWG.get_instr().set(wf_name_GI, GI)
        self.AWG.get_instr().set(wf_name_GQ, GQ)
        self.AWG.get_instr().set(wf_name_DI, DI)
        self.AWG.get_instr().set(wf_name_DQ, DQ)

    ##########################################################################
    # AWG8_MW_LutMan overrides
    ##########################################################################

    def _set_channel_amp(self, val):
        AWG = self.AWG.get_instr()
        for awg_ch in [self.channel_GI(), self.channel_GQ(),
                       self.channel_DI(), self.channel_DQ()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            AWG.set('awgs_{}_outputs_{}_amplitude'.format(awg_nr, ch_pair), val)  # FIXME: awgs_{}_outputs_{}_amplitude superceeded by awgs_{}_outputs_{}_

    def _get_channel_amp(self):
        AWG = self.AWG.get_instr()
        vals = []
        for awg_ch in [self.channel_GI(), self.channel_GQ(),
                       self.channel_DI(), self.channel_DQ()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            vals.append(
                AWG.get('awgs_{}_outputs_{}_amplitude'.format(awg_nr, ch_pair)))
        assert vals[0] == vals[1] == vals[2] == vals[3]
        return vals[0]

    def _add_mixer_corr_pars(self):
        self.add_parameter(
            'G_mixer_alpha',
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
            initial_value=1.0
        )
        self.add_parameter(
            'G_mixer_phi',
            vals=vals.Numbers(),
            unit='deg',
            parameter_class=ManualParameter,
            initial_value=0.0
        )
        self.add_parameter(
            'D_mixer_alpha',
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
            initial_value=1.0
        )
        self.add_parameter(
            'D_mixer_phi',
            vals=vals.Numbers(),
            unit='deg',
            parameter_class=ManualParameter,
            initial_value=0.0
        )

        self.add_parameter(
            'mixer_apply_predistortion_matrix',
            vals=vals.Bool(),
            docstring=(
                'If True applies a mixer correction using mixer_phi and '
                'mixer_alpha to all microwave pulses using.'),
            parameter_class=ManualParameter,
            initial_value=True
        )

    def apply_mixer_predistortion_corrections(self, wave_dict):

        M_G = wf.mixer_predistortion_matrix(
            self.G_mixer_alpha(),
            self.G_mixer_phi()
        )
        M_D = wf.mixer_predistortion_matrix(
            self.D_mixer_alpha(),
            self.D_mixer_phi()
        )

        for key, val in wave_dict.items():
            GI, GQ = np.dot(M_G, val[0:2])  # Mixer correction Gaussian comp.
            DI, DQ = np.dot(M_D, val[2:4])  # Mixer correction Derivative comp.
            wave_dict[key] = GI, GQ, DI, DQ
        return wave_dict

class QWG_MW_LutMan_VQE(QWG_MW_LutMan):
    def __init__(self, name, **kw):
        """
        Waveform allocation strategy for VQE.

        |q0> ---- Y ----x---- T1' --- M
                        |
        |q1> -----------x---- T2 ---- M


        Waveform table for |q0>
        +------+------+---------------+-------------+--------------------+
        | Wave | Gate | Amp,phase     | Used in VQE | Used for full-tomo |
        +------+------+---------------+-------------+--------------------+
        | 0    |  I   | 0,0           |      X      |                    |
        | 1    |  X'  | pi,phi        |      X      |                    |
        | 2    |  Y'  | pi,phi        |             |          X         |
        | 3    |  x'  | pi/2,phi      |      X      |                    |
        | 4    | -x'  | -pi/2,phi     |      X      |                    |
        | 5    |  y'  | pi/2,phi+90   |      X      |                    |
        | 6    | -y'  | -pi/2,phi+90  |      X      |                    |
        | 7    |  Y   | pi,90         |      X      |                    |
        | 8    |      |               |             |                    |
        +------+------+---------------+-------------+--------------------+

        # NOTE: prime stands for gate compiled along with Z(phi).

        Waveform table for |q1>
        +------+------+---------------+-------------+--------------------+
        | Wave | Gate | Amp,phase     | Used in VQE | Used for full-tomo |
        +------+------+---------------+-------------+--------------------+
        | 0    |  I   | 0,0           |      X      |                    |
        | 1    |  X   | pi,0          |      X      |                    |
        | 2    |  Y   | pi,0          |             |          X         |
        | 3    |  x   | pi/2,0        |      X      |                    |
        | 4    | -x   | -pi/2,0       |      X      |                    |
        | 5    |  y   | pi/2,90       |      X      |                    |
        | 6    | -y   | -pi/2,90      |      X      |                    |
        | 7    |      |               |             |                    |
        | 8    |      |               |             |                    |
        +------+------+---------------+-------------+--------------------+
        """
        self._num_channels = 4
        super().__init__(name, **kw)

        # sacrifices last pulse 'Spec' from std list to have 3 bit (8)
        self._def_lm = ['I', 'rX180',  'rY180', 'rX90',
                        'rY90', 'rXm90',  'rYm90', 'rPhi90']
        self.set_default_lutmap()

        self._vqe_lm = ['I', 'X180c',  'Y180c',
                        'X90c',  'Xm90c', 'Y90c',  'Y90c', 'rY180']

    ##########################################################################
    # Base_LutMan overrides
    ##########################################################################

    def _add_waveform_parameters(self):
        super()._add_waveform_parameters()
        # parameters related to codeword bits
        self.add_parameter(
            'bit_shift',
            unit='',
            vals=vals.Ints(0, 4),
            parameter_class=ManualParameter,
            initial_value=0
        )
        self.add_parameter(
            'bit_width',
            unit='', vals=vals.Ints(0, 4),
            parameter_class=ManualParameter,
            initial_value=0
        )
        # parameters related to phase compilation
        self.add_parameter(
            'phi',
            unit='rad',  # FIXME: does not match vals values
            vals=vals.Numbers(0, 360),
            parameter_class=ManualParameter,
            initial_value=0
        )
        # parameters related to timings
        self.add_parameter(
            'pulse_delay',
            unit='s',
            vals=vals.Numbers(0, 1e-6),
            parameter_class=ManualParameter,
            initial_value=0
        )

    def generate_standard_waveforms(self):
        self._wave_dict = {}
        if self.cfg_sideband_mode() == 'static':
            f_modulation = self.mw_modulation()
        else:
            f_modulation = 0
            self.AWG.get_instr().set('ch_pair{}_sideband_frequency'.format(self.channel_I()), self.mw_modulation())
            self.AWG.get_instr().syncSidebandGenerators()

        ########################################
        # STD waveforms
        ########################################
        # FIXME: this creates _wave_dict, independent of LutMap
        self._wave_dict['I'] = self.wf_func(
            amp=0, sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=0, delay=self.pulse_delay())
        self._wave_dict['rX180'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rY180'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rX90'] = self.wf_func(
            amp=self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rY90'] = self.wf_func(
            amp=self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rXm90'] = self.wf_func(
            amp=-1*self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rYm90'] = self.wf_func(
            amp=-1*self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rPhi180'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.mw_phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rPhi90'] = self.wf_func(
            amp=self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.mw_phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rPhim90'] = self.wf_func(
            amp=-1*self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.mw_phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['spec'] = self.spec_func(
            amp=self.spec_amp(),
            length=self.spec_length(),
            sampling_rate=self.sampling_rate(),
            delay=0,
            phase=0)

        for i in range(18):
            angle = i * 20
            self._wave_dict['r{}_90'.format(angle)] = self.wf_func(
                amp=self.mw_amp180()*self.mw_amp90_scale(),
                sigma_length=self.mw_gauss_width(),
                f_modulation=f_modulation,
                sampling_rate=self.sampling_rate(), phase=angle,
                motzoi=self.mw_motzoi(), delay=self.pulse_delay())

        ########################################
        # compiled waveforms
        ########################################
        self._wave_dict['X180c'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rY180'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rY180c'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90+self.phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rX90c'] = self.wf_func(
            amp=self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rY90c'] = self.wf_func(
            amp=self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90+self.phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rXm90c'] = self.wf_func(
            amp=-1*self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())
        self._wave_dict['rYm90c'] = self.wf_func(
            amp=-1*self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90+self.phi(),
            motzoi=self.mw_motzoi(), delay=self.pulse_delay())

        if self.mixer_apply_predistortion_matrix():
            self._wave_dict = self.apply_mixer_predistortion_corrections(self._wave_dict)
        return self._wave_dict

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str, regenerate_waveforms: bool=False):
        if regenerate_waveforms:
            self.generate_standard_waveforms()
        # waveform_name is pulse string (i.e. 'X180')
        # waveforms is the tuple (G,D) with the actual pulseshape array
        # codewords is the tuple (A,B) with the strings wave_chY_cwXXX
        # codeword_int is the number XXX (see line above)
        waveforms = self._wave_dict[waveform_name]
        codewords = self.LutMap()[waveform_name]
        codeword_int = int(codewords[0][-3:])
        # get redundant codewords as a list
        redundant_cw_list = get_redundant_codewords(codeword_int,
                                                    bit_width=self.bit_width(),
                                                    bit_shift=self.bit_shift())
        # update all of them
        for redundant_cw_idx in redundant_cw_list:
            redundant_cw_I = 'wave_ch{}_cw{:03}'.format(self.channel_I(), redundant_cw_idx)
            self.AWG.get_instr().set(redundant_cw_I, waveforms[0])
            redundant_cw_Q = 'wave_ch{}_cw{:03}'.format(self.channel_Q(), redundant_cw_idx)
            self.AWG.get_instr().set(redundant_cw_Q, waveforms[1])

    ##########################################################################
    # Functions
    # FIXME: these provide an undesired backdoor
    ##########################################################################

    def set_VQE_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        vqe_lm = self._vqe_lm
        LutMap = {}
        for cw_idx, cw_key in enumerate(vqe_lm):
            LutMap[cw_key] = (
                'wave_ch{}_cw{:03}'.format(self.channel_I(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_Q(), cw_idx))
        self.LutMap(LutMap)

# Not the cleanest inheritance but whatever - MAR Nov 2017
class QWG_VSM_MW_LutMan(AWG8_VSM_MW_LutMan):

    ##########################################################################
    # Base_LutMan overrides
    ##########################################################################

    def load_waveforms_onto_AWG_lookuptable(self, regenerate_waveforms: bool=True,  stop_start: bool = True):
        AWG = self.AWG.get_instr()
        if self.cfg_sideband_mode() == 'real-time':
            AWG.ch_pair1_sideband_frequency(self.mw_modulation())
            AWG.ch_pair3_sideband_frequency(self.mw_modulation())
        else:
            AWG.ch_pair1_sideband_frequency(0)
            AWG.ch_pair3_sideband_frequency(0)

        for ch in range(1, 5):
            # ensures amplitude specified is in
            AWG.set('ch{}_amp'.format(ch), 1)
        return Base_MW_LutMan.load_waveforms_onto_AWG_lookuptable(
            self=self,
            regenerate_waveforms=regenerate_waveforms, stop_start=stop_start)

    def _add_channel_params(self):
        # all channels are used and hardcoded in functionality
        pass

    # def set_default_lutmap(self):
    #     """
    #     Set's the default lutmap for standard microwave drive pulses.
    #     """
    #     def_lm = self._def_lm
    #     LutMap = OrderedDict()
    #     for cw_idx, cw_key in enumerate(def_lm):
    #         LutMap[cw_key] = (
    #             'wave_ch1_cw{:03}'.format(cw_idx),
    #             'wave_ch2_cw{:03}'.format(cw_idx),
    #             'wave_ch3_cw{:03}'.format(cw_idx),
    #             'wave_ch4_cw{:03}'.format(cw_idx))
    #     self.LutMap(LutMap)
