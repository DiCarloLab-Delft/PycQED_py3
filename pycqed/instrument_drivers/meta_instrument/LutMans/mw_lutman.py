from .base_lutman import Base_LutMan, get_redundant_codewords, get_wf_idx_from_name
import numpy as np
from collections.abc import Iterable
from collections import OrderedDict
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from pycqed.measurement.waveform_control_CC import waveform as wf
import time

default_mw_lutmap = {
    0  : {"name" : "I"     , "theta" : 0        , "phi" : 0 , "type" : "ge"},
    1  : {"name" : "rX180" , "theta" : 180      , "phi" : 0 , "type" : "ge"},
    2  : {"name" : "rY180" , "theta" : 180      , "phi" : 90, "type" : "ge"},
    3  : {"name" : "rX90"  , "theta" : 90       , "phi" : 0 , "type" : "ge"},
    4  : {"name" : "rY90"  , "theta" : 90       , "phi" : 90, "type" : "ge"},
    5  : {"name" : "rXm90" , "theta" : -90      , "phi" : 0 , "type" : "ge"},
    6  : {"name" : "rYm90" , "theta" : -90      , "phi" : 90, "type" : "ge"},
    7  : {"name" : "rPhi90", "theta" : 90       , "phi" : 0 , "type" : "ge"},
    8  : {"name" : "spec"  , "type"  : "spec"}  ,
    9  : {"name" : "rX12"  , "theta" : 180      , "phi" : 0 , "type" : "ef"},
    10 : {"name" : "square", "type"  : "square"},
    11 : {"name" : "rY45"  , "theta" : 45       , "phi" : 90, "type" : "ge"},
    12 : {"name" : "rYm45" , "theta" : -45      , "phi" : 90, "type" : "ge"},
    13 : {"name" : "rX45"  , "theta" : 45       , "phi" : 0 , "type" : "ge"},
    14 : {"name" : "rXm45" , "theta" : -45      , "phi" : 0 , "type" : "ge"}
}

inspire_mw_lutmap = {
    0  : {"name" : "I"     , "theta" : 0        , "phi" : 0  , "type" : "ge"}, # I for CW compatibility
    1  : {"name" : "rX180" , "theta" : 180      , "phi" : 0  , "type" : "ge"}, # rX180 for CW compatibility
    2  : {"name" : "rY180" , "theta" : 180      , "phi" : 90 , "type" : "ge"}, # rY180 for CW compatibility
    3  : {"name" : "rX90"  , "theta" : 90       , "phi" : 0  , "type" : "ge"}, # rX90 for CW compatibility
    4  : {"name" : "rY90"  , "theta" : 90       , "phi" : 90 , "type" : "ge"}, # rY90 for CW compatibility
    5  : {"name" : "rX270" , "theta" : 270      , "phi" : 0  , "type" : "ge"}, # rXm90 for CW compatibility
    6  : {"name" : "rY270" , "theta" : 270      , "phi" : 90 , "type" : "ge"}, # rYm90 for CW compatibility
    7  : {"name" : "rX5"   , "theta" : 5.625    , "phi" : 0  , "type" : "ge"},
    8  : {"name" : "rX11"  , "theta" : 11.25    , "phi" : 0  , "type" : "ge"},
    9  : {"name" : "rX12"  , "theta" : 180      , "phi" : 0  , "type" : "ef"}, # rX12 for CW compatibility
    10 : {"name" : "rX16"  , "theta" : 16.875   , "phi" : 0  , "type" : "ge"},
    11 : {"name" : "rY45"  , "theta" : 45       , "phi" : 90 , "type" : "ge"}, # rY45 for CW compatibility
    12 : {"name" : "rY315" , "theta" : -45      , "phi" : 90 , "type" : "ge"}, # rYm45 for CW compatibility
    13 : {"name" : "rX45"  , "theta" : 45       , "phi" : 0  , "type" : "ge"}, # rX45 for CW compatibility
    14 : {"name" : "rX315" , "theta" : -45      , "phi" : 0  , "type" : "ge"}, # rXm45 for CW compatibility
    15 : {"name" : "rX22"  , "theta" : 22.5     , "phi" : 0  , "type" : "ge"},
    16 : {"name" : "rX28"  , "theta" : 28.125   , "phi" : 0  , "type" : "ge"},
    17 : {"name" : "rX33"  , "theta" : 33.75    , "phi" : 0  , "type" : "ge"},
    18 : {"name" : "rX39"  , "theta" : 39.375   , "phi" : 0  , "type" : "ge"},
    19 : {"name" : "rX50"  , "theta" : 50.625   , "phi" : 0  , "type" : "ge"},
    20 : {"name" : "rX56"  , "theta" : 56.25    , "phi" : 0  , "type" : "ge"},
    21 : {"name" : "rX61"  , "theta" : 61.875   , "phi" : 0  , "type" : "ge"},
    22 : {"name" : "rX67"  , "theta" : 67.5     , "phi" : 0  , "type" : "ge"},
    23 : {"name" : "rX73"  , "theta" : 73.125   , "phi" : 0  , "type" : "ge"},
    24 : {"name" : "rX78"  , "theta" : 78.75    , "phi" : 0  , "type" : "ge"},
    25 : {"name" : "rX84"  , "theta" : 84.375   , "phi" : 0  , "type" : "ge"},
    26 : {"name" : "rX95"  , "theta" : 95.625   , "phi" : 0  , "type" : "ge"},
    27 : {"name" : "rX101" , "theta" : 101.25   , "phi" : 0  , "type" : "ge"},
    28 : {"name" : "rX106" , "theta" : 106.875  , "phi" : 0  , "type" : "ge"},
    29 : {"name" : "rX112" , "theta" : 112.5    , "phi" : 0  , "type" : "ge"},
    30 : {"name" : "rX118" , "theta" : 118.125  , "phi" : 0  , "type" : "ge"},
    31 : {"name" : "rX123" , "theta" : 123.75   , "phi" : 0  , "type" : "ge"},
    32 : {"name" : "rX129" , "theta" : 129.375  , "phi" : 0  , "type" : "ge"},
    33 : {"name" : "rX135" , "theta" : 135      , "phi" : 0  , "type" : "ge"},
    34 : {"name" : "rX140" , "theta" : 140.625  , "phi" : 0  , "type" : "ge"},
    35 : {"name" : "rX146" , "theta" : 146.25   , "phi" : 0  , "type" : "ge"},
    36 : {"name" : "rX151" , "theta" : 151.875  , "phi" : 0  , "type" : "ge"},
    37 : {"name" : "rX157" , "theta" : 157.5    , "phi" : 0  , "type" : "ge"},
    38 : {"name" : "rX163" , "theta" : 163.125  , "phi" : 0  , "type" : "ge"},
    39 : {"name" : "rX168" , "theta" : 168.75   , "phi" : 0  , "type" : "ge"},
    40 : {"name" : "rX174" , "theta" : 174.375  , "phi" : 0  , "type" : "ge"},
    41 : {"name" : "rX185" , "theta" : -174.375 , "phi" : 0  , "type" : "ge"},
    42 : {"name" : "rX191" , "theta" : -168.75  , "phi" : 0  , "type" : "ge"},
    43 : {"name" : "rX196" , "theta" : -163.125 , "phi" : 0  , "type" : "ge"},
    44 : {"name" : "rX202" , "theta" : -157.5   , "phi" : 0  , "type" : "ge"},
    45 : {"name" : "rX208" , "theta" : -151.875 , "phi" : 0  , "type" : "ge"},
    46 : {"name" : "rX213" , "theta" : -146.25  , "phi" : 0  , "type" : "ge"},
    47 : {"name" : "rX219" , "theta" : -140.625 , "phi" : 0  , "type" : "ge"},
    48 : {"name" : "rX225" , "theta" : -135     , "phi" : 0  , "type" : "ge"},
    49 : {"name" : "rX230" , "theta" : -129.375 , "phi" : 0  , "type" : "ge"},
    50 : {"name" : "rX236" , "theta" : -123.75  , "phi" : 0  , "type" : "ge"},
    51 : {"name" : "rX241" , "theta" : -118.125 , "phi" : 0  , "type" : "ge"},
    52 : {"name" : "rX247" , "theta" : -112.5   , "phi" : 0  , "type" : "ge"},
    53 : {"name" : "rX253" , "theta" : -106.875 , "phi" : 0  , "type" : "ge"},
    54 : {"name" : "rX258" , "theta" : -101.25  , "phi" : 0  , "type" : "ge"},
    55 : {"name" : "rX264" , "theta" : -95.625  , "phi" : 0  , "type" : "ge"},
    56 : {"name" : "rX275" , "theta" : -84.375  , "phi" : 0  , "type" : "ge"},
    57 : {"name" : "rX281" , "theta" : -78.75   , "phi" : 0  , "type" : "ge"},
    58 : {"name" : "rX286" , "theta" : -73.125  , "phi" : 0  , "type" : "ge"},
    59 : {"name" : "rX292" , "theta" : -67.5    , "phi" : 0  , "type" : "ge"},
    60 : {"name" : "rX298" , "theta" : -61.875  , "phi" : 0  , "type" : "ge"},
    61 : {"name" : "rX303" , "theta" : -56.25   , "phi" : 0  , "type" : "ge"},
    62 : {"name" : "rX309" , "theta" : -50.625  , "phi" : 0  , "type" : "ge"},
    63 : {"name" : "rX320" , "theta" : -39.375  , "phi" : 0  , "type" : "ge"},
    64 : {"name" : "rX326" , "theta" : -33.75   , "phi" : 0  , "type" : "ge"},
    65 : {"name" : "rX331" , "theta" : -28.125  , "phi" : 0  , "type" : "ge"},
    66 : {"name" : "rX337" , "theta" : -22.5    , "phi" : 0  , "type" : "ge"},
    67 : {"name" : "rX343" , "theta" : -16.875  , "phi" : 0  , "type" : "ge"},
    68 : {"name" : "rX348" , "theta" : -11.25   , "phi" : 0  , "type" : "ge"},
    69 : {"name" : "rX354" , "theta" : -5.625   , "phi" : 0  , "type" : "ge"},
    70 : {"name" : "rY5"   , "theta" : 5.625    , "phi" : 90 , "type" : "ge"},
    71 : {"name" : "rY11"  , "theta" : 11.25    , "phi" : 90 , "type" : "ge"},
    72 : {"name" : "rY16"  , "theta" : 16.875   , "phi" : 90 , "type" : "ge"},
    73 : {"name" : "rY22"  , "theta" : 22.5     , "phi" : 90 , "type" : "ge"},
    74 : {"name" : "rY28"  , "theta" : 28.125   , "phi" : 90 , "type" : "ge"},
    75 : {"name" : "rY33"  , "theta" : 33.75    , "phi" : 90 , "type" : "ge"},
    76 : {"name" : "rY39"  , "theta" : 39.375   , "phi" : 90 , "type" : "ge"},
    77 : {"name" : "rY50"  , "theta" : 50.625   , "phi" : 90 , "type" : "ge"},
    78 : {"name" : "rY56"  , "theta" : 56.25    , "phi" : 90 , "type" : "ge"},
    79 : {"name" : "rY61"  , "theta" : 61.875   , "phi" : 90 , "type" : "ge"},
    80 : {"name" : "rY67"  , "theta" : 67.5     , "phi" : 90 , "type" : "ge"},
    81 : {"name" : "rY73"  , "theta" : 73.125   , "phi" : 90 , "type" : "ge"},
    82 : {"name" : "rY78"  , "theta" : 78.75    , "phi" : 90 , "type" : "ge"},
    83 : {"name" : "rY84"  , "theta" : 84.375   , "phi" : 90 , "type" : "ge"},
    84 : {"name" : "rY95"  , "theta" : 95.625   , "phi" : 90 , "type" : "ge"},
    85 : {"name" : "rY101" , "theta" : 101.25   , "phi" : 90 , "type" : "ge"},
    86 : {"name" : "rY106" , "theta" : 106.875  , "phi" : 90 , "type" : "ge"},
    87 : {"name" : "rY112" , "theta" : 112.5    , "phi" : 90 , "type" : "ge"},
    88 : {"name" : "rY118" , "theta" : 118.125  , "phi" : 90 , "type" : "ge"},
    89 : {"name" : "rY123" , "theta" : 123.75   , "phi" : 90 , "type" : "ge"},
    90 : {"name" : "rY129" , "theta" : 129.375  , "phi" : 90 , "type" : "ge"},
    91 : {"name" : "rY135" , "theta" : 135      , "phi" : 90 , "type" : "ge"},
    92 : {"name" : "rY140" , "theta" : 140.625  , "phi" : 90 , "type" : "ge"},
    93 : {"name" : "rY146" , "theta" : 146.25   , "phi" : 90 , "type" : "ge"},
    94 : {"name" : "rY151" , "theta" : 151.875  , "phi" : 90 , "type" : "ge"},
    95 : {"name" : "rY157" , "theta" : 157.5    , "phi" : 90 , "type" : "ge"},
    96 : {"name" : "rY163" , "theta" : 163.125  , "phi" : 90 , "type" : "ge"},
    97 : {"name" : "rY168" , "theta" : 168.75   , "phi" : 90 , "type" : "ge"},
    98 : {"name" : "rY174" , "theta" : 174.375  , "phi" : 90 , "type" : "ge"},
    99 : {"name" : "rY185" , "theta" : -174.375 , "phi" : 90 , "type" : "ge"},
    100: {"name" : "rY191" , "theta" : -168.75  , "phi" : 90 , "type" : "ge"},
    101: {"name" : "rY196" , "theta" : -163.125 , "phi" : 90 , "type" : "ge"},
    102: {"name" : "rY202" , "theta" : -157.5   , "phi" : 90 , "type" : "ge"},
    103: {"name" : "rY208" , "theta" : -151.875 , "phi" : 90 , "type" : "ge"},
    104: {"name" : "rY213" , "theta" : -146.25  , "phi" : 90 , "type" : "ge"},
    105: {"name" : "rY219" , "theta" : -140.625 , "phi" : 90 , "type" : "ge"},
    106: {"name" : "rY225" , "theta" : -135     , "phi" : 90 , "type" : "ge"},
    107: {"name" : "rY230" , "theta" : -129.375 , "phi" : 90 , "type" : "ge"},
    108: {"name" : "rY236" , "theta" : -123.75  , "phi" : 90 , "type" : "ge"},
    109: {"name" : "rY241" , "theta" : -118.125 , "phi" : 90 , "type" : "ge"},
    110: {"name" : "rY247" , "theta" : -112.5   , "phi" : 90 , "type" : "ge"},
    111: {"name" : "rY253" , "theta" : -106.875 , "phi" : 90 , "type" : "ge"},
    112: {"name" : "rY258" , "theta" : -101.25  , "phi" : 90 , "type" : "ge"},
    113: {"name" : "rY264" , "theta" : -95.625  , "phi" : 90 , "type" : "ge"},
    114: {"name" : "rY275" , "theta" : -84.375  , "phi" : 90 , "type" : "ge"},
    115: {"name" : "rY281" , "theta" : -78.75   , "phi" : 90 , "type" : "ge"},
    116: {"name" : "rY286" , "theta" : -73.125  , "phi" : 90 , "type" : "ge"},
    117: {"name" : "rY292" , "theta" : -67.5    , "phi" : 90 , "type" : "ge"},
    118: {"name" : "rY298" , "theta" : -61.875  , "phi" : 90 , "type" : "ge"},
    119: {"name" : "rY303" , "theta" : -56.25   , "phi" : 90 , "type" : "ge"},
    120: {"name" : "rY309" , "theta" : -50.625  , "phi" : 90 , "type" : "ge"},
    121: {"name" : "rY320" , "theta" : -39.375  , "phi" : 90 , "type" : "ge"},
    122: {"name" : "rY326" , "theta" : -33.75   , "phi" : 90 , "type" : "ge"},
    123: {"name" : "rY331" , "theta" : -28.125  , "phi" : 90 , "type" : "ge"},
    124: {"name" : "rY337" , "theta" : -22.5    , "phi" : 90 , "type" : "ge"},
    125: {"name" : "rY343" , "theta" : -16.875  , "phi" : 90 , "type" : "ge"},
    126: {"name" : "rY348" , "theta" : -11.25   , "phi" : 90 , "type" : "ge"},
    127: {"name" : "rY354" , "theta" : -5.625   , "phi" : 90 , "type" : "ge"}
}

valid_types = {'ge', 'ef', 'spec', 'raw-drag', 'ef-raw', 'square'}

# _def_lm = ['I', 'rX180',  'rY180', 'rX90',  'rY90',
#            'rXm90',  'rYm90', 'rPhi90', 'spec']
# # use remaining codewords to set pi/2 gates for various angles
# for i in range(18):
#     angle = i * 20
#     _def_lm.append('r{}_90'.format(angle))


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

    def set_default_lutmap(self):
        """Set the default lutmap for standard microwave drive pulses."""
        self.LutMap(default_mw_lutmap.copy())

    def set_inspire_lutmap(self):
        """Set the default lutmap for standard microwave drive pulses."""
        self.LutMap(inspire_mw_lutmap.copy())

    def codeword_idx_to_parnames(self, cw_idx: int):
        """Convert a codeword_idx to a list of par names for the waveform."""
        # the possible channels way of doing this is to make it work both for
        # VSM style lutmans and no VSM style lutmans.
        possible_channels = ('channel_GI', 'channel_GQ',
                             'channel_DI', 'channel_DQ',
                             'channel_I', 'channel_Q')
        codewords = ['wave_ch{}_cw{:03}'.format(self[ch](), cw_idx)
                     for ch in possible_channels if hasattr(self, ch)]
        return codewords

    def _add_waveform_parameters(self):
        # defined here so that the VSM based LutMan can overwrite this
        self.wf_func = wf.mod_gauss
        self.spec_func = wf.block_pulse

        self._add_channel_params()
        self.add_parameter('cfg_sideband_mode',
                           vals=vals.Enum('real-time', 'static'),
                           initial_value='static',
                           parameter_class=ManualParameter)
        self.add_parameter('mw_amp180', unit='frac', vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.1)
        self.add_parameter('mw_amp90_scale',
                           vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.5)
        self.add_parameter('mw_motzoi', vals=vals.Numbers(-2, 2),
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('mw_gauss_width',
                           vals=vals.Numbers(min_value=1e-9), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=4e-9)
        self.add_parameter('mw_phi', label='Phase of Rphi pulse',
                           vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0)

        self.add_parameter('spec_length',
                           vals=vals.Numbers(), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=20e-9)
        self.add_parameter('spec_amp',
                           vals=vals.Numbers(), unit='frac',
                           parameter_class=ManualParameter,
                           initial_value=1)
        # parameters related to timings
        self.add_parameter('pulse_delay', unit='s', vals=vals.Numbers(0, 1e-6),
                           parameter_class=ManualParameter,
                           initial_value=0)

        self.add_parameter(
            'mw_modulation', vals=vals.Numbers(), unit='Hz',
            docstring=('Modulation frequency for qubit driving pulses. Note'
                       ' that when using an AWG with build in modulation this'
                       ' should be set to 0.'),
            parameter_class=ManualParameter, initial_value=50.0e6)
        self._add_mixer_corr_pars()

        self.add_parameter('mw_ef_modulation', vals=vals.Numbers(), unit='Hz',
                           docstring=('Modulation frequency for driving pulses to the '
                                      'second excited-state.'),
                           parameter_class=ManualParameter, initial_value=50.0e6)
        self.add_parameter('mw_ef_amp180', unit='frac',
                           docstring=(
                               'Pulse amplitude for pulsing the ef/12 transition'),
                           vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter, initial_value=.2)

    def _add_mixer_corr_pars(self):
        self.add_parameter('mixer_alpha', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('mixer_phi', vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter(
            'mixer_apply_predistortion_matrix', vals=vals.Bool(), docstring=(
                'If True applies a mixer correction using mixer_phi and '
                'mixer_alpha to all microwave pulses using.'),
            parameter_class=ManualParameter, initial_value=True)

    def _add_channel_params(self):
        self.add_parameter('channel_I',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, self._num_channels))

        self.add_parameter('channel_Q',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, self._num_channels))

    def generate_standard_waveforms(
            self, apply_predistortion_matrix: bool=True):
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
                if 'sq_G_amp' in self.parameters:
                    self._wave_dict[idx] = wf.mod_square_VSM(
                        amp_G=self.sq_G_amp(), amp_D=self.sq_D_amp(),
                        length=self.mw_gauss_width()*4,
                        f_modulation=self.mw_modulation(),
                        sampling_rate=self.sampling_rate())
                elif 'sq_amp' in self.parameters:
                    self._wave_dict[idx] = wf.mod_square(
                        amp=self.sq_amp(), length=self.mw_gauss_width()*4,
                        f_modulation=self.mw_modulation(),  phase=0,
                        motzoi=0, sampling_rate=self.sampling_rate())
                else:
                    raise KeyError('Expected parameter "sq_amp" to exist')
            else:
                raise ValueError

        # Add predistortions + test
        if (self.mixer_apply_predistortion_matrix()
                and apply_predistortion_matrix):
            self._wave_dict = self.apply_mixer_predistortion_corrections(
                self._wave_dict)
        return self._wave_dict

    def apply_mixer_predistortion_corrections(self, wave_dict):
        M = wf.mixer_predistortion_matrix(self.mixer_alpha(),
                                          self.mixer_phi())
        for key, val in wave_dict.items():
            wave_dict[key] = np.dot(M, val)
        return wave_dict

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
        if regenerate_waveforms:
            self.generate_standard_waveforms()

        # FIXME: type mismatch with function parameter, misleading name
        if isinstance(waveform_name, int):
            cw_idx = waveform_name
        else:
            raise DeprecationWarning

        waveforms = self._wave_dict[cw_idx]
        codewords = self.codeword_idx_to_parnames(cw_idx)

        for waveform, cw in zip(waveforms, codewords):
            self.AWG.get_instr().set(cw, waveform)

    def load_phase_pulses_to_AWG_lookuptable(self,
                                             phases=np.arange(0, 360, 20)):
        """
        Loads rPhi90 pulses onto the AWG lookuptable.
        """

        if (len(phases) > 18):
            raise ValueError('max 18 amplitude values can be provided')
        lm = self.LutMap()
        for i, (phase) in enumerate(phases):
            lm[i+9] = {"name": "rPhi90",    "theta": 90,
                       "phi": phase, "type": "ge"}
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
        lm = self.LutMap()
        for i, (amp, mod_freq) in enumerate(zip(amps, mod_freqs)):
            lm[i+9] = {"name": "", "type": "raw-drag",
                       "drag_pars": {
                           "amp": amp, "f_modulation": mod_freq,
                           "sigma_length": self.mw_gauss_width(),
                           "sampling_rate": self.sampling_rate(),
                           "motzoi": 0}
                       }

        # 3. generate and upload waveforms
        self.load_waveforms_onto_AWG_lookuptable(regenerate_waveforms=True)


class CBox_MW_LutMan(Base_MW_LutMan):
    _def_lm = ['I', 'rX180',  'rY180', 'rX90',  'rY90',
               'rXm90',  'rYm90', 'rPhi90', 'spec']
    # use remaining codewords to set pi/2 gates for various angles
    for i in range(18):
        angle = i * 20
        _def_lm.append('r{}_90'.format(angle))

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

    def _add_channel_params(self):
        # CBox channels come in pairs defined in the AWG nr
        self.add_parameter('awg_nr', parameter_class=ManualParameter,
                           initial_value=0, vals=vals.Numbers(0, 2))

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
        if regenerate_waveforms:
            self.generate_standard_waveforms()
        I_wave, Q_wave = self._wave_dict[waveform_name]
        codeword = self.LutMap()[waveform_name]

        self.AWG.get_instr().set_awg_lookuptable(self.awg_nr(),
                                                 codeword, 0, I_wave)
        self.AWG.get_instr().set_awg_lookuptable(self.awg_nr(),
                                                 codeword, 1, Q_wave)

    def set_default_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        def_lm = self._def_lm
        LutMap = OrderedDict()
        for cw_idx, cw_key in enumerate(def_lm):
            max_cw_cbox = 8
            if cw_idx < max_cw_cbox:
                LutMap[cw_key] = cw_idx
        self.LutMap(LutMap)


class QWG_MW_LutMan(Base_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 4
        super().__init__(name, **kw)

    def _add_channel_params(self):
        super()._add_channel_params()
        self.add_parameter('channel_amp',
                           unit='a.u.',
                           vals=vals.Numbers(-1.8, 1.8),
                           set_cmd=self._set_channel_amp,
                           get_cmd=self._get_channel_amp,
                           docstring=('using the channel amp as additional'
                                      'parameter to allow rabi-type experiments without'
                                      'wave reloading. Should not be using VSM'))
        # parameters related to codeword bits
        self.add_parameter('bit_shift', unit='', vals=vals.Ints(0, 8),
                           parameter_class=ManualParameter,
                           initial_value=0)
        self.add_parameter('bit_width', unit='', vals=vals.Ints(0, 8),
                           parameter_class=ManualParameter,
                           initial_value=0)

    def _add_waveform_parameters(self):
        super()._add_waveform_parameters()
        # Parameters for a square pulse
        self.add_parameter('sq_amp',
                           unit='frac', vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.5)

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


class AWG8_MW_LutMan(Base_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 8
        super().__init__(name, **kw)
        self.sampling_rate(2.4e9)

    def _add_channel_params(self):
        super()._add_channel_params()
        self.add_parameter(
            'channel_amp', unit='a.u.', vals=vals.Numbers(0, 1),
            set_cmd=self._set_channel_amp, get_cmd=self._get_channel_amp,
            docstring=('using the channel amp as additional'
                       'parameter to allow rabi-type experiments without'
                       'wave reloading. Should not be using VSM'))


    def _add_waveform_parameters(self):
        super()._add_waveform_parameters()
        # Parameters for a square pulse
        self.add_parameter('sq_amp', unit='frac', vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.5)   

    def _set_channel_amp(self, val):
        AWG = self.AWG.get_instr()
        for awg_ch in [self.channel_I(), self.channel_Q()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            AWG.set('awgs_{}_outputs_{}_amplitude'.format(awg_nr, ch_pair), val)

    def _get_channel_amp(self):
        AWG = self.AWG.get_instr()
        vals = []
        for awg_ch in [self.channel_I(), self.channel_Q()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            vals.append(
                AWG.get('awgs_{}_outputs_{}_amplitude'.format(awg_nr, ch_pair)))
        print(vals)
        assert vals[0] == vals[1]
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
            self, regenerate_waveforms: bool=True, stop_start: bool = True,
            force_load_sequencer_program: bool=False):
        """
        Loads all waveforms specified in the LutMap to an AWG.

        Args:
            regenerate_waveforms (bool): if True calls
                generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.
            force_load_sequencer_program (bool): if True forces a new compilation
                and upload of the program on the sequencer.
        """
        # Uploading the codeword program (again) is needed to link the new
        # waveforms in case the user has changed the codeword mode.
        if force_load_sequencer_program:
            # This ensures only the channels that are relevant get reconfigured
            if 'channel_GI' in self.parameters:
                awgs = [self.channel_GI()//2, self.channel_DI()//2]
            else:
                awgs = [self.channel_I()//2]

            self.AWG.get_instr().upload_codeword_program(awgs=awgs)

        super().load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=regenerate_waveforms,
            stop_start=stop_start)

class AWG8_VSM_MW_LutMan(AWG8_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 8
        super().__init__(name, **kw)
        self.wf_func = wf.mod_gauss_VSM
        self.spec_func = wf.block_pulse_vsm

    def _add_waveform_parameters(self):
        super()._add_waveform_parameters()
        # Base_MW_LutMan._add_waveform_parameters(self)
        # Parameters for a square pulse
        self.add_parameter('sq_G_amp', unit='frac', vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.5)
        self.add_parameter('sq_D_amp', unit='frac', vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0)

    def _add_channel_params(self):
        self.add_parameter(
            'channel_amp', unit='a.u.', vals=vals.Numbers(0, 1),
            set_cmd=self._set_channel_amp,
            get_cmd=self._get_channel_amp,
            docstring=('using the channel amp as additional'
                       'parameter to allow rabi-type experiments without'
                       'wave reloading. Should not be using VSM'))

        for ch in ['GI', 'GQ', 'DI', 'DQ']:
            self.add_parameter(
                'channel_{}'.format(ch), parameter_class=ManualParameter,
                vals=vals.Numbers(1, self._num_channels))

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

    def _set_channel_amp(self, val):
        AWG = self.AWG.get_instr()
        for awg_ch in [self.channel_GI(), self.channel_GQ(),
                       self.channel_DI(), self.channel_DQ()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            AWG.set('awgs_{}_outputs_{}_amplitude'.format(awg_nr, ch_pair), val)

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
        self.add_parameter('G_mixer_alpha', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('G_mixer_phi', vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('D_mixer_alpha', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('D_mixer_phi', vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)

        self.add_parameter(
            'mixer_apply_predistortion_matrix', vals=vals.Bool(), docstring=(
                'If True applies a mixer correction using mixer_phi and '
                'mixer_alpha to all microwave pulses using.'),
            parameter_class=ManualParameter, initial_value=True)

    def apply_mixer_predistortion_corrections(self, wave_dict):

        M_G = wf.mixer_predistortion_matrix(self.G_mixer_alpha(),
                                            self.G_mixer_phi())
        M_D = wf.mixer_predistortion_matrix(self.D_mixer_alpha(),
                                            self.D_mixer_phi())

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

    def _add_waveform_parameters(self):
        super()._add_waveform_parameters()
        # parameters related to codeword bits
        self.add_parameter('bit_shift', unit='', vals=vals.Ints(0, 4),
                           parameter_class=ManualParameter,
                           initial_value=0)
        self.add_parameter('bit_width', unit='', vals=vals.Ints(0, 4),
                           parameter_class=ManualParameter,
                           initial_value=0)
        # parameters related to phase compilation
        self.add_parameter('phi', unit='rad', vals=vals.Numbers(0, 360),
                           parameter_class=ManualParameter,
                           initial_value=0)
        # parameters related to timings
        self.add_parameter('pulse_delay', unit='s', vals=vals.Numbers(0, 1e-6),
                           parameter_class=ManualParameter,
                           initial_value=0)

    def generate_standard_waveforms(self):
        self._wave_dict = {}
        if self.cfg_sideband_mode() == 'static':
            f_modulation = self.mw_modulation()
        else:
            f_modulation = 0
            self.AWG.get_instr().set('ch_pair{}_sideband_frequency'.format(self.channel_I()),
                                     self.mw_modulation())
            self.AWG.get_instr().syncSidebandGenerators()

        ########################################
        # STD waveforms
        ########################################
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
            self._wave_dict = self.apply_mixer_predistortion_corrections(
                self._wave_dict)
        return self._wave_dict

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
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
            redundant_cw_I = 'wave_ch{}_cw{:03}'.format(self.channel_I(),
                                                        redundant_cw_idx)
            self.AWG.get_instr().set(redundant_cw_I, waveforms[0])
            redundant_cw_Q = 'wave_ch{}_cw{:03}'.format(self.channel_Q(),
                                                        redundant_cw_idx)
            self.AWG.get_instr().set(redundant_cw_Q, waveforms[1])


# Not the cleanest inheritance but whatever - MAR Nov 2017
class QWG_VSM_MW_LutMan(AWG8_VSM_MW_LutMan):

    def load_waveforms_onto_AWG_lookuptable(
            self, regenerate_waveforms: bool=True,  stop_start: bool = True):
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
