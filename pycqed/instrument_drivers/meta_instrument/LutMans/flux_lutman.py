from .base_lutman import Base_LutMan
import numpy as np
import logging
from copy import copy
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from qcodes.utils import validators as vals
from pycqed.instrument_drivers.pq_parameters import NP_NANs
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.measurement.waveform_control_CC import waveforms_flux as wfl
try:
    from pycqed.measurement.openql_experiments.openql_helpers import clocks_to_s
except ImportError:
    pass # This is to make the lutman work if no OpenQL is installed.
from qcodes.plots.pyqtgraph import QtPlot
import matplotlib.pyplot as plt
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
import time

_def_lm = {
    0: {"name": "i", "type": "idle"},
    1: {"name": "cz_NE", "type": "idle_z", "which": "NE"},
    2: {"name": "cz_SE", "type": "cz", "which": "SE"},
    3: {"name": "cz_SW", "type": "cz", "which": "SW"},
    4: {"name": "cz_NW", "type": "idle_z", "which": "NW"},
    5: {"name": "park", "type": "square"},
    6: {"name": "square", "type": "square"},
    7: {"name": "custom_wf", "type": "custom"}
}

valid_types = {'idle', 'cz', 'idle_z', 'square', 'custom'}

def flux_lutmap_is_valid(lutmap: dict) -> bool:
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


class Base_Flux_LutMan(Base_LutMan):
    """
    The default scheme of this LutMap allows for 4 different 2Q gates.

    NW         NE
      \       /
       \     /
        \   /
          Q
        /   \
       /     \
      /       \ 
    SW         SE

    """
    def render_wave(self, wave_name, show=True, time_units='s',
                    reload_pulses: bool = True, render_distorted_wave: bool = True,
                    QtPlot_win=None):
        """
        Renders a waveform
        """
        if reload_pulses:
            self.generate_standard_waveforms()

        x = np.arange(len(self._wave_dict[wave_name]))
        y = self._wave_dict[wave_name]

        if time_units == 'lut_index':
            xlab = ('Lookuptable index', 'i')
        elif time_units == 's':
            x = x/self.sampling_rate()
            xlab = ('Time', 's')

        if QtPlot_win is None:
            QtPlot_win = QtPlot(window_title=wave_name,
                                figsize=(600, 400))

        if render_distorted_wave:
            if wave_name in self._wave_dict_dist.keys():
                x2 = np.arange(len(self._wave_dict_dist[wave_name]))
                if time_units == 's':
                    x2 = x2/self.sampling_rate()

                y2 = self._wave_dict_dist[wave_name]
                QtPlot_win.add(
                    x=x2, y=y2, name=wave_name+' distorted',
                    symbol='o', symbolSize=5,
                    xlabel=xlab[0], xunit=xlab[1], ylabel='Amplitude',
                    yunit='dac val.')
            else:
                logging.warning('Wave not in distorted wave dict')
        # Plotting the normal one second ensures it is on top.
        QtPlot_win.add(
            x=x, y=y, name=wave_name,
            symbol='o', symbolSize=5,
            xlabel=xlab[0], xunit=xlab[1], ylabel='Amplitude', yunit='V')

        return QtPlot_win

class HDAWG_Flux_LutMan(Base_Flux_LutMan):
    
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._wave_dict_dist = dict()
        self.sampling_rate(2.4e9)
        self._add_qubit_parameters()

    def set_default_lutmap(self):
        """Set the default lutmap for standard microwave drive pulses."""
        self.LutMap(_def_lm.copy())

    def generate_standard_waveforms(self):
        """
        Generates all the standard waveforms and populates self._wave_dict
        """
        self._wave_dict = {}
        # N.B. the  naming convention ._gen_{waveform_name} must be preserved
        # as it is used in the load_waveform_onto_AWG_lookuptable method.
        self._wave_dict['i'] = self._gen_i()
        self._wave_dict['square'] = self._gen_square()
        self._wave_dict['park'] = self._gen_park()
        self._wave_dict['custom_wf'] = self._gen_custom_wf()


        for idx, waveform in self.LutMap().items():
            wave_name = waveform['name']
            if waveform['type']=='cz' or waveform['type']=='idle_z':
                which_gate = waveform['which']
            if waveform['type']=='cz':
                self._wave_dict[wave_name] = self._gen_cz(which_gate=which_gate)
            elif waveform['type']=='idle_z':
                self._wave_dict[wave_name] = self._gen_idle_z(which_gate=which_gate)

    def _gen_i(self):
        return np.zeros(int(self.idle_pulse_length()*self.sampling_rate()))

    def _gen_square(self):
        return wf.single_channel_block(
            amp=self.sq_amp(), length=self.sq_length(),
            sampling_rate=self.sampling_rate(), delay=0)

    def _gen_park(self):
        return self.park_amp()*np.ones(int(self.park_length()*self.sampling_rate()))

    def _add_qubit_parameters(self):
        """
        Adds parameters responsible for keeping track of qubit frequencies,
        coupling strengths etc.
        """
        self.add_parameter(
            'q_polycoeffs_freq_01_det',
            docstring='Coefficients of the polynomial used to convert '
            'amplitude in V to detuning in Hz. \nN.B. it is important to '
            'include both the AWG range and channel amplitude in the params.\n'
            'N.B.2 Sign convention: positive detuning means frequency is '
            'higher than current  frequency, negative detuning means its '
            'smaller.\n'
            'In order to convert a set of cryoscope flux arc coefficients to '
            ' units of Volts they can be rescaled using [c0*sc**2, c1*sc, c2]'
            ' where sc is the desired scaling factor that includes the sq_amp '
            'used and the range of the AWG (5 in amp mode).',
            vals=vals.Arrays(),
            # initial value is chosen to not raise errors
            initial_value=np.array([-2e9, 0, 0]),
            parameter_class=ManualParameter)
        self.add_parameter(
            'q_polycoeffs_anharm',
            docstring='coefficients of the polynomial used to calculate '
            'the anharmonicity (Hz) as a function of amplitude in V. '
            'N.B. it is important to '
            'include both the AWG range and channel amplitude in the params.\n',
            vals=vals.Arrays(),
            # initial value sets a flux independent anharmonicity of 300MHz
            initial_value=np.array([0, 0, -300e6]),
            parameter_class=ManualParameter)

        self.add_parameter('q_freq_01', vals=vals.Numbers(),
                           docstring='Current operating frequency of qubit',
                           # initial value is chosen to not raise errors
                           initial_value=6e9,
                           unit='Hz', parameter_class=ManualParameter)

        for this_cz in ['NE','NW','SW','SE']:
            self.add_parameter('q_freq_10_%s'%this_cz, vals=vals.Numbers(),
                               docstring='Current operating frequency of qubit'
                               ' with which a CZ gate can be performed.',
                               # initial value is chosen to not raise errors
                               initial_value=6e9,
                               unit='Hz', parameter_class=ManualParameter)
            self.add_parameter('q_J2_%s'%this_cz, vals=vals.Numbers(), unit='Hz',
                               docstring='effective coupling between the 11 and '
                               '02 states.',
                               # initial value is chosen to not raise errors
                               initial_value=15e6,
                               parameter_class=ManualParameter)

    def _gen_idle_z(self, which_gate):
        cz_length = self.get('cz_length_%s'%which_gate)
        idle_z = self._get_phase_corrected_pulse(
            base_wf=np.zeros(int(cz_length*self.sampling_rate()+1)),
            which_gate=which_gate)

        return idle_z

    def _add_waveform_parameters(self):
        # CODEWORD 1: Idling
        self.add_parameter('idle_pulse_length', unit='s',
                           label='Idling pulse length',
                           initial_value=40e-9,
                           vals=vals.Numbers(0, 100e-6),
                           parameter_class=ManualParameter)
        # CODEWORDS 1-4: CZ
        for this_cz in ['NE','NW','SW','SE']:
            self.add_parameter('czd_double_sided_%s'%this_cz,
                               initial_value=False,
                               vals=vals.Bool(),
                               parameter_class=ManualParameter)
            self.add_parameter('disable_cz_only_z_%s'%this_cz,
                               initial_value=False,
                               vals=vals.Bool(),
                               parameter_class=ManualParameter)

            self.add_parameter(
                'czd_net_integral_%s'%this_cz,
                docstring='Used determine what the integral of'
                ' the CZ waveform should evaluate to. This is realized by adding'
                ' an offset to the phase correction pulse.\nBy setting this '
                'parameter to np.nan no offset correction is performed.',
                initial_value=np.nan,
                unit='dac value * samples',
                vals=vals.MultiType(vals.Numbers(), NP_NANs()),
                parameter_class=ManualParameter)

            self.add_parameter('cz_phase_corr_length_%s'%this_cz,
                               unit='s',
                               initial_value=5e-9, vals=vals.Numbers(),
                               parameter_class=ManualParameter)
            self.add_parameter('cz_phase_corr_amp_%s'%this_cz,
                               unit='dac value',
                               initial_value=0,
                               vals=vals.Numbers(),
                               parameter_class=ManualParameter)
            self.add_parameter('cz_length_%s'%this_cz,
                               vals=vals.Numbers(),
                               unit='s', initial_value=35e-9,
                               parameter_class=ManualParameter)
            self.add_parameter('cz_lambda_2_%s'%this_cz,
                               vals=vals.Numbers(),
                               initial_value=0,
                               parameter_class=ManualParameter)
            self.add_parameter('cz_lambda_3_%s'%this_cz,
                               vals=vals.Numbers(),
                               initial_value=0,
                               parameter_class=ManualParameter)
            self.add_parameter('cz_theta_f_%s'%this_cz,
                               vals=vals.Numbers(),
                               unit='deg',
                               initial_value=80,
                               parameter_class=ManualParameter)
            self.add_parameter(
                'czd_lambda_2_%s'%this_cz,
                docstring='lambda_2 parameter of the negative part of the cz pulse'
                ' if set to np.nan will default to the value of the main parameter',
                vals=vals.MultiType(vals.Numbers(), NP_NANs()),
                initial_value=np.nan,
                parameter_class=ManualParameter)

            self.add_parameter(
                'czd_lambda_3_%s'%this_cz,
                docstring='lambda_3 parameter of the negative part of the cz pulse'
                ' if set to np.nan will default to the value of the main parameter',
                vals=vals.MultiType(vals.Numbers(), NP_NANs()),
                initial_value=np.nan,
                parameter_class=ManualParameter)
            self.add_parameter(
                'czd_theta_f_%s'%this_cz,
                docstring='theta_f parameter of the negative part of the cz pulse'
                ' if set to np.nan will default to the value of the main parameter',
                vals=vals.MultiType(vals.Numbers(), NP_NANs()),
                unit='deg',
                initial_value=np.nan,
                parameter_class=ManualParameter)


            self.add_parameter('czd_amp_ratio_%s'%this_cz,
                               docstring='Amplitude ratio for double sided CZ gate',
                               initial_value=1,
                               vals=vals.Numbers(),
                               parameter_class=ManualParameter)

            self.add_parameter('czd_amp_offset_%s'%this_cz,
                               docstring='used to add an offset to the negative '
                               ' pulse that is used in the net-zero cz gate',
                               initial_value=0,
                               unit='dac value',
                               vals=vals.Numbers(),
                               parameter_class=ManualParameter)
            self.add_parameter(
                'czd_signs_%s'%this_cz, initial_value=['+', '-'],
                docstring='Used to determine the sign of the two parts of the '
                'double sided CZ pulse. This should be a list of two elements,'
                ' where "+" is a positive pulse, "-" a negative amplitude and "0" '
                'a disabled pulse.',
                vals=vals.Lists(vals.Enum('+', '-', 0)),
                parameter_class=ManualParameter)
            self.add_parameter('czd_length_ratio_%s'%this_cz,
                               vals=vals.MultiType(vals.Numbers(0, 1),
                                                   vals.Enum('auto')),
                               initial_value=0.5,
                               docstring='When using a net-zero pulse, this '
                               'parameter is used to determine the length ratio'
                               ' of the positive and negative parts of the pulse.'
                               'If this is set to "auto", the ratio will be '
                               'automatically determined to ensure the integral '
                               'of the net-zero pulse is close to zero.',
                               parameter_class=ManualParameter)


        # CODEWORD 6: SQUARE
        self.add_parameter('sq_amp', initial_value=.5,
                           # units is part of the total range of AWG8
                           label='Square pulse amplitude',
                           unit='dac value', vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('sq_length', unit='s',
                           label='Square pulse length',
                           initial_value=40e-9,
                           vals=vals.Numbers(0, 100e-6),
                           parameter_class=ManualParameter)

        # CODEWORD 1: Idling
        self.add_parameter('park_length', unit='s',
                           label='Parking pulse length',
                           initial_value=40e-9,
                           vals=vals.Numbers(0, 100e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('park_amp', initial_value=0,
                           # units is part of the total range of AWG8
                           label='Parking pulse amplitude',
                           unit='dac value', vals=vals.Numbers(),
                           parameter_class=ManualParameter)

        # CODEWORD 7: CUSTOM

        self.add_parameter(
            'custom_wf',
            initial_value=np.array([]),
            label='Custom waveform',
            docstring=('Specifies a custom waveform, note that '
                       '`custom_wf_length` is used to cut of the waveform if'
                       'it is set.'),
            parameter_class=ManualParameter,
            vals=vals.Arrays())
        self.add_parameter(
            'custom_wf_length',
            unit='s',
            label='Custom waveform length',
            initial_value=np.inf,
            docstring=('Used to determine at what sample the custom waveform '
                       'is forced to zero. This is used to facilitate easy '
                       'cryoscope measurements of custom waveforms.'),
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=0))


    def _get_phase_corrected_pulse(self, base_wf, which_gate):
        """
        Creates a phase correction pulse using a cosine with an offset
        to correct any picked up phase.

        Two properties are obeyed.
            - The net-integral (if net-zero) is set to 'czd_net_integral'
            - The amplitude of the cosine is set to 'cz_phase_corr_amp'
        """
        is_double_sided = self.get('czd_double_sided_%s'%which_gate)
        disable_cz_only_z = self.get('disable_cz_only_z_%s'%which_gate)
        cz_integral = self.get('czd_net_integral_%s'%which_gate)
        corr_len = self.get('cz_phase_corr_length_%s'%which_gate)
        corr_amp = self.get('cz_phase_corr_amp_%s'%which_gate)


        corr_samples = int(corr_len*self.sampling_rate())

        # First the offset to guarantee net-zero integral
        if is_double_sided and not np.isnan(cz_integral):
            curr_int = np.sum(base_wf)
            corr_int = cz_integral-curr_int

            corr_pulse = phase_corr_square(
                int_val=corr_int, nr_samples=corr_samples)
            if np.max(corr_pulse) > 0.5:
                logging.warning('net-zero integral correction({:.2f}) larger than 0.4'.format(
                    np.max(corr_pulse)))
        else:
            corr_pulse = np.zeros(corr_samples)

        # Now the sinusoidal step for phase acquisition
        if is_double_sided:
            corr_pulse += phase_corr_sine_series([corr_amp],
                                                 corr_samples)
        else:
            corr_pulse += phase_corr_sine_series_half([corr_amp],
                                                      corr_samples)

        if disable_cz_only_z:
            modified_wf = np.concatenate([base_wf*0, corr_pulse])
        else:
            modified_wf = np.concatenate([base_wf, corr_pulse])
        return modified_wf

    def _gen_cz(self, which_gate, regenerate_cz=True):
        gate_str = 'cz_%s'%which_gate
        if regenerate_cz:
            self._wave_dict[gate_str] = self._gen_adiabatic_pulse(which_gate=which_gate)

        # Commented out snippet is old (deprecated ) phase corr 19/6/2018 MAR
        # phase_corr = self._gen_phase_corr(cz_offset_comp=True)
        # # CZ with phase correction
        # cz_z = np.concatenate([self._wave_dict['cz'], phase_corr])

        cz_pulse = self._get_phase_corrected_pulse(base_wf=self._wave_dict[gate_str],
                                                   which_gate=which_gate)

        return cz_pulse

    def _gen_adiabatic_pulse(self, which_gate):
        """
        Generates the CZ waveform.
        """

        # getting the right parameters for the gate
        is_double_sided = self.get('czd_double_sided_%s'%which_gate)
        cz_length = self.get('cz_length_%s'%which_gate)
        cz_theta_f = self.get('cz_theta_f_%s'%which_gate)
        cz_lambda_2 = self.get('cz_lambda_2_%s'%which_gate)
        cz_lambda_3 = self.get('cz_lambda_3_%s'%which_gate)
        q_J2 = self.get('q_J2_%s'%which_gate)
        czd_signs = self.get('czd_signs_%s'%which_gate)

        czd_theta_f = self.get('czd_theta_f_%s'%which_gate)
        czd_lambda_2 = self.get('czd_lambda_2_%s'%which_gate)
        czd_lambda_3 = self.get('czd_lambda_3_%s'%which_gate)

        czd_amp_ratio = self.get('czd_amp_ratio_%s'%which_gate)
        czd_amp_offset = self.get('czd_amp_offset_%s'%which_gate)

        dac_scalefactor = self.get_amp_to_dac_val_scalefactor()
        eps_i = self.calc_amp_to_eps(0, state_A='11',
                                     state_B='02',
                                     which_gate=which_gate)
        # Beware theta in radian!
        theta_i = wfl.eps_to_theta(eps_i, g=q_J2)

        if not is_double_sided:
            CZ_theta = wfl.martinis_flux_pulse(
                cz_length, theta_i=theta_i,
                theta_f=np.deg2rad(cz_theta_f),
                lambda_2=cz_lambda_2, lambda_3=cz_lambda_3)
            CZ_eps = wfl.theta_to_eps(CZ_theta, g=q_J2)
            CZ_amp = self.calc_eps_to_amp(CZ_eps, state_A='11',
                                          state_B='02',
                                          which_gate=which_gate)

            # convert amplitude in V to amplitude in awg dac value
            CZ = dac_scalefactor*CZ_amp
            return CZ

        else:
            signs = czd_signs

            # Simple double sided CZ pulse implemented in most basic form.
            # repeats the same CZ gate twice and sticks it together.
            length_ratio = self.calc_net_zero_length_ratio(which_gate=which_gate)

            CZ_theta_A = wfl.martinis_flux_pulse(
                cz_length*length_ratio, theta_i=theta_i,
                theta_f=np.deg2rad(cz_theta_f),
                lambda_2=cz_lambda_2, lambda_3=cz_lambda_3)
            CZ_eps_A = wfl.theta_to_eps(CZ_theta_A, g=q_J2)

            CZ_amp_A = self.calc_eps_to_amp(CZ_eps_A, state_A='11',
                                            state_B='02',
                                            positive_branch=(signs[0] == '+'),
                                            which_gate=which_gate)

            CZ_A = dac_scalefactor*CZ_amp_A
            if signs[0] == 0:
                CZ_A *= 0

            # Generate the second CZ pulse. If the params are np.nan, default
            # to the main parameter
            if not np.isnan(czd_theta_f):
                d_theta_f = czd_theta_f
            else:
                d_theta_f = cz_theta_f

            if not np.isnan(czd_lambda_2):
                d_lambda_2 = czd_lambda_2
            else:
                d_lambda_2 = cz_lambda_2
            if not np.isnan(czd_lambda_3):
                d_lambda_3 = czd_lambda_3
            else:
                d_lambda_3 = cz_lambda_3

            CZ_theta_B = wfl.martinis_flux_pulse(
                cz_length*(1-length_ratio), theta_i=theta_i,
                theta_f=np.deg2rad(d_theta_f),
                lambda_2=d_lambda_2, lambda_3=d_lambda_3)
            CZ_eps_B = wfl.theta_to_eps(CZ_theta_B, g=q_J2)
            CZ_amp_B = self.calc_eps_to_amp(CZ_eps_B,
                                            state_A='11', state_B='02',
                                            positive_branch=(signs[1] == '+'),
                                            which_gate=which_gate)

            CZ_B = dac_scalefactor*CZ_amp_B
            if signs[1] == 0:
                CZ_B *= 0
            # Combine both halves of the double sided CZ gate
            amp_rat = czd_amp_ratio
            waveform = np.concatenate(
                [CZ_A, amp_rat*CZ_B + czd_amp_offset])

            return waveform

    def calc_amp_to_eps(self, amp: float,
                        state_A: str = '01',
                        state_B: str = '02',
                        which_gate: str = 'NE'):
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
        polycoeffs_A = self.get_polycoeffs_state(state=state_A,
                                                 which_gate=which_gate)
        polycoeffs_B = self.get_polycoeffs_state(state=state_B,
                                                 which_gate=which_gate)
        polycoeffs = polycoeffs_B - polycoeffs_A
        return np.polyval(polycoeffs, amp)

    def calc_eps_to_amp(self, eps,
                        state_A: str = '01',
                        state_B: str = '02',
                        which_gate: str = 'NE',
                        positive_branch=True):
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
            return np.array([self.calc_eps_to_amp(
                eps=e, state_A=state_A, state_B=state_B, which_gate=which_gate,
                positive_branch=positive_branch) for e in eps])

        polycoeffs_A = self.get_polycoeffs_state(state=state_A,
                                                 which_gate=which_gate)
        if state_B is not None:
            polycoeffs_B = self.get_polycoeffs_state(state=state_B,
                                                     which_gate=which_gate)
            polycoeffs = polycoeffs_B - polycoeffs_A
        else:
            polycoeffs = copy(polycoeffs_A)
            polycoeffs[-1] = 0

        p = np.poly1d(polycoeffs)
        sols = (p-eps).roots

        # sols returns 2 solutions (for a 2nd order polynomial)
        if positive_branch:
            sol = np.max(sols)
        else:
            sol = np.min(sols)

        # imaginary part is ignored, instead sticking to closest real value
        # float is because of a typecasting bug in np 1.12 (solved in 1.14)
        return float(np.real(sol))

    def calc_net_zero_length_ratio(self, which_gate: str = 'NE'):
        """
        Determines the lenght ratio of the net-zero pulses based on the
        parameter "czd_length_ratio".

        If czd_length_ratio is set to auto, uses the interaction amplitudes
        to determine the scaling of lengths. Note that this is a coarse
        approximation.
        """
        czd_length_ratio = self.get('czd_length_ratio_%s'%which_gate)
        if czd_length_ratio != 'auto':
            return czd_length_ratio
        else:
            amp_J2_pos = self.calc_eps_to_amp(0, state_A='11', state_B='02',
                                              which_gate=which_gate,
                                              positive_branch=True)
            amp_J2_neg = self.calc_eps_to_amp(0, state_A='11', state_B='02',
                                              which_gate=which_gate,
                                              positive_branch=False)

            # lr chosen to satisfy (amp_pos*lr + amp_neg*(1-lr) = 0 )
            lr = - amp_J2_neg/(amp_J2_pos-amp_J2_neg)
            return lr

    def get_polycoeffs_state(self, state: str, which_gate: str = 'NE'):
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
        freq_10 = self.get('q_freq_10_%s'%which_gate)
        if state == '00':
            pass
        elif state == '01':
            polycoeffs += self.q_polycoeffs_freq_01_det()
            polycoeffs[2] += self.q_freq_01()
        elif state == '02':
            polycoeffs += 2*self.q_polycoeffs_freq_01_det()
            polycoeffs += self.q_polycoeffs_anharm()
            polycoeffs[2] += 2*self.q_freq_01()
        elif state == '10':
            polycoeffs[2] += freq_10
        elif state == '11':
            polycoeffs += self.q_polycoeffs_freq_01_det()
            polycoeffs[2] += self.q_freq_01() + freq_10
        else:
            raise ValueError('State {} not recognized'.format(state))
        return polycoeffs

    def _get_awg_channel_amplitude(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel()-1  # -1 is to account for starting at 1
        awg_nr = awg_ch//2
        ch_pair = awg_ch % 2
        for i in range(5):
            channel_amp = AWG.get('awgs_{}_outputs_{}_amplitude'.format(
                awg_nr, ch_pair))
            if channel_amp is not None:
                break
            time.sleep(0.5)
        return channel_amp

    def _get_awg_channel_range(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel()-1  # -1 is to account for starting at 1
        # channel range of 5 corresponds to -2.5V to +2.5V
        for i in range(5):
            channel_range_pp = AWG.get('sigouts_{}_range'.format(awg_ch))
            if channel_range_pp is not None:
                break
            time.sleep(0.5)
        return channel_range_pp

    def _gen_composite_wf(self, primitive_waveform_name: str,
                          time_tuples: list):
        """
        Generates a composite waveform based on a timetuple.
        Only relies on the first element of the timetuple which is expected
        to be the starting time of the pulse in clock cycles.


        N.B. No waveforms are regenerated here!
        This relies on the base waveforms being up to date in self._wave_dict

        """

        max_nr_samples = int(self.cfg_max_wf_length()*self.sampling_rate())
        waveform = np.zeros(max_nr_samples)

        for i, tt in enumerate(time_tuples):
            t_start = clocks_to_s(tt[0])
            sample = self.time_to_sample(t_start)
            if sample > max_nr_samples:
                raise ValueError('Waveform longer than max wf lenght')

            if (primitive_waveform_name == 'cz_z' or
                    primitive_waveform_name == 'idle_z'):
                phase_corr = wf.single_channel_block(
                    amp=self.get('cz_phase_corr_amp'),
                    length=self.cz_phase_corr_length(),
                    sampling_rate=self.sampling_rate(), delay=0)
                # phase_corr = wf.single_channel_block(
                #     amp=self.get('mcz_phase_corr_amp_{}'.format(i+1)),
                #     length=self.cz_phase_corr_length(),
                #     sampling_rate=self.sampling_rate(), delay=0)
                if primitive_waveform_name == 'cz_z':
                    prim_wf = np.concatenate(
                        [self._wave_dict['cz'], phase_corr])
                elif primitive_waveform_name == 'idle_z':
                    prim_wf = np.concatenate(
                        [np.zeros(len(self._wave_dict['cz'])), phase_corr])
            else:
                prim_wf = self._wave_dict[primitive_waveform_name]
            waveform[sample:sample+len(prim_wf)] += prim_wf

        return waveform

    def _get_wf_name_from_cw(self, codeword: int):
        for idx, waveform in self.LutMap().items():
            if int(idx) == codeword:
                return waveform['name']
        raise ValueError("Codeword {} not specified"
                         " in LutMap".format(codeword))

    def _get_cw_from_wf_name(self, wf_name: str):
        for idx, waveform in self.LutMap().items():
            if wf_name == waveform['name']:
                return int(idx)
        raise ValueError("Waveform {} not specified"
                         " in LutMap".format(wf_name))

    def _gen_custom_wf(self):
        base_wf = copy(self.custom_wf())

        if self.custom_wf_length() != np.inf:
            # cuts of the waveform at a certain length by setting
            # all subsequent samples to 0.
            max_sample = int(self.custom_wf_length()*self.sampling_rate())
            base_wf[max_sample:] = 0
        return base_wf


    def calc_freq_to_amp(self, freq: float, state: str = '01',
                         which_gate: str = 'NE',
                         positive_branch=True):
        """
        Calculates amplitude in Volt corresponding to the energy of a state
        in Hz.

        N.B. this method assumes that the polycoeffs are with respect to the
            amplitude in units of V, including rescaling due to the channel
            amplitude and range settings of the AWG8.
            See also `self.get_dac_val_to_amp_scalefactor`.

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """

        return self.calc_eps_to_amp(eps=freq, state_B=state, state_A='00',
                                    positive_branch=positive_branch, which_gate=which_gate)


    """
    UNTOUCHED
    """

    def _add_cfg_parameters(self):

        self.add_parameter('cfg_awg_channel',
                           initial_value=1,
                           vals=vals.Ints(1, 8),
                           parameter_class=ManualParameter)
        self.add_parameter('cfg_distort',
                           initial_value=True,
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter(
            'cfg_append_compensation', docstring=(
                'If True compensation pulses will be added to individual '
                ' waveforms creating very long waveforms for each codeword'),
            initial_value=True, vals=vals.Bool(),
            parameter_class=ManualParameter)
        self.add_parameter('cfg_compensation_delay',
                           parameter_class=ManualParameter,
                           initial_value=3e-6,
                           unit='s',
                           vals=vals.Numbers())

        self.add_parameter(
            'cfg_pre_pulse_delay', unit='s', label='Pre pulse delay',
            docstring='This parameter is used for fine timing corrections, the'
                      ' correction is applied in distort_waveform.',
            initial_value=0e-9,
            vals=vals.Numbers(0, 1e-6),
            parameter_class=ManualParameter)

        self.add_parameter('instr_distortion_kernel',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_partner_lutman',
                           docstring='LutMan responsible for the corresponding'
                           'channel in the AWG8 channel pair. '
                           'Reference is used when uploading waveforms',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter(
            '_awgs_fl_sequencer_program_expected_hash',
            docstring='crc32 hash of the awg8 sequencer program. '
            'This parameter is used to dynamically determine '
            'if the program needs to be uploaded. The initial_value is'
            ' None, indicating that the program needs to be uploaded.'
            ' After the first program is uploaded, the value is set.',
            parameter_class=ManualParameter, initial_value=None,
            vals=vals.Ints())

        self.add_parameter(
            'cfg_operating_mode',
            initial_value='Codeword_normal',
            vals=vals.Enum('Codeword_normal'),
            # 'CW_single_01', 'CW_single_02',
            # 'CW_single_03', 'CW_single_04',
            # 'CW_single_05', 'CW_single_06'),
            docstring='Used to determine what program to load in the AWG8. '
            'If set to "Codeword_normal" it does codeword triggering, '
            'other modes exist to play only a specific single waveform.',
            set_cmd=self._set_cfg_operating_mode,
            get_cmd=self._get_cfg_operating_mode)
        self._cfg_operating_mode = 'Codeword_normal'

        self.add_parameter('cfg_max_wf_length',
                           parameter_class=ManualParameter,
                           initial_value=10e-6,
                           unit='s', vals=vals.Numbers(0, 100e-6))
        self.add_parameter('cfg_awg_channel_range',
                           docstring='peak peak value, channel range of 5 corresponds to -2.5V to +2.5V',
                           get_cmd=self._get_awg_channel_range,
                           unit='V_pp')
        self.add_parameter('cfg_awg_channel_amplitude',
                           docstring='digital scale factor between 0 and 1',
                           get_cmd=self._get_awg_channel_amplitude,
                           unit='a.u.', vals=vals.Numbers(0, 1))

    def _set_cfg_operating_mode(self, val):
        self._cfg_operating_mode = val
        # this is to ensure changing the mode requires reuploading the program
        self._awgs_fl_sequencer_program_expected_hash(101)

    def _get_cfg_operating_mode(self):
        return self._cfg_operating_mode

    def get_dac_val_to_amp_scalefactor(self):
        """
        Returns the scale factor to transform an amplitude in 'dac value' to an
        amplitude in 'V'.

        "dac_value" refers to the value between -1 and +1 that is set in a
        waveform.

        N.B. the implementation is specific to this type of AWG
        """
        if self.AWG() is None:
            logging.warning('No AWG present, returning unity scale factor.')
            return 1
        channel_amp = self.cfg_awg_channel_amplitude()
        channel_range_pp = self.cfg_awg_channel_range()
        # channel range of 5 corresponds to -2.5V to +2.5V
        scalefactor = channel_amp*(channel_range_pp/2)
        return scalefactor

    def get_amp_to_dac_val_scalefactor(self):
        if self.get_dac_val_to_amp_scalefactor() == 0:
                # Give a warning and don't raise an error as things should not
                # break because of this.
            logging.warning('AWG amp to dac scale factor is 0, check "{}" '
                            'output amplitudes'.format(self.AWG()))
            return 1
        return 1/self.get_dac_val_to_amp_scalefactor()

    def calc_amp_to_freq(self, amp: float, state: str = '01', which_gate: str = 'NE'):
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
    ###########################################################
    #  Waveform generation net-zero phase correction methods  #
    ###########################################################
    def _calc_modified_wf(self, base_wf, a_i, corr_samples):

        if not np.isnan(self.czd_net_integral()):
            curr_int = np.sum(base_wf)
            corr_int = self.czd_net_integral()-curr_int
            #corr_pulse = phase_corr_triangle(
            #    int_val=corr_int, nr_samples=corr_samples)

            corr_pulse = phase_corr_square(
                int_val=corr_int, nr_samples=corr_samples)
            if np.max(corr_pulse) > 0.5:
                logging.warning('net-zero integral correction({:.2f}) larger than 0.5'.format(
                    np.max(corr_pulse)))
        else:
            corr_pulse = np.zeros(corr_samples)

        corr_pulse += phase_corr_sine_series(a_i, corr_samples)

        modified_wf = np.concatenate([base_wf, corr_pulse])
        return modified_wf

    def _phase_corr_cost_func(self, base_wf, a_i, corr_samples,
                              print_result=False):
        """
        The cost function of the cz_z waveform is designed to meet
        the following criteria
        1. Net-zero character of waveform
            Integral of wf = 0
        2. Single qubit phase correction
            Integral of phase_corr_part**2 = desired constant
        3. Target_wf ends at 0
        4. No-distortions present after cutting of waveform
            predistorted_wf ends at 0 smoothly (as many derivatives as possible 0)

        5. Minimize the maximum amplitude
            Prefer small non-violent pulses
        """
        # samples to quanitify leftover distortions
        tail_samples = 500

        target_wf = self._calc_modified_wf(
            base_wf, a_i=a_i, corr_samples=corr_samples)
        if self.cfg_distort():
            k0 = self.instr_distortion_kernel.get_instr()
            predistorted_wf = k0.distort_waveform(target_wf,
                                                  len(target_wf)+tail_samples)
        else:
            predistorted_wf = target_wf

        # 2. Phase correction pulse
        phase_corr_int = np.sum(
            target_wf[len(base_wf):len(base_wf)+corr_samples]**2)/corr_samples
        cv_2 = ((phase_corr_int - self.cz_phase_corr_amp()**2)*1000)**2

        # 4. No-distortions present after cutting of waveform
        cv_4 = np.sum(abs(predistorted_wf[-tail_samples+50:]))*20
        # 5. no violent waveform
        cv_5 = np.max(abs(target_wf)*100)**2

        cost_val = cv_2 + cv_4+cv_5

        #     if print_result:
        #         print("Cost function value")''

        # #         print("cv_1 net_zero_character: {:.6f}".format(cv_1))
        #         print("cv_2 phase corr pulse  : {:.6f}".format(cv_2))
        # #         print("cv_3 ends at 0         : {:.6f}".format(cv_3))
        #         print("cv_4 distortions tail  : {:.6f}".format(cv_4))
        #         print("cv_5 non violent       : {:.6f}".format(cv_5))

        return cost_val


    #################################
    #  Waveform loading methods     #
    #################################

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool = False):
        """
        Loads a specific waveform to the AWG
        """
        if regenerate_waveforms:
            # only regenerate the one waveform that is desired
            gen_wf_func = getattr(self, '_gen_{}'.format(waveform_name))
            self._wave_dict[waveform_name] = gen_wf_func()

        waveform = self._wave_dict[waveform_name]
        codeword = self._get_cw_from_wf_name(waveform_name)
        codeword_str = 'wave_ch{}_cw{:03}'.format(self.cfg_awg_channel(),codeword)

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
            self, regenerate_waveforms: bool = True, stop_start: bool = True,
            force_load_sequencer_program: bool = False):
        """
        Loads all waveforms specified in the LutMap to an AWG for both this
        LutMap and the partner LutMap.

        Args:
            regenerate_waveforms (bool): if True calls
                generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.


        Because of realtime loading vs the DIO sequencer program the uploading
        flow for the AWG8 is slightly different.
        """

        # Generate the waveforms and link them to the AWG8 parameters
        lutmans = [self]
        # If statement is here because it should be possible to function
        # independently
        if self.instr_partner_lutman() is None:
            logging.warning('No partner_lutman specified')
        else:
            lutmans += [self.instr_partner_lutman.get_instr()]

        if self.cfg_operating_mode() != 'Codeword_normal':
            # Regenerating only one waveform in this case, see below
            regenerate_waveforms_realtime = regenerate_waveforms
            regenerate_waveforms = False

        for lm in lutmans:
            if regenerate_waveforms:
                lm.generate_standard_waveforms()
                for idx, waveform in lm.LutMap().items():
                    waveform_name = waveform['name']
                    lm.load_waveform_onto_AWG_lookuptable(waveform_name)

        # Uploading the codeword program if required
        if self._program_hash_differs() or force_load_sequencer_program:
            # FIXME: List of conditions in which reloading is required
            # - program hash differs
            # - max waveform length has changed
            # N.B. these other conditions need to be included here.
            # This ensures only the channels that are relevant get reconfigured
            awg_nr = (self.cfg_awg_channel()-1)//2
            self._upload_codeword_program(awg_nr)

        if self.cfg_operating_mode() == 'Codeword_normal':
            for idx, waveform in lm.LutMap().items():
                self.load_waveform_realtime(
                    waveform_name=waveform['name'],
                    wf_nr=None, regenerate_waveforms=False)
        else:
            # Only load one waveform and do it in realtime.
            cw_idx = int(self.cfg_operating_mode()[-2:])
            waveform_name = self._get_wf_name_from_cw(cw_idx)
            self.load_waveform_realtime(
                waveform_name=waveform_name,
                wf_nr=None, regenerate_waveforms=regenerate_waveforms_realtime)
        #updating channel amplitude and range
        self.cfg_awg_channel_amplitude()
        self.cfg_awg_channel_range()
        self._update_expected_program_hash()

    def load_waveform_realtime(self, waveform_name: str,
                               wf_nr: int = None,
                               regenerate_waveforms: bool = True):
        """
        Args:
            waveform_name:        (str) : name of the waveform
            wf_nr                 (int) : what codeword to load the pulse onto
                if set to None, will determine awg_nr based on self.LutMap
            regenerate_waveforms (bool) : if True regenerates all waveforms

        """

        if wf_nr is None:
            wf_nr = self._get_cw_from_wf_name(waveform_name)

        if regenerate_waveforms:
            gen_wf_func = getattr(self, '_gen_{}'.format(waveform_name))
            waveform = gen_wf_func()
            if self.cfg_append_compensation():
                waveform = self.add_compensation_pulses(waveform)
            if self.cfg_distort():
                waveform = self.distort_waveform(waveform)
                self._wave_dict_dist[waveform_name] = waveform
            else:
                waveform = self._append_zero_samples(waveform)
                self._wave_dict_dist[waveform_name] = waveform

        waveform = self._wave_dict_dist[waveform_name]
        codeword = wf_nr
        self.AWG.get_instr().set('wave_ch{}_cw{:03}'.format(self.cfg_awg_channel(),codeword), waveform)

        if self.instr_partner_lutman() is None:
            logging.warning('no partner lutman specified')
            other_waveform = np.zeros(len(waveform))
        else:
            partner_lm = self.instr_partner_lutman.get_instr()
            prtnr_wf_name = partner_lm._get_wf_name_from_cw(codeword=wf_nr)
            if regenerate_waveforms:
                gen_wf_func = getattr(partner_lm,
                                      '_gen_{}'.format(prtnr_wf_name))
                prtnr_wf = gen_wf_func()

                if partner_lm.cfg_append_compensation():
                    prtnr_wf = partner_lm.add_compensation_pulses(prtnr_wf)

                if partner_lm.cfg_distort():
                    prtnr_wf = partner_lm.distort_waveform(prtnr_wf)
                    partner_lm._wave_dict_dist[prtnr_wf_name] = prtnr_wf

            other_waveform = partner_lm._wave_dict_dist[prtnr_wf_name]

        if len(waveform) != len(other_waveform):
            other_waveform = np.zeros(len(waveform))
            msg = ('Lenghts of waveforms "{}" ({}) and "{}" ({}) do '
                   'not match.'.format(waveform_name, len(waveform),
                                       prtnr_wf_name, len(other_waveform)))
            logging.warning(msg)
            logging.warning('Setting "other waveform" to array of zeros.')

        awg_ch = self.cfg_awg_channel()-1  # -1 is to account for starting at 1
        ch_pair = awg_ch % 2
        awg_nr = awg_ch//2

        if ch_pair == 0:
            waveforms = (waveform, other_waveform)
        else:
            waveforms = (other_waveform, waveform)

        self.AWG.get_instr().upload_waveform_realtime(
            w0=waveforms[0], w1=waveforms[1], awg_nr=awg_nr, wf_nr=wf_nr)

    def _generate_single_cw_program(self, cw_idx):
        devname = self.AWG.get_instr()._devname
        ch = self.cfg_awg_channel() - (self.cfg_awg_channel()+1) % 2
        awg_single_wf_program = (
            '\nwhile (1) {\n' +
            'waitDIOTrigger();\n' +
            'playWave("{}_wave_ch{}_cw{:03}", '.format(devname, ch, cw_idx) +
            '"{}_wave_ch{}_cw{:03}");'.format(devname, ch+1, cw_idx) + '\n}')
        return awg_single_wf_program

    def _upload_codeword_program(self, awg_nr):
        awg = self.AWG.get_instr()
        if self.cfg_operating_mode() == 'Codeword_normal':
            awg.upload_codeword_program(awgs=[awg_nr])
        else:
            cw_idx = int(self.cfg_operating_mode()[-2:])
            single_cw_program = self._generate_single_cw_program(cw_idx)
            awg.configure_awg_from_string(awg_nr, single_cw_program)
            awg.configure_codeword_protocol()
            awg.start()

    def _append_zero_samples(self, waveform):
        """
        Helper method to ensure waveforms have the desired length
        """
        length_samples = roundup1024(int(self.sampling_rate()*self.cfg_max_wf_length()))
        extra_samples = length_samples - len(waveform)
        if extra_samples >= 0:
            y_sig = np.concatenate([waveform, np.zeros(extra_samples)])
        else:
            y_sig = waveform[:extra_samples]
        return y_sig

    def _update_expected_program_hash(self):
        """
        Updates the expected AWG sequencer program hash with the current
        hash of the sequencer program. This is intended to be called after
        setting the hash.
        """
        awg_nr = (self.cfg_awg_channel()-1)//2
        hash = self.AWG.get_instr().get(
            'awgs_{}_sequencer_program_crc32_hash'.format(awg_nr))
        self._awgs_fl_sequencer_program_expected_hash(hash)

    def _program_hash_differs(self)-> bool:
        """
        Args:
            --
        Returns:
            hash_different (bool): returns True if one of the hashes does not
                correspond to the expected hash.

        Compares current AWG sequencer program hash for the relevant channels
        with the expected program hash and, returns True if one ore more
        hashes do not match, indicating that uploading the programs is
        required.
        """
        awg_nr = (self.cfg_awg_channel()-1)//2
        hash = self.AWG.get_instr().get(
            'awgs_{}_sequencer_program_crc32_hash'.format(awg_nr))
        expected_hash = self._awgs_fl_sequencer_program_expected_hash()
        hash_differs = (hash != expected_hash)

        return hash_differs

    def load_composite_waveform_onto_AWG_lookuptable(
        self,
            primitive_waveform_name: str,
            time_tuples: list,
            codeword: int):
        """
        Creates a composite waveform based on time_tuples extracted from a qisa
        file.
        """
        waveform_name = 'comp_{}_cw{:03}'.format(primitive_waveform_name,
                                                 codeword)

        # assigning for post loading rendering purposes
        self._wave_dict[waveform_name] = self._gen_composite_wf(
            primitive_waveform_name,  time_tuples)
        waveform = self._wave_dict[waveform_name]

        codeword = 'wave_ch{}_cw{:03}'.format(self.cfg_awg_channel(),
                                              codeword)

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

        self.AWG.get_instr().set(codeword, waveform)

    def add_compensation_pulses(self, waveform):
        """
        Adds the inverse of the pulses at the end of a waveform to
        ensure flux discharging.
        """
        wf = np.array(waveform)  # catches a rare bug when wf is a list
        delay_samples = np.zeros(int(self.sampling_rate() *
                                     self.cfg_compensation_delay()))
        comp_wf = np.concatenate([wf, delay_samples, -1*wf])
        return comp_wf

    def distort_waveform(self, waveform, inverse=False):
        """
        Modifies the ideal waveform to correct for distortions and correct
        fine delays.
        Distortions are corrected using the kernel object.
        """
        k = self.instr_distortion_kernel.get_instr()

        # Prepend zeros to delay waveform to correct for fine timing
        delay_samples = int(self.cfg_pre_pulse_delay()*self.sampling_rate())
        waveform = np.pad(waveform, (delay_samples, 0), 'constant')

        # duck typing the distort waveform method
        if hasattr(k, 'distort_waveform'):
            distorted_waveform = k.distort_waveform(
                waveform,
                length_samples=int(
                    roundup1024(self.cfg_max_wf_length()*self.sampling_rate())),
                inverse=inverse)
        else:  # old kernel object does not have this method
            if inverse:
                raise NotImplementedError()
            distorted_waveform = k.convolve_kernel(
                [k.kernel(), waveform],
                length_samples=int(self.cfg_max_wf_length() *
                                   self.sampling_rate()))
        return distorted_waveform

    #################################
    #  Plotting methods            #
    #################################

    def plot_cz_trajectory(self, axs=None, show=True,
                           extra_plot_samples: int = 50, which_gate='NE'):
        """
        Plots the cz trajectory in frequency space.
        """
        cz_length = self.get('cz_length_%s'%which_gate)
        q_J2 = self.get('q_J2_%s'%which_gate)
        sampling_rate = self.get('sampling_rate')
        cz_phase_corr_length = self.get('cz_phase_corr_length_%s'%which_gate)



        if axs is None:
            f, axs = plt.subplots(figsize=(5, 7), nrows=3, sharex=True)
        nr_plot_samples = int((cz_length+cz_phase_corr_length) *
                               sampling_rate + extra_plot_samples)

        dac_amps = self._wave_dict['cz_%s'%which_gate][:nr_plot_samples]
        t = np.arange(0, len(dac_amps))*1/self.sampling_rate()

        CZ_amp = dac_amps*self.get_dac_val_to_amp_scalefactor()
        CZ_eps = self.calc_amp_to_eps(CZ_amp, '11', '02', which_gate=which_gate)
        CZ_theta = wfl.eps_to_theta(CZ_eps, q_J2)

        axs[0].plot(t, np.rad2deg(CZ_theta), marker='.')
        axs[0].fill_between(t, np.rad2deg(CZ_theta), color='C0', alpha=.5)
        set_ylabel(axs[0], r'$\theta$', 'deg')

        axs[1].plot(t, CZ_eps, marker='.')
        axs[1].fill_between(t, CZ_eps, color='C0', alpha=.5)
        set_ylabel(axs[1], r'$\epsilon_{11-02}$', 'Hz')

        axs[2].plot(t, CZ_amp, marker='.')
        axs[2].fill_between(t, CZ_amp, color='C0', alpha=.1)
        set_xlabel(axs[2], 'Time', 's')
        set_ylabel(axs[2], r'Amp.', 'V')
        # axs[2].set_ylim(-1, 1)
        axs[2].axhline(0, lw=.2, color='grey')
        CZ_amp_pred = self.distort_waveform(CZ_amp)[:len(CZ_amp)]
        axs[2].plot(t, CZ_amp_pred, marker='.')
        axs[2].fill_between(t, CZ_amp_pred, color='C1', alpha=.3)
        if show:
            plt.show()
        return axs

    def plot_level_diagram(self, ax=None, show=True, which_gate='NE'):
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
        freqs = self.calc_amp_to_freq(amps, state='01', which_gate=which_gate)
        ax.plot(amps, freqs, label='$f_{01}$')
        ax.text(0, self.calc_amp_to_freq(0, state='01', which_gate=which_gate), '01', color='C0',
                ha='left', va='bottom', clip_on=True)

        freqs = self.calc_amp_to_freq(amps, state='02', which_gate=which_gate)
        ax.plot(amps, freqs, label='$f_{02}$')
        ax.text(0, self.calc_amp_to_freq(0, state='02', which_gate=which_gate), '02', color='C1',
                ha='left', va='bottom', clip_on=True)

        freqs = self.calc_amp_to_freq(amps, state='10', which_gate=which_gate)
        ax.plot(amps, freqs, label='$f_{10}$')
        ax.text(0, self.calc_amp_to_freq(0, state='10', which_gate=which_gate), '10', color='C2',
                ha='left', va='bottom', clip_on=True)

        freqs = self.calc_amp_to_freq(amps, state='11', which_gate=which_gate)
        ax.plot(amps, freqs, label='$f_{11}$')
        ax.text(0, self.calc_amp_to_freq(0, state='11', which_gate=which_gate), '11', color='C3',
                ha='left', va='bottom', clip_on=True)

        # 2. Annotating feature of interest
        ax.axvline(0, 0, 1e10, linestyle='dotted', c='grey')

        amp_J2 = self.calc_eps_to_amp(0, state_A='11', state_B='02', which_gate=which_gate)
        amp_J1 = self.calc_eps_to_amp(0, state_A='10', state_B='01', which_gate=which_gate)

        ax.axvline(amp_J2, ls='--', lw=1, c='C4')
        ax.axvline(amp_J1, ls='--', lw=1, c='C6')

        f_11_02 = self.calc_amp_to_freq(amp_J2, state='11', which_gate=which_gate)
        ax.plot([amp_J2], [f_11_02],
                color='C4', marker='o', label='11-02')
        ax.text(amp_J2, f_11_02,
                '({:.4f},{:.2f})'.format(amp_J2, f_11_02*1e-9),
                color='C4',
                ha='left', va='bottom', clip_on=True)

        f_10_01 = self.calc_amp_to_freq(amp_J1, state='01', which_gate=which_gate)

        ax.plot([amp_J1], [f_10_01],
                color='C5', marker='o', label='10-01')
        ax.text(amp_J1, f_10_01,
                '({:.4f},{:.2f})'.format(amp_J1, f_10_01*1e-9),
                color='C5', ha='left', va='bottom', clip_on=True)

        # 3. Adding legend etc.
        title = ('Calibration visualization\n{}\nchannel {}'.format(
            self.AWG(), self.cfg_awg_channel()))
        leg = ax.legend(title=title, loc=(1.05, .3))
        leg._legend_box.align = 'center'
        set_xlabel(ax, 'AWG amplitude', 'V')
        set_ylabel(ax, 'Frequency', 'Hz')
        ax.set_xlim(-2.5, 2.5)

        ax.set_ylim(0, self.calc_amp_to_freq(0, state='02', which_gate=which_gate)*1.1)

        # 4. Add a twin x-axis to denote scale in dac amplitude
        dac_val_axis = ax.twiny()
        dac_ax_lims = np.array(ax.get_xlim()) * \
            self.get_amp_to_dac_val_scalefactor()
        dac_val_axis.set_xlim(dac_ax_lims)
        set_xlabel(dac_val_axis, 'AWG amplitude', 'dac')

        dac_val_axis.axvspan(1, 1000, facecolor='.5', alpha=0.5)
        dac_val_axis.axvspan(-1000, -1, facecolor='.5', alpha=0.5)
        # get figure is here in case an axis object was passed as input
        f = ax.get_figure()
        f.subplots_adjust(right=.7)
        if show:
            plt.show()
        return ax

#########################################################################
# Legacy classes below
#########################################################################

class AWG8_Flux_LutMan(Base_Flux_LutMan):
    _def_lm = ['i', 'cz_z', 'square', 'park', 'multi_cz', 'custom_wf']
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._wave_dict_dist = dict()
        self._def_lm = _def_lm
        self.sampling_rate(2.4e9)
        self._add_qubit_parameters()

    def _add_qubit_parameters(self):
        """
        Adds parameters responsible for keeping track of qubit frequencies,
        coupling strengths etc.

        N.B. Currently this is geared towards a 2-qubit device. Ideally,
        these parameters would be extracted from the relevant qubit objects
        in a way that does not violate the layers of abstraction.
        """
        self.add_parameter(
            'q_polycoeffs_freq_01_det',
            docstring='Coefficients of the polynomial used to convert '
            'amplitude in V to detuning in Hz. \nN.B. it is important to '
            'include both the AWG range and channel amplitude in the params.\n'
            'N.B.2 Sign convention: positive detuning means frequency is '
            'higher than current  frequency, negative detuning means its '
            'smaller.\n'
            'In order to convert a set of cryoscope flux arc coefficients to '
            ' units of Volts they can be rescaled using [c0*sc**2, c1*sc, c2]'
            ' where sc is the desired scaling factor that includes the sq_amp '
            'used and the range of the AWG (5 in amp mode).',
            vals=vals.Arrays(),
            # initial value is chosen to not raise errors
            initial_value=np.array([-2e9, 0, 0]),
            parameter_class=ManualParameter)
        self.add_parameter(
            'q_polycoeffs_anharm',
            docstring='coefficients of the polynomial used to calculate '
            'the anharmonicity (Hz) as a function of amplitude in V. '
            'N.B. it is important to '
            'include both the AWG range and channel amplitude in the params.\n',
            vals=vals.Arrays(),
            # initial value sets a flux independent anharmonicity of 300MHz
            initial_value=np.array([0, 0, -300e6]),
            parameter_class=ManualParameter)

        self.add_parameter('q_freq_01', vals=vals.Numbers(),
                           docstring='Current operating frequency of qubit',
                           # initial value is chosen to not raise errors
                           initial_value=6e9,
                           unit='Hz', parameter_class=ManualParameter)

        self.add_parameter('q_freq_10', vals=vals.Numbers(),
                           docstring='Current operating frequency of qubit'
                           ' with which a CZ gate can be performed.',
                           # initial value is chosen to not raise errors
                           initial_value=6e9,
                           unit='Hz', parameter_class=ManualParameter)
        self.add_parameter('q_J2', vals=vals.Numbers(), unit='Hz',
                           docstring='effective coupling between the 11 and '
                           '02 states.',
                           # initial value is chosen to not raise errors
                           initial_value=15e6,
                           parameter_class=ManualParameter)

    def _add_cfg_parameters(self):

        self.add_parameter('cfg_awg_channel',
                           initial_value=1,
                           vals=vals.Ints(1, 8),
                           parameter_class=ManualParameter)
        self.add_parameter('cfg_distort',
                           initial_value=True,
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter(
            'cfg_append_compensation', docstring=(
                'If True compensation pulses will be added to individual '
                ' waveforms creating very long waveforms for each codeword'),
            initial_value=True, vals=vals.Bool(),
            parameter_class=ManualParameter)
        self.add_parameter('cfg_compensation_delay',
                           parameter_class=ManualParameter,
                           initial_value=3e-6,
                           unit='s',
                           vals=vals.Numbers())

        self.add_parameter(
            'cfg_pre_pulse_delay', unit='s', label='Pre pulse delay',
            docstring='This parameter is used for fine timing corrections, the'
                      ' correction is applied in distort_waveform.',
            initial_value=0e-9,
            vals=vals.Numbers(0, 1e-6),
            parameter_class=ManualParameter)

        self.add_parameter('instr_distortion_kernel',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_partner_lutman',
                           docstring='LutMan responsible for the corresponding'
                           'channel in the AWG8 channel pair. '
                           'Reference is used when uploading waveforms',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter(
            '_awgs_fl_sequencer_program_expected_hash',
            docstring='crc32 hash of the awg8 sequencer program. '
            'This parameter is used to dynamically determine '
            'if the program needs to be uploaded. The initial_value is'
            ' None, indicating that the program needs to be uploaded.'
            ' After the first program is uploaded, the value is set.',
            parameter_class=ManualParameter, initial_value=None,
            vals=vals.Ints())

        self.add_parameter(
            'cfg_operating_mode',
            initial_value='Codeword_normal',
            vals=vals.Enum('Codeword_normal'),
            # 'CW_single_01', 'CW_single_02',
            # 'CW_single_03', 'CW_single_04',
            # 'CW_single_05', 'CW_single_06'),
            docstring='Used to determine what program to load in the AWG8. '
            'If set to "Codeword_normal" it does codeword triggering, '
            'other modes exist to play only a specific single waveform.',
            set_cmd=self._set_cfg_operating_mode,
            get_cmd=self._get_cfg_operating_mode)
        self._cfg_operating_mode = 'Codeword_normal'

        self.add_parameter('cfg_max_wf_length',
                           parameter_class=ManualParameter,
                           initial_value=10e-6,
                           unit='s', vals=vals.Numbers(0, 100e-6))
        self.add_parameter('cfg_awg_channel_range',
                           docstring='peak peak value, channel range of 5 corresponds to -2.5V to +2.5V',
                           get_cmd=self._get_awg_channel_range,
                           unit='V_pp')
        self.add_parameter('cfg_awg_channel_amplitude',
                           docstring='digital scale factor between 0 and 1',
                           get_cmd=self._get_awg_channel_amplitude,
                           unit='a.u.', vals=vals.Numbers(0, 1))

    def _set_cfg_operating_mode(self, val):
        self._cfg_operating_mode = val
        # this is to ensure changing the mode requires reuploading the program
        self._awgs_fl_sequencer_program_expected_hash(101)

    def _get_cfg_operating_mode(self):
        return self._cfg_operating_mode

    def get_polycoeffs_state(self, state: str):
        """
        Args:
            state (str) : string of 2 numbers denoting the state. The numbers
                correspond to the number of excitations in each qubits.
                The LSQ (right) corresponds to the qubit being fluxed and
                under control of this flux lutman.

        Get's the polynomial coefficients that are used to calculate the
        energy levels of specific states.
        Note that avoided crossings are not taken into account here.


        """
        polycoeffs = np.zeros(3)
        if state == '00':
            pass
        elif state == '01':
            polycoeffs += self.q_polycoeffs_freq_01_det()
            polycoeffs[2] += self.q_freq_01()
        elif state == '02':
            polycoeffs += 2*self.q_polycoeffs_freq_01_det()
            polycoeffs += self.q_polycoeffs_anharm()
            polycoeffs[2] += 2*self.q_freq_01()
        elif state == '10':
            polycoeffs[2] += self.q_freq_10()
        elif state == '11':
            polycoeffs += self.q_polycoeffs_freq_01_det()
            polycoeffs[2] += self.q_freq_01() + self.q_freq_10()
        else:
            raise ValueError('State {} not recognized'.format(state))
        return polycoeffs

    def calc_amp_to_freq(self, amp: float, state: str = '01'):
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

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """
        polycoeffs = self.get_polycoeffs_state(state=state)

        return np.polyval(polycoeffs, amp)

    def calc_freq_to_amp(self, freq: float, state: str = '01',
                         positive_branch=True):
        """
        Calculates amplitude in Volt corresponding to the energy of a state
        in Hz.

        N.B. this method assumes that the polycoeffs are with respect to the
            amplitude in units of V, including rescaling due to the channel
            amplitude and range settings of the AWG8.
            See also `self.get_dac_val_to_amp_scalefactor`.

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """

        return self.calc_eps_to_amp(eps=freq, state_B=state, state_A='00',
                                    positive_branch=positive_branch)

    def calc_amp_to_eps(self, amp: float,
                        state_A: str = '01', state_B: str = '02'):
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
        polycoeffs_A = self.get_polycoeffs_state(state=state_A)
        polycoeffs_B = self.get_polycoeffs_state(state=state_B)
        polycoeffs = polycoeffs_B - polycoeffs_A
        return np.polyval(polycoeffs, amp)

    def calc_eps_to_amp(self, eps,
                        state_A: str = '01', state_B: str = '02',
                        positive_branch=True):
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
            return np.array([self.calc_eps_to_amp(
                eps=e, state_A=state_A, state_B=state_B,
                positive_branch=positive_branch) for e in eps])

        polycoeffs_A = self.get_polycoeffs_state(state=state_A)
        if state_B is not None:
            polycoeffs_B = self.get_polycoeffs_state(state=state_B)
            polycoeffs = polycoeffs_B - polycoeffs_A
        else:
            polycoeffs = copy(polycoeffs_A)
            polycoeffs[-1] = 0

        p = np.poly1d(polycoeffs)
        sols = (p-eps).roots

        # sols returns 2 solutions (for a 2nd order polynomial)
        if positive_branch:
            sol = np.max(sols)
        else:
            sol = np.min(sols)

        # imaginary part is ignored, instead sticking to closest real value
        # float is because of a typecasting bug in np 1.12 (solved in 1.14)
        return float(np.real(sol))

    def calc_net_zero_length_ratio(self):
        """
        Determines the lenght ratio of the net-zero pulses based on the
        parameter "czd_length_ratio".

        If czd_length_ratio is set to auto, uses the interaction amplitudes
        to determine the scaling of lengths. Note that this is a coarse
        approximation.
        """
        if self.czd_length_ratio() != 'auto':
            return self.czd_length_ratio()
        else:
            amp_J2_pos = self.calc_eps_to_amp(0, state_A='11', state_B='02',
                                              positive_branch=True)
            amp_J2_neg = self.calc_eps_to_amp(0, state_A='11', state_B='02',
                                              positive_branch=False)

            # lr chosen to satisfy (amp_pos*lr + amp_neg*(1-lr) = 0 )
            lr = - amp_J2_neg/(amp_J2_pos-amp_J2_neg)
            return lr

    def _get_awg_channel_amplitude(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel()-1  # -1 is to account for starting at 1
        awg_nr = awg_ch//2
        ch_pair = awg_ch % 2
        for i in range(5):
            channel_amp = AWG.get('awgs_{}_outputs_{}_amplitude'.format(
                awg_nr, ch_pair))
            if channel_amp is not None:
                break
            time.sleep(0.5)
        return channel_amp

    def _get_awg_channel_range(self):
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel()-1  # -1 is to account for starting at 1
        # channel range of 5 corresponds to -2.5V to +2.5V
        for i in range(5):
            channel_range_pp = AWG.get('sigouts_{}_range'.format(awg_ch))
            if channel_range_pp is not None:
                break
            time.sleep(0.5)
        return channel_range_pp

    def get_dac_val_to_amp_scalefactor(self):
        """
        Returns the scale factor to transform an amplitude in 'dac value' to an
        amplitude in 'V'.

        "dac_value" refers to the value between -1 and +1 that is set in a
        waveform.

        N.B. the implementation is specific to this type of AWG
        """
        if self.AWG() is None:
            logging.warning('No AWG present, returning unity scale factor.')
            return 1
        channel_amp = self.cfg_awg_channel_amplitude()
        channel_range_pp = self.cfg_awg_channel_range()
        # channel range of 5 corresponds to -2.5V to +2.5V
        scalefactor = channel_amp*(channel_range_pp/2)
        return scalefactor

    def get_amp_to_dac_val_scalefactor(self):
        if self.get_dac_val_to_amp_scalefactor() == 0:
                # Give a warning and don't raise an error as things should not
                # break because of this.
            logging.warning('AWG amp to dac scale factor is 0, check "{}" '
                            'output amplitudes'.format(self.AWG()))
            return 1
        return 1/self.get_dac_val_to_amp_scalefactor()

    def set_default_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        def_lm = self._def_lm
        LutMap = {}
        for cw_idx, cw_key in enumerate(def_lm):
            LutMap[cw_key] = 'wave_ch{}_cw{:03}'.format(
                self.cfg_awg_channel(), cw_idx)
        self.LutMap(LutMap)

    def _get_wf_name_from_cw(self, codeword: int):
        for wf_name, val in self.LutMap().items():
            if int(val[-3:]) == codeword:
                return wf_name
        raise ValueError("Codeword {} not specified"
                         " in LutMap".format(codeword))

    def _add_waveform_parameters(self):
        """
        Adds the parameters required to generate the standard waveforms

        The following prefixes are used for waveform parameters
            sq
            cz
            czd
            mcz
            custom

        """
        self.add_parameter('sq_amp', initial_value=.5,
                           # units is part of the total range of AWG8
                           label='Square pulse amplitude',
                           unit='dac value', vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('sq_length', unit='s',
                           label='Square pulse length',
                           initial_value=40e-9,
                           vals=vals.Numbers(0, 100e-6),
                           parameter_class=ManualParameter)

        self.add_parameter('cz_length', vals=vals.Numbers(),
                           unit='s', initial_value=35e-9,
                           parameter_class=ManualParameter)
        self.add_parameter('cz_lambda_2', vals=vals.Numbers(),
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('cz_lambda_3', vals=vals.Numbers(),
                           initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('cz_theta_f', vals=vals.Numbers(),
                           unit='deg',
                           initial_value=80,
                           parameter_class=ManualParameter)

        self.add_parameter('czd_length_ratio',
                           vals=vals.MultiType(vals.Numbers(0, 1),
                                               vals.Enum('auto')),
                           initial_value=0.5,
                           docstring='When using a net-zero pulse, this '
                           'parameter is used to determine the length ratio'
                           ' of the positive and negative parts of the pulse.'
                           'If this is set to "auto", the ratio will be '
                           'automatically determined to ensure the integral '
                           'of the net-zero pulse is close to zero.',
                           parameter_class=ManualParameter)
        self.add_parameter(
            'czd_lambda_2',
            docstring='lambda_2 parameter of the negative part of the cz pulse'
            ' if set to np.nan will default to the value of the main parameter',
            vals=vals.MultiType(vals.Numbers(), NP_NANs()),
            initial_value=np.nan,
            parameter_class=ManualParameter)

        self.add_parameter(
            'czd_lambda_3',
            docstring='lambda_3 parameter of the negative part of the cz pulse'
            ' if set to np.nan will default to the value of the main parameter',
            vals=vals.MultiType(vals.Numbers(), NP_NANs()),
            initial_value=np.nan,
            parameter_class=ManualParameter)
        self.add_parameter(
            'czd_theta_f',
            docstring='theta_f parameter of the negative part of the cz pulse'
            ' if set to np.nan will default to the value of the main parameter',
            vals=vals.MultiType(vals.Numbers(), NP_NANs()),
            unit='deg',
            initial_value=np.nan,
            parameter_class=ManualParameter)

        self.add_parameter('cz_phase_corr_length', unit='s',
                           initial_value=5e-9, vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('cz_phase_corr_amp',
                           unit='dac value',
                           initial_value=0, vals=vals.Numbers(),
                           parameter_class=ManualParameter)

        self.add_parameter('czd_amp_ratio',
                           docstring='Amplitude ratio for double sided CZ gate',
                           initial_value=1,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)

        self.add_parameter('czd_amp_offset',
                           docstring='used to add an offset to the negative '
                           ' pulse that is used in the net-zero cz gate',
                           initial_value=0,
                           unit='dac value',
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)

        self.add_parameter(
            'czd_net_integral', docstring='Used determine what the integral of'
            ' the CZ waveform should evaluate to. This is realized by adding'
            ' an offset to the phase correction pulse.\nBy setting this '
            'parameter to np.nan no offset correction is performed.',
            initial_value=np.nan,
            unit='dac value * samples',
            vals=vals.MultiType(vals.Numbers(), NP_NANs()),
            parameter_class=ManualParameter)
        self.add_parameter('disable_cz_only_z',
                           initial_value=False,
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)

        self.add_parameter('czd_double_sided',
                           initial_value=False,
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)

        self.add_parameter(
            'czd_signs', initial_value=['+', '-'],
            docstring='Used to determine the sign of the two parts of the '
            'double sided CZ pulse. This should be a list of two elements,'
            ' where "+" is a positive pulse, "-" a negative amplitude and "0" '
            'a disabled pulse.',
            vals=vals.Lists(vals.Enum('+', '-', 0)),
            parameter_class=ManualParameter)

        self.add_parameter('mcz_nr_of_repeated_gates',
                           initial_value=1, vals=vals.PermissiveInts(1, 40),
                           parameter_class=ManualParameter)
        self.add_parameter('mcz_gate_separation', unit='s',
                           label='Gate separation',  initial_value=0,
                           docstring=('Separtion between the start of CZ gates'
                                      'in the "multi_cz" gate'),
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(min_value=0))

        self.add_parameter(
            'custom_wf',
            initial_value=np.array([]),
            label='Custom waveform',
            docstring=('Specifies a custom waveform, note that '
                       '`custom_wf_length` is used to cut of the waveform if'
                       'it is set.'),
            parameter_class=ManualParameter,
            vals=vals.Arrays())
        self.add_parameter(
            'custom_wf_length',
            unit='s',
            label='Custom waveform length',
            initial_value=np.inf,
            docstring=('Used to determine at what sample the custom waveform '
                       'is forced to zero. This is used to facilitate easy '
                       'cryoscope measurements of custom waveforms.'),
            parameter_class=ManualParameter,
            vals=vals.Numbers(min_value=0))

    #################################
    #  Waveform generation methods  #
    #################################

    def generate_standard_waveforms(self):
        """
        Generates all the standard waveforms and populates self._wave_dict
        """
        self._wave_dict = {}
        # N.B. the  naming convention ._gen_{waveform_name} must be preserved
        # as it is used in the load_waveform_onto_AWG_lookuptable method.
        self._wave_dict['i'] = self._gen_i()
        self._wave_dict['square'] = self._gen_square()
        self._wave_dict['park'] = self._gen_park()

        # FIXME: reenable this
        self._wave_dict['cz'] = self._gen_cz()
        self._wave_dict['cz_z'] = self._gen_cz_z(regenerate_cz=False)

        self._wave_dict['idle_z'] = self._gen_idle_z()
        self._wave_dict['custom_wf'] = self._gen_custom_wf()
        self._wave_dict['multi_square'] = self._gen_multi_square(
            regenerate_square=False)
        # multi_cz is used because there is no real-time flux correction yet
        self._wave_dict['multi_cz'] = self._gen_multi_cz(regenerate_cz=False)
        self._wave_dict['multi_idle_z'] = self._gen_multi_idle_z(
            regenerate_cz=False)

    def _gen_i(self):
        return np.zeros(42)

    def _gen_square(self):
        return wf.single_channel_block(
            amp=self.sq_amp(), length=self.sq_length(),
            sampling_rate=self.sampling_rate(), delay=0)

    def _gen_park(self):
        return np.zeros(42)

    def _gen_cz(self):
        """
        Generates the CZ waveform.
        """

        dac_scalefactor = self.get_amp_to_dac_val_scalefactor()
        eps_i = self.calc_amp_to_eps(0, state_A='11', state_B='02')
        # Beware theta in radian!
        theta_i = wfl.eps_to_theta(eps_i, g=self.q_J2())

        if not self.czd_double_sided():
            CZ_theta = wfl.martinis_flux_pulse(
                self.cz_length(), theta_i=theta_i,
                theta_f=np.deg2rad(self.cz_theta_f()),
                lambda_2=self.cz_lambda_2(), lambda_3=self.cz_lambda_3())
            CZ_eps = wfl.theta_to_eps(CZ_theta, g=self.q_J2())
            CZ_amp = self.calc_eps_to_amp(CZ_eps, state_A='11', state_B='02')

            # convert amplitude in V to amplitude in awg dac value
            CZ = dac_scalefactor*CZ_amp
            return CZ

        else:
            signs = self.czd_signs()

            # Simple double sided CZ pulse implemented in most basic form.
            # repeats the same CZ gate twice and sticks it together.
            length_ratio = self.calc_net_zero_length_ratio()

            CZ_theta_A = wfl.martinis_flux_pulse(
                self.cz_length()*length_ratio, theta_i=theta_i,
                theta_f=np.deg2rad(self.cz_theta_f()),
                lambda_2=self.cz_lambda_2(), lambda_3=self.cz_lambda_3())
            CZ_eps_A = wfl.theta_to_eps(CZ_theta_A, g=self.q_J2())

            CZ_amp_A = self.calc_eps_to_amp(
                CZ_eps_A, state_A='11', state_B='02',
                positive_branch=(signs[0] == '+'))

            CZ_A = dac_scalefactor*CZ_amp_A
            if signs[0] == 0:
                CZ_A *= 0

            # Generate the second CZ pulse. If the params are np.nan, default
            # to the main parameter
            if not np.isnan(self.czd_theta_f()):
                d_theta_f = self.czd_theta_f()
            else:
                d_theta_f = self.cz_theta_f()

            if not np.isnan(self.czd_lambda_2()):
                d_lambda_2 = self.czd_lambda_2()
            else:
                d_lambda_2 = self.cz_lambda_2()
            if not np.isnan(self.czd_lambda_3()):
                d_lambda_3 = self.czd_lambda_3()
            else:
                d_lambda_3 = self.cz_lambda_3()

            CZ_theta_B = wfl.martinis_flux_pulse(
                self.cz_length()*(1-length_ratio), theta_i=theta_i,
                theta_f=np.deg2rad(d_theta_f),
                lambda_2=d_lambda_2, lambda_3=d_lambda_3)
            CZ_eps_B = wfl.theta_to_eps(CZ_theta_B, g=self.q_J2())
            CZ_amp_B = self.calc_eps_to_amp(
                CZ_eps_B, state_A='11', state_B='02',
                positive_branch=(signs[1] == '+'))

            CZ_B = dac_scalefactor*CZ_amp_B
            if signs[1] == 0:
                CZ_B *= 0
            # Combine both halves of the double sided CZ gate
            amp_rat = self.czd_amp_ratio()
            waveform = np.concatenate(
                [CZ_A, amp_rat*CZ_B + self.czd_amp_offset()])

            return waveform

    def _get_phase_corr_offset(self):
        wf_integral = np.sum(self._wave_dict['cz'])
        corr_samples = self.sampling_rate()*self.cz_phase_corr_length()

        if np.isnan(self.czd_net_integral()):
            offset = 0
        else:
            offset = (self.czd_net_integral()-wf_integral)/corr_samples
        if abs(offset) > 0.4:
            # if this warning is raised, it is recommended to play with
            # the cz_lenght ratio parameter.
            logging.warning('net-zero offset ({:.2f}) larger than 0.4'.format(
                            offset))
        return offset

    def _gen_phase_corr(self, phase_corr_amp: float = None,
                        cz_offset_comp: bool = False):
        """
        Generates a phase correction pulse.
        Amp can be given as an argument in order to use it in the multi-pulse
        if different amps are desired.

        cz_offset_comp can be given to ensure a zero net flux
        pulse. This should be False when generating phase correction for
        an "idle" pulse.

        """
        raise NotImplementedError("Method is deprecated")
        # if phase_corr_amp is None:
        #     phase_corr_amp = self.cz_phase_corr_amp()

        # if not self.czd_double_sided():
        #     phase_corr_wf = wf.single_channel_block(
        #         amp=phase_corr_amp, length=self.cz_phase_corr_length(),
        #         sampling_rate=self.sampling_rate(), delay=0)
        #     return phase_corr_wf

        # else:
        #     block = wf.single_channel_block(
        #         amp=phase_corr_amp,
        #         length=self.cz_phase_corr_length()/2,
        #         sampling_rate=self.sampling_rate(), delay=0)

        #     if cz_offset_comp:
        #         phase_corr_offset = self._get_phase_corr_offset()
        #     else:
        #         phase_corr_offset = 0

        #     phase_corr_wf = np.concatenate(
        #         [block, -1*block]) + phase_corr_offset
        #     return phase_corr_wf

    def _gen_cz_z(self, regenerate_cz=True):
        if regenerate_cz:
            self._wave_dict['cz'] = self._gen_cz()

        # Commented out snippet is old (deprecated ) phase corr 19/6/2018 MAR
        # phase_corr = self._gen_phase_corr(cz_offset_comp=True)
        # # CZ with phase correction
        # cz_z = np.concatenate([self._wave_dict['cz'], phase_corr])

        cz_z = self._get_phase_corrected_pulse(base_wf=self._wave_dict['cz'])

        return cz_z

    def _gen_multi_square(self, regenerate_square=True):
        """
        Composite waveform containing multiple cz gates
        """
        if regenerate_square:
            self._wave_dict['square'] = self._gen_square()

        max_nr_samples = int(self.cfg_max_wf_length()*self.sampling_rate())

        waveform = np.zeros(max_nr_samples)
        for i in range(self.mcz_nr_of_repeated_gates()):
            sample_start_idx = int(self.mcz_gate_separation() *
                                   self.sampling_rate())*i
            sq = self._wave_dict['square']
            try:
                waveform[sample_start_idx:sample_start_idx+len(sq)] += sq
            except ValueError as e:
                logging.warning('Could not add square pulse {} in {}'.format(
                    i, self.name))
                logging.warning(e)
                break
        return waveform

    def _gen_multi_cz(self, regenerate_cz=True):
        """
        Composite waveform containing multiple cz gates
        """
        if regenerate_cz:
            self._wave_dict['cz'] = self._gen_cz()
            self._wave_dict['cz_z'] = self._gen_cz_z()

        max_nr_samples = int(self.cfg_max_wf_length()*self.sampling_rate())

        waveform = np.zeros(max_nr_samples)
        for i in range(self.mcz_nr_of_repeated_gates()):
            cz_z = self._wave_dict['cz_z']
            sample_start_idx = int(self.mcz_gate_separation() *
                                   self.sampling_rate())*i
            try:
                waveform[sample_start_idx:sample_start_idx+len(cz_z)] += cz_z
            except ValueError as e:
                logging.warning('Could not add cz_z pulse {} in {}'.format(
                    i, self.name))
                logging.warning(e)
                break
        # CZ with phase correction
        return waveform

    def _gen_custom_wf(self):
        base_wf = copy(self.custom_wf())

        if self.custom_wf_length() != np.inf:
            # cuts of the waveform at a certain length by setting
            # all subsequent samples to 0.
            max_sample = int(self.custom_wf_length()*self.sampling_rate())
            base_wf[max_sample:] = 0
        return base_wf

    def _gen_composite_wf(self, primitive_waveform_name: str,
                          time_tuples: list):
        """
        Generates a composite waveform based on a timetuple.
        Only relies on the first element of the timetuple which is expected
        to be the starting time of the pulse in clock cycles.


        N.B. No waveforms are regenerated here!
        This relies on the base waveforms being up to date in self._wave_dict

        """

        max_nr_samples = int(self.cfg_max_wf_length()*self.sampling_rate())
        waveform = np.zeros(max_nr_samples)

        for i, tt in enumerate(time_tuples):
            t_start = clocks_to_s(tt[0])
            sample = self.time_to_sample(t_start)
            if sample > max_nr_samples:
                raise ValueError('Waveform longer than max wf lenght')

            if (primitive_waveform_name == 'cz_z' or
                    primitive_waveform_name == 'idle_z'):
                phase_corr = wf.single_channel_block(
                    amp=self.get('cz_phase_corr_amp'),
                    length=self.cz_phase_corr_length(),
                    sampling_rate=self.sampling_rate(), delay=0)
                # phase_corr = wf.single_channel_block(
                #     amp=self.get('mcz_phase_corr_amp_{}'.format(i+1)),
                #     length=self.cz_phase_corr_length(),
                #     sampling_rate=self.sampling_rate(), delay=0)
                if primitive_waveform_name == 'cz_z':
                    prim_wf = np.concatenate(
                        [self._wave_dict['cz'], phase_corr])
                elif primitive_waveform_name == 'idle_z':
                    prim_wf = np.concatenate(
                        [np.zeros(len(self._wave_dict['cz'])), phase_corr])
            else:
                prim_wf = self._wave_dict[primitive_waveform_name]
            waveform[sample:sample+len(prim_wf)] += prim_wf

        return waveform

    def _gen_idle_z(self):
        idle_z = self._get_phase_corrected_pulse(
            base_wf=np.zeros(int(self.cz_length()*self.sampling_rate()+1)))

        return idle_z

    def _gen_multi_idle_z(self, regenerate_cz=False):
        """
        Composite waveform containing multiple cz gates
        """
        if regenerate_cz:
            self._wave_dict['idle_z'] = self._gen_cz(
                regenerate_cz=regenerate_cz)
        idle_z = self._wave_dict['idle_z']
        max_nr_samples = int(self.cfg_max_wf_length()*self.sampling_rate())

        waveform = np.zeros(max_nr_samples)
        for i in range(self.mcz_nr_of_repeated_gates()):
            # if self.mcz_identical_phase_corr():
            # phase_corr = self._gen_phase_corr(cz_offset_comp=False)
            # else:
            #     phase_corr = wf.single_channel_block(
            #         amp=self.get('mcz_phase_corr_amp_{}'.format(i+1)),
            #         length=self.cz_phase_corr_length(),
            #         sampling_rate=self.sampling_rate(), delay=0)
            # idle_z = np.concatenate([np.zeros(len(self._wave_dict['cz'])),
            #                        phase_corr])
            sample_start_idx = int(self.mcz_gate_separation() *
                                   self.sampling_rate())*i
            try:
                waveform[sample_start_idx:sample_start_idx +
                         len(idle_z)] += idle_z
            except ValueError as e:
                logging.warning('Could not add idle_z pulse {} in {}'.format(
                    i, self.name))
                logging.warning(e)
                break
        # CZ with phase correction
        return waveform

    ###########################################################
    #  Waveform generation net-zero phase correction methods  #
    ###########################################################
    def _calc_modified_wf(self, base_wf, a_i, corr_samples):

        if not np.isnan(self.czd_net_integral()):
            curr_int = np.sum(base_wf)
            corr_int = self.czd_net_integral()-curr_int
            #corr_pulse = phase_corr_triangle(
            #    int_val=corr_int, nr_samples=corr_samples)

            corr_pulse = phase_corr_square(
                int_val=corr_int, nr_samples=corr_samples)
            if np.max(corr_pulse) > 0.5:
                logging.warning('net-zero integral correction({:.2f}) larger than 0.5'.format(
                    np.max(corr_pulse)))
        else:
            corr_pulse = np.zeros(corr_samples)

        corr_pulse += phase_corr_sine_series(a_i, corr_samples)

        modified_wf = np.concatenate([base_wf, corr_pulse])
        return modified_wf

    def _phase_corr_cost_func(self, base_wf, a_i, corr_samples,
                              print_result=False):
        """
        The cost function of the cz_z waveform is designed to meet
        the following criteria
        1. Net-zero character of waveform
            Integral of wf = 0
        2. Single qubit phase correction
            Integral of phase_corr_part**2 = desired constant
        3. Target_wf ends at 0
        4. No-distortions present after cutting of waveform
            predistorted_wf ends at 0 smoothly (as many derivatives as possible 0)

        5. Minimize the maximum amplitude
            Prefer small non-violent pulses
        """
        # samples to quanitify leftover distortions
        tail_samples = 500

        target_wf = self._calc_modified_wf(
            base_wf, a_i=a_i, corr_samples=corr_samples)
        if self.cfg_distort():
            k0 = self.instr_distortion_kernel.get_instr()
            predistorted_wf = k0.distort_waveform(target_wf,
                                                  len(target_wf)+tail_samples)
        else:
            predistorted_wf = target_wf

        # 2. Phase correction pulse
        phase_corr_int = np.sum(
            target_wf[len(base_wf):len(base_wf)+corr_samples]**2)/corr_samples
        cv_2 = ((phase_corr_int - self.cz_phase_corr_amp()**2)*1000)**2

        # 4. No-distortions present after cutting of waveform
        cv_4 = np.sum(abs(predistorted_wf[-tail_samples+50:]))*20
        # 5. no violent waveform
        cv_5 = np.max(abs(target_wf)*100)**2

        cost_val = cv_2 + cv_4+cv_5

    #     if print_result:
    #         print("Cost function value")''

    # #         print("cv_1 net_zero_character: {:.6f}".format(cv_1))
    #         print("cv_2 phase corr pulse  : {:.6f}".format(cv_2))
    # #         print("cv_3 ends at 0         : {:.6f}".format(cv_3))
    #         print("cv_4 distortions tail  : {:.6f}".format(cv_4))
    #         print("cv_5 non violent       : {:.6f}".format(cv_5))

        return cost_val

    def _get_phase_corrected_pulse(self, base_wf):
        """
        Creates a phase correction pulse using a cosine with an offset
        to correct any picked up phase.

        Two properties are obeyed.
            - The net-integral (if net-zero) is set to 'czd_net_integral'
            - The amplitude of the cosine is set to 'cz_phase_corr_amp'
        """
        corr_samples = int(self.cz_phase_corr_length()*self.sampling_rate())

        if self.czd_double_sided() and not np.isnan(self.czd_net_integral()):
            curr_int = np.sum(base_wf)
            corr_int = self.czd_net_integral()-curr_int

#            corr_pulse = phase_corr_triangle(
#                int_val=corr_int, nr_samples=corr_samples)
            corr_pulse = phase_corr_square(
                int_val=corr_int, nr_samples=corr_samples)
            if np.max(corr_pulse) > 0.5:
                logging.warning('net-zero integral correction({:.2f}) larger than 0.4'.format(
                    np.max(corr_pulse)))
        else:
            corr_pulse = np.zeros(corr_samples)

        if self.czd_double_sided():
            corr_pulse += phase_corr_sine_series([self.cz_phase_corr_amp()],
                                                 corr_samples)
        else:
            corr_pulse += phase_corr_sine_series_half([self.cz_phase_corr_amp()],
                                                      corr_samples)

        if self.disable_cz_only_z():
            modified_wf = np.concatenate([base_wf*0, corr_pulse])
        else:
            modified_wf = np.concatenate([base_wf, corr_pulse])
        return modified_wf

    #################################
    #  Waveform loading methods     #
    #################################

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool = False):
        """
        Loads a specific waveform to the AWG
        """
        if regenerate_waveforms:
            # only regenerate the one waveform that is desired
            gen_wf_func = getattr(self, '_gen_{}'.format(waveform_name))
            self._wave_dict[waveform_name] = gen_wf_func()

        waveform = self._wave_dict[waveform_name]
        codeword = self.LutMap()[waveform_name]

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

        self.AWG.get_instr().set(codeword, waveform)

    def load_waveforms_onto_AWG_lookuptable(
            self, regenerate_waveforms: bool = True, stop_start: bool = True,
            force_load_sequencer_program: bool = False):
        """
        Loads all waveforms specified in the LutMap to an AWG for both this
        LutMap and the partner LutMap.

        Args:
            regenerate_waveforms (bool): if True calls
                generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.


        Because of realtime loading vs the DIO sequencer program the uploading
        flow for the AWG8 is slightly different.
        """

        # Generate the waveforms and link them to the AWG8 parameters
        lutmans = [self]
        # If statement is here because it should be possible to function
        # independently
        if self.instr_partner_lutman() is None:
            logging.warning('No partner_lutman specified')
        else:
            lutmans += [self.instr_partner_lutman.get_instr()]

        if self.cfg_operating_mode() != 'Codeword_normal':
            # Regenerating only one waveform in this case, see below
            regenerate_waveforms_realtime = regenerate_waveforms
            regenerate_waveforms = False

        for lm in lutmans:
            if regenerate_waveforms:
                lm.generate_standard_waveforms()
                for waveform_name, lookuptable in lm.LutMap().items():
                    lm.load_waveform_onto_AWG_lookuptable(waveform_name)

        # Uploading the codeword program if required
        if self._program_hash_differs() or force_load_sequencer_program:
            # FIXME: List of conditions in which reloading is required
            # - program hash differs
            # - max waveform length has changed
            # N.B. these other conditions need to be included here.
            # This ensures only the channels that are relevant get reconfigured
            awg_nr = (self.cfg_awg_channel()-1)//2
            self._upload_codeword_program(awg_nr)

        if self.cfg_operating_mode() == 'Codeword_normal':
            for waveform_name in self.LutMap().keys():
                self.load_waveform_realtime(
                    waveform_name=waveform_name,
                    wf_nr=None, regenerate_waveforms=False)
        else:
            # Only load one waveform and do it in realtime.
            cw_idx = int(self.cfg_operating_mode()[-2:])
            waveform_name = self._get_wf_name_from_cw(cw_idx)
            self.load_waveform_realtime(
                waveform_name=waveform_name,
                wf_nr=None, regenerate_waveforms=regenerate_waveforms_realtime)
        #updating channel amplitude and range
        self.cfg_awg_channel_amplitude()
        self.cfg_awg_channel_range()
        self._update_expected_program_hash()

    def _generate_single_cw_program(self, cw_idx):
        devname = self.AWG.get_instr()._devname
        ch = self.cfg_awg_channel() - (self.cfg_awg_channel()+1) % 2
        awg_single_wf_program = (
            '\nwhile (1) {\n' +
            'waitDIOTrigger();\n' +
            'playWave("{}_wave_ch{}_cw{:03}", '.format(devname, ch, cw_idx) +
            '"{}_wave_ch{}_cw{:03}");'.format(devname, ch+1, cw_idx) + '\n}')
        return awg_single_wf_program

    def _upload_codeword_program(self, awg_nr):
        awg = self.AWG.get_instr()
        if self.cfg_operating_mode() == 'Codeword_normal':
            awg.upload_codeword_program(awgs=[awg_nr])
        else:
            cw_idx = int(self.cfg_operating_mode()[-2:])
            single_cw_program = self._generate_single_cw_program(cw_idx)
            awg.configure_awg_from_string(awg_nr, single_cw_program)
            awg.configure_codeword_protocol()
            awg.start()

    def _append_zero_samples(self, waveform):
        """
        Helper method to ensure waveforms have the desired length
        """
        length_samples = roundup1024(int(self.sampling_rate()*self.cfg_max_wf_length()))
        extra_samples = length_samples - len(waveform)
        if extra_samples >= 0:
            y_sig = np.concatenate([waveform, np.zeros(extra_samples)])
        else:
            y_sig = waveform[:extra_samples]
        return y_sig

    def _update_expected_program_hash(self):
        """
        Updates the expected AWG sequencer program hash with the current
        hash of the sequencer program. This is intended to be called after
        setting the hash.
        """
        awg_nr = (self.cfg_awg_channel()-1)//2
        hash = self.AWG.get_instr().get(
            'awgs_{}_sequencer_program_crc32_hash'.format(awg_nr))
        self._awgs_fl_sequencer_program_expected_hash(hash)

    def _program_hash_differs(self)-> bool:
        """
        Args:
            --
        Returns:
            hash_different (bool): returns True if one of the hashes does not
                correspond to the expected hash.

        Compares current AWG sequencer program hash for the relevant channels
        with the expected program hash and, returns True if one ore more
        hashes do not match, indicating that uploading the programs is
        required.
        """
        awg_nr = (self.cfg_awg_channel()-1)//2
        hash = self.AWG.get_instr().get(
            'awgs_{}_sequencer_program_crc32_hash'.format(awg_nr))
        expected_hash = self._awgs_fl_sequencer_program_expected_hash()
        hash_differs = (hash != expected_hash)

        return hash_differs

    def load_composite_waveform_onto_AWG_lookuptable(
        self,
            primitive_waveform_name: str,
            time_tuples: list,
            codeword: int):
        """
        Creates a composite waveform based on time_tuples extracted from a qisa
        file.
        """
        waveform_name = 'comp_{}_cw{:03}'.format(primitive_waveform_name,
                                                 codeword)

        # assigning for post loading rendering purposes
        self._wave_dict[waveform_name] = self._gen_composite_wf(
            primitive_waveform_name,  time_tuples)
        waveform = self._wave_dict[waveform_name]

        codeword = 'wave_ch{}_cw{:03}'.format(self.cfg_awg_channel(),
                                              codeword)

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

        self.AWG.get_instr().set(codeword, waveform)

    def load_waveform_realtime(self, waveform_name: str,
                               wf_nr: int = None,
                               regenerate_waveforms: bool = True):
        """
        Args:
            waveform_name:        (str) : name of the waveform
            wf_nr                 (int) : what codeword to load the pulse onto
                if set to None, will determine awg_nr based on self.LutMap
            regenerate_waveforms (bool) : if True regenerates all waveforms

        """

        if wf_nr is None:
            wf_nr = int(self.LutMap()[waveform_name][-3:])

        if regenerate_waveforms:
            gen_wf_func = getattr(self, '_gen_{}'.format(waveform_name))
            waveform = gen_wf_func()
            if self.cfg_append_compensation():
                waveform = self.add_compensation_pulses(waveform)
            if self.cfg_distort():
                waveform = self.distort_waveform(waveform)
                self._wave_dict_dist[waveform_name] = waveform
            else:
                waveform = self._append_zero_samples(waveform)
                self._wave_dict_dist[waveform_name] = waveform

        waveform = self._wave_dict_dist[waveform_name]
        codeword = self.LutMap()[waveform_name]
        self.AWG.get_instr().set(codeword, waveform)

        if self.instr_partner_lutman() is None:
            logging.warning('no partner lutman specified')
            other_waveform = np.zeros(len(waveform))
        else:
            partner_lm = self.instr_partner_lutman.get_instr()
            prtnr_wf_name = partner_lm._get_wf_name_from_cw(codeword=wf_nr)
            if regenerate_waveforms:
                gen_wf_func = getattr(partner_lm,
                                      '_gen_{}'.format(prtnr_wf_name))
                prtnr_wf = gen_wf_func()

                if partner_lm.cfg_append_compensation():
                    prtnr_wf = partner_lm.add_compensation_pulses(prtnr_wf)

                if partner_lm.cfg_distort():
                    prtnr_wf = partner_lm.distort_waveform(prtnr_wf)
                    partner_lm._wave_dict_dist[prtnr_wf_name] = prtnr_wf

            other_waveform = partner_lm._wave_dict_dist[prtnr_wf_name]

        if len(waveform) != len(other_waveform):
            other_waveform = np.zeros(len(waveform))
            msg = ('Lenghts of waveforms "{}" ({}) and "{}" ({}) do '
                   'not match.'.format(waveform_name, len(waveform),
                                       prtnr_wf_name, len(other_waveform)))
            logging.warning(msg)
            logging.warning('Setting "other waveform" to array of zeros.')

        awg_ch = self.cfg_awg_channel()-1  # -1 is to account for starting at 1
        ch_pair = awg_ch % 2
        awg_nr = awg_ch//2

        if ch_pair == 0:
            waveforms = (waveform, other_waveform)
        else:
            waveforms = (other_waveform, waveform)

        self.AWG.get_instr().upload_waveform_realtime(
            w0=waveforms[0], w1=waveforms[1], awg_nr=awg_nr, wf_nr=wf_nr)

    def add_compensation_pulses(self, waveform):
        """
        Adds the inverse of the pulses at the end of a waveform to
        ensure flux discharging.
        """
        wf = np.array(waveform)  # catches a rare bug when wf is a list
        delay_samples = np.zeros(int(self.sampling_rate() *
                                     self.cfg_compensation_delay()))
        comp_wf = np.concatenate([wf, delay_samples, -1*wf])
        return comp_wf

    def distort_waveform(self, waveform, inverse=False):
        """
        Modifies the ideal waveform to correct for distortions and correct
        fine delays.
        Distortions are corrected using the kernel object.
        """
        k = self.instr_distortion_kernel.get_instr()

        # Prepend zeros to delay waveform to correct for fine timing
        delay_samples = int(self.cfg_pre_pulse_delay()*self.sampling_rate())
        waveform = np.pad(waveform, (delay_samples, 0), 'constant')

        # duck typing the distort waveform method
        if hasattr(k, 'distort_waveform'):
            distorted_waveform = k.distort_waveform(
                waveform,
                length_samples=int(
                    roundup1024(self.cfg_max_wf_length()*self.sampling_rate())),
                inverse=inverse)
        else:  # old kernel object does not have this method
            if inverse:
                raise NotImplementedError()
            distorted_waveform = k.convolve_kernel(
                [k.kernel(), waveform],
                length_samples=int(self.cfg_max_wf_length() *
                                   self.sampling_rate()))
        return distorted_waveform

    #################################
    #  Plotting methods            #
    #################################

    def plot_cz_trajectory(self, axs=None, show=True,
                           extra_plot_samples: int = 50):
        """
        Plots the cz trajectory in frequency space.
        """
        if axs is None:
            f, axs = plt.subplots(figsize=(5, 7), nrows=3, sharex=True)
        nr_plot_samples = int((self.cz_length()+self.cz_phase_corr_length()) *
                              self.sampling_rate() + extra_plot_samples)

        dac_amps = self._wave_dict['cz_z'][:nr_plot_samples]
        t = np.arange(0, len(dac_amps))*1/self.sampling_rate()

        CZ_amp = dac_amps*self.get_dac_val_to_amp_scalefactor()
        CZ_eps = self.calc_amp_to_eps(CZ_amp, '11', '02')
        CZ_theta = wfl.eps_to_theta(CZ_eps, self.q_J2())

        axs[0].plot(t, np.rad2deg(CZ_theta), marker='.')
        axs[0].fill_between(t, np.rad2deg(CZ_theta), color='C0', alpha=.5)
        set_ylabel(axs[0], r'$\theta$', 'deg')

        axs[1].plot(t, CZ_eps, marker='.')
        axs[1].fill_between(t, CZ_eps, color='C0', alpha=.5)
        set_ylabel(axs[1], r'$\epsilon_{11-02}$', 'Hz')

        axs[2].plot(t, CZ_amp, marker='.')
        axs[2].fill_between(t, CZ_amp, color='C0', alpha=.1)
        set_xlabel(axs[2], 'Time', 's')
        set_ylabel(axs[2], r'Amp.', 'V')
        # axs[2].set_ylim(-1, 1)
        axs[2].axhline(0, lw=.2, color='grey')
        CZ_amp_pred = self.distort_waveform(CZ_amp)[:len(CZ_amp)]
        axs[2].plot(t, CZ_amp_pred, marker='.')
        axs[2].fill_between(t, CZ_amp_pred, color='C1', alpha=.3)
        if show:
            plt.show()
        return axs

    def plot_level_diagram(self, ax=None, show=True):
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
        freqs = self.calc_amp_to_freq(amps, state='01')
        ax.plot(amps, freqs, label='$f_{01}$')
        ax.text(0, self.calc_amp_to_freq(0, state='01'), '01', color='C0',
                ha='left', va='bottom', clip_on=True)

        freqs = self.calc_amp_to_freq(amps, state='02')
        ax.plot(amps, freqs, label='$f_{02}$')
        ax.text(0, self.calc_amp_to_freq(0, state='02'), '02', color='C1',
                ha='left', va='bottom', clip_on=True)

        freqs = self.calc_amp_to_freq(amps, state='10')
        ax.plot(amps, freqs, label='$f_{10}$')
        ax.text(0, self.calc_amp_to_freq(0, state='10'), '10', color='C2',
                ha='left', va='bottom', clip_on=True)

        freqs = self.calc_amp_to_freq(amps, state='11')
        ax.plot(amps, freqs, label='$f_{11}$')
        ax.text(0, self.calc_amp_to_freq(0, state='11'), '11', color='C3',
                ha='left', va='bottom', clip_on=True)

        # 2. Annotating feature of interest
        ax.axvline(0, 0, 1e10, linestyle='dotted', c='grey')

        amp_J2 = self.calc_eps_to_amp(0, state_A='11', state_B='02')
        amp_J1 = self.calc_eps_to_amp(0, state_A='10', state_B='01')

        ax.axvline(amp_J2, ls='--', lw=1, c='C4')
        ax.axvline(amp_J1, ls='--', lw=1, c='C6')

        f_11_02 = self.calc_amp_to_freq(amp_J2, state='11')
        ax.plot([amp_J2], [f_11_02],
                color='C4', marker='o', label='11-02')
        ax.text(amp_J2, f_11_02,
                '({:.4f},{:.2f})'.format(amp_J2, f_11_02*1e-9),
                color='C4',
                ha='left', va='bottom', clip_on=True)

        f_10_01 = self.calc_amp_to_freq(amp_J1, state='01')

        ax.plot([amp_J1], [f_10_01],
                color='C5', marker='o', label='10-01')
        ax.text(amp_J1, f_10_01,
                '({:.4f},{:.2f})'.format(amp_J1, f_10_01*1e-9),
                color='C5', ha='left', va='bottom', clip_on=True)

        # 3. Adding legend etc.
        title = ('Calibration visualization\n{}\nchannel {}'.format(
            self.AWG(), self.cfg_awg_channel()))
        leg = ax.legend(title=title, loc=(1.05, .3))
        leg._legend_box.align = 'center'
        set_xlabel(ax, 'AWG amplitude', 'V')
        set_ylabel(ax, 'Frequency', 'Hz')
        ax.set_xlim(-2.5, 2.5)

        ax.set_ylim(0, self.calc_amp_to_freq(0, state='02')*1.1)

        # 4. Add a twin x-axis to denote scale in dac amplitude
        dac_val_axis = ax.twiny()
        dac_ax_lims = np.array(ax.get_xlim()) * \
            self.get_amp_to_dac_val_scalefactor()
        dac_val_axis.set_xlim(dac_ax_lims)
        set_xlabel(dac_val_axis, 'AWG amplitude', 'dac')

        dac_val_axis.axvspan(1, 1000, facecolor='.5', alpha=0.5)
        dac_val_axis.axvspan(-1000, -1, facecolor='.5', alpha=0.5)
        # get figure is here in case an axis object was passed as input
        f = ax.get_figure()
        f.subplots_adjust(right=.7)
        if show:
            plt.show()
        return ax


class QWG_Flux_LutMan(AWG8_Flux_LutMan):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._wave_dict_dist = dict()
        self.sampling_rate(1e9)
        self.add_parameter('cfg_oldstyle_kernel_enabled',
                           initial_value=False,
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool = False):
        """
        Loads a specific waveform to the AWG
        """
        if regenerate_waveforms:
            # only regenerate the one waveform that is desired
            gen_wf_func = getattr(self, '_gen_{}'.format(waveform_name))
            self._wave_dict[waveform_name] = gen_wf_func()

        waveform = self._wave_dict[waveform_name]
        codeword = self.LutMap()[waveform_name]

        if self.cfg_append_compensation():
            waveform = self.add_compensation_pulses(waveform)

        if self.cfg_distort():
            waveform = self.distort_waveform(waveform)
            self._wave_dict_dist[waveform_name] = waveform
        else:
            waveform = self._append_zero_samples(waveform)
            self._wave_dict_dist[waveform_name] = waveform

        # N.B. method identical to AWG8 version with the exception 
        # of AWG.stop() and AWG.start() 
        self.AWG.get_instr().stop()
        self.AWG.get_instr().set(codeword, waveform)
        self.AWG.get_instr().start()

    def distort_waveform(self, waveform):
        """
        Modifies the ideal waveform to correct for distortions and correct
        fine delays.
        Distortions are corrected using the kernel object.
        Modified to implement also normal kernels.
        """

        # FIXME: this function looks identical to the one in the parent class 
        # maybe this should be deleted/cleaned up? -MAR Jan 2019
        k = self.instr_distortion_kernel.get_instr()

        # Prepend zeros to delay waveform to correct for fine timing
        delay_samples = int(self.cfg_pre_pulse_delay()*self.sampling_rate())
        waveform = np.pad(waveform, (delay_samples, 0), 'constant')

        # duck typing the distort waveform method
        if hasattr(k, 'distort_waveform'):
            distorted_waveform = k.distort_waveform(
                waveform,
                length_samples=int(
                    self.cfg_max_wf_length()*self.sampling_rate()))
        else:  # old kernel object does not have this method
            distorted_waveform = k.convolve_kernel(
                [k.kernel(), waveform],
                length_samples=int(self.cfg_max_wf_length() *
                                   self.sampling_rate()))

        if self.cfg_oldstyle_kernel_enabled():
            # hotfix: to distort adding abounce model
            kernel_oldstyle = self.kernel_oldstyle
            distorted_waveform = self.convolve_kernel(
                [kernel_oldstyle, distorted_waveform],
                length_samples=int(self.cfg_max_wf_length() *
                                   self.sampling_rate()))

        return distorted_waveform

    def convolve_kernel(self, kernel_list, length_samples=None):
        """
        kernel_list : (list of arrays)
        length_samples      : (int) maximum for convolution
        Performs a convolution of different kernels
        """
        kernels = kernel_list[0]
        for k in kernel_list[1:]:
            kernels = np.convolve(k, kernels)[
                :max(len(k), int(length_samples))]
        if length_samples is not None:
            return kernels[:int(length_samples)]
        return kernels

    def get_dac_val_to_amp_scalefactor(self):
        """
        Returns the scale factor to transform an amplitude in 'dac value' to an
        amplitude in 'V'.

        N.B. the implementation is specific to this type of AWG (QWG)
        """
        AWG = self.AWG.get_instr()
        awg_ch = self.cfg_awg_channel()

        channel_amp = AWG.get('ch{}_amp'.format(awg_ch))
        scalefactor = channel_amp
        return scalefactor



#########################################################################
# Convenience functions below
#########################################################################


def phase_corr_triangle(int_val, nr_samples):
    """
    Creates an offset triangle with desired integrated value
    """
    x = np.arange(nr_samples)
    # nr_samples+1 is because python counting starts at 0
    b = 2*int_val/(nr_samples+1)
    a = -b/nr_samples
    y = a*x+b
    return y


def phase_corr_square(int_val, nr_samples):
    """
    Creates an offset square with desired integrated value
    """
    x = np.arange(nr_samples)
    # nr_samples+1 is because python counting starts at 0
    a = int_val/(nr_samples+1)
    y = a*np.ones(len(x))
    return y


def phase_corr_sine_series(a_i, nr_samples):
    """
    Phase correction pulse as a fourier sine series.

    The integeral (sum) of this waveform is
    gauranteed to be equal to zero (within rounding error)
    by the choice of function.
    """
    x = np.linspace(0, 2*np.pi, nr_samples)
    s = np.zeros(nr_samples)

    for i, a in enumerate(a_i):
        s += a*np.sin((i+1)*x)
    return s


def phase_corr_sine_series_half(a_i, nr_samples):
    """
    Phase correction pulse as a fourier sine series.

    The integeral (sum) of this waveform is
    gauranteed to be equal to zero (within rounding error)
    by the choice of function.
    """
    x = np.linspace(0, 2*np.pi, nr_samples)
    s = np.zeros(nr_samples)

    for i, a in enumerate(a_i):
        s += a*np.sin(((i+1)*x)/2)
    return s

def roundup1024(n):
    return int(np.ceil(n/1024)*1024)