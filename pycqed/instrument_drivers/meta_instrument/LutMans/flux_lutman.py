from .base_lutman import Base_LutMan
import numpy as np
import logging
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from qcodes.utils import validators as vals
from pycqed.instrument_drivers.pq_parameters import NP_NANs
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.measurement.openql_experiments.openql_helpers import clocks_to_s
from qcodes.plots.pyqtgraph import QtPlot
import matplotlib.pyplot as plt
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel


class Base_Flux_LutMan(Base_LutMan):
    # this default lutman is if a flux pulse can be done with only one
    # other qubit. this needs to be expanded if there are more qubits
    # to interact with.
    _def_lm = ['i', 'cz_z', 'square', 'park', 'multi_cz']

    def render_wave(self, wave_name, show=True, time_units='s',
                    reload_pulses: bool=True, render_distorted_wave: bool=True,
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


class AWG8_Flux_LutMan(Base_Flux_LutMan):

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._wave_dict_dist = dict()
        self.sampling_rate(2.4e9)

    def _add_cfg_parameters(self):

        self.add_parameter(
            'polycoeffs_freq_conv',
            docstring='coefficients of the polynomial used to convert '
            'amplitude in V to detuning in Hz. N.B. it is important to '
            'include both the AWG range and channel amplitude in the params.\n'
            'In order to convert a set of cryoscope flux arc coefficients to '
            ' units of Volts they can be rescaled using [c0*sc**2, c1*sc, c2]'
            ' where sc is the desired scaling factor that includes the sq_amp '
            'used and the range of the AWG (5 in amp mode).',
            vals=vals.Arrays(),
            initial_value=np.array([0, 0, 0]),
            parameter_class=ManualParameter)

        self.add_parameter('cfg_operating_mode',
                           initial_value='Codeword',
                           vals=vals.Enum('Codeword', 'LongSeq'),
                           parameter_class=ManualParameter)
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

        self.add_parameter('cfg_max_wf_length',
                           parameter_class=ManualParameter,
                           initial_value=100e-6,
                           unit='s', vals=vals.Numbers(0, 100e-6))

    def amp_to_detuning(self, amp):
        """
        Converts amplitude to detuning in Hz.

        Requires "polycoeffs_freq_conv" to be set to the polynomial values
        extracted from the cryoscope flux arc.

        N.B. this method assumes that the polycoeffs are with respect to the
            amplitude in units of V, including rescaling due to the channel
            amplitude and range settings of the AWG8.
            See also `self.get_dac_val_to_amp_scalefactor`.

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """
        return np.polyval(self.polycoeffs_freq_conv(), amp)

    def detuning_to_amp(self, freq, positive_branch=True):
        """
        Converts detuning in Hz to amplitude.

        Requires "polycoeffs_freq_conv" to be set to the polynomial values
        extracted from the cryoscope flux arc.


        N.B. this method assumes that the polycoeffs are with respect to the
            amplitude in units of V, including rescaling due to the channel
            amplitude and range settings of the AWG8.
            See also `self.get_dac_val_to_amp_scalefactor`.

                amp_Volts = amp_dac_val * channel_amp * channel_range
        """
        # recursive allows dealing with an array of freqs
        if isinstance(freq, (list, np.ndarray)):
            return np.array([self.detuning_to_amp(
                f, positive_branch=positive_branch) for f in freq])
        p = np.poly1d(self.polycoeffs_freq_conv())
        sols = (p-freq).roots

        # sols returns 2 solutions (for a 2nd order polynomial)
        if positive_branch:
            sol = np.max(sols)
        else:
            sol = np.min(sols)

        # imaginary part is ignored, instead sticking to closest real value
        return np.real(sol)

    def get_dac_val_to_amp_scalefactor(self):
        """
        Returns the scale factor to transform an amplitude in 'dac value' to an
        amplitude in 'V'.

        "dac_value" refers to the value between -1 and +1 that is set in a
        waveform.

        N.B. the implementation is specific to this type of AWG
        """
        AWG = self.AWG.get_instr()

        awg_ch = self.cfg_awg_channel()-1  # -1 is to account for starting at 1
        awg_nr = awg_ch//2
        ch_pair = awg_ch % 2

        channel_amp = AWG.get('awgs_{}_outputs_{}_amplitude'.format(
            awg_nr, ch_pair))

        # channel range of 5 corresponds to -2.5V to +2.5V
        channel_range_pp = AWG.get('sigouts_{}_range'.format(awg_ch))
        # direct_mode = AWG.get('sigouts_{}_direct'.format(awg_ch))
        scale_factor = channel_amp*(channel_range_pp/2)
        return scale_factor

    def get_amp_to_dac_val_scale_factor(self):
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

    def _add_waveform_parameters(self):
        """
        Adds the parameters required to generate the standard waveforms
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

        self.add_parameter('czd_length_ratio', vals=vals.Numbers(0, 1),
                           initial_value=0.5,
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

        self.add_parameter('cz_freq_01_max', vals=vals.Numbers(),
                           unit='Hz', parameter_class=ManualParameter)
        self.add_parameter('cz_J2', vals=vals.Numbers(), unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('cz_freq_interaction', vals=vals.Numbers(),
                           unit='Hz',
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

        self.add_parameter('czd_double_sided',
                           initial_value=False,
                           vals=vals.Bool(),
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
        self._wave_dict['cz'] = self._gen_cz()
        self._wave_dict['cz_z'] = self._gen_cz_z(regenerate_cz=False)
        self._wave_dict['idle_z'] = self._gen_idle_z()

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

        dac_scale_factor = self.get_amp_to_dac_val_scale_factor()

        if not self.czd_double_sided():
            CZ = wf.martinis_flux_pulse(
                length=self.cz_length(),
                lambda_2=self.cz_lambda_2(),
                lambda_3=self.cz_lambda_3(),
                theta_f=self.cz_theta_f(),
                f_01_max=self.cz_freq_01_max(),
                J2=self.cz_J2(),
                f_interaction=self.cz_freq_interaction(),
                sampling_rate=self.sampling_rate(),
                return_unit='f01')
            return dac_scale_factor*self.detuning_to_amp(
                self.cz_freq_01_max() - CZ)
        else:
            # Simple double sided CZ pulse implemented in most basic form.
            # repeats the same CZ gate twice and sticks it together.
            half_CZ_A = wf.martinis_flux_pulse(
                length=self.cz_length()*self.czd_length_ratio(),
                lambda_2=self.cz_lambda_2(),
                lambda_3=self.cz_lambda_3(),
                theta_f=self.cz_theta_f(),
                f_01_max=self.cz_freq_01_max(),
                # V_per_phi0=self.cz_V_per_phi0(),
                J2=self.cz_J2(),
                # E_c=self.cz_E_c(),
                f_interaction=self.cz_freq_interaction(),
                sampling_rate=self.sampling_rate(),
                return_unit='f01')
            half_CZ_A = dac_scale_factor*self.detuning_to_amp(
                self.cz_freq_01_max() - half_CZ_A)

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

            half_CZ_B = wf.martinis_flux_pulse(
                length=self.cz_length()*(1-self.czd_length_ratio()),
                lambda_2=d_lambda_2,
                lambda_3=d_lambda_3,
                theta_f=d_theta_f,
                f_01_max=self.cz_freq_01_max(),
                # V_per_phi0=self.cz_V_per_phi0(),
                J2=self.cz_J2(),
                # E_c=self.cz_E_c(),
                f_interaction=self.cz_freq_interaction(),
                sampling_rate=self.sampling_rate(),
                return_unit='f01')
            half_CZ_B = dac_scale_factor*self.detuning_to_amp(
                self.cz_freq_01_max() - half_CZ_B, positive_branch=False)

            amp_rat = self.czd_amp_ratio()
            waveform = np.concatenate(
                [half_CZ_A, amp_rat*half_CZ_B + self.czd_amp_offset()])

            return waveform

    def _get_phase_corr_offset(self):
        net_charge = np.sum(self._wave_dict['cz'])
        corr_samples = self.sampling_rate()*self.cz_phase_corr_length()
        offset = -net_charge/corr_samples
        return offset

    def _gen_phase_corr(self, phase_corr_amp: float = None,
                        cz_offset_comp: bool =False):
        """
        Generates a phase correction pulse.
        Amp can be given as an argument in order to use it in the multi-pulse
        if different amps are desired.

        cz_offset_comp can be given to ensure a zero net flux
        pulse. This should be False when generating phase correction for
        an "idle" pulse.

        """
        if phase_corr_amp is None:
            phase_corr_amp = self.cz_phase_corr_amp()

        if not self.czd_double_sided():
            phase_corr_wf = wf.single_channel_block(
                amp=phase_corr_amp, length=self.cz_phase_corr_length(),
                sampling_rate=self.sampling_rate(), delay=0)
            return phase_corr_wf

        else:
            block = wf.single_channel_block(
                amp=phase_corr_amp,
                length=self.cz_phase_corr_length()/2,
                sampling_rate=self.sampling_rate(), delay=0)

            if cz_offset_comp:
                phase_corr_offset = self._get_phase_corr_offset()
            else:
                phase_corr_offset = 0

            phase_corr_wf = np.concatenate(
                [block, -1*block]) + phase_corr_offset
            return phase_corr_wf

    def _gen_cz_z(self, regenerate_cz=True):
        if regenerate_cz:
            self._wave_dict['cz'] = self._gen_cz()
        phase_corr = self._gen_phase_corr(cz_offset_comp=False)
        # CZ with phase correction
        cz_z = np.concatenate([self._wave_dict['cz'], phase_corr])

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
            waveform[sample_start_idx:sample_start_idx+len(sq)] += sq
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
            waveform[sample_start_idx:sample_start_idx+len(cz_z)] += cz_z
        # CZ with phase correction
        return waveform

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

    def _gen_idle_z(self, regenerate_cz=True):
        if regenerate_cz:
            self._wave_dict['cz'] = self._gen_cz()
        phase_corr = self._gen_phase_corr()
        # Only apply phase correction component after idle for duration of CZ
        return np.concatenate(
            [np.zeros(len(self._wave_dict['cz'])), phase_corr])

    def _gen_multi_idle_z(self, regenerate_cz=False):
        """
        Composite waveform containing multiple cz gates
        """
        if regenerate_cz:
            self._wave_dict['cz'] = self._gen_cz()

        max_nr_samples = int(self.cfg_max_wf_length()*self.sampling_rate())

        waveform = np.zeros(max_nr_samples)
        for i in range(self.mcz_nr_of_repeated_gates()):
            # if self.mcz_identical_phase_corr():
            phase_corr = self._gen_phase_corr(cz_offset_comp=False)
            # else:
            #     phase_corr = wf.single_channel_block(
            #         amp=self.get('mcz_phase_corr_amp_{}'.format(i+1)),
            #         length=self.cz_phase_corr_length(),
            #         sampling_rate=self.sampling_rate(), delay=0)
            cz_z = np.concatenate([np.zeros(len(self._wave_dict['cz'])),
                                   phase_corr])
            sample_start_idx = int(self.mcz_gate_separation() *
                                   self.sampling_rate())*i
            waveform[sample_start_idx:sample_start_idx+len(cz_z)] += cz_z
        # CZ with phase correction
        return waveform

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
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
        self.AWG.get_instr().set(codeword, waveform)

    def load_composite_waveform_onto_AWG_lookuptable(self,
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
            waveform = self.distort_waveform(waveform)
            self._wave_dict_dist[waveform_name] = waveform
        self.AWG.get_instr().set(codeword, waveform)

    def load_waveform_realtime(self, waveform_name: str,
                               other_waveform=None):
        """
        Warning! Care should be taken when using this method.
        It makes certain assumptions about the program being used
        and is optimized for speed.

        Args:
            waveform_name (str) : name of the waveform to reload
            other_waveform (array): used to fill the other channel,
                                    filled with zeros if nothing is specified
        It assumes
            - an AWG8 program with only a single waveform is played
            - it will always regenerate the desired waveform.
            - it does not update the _wave_dict
        """
        gen_wf_func = getattr(self, '_gen_{}'.format(waveform_name))
        waveform = gen_wf_func()

        if self.cfg_append_compensation():
            waveform = self.add_compensation_pulses(waveform)
        if self.cfg_distort():
            waveform = self.distort_waveform(waveform)
            self._wave_dict_dist[waveform_name] = waveform

        awg_ch = self.cfg_awg_channel()-1  # -1 is to account for starting at 1
        ch_pair = awg_ch % 2
        awg_nr = awg_ch//2

        if other_waveform is None:
            other_waveform = np.zeros(len(waveform))

        if ch_pair == 0:
            waveforms = (waveform, other_waveform)
        else:
            waveforms = (other_waveform, waveform)

        # wf_nr = 1 is the assumption of only a sinlge waveform in the program
        self.AWG.get_instr().upload_waveform_realtime(
            w0=waveforms[0], w1=waveforms[1], awg_nr=awg_nr, wf_nr=1)

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

    def distort_waveform(self, waveform):
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
                    self.cfg_max_wf_length()*self.sampling_rate()))
        else:  # old kernel object does not have this method
            distorted_waveform = k.convolve_kernel(
                [k.kernel(), waveform],
                length_samples=int(self.cfg_max_wf_length() *
                                   self.sampling_rate()))
        return distorted_waveform

    def plot_cz_trajectory(self, ax=None, show=True):
        """
        Plots the cz trajectory in frequency space.
        """
        if ax is None:
            f, ax = plt.subplots()
        extra_samples = 10
        nr_plot_samples = int((self.cz_length()+self.cz_phase_corr_length()) *
                              self.sampling_rate() + extra_samples)
        dac_amps = self._wave_dict['cz_z'][:nr_plot_samples]
        samples = np.arange(len(dac_amps))
        amps = dac_amps*self.get_dac_val_to_amp_scalefactor()
        deltas = self.amp_to_detuning(amps)
        freqs = self.cz_freq_01_max()-deltas
        ax.scatter(amps, freqs, c=samples, label='CZ trajectory')
        if show:
            plt.show()
        return ax

    def plot_flux_arc(self, ax=None, show=True,
                      plot_cz_trajectory=False):
        """
        Plots the flux arc as used in the lutman based on the polynomial
        coefficients
        """

        if ax is None:
            f, ax = plt.subplots()
        amps = np.linspace(-2.5, 2.5, 101)  # maximum voltage of AWG amp mode
        deltas = self.amp_to_detuning(amps)
        freqs = self.cz_freq_01_max()-deltas

        ax.plot(amps, freqs, label='$f_{01}$')
        ax.axhline(self.cz_freq_interaction(), -5, 5,
                   label='$f_{\mathrm{int.}}$:'+' {:.3f} GHz'.format(
            self.cz_freq_interaction()*1e-9),
            c='C1')

        ax.axvline(0, 0, 1e10, linestyle='dotted', c='grey')
        ax.fill_between(
            x=[-5, 5],
            y1=[self.cz_freq_interaction()-self.cz_J2()]*2,
            y2=[self.cz_freq_interaction()+self.cz_J2()]*2,
            label='$J_{\mathrm{2}}/2\pi$:'+' {:.3f} MHz'.format(
                self.cz_J2()*1e-6),
            color='C1', alpha=0.25)

        title = ('Calibration visualization\n{}\nchannel {}'.format(
            self.AWG(), self.cfg_awg_channel()))
        if plot_cz_trajectory:
            self.plot_cz_trajectory(ax=ax, show=False)
        leg = ax.legend(title=title, loc=(1.05, .7))
        leg._legend_box.align = 'center'
        set_xlabel(ax, 'AWG amplitude', 'V')
        set_ylabel(ax, 'Frequency', 'Hz')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(3e9, np.max(freqs)+500e6)

        dac_val_axis = ax.twiny()
        dac_ax_lims = np.array(ax.get_xlim()) * \
            self.get_amp_to_dac_val_scale_factor()
        dac_val_axis.set_xlim(dac_ax_lims)
        set_xlabel(dac_val_axis, 'AWG amplitude', 'dac')

        dac_val_axis.axvspan(1, 1000, facecolor='.5', alpha=0.5)
        dac_val_axis.axvspan(-1000, -1, facecolor='.5', alpha=0.5)
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
                                           regenerate_waveforms: bool=False):
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
        scale_factor = channel_amp
        return scale_factor