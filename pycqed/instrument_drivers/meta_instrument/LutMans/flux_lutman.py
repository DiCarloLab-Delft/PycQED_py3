from .base_lutman import Base_LutMan
import numpy as np
import logging
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
from qcodes.utils import validators as vals
from pycqed.measurement.waveform_control_CC import waveform as wf
from qcodes.plots.pyqtgraph import QtPlot


class Base_Flux_LutMan(Base_LutMan):
    # this default lutman is if a flux pulse can be done with only one
    # other qubit. this needs to be expanded if there are more qubits
    # to interact with.
    _def_lm = ['i', 'cz_z', 'square', 'park']

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
                    yunit='V')
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

        self.add_parameter('instr_distortion_kernel',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter('cfg_max_wf_length',
                           parameter_class=ManualParameter,
                           initial_value=100e-6,
                           unit='s', vals=vals.Numbers(0, 100e-6))

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
                           unit='a.u.', vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('sq_length', unit='s',
                           label='Square pulse length',
                           initial_value=40e-9,
                           vals=vals.Numbers(0, 100e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('sq_delay', unit='s',
                           label='Square pulse length',
                           initial_value=0e-9,
                           vals=vals.Numbers(0, 12e-6),
                           # 12us is current max of AWG8
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
        self.add_parameter('cz_V_per_phi0', vals=vals.Numbers(),
                           unit='V', initial_value=1,
                           parameter_class=ManualParameter)
        self.add_parameter('cz_freq_01_max', vals=vals.Numbers(),
                           unit='Hz', parameter_class=ManualParameter)
        self.add_parameter('cz_J2', vals=vals.Numbers(),
                           unit='Hz',
                           initial_value=50e6,
                           parameter_class=ManualParameter)
        self.add_parameter('cz_freq_interaction', vals=vals.Numbers(),
                           unit='Hz',
                           parameter_class=ManualParameter)
        self.add_parameter('cz_phase_corr_length',
                           unit='s',
                           initial_value=5e-9, vals=vals.Numbers(),
                           parameter_class=ManualParameter)

        self.add_parameter('cz_phase_corr_amp',
                           unit='V',
                           initial_value=0, vals=vals.Numbers(),
                           parameter_class=ManualParameter)

    def generate_standard_waveforms(self):
        """
        Generates all the standard waveforms and populates self._wave_dict

        """
        self._wave_dict = {}
        # FIXME: only implemented square wave for now (MAR)
        self._wave_dict['i'] = np.zeros(42)
        self._wave_dict['square'] = wf.single_channel_block(
            amp=self.sq_amp(), length=self.sq_length(),
            sampling_rate=self.sampling_rate(), delay=self.sq_delay())
        self._wave_dict['park'] = np.zeros(42)

        self._wave_dict['cz'] = wf.martinis_flux_pulse(
            length=self.cz_length(),
            lambda_2=self.cz_lambda_2(),
            lambda_3=self.cz_lambda_3(),
            theta_f=self.cz_theta_f(),
            f_01_max=self.cz_freq_01_max(),
            V_per_phi0=self.cz_V_per_phi0(),
            J2=self.cz_J2(),
            f_interaction=self.cz_freq_interaction(),
            sampling_rate=self.sampling_rate(),
            return_unit='V')

        phase_corr = wf.single_channel_block(
            amp=self.cz_phase_corr_amp(), length=self.cz_phase_corr_length(),
            sampling_rate=self.sampling_rate(), delay=0)

        # CZ with phase correction
        self._wave_dict['cz_z'] = np.concatenate(
            [self._wave_dict['cz'], phase_corr])

        # Only apply phase correction component after idle for duration of CZ
        self._wave_dict['idle_z'] = np.concatenate(
            [np.zeros(len(self._wave_dict['cz'])), phase_corr])

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
        """
        Loads a specific waveform to the AWG
        """
        if regenerate_waveforms:
            self.generate_standard_waveforms()
        waveform = self._wave_dict[waveform_name]
        codeword = self.LutMap()[waveform_name]

        if self.cfg_append_compensation():
            waveform = self.add_compensation_pulses(waveform)

        if self.cfg_distort():
            waveform = self.distort_waveform(waveform)
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

    def distort_waveform(self, waveform):
        """
        uses the Kernel object to distort waveforms
        """
        k = self.instr_distortion_kernel.get_instr()
        distorted_waveform = k.convolve_kernel(
            [k.kernel(), waveform],
            length_samples=int(self.cfg_max_wf_length()*self.sampling_rate()))
        return distorted_waveform
