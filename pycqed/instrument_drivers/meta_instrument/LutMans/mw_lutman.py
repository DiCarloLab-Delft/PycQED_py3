from .base_lutman import Base_LutMan
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from pycqed.measurement.waveform_control_CC import waveform as wf


class Base_MW_LutMan(Base_LutMan):

    def _add_waveform_parameters(self):
        # defined here so that the VSM based LutMan can overwrite this
        self.wf_func = wf.mod_gauss

        self._add_channel_params()
        self.add_parameter('Q_amp180', unit='V', vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.1)
        self.add_parameter('Q_amp90_scale',
                           vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.5)
        self.add_parameter('Q_motzoi', vals=vals.Numbers(-2, 2),
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('Q_gauss_width',
                           vals=vals.Numbers(min_value=1e-9), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=4e-9)
        self.add_parameter('Q_phi', label='Phase of Rphi pulse',
                           vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0)

        self.add_parameter(
            'Q_modulation', vals=vals.Numbers(), unit='Hz',
            docstring=('Modulation frequency for qubit driving pulses. Note'
                       ' that when using an AWG with build in modulation this'
                       ' should be set to 0.'),
            parameter_class=ManualParameter, initial_value=50.0e6)
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
            parameter_class=ManualParameter, initial_value=False)

    def _add_channel_params(self):
        self.add_parameter('channel_I',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, self._num_channels))

        self.add_parameter('channel_Q',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, self._num_channels))

    def generate_standard_waveforms(self):
        self._wave_dict = {}

        self._wave_dict['I'] = self.wf_func(
            amp=0, sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=0)
        self._wave_dict['rX180'] = self.wf_func(
            amp=self.Q_amp180(), sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.Q_motzoi())
        self._wave_dict['rY180'] = self.wf_func(
            amp=self.Q_amp180(), sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.Q_motzoi())
        self._wave_dict['rX90'] = self.wf_func(
            amp=self.Q_amp180()*self.Q_amp90_scale(),
            sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.Q_motzoi())
        self._wave_dict['rY90'] = self.wf_func(
            amp=self.Q_amp180()*self.Q_amp90_scale(),
            sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.Q_motzoi())
        self._wave_dict['rXm90'] = self.wf_func(
            amp=-1*self.Q_amp180()*self.Q_amp90_scale(),
            sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.Q_motzoi())
        self._wave_dict['rYm90'] = self.wf_func(
            amp=-1*self.Q_amp180()*self.Q_amp90_scale(),
            sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.Q_motzoi())

        self._wave_dict['rPhi180'] = self.wf_func(
            amp=self.Q_amp180(), sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=self.Q_phi(),
            motzoi=self.Q_motzoi())
        self._wave_dict['rPhi90'] = self.wf_func(
            amp=self.Q_amp180()*self.Q_amp90_scale(),
            sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=self.Q_phi(),
            motzoi=self.Q_motzoi())
        self._wave_dict['rPhim90'] = self.wf_func(
            amp=-1*self.Q_amp180()*self.Q_amp90_scale(),
            sigma_length=self.Q_gauss_width(),
            f_modulation=self.Q_modulation(),
            sampling_rate=self.sampling_rate(), phase=self.Q_phi(),
            motzoi=self.Q_motzoi())

        if self.mixer_apply_predistortion_matrix():
            M = wf.mixer_predistortion_matrix(self.mixer_alpha(),
                                              self.mixer_phi())
            for key, val in self._wave_dict.items():
                self._wave_dict[key] = np.dot(M, val)

        return self._wave_dict

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
            if regenerate_waveforms:
                self.generate_standard_waveforms()
            waveforms = self._wave_dict[waveform_name]
            codewords = self.LutMap()[waveform_name]
            for waveform, cw in zip(waveforms, codewords):
                self.AWG.get_instr().set(cw, waveform)


class CBox_MW_LutMan(Base_MW_LutMan):

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
        def_lm = ['I', 'rX180',  'rY180', 'rX90',  'rY90',
                  'rXm90',  'rYm90', 'rPhi90', 'rPhim90']
        LutMap = {}
        for cw_idx, cw_key in enumerate(def_lm):
            LutMap[cw_key] = cw_idx
        self.LutMap(LutMap)


class QWG_MW_LutMan(Base_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 4
        super().__init__(name, **kw)

    def set_default_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        def_lm = ['I', 'rX180',  'rY180', 'rX90',  'rY90',
                  'rXm90',  'rYm90', 'rPhi180', 'rPhi90', 'rPhim90']
        LutMap = {}
        # N.B. The main difference between the AWG8 and QWG in the lutmap
        # is the naming of the codeword params.
        for cw_idx, cw_key in enumerate(def_lm):
            LutMap[cw_key] = (
                'codeword_{}_ch{}_waveform'.format(cw_idx, self.channel_I()),
                'codeword_{}_ch{}_waveform'.format(cw_idx, self.channel_Q()))
        self.LutMap(LutMap)

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
        raise NotImplementedError('Awaits resolution of issue #322')


class AWG8_MW_LutMan(Base_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 8
        super().__init__(name, **kw)

    def set_default_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        def_lm = ['I', 'rX180',  'rY180', 'rX90',  'rY90',
                  'rXm90',  'rYm90', 'rPhi90', 'rPhim90']
        LutMap = {}
        for cw_idx, cw_key in enumerate(def_lm):
            LutMap[cw_key] = (
                'wave_ch{}_cw{:03}'.format(self.channel_I(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_Q(), cw_idx))
        self.LutMap(LutMap)


class AWG8_VSM_MW_LutMan(Base_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 8
        super().__init__(name, **kw)
        self.wf_func = wf.mod_gauss_VSM

    def set_default_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        def_lm = ['I', 'rX180',  'rY180', 'rX90',  'rY90',
                  'rXm90',  'rYm90', 'rPhi90', 'rPhim90']
        LutMap = {}
        for cw_idx, cw_key in enumerate(def_lm):
            LutMap[cw_key] = (
                'wave_ch{}_cw{:03}'.format(self.channel_GI(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_GQ(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_DI(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_DQ(), cw_idx))
        self.LutMap(LutMap)

    def _add_channel_params(self):
        self.add_parameter('channel_GI',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, self._num_channels))
        self.add_parameter('channel_GQ',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, self._num_channels))
        self.add_parameter('channel_DI',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, self._num_channels))
        self.add_parameter('channel_DQ',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(1, self._num_channels))


