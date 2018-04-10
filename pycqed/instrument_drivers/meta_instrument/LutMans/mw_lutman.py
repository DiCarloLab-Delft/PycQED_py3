from .base_lutman import Base_LutMan, get_redundant_codewords
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from pycqed.measurement.waveform_control_CC import waveform as wf


class Base_MW_LutMan(Base_LutMan):
    _def_lm = ['I', 'rX180',  'rY180', 'rX90',  'rY90',
               'rXm90',  'rYm90', 'rPhi90', 'spec']
    # use remaining codewords to set pi/2 gates for various angles
    for i in range(18):
        angle = i * 20
        _def_lm.append('r{}_90'.format(angle))

    def set_default_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        def_lm = self._def_lm
        LutMap = {}
        for cw_idx, cw_key in enumerate(def_lm):
            LutMap[cw_key] = (
                'wave_ch{}_cw{:03}'.format(self.channel_I(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_Q(), cw_idx))
        self.LutMap(LutMap)

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

        self.add_parameter(
            'mw_modulation', vals=vals.Numbers(), unit='Hz',
            docstring=('Modulation frequency for qubit driving pulses. Note'
                       ' that when using an AWG with build in modulation this'
                       ' should be set to 0.'),
            parameter_class=ManualParameter, initial_value=50.0e6)
        self._add_mixer_corr_pars()

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

    def generate_standard_waveforms(self):
        self._wave_dict = {}
        if self.cfg_sideband_mode() == 'static':
            f_modulation = self.mw_modulation()
        else:
            f_modulation = 0

        self._wave_dict['I'] = self.wf_func(
            amp=0, sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=0)
        self._wave_dict['rX180'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.mw_motzoi())
        self._wave_dict['rY180'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.mw_motzoi())
        self._wave_dict['rX90'] = self.wf_func(
            amp=self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.mw_motzoi())
        self._wave_dict['rY90'] = self.wf_func(
            amp=self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.mw_motzoi())
        self._wave_dict['rXm90'] = self.wf_func(
            amp=-1*self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=0,
            motzoi=self.mw_motzoi())
        self._wave_dict['rYm90'] = self.wf_func(
            amp=-1*self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=90,
            motzoi=self.mw_motzoi())

        self._wave_dict['rPhi180'] = self.wf_func(
            amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.mw_phi(),
            motzoi=self.mw_motzoi())
        self._wave_dict['rPhi90'] = self.wf_func(
            amp=self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.mw_phi(),
            motzoi=self.mw_motzoi())
        self._wave_dict['rPhim90'] = self.wf_func(
            amp=-1*self.mw_amp180()*self.mw_amp90_scale(),
            sigma_length=self.mw_gauss_width(),
            f_modulation=f_modulation,
            sampling_rate=self.sampling_rate(), phase=self.mw_phi(),
            motzoi=self.mw_motzoi())
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
                motzoi=self.mw_motzoi())

        if self.mixer_apply_predistortion_matrix():
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
        def_lm = self._def_lm
        LutMap = {}
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
                           vals=vals.Numbers(0, 1.8),
                           set_cmd=self._set_channel_amp,
                           get_cmd=self._get_channel_amp,
                           docstring=('using the channel amp as additional'
                                      'parameter to allow rabi-type experiments without'
                                      'wave reloading. Should not be using VSM'))

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

    def _add_channel_params(self):
        super()._add_channel_params()
        self.add_parameter('channel_amp',
                           unit='a.u.',
                           vals=vals.Numbers(0, 1),
                           set_cmd=self._set_channel_amp,
                           get_cmd=self._get_channel_amp,
                           docstring=('using the channel amp as additional'
                                      'parameter to allow rabi-type experiments without'
                                      'wave reloading. Should not be using VSM'))

    def _set_channel_amp(self, val):
        AWG = self.AWG.get_instr()
        for awg_ch in [self.channel_I(), self.channel_Q()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            AWG.set('AWGS/{}/OUTPUTS/{}/AMPLITUDE'.format(awg_nr, ch_pair), val)

    def _get_channel_amp(self):
        AWG = self.AWG.get_instr()
        vals = []
        for awg_ch in [self.channel_I(), self.channel_Q()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            vals.append(
                AWG.get('AWGS/{}/OUTPUTS/{}/AMPLITUDE'.format(awg_nr, awg_ch)))
        assert vals[0] == vals[1]
        return vals[0]

    def load_waveforms_onto_AWG_lookuptable(
            self, regenerate_waveforms: bool=True, stop_start: bool = True):
        """
        Loads all waveforms specified in the LutMap to an AWG.

        Args:
            regenerate_waveforms (bool): if True calls
                generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.
        """
        super().load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=regenerate_waveforms,
            stop_start=stop_start)
        # Uploading the codeword program (again) is needed to link the new
        # waveforms.
        # the False prevents reconfiguring the DIO timings. This
        # needs to be fixed in the AWG8 driver (MAR Oct 2017)
        self.AWG.get_instr().upload_codeword_program()


class AWG8_VSM_MW_LutMan(AWG8_MW_LutMan):

    def __init__(self, name, **kw):
        self._num_channels = 8
        super().__init__(name, **kw)
        self.wf_func = wf.mod_gauss_VSM
        self.spec_func = wf.block_pulse_vsm

    def set_default_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        def_lm = self._def_lm
        LutMap = {}
        for cw_idx, cw_key in enumerate(def_lm):
            LutMap[cw_key] = (
                'wave_ch{}_cw{:03}'.format(self.channel_GI(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_GQ(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_DI(), cw_idx),
                'wave_ch{}_cw{:03}'.format(self.channel_DQ(), cw_idx))
        self.LutMap(LutMap)

    def _add_channel_params(self):
        # FIXME: add parameter channel amp that sets the ouput amplitude of
        # all channels used for this pulse
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
        self.add_parameter('channel_amp',
                           unit='a.u.',
                           vals=vals.Numbers(0, 1),
                           set_cmd=self._set_channel_amp,
                           get_cmd=self._get_channel_amp,
                           docstring=('using the channel amp as additional'
                                      'parameter to allow rabi-type experiments without'
                                      'wave reloading. Should not be using VSM'))

    def _set_channel_amp(self, val):
        AWG = self.AWG.get_instr()
        for awg_ch in [self.channel_GI(), self.channel_GQ(),
                       self.channel_DI(), self.channel_DQ()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            AWG.set('AWGS/{}/OUTPUTS/{}/AMPLITUDE'.format(awg_nr, ch_pair), val)

    def _get_channel_amp(self):
        AWG = self.AWG.get_instr()
        vals = []
        for awg_ch in [self.channel_GI(), self.channel_GQ(),
                       self.channel_DI(), self.channel_DQ()]:
            awg_nr = (awg_ch-1)//2
            ch_pair = (awg_ch-1) % 2
            vals.append(
                AWG.get('AWGS/{}/OUTPUTS/{}/AMPLITUDE'.format(awg_nr, awg_ch)))
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
        M_D = wf.mixer_predistortion_matrix(self.G_mixer_alpha(),
                                            self.G_mixer_phi())

        for key, val in wave_dict.items():
            GI, GQ = np.dot(M_G, val[0:2])  # Mixer correction Gaussian comp.
            DI, DQ = np.dot(M_D, val[2:4])  # Mixer correction Derivative comp.
            wave_dict[key] = GI, GQ, DI, DQ
        return wave_dict


class QWG_MW_LutMan_VQE(QWG_MW_LutMan):
    """
    WORK ONGOING:
    4. define the waveforms accordingly to the VQE lutmap.
          (use parameters from _add_waveform_parameters.)

    """

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
        self._def_lm = ['I', 'rX180',  'rY180', 'rX90',  'rY90',
                        'rXm90',  'rYm90', 'rPhi90']
        self.set_default_lutmap()

        self._vqe_lm = ['I', 'X180t',  'Y180t', 'X90t',  'Xm90t',
                        'Y90t',  'Y90t', 'Y180']

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
        self.add_parameter('phi1', unit='rad', vals=vals.Numbers(0, 2*np.pi),
                           parameter_class=ManualParameter,
                           initial_value=0)
        self.add_parameter('phi2', unit='rad', vals=vals.Numbers(0, 2*np.pi),
                           parameter_class=ManualParameter,
                           initial_value=0)

    def generate_standard_waveforms(self):
        # do not apply mixer predistortions until the end
        old_mixer_setting = self.mixer_apply_predistortion_matrix()
        self.mixer_apply_predistortion_matrix(False)
        # generate waveforms for standard mapping
        super().generate_standard_waveforms()
        self.mixer_apply_predistortion_matrix(old_mixer_setting)

        if self.cfg_sideband_mode() == 'static':
            f_modulation = self.mw_modulation()
        else:
            f_modulation = 0

        # """
        # This pulses still need to be cross-checked.
        # """
        # self._wave_dict['cX180_phi'] = self.wf_func(
        #     amp=self.mw_amp180(), sigma_length=self.mw_gauss_width(),
        #     f_modulation=f_modulation,
        #     sampling_rate=self.sampling_rate(), phase=0,
        #     motzoi=self.mw_motzoi())

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
        print(redundant_cw_list)
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

    def set_default_lutmap(self):
        """
        Set's the default lutmap for standard microwave drive pulses.
        """
        def_lm = self._def_lm
        LutMap = {}
        for cw_idx, cw_key in enumerate(def_lm):
            LutMap[cw_key] = (
                'wave_ch1_cw{:03}'.format(cw_idx),
                'wave_ch2_cw{:03}'.format(cw_idx),
                'wave_ch3_cw{:03}'.format(cw_idx),
                'wave_ch4_cw{:03}'.format(cw_idx))
        self.LutMap(LutMap)
