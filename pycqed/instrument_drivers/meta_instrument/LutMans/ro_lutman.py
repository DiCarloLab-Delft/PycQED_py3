from .base_lutman import Base_LutMan, get_wf_idx_from_name
from pycqed.measurement.waveform_control_CC import waveform as wf
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import numpy as np
import copy as copy


def create_pulse(shape: str,
                 amplitude: float,
                 length: float,
                 phase: float,
                 delay: float=0,
                 sampling_rate: float=1.8e9):
    kw = {}
    if shape == 'square':
        shape_function = wf.block_pulse
        kw['length'] = length
    elif shape == 'gaussian':
        shape_function = wf.gauss_pulse
        nr_sigma = 4
        kw['sigma_length'] = length/nr_sigma
        kw['nr_sigma'] = nr_sigma
    else:
        raise NotImplementedError('Primitive pulse shape ' +
                                  shape +
                                  ' not implemented.')

    return shape_function(amp=amplitude,
                          sampling_rate=sampling_rate,
                          delay=delay,
                          phase=phase, **kw)


class Base_RO_LutMan(Base_LutMan):

    def __init__(self, name, num_res=2, feedline_number: int=0,
                 feedline_map='S7', **kw):
        if num_res > 10:
            raise ValueError('At most 10 resonators can be read out.')
        self._num_res = num_res
        self._feedline_number = feedline_number

        if feedline_map == 'S5':
            if self._feedline_number == 0:
                self._resonator_codeword_bit_mapping = [0, 2, 3, 4]
            elif self._feedline_number == 1:
                self._resonator_codeword_bit_mapping = [1]
            else:
                raise NotImplementedError(
                    'Hardcoded for feedline 0 and 1 of Surface-5')
        elif feedline_map == 'S7':
            if self._feedline_number == 0:
                self._resonator_codeword_bit_mapping = [0, 2, 3, 5, 6]
            elif self._feedline_number == 1:
                self._resonator_codeword_bit_mapping = [1, 4]
            else:
                raise NotImplementedError(
                    'Hardcoded for feedline 0 and 1 of Surface-7')
        elif feedline_map == 'S17':

            if self._feedline_number == 0:
                self._resonator_codeword_bit_mapping = [13, 16]
            elif self._feedline_number == 1:
                self._resonator_codeword_bit_mapping = [
                    1, 4, 5, 7, 8, 10, 11, 14, 15]
            elif self._feedline_number == 2:
                self._resonator_codeword_bit_mapping = [0, 2, 3, 6, 9, 12]
            else:
                # FIXME: copy/paste error
                raise NotImplementedError(
                    'Hardcoded for feedline 0, 1 and 2 of Surface-17')
        else:
            raise ValueError('Feedline map not in {"S5", "S7", "S17"}.')

        # capping the resonator bit mapping in case a limited number of resonators is used
        self._resonator_codeword_bit_mapping = self._resonator_codeword_bit_mapping[
            :self._num_res]
        # Initial values on parameters that otherwise depend on each other
        self._resonator_combinations = [
            [self._resonator_codeword_bit_mapping[0]]]
        self._pulse_type = 'M_simple'
        super().__init__(name, **kw)

    def _set_resonator_combinations(self, value):
        self._resonator_combinations = value
        self.set_default_lutmap()

    def _get_resonator_combinations(self):
        return self._resonator_combinations

    def _set_pulse_type(self, value):
        self._pulse_type = value
        self.set_default_lutmap()

    def _get_pulse_type(self):
        return self._pulse_type

    def _add_waveform_parameters(self):
        # mixer corrections are done globally, can be specified per resonator
        self.add_parameter('mixer_apply_predistortion_matrix',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=False)
        self.add_parameter('gaussian_convolution',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=False)
        self.add_parameter('gaussian_convolution_sigma', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=5.0e-9,
                           unit='s')
        self.add_parameter('mixer_alpha', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('mixer_phi', vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('mixer_offs_I', unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_Q', unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        comb_msg = (
            'Resonator combinations specifies which pulses are uploaded to'
            'the device. Given as a list of lists:'
            'e.g. [[0], [2], [0, 2]] specifies that pulses for readout'
            'of resonator 0, 2, and a pulse for mux readout on both should be'
            'uploaded.')
        self.add_parameter('resonator_combinations',
                           vals=vals.Lists(),
                           docstring=comb_msg,
                           set_cmd=self._set_resonator_combinations,
                           get_cmd=self._get_resonator_combinations)
        self.add_parameter('pulse_type', vals=vals.Enum(
            'M_up_down_down', 'M_simple', 'M_up_down_down_final'),
            set_cmd=self._set_pulse_type,
            get_cmd=self._get_pulse_type,
            docstring='defines sequence of segments of the pulse')
        self.add_parameter('pulse_primitive_shape', vals=vals.Enum(
            'square', 'gaussian'),
            parameter_class=ManualParameter,
            docstring='defines the shape of the segments of the pulse',
            initial_value='square')
        for res in self._resonator_codeword_bit_mapping:
            self.add_parameter('M_modulation_R{}'.format(res),
                               vals=vals.Numbers(), unit='Hz',
                               parameter_class=ManualParameter,
                               initial_value=20.0e6)
            self.add_parameter('M_length_R{}'.format(res), unit='s',
                               vals=vals.Numbers(1e-9, 8000e-9),
                               parameter_class=ManualParameter,
                               initial_value=2000e-9)
            self.add_parameter('M_amp_R{}'.format(res), unit='V',
                               vals=vals.Numbers(0, 1),
                               parameter_class=ManualParameter,
                               initial_value=0.1)
            self.add_parameter('M_delay_R{}'.format(res), unit='s',
                               vals=vals.Numbers(0, 1e-6),
                               parameter_class=ManualParameter,
                               initial_value=0)
            self.add_parameter('M_final_amp_R{}'.format(res), unit='V',
                               vals=vals.Numbers(0, 1),
                               parameter_class=ManualParameter,
                               initial_value=0.1)
            self.add_parameter('M_final_length_R{}'.format(res), unit='s',
                               vals=vals.Numbers(1e-9, 8000e-9),
                               parameter_class=ManualParameter,
                               initial_value=1000e-9)
            self.add_parameter('M_final_delay_R{}'.format(res), unit='s',
                               vals=vals.Numbers(1e-9, 8000e-9),
                               parameter_class=ManualParameter,
                               initial_value=200e-9)
            self.add_parameter('M_phi_R{}'.format(res), unit='deg',
                               vals=vals.Numbers(0, 360),
                               parameter_class=ManualParameter,
                               initial_value=0.0)
            self.add_parameter('M_down_length0_R{}'.format(res), unit='s',
                               vals=vals.Numbers(1e-9, 8000e-9),
                               parameter_class=ManualParameter,
                               initial_value=200.0e-9)
            self.add_parameter('M_down_length1_R{}'.format(res), unit='s',
                               vals=vals.Numbers(1e-9, 8000e-9),
                               parameter_class=ManualParameter,
                               initial_value=200.0e-9)
            self.add_parameter('M_down_amp0_R{}'.format(res), unit='V',
                               vals=vals.Numbers(-1, 1),
                               parameter_class=ManualParameter,
                               initial_value=0.1)
            self.add_parameter('M_down_amp1_R{}'.format(res), unit='V',
                               vals=vals.Numbers(-1, 1),
                               parameter_class=ManualParameter,
                               initial_value=0.1)
            self.add_parameter('M_down_phi0_R{}'.format(res), unit='deg',
                               parameter_class=ManualParameter,
                               initial_value=180.0)
            self.add_parameter('M_down_phi1_R{}'.format(res), unit='deg',
                               parameter_class=ManualParameter,
                               initial_value=180.0)

    def set_default_lutmap(self):
        """
        Updates the LutMap based on the "resonator_combinations" and "pulse_type"
        parameters.
        """
        lm = {}

        # LutMap format:
        # CW: {'name': name}
        # As the LutMap is going to contain a list of the actual codewords that are used,
        # we start by iterating through the used read-out combinations.
        for resonator_combination in self._resonator_combinations:
            case = 0

            if not resonator_combination:
                wavename = 'empty'
            else:
                wavename = self._pulse_type
                for resonator in resonator_combination:
                    wavename += '_R' + str(resonator)
                    try:
                        case += 2**self._resonator_codeword_bit_mapping.index(
                            resonator)
                    except ValueError:
                        # The allowed resonators is determined by the feedline
                        raise ValueError(
                            "{} not in allowed resonators ({}).".format(
                                resonator, self._resonator_codeword_bit_mapping))

            lm[case] = {'name': wavename, 'resonators': resonator_combination}

        self.LutMap(lm)

    def generate_standard_waveforms(self):
        """
        Generates waveforms for reading out from each individual resonator.
        """
        # Only generate the combinations required/specified in the LutMap
        # RO pulses
        sampling_rate = self.get('sampling_rate')

        # Prepare gauss pulse for convolution (if desired)
        if self.gaussian_convolution():
            if self.pulse_primitive_shape() != 'square':
                print("Warning: recommend to set pulse_primitive_shape to " +
                      "'square' when using gaussian_convolution.")
            sigma = self.gaussian_convolution_sigma()

            def norm_gauss(x, mu, sigma):
                n = 1/(sigma*np.sqrt(2*np.pi))
                return n*np.exp((-((x-mu)/sigma)**2)/2)

            gauss_length = 6*sigma  # in units of time
            hgsl = int(gauss_length*sampling_rate/2)  # half no. of samples
            gauss_sample_length = hgsl*2  # no. of samples
            gauss_samples = np.arange(0, gauss_sample_length, 1)/sampling_rate
            norm_gauss_p = norm_gauss(x=gauss_samples,
                                      mu=gauss_length/2, sigma=sigma)
        else:
            gauss_length = 0

        # Prepare pulses for all resonators
        self._wave_dict = {}
        for res in self._resonator_codeword_bit_mapping:
            res_wave_dict = {}
            # 1. Generate Pulse envelopes
            # Simple pulse
            up_len = self.get('M_length_R{}'.format(res))-gauss_length
            M = create_pulse(shape=self.pulse_primitive_shape(),
                             amplitude=self.get('M_amp_R{}'.format(res)),
                             length=up_len,
                             delay=self.get('M_delay_R{}'.format(res)),
                             phase=self.get('M_phi_R{}'.format(res)),
                             sampling_rate=sampling_rate)
            res_wave_dict['M_simple_R{}'.format(res)] = M

            # 3-step RO pulse with ramp-up and double depletion
            up_len = self.get('M_length_R{}'.format(res))-gauss_length/2
            M_up = create_pulse(shape=self.pulse_primitive_shape(),
                                amplitude=self.get('M_amp_R{}'.format(res)),
                                length=up_len,
                                delay=0,
                                phase=self.get('M_phi_R{}'.format(res)),
                                sampling_rate=sampling_rate)

            M_down0 = create_pulse(shape=self.pulse_primitive_shape(),
                                   amplitude=self.get(
                                       'M_down_amp0_R{}'.format(res)),
                                   length=self.get(
                                       'M_down_length0_R{}'.format(res)),  # ns
                                   delay=0,
                                   phase=self.get(
                                       'M_down_phi0_R{}'.format(res)),
                                   sampling_rate=sampling_rate)

            down1_len = self.get(
                'M_down_length1_R{}'.format(res))-gauss_length/2
            M_down1 = create_pulse(shape=self.pulse_primitive_shape(),
                                   amplitude=self.get(
                                       'M_down_amp1_R{}'.format(res)),
                                   length=down1_len,
                                   delay=0,
                                   phase=self.get(
                                       'M_down_phi1_R{}'.format(res)),
                                   sampling_rate=sampling_rate)

            M_up_down_down = (np.concatenate((M_up[0], M_down0[0], M_down1[0])),
                              np.concatenate((M_up[1], M_down0[1], M_down1[1])))
            res_wave_dict['M_up_down_down_R{}'.format(res)] = M_up_down_down

            # pulse with up, down, down depletion with an additional final
            # strong measurement at some delay
            M_final = create_pulse(shape=self.pulse_primitive_shape(),
                                   amplitude=self.get(
                                       'M_final_amp_R{}'.format(res)),
                                   length=self.get(
                                       'M_final_length_R{}'.format(res)),  # ns
                                   delay=self.get(
                                       'M_final_delay_R{}'.format(res)),
                                   phase=self.get('M_phi_R{}'.format(res)),
                                   sampling_rate=sampling_rate)

            M_up_down_down_final = (np.concatenate((M_up_down_down[0], M_final[0])),
                                    np.concatenate((M_up_down_down[1], M_final[1])))
            res_wave_dict['M_up_down_down_final_R{}'.format(
                res)] = M_up_down_down_final

            # 2. convolve with gaussian (if desired)
            if self.gaussian_convolution():
                for key, val in res_wave_dict.items():
                    M_conv0 = np.convolve(val[0], norm_gauss_p)
                    M_conv1 = np.convolve(val[1], norm_gauss_p)
                    #M_conv0 = M_conv0[hgsl: -hgsl+1]
                    #M_conv1 = M_conv1[hgsl: -hgsl+1]
                    res_wave_dict[key] = (
                        M_conv0/sampling_rate, M_conv1/sampling_rate)

            # 3. modulation with base frequency
            for key, val in res_wave_dict.items():
                res_wave_dict[key] = wf.mod_pulse(pulse_I=val[0], pulse_Q=val[1],
                                                  f_modulation=self.get(
                                                      'M_modulation_R{}'.format(res)),
                                                  sampling_rate=self.get('sampling_rate'))

            # 4. apply mixer predistortion
            if self.mixer_apply_predistortion_matrix():
                Mat = wf.mixer_predistortion_matrix(
                    self.mixer_alpha(), self.mixer_phi())
                for key, val in res_wave_dict.items():
                    res_wave_dict[key] = np.dot(Mat, val)

            # 5. Add to global dict
            self._wave_dict.update(**res_wave_dict)

        return self._wave_dict


class UHFQC_RO_LutMan(Base_RO_LutMan):

    def __init__(self, name, num_res: int=1, feedline_number: int=0,
                 feedline_map='S7', **kw):
        super().__init__(name, num_res=num_res,
                         feedline_number=feedline_number,
                         feedline_map=feedline_map, **kw)
        self.add_parameter('acquisition_delay',
                           vals=vals.Numbers(min_value=0), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=270e-9)
        self.add_parameter('timeout',
                           vals=vals.Numbers(min_value=0), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=5)

        # Set to a default because box is not expected to change
        self._voltage_min = -1.0
        self._voltage_max = 1.0-1.0/2**13
        # Used to select which waveform to upload in the 'single_pulse' mode.
        self._single_pulse_name = None
        # By default, use the DIO triggered mode
        self._mode = 'DIO_triggered'
        # Sample rate of the instrument
        self.sampling_rate(1.8e9)

    def load_single_pulse_sequence_onto_UHFQC(self, pulse_name,
                                              regenerate_waveforms=True):
        '''
        Load a single pulse to the lookuptable, it uses the lut_mapping to
            determine which lookuptable to load to.
        '''
        wf_idx = get_wf_idx_from_name(pulse_name, self.LutMap())
        if not wf_idx:
            raise KeyError(
                'Waveform named "{}" does not exist in the LutMap! Make sure that "pulse_type" "resonator_combinations" is set correctly.')

        self._mode = 'single_pulse'
        self._single_pulse_name = pulse_name
        self.load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=regenerate_waveforms)

    def load_DIO_triggered_sequence_onto_UHFQC(self,
                                               regenerate_waveforms=True,
                                               timeout=5):
        '''
        Load a single pulse to the lookuptable.
        '''
        self._mode = 'DIO_triggered'
        self.timeout(timeout)
        self.load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=regenerate_waveforms)

    def set_mixer_offsets(self):
        UHFQC = self.AWG.get_instr()
        UHFQC.sigouts_0_offset(self.mixer_offs_I())
        UHFQC.sigouts_1_offset(self.mixer_offs_Q())

    def load_waveform_onto_AWG_lookuptable(
            self, wave_id: str, regenerate_waveforms: bool=False):
        """
        Load a waveform into the AWG.

        Args:
            wave_id: can be either the "name" of a waveform or
                the integer codeword corresponding to a combination of
                readout waveforms.
            regenerate_waveforms (bool) : if True regenerates all waveforms
        """
        if regenerate_waveforms:
            self.generate_standard_waveforms()

        if wave_id not in self.LutMap().keys():
            wave_id = get_wf_idx_from_name(wave_id, self.LutMap())

        # Create the combined waveform
        I_wave = np.array([])
        Q_wave = np.array([])
        for resonator in self.LutMap()[wave_id]['resonators']:
            # Create the waveform name
            wavename = self.pulse_type() + '_R' + str(resonator)
            # adding new wave (not necessarily same length)
            I_wave = add_waves_different_length(
                I_wave, self._wave_dict[wavename][0])
            Q_wave = add_waves_different_length(
                Q_wave, self._wave_dict[wavename][1])

        # clipping the waveform
        I_wave = np.clip(I_wave, self._voltage_min, self._voltage_max)
        Q_wave = np.clip(Q_wave, self._voltage_min, self._voltage_max)

        # I and Q hardcoded to channels 1 and 2, codeword hardcoded to zero
        # in 'single_pulse' mode. In this mode we only ever upload a single
        # waveform to the instrument!
        if self._mode == 'single_pulse':
            # With this extra if-statement we prevent uploading to other waveforms
            # than codeword 0 with the waveform selected by the user.
            if self._single_pulse_name == self.LutMap()[wave_id]['name']:
                self.AWG.get_instr().set('wave_ch1_cw000', I_wave)
                self.AWG.get_instr().set('wave_ch2_cw000', Q_wave)
        else:
            self.AWG.get_instr().set(
                'wave_ch1_cw{:03}'.format(wave_id), I_wave)
            self.AWG.get_instr().set(
                'wave_ch2_cw{:03}'.format(wave_id), Q_wave)

    def load_waveforms_onto_AWG_lookuptable(
            self,
            regenerate_waveforms: bool=True,
            stop_start: bool = True,
            # FIXME, force load should be False but is here now to hack around the _upload_updated_waveforms
            force_load_sequencer_program: bool=True):
        # Uploading the codeword program (again) is needed to if the user
        # has changed the mode of the instrument.
        if force_load_sequencer_program:
            if self._mode == 'single_pulse':
                self.AWG.get_instr().awg_sequence_acquisition_and_pulse(
                    acquisition_delay=self.acquisition_delay())
            else:
                self.AWG.get_instr().awg_sequence_acquisition_and_DIO_triggered_pulse(
                    cases=list(self.LutMap().keys()),
                    acquisition_delay=self.acquisition_delay(),
                    timeout=self.timeout())

        super().load_waveforms_onto_AWG_lookuptable(
            regenerate_waveforms=regenerate_waveforms,
            stop_start=stop_start)


def add_waves_different_length(a, b):
    """
    Helper method used to add two arrays of different lengths.
    """
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c
