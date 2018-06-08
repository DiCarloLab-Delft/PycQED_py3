from .base_lutman import Base_LutMan
from pycqed.measurement.waveform_control_CC import waveform as wf
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import numpy as np
import copy as copy


class Base_RO_LutMan(Base_LutMan):

    def __init__(self, name, num_res=2, feedline_number: int=0, **kw):
        if num_res > 9:
            raise ValueError('At most 9 resonators can be read out.')
        self._num_res = num_res
        self._feedline_number = feedline_number
        if self._feedline_number == 0:
            self._resonator_codeword_bit_mapping = [0, 2, 3, 5, 6]
        elif self._feedline_number == 1:
            self._resonator_codeword_bit_mapping = [1, 4]
        else:
            raise NotImplementedError(
              'hardcoded for feedline 0 and 1 of Surface-7')
        #capping the resonator bit mapping in case a limited number of resonators is used
        self._resonator_codeword_bit_mapping = self._resonator_codeword_bit_mapping[:self._num_res]
        super().__init__(name, **kw)

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
        self.add_parameter('resonator_combinations', vals=vals.Lists(),
                           parameter_class=ManualParameter,
                           docstring=comb_msg,
                           initial_value=[[0], [2], [0, 2]])
        self.add_parameter('pulse_type', vals=vals.Enum(
            'M_up_down_down', 'M_simple', 'M_up_down_down_final'),
            parameter_class=ManualParameter,
            docstring='defines sequence of segments of the pulse',
            initial_value='M_simple')
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

    def add_pulse(self, amplitude: float, length: float, phase: float,
                  delay: float=0):
        kw = {}
        if self.pulse_primitive_shape() == 'square':
            shape_function = wf.block_pulse
            kw['length'] = length
        elif self.pulse_primitive_shape() == 'gaussian':
            shape_function = wf.gauss_pulse
            nr_sigma = 4
            kw['sigma_length'] = length/nr_sigma
            kw['nr_sigma'] = nr_sigma
        else:
            raise NotImplementedError('Primitive pulse shape ' +
                                      self.pulse_primitive_shape() +
                                      ' not implemented.')

        M = shape_function(amp=amplitude,
                           sampling_rate=self.get('sampling_rate'),
                           delay=delay,
                           phase=phase, **kw)
        return M

    def generate_standard_waveforms(self):
        """
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

            gauss_length = 6*sigma# in units of time
            hgsl = int(gauss_length*sampling_rate/2)# half no. of samples
            gauss_sample_length = hgsl*2# no. of samples
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
            ## Simple pulse
            up_len = self.get('M_length_R{}'.format(res))-gauss_length
            M = self.add_pulse(self.get('M_amp_R{}'.format(res)), up_len,
                               delay=0,
                               phase=self.get('M_phi_R{}'.format(res)))
            res_wave_dict['M_simple_R{}'.format(res)] = M

            ## 3-step RO pulse with ramp-up and double depletion
            up_len = self.get('M_length_R{}'.format(res))-gauss_length/2
            M_up = self.add_pulse(self.get('M_amp_R{}'.format(res)),
                                  up_len, delay=0,
                                  phase=self.get('M_phi_R{}'.format(res)))
            M_down0 = self.add_pulse(self.get('M_down_amp0_R{}'.format(res)),
                                     self.get('M_down_length0_R{}'.format(res)),  # ns
                                     delay=0,
                                     phase=self.get('M_down_phi0_R{}'.format(res)))

            down1_len = self.get('M_down_length1_R{}'.format(res))-gauss_length/2
            M_down1 = self.add_pulse(self.get('M_down_amp1_R{}'.format(res)),
                                     down1_len, delay=0,
                                     phase=self.get('M_down_phi1_R{}'.format(res)))

            M_up_down_down = (np.concatenate((M_up[0], M_down0[0], M_down1[0])),
                              np.concatenate((M_up[1], M_down0[1], M_down1[1])))
            res_wave_dict['M_up_down_down_R{}'.format(res)] = M_up_down_down

            ## pulse with up, down, down depletion with an additional final
            ## strong measurement at some delay
            M_final = self.add_pulse(self.get('M_final_amp_R{}'.format(res)),
                               self.get('M_final_length_R{}'.format(res)),  # ns
                               delay=self.get('M_final_delay_R{}'.format(res)),
                               phase=self.get('M_phi_R{}'.format(res)))

            M_up_down_down_final = (np.concatenate((M_up_down_down[0], M_final[0])),
                                    np.concatenate((M_up_down_down[1], M_final[1])))
            res_wave_dict['M_up_down_down_final_R{}'.format(res)] = M_up_down_down_final

            # 2. convolve with gaussian (if desired)
            if self.gaussian_convolution():
                for key, val in res_wave_dict.items():
                    M_conv0 = np.convolve(val[0], norm_gauss_p)
                    M_conv1 = np.convolve(val[1], norm_gauss_p)
                    #M_conv0 = M_conv0[hgsl: -hgsl+1]
                    #M_conv1 = M_conv1[hgsl: -hgsl+1]
                    res_wave_dict[key] = (M_conv0/sampling_rate, M_conv1/sampling_rate)

            # 3. modulation with base frequency
            for key, val in res_wave_dict.items():
                res_wave_dict[key] = wf.mod_pulse(pulse_I=val[0], pulse_Q=val[1],
                                        f_modulation=self.get('M_modulation_R{}'.format(res)),
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

    def __init__(self, name, num_res: int=1, **kw):
        super().__init__(name, num_res=num_res, **kw)
        self.add_parameter('acquisition_delay',
                           vals=vals.Numbers(min_value=0), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=270e-9)

        # Set to a default because box is not expected to change
        self._voltage_min = -1.0
        self._voltage_max = 1.0-1.0/2**13
        self.sampling_rate(1.8e9)


    def set_default_lutmap(self):
        """
        The LutMap parameter is not used for the UHFQC. Instead this is
        determined on the fly using the "resonator_combinations" and "pulse_type"
        parameters.
        """
        self.LutMap({})

    def load_single_pulse_sequence_onto_UHFQC(self, pulse_name,
                                              regenerate_waveforms=True):
        '''
        Load a single pulse to the lookuptable, it uses the lut_mapping to
            determine which lookuptable to load to.
        '''
        if regenerate_waveforms:
            wave_dict = self.generate_standard_waveforms()
        else:
            wave_dict = self._wave_dict

        I_wave = np.clip(wave_dict[pulse_name][0],
                         self._voltage_min, self._voltage_max)
        Q_wave = np.clip(wave_dict[pulse_name][1], self._voltage_min,
                         self._voltage_max)
        self.AWG.get_instr().awg_sequence_acquisition_and_pulse(
            I_wave, Q_wave, self.acquisition_delay())

    def load_DIO_triggered_sequence_onto_UHFQC(self,
                                               regenerate_waveforms=True,
                                               timeout=5):
        '''
        Load a single pulse to the lookuptable, it uses the lut_mapping to
            determine which lookuptable to load to.

        hardcode_case_0 is a workaround as long as it's not clear how
        to make the qisa assembler output the codeword mask on DIO.
        The first element of self.resonator_combinations is linked
        to codeword 0, the others are not uploaded.
        '''
        resonator_combinations = self.resonator_combinations()

        pulse_type = self.pulse_type()
        if regenerate_waveforms:
            wave_dict = self.generate_standard_waveforms()
        else:
            wave_dict = self._wave_dict
        I_waves = []
        Q_waves = []
        cases = np.zeros([len(resonator_combinations)])

        for i, resonator_combination in enumerate(resonator_combinations):
            if not resonator_combination:
                # empty combination, generating empty 20 ns pulse
                I_waves.append(np.zeros(36))
                Q_waves.append(np.zeros(36))
            else:
                for j, resonator in enumerate(resonator_combination):
                    wavename = pulse_type+'_R'+str(resonator)
                    if j == 0:
                        # starting the entry
                        Icopy = copy.deepcopy(wave_dict[wavename][0])
                        Qcopy = copy.deepcopy(wave_dict[wavename][1])
                        I_waves.append(Icopy)
                        Q_waves.append(Qcopy)
                    else:
                        # adding new wave (not necessarily same length)
                        I_waves[i] = add_waves_different_length(
                            I_waves[i], wave_dict[wavename][0])
                        Q_waves[i] = add_waves_different_length(
                            Q_waves[i], wave_dict[wavename][1])
                    cases[i] += 2**self._resonator_codeword_bit_mapping.index(resonator)

            # clipping the waveform
            I_waves[i] = np.clip(I_waves[i],
                                 self._voltage_min, self._voltage_max)
            Q_waves[i] = np.clip(Q_waves[i], self._voltage_min,
                                 self._voltage_max)

        self.AWG.get_instr().awg_sequence_acquisition_and_DIO_triggered_pulse(
            I_waves, Q_waves, cases, self.acquisition_delay(), timeout=timeout)

    def set_mixer_offsets(self):
        UHFQC = self.AWG.get_instr()
        UHFQC.sigouts_0_offset(self.mixer_offs_I())
        UHFQC.sigouts_1_offset(self.mixer_offs_Q())

    def load_waveforms_onto_AWG_lookuptable(
            self, regenerate_waveforms: bool=True,
            stop_start: bool = True):
        raise NotImplementedError(
            'UHFQC needs a full sequence, use '
            '"load_DIO_triggered_sequence_onto_UHFQC"')


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
