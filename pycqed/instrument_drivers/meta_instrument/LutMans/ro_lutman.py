from .base_lutman import Base_LutMan
from pycqed.measurement.waveform_control_CC import waveform as wf
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import numpy as np
import copy as copy


class Base_RO_LutMan(Base_LutMan):

    def __init__(self, name, num_res: int=1, **kw):
        if num_res > 9:
            raise ValueError('At most 9 resonators can be read out.')
        self._num_res = num_res
        super().__init__(name, **kw)

    def _add_waveform_parameters(self):
        # mixer corrections are done globally, can be specified per resonator
        self.add_parameter('mixer_apply_predistortion_matrix',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=False)
        self.add_parameter('mixer_alpha', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('mixer_phi', vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        comb_msg = (
            'Resonator combinations specifies blablab needs to be format like bla example ablabj ')
        self.add_parameter('resonator_combinations', vals=vals.Lists(),
                           parameter_class=ManualParameter,
                           docstring=comb_msg,
                           initial_value=[[0]])
        self.add_parameter('pulse_type', vals=vals.Enum(
            'M_up_down_down', 'M_square'),
            parameter_class=ManualParameter,
            docstring=comb_msg,
            initial_value='M_square')
        for res in range(self._num_res):
            self.add_parameter('M_modulation_R{}'.format(res),
                               vals=vals.Numbers(), unit='Hz',
                               parameter_class=ManualParameter,
                               initial_value=20.0e6)
            self.add_parameter('M_length_R{}'.format(res), unit='s',
                               vals=vals.Numbers(1e-9, 8000e-9),
                               parameter_class=ManualParameter,
                               initial_value=300e-9)
            self.add_parameter('M_amp_R{}'.format(res), unit='V',
                               vals=vals.Numbers(0, 1),
                               parameter_class=ManualParameter,
                               initial_value=0.1)
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

    def generate_standard_waveforms(self):
        """
        """
        # Only generate the combinations required/specified in the LutMap
        # RO pulses
        self._wave_dict = {}
        for res in range(self._num_res):
            M = wf.block_pulse(self.get('M_amp_R{}'.format(res)),
                               self.get('M_length_R{}'.format(res)),  # ns
                               sampling_rate=self.get('sampling_rate'),
                               delay=0,
                               phase=self.get('M_phi_R{}'.format(res)))
            Mod_M = wf.mod_pulse(M[0], M[1],
                                 f_modulation=self.get(
                                     'M_modulation_R{}'.format(res)),
                                 sampling_rate=self.get('sampling_rate'))

            # 3-step RO pulse with ramp-up and double depletion
            M_down0 = wf.block_pulse(self.get('M_down_amp0_R{}'.format(res)),
                                     self.get(
                                         'M_down_length0_R{}'.format(res)),  # ns
                                     sampling_rate=self.get('sampling_rate'),
                                     delay=0,
                                     phase=self.get('M_down_phi0_R{}'.format(res)))

            M_down1 = wf.block_pulse(self.get('M_down_amp1_R{}'.format(res)),
                                     self.get(
                                         'M_down_length1_R{}'.format(res)),  # ns
                                     sampling_rate=self.get('sampling_rate'),
                                     delay=0,
                                     phase=self.get('M_down_phi1_R{}'.format(res)))

            # concatenating up, down, down depletion
            M_up_down_down = (np.concatenate((M[0], M_down0[0], M_down1[0])),
                              np.concatenate((M[1], M_down0[1], M_down1[1])))
            Mod_M_up_down_down = wf.mod_pulse(M_up_down_down[0], M_up_down_down[1],
                                              f_modulation=self.get(
                                                  'M_modulation_R{}'.format(res)),
                                              sampling_rate=self.sampling_rate())
            self._wave_dict.update({'M_square_R{}'.format(res): Mod_M,
                                    'M_up_down_down_R{}'.format(res): Mod_M_up_down_down})

        if self.mixer_apply_predistortion_matrix():
            M = wf.mixer_predistortion_matrix(
                self.mixer_alpha(), self.mixer_phi())
            for key, val in self._wave_dict.items():
                self._wave_dict[key] = np.dot(M, val)
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

        self.add_parameter('hardcode_cases', vals=vals.Lists(),
                           parameter_class=ManualParameter,
                           initial_value=[])


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
                                               hardcode_cases=None,
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

        # TODO: automatically assign right codeword for resonator combinations
        # 1. convert each combination to binary to extract the expected CW
        # 2. create a list of these binary numbers or put the combinations in
        #   a dict with these numbers as keys.
        # 3. When uploading, ensure that the pulses are loaded to the right
        # number as specified in 2.

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

                    cases[i] += 2**resonator

            # clipping the waveform
            I_waves[i] = np.clip(I_waves[i],
                                 self._voltage_min, self._voltage_max)
            Q_waves[i] = np.clip(Q_waves[i], self._voltage_min,
                                 self._voltage_max)

        if self.hardcode_cases() != []:
            cases = self.hardcode_cases()

        self.AWG.get_instr().awg_sequence_acquisition_and_DIO_triggered_pulse(
            I_waves, Q_waves, cases, self.acquisition_delay(), timeout=timeout)

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
