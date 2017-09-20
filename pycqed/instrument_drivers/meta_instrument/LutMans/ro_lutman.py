from .base_lutman import Base_LutMan
from pycqed.measurement.waveform_control_CC import waveform as wf


class Base_RO_LutMan(Base_LutMan):
    def __init__(self, name, num_res: int=1, **kw):
        super().__init__(name, **kw)
        self._num_res = num_res

    def _add_waveform_parameters(self):
        #mixer corrections are done globally, can be specified per resonator
        self.add_parameter('mixer_apply_predistortion_matrix', vals=vals.Bool(),
                    parameter_class=ManualParameter,
                    initial_value=False)
        self.add_parameter('mixer_alpha', vals=vals.Numbers(),
                        parameter_class=ManualParameter,
                        initial_value=1.0)
        self.add_parameter('mixer_phi', vals=vals.Numbers(), unit='deg',
                        parameter_class=ManualParameter,
                        initial_value=0.0)
        for res in range(self._num_res):
            self.add_parameter('modulation_R{}'.format(res), vals=vals.Numbers(), unit='Hz',
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
                        vals=vals.Numbers(0, 1),
                        parameter_class=ManualParameter,
                               initial_value=0.1)
            self.add_parameter('M_down_amp1_R{}'.format(res), unit='V',
                        vals=vals.Numbers(0, 1),
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
        for res in range(self._num_res)
            M = wf.block_pulse(self.get('M_amp_R{}'.format(res)), self.get('M_length_R{}'.format(res)),  # ns
                               sampling_rate=self.get('sampling_rate'),
                               delay=0,
                               phase=self.get('M_phi_R{}'.format(res)))
            Mod_M = wf.mod_pulse(M[0], M[1],
                                 f_modulation=self.get('M_modulation_R{}'.format(res)),
                                 sampling_rate=self.get('sampling_rate'))
            
            # 3-step RO pulse with ramp-up and double depletion
            M_down0 = wf.block_pulse(self.get('M_down_amp0_R{}'.format(res)), self.get('M_down_length0_R{}'.format(res)),  # ns
                                     sampling_rate=self.get('sampling_rate'),
                                     delay=0,
                                     phase=self.get('M_down_phi0_R{}'.format(res)))

            M_down1 = wf.block_pulse(self.get('M_down_amp1{}'.format(res)), self.get('M_down_length1_R{}'.format(res)),  # ns
                                     sampling_rate=self.get('sampling_rate'),
                                     delay=0,
                                     phase=self.get('M_down_phi1_R{}'.format(res)))

            # concatenating up, down, down depletion
            M_up_down_down = (np.concatenate(M[0], M_down0[0], M_down1[0]),
                              np.concatenate(M[1], M_down0[1], M_down1[1]))
            Mod_M_up_down_down = wf.mod_pulse(M_up_down_down[0], M_up_down_down[1],
                                 f_modulation=self.get('M_modulation_R{}'.format(res)),
                                 sampling_rate=self.sampling_rate())
            self._wave_dict.update({'M_R{}'.format(res): Mod_M,
                               'M_up_down_down_R{}'.format(res): Mod_M_up_down_down})

        if self.mixer_apply_predistortion_matrix():
            M = wf.mixer_predistortion_matrix(self.mixer_alpha(), self.mixer_phi())
            for key, val in self._wave_dict.items():
                    self._wave_dict[key] = np.dot(M, val)
        return self._wave_dict

class UHFQC_RO_LutMan(Base_RO_LutMan):
        self.add_parameter('acquisition_delay', vals=vals.Numbers(), unit='ns',
                        parameter_class=ManualParameter,
                        initial_value=270e-9)
                # Set to a default because box is not expected to change
        self._voltage_min = -1.0
        self._voltage_max = 1.0-1.0/2**13
        def load_single_pulse_sequence_onto_UHFQC(self, pulse_name,
                                        regenerate_pulses=True):
            '''
            Load a single pulse to the lookuptable, it uses the lut_mapping to
                determine which lookuptable to load to.
            '''
            if regenerate_pulses:
                wave_dict = self.generate_standard_pulses()
            else:
                wave_dict = self._wave_dict

            I_wave = np.clip(wave_dict[pulse_name][0],
                             self._voltage_min, self._voltage_max)
            Q_wave = np.clip(wave_dict[pulse_name][1], self._voltage_min,
                             self._voltage_max)
            self.UHFQC.awg_sequence_acquisition_and_pulse(I_wave, Q_wave,
                                                          self.acquisition_delay())

        def load_DIO_triggered_sequence_onto_UHFQC(self, pulse_type, resonatorcombinations,
                                        regenerate_pulses=True):
            '''
            Load a single pulse to the lookuptable, it uses the lut_mapping to
                determine which lookuptable to load to.
            '''
            if regenerate_pulses:
                wave_dict = self.generate_standard_pulses()
            else:
                wave_dict = self._wave_dict
            I_waves = []
            Q_waves = []
            for i,resonatorcombination in np.enumerate(resonatorcombinations):
                for j,resonator in np.enumerate(resonatorcombination):
                    wavename=pulse_type+resonator
                    print(wavename)
                    if j==0:
                        #starting the entry
                        I_waves.append(wave_dict[wavename][0])
                        Q_waves.append(wave_dict[wavename][1])
                    else:
                        #adding new wave (not necessarily same length (have to be same length for now)
                        I_waves[i]+=wave_dict[wavename][0]
                        Q_waves[i]+=wave_dict[wavename][1]
                #clipping the waveform
                I_waves[i]=np.clip(I_waves[i],
                             self._voltage_min, self._voltage_max)
                Q_waves[i]=np.clip(Q_waves[i], self._voltage_min,
                             self._voltage_max)
            self.UHFQC.awg_sequence_acquisition_and_DIO_triggered_pulse(I_waves, Q_waves,
                                                          self.acquisition_delay())
    pass