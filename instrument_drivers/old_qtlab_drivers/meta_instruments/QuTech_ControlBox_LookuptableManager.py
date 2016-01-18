
import qt
import time
import numpy as np
import sys
import serial
from instrument import Instrument
import logging
from bitstring import BitArray
import defHeaders  # File containing translation from commands to bytestrings
from modules.measurement import CBox_Pulse_Generator as PG

import matplotlib.pyplot as plt
import imp
imp.reload(PG)


class QuTech_ControlBox_LookuptableManager(Instrument):
    '''
    meta-instrument that handles loading pulses into the CBox lookuptables
    and holds their parameters so that they can be sweeped and are logged.

    For now this is a test version that only stores the parameters for a
    specific set of pulses.
    '''

    def __init__(self, name, CBox='CBox', reset=False):

        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['Meta-Instrument'])

        self.CBox = qt.instruments[CBox]

        self.add_parameter('amp180', units='mV',
                           type=float, flag=Instrument.FLAG_GETSET)
        self.add_parameter('amp90', units='mV',
                           type=float, flag=Instrument.FLAG_GETSET)
        self.add_parameter('ampCW', units='mV',
                           type=float, flag=Instrument.FLAG_GETSET)
        self.add_parameter('motzoi_parameter', type=float,
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('QI_amp_ratio', type=float,
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('IQ_phase_skewness', type=float, units='deg',
                           flag=Instrument.FLAG_GETSET)

        self.add_parameter('gauss_width', type=float, units='ns',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('f_modulation', type=float, units='GHz',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('f_modulation_ground', type=float, units='GHz',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('f_modulation_excited', type=float, units='GHz',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('sampling_rate', type=float, units='GHz',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_delay', type=float, units='ns',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_length', type=float, units='ns',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_length_CLEAR_unc', type=float, units='ns',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_length_CLEAR_c', type=float, units='ns',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_amp', type=float, units='mV',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_amp_CLEAR_1', type=float, units='mV',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_amp_CLEAR_2', type=float, units='mV',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_amp_CLEAR_a1', type=float, units='mV',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_amp_CLEAR_a2', type=float, units='mV',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_amp_CLEAR_b1', type=float, units='mV',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_amp_CLEAR_b2', type=float, units='mV',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_phase_CLEAR_1', type=float, units='degrees',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_phase_CLEAR_2', type=float, units='degrees',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_phase_CLEAR_a1', type=float, units='degrees',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_phase_CLEAR_a2', type=float, units='degrees',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_phase_CLEAR_b1', type=float, units='degrees',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('M_phase_CLEAR_b2', type=float, units='degrees',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('feedback_pulse_delay', type=float, units='ns',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('Ramsey_idling', type=float, units='ns',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('lut_mapping', type=list,
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('CLEAR_double_segment', type=bool,
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('CLEAR_double_frequency', type=bool,
                           flag=Instrument.FLAG_GETSET)
        # These parameters are added for mixer skewness correction.
        # They are intended to be renamed such that they can be combined with
        # QI_amp_ratio and IQ_phase_skewness.
        self.add_parameter('alpha', type=float,
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('phi', type=float, units='deg',
                           flag=Instrument.FLAG_GETSET)
        self.add_parameter('apply_predistortion_matrix', type=bool,
                           flag=Instrument.FLAG_GETSET)

        self.set_lut_mapping(['I', 'X180', 'Y180', 'X90', 'Y90', 'I',  # 'Block',
                             'X180_delayed'])
        # Set to a default because box is not expected to change
        self.set_sampling_rate(0.2)
        self.set_QI_amp_ratio(1)
        self._do_set_IQ_phase_skewness(0)
        self.set_apply_predistortion_matrix(False)
        self.set_M_length_CLEAR_unc(0)
        self.set_M_length_CLEAR_c(0)
        self.set_M_length(0)
        self.set_M_amp(0)
        self.set_CLEAR_double_segment(0)
        self.set_M_delay(0)
        self.set_M_amp_CLEAR_1(0)
        self.set_M_amp_CLEAR_2(0)
        self.set_M_amp_CLEAR_a1(0)
        self.set_M_amp_CLEAR_a2(0)
        self.set_M_amp_CLEAR_b1(0)
        self.set_M_amp_CLEAR_b2(0)
        self.set_M_phase_CLEAR_1(0)
        self.set_M_phase_CLEAR_2(0)
        self.set_M_phase_CLEAR_a1(0)
        self.set_M_phase_CLEAR_a2(0)
        self.set_M_phase_CLEAR_b1(0)
        self.set_M_phase_CLEAR_b2(0)
        self.set_CLEAR_double_segment(0)
        self.set_CLEAR_double_frequency(False)
        self.set_amp180(0)
        self.set_amp90(0)
        self.set_ampCW(0)
        self.set_gauss_width(10)
        self.set_f_modulation(-0.02)
        self.set_f_modulation_ground(0.0)
        self.set_f_modulation_excited(0.0)
        self.set_feedback_pulse_delay(0)
        self.set_motzoi_parameter(0)
        self.set_Ramsey_idling(0)
        self.voltage_min = -1000
        self.voltage_max = 1000-1000./2**13

    def generate_standard_pulses(self):
        '''
        Generates a basic set of pulses (I, X-180, Y-180, x-90, y-90, Block,
                                         X180_delayed)
        using the parameters set on this meta-instrument and returns the
        corresponding waveforms for both I and Q channels as a dict.

        Note the primitive set is a different set than the one used in
        Serwan's thesis.
        '''
        # Standard qubit pulses
        Wave_I = [np.zeros(10), np.zeros(10)]
        Wave_X_180 = PG.mod_gauss(self.amp180, self.gauss_width,
                                  self.f_modulation, axis='x',
                                  motzoi=self.motzoi_parameter,
                                  sampling_rate=self.sampling_rate,
                                  Q_phase_delay=self.IQ_phase_skewness)
        Wave_X_90 = PG.mod_gauss(self.amp90, self.gauss_width,
                                 self.f_modulation, axis='x',
                                 motzoi=self.motzoi_parameter,
                                 sampling_rate=self.sampling_rate,
                                 Q_phase_delay=self.IQ_phase_skewness)

        Wave_Y_180 = PG.mod_gauss(self.amp180, self.gauss_width,
                                  self.f_modulation, axis='y',
                                  motzoi=self.motzoi_parameter,
                                  sampling_rate=self.sampling_rate,
                                  Q_phase_delay=self.IQ_phase_skewness)
        Wave_Y_90 = PG.mod_gauss(self.amp90, self.gauss_width,
                                 self.f_modulation, axis='y',
                                 motzoi=self.motzoi_parameter,
                                 sampling_rate=self.sampling_rate,
                                 Q_phase_delay=self.IQ_phase_skewness)

        Wave_X_180_delayed = PG.mod_gauss(self.amp180, self.gauss_width,
                                          self.f_modulation, axis='x',
                                          motzoi=self.motzoi_parameter,
                                          sampling_rate=self.sampling_rate,
                                          Q_phase_delay=self.IQ_phase_skewness,
                                          delay=self.feedback_pulse_delay)
        Wave_mX90 = PG.mod_gauss(-self.amp90, self.gauss_width,
                                 self.f_modulation, axis='x',
                                 motzoi=self.motzoi_parameter,
                                 sampling_rate=self.sampling_rate,
                                 Q_phase_delay=self.IQ_phase_skewness)

        mod_block, M0, M1, M2, M3_a, M3_b = self.generate_resonator_pulses()
        (Wave_X90_X90, Wave_X90_mX90,
         Wave_X90_X180_mX90, Wave_X90_X180_X90, Wave_3X180) = \
            self.generate_composite_qubit_pulses(Wave_X_90, Wave_mX90,
                                                 Wave_X_180)
        self.wave_dict = {'I': Wave_I,
                          'X180': Wave_X_180, 'Y180': Wave_Y_180,
                          'X90': Wave_X_90, 'Y90': Wave_Y_90,
                          'Block': mod_block,
                          'X180_delayed': Wave_X_180_delayed,
                          'X90_X90': Wave_X90_X90,
                          'X90_min_X90': Wave_X90_mX90,
                          'X90_X180_mX90': Wave_X90_X180_mX90,
                          'X90_X180_X90': Wave_X90_X180_X90,
                          '3X180': Wave_3X180,
                          'M0': M0, 'M1': M1, 'M2': M2,
                          'M3_a': M3_a, 'M3_b': M3_b}

        if self.apply_predistortion_matrix:
            M = self.get_mixer_predistortion_matrix()
            for key, val in self.wave_dict.items():
                self.wave_dict[key] = np.dot(M, val)

        return self.wave_dict

    def generate_resonator_pulses(self):
        '''
        11/2015 MAR: Moved over the pulses that are used to modulate the
        resonator tone here as the lut man was getting rather convoluted
        '''
        Block_envelope = PG.block_pulse(amp=self.ampCW,
                                        length=200, delay=0,
                                        phase=45)
        # Block pulse is used for mixer calibratino, 45 degrees to have
        # equal signal on both I and Q

        # generating measurement envelope
        M_envelope = PG.block_pulse(amp=self.M_amp, length=self.M_length,
                                    delay=0)


        # concatening unconditional CLEAR pulse envelopes
        if not self.CLEAR_double_frequency:
                    # generating unconditional CLEAR pulse segment envelopes
            M_CLEAR_envelope_1 = PG.block_pulse(amp=self.M_amp_CLEAR_1,
                                                length=self.M_length_CLEAR_unc/2,
                                                delay=0, phase=self.M_phase_CLEAR_1)

            M_CLEAR_envelope_2 = PG.block_pulse(amp=self.M_amp_CLEAR_2,
                                                length=self.M_length_CLEAR_unc/2,
                                                delay=0, phase=self.M_phase_CLEAR_2)
            M_CLEAR_envelope_com = (np.concatenate((M_CLEAR_envelope_1[0],
                                                    M_CLEAR_envelope_2[0])),
                                    np.concatenate((M_CLEAR_envelope_1[1],
                                                   M_CLEAR_envelope_2[1])))

            # concatening measurement pulse and unconditional CLEAR pulse envelopes
            M_envelope_com = (np.concatenate((M_envelope[0],
                                              M_CLEAR_envelope_com[0])),
                              np.concatenate((M_envelope[1],
                                              M_CLEAR_envelope_com[1])))

            # modulating the measurement CLEAR pulse
            M_com = PG.mod_pulse(M_envelope_com[0],
                                 M_envelope_com[1],
                                 f_modulation=self.f_modulation,
                                 sampling_rate=self.sampling_rate,
                                 Q_phase_delay=self.IQ_phase_skewness)
        else:
            # generating unconditional CLEAR pulse segment envelopes with double frequency
            M_CLEAR_envelope_1 = PG.block_pulse(amp=self.M_amp_CLEAR_1,
                                                length=self.M_length_CLEAR_unc,
                                                delay=0, phase=self.M_phase_CLEAR_1)

            M_CLEAR_envelope_2 = PG.block_pulse(amp=self.M_amp_CLEAR_2,
                                                length=self.M_length_CLEAR_unc,
                                                delay=0, phase=self.M_phase_CLEAR_2)

            M_CLEAR_envelope_1_mod = PG.mod_pulse(M_CLEAR_envelope_1[0],
                                 M_CLEAR_envelope_1[1],
                                 f_modulation=self.f_modulation_excited,
                                 sampling_rate=self.sampling_rate,
                                 Q_phase_delay=self.IQ_phase_skewness)
            M_CLEAR_envelope_2_mod = PG.mod_pulse(M_CLEAR_envelope_2[0],
                                 M_CLEAR_envelope_2[1],
                                 f_modulation=self.f_modulation_ground,
                                 sampling_rate=self.sampling_rate,
                                 Q_phase_delay=self.IQ_phase_skewness)

            # concatening measurement pulse and unconditional CLEAR pulse envelopes
            M_CLEAR_envelope_mod = (np.add(M_CLEAR_envelope_1_mod[0],
                                            M_CLEAR_envelope_2_mod[0]),
                                    np.add(M_CLEAR_envelope_1_mod[1],
                                              M_CLEAR_envelope_2_mod[1]))


            # modulating the measurement CLEAR pulse
            M_mod = PG.mod_pulse(M_envelope[0],
                                 M_envelope[1],
                                 f_modulation=self.f_modulation,
                                 sampling_rate=self.sampling_rate,
                                 Q_phase_delay=self.IQ_phase_skewness)

            M_com = (np.concatenate((M_mod[0], M_CLEAR_envelope_mod[0])),
                     np.concatenate((M_mod[1], M_CLEAR_envelope_mod[1])))

        # setting delay before the modulated pulse
        M_delay = [np.zeros(self.M_delay*self.sampling_rate),
                   np.zeros(self.M_delay*self.sampling_rate)]

        M = (np.concatenate((M_delay[0], M_com[0])),
             np.concatenate((M_delay[1], M_com[1])))

        #  generating the pulse envelopes for conditional CLEAR pulses

        M_CLEAR_envelope_a1 = PG.block_pulse(amp=self.M_amp_CLEAR_a1,
                                             length=self.M_length_CLEAR_c/2,
                                             delay=0, phase=self.M_phase_CLEAR_a1)
        M_CLEAR_envelope_a2 = PG.block_pulse(amp=self.M_amp_CLEAR_a2,
                                             length=self.M_length_CLEAR_c/2,
                                             delay=0, phase=self.M_phase_CLEAR_a2)
        M_CLEAR_envelope_com_a = (np.concatenate((M_CLEAR_envelope_a1[0],
                                                  M_CLEAR_envelope_a2[0])),
                                  np.concatenate((M_CLEAR_envelope_a1[1],
                                                 M_CLEAR_envelope_a2[1])))

        M_CLEAR_envelope_b1 = PG.block_pulse(amp=self.M_amp_CLEAR_b1,
                                             length=self.M_length_CLEAR_c/2,
                                             delay=0, phase=self.M_phase_CLEAR_b1)
        M_CLEAR_envelope_b2 = PG.block_pulse(amp=self.M_amp_CLEAR_b2,
                                             length=self.M_length_CLEAR_c/2,
                                             delay=0, phase=self.M_phase_CLEAR_b2)
        M_CLEAR_envelope_com_b = (np.concatenate((M_CLEAR_envelope_b1[0],
                                                  M_CLEAR_envelope_b2[0])),
                                  np.concatenate((M_CLEAR_envelope_b1[1],
                                                  M_CLEAR_envelope_b2[1])))
        # modulating the conditional CLEAR pulses
        if not self.CLEAR_double_frequency:
            M3_a = PG.mod_pulse(M_CLEAR_envelope_com_a[0],
                                M_CLEAR_envelope_com_a[1],
                                f_modulation=self.f_modulation,
                                sampling_rate=self.sampling_rate,
                                Q_phase_delay=self.IQ_phase_skewness)
            M3_b = PG.mod_pulse(M_CLEAR_envelope_com_b[0],
                                M_CLEAR_envelope_com_b[1],
                                f_modulation=self.f_modulation,
                                sampling_rate=self.sampling_rate,
                                Q_phase_delay=self.IQ_phase_skewness)
        else:
            M3_a = PG.mod_pulse(M_CLEAR_envelope_com_a[0],
                                M_CLEAR_envelope_com_a[1],
                                f_modulation=self.f_modulation_excited,
                                sampling_rate=self.sampling_rate,
                                Q_phase_delay=self.IQ_phase_skewness)
            M3_b = PG.mod_pulse(M_CLEAR_envelope_com_b[0],
                                M_CLEAR_envelope_com_b[1],
                                f_modulation=self.f_modulation_ground,
                                sampling_rate=self.sampling_rate,
                                Q_phase_delay=self.IQ_phase_skewness)

        mod_block = PG.simple_mod_pulse(Block_envelope[0], Block_envelope[1],
                                        f_modulation=self.f_modulation,
                                        sampling_rate=self.sampling_rate,
                                        Q_phase_delay=self.IQ_phase_skewness)

        # dividing the measurement/CLEAR_uncond wave over 3 different entries
        M_samples = (self.M_delay + self.M_length +
                     self.M_length_CLEAR_unc)*self.sampling_rate
        if np.mod(M_samples*5, 15) != 0.0:
            print(M_samples*5)
            print(np.mod(M_samples*5, 15))
            raise ValueError(
                "sum of M_delay, M_length, M_length_CLEAR_unc is not multiple of 15 ns")
        if M_samples > 128*3:
            print(M_samples)
            raise ValueError(
                "sum of M_delay, M_length, M_length_CLEAR_unc is larger than 1920")

        M0 = (M[0][:int(M_samples/3)], M[1][:int(M_samples/3)])
        M1 = (M[0][int(M_samples/3):int(2*M_samples/3)],
              M[1][int(M_samples/3):int(2*M_samples/3)])
        M2 = (M[0][int(2*M_samples/3):], M[1][int(2*M_samples/3):])

        return mod_block, M0, M1, M2, M3_a, M3_b

    def generate_composite_qubit_pulses(self, Wave_X_90, Wave_mX90,
                                        Wave_X_180):
        '''
        Generates "composite pulses" that do load more than one pulse in the
        lut manager
        11/2015 MAR: Separated from the generate standard pulses to make code
        more readable.
        '''

        I_tau = [np.zeros((2*self.gauss_width+self.Ramsey_idling)
                          * self.sampling_rate),
                 np.zeros((2*self.gauss_width+self.Ramsey_idling)
                          * self.sampling_rate)]
        # I_tau2 adds to waits of tau/2 around the pulses.
        # FIXME: gauss width should be changed in these expressions still
        # like this for backwards compatibility reasons.
        if self.Ramsey_idling/2.0 > self.gauss_width:
            I_tau2 = [np.zeros((self.Ramsey_idling/2.0-self.gauss_width)
                      * self.sampling_rate),
                      np.zeros((self.Ramsey_idling/2.0-self.gauss_width)
                               * self.sampling_rate)]
        else:
            # No warning raised as the Echo is often not used and we would
            # otherwise spam the terminal.
            I_tau2 = [np.array([]), np.array([])]

        Wave_X90_X90 = (np.concatenate((Wave_X_90[0], I_tau[0], Wave_X_90[0])),
                        np.concatenate((Wave_X_90[1], I_tau[1], Wave_X_90[1])))

        Wave_X90_mX90 = (np.concatenate((Wave_X_90[0], I_tau[0], Wave_mX90[0])),
                         np.concatenate((Wave_X_90[1], I_tau[1], Wave_mX90[1])))

        # Echo pulse that flips between rounds
        Wave_X90_X180_mX90 = (
            np.concatenate((Wave_X_90[0], I_tau2[0], Wave_X_180[0],
                            I_tau2[0], Wave_mX90[0])),
            np.concatenate((Wave_X_90[1], I_tau2[1], Wave_X_180[1],
                            I_tau2[1], Wave_mX90[1])))
        Wave_X90_X180_X90 = (
            np.concatenate((Wave_X_90[0], I_tau2[0], Wave_X_180[0],
                            I_tau2[0], Wave_X_90[0])),
            np.concatenate((Wave_X_90[1], I_tau2[1], Wave_X_180[1],
                            I_tau2[1], Wave_X_90[1])))

        Wave_3X180 = (
            np.concatenate((Wave_X_180[0], I_tau2[0], Wave_X_180[0],
                            I_tau2[0], Wave_X_180[0])),
            np.concatenate((Wave_X_180[1], I_tau2[1], Wave_X_180[1],
                            I_tau2[1], Wave_X_180[1])))
        return (Wave_X90_X90, Wave_X90_mX90,
                Wave_X90_X180_mX90, Wave_X90_X180_X90, Wave_3X180)

    def render_wave(self, wave_name, show=True, time_units='lut_index',
                    reload_pulses=True):
        if reload_pulses:
            self.generate_standard_pulses()
        fig, ax = plt.subplots(1, 1)
        if time_units == 'lut_index':
            x = np.arange(len(self.wave_dict[wave_name][0]))
            ax.set_xlabel('Lookuptable index (i)')
            ax.vlines(128, self.voltage_min, self.voltage_max, linestyle='--')
        elif time_units == 'ns':
            x = (np.arange(len(self.wave_dict[wave_name][0]))
                 / self.get_sampling_rate())
            ax.set_xlabel('time (ns)')
            ax.vlines(128 / self.get_sampling_rate(),
                      self.voltage_min, self.voltage_max, linestyle='--')

        ax.plot(x, self.wave_dict[wave_name][0],
                marker='o', label=wave_name+' chI')
        ax.plot(x, self.wave_dict[wave_name][1],
                marker='o', label=wave_name+' chQ')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_axis_bgcolor('gray')
        ax.axhspan(self.voltage_min, self.voltage_max, facecolor='w',
                   linewidth=0)
        ax.legend()
        ax.set_ylim(self.voltage_min*1.1, self.voltage_max*1.1)
        ax.set_xlim(0, x[-1])
        if show:
            plt.show()
        return fig, ax

    def get_mixer_predistortion_matrix(self):
        '''
        predistortion matrix correcting for a mixer with amplitude
        mismatch "alpha" and skewness "phi"

        M = [ 1            tan(phi) ]
            [ 0   1/alpha * sec(phi)]

        Notes on the procedure for acquiring this matrix can be found in
        PycQED/docs/notes/MixerSkewnessCalibration_LDC_150629.pdf
        '''

        mixer_pre_distortion_matrix = np.array(
            ((1,  np.tan(self.phi*2*np.pi/360)),
             (0, 1/self.alpha * 1/np.cos(self.phi*2*np.pi/360))))
        return mixer_pre_distortion_matrix

    def load_pulses_onto_AWG_lookuptable(self, awg_nr):
        '''
        Loads the pulses to the lookuptables, it uses the lut_mapping to
        determine what pulse to load to which lookuptable.
        '''
        self.generate_standard_pulses()
        for i, pulse_name in enumerate(self.lut_mapping):
            self.load_pulse_onto_AWG_lookuptable(pulse_name, awg_nr,
                                                 regenerate_pulses=False)

    def load_pulse_onto_AWG_lookuptable(self, pulse_name, awg_nr,
                                        regenerate_pulses=True):
        '''
        Load a pulses to the lookuptable, it uses the lut_mapping to
        determine which lookuptable to load to.
        '''
        if regenerate_pulses:
            wave_dict = self.generate_standard_pulses()
        else:
            wave_dict = self.wave_dict
        # This is to account for the odd definition in CBox (see CBox Issue #13)
        I_ch = 1
        Q_ch = 0
        I_wave = np.clip(wave_dict[pulse_name][0],
                         self.voltage_min, self.voltage_max)
        Q_wave = np.clip(np.multiply(self.QI_amp_ratio,
                         wave_dict[pulse_name][1]), self.voltage_min,
                         self.voltage_max)
        # To account for multiple occurences in lut mapping
        indices = [i for i, x in enumerate(self.lut_mapping) if x == pulse_name]
        for i in indices:
            self.CBox.set_awg_lookuptable(awg_nr, i, I_ch, I_wave)
            self.CBox.set_awg_lookuptable(awg_nr, i, Q_ch, Q_wave)

    #######################################
    # get/set functions for 'simple' pars #
    #######################################
    def _do_set_amp180(self, val):
        self.amp180 = val

    def _do_get_amp180(self):
        return self.amp180

    def _do_set_lut_mapping(self, val):
        self.lut_mapping = val

    def _do_get_lut_mapping(self):
        return self.lut_mapping

    def _do_set_amp90(self, val):
        self.amp90 = val

    def _do_get_amp90(self):
        return self.amp90

    def _do_set_ampCW(self, val):
        self.ampCW = val

    def _do_get_ampCW(self):
        return self.ampCW

    def _do_set_M_amp(self, val):
        self.M_amp = val

    def _do_get_M_amp(self):
        return self.M_amp

    def _do_set_M_amp_CLEAR_1(self, val):
        if not self.CLEAR_double_segment:
            self.M_amp_CLEAR_2 = val
        self.M_amp_CLEAR_1 = val

    def _do_get_M_amp_CLEAR_1(self):
        return self.M_amp_CLEAR_1

    def _do_set_M_amp_CLEAR_2(self, val):
        self.M_amp_CLEAR_2 = val

    def _do_get_M_amp_CLEAR_2(self):
        return self.M_amp_CLEAR_2

    def _do_set_M_amp_CLEAR_a1(self, val):
        if not self.CLEAR_double_segment:
            self.M_amp_CLEAR_a2 = val
        self.M_amp_CLEAR_a1 = val

    def _do_get_M_amp_CLEAR_a1(self):
        return self.M_amp_CLEAR_a1

    def _do_set_M_amp_CLEAR_a2(self, val):
        self.M_amp_CLEAR_a2 = val

    def _do_get_M_amp_CLEAR_a2(self):
        return self.M_amp_CLEAR_a2

    def _do_set_M_amp_CLEAR_b1(self, val):
        if not self.CLEAR_double_segment:
            self.M_amp_CLEAR_b2 = val
        self.M_amp_CLEAR_b1 = val

    def _do_get_M_amp_CLEAR_b1(self):
        return self.M_amp_CLEAR_b1

    def _do_set_M_amp_CLEAR_b2(self, val):
        self.M_amp_CLEAR_b2 = val

    def _do_get_M_amp_CLEAR_b2(self):
        return self.M_amp_CLEAR_b2

    def _do_set_M_phase_CLEAR_1(self, val):
        if not self.CLEAR_double_segment:
            self.M_phase_CLEAR_2= val
        self.M_phase_CLEAR_1 = val

    def _do_get_M_phase_CLEAR_1(self):
        return self.M_phase_CLEAR_1

    def _do_set_M_phase_CLEAR_2(self, val):
        self.M_phase_CLEAR_2 = val

    def _do_get_M_phase_CLEAR_2(self):
        return self.M_phase_CLEAR_2

    def _do_set_M_phase_CLEAR_a1(self, val):
        if not self.CLEAR_double_segment:
            self.M_phase_CLEAR_a2 = val
        self.M_phase_CLEAR_a1 = val

    def _do_get_M_phase_CLEAR_a1(self):
        return self.M_phase_CLEAR_a1

    def _do_set_M_phase_CLEAR_a2(self, val):
        self.M_phase_CLEAR_a2 = val

    def _do_get_M_phase_CLEAR_a2(self):
        return self.M_phase_CLEAR_a2

    def _do_set_M_phase_CLEAR_b1(self, val):
        if not self.CLEAR_double_segment:
            self.M_phase_CLEAR_b2 = val
        self.M_phase_CLEAR_b1 = val

    def _do_get_M_phase_CLEAR_b1(self):
        return self.M_phase_CLEAR_b1

    def _do_set_M_phase_CLEAR_b2(self, val):
        self.M_phase_CLEAR_b2 = val

    def _do_get_M_phase_CLEAR_b2(self):
        return self.M_phase_CLEAR_b2

    def _do_set_motzoi_parameter(self, val):
        self.motzoi_parameter = val

    def _do_get_motzoi_parameter(self):
        return self.motzoi_parameter

    def _do_set_gauss_width(self, val):
        self.gauss_width = val

    def _do_get_gauss_width(self):
        return self.gauss_width

    def _do_set_f_modulation(self, val):
        self.f_modulation = val

    def _do_get_f_modulation(self):
        return self.f_modulation

    def _do_set_f_modulation_ground(self, val):
        self.f_modulation_ground = val

    def _do_get_f_modulation_ground(self):
        return self.f_modulation_ground

    def _do_set_f_modulation_excited(self, val):
        self.f_modulation_excited = val

    def _do_get_f_modulation_excited(self):
        return self.f_modulation_excited

    def _do_set_M_delay(self, val):
        self.M_delay = val

    def _do_get_M_delay(self):
        return self.M_delay

    def _do_set_M_length(self, val):
        self.M_length = val

    def _do_get_M_length(self):
        return self.M_length

    def _do_set_M_length_CLEAR_unc(self, val):
        self.M_length_CLEAR_unc = val

    def _do_get_M_length_CLEAR_unc(self):
        return self.M_length_CLEAR_unc

    def _do_set_M_length_CLEAR_c(self, val):
        self.M_length_CLEAR_c = val

    def _do_get_M_length_CLEAR_c(self):
        return self.M_length_CLEAR_c

    def _do_set_QI_amp_ratio(self, val):
        self.QI_amp_ratio = val

    def _do_get_QI_amp_ratio(self):
        return self.QI_amp_ratio

    def _do_set_IQ_phase_skewness(self, val):
        self.IQ_phase_skewness = val

    def _do_get_IQ_phase_skewness(self):
        return self.IQ_phase_skewness

    def _do_set_sampling_rate(self, val):
        self.sampling_rate = val

    def _do_get_sampling_rate(self):
        return self.sampling_rate

    def _do_set_alpha(self, val):
        self.alpha = val

    def _do_get_alpha(self):
        return self.alpha

    def _do_set_phi(self, val):
        self.phi = val

    def _do_get_phi(self):
        return self.phi

    def _do_set_feedback_pulse_delay(self, val):
        self.feedback_pulse_delay = val

    def _do_get_feedback_pulse_delay(self):
        return self.feedback_pulse_delay

    def _do_set_Ramsey_idling(self, val):
        self.Ramsey_idling = val

    def _do_get_Ramsey_idling(self):
        return self.Ramsey_idling

    def _do_set_apply_predistortion_matrix(self, val):
        self.apply_predistortion_matrix = val

    def _do_get_apply_predistortion_matrix(self):
        return self.apply_predistortion_matrix

    def _do_set_CLEAR_double_segment(self, val):
        self.CLEAR_double_segment = val

    def _do_get_CLEAR_double_segment(self):
        return self.CLEAR_double_segment

    def _do_set_CLEAR_double_frequency(self, val):
        self.CLEAR_double_frequency = val

    def _do_get_CLEAR_double_frequency(self):
        return self.CLEAR_double_frequency
