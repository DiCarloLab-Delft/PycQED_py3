
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import logging
from pycqed.measurement import Pulse_Generator as PG
import unittest
import matplotlib.pyplot as plt
import imp
from pycqed.analysis.fit_toolbox import functions as func

imp.reload(PG)

global lm  # Global used for passing value to the testsuite


class UHFQC_LookuptableManager(Instrument):

    '''
    meta-instrument that handles loading pulses into the UHFQC lookuptables
    and holds their parameters so that they can be sweeped and are logged.

    For now this is a test version that only stores the parameters for a
    specific set of pulses.
    '''
    shared_kwargs = ['UHFQC']

    def __init__(self, name, UHFQC, **kw):
        logging.warning('The UHFQC_LookuptableManager is deprecated')

        logging.info(__name__ + ' : Initializing instrument')
        super().__init__(name, **kw)

        self.UHFQC = UHFQC

        self.add_parameter('Q_amp180',
                           unit='V',
                           vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.1)
        self.add_parameter('Q_amp90', unit='V',
                           vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.05)
        self.add_parameter('Q_ampCW', unit='V',
                           vals=vals.Numbers(-1, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.05)
        self.add_parameter('Q_block_length', unit='s',
                           vals=vals.Numbers(1e-9, 640e-9),
                           parameter_class=ManualParameter,
                           initial_value=500e-9)
        self.add_parameter('Q_motzoi_parameter', vals=vals.Numbers(-2, 2),
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('Q_gauss_width', vals=vals.Numbers(), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=10e-9)
        self.add_parameter('Q_gauss_nr_sigma', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=4)
        self.add_parameter('mixer_QI_amp_ratio', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('mixer_IQ_phase_skewness', vals=vals.Numbers(),
                           unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('Q_modulation', vals=vals.Numbers(), unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=20.0e6)
        self.add_parameter('sampling_rate', vals=vals.Numbers(), unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=1.8e9)
        # These parameters are added for mixer skewness correction.
        # They are intended to be renamed such that they can be combined with
        # mixer_QI_amp_ratio and mixer_IQ_phase_skewness.
        self.add_parameter('mixer_alpha', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('mixer_phi', vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('mixer_apply_predistortion_matrix', vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=False)
        self.add_parameter('M_modulation', vals=vals.Numbers(), unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=20.0e6)
        self.add_parameter('M_length', unit='s',
                           vals=vals.Numbers(1e-9, 8000e-9),
                           parameter_class=ManualParameter,
                           initial_value=300e-9)
        self.add_parameter('M_amp', unit='V',
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.1)
        self.add_parameter('M_phi', unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('M_up_length', unit='s',
                           vals=vals.Numbers(1e-9, 640e-9),
                           parameter_class=ManualParameter,
                           initial_value=100.0e-9)
        self.add_parameter('M_up_amp', unit='V',
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.1)
        self.add_parameter('M_up_phi', unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('M_down_length', unit='s',
                           vals=vals.Numbers(1e-9, 640e-9),
                           parameter_class=ManualParameter,
                           initial_value=200.0e-9)
        self.add_parameter('M_down_amp0', unit='V',
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.1)
        self.add_parameter('M_down_amp1', unit='V',
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.1)
        self.add_parameter('M_down_phi0', unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=180.0)
        self.add_parameter('M_down_phi1', unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=180.0)
        self.add_parameter('M0_modulation', vals=vals.Numbers(), unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=20.0e6)
        self.add_parameter('M1_modulation', vals=vals.Numbers(), unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=20.0e6)
        self.add_parameter('acquisition_delay', vals=vals.Numbers(), unit='ns',
                           parameter_class=ManualParameter,
                           initial_value=270e-9)

        # Set to a default because box is not expected to change
        self._voltage_min = -1.0
        self._voltage_max = 1.0-1.0/2**13

    def run_test_suite(self):
        # pass the UHFQC to the module so it can be used in the tests
        from importlib import reload
        from .tests import test_suite
        reload(test_suite)
        test_suite.lm = self
        suite = unittest.TestLoader().loadTestsFromTestCase(
            test_suite.LutManTests)
        unittest.TextTestRunner(verbosity=2).run(suite)

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
        Wave_X_180 = PG.mod_gauss(self.get('Q_amp180'), self.get('Q_gauss_width'),
                                  self.get('Q_modulation'), axis='x',
                                  motzoi=self.get('Q_motzoi_parameter'),
                                  sampling_rate=self.get('sampling_rate'),
                                  Q_phase_delay=self.get(
                                      'mixer_IQ_phase_skewness'),
                                  nr_sigma=self.Q_gauss_nr_sigma())
        Wave_X_90 = PG.mod_gauss(self.get('Q_amp90'), self.get('Q_gauss_width'),
                                 self.get('Q_modulation'), axis='x',
                                 motzoi=self.get('Q_motzoi_parameter'),
                                 sampling_rate=self.get('sampling_rate'),
                                 Q_phase_delay=self.get(
                                     'mixer_IQ_phase_skewness'),
                                 nr_sigma=self.Q_gauss_nr_sigma())

        Wave_Y_180 = PG.mod_gauss(self.get('Q_amp180'), self.get('Q_gauss_width'),
                                  self.get('Q_modulation'), axis='y',
                                  motzoi=self.get('Q_motzoi_parameter'),
                                  sampling_rate=self.get('sampling_rate'),
                                  Q_phase_delay=self.get(
                                      'mixer_IQ_phase_skewness'),
                                  nr_sigma=self.Q_gauss_nr_sigma())
        Wave_Y_90 = PG.mod_gauss(self.get('Q_amp90'), self.get('Q_gauss_width'),
                                 self.get('Q_modulation'), axis='y',
                                 motzoi=self.get('Q_motzoi_parameter'),
                                 sampling_rate=self.get('sampling_rate'),
                                 Q_phase_delay=self.get(
                                     'mixer_IQ_phase_skewness'),
                                 nr_sigma=self.Q_gauss_nr_sigma())

        Wave_mX90 = PG.mod_gauss(-self.get('Q_amp90'), self.get('Q_gauss_width'),
                                 self.get('Q_modulation'), axis='x',
                                 motzoi=self.get('Q_motzoi_parameter'),
                                 sampling_rate=self.get('sampling_rate'),
                                 Q_phase_delay=self.get(
                                     'mixer_IQ_phase_skewness'),
                                 nr_sigma=self.Q_gauss_nr_sigma())

        Wave_mY90 = PG.mod_gauss(-self.get('Q_amp90'), self.get('Q_gauss_width'),
                                 self.get('Q_modulation'), axis='y',
                                 motzoi=self.get('Q_motzoi_parameter'),
                                 sampling_rate=self.get('sampling_rate'),
                                 Q_phase_delay=self.get(
                                     'mixer_IQ_phase_skewness'),
                                 nr_sigma=self.Q_gauss_nr_sigma())
        Block = PG.block_pulse(self.get('Q_ampCW'), self.Q_block_length.get(),  # ns
                               sampling_rate=self.get('sampling_rate'),
                               delay=0,
                               phase=0)
        ModBlock = PG.mod_pulse(Block[0], Block[1],
                                f_modulation=self.Q_modulation.get(),
                                sampling_rate=self.sampling_rate.get(),
                                Q_phase_delay=self.mixer_IQ_phase_skewness.get())

        # RO pulses
        M = PG.block_pulse(self.get('M_amp'), self.M_length.get(),  # ns
                           sampling_rate=self.get('sampling_rate'),
                           delay=0,
                           phase=self.get('M_phi'))
        Mod_M = PG.mod_pulse(M[0], M[1],
                             f_modulation=self.M_modulation.get(),
                             sampling_rate=self.sampling_rate.get(),
                             Q_phase_delay=self.mixer_IQ_phase_skewness.get())
        # advanced RO pulses
        # with ramp-up
        M_up = PG.block_pulse(self.get('M_up_amp'), self.M_up_length.get(),  # ns
                              sampling_rate=self.get('sampling_rate'),
                              delay=0,
                              phase=self.get('M_up_phi'))

        M_up_mid = (np.concatenate((M_up[0], M[0])),
                    np.concatenate((M_up[1], M[1])))

        Mod_M_up_mid = PG.mod_pulse(M_up_mid[0], M_up_mid[1],
                                    f_modulation=self.get('M_modulation'),
                                    sampling_rate=self.get('sampling_rate'),
                                    Q_phase_delay=self.get('mixer_IQ_phase_skewness'))

        # with ramp-up and double frequency depletion
        M_down0 = PG.block_pulse(self.get('M_down_amp0'), self.get('M_down_length'),  # ns
                                 sampling_rate=self.get('sampling_rate'),
                                 delay=0,
                                 phase=self.get('M_down_phi0'))

        M_down1 = PG.block_pulse(self.get('M_down_amp1'), self.get('M_down_length'),  # ns
                                 sampling_rate=self.get('sampling_rate'),
                                 delay=0,
                                 phase=self.get('M_down_phi1'))
        Mod_M_down0 = PG.mod_pulse(M_down0[0],
                                   M_down1[1],
                                   f_modulation=self.get('M0_modulation'),
                                   sampling_rate=self.get('sampling_rate'),
                                   Q_phase_delay=self.get('mixer_IQ_phase_skewness'))
        Mod_M_down1 = PG.mod_pulse(M_down1[0],
                                   M_down1[1],
                                   f_modulation=self.get('M1_modulation'),
                                   sampling_rate=self.get('sampling_rate'),
                                   Q_phase_delay=self.get('mixer_IQ_phase_skewness'))

        # summing the depletion components
        Mod_M_down = (np.add(Mod_M_down0[0],
                             Mod_M_down1[0]),
                      np.add(Mod_M_down0[1],
                             Mod_M_down1[1]))

        # concatenating up, mid and depletion
        Mod_M_up_mid_down = (np.concatenate((Mod_M_up_mid[0], Mod_M_down[0])),
                             np.concatenate((Mod_M_up_mid[1], Mod_M_down[1])))

        self._wave_dict = {'I': Wave_I,
                           'X180': Wave_X_180, 'Y180': Wave_Y_180,
                           'X90': Wave_X_90, 'Y90': Wave_Y_90,
                           'mX90': Wave_mX90, 'mY90': Wave_mY90,
                           'Block': Block,
                           'ModBlock': ModBlock,
                           'M_square': Mod_M,
                           'M_up_mid': Mod_M_up_mid,
                           'M_up_mid_double_dep': Mod_M_up_mid_down
                           }

        if self.mixer_apply_predistortion_matrix():
            M = self.get_mixer_predistortion_matrix()
            for key, val in self._wave_dict.items():
                self._wave_dict[key] = np.dot(M, val)

        return self._wave_dict

    def render_wave(self, wave_name, show=True, time_units='lut_index',
                    reload_pulses=True):
        if reload_pulses:
            self.generate_standard_pulses()
        fig, ax = plt.subplots(1, 1)
        if time_units == 'lut_index':
            x = np.arange(len(self._wave_dict[wave_name][0]))
            ax.set_xlabel('Lookuptable index (i)')
            ax.vlines(
                2048, self._voltage_min, self._voltage_max, linestyle='--')
        elif time_units == 's':
            x = (np.arange(len(self._wave_dict[wave_name][0]))
                 / self.sampling_rate.get())
            ax.set_xlabel('time (s)')
            ax.vlines(2048 / self.sampling_rate.get(),
                      self._voltage_min, self._voltage_max, linestyle='--')
        print(wave_name)
        ax.set_title(wave_name)
        ax.plot(x, self._wave_dict[wave_name][0],
                marker='o', label='chI')
        ax.plot(x, self._wave_dict[wave_name][1],
                marker='o', label='chQ')
        ax.set_ylabel('Amplitude (V)')
        ax.set_axis_bgcolor('gray')
        ax.axhspan(self._voltage_min, self._voltage_max, facecolor='w',
                   linewidth=0)
        ax.legend()
        ax.set_ylim(self._voltage_min*1.1, self._voltage_max*1.1)
        ax.set_xlim(0, x[-1])
        if show:
            plt.show()
        return fig, ax

    def render_wave_PSD(self, wave_name, show=True, reload_pulses=True, f_bounds=None, y_bounds=None):
        if reload_pulses:
            self.generate_standard_pulses()
        fig, ax = plt.subplots(1, 1)
        f_axis, PSD_I = func.PSD(
            self._wave_dict[wave_name][0], 1/self.sampling_rate())
        f_axis, PSD_Q = func.PSD(
            self._wave_dict[wave_name][1], 1/self.sampling_rate())

        ax.set_xlabel('frequency (Hz)')
        ax.set_title(wave_name)
        ax.plot(f_axis, PSD_I,
                marker='o', label='chI')
        ax.plot(f_axis, PSD_Q,
                marker='o', label='chQ')
        ax.set_ylabel('Spectral density (V^2/Hz)')
        ax.legend()

        ax.set_yscale("log", nonposy='clip')
        if y_bounds != None:
            ax.set_ylim(y_bounds[0], y_bounds[1])
        if f_bounds != None:
            ax.set_xlim(f_bounds[0], f_bounds[1])
        if show:
            plt.show()
        return fig, ax

    def get_mixer_predistortion_matrix(self):
        '''
        predistortion matrix correcting for a mixer with amplitude
        mismatch "mixer_alpha" and skewness "phi"

        M = [ 1            tan(phi) ]
            [ 0   1/mixer_alpha * sec(phi)]

        Notes on the procedure for acquiring this matrix can be found in
        PycQED/docs/notes/MixerSkewnessCalibration_LDC_150629.pdf

        Note: The same effect as the predistortion matrix can also be achieved
        by setting the IQ-phase skewness and QI-amp-ratio paramters.
        '''

        mixer_pre_distortion_matrix = np.array(
            ((1,  np.tan(self.get('mixer_phi')*2*np.pi/360)),
             (0, 1/self.get('mixer_alpha') * 1/np.cos(self.get('mixer_phi')*2*np.pi/360))))
        return mixer_pre_distortion_matrix

    # TODO: fix this function
    # def load_pulses_onto_AWG_lookuptable(self, regenerate_pulses=True):
    #     pulse_name = self.
    #     self.load_pulse_onto_AWG_lookuptable(pulse_name, regenerate_pulses)

    def load_pulse_onto_AWG_lookuptable(self, pulse_name,
                                        regenerate_pulses=True):
        '''
        Load a pulses to the lookuptable, it uses the lut_mapping to
        determine which lookuptable to load to.
        '''
        if regenerate_pulses:
            wave_dict = self.generate_standard_pulses()
        else:
            wave_dict = self._wave_dict

        I_wave = np.clip(wave_dict[pulse_name][0],
                         self._voltage_min, self._voltage_max)
        Q_wave = np.clip(np.multiply(self.get('mixer_QI_amp_ratio'),
                                     wave_dict[pulse_name][1]), self._voltage_min,
                         self._voltage_max)
        self.UHFQC.awg_sequence_acquisition_and_pulse(I_wave, Q_wave,
                                                      self.acquisition_delay())

    def give_back_wave_forms(self, pulse_name, regenerate_pulses=True):
        '''
        Load a pulses to the lookuptable, it uses the lut_mapping to
        determine which lookuptable to load to.
        '''
        if regenerate_pulses:
            wave_dict = self.generate_standard_pulses()
        else:
            wave_dict = self._wave_dict
        I_ch = 0
        Q_ch = 1
        I_wave = np.clip(wave_dict[pulse_name][0],
                         self._voltage_min, self._voltage_max)
        Q_wave = np.clip(np.multiply(self.get('mixer_QI_amp_ratio'),
                                     wave_dict[pulse_name][1]), self._voltage_min,
                         self._voltage_max)
        return I_wave, Q_wave
