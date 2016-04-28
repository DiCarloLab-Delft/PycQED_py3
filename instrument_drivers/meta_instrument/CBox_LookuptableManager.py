
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
import logging
from modules.measurement import CBox_Pulse_Generator as PG
import unittest
import matplotlib.pyplot as plt
import imp
imp.reload(PG)

global lm  # Global used for passing value to the testsuite


class QuTech_ControlBox_LookuptableManager(Instrument):
    '''
    meta-instrument that handles loading pulses into the CBox lookuptables
    and holds their parameters so that they can be sweeped and are logged.

    For now this is a test version that only stores the parameters for a
    specific set of pulses.

    todo:
        Add RO-tones to lut (or maybe to a child class?)
        Convert all units to SI (s and Hz instead of ns and GHz)
    Note: I did not port over the depletion pulses (MAR 7-1-2016)
    '''
    shared_kwargs = ['CBox']

    def __init__(self, name, CBox, **kw):

        logging.info(__name__ + ' : Initializing instrument')
        super().__init__(name, **kw)

        self.CBox = CBox

        self.add_parameter('amp180', units='mV',
                           vals=vals.Numbers(-1000, 1000),
                           get_cmd=self._do_get_amp180,
                           set_cmd=self._do_set_amp180)
        self.add_parameter('amp90', units='mV',
                           vals=vals.Numbers(-1000, 1000),
                           get_cmd=self._do_get_amp90,
                           set_cmd=self._do_set_amp90)
        self.add_parameter('ampCW', units='mV',
                           vals=vals.Numbers(-1000, 1000),
                           get_cmd=self._do_get_ampCW,
                           set_cmd=self._do_set_ampCW)
        self.add_parameter('block_length', units='ns',
                           vals=vals.Numbers(1, 500),
                           get_cmd=self._do_get_block_length,
                           set_cmd=self._do_set_block_length)

        self.add_parameter('motzoi_parameter', vals=vals.Numbers(-2, 2),
                           get_cmd=self._do_get_motzoi_parameter,
                           set_cmd=self._do_set_motzoi_parameter)
        self.add_parameter('QI_amp_ratio', vals=vals.Numbers(),
                           get_cmd=self._do_get_QI_amp_ratio,
                           set_cmd=self._do_set_QI_amp_ratio)
        self.add_parameter('IQ_phase_skewness', vals=vals.Numbers(),
                           units='deg',
                           get_cmd=self._do_get_IQ_phase_skewness,
                           set_cmd=self._do_set_IQ_phase_skewness)

        self.add_parameter('gauss_width', vals=vals.Numbers(), units='ns',
                           get_cmd=self._do_get_gauss_width,
                           set_cmd=self._do_set_gauss_width)
        self.add_parameter('f_modulation', vals=vals.Numbers(), units='GHz',
                           get_cmd=self._do_get_f_modulation,
                           set_cmd=self._do_set_f_modulation)
        self.add_parameter('sampling_rate', vals=vals.Numbers(), units='GHz',
                           set_cmd=self._do_set_sampling_rate,
                           get_cmd=self._do_get_sampling_rate)
        self.add_parameter('lut_mapping',
                           size=8,
                           get_cmd=self._do_get_lut_mapping,
                           set_cmd=self._do_set_lut_mapping,
                           vals=vals.Anything())


        # These parameters are added for mixer skewness correction.
        # They are intended to be renamed such that they can be combined with
        # QI_amp_ratio and IQ_phase_skewness.
        self.add_parameter('alpha', vals=vals.Numbers(),
                           get_cmd=self._do_get_alpha,
                           set_cmd=self._do_set_alpha)
        self.add_parameter('phi', vals=vals.Numbers(), units='deg',
                           get_cmd=self._do_get_phi,
                           set_cmd=self._do_set_phi)
        self.add_parameter('apply_predistortion_matrix', type=bool,
                           set_cmd=self._do_set_apply_predistortion_matrix,
                           get_cmd=self._do_get_apply_predistortion_matrix)

        self.set('lut_mapping', ['I', 'X180', 'Y180', 'X90', 'Y90', 'mX90',
                                 'mY90', 'ModBlock'])
        # Set to a default because box is not expected to change
        self.set('sampling_rate', 0.2)
        self.set('QI_amp_ratio', 1)
        self.set('alpha', 1)
        self.set('phi', 0)
        self.set('IQ_phase_skewness', 0)
        self.set('apply_predistortion_matrix', False)
        self.set('amp180', 30)
        self.set('amp90', 15)
        self.set('ampCW', 10)
        self.set('block_length', 50)
        self.set('gauss_width', 10)
        self.set('f_modulation', -0.02)
        self.set('motzoi_parameter', 0)
        self._voltage_min = -1000
        self._voltage_max = 1000-1000./2**13

    def run_test_suite(self):
            # pass the CBox to the module so it can be used in the tests
            from importlib import reload
            from .tests import test_suite; reload(test_suite)
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
        Wave_X_180 = PG.mod_gauss(self.get('amp180'), self.get('gauss_width'),
                                  self.get('f_modulation'), axis='x',
                                  motzoi=self.get('motzoi_parameter'),
                                  sampling_rate=self.get('sampling_rate'),
                                  Q_phase_delay=self.get('IQ_phase_skewness'))
        Wave_X_90 = PG.mod_gauss(self.get('amp90'), self.get('gauss_width'),
                                 self.get('f_modulation'), axis='x',
                                 motzoi=self.get('motzoi_parameter'),
                                 sampling_rate=self.get('sampling_rate'),
                                 Q_phase_delay=self.get('IQ_phase_skewness'))

        Wave_Y_180 = PG.mod_gauss(self.get('amp180'), self.get('gauss_width'),
                                  self.get('f_modulation'), axis='y',
                                  motzoi=self.get('motzoi_parameter'),
                                  sampling_rate=self.get('sampling_rate'),
                                  Q_phase_delay=self.get('IQ_phase_skewness'))
        Wave_Y_90 = PG.mod_gauss(self.get('amp90'), self.get('gauss_width'),
                                 self.get('f_modulation'), axis='y',
                                 motzoi=self.get('motzoi_parameter'),
                                 sampling_rate=self.get('sampling_rate'),
                                 Q_phase_delay=self.get('IQ_phase_skewness'))

        Wave_mX90 = PG.mod_gauss(-self.get('amp90'), self.get('gauss_width'),
                                 self.get('f_modulation'), axis='x',
                                 motzoi=self.get('motzoi_parameter'),
                                 sampling_rate=self.get('sampling_rate'),
                                 Q_phase_delay=self.get('IQ_phase_skewness'))

        Wave_mY90 = PG.mod_gauss(-self.get('amp90'), self.get('gauss_width'),
                                 self.get('f_modulation'), axis='y',
                                 motzoi=self.get('motzoi_parameter'),
                                 sampling_rate=self.get('sampling_rate'),
                                 Q_phase_delay=self.get('IQ_phase_skewness'))

        Block = PG.block_pulse(self.get('ampCW'), self.block_length.get(),  #ns
                               sampling_rate=self.get('sampling_rate'),
                               delay=0,
                               phase=0)
        ModBlock = PG.mod_pulse(Block[0], Block[1],
                                f_modulation=self.f_modulation.get(),
                                sampling_rate=self.sampling_rate.get(),
                                Q_phase_delay=self.IQ_phase_skewness.get())

        self._wave_dict = {'I': Wave_I,
                           'X180': Wave_X_180, 'Y180': Wave_Y_180,
                           'X90': Wave_X_90, 'Y90': Wave_Y_90,
                           'mX90': Wave_mX90, 'mY90': Wave_mY90,
                           'Block': Block,
                           'ModBlock': ModBlock}
        if self.apply_predistortion_matrix:
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
            ax.vlines(128, self._voltage_min, self._voltage_max, linestyle='--')
        elif time_units == 'ns':
            x = (np.arange(len(self._wave_dict[wave_name][0]))
                 / self.sampling_rate.get())
            ax.set_xlabel('time (ns)')
            ax.vlines(128 / self.sampling_rate.get(),
                      self._voltage_min, self._voltage_max, linestyle='--')

        ax.plot(x, self._wave_dict[wave_name][0],
                marker='o', label=wave_name+' chI')
        ax.plot(x, self._wave_dict[wave_name][1],
                marker='o', label=wave_name+' chQ')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_axis_bgcolor('gray')
        ax.axhspan(self._voltage_min, self._voltage_max, facecolor='w',
                   linewidth=0)
        ax.legend()
        ax.set_ylim(self._voltage_min*1.1, self._voltage_max*1.1)
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

        Note: The same effect as the predistortion matrix can also be achieved
        by setting the IQ-phase skewness and QI-amp-ratio paramters.
        '''

        mixer_pre_distortion_matrix = np.array(
            ((1,  np.tan(self.get('phi')*2*np.pi/360)),
             (0, 1/self.get('alpha') * 1/np.cos(self.get('phi')*2*np.pi/360))))
        return mixer_pre_distortion_matrix

    def load_pulses_onto_AWG_lookuptable(self, awg_nr):
        '''
        Loads the pulses to the lookuptables, it uses the lut_mapping to
        determine what pulse to load to which lookuptable.
        '''
        self.generate_standard_pulses()
        for i, pulse_name in enumerate(self.get('lut_mapping')):
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
            wave_dict = self._wave_dict
        # This is to account for the odd definition in CBox (see CBox Issue #13)
        I_ch = 1
        Q_ch = 0
        I_wave = np.clip(wave_dict[pulse_name][0],
                         self._voltage_min, self._voltage_max)
        Q_wave = np.clip(np.multiply(self._QI_amp_ratio,
                         wave_dict[pulse_name][1]), self._voltage_min,
                         self._voltage_max)
        # To account for multiple occurences in lut mapping
        indices = [i for i, x in enumerate(self.get('lut_mapping')) if
                   x == pulse_name]
        for i in indices:
            self.CBox.set_awg_lookuptable(awg_nr, i, I_ch, I_wave)
            self.CBox.set_awg_lookuptable(awg_nr, i, Q_ch, Q_wave)

    #######################################
    # get/set functions for 'simple' pars #
    #######################################
    def _do_set_amp180(self, val):
        self._amp180 = val

    def _do_get_amp180(self):
        return self._amp180

    def _do_set_lut_mapping(self, val):
        self._lut_mapping = val

    def _do_get_lut_mapping(self):
        return self._lut_mapping

    def _do_set_amp90(self, val):
        self._amp90 = val

    def _do_get_amp90(self):
        return self._amp90

    def _do_set_ampCW(self, val):
        self._ampCW = val

    def _do_get_ampCW(self):
        return self._ampCW

    def _do_set_block_length(self, val):
        self._block_length = val

    def _do_get_block_length(self):
        return self._block_length

    def _do_set_motzoi_parameter(self, val):
        self._motzoi_parameter = val

    def _do_get_motzoi_parameter(self):
        return self._motzoi_parameter

    def _do_set_gauss_width(self, val):
        self._gauss_width = val

    def _do_get_gauss_width(self):
        return self._gauss_width

    def _do_set_f_modulation(self, val):
        self._f_modulation = val

    def _do_get_f_modulation(self):
        return self._f_modulation

    def _do_set_f_modulation_ground(self, val):
        self._f_modulation_ground = val

    def _do_get_f_modulation_ground(self):
        return self._f_modulation_ground

    def _do_set_f_modulation_excited(self, val):
        self._f_modulation_excited = val

    def _do_get_f_modulation_excited(self):
        return self._f_modulation_excited

    def _do_set_QI_amp_ratio(self, val):
        self._QI_amp_ratio = val

    def _do_get_QI_amp_ratio(self):
        return self._QI_amp_ratio

    def _do_set_IQ_phase_skewness(self, val):
        self._IQ_phase_skewness = val

    def _do_get_IQ_phase_skewness(self):
        return self._IQ_phase_skewness

    def _do_set_sampling_rate(self, val):
        self._sampling_rate = val

    def _do_get_sampling_rate(self):
        return self._sampling_rate

    def _do_set_alpha(self, val):
        self._alpha = val

    def _do_get_alpha(self):
        return self._alpha

    def _do_set_phi(self, val):
        self._phi = val

    def _do_get_phi(self):
        return self._phi

    def _do_set_apply_predistortion_matrix(self, val):
        self._apply_predistortion_matrix = val

    def _do_get_apply_predistortion_matrix(self):
        return self._apply_predistortion_matrix


