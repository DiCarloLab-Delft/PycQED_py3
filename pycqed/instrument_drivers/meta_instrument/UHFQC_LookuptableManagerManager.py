
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


class UHFQC_LookuptableManagerManager(Instrument):
    '''
    meta-instrument that can produce multiplexed pulses by adding pulses
    from different lookupotable managers.

    For now this is a test version that only stores the parameters for a
    specific set of pulses.
    '''
    shared_kwargs = ['UHFQC']

    def __init__(self, name, UHFQC, **kw):

        logging.info(__name__ + ' : Initializing instrument')
        super().__init__(name, **kw)

        self.UHFQC = UHFQC
        self.add_parameter('mixer_QI_amp_ratio', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('mixer_IQ_phase_skewness', vals=vals.Numbers(),
                           unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
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

        self.add_parameter('acquisition_delay', vals=vals.Numbers(), unit='s',
                           parameter_class=ManualParameter,
                           initial_value=270e-9)
        self.add_parameter('LutMans', vals=vals.Anything(),
                   set_cmd=self._attach_lutmans_to_Lutmanman)
        self.add_parameter('sampling_rate', vals=vals.Numbers(), unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=1.8e9)

        # Set to a default because box is not expected to change
        self._voltage_min = -1.0
        self._voltage_max = 1.0-1.0/2**13

    def _attach_lutmans_to_Lutmanman(self, LutMans):
        for LutMan in LutMans:
            LutManthis=self.find_instrument(LutMan)
            setattr(self, LutMan, LutManthis) #equivalent to: self.LutMan= LutManthis


    def generate_multiplexed_pulse(self, multiplexed_wave):
        '''
        Generates a basic set of pulses (I, X-180, Y-180, x-90, y-90, Block,
                                         X180_delayed)
        using the parameters set on this meta-instrument and returns the
        corresponding waveforms for both I and Q channels as a dict.

        Note the primitive set is a different set than the one used in
        Serwan's thesis.
        '''
        # Standard qubit pulses

        ###### RO pulses
        Wave_multi_I=np.array(np.zeros(2))
        Wave_multi_Q=np.array(np.zeros(2))
        for LutMan, pulse in multiplexed_wave:
            print("loading {} from {}".format(pulse, LutMan))
            LutManthis=self.find_instrument(LutMan)
            Wave_element_I, Wave_element_Q  = LutManthis.give_back_wave_forms(pulse_name=pulse)
            # Add to Wave_multi_I resize to check that the lengths are OK
            if len(Wave_element_I)>len(Wave_multi_I):
               Wave_multi_I.resize(Wave_element_I.shape)
               Wave_multi_I = Wave_multi_I+Wave_element_I
               Wave_multi_Q.resize(Wave_element_Q.shape)
               Wave_multi_Q = Wave_multi_Q+Wave_element_Q
            else:
               Wave_element_I.resize(Wave_multi_I.shape)
               Wave_multi_I = Wave_multi_I+Wave_element_I
               Wave_element_Q.resize(Wave_multi_Q.shape)
               Wave_multi_Q = Wave_multi_Q+Wave_element_Q

        Wave_multi = [Wave_multi_I, Wave_multi_Q]
        self._wave_dict = {'Multiplexed_pulse': Wave_multi}

        if self.mixer_apply_predistortion_matrix():
            M = self.get_mixer_predistortion_matrix()
            for key, val in self._wave_dict.items():
                self._wave_dict[key] = np.dot(M, val)

        return self._wave_dict

    def render_wave(self, wave_name, show=True, time_unit='lut_index'):
        fig, ax = plt.subplots(1, 1)
        if time_unit == 'lut_index':
            x = np.arange(len(self._wave_dict[wave_name][0]))
            ax.set_xlabel('Lookuptable index (i)')
            ax.vlines(2048, self._voltage_min, self._voltage_max, linestyle='--')
        elif time_unit == 's':
            x = (np.arange(len(self._wave_dict[wave_name][0]))
                 / self.sampling_rate.get())
            ax.set_xlabel('Time (s)')
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

    def render_wave_PSD(self, wave_name, show=True, f_bounds=None, y_bounds=None):
        fig, ax = plt.subplots(1, 1)
        f_axis, PSD_I = func.PSD(self._wave_dict[wave_name][0], 1/self.sampling_rate())
        f_axis, PSD_Q = func.PSD(self._wave_dict[wave_name][1], 1/self.sampling_rate())

        ax.set_xlabel('frequency (Hz)')
        ax.set_title(wave_name)
        ax.plot(f_axis, PSD_I,
                marker='o', label='chI')
        ax.plot(f_axis, PSD_Q,
                marker='o', label='chQ')
        ax.set_ylabel('Spectral density (V^2/Hz)')
        ax.legend()

        ax.set_yscale("log", nonposy='clip')
        if y_bounds!=None:
          ax.set_ylim(y_bounds[0],y_bounds[1])
        if f_bounds!=None:
          ax.set_xlim(f_bounds[0],f_bounds[1])
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

    def load_pulse_onto_AWG_lookuptable(self, pulse_name):
        '''
        Load a pulses to the lookuptable, it uses the lut_mapping to
        determine which lookuptable to load to.
        '''

        wave_dict = self._wave_dict
        I_ch = 0
        Q_ch = 1
        I_wave = np.clip(wave_dict[pulse_name][0],
                         self._voltage_min, self._voltage_max)
        Q_wave = np.clip(np.multiply(self.get('mixer_QI_amp_ratio'),
                         wave_dict[pulse_name][1]), self._voltage_min,
                         self._voltage_max)
        self.UHFQC.awg_sequence_acquisition_and_pulse(I_wave, Q_wave, self.acquisition_delay())
        print('wave {} should be loaded in UHFQC'.format(pulse_name))

    def give_back_wave_forms(self, pulse_name):
        '''
        Load a pulses to the lookuptable, it uses the lut_mapping to
        determine which lookuptable to load to.
        '''
        wave_dict = self._wave_dict
        I_ch = 0
        Q_ch = 1
        I_wave = np.clip(wave_dict[pulse_name][0],
                         self._voltage_min, self._voltage_max)
        Q_wave = np.clip(np.multiply(self.get('mixer_QI_amp_ratio'),
                         wave_dict[pulse_name][1]), self._voltage_min,
                         self._voltage_max)
        return I_wave, Q_wave



