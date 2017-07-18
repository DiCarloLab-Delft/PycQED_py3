from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import logging
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
import numpy as np
from qcodes.plots.pyqtgraph import QtPlot


class QWG_LookuptableManager(Instrument):

    def __init__(self, name, QWG, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        super().__init__(name, **kw)
        self.add_parameter('QWG', parameter_class=InstrumentParameter)

        self.add_parameter('Q_amp180',
                           unit='V',
                           vals=vals.Numbers(-1, 1),
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

        self.add_parameter('spec_pulse_type',
                           vals=vals.Enum('block', 'gauss'),
                           parameter_class=ManualParameter,
                           initial_value='block')
        self.add_parameter('spec_amp',
                           unit='V',
                           vals=vals.Numbers(0, 1),
                           parameter_class=ManualParameter,
                           initial_value=0.4)
        self.add_parameter(
            'spec_length', vals=vals.Numbers(min_value=1e-9), unit='s',
            parameter_class=ManualParameter,
            docstring=('length of the block pulse if spec_pulse_type' +
                       'is "block", gauss_width if spec_pulse_type is gauss.'),
            initial_value=100e-9)

    def load_pulses_onto_AWG_lookuptable(self):
        self.QWG.get_instr().stop()

        # Microwave pulses
        G_amp = self.Q_amp180()/self.QWG.get_instr().get('ch{}_amp'.format(1))
        # Amplitude is set using the channel amplitude (at least for now)
        G, D = wf.gauss_pulse(G_amp, self.Q_gauss_width(),
                              motzoi=self.Q_motzoi(),
                              sampling_rate=1e9)  # sampling rate of QWG
        self.QWG.get_instr().deleteWaveformAll()
        self.QWG.get_instr().createWaveformReal('X180_q0_I', G)
        self.QWG.get_instr().createWaveformReal('X180_q0_Q', D)
        self.QWG.get_instr().createWaveformReal('X90_q0_I',
                                                self.Q_amp90_scale()*G)
        self.QWG.get_instr().createWaveformReal('X90_q0_Q',
                                                self.Q_amp90_scale()*D)

        self.QWG.get_instr().createWaveformReal('Y180_q0_I', D)
        self.QWG.get_instr().createWaveformReal('Y180_q0_Q', -G)
        self.QWG.get_instr().createWaveformReal('Y90_q0_I',
                                                self.Q_amp90_scale()*D)
        self.QWG.get_instr().createWaveformReal('Y90_q0_Q',
                                                -self.Q_amp90_scale()*G)

        self.QWG.get_instr().createWaveformReal('mX90_q0_I',
                                                -self.Q_amp90_scale()*G)
        self.QWG.get_instr().createWaveformReal('mX90_q0_Q',
                                                -self.Q_amp90_scale()*D)
        self.QWG.get_instr().createWaveformReal('mY90_q0_I',
                                                -self.Q_amp90_scale()*D)
        self.QWG.get_instr().createWaveformReal('mY90_q0_Q',
                                                self.Q_amp90_scale()*G)

        # Spec pulse
        if self.spec_pulse_type() == 'gauss':
            spec_G, spec_Q = wf.gauss_pulse(self.spec_amp(),
                                            self.spec_length(),
                                            motzoi=0, sampling_rate=1e9)
        elif self.spec_pulse_type() == 'block':
            spec_G, spec_Q = wf.block_pulse(self.spec_amp(),
                                            self.spec_length(),
                                            sampling_rate=1e9)
        self.QWG.get_instr().createWaveformReal('spec_q0_I', spec_G)
        self.QWG.get_instr().createWaveformReal('spec_q0_Q', spec_Q)

        # Filler waveform

        self.QWG.get_instr().createWaveformReal('zero', [0]*4)
        self.QWG.get_instr().codeword_0_ch1_waveform('X180_q0_I')
        self.QWG.get_instr().codeword_0_ch2_waveform('X180_q0_Q')
        self.QWG.get_instr().codeword_0_ch3_waveform('X180_q0_I')
        self.QWG.get_instr().codeword_0_ch4_waveform('X180_q0_Q')

        self.QWG.get_instr().codeword_1_ch1_waveform('Y180_q0_I')
        self.QWG.get_instr().codeword_1_ch2_waveform('Y180_q0_Q')
        self.QWG.get_instr().codeword_1_ch3_waveform('Y180_q0_I')
        self.QWG.get_instr().codeword_1_ch4_waveform('Y180_q0_Q')

        self.QWG.get_instr().codeword_2_ch1_waveform('X90_q0_I')
        self.QWG.get_instr().codeword_2_ch2_waveform('X90_q0_Q')
        self.QWG.get_instr().codeword_2_ch3_waveform('X90_q0_I')
        self.QWG.get_instr().codeword_2_ch4_waveform('X90_q0_Q')

        self.QWG.get_instr().codeword_3_ch1_waveform('Y90_q0_I')
        self.QWG.get_instr().codeword_3_ch2_waveform('Y90_q0_Q')
        self.QWG.get_instr().codeword_3_ch3_waveform('Y90_q0_I')
        self.QWG.get_instr().codeword_3_ch4_waveform('Y90_q0_Q')

        self.QWG.get_instr().codeword_4_ch1_waveform('mX90_q0_I')
        self.QWG.get_instr().codeword_4_ch2_waveform('mX90_q0_Q')
        self.QWG.get_instr().codeword_4_ch3_waveform('mX90_q0_I')
        self.QWG.get_instr().codeword_4_ch4_waveform('mX90_q0_Q')

        self.QWG.get_instr().codeword_5_ch1_waveform('mY90_q0_I')
        self.QWG.get_instr().codeword_5_ch2_waveform('mY90_q0_Q')
        self.QWG.get_instr().codeword_5_ch3_waveform('mY90_q0_I')
        self.QWG.get_instr().codeword_5_ch4_waveform('mY90_q0_Q')

        self.QWG.get_instr().codeword_6_ch1_waveform('spec_q0_I')
        self.QWG.get_instr().codeword_6_ch2_waveform('spec_q0_Q')
        self.QWG.get_instr().codeword_6_ch3_waveform('spec_q0_I')
        self.QWG.get_instr().codeword_6_ch4_waveform('spec_q0_Q')

        self.QWG.get_instr().start()
        self.QWG.get_instr().getOperationComplete()


class QWG_FluxLookuptableManager(Instrument):

    def __init__(self, name, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        super().__init__(name, **kw)
        self.add_parameter('QWG', parameter_class=InstrumentParameter)
        self.add_parameter('F_kernel_instr',
                           parameter_class=InstrumentParameter)

        self.add_parameter('F_amp', unit='frac',
                           docstring=('Amplitude of flux pulse as fraction of '
                                      'the peak amplitude. Beware of factor 2'
                                      ' with Vpp in the QWG'),
                           initial_value=0.5,
                           parameter_class=ManualParameter)
        self.add_parameter('F_length', unit='s',
                           parameter_class=ManualParameter,
                           initial_value=1e-6)
        self.add_parameter('F_ch', label='Flux channel',
                           vals=vals.Ints(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_delay', label='Flux pulse delay',
                           unit='s',
                           initial_value=0,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_compensation_delay',
                           label='compens. pulse delay',
                           unit='s',
                           initial_value=4e-6,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_lambda_1',
                           docstring='first lambda coef. for martinis pulse',
                           label='Lambda_1',
                           unit='',
                           initial_value=0,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_lambda_2',
                           docstring='second lambda coef. for martinis pulse',
                           label='Lambda_2',
                           unit='',
                           initial_value=0,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_lambda_3',
                           docstring='third lambda coef. for martinis pulse',
                           label='Lambda_3',
                           unit='',
                           initial_value=0,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_theta_f',
                           docstring='theta_f for martinis pulse',
                           label='theta_f',
                           unit='deg',
                           initial_value=90,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_J2',
                           docstring='coupling between 11-02',
                           label='J2',
                           unit='Hz',
                           initial_value=10e6,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_f_interaction',
                           label='interaction frequency',
                           unit='Hz',
                           initial_value=5e9,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_dac_flux_coef',
                           docstring='conversion factor AWG voltage to flux',
                           label='dac flux coef',
                           unit='(V^-1)',
                           initial_value=1.0,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_E_c',
                           label='qubit E_c',
                           unit='Hz',
                           initial_value=250e6,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_f_01_max',
                           label='sweet spot freq',
                           unit='Hz',
                           initial_value=6e9,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_asymmetry',
                           label='qubit asymmetry',
                           unit='Hz',
                           initial_value=0.0,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('codeword_dict',
                           docstring='Dict assigning codewords to pulses',
                           label='codeword dict',
                           unit='',
                           initial_value={},
                           vals=vals.Anything(),
                           parameter_class=ManualParameter)
        self.add_parameter('sampling_rate',
                           docstring='Sampling rate of the QWG',
                           label='sampling rate',
                           unit='Hz',
                           initial_value=1.0e9,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('Z_length',
                           docstring=('Duration of single qubit Z pulse in'
                                      ' seconds'),
                           label='Z length',
                           unit='s',
                           initial_value=10e-9,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('Z_amp',
                           docstring=('Amplitude of flux pulse as fraction of '
                                      'the peak amplitude. Beware of factor '
                                      '2 with Vpp in the QWG'),
                           label='Z amplitude',
                           unit='frac',
                           initial_value=0.0,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)

        self.add_parameter('max_waveform_length', unit='s',
                           parameter_class=ManualParameter,
                           initial_value=30e-6)
        self.add_parameter('codeword_channels',
                           parameter_class=ManualParameter,
                           docstring='Channels used for triggering specific '
                                     'codewords.',
                           vals=vals.Lists(vals.Ints()))

        self.add_parameter('disable_CZ',
                           parameter_class=ManualParameter,
                           vals=vals.Bool(),
                           initial_value=False)
        self.add_parameter('pulse_map',
                           initial_value={'cz': 'adiabatic_Z',
                                          'square': 'square'},
                           parameter_class=ManualParameter,
                           # to be replaced with vals.Dict)
                           vals=vals.Anything())
        self.add_parameter('wave_dict_unit',
                           get_cmd=self._get_wave_dict_unit(),
                           set_cmd=self._set_wave_dict_unit(),
                           docstring='Unit in which the waveforms are '
                           'specified.\n'
                           '"frac" means "fraction of the maximum QWG '
                           'range".\n'
                           '"V" means volts.',
                           vals=vals.Enum('frac', 'V'))
        self._wave_dict_unit = 'frac'  # Initial value for wave_dict_unit
        self.add_parameter('V_offset',
                           unit='V',
                           label='V offset',
                           docstring='pulsed sweet spot offset',
                           parameter_class=ManualParameter,
                           initial_value=0,
                           vals=vals.Numbers())
        self.add_parameter('V_per_phi0',
                           unit='V',
                           label='V per phi_0',
                           docstring='pulsed voltage required for one phi0 '
                                     'of flux',
                           parameter_class=ManualParameter,
                           initial_value=1,
                           vals=vals.Numbers())

        self._wave_dict = {}

    def _get_wave_dict_unit(self):
        return self._wave_dict_unit

    def _set_wave_dict_unit(self, val):
        self.F_amp.unit = val
        self._wave_dict_unit = val

    def standard_waveforms(self):
        '''
        Returns standard waveforms, without delays or distortions applied.
        '''
        # Block pulses
        # Pulse is not IQ modulated, so block_Q is not used.
        block = wf.single_channel_block(self.F_amp(), self.F_length(),
                                        sampling_rate=self.sampling_rate())

        # Fast adiabatic pulse
        # Old version, commented out because we might still need it for
        # debugging (13 June 2017)
        # martinis_pulse = wf.martinis_flux_pulse(
        #     length=self.F_length(),
        #     lambda_coeffs=[self.F_lambda_1(), self.F_lambda_2(),
        #                    self.F_lambda_3()],
        #     theta_f=self.F_theta_f()*2*np.pi/360,
        #     f_01_max=self.F_f_01_max(),
        #     g2=self.F_J2(),
        #     E_c=self.F_E_c(),
        #     dac_flux_coefficient=self.F_dac_flux_coef(),
        #     f_interaction=self.F_f_interaction(),
        #     f_bus=None,
        #     asymmetry=self.F_asymmetry(),
        #     sampling_rate=self.sampling_rate(),
        #     return_unit='V')

        # New version of fast adiabatic pulse
        martinis_pulse_v2 = wf.martinis_flux_pulse_v2(
            length=self.F_length(),
            lambda_2=self.F_lambda_2(),
            lambda_3=self.F_lambda_3(),
            theta_f=self.F_theta_f(),
            f_01_max=self.F_f_01_max(),
            J2=self.F_J2(),
            E_c=self.F_E_c(),
            V_per_phi0=self.V_per_phi0(),
            V_offset=self.V_offset(),
            f_interaction=self.F_f_interaction(),
            f_bus=None,
            asymmetry=self.F_asymmetry(),
            sampling_rate=self.sampling_rate(),
            return_unit='V')

        # Flux pulse for single qubit phase correction
        z_nr_samples = int(np.round(self.Z_length() * self.sampling_rate()))
        single_qubit_phase_correction = np.ones(z_nr_samples) * self.Z_amp()

        if self.disable_CZ():
            martinis_pulse_v2 *= 0
        # Construct phase corrected pulses
        martinis_phase_corrected = np.concatenate(
            [martinis_pulse_v2, single_qubit_phase_correction])

        return {'square': block,
                'adiabatic': martinis_pulse_v2,
                'adiabatic_Z': martinis_phase_corrected}

    def generate_standard_pulses(self):
        '''
        Generates all flux pulses that are defined in the method
        'standard_waveforms'.

        The complete pulses with delays and distortions are stored in
        self._wave_dict, which is also returned by this method.
        '''
        self._wave_dict = {}

        waveforms = self.standard_waveforms()

        # Insert delays and compensation pulses, apply distortions
        wait_samples = np.zeros(int(np.round(self.F_delay() *
                                             self.sampling_rate())))
        wait_samples_2 = np.zeros(int(np.round(self.F_compensation_delay() *
                                               self.sampling_rate())))

        for key in waveforms.keys():
            delayed_wave = np.concatenate(
                [wait_samples, np.array(waveforms[key]), wait_samples_2,

                 -1 * np.array(waveforms[key])])
            if self.F_kernel_instr() is not None:
                k = self.F_kernel_instr.get_instr()
                self._wave_dict[key] = k.convolve_kernel(
                    [k.kernel(), delayed_wave],
                    length_samples=int(np.round(self.max_waveform_length() *
                                                self.sampling_rate())))
                # the waveform is cut off after length_samples.
                # this is particularly important considering hardware (memory)
                # limits of the QWG.
            else:
                logging.warning('No distortion kernel specified,'
                                ' not distorting flux pulse')
                self._wave_dict[key] = delayed_wave

        return self._wave_dict

    def generate_composite_flux_pulse(self, time_tuples: list,
                                      end_time_ns: float):
        """
        takes a list time_tuples (time_in_ns, pulse_name) and creates a
        composite pulse using the parameters stored in the flux lutman.

        This pulse does not include distortions and compensation pulses as
        these are added when the pulse is uploaded using
        "load_custom_pulse_onto_AWG_lookuptable".
        """

        # only intended for QWG which has a sampling rate of 1GSps
        if self.sampling_rate() != 1e9:
            raise ValueError('this function only works for 1GSps '
                             'sampling rate')
        end_sample = end_time_ns

        pulse_map = self.pulse_map()
        base_waveforms = self.standard_waveforms()

        composite_waveform = np.zeros(end_sample)

        for time_ns, operation in time_tuples:
            waveform = base_waveforms[pulse_map[operation]]
            composite_waveform[time_ns:time_ns+len(waveform)] += waveform

        return composite_waveform

    def regenerate_pulse(self, pulse_name):
        '''
        Regenerates a single pulse. The pulse is updated in self._wave_dict
        and also returned by this method.

        Args:
            pulse_name      (str): name of the pulse to regenerate
        '''
        if not self._wave_dict:
            # Pulses have never been generated since instatiation.
            self.generate_standard_pulses()

        if pulse_name not in self._wave_dict.keys():
            raise KeyError(
                'Pulse {} not in wave dictionary.'.format(pulse_name))

        # Get the plain waveform
        waveform = self.standard_waveforms()[pulse_name]

        # Insert delays and compensation pulses, apply distortions
        wait_samples = np.zeros(int(self.F_delay()*self.sampling_rate()))
        wait_samples_comp_pulses = np.zeros(int(self.F_compensation_delay()
                                                * self.sampling_rate()))
        k = self.F_kernel_instr.get_instr()

        delayed_wave = np.concatenate([wait_samples, np.array(waveform),
                                       wait_samples_comp_pulses,
                                       -1*np.array(waveform)])
        distorted_wave = k.convolve_kernel(
            [k.kernel(), delayed_wave],
            length_samples=int(np.round(self.max_waveform_length()
                               * self.sampling_rate())))

        self._wave_dict[pulse_name] = distorted_wave
        return distorted_wave

    def load_pulse_onto_AWG_lookuptable(self, pulse_name,
                                        regenerate_pulse=True):
        '''
        Load a specific pulse onto the lookuptable.

        Args:
            pulse_name          (str): Name of the pulse to be loaded
            regenerate_pulses   (bool): should the pulses be generated again
        '''
        if regenerate_pulse:
            self.regenerate_pulse(pulse_name)

        if pulse_name not in self.codeword_dict().keys():
            raise KeyError(
                'Pulse {} not in codeword mapping.'.format(pulse_name))

        if self.wave_dict_unit() == 'V':
            # Rescale wave by Vpp/2 of the QWG, such that the output wave
            # amplitude is actually as specified in the wave dictionary.
            V_out = self.QWG.get_instr().get('ch{}_amp'
                                             .format(self.F_ch())) / 2
            outWave = self._wave_dict[pulse_name] / V_out
            if np.any(np.abs(outWave) > 1):
                raise RuntimeError('Waveform values out of range [-1, 1]. '
                                   'Try increasing QWG.ch{}_amp.'
                                   .format(self.F_ch()))

        elif self.wave_dict_unit() == 'frac':
            # Do not rescale; wave dictionary is in units of
            # "fraction of Vpp/2"
            outWave = self._wave_dict[pulse_name]

        self.QWG.get_instr().createWaveformReal(
            pulse_name+self.name, outWave)
        self.QWG.get_instr().set('codeword_{}_ch{}_waveform'.format(
            self.codeword_dict()[pulse_name], self.F_ch()),
                                 pulse_name+self.name)

    def load_pulses_onto_AWG_lookuptable(self, regenerate_pulses=True):
        '''
        Load all standard pulses to the QWG lookuptable.

        Args:
            regenerate_pulses   (bool): should the pulses be generated again
        '''
        self.QWG.get_instr().stop()
        if regenerate_pulses:
            self.generate_standard_pulses()

        for pulse_name in self._wave_dict:
            self.load_pulse_onto_AWG_lookuptable(
                pulse_name, regenerate_pulse=False)
        self.QWG.get_instr().start()
        self.QWG.get_instr().getOperationComplete()

    def load_custom_pulse_onto_AWG_lookuptable(self, waveform,
                                               append_compensation: bool=True,
                                               distort: bool=True,
                                               pulse_name='custom',
                                               codeword=0):
        '''
        Load a user-defined waveform onto the QWG. The pulse is delayed by
        self.F_delay() and a compensation pulse is added after
        self.F_compensation_delay() if append_compensation is True.

        Args:
            waveform (array):
                    Samples of the pulse that will be loaded onto QWG.
            append_compensation (bool):
                    Should the compensation pulse be added.
            pulse_name (string):
                    The name of the pulse. This is also used in the QWG.
            codeword (int):
                    QWG codeword used for the pulse.
        '''
        self.QWG.get_instr().stop()

        # Insert delays and compensation pulses, apply distortions
        wait_samples = np.zeros(int(self.F_delay()*self.sampling_rate()))
        wait_samples_comp_pulses = np.zeros(int(self.F_compensation_delay()
                                                * self.sampling_rate()))

        delayed_wave = np.concatenate([wait_samples, np.array(waveform)])
        if append_compensation:
            delayed_wave = np.concatenate(
                [delayed_wave,
                 wait_samples_comp_pulses, -1 * np.array(waveform)])

        if self.F_kernel_instr() is not None:
            k = self.F_kernel_instr.get_instr()
            distorted_wave = k.convolve_kernel(
                [k.kernel(), delayed_wave],
                length_samples=int(self.max_waveform_length() *
                                   self.sampling_rate()))
        else:
            logging.warning('No distortion kernel specified, not distorting '
                            'flux pulse.')
            distorted_wave = delayed_wave

        # Scale to Vpp if necessary
        if self.wave_dict_unit() == 'V':
            # Rescale wave by Vpp/2 of the QWG, such that the output wave
            # amplitude is actually as specified in the wave dictionary.
            V_out = self.QWG.get_instr().get('ch{}_amp'
                                             .format(self.F_ch())) / 2
            outWave = distorted_wave / V_out
            if np.any(np.abs(outWave) > 1):
                raise RuntimeError('Waveform values out of range [-1, 1]. '
                                   'Try increasing QWG.ch{}_amp.'
                                   .format(self.F_ch()))
        elif self.wave_dict_unit() == 'frac':
            # Do not rescale; wave dictionary is in units of
            # "fraction of Vpp/2"
            outWave = distorted_wave

        # Upload pulse
        self.QWG.get_instr().createWaveformReal(pulse_name, outWave)
        self.QWG.get_instr().set('codeword_{}_ch{}_waveform'.format(
            codeword, self.F_ch()), pulse_name)

        self.QWG.get_instr().start()
        self.QWG.get_instr().getOperationComplete()

    def render_wave(self, wave_name, show=True, QtPlot_win=None):
        '''
        Plots the specified wave.

        Args:
        '''
        x = (np.arange(len(self._wave_dict[wave_name])))*1e-9
        y = self._wave_dict[wave_name]

        if QtPlot_win is None:
            QtPlot_win = QtPlot(window_title=wave_name,
                                figsize=(600, 400))
        QtPlot_win.add(
            x=x, y=y, name=wave_name,
            symbol='o', symbolSize=5,
            xlabel='Time', xunit='s', ylabel='Amplitude', yunit='V')

        return QtPlot_win
