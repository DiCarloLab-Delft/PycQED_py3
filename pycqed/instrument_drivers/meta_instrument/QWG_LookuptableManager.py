from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import logging
from pycqed.measurement.waveform_control_CC import waveform as wf
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
import numpy as np


class QWG_LookuptableManager(Instrument):

    def __init__(self, name, QWG, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        super().__init__(name, **kw)
        self.QWG = QWG

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
        self.QWG.stop()

        # Microwave pulses
        G_amp = self.Q_amp180()/self.QWG.get('ch{}_amp'.format(1))
        # Amplitude is set using the channel amplitude (at least for now)
        G, D = wf.gauss_pulse(G_amp, self.Q_gauss_width(),
                              motzoi=self.Q_motzoi(),
                              sampling_rate=1e9)  # sampling rate of QWG
        self.QWG.deleteWaveformAll()
        self.QWG.createWaveformReal('X180_q0_I', G)
        self.QWG.createWaveformReal('X180_q0_Q', D)
        self.QWG.createWaveformReal('X90_q0_I', self.Q_amp90_scale()*G)
        self.QWG.createWaveformReal('X90_q0_Q', self.Q_amp90_scale()*D)

        self.QWG.createWaveformReal('Y180_q0_I', D)
        self.QWG.createWaveformReal('Y180_q0_Q', -G)
        self.QWG.createWaveformReal('Y90_q0_I', self.Q_amp90_scale()*D)
        self.QWG.createWaveformReal('Y90_q0_Q', -self.Q_amp90_scale()*G)

        self.QWG.createWaveformReal('mX90_q0_I', -self.Q_amp90_scale()*G)
        self.QWG.createWaveformReal('mX90_q0_Q', -self.Q_amp90_scale()*D)
        self.QWG.createWaveformReal('mY90_q0_I', -self.Q_amp90_scale()*D)
        self.QWG.createWaveformReal('mY90_q0_Q', self.Q_amp90_scale()*G)

        # Spec pulse
        if self.spec_pulse_type() == 'gauss':
            spec_G, spec_Q = wf.gauss_pulse(self.spec_amp(),
                                            self.spec_length(),
                                            motzoi=0, sampling_rate=1e9)
        elif self.spec_pulse_type() == 'block':
            spec_G, spec_Q = wf.block_pulse(self.spec_amp(),
                                            self.spec_length(),
                                            sampling_rate=1e9)
        self.QWG.createWaveformReal('spec_q0_I', spec_G)
        self.QWG.createWaveformReal('spec_q0_Q', spec_Q)

        # Filler waveform
        self.QWG.createWaveformReal('zero', [0]*4)
        self.QWG.codeword_0_ch1_waveform('X180_q0_I')
        self.QWG.codeword_0_ch2_waveform('X180_q0_Q')
        self.QWG.codeword_0_ch3_waveform('X180_q0_I')
        self.QWG.codeword_0_ch4_waveform('X180_q0_Q')

        self.QWG.codeword_1_ch1_waveform('Y180_q0_I')
        self.QWG.codeword_1_ch2_waveform('Y180_q0_Q')
        self.QWG.codeword_1_ch3_waveform('Y180_q0_I')
        self.QWG.codeword_1_ch4_waveform('Y180_q0_Q')

        self.QWG.codeword_2_ch1_waveform('X90_q0_I')
        self.QWG.codeword_2_ch2_waveform('X90_q0_Q')
        self.QWG.codeword_2_ch3_waveform('X90_q0_I')
        self.QWG.codeword_2_ch4_waveform('X90_q0_Q')

        self.QWG.codeword_3_ch1_waveform('Y90_q0_I')
        self.QWG.codeword_3_ch2_waveform('Y90_q0_Q')
        self.QWG.codeword_3_ch3_waveform('Y90_q0_I')
        self.QWG.codeword_3_ch4_waveform('Y90_q0_Q')

        self.QWG.codeword_4_ch1_waveform('mX90_q0_I')
        self.QWG.codeword_4_ch2_waveform('mX90_q0_Q')
        self.QWG.codeword_4_ch3_waveform('mX90_q0_I')
        self.QWG.codeword_4_ch4_waveform('mX90_q0_Q')

        self.QWG.codeword_5_ch1_waveform('mY90_q0_I')
        self.QWG.codeword_5_ch2_waveform('mY90_q0_Q')
        self.QWG.codeword_5_ch3_waveform('mY90_q0_I')
        self.QWG.codeword_5_ch4_waveform('mY90_q0_Q')

        self.QWG.codeword_6_ch1_waveform('spec_q0_I')
        self.QWG.codeword_6_ch2_waveform('spec_q0_Q')
        self.QWG.codeword_6_ch3_waveform('spec_q0_I')
        self.QWG.codeword_6_ch4_waveform('spec_q0_Q')

        self.QWG.start()
        self.QWG.getOperationComplete()


class QWG_FluxLookuptableManager(Instrument):

    def __init__(self, name, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        super().__init__(name, **kw)
        self.add_parameter('QWG', parameter_class=InstrumentParameter)
        self.add_parameter('F_kernel_instr',
                           parameter_class=InstrumentParameter)

        self.add_parameter('F_amp', unit='V', parameter_class=ManualParameter)
        self.add_parameter('F_length', unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('F_CW', label='Flux pulse codeword',
                           vals=vals.Ints(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_ch', label='Flux channel',
                           vals=vals.Ints(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_delay', label='Flux pulse delay',
                           unit='s',
                           initial_value=0,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('F_compensation_delay',
                           label='Delay before compensation pulses',
                           unit='s',
                           initial_value=4e-6,
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)

    def load_pulses_onto_AWG_lookuptable(self):
        sampling_rate = 1e9
        block_I, block_Q = wf.block_pulse(self.F_amp(), self.F_length(),
                                          sampling_rate=1e9)
        wait_samples = np.zeros(int(self.F_delay()*sampling_rate))
        wait_samples_2 = np.zeros(int(self.F_compensation_delay()
                                      * sampling_rate))

        block_I = np.concatenate([wait_samples, np.array(block_I),
                                 wait_samples_2, -1*np.array(block_I)])
        block_Q = np.concatenate([wait_samples, np.array(block_Q),
                                 wait_samples_2, -1*np.array(block_Q)])
        k = self.F_kernel_instr.get_instr()
        distorted_I = k.convolve_kernel(
            [k.kernel(), block_I], length_samples=60e3)
        # hardcoded length for the distortions

        self.QWG.get_instr().createWaveformReal(
            'Square_flux_pulse', distorted_I)
        self.QWG.get_instr().set('codeword_{}_ch{}_waveform'.format(
            self.F_CW(), self.F_ch()), 'Square_flux_pulse')
