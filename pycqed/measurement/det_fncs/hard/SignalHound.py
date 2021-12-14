"""
SignalHound related detector functions
extracted from pycqed/measurement/detector_functions.py commit 0da380ad2adf2dc998f5effef362cdf264b87948
"""

import logging
import time
from packaging import version

import qcodes as qc

from pycqed.measurement.det_fncs.Base import Soft_Detector, Hard_Detector

from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control import sequence

# import instruments for type annotations
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController import UHFQC


log = logging.getLogger(__name__)


class Signal_Hound_fixed_frequency(Soft_Detector):

    def __init__(
            self,
            signal_hound,
            frequency=None,
            Navg=1,
            delay=0.1,
            prepare_for_each_point=False,
            prepare_function=None,
            prepare_function_kwargs: dict = {}
    ):
        super().__init__()
        self.frequency = frequency
        self.name = 'SignalHound_fixed_frequency'
        self.value_names = ['Power']
        self.value_units = ['dBm']
        self.delay = delay
        self.SH = signal_hound
        if frequency is not None:
            self.SH.frequency(frequency)
        self.Navg = Navg
        self.prepare_for_each_point = prepare_for_each_point
        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs

    def acquire_data_point(self, **kw):
        if self.prepare_for_each_point:
            self.prepare()
        time.sleep(self.delay)
        if version.parse(qc.__version__) < version.parse('0.1.11'):
            return self.SH.get_power_at_freq(Navg=self.Navg)
        else:
            self.SH.avg(self.Navg)
            return self.SH.power()

    def prepare(self, **kw):
        if qc.__version__ < '0.1.11':
            self.SH.prepare_for_measurement()
        if self.prepare_function is not None:
            self.prepare_function(**self.prepare_function_kwargs)

    def finish(self, **kw):
        self.SH.abort()


class Signal_Hound_sweeped_frequency(Hard_Detector):

    def __init__(
            self,
            signal_hound,
            Navg=1,
            delay=0.1,
            **kw
    ):
        super().__init__()
        self.name = 'SignalHound_fixed_frequency'
        self.value_names = ['Power']
        self.value_units = ['dBm']
        self.delay = delay
        self.SH = signal_hound
        self.Navg = Navg

    def acquire_data_point(self, **kw):
        frequency = self.swp.pop()
        self.SH.set('frequency', frequency)
        self.SH.prepare_for_measurement()
        time.sleep(self.delay)
        return self.SH.get_power_at_freq(Navg=self.Navg)

    def get_values(self):
        return ([self.acquire_data_point()])

    def prepare(self, sweep_points):
        self.swp = list(sweep_points)
        # self.SH.prepare_for_measurement()

    def finish(self, **kw):
        self.SH.abort()


class SH_mixer_skewness_det(Soft_Detector):
    '''
    Based on the "Signal_Hound_fixed_frequency" detector.
    generates an AWG seq to measure sideband transmission

    Inputs:
        frequency       (Hz)
        QI_amp_ratio    (parameter)
        IQ_phase        (parameter)
        SH              (instrument)
        f_mod           (Hz)

    '''

    def __init__(
            self,
            frequency,
            QI_amp_ratio,
            IQ_phase,
            SH,
            I_ch, Q_ch,
            station,
            Navg=1,
            delay=0.1,
            f_mod=10e6,
            verbose=False,
            **kw):
        super(SH_mixer_skewness_det, self).__init__()
        self.SH = SH
        self.frequency = frequency
        self.name = 'SignalHound_mixer_skewness_det'
        self.value_names = ['Power']
        self.value_units = ['dBm']
        self.delay = delay
        self.SH.frequency.set(frequency)  # Accepts input in Hz
        self.Navg = Navg
        self.QI_amp_ratio = QI_amp_ratio
        self.IQ_phase = IQ_phase
        self.pulsar = station.pulsar
        self.f_mod = f_mod
        self.I_ch = I_ch
        self.Q_ch = Q_ch
        self.verbose = verbose

    def acquire_data_point(self, **kw):
        QI_ratio = self.QI_amp_ratio.get()
        skewness = self.IQ_phase.get()
        if self.verbose:
            print('QI ratio: %.3f' % QI_ratio)
            print('skewness: %.3f' % skewness)
        self.generate_awg_seq(QI_ratio, skewness, self.f_mod)
        self.pulsar.AWG.start()
        time.sleep(self.delay)
        return self.SH.get_power_at_freq(Navg=self.Navg)

    def generate_awg_seq(self, QI_ratio, skewness, f_mod):
        SSB_modulation_el = element.Element('SSB_modulation_el',
                                            pulsar=self.pulsar)
        cos_pulse = pulse.CosPulse(channel=self.I_ch, name='cos_pulse')
        sin_pulse = pulse.CosPulse(channel=self.Q_ch, name='sin_pulse')

        SSB_modulation_el.add(pulse.cp(cos_pulse, name='cos_pulse',
                                       frequency=f_mod, amplitude=0.15,
                                       length=1e-6, phase=0))
        SSB_modulation_el.add(pulse.cp(sin_pulse, name='sin_pulse',
                                       frequency=f_mod, amplitude=0.15 *
                                                                  QI_ratio,
                                       length=1e-6, phase=90 + skewness))

        seq = sequence.Sequence('Sideband_modulation_seq')
        seq.append(name='SSB_modulation_el', wfname='SSB_modulation_el',
                   trigger_wait=False)
        self.pulsar.program_awgs(seq, SSB_modulation_el)

    def prepare(self, **kw):
        self.SH.prepare_for_measurement()

    def finish(self, **kw):
        self.SH.abort()
