import numpy as np
from time import time
from modules.measurement.waveform_control import pulsar
from modules.measurement.waveform_control import pulse
from modules.measurement.waveform_control import element
from modules.measurement.waveform_control import sequence
from modules.measurement.waveform_control.viewer import show_element, show_wf
from modules.measurement.waveform_control import pulse_library as pl

reload(element)
reload(pl)
pi_amp = 1.2
times = np.around(np.linspace(5e-9, 20e-6, 60), decimals=9)
sigma = 20e-9
mod_frequency = 20e6
meas_pulse_delay = 200e-9
RO_pulse_duration = 2e-6
ATS_trig_delay = -200e-9


reload(pulsar)
station.pulsar = pulsar.Pulsar()
station.pulsar.AWG = station.instruments['AWG']
station.pulsar.define_channel(id='ch1', name='I', type='analog',
                              high=1, low=-1,
                              offset=0.0, delay=0, active=True)
station.pulsar.define_channel(id='ch2', name='Q', type='analog',
                              high=1, low=-1,
                              offset=0.0, delay=0, active=True)

station.pulsar.define_channel(id='ch1_marker1', name='ATS-marker', type='marker',
                              high=1.0, low=0, offset=0.,
                              delay=0, active=True)
station.pulsar.define_channel(id='ch1_marker2', name='RF-marker', type='marker',
                              high=1.0, low=0, offset=0.,
                              delay=0, active=True)


def generate_T1_element(tau, amp=1, sigma=5e-9, mod_frequency=10e6,
                        meas_pulse_delay=20e-9, RO_pulse_duration=2e-6):
    # make sure tau is a multiple of 1 ns, if it is not the fixed point will
    # not be able to be computed.

    T1_element = element.Element(name=('T1_element %s' % tau), pulsar=qt.pulsar)
    Drag_pulse = pl.SSB_DRAG_pulse(name='DRAG', I_channel='I', Q_channel='Q')

    T1_element.add(pulse.cp(Drag_pulse, mod_frequency=mod_frequency,
                            amplitude=amp, motzoi=.1, sigma=sigma),
                   name='initial_pi', start=50e-9)

    T1_element.add(pulse.SquarePulse(
                   channel='RF-marker',
                   amplitude=1,
                   length=RO_pulse_duration),
                   name='Readout tone',
                   start=(meas_pulse_delay+tau), refpulse='initial_pi',
                   fixed_point_freq=100e6)
    T1_element.add(pulse.SquarePulse(name='ATS-marker', channel='ATS-marker',
                   amplitude=1,
                   length=100e-9),
                   start=ATS_trig_delay, refpoint='start',
                   refpulse='Readout tone')

    return T1_element


def generate_and_upload_T1_sequence():
    seq = sequence.Sequence('T1 sequence')
    el_list = []

    for i, tau in enumerate(times):
        el = generate_T1_element(tau, pi_amp, sigma, mod_frequency,
                                 meas_pulse_delay)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    # show_element(el)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list)

if __name__ == '__main__':
    t0 = time()
    generate_and_upload_T1_sequence()
    print 'total time:',  time() - t0
