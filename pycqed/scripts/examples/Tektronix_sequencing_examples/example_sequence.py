import numpy as np
import time
from pycqed.measurement.waveform_control import pulsar
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import element
import pprint
import imp
import qcodes as qc
station = qc.station


imp.reload(pulse)
imp.reload(element)
imp.reload(pulsar)

# Making sure we have our global pulsar configured
# Warning! This does overwrite your pulsar settings (if such a device exists)
station.pulsar = pulsar.Pulsar()
station.pulsar.AWG = station.components['AWG']
station.pulsar.define_channel(id='ch1', name='RF', type='analog',
                              high=0.541, low=-0.541,
                              offset=0., delay=211e-9, active=True)
station.pulsar.define_channel(id='ch1_marker1', name='MW_pulsemod', type='marker',
                              high=2.0, low=0, offset=0.,
                              delay=(44+166-8)*1e-9, active=True)

station.pulsar.AWG_sequence_cfg = {
    'SAMPLING_RATE': 1e9,
    'CLOCK_SOURCE': 1,  # Internal | External
    'REFERENCE_SOURCE':   1,  # Internal | External
    'EXTERNAL_REFERENCE_TYPE':   1,  # Fixed | Variable
    'REFERENCE_CLOCK_FREQUENCY_SELECTION': 1,  # 10 MHz | 20 MHz | 100 MHz
    'TRIGGER_SOURCE':   1,  # External | Internal
    'TRIGGER_INPUT_IMPEDANCE':   1,  # 50 ohm | 1 kohm
    'TRIGGER_INPUT_SLOPE':   1,  # Positive | Negative
    'TRIGGER_INPUT_POLARITY':   1,  # Positive | Negative
    'TRIGGER_INPUT_THRESHOLD':   0.6,  # V
    'EVENT_INPUT_IMPEDANCE':   2,  # 50 ohm | 1 kohm
    'EVENT_INPUT_POLARITY':   1,  # Positive | Negative
    'EVENT_INPUT_THRESHOLD':   1.4,  # V
    'JUMP_TIMING':   1,  # Sync | Async
    'RUN_MODE':   4,  # Continuous | Triggered | Gated | Sequence
    'RUN_STATE':   0,  # On | Off
}


# Generating an example sequence
test_element = element.Element('a_test_element', pulsar=station.pulsar)
# we copied the channel definition from out global pulsar
print('Channel definitions: ')
pprint.pprint(test_element._channels)
print()

# define some bogus pulses.
sin_pulse = pulse.SinePulse(channel='RF', name='A sine pulse on RF')
sq_pulse = pulse.SquarePulse(channel='MW_pulsemod',
                             name='A square pulse on MW pmod')

special_pulse = pulse.SinePulse(channel='RF', name='special pulse')
special_pulse.amplitude = 0.2
special_pulse.length = 2e-6
special_pulse.frequency = 10e6
special_pulse.phase = 0

# create a few of those
test_element.add(pulse.cp(sin_pulse, frequency=1e6, amplitude=1, length=1e-6),
                 name='first pulse')
test_element.add(pulse.cp(sq_pulse, amplitude=1, length=1e-6),
                 name='second pulse', refpulse='first pulse', refpoint='end')
test_element.add(pulse.cp(sin_pulse, frequency=2e6, amplitude=0.5, length=1e-6),
                 name='third pulse', refpulse='second pulse', refpoint='end')

print('Element overview:')
# test_element.print_overview()
print()

special_element = element.Element('Another_element', pulsar=station.pulsar)
special_element.add(special_pulse)

# create the sequnce
# note that we re-use the same waveforms (just with different identifier
# names)
seq = pulsar.Sequence('A Sequence')
seq.append(name='first_element', wfname='a_test_element', trigger_wait=True,
           goto_target='first_element', jump_target='first special element')
seq.append('first special element', 'Another_element',
           repetitions=5)
seq.append('third element', 'a_test_element', trigger_wait=True,
           goto_target='third element', jump_target='second special element')
seq.append('second special element', 'Another_element',
           repetitions=5)

# program the Sequence
station.pulsar.program_awg(seq, test_element, special_element)
