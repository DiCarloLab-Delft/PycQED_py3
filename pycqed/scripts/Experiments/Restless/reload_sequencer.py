from pycqed.measurement.waveform_control import pulse
reload(pulse)

from pycqed.measurement.waveform_control import pulse_library as pulselib
reload(pulselib)
reload(ps)
from pycqed.measurement.waveform_control import element
reload(element)
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
reload(sq)
from pycqed.measurement.pulse_sequences import single_qubit_2nd_exc_seqs as sq2
reload(sq2)

from pycqed.measurement.pulse_sequences import standard_elements as ste
reload(ste)

reload(awg_swf)

station.pulsar = ps.Pulsar()
station.pulsar.AWG = station.components['AWG']
for i in range(4):
    # Note that these are default parameters and should be kept so.
    # the channel offset is set in the AWG itself. For now the amplitude is
    # hardcoded. You can set it by hand but this will make the value in the
    # sequencer different.
    station.pulsar.define_channel(id='ch{}'.format(i+1),
                                  name='ch{}'.format(i+1), type='analog',
                                  # max safe IQ voltage
                                  high=.5, low=-.5,
                                  offset=0.0, delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                  name='ch{}_marker1'.format(i+1),
                                  type='marker',
                                  high=2.0, low=0, offset=0.,
                                  delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                  name='ch{}_marker2'.format(i+1),
                                  type='marker',
                                  high=2.0, low=0, offset=0.,
                                  delay=0, active=True)

sq.station=station
sq2.station = station
cal_elts.station = station


print('reloaded sequences')
