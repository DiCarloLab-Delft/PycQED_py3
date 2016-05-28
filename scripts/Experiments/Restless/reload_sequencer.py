from modules.measurement.waveform_control import pulse
reload(pulse)

from modules.measurement.waveform_control import pulse_library as pulselib
reload(pulselib)

from modules.measurement.waveform_control import element
reload(element)
from modules.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
reload(sq)

from modules.measurement.pulse_sequences import standard_elements as ste
reload(ste)

reload(awg_swf)

sq.station=station


print('reloaded sequences')