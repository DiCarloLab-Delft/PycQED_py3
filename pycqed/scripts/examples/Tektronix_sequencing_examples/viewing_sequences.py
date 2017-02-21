# Example of how to view a tektronix sequence
import qcodes as qc
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
from pycqed.measurement.waveform_control import viewer
fsqs.station = qc.station


seq, elts = fsqs.Ram_Z_seq(AncT.get_operation_dict(), 'AncT',
               dist_dict, t_dict, upload=False, return_seq=True)


ax = viewer.show_element_dclab(element=elts[0])
ax = viewer.show_element_dclab(element=elts[4], ax=ax)
ax = viewer.show_element_dclab(element=elts[10], ax=ax)


ax.set_xlim(0, 2600)
ax.set_ylim(-3, 3)
