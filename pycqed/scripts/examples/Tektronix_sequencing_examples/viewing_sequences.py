# Example of how to view a tektronix sequence
import qcodes as qc
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs

fsqs.station = qc.station


seq, elts = fsqs.Ram_Z_seq(AncT.get_operation_dict(), 'AncT',
               dist_dict, t_dict, upload=False, return_seq=True)


ax = seq_viewer.show_element_dclab(element=elts[0])
ax = seq_viewer.show_element_dclab(element=elts[4], ax=ax)
ax = seq_viewer.show_element_dclab(element=elts[10], ax=ax)


ax.set_xlim(0, 2600)
ax.set_ylim(-3, 3)
