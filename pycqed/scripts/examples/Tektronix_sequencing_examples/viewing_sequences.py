# Example of how to view a tektronix sequence
import qcodes as qc
from pycqed.measurement.waveform_control import viewer
reload(viewer)
station = qc.station

# Shows the first two elements of the last uploaded AWG sequence

# By uncommenting this you can reuse the plotting window
try:
    vw.clear()
except:
    vw = None
for i, elt_idx in enumerate([0, 1]):
    vw = viewer.show_element_pyqt(
        station.pulsar.last_elements[elt_idx], vw,
        color_idx= i % len(viewer.color_cycle))
