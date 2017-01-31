# Example of how to view a tektronix sequence
import qcodes as qc
from pycqed.measurement.waveform_control import viewer
station = qc.station

# Shows the first two elements of the last uploaded AWG sequence

# By uncommenting this you can reuse the plotting window
vw = None
# vw.clear()
for i in [0, 1]:
    vw = viewer.show_element_pyqt(
        station.pulsar.last_elements[i], vw, color_idx=i)
