import qcodes as qc
from modules.measurement import sweep_functions as swf
from modules.measurement import detector_functions as det

station = qc.station
MC = station.MC
CBox = station.components['CBox']

MC.set_sweep_function(swf.None_Sweep(sweep_control='hard'))
MC.set_detector_function(det.CBox)
