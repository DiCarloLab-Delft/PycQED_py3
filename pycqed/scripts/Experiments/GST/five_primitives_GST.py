import qcodes as qc
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det
import measurement.gate_set_tomography.gate_set_tomography as gsts
reload(gsts)
gsts.station = station


station = station
MC = MC
CBox = station.components['CBox']

# calibrate_RO_threshold_no_rotation()

exec(open(PyCQEDpath+'\scripts\Experiments\Restless\prepare_for_restless.py').read())

# Defined in prepare for CLEAR
measure_GST(upload=True, l=512, nr_elts=6103, nr_logs=40)
