from pycqed.measurement import multi_qubit_module as mq_mod
from pycqed.measurement import single_qubit_fluxing_module as sqf_mod
import qcodes as qc

station = qc.station
S5 =station.components['S5']

mq_mod.measure_two_qubit_AllXY(S5, DataT.name, AncT.name)

