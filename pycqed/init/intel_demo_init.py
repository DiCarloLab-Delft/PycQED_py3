
import time
import logging
t0 = time.time()  # to print how long init takes

from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib.pyplot as plt

import qcodes as qc
# Globally defined config

qc_config = {'datadir': r'D:\Experiments\\1611_intel_demo\data',
             'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}



# General PycQED modules
from pycqed.measurement import measurement_control as mc
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det

# QASM modules
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta
from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sq_qasm
from pycqed.instrument_drivers.physical_instruments._controlbox \
    import Assembler

# Importing instruments
from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb
import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as lm
from pycqed.instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt
from pycqed.instrument_drivers.meta_instrument import CBox_LookuptableManager as cbl


station = qc.Station()

CBox = qcb.QuTech_ControlBox_v3('CBox', address='Com7', run_tests=False, server_name=None)
station.add_component(CBox)
LutMan = cbl.QuTech_ControlBox_LookuptableManager('Lutman', CBox)
station.add_component(LutMan)

MC = mc.MeasurementControl('MC')
MC.station = station

