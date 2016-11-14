
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

# Importing instruments
from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb
import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as lm
from pycqed.instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt


station = qc.Station()

MC = mc.MeasurementControl('MC')
MC.station = station