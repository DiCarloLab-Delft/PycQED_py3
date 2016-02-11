# General imports
import time
t0 = time.time()
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload  # Useful for reloading during testin
# Qcodes
import qcodes as qc
qc.set_mp_method('spawn')  # force Windows behavior on mac

# Globally defined config
qc_config = {'datadir': '/Users/Adriaan/Documents/Testing',
             'PycQEDdir': '/Users/Adriaan/GitHubRepos/DiCarloLabRepositories/PycQED_py3'}

# General PycQED modules
from modules.measurement import measurement_control as mc
from modules.measurement import sweep_functions as swf
from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.analysis import measurement_analysis as ma
from modules.measurement import mc_parameter_wrapper as pw

# Initializing instruments
station = qc.Station()
MC = mc.MeasurementControl('MC')
MC.station = station
station.MC = MC

t1 = time.time()


print('Ran initialization in %.2fs' % (t1-t0))
