
import time
t0 = time.time()  # to print how long init takes
from instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt

from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Qcodes
import qcodes as qc
qc.set_mp_method('spawn')  # force Windows behavior on mac
qc.show_subprocess_widget()
# Globally defined config
qc_config = {'datadir': 'D:\Experiments\Simultaneous_Driving\data',
             'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}
# General PycQED modules
from measurement import measurement_control as mc
from measurement import sweep_functions as swf
from measurement import awg_sweep_functions as awg_swf
from measurement import detector_functions as det
from measurement import composite_detector_functions as cdet
from measurement import calibration_toolbox as cal_tools
from measurement import mc_parameter_wrapper as pw
from measurement import CBox_sweep_functions as cb_swf
from measurement.optimization import nelder_mead
from analysis import measurement_analysis as ma
from analysis import analysis_toolbox as a_tools
from utilities import general as gen

t1 = time.time()
print('init took {:.2f}s'.format(t1-t0))
