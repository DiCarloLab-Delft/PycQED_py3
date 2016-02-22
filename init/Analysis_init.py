# General imports
import time
t0 = time.time()  # to print how long init takes
from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Qcodes
import qcodes as qc
qc.set_mp_method('spawn')  # force Windows behavior on mac

# Globally defined config
qc_config = {'datadir': 'D:\Experiments\Simultaneous_Driving\data',
             'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}


# General PycQED modules
from modules.analysis import measurement_analysis as ma
from modules.analysis import analysis_toolbox as a_tools
from modules.utilities import general as gen


t1 = time.time()


print('Ran initialization in %.2fs' % (t1-t0))
