# General imports

import time
import logging
t0 = time.time()  # to print how long init takes
from pycqed.instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt

from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab

# Qcodes
import qcodes as qc
station = qc.Station()
qc.station = station

from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb
CBox = qcb.QuTech_ControlBox_v3(
    'CBox', address='Com6', run_tests=False, server_name=None)
station.add_component(CBox)
# Initializaing ATS,


from pycqed.instrument_drivers.physical_instruments import QuTech_AWG_Module as qwg
QWG = qwg.QuTech_AWG_Module(
    'QWG', address='192.168.0.10',
    port=5025, server_name=None)
station.add_component(QWG)