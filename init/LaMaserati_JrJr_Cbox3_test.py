"""
This scripts initializes the instruments and imports the modules
"""


# General imports

import time
import logging
t0 = time.time()  # to print how long init takes
from instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt

from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Qcodes
import qcodes as qc
# currently on a Windows machine
# qc.set_mp_method('spawn')  # force Windows behavior on mac
qc.show_subprocess_widget()
# Globally defined config
# qc_config = {'datadir': r'D:\\Experimentsp7_Qcodes_5qubit',
#              'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}
qc_config = {'datadir': r'D:\Experiments\\1607_Qcodes_5qubit\data',
             'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}

# makes sure logging messages show up in the notebook
root = logging.getLogger()
root.addHandler(logging.StreamHandler())

# General PycQED modules
from modules.measurement import measurement_control as mc
from modules.measurement import sweep_functions as swf
from modules.measurement import awg_sweep_functions as awg_swf
from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.measurement import calibration_toolbox as cal_tools
from modules.measurement import mc_parameter_wrapper as pw
from modules.measurement import CBox_sweep_functions as cb_swf
from modules.measurement.optimization import nelder_mead
from modules.analysis import measurement_analysis as ma
from modules.analysis import analysis_toolbox as a_tools



from modules.utilities import general as gen
# Standarad awg sequences
from modules.measurement.waveform_control import pulsar as ps
from modules.measurement.pulse_sequences import standard_sequences as st_seqs
from modules.measurement.pulse_sequences import calibration_elements as cal_elts
from modules.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq

# Instrument drivers
from qcodes.instrument_drivers.rohde_schwarz import SGS100A as rs
import qcodes.instrument_drivers.signal_hound.USB_SA124B as sh
import qcodes.instrument_drivers.QuTech.IVVI as iv
from qcodes.instrument_drivers.tektronix import AWG5014 as tek
from qcodes.instrument_drivers.tektronix import AWG520 as tk520
from qcodes.instrument_drivers.agilent.E8527D import Agilent_E8527D

from instrument_drivers.physical_instruments import QuTech_ControlBoxdriver as qcb
import instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from instrument_drivers.meta_instrument import heterodyne as hd
import instrument_drivers.meta_instrument.CBox_LookuptableManager as lm

from instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as qb
from instrument_drivers.physical_instruments import QuTech_Duplexer as qdux


# Initializing instruments


station = qc.Station()

from instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb_v3


CBox_v3 = qcb_v3.QuTech_ControlBox_v3('CBox_v3', address='Com6', run_tests=False, server_name=None)
station.add_component(CBox_v3)

CBox_v3.set('measurement_timeout', 120)
CBox_v3.set('acquisition_mode', 'idle')
CBox_v3.set('run_mode', 0)
CBox_v3.set('signal_delay', 0)
CBox_v3.set('integration_length', 100)
CBox_v3.set('adc_offset', 0)
CBox_v3.set('log_length', 100)
CBox_v3.set('nr_samples', 100)
CBox_v3.set('nr_averages', 512)
CBox_v3.set('lin_trans_coeffs', [1, 0, 0, 1])