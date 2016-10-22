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
# qc.show_subprocess_widget()
# Globally defined config
# qc_config = {'datadir': r'D:\\Experimentsp7_Qcodes_5qubit',
#              'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}
qc_config = {'datadir': r'D:\Experiments\\1607_Qcodes_5qubit\data',
             'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}

# makes sure logging messages show up in the notebook
root = logging.getLogger()
root.addHandler(logging.StreamHandler())

# General PycQED modules
from pycqed.measurement import measurement_control as mc
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.measurement import calibration_toolbox as cal_tools
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement import CBox_sweep_functions as cb_swf
from pycqed.measurement.optimization import nelder_mead
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_swf_m
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as sq_m


from pycqed.utilities import general as gen
# Standarad awg sequences
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
from pycqed.measurement.pulse_sequences import calibration_elements as cal_elts
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq

# Instrument drivers
from qcodes.instrument_drivers.rohde_schwarz import SGS100A as rs
import qcodes.instrument_drivers.signal_hound.USB_SA124B as sh
import qcodes.instrument_drivers.QuTech.IVVI as iv
from qcodes.instrument_drivers.tektronix import AWG5014 as tek
from qcodes.instrument_drivers.tektronix import AWG520 as tk520
from qcodes.instrument_drivers.agilent.E8527D import Agilent_E8527D

from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBoxdriver as qcb
import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from pycqed.instrument_drivers.meta_instrument import heterodyne as hd
import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as lm

from pycqed.instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as qb
from pycqed.instrument_drivers.physical_instruments import QuTech_Duplexer as qdux
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.physical_instruments import Weinschel_8320_novisa

# Initializing instruments


station = qc.Station()
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
                            address='GPIB0::6::INSTR', server_name=None)
station.add_component(AWG)
AWG.timeout(180)

MC = mc.MeasurementControl('MC')

MC.station = station
station.MC = MC
nested_MC = mc.MeasurementControl('nested_MC')
nested_MC.station = station

# The AWG sequencer
station.pulsar = ps.Pulsar()
station.pulsar.AWG = station.components['AWG']
marker1highs=[2,2,2.7,2]
for i in range(4):
    # Note that these are default parameters and should be kept so.
    # the channel offset is set in the AWG itself. For now the amplitude is
    # hardcoded. You can set it by hand but this will make the value in the
    # sequencer different.
    station.pulsar.define_channel(id='ch{}'.format(i+1),
                                  name='ch{}'.format(i+1), type='analog',
                                  # max safe IQ voltage
                                  high=.7, low=-.7,
                                  offset=0.0, delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                  name='ch{}_marker1'.format(i+1),
                                  type='marker',
                                  high=marker1highs[i], low=0, offset=0.,
                                  delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                  name='ch{}_marker2'.format(i+1),
                                  type='marker',
                                  high=2.0, low=0, offset=0.,
                                  delay=0, active=True)
# to make the pulsar available to the standard awg seqs
st_seqs.station = station
sq.station = station
cal_elts.station = station

t1 = time.time()

#manually setting the clock, to be done automatically
AWG.clock_freq(1e9)


print('Ran initialization in %.2fs' % (t1-t0))
