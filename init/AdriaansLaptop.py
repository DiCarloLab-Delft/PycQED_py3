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
from modules.analysis import analysis_toolbox as a_tools

# Initializing instruments
station = qc.Station()
MC = mc.MeasurementControl('MC')
MC.station = station
station.MC = MC

from qcodes.tests.instrument_mocks import MockParabola

ParabInstr = MockParabola('ParabInstr')
station.add_instrument(ParabInstr)

# station.pulsar = ps.Pulsar()
# # station.pulsar.AWG = station.instruments['AWG']
# for i in range(4):
#     # Note that these are default parameters and should be kept so.
#     # the channel offset is set in the AWG itself. For now the amplitude is
#     # hardcoded. You can set it by hand but this will make the value in the
#     # sequencer different.
#     station.pulsar.define_channel(id='ch{}'.format(i+1),
#                                   name='ch{}'.format(i+1), type='analog',
#                                   # max safe IQ voltage
#                                   high=.5, low=-.5,
#                                   offset=0.0, delay=0, active=True)
#     station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
#                                   name='ch{}_marker1'.format(i+1),
#                                   type='marker',
#                                   high=2.0, low=0, offset=0.,
#                                   delay=0, active=True)
#     station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
#                                   name='ch{}_marker2'.format(i+1),
#                                   type='marker',
#                                   high=2.0, low=0, offset=0.,
#                                   delay=0, active=True)



t1 = time.time()
print('Ran initialization in %.2fs' % (t1-t0))
