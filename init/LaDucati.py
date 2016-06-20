"""
This scripts initializes the instruments and imports the modules
"""


# General imports

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

AWG520 = tk520.Tektronix_AWG520('AWG520', address='GPIB0::17::INSTR',
                                server_name='')
# SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None) #commented because of 8s load time

LO = rs.RohdeSchwarz_SGS100A(name='LO', address='TCPIP0::192.168.0.77',
    server_name=None)  # left
RF = rs.RohdeSchwarz_SGS100A(name='RF', address='TCPIP0::192.168.0.78',
    server_name=None)  # right
Qubit_LO = rs.RohdeSchwarz_SGS100A(name='Qubit_LO', address='TCPIP0::192.168.0.11',
    server_name=None)  # top
Pump = Agilent_E8527D(name='Pump', address='TCPIP0::192.168.0.13',
                        server_name=None)
CBox = qcb.QuTech_ControlBox('CBox', address='Com3', run_tests=False)
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
                            address='TCPIP0::192.168.0.9', server_name=None)
AWG.timeout(180)
IVVI = iv.IVVI('IVVI', address='ASRL1', numdacs=16, server_name=None)
Dux = qdux.QuTech_Duplexer('Dux', address='TCPIP0::192.168.0.101',
                            server_name=None)

# Meta-instruments
HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, CBox=CBox, AWG=AWG,
                             server_name=None)
LutMan = lm.QuTech_ControlBox_LookuptableManager('LutMan', CBox=CBox,
                                                 server_name=None)
                                                 # server_name='metaLM')

MC = mc.MeasurementControl('MC')
VIP_mon_2_tek = qbt.Tektronix_driven_transmon('VIP_mon_2_tek',
                                              LO=LO,
                                              cw_source=Qubit_LO,
                                              td_source=Qubit_LO,
                                              IVVI=IVVI, rf_RO_source=RF,
                                              AWG=AWG,
                                              CBox=CBox, heterodyne_instr=HS,
                                              MC=MC,
                                              server_name=None)

VIP_mon_4_tek = qbt.Tektronix_driven_transmon('VIP_mon_4_tek',
                                              LO=LO,
                                              cw_source=Qubit_LO,
                                              td_source=Qubit_LO,
                                              IVVI=IVVI, rf_RO_source=RF,
                                              AWG=AWG,
                                              CBox=CBox, heterodyne_instr=HS,
                                              MC=MC,
                                              server_name=None)

VIP_mon_2_dux = dt.Duplexer_tek_transmon('VIP_mon_2_dux', LO=LO,
                                         cw_source=Qubit_LO,
                                         td_source=Qubit_LO,
                                         IVVI=IVVI, AWG=AWG, CBox=CBox,
                                         heterodyne_instr=HS, MC=MC, Mux=Dux,
                                         rf_RO_source=RF, server_name=None)

gen.load_settings_onto_instrument(VIP_mon_2_tek)
gen.load_settings_onto_instrument(VIP_mon_4_tek)

station = qc.Station(LO, RF, Qubit_LO, IVVI, Dux, Pump,
                     AWG, AWG520, HS, CBox, LutMan,
                     VIP_mon_2_dux,
                     VIP_mon_2_tek, VIP_mon_4_tek)
MC.station = station
station.MC = MC
nested_MC = mc.MeasurementControl('nested_MC')
nested_MC.station = station

# The AWG sequencer
station.pulsar = ps.Pulsar()
station.pulsar.AWG = station.components['AWG']
for i in range(4):
    # Note that these are default parameters and should be kept so.
    # the channel offset is set in the AWG itself. For now the amplitude is
    # hardcoded. You can set it by hand but this will make the value in the
    # sequencer different.
    station.pulsar.define_channel(id='ch{}'.format(i+1),
                                  name='ch{}'.format(i+1), type='analog',
                                  # max safe IQ voltage
                                  high=.5, low=-.5,
                                  offset=0.0, delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                  name='ch{}_marker1'.format(i+1),
                                  type='marker',
                                  high=2.0, low=0, offset=0.,
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



print('Ran initialization in %.2fs' % (t1-t0))
