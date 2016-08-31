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
#LO = rs.RohdeSchwarz_SGS100A(name='LO', address='TCPIP0::192.168.0.73', server_name=None)  #
#station.add_component(LO)
#RF = rs.RohdeSchwarz_SGS100A(name='RF', address='TCPIP0::192.168.0.74', server_name=None)  #
#station.add_component(RF)
#Spec_source = rs.RohdeSchwarz_SGS100A(name='Spec_source', address='TCPIP0::192.168.0.87', server_name=None)  #
#station.add_component(Spec_source)
#Qubit_LO = rs.RohdeSchwarz_SGS100A(name='Qubit_LO', address='TCPIP0::192.168.0.86', server_name=None)  #
#station.add_component(Qubit_LO)
#TWPA_Pump = rs.RohdeSchwarz_SGS100A(name='TWPA_Pump', address='TCPIP0::192.168.0.90', server_name=None)  #
#station.add_component(TWPA_Pump)
CBox = qcb.QuTech_ControlBox('CBox', address='Com5', run_tests=False, server_name=None)
station.add_component(CBox)
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
                            address='GPIB0::6::INSTR', server_name=None)
station.add_component(AWG)
AWG.timeout(180)
#AWG520 = tk520.Tektronix_AWG520('AWG520', address='GPIB0::17::INSTR',
#                                server_name='')
#station.add_component(AWG520)
#IVVI = iv.IVVI('IVVI', address='COM4', numdacs=16, server_name=None)
#station.add_component(IVVI)
# Dux = qdux.QuTech_Duplexer('Dux', address='TCPIP0::192.168.0.101',
#                             server_name=None)
# SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None) #commented because of 8s load time

# Meta-instruments
#HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, CBox=CBox, AWG=AWG,
#                             server_name=None)
#station.add_component(HS)
# LutMan = lm.QuTech_ControlBox_LookuptableManager('LutMan', CBox=CBox,
#                                                  server_name=None)
                                                 # server_name='metaLM')
MC = mc.MeasurementControl('MC')



# AncB = qbt.Tektronix_driven_transmon('AncB', LO=LO, cw_source=Spec_source,
#                                               td_source=Qubit_LO,
#                                               IVVI=IVVI, rf_RO_source=RF,
#                                               AWG=AWG,
#                                               CBox=CBox, heterodyne_instr=HS,
#                                               MC=MC,
#                                               server_name=None)
# station.add_component(AncB)
# AncT = qbt.Tektronix_driven_transmon('AncT', LO=LO, cw_source=Spec_source,
#                                               td_source=Qubit_LO,
#                                               IVVI=IVVI, rf_RO_source=RF,
#                                               AWG=AWG,
#                                               CBox=CBox, heterodyne_instr=HS,
#                                               MC=MC,
#                                               server_name=None)
# station.add_component(AncT)
# DataB = qbt.Tektronix_driven_transmon('DataB', LO=LO, cw_source=Spec_source,
#                                               td_source=Qubit_LO,
#                                               IVVI=IVVI, rf_RO_source=RF,
#                                               AWG=AWG,
#                                               CBox=CBox, heterodyne_instr=HS,
#                                               MC=MC,
#                                               server_name=None)
# station.add_component(DataB)
# DataM = qbt.Tektronix_driven_transmon('DataM', LO=LO, cw_source=Spec_source,
#                                               td_source=Qubit_LO,
#                                               IVVI=IVVI, rf_RO_source=RF,
#                                               AWG=AWG,
#                                               CBox=CBox, heterodyne_instr=HS,
#                                               MC=MC,
#                                               server_name=None)
# station.add_component(DataM)
# DataT = qbt.Tektronix_driven_transmon('DataT', LO=LO, cw_source=Spec_source,
#                                               td_source=Qubit_LO,
#                                               IVVI=IVVI, rf_RO_source=RF,
#                                               AWG=AWG,
#                                               CBox=CBox, heterodyne_instr=HS,
#                                               MC=MC,
#                                               server_name=None)
# station.add_component(DataT)

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
                                  high=.7, low=-.7,
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

# def all_sources_off():
#     LO.off()
#     RF.off()
#     Spec_source.off()
#     Qubit_LO.off()
#     TWPA_Pump.off()


def print_instr_params(instr):
    snapshot = instr.snapshot()
    for par in snapshot['parameters']:
        print('{}: {} {}'.format(snapshot['parameters'][par]['name'],
                                 snapshot['parameters'][par]['value'],
                                 snapshot['parameters'][par]['units']))

def set_integration_weights():
    trace_length = 512
    tbase = np.arange(0, 5*trace_length, 5)*1e-9
    cosI = np.floor(127.*np.cos(2*np.pi*AncB.get('f_RO_mod')*tbase))
    sinI = np.floor(127.*np.sin(2*np.pi*AncB.get('f_RO_mod')*tbase))
    CBox.sig0_integration_weights(cosI)
    CBox.sig1_integration_weights(sinI)
from scripts.Experiments.FiveQubits import common_functions as cfct
cfct.set_AWG_limits(station,1.7)