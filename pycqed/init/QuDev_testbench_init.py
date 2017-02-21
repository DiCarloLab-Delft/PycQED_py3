# General imports
import time
t0 = time.time()  # to print how long init takes
import logging

from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Qcodes
import sys
import qcodes as qc

# makes sure logging messages show up in the notebook
root = logging.getLogger()
root.addHandler(logging.StreamHandler())

# General PycQED modules
from pycqed.utilities import general as gen
from pycqed.measurement import measurement_control as mc
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement import detector_functions as det
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools

# Instrument drivers
from qcodes.instrument_drivers.rohde_schwarz import SGS100A as rs
from qcodes.instrument_drivers.tektronix import AWG5014 as tek
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments \
    import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.meta_instrument import heterodyne as hd
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon

station = qc.Station()
LO = rs.RohdeSchwarz_SGS100A(name='LO', address='TCPIP0::192.168.1.14',
                             server_name=None)
RF = rs.RohdeSchwarz_SGS100A(name='RF', address='TCPIP0::192.168.1.16',
                             server_name=None)
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder="Setup_Folder", timeout=20,
                            address='TCPIP0::192.168.1.15::inst0::INSTR',
                            server_name=None)
UHFQC_1 = ZI_UHFQC.UHFQC('UHFQC_1', device='dev2204', server_name=None,
                         port=8004)
#HS = hd.LO_modulated_Heterodyne('HS', LO=LO, AWG=AWG,
#                                acquisition_instr=UHFQC_1.name,
#                                server_name=None)
HS = hd.HeterodyneInstrument('HS', RF=RF, LO=LO, AWG=AWG,
                                acquisition_instr=UHFQC_1.name,
                                server_name=None)
# CW = rs.RohdeSchwarz_SGS100A(name='CW', address='TCPIP0::192.168.1.16',
#                              server_name=None)  # CW source for probing qubit

station.add_component(LO)
station.add_component(AWG)
station.add_component(UHFQC_1)
station.add_component(HS)
# station.add_component(CW)

MC = mc.MeasurementControl('MC')
MC.station = station

station.pulsar = ps.Pulsar()
station.pulsar.AWG = station.components['AWG']

AWG.stop()
# set internal reference clock
AWG.write('SOUR1:ROSC:SOUR INT')

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

AWG.ch1_offset(0.0)
AWG.ch2_offset(0.0)
AWG.ch3_offset(0.0)
AWG.ch4_offset(0.0)
AWG.ch1_amp(1.4)
AWG.ch2_amp(1.4)
AWG.ch3_amp(1.4)
AWG.ch4_amp(1.4)
AWG.ch1_state(1)
AWG.ch2_state(1)
AWG.ch3_state(1)
AWG.ch4_state(1)

from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
st_seqs.station = station

# set up the UHFQC instrument
# set awg digital trigger 1 to trigger input 1
UHFQC_1.awgs_0_auxtriggers_0_channel(0)
# set outputs to 50 Ohm
UHFQC_1.sigouts_0_imp50(1)
UHFQC_1.sigouts_1_imp50(1)
# set awg output to 1:1 scale
UHFQC_1.awgs_0_outputs_0_amplitude(1)
UHFQC_1.awgs_0_outputs_1_amplitude(1)

# Set up heterodyne instrument parameters
LO_power = 18 #dBm
f_RO_mod = 100e6 #Hz
averages = 1
RO_length = 500e-9
mod_amp = 1
RF_power = 10

HS.LO.power(LO_power)
HS.f_RO_mod(f_RO_mod)
HS.nr_averages(averages)
HS.RO_length(RO_length)
#HS.mod_amp(mod_amp)
HS.RF_power(RF_power)

# Create qubit object
qubit = QuDev_transmon("qubit", MC, HS, None)

# load settings onto the qubit from the latest hdf5 file
gen.load_settings_onto_instrument(qubit)

t1 = time.time()
print('Ran initialization in %.2fs' % (t1-t0))
