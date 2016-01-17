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
qc_config = {'datadir': 'D:\Experiments\Simultaneous_Driving\data',
             'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}

# General PycQED modules
from modules.measurement import measurement_control as mc
from modules.measurement import sweep_functions as swf
from modules.measurement import detector_functions as det
from modules.analysis import measurement_analysis as ma
from modules.measurement import calibration_toolbox as cal_tools
from modules.measurement import mc_parameter_wrapper as pw
# Standarad awg sequences
from modules.measurement.waveform_control import pulsar as ps
from modules.measurement.waveform_control import standard_sequences as st_seqs


# Instrument drivers
from qcodes.instrument_drivers import RS_SGS100A as rs
import qcodes.instrument_drivers.SignalHound_USB_SA124B as sh
import qcodes.instrument_drivers.IVVI as iv
from qcodes.instrument_drivers import Tektronix_AWG5014 as tek
from instrument_drivers.physical_instruments import QuTech_ControlBoxdriver as qcb
from instrument_drivers.meta_instrument import heterodyne as hd

# Initializing instruments
try:
    SH = sh.SignalHound_USB_SA124B('Signal hound')
except:
    SH.closeDevice()
    SH = sh.SignalHound_USB_SA124B('Signal hound')

CBox = qcb.QuTech_ControlBox('CBox', address='Com3', run_tests=False)
LO = rs.RS_SGS100A(name='LO', address='TCPIP0::192.168.0.77')
S1 = rs.RS_SGS100A(name='S1', address='TCPIP0::192.168.0.78')
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None,
                            address='TCPIP0::192.168.0.9')
IVVI = iv.IVVI('IVVI', address='ASRL1', numdacs=16)
HS = hd.LO_modulated_Heterodyne('HS', LO=LO, CBox=CBox, AWG=AWG)

MC = mc.MeasurementControl('MC')
station = qc.Station(LO, S1, AWG, CBox, HS, SH)
MC.station = station
station.MC = MC

# The AWG sequencer
station.pulsar = ps.Pulsar()
station.pulsar.AWG = station.instruments['AWG']
for i in range(4):
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

HS.set('mod_amp', .1)  # low power regime of VIPmon2
LO.set('power', 16)  # splitting gives 13dBm at both mixer LO ports
CBox.set('nr_averages', 2**12)
# this is the max nr of averages that does not slow down the heterodyning
CBox.set('nr_samples', 100)  # sets 500ns of integration in heterodyne

# !Not calibrated
CBox.set_dac_offset(0, 1, 0) #I channel qubit drive AWG
CBox.set_dac_offset(0, 0, 0) #Q channel
CBox.set_dac_offset(1, 1, 0) #I channel
CBox.set_dac_offset(1, 0, 0) #Q channel readout AWG


t1 = time.time()
print('Ran initialization in %.2fs' % (t1-t0))
