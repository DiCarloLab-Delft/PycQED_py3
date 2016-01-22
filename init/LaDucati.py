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
from modules.measurement import awg_sweep_functions as awg_swf
from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.analysis import measurement_analysis as ma
from modules.measurement import calibration_toolbox as cal_tools
from modules.measurement import mc_parameter_wrapper as pw
# Standarad awg sequences
from modules.measurement.waveform_control import pulsar as ps
from modules.measurement.pulse_sequences import standard_sequences as st_seqs

# Instrument drivers
from qcodes.instrument_drivers import RS_SGS100A as rs
import qcodes.instrument_drivers.SignalHound_USB_SA124B as sh
import qcodes.instrument_drivers.IVVI as iv
from qcodes.instrument_drivers import Tektronix_AWG5014 as tek
from instrument_drivers.physical_instruments import QuTech_ControlBoxdriver as qcb
from instrument_drivers.meta_instrument import heterodyne as hd
import instrument_drivers.meta_instrument.CBox_LookuptableManager as lm

# Initializing instruments

# SH = sh.SignalHound_USB_SA124B('Signal hound') #commented because of 8s load time
CBox = qcb.QuTech_ControlBox('CBox', address='Com3', run_tests=False)
S1 = rs.RS_SGS100A('S1', address='GPIB0::11::INSTR') #located on top of rack
LO = rs.RS_SGS100A(name='LO', address='TCPIP0::192.168.0.77')  # left of s2
S2 = rs.RS_SGS100A(name='S1', address='TCPIP0::192.168.0.78')  # right
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None,
                            address='TCPIP0::192.168.0.9')
IVVI = iv.IVVI('IVVI', address='ASRL1', numdacs=16)

# Meta-instruments
HS = hd.LO_modulated_Heterodyne('HS', LO=LO, CBox=CBox, AWG=AWG)
LutMan = lm.QuTech_ControlBox_LookuptableManager('LutMan', CBox)
station = qc.Station(LO, S1, S2, IVVI,
                     HS,
                     AWG, CBox, LutMan)
MC = mc.MeasurementControl('MC')
MC.station = station
station.MC = MC
nested_MC = mc.MeasurementControl('nested_MC')
nested_MC.station = station

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

IVVI.set('dac2', 0)
IVVI.set('dac5', 95.0)

RO_freq = 6.8482e9
qubit_freq = 6.4718e9 - 40e6  # as measured by my Ramsey
IF = -20e6        # RO modulation frequency
mod_freq = -40e6  # Qubit pulse modulation frequency

HS.set('mod_amp', .13)  # low power regime of VIPmon2
HS.set('IF', IF)
HS.set('frequency', RO_freq)  # Frequence of the RO resonator of qubit 2
LO.set('power', 16)  # splitting gives 13dBm at both mixer LO ports
LO.off()

S1.off()
S2.set('power', 14)
S2.set('frequency', qubit_freq - mod_freq)
S2.off()

LutMan.set('lut_mapping',
           ['I', 'X180', 'Y180', 'X90', 'Y90', 'I', 'I', 'I'])
LutMan.set('f_modulation', mod_freq*1e-9)  # Lutman works in ns and GHz
LutMan.set('gauss_width', 10)
amp180 = 50
# Need to add attenuation to ensure a more sensible value is used (e.g. 300)
LutMan.set('amp180', amp180)
LutMan.set('amp90', amp180/2)

# Calibrated at 6.5GHz (18-1-2016)
CBox.set_dac_offset(0, 1, 18.81)  # I channel qubit drive AWG
CBox.set_dac_offset(0, 0, -24.938)  # Q channel

CBox.set_dac_offset(1, 1, 0)  # I channel
CBox.set_dac_offset(1, 0, 0)  # Q channel readout AWG

t_base = np.arange(512)*5e-9

cosI = np.cos(2*np.pi * t_base*IF)
sinI = np.sin(2*np.pi * t_base*IF)
w0 = np.round(cosI*120)
w1 = np.round(sinI*120)

CBox.set('sig0_integration_weights', w0)
CBox.set('sig1_integration_weights', w1)

CBox.set('nr_averages', 2048)
# this is the max nr of averages that does not slow down the heterodyning
CBox.set('nr_samples', 75)  # Shorter because of min marker spacing
CBox.set('integration_length', 70)
CBox.set('acquisition_mode', 0)
CBox.set('lin_trans_coeffs', [1, 0, 0, 1])
CBox.set('log_length', 8000)

CBox.set('AWG0_mode', 'Tape')
CBox.set('AWG1_mode', 'Tape')
CBox.set('AWG0_tape', [1, 1])
CBox.set('AWG1_tape', [1, 1])

t1 = time.time()


print('Ran initialization in %.2fs' % (t1-t0))
