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

qc_config = {'datadir': r'E:\Control software\data',
             'PycQEDdir': r'E:\Control software\PycQED_py3'}

# makes sure logging messages show up in the notebook
root = logging.getLogger()
root.addHandler(logging.StreamHandler())

# General PycQED modules
from pycqed.utilities import general as gen
from pycqed.measurement import measurement_control as mc
from pycqed.measurement import mc_parameter_wrapper as pw
from pycqed.measurement import detector_functions as det
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement import awg_sweep_functions as awg_swf
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
print("initializing SGS100A's")
drive_LO = rs.RohdeSchwarz_SGS100A(name='drive_LO', address='TCPIP0::192.168.1.35')
readout_LO = rs.RohdeSchwarz_SGS100A(name='readout_LO', address='TCPIP0::192.168.1.31')
readout_RF = rs.RohdeSchwarz_SGS100A(name='readout_RF', address='TCPIP0::192.168.1.37')
#cw_source = rs.RohdeSchwarz_SGS100A(name='cw_source', address='TCPIP0::192.168.0.73')
print("initializing AWG5014")
AWG = tek.Tektronix_AWG5014(name='AWG', timeout=20,
                            address='TCPIP0::192.168.1.4')
print("initializing UHFQC")
UHFQC = ZI_UHFQC.UHFQC('UHFQC', device='dev2204', port=8004)
print("initializing heterodynes")
#homodyne = hd.LO_modulated_Heterodyne('homodyne', LO=readout_LO, AWG=AWG,
#                                      acquisition_instr=UHFQC.name)
heterodyne = hd.HeterodyneInstrument('heterodyne', RF=readout_RF, LO=readout_LO,
                                     AWG=AWG, acquisition_instr=UHFQC.name)
print("initializing qubit")
MC = mc.MeasurementControl('MC')
MC.station = station
qb2 = QuDev_transmon('qb2', MC,
                       heterodyne = heterodyne,
                       cw_source = drive_LO,
                       readout_LO = readout_LO,
                       readout_RF = readout_RF,
                       drive_LO = drive_LO,
                       AWG = AWG,
                       UHFQC = UHFQC)

print('configuring parameters')
station.add_component(drive_LO)
station.add_component(readout_LO)
station.add_component(readout_RF)
#station.add_component(cw_source)
station.add_component(AWG)
station.add_component(UHFQC)
#station.add_component(homodyne)
station.add_component(heterodyne)
station.add_component(qb2)

station.pulsar = ps.Pulsar()
station.pulsar.AWG = AWG

from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
from pycqed.measurement.pulse_sequences import calibration_elements as cal_elts
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
st_seqs.station = station
sq.station = station
cal_elts.station = station
fsqs.station = station
mqs.station = station

# configure AWG
for i in range(4):
    # Note that these are default parameters and should be kept so.
    # the channel offset is set in the AWG itself. For now the amplitude is
    # hardcoded. You can set it by hand but this will make the value in the
    # sequencer different.
    station.pulsar.define_channel(id='ch{}'.format(i+1),
                                  name='ch{}'.format(i+1), type='analog',
                                  # max safe IQ voltage
                                  high=1, low=-1,
                                  offset=0.0, delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                  name='ch{}_marker1'.format(i+1),
                                  type='marker',
                                  high=2.7, low=0, offset=0.,
                                  delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                  name='ch{}_marker2'.format(i+1),
                                  type='marker',
                                  high=2.7, low=0, offset=0.,
                                  delay=0, active=True)
#station.pulsar.set_channel_opt("ch2_marker1", "high", 3)

AWG.clock_freq(1e9)
AWG.stop()
AWG.ch1_offset(0.0)
AWG.ch2_offset(0.0)
AWG.ch3_offset(0.0)
AWG.ch4_offset(0.0)
AWG.ch1_amp(2)
AWG.ch2_amp(2)
AWG.ch3_amp(2)
AWG.ch4_amp(2)
AWG.ch1_state(1)
AWG.ch2_state(1)
AWG.ch3_state(1)
AWG.ch4_state(1)

AWG.set_current_folder_name(r"C:\temp\PycQEDwaveforms")

# set up the UHFQC instrument
# set awg digital trigger 1 to trigger input 1
UHFQC.awgs_0_auxtriggers_0_channel(0)
# set outputs to 50 Ohm
UHFQC.sigouts_0_imp50(1)
UHFQC.sigouts_1_imp50(1)
# set awg output to 1:1 scale
UHFQC.awgs_0_outputs_0_amplitude(1)
UHFQC.awgs_0_outputs_1_amplitude(1)


# configure heterodyne instrument parameters
readout_LO.power(19) #dBm
heterodyne.RF_power(-40) #dBm

for hdyne in [heterodyne]:#, homodyne]:
    hdyne.f_RO_mod(25e6) #Hz
    hdyne.nr_averages(4096)
    hdyne.RO_length(2.2e-6) #s
    hdyne.trigger_separation(4e-6) #s
    hdyne.acq_marker_channels("ch1_marker2")
    hdyne.frequency(7.1903e9) #Hz
#homodyne.mod_amp(1) #V

# configure qubit 2 parameters
qb2.f_RO_resonator(0) #Hz
qb2.Q_RO_resonator(0)
qb2.optimal_acquisition_delay(0) #s
qb2.f_qubit(6023626529.174219) #Hz
qb2.spec_pow(-40) #dBm
qb2.f_RO(7.1903e9) #Hz
qb2.drive_LO_pow(19) #dBm
qb2.pulse_I_offset(0) #V
qb2.pulse_Q_offset(0) #V
qb2.RO_pulse_power(-20) #dBm
qb2.RO_I_offset(0) #V
qb2.RO_Q_offset(0) #V
qb2.RO_acq_averages(4096)
qb2.RO_acq_integration_length(2.2e-6)
qb2.RO_acq_weight_function_I(0)
qb2.RO_acq_weight_function_Q(1)

qb2.spec_pulse_type('SquarePulse')
qb2.spec_pulse_marker_channel('ch3_marker1') # gate drive LO = cw_source = MWG5
qb2.spec_pulse_amp(1) #V
qb2.spec_pulse_length(10e-6) #s
qb2.spec_pulse_depletion_time(2e-6) #s

qb2.RO_pulse_type('Gated_MW_RO_pulse')
qb2.RO_I_channel()
qb2.RO_Q_channel()
qb2.RO_pulse_marker_channel('ch2_marker1') # gates readout RF = MWG7
qb2.RO_amp(2.7) #V
qb2.RO_pulse_length(800e-9) #s
qb2.RO_pulse_delay(1e-6) #s
qb2.f_RO_mod(25e6) #Hz
qb2.RO_acq_marker_delay(-800e-9) #s
qb2.RO_acq_marker_channel('ch1_marker2') # triggers UHFLI
qb2.RO_pulse_phase(0) #rad

qb2.pulse_type('SSB_DRAG_pulse')
qb2.pulse_I_channel('ch1')
qb2.pulse_Q_channel('ch2')
qb2.amp180(0.36428994664187847) #V
qb2.amp90_scale(0.5119350936368918)
qb2.pulse_delay(0)
qb2.gauss_sigma(20e-9) #s
qb2.nr_sigma(6)
qb2.motzoi(0)
qb2.f_pulse_mod(100e6) #Hz
qb2.phi_skew(0)
qb2.alpha(1)
qb2.X_pulse_phase(0) #rad

t1 = time.time()
print('Ran initialization in %.2fs' % (t1-t0))
