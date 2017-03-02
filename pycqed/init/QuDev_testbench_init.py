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
drive_LO = rs.RohdeSchwarz_SGS100A(name='drive_LO', address='')
readout_LO = rs.RohdeSchwarz_SGS100A(name='readout_LO', address='')
readout_RF = rs.RohdeSchwarz_SGS100A(name='readout_RF', address='')
cw_source = rs.RohdeSchwarz_SGS100A(name='cw_source', address='')
print("initializing AWG5014")
AWG = tek.Tektronix_AWG5014(name='AWG', timeout=20,
                            address='')
print("initializing UHFQC")
UHFQC = ZI_UHFQC.UHFQC('UHFQC', device='dev2204', port=8004)
print("initializing heterodynes")
homodyne = hd.LO_modulated_Heterodyne('homodyne', LO=readout_LO, AWG=AWG,
                                      acquisition_instr=UHFQC.name)
heterodyne = hd.HeterodyneInstrument('heterodyne', RF=readout_RF, LO=readout_LO,
                                     AWG=AWG, acquisition_instr=UHFQC.name)
print("initializing qubit")
MC = mc.MeasurementControl('MC')
MC.station = station
qubit = QuDev_transmon('qubit', MC,
                       heterodyne = heterodyne,
                       cw_source = cw_source,
                       readout_LO = readout_LO,
                       readout_RF = readout_RF,
                       drive_LO = drive_LO,
                       AWG = AWG,
                       UHFQC = UHFQC)

print('configuring parameters')
station.add_component(drive_LO)
station.add_component(readout_LO)
station.add_component(readout_RF)
station.add_component(cw_source)
station.add_component(AWG)
station.add_component(UHFQC)
station.add_component(homodyne)
station.add_component(heterodyne)
station.add_component(qubit)

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
                                  high=.7, low=-.7,
                                  offset=0.0, delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                  name='ch{}_marker1'.format(i+1),
                                  type='marker',
                                  high=2, low=0, offset=0.,
                                  delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                  name='ch{}_marker2'.format(i+1),
                                  type='marker',
                                  high=2.0, low=0, offset=0.,
                                  delay=0, active=True)

AWG.stop()
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
readout_LO.power(18) #dBm
readout_RF.power(10) #dBm

for hdyne in [heterodyne, homodyne]:
    hdyne.f_RO_mod(100e6) #Hz
    hdyne.nr_averages(1024)
    hdyne.RO_length(500e-9) #s
    hdyne.trigger_separation(3e-6) #s
homodyne.mod_amp(1) #V

#configure qubit parameters
qubit.f_RO_resonator() #Hz
qubit.Q_RO_resonator()
qubit.optimal_acquisition_delay() #s
qubit.f_qubit() #Hz
qubit.spec_pow() #dBm
qubit.spec_pow_pulsed() #dBm
qubit.f_RO() #Hz
qubit.drive_LO_pow() #dBm
qubit.pulse_I_offset() #V
qubit.pulse_Q_offset() #V
qubit.RO_pulse_power() #dBm
qubit.RO_I_offset() #V
qubit.RO_Q_offset() #V

qubit.spec_pulse_type('SquarePulse')
qubit.spec_pulse_marker_channel()
qubit.spec_pulse_amp() #V
qubit.spec_pulse_length() #s
qubit.spec_pulse_depletion_time() #s

qubit.RO_pulse_type('MW_IQmod_pulse_UHFQC')
qubit.RO_I_channel()
qubit.RO_Q_channel()
qubit.RO_pulse_marker_channel()
qubit.RO_amp() #V
qubit.RO_pulse_length() #s
qubit.RO_pulse_delay() #s
qubit.f_RO_mod() #Hz
qubit.RO_acq_marker_delay() #s
qubit.RO_acq_marker_channel()
qubit.RO_pulse_phase(0) #rad

qubit.pulse_type('SSB_DRAG_pulse')
qubit.pulse_I_channel()
qubit.pulse_Q_channel()
qubit.amp180() #V
qubit.amp90_scale(0.5)
qubit.pulse_delay()
qubit.gauss_sigma() #s
qubit.nr_sigma(4)
qubit.motzoi(0)
qubit.f_pulse_mod() #Hz
qubit.phi_skew(0)
qubit.alpha(1)
qubit.X_pulse_phase(0) #rad

t1 = time.time()
print('Ran initialization in %.2fs' % (t1-t0))
