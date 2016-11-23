"""
This scripts initializes the instruments and imports the modules
"""


# General imports

import time
import logging
t0 = time.time()  # to print how long init takes

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
qc_config = {'datadir': r'D:\\Experiments\\1611_Starmon\\Data',
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
from qcodes.instrument_drivers.agilent.E8527D import Agilent_E8527D
from qcodes.instrument_drivers.rohde_schwarz import ZNB20 as ZNB20
from qcodes.instrument_drivers.weinschel import Weinschel_8320 as Weinschel_8320
from pycqed.instrument_drivers.physical_instruments import Weinschel_8320_novisa
# from pycqed.instrument_drivers.physical_instruments import Fridge_monitor as fm
from qcodes.instrument_drivers.tektronix import AWG5014 as tek
# from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBoxdriver as qcb

# from qcodes.instrument_drivers.tektronix import AWG520 as tk520
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.meta_instrument import Flux_Control as FluxCtrl

import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from pycqed.instrument_drivers.meta_instrument import heterodyne as hd
#import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as lm

############################
# Initializing instruments #
############################
station = qc.Station()


###########
# Sources #
###########
LO = Agilent_E8527D(name='LO', address='GPIB0::25', server_name=None)  #
station.add_component(LO)
RF = rs.RohdeSchwarz_SGS100A(name='RF', address='TCPIP0::192.168.0.80', server_name=None)  #
station.add_component(RF)
QL_LO = rs.RohdeSchwarz_SGS100A(name='QL_LO', address='TCPIP0::192.168.0.71', server_name=None)  #
station.add_component(QL_LO)
QR_LO = rs.RohdeSchwarz_SGS100A(name='QR_LO', address='TCPIP0::192.168.0.72', server_name=None)  #
station.add_component(QR_LO)
Spec_source = rs.RohdeSchwarz_SGS100A(name='Spec_source', address='TCPIP0::192.168.0.79', server_name=None)  #
station.add_component(Spec_source)

# VNA
#VNA = ZNB20.ZNB20(name='VNA', address='TCPIP0::192.168.0.55', server_name=None)  #
#station.add_component(VNA)
# Fridge_mon = fm.Fridge_Monitor('Fridge monitor', 'LaMaserati')
# station.add_component(Fridge_mon)


#Initializing UHFQC
UHFQC_1 = ZI_UHFQC.UHFQC('UHFQC_1', device='dev2214', server_name=None)
station.add_component(UHFQC_1)

#Initializing CBox
# CBox = qcb.QuTech_ControlBox('CBox', address='COM3', run_tests=False, server_name=None)
# station.add_component(CBox)


MC = mc.MeasurementControl('MC')

MC.station = station
station.MC = MC

AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
                            address='GPIB0::8::INSTR', server_name=None)
station.add_component(AWG)
AWG.timeout(180)
# AWG520 = tk520.Tektronix_AWG520('AWG520', address='GPIB0::17::INSTR',
#                                 server_name='')
# station.add_component(AWG520)
IVVI = iv.IVVI('IVVI', address='COM6', numdacs=8, server_name=None)
station.add_component(IVVI)
Flux_Control = FluxCtrl.Flux_Control(name='FluxControl',IVVI=station.IVVI)
station.add_component(Flux_Control)

# # Dux = qdux.QuTech_Duplexer('Dux', address='TCPIP0::192.168.0.101',
# #                             server_name=None)
# # SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None) #commented because of 8s load time

# Meta-instruments
HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, AWG=AWG, acquisition_instr=UHFQC_1.name,
                             server_name=None)
station.add_component(HS)

QL = qbt.Tektronix_driven_transmon('QL', LO=LO, cw_source=Spec_source,
                                              td_source=QL_LO,
                                              IVVI=IVVI, rf_RO_source=RF,
                                              AWG=AWG,
                                              heterodyne_instr=HS,
                                              MC=MC,
                                              FluxCtrl=Flux_Control,
                                              server_name=None)
station.add_component(QL)
QR = qbt.Tektronix_driven_transmon('QR', LO=LO, cw_source=Spec_source,
                                              td_source=QR_LO,
                                              IVVI=IVVI, rf_RO_source=RF,
                                              AWG=AWG,
                                              heterodyne_instr=HS,
                                              MC=MC,
                                              FluxCtrl=Flux_Control,
                                              server_name=None)
station.add_component(QR)

# # load settings onto qubits
# gen.load_settings_onto_instrument(QL)
# gen.load_settings_onto_instrument(QR)
# gen.load_settings_onto_instrument(HS)


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

def all_sources_off():
    LO.off()
    RF.off()
    Spec_source.off()
    QL_LO.off()
    QR_LO.off()


def print_instr_params(instr):
    snapshot = instr.snapshot()
    for par in sorted(snapshot['parameters']):
        print('{}: {} {}'.format(snapshot['parameters'][par]['name'],
                                 snapshot['parameters'][par]['value'],
                                 snapshot['parameters'][par]['units']))


# from scripts.Experiments.FiveQubits import common_functions as cfct
# cfct.set_AWG_limits(station,1.7)



def switch_to_pulsed_RO_UHFQC(qubit):
    UHFQC_1.awg_sequence_acquisition()
    qubit.RO_pulse_type('Gated_MW_RO_pulse')
    qubit.RO_acq_marker_delay(75e-9)
    qubit.acquisition_instr('UHFQC_1')
    qubit.RO_acq_marker_channel('ch3_marker2')
    qubit.RO_acq_weight_function_I(0)
    qubit.RO_acq_weight_function_Q(1)

def switch_to_IQ_mod_RO_UHFQC(qubit):
    UHFQC_1.awg_sequence_acquisition_and_pulse_SSB(f_RO_mod=qubit.f_RO_mod(),
                RO_amp=qubit.RO_amp(), RO_pulse_length=qubit.RO_pulse_length(),
                acquisition_delay=285e-9)
    qubit.RO_pulse_type('MW_IQmod_pulse_UHFQC')
    qubit.RO_acq_marker_delay(-200e-9)
    qubit.acquisition_instr('UHFQC_1')
    qubit.RO_acq_marker_channel('ch3_marker2')
    qubit.RO_I_channel('0')
    qubit.RO_Q_channel('1')
    qubit.RO_acq_weight_function_I(0)
    qubit.RO_acq_weight_function_Q(1)

def switch_to_pulsed_RO_CBox(qubit):
    UHFQC_1.awg_sequence_acquisition()
    qubit.RO_pulse_type('Gated_MW_RO_pulse')
    qubit.RO_acq_marker_delay(155e-9)
    qubit.acquisition_instr('CBox')
    qubit.RO_acq_marker_channel('ch3_marker1')
    qubit.RO_acq_weight_function_I(0)
    qubit.RO_acq_weight_function_Q(1)

# #preparing UHFQC readout with IQ mod pulses

switch_to_pulsed_RO_UHFQC(QL)
switch_to_pulsed_RO_UHFQC(QR)
