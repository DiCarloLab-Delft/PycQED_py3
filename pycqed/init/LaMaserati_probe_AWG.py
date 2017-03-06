"""
This scripts initializes the instruments and imports the modules
"""

UHFQC = True

# General imports

import time
import logging
t0 = time.time()
from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab

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
from pycqed.measurement import multi_qubit_module as mq_mod
from pycqed.analysis import tomography as tomo
import pycqed.scripts.Experiments.Five_Qubits.cost_functions_Leo_optimization as ca
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

from pycqed.instrument_drivers.meta_instrument import heterodyne as hd
import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as lm

from pycqed.instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as qb
from pycqed.instrument_drivers.physical_instruments import QuTech_Duplexer as qdux
if UHFQC:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.physical_instruments import Weinschel_8320_novisa
# for multiplexed readout
import pycqed.instrument_drivers.meta_instrument.UHFQC_LookuptableManager as lm_UHFQC
import pycqed.instrument_drivers.meta_instrument.UHFQC_LookuptableManagerManager as lmm_UHFQC
from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_swf_m
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as sq_m
import pycqed.scripts.personal_folders.Niels.two_qubit_readout_analysis as Niels

# for flux pulses
from pycqed.scripts.Experiments.Five_Qubits import cost_functions_Leo_optimization as cl
import pycqed.instrument_drivers.meta_instrument.kernel_object as k_obj
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
# Initializing instruments


station = qc.Station()
qc.station = station  # makes it easily findable from inside files
LO = rs.RohdeSchwarz_SGS100A(
    name='LO', address='TCPIP0::192.168.0.73', server_name=None)  #
station.add_component(LO)
RF = rs.RohdeSchwarz_SGS100A(
    name='RF', address='TCPIP0::192.168.0.74', server_name=None)  #
station.add_component(RF)
QL_LO = rs.RohdeSchwarz_SGS100A(
    name='QL_LO', address='TCPIP0::192.168.0.86', server_name=None)  #
station.add_component(QL_LO)
QR_LO = rs.RohdeSchwarz_SGS100A(
    name='QR_LO', address='TCPIP0::192.168.0.87', server_name=None)  #
station.add_component(QR_LO)
# TWPA_Pump = rs.RohdeSchwarz_SGS100A(name='TWPA_Pump', address='TCPIP0::192.168.0.90', server_name=None)  #
# station.add_component(TWPA_Pump)
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
                            address='TCPIP0::192.168.0.99::INSTR', server_name=None)
station.add_component(AWG)
AWG.timeout(180)  # timeout long for uploading wait.
# AWG520 = tk520.Tektronix_AWG520('AWG520', address='GPIB0::17::INSTR',
#                                 server_name='')
# station.add_component(AWG520)
# CBox = qcb.QuTech_ControlBox('CBox', address='Com5', run_tests=False, server_name=None)
# station.add_component(CBox)
IVVI = iv.IVVI('IVVI', address='COM8', numdacs=16, server_name=None)
station.add_component(IVVI)


from pycqed.instrument_drivers.meta_instrument.flux_control import Flux_Control
FC = Flux_Control('FC', 2, IVVI.name)
station.add_component(FC)
# gen.load_settings_onto_instrument(FC)
dac_offsets= np.array([75, -16.59])

dac_mapping = np.array([2, 1])
A = np.array([[-1.71408478e-13,   1.00000000e+00],
              [1.00000000e+00,  -1.32867524e-14]])
FC.transfer_matrix(A)
FC.dac_mapping(dac_mapping)
FC.dac_offsets(dac_offsets)


# Initializing UHFQC
UHFQC_1 = ZI_UHFQC.UHFQC('UHFQC_1', device='dev2178', server_name=None)
station.add_component(UHFQC_1)
UHFQC_1.sigins_0_ac()
UHFQC_1.sigins_1_ac()
# preparing the lookuptables for readout
LutMan0 = lm_UHFQC.UHFQC_LookuptableManager('LutMan0', UHFQC=UHFQC_1,
                                            server_name=None)
station.add_component(LutMan0)
LutMan1 = lm_UHFQC.UHFQC_LookuptableManager('LutMan1', UHFQC=UHFQC_1,
                                            server_name=None)
station.add_component(LutMan1)

LutManMan = lmm_UHFQC.UHFQC_LookuptableManagerManager('LutManMan', UHFQC=UHFQC_1,
                                                      server_name=None)
station.add_component(LutManMan)
LutManMan.LutMans(
    [LutMan0.name, LutMan1.name])


# SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None)
# #commented because of 8s load time

# Meta-instruments
HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, AWG=AWG, acquisition_instr=UHFQC_1.name,
                             server_name=None)
station.add_component(HS)
LutMan = lm.QuTech_ControlBox_LookuptableManager('LutMan', CBox=None,
                                                 server_name=None)

MC = mc.MeasurementControl('MC')
station.add_component(MC)
# HS = None
import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
QL = qbt.Tektronix_driven_transmon('QL', LO=LO, cw_source=QR_LO,
                                   td_source=QL_LO,
                                   IVVI=IVVI, rf_RO_source=RF,
                                   AWG=AWG,
                                   heterodyne_instr=HS,
                                   FluxCtrl=FC,
                                   MC=MC,
                                   server_name=None)
station.add_component(QL)
QR = qbt.Tektronix_driven_transmon('QR', LO=LO, cw_source=QL_LO,
                                   td_source=QR_LO,
                                   IVVI=IVVI, rf_RO_source=RF,
                                   AWG=AWG,
                                   heterodyne_instr=HS,
                                   FluxCtrl=FC,
                                   MC=MC,
                                   server_name=None)
station.add_component(QR)

MC.station = station
station.MC = MC
nested_MC = mc.MeasurementControl('nested_MC')
nested_MC.station = station

# The AWG sequencer
station.pulsar = ps.Pulsar()
station.pulsar.AWG = station.components['AWG']
markerhighs = [2, 2, 2.7, 2]
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
                                  high=markerhighs[i], low=0, offset=0.,
                                  delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                  name='ch{}_marker2'.format(i+1),
                                  type='marker',
                                  high=markerhighs[i], low=0, offset=0.,
                                  delay=0, active=True)
# to make the pulsar available to the standard awg seqs
st_seqs.station = station
sq.station = station
awg_swf.fsqs.station = station
cal_elts.station = station
mqs.station = station


t1 = time.time()

# manually setting the clock, to be done automatically
AWG.clock_freq(1e9)


print('Ran initialization in %.2fs' % (t1-t0))


def all_sources_off():
    LO.off()
    RF.off()
    Spec_source.off()
    Qubit_LO.off()
    # TWPA_Pump.off()


def print_instr_params(instr):
    snapshot = instr.snapshot()
    for par in sorted(snapshot['parameters']):
        print('{}: {} {}'.format(snapshot['parameters'][par]['name'],
                                 snapshot['parameters'][par]['value'],
                                 snapshot['parameters'][par]['units']))


# from scripts.Experiments.FiveQubits import common_functions as cfct
# cfct.set_AWG_limits(station,1.7)

if UHFQC:
    def switch_to_pulsed_RO_CBox(qubit):
        UHFQC_1.awg_sequence_acquisition()
        qubit.RO_pulse_type('Gated_MW_RO_pulse')
        qubit.RO_acq_marker_delay(155e-9)
        qubit.acquisition_instr('CBox')
        qubit.RO_acq_marker_channel('ch3_marker1')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)

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
                                                       RO_amp=qubit.RO_amp(),
                                                       RO_pulse_length=qubit.RO_pulse_length(),
                                                       acquisition_delay=270e-9)
        qubit.RO_pulse_type('MW_IQmod_pulse_UHFQC')  # changed to satisfy validator
        qubit.RO_acq_marker_delay(-165e-9)
        qubit.acquisition_instr('UHFQC_1')
        qubit.RO_acq_marker_channel('ch3_marker2')
        qubit.RO_I_channel('0')
        qubit.RO_Q_channel('1')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)
else:
    def switch_to_pulsed_RO_CBox(qubit):
        qubit.RO_pulse_type('Gated_MW_RO_pulse')
        qubit.RO_acq_marker_delay(155e-9)
        qubit.acquisition_instr('CBox')
        qubit.RO_acq_marker_channel('ch3_marker1')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)


from pycqed.instrument_drivers.physical_instruments.Fridge_monitor import Fridge_Monitor
Maserati_fridge_mon = Fridge_Monitor('Maserati_fridge_mon', 'LaMaserati')
station.add_component(Maserati_fridge_mon)

# def reload_mod_stuff():

# preparing UHFQC readout with IQ mod pulses

list_qubits = [QL, QR]
for qubit in list_qubits:
    qubit.RO_pulse_delay(20e-9)
    # qubit.RO_acq_averages(2**13)

# switch_to_pulsed_RO_CBox(AncT)
# switch_to_pulsed_RO_CBox(DataT)


def reload_mod_stuff():
    from pycqed.measurement.waveform_control import pulse_library as pl
    reload(pl)
    from pycqed.measurement.waveform_control import pulsar as ps
    reload(ps)
    from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
    reload(sq)
    # The AWG sequencer
    qc.station.pulsar = ps.Pulsar()
    sq.station = station
    sq.station.pulsar = qc.station.pulsar
    station.pulsar.AWG = station.components['AWG']
    markerhighs = [2, 2, 2.7, 2]
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
                                      high=markerhighs[i], low=0, offset=0.,
                                      delay=0, active=True)
        station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                      name='ch{}_marker2'.format(i+1),
                                      type='marker',
                                      high=markerhighs[i], low=0, offset=0.,
                                      delay=0, active=True)

    from pycqed.measurement.waveform_control_CC import waveform as wf
    reload(wf)

    from pycqed.measurement.pulse_sequences import fluxing_sequences as fqqs
    reload(fqqs)
    from pycqed.scripts.Experiments.Five_Qubits import cost_functions_Leo_optimization as ca
    reload(ca)
    from pycqed.measurement.waveform_control import pulse_library as pl
    reload(pl)
    from pycqed.measurement.pulse_sequences import standard_elements as ste
    reload(ste)

    from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
    reload(mqs)
    from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_mswf
    reload(awg_mswf)
    reload(awg_swf)
    mqs.station = station
    fqqs.station = station
    reload(mq_mod)
    mq_mod.station = station

    reload(fsqs)
    reload(awg_swf)
    fsqs.station = station
    reload(det)
    reload(ca)
reload_mod_stuff()


################################
# Reloading qubit snippet
################################
import qcodes as qc
station = qc.station
from qcodes.utils import validators as vals
from pycqed.instrument_drivers.meta_instrument.qubit_objects import qubit_object as qo
from pycqed.instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as cbt
from pycqed.instrument_drivers.meta_instrument.qubit_objects import Tektronix_driven_transmon as qbt

QL.RO_acq_weight_function_I(0)
QL.RO_acq_weight_function_Q(0)
QR.RO_acq_weight_function_I(1)
QR.RO_acq_weight_function_Q(1)


gen.load_settings_onto_instrument(QL)
gen.load_settings_onto_instrument(QR)


station.sequencer_config = {'RO_fixed_point': 1e-6,
                                'Buffer_Flux_Flux': 0,
                                'Buffer_Flux_MW': 0,
                                'Buffer_Flux_RO': 0,
                                'Buffer_MW_Flux': 0,
                                'Buffer_MW_MW': 0,
                                'Buffer_MW_RO': 0,
                                'Buffer_RO_Flux': 0,
                                'Buffer_RO_MW': 0,
                                'Buffer_RO_RO': 0,
                                'Flux_comp_dead_time': 3e-6,
                                }
