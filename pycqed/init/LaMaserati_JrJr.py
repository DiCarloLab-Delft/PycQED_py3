"""
This scripts initializes the instruments and imports the modules
"""

UHFQC = True

# General imports

import time
import logging
t0 = time.time()  # to print how long init takes
from pycqed.instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt

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
import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from pycqed.instrument_drivers.meta_instrument import heterodyne as hd
import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as lm

from pycqed.instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as qb
from pycqed.instrument_drivers.physical_instruments import QuTech_Duplexer as qdux
if UHFQC:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.physical_instruments import Weinschel_8320_novisa
from pycqed.instrument_drivers.meta_instrument import Flux_Control as fc
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
Spec_source = rs.RohdeSchwarz_SGS100A(
    name='Spec_source', address='TCPIP0::192.168.0.87', server_name=None)  #
station.add_component(Spec_source)
Qubit_LO = rs.RohdeSchwarz_SGS100A(
    name='Qubit_LO', address='TCPIP0::192.168.0.86', server_name=None)  #
station.add_component(Qubit_LO)
# TWPA_Pump = rs.RohdeSchwarz_SGS100A(name='TWPA_Pump', address='TCPIP0::192.168.0.90', server_name=None)  #
# station.add_component(TWPA_Pump)
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
                            address='TCPIP0::192.168.0.99', server_name=None)
station.add_component(AWG)
AWG.timeout(180)  # timeout long for uploading wait.
# AWG520 = tk520.Tektronix_AWG520('AWG520', address='GPIB0::17::INSTR',
#                                 server_name='')
# station.add_component(AWG520)
CBox = qcb.QuTech_ControlBox('CBox', address='Com5', run_tests=False, server_name=None)
station.add_component(CBox)
IVVI = iv.IVVI('IVVI', address='COM4', numdacs=16, server_name=None)
station.add_component(IVVI)

# flux pulsing
k1 = k_obj.Distortion(name='k1')
station.add_component(k1)
k0 = k_obj.Distortion(name='k0')
station.add_component(k0)

if UHFQC:
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
    LutMan2 = lm_UHFQC.UHFQC_LookuptableManager('LutMan2', UHFQC=UHFQC_1,
                                                server_name=None)
    station.add_component(LutMan2)
    LutMan3 = lm_UHFQC.UHFQC_LookuptableManager('LutMan3', UHFQC=UHFQC_1,
                                                server_name=None)
    station.add_component(LutMan3)
    LutMan4 = lm_UHFQC.UHFQC_LookuptableManager('LutMan4', UHFQC=UHFQC_1,
                                                server_name=None)
    station.add_component(LutMan4)
    LutManMan = lmm_UHFQC.UHFQC_LookuptableManagerManager('LutManMan', UHFQC=UHFQC_1,
                                                          server_name=None)
    station.add_component(LutManMan)
    LutManMan.LutMans(
        [LutMan0.name, LutMan1.name, LutMan2.name, LutMan3.name, LutMan4.name])

else:
    UHFQC_1 = None


Flux_Control = fc.Flux_Control(name='FluxControl', IVVI=station.IVVI)
station.add_component(Flux_Control)

transfer_matrix_dec = np.array([[4.70306717e-04,  -8.41312977e-05,   3.64442804e-05,  -1.00489353e-05,
                                 -2.36455362e-05],
                                [-6.70464355e-05,   6.39386703e-04,  -4.37263640e-05,  -2.01374983e-05,
                                 1.77516922e-05],
                                [7.69376917e-06,  -4.09893480e-05,   5.35184092e-04,  -2.36755094e-05,
                                 -5.34108608e-05],
                                [3.08518924e-05,   1.11315677e-05,   7.36191927e-05,   4.09078121e-04,
                                 -2.63031372e-05],
                                [-4.51217544e-05,  -1.35430841e-05,  -9.52349548e-05,  -4.18415379e-05,
                                 4.09962523e-04]])
invA = np.array([[2.17320666e+03, 2.79414032e+02, -1.16652799e+02, 7.08814870e+01, 1.02595827e+02],
                 [2.16689677e+02, 1.59642752e+03, 9.70544635e+01,
                     8.55894771e+01, -3.84925724e+01],
                 [1.52695260e+00, 1.25113953e+02, 1.90389457e+03,
                     1.42143094e+02, 2.51834186e+02],
                 [-1.55226336e+02, -8.03197377e+01, -3.10695549e+02,
                     2.43001891e+03, 1.09956406e+02],
                 [2.30860259e+02, 1.04357646e+02, 4.00934628e+02, 2.91661201e+02, 2.51899161e+03]])
Flux_Control.transfer_matrix(transfer_matrix_dec)
Flux_Control.inv_transfer_matrix(invA)

Flux_Control.dac_mapping([1, 2, 3, 4, 5])


# sweet_spots_mv = [-55.265, 49.643, -38.5, 13.037, 49.570]
sweet_spots_mv = [-31.5251, 54.1695, -0.3967, 4.9744, 60.3341]
offsets = np.dot(Flux_Control.transfer_matrix(), sweet_spots_mv)
Flux_Control.flux_offsets(-offsets)


# ATT = Weinschel_8320_novisa.Weinschel_8320(name='ATT',address='192.168.0.54', server_name=None)
# station.add_component(ATT)
# Dux = qdux.QuTech_Duplexer('Dux', address='TCPIP0::192.168.0.101',
#                             server_name=None)
# SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None)
# #commented because of 8s load time

# Meta-instruments
HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, AWG=AWG, acquisition_instr=UHFQC_1.name,
                             server_name=None)
station.add_component(HS)
LutMan = lm.QuTech_ControlBox_LookuptableManager('LutMan', CBox=CBox,
                                                 server_name=None)

MC = mc.MeasurementControl('MC')
station.add_component(MC)
# HS = None

AncB = qbt.Tektronix_driven_transmon('AncB', LO=LO, cw_source=Spec_source,
                                     td_source=Qubit_LO,
                                     IVVI=IVVI, rf_RO_source=RF,
                                     AWG=AWG,
                                     heterodyne_instr=HS,
                                     FluxCtrl=Flux_Control,
                                     MC=MC,
                                     server_name=None)
station.add_component(AncB)
AncT = qbt.Tektronix_driven_transmon('AncT', LO=LO, cw_source=Spec_source,
                                     td_source=Qubit_LO,
                                     IVVI=IVVI, rf_RO_source=RF,
                                     AWG=AWG,
                                     heterodyne_instr=HS,
                                     FluxCtrl=Flux_Control,
                                     MC=MC,
                                     server_name=None)
station.add_component(AncT)
DataB = qbt.Tektronix_driven_transmon('DataB', LO=LO, cw_source=Spec_source,
                                      td_source=Qubit_LO,
                                      IVVI=IVVI, rf_RO_source=RF,
                                      AWG=AWG,
                                      heterodyne_instr=HS,
                                      FluxCtrl=Flux_Control,
                                      MC=MC,
                                      server_name=None)
station.add_component(DataB)
DataM = qbt.Tektronix_driven_transmon('DataM', LO=LO, cw_source=Spec_source,
                                      td_source=Qubit_LO,
                                      IVVI=IVVI, rf_RO_source=RF,
                                      AWG=AWG,
                                      heterodyne_instr=HS,
                                      FluxCtrl=Flux_Control,
                                      MC=MC,
                                      server_name=None)
station.add_component(DataM)
DataT = qbt.Tektronix_driven_transmon('DataT', LO=LO, cw_source=Spec_source,
                                      td_source=Qubit_LO,
                                      IVVI=IVVI, rf_RO_source=RF,
                                      AWG=AWG,
                                      heterodyne_instr=HS,
                                      FluxCtrl=Flux_Control,
                                      MC=MC,
                                      server_name=None)
station.add_component(DataT)

# load settings onto qubits
gen.load_settings_onto_instrument(AncB)
gen.load_settings_onto_instrument(AncT)
gen.load_settings_onto_instrument(DataB)
gen.load_settings_onto_instrument(DataM)
gen.load_settings_onto_instrument(DataT)
gen.load_settings_onto_instrument(HS)
gen.load_settings_onto_instrument(k0)
gen.load_settings_onto_instrument(k1)


AncT.E_c(0.28e9)
AncT.asymmetry(0)
AncT.dac_flux_coefficient(0.0014832606276941286)
AncT.dac_sweet_spot(-80.843401134877467)
AncT.f_max(5.942865842632016e9)
AncT.f_qubit_calc('flux')

AncB.E_c(0.28e9)
AncB.asymmetry(0)
AncB.dac_flux_coefficient(0.0020108167368328178)
AncB.dac_sweet_spot(46.64580507835808)
AncB.f_max(6.3772306731019359e9)
AncB.f_qubit_calc('flux')

DataT.E_c(0.28e9)
DataT.asymmetry(0)
DataT.dac_flux_coefficient(0.0016802077647335939)
DataT.dac_sweet_spot(-59.871260477923215)
DataT.f_max(5.6884932787721443e9)
DataT.f_qubit_calc('flux')


DataM.E_c(0.28e9)
DataM.asymmetry(0)
DataM.dac_flux_coefficient(0.0013648395455073477)
DataM.dac_sweet_spot(23.632250360310309)
DataM.f_max(6.1091409419040268e9)
DataM.f_qubit_calc('flux')

DataB.E_c(0.28e9)
DataB.asymmetry(0)
DataB.dac_flux_coefficient(0.00076044591994623627)
DataB.dac_sweet_spot(89.794843711783415)
DataB.f_max(6.7145280717783091e9)
DataB.f_qubit_calc('flux')


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
mqs.station=station


t1 = time.time()

# manually setting the clock, to be done automatically
AWG.clock_freq(1e9)


print('Ran initialization in %.2fs' % (t1-t0))


def all_sources_off():
    LO.off()
    RF.off()
    Spec_source.off()
    Qubit_LO.off()
    TWPA_Pump.off()


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
                                                       RO_amp=qubit.RO_amp(), RO_pulse_length=qubit.RO_pulse_length(),
                                                       acquisition_delay=270e-9)
        qubit.RO_pulse_type('MW_IQmod_pulse_UHFQC')
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


# preparing UHFQC readout with IQ mod pulses

list_qubits = [DataT, AncT, DataM, AncB,  DataB]
for qubit in list_qubits:
    qubit.RO_pulse_delay(20e-9)
    # qubit.RO_acq_averages(2**13)

q1, q0, q3, q2, q4 = AncT, DataT, AncB, DataM, DataB

# switch_to_pulsed_RO_CBox(AncT)
# switch_to_pulsed_RO_CBox(DataT)












#######################################################
# These settings don't get automotically restored upon init
#######################################################

k0.channel(4)
k0.kernel_dir_path(
    r'D:\GitHubRepos\iPython-Notebooks\Experiments\1607_Qcodes_5qubit\kernels')
k0.kernel_list(['precompiled_RT_20161206.txt'])

k1.channel(3)
k1.kernel_dir_path(
    r'D:\GitHubRepos\iPython-Notebooks\Experiments\1607_Qcodes_5qubit\kernels')
k1.kernel_list(['precompiled_AncT_RT_20161203.txt',
                'kernel_fridge_lowpass_20161024_1.00.txt',
                'kernel_skineffect_0.7.txt',
                'kernel_fridge_slow1_20161203_15_-0.013.txt'])

k1.bounce_tau_1(16)
k1.bounce_amp_1(-0.03)

k1.bounce_tau_2(1)
k1.bounce_amp_2(-0.04)

dist_dict = {'ch_list': ['ch4', 'ch3'],
             'ch4': k0.kernel(),
             'ch3': k1.kernel()}



def reload_mod_stuff():
    from pycqed.measurement.waveform_control import pulse_library as pl
    reload(pl)
    from pycqed.measurement.waveform_control import pulsar as ps
    reload(ps)
    # The AWG sequencer
    qc.station.pulsar = ps.Pulsar()
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
    fsqs.station=station
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

##
AncT.add_operation('CZ')
# AncT.add_operation('CZ_phase_corr') # to be added as separate later
AncT.add_pulse_parameter('CZ', 'CZ_pulse_amp', 'amplitude', initial_value=.5)
AncT.add_pulse_parameter('CZ', 'fluxing_operation_type', 'operation_type',
                         initial_value='Flux', vals=vals.Strings())
AncT.add_pulse_parameter('CZ', 'CZ_channel_amp', 'channel_amplitude',
                         initial_value=2.)
AncT.link_param_to_operation('CZ', 'fluxing_channel', 'channel')
AncT.link_param_to_operation('CZ', 'E_c', 'E_c')
AncT.add_pulse_parameter('CZ', 'CZ_pulse_type', 'pulse_type',
                         initial_value='MartinisFluxPulse', vals=vals.Strings())
AncT.add_pulse_parameter('CZ', 'CZ_dac_flux_coeff', 'dac_flux_coefficient',
                         initial_value=1.358)
AncT.add_pulse_parameter('CZ', 'CZ_dead_time', 'dead_time',
                         initial_value=3e-6)
AncT.link_param_to_operation('CZ', 'f_qubit', 'f_01_max')
AncT.add_pulse_parameter('CZ', 'CZ_bus', 'f_bus', 4.8e9)
AncT.add_pulse_parameter('CZ', 'CZ_length', 'length', 40e-9)
# AncT.link_param_to_operation('CZ', 'CZ_length', 'flux_pulse_length')

AncT.add_pulse_parameter('CZ', 'g2', 'g2', 33.3e6)
AncT.add_pulse_parameter('CZ', 'CZ_lambda_coeffs', 'lambda_coeffs',
                         np.array([1, 0, 0]),
                         vals=vals.Arrays())
AncT.link_param_to_operation('CZ', 'mw_to_flux_delay', 'mw_to_flux_delay')#, 0)


AncT.add_pulse_parameter('CZ', 'CZ_pulse_delay',
                         'pulse_delay', 0e-9)
AncT.add_pulse_parameter('CZ', 'CZ_refpoint',
                         'refpoint', 'end', vals=vals.Strings())

# AncT.add_pulse_parameter('CZ', 'CZ_square_pulse_buffer',
#                          'square_pulse_buffer', 100e-9)
# AncT.add_pulse_parameter('CZ', 'CZ_square_pulse_length',
#                          'square_pulse_length', 40e-9)
AncT.add_pulse_parameter('CZ', 'CZ_theta', 'theta_f', np.pi/2)


AncT.add_operation('CZ_corr')
AncT.link_param_to_operation('CZ_corr', 'fluxing_operation_type', 'operation_type')
AncT.link_param_to_operation('CZ_corr', 'fluxing_channel', 'channel')

AncT.link_param_to_operation('CZ_corr', 'CZ_refpoint', 'refpoint')

AncT.add_pulse_parameter('CZ_corr', 'CZ_corr_amp', 'amplitude', 0)
AncT.add_pulse_parameter('CZ_corr', 'CZ_corr_length',
                         'length', 10e-9)
#
AncT.add_pulse_parameter('CZ_corr', 'CZ_corr_pulse_type', 'pulse_type',
                         initial_value='SquarePulse',
                         vals=vals.Strings())
AncT.add_pulse_parameter('CZ_corr', 'CZ_corr_pulse_delay',
                         'pulse_delay', 0)

DataT.add_operation('SWAP')
DataT.add_pulse_parameter('SWAP', 'fluxing_operation_type', 'operation_type',
                          initial_value='Flux', vals=vals.Strings())
DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_amp', 'amplitude',
                          initial_value=0.5)
DataT.link_param_to_operation('SWAP', 'fluxing_channel', 'channel')

DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_type', 'pulse_type',
                          initial_value='SquarePulse', vals=vals.Strings())
DataT.add_pulse_parameter('SWAP', 'SWAP_refpoint',
                          'refpoint', 'end', vals=vals.Strings())
DataT.link_param_to_operation('SWAP', 'SWAP_amp', 'SWAP_amp')
DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_buffer',
                          'pulse_buffer', 0e-9)

DataT.link_param_to_operation('SWAP', 'SWAP_time', 'length')


DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_delay',
                          'pulse_delay', 0e-9)

DataT.add_operation('SWAP_corr')
DataT.add_pulse_parameter(
    'SWAP_corr', 'SWAP_corr_amp', 'amplitude', 0)
DataT.link_param_to_operation('SWAP_corr', 'fluxing_operation_type', 'operation_type')
DataT.link_param_to_operation('SWAP_corr', 'fluxing_channel', 'channel')
DataT.link_param_to_operation('SWAP_corr', 'SWAP_refpoint', 'refpoint')
DataT.add_pulse_parameter('SWAP_corr', 'SWAP_corr_length',
                          'length', 10e-9)
# DataT.link_param_to_operation('SWAP_corr', 'SWAP_corr_amp', 'amplitude')
# DataT.link_param_to_operation('SWAP_corr', 'SWAP_corr_length', 'square_pulse_length')
DataT.add_pulse_parameter('SWAP_corr', 'SWAP_corr_pulse_type', 'pulse_type',
                          initial_value='SquarePulse', vals=vals.Strings())
DataT.add_pulse_parameter('SWAP_corr', 'SWAP_corr_pulse_delay',
                          'pulse_delay', 0)

DataT.add_operation('rSWAP')
DataT.link_param_to_operation('rSWAP', 'fluxing_operation_type', 'operation_type')
DataT.link_param_to_operation('rSWAP', 'fluxing_channel', 'channel')
DataT.link_param_to_operation('rSWAP', 'SWAP_refpoint', 'refpoint')
DataT.link_param_to_operation('rSWAP', 'SWAP_pulse_type', 'pulse_type')
DataT.link_param_to_operation('rSWAP', 'SWAP_amp', 'SWAP_amp')


DataT.link_param_to_operation('rSWAP', 'SWAP_pulse_buffer', 'pulse_buffer')
DataT.link_param_to_operation('rSWAP', 'SWAP_pulse_delay', 'pulse_delay')

DataT.add_pulse_parameter('rSWAP', 'rSWAP_time', 'length',
                          initial_value=10e-9)
DataT.add_pulse_parameter('rSWAP', 'rSWAP_pulse_amp', 'amplitude',
                          initial_value=0.5)


gen.load_settings_onto_instrument(AncT)
gen.load_settings_onto_instrument(DataT)


DataT.RO_acq_weight_function_I(0)
DataT.RO_acq_weight_function_Q(0)
AncT.RO_acq_weight_function_I(1)
AncT.RO_acq_weight_function_Q(1)


# Reloading device type object
from pycqed.instrument_drivers.meta_instrument import device_object as do
reload(do)
# print(S5)
try:
    S5 = station.components['S5']
    S5.close()
    del station.components['S5']
except:
    pass
S5 = do.DeviceObject('S5')
station.add_component(S5)
S5.add_qubits([AncT, DataT])

S5.Buffer_Flux_Flux(10e-9)
S5.Buffer_Flux_MW(40e-9)
S5.Buffer_MW_MW(10e-9)
S5.Buffer_MW_Flux(10e-9)
station.sequencer_config = S5.get_operation_dict()['sequencer_config']


# Required for the Niels naming scheme
q0 = DataT
q1 = AncT


dist_dict = {'ch_list': ['ch4', 'ch3'],
             'ch4': k0.kernel(),
             'ch3': k1.kernel()}

DataT.dist_dict(dist_dict)
AncT.dist_dict(dist_dict)

AWG.ch4_amp(DataT.SWAP_amp())
AWG.ch3_amp(AncT.CZ_channel_amp())