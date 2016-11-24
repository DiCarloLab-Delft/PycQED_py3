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
import qcodes as qc
# Globally defined config

qc_config = {'datadir': r'D:\Experiments\\1611_intel_demo\data',
             'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}

# makes sure logging messages show up in the notebook
root = logging.getLogger()
root.addHandler(logging.StreamHandler())


# General PycQED modules
from pycqed.measurement import measurement_control as mc
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det

# QASM modules
from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta
from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sq_qasm
from pycqed.instrument_drivers.physical_instruments._controlbox \
    import Assembler

# Importing instruments
from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb
from pycqed.instrument_drivers.meta_instrument import CBox_LookuptableManager as cbl


# General PycQED modules
from pycqed.measurement import measurement_control as mc
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import detector_functions as det
from pycqed.measurement import composite_detector_functions as cdet
from pycqed.measurement import calibration_toolbox as cal_tools
from pycqed.measurement import CBox_sweep_functions as cb_swf
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools


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


import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from pycqed.instrument_drivers.meta_instrument import heterodyne as hd

from pycqed.instrument_drivers.physical_instruments import Weinschel_8320_novisa
from pycqed.instrument_drivers.meta_instrument import Flux_Control as FluxCtrl


station = qc.Station()


MC = mc.MeasurementControl('MC')
MC.station = station
station.MC = MC

# Initializing instruments

# Microwave sources
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
TWPA_Pump = rs.RohdeSchwarz_SGS100A(
    name='TWPA_Pump', address='TCPIP0::192.168.0.90', server_name=None)  #
station.add_component(TWPA_Pump)

# step attenuator
ATT = Weinschel_8320_novisa.Weinschel_8320(
    name='ATT', address='192.168.0.54', server_name=None)
station.add_component(ATT)

# AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
#                             address='GPIB0::6::INSTR', server_name=None)
# station.add_component(AWG)
# Current sources
IVVI = iv.IVVI('IVVI', address='COM4', numdacs=16, server_name=None)
station.add_component(IVVI)


Flux_Control = FluxCtrl.Flux_Control(
    name='FluxControl', IVVI=station.IVVI, server_name=None)
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


sweet_spots_mv = [85.265, -49.643, 60.893, -13.037, -49.570]
offsets = np.dot(Flux_Control.transfer_matrix(), sweet_spots_mv)
Flux_Control.flux_offsets(offsets)


# CBox
CBox = qcb.QuTech_ControlBox_v3(
    'CBox', address='Com6', run_tests=False, server_name=None)
station.add_component(CBox)
LutMan = cbl.QuTech_ControlBox_LookuptableManager(
    'Lutman', CBox, server_name=None)
station.add_component(LutMan)


# Dux = qdux.QuTech_Duplexer('Dux', address='TCPIP0::192.168.0.101',
#                             server_name=None)
# SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None)
# #commented because of 8s load time

# Meta-instruments
print('starting meta instruments')
# HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, AWG=AWG, acquisition_instr=CBox.name,
#                              server_name=None)
# station.add_component(HS)
print('starting qubit objects')
# AncB = qbt.Tektronix_driven_transmon('AncB', LO=LO, cw_source=Spec_source,
#                                      td_source=Qubit_LO,
#                                      IVVI=IVVI, rf_RO_source=RF,
#                                      AWG=AWG,
#                                      heterodyne_instr=HS,
#                                      FluxCtrl=Flux_Control,
#                                      MC=MC,
#                                      server_name=None)
# station.add_component(AncB)
from pycqed.instrument_drivers.meta_instrument.qubit_objects import CBox_v3_driven_transmon as cq

AncT_CB = cq.CBox_v3_driven_transmon('AncT_CB', LO=LO, cw_source=Spec_source,
                                     td_source=Qubit_LO, IVVI=IVVI,
                                     LutMan=LutMan, CBox=CBox, MC=MC)
# AncT = qbt.Tektronix_driven_transmon('AncT', LO=LO, cw_source=Spec_source,
#                                      td_source=Qubit_LO,
#                                      IVVI=IVVI, rf_RO_source=RF,
#                                      AWG=AWG,
#                                      heterodyne_instr=HS,
#                                      FluxCtrl=Flux_Control,
#                                      MC=MC,
#                                      server_name=None)
# station.add_component(AncT)
# DataB = qbt.Tektronix_driven_transmon('DataB', LO=LO, cw_source=Spec_source,
#                                       td_source=Qubit_LO,
#                                       IVVI=IVVI, rf_RO_source=RF,
#                                       AWG=AWG,
#                                       heterodyne_instr=HS,
#                                       FluxCtrl=Flux_Control,
#                                       MC=MC,
#                                       server_name=None)
# station.add_component(DataB)
# DataM = qbt.Tektronix_driven_transmon('DataM', LO=LO, cw_source=Spec_source,
#                                       td_source=Qubit_LO,
#                                       IVVI=IVVI, rf_RO_source=RF,
#                                       AWG=AWG,
#                                       heterodyne_instr=HS,
#                                       FluxCtrl=Flux_Control,
#                                       MC=MC,
#                                       server_name=None)
# station.add_component(DataM)
# DataT = qbt.Tektronix_driven_transmon('DataT', LO=LO, cw_source=Spec_source,
#                                       td_source=Qubit_LO,
#                                       IVVI=IVVI, rf_RO_source=RF,
#                                       AWG=AWG,
#                                       heterodyne_instr=HS,
#                                       FluxCtrl=Flux_Control,
#                                       MC=MC,
#                                       server_name=None)
# station.add_component(DataT)

# load settings onto qubits
# gen.load_settings_onto_instrument(AncT)  # , timestamp='20161111_165442')
# gen.load_settings_onto_instrument(HS)

# AncT.E_c(0.28e9)
# AncT.asymmetry(0)
# AncT.dac_flux_coefficient(0.0014870990568855082)
# AncT.dac_sweet_spot(-85.264803738253249)
# AncT.f_max(5.9425723344778003e9)
# AncT.f_qubit_calc('flux')


# nested_MC = mc.MeasurementControl('nested_MC')
# nested_MC.station = station

# The AWG sequencer
# station.pulsar = ps.Pulsar()
# station.pulsar.AWG = station.components['AWG']
# marker1highs = [2, 2, 2.7, 2]
# for i in range(4):
#     # Note that these are default parameters and should be kept so.
#     # the channel offset is set in the AWG itself. For now the amplitude is
#     # hardcoded. You can set it by hand but this will make the value in the
#     # sequencer different.
#     station.pulsar.define_channel(id='ch{}'.format(i+1),
#                                   name='ch{}'.format(i+1), type='analog',
#                                   # max safe IQ voltage
#                                   high=.7, low=-.7,
#                                   offset=0.0, delay=0, active=True)
#     station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
#                                   name='ch{}_marker1'.format(i+1),
#                                   type='marker',
#                                   high=marker1highs[i], low=0, offset=0.,
#                                   delay=0, active=True)
#     station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
#                                   name='ch{}_marker2'.format(i+1),
#                                   type='marker',
#                                   high=2.0, low=0, offset=0.,
#                                   delay=0, active=True)
# # to make the pulsar available to the standard awg seqs
# st_seqs.station = station
# sq.station = station
# awg_swf.fsqs.station = station
# cal_elts.station = station

t1 = time.time()

# manually setting the clock, to be done automatically
# AWG.clock_freq(1e9)


print('Ran initialization in %.2fs' % (t1-t0))

#########################################
# Demo parameters
#########################################

from pycqed.measurement.waveform_control_CC import single_qubit_qasm_seqs as sqs
import pycqed.measurement.CBox_sweep_functions as cbs
from pycqed.scripts.Experiments.intel_demo import qasm_helpers as qh

# AncT.f_modulation(-20e-6)

ATT.attenuation(10)

qubit_name = 'AncT_CB'

modulation_frequency = -20e6
try:
    CBox.integration_length(120)  # samples
except:
    logging.warning('checksum error caught?')
    CBox.integration_length(120)
CBox.AWG0_dac0_offset(-11.10517597)
CBox.AWG0_dac1_offset(-21.095)
CBox.upload_standard_weights(modulation_frequency)

LutMan.Q_modulation(modulation_frequency)
LutMan.Q_gauss_width(10e-9)
LutMan.Q_amp180(0.249955)
LutMan.Q_amp90(LutMan.Q_amp180()/2)

LutMan.lut_mapping(['I', 'X180', 'Y180', 'X90', 'Y90', 'mX90', 'mY90',
                    'M_square'])
LutMan.M_modulation(modulation_frequency)
LutMan.M_amp(0.2)
LutMan.M_length(600e-9)

LutMan.load_pulses_onto_AWG_lookuptable(0)


AncT_CB.f_RO()
AncT_CB.f_qubit(5941438225.31)
AncT_CB.f_pulse_mod(-20e6)
AncT_CB.f_RO(7094199808.0)
AncT_CB.f_RO_mod(-20e6)

AncT_CB.f_pulse_mod(modulation_frequency)
AncT_CB.gauss_width(10e-9)
AncT_CB.amp180(0.249955)
AncT_CB.amp90(LutMan.Q_amp180()/2)

AncT_CB.RO_pulse_delay(60*5e-9)
AncT_CB.RO_pulse_length(50*5e-9)
AncT_CB.RO_amp(0.2)
AncT_CB.td_source_pow(16)


op_dict = AncT_CB.get_operation_dict()
# op_dict = qh.create_CBox_op_dict(qubit_name, pulse_length=10, RO_length=50,
#                                  RO_delay=60, modulated_RO=True)


##############
# Defining useful function
###############
def all_sources_off():
    LO.off()
    RF.off()
    Spec_source.off()
    Qubit_LO.off()
    TWPA_Pump.off()


def print_instr_params(instr):
    snapshot = instr.snapshot()
    print('{0:23} {1} \t ({2})'.format('\t parameter ', 'value', 'units'))
    print('-'*80)
    for par in sorted(snapshot['parameters']):
        print('{0:25}: \t{1}\t ({2})'.format(
            snapshot['parameters'][par]['name'],
            snapshot['parameters'][par]['value'],
            snapshot['parameters'][par]['units']))


# All the standard sequences

def prepare_rabi_seq_CC(qubit_name, op_dict):
    single_pulse_elt = sqs.single_elt_on('AncT_CB')
    single_pulse_asm = qta.qasm_to_asm(single_pulse_elt.name, op_dict)
    asm_file = single_pulse_asm
    CBox.load_instructions(asm_file.name)


def measure_rabi(qubit_name, op_dict):
    prepare_rabi_seq_CC(qubit_name=qubit_name, op_dict=op_dict)
    amp_swf = cbs.Lutman_par_with_reload_single_pulse(
        LutMan=LutMan, parameter=LutMan.Q_amp180,
        pulse_names=['X180'], awg_nrs=[0])
    d = qh.CBox_single_integration_average_det_CC(
        CBox, nr_averages=128, seg_per_point=1)

    MC.set_sweep_function(amp_swf)
    MC.set_sweep_points(np.linspace(-.5, .5, 41))
    MC.set_detector_function(d)
    MC.run('ASM_rabi')
    a = ma.Rabi_Analysis(close_fig=False, label='ASM_rabi')
    return a


def measure_T1():
    MC.soft_avg(5)
    times = np.arange(50e-9, 80e-6, 3e-6)
    T1 = sq_qasm.T1(qubit_name, times=times)
    s = qh.QASM_Sweep(T1.name, CBox, op_dict)
    d = qh.CBox_integrated_average_detector_CC(CBox, nr_averages=512)
    MC.set_sweep_function(s)
    MC.set_sweep_points(times)
    MC.set_detector_function(d)
    MC.run('T1_qasm')
    ma.T1_Analysis(close_fig=True)


def measure_ramsey():
    MC.soft_avg(5)
    times = np.arange(50e-9, 50e-6, 1e-6)
    Ramsey = sq_qasm.Ramsey(qubit_name, times=times, artificial_detuning=None)
    s = qh.QASM_Sweep(Ramsey.name, CBox, op_dict)
    d = qh.CBox_integrated_average_detector_CC(CBox, nr_averages=128)
    MC.set_sweep_function(s)
    MC.set_sweep_points(times)
    MC.set_detector_function(d)
    MC.run('Ramsey_qasm')
    ma.Ramsey_Analysis(close_fig=True, label='Ramsey')


def measure_echo():
    MC.soft_avg(5)

    times = np.arange(50e-9, 80e-6, 3e-6)
    Ecjp = sq_qasm.echo(qubit_name, times=times, artificial_detuning=None)
    s = qh.QASM_Sweep(Ecjp.name, CBox, op_dict)
    d = qh.CBox_integrated_average_detector_CC(CBox, nr_averages=128)
    MC.set_sweep_function(s)
    MC.set_sweep_points(times)
    MC.set_detector_function(d)
    MC.run('Echo')
    ma.Echo_analysis(close_fig=True)


def measure_AllXY():
    MC.soft_avg(5)
    reload(qh)
    AllXY = sq_qasm.AllXY(qubit_name, double_points=True)
    s = qh.QASM_Sweep(AllXY.name, CBox, op_dict)
    d = qh.CBox_integrated_average_detector_CC(CBox, nr_averages=256)
    MC.set_sweep_function(s)
    MC.set_sweep_points(np.arange(42))
    MC.set_detector_function(d)
    MC.run('AllXY_qasm')

    ma.AllXY_Analysis(close_fig=False)


def measure_ssro():
    MC.soft_avg(1)
    reload(qh)
    CBox.log_length(8000)
    off_on = sq_qasm.off_on(qubit_name)
    s = qh.QASM_Sweep(off_on.name, CBox, op_dict)
    d = qh.CBox_integration_logging_det_CC(CBox, )
    MC.set_sweep_function(s)
    MC.set_sweep_points(np.arange(8000))
    MC.set_detector_function(d)
    dat = MC.run('off_on')
    ma.SSRO_Analysis(close_fig=False)


def prepare_motzoi_seq_CC(qubit_name, op_dict):
    motzoi_elt = sqs.two_elt_MotzoiXY(qubit_name)
    single_pulse_asm = qta.qasm_to_asm(motzoi_elt.name, op_dict)
    asm_file = single_pulse_asm
    CBox.load_instructions(asm_file.name)


def measure_motzoi():
    prepare_motzoi_seq_CC(qubit_name, op_dict)
    MC.soft_avg(5)
    motzois = np.repeat(np.linspace(-.4, .2, 41), 2)
    motzoi_swf = cbs.Lutman_par_with_reload_single_pulse(LutMan=LutMan, parameter=LutMan.Q_motzoi_parameter,
                                                         pulse_names=['X180', 'X90', 'Y180', 'Y90'], awg_nrs=[0])
    d = qh.CBox_single_integration_average_det_CC(
        CBox, seg_per_point=2, nr_averages=512)

    MC.set_sweep_function(motzoi_swf)
    MC.set_sweep_points(motzois)
    MC.set_detector_function(d)
    MC.run('ASM_motzoi_single')
    a = ma.MeasurementAnalysis(close_fig=False, label='ASM_motzoi_single')


AncT_CB.prepare_for_timedomain()
