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
qc_config = {'datadir': r'D:\\Experiments\\1702_Starmon\\data',
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
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_swf_m
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as sq_m
from pycqed.measurement import multi_qubit_module as mq_mod
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

from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb
import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from pycqed.instrument_drivers.meta_instrument import heterodyne as hd
from pycqed.instrument_drivers.meta_instrument.qubit_objects import CC_transmon as qb
from pycqed.instrument_drivers.physical_instruments import QuTech_Duplexer as qdux
if UHFQC:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.physical_instruments import Weinschel_8320_novisa
from pycqed.instrument_drivers.meta_instrument import Flux_Control as fc
# for multiplexed readout

from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_swf_m
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as sq_m
import pycqed.scripts.personal_folders.Niels.two_qubit_readout_analysis as Niels


# import for ATS
import qcodes.instrument.parameter as parameter
import qcodes.instrument_drivers.AlazarTech.ATS9870 as ATSdriver
import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ats_contr

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


# AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
#                             address='TCPIP0::192.168.0.99::INSTR', server_name=None)
# station.add_component(AWG)
# AWG.timeout(180)  # timeout long for uploading wait.

IVVI = iv.IVVI('IVVI', address='COM8', numdacs=16, server_name=None)
station.add_component(IVVI)

MC = mc.MeasurementControl('MC')
station.add_component(MC)
# HS = None

# Initializing UHFQC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC

import pycqed.instrument_drivers.meta_instrument.UHFQC_LookuptableManager as lm_UHFQC
import pycqed.instrument_drivers.meta_instrument.UHFQC_LookuptableManagerManager as lmm_UHFQC

UHFQC_1 = ZI_UHFQC.UHFQC('UHFQC_1', device='dev2178', server_name=None)
station.add_component(UHFQC_1)


UHFQC_1.sigins_0_ac()
UHFQC_1.sigins_1_ac()
# preparing the lookuptables for readout
LutMan0 = lm_UHFQC.UHFQC_LookuptableManager('LutMan0', UHFQC=UHFQC_1,
                                            server_name=None)
station.add_component(LutMan0)
LutMan1 = lm_UHFQC.UHFQC_LookuptableManager('LutMan1', UHFQC=UHFQC_1)
station.add_component(LutMan1)
LutManMan = lmm_UHFQC.UHFQC_LookuptableManagerManager(
    'LutManMan', UHFQC=UHFQC_1)
station.add_component(LutManMan)
LutManMan.LutMans(
    [LutMan0.name, LutMan1.name])



from pycqed.instrument_drivers.physical_instruments import QuTech_AWG_Module as qwg
QWG = qwg.QuTech_AWG_Module(
                'QWG', address='192.168.0.10',
                port=5025, server_name=None)
station.add_component(QWG)

from pycqed.instrument_drivers.meta_instrument import QWG_LookuptableManager as qlm
QL_LutMan = qlm.QWG_LookuptableManager('QL_LutMan', QWG=QWG)
station.add_component(QL_LutMan)
QR_LutMan = qlm.QWG_LookuptableManager('QR_LutMan', QWG=QWG)
station.add_component(QR_LutMan)



CBox = qcb.QuTech_ControlBox_v3(
    'CBox', address='Com6', run_tests=False, server_name=None)
station.add_component(CBox)
# Initializaing ATS,
# ATSdriver.AlazarTech_ATS.find_boards()
# ATS = ATSdriver.AlazarTech_ATS9870(name='ATS', server_name=None)
# station.add_component(ATS)

# # Configure all settings in the ATS
# ATS.config(clock_source='INTERNAL_CLOCK',
#            sample_rate=100000000,
#            clock_edge='CLOCK_EDGE_RISING',
#            decimation=0,
#            coupling=['AC', 'AC'],
#            channel_range=[2., 2.],
#            impedance=[50, 50],
#            bwlimit=['DISABLED', 'DISABLED'],
#            trigger_operation='TRIG_ENGINE_OP_J',
#            trigger_engine1='TRIG_ENGINE_J',
#            trigger_source1='EXTERNAL',
#            trigger_slope1='TRIG_SLOPE_POSITIVE',
#            trigger_level1=128,
#            trigger_engine2='TRIG_ENGINE_K',
#            trigger_source2='DISABLE',
#            trigger_slope2='TRIG_SLOPE_POSITIVE',
#            trigger_level2=128,
#            external_trigger_coupling='AC',
#            external_trigger_range='ETR_5V',
#            trigger_delay=0,
#            timeout_ticks=0)

# # demodulation frequcnye is first set to 10 MHz
# ATS_controller = ats_contr.Demodulation_AcquisitionController(name='ATS_controller',
#                                                               demodulation_frequency=10e6,
#                                                               alazar_name='ATS',
#                                                               server_name=None)
# station.add_component(ATS_controller)

# # configure the ATS controller
# ATS_controller.update_acquisitionkwargs(  # mode='NPT',
#     samples_per_record=64*1000,  # 4992,
#     records_per_buffer=8,  # 70, segmments
#     buffers_per_acquisition=8,
#     channel_selection='AB',
#     transfer_offset=0,
#     external_startcapture='ENABLED',
#     enable_record_headers='DISABLED',
#     alloc_buffers='DISABLED',
#     fifo_only_streaming='DISABLED',
#     interleave_samples='DISABLED',
#     get_processed_data='DISABLED',
#     allocated_buffers=100,
#     buffer_timeout=1000)

# Meta-instruments
HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, AWG=None,
                             acquisition_instr=UHFQC_1.name,
                             # acquisition_instr_controller=ATS_controller.name,
                             server_name=None)
station.add_component(HS)


QL = qb.QWG_driven_transmon('QL', LO, cw_source=QL_LO, td_source=QL_LO,
                                IVVI=IVVI, QWG=QWG, CBox=CBox,
                                RO_LutMan=LutMan0, Q_LutMan=QL_LutMan,
                                MC=MC)
station.add_component(QL)

QR = qb.QWG_driven_transmon('QR', LO, cw_source=QR_LO, td_source=QR_LO,
                                IVVI=IVVI, QWG=QWG, CBox=CBox,
                                RO_LutMan=LutMan1, Q_LutMan=QR_LutMan,
                                MC=MC)
station.add_component(QR)

# load settings onto qubits
gen.load_settings_onto_instrument(QL)
gen.load_settings_onto_instrument(QR)
# gen.load_settings_onto_instrument(HS)


# AncT.E_c(0.28e9)
# AncT.asymmetry(0)
# AncT.dac_flux_coefficient(0.0014832606276941286)
# AncT.dac_sweet_spot(-80.843401134877467)
# AncT.f_max(5.942865842632016e9)
# AncT.f_qubit_calc('flux')

# AncB.E_c(0.28e9)
# AncB.asymmetry(0)
# AncB.dac_flux_coefficient(0.0020108167368328178)
# AncB.dac_sweet_spot(46.64580507835808)
# AncB.f_max(6.3772306731019359e9)
# AncB.f_qubit_calc('flux')

MC.station = station
station.MC = MC
nested_MC = mc.MeasurementControl('nested_MC')
nested_MC.station = station


t1 = time.time()

# manually setting the clock, to be done automatically


print('Ran initialization in %.2fs' % (t1-t0))


def all_sources_off():
    LO.off()
    RF.off()
    QL_LO.off()
    QR_LO.off()

# if UHFQC:
#     def switch_to_pulsed_RO_CBox(qubit):
#         UHFQC_1.awg_sequence_acquisition()
#         qubit.RO_pulse_type('Gated_MW_RO_pulse')
#         qubit.RO_acq_marker_delay(155e-9)
#         qubit.acquisition_instr('CBox')
#         qubit.RO_acq_marker_channel('ch3_marker1')
#         qubit.RO_acq_weight_function_I(0)
#         qubit.RO_acq_weight_function_Q(1)

#     def switch_to_pulsed_RO_UHFQC(qubit):
#         UHFQC_1.awg_sequence_acquisition()
#         qubit.RO_pulse_type('Gated_MW_RO_pulse')
#         qubit.RO_acq_marker_delay(75e-9)
#         qubit.acquisition_instr('UHFQC_1')
#         qubit.RO_acq_marker_channel('ch3_marker2')
#         qubit.RO_acq_weight_function_I(0)
#         qubit.RO_acq_weight_function_Q(1)

#     def switch_to_IQ_mod_RO_UHFQC(qubit):
#         UHFQC_1.awg_sequence_acquisition_and_pulse_SSB(f_RO_mod=qubit.f_RO_mod(),
#                                                        RO_amp=qubit.RO_amp(), RO_pulse_length=qubit.RO_pulse_length(),
#                                                        acquisition_delay=270e-9)
#         qubit.RO_pulse_type('MW_IQmod_pulse_UHFQC')
#         qubit.RO_acq_marker_delay(-165e-9)
#         qubit.acquisition_instr('UHFQC_1')
#         qubit.RO_acq_marker_channel('ch3_marker2')
#         qubit.RO_I_channel('0')
#         qubit.RO_Q_channel('1')
#         qubit.RO_acq_weight_function_I(0)
#         qubit.RO_acq_weight_function_Q(1)
# else:
#     def switch_to_pulsed_RO_CBox(qubit):
#         qubit.RO_pulse_type('Gated_MW_RO_pulse')
#         qubit.RO_acq_marker_delay(155e-9)
#         qubit.acquisition_instr('CBox')
#         qubit.RO_acq_marker_channel('ch3_marker1')
#         qubit.RO_acq_weight_function_I(0)
#         qubit.RO_acq_weight_function_Q(1)


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
    fsqs.station = station
    reload(det)
    reload(ca)
reload_mod_stuff()
