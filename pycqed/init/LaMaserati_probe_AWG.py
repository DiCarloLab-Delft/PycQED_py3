"""
This scripts initializes the instruments and imports the modules
"""


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
from qcodes.utils import validators as vals
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

from pycqed.instrument_drivers.meta_instrument import heterodyne as hd
import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as lm

from pycqed.instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as qb

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

from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
# Initializing instruments


station = qc.Station()
qc.station = station  # makes it easily findable from inside files
LO = rs.RohdeSchwarz_SGS100A(name='LO', address='TCPIP0::192.168.0.73')  #
station.add_component(LO)
RF = rs.RohdeSchwarz_SGS100A(name='RF', address='TCPIP0::192.168.0.74')  #
station.add_component(RF)
S86 = rs.RohdeSchwarz_SGS100A(name='S86', address='TCPIP0::192.168.0.86')  #
station.add_component(S86)
S87 = rs.RohdeSchwarz_SGS100A(name='S87', address='TCPIP0::192.168.0.87')  #
station.add_component(S87)
AWG = tek.Tektronix_AWG5014(name='AWG', timeout=2,
                            address='TCPIP0::192.168.0.9::INSTR')

station.add_component(AWG)
AWG.timeout(15)  # timeout long for uploading wait.

IVVI = iv.IVVI('IVVI', address='COM8', numdacs=16, safe_version=False)
station.add_component(IVVI)


from pycqed.instrument_drivers.meta_instrument.Flux_Control import Flux_Control
FC = Flux_Control('FC', 2, IVVI.name)
station.add_component(FC)
# gen.load_settings_onto_instrument(FC)
dac_offsets = np.array([75, -16.59])

dac_mapping = np.array([2, 1])
A = np.array([[-1.71408478e-13,   1.00000000e+00],
              [1.00000000e+00,  -1.32867524e-14]])
FC.transfer_matrix(A)
FC.dac_mapping(dac_mapping)
FC.dac_offsets(dac_offsets)


# # Initializing UHFQC
UHFQC2178 = ZI_UHFQC.UHFQC('UHFQC2178', device='dev2178', server_name=None)
station.add_component(UHFQC2178)
UHFQC2178.sigins_0_ac()
UHFQC2178.sigins_1_ac()
# preparing the lookuptables for readout
LutMan0 = lm_UHFQC.UHFQC_LookuptableManager('LutMan0', UHFQC=UHFQC2178,
                                            server_name=None)
station.add_component(LutMan0)
LutMan1 = lm_UHFQC.UHFQC_LookuptableManager('LutMan1', UHFQC=UHFQC2178,
                                            server_name=None)
station.add_component(LutMan1)

LutManMan = lmm_UHFQC.UHFQC_LookuptableManagerManager('LutManMan', UHFQC=UHFQC2178,
                                                      server_name=None)
station.add_component(LutManMan)
LutManMan.LutMans(
    [LutMan0.name, LutMan1.name])


# SH = sh.SignalHound_USB_SA124B('Signal hound')

# Meta-instruments
HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, AWG=AWG,
                             acquisition_instr=UHFQC2178.name)
station.add_component(HS)

MC = mc.MeasurementControl('MC')
station.add_component(MC)
# HS = None

#####################################
# Qubit objects
#####################################

import pycqed.instrument_drivers.meta_instrument.kernel_object as k_obj
k0 = k_obj.Distortion(name='k0')
station.add_component(k0)
k0.channel(2)
k0.kernel_dir(
    r'D:\Experiments\1702_Starmon\kernels')
k0.kernel_list(['id_kernel.txt'])



from pycqed.instrument_drivers.meta_instrument import device_object as do
Starmon = do.DeviceObject('Starmon')
station.add_component(Starmon)


# from pycqed.instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt
# QL = dt.Duplexer_tek_transmon('QL')
# station.add_component(QL)
# gen.load_settings_onto_instrument(QL)


from pycqed.instrument_drivers.meta_instrument.qubit_objects import CC_transmon as qb
QL = qb.CBox_v3_driven_transmon('QL')
station.add_component(QL)


import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt

QR = qb.CBox_v3_driven_transmon('QR')
station.add_component(QR)

# QR = qbt.Tektronix_driven_transmon('QR')
# station.add_component(QR)

# QR.add_operation('SWAP')
# QR.add_pulse_parameter('SWAP', 'fluxing_operation_type', 'operation_type',
#                        initial_value='Flux', vals=vals.Strings())
# QR.add_pulse_parameter('SWAP', 'SWAP_pulse_amp', 'amplitude',
#                        initial_value=0.5)
# QR.link_param_to_operation('SWAP', 'fluxing_channel', 'channel')

# QR.add_pulse_parameter('SWAP', 'SWAP_pulse_type', 'pulse_type',
#                        initial_value='SquareFluxPulse', vals=vals.Strings())
# QR.add_pulse_parameter('SWAP', 'SWAP_refpoint',
#                        'refpoint', 'end', vals=vals.Strings())
# QR.link_param_to_operation('SWAP', 'SWAP_amp', 'SWAP_amp')
# QR.add_pulse_parameter('SWAP', 'SWAP_pulse_buffer',
#                        'pulse_buffer', 0e-9)

# QR.link_param_to_operation('SWAP', 'SWAP_time', 'square_pulse_length')


# QR.add_pulse_parameter('SWAP', 'SWAP_pulse_delay',
#                        'pulse_delay', 0e-9)

# gen.load_settings_onto_instrument(QR)
# k0.kernel_list(['RT_Compiled_170308.txt'])
# QR.dist_dict({'ch_list': ['ch2'], 'ch2': k0.kernel()})

# k0.decay_amp_1(-1.)
# k0.decay_tau_1(1.)
# QR.dist_dict({'ch_list': ['ch2'], 'ch2': k0.kernel()})

# QR.dac_channel(1)


Starmon.add_qubits([QL, QR])
gen.load_settings_onto_instrument(Starmon)

gen.load_settings_onto_instrument(QL)
# station.sequencer_config = Starmon.get_operation_dict()['sequencer_config']


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
    S86.off()
    S87.off()

    def switch_to_pulsed_RO_UHFQC(qubit):
        UHFQC2178.awg_sequence_acquisition()
        qubit.RO_pulse_type('Gated_MW_RO_pulse')
        qubit.RO_acq_marker_delay(75e-9)
        qubit.acquisition_instr('UHFQC2178')
        qubit.RO_acq_marker_channel('ch3_marker2')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)

    def switch_to_IQ_mod_RO_UHFQC(qubit):
        UHFQC2178.awg_sequence_acquisition_and_pulse_SSB(
            f_RO_mod=qubit.f_RO_mod(),
            RO_amp=qubit.RO_amp(),
            RO_pulse_length=qubit.RO_pulse_length(),
            acquisition_delay=270e-9)
        # changed to satisfy validator
        qubit.RO_pulse_type('MW_IQmod_pulse_UHFQC')
        qubit.RO_acq_marker_delay(-165e-9)
        qubit.acquisition_instr('UHFQC_1')
        qubit.RO_acq_marker_channel('ch3_marker2')
        qubit.RO_I_channel('0')
        qubit.RO_Q_channel('1')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)

from pycqed.instrument_drivers.physical_instruments.Fridge_monitor import Fridge_Monitor
Maserati_fridge_mon = Fridge_Monitor('Maserati_fridge_mon', 'LaMaserati')
station.add_component(Maserati_fridge_mon)


from pycqed.instrument_drivers.virtual_instruments import instrument_monitor as im
IM = im.InstrumentMonitor('IM', station)
station.add_component(IM)
MC.instrument_monitor(IM.name)
nested_MC.instrument_monitor(IM.name)


###############################################################################
###############################################################################
###############################################################################

# from importlib import reload
# import qcodes as qc
# import logging
# station = qc.station
# from pycqed.utilities import general as gen


# def reload_CC_qubit(qubit):
#     reload(qb)
#     try:
#         qubit_name = qubit.name
#         qubit.close()
#         del station.components[qubit_name]

#     except Exception as e:
#         logging.warning(e)
#     qubit = qb.CBox_v3_driven_transmon(qubit_name)
#     station.add_component(qubit)
#     gen.load_settings_onto_instrument(qubit)
#     return qubit


# # from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBoxdriver as qcb
# # reload(qcb)
from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb
# # reload(qcb)
# # CBox.close()
# # del station.components['CBox']
CBox = qcb.QuTech_ControlBox_v3(
    'CBox', address='Com6', run_tests=False, server_name=None)
station.add_component(CBox)

import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as cbl
CBox_LutMan = cbl.ControlBox_LookuptableManager('CBox_LutMan')
CBox_LutMan.CBox(CBox.name)
station.add_component(CBox_LutMan)

# UHFQC_1.timeout(2)
from pycqed.instrument_drivers.physical_instruments import QuTech_AWG_Module as qwg
QWG = qwg.QuTech_AWG_Module('QWG', address='192.168.0.10', port=5025)
station.add_component(QWG)

from pycqed.instrument_drivers.physical_instruments import QuTech_Duplexer as qdux
VSM = qdux.QuTech_Duplexer('VSM', address='TCPIP0::192.168.0.100')
station.add_component(VSM)
