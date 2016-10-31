"""
This scripts initializes the instruments and imports the modules
"""

UHFQC=False


# General imports

import time
import logging
t0 = time.time()  # to print how long init takes
from instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt

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
from pycqed.instrument_drivers.meta_instrument import Flux_Control as FluxCtrl
# Initializing instruments


station = qc.Station()
LO = rs.RohdeSchwarz_SGS100A(name='LO', address='TCPIP0::192.168.0.73', server_name=None)  #
station.add_component(LO)
RF = rs.RohdeSchwarz_SGS100A(name='RF', address='TCPIP0::192.168.0.74', server_name=None)  #
station.add_component(RF)
Spec_source = rs.RohdeSchwarz_SGS100A(name='Spec_source', address='TCPIP0::192.168.0.87', server_name=None)  #
station.add_component(Spec_source)
Qubit_LO = rs.RohdeSchwarz_SGS100A(name='Qubit_LO', address='TCPIP0::192.168.0.86', server_name=None)  #
station.add_component(Qubit_LO)
TWPA_Pump = rs.RohdeSchwarz_SGS100A(name='TWPA_Pump', address='TCPIP0::192.168.0.90', server_name=None)  #
station.add_component(TWPA_Pump)
CBox = qcb.QuTech_ControlBox('CBox', address='Com5', run_tests=False, server_name=None)
station.add_component(CBox)
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
                            address='GPIB0::6::INSTR', server_name=None)
station.add_component(AWG)
AWG.timeout(180)
AWG520 = tk520.Tektronix_AWG520('AWG520', address='GPIB0::17::INSTR',
                                server_name='')
station.add_component(AWG520)
IVVI = iv.IVVI('IVVI', address='COM4', numdacs=16, server_name=None)
station.add_component(IVVI)

if UHFQC:
    #Initializing UHFQC
    UHFQC_1 = ZI_UHFQC.UHFQC('UHFQC_1', device='dev2178', server_name=None)
    station.add_component(UHFQC_1)
else:
    UHFQC_1=None


Flux_Control = FluxCtrl.Flux_Control(name='FluxControl',IVVI=station.IVVI)
station.add_component(Flux_Control)

transfer_matrix_dec = np.array([[  4.70306717e-04,  -8.41312977e-05,   3.64442804e-05,  -1.00489353e-05,
   -2.36455362e-05],
 [ -6.70464355e-05,   6.39386703e-04,  -4.37263640e-05,  -2.01374983e-05,
    1.77516922e-05],
 [  7.69376917e-06,  -4.09893480e-05,   5.35184092e-04,  -2.36755094e-05,
   -5.34108608e-05],
 [  3.08518924e-05,   1.11315677e-05,   7.36191927e-05,   4.09078121e-04,
   -2.63031372e-05],
 [ -4.51217544e-05,  -1.35430841e-05,  -9.52349548e-05,  -4.18415379e-05,
    4.09962523e-04]])
invA = np.array([[  2.17320666e+03,2.79414032e+02,-1.16652799e+02,7.08814870e+01,1.02595827e+02],
                 [2.16689677e+02,1.59642752e+03,9.70544635e+01,8.55894771e+01,-3.84925724e+01],
                 [1.52695260e+00,1.25113953e+02,1.90389457e+03,1.42143094e+02,2.51834186e+02],
                 [-1.55226336e+02,-8.03197377e+01,-3.10695549e+02,2.43001891e+03,1.09956406e+02],
                 [2.30860259e+02,1.04357646e+02,4.00934628e+02,2.91661201e+02,2.51899161e+03]])
Flux_Control.transfer_matrix(transfer_matrix_dec)
Flux_Control.inv_transfer_matrix(invA)

Flux_Control.dac_mapping([1, 2, 3, 4, 5])

Flux_Control.flux_offsets(np.array([3.21499683e-02,-2.91992550e-02,2.88520021e-02,-2.26225717e-06,-9.35805778e-03]))




# ATT = Weinschel_8320_novisa.Weinschel_8320(name='ATT',address='192.168.0.54', server_name=None)
# station.add_component(ATT)
# Dux = qdux.QuTech_Duplexer('Dux', address='TCPIP0::192.168.0.101',
#                             server_name=None)
# SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None) #commented because of 8s load time

# Meta-instruments
HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, AWG=AWG, acquisition_instr=CBox.name,
                             server_name=None)
station.add_component(HS)
# LutMan = lm.QuTech_ControlBox_LookuptableManager('LutMan', CBox=CBox,
#                                                  server_name=None)
                                                 # server_name='metaLM')
MC = mc.MeasurementControl('MC')



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

DataT.E_c(0.28e9)
DataT.asymmetry(0)
DataT.dac_flux_coefficient(0.0016813942523375956)
DataT.dac_sweet_spot(-53.472554718672427)
DataT.f_max(5.688884012383026e9)
DataT.f_qubit_calc('flux')

AncB.E_c(0.28e9)
AncB.asymmetry(0)
AncB.dac_flux_coefficient(0.002028986705064149)
AncB.dac_sweet_spot(36.460579336820274)
AncB.f_max(6.381268822811037e9)
AncB.f_qubit_calc('flux')

AncT.E_c(0.28e9)
AncT.asymmetry(0)
AncT.dac_flux_coefficient(0.0015092699034525462)
AncT.dac_sweet_spot(-64.682660992718183)
AncT.f_max(5.9419418666592483e9)
AncT.f_qubit_calc('flux')

DataM.E_c(0.28e9)
DataM.asymmetry(0)
DataM.dac_flux_coefficient(0.0012685027014113798)
DataM.dac_sweet_spot(2.4196012752483966)
DataM.f_max(6.1113712558694182)
DataM.f_qubit_calc('flux')

DataB.E_c(0.28e9)
DataB.asymmetry(0)
DataB.dac_flux_coefficient(0.00094498809508039799)
DataB.dac_sweet_spot(31.549597601272581)
DataB.f_max(6.7138650690678894)
DataB.f_qubit_calc('flux')


MC.station = station
station.MC = MC
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
    Qubit_LO.off()
    TWPA_Pump.off()


def print_instr_params(instr):
    snapshot = instr.snapshot()
    for par in sorted(snapshot['parameters']):
        print('{}: {} {}'.format(snapshot['parameters'][par]['name'],
                                 snapshot['parameters'][par]['value'],
                                 snapshot['parameters'][par]['units']))


from scripts.Experiments.FiveQubits import common_functions as cfct
cfct.set_AWG_limits(station,1.7)


if UHFQC:
    def switch_to_pulsed_RO_CBox(qubit):
        UHFQC_1.awg_sequence_acquisition()
        qubit.RO_pulse_type('Gated_MW_RO_pulse')
        qubit.RO_acq_marker_delay(175e-9)
        qubit.acquisition_instr('CBox')
        qubit.RO_acq_marker_channel('ch3_marker1')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)

    def switch_to_pulsed_RO_UHFQC(qubit):
        UHFQC_1.awg_sequence_acquisition()
        qubit.RO_pulse_type('Gated_MW_RO_pulse')
        qubit.RO_acq_marker_delay(175e-9)
        qubit.acquisition_instr('UHFQC_1')
        qubit.RO_acq_marker_channel('ch3_marker2')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)


    def switch_to_IQ_mod_RO_UHFQC(qubit):
        UHFQC_1.awg_sequence_acquisition_and_pulse_SSB(f_RO_mod=qubit.f_RO_mod(),
                    RO_amp=qubit.RO_amp(), RO_pulse_length=qubit.RO_pulse_length(),
                    acquisition_delay=270e-9)
        qubit.RO_pulse_type('MW_IQmod_pulse_UHFQC')
        qubit.RO_acq_marker_delay(-100e-9)
        qubit.acquisition_instr('UHFQC_1')
        qubit.RO_acq_marker_channel('ch3_marker2')
        qubit.RO_I_channel('0')
        qubit.RO_Q_channel('1')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)
else:
    def switch_to_pulsed_RO_CBox(qubit):
        qubit.RO_pulse_type('Gated_MW_RO_pulse')
        qubit.RO_acq_marker_delay(175e-9)
        qubit.acquisition_instr('CBox')
        qubit.RO_acq_marker_channel('ch3_marker1')
        qubit.RO_acq_weight_function_I(0)
        qubit.RO_acq_weight_function_Q(1)


q0 = AncT
q1 = DataT

#preparing UHFQC readout with IQ mod pulses

switch_to_pulsed_RO_CBox(AncT)
switch_to_pulsed_RO_CBox(DataT)
