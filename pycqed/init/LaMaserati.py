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
qc_config = {'datadir': r'D:\\Experiments\\1702_S7m_W16_NW22\\Data',
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
from pycqed.measurement.optimization import nelder_mead
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_swf_m
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as sq_m

from pycqed.instrument_drivers.physical_instruments import Fridge_monitor as fm
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

from qcodes.instrument_drivers.tektronix import AWG5014 as tek
# from qcodes.instrument_drivers.tektronix import AWG520 as tk520



import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from pycqed.instrument_drivers.meta_instrument import heterodyne as hd
import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as lm


#import for UHFKLI for UHFLI
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
# import for ATS
import qcodes.instrument.parameter as parameter
import qcodes.instrument_drivers.AlazarTech.ATS9870 as ATSdriver
import qcodes.instrument_drivers.AlazarTech.ATS_acquisition_controllers as ats_contr
from pycqed.instrument_drivers.meta_instrument import Flux_Control as FluxCtrl


t0 = time.time()  # to print how long init takes
############################
# Initializing instruments #
############################
station = qc.Station()

Fridge_mon = fm.Fridge_Monitor('Fridge monitor', 'LaMaserati')
station.add_component(Fridge_mon)

###########
# Sources #
###########
LO = rs.RohdeSchwarz_SGS100A(name='LO', address='TCPIP0::192.168.0.71', server_name=None)  #
station.add_component(LO)
RF = rs.RohdeSchwarz_SGS100A(name='RF', address='TCPIP0::192.168.0.72', server_name=None)  #
station.add_component(RF)
cw_source = rs.RohdeSchwarz_SGS100A(name='cw_source', address='TCPIP0::192.168.0.85', server_name=None)  #
station.add_component(cw_source)
Qubit_LO = rs.RohdeSchwarz_SGS100A(name='Qubit_LO', address='TCPIP0::192.168.0.90', server_name=None)  #
station.add_component(Qubit_LO)


#Initializing UHFQC
UHFQC_1 = ZI_UHFQC.UHFQC('UHFQC_1', device='dev2209', server_name=None)
station.add_component(UHFQC_1)

#initializing AWG
AWG = tek.Tektronix_AWG5014(name='AWG',  timeout=2,
                            address='GPIB0::8::INSTR', server_name=None)
station.add_component(AWG)
AWG.timeout(180)  # timeout long for uploading wait.

#IVVI
IVVI = iv.IVVI('IVVI', address='COM10', numdacs=16, server_name=None)
station.add_component(IVVI)
Flux_Control = FluxCtrl.Flux_Control(name='FluxControl',IVVI=station.IVVI, num_channels=16)
station.add_component(Flux_Control)


#Initializaing ATS,
ATSdriver.AlazarTech_ATS.find_boards()
ATS = ATSdriver.AlazarTech_ATS9870(name='ATS', server_name=None)
station.add_component(ATS)

# Configure all settings in the ATS
ATS.config(clock_source='INTERNAL_CLOCK',
                sample_rate=100000000,
                clock_edge='CLOCK_EDGE_RISING',
                decimation=0,
                coupling=['AC','AC'],
                channel_range=[2.,2.],
                impedance=[50,50],
                bwlimit=['DISABLED','DISABLED'],
                trigger_operation='TRIG_ENGINE_OP_J',
                trigger_engine1='TRIG_ENGINE_J',
                trigger_source1='EXTERNAL',
                trigger_slope1='TRIG_SLOPE_POSITIVE',
                trigger_level1=128,
                trigger_engine2='TRIG_ENGINE_K',
                trigger_source2='DISABLE',
                trigger_slope2='TRIG_SLOPE_POSITIVE',
                trigger_level2=128,
                external_trigger_coupling='AC',
                external_trigger_range='ETR_5V',
                trigger_delay=0,
                timeout_ticks=0
)

#demodulation frequcnye is first set to 10 MHz
ATS_controller = ats_contr.Demodulation_AcquisitionController(name='ATS_controller',
                                                                      demodulation_frequency=10e6,
                                                                      alazar_name='ATS',
                                                                      server_name=None)
station.add_component(ATS_controller)

# configure the ATS controller
ATS_controller.update_acquisitionkwargs(#mode='NPT',
                 samples_per_record=64*1000,#4992,
                 records_per_buffer=8,#70, segmments
                 buffers_per_acquisition=8,
                 channel_selection='AB',
                 transfer_offset=0,
                 external_startcapture='ENABLED',
                 enable_record_headers='DISABLED',
                 alloc_buffers='DISABLED',
                 fifo_only_streaming='DISABLED',
                 interleave_samples='DISABLED',
                 get_processed_data='DISABLED',
                 allocated_buffers=100,
                 buffer_timeout=1000
)

HS = hd.HeterodyneInstrument('HS', LO=LO, RF=RF, AWG=None,
                             acquisition_instr=ATS.name,
                             acquisition_instr_controller=ATS_controller.name,
                             server_name=None)
station.add_component(HS)


MC = mc.MeasurementControl('MC')

MC.station = station
station.MC = MC
station.add_component(MC)


#qubit objects
QR1 = qbt.Tektronix_driven_transmon('QR1', LO=LO, cw_source=cw_source,
                                              td_source=Qubit_LO,
                                              IVVI=IVVI,
                                              rf_RO_source=RF,
                                              AWG=AWG,
                                              heterodyne_instr=HS,
                                              MC=MC,
                                              FluxCtrl=None,
                                              server_name=None)
station.add_component(QR1)
gen.load_settings_onto_instrument(QR1)
QR1.acquisition_instr('ATS')

QR2 = qbt.Tektronix_driven_transmon('QR2', LO=LO, cw_source=cw_source,
                                              td_source=Qubit_LO,
                                              IVVI=IVVI,
                                              rf_RO_source=RF,
                                              AWG=AWG,
                                              heterodyne_instr=HS,
                                              MC=MC,
                                              FluxCtrl=None,
                                              server_name=None)
station.add_component(QR2)
gen.load_settings_onto_instrument(QR2)
QR2.acquisition_instr('ATS')

QR3 = qbt.Tektronix_driven_transmon('QR3', LO=LO, cw_source=cw_source,
                                              td_source=Qubit_LO,
                                              IVVI=IVVI,
                                              rf_RO_source=RF,
                                              AWG=AWG,
                                              heterodyne_instr=HS,
                                              MC=MC,
                                              FluxCtrl=None,
                                              server_name=None)
station.add_component(QR3)
gen.load_settings_onto_instrument(QR3)
QR3.acquisition_instr('ATS')

QR4 = qbt.Tektronix_driven_transmon('QR4', LO=LO, cw_source=cw_source,
                                              td_source=Qubit_LO,
                                              IVVI=IVVI,
                                              rf_RO_source=RF,
                                              AWG=AWG,
                                              heterodyne_instr=HS,
                                              MC=MC,
                                              FluxCtrl=None,
                                              server_name=None)
station.add_component(QR4)
gen.load_settings_onto_instrument(QR4)
QR4.acquisition_instr('ATS')

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


def print_instr_params(instr):
    snapshot = instr.snapshot()
    print('{0:23} {1} \t ({2})'.format('\t parameter ', 'value', 'units'))
    print('-'*80)
    for par in sorted(snapshot['parameters']):
        print('{0:25}: \t{1}\t ({2})'.format(
            snapshot['parameters'][par]['name'],
            snapshot['parameters'][par]['value'],
            snapshot['parameters'][par]['units']))


def switch_to_pulsed_RO_UHFQC(qubit):

    qubit.RO_pulse_type('Gated_MW_RO_pulse')
    qubit.acquisition_instr('UHFQC_1')
    qubit.RO_acq_marker_channel('ch3_marker2')
    qubit.RO_acq_weight_function_I(0)
    qubit.RO_acq_weight_function_Q(1)
    qubit.spec_pulse_depletion_time()
    UHFQC_1.prepare_DSB_weight_and_rotation(qubit.f_RO_mod())
    qubit.pulse_I_offset(11e-3)
    qubit.pulse_Q_offset(-10e-3)
    qubit.RO_acq_averages(1024)
    qubit.RO_pulse_length(2e-6)
    qubit.RO_pulse_marker_channel('ch1_marker2')
    qubit.f_pulse_mod(50e6)
    qubit.RO_acq_averages(1024)
    qubit.RO_acq_marker_delay(100e-9)
    qubit.RO_pulse_delay(100e-9)
    qubit.spec_pulse_depletion_time(5e-6)
    qubit.spec_pulse_length(10e-6)
    qubit.spec_pulse_marker_channel('ch1_marker1')
    qubit.td_source_pow(16)
    qubit.RO_acq_integration_length(2e-6)
list_qubits = [QR1, QR2, QR3, QR4]
for qubit in list_qubits:
    switch_to_pulsed_RO_UHFQC(qubit)
UHFQC_1.awg_sequence_acquisition()