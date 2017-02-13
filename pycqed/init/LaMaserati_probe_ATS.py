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
qc_config = {'datadir': r'D:\\Experiments\\1701_Vertical_IO_W16_chipA\\Data',
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

#Initializing UHFQC
# UHFQC_1 = ZI_UHFQC.UHFQC('UHFQC_1', device='dev2209', server_name=None)
# station.add_component(UHFQC_1)

#initializing AWG
# AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None, timeout=2,
#                             address='GPIB0::8::INSTR', server_name=None)

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

# VNA
# VNA = ZNB20.ZNB20(name='VNA', address='TCPIP0::192.168.0.55', server_name=None)  #
# station.add_component(VNA)

# variable attenuator
# ATT = Weinschel_8320_novisa.Weinschel_8320(name='ATT',address='192.168.0.54', server_name=None)
# station.add_component(ATT)


MC = mc.MeasurementControl('MC')

MC.station = station
station.MC = MC
station.add_component(MC)


def print_instr_params(instr):
    snapshot = instr.snapshot()
    print('{0:23} {1} \t ({2})'.format('\t parameter ', 'value', 'units'))
    print('-'*80)
    for par in sorted(snapshot['parameters']):
        print('{0:25}: \t{1}\t ({2})'.format(
            snapshot['parameters'][par]['name'],
            snapshot['parameters'][par]['value'],
            snapshot['parameters'][par]['units']))