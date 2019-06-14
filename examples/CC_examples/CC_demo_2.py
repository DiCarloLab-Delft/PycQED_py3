#!/usr/bin/python

import os
import logging
import sys
import numpy as np

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan

from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo

from qcodes import station

##########################################
# Constants
##########################################
# instrument info
conf = lambda:0 # create empty 'struct'
conf.ro_0 = 'dev2209'
conf.mw_0 = 'dev8078'
conf.flux_0 = ''
conf.cc_ip = '192.168.0.241'

qubit_idx = 10
curdir = os.path.dirname(__file__)
cfg_openql_platform_fn = os.path.join(curdir, 'demo1_cfg.json')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


##########################################
# Open physical instruments
##########################################
station = station.Station()
instr = lambda:0 # create empty 'struct'

if conf.mw_0 != '':
    instr.mw_0 = ZI_HDAWG8.ZI_HDAWG8('mw_0', device=conf.mw_0)
    station.add_component(instr.mw_0)

if conf.flux_0 != '':
    instr.flux_0 = ZI_HDAWG8.ZI_HDAWG8('flux_0', device=conf.flux_0)
    station.add_component(instr.flux_0)

instr.ro_0 = ZI_UHFQC.UHFQC('ro_0', device=conf.ro_0)
station.add_component(instr.ro_0)

log.debug('connecting to CC')
instr.cc = QuTechCC('cc', IPTransport(conf.cc_ip))
instr.cc.reset()
instr.cc.clear_status()
instr.cc.set_status_questionable_frequency_enable(0x7FFF)



##########################################
# Open virtual instruments
##########################################
rolut = UHFQC_RO_LutMan('rolut', num_res=7)
#rolut.AWG(UHFQC.name)
station.add_component(rolut)


### calibrate DIO





##########################################
# Configure AWG8 waveforms
##########################################

# This creates a staircase pattern

waveform_type = 'square'
# waveform_type = 'cos'

if waveform_type =='square':
    for ch in range(8):
        for i in range(sequence_length):
            AWG8.set('wave_ch{}_cw{:03}'.format(ch+1, i), (np.ones(48)*i/(sequence_length-1)))
elif waveform_type == 'cos':
    for ch in range(8):
        for i in range(sequence_length):
            AWG8.set('wave_ch{}_cw{:03}'.format(ch+1, i), (np.cos(np.arange(48)/2)*i/(sequence_length-1)))
else:
    raise KeyError()

# this makes the program a bit smaller
AWG8.cfg_num_codewords(sequence_length)

##########################################
#  4. Configuring the DIO protocol       #
##########################################

if conf.mw_0 != '':
    instr.mw_0.cfg_codeword_protocol('microwave')
    instr.mw_0.configure_codeword_protocol()
    instr.mw_0.upload_codeword_program()
    #AWG8.calibrate_dio_protocol() # aligns the different bits in the codeword protocol



















if sel==0:  # ALLXY
    # based on CCL_Transmon.py::measure_allxy()
    log.debug('compiling allxy')
    p = sqo.AllXY(qubit_idx=qubit_idx, double_points=True, platf_cfg=cfg_openql_platform_fn)
    print(p.filename)

if sel==1:  # Ramsey
    # based on CCL_Transmon.py::measure_ramsey()
    # funny default is because there is no real time sideband
    # modulation
    log.debug('compiling Ramsey')
    T2_star = 20e-6
    cfg_cycle_time = 20e-9
    stepsize = (T2_star * 4 / 61) // (abs(cfg_cycle_time)) \
               * abs(cfg_cycle_time)
    times = np.arange(0, T2_star * 4, stepsize)
    p = sqo.Ramsey(times, qubit_idx=qubit_idx, platf_cfg=cfg_openql_platform_fn)
    print(p.filename)

if sel==2:  # Rabi
    # based on CCL_Transmon.py::measure_rabi_channel_amp()
    log.debug('compiling Rabi')
    p = sqo.off_on(
        qubit_idx=qubit_idx, pulse_comb='on',
        initialize=False,
        platf_cfg=cfg_openql_platform_fn)
    print(p.filename)


if 1:

    if 1:
        cc.debug_marker_out(0, cc.UHFQA_TRIG) # UHF-QA trigger
        cc.debug_marker_out(8, cc.HDAWG_TRIG) # HDAWG trigger


    log.debug('uploading {}'.format(p.filename))
    if 0:
        cc.eqasm_program = p.filename
    else:
        with open(p.filename, 'r') as f:
            prog = f.read()
        cc.sequence_program(prog)

    err_cnt = cc.get_system_error_count()
    for i in range(err_cnt):
        print(cc.get_error())

    log.debug('starting CC')
    cc.start()
