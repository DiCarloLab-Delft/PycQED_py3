#!/usr/bin/python

### setup logging before all imports (before any logging is done as to prevent a default root logger)
import logging
# configure root logger
root_logger = logging.getLogger('')
root_formatter = logging.Formatter('%(asctime)s.%(msecs)d %(levelname)s %(message)s [%(name)s]', '%Y%m%d %H:%M:%S')
root_sh = logging.StreamHandler()
root_sh.setLevel(logging.DEBUG)
root_sh.setFormatter(root_formatter)
root_logger.addHandler(root_sh)
# configure pycqed logger
pycqed_logger = logging.getLogger('pycqed')
pycqed_logger.setLevel(logging.DEBUG)  # FIXME: needed to get output, but why
# configure print logger
print_logger = logging.getLogger('print')
print_logger.setLevel(logging.DEBUG)  # FIXME: needed to get output, but why
# configure our logger
log = logging.getLogger('demo_2')
log.setLevel(logging.DEBUG)  # FIXME: needed to get output, but why
log.debug('starting')

# redirect print statements to log
if 0:  # does not catch prints from ZI
    import sys
    print = log.info
if 1:  # creates multiple lines in one entry, probably lines from ziPython which do not call print
    import sys
    sys.stdout.write = print_logger.info
if 0:
    print = lambda *tup : log.info(str(" ".join([str(x) for x in tup])))
if 0:  # does not catch prints from ZI
    import functools
    print = functools.partial(print, flush=True)
if 0:   # shows prints after log
    _orig_print = print
    def print(*args, **kwargs):
        _orig_print(*args, flush=True, **kwargs)
if 0:
    # make print unbuffered
    _orig_print = print
    def print(*args, **kwargs):
        _orig_print(*args, flush=True, **kwargs)
    # and redirect it to logging
    import sys
    sys.stdout.write = log.info


### imports
import sys
import os
from pathlib import Path
import numpy as np

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.wouter import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.wouter import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan

import pycqed.measurement.openql_experiments.openql_helpers as oqh
from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo

from qcodes import station



def set_waveforms(awg, waveform_type, sequence_length):
    if waveform_type == 'square':
        for ch in range(8):
            for i in range(sequence_length):
                awg.set('wave_ch{}_cw{:03}'.format(ch + 1, i), np.ones(48) * i / (sequence_length - 1))
    elif waveform_type == 'cos':
        for ch in range(8):
            for i in range(sequence_length):
                awg.set('wave_ch{}_cw{:03}'.format(ch + 1, i), np.cos(np.arange(48) / 2) * i / (sequence_length - 1))
    else:
        raise KeyError()

# parameter handling
log.debug('started')
sel = 0
if len(sys.argv)>1:
    sel = int(sys.argv[1])

##########################################
# Constants
##########################################
# instrument info
conf = lambda:0 # create empty 'struct'
conf.ro_0 = ''
conf.ro_0 = 'dev2295'
conf.mw_0 = 'dev8079'
conf.flux_0 = ''
conf.cc_ip = '192.168.0.241'

qubit_idx = 3 # NB: connects to AWG8'mw_0'  in slot 3
slot_ro_1 = 1
slot_mw_0 = 4
curdir = os.path.dirname(__file__)
cfg_openql_platform_fn = str(Path("../../pycqed/tests/openql/test_cfg_cc.json"))
print(cfg_openql_platform_fn)

##########################################
# OpenQL program
##########################################

if sel==0:  # ALLXY
    # based on CCL_Transmon.py::measure_allxy()
    log.debug('compiling allxy')
    p = sqo.AllXY(qubit_idx=qubit_idx, double_points=True, platf_cfg=cfg_openql_platform_fn)
    cc_file_name = p.filename

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
    cc_file_name = p.filename

if sel==2:  # Rabi
    # based on CCL_Transmon.py::measure_rabi_channel_amp()
    log.debug('compiling Rabi')
    p = sqo.off_on(
        qubit_idx=qubit_idx, pulse_comb='on',
        initialize=False,
        platf_cfg=cfg_openql_platform_fn)
    cc_file_name = p.filename

if sel==3:  # Quantum staircase
    # based on CCL_Transmon.py::measure_rabi_channel_amp()
    log.debug('compiling Quantum staircase')

    p = oqh.create_program('QuantumStaircase', cfg_openql_platform_fn)

    k = oqh.create_kernel("Main", p)
    # gates with codewords [1:6]
    k.gate('rx180', [qubit_idx])
    k.gate('ry180', [qubit_idx])
    k.gate('rx90', [qubit_idx])
    k.gate('ry90', [qubit_idx])
    k.gate('rxm90', [qubit_idx])
    k.gate('rym90', [qubit_idx])
    p.add_kernel(k)

    p = oqh.compile(p)
    cc_file_name = p.filename


log.debug("File for CC = '{}'".format(cc_file_name))

##########################################
# Open physical instruments
##########################################
#station = station.Station()

instr = lambda:0 # create empty 'struct'
if conf.mw_0 != '':
    log.debug('connecting to mw_0')
    instr.mw_0 = ZI_HDAWG8.ZI_HDAWG8('mw_0', device=conf.mw_0)
    #station.add_component(instr.mw_0)

if conf.flux_0 != '':
    log.debug('connecting to flux_0')
    instr.flux_0 = ZI_HDAWG8.ZI_HDAWG8('flux_0', device=conf.flux_0)
    #station.add_component(instr.flux_0)

if conf.ro_0 != '':
    log.debug('connecting to ro_0')
    instr.ro_0 = ZI_UHFQC.UHFQC('ro_0', device=conf.ro_0)
    #station.add_component(instr.ro_0)

log.debug('connecting to CC')
instr.cc = QuTechCC('cc', IPTransport(conf.cc_ip))
instr.cc.reset()
instr.cc.clear_status()
instr.cc.set_status_questionable_frequency_enable(0x7FFF)

##########################################
# Open virtual instruments
##########################################
rolut = UHFQC_RO_LutMan('rolut', num_res=7)
#station.add_component(rolut)

##########################################
#  Configure AWGs
##########################################
if conf.mw_0 != '':
    log.debug('configuring mw_0')
    # define sequence
    sequence_length = 32

    # configure instrument
    instr.mw_0.load_default_settings()
    instr.mw_0.assure_ext_clock()
    set_waveforms(instr.mw_0, 'square', sequence_length)
    instr.mw_0.cfg_num_codewords(sequence_length)  # this makes the seqC program a bit smaller
    instr.mw_0.cfg_codeword_protocol('microwave')
    instr.mw_0.upload_codeword_program()
    if 1:   # FIXME: should be moved to driver
        instr.mw_0._dev.setd('raw/dios/0/extclk', 1)  # enable 50 MHz sampling of DIO inputs
        for awg in range(4):
            instr.mw_0.set('awgs_{}_dio_strobe_slope'.format(awg), 0)  # disable strobe triggering

    #AWG8.calibrate_dio_protocol() # aligns the different bits in the codeword protocol
    if 0:
        delay = 1   # OK: [1:2] in our particular configuration, with old AWG8 firmware (not yet sampling at 50 MHz)
        for awg in range(4):
            instr.mw_0._set_dio_delay(awg, 0x40000000, 0xBFFFFFFF, delay)  # skew TOGGLE_DS versus rest
    else:
        delay = 1  # firmware 62730, LabOne LabOneEarlybird64-19.05.62848.msi
        instr.mw_0._dev.setd('raw/dios/0/delays/*/value', delay)  # new interface?, range [0:15]
        for awg in range(4):
            dio_timing_errors = instr.mw_0._dev.geti('awgs/{}/dio/error/timing'.format(awg))
            log.debug('DIO timing errors on AWG {}: {}'.format(awg,dio_timing_errors))

"""
    *Before* adding 'raw/dios/0/extclk'=1 and 'awgs_{}_dio_strobe_slope'=0
    delay   stable  timing errors   scope delta T between CC marker rising and AWG8 signal falling [ns]
    0       +       0/0/0/0         60
    1       +       0/0/0/0         60
    2       +       0/1/1/1         66
    3       +       1/1/1/1         66
    4       +       1/0/0/0         73
    5       +       0/0/0/0         73
    
    6       +       0/0/0/0         80
    7       +       0/0/0/0         80
    8       +       0/1/1/1         86
    9       +       1/1/1/1         86
    10      +       1/0/0/0         93
    11      +       0/0/0/0         93
    
    12      +       0/0/0/0         99
    13      +       0/0/0/0         99
    14      +       0/1/1/1        106
    15      +       1/1/1/1        106
    
    Analysis:
    - delay steps are 3.33 ns each
    - 15 steps == 50 ns
    - 6 steps == 20 ns, pattern repeats after that


    *After* adding 'raw/dios/0/extclk'=1 and 'awgs_{}_dio_strobe_slope'=0
    delay   stable  timing errors   scope delta T between CC marker rising and AWG8 signal falling [ns]
    0       +       0/0/0/0         66
    1       +       0/0/0/0         66
    2       +       0/0/0/0         66
    
    3       +       1/0/0/0         86
    4       +       1/0/0/0         86
    5       +       0/0/0/0         86
    6       +       0/0/0/0         86
    7       +       0/0/0/0         86
    8       +       0/0/0/0         86
    
    9       +       1/0/0/0        106
    ...
    
    
"""

if conf.flux_0 != '':
    log.debug('configuring flux_0')
    # define sequence
    sequence_length = 8

    # configure instrument
    instr.mw_0.load_default_settings()
    instr.flux_0.assure_ext_clock()
    set_waveforms(instr.flux_0, 'square', sequence_length)
    instr.flux_0.cfg_num_codewords(sequence_length)  # this makes the seqC program a bit smaller
    instr.flux_0.cfg_codeword_protocol('flux')
    instr.flux_0.upload_codeword_program()
    #AWG8.calibrate_dio_protocol() # aligns the different bits in the codeword protocol

##########################################
#  Configure UHFQA's
##########################################

if conf.ro_0 != '':
    log.debug('configuring ro_0')
    instr.mw_0.load_default_settings() # FIXME: also done at init?
    rolut.AWG(instr.mw_0.name)




##########################################
#  Configure CC
##########################################

log.debug('configuring CC')
instr.cc.debug_marker_out(slot_ro_1, instr.cc.UHFQA_TRIG) # UHF-QA trigger
instr.cc.debug_marker_out(slot_mw_0, instr.cc.HDAWG_TRIG) # HDAWG trigger

log.debug("uploading '{}' to CC".format(p.filename))
instr.cc.eqasm_program(p.filename)

if 0:
    err_cnt = instr.cc.get_system_error_count()
    if err_cnt>0:
        log.warning('CC status after upload')
    for i in range(err_cnt):
        print(instr.cc.get_error())

log.debug('starting CC')
instr.cc.start()

if 0:
    err_cnt = instr.cc.get_system_error_count()
    if err_cnt>0:
        log.warning('CC status after start')
    for i in range(err_cnt):
        print(instr.cc.get_error())

