#!/usr/bin/python

### setup logging before all imports (before any logging is done as to prevent a default root logger)
import CC_logging

### imports
import sys
import os
import logging
import numpy as np
from pathlib import Path

from pycqed.instrument_drivers.physical_instruments.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTechCC import QuTechCC
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC

from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan

import pycqed.measurement.openql_experiments.openql_helpers as oqh
from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo

from qcodes import station

# configure our logger
log = logging.getLogger('demo_2')
log.setLevel(logging.DEBUG)
log.debug('starting')


def set_waveforms(instr_awg, waveform_type, sequence_length):
    if waveform_type == 'square':
        for i in range(sequence_length):
            wav = np.ones(48) * i / (sequence_length - 1)
            for ch in range(8):
                instr_awg.set('wave_ch{}_cw{:03}'.format(ch + 1, i), wav)
    elif waveform_type == 'cos':
        for i in range(sequence_length):
            wav = np.cos(np.arange(48) / 2) * i / (sequence_length - 1)
            for ch in range(8):
                instr_awg.set('wave_ch{}_cw{:03}'.format(ch + 1, i), wav)
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
conf.ro = [] # 'dev2312'   # 'dev2295'
conf.mw = ['dev8068', 'dev8079']
conf.flux = []
conf.cc_ip = '192.168.0.241'

qubit_idx = 3 # NB: connects to AWG8'mw'  in slot 3
slot_ro_1 = 1
slot_mw = 4
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
instr.mw = []
instr.flux = []
instr.ro = []
for i, dev in enumerate(conf.mw):
    name = 'mw_'+str(i)
    log.debug(f'connecting to microwave AWG8 {name}={dev}')
    instr.mw.append(ZI_HDAWG8.ZI_HDAWG8(name, device=dev))
    log.debug(f'connected to mw AWG8 {name}={dev}')
    #station.add_component(instr.mw[i])

for i, dev in enumerate(conf.flux):
    name = 'flux_'+str(i)
    log.debug(f'connecting to flux AWG8 {name}={dev}')
    instr.flux.append(ZI_HDAWG8.ZI_HDAWG8(name, device=dev))
    #station.add_component(instr.flux[i])

for i, dev in enumerate(conf.ro):
    name = 'ro_'+str(i)
    log.debug(f'connecting to readout UHFQA {name}={dev}')
    instr.ro.append(ZI_UHFQC.UHFQC(name, device=dev))
    #station.add_component(instr.ro[i])

log.debug('connecting to CC')
instr.cc = QuTechCC('cc', IPTransport(conf.cc_ip))
instr.cc.reset()
instr.cc.clear_status()
instr.cc.status_preset()
log.debug('connected to CC, set to defaults')

##########################################
# Open virtual instruments
##########################################
rolut = UHFQC_RO_LutMan('rolut', num_res=7)
#station.add_component(rolut)

##########################################
#  Configure AWGs
##########################################
for i, dev in enumerate(conf.mw):
    log.debug(f'configuring microwave HDAWG {dev}')
    # define sequence
    sequence_length = 32

    # configure instrument
    instr.mw[i].clear_errors()
    instr.mw[i].load_default_settings()
    instr.mw[i].assure_ext_clock()
    set_waveforms(instr.mw[i], 'square', sequence_length)

    # set DIO protocol
    codeword_protocol = 'microwave'
    instr.mw[i].cfg_codeword_protocol.set(codeword_protocol)
    instr.mw[i].upload_codeword_program()  # FIXME: also done by calibrate_dio_protocol?

    if 1:  # DIO calibration
        instr.cc.output_dio_calibration_data(codeword_protocol)

        try:
            instr.mw[i].calibrate_dio_protocol(verbose=True)
        except:
            log.warning('calibrate_dio_protocol raised exception')
            instr.mw[i].plot_dio_snapshot()
            raise

    instr.mw[i].start()


for i, dev in enumerate(conf.flux):
    log.debug(f'configuring flux HDAWG {dev}')
    # define sequence
    sequence_length = 8

    # configure instrument
    instr.flux[i].clear_errors()
    instr.flux[i].load_default_settings()
    instr.flux[i].assure_ext_clock()
    set_waveforms(instr.flux[i], 'square', sequence_length)
    instr.flux[i].cfg_codeword_protocol.set('flux')
    instr.flux[i].upload_codeword_program()
    #AWG8.calibrate_dio_protocol() # aligns the different bits in the codeword protocol

##########################################
#  Configure UHFQA's
##########################################

for i, dev in enumerate(conf.ro):
    log.debug(f'configuring readout UHFQA {dev}')
    instr.ro[i].load_default_settings() # FIXME: also done at init?

    # configure UHFQC to generate codeword based readout signals
    instr.ro[i].quex_rl_length(1)
    instr.ro[i].quex_wint_length(int(600e-9 * 1.8e9))

    if 1:
        # generate waveforms and ZIseqC program using rolut
        amps = [0.1, 0.2, 0.3, 0.4, 0.5]
        resonator_codeword_bit_mapping = [0, 2, 3, 5, 6]   # FIXME: default Base_RO_LutMan _resonator_codeword_bit_mapping

        for i,res in enumerate(resonator_codeword_bit_mapping):
            rolut.set('M_amp_R{}'.format(res), amps[i])
            rolut.set('M_phi_R{}'.format(res), -45)

            rolut.set('M_down_amp0_R{}'.format(res), amps[i] / 2)
            rolut.set('M_down_amp1_R{}'.format(res), -amps[i] / 2)
            rolut.set('M_down_phi0_R{}'.format(res), -45 + 180)
            rolut.set('M_down_phi1_R{}'.format(res), -45)

            rolut.set('M_length_R{}'.format(res), 500e-9)
            rolut.set('M_down_length0_R{}'.format(res), 200e-9)
            rolut.set('M_down_length1_R{}'.format(res), 200e-9)
            rolut.set('M_modulation_R{}'.format(res), 0)

        rolut.acquisition_delay(200e-9)
        rolut.AWG(instr.ro[i].name)
        rolut.sampling_rate(1.8e9)
        rolut.generate_standard_waveforms()
        rolut.pulse_type('M_up_down_down')
        #rolut.resonator_combinations([[0], [2], [3], [5], [6]])  # FIXME: must use resonators from resonator_codeword_bit_mapping
        rolut.resonator_combinations([[0,2,3,5,6]])  # FIXME: must use resonators from resonator_codeword_bit_mapping
        rolut.load_DIO_triggered_sequence_onto_UHFQC()  # upload waveforms and ZIseqC program

        instr.ro[i].awgs_0_userregs_0(1024)  # loop_cnt, see UHFQC driver (awg_sequence_acquisition_and_DIO_triggered_pulse)

##########################################
#  Configure CC
##########################################

log.debug('configuring CC')
instr.cc.debug_marker_out(slot_ro_1, instr.cc.UHFQA_TRIG) # UHF-QA trigger
instr.cc.debug_marker_out(slot_mw, instr.cc.HDAWG_TRIG) # HDAWG trigger

log.debug(f"uploading '{p.filename}' to CC")
instr.cc.eqasm_program(p.filename)

log.debug("printing CC errors")
err_cnt = instr.cc.get_system_error_count()
if err_cnt>0:
    log.warning('CC status after upload')
for i in range(err_cnt):
    print(instr.cc.get_error())

log.debug('starting CC')
instr.cc.start()

err_cnt = instr.cc.get_system_error_count()
if err_cnt>0:
    log.warning('CC status after start')
for i in range(err_cnt):
    print(instr.cc.get_error())









    """
    if 0:  # manual DIO delay
        if 0:
            delay = 1  # OK: [1:2] in our particular configuration, with old AWG8 firmware (not yet sampling at 50 MHz)
            for awg in range(4):
                instr.mw[i]._set_dio_delay(awg, 0x40000000, 0xBFFFFFFF, delay)  # skew TOGGLE_DS versus rest
        else:
            delay = 0  # firmware 62730, LabOne LabOneEarlybird64-19.05.62848.msi
            instr.mw[i].setd('raw/dios/0/delays/*/value', delay)  # new interface?, range [0:15]
            for awg in range(4):
                dio_timing_errors = instr.mw[i].geti('awgs/{}/dio/error/timing'.format(awg))
                log.debug('DIO timing errors on AWG {}: {}'.format(awg, dio_timing_errors))

    dev8079 *After* adding 'raw/dios/0/extclk'=1 and 'awgs_{}_dio_strobe_slope'=0
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



    dev8079 *Before* adding 'raw/dios/0/extclk'=1 and 'awgs_{}_dio_strobe_slope'=0
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
    ...


"""
