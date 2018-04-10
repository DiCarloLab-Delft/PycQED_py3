"""
@author: Ramiro
@date: 10-04-2018

Adapted from AWG8_staircase_test.py
@author: Adriaan
@date: 15-12-2017

This contains a test program for the QWG that shows a staircase pattern.

It is made up of 3 parts to be self contained. Not all parts are needed
if one wants to use this in a running experiment.

1. General import statements and instantiating the required instruments.
2. Uploading the QWG test program to the CCLight
3. Uploading the staircase waveforms to the QWG

"""


##########################################
#  1. Instantiating instruments          #
##########################################

import numpy as np
import os
import pycqed as pq
from importlib import reload
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan
from pycqed.instrument_drivers.meta_instrument.LutMans.mw_lutman import get_redundant_codewords

from pycqed.instrument_drivers.physical_instruments import QuTech_AWG_Module as qwg
from pycqed.instrument_drivers.physical_instruments import QuTech_CCL
reload(QuTech_CCL)

CCL = QuTech_CCL.CCL('CCL', address='192.168.0.11', port=5025)
cs_filepath = os.path.join(pq.__path__[0], 'measurement', 'openql_experiments',
                           'output', 'cs.txt')

CCL.control_store(cs_filepath)

QWG = qwg.QuTech_AWG_Module('QWG_MW', address='192.168.0.190', port=5025, numCodewords=256)


##########################################
#  2. Starting AWG8 test program in CCL  #
##########################################

AWG_type = 'microwave'
# AWG_type = 'flux'

if AWG_type == 'microwave':
    example_fp = os.path.abspath(
        os.path.join(pq.__path__[0], '..', 'examples', 'CCLight_example',
                     'qisa_test_assembly', 'consecutive_cws_double.qisa'))
elif AWG_type == 'flux':
    example_fp = os.path.abspath(os.path.join(pq.__path__[0], '..',
                                              'examples', 'CCLight_example',
                                              'qisa_test_assembly', 'consecutive_cws_flux.qisa'))

print(example_fp)
CCL.eqasm_program(example_fp)
CCL.start()


##########################################
#  3. Configuring the DIO protocol       #
##########################################

# This creates a staircase pattern
import numpy as np

waveform_type = 'square'
# waveform_type = 'cos'

#######################
# CHANNELS 1,2 FOR s0 #
# CHANNELS 3,4 FOR s2 #
#######################

if waveform_type == 'square':
    for ch in range(8):
        for codeword_int in range(256):
            # FOR s0
            redundant_cw_list = get_redundant_codewords(codeword_int,
                                                        bit_width=4,
                                                        bit_shift=4)
            # update all of them
            wave_I = np.ones(50)*codeword_int/256
            wave_Q = -np.ones(50)*codeword_int/256
            for redundant_cw_idx in redundant_cw_list:
                redundant_cw_I = 'wave_ch1_cw{:03}'.format(redundant_cw_idx)
                QWG.get_instr().set(redundant_cw_I, wave_I)
                redundant_cw_Q = 'wave_ch2_cw{:03}'.format(redundant_cw_idx)
                QWG.get_instr().set(redundant_cw_Q, wave_Q)

            # FOR s1
            redundant_cw_list = get_redundant_codewords(codeword_int,
                                                        bit_width=4,
                                                        bit_shift=0)
            # update all of them
            wave_I = np.ones(50)*codeword_int/256
            wave_Q = -np.ones(50)*codeword_int/256
            for redundant_cw_idx in redundant_cw_list:
                redundant_cw_I = 'wave_ch3_cw{:03}'.format(redundant_cw_idx)
                QWG.get_instr().set(redundant_cw_I, wave_I)
                redundant_cw_Q = 'wave_ch4_cw{:03}'.format(redundant_cw_idx)
                QWG.get_instr().set(redundant_cw_Q, wave_Q)


            QWG.set('wave_ch{}_cw{:03}'.format(ch+1, i), (np.ones(50)*i/256))
elif waveform_type == 'cos':
    for ch in range(8):
        for i in range(256):
            QWG.set('wave_ch{}_cw{:03}'.format(ch+1, i),
                     (np.cos(np.arange(50)/2)*i/32))
else:
    raise KeyError()

