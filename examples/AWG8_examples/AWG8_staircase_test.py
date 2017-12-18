"""
@author: Adriaan
@date: 15-12-2017
This contains a test program for the AWG8 that shows a staircase pattern.
Additionaly it also contains the DIO timing calibration protocol.

It is made up of X parts to be self contained. Not all parts are needed
if one wants to use this in a running experiment.

1. General import statements and instantiating the required instruments.
2. Uploading the AWG8 test program to the CCLight
3. Uploading the staircase waveforms to the AWG8
4. configuring the protocol

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

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_HDAWG8
from pycqed.instrument_drivers.physical_instruments import QuTech_CCL
reload(QuTech_CCL)

CCL = QuTech_CCL.CCL('CCL', address='192.168.0.11', port=5025)
cs_filepath = os.path.join(pq.__path__[0], 'measurement','openql_experiments',
                           'output','cs.txt')

CCL.upload_microcode(cs_filepath)

# AWG8 = ZI_HDAWG8.ZI_HDAWG8('AWG8_8003', device='dev8003')
AWG8 = ZI_HDAWG8.ZI_HDAWG8('AWG8_8004', device='dev8004')
#AWG8 = ZI_HDAWG8.ZI_HDAWG8('AWG8_8005', device='dev8005')
#AWG8 = ZI_HDAWG8.ZI_HDAWG8('AWG8_8006', device='dev8006')
# AWG8 = ZI_HDAWG8.ZI_HDAWG8('AWG8_8008', device='dev8008')


##########################################
#  2. Starting AWG8 test program in CCL  #
##########################################

AWG_type = 'microwave'
# AWG_type = 'flux'

if AWG_type == 'microwave':
    example_fp = os.path.abspath(
        os.path.join(pq.__path__[0], '..','examples','CCLight_example',
                     'qisa_test_assembly','consecutive_cws_double.qisa'))
elif AWG_type == 'flux':
    example_fp = os.path.abspath(os.path.join(pq.__path__[0], '..',
        'examples','CCLight_example',
        'qisa_test_assembly','consecutive_cws_flux.qisa'))

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

if waveform_type =='square':
    for ch in range(8):
        for i in range(32):
            AWG8.set('wave_ch{}_cw{:03}'.format(ch+1, i), (np.ones(48)*i/32))
elif waveform_type == 'cos':
    for ch in range(8):
        for i in range(32):
            AWG8.set('wave_ch{}_cw{:03}'.format(ch+1, i), (np.cos(np.arange(48)/2)*i/32))
else:
    raise KeyError()

# this makes the program a bit smaller
AWG8.cfg_num_codewords(32)
# and this configures
AWG8.upload_codeword_program()



##########################################
#  4. Configuring the DIO protocol       #
##########################################
AWG8.cfg_codeword_protocol('microwave') # <- ensures all bits are uploaded
AWG8.configure_codeword_protocol()
AWG8.upload_codeword_program()
AWG8.calibrate_dio_protocol()