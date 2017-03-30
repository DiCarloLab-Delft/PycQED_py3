# remember to turn off AWG and IVVI rack while reiwiring for distortion calibration

AWG.all_channels_off()
IVVI.dac1(0.)
IVVI.dac2(0.)

# define a distortions object

import pycqed.instrument_drivers.meta_instrument.kernel_object as k_obj
k0 = k_obj.Distortion(name='k0')
station.add_component(k0)
k0.channel(2)
k0.kernel_dir(
    r'D:\Experiments\1702_Starmon\kernels')

# define an identity distortion to start
dist_dict = {'ch_list': ['ch2'],
             'ch2': [1,0]}

# after starting you want to redefine the distortion dictionary as


k0.kernel_list(['kernel_170307_120012_fitSPsaveSP.txt',
                'kernel_170308_100242_biastee_quad10965_45000.txt',
                'kernel_170308_114945_fitSPsaveSP.txt',
                'kernel_170308_115508_fitSPsaveSP.txt',
                'kernel_170308_120149_fitSPsaveSP.txt',
                'kernel_170308_120821_B_fitSPsaveSP.txt',
                'kernel_170308_121552_B_fitSPsaveSP.txt',
                'kernel_170308_122054_fast_corrections.txt'])
dist_dict = {'ch_list': ['ch2'],
             'ch2': k0.kernel()}

# generate the flux pulse to calibrate

flux_pulse_pars = {'pulse_type': 'SquarePulse',
              'pulse_delay': .1e-6,
              'channel': 'ch2',
              'amplitude': .23,
              'length': 40e-6,
              'dead_time_length': 10e-6}
# upload sequence
AWG.ch2_amp(2.)
seq_square = awg_swf.fsqs.single_pulse_seq(flux_pulse_pars,comp_pulse=True,
                                           distortion_dict=dist_dict, return_seq=True)
AWG.start()

# after converging, compile all the kernels toghether

k_list = ['kernel_170307_120012_fitSPsaveSP.txt',
                'kernel_170308_100242_biastee_quad10965_45000.txt',
                'kernel_170308_114945_fitSPsaveSP.txt','kernel_170308_115508_fitSPsaveSP.txt',
                'kernel_170308_120149_fitSPsaveSP.txt',
                'kernel_170308_120821_B_fitSPsaveSP.txt',
                'kernel_170308_121552_B_fitSPsaveSP.txt',
                'kernel_170308_122054_fast_corrections.txt']
length = 100000
kernels = np.loadtxt(k0.kernel_dir()+r'/'+k_list[0])
for k in k_list[1:]:
    v = np.loadtxt(k0.kernel_dir()+r'/'+k)
    kernels = np.convolve(v,kernels)[:max(len(v),len(kernels))]
# kernels /= np.sum(kernels)
np.sum(kernels)

np.savetxt(k0.kernel_dir()+r'/RT_Compiled_170308.txt',kernels)

np.savetxt(k0.kernel_dir()+r'/RT_Compiled_norm_170308.txt',kernels/np.sum(kernels))


import qcodes as qc
station = qc.station
from qcodes.utils import validators as vals
from pycqed.instrument_drivers.meta_instrument.qubit_objects import qubit_object as qo
from pycqed.instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as cbt
from pycqed.instrument_drivers.meta_instrument.qubit_objects import Tektronix_driven_transmon as qbt

QR.add_operation('SWAP')
QR.add_pulse_parameter('SWAP', 'fluxing_operation_type', 'operation_type',
                          initial_value='Flux', vals=vals.Strings())
QR.add_pulse_parameter('SWAP', 'SWAP_pulse_amp', 'amplitude',
                          initial_value=0.5)
QR.link_param_to_operation('SWAP', 'fluxing_channel', 'channel')

QR.add_pulse_parameter('SWAP', 'SWAP_pulse_type', 'pulse_type',
                          initial_value='SquareFluxPulse', vals=vals.Strings())
QR.add_pulse_parameter('SWAP', 'SWAP_refpoint',
                          'refpoint', 'end', vals=vals.Strings())
QR.link_param_to_operation('SWAP', 'SWAP_amp', 'SWAP_amp')
QR.add_pulse_parameter('SWAP', 'SWAP_pulse_buffer',
                          'pulse_buffer', 0e-9)

QR.link_param_to_operation('SWAP', 'SWAP_time', 'square_pulse_length')


QR.add_pulse_parameter('SWAP', 'SWAP_pulse_delay',
                          'pulse_delay', 0e-9)