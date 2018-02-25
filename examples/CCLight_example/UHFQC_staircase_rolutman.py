import os
import pycqed as pq
import qcodes as qc
station = qc.station


# making full DIO triggered lookuptable
rolut = station.components['RO_lutman']
UHFQC = station.components['UHFQC']
CCL = station.components['CCL']

# This load the CCL sequence that triggers consecutive readouts
example_fp = os.path.abspath(os.path.join(
    pq.__path__[0], '..', 'examples',
    'CCLight_example', 'qisa_test_assembly', 'UHFQC_ro_example.qisa'))
CCL.eqasm_program(example_fp)
CCL.start()

amps = [0.1, 0.2, 0.3, 0.4, 0.5]
rolut.sampling_rate(1.8e9)
# testing single pulse sequence
for i in range(5):
    rolut.set('M_amp_R{}'.format(i), amps[i])
    rolut.set('M_phi_R{}'.format(i), -45)

    rolut.set('M_down_amp0_R{}'.format(i), amps[i]/2)
    rolut.set('M_down_amp1_R{}'.format(i), -amps[i]/2)
    rolut.set('M_down_phi0_R{}'.format(i), -45+180)
    rolut.set('M_down_phi1_R{}'.format(i), -45)

    rolut.set('M_length_R{}'.format(i), 500e-9)
    rolut.set('M_down_length0_R{}'.format(i), 200e-9)
    rolut.set('M_down_length1_R{}'.format(i), 200e-9)
    rolut.set('M_modulation_R{}'.format(i), 0)

rolut.acquisition_delay(200e-9)
UHFQC.quex_rl_length(1)
UHFQC.quex_wint_length(int(600e-9*1.8e9))
UHFQC.awgs_0_enable(1)
rolut.AWG('UHFQC')
rolut.generate_standard_waveforms()
rolut.pulse_type('M_up_down_down')
rolut.resonator_combinations([[0], [1], [2], [3], [4]])
rolut.load_DIO_triggered_sequence_onto_UHFQC()
UHFQC.awgs_0_userregs_0(1024)
UHFQC.awgs_0_enable(1)
