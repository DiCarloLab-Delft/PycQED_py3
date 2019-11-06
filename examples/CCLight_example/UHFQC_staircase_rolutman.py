import os
import pycqed as pq
import qcodes as qc

station = qc.Station()

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import UHFQuantumController as ZI_UHFQC

UHFQC = ZI_UHFQC.UHFQC('UHFQC', device='dev2320', server_name=None)
station.add_component(UHFQC)
UHFQC.timeout(60)

from pycqed.instrument_drivers.physical_instruments import QuTech_CCL

CCL = QuTech_CCL.CCL('CCL', address='192.168.0.11', port=5025)
cs_filepath = os.path.join(pq.__path__[0], 'measurement', 'openql_experiments',
                           'output', 'cs.txt')
print("cs_file_path", cs_filepath)
opc_filepath = os.path.join(pq.__path__[0], 'measurement',
                            'openql_experiments', 'output',
                            'qisa_opcodes.qmap')
CCL.control_store(cs_filepath)
CCL.qisa_opcode(opc_filepath)
CCL.num_append_pts(2)
station.add_component(CCL)

from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman import UHFQC_RO_LutMan
RO_lutman = UHFQC_RO_LutMan('RO_lutman', num_res=7)
RO_lutman.AWG(UHFQC.name)
station.add_component(RO_lutman)

# making full DIO triggered lookuptable
rolut = station.components['RO_lutman']
UHFQC = station.components['UHFQC']
CCL = station.components['CCL']

# This load the CCL sequence that triggers consecutive readouts
example_fp = os.path.abspath(
    os.path.join(pq.__path__[0], '..', 'examples', 'CCLight_example',
                 'qisa_test_assembly', 'UHFQC_ro_example.qisa'))
CCL.eqasm_program(example_fp)
CCL.start()

amps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
rolut.sampling_rate(1.8e9)
# testing single pulse sequence
for i in range(5):
    rolut.set('M_amp_R{}'.format(i), amps[i])
    rolut.set('M_phi_R{}'.format(i), -45)

    rolut.set('M_down_amp0_R{}'.format(i), amps[i] / 2)
    rolut.set('M_down_amp1_R{}'.format(i), -amps[i] / 2)
    rolut.set('M_down_phi0_R{}'.format(i), -45 + 180)
    rolut.set('M_down_phi1_R{}'.format(i), -45)

    rolut.set('M_length_R{}'.format(i), 500e-9)
    rolut.set('M_down_length0_R{}'.format(i), 200e-9)
    rolut.set('M_down_length1_R{}'.format(i), 200e-9)
    rolut.set('M_modulation_R{}'.format(i), 0)

rolut.acquisition_delay(200e-9)

UHFQC.qas_0_result_length(1)
UHFQC.qas_0_integration_length(int(600e-9 * 1.8e9))
rolut.AWG('UHFQC')
rolut.generate_standard_waveforms()
rolut.pulse_type('M_up_down_down')
rolut.resonator_combinations([[0], [1], [2], [3], [4], [5], [6]])
UHFQC.awgs_0_userregs_0(1024)

UHFQC.awgs_0_enable(0)
rolut.load_DIO_triggered_sequence_onto_UHFQC()
UHFQC.awgs_0_enable(1)