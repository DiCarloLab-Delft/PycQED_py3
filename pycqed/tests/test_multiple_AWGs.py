import unittest

import qcodes as qc
from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014
import pycqed.measurement.waveform_control.pulsar as ps
import pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts as sqs


default_pulse_pars = {
    'I_channel': 'ch1',
    'Q_channel': 'ch2',
    'amplitude': 0.25,
    'amp90_scale': 0.5,
    'sigma': 10e-9,
    'nr_sigma': 4,
    'motzoi': 0,
    'mod_frequency': -100e6,
    'pulse_delay': 0,
    'phi_skew': 0,
    'alpha': 1,
    'phase': 0,
    'operation_type': 'MW',
    'target_qubit': 'qubit1',
    'pulse_type': 'SSB_DRAG_pulse'}

default_RO_pars = {
    'I_channel': 'ch3',
    'Q_channel': 'ch4',
    'RO_pulse_marker_channel': 'ch3_marker1',
    'amplitude': 0.5,
    'length': 500e-9,
    'pulse_delay': 0,
    'mod_frequency': 25e6,
    'acq_marker_delay': 0,
    'acq_marker_channel': 'ch4_marker2',
    'phase': 0,
    'operation_type': 'RO',
    'target_qubit': 'qubit1',
    'pulse_type': 'Gated_MW_RO_pulse'}

default_sequencer_config = {
    'RO_fixed_point': 1e-6,
    'Buffer_Flux_Flux': 0,
    'Buffer_Flux_MW': 0,
    'Buffer_Flux_RO': 0,
    'Buffer_MW_Flux': 0,
    'Buffer_MW_MW': 0,
    'Buffer_MW_RO': 0,
    'Buffer_RO_Flux': 0,
    'Buffer_RO_MW': 0,
    'Buffer_RO_RO': 0,
    'Flux_comp_dead_time': 3e-6,
    'slave_AWG_trig_channels': [],
}

class TestMultipleAWGs(unittest.TestCase):

    def test_with_single_AWG(self):
        self.station = qc.Station()
        self.station.sequencer_config = default_sequencer_config.copy()
        sqs.station = self.station
        self.AWG = VirtualAWG5014("AWG")
        #self.AWG = tek.Tektronix_AWG5014(
        #    name='AWG', timeout=20,
        #    address='TCPIP0::192.168.1.15::inst0::INSTR', server_name=None)

        self.station.add_component(self.AWG)
        self.station.pulsar = ps.Pulsar('Pulsar1', default_AWG=self.AWG.name)
        for i in range(4):
            self.station.pulsar.define_channel(id='ch{}'.format(i+1),
                                               name='ch{}'.format(i+1),
                                               type='analog', high=1., low=-1.,
                                               offset=0.0, delay=0, active=True)
            self.station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                               name='ch{}_marker1'.format(i+1),
                                               type='marker', high=2, low=0,
                                               offset=0., delay=0, active=True)
            self.station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                               name='ch{}_marker2'.format(i+1),
                                               type='marker', high=2, low=0,
                                               offset=0., delay=0, active=True)

        self.pulse_pars = default_pulse_pars.copy()
        self.RO_pars = default_RO_pars.copy()

        sqs.Rabi_seq([0.3, 0.6], self.pulse_pars, self.RO_pars)

        pwfs = self.AWG.file['p_wfs']
        self.assertEqual(max(pwfs['2-pulse-elt_0_ch1']), 10286)
        self.assertEqual(min(pwfs['2-pulse-elt_0_ch1']), 6384)
        self.assertEqual(max(pwfs['2-pulse-elt_0_ch2']), 10211)
        self.assertEqual(min(pwfs['2-pulse-elt_0_ch2']), 6171)
        self.assertEqual(max(pwfs['2-pulse-elt_0_ch3']), 24575)
        self.assertEqual(min(pwfs['2-pulse-elt_0_ch3']), 8191)
        self.assertEqual(max(pwfs['2-pulse-elt_0_ch4']), 40959)
        self.assertEqual(min(pwfs['2-pulse-elt_0_ch4']), 8191)
        self.assertEqual(max(pwfs['2-pulse-elt_1_ch1']), 12382)
        self.assertEqual(min(pwfs['2-pulse-elt_1_ch1']), 4578)
        self.assertEqual(max(pwfs['2-pulse-elt_1_ch2']), 12231)
        self.assertEqual(min(pwfs['2-pulse-elt_1_ch2']), 4151)
        self.assertEqual(max(pwfs['2-pulse-elt_1_ch3']), 24575)
        self.assertEqual(min(pwfs['2-pulse-elt_1_ch3']), 8191)
        self.assertEqual(max(pwfs['2-pulse-elt_1_ch4']), 40959)
        self.assertEqual(min(pwfs['2-pulse-elt_1_ch4']), 8191)

    def test_with_multiple_AWGs(self):
        self.station = qc.Station()
        sqs.station = self.station
        self.station.sequencer_config = default_sequencer_config.copy()
        self.AWG1 = VirtualAWG5014("AWG1")
        self.AWG2 = VirtualAWG5014("AWG2")
        self.station.add_component(self.AWG1)
        self.station.add_component(self.AWG2)
        self.station.pulsar = ps.Pulsar('Pulsar2')
        for i in range(1,3):
            for j in range(1,5):
                self.station.pulsar.define_channel(
                    id='ch{}'.format(j),
                    name='AWG{} ch{}'.format(i, j),
                    type='analog', high=1., low=-1.,
                    offset=0.0, delay=0, active=True,
                    AWG='AWG{}'.format(i))
                self.station.pulsar.define_channel(
                    id='ch{}_marker1'.format(j),
                    name='AWG{} ch{}_marker1'.format(i, j),
                    type='marker', high=2, low=0,
                    offset=0., delay=0, active=True,
                    AWG='AWG{}'.format(i))
                self.station.pulsar.define_channel(
                    id='ch{}_marker2'.format(j),
                    name='AWG{} ch{}_marker2'.format(i, j),
                    type='marker', high=2, low=0,
                    offset=0., delay=0, active=True,
                    AWG='AWG{}'.format(i))

        self.pulse_pars = default_pulse_pars.copy()
        self.RO_pars = default_RO_pars.copy()

        self.pulse_pars['I_channel'] = 'AWG1 ch1'
        self.pulse_pars['Q_channel'] = 'AWG2 ch2'

        self.RO_pars['RO_pulse_marker_channel'] = 'AWG1 ch3_marker1'
        self.RO_pars['acq_marker_channel'] = 'AWG2 ch4_marker2'

        sqs.Rabi_seq([0.3, 0.6], self.pulse_pars, self.RO_pars)

        pwfs1 = self.AWG1.file['p_wfs']
        self.assertEqual(max(pwfs1['2-pulse-elt_0_ch1']), 10286)
        self.assertEqual(min(pwfs1['2-pulse-elt_0_ch1']), 6384)
        self.assertEqual(max(pwfs1['2-pulse-elt_0_ch2']), 8191)
        self.assertEqual(min(pwfs1['2-pulse-elt_0_ch2']), 8191)
        self.assertEqual(max(pwfs1['2-pulse-elt_0_ch3']), 24575)
        self.assertEqual(min(pwfs1['2-pulse-elt_0_ch3']), 8191)
        self.assertEqual(max(pwfs1['2-pulse-elt_0_ch4']), 8191)
        self.assertEqual(min(pwfs1['2-pulse-elt_0_ch4']), 8191)
        self.assertEqual(max(pwfs1['2-pulse-elt_1_ch1']), 12382)
        self.assertEqual(min(pwfs1['2-pulse-elt_1_ch1']), 4578)
        self.assertEqual(max(pwfs1['2-pulse-elt_1_ch2']), 8191)
        self.assertEqual(min(pwfs1['2-pulse-elt_1_ch2']), 8191)
        self.assertEqual(max(pwfs1['2-pulse-elt_1_ch3']), 24575)
        self.assertEqual(min(pwfs1['2-pulse-elt_1_ch3']), 8191)
        self.assertEqual(max(pwfs1['2-pulse-elt_1_ch4']), 8191)
        self.assertEqual(min(pwfs1['2-pulse-elt_1_ch4']), 8191)

        pwfs2 = self.AWG2.file['p_wfs']
        self.assertEqual(max(pwfs2['2-pulse-elt_0_ch1']), 8191)
        self.assertEqual(min(pwfs2['2-pulse-elt_0_ch1']), 8191)
        self.assertEqual(max(pwfs2['2-pulse-elt_0_ch2']), 10211)
        self.assertEqual(min(pwfs2['2-pulse-elt_0_ch2']), 6171)
        self.assertEqual(max(pwfs2['2-pulse-elt_0_ch3']), 8191)
        self.assertEqual(min(pwfs2['2-pulse-elt_0_ch3']), 8191)
        self.assertEqual(max(pwfs2['2-pulse-elt_0_ch4']), 40959)
        self.assertEqual(min(pwfs2['2-pulse-elt_0_ch4']), 8191)
        self.assertEqual(max(pwfs2['2-pulse-elt_1_ch1']), 8191)
        self.assertEqual(min(pwfs2['2-pulse-elt_1_ch1']), 8191)
        self.assertEqual(max(pwfs2['2-pulse-elt_1_ch2']), 12231)
        self.assertEqual(min(pwfs2['2-pulse-elt_1_ch2']), 4151)
        self.assertEqual(max(pwfs2['2-pulse-elt_1_ch3']), 8191)
        self.assertEqual(min(pwfs2['2-pulse-elt_1_ch3']), 8191)
        self.assertEqual(max(pwfs2['2-pulse-elt_1_ch4']), 40959)
        self.assertEqual(min(pwfs2['2-pulse-elt_1_ch4']), 8191)
