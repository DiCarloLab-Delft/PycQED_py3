import unittest
import numpy as np

import qcodes as qc
from qcodes.instrument.base import Instrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
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


class VirtualAWG5014(Tektronix_AWG5014):

    def __init__(self, name):
        Instrument.__init__(self, name)
        self.add_parameter('timeout', unit='s', initial_value=5,
                           parameter_class=ManualParameter,
                           vals=vals.MultiType(vals.Numbers(min_value=0),
                                               vals.Enum(None)))

        for i in range(1, 5):
            self.add_parameter('ch{}_state'.format(i), initial_value=1,
                               label='Status channel {}'.format(i),
                               parameter_class=ManualParameter,
                               vals=vals.Ints(0, 1))
            self.add_parameter('ch{}_amp'.format(i), initial_value=1,
                               label='Amplitude channel {}'.format(i),
                               unit='Vpp', parameter_class=ManualParameter,
                               vals=vals.Numbers(0.02, 4.5))
            self.add_parameter('ch{}_offset'.format(i), initial_value=0,
                               label='Offset channel {}'.format(i), unit='V',
                               parameter_class=ManualParameter,
                               vals=vals.Numbers(-.1, .1))
            self.add_parameter('ch{}_waveform'.format(i), initial_value="",
                               label='Waveform channel {}'.format(i),
                               parameter_class=ManualParameter,
                               vals=vals.Strings())
            self.add_parameter('ch{}_direct_output'.format(i), initial_value=1,
                               label='Direct output channel {}'.format(i),
                               parameter_class=ManualParameter,
                               vals=vals.Ints(0, 1))
            self.add_parameter('ch{}_filter'.format(i), initial_value='INF',
                               label='Low pass filter channel {}'.format(i),
                               unit='Hz', parameter_class=ManualParameter,
                               vals=vals.Enum(20e6, 100e6, 9.9e37, 'INF',
                                              'INFinity'))
            self.add_parameter('ch{}_DC_out'.format(i), initial_value=0,
                               label='DC output level channel {}'.format(i),
                               unit='V', parameter_class=ManualParameter,
                               vals=vals.Numbers(-3, 5))

            for j in range(1, 3):
                self.add_parameter(
                    'ch{}_m{}_del'.format(i, j), initial_value=0,
                    label='Channel {} Marker {} delay'.format(i, j),
                    unit='ns', parameter_class=ManualParameter,
                    vals=vals.Numbers(0, 1))
                self.add_parameter(
                    'ch{}_m{}_high'.format(i, j), initial_value=2,
                    label='Channel {} Marker {} high level'.format(i, j),
                    unit='V', parameter_class=ManualParameter,
                    vals=vals.Numbers(-2.7, 2.7))
                self.add_parameter(
                    'ch{}_m{}_low'.format(i, j), initial_value=0,
                    label='Channel {} Marker {} low level'.format(i, j),
                    unit='V', parameter_class=ManualParameter,
                    vals=vals.Numbers(-2.7, 2.7))

        self.add_parameter('clock_freq', label='Clock frequency',
                           unit='Hz', vals=vals.Numbers(1e6, 1.2e9),
                           parameter_class=ManualParameter, initial_value=1.2e9)

        self.awg_files = {}
        self.file = None

    def stop(self):
        pass

    def pack_waveform(self, wf, m1, m2):
        # Input validation
        if (not((len(wf) == len(m1)) and ((len(m1) == len(m2))))):
            raise Exception('error: sizes of the waveforms do not match')
        if min(wf) < -1 or max(wf) > 1:
            raise TypeError('Waveform values out of bonds.' +
                            ' Allowed values: -1 to 1 (inclusive)')
        if (list(m1).count(0)+list(m1).count(1)) != len(m1):
            raise TypeError('Marker 1 contains invalid values.' +
                            ' Only 0 and 1 are allowed')
        if (list(m2).count(0)+list(m2).count(1)) != len(m2):
            raise TypeError('Marker 2 contains invalid values.' +
                            ' Only 0 and 1 are allowed')

        wflen = len(wf)
        packed_wf = np.zeros(wflen, dtype=np.uint16)
        packed_wf += np.uint16(np.round(wf * 8191) + 8191 +
                               np.round(16384 * m1) +
                               np.round(32768 * m2))
        if len(np.where(packed_wf == -1)[0]) > 0:
            print(np.where(packed_wf == -1))
        return packed_wf

    def generate_awg_file(self, packed_waveforms, wfname_l, nrep_l, wait_l,
                          goto_l, logic_jump_l, channel_cfg, sequence_cfg=None,
                          preservechannelsettings=False):
        return {
            'p_wfs': packed_waveforms,
            'names': wfname_l,
            'nreps': nrep_l,
            'waits': wait_l,
            'gotos': goto_l,
            'jumps': logic_jump_l,
            'chans': channel_cfg
        }
    def send_awg_file(self, filename, awg_file):
        self.awg_files[filename] = awg_file

    def load_awg_file(self, filename):
        self.file = self.awg_files[filename]

    def __str__(self):
        if self.file is None:
            return "{}: no file loaded".format(self.name)
        else:
            return """{}:
    wf_names: {}
              {}
              {}
              {}""".format(self.name, self.file['names'][0],
                           self.file['names'][1], self.file['names'][2],
                           self.file['names'][3])

    def is_awg_ready(self):
        return True

class TestMultipleAWGs(unittest.TestCase):

    def test_with_single_AWG(self):
        self.station = qc.Station()
        sqs.station = self.station
        self.AWG = VirtualAWG5014("AWG")
        #self.AWG = tek.Tektronix_AWG5014(
        #    name='AWG', timeout=20,
        #    address='TCPIP0::192.168.1.15::inst0::INSTR', server_name=None)

        self.station.add_component(self.AWG)
        self.station.pulsar = ps.Pulsar('Pulsar1', default_AWG=self.AWG.name)
        for i in range(4):
            self.station.pulsar.define_channel(cid='ch{}'.format(i+1),
                                               name='ch{}'.format(i+1),
                                               type='analog', high=.7, low=-.7,
                                               offset=0.0, delay=0, active=True)
            self.station.pulsar.define_channel(cid='ch{}_marker1'.format(i+1),
                                               name='ch{}_marker1'.format(i+1),
                                               type='marker', high=2, low=0,
                                               offset=0., delay=0, active=True)
            self.station.pulsar.define_channel(cid='ch{}_marker2'.format(i+1),
                                               name='ch{}_marker2'.format(i+1),
                                               type='marker', high=2, low=0,
                                               offset=0., delay=0, active=True)

        self.pulse_pars = default_pulse_pars.copy()
        self.RO_pars = default_RO_pars.copy()

        sqs.Rabi_seq([0.3, 0.6], self.pulse_pars, self.RO_pars)

        self.assertEqual(max(self.AWG.file['p_wfs']['2-pulse-elt_0_ch1']),
                         12369)
        self.assertEqual(min(self.AWG.file['p_wfs']['2-pulse-elt_0_ch1']), 4591)
        self.assertEqual(max(self.AWG.file['p_wfs']['2-pulse-elt_1_ch1']),
                         16382)
        self.assertEqual(min(self.AWG.file['p_wfs']['2-pulse-elt_1_ch1']), 990)
        self.assertEqual(max(self.AWG.file['p_wfs']['2-pulse-elt_0_ch2']),
                         12072)
        self.assertEqual(min(self.AWG.file['p_wfs']['2-pulse-elt_0_ch2']), 4310)
        self.assertEqual(max(self.AWG.file['p_wfs']['2-pulse-elt_1_ch2']),
                         15953)
        self.assertEqual(min(self.AWG.file['p_wfs']['2-pulse-elt_1_ch2']), 429)
        self.assertEqual(max(self.AWG.file['p_wfs']['2-pulse-elt_0_ch3']),
                         24575)
        self.assertEqual(min(self.AWG.file['p_wfs']['2-pulse-elt_0_ch3']), 8191)
        self.assertEqual(max(self.AWG.file['p_wfs']['2-pulse-elt_1_ch3']),
                         24575)
        self.assertEqual(min(self.AWG.file['p_wfs']['2-pulse-elt_1_ch3']), 8191)
        self.assertEqual(max(self.AWG.file['p_wfs']['2-pulse-elt_0_ch4']),
                         40959)
        self.assertEqual(min(self.AWG.file['p_wfs']['2-pulse-elt_0_ch4']), 8191)
        self.assertEqual(max(self.AWG.file['p_wfs']['2-pulse-elt_1_ch4']),
                         40959)
        self.assertEqual(min(self.AWG.file['p_wfs']['2-pulse-elt_1_ch4']), 8191)

    def test_with_multiple_AWGs(self):
        self.station = qc.Station()
        sqs.station = self.station
        self.AWG1 = VirtualAWG5014("AWG1")
        self.AWG2 = VirtualAWG5014("AWG2")
        self.station.add_component(self.AWG1)
        self.station.add_component(self.AWG2)
        self.station.pulsar = ps.Pulsar('Pulsar2')
        for i in range(1,3):
            for j in range(1,5):
                self.station.pulsar.define_channel(
                    cid='ch{}'.format(j),
                    name='AWG{} ch{}'.format(i, j),
                    type='analog', high=.7, low=-.7,
                    offset=0.0, delay=0, active=True,
                    AWG='AWG{}'.format(i))
                self.station.pulsar.define_channel(
                    cid='ch{}_marker1'.format(j),
                    name='AWG{} ch{}_marker1'.format(i, j),
                    type='marker', high=2, low=0,
                    offset=0., delay=0, active=True,
                    AWG='AWG{}'.format(i))
                self.station.pulsar.define_channel(
                    cid='ch{}_marker2'.format(j),
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

        # AWG1
        self.assertEqual(max(self.AWG1.file['p_wfs']['2-pulse-elt_0_ch1']),
                         12369)
        self.assertEqual(min(self.AWG1.file['p_wfs']['2-pulse-elt_0_ch1']),
                         4591)
        self.assertEqual(max(self.AWG1.file['p_wfs']['2-pulse-elt_1_ch1']),
                         16382)
        self.assertEqual(min(self.AWG1.file['p_wfs']['2-pulse-elt_1_ch1']), 990)
        self.assertEqual(max(self.AWG1.file['p_wfs']['2-pulse-elt_0_ch2']),
                         8191)
        self.assertEqual(min(self.AWG1.file['p_wfs']['2-pulse-elt_0_ch2']),
                         8191)
        self.assertEqual(max(self.AWG1.file['p_wfs']['2-pulse-elt_1_ch2']),
                         8191)
        self.assertEqual(min(self.AWG1.file['p_wfs']['2-pulse-elt_1_ch2']),
                         8191)
        self.assertEqual(max(self.AWG1.file['p_wfs']['2-pulse-elt_0_ch3']),
                         24575)
        self.assertEqual(min(self.AWG1.file['p_wfs']['2-pulse-elt_0_ch3']),
                         8191)
        self.assertEqual(max(self.AWG1.file['p_wfs']['2-pulse-elt_1_ch3']),
                         24575)
        self.assertEqual(min(self.AWG1.file['p_wfs']['2-pulse-elt_1_ch3']),
                         8191)
        self.assertEqual(max(self.AWG1.file['p_wfs']['2-pulse-elt_0_ch4']),
                         8191)
        self.assertEqual(min(self.AWG1.file['p_wfs']['2-pulse-elt_0_ch4']),
                         8191)
        self.assertEqual(max(self.AWG1.file['p_wfs']['2-pulse-elt_1_ch4']),
                         8191)
        self.assertEqual(min(self.AWG1.file['p_wfs']['2-pulse-elt_1_ch4']),
                         8191)

        # AWG2
        self.assertEqual(max(self.AWG2.file['p_wfs']['2-pulse-elt_0_ch1']),
                         8191)
        self.assertEqual(min(self.AWG2.file['p_wfs']['2-pulse-elt_0_ch1']),
                         8191)
        self.assertEqual(max(self.AWG2.file['p_wfs']['2-pulse-elt_1_ch1']),
                         8191)
        self.assertEqual(min(self.AWG2.file['p_wfs']['2-pulse-elt_1_ch1']),
                         8191)
        self.assertEqual(max(self.AWG2.file['p_wfs']['2-pulse-elt_0_ch2']),
                         12072)
        self.assertEqual(min(self.AWG2.file['p_wfs']['2-pulse-elt_0_ch2']),
                         4310)
        self.assertEqual(max(self.AWG2.file['p_wfs']['2-pulse-elt_1_ch2']),
                         15953)
        self.assertEqual(min(self.AWG2.file['p_wfs']['2-pulse-elt_1_ch2']), 429)
        self.assertEqual(max(self.AWG2.file['p_wfs']['2-pulse-elt_0_ch3']),
                         8191)
        self.assertEqual(min(self.AWG2.file['p_wfs']['2-pulse-elt_0_ch3']),
                         8191)
        self.assertEqual(max(self.AWG2.file['p_wfs']['2-pulse-elt_1_ch3']),
                         8191)
        self.assertEqual(min(self.AWG2.file['p_wfs']['2-pulse-elt_1_ch3']),
                         8191)
        self.assertEqual(max(self.AWG2.file['p_wfs']['2-pulse-elt_0_ch4']),
                         40959)
        self.assertEqual(min(self.AWG2.file['p_wfs']['2-pulse-elt_0_ch4']),
                         8191)
        self.assertEqual(max(self.AWG2.file['p_wfs']['2-pulse-elt_1_ch4']),
                         40959)
        self.assertEqual(min(self.AWG2.file['p_wfs']['2-pulse-elt_1_ch4']),
                         8191)