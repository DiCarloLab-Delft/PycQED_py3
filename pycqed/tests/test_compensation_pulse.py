import numpy as np
import sys
import logging
from copy import deepcopy
import matplotlib.pyplot as plt
import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulse as bpl  # base pulse lib
import pycqed.measurement.waveform_control.pulsar as ps
import pycqed.measurement.waveform_control.segment as segment
import pycqed.measurement.waveform_control.sequence as sequence
from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014
import unittest
import qcodes as qc

AWG1 = VirtualAWG5014(name='AWG1')
AWGm = VirtualAWG5014(name='AWGm')
station = qc.Station()
pulsar = ps.Pulsar()
station.add_component(AWG1)
station.add_component(pulsar)
station.pulsar = pulsar

delays = {'AWG1': 300e-9, 'AWGm': 0}
offset_mode = {'AWG1': 'hardware', 'AWGm': 'hardware'}

for AWG in [AWG1, AWGm]:
    pulsar.define_awg_channels(AWG)
    pulsar.set('{}_delay'.format(AWG.name), delays[AWG.name])
    pulsar.set('{}_compensation_pulse_min_length'.format(AWG.name),
               4 / (1.2e9))
    for ch in pulsar.find_awg_channels(AWG.name):
        if pulsar.get('{}_type'.format(ch)) != 'analog':
            continue
        pulsar.set('{}_offset_mode'.format(ch), offset_mode[AWG.name])
        pulsar.set('{}_compensation_pulse_delay'.format(ch), 4 / (1.2e9))
        pulsar.set('{}_charge_buildup_compensation'.format(ch), True)

pulsar.AWG1_trigger_channels(['AWGm_ch1m1'])
pulsar.master_awg(AWGm.name)


class TestChargeCompensation(unittest.TestCase):
    def tearDown(self):
        del self.seg
        del self.seq

    def test_compensation_5k(self):
        seq_name = 'test_sequence'
        self.seq = sequence.Sequence(seq_name, station.pulsar)

        Square_pulse_pars = {
            "reference_pulse": "segment_start",
            "pulse_type": "SquarePulse",
            "pulse_delay": 25e-9,
            "channels": ["AWG1_ch1"],
            "length": 25e-9,
            "element_name": "element0",
            "amplitude": 1
        }

        pulses = [Square_pulse_pars]

        for i in range(83):
            pulse_pars2 = deepcopy(Square_pulse_pars)
            pulse_pars2['reference_pulse'] = 'previous_pulse'
            pulse_pars2['element_name'] = 'element{}'.format(int(i / 5) + 1)
            pulses += [pulse_pars2]

        self.seg = segment.Segment('segment', station.pulsar, pulses)
        self.seq.add(self.seg)

        pulsar.program_awgs(self.seq)
        cids = []
        for i in range(1, 5):
            cids.append('ch{}'.format(i))

        for cid in cids:
            ydata = 0
            try:
                for seg in range(len(AWG1.file['names'][int(cid[2]) - 1])):
                    el_name = AWG1.file['names'][int(cid[2]) - 1][seg]
                    pwfs = AWG1.file['p_wfs'][el_name]
                    ydata += np.sum(
                        (np.float_(np.bitwise_and(pwfs, 16383)) - 8191) / 8191)
            except IndexError:
                continue
            print(abs(ydata))
            # 8 Bit amplitude values --> 1/(*1.2*1e9) = 1e-9
            self.assertGreater(1, abs(ydata))

    def test_compensation_SBB_DRAG(self):
        seq_name = 'test_sequence'
        self.seq = sequence.Sequence(seq_name, station.pulsar)

        SSB_pulse_pars = {
            "pulse_type": "SSB_DRAG_pulse",
            "reference_pulse": "segment_start",
            "pulse_delay": 1e-8,
            "I_channel": "AWG1_ch1",
            "Q_channel": "AWG1_ch2",
            "sigma": 50e-9,
            "nr_sigma": 5,
            "mod_frequency": 100e6,
            "amp90_scale": 0.5,
            "phase": 0,
            "element_name": "element0",
        }

        pulses = [SSB_pulse_pars]

        for i in range(83):
            pulse_pars = deepcopy(SSB_pulse_pars)
            pulse_pars['reference_pulse'] = 'previous_pulse'
            pulse_pars['element_name'] = "element{}".format(int(i / 5) + 1)
            pulses += [pulse_pars]

        self.seg = segment.Segment('segment', station.pulsar, pulses)
        self.seq.add(self.seg)

        pulsar.program_awgs(self.seq)
        cids = []
        for i in range(1, 5):
            cids.append('ch{}'.format(i))

        for cid in cids:
            ydata = 0
            try:
                for seg in range(len(AWG1.file['names'][int(cid[2]) - 1])):
                    el_name = AWG1.file['names'][int(cid[2]) - 1][seg]
                    pwfs = AWG1.file['p_wfs'][el_name]
                    ydata += np.sum(
                        (np.float_(np.bitwise_and(pwfs, 16383)) - 8191) / 8191)
            except IndexError:
                continue
            print(abs(ydata))
            # 8 Bit amplitude values --> 1/(*1.2*1e9) = 1e-9
            self.assertGreater(1, abs(ydata))

    def test_compensation_500k(self):
        '''
        Starts with segment of length 24k samples and linearly increases the
        length up to 500k samples.
        '''

        cids = []
        for i in range(1, 5):
            cids.append('ch{}'.format(i))

        errors = {}
        for cid in cids:
            errors[cid] = []
        for j in range(1, 11):
            seq_name = 'test_sequence'
            self.seq = sequence.Sequence(seq_name, station.pulsar)

            # 25e-9 equals 30 samples
            Square_pulse_pars = {
                "reference_pulse": "segment_start",
                "pulse_type": "SquarePulse",
                "pulse_delay": j * 25e-9,
                "channels": ["AWG1_ch1"],
                "length": j * 25e-9,
                "element_name": "element0",
                "amplitude": 1
            }

            pulses = [Square_pulse_pars]

            for i in range(400):
                pulse_pars2 = deepcopy(Square_pulse_pars)
                pulse_pars2['reference_pulse'] = 'previous_pulse'
                pulse_pars2['element_name'] = 'element{}'.format(
                    int(i / 5) + 1)
                pulses += [pulse_pars2]

            self.seg = segment.Segment('segment', station.pulsar, pulses)
            self.seq.add(self.seg)

            pulsar.program_awgs(self.seq)

            # dictionary containg the errors for various segment lengths
            # and cids

            for cid in cids:
                ydata = 0
                try:
                    for seg in range(len(AWG1.file['names'][int(cid[2]) - 1])):
                        el_name = AWG1.file['names'][int(cid[2]) - 1][seg]
                        pwfs = AWG1.file['p_wfs'][el_name]
                        ydata += np.sum(
                            (np.float_(np.bitwise_and(pwfs, 16383)) - 8191) /
                            8191)
                except IndexError:
                    continue

                print(abs(ydata))
                self.assertGreater(20, abs(ydata))
                errors[cid].append(abs(ydata))

        for cid in cids:
            plt.plot(errors[cid])
