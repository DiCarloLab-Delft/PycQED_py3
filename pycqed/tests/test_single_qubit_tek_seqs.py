import numpy as np
import unittest

from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control import element
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sqs


class Test_SingleQubitTek(unittest.TestCase):

    def setUp(self):
        # set up a pulsar with some mock settings for the element
        self.pulsar = Pulsar()
        for i in range(4):
            self.pulsar.define_channel(id='ch{}'.format(i+1),
                                          name='ch{}'.format(i+1),
                                          type='analog',
                                          # max safe IQ voltage
                                          high=.7, low=-.7,
                                          offset=0.0, delay=0, active=True)
            self.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                          name='ch{}_marker1'.format(i+1),
                                          type='marker',
                                          high=2.0, low=0, offset=0.,
                                          delay=0, active=True)
            self.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                          name='ch{}_marker2'.format(i+1),
                                          type='marker',
                                          high=2.0, low=0, offset=0.,
                                          delay=0, active=True)

        self.pulse_pars = {
            'I_channel': 'ch1',
            'Q_channel': 'ch2',
            'amplitude': .5,
            'amp90_scale': .5,
            'sigma': 10e-9,
            'nr_sigma': 4,
            'motzoi': .8,
            'mod_frequency': 100e-6,
            'pulse_delay': 0,
            'phi_skew': 0,
            'alpha': 1,
            'phase': 0,
            'pulse_type': 'SSB_DRAG_pulse'}

        self.RO_pars = {
            'I_channel': 'ch3',
            'Q_channel': 'ch4',
            'RO_pulse_marker_channel': 'ch3_marker1',
            'amplitude': '.5',
            'length': 300e-9,
            'pulse_delay': 0,
            'mod_frequency': 50e6,
            'fixed_point_frequency': -50e6,
            'acq_marker_delay': 0,
            'acq_marker_channel': 'ch1_marker1',
            'phase': 0,
            'pulse_type': 'MW_IQmod_pulse_tek'}

        station = Bunch(pulsar=self.pulsar)
        sqs.station = station

    def test_ramsey_no_detuning(self):
        times = np.linspace(0, 5e-6, 41)
        f_fix_pt = self.RO_pars['fixed_point_frequency']

        # Sequence with no artificial detuning
        seq, el_list = sqs.Ramsey_seq(times, self.pulse_pars, self.RO_pars,
                                      artificial_detuning=None,
                                      cal_points=True,
                                      verbose=False,
                                      upload=False,
                                      return_seq=True)
        self.assertEqual(len(times), len(seq.elements))
        self.assertEqual(len(times), len(el_list))
        for i, el in enumerate(el_list):
            t_RO = el.effective_pulse_start_time('RO_tone-0', 'ch1')
            t_ROm = el.effective_pulse_start_time('Acq-trigger-0', 'ch1')
            self.assertAlmostEqual(t_RO, t_ROm, places=10)
            # test if fix point put pulses at the right spot.
            self.assertTrue(element.is_divisible_by_clock(t_RO, abs(f_fix_pt)))
            # Check pulse delay
            if i < (len(times)-4):
                t0 = el.effective_pulse_start_time('pulse_0-0', 'ch1')
                t1 = el.effective_pulse_start_time('pulse_1-0', 'ch1')
                self.assertAlmostEqual(t1-t0, times[i], places=10)
                p0 = el.pulses['pulse_0-0']
                self.assertEqual(p0.phase, 0)
                p1 = el.pulses['pulse_1-0']
                self.assertEqual(p1.phase, 0)
            else:
                # Calibration points do not have two pulses
                with self.assertRaises(KeyError):
                    t1 = el.effective_pulse_start_time('pulse_1-0', 'ch1')

    def test_ramsey_freq_detuning(self):
        times = np.linspace(0, 5e-6, 41)
        for f_fix_pt in [50e-6, -50e-6]:
            self.RO_pars['fixed_point_frequency'] = f_fix_pt
            for RO_pulse_type in ['Gated_MW_RO_pulse', 'MW_IQmod_pulse_tek']:
                self.RO_pars['pulse_type'] = RO_pulse_type
                f_detuning = 300e3  # 300 kHz detuning
                # Sequence with artificial detuning specified in Hz
                seq, el_list = sqs.Ramsey_seq(times, self.pulse_pars,
                                              self.RO_pars,
                                              artificial_detuning=f_detuning,
                                              cal_points=True,
                                              verbose=False,
                                              upload=False,
                                              return_seq=True)
                self.assertEqual(len(times), len(seq.elements))
                self.assertEqual(len(times), len(el_list))
                for i, el in enumerate(el_list):
                    if RO_pulse_type == 'MW_IQmod_pulse_tek':
                        t_RO = el.effective_pulse_start_time(
                            'RO_tone-0', 'ch1')
                    else:
                        t_RO = el.effective_pulse_start_time(
                            'RO_marker-0', 'ch1')
                    t_ROm = el.effective_pulse_start_time(
                        'Acq-trigger-0', 'ch1')
                    self.assertAlmostEqual(t_RO, t_ROm, places=10)

                    # test if fix point put pulses at the right spot.
                    self.assertTrue(
                        element.is_divisible_by_clock(t_RO, f_fix_pt))

                    # Check Ramsey pulse spacing
                    if i < (len(times)-4):
                        t0 = el.effective_pulse_start_time(
                            'pulse_0-0', 'ch1')
                        t1 = el.effective_pulse_start_time(
                            'pulse_1-0', 'ch1')
                        self.assertAlmostEqual(t1-t0, times[i], places=10)
                        p0 = el.pulses['pulse_0-0']
                        self.assertEqual(p0.phase, 0)
                        p1 = el.pulses['pulse_1-0']
                        exp_phase = (360*f_detuning*(t1-t0)) % 360
                        if exp_phase == 360:
                            exp_phase = 0
                        self.assertAlmostEqual(p1.phase, exp_phase, places=3)
                    else:
                        # Calibration points do not have two pulses
                        with self.assertRaises(KeyError):
                            t1 = el.effective_pulse_start_time(
                                'pulse_1-0', 'ch1')


class Bunch:

    def __init__(self, **kwds):
        self.__dict__.update(kwds)
