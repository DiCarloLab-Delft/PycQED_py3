import numpy as np
import matplotlib.pyplot as plt

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014


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
            self.add_parameter('ch{}_amp'.format(i), initial_value=1.,
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
        if not ((len(wf) == len(m1)) and ((len(m1) == len(m2)))):
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

    def plot_waveforms(self, seg=0, cids='all'):
        if cids == 'all':
            cids = []
            for i in range(1, 5):
                cids.append('ch{}'.format(i))
                cids.append('ch{}_marker1'.format(i))
                cids.append('ch{}_marker2'.format(i))

        fig, axs = plt.subplots(len(cids), 1, sharex=True)

        i = 0
        for cid in cids:
            el_name = self.file['names'][int(cid[2])-1][seg]
            pwfs = self.file['p_wfs'][el_name]

            ax = axs[i]
            i += 1

            if cid[4:] == 'marker1':
                ydata = np.float_((pwfs // 16384) % 2)
            elif cid[4:] == 'marker2':
                ydata = np.float_((pwfs // 32768) % 2)
            else:
                ydata = (np.float_(np.bitwise_and(pwfs, 16383)) - 8191)/8191
            xdata = np.arange(len(ydata))/self.clock_freq()

            ax.plot(xdata, ydata)
            ax.set_ylabel(cid)
            if i == len(cids):
                ax.set_xlabel('Time')
