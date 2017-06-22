# Originally by Wolfgang Pfaff
# Modified by Adriaan Rol 9/2015
# Modified by Ants Remm 5/2017

import numpy as np
import logging
from qcodes.instrument.base import Instrument
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
import time
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.\
        UHFQuantumController import UHFQC
except ModuleNotFoundError:
    UHFQC = type(None)


# some pulses use rounding when determining the correct sample at which to
# insert a particular value. this might require correct rounding -- the pulses
# are typically specified on short time scales, but the time unit we use is
# seconds. therefore we need a suitably chosen digit on which to round. 9 would
# round a pulse to 1 ns precision. 11 is 10 ps, and therefore probably beyond
# the lifetime of this code (no 10ps AWG available yet :))
SIGNIFICANT_DIGITS = 11


class Pulsar(Instrument):
    """
    A meta-instrument responsible for all communication with the AWGs.
    Contains information about all the available awg-channels in the setup.
    Starting, stopping and programming and changing the parameters of the AWGs
    should be done through Pulsar. Supports Tektronix AWG5014 and ZI UHFLI.

    Args:
        default_AWG: Name of the AWG that new channels get defined on if no
                     AWG is specified
        master_AWG: Name of the AWG that triggers all the other AWG-s and
                    should be started last (after other AWG-s are already
                    waiting for a trigger.
    """
    def __init__(self, name='Pulsar', default_AWG=None, master_AWG=None):
        super().__init__(name)

        # for compatibility with old code, the default AWG name is stored in
        # self.AWG.name
        if default_AWG is not None:
            self.AWG = self.AWG_obj(AWG=default_AWG)
        else:
            class Object(object):
                pass
            self.AWG = Object()
            self.AWG.name = None

        self.add_parameter('default_AWG',
                           set_cmd=self._set_default_AWG,
                           get_cmd=self._get_default_AWG)
        self.add_parameter('master_AWG', parameter_class=InstrumentParameter,
                           initial_value=master_AWG)

        self.channels = {}
        self.last_sequence = None
        self.last_elements = None

    # channel handling
    def define_channel(self, id, name, type, delay, offset,
                       high, low, active, AWG=None):
        """
        The AWG object must be created before creating channels for that AWG

        Args:
            id: channel id. For the Tektronix 5014 must be of the form
                ch#(_marker#) with # a number and the part in () optional.
                For UHFQC must be 'ch1' or 'ch2'.
            name: This name must be specified in pulses for them to play on
                  this channel.
            type: marker/analog/readout
            delay: global delay applied to this channel (positive values move
                   pulses on this channel forward in time)
            offset: a (software implemented) offset voltage that is added to
                    all of the waveforms (analog channel only)
            high: maximal output value
            low: minimal output value
            active: whether this channel will be programmed
            AWG: name of the AWG this channel is on
        """
        if AWG is None:
            AWG = self.default_AWG()

        _doubles = []
        for c_name, c_dict in self.channels.items():
            if c_dict['id'] == id and c_dict['AWG'] == AWG:
                logging.warning("Channel '{}' on '{}' already in use, {} will "
                                "overwrite {}.".format(id, AWG, name, c_name))
                _doubles.append(c_name)
        for c in _doubles:
            del self.channels[c]

        self.channels[name] = {'id': id,
                               'type': type,
                               'delay': delay,
                               'offset': offset,
                               'high': high,
                               'low': low,
                               'active': active,
                               'AWG': AWG}

    def AWG_obj(self, **kw):
        """
        Return the AWG object corresponding to a channel or an AWG name.

        Args:
            AWG: Name of the AWG Instrument.
            channel: Name of the channel

        Returns: An instance of Instrument class corresponding to the AWG
                 requested.
        """
        AWG = kw.get('AWG', None)
        chan = kw.get('channel', None)
        if AWG is not None and chan is not None:
            raise ValueError('Both `AWG` and `channel` arguments passed to '
                             'Pulsar.AWG_obj()')
        elif AWG is None and chan is not None:
            name = self.channels[chan]['AWG']
        elif AWG is not None and chan is None:
            name = AWG
        else:
            raise ValueError('Either `AWG` or `channel` argument needs to be '
                             'passed to Pulsar.AWG_obj()')
        return Instrument.find_instrument(name)

    def clock(self, channel):
        """
        Returns the clock rate of channel `channel`
        Args:
            channel: name of the channel
        Returns: clock rate in samples per second
        """
        obj = self.AWG_obj(channel=channel)
        return obj.clock_freq()

    def channel_opt(self, name, option, value=None):
        """
        Convenience function to get or set a channel option.
        Args:
            name: Name of the channel.
            option: Name of the option. Available options:
                * 'id'
                * 'type'
                * 'delay'
                * 'offset'
                * 'high'
                * 'low'
                * 'active'
                * 'AWG'
            value: New value for the option.
        """
        if value is not None:
            self.channels[name][option] = value
        else:
            return self.channels[name][option]

    def used_AWGs(self):
        """
        Returns:
            A set of the names of the active AWGs registered
        """
        res = set()
        for cdict in self.channels.values():
            if cdict['active']:
                res.add(cdict['AWG'])
        return res

    def start(self):
        """
        Start the active AWGs.
        """
        if self.master_AWG() is None:
            for AWG in self.used_AWGs():
                self._start_AWG(AWG)
        else:
            for AWG in self.used_AWGs():
                if AWG != self.master_AWG():
                    self._start_AWG(AWG)
            time.sleep(0.2)  # wait 0.2 second for all other awg-s to start
            self._start_AWG(self.master_AWG())

    def stop(self):
        """
        Stop all active AWGs.
        """
        for AWG in self.used_AWGs():
            self._stop_AWG(AWG)

    def program_awgs(self, sequence, *elements, AWGs='all', channels='all',
                     loop=True, allow_first_nonzero=False, verbose=False):
        """
        Args:
            sequence: The `Sequence` object that determines the segment order,
                      repetition and trigger wait.
            *elements: The `Element` objects to program to the AWGs.
            AWGs: List of names of the AWGs to program. Default is 'all'.
            channels: List of names of the channels that should be programmed.
                      Default is `'all'`.
            loop: Boolean flag, whether the segments should be looped over.
                  Default is `True`.
            allow_first_nonzero: Boolean flag, whether to allow the first point
                                 of the element to be nonzero if the segment
                                 waits for a trigger. In Tektronix AWG5014,
                                 the output is set to the first value of the
                                 segment while waiting for the trigger. Default
                                 is `False`.
             verbose: Currently unused.
        """
        # Stores the last uploaded elements for easy access and plotting
        self.last_sequence = sequence
        self.last_elements = elements

        if AWGs == 'all':
            AWGs = self.used_AWGs()
        if channels == 'all':
            channels = self.channels.keys()

        # dict(name of AWG ->
        #      dict(element name ->
        #           dict(channel id ->
        #                waveform data)))
        AWG_wfs = {}

        for el in elements:
            tvals, waveforms = el.normalized_waveforms()
            for cname in waveforms:
                if cname not in channels:
                    continue
                if not self.channels[cname]['active']:
                    continue
                cAWG = self.channels[cname]['AWG']
                cid = self.channels[cname]['id']
                if cAWG not in AWGs:
                    continue
                if cAWG not in AWG_wfs:
                    AWG_wfs[cAWG] = {}
                if el.name not in AWG_wfs[cAWG]:
                    AWG_wfs[cAWG][el.name] = {}
                AWG_wfs[cAWG][el.name][cid] = waveforms[cname]

        self.update_AWG5014_settings()
        for AWG in AWG_wfs:
            obj = self.AWG_obj(AWG=AWG)
            if isinstance(obj, Tektronix_AWG5014):
                self._program_AWG5014(obj, sequence, AWG_wfs[AWG], loop=loop,
                                      allow_first_nonzero=allow_first_nonzero)
            elif isinstance(obj, UHFQC):
                self._program_UHFQC(obj, sequence, AWG_wfs[AWG], loop=loop,
                                    allow_first_nonzero=allow_first_nonzero)
            else:
                raise TypeError('Unsupported AWG instrument: {} of type {}'
                                .format(AWG, type(obj)))

    def _program_AWG5014(self, obj, sequence, el_wfs, loop=True,
                         allow_first_nonzero=False):
        """
        Program the AWG with a sequence of segments.

        Args:
            obj: the instance of the AWG to program
            sequence: the `Sequence` object that determines the segment order,
                      repetition and trigger wait
            el_wfs: A dictionary from element name to a dictionary from channel
                    id to the waveform.
            loop: Boolean flag, whether the segments should be looped over.
                  Default is `True`.
            allow_first_nonzero: Boolean flag, whether to allow the first point
                                 of the element to be nonzero if the segment
                                 waits for a trigger. In Tektronix AWG5014,
                                 the output is set to the first value of the
                                 segment while waiting for the trigger. Default
                                 is `False`.
        """

        old_timeout = obj.timeout()
        obj.timeout(max(180, old_timeout))

        # determine which channel groups are involved in the sequence
        grps = set()
        for cid_wfs in el_wfs.values():
            for cid in cid_wfs:
                grps.add(cid[:3])
        grps = list(grps)
        grps.sort()

        # create a packed waveform for each element for each channel group
        # in the sequence
        packed_waveforms = {}
        elements_with_non_zero_first_points = set()
        for el, cid_wfs in el_wfs.items():
            maxlen = 0
            for wf in cid_wfs.values():
                if len(wf) > maxlen:
                    maxlen = len(wf)
            for grp in grps:
                grp_wfs = {}
                # arrange waveforms from input data and pad with zeros for
                # equal length
                for cid in self._AWG5014_group_ids(grp):
                    grp_wfs[cid] = cid_wfs.get(cid, np.zeros(1))
                    cname = self._AWG5014_id_channel(cid, obj.name)
                    if cid[4:-1] == 'marker' or cname is None:
                        cval = 0
                    else:
                        cval = self.channels[cname]['offset']
                        hi = self.channels[cname]['high']
                        lo = self.channels[cname]['low']
                        cval = (2*cval - hi - lo)/(hi - lo)
                    grp_wfs[cid] = np.pad(grp_wfs[cid],
                                          (0, maxlen - len(grp_wfs[cid])),
                                          'constant',
                                          constant_values=cval)
                    if grp_wfs[cid][0] != 0.:
                        elements_with_non_zero_first_points.add(el)
                wfname = el + '_' + grp

                packed_waveforms[wfname] = obj.pack_waveform(
                    grp_wfs[grp],
                    grp_wfs[grp + '_marker1'],
                    grp_wfs[grp + '_marker2'])

        # sequence programming
        _t0 = time.time()
        if sequence.element_count() > 8000:
            logging.warning("Error: trying to program '{:s}' ({:d}'".format(
                            sequence.name, sequence.element_count()) +
                            " element(s))...\n Sequence contains more than " +
                            "8000 elements, Aborting", end=' ')
            return

        print("Programming {} sequence '{}' ({} element(s)) \t".format(
            obj.name, sequence.name, sequence.element_count()), end=' ')

        # Create lists with sequence information:
        # wfname_l = list of waveform names [[wf1_ch1,wf2_ch1..],
        #                                    [wf1_ch2,wf2_ch2..], ...]
        # nrep_l = list specifying the number of reps for each seq element
        # wait_l = idem for wait_trigger_state
        # goto_l = idem for goto_state (goto is the element where it hops to in
        # case the element is finished)

        wfname_l = []
        nrep_l = []
        wait_l = []
        goto_l = []
        logic_jump_l = []

        for grp in grps:
            grp_wfnames = []
            # add all wf names of channel
            for el in el_wfs:
                wfname = el + '_' + grp
                grp_wfnames.append(wfname)
            wfname_l.append(grp_wfnames)

        for el in sequence.elements:
            nrep_l.append(el['repetitions'])
            if (el['repetitions'] < 1) or (el['repetitions'] > 65536):
                raise Exception(
                    'Pulsar: The number of repetitions of AWG "{}" element "{}"'
                    ' are out of range. Valid range = 1 to 65536 ("{}" received'
                    ')'.format(obj.name, el['wfname'], el['repetitions'])
                )
            if el['goto_target'] is not None:
                goto_l.append(sequence.element_index(el['goto_target']))
            else:
                goto_l.append(0)
            if el['jump_target'] is not None:
                logic_jump_l.append(sequence.element_index(el['jump_target']))
            else:
                logic_jump_l.append(0)
            if el['trigger_wait']:
                wait_l.append(1)
                if el['wfname'] in elements_with_non_zero_first_points and \
                        not allow_first_nonzero:
                    raise Exception('Pulsar: Trigger wait set for element {} '
                                    'with a non-zero first point'.format(
                                        el['wfname']))
            else:
                wait_l.append(0)
        if loop and len(goto_l) > 0:
            goto_l[-1] = 1

        if len(wfname_l) > 0:
            filename = sequence.name + '_FILE.AWG'
            awg_file = obj.generate_awg_file(packed_waveforms,
                                             np.array(wfname_l), nrep_l, wait_l,
                                             goto_l, logic_jump_l,
                                             self._AWG5014_chan_cfg(obj.name))
            obj.send_awg_file(filename, awg_file)
            obj.load_awg_file(filename)
        else:
            awg_file = None

        obj.timeout(old_timeout)

        time.sleep(.1)
        # Waits for AWG to be ready
        obj.is_awg_ready()

        self._AWG5014_activate_channels(grps, obj.name)

        _t = time.time() - _t0
        print(" finished in {:.2f} seconds.".format(_t))
        return awg_file

    def _program_UHFQC(self, obj, sequence, el_wfs, loop=True,
                       allow_first_nonzero=False):
        header = """const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x1f0000;
setTrigger(WINT_EN);
var loop_cnt = getUserReg(0);
var RO_TRIG;
if (getUserReg(1)) {
  RO_TRIG=IAVG_TRIG;
}else{
  RO_TRIG=WINT_TRIG;
}
\n"""

        if loop:
            main_loop = 'while(1) {\n'
            footer = '}\n'
        else:
            main_loop = ''
            footer = ''
        main_loop += 'repeat (loop_cnt) {\n'

        footer += """}
wait(1000);
setTrigger(0);
"""

        # parse elements
        elements_with_non_zero_first_points = []
        wfnames = {'ch1': [], 'ch2': []}
        wfdata = {'ch1': [], 'ch2': []}
        i = 1
        for el in el_wfs:
            for cid in ['ch1', 'ch2']:

                if cid in el_wfs[el]:
                    wfname = el + '_' + cid
                    cid_wf = el_wfs[el][cid]
                    wfnames[cid].append(wfname)
                    wfdata[cid].append(cid_wf)
                    if cid_wf[0] != 0.:
                        elements_with_non_zero_first_points.append(el)
                    header += 'wave {} = ramp({}, 0, {});\n'.format(
                        wfname, len(cid_wf), 1 / i
                    )
                    i += 1
                else:
                    wfnames[cid].append(None)
                    wfdata[cid].append(None)

        # create waveform playback code
        for i, el in enumerate(sequence.elements):
            if el['goto_target'] is not None:
                raise NotImplementedError(
                    'UHFQC sequencer does not yet support nontrivial goto-s.')
            if el['jump_target'] is not None:
                raise NotImplementedError('UHFQC sequencer does not support'
                                          ' jump events.')
            if el['trigger_wait']:
                if el['wfname'] in elements_with_non_zero_first_points and \
                        not allow_first_nonzero:
                    raise Exception(
                        'Pulsar: Trigger wait set for element {} '
                        'with a non-zero first point'.format(el['wfname']))
            name_ch1 = wfnames['ch1'][i]
            name_ch2 = wfnames['ch2'][i]
            main_loop += self._UHFQC_element_seqc(el['repetitions'],
                                                  el['trigger_wait'],
                                                  name_ch1, name_ch2, True)
        awg_str = header + main_loop + footer
        obj.awg_string(awg_str)

        # populate the waveforms with data
        i = 0
        for data1, data2 in zip(wfdata['ch1'], wfdata['ch2']):
            if data1 is None and data2 is None:
                continue
            elif data1 is None:
                obj.awg_update_waveform(i, data2)
                i += 1
            elif data2 is None:
                obj.awg_update_waveform(i, data1)
                i += 1
            else:
                data12 = np.vstack((data1, data2,)).reshape((-1,), order='F')
                obj.awg_update_waveform(i, data12)
                i += 1

        return awg_str

    def _start_AWG(self, AWG):
        obj = self.AWG_obj(AWG=AWG)
        if isinstance(obj, Tektronix_AWG5014):
            obj.start()
        elif isinstance(obj, UHFQC):
            obj.acquisition_arm()
        else:
            raise ValueError('Unsupported AWG type: {}'.format(type(obj)))

    def _stop_AWG(self, AWG):
        obj = self.AWG_obj(AWG=AWG)
        if isinstance(obj, Tektronix_AWG5014):
            obj.stop()
        elif isinstance(obj, UHFQC):
            obj._daq.syncSetInt('/' + obj._device + '/awgs/0/enable', 0)
        else:
            raise ValueError('Unsupported AWG type: {}'.format(type(obj)))

    def _set_default_AWG(self, AWG):
        self.AWG = self.AWG_obj(AWG=AWG)

    def _get_default_AWG(self):
        return self.AWG.name

    ###################################
    # AWG5014 specific helper functions

    @staticmethod
    def update_channel_settings():
        logging.error('Pulsar.update_channel_settings() is deprecated with the'
                      ' multi-AWG support. Please update your code.')

    def update_AWG5014_settings(self, AWGs='all'):
        """
        Updates the AWG5014 parameters to the values in
        `self.channels`

        Args:
            AWGs: A list of AWG names to update or 'all'. Default 'all'.
        """
        for cname, cdict in self.channels.items():
            if AWGs == 'all' or cdict['AWG'] in AWGs:
                obj = self.AWG_obj(channel=cname)
                if not isinstance(obj, Tektronix_AWG5014):
                    continue
                if cdict['type'] == 'analog':
                    amp = cdict['high'] - cdict['low']
                    offset = (cdict['low'] + cdict['high'])/2
                    obj.set('{}_amp'.format(cdict['id']), amp)
                    obj.set('{}_offset'.format(cdict['id']), offset)
                else:  # c_dict['type'] == 'marker'
                    cid = cdict['id']
                    low_par = 'ch{}_m{}_low'.format(cid[2], cid[-1])
                    high_par = 'ch{}_m{}_high'.format(cid[2], cid[-1])
                    obj.set(low_par, cdict['low'])
                    obj.set(high_par, cdict['high'])

    @staticmethod
    def _AWG5014_group_ids(cid):
        """
        Returns all id-s corresponding to a single channel group.
        For example `Pulsar._AWG5014_group_ids('ch2')` returns `['ch2',
        'ch2_marker1', 'ch2_marker2']`.

        Args:
            cid: An id of one of the AWG5014 channels.

        Returns: A list of id-s corresponding to the same group as `cid`.
        """
        return [cid[:3], cid[:3] + '_marker1', cid[:3] + '_marker2']

    def _AWG5014_id_channel(self, cid, AWG):
        """
        Returns the channel name corresponding to the channel with id `cid` on
        the AWG `AWG`.

        Args:
            cid: An id of one of the AWG5014 channels.
            AWG: The name of the AWG.

        Returns: The corresponding channel name. If the channel is not found,
                 returns `None`.
        """
        for cname, cdict in self.channels.items():
            if cdict['AWG'] == AWG and cdict['id'] == cid:
                return cname
        return None

    def _AWG5014_activate_channels(self, grps, AWG):
        """
        Turns on AWG5014 channel groups.

        Args:
            grps: An iterable of channel group id-s to turn on.
            AWG: The name of the AWG.
        """
        for grp in grps:
            self.AWG_obj(AWG=AWG).set('{}_state'.format(grp), 1)

    def _AWG5014_chan_cfg(self, AWG):
        channel_cfg = {}
        for cdict in self.channels.values():
            if cdict['AWG'] != AWG:
                continue
            cid = cdict['id']
            if cdict['type'] == 'analog':
                channel_cfg['ANALOG_METHOD_' + cid[2]] = 1
                amp = cdict['high'] - cdict['low']
                off = (cdict['high'] + cdict['low'])/2.
                channel_cfg['ANALOG_AMPLITUDE_' + cid[2]] = amp
                channel_cfg['ANALOG_OFFSET_' + cid[2]] = off
            elif cdict['type'] == 'marker':
                channel_cfg['MARKER1_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER2_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER{}_LOW_{}'.format(cid[-1], cid[2])] = \
                    cdict['low']
                channel_cfg['MARKER{}_HIGH_{}'.format(cid[-1], cid[2])] = \
                    cdict['high']
            channel_cfg['CHANNEL_STATE_' + cid[2]] = 0
        # activate only active channels
        for cdict in self.channels.values():
            if cdict['AWG'] != AWG:
                continue
            cid = cdict['id']
            if cdict['active']:
                channel_cfg['CHANNEL_STATE_' + cid[2]] = 1
        return channel_cfg

    ###################################
    # UHFQC specific helper functions

    @staticmethod
    def _UHFQC_element_seqc(reps, wait, name1, name2, readout):
        """
        Generates a part of the sequence code responsible for playing back a
        single element

        Args:
            reps: number of repetitions for this code
            wait: boolean flag, whether to wait for trigger
            name1: name of the wave to be played on channel 1
            name2: name of the wave to be played on channel 2
            readout: boolean flag, whether to acquire a datapoint after the
                     element
        Returns:
            string for playing back an element
        """
        repeat_open_str = '\trepeat ({}) {{\n'.format(reps) if reps != 0 else ''
        wait_wave_str = '\t\twaitWave();\n' if wait else ''
        trigger_str = '\t\twaitDigTrigger(1, 1);\n' if wait else ''
        if name1 is None:
            play_str = '\t\tplayWave(2, {});\n'.format(name2)
        elif name2 is None:
            play_str = '\t\tplayWave(1, {});\n'.format(name1)
        else:
            play_str = '\t\tplayWave({}, {});\n'.format(name1, name2)
        readout_str = '\t\tsetTrigger(WINT_EN+RO_TRIG);\n' if readout else ''
        readout_str += '\t\tsetTrigger(WINT_EN);\n' if readout else ''
        repeat_close_str = '\t}\n' if reps != 0 else ''
        return repeat_open_str + trigger_str + play_str + readout_str + \
            wait_wave_str + repeat_close_str
