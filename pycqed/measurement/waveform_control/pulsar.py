# Originally by Wolfgang Pfaff
# Modified by Adriaan Rol 9/2015

# this module contains the sequencer and sequence classes

# TODO this would be nice to have also accessible from clients.
# should therefore be a SharedObject
# TODO in principle that could be generalized for other
# sequencing hardware i guess

import numpy as np
import logging
from qcodes.instrument.base import Instrument
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.\
    UHFQuantumController import UHFQC
import time

# some pulses use rounding when determining the correct sample at which to
# insert a particular value. this might require correct rounding -- the pulses
# are typically specified on short time scales, but the time unit we use is
# seconds. therefore we need a suitably chosen digit on which to round. 9 would
# round a pulse to 1 ns precision. 11 is 10 ps, and therefore probably beyond
# the lifetime of this code (no 10ps AWG available yet :))
SIGNIFICANT_DIGITS = 11


class Pulsar:
    """
    A meta-instrument responsible for all communication with the AWGs.
    Contains information about all the available awg-channels in the setup.
    Starting, stopping and programming the AWGs should be done through Pulsar.
    TODO: Supports Tektronix AWG5014 and ZI UHFLI.
    """
    def __init__(self):
        self.default_AWG = None  # the default AWG object
        self._AWG_obj = {}  # dictionary from AWG name to AWG object
        self.channels = {}

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
            delay: global delay applied to this channel
            offset: a (software implemented) offset voltage that is added to
                    all of the waveforms (analog channel only)
            high: maximal output value
            low: minimal output value
            active: whether this channel will be programmed
            AWG: name of the AWG this channel is on
        """
        if AWG is None:
            AWG = self.default_AWG.name

        self._AWG_obj[AWG] = Instrument.find_instrument(AWG)

        _doubles = []
        for c_name, c_dict in self.channels.items():
            if c_dict['id'] == id and c_dict['AWG'] == AWG:
                logging.warning("Channel '{}' on {} already in use, {} will "
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

    def get_AWG_object(self, channel):
        """
        Returns the AWG object for the channel

        Args:
            channel: Name of the channel
        Returns:
            the AWG object
        """
        return self._AWG_obj[self.channels[channel]['AWG']]

    def set_channel_opt(self, name, option, value):
        """
        Set a channel option

        Args:
            name: name of the channel
            option: name of the option
            value: new value for the option
        """
        self.channels[name][option] = value

    def _get_5014_subchannels(self, id):
        """
        Returns a list with all subchannels corresponding to AWG5014 channel id

        Args:
            id: ch#(_marker#)
        Returns:
            [ch#, ch#_marker1, ch#_marker2]
        """
        return [id[:3], id[:3]+'_marker1', id[:3]+'_marker2']

    def get_used_AWGs(self):
        """
        Returns:
            A list of the names of the AWGs in use
        """
        return self.AWGs.keys()

    def get_active_AWGs(self):
        """
        Returns:
            A set of the names of the AWGs in use with active channels
        """
        active_AWGs = set()
        for c_dict in self.channels.values():
            if c_dict['active']:
                active_AWGs.add(c_dict['AWG'])
        return active_AWGs

    def _get_channel_names_by_5014_id(self, id, AWG=None):
        """
        Args:
            id: ch#
            AWG: name of the AWG to look the channels for. If `None` uses
                 the default AWG
        Returns:
            dict of channel names with the given id
        """
        if AWG is None:
            AWG = self.default_AWG.name
        chans = {id: None, id+'_marker1': None, id+'_marker2': None}
        for c_name, c_dict in self.channels.items():
            if c_dict['id'] in chans and c_dict['AWG'] == AWG:
                chans[c_dict['id']] = c_name
        return chans

    def get_channel_name_by_id(self, id, AWG=None):
        """
        Args:
            id: ch#
            AWG: name of the AWG to look the channels for. If `None` uses
                 the default AWG.
        Returns:
            channel name with the given id
        """
        if AWG is None:
            AWG = self.default_AWG.name
        for c_name, c_dict in self.channels.items():
            if c_dict['id'] == id and c_dict['AWG'] == AWG:
                return c_name

    def get_used_subchannel_ids(self, all_subchannels=True, AWG=None):
        """
        Args:
            all_subchannels: Boolean flag to return all subchannel id-s if one
                             is enabled (applicable to AWG5014)
            AWG: The name of the AWG to look for the channels
        Returns:
            id-s of all the used channels (analog and marker) on the given AWG.
        """
        if AWG is None:
            AWG = self.default_AWG.name
        chans = set()
        for c_dict in self.channels.values():
            if c_dict['active'] and c_dict['AWG'] == AWG:
                if all_subchannels:
                    for id in self._get_5014_subchannels(c_dict['id']):
                        chans.add(id)
                else:
                    chans.add(c_dict['id'])
        return chans

    def get_used_channel_ids(self, AWG=None):
        """
        Args:
            AWG: The name of the AWG to look for the channels
        Returns:
            id-s of all the used channels on the given AWG. Marker channels on
             the AWG5014 are considered part of the corresponding analog channel
        """
        if AWG is None:
            AWG = self.default_AWG.name
        chans = set()
        for c_dict in self.channels.values():
            if c_dict['active'] and c_dict['AWG'] == AWG:
                chans.add(c_dict['id'][:3])
        return chans

    def activate_channels(self, channels='all', AWGs='all'):
        """
        :param channels: list of channel ids to turn on (if they are defined
                         and set as active)
        :param AWGs: a list of names of AWGs on which to turn channels on
                     or 'all'
        sets the state to 'on' for the active channels
        """
        if AWGs == 'all':
            AWGs = self.get_used_AWGs()
        for AWG in AWGs:
            ids = self.get_used_channel_ids(AWG)
            for id in ids:
                output = False
                names = self.get_channel_names_by_id(id, AWG)
                for sid in names:
                    if names[sid] is None:
                        continue
                    if channels != 'all' and names[sid] not in channels:
                        continue
                    if self.channels[names[sid]]['active']:
                        output = True
                if output:
                    self.AWGs[AWG].set('{}_state'.format(id), 1)

    def get_awg_channel_cfg(self, AWG=None):
        """
        Args:
            AWG: the name of the AWG
        Returns:
            A dictionary with the configuration parameters for the AWG
        """
        if AWG is None:
            AWG = self.default_AWG.name
        channel_cfg = {}

        for c_dict in self.channels.values():
            if c_dict['AWG'] != AWG:
                continue
            ch_id = c_dict['id']
            if c_dict['type'] == 'analog':
                channel_cfg['ANALOG_METHOD_%s' % ch_id[-1]] = 1
                a = c_dict['high'] - c_dict['low']
                o = (c_dict['high'] + c_dict['low'])/2.
                channel_cfg['ANALOG_AMPLITUDE_%s' % ch_id[-1]] = a
                channel_cfg['ANALOG_OFFSET_%s' % ch_id[-1]] = o
            elif c_dict['type'] == 'marker':
                channel_cfg['MARKER1_METHOD_%s' % ch_id[2]] = 2
                channel_cfg['MARKER2_METHOD_%s' % ch_id[2]] = 2
                channel_cfg['MARKER%s_LOW_%s' % (ch_id[-1], ch_id[2])] = \
                    c_dict['low']
                channel_cfg['MARKER%s_HIGH_%s' % (ch_id[-1], ch_id[2])] = \
                    c_dict['high']

            channel_cfg['CHANNEL_STATE_%s' % ch_id[2]] = 0

        # activate the used channels
        active_chans = self.get_used_channel_ids(AWG)
        for chan_id in active_chans:
            channel_cfg['CHANNEL_STATE_%s' % chan_id[2]] = 1
        return channel_cfg

    # waveform/file handling
    def delete_all_waveforms(self, AWG=None):
        if AWG is None:
            AWG = self.default_AWG.name
        self._AWG_obj[AWG].delete_all_waveforms_from_list()

    def program_awgs(self, sequence, *elements, **kw):
        """
        Args:
            sequence: The `Sequence` object telling the order in which to play
                      the elements
            *elements: The `Element` objects to program to the AWGs.
            AWGs: names of the AWGs to program. Defaults to 'all'.
        """
        # Stores the last uploaded elements for easy access and plotting
        self.last_sequence = sequence
        self.last_elements = elements

        AWGs = kw.pop('AWGs', 'all')

        if AWGs == 'all':
            AWGs = self.get_used_AWGs()

        for AWGn in AWGs:
            AWG = self._AWG_obj[AWGn]
            if isinstance(AWG, Tektronix_AWG5014):
                self._program_AWG5014(AWG, sequence, *elements, **kw)
            elif isinstance(AWG, UHFQC):
                self._program_UHFQC(AWG, sequence, *elements, **kw)
            else:
                raise TypeError('Unsupported AWG intrument: {} of type {}'
                                .format(AWGn, type(AWG)))


    def _program_AWG5014(self, AWG, sequence, *elements, **kw):
        """
        Upload a single file to the AWG (.awg) which contains all waveforms
        AND sequence information (i.e. nr of repetitions, event jumps etc)
        Advantage is that it's much faster, since sequence information is sent
        to the AWG in a single file.
        """
        verbose = kw.pop('verbose', False)
        debug = kw.pop('debug', False)
        channels = kw.pop('channels', 'all')
        loop = kw.pop('loop', True)
        allow_non_zero_first_point_on_trigger_wait = \
            kw.pop('allow_first_nonzero', False)
        elt_cnt = len(elements)

        packed_waveforms = {}

        # determine which channels are involved in the sequence
        if channels == 'all':
            chan_ids = self.get_used_channel_ids(AWG.name)
        else:
            chan_ids = set()
            for c in channels:
                if self.channels[c]['AWG'] == AWG.name:
                    chan_ids.add(self.channels[c]['id'][:3])
        chan_ids = list(chan_ids) # to make sure we iterate in a certain order

        old_timeout = AWG.timeout()
        AWG.timeout(max(180, old_timeout))

        elements_with_non_zero_first_points = []

        # order the waveforms according to physical AWG channels and
        # make empty sequences where necessary
        _t0 = time.time()
        for i, element in enumerate(elements):
            if verbose:
                print("Generate %s element %d / %d: %s (%d samples)... " %
                      (AWG.name, i+1, elt_cnt, element.name,
                       element.samples()), end='')
            _t0 = time.time()

            tvals, wfs = element.normalized_waveforms()

            for id in chan_ids:
                wfname = element.name + '_%s' % id

                # determine if we actually want to upload this channel
                if channels != 'all':
                    upload = False
                    for c in channels:
                        if self.channels[c]['id'][:3] == id and \
                                self.channels[c]['AWG'] == AWG.name:
                            upload = True
                    if not upload:
                        continue

                chan_wfs = {id: None,
                            id+'_marker1': None,
                            id+'_marker2': None}
                grp = self.get_channel_names_by_id(id, AWG.name)

                for sid in grp:
                    if grp[sid] != None and grp[sid] in wfs:
                        chan_wfs[sid] = wfs[grp[sid]]
                        if chan_wfs[sid][0] != 0.:
                            elements_with_non_zero_first_points.append(
                                element.name)
                    else:
                        chan_wfs[sid] = np.zeros(element.samples())

                # Create wform files
                packed_waveforms[wfname] = AWG.pack_waveform(
                    chan_wfs[id],
                    chan_wfs[id+'_marker1'],
                    chan_wfs[id+'_marker2'])

        _t = time.time() - _t0

        if verbose:
            print("finished in %.2f seconds." % _t)

        # sequence programming
        _t0 = time.time()
        if sequence.element_count() > 8000:
            logging.warning("Error: trying to program '{:s}' ({:d}'".format(
                            sequence.name, sequence.element_count()) +
                            " element(s))...\n Sequence contains more than " +
                            "8000 elements, Aborting", end=' ')
            return

        print("Programming %s sequence '%s' (%d element(s)) \t"
              % (AWG.name, sequence.name, sequence.element_count()), end=' ')

        # Create lists with sequence information:
        # wfname_l = list of waveform names [[wf1_ch1,wf2_ch1..],[wf1_ch2,wf2_ch2..],...]
        # nrep_l = list specifying the number of reps for each seq element
        # wait_l = idem for wait_trigger_state
        # goto_l = idem for goto_state (goto is the element where it hops to in case the element is finished)

        wfname_l = []
        nrep_l = []
        wait_l = []
        goto_l = []
        logic_jump_l = []

        for id in chan_ids:
            # set all the waveforms
            el_wfnames = []
            # add all wf names of channel
            for elt in sequence.elements:
                el_wfnames.append(elt['wfname'] + '_%s' % id)
                #  should the name include id nr?
            wfname_l.append(el_wfnames)

        for elt in sequence.elements:
            nrep_l.append(elt['repetitions'])
            if (elt['repetitions'] < 1) or (elt['repetitions'] > 65536):
                raise Exception('pulsar: The number of repetitions of ' +
                                'AWG element "%s" are out of range. Valid '
                                % elt['wfname'] +
                                'range = 1 to 65536 ("%s" recieved)'
                                % elt['repetitions'])

            if elt['goto_target'] != None:
                goto_l.append(sequence.element_index(elt['goto_target']))
            else:
                goto_l.append(0)
            if elt['jump_target'] != None:
                logic_jump_l.append(sequence.element_index(elt['jump_target']))
            else:
                logic_jump_l.append(0)
            if elt['trigger_wait']:
                wait_l.append(1)
                if elt['wfname'] in elements_with_non_zero_first_points and \
                        not allow_non_zero_first_point_on_trigger_wait:
                    raise Exception('pulsar: Trigger wait set for element {} '
                                    'with a non-zero first point'
                                    .format(elt['wfname']))
            else:
                wait_l.append(0)

        if loop and len(goto_l) > 0:
            goto_l[-1] = 1

        if debug:
            self.check_sequence_consistency(packed_waveforms,
                                            wfname_l,
                                            nrep_l, wait_l, goto_l,
                                            logic_jump_l)

        if len(wfname_l) > 0:
            filename = sequence.name+'_FILE.AWG'

            awg_file = AWG.generate_awg_file(
                packed_waveforms,
                np.array(wfname_l),
                nrep_l, wait_l, goto_l, logic_jump_l,
                self.get_awg_channel_cfg(AWG.name))
            AWG.send_awg_file(filename, awg_file)
            AWG.load_awg_file(filename)
        else:
            awg_file = None
        AWG.timeout(old_timeout)

        time.sleep(.1)
        # Waits for AWG to be ready
        AWG.is_awg_ready()

        self.activate_channels(channels, AWGs=[AWG.name])

        _t = time.time() - _t0
        print(" finished in %.2f seconds." % _t)
        return awg_file

    def _program_UHFQC(self, AWG, sequence, *elements, **kw):
        channels = kw.pop('channels', 'all')
        allow_non_zero_first_point_on_trigger_wait = \
            kw.pop('allow_first_nonzero', False)

        header = """const TRIGGER1  = 0x000001;
const WINT_TRIG = 0x000010;
const IAVG_TRIG = 0x000020;
const WINT_EN   = 0x1f0000;
setTrigger(WINT_EN);
var loop_cnt = getUserReg(0);
var RO_TRIG;
if(getUserReg(1)){
  RO_TRIG=IAVG_TRIG;
}else{
  RO_TRIG=WINT_TRIG;
}
\n"""

        main_loop = 'repeat(loop_cnt) {\n'


        footer = """}
wait(1000);
setTrigger(0);
"""

        # parse elements
        elements_with_non_zero_first_points = []
        wfnames = {'ch1': [], 'ch2': []}
        wfdata = []
        for element in elements:
            tvals, wfs = element.normalized_waveforms()

            for id in ['ch1', 'ch2']:
                wfname = element.name + '_%s' % id

                # determine if we actually want to upload this channel
                if channels != 'all':
                    upload = False
                    for c in channels:
                        if self.channels[c]['id'][:3] == id and \
                                self.channels[c]['AWG'] == AWG.name and \
                                self.channels[c]['active']:
                            upload = True
                            c_name = c
                    if not upload:
                        wfnames[id].append(None)
                        continue
                else:
                    c_name = self.get_channel_name_by_id(id, AWG.name)

                if c_name in wfs:
                    chan_wf = wfs[c_name]
                    if chan_wf[0] != 0.:
                        elements_with_non_zero_first_points.append(
                            element.name)
                    wfnames[id].append(wfname)
                    header += 'wave {} = zeros({})'.format(wfname,
                                                           element.samples)
                    wfdata.append(chan_wf)
                else:
                    wfnames[id].append(None)

        # create waveform playback code
        for i, elt in enumerate(sequence.elements):
            if elt['goto_target'] != 0:
                raise NotImplementedError('UHFQC sequencer does not yet support'
                                          ' nontrivial goto-s.')
            if elt['jump_target'] != 0:
                raise NotImplementedError('UHFQC sequencer does not support'
                                          ' jump events.')
            if elt['trigger_wait']:
                if elt['wfname'] in elements_with_non_zero_first_points and \
                        not allow_non_zero_first_point_on_trigger_wait:
                    raise Exception('pulsar: Trigger wait set for element {} '
                                    'with a non-zero first point'
                                    .format(elt['wfname']))
            name_ch1 = wfnames['ch1'][i]
            name_ch2 = wfnames['ch2'][i]
            main_loop += self._UHFQC_element_seqc(elt['repetitions'],
                                                  elt['trigger_wait'],
                                                  name_ch1, name_ch2, True)

        awg_str = header + main_loop + footer
        AWG.awg_string(awg_str)

        # populate the waveforms with data
        for i, data in enumerate(wfdata):
            AWG.awg_update_waveform(i, data)


    def _UHFQC_element_seqc(self, reps, wait, name1, name2, readout):
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
        repeat_open_str = '\trepeat({}) {\n'.format(reps) if reps != 0 else ''
        trigger_str = '\t\twaitDigTrigger(1, 1);\n' if wait else ''
        if name1 is None:
            play_str = '\t\tplayWave(2, {});\n'.format(name2)
        elif name2 is None:
            play_str = '\t\tplayWave(1, {});\n'.format(name1)
        else:
            play_str = '\t\tplayWave({}, {});\n'.format(name1, name2)
        readout_str = '\t\tsetTrigger(WINT_EN +RO_TRIG);\n' if readout else ''
        readout_str += '\t\tsetTrigger(WINT_EN);\n' if readout else ''
        wait_wave_str = '\t\twaitWave();\n'
        repeat_close_str = '\t}\n' if reps != 0 else ''
        return repeat_open_str + trigger_str + play_str + readout_str + \
               wait_wave_str + repeat_close_str

    def check_sequence_consistency(self, packed_waveforms,
                                   wfname_l,
                                   nrep_l, wait_l, goto_l, logic_jump_l):
        '''
        Specific for 4 channel tektronix 5014 where all channels are used.
        '''
        if not (len(wfname_l[0]) == len(wfname_l[1]) ==
                len(wfname_l[2]) == len(wfname_l[3]) ==
                len(nrep_l) == len(wait_l) == len(goto_l) ==
                len(logic_jump_l)):
            raise Exception('pulsar: sequence list of elements/properties has '
                            'unequal length')
        ch = 0
        for ch_wf in wfname_l:
            ch += 1
            el = 0
            for wfname in ch_wf:
                el += 1
                if wfname not in list(packed_waveforms.keys()):
                    raise Exception('pulsar: waveform name ' + wfname +
                                    ' , in position ' + str(el) +
                                    ' , channel ' + str(ch) +
                                    ' does not exist in waveform dictionary')

    def load_awg_file(self, filename, AWG=None, **kw):
        """
        Function to load an AWG sequence from its internal hard drive
        No possibility for jump statements
        """
        if AWG is None:
            AWG = self.AWG
        else:
            AWG = self.AWGs[AWG]
        old_timeout = AWG.timeout()
        AWG.timeout(max(180, old_timeout))
        channels = kw.pop('channels', 'all')
        chan_ids = self.get_used_channel_ids(AWG.name)
        _t0 = time.time()

        # Store offset settings to restore them after upload the seq
        # Note that this is the AWG setting offset, as distinct from the
        # channel parameter offset.
        offsets = {}
        for c_dict in self.channels.values():
            if c_dict['type'] == 'analog' and c_dict['AWG'] == AWG.name:
                offsets[c_dict['id']] = AWG.get(c_dict['id']+'_offset')

        AWG.load_awg_file(filename)
        AWG.timeout(old_timeout)

        time.sleep(.1)
        # Waits for AWG to be ready
        AWG.is_awg_ready()

        for channel, offset in offsets.items():
            AWG.set(channel+'_offset', offset)

        self.activate_channels(channels, AWG.name)

        _t = time.time() - _t0
        print(" finished in %.2f seconds." % _t)

    def start(self):
        if self.master_AWG is None:
            for AWG in self.get_active_AWGs():
                self.AWGs[AWG].start()
        else:
            for AWG in self.get_active_AWGs():
                if AWG != self.master_AWG:
                    self.AWGs[AWG].start()
            time.sleep(1) # wait 1 second for all other awg-s to start
            self.AWGs[self.master_AWG].start()

    def stop(self):
        for AWG in self.AWGs.values():
            AWG.stop()

    def update_channel_settings(self, AWGs='all'):
        """
        Deprecated with multi-AWG support. AWG settings are set to correct
        values while programming.

        gets the latest offset and amplitude from the used AWG analog channels,
        updates the self.channels dictionary accordingly (high = -low = amp/2)
        and returns the channel dictionary and a separate dictionary of offsets
        """
        offsets = {}

        for c_name, c_dict in self.channels.items():
            if AWGs == 'all' or c_dict['AWG'] in AWGs:
                if c_dict['type'] == 'analog':
                    AWG = Instrument.find_instrument(c_dict['AWG'])
                    exec('offsets[c_name] = AWG.{}_offset.get_latest()'
                         .format(c_dict['id']))
                    if offsets[c_name] is None:
                        offsets[c_name] = AWG.get(c_dict['id']+'_offset')
                    ch_amp = None  # to prevent linting error showing up
                    exec('ch_amp = AWG.{}_amp.get_latest()'
                         .format(c_dict['id']))
                    if ch_amp is None:
                        ch_amp = AWG.get('{}_amp'.format(c_dict['id']))
                    c_dict['low'] = -ch_amp/2
                    c_dict['high'] = ch_amp/2

        return self.channels, offsets
