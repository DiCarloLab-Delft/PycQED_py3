# Originally by Wolfgang Pfaff
# Modified by Adriaan Rol 9/2015

# this module contains the sequencer and sequence classes

# TODO this would be nice to have also accessible from clients.
# should therefore be a SharedObject
# TODO in principle that could be generalized for other
# sequencing hardware i guess

import time
import numpy as np
import logging
import warnings
from qcodes.instrument.base import Instrument

# some pulses use rounding when determining the correct sample at which to
# insert a particular value. this might require correct rounding -- the pulses
# are typically specified on short time scales, but the time unit we use is
# seconds. therefore we need a suitably chosen digit on which to round. 9 would
# round a pulse to 1 ns precision. 11 is 10 ps, and therefore probably beyond
# the lifetime of this code (no 10ps AWG available yet :))
SIGNIFICANT_DIGITS = 11


class Pulsar:
    """
    This is the object that communicates with the AWG.
    """

    # the default AWG instance
    AWG = None
    AWG_type = 'regular'  # other option at this point is 'opt09'

    def __init__(self, clock=1e9):
        self.channels = {}
        self.clock = clock

    # channel handling
    def define_channel(self, id, name, type, delay, offset,
                       high, low, active, AWG=None):
        if AWG is None:
            AWG = self.AWG.name

        _doubles = []
        for c_name, c_dict in self.channels.items():
            if c_dict['id'] == id and c_dict['AWG'] == AWG:
                logging.warning("Channel '{}' on {} already in use, will "
                                "overwrite.".format(id, AWG))
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

    def set_channel_opt(self, name, option, value):
        self.channels[name][option] = value

    def get_subchannels(self, id):
        return [id[:3], id[:3]+'_marker1', id[:3]+'_marker2']

    def get_used_AWGs(self):
        AWGs = set()
        for c_dict in self.channels.values():
            AWGs.add(c_dict['AWG'])
        return AWGs


    def get_channel_names_by_id(self, id, AWG=None):
        if AWG is None:
            AWG = self.AWG.name
        chans = {id: None, id+'_marker1': None, id+'_marker2': None}
        for c_name, c_dict in self.channels.items():
            if c_dict['id'] in chans and c_dict['AWG'] == AWG:
                chans[c_dict['id']] = c_name
        return chans

    def get_channel_name_by_id(self, id, AWG=None):
        if AWG is None:
            AWG = self.AWG.name
        for c_dict in self.channels.values():
            if c_dict['id'] == id and c_dict['AWG'] == AWG:
                return c

    def get_used_subchannel_ids(self, all_subchannels=True, AWG=None):
        if AWG is None:
            AWG = self.AWG.name
        chans = set()
        for c_dict in self.channels.values():
            if c_dict['active'] and c_dict['AWG'] == AWG:
                if all_subchannels:
                    for id in self.get_subchannels(c_dict['id']):
                        chans.add(id)
                else:
                    chans.add(c_dict['id'])
        return chans

    def get_used_channel_ids(self, AWG=None):
        if AWG is None:
            AWG = self.AWG.name
        chans = set()
        for c_dict in self.channels.values():
            if c_dict['active'] and c_dict['AWG'] == AWG:
                chans.add(c_dict['id'][:3])
        return chans

    def activate_channels(self, channels='all', AWGs='all'):
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
                    Instrument.find_instrument(AWG).set('{}_state'.format(id), 1)

    def get_awg_channel_cfg(self, AWG=None):
        if AWG is None:
            AWG = self.AWG.name
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
            AWG = self.AWG
        else:
            AWG = Instrument.find_instrument(AWG)
        AWG.delete_all_waveforms_from_list()

    def program_awgs(self, sequence, *elements, **kw):
        # Stores the last uploaded elements for easy access and plotting
        self.last_sequence = sequence
        self.last_elements = elements

        AWGs = kw.pop('AWGs', 'all')

        if AWGs == 'all':
            AWGs = self.get_used_AWGs()

        # Store offset settings to restore them after upload the seq
        # Note that this is the AWG setting offset, as distinct from the
        # channel parameter offset.
        chn, offsets = self.update_channel_settings()

        for AWGn in AWGs:
            AWG = Instrument.find_instrument(AWGn)
            AWG.stop()
            self._program_awg(AWG, sequence, *elements, **kw)

        for c, offset in offsets.items():
            Instrument.find_instrument(self.channels[c]['AWG']).\
                set(self.channels[c]['id']+'_offset', offset)

    def _program_awg(self, AWG, sequence, *elements, **kw):
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
            chan_ids = {}
            for c in channels:
                if self.channels[c]['AWG'] == AWG.name:
                    chan_ids.add(self.channels[c]['id'][:3])
        chan_ids = list(chan_ids) # to make sure we iterate in a certain order

        old_timeout = AWG.timeout()
        AWG.timeout(max(180, old_timeout))

        elements_with_non_zero_first_points = []

        # order the waveforms according to physical AWG channels and
        # make empty sequences where necessary
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

        # setting jump modes and loading the djump table
        if sequence.djump_table != None:
            if self.AWG_type not in ['opt09']:
                raise Exception('pulsar: The AWG configured does not support'
                                ' dynamic jumping')
            else:
                # TODO as self.AWG_sequence_cfg no longer exists but is
                # generated from the sequence_cfg file, make these set the
                # values on the AWG itself.
                logging.warning("Dynamical jumping not implemented in current "
                                "implementation of Pulsar class")
                # self.AWG.set_event_jump_mode('DJUM')
                # self.AWG_sequence_cfg['EVENT_JUMP_MODE'] = 2  # DYNAMIC JUMP
                # print('AWG set to dynamical jump')
                # awg_djump_table = np.zeros(16, dtype='l')
                # for i in list(sequence.djump_table.keys()):
                    # el_idx = sequence.element_index(sequence.djump_table[i])
                    # awg_djump_table[i] = el_idx
                # self.AWG_sequence_cfg['TABLE_JUMP_DEFINITION'] = awg_djump_table
            # else:
                # self.AWG_sequence_cfg['EVENT_JUMP_MODE'] = 1  # EVENT JUMP

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
        AWG.timeout(old_timeout)

        time.sleep(.1)
        # Waits for AWG to be ready
        AWG.is_awg_ready()

        self.activate_channels(channels, AWGs=[AWG.name])

        _t = time.time() - _t0
        print(" finished in %.2f seconds." % _t)
        return awg_file

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
            AWG = Instrument.find_instrument(AWG)
        old_timeout = AWG.timeout()
        AWG.timeout(max(180, old_timeout))
        channels = kw.pop('channels', 'all')
        chan_ids = self.get_used_channel_ids(AWG.name)
        _t0 = time.time()

        # Store offset settings to restore them after upload the seq
        # Note that this is the AWG setting offset, as distinct from the
        # channel parameter offset.
        offsets = {}
        for c_dict in self.channels.items():
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

    def update_channel_settings(self, AWGs='all'):
        """
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
