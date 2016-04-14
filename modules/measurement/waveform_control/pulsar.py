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

    AWG = None
    AWG_type = 'regular'  # other option at this point is 'opt09'
    clock = 1e9
    channel_ids = ['ch1', 'ch1_marker1', 'ch1_marker2',
                   'ch2', 'ch2_marker1', 'ch2_marker2',
                   'ch3', 'ch3_marker1', 'ch3_marker2',
                   'ch3', 'ch3_marker1', 'ch3_marker2']
    AWG_sequence_cfg = {}

    def __init__(self):
        self.channels = {}

    # channel handling
    def define_channel(self, id, name, type, delay, offset,
                       high, low, active):

        _doubles = []
        for c in self.channels:
            if self.channels[c]['id'] == id:
                logging.warning(
                    "Channel '%s' already in use, will overwrite." % id)
                _doubles.append(c)
        for c in _doubles:
            del self.channels[c]

        self.channels[name] = {'id': id,
                               'type': type,
                               'delay': delay,
                               'offset': offset,
                               'high': high,
                               'low': low,
                               'active': active}

    def set_channel_opt(self, name, option, value):
        self.channels[name][option] = value

    def get_subchannels(self, id):
        return [id[:3], id[:3]+'_marker1', id[:3]+'_marker2']

    def get_channel_names_by_id(self, id):
        chans = {id: None, id+'_marker1': None, id+'_marker2': None}
        for c in self.channels:
            if self.channels[c]['id'] in chans:
                chans[self.channels[c]['id']] = c
        return chans

    def get_channel_name_by_id(self, id):
        for c in self.channels:
            if self.channels[c]['id'] == id:
                return c

    def get_used_subchannel_ids(self, all_subchannels=True):
        chans = []

        for c in self.channels:
            if self.channels[c]['active'] and \
                    self.channels[c]['id'] not in chans:
                if all_subchannels:
                    [chans.append(id) for id in
                        self.get_subchannels(self.channels[c]['id'])]
                else:
                    chans.append(self.channels[c]['id'])

        return chans

    def get_used_channel_ids(self):
        chans = []
        for c in self.channels:
            if self.channels[c]['active'] and \
                    self.channels[c]['id'][:3] not in chans:
                chans.append(self.channels[c]['id'][:3])
        return chans

    def setup_channels(self, output=False, reset_unused=True):
        '''
        Function seems to be unused (10/09/15) remove?
        '''

        for n in self.channel_ids:
            getattr(self.AWG, 'set_%s_status' % n[:3])('off')

        if reset_unused:
            for n in self.channel_ids:
                if 'marker' in n:
                    getattr(self.AWG, 'set_%s_low' % n)(0)
                    getattr(self.AWG, 'set_%s_high' % n)(1)
                else:
                    getattr(self.AWG, 'set_%s_amplitude' % n)(1.)
                    getattr(self.AWG, 'set_%s_offset' % n)(0.)

        for c in self.channels:
            n = self.channels[c]['id']

            # set correct bounds
            if self.channels[c]['type'] == 'analog':
                a = self.channels[c]['high'] - self.channels[c]['low']
                o = (self.channels[c]['high'] + self.channels[c]['low'])/2.
                getattr(self.AWG, 'set_%s_amplitude' % n)(a)
                getattr(self.AWG, 'set_%s_offset' % n)(o)
            elif self.channels[c]['type'] == 'marker':
                getattr(self.AWG, 'set_%s_low' % n)(self.channels[c]['low'])
                getattr(self.AWG, 'set_%s_high' % n)(self.channels[c]['high'])

            # turn on the used channels
            if output and self.channels[c]['active']:
                getattr(self.AWG, 'set_%s_status' % n[:3])('on')

    def activate_channels(self, channels='all'):
        ids = self.get_used_channel_ids()
        for id in ids:
            output = False
            names = self.get_channel_names_by_id(id)
            for sid in names:
                if names[sid] is None:
                    continue
                if channels != 'all' and names[sid] not in channels:
                    continue
                if self.channels[names[sid]]['active']:
                    output = True
            if output:
                self.AWG.set('{}_state'.format(id), 1)

    def get_awg_channel_cfg(self):
        channel_cfg = {}

        for c in self.channels:
            ch_id = self.channels[c]['id']
            if self.channels[c]['type'] == 'analog':
                channel_cfg['ANALOG_METHOD_%s' % ch_id[-1]] = 1
                a = self.channels[c]['high'] - self.channels[c]['low']
                o = (self.channels[c]['high'] + self.channels[c]['low'])/2.
                channel_cfg['ANALOG_AMPLITUDE_%s' % ch_id[-1]] = a
                channel_cfg['ANALOG_OFFSET_%s' % ch_id[-1]] = o
            elif self.channels[c]['type'] == 'marker':
                channel_cfg['MARKER1_METHOD_%s' % ch_id[2]] = 2
                channel_cfg['MARKER2_METHOD_%s' % ch_id[2]] = 2
                channel_cfg['MARKER%s_LOW_%s' % (ch_id[-1], ch_id[2])] = \
                    self.channels[c]['low']
                channel_cfg['MARKER%s_HIGH_%s' % (ch_id[-1], ch_id[2])] = \
                    self.channels[c]['high']

            # activate the used channels
            if self.channels[c]['active']:
                channel_cfg['CHANNEL_STATE_%s' % ch_id[-1]] = 1
            else:
                channel_cfg['CHANNEL_STATE_%s' % ch_id[-1]] = 0
        return channel_cfg

    # waveform/file handling

    def delete_all_waveforms(self):
        self.AWG.delete_all_waveforms_from_list()

    def program_awg(self, sequence, *elements, **kw):
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
            kw.pop('allow_first_zero', False)
        elt_cnt = len(elements)
        chan_ids = self.get_used_channel_ids()
        packed_waveforms = {}

        # Store offset settings to restore them after upload the seq
        # Note that this is the AWG setting offset, as distinct from the
        # channel parameter offset.
        offsets = {}
        for c in self.channels:
            if self.channels[c]['type'] == 'analog':
                offsets[c] = self.AWG.get(c+'_offset')

        elements_with_non_zero_first_points = []

        # order the waveforms according to physical AWG channels and
        # make empty sequences where necessary
        for i, element in enumerate(elements):
            if verbose:
                print("%d / %d: %s (%d samples)... " % \
                    (i+1, elt_cnt, element.name, element.samples()))
                print("Generate/upload '%s' (%d samples)... " \
                    % (element.name, element.samples()), end=' ')
            _t0 = time.time()

            tvals, wfs = element.normalized_waveforms()
            '''
            channels_to_print=['MW_Imod']
            for i in channels_to_print:
                if len(np.where(wfs[i]>0)[0]) !=0:
                    print i, np.where(wfs[i]>0)
            '''
            for id in chan_ids:
                wfname = element.name + '_%s' % id

                # determine if we actually want to upload this channel
                upload = False
                if channels == 'all':
                    upload = True
                else:
                    for c in channels:
                        if self.channels[c]['id'][:3] == id:
                            upload = True
                    if not upload:
                        continue

                chan_wfs = {id: None,
                            id+'_marker1': None,
                            id+'_marker2': None}
                grp = self.get_channel_names_by_id(id)

                for sid in grp:
                    if grp[sid] != None and grp[sid] in wfs:
                        chan_wfs[sid] = wfs[grp[sid]]
                        if chan_wfs[sid][0] != 0.:
                            elements_with_non_zero_first_points.append(
                                element.name)
                    else:
                        chan_wfs[sid] = np.zeros(element.samples())

                # Create wform files
                packed_waveforms[wfname] = self.AWG.pack_waveform(
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

        print("Programming '%s' (%d element(s)) \t"
              % (sequence.name, sequence.element_count()), end=' ')

        # determine which channels are involved in the sequence
        if channels == 'all':
            chan_ids = self.get_used_channel_ids()
        else:
            chan_ids = []
            for c in channels:
                if self.channels[c]['id'][:3] not in chan_ids:
                    chan_ids.append(self.channels[c]['id'][:3])

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
                # if (elt['wfname'] in elements_with_non_zero_first_points) and not(allow_non_zero_first_point_on_trigger_wait):
                #     print 'warning Trigger wait set for element with a non-zero first point'
                #     raise Exception('pulsar: Trigger wait set for element {} with a non-zero first point'.format(elt['wfname']))
            else:
                wait_l.append(0)

        if loop:
            goto_l[-1] = 1

        # setting jump modes and loading the djump table
        if sequence.djump_table != None and self.AWG_type not in ['opt09']:
            raise Exception('pulsar: The AWG configured does not support dynamic jumping')

        if self.AWG_type in ['opt09']:
            # TODO as self.AWG_sequence_cfg no longer exists but is generated
            # from the sequence_cfg file, make these set the values on the AWG
            # itself.
            if sequence.djump_table != None:
                # self.AWG.set_event_jump_mode('DJUM')
                self.AWG_sequence_cfg['EVENT_JUMP_MODE'] = 2  # DYNAMIC JUMP
                print('AWG set to dynamical jump')
                awg_djump_table = np.zeros(16, dtype='l')
                for i in list(sequence.djump_table.keys()):
                    el_idx = sequence.element_index(sequence.djump_table[i])
                    awg_djump_table[i] = el_idx
                self.AWG_sequence_cfg['TABLE_JUMP_DEFINITION'] = awg_djump_table

            else:
                self.AWG_sequence_cfg['EVENT_JUMP_MODE'] = 1  # EVENT JUMP

        if debug:
            self.check_sequence_consistency(packed_waveforms,
                                            wfname_l,
                                            nrep_l, wait_l, goto_l,
                                            logic_jump_l)

        filename = sequence.name+'_FILE.AWG'
        awg_file = self.AWG.generate_awg_file(
            packed_waveforms,
            np.array(wfname_l),
            nrep_l, wait_l, goto_l, logic_jump_l,
            self.get_awg_channel_cfg())
        self.AWG.send_awg_file(filename, awg_file)
        self.AWG.load_awg_file(filename)

        time.sleep(.1)
        # Waits for AWG to be ready
        self.AWG.is_awg_ready()

        for channel, offset in offsets.items():
            self.AWG.set(channel+'_offset', offset)

        self.activate_channels(channels)

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
            raise Exception('pulsar: sequence list of elements/properties has unequal length')
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

