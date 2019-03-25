# Originally by Wolfgang Pfaff
# Modified by Adriaan Rol 9/2015
# Modified by Ants Remm 5/2017



import numpy as np
import logging
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
import qcodes.utils.validators as vals
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter
import pycqed.measurement.waveform_control.element as element
import string
import time
import pprint
import base64
import hashlib

from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014
from pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 import \
    VirtualAWG8
# exception catching removed because it does not work in python versions before
# 3.6
try:
    from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
except Exception:
    Tektronix_AWG5014 = type(None)
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.\
        UHFQuantumController import UHFQC
except Exception:
    UHFQC = type(None)
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments. \
        ZI_HDAWG8 import ZI_HDAWG8
except Exception:
    ZI_HDAWG8 = type(None)
log = logging.getLogger(__name__)


class UHFQCPulsar:
    """
    Defines the Zurich Instruments UHFQC specific functionality for the Pulsar
    class
    """
    _supportedAWGtypes = (UHFQC,)

    def _create_parameters(self, name, id, obj, options):

        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._create_parameters(name, id, obj)

        if id not in {'ch1', 'ch2'}:
            raise KeyError('Invalid UHFQC channel id: {}'.format(id))

        self.add_parameter('{}_id'.format(name),
                           get_cmd=(lambda _=id: _))
        self.add_parameter('{}_AWG'.format(name),
                           get_cmd=lambda _=obj.name: _)
        self.add_parameter('{}_options'.format(name),
                           get_cmd=lambda _=options.copy(): _)
        self.add_parameter('{}_type'.format(name),
                           get_cmd=lambda: 'analog')
        self.add_parameter('{}_granularity'.format(name),
                           get_cmd=lambda: 8)
        self.add_parameter('{}_min_length'.format(name),
                           get_cmd=lambda: 8 / 1.8e9)
        self.add_parameter('{}_inter_element_deadtime'.format(name),
                           get_cmd=lambda: 8 / 1.8e9)
        self.add_parameter('{}_precompile'.format(name),
                           get_cmd=lambda: False)
        self.add_parameter('{}_delay'.format(name), initial_value=0,
                           label='{} delay'.format(name), unit='s',
                           parameter_class=ManualParameter,
                           docstring="Global delay applied to this channel. "
                                     "Positive values move pulses on this "
                                     "channel forward in time")
        self.add_parameter('{}_offset'.format(name), initial_value=0,
                           label='{} offset'.format(name), unit='V',
                           parameter_class=ManualParameter)
        self.add_parameter('{}_amp'.format(name), initial_value=1,
                           label='{} amplitude'.format(name), unit='V',
                           parameter_class=ManualParameter)
        self.add_parameter('{}_active'.format(name), initial_value=True,
                           label='{} active'.format(name), vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_distortion'.format(name),
                           label='{} distortion mode'.format(name),
                           initial_value='off',
                           vals=vals.Enum('off', 'precalculate'),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_distortion_dict'.format(name),
                           label='{} distortion dictionary'.format(name),
                           vals=vals.Dict(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_charge_buildup_compensation'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.Bool(), initial_value=False)
        self.add_parameter('{}_discharge_timescale'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.MultiType(vals.Numbers(0),
                                               vals.Enum(None)),
                           initial_value=None)
        self.add_parameter('{}_compensation_pulse_scale'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0., 1.), initial_value=0.5)

    def _program_awg(self, obj, sequence, el_wfs, loop=True):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._program_awg(obj, sequence, el_wfs, loop)

        header = "const TRIGGER1  = 0x00000001;\n" \
                 "const TRIGGER2  = 0x00000002;\n" \
                 "const WINT_TRIG = 0x00000010;\n" \
                 "const IAVG_TRIG = 0x00000020;\n" \
                 "const WINT_EN   = 0x01ff0000;\n" \
                 "setTrigger(WINT_EN);\n" \
                 "var loop_cnt = getUserReg(0);\n" \
                 "var RO_TRIG;\n" \
                 "if (getUserReg(1)) {\n" \
                 "  RO_TRIG=IAVG_TRIG;\n" \
                 "} else {\n" \
                 "  RO_TRIG=WINT_TRIG;\n" \
                 "}\n\n"

        if loop:
            main_loop = "while(1) {\n"
            footer = "}\n"
        else:
            main_loop = ""
            footer = ""
        main_loop += "repeat (loop_cnt) {\n"

        footer += "}\n" \
                  "wait(1000);\n" \
                  "setTrigger(0);\n"

        # parse elements
        elements_with_non_zero_first_points = []
        wfnames = {'ch1': [], 'ch2': []}
        wfdata = {'ch1': [], 'ch2': []}
        i = 1
        for (_, el), cid_wfs in sorted(el_wfs.items()):
            for cid in ['ch1', 'ch2']:
                if cid in cid_wfs:
                    wfname = el + '_' + cid
                    cid_wf = cid_wfs[cid]
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
            if el['wfname'] == 'codeword':
                raise NotImplementedError("UHFQC sequencer does not currently "
                                          "support codeword triggering.")
            if el['goto_target'] is not None:
                raise NotImplementedError("UHFQC sequencer does not yet support"
                                          " nontrivial goto-s.")
            if el['trigger_wait']:
                if el['wfname'] in elements_with_non_zero_first_points:
                    log.warning('Pulsar: Trigger wait set for element {} '
                        'with a non-zero first point'.format(el['wfname']))
            name_ch1 = el['wfname'] + '_ch1'
            name_ch2 = el['wfname'] + '_ch2'
            if name_ch1 not in wfnames['ch1']: name_ch1 = None
            if name_ch2 not in wfnames['ch2']: name_ch2 = None
            main_loop += self._UHFQC_element_seqc(el['repetitions'],
                                                  el['trigger_wait'],
                                                  name_ch1, name_ch2,
                                                  'readout' in el['flags'])
        awg_str = header + main_loop + footer
        obj.awg_string(awg_str)

        # populate the waveforms with data
        i = 0
        for data1, data2 in zip(wfdata['ch1'], wfdata['ch2']):
            if data1 is None and data2 is None:
                continue
            obj.awg_update_waveform(i, data1, data2)
            i += 1
        return awg_str

    def _is_AWG_running(self, obj):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._is_AWG_running(obj)

        return obj.awgs_0_enable() != 0

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
        repeat_open_str = '\trepeat ({}) {{\n'.format(
            reps) if reps != 1 else ''
        wait_wave_str = '\t\twaitWave();\n'
        trigger_str = '\t\twaitDigTrigger(1, 1);\n' if wait else ''
        if name1 is None:
            play_str = '\t\tplayWave(2, {});\n'.format(name2)
        elif name2 is None:
            play_str = '\t\tplayWave(1, {});\n'.format(name1)
        else:
            play_str = '\t\tplayWave({}, {});\n'.format(name1, name2)
        readout_str = '\t\tsetTrigger(WINT_EN+RO_TRIG);\n' if readout else ''
        readout_str += '\t\tsetTrigger(WINT_EN);\n' if readout else ''
        repeat_close_str = '\t}\n' if reps != 1 else ''
        return repeat_open_str + wait_wave_str + trigger_str + play_str + \
               readout_str + repeat_close_str

    def _clock(self, obj, cid):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._clock(obj, cid)
        return obj.clock_freq()

class HDAWG8Pulsar:
    """
    Defines the Zurich Instruments HDAWG8 specific functionality for the Pulsar
    class
    """
    _supportedAWGtypes = (ZI_HDAWG8, VirtualAWG8, )

    def _create_parameters(self, name, id, obj, options):

        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._create_parameters(name, id, obj)

        if id not in {'ch{}'.format(i) for i in range(1, 9)}:
            raise KeyError('Invalid HDAWG8 channel id: {}'.format(id))

        self.add_parameter('{}_id'.format(name),
                           get_cmd=lambda _=id: _)
        self.add_parameter('{}_AWG'.format(name),
                           get_cmd=lambda _=obj.name: _)
        self.add_parameter('{}_options'.format(name),
                           get_cmd=lambda _=options.copy(): _)
        self.add_parameter('{}_type'.format(name),
                           get_cmd=lambda: 'analog')
        self.add_parameter('{}_granularity'.format(name),
                           get_cmd=lambda: 16)
        self.add_parameter('{}_min_length'.format(name),
                           get_cmd=lambda: 8 / 2.4e9)
        self.add_parameter('{}_inter_element_deadtime'.format(name),
                           # get_cmd=lambda: 80 / 2.4e9)
                           get_cmd=lambda: 8 / 2.4e9)
                           # get_cmd=lambda: 0 / 2.4e9)
        self.add_parameter('{}_precompile'.format(name), initial_value=False,
                           label='{} precompile segments'.format(name),
                           parameter_class=ManualParameter, vals=vals.Bool())
        self.add_parameter('{}_delay'.format(name), initial_value=0,
                           label='{} delay'.format(name), unit='s',
                           parameter_class=ManualParameter,
                           docstring="Global delay applied to this channel. "
                                     "Positive values move pulses on this "
                                     "channel forward in time")
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._HDAWG8_setter(obj, id, 'offset'),
                           get_cmd=self._HDAWG8_getter(obj, id, 'offset'),
                           vals=vals.Numbers())
        self.add_parameter('{}_amp'.format(name),
                           label='{} amplitude'.format(name), unit='V',
                           set_cmd=self._HDAWG8_setter(obj, id, 'amp'),
                           get_cmd=self._HDAWG8_getter(obj, id, 'amp'),
                           vals=vals.Numbers(0.01, 5.0))
        self.add_parameter('{}_active'.format(name), initial_value=True,
                           label='{} active'.format(name), vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_distortion'.format(name),
                           label='{} distortion mode'.format(name),
                           initial_value='off',
                           vals=vals.Enum('off', 'precalculate'),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_distortion_dict'.format(name),
                           label='{} distortion dictionary'.format(name),
                           vals=vals.Dict(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_charge_buildup_compensation'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.Bool(), initial_value=False)
        self.add_parameter('{}_discharge_timescale'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.MultiType(vals.Numbers(0),
                                               vals.Enum(None)),
                           initial_value=None)
        self.add_parameter('{}_compensation_pulse_scale'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0., 1.), initial_value=0.5)

    @staticmethod
    def _HDAWG8_setter(obj, id, par):
        if par == 'offset':
            def s(val):
                obj.set('sigouts_{}_offset'.format(int(id[2])-1), val)
        elif par == 'amp':
            def s(val):
                obj.set('sigouts_{}_range'.format(int(id[2])-1), 2*val)
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _HDAWG8_getter(self, obj, id, par):
        if par == 'offset':
            def g():
                return obj.get('sigouts_{}_offset'.format(int(id[2])-1))
        elif par == 'amp':
            def g():
                if self._AWGs_prequeried_state:
                    return obj.parameters['sigouts_{}_range'
                        .format(int(id[2])-1)].get_latest()/2
                else:
                    return obj.get('sigouts_{}_range'.format(int(id[2])-1))/2
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return g

    def _program_awg(self, obj, sequence, el_wfs, loop=True):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._program_awg(obj, sequence, el_wfs, loop)

        # delete previous cache files
        obj._delete_chache_files()
        print('Previous AWG8 cache files have been deleted.')
        # delete old csv files
        obj._delete_csv_files()
        print('Previous AWG8 csv files have been deleted.')

        for awg_nr in [0, 1, 2, 3]:
            ch1id = 'ch{}'.format(awg_nr * 2 + 1)
            ch2id = 'ch{}'.format(awg_nr * 2 + 2)

            prev_dio_valid_polarity = obj.get('awgs_{}_dio_valid_polarity'.format(awg_nr))

            # Create waveform definitions
            header = ""
            elements_with_non_zero_first_points = []

            unique_waveforms = {} # maps list hashes to waveform names
            wf_name_map = {} # maps waveform name from element to an older copy
            # i = 0
            for (_, el), cid_wfs in sorted(el_wfs.items()):
                for cid in [ch1id, ch2id]:
                    if cid in cid_wfs:
                        wfname = str(el) + '_' + cid
                        cid_wf = cid_wfs[cid]

                        # TODO - Restructure for hashing without data copy
                        # hashing list using tuple makes a copy of the data
                        # and also __eq__ is expansive
                        # thus it would be better to restructure the code
                        # and hash before evaluating numerically - JH
                        wfhashable = tuple(cid_wf)
                        # if wfhashable in unique_waveforms:
                        #     # we already have such a waveform
                        #     wf_name_map[wfname] = \
                        #         unique_waveforms[wfhashable]
                        # else:
                        #     # new waveform for this subAWG
                        #     unique_waveforms[wfhashable] = wfname
                        #     wf_name_map[wfname] = wfname

                        # add all waveforms
                        unique_waveforms[wfname] = wfhashable
                        wf_name_map[wfname] = wfname

                        if len(cid_wf) > 0 and cid_wf[0] != 0.:
                            elements_with_non_zero_first_points.append(el)
                        #header += 'wave {} = ramp({}, 0, {});\n'.format(
                        #    simplify_name(wfname), len(cid_wf), 1 / i
                        #)
                        #i += 1

            # Create waveform table
            waveform_table = ""
            for cw, wfname in sequence.codewords.items():
                for (_, el), cid_wfs in sorted(el_wfs.items()):
                    if el == wfname:
                        if ch1id in cid_wfs and ch2id in cid_wfs:
                            wfname1 = wf_name_map[wfname + '_' + ch1id]
                            wfname2 = wf_name_map[wfname + '_' + ch2id]
                            waveform_table += \
                                'setWaveDIO({0}, "{1}_{2}", "{1}_{3}");\n' \
                                .format(cw, obj._devname,
                                        simplify_name(wfname1),
                                        simplify_name(wfname2))
                        elif ch1id in cid_wfs and ch2id not in cid_wfs:
                            wfname1 = wf_name_map[wfname + '_' + ch1id]
                            waveform_table += \
                                'setWaveDIO({}, 1, "{}_{}");\n' \
                                .format(cw, obj._devname,
                                        simplify_name(wfname1))
                        elif ch1id not in cid_wfs and ch2id in cid_wfs:
                            wfname2 = wf_name_map[wfname + '_' + ch2id]
                            waveform_table += \
                                'setWaveDIO({}, 2, "{}_{}");\n' \
                                .format(cw, obj._devname,
                                        simplify_name(wfname2))

            # Create main loop and footer
            if loop:
                main_loop = "while(1) {\n"
                footer = "}\n"
            else:
                main_loop = ""
                footer = ""
            footer += "wait(1000);\n"
            # waveform playback code
            for i, el in enumerate(sequence.elements):
                if el['goto_target'] is not None:
                    raise NotImplementedError("HDAWG8 sequencer does not yet "
                                              "support nontrivial goto-s.")
                if el['wfname'] == 'codeword':
                    main_loop += self._HDAWG8_element_seqc(el['repetitions'],
                        el['trigger_wait'], None, None)
                else:
                    if el['trigger_wait']:
                        if el['wfname'] in elements_with_non_zero_first_points:
                            log.warning('Pulsar: Trigger wait set for element '
                                        '{} with a non-zero first '
                                        'point'.format(el['wfname']))

                    # Expected element names
                    name_ch1 = str(el['wfname']) + '_' + ch1id
                    name_ch2 = str(el['wfname']) + '_' + ch2id

                    # Check if element with such a name exists
                    # Ch1
                    if name_ch1 in wf_name_map:
                        name_ch1 = wf_name_map[name_ch1]
                    else:
                        name_ch1 = None
                    # Ch2
                    if name_ch2 in wf_name_map:
                        name_ch2 = wf_name_map[name_ch2]
                    else:
                        name_ch2 = None

                    if name_ch1 is not None:
                        name_ch1 = '"{}_{}"'.format(obj._devname,
                                                    simplify_name(name_ch1))
                    if name_ch2 is not None:
                        name_ch2 = '"{}_{}"'.format(obj._devname,
                                                    simplify_name(name_ch2))
                    if name_ch1 is not None or name_ch2 is not None:
                        main_loop += self._HDAWG8_element_seqc(
                            el['repetitions'], el['trigger_wait'],
                            name_ch1, name_ch2)
            awg_str = header + waveform_table + main_loop + footer

            # write the waveforms to csv files
            # for data, wfname in unique_waveforms.items():
            for wfname, data in unique_waveforms.items():
                obj._write_csv_waveform(simplify_name(wfname), np.array(data))

            print("Programming {} vawg{} sequence '{}' ({} "
                  "element(s)) \t".format(obj.name, awg_nr, sequence.name,
                                          sequence.element_count()), end=' ')

            # here we want to use a timeout value longer than the obj.timeout()
            # as programming the AWGs takes more time than normal communications
            obj.configure_awg_from_string(awg_nr, awg_str, timeout=300)

            obj.set('awgs_{}_dio_valid_polarity'.format(awg_nr),
                    prev_dio_valid_polarity)

            # Turn on active channels:
            for ch in range(8):
                cname = self._id_channel('ch{}'.format(ch + 1), obj.name)
                obj.set('sigouts_{}_on'.format(ch), self.get(cname + '_active'))


        return awg_str

    def _is_AWG_running(self, obj):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._is_AWG_running(obj)

        return all([obj.get('awgs_{}_enable'.format(awg_nr)) for awg_nr in
                    self._HDAWG8_active_AWGs(obj)])

    def _clock(self, obj, cid):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._clock(obj, cid)
        return obj.clock_freq((int(cid[2])-1)//2)

    def _HDAWG8_element_seqc(self, reps, wait, name1, name2):
        """
        Generates a part of the sequence code responsible for playing back a
        single element

        Args:
            reps: number of repetitions for this code
            wait: boolean flag, whether to wait for trigger
            name1: name of the wave to be played on channel 1
            name2: name of the wave to be played on channel 2
        Returns:
            string for playing back an element
        """
        repeat_open_str = 'repeat ({}) {{\n'.format(
            reps) if reps != 1 else ''
        trigger_str = 'waitDigTrigger(1);\n'
        if name1 is None and name2 is None:
            prefetch_str = ''
            play_str = 'playWaveDIO();\n'
        elif name1 is None:
            prefetch_str = ''
            play_str = 'playWave(2, {});\n'.format(name2)
        elif name2 is None:
            prefetch_str = ''
            play_str = 'playWave(1, {});\n'.format(name1)
        else:
            prefetch_str = 'prefetch({}, {});\n'.format(name1, name2)
            play_str = 'playWave({}, {});\n'.format(name1, name2)
        repeat_close_str = '}\n' if reps != 1 else ''
        return repeat_open_str + prefetch_str+ trigger_str + \
               play_str + repeat_close_str

    def _HDAWG8_active_AWGs(self, obj):
        result = set()
        for channel in self.channels:
            if not self.get('{}_active'.format(channel)):
                continue
            if self.get('{}_AWG'.format(channel)) != obj.name:
                continue
            ch_nr = int(self.get('{}_id'.format(channel))[2])
            result.add((ch_nr - 1)//2)
        return result

class AWG5014Pulsar:
    """
    Defines the Tektronix AWG5014 specific functionality for the Pulsar class
    """
    _supportedAWGtypes = (Tektronix_AWG5014, VirtualAWG5014, )

    def _program_awg(self, obj, sequence, el_wfs, loop=True):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._program_awg(obj, sequence, el_wfs, loop)

        pars = {'ch{}_m{}_low'.format(ch+1, m+1) for ch in range(4)
                for m in range(2)}
        pars |= {'ch{}_m{}_high'.format(ch+1, m+1) for ch in range(4)
                for m in range(2)}
        old_vals = {}
        for par in pars:
            old_vals[par] = obj.get(par)

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
        # in the sequence0
        packed_waveforms = {}
        elements_with_non_zero_first_points = set()
        for (_, el), cid_wfs in sorted(el_wfs.items()):
            maxlen = 1
            for wf in cid_wfs.values():
                maxlen = max(maxlen, len(wf))
            for grp in grps:
                grp_wfs = {}
                # arrange waveforms from input data and pad with zeros for
                # equal length
                for cid in self._AWG5014_group_ids(grp):
                    grp_wfs[cid] = cid_wfs.get(cid, np.zeros(maxlen))
                    grp_wfs[cid] = np.pad(grp_wfs[cid],
                                          (0, maxlen - len(grp_wfs[cid])),
                                          'constant',
                                          constant_values=0)

                    if grp_wfs[cid][0] != 0.:
                        elements_with_non_zero_first_points.add(el)
                wfname = str(el) + '_' + grp

                packed_waveforms[wfname] = obj.pack_waveform(
                    grp_wfs[grp],
                    grp_wfs[grp + '_m1'],
                    grp_wfs[grp + '_m2'])

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
            for el in sequence.elements:
                if el['wfname'] == 'codeword':
                    # pick a random codeword element.
                    # all of them should be 0.
                    for elwfname in sequence.codewords.values():
                        wfname = str(elwfname) + '_' + grp
                        break
                else:
                    wfname = str(el['wfname']) + '_' + grp
                grp_wfnames.append(wfname)
            wfname_l.append(grp_wfnames)


        for el in sequence.elements:
            nrep_l.append(el['repetitions'])
            if (el['repetitions'] < 1) or (el['repetitions'] > 65536):
                raise Exception(
                    "The number of repetitions of AWG '{}' element '{}' are out"
                    " of range. Valid range = 1 to 65536 ('{}' received)"
                        .format(obj.name, el['wfname'], el['repetitions'])
                )
            if el['goto_target'] is not None:
                goto_l.append(sequence.element_index(el['goto_target']))
            else:
                goto_l.append(0)
            logic_jump_l.append(0)
            if el['trigger_wait']:
                wait_l.append(1)
                if el['wfname'] in elements_with_non_zero_first_points:
                    log.warning('Pulsar: Trigger wait set for element {} '
                                'with a non-zero first point'.format(
                                    el['wfname']))
            else:
                wait_l.append(0)
        if loop and len(goto_l) > 0:
            goto_l[-1] = 1

        prev_offsets = {ch: obj.get('{}_offset'.format(ch)) for ch in
                        ['ch1', 'ch2', 'ch3', 'ch4']}

        self.wfname_l = wfname_l
        self.packed_waveforms = packed_waveforms
        self.nrep_l = nrep_l
        self.wait_l = wait_l
        self.goto_l = goto_l
        self.logic_jump_l = logic_jump_l

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

        for ch, off in prev_offsets.items():
            obj.set(ch + '_offset', off)

        obj.timeout(old_timeout)
        for par in pars:
            obj.set(par, old_vals[par])

        time.sleep(.1)
        # Waits for AWG to be ready
        obj.is_awg_ready()

        self._AWG5014_activate_channels(grps, obj.name)

        hardware_offsets = False
        for grp in grps:
            cname = self._id_channel(grp, obj.name)
            options = self.get('{}_options'.format(cname))
            offset_mode = options.get('offset_mode', 'software')
            if offset_mode == 'hardware':
                hardware_offsets = True
        if hardware_offsets:
            obj.DC_output(1)
        else:
            obj.DC_output(0)

        _t = time.time() - _t0
        print(" finished in {:.2f} seconds.".format(_t))
        return awg_file

    def _is_AWG_running(self, obj):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._is_AWG_running(obj)

        return obj.get_state() != 'Idle'

    def _clock(self, obj, cid):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._clock(obj, cid)
        return obj.clock_freq()

    def _create_parameters(self, name, id, obj, options):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._create_parameters(name, id, obj, options)

        offset_mode = options.get('offset_mode', 'software')

        if id not in ['ch1', 'ch1_m1', 'ch1_m2',
                      'ch2', 'ch2_m1', 'ch2_m2',
                      'ch3', 'ch3_m1', 'ch3_m2',
                      'ch4', 'ch4_m1', 'ch4_m2']:
            raise KeyError('Invalid AWG5014 channel id: {}'.format(id))

        self.add_parameter('{}_id'.format(name),
                           get_cmd=lambda _=id: _)
        self.add_parameter('{}_AWG'.format(name),
                           get_cmd=lambda _=obj.name: _)
        self.add_parameter('{}_options'.format(name),
                           get_cmd=lambda _=options.copy(): _)
        self.add_parameter('{}_delay'.format(name), initial_value=0,
                           label='{} delay'.format(name), unit='s',
                           parameter_class=ManualParameter,
                           docstring="Global delay applied to this channel. "
                                     "Positive values move pulses on this "
                                     "channel forward in  time")
        self.add_parameter('{}_granularity'.format(name),
                           get_cmd=lambda: 4)
        self.add_parameter('{}_min_length'.format(name),
                           get_cmd=lambda: 210e-9)  # Can not be triggered faster
                                                    # than 210 ns.
        self.add_parameter('{}_inter_element_deadtime'.format(name),
                           get_cmd=lambda: 0)
        self.add_parameter('{}_precompile'.format(name), initial_value=False,
                           label='{} precompile segments'.format(name),
                           parameter_class=ManualParameter, vals=vals.Bool())
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:  # analog
            self.add_parameter('{}_type'.format(name),
                               get_cmd=lambda: 'analog')
            self.add_parameter('{}_offset'.format(name),
                               label='{} offset'.format(name), unit='V',
                               set_cmd=self._AWG5014_setter(obj, id, 'offset',
                                                            offset_mode),
                               get_cmd=self._AWG5014_getter(obj, id, 'offset',
                                                            offset_mode),
                               vals=vals.Numbers())
            self.add_parameter('{}_amp'.format(name), initial_value=1,
                               label='{} amplitude'.format(name), unit='V',
                               set_cmd=self._AWG5014_setter(obj, id, 'amp'),
                               get_cmd=self._AWG5014_getter(obj, id, 'amp'),
                               vals=vals.Numbers(0.01, 2.25))
            self.add_parameter('{}_distortion'.format(name),
                               label='{} distortion mode'.format(name),
                               initial_value='off',
                               vals=vals.Enum('off', 'precalculate'),
                               parameter_class=ManualParameter)
            self.add_parameter('{}_distortion_dict'.format(name),
                               label='{} distortion dictionary'.format(name),
                               vals=vals.Dict(),
                               parameter_class=ManualParameter)
            self.add_parameter('{}_charge_buildup_compensation'.format(name),
                               parameter_class=ManualParameter,
                               vals=vals.Bool(), initial_value=False)
            self.add_parameter('{}_discharge_timescale'.format(name),
                               parameter_class=ManualParameter,
                               vals=vals.MultiType(vals.Numbers(0),
                                                   vals.Enum(None)),
                               initial_value=None)
            self.add_parameter('{}_compensation_pulse_scale'.format(name),
                               parameter_class=ManualParameter,
                               vals=vals.Numbers(0., 1.), initial_value=0.5)
            self.add_parameter('{}_compensation_pulse_delay'.format(name),
                               parameter_class=ManualParameter,
                               vals=vals.Numbers(min_value=0.),
                               initial_value=500e-9)

        else:  # marker
            self.add_parameter('{}_type'.format(name),
                               get_cmd=lambda: 'marker')
            self.add_parameter('{}_offset'.format(name),
                               label='{} offset'.format(name), unit='V',
                               set_cmd=self._AWG5014_setter(obj, id, 'offset'),
                               get_cmd=self._AWG5014_getter(obj, id, 'offset'),
                               vals=vals.Numbers(-2.7, 2.7))
            self.add_parameter('{}_amp'.format(name), initial_value=1,
                               label='{} amplitude'.format(name), unit='V',
                               set_cmd=self._AWG5014_setter(obj, id, 'amp'),
                               get_cmd=self._AWG5014_getter(obj, id, 'amp'),
                               vals=vals.Numbers(-5.4, 5.4))
        self.add_parameter('{}_active'.format(name), initial_value=True,
                           label='{} active'.format(name), vals=vals.Bool(),
                           parameter_class=ManualParameter)

    @staticmethod
    def _AWG5014_setter(obj, id, par, offset_mode=None):
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            if par == 'offset':
                if offset_mode == 'software':
                    def s(val):
                        obj.set('{}_offset'.format(id), val)
                elif offset_mode == 'hardware':
                    def s(val):
                        obj.set('{}_DC_out'.format(id), val)
                else:
                    raise ValueError('Invalid offset mode for AWG5014: '
                                     '{}'.format(offset_mode))
            elif par == 'amp':
                def s(val):
                    obj.set('{}_amp'.format(id), 2*val)
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        else:
            if par == 'offset':
                def s(val):
                    h = obj.get('{}_high'.format(id))
                    l = obj.get('{}_low'.format(id))
                    obj.set('{}_high'.format(id), val + h - l)
                    obj.set('{}_low'.format(id), val)
            elif par == 'amp':
                def s(val):
                    l = obj.get('{}_low'.format(id))
                    obj.set('{}_high'.format(id), l + val)
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _AWG5014_getter(self, obj, id, par, offset_mode=None):
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            if par == 'offset':
                if offset_mode == 'software':
                    def g():
                        return obj.get('{}_offset'.format(id))
                elif offset_mode == 'hardware':
                    def g():
                        return obj.get('{}_DC_out'.format(id))
                else:
                    raise ValueError('Invalid offset mode for AWG5014: '
                                     '{}'.format(offset_mode))
            elif par == 'amp':
                def g():
                    if self._AWGs_prequeried_state:
                        return obj.parameters['{}_amp'.format(id)] \
                                   .get_latest()/2
                    else:
                        return obj.get('{}_amp'.format(id))/2
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        else:
            if par == 'offset':
                def g():
                    return obj.get('{}_low'.format(id))
            elif par == 'amp':
                def g():
                    if self._AWGs_prequeried_state:
                        h = obj.get('{}_high'.format(id))
                        l = obj.get('{}_low'.format(id))
                    else:
                        h = obj.parameters['{}_high'.format(id)].get_latest()
                        l = obj.parameters['{}_low'.format(id)].get_latest()
                    return h - l
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        return g

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
        return [cid[:3], cid[:3] + '_m1', cid[:3] + '_m2']

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
        for channel in self.channels:
            if self.get('{}_AWG'.format(channel)) != AWG:
                continue
            options = self.get('{}_options'.format(channel))
            offset_mode = options.get('offset_mode', 'software')
            cid = self.get('{}_id'.format(channel))
            amp = self.get('{}_amp'.format(channel)) * 2
            off = self.get('{}_offset'.format(channel))
            if cid in ['ch1', 'ch2', 'ch3', 'ch4']:
                channel_cfg['ANALOG_METHOD_' + cid[2]] = 1
                channel_cfg['ANALOG_AMPLITUDE_' + cid[2]] = amp
                if offset_mode == 'software':
                    channel_cfg['ANALOG_OFFSET_' + cid[2]] = off
                    channel_cfg['DC_OUTPUT_LEVEL_' + cid[2]] = 0
                    channel_cfg['EXTERNAL_ADD_' + cid[2]] = 0
                else:
                    channel_cfg['ANALOG_OFFSET_' + cid[2]] = 0
                    channel_cfg['DC_OUTPUT_LEVEL_' + cid[2]] = off
                    channel_cfg['EXTERNAL_ADD_' + cid[2]] = 1
            else:
                channel_cfg['MARKER1_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER2_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER{}_LOW_{}'.format(cid[-1], cid[2])] = \
                    off
                channel_cfg['MARKER{}_HIGH_{}'.format(cid[-1], cid[2])] = \
                    off + amp
            channel_cfg['CHANNEL_STATE_' + cid[2]] = 0
        for channel in self.channels:
            if self.get('{}_AWG'.format(channel)) != AWG:
                continue
            if self.get('{}_active'.format(channel)):
                cid = self.get('{}_id'.format(channel))
                channel_cfg['CHANNEL_STATE_' + cid[2]] = 1
        return channel_cfg


class Pulsar(AWG5014Pulsar, HDAWG8Pulsar, UHFQCPulsar, Instrument):
    """
    A meta-instrument responsible for all communication with the AWGs.
    Contains information about all the available awg-channels in the setup.
    Starting, stopping and programming and changing the parameters of the AWGs
    should be done through Pulsar. Supports Tektronix AWG5014 and partially
    ZI UHFLI.

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
                           initial_value=master_AWG, vals=vals.Strings())
        self.add_parameter('inter_element_spacing',
                           vals=vals.MultiType(vals.Numbers(0),
                                               vals.Enum('auto')),
                           set_cmd=self._set_inter_element_spacing,
                           get_cmd=self._get_inter_element_spacing)
        self._inter_element_spacing = 'auto'
        self.channels = set()
        self.last_sequence = None
        self.last_elements = None

        self._AWGs_prequeried_state = False

    # channel handling
    def define_channel(self, name, id, AWG=None, options=None):
        """
        The AWG object must be created before creating channels for that AWG

        Args:
            name: This name must be specified in pulses for them to play on
                  this channel.
            id:   channel id. For the Tektronix 5014 must be of the form
                  ch#(_m#) with # a number and the part in () optional.
                  For UHFQC must be 'ch1' or 'ch2'.
            AWG:  name of the AWG this channel is on
            options: extra options for this channel.
        """
        if options is None:
            options = {}

        if AWG is None:
            AWG = self.default_AWG()

        if name in self.channels:
            raise KeyError("Channel named '{}' already defined".format(name))

        for c_name in self.channels:
            if id == self.get('{}_id'.format(c_name)) and \
               AWG == self.get('{}_AWG'.format(c_name)):
                raise KeyError("Channel '{}' on '{}' already in use in channel"
                               " {}, can not add channel {}."
                               .format(id, AWG, c_name, name))

        obj = self.AWG_obj(AWG=AWG)
        fail = None
        try:
            super()._create_parameters(name, id, obj, options)
        except AttributeError as e:
            fail = e
        if fail is not None:
            raise TypeError('Unsupported AWG instrument: {} of type {}. '
                            .format(obj.name, type(obj)) + str(fail))
        self.channels.add(name)


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
            name = self.get('{}_AWG'.format(chan))
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
        if self._AWGs_prequeried_state:
            return self._clocks[channel]
        else:
            fail = None
            obj = self.AWG_obj(channel=channel)
            cid = self.get('{}_id'.format(channel))
            try:
                return super()._clock(obj, cid)
            except AttributeError as e:
                fail = e
            if fail is not None:
                raise TypeError('Unsupported AWG instrument: {} of type {}. '
                                .format(obj.name, type(obj)) + str(fail))

    def used_AWGs(self):
        """
        Returns:
            A set of the names of the active AWGs registered
        """
        res = set()
        for chan in self.channels:
            if self.get('{}_active'.format(chan)):
                res.add(self.get('{}_AWG'.format(chan)))
        return res

    def start(self):
        """
        Start the active AWGs. If multiple AWGs are used in a setup where the
        slave AWGs are triggered by the master AWG, then the slave AWGs must be
        running and waiting for trigger when the master AWG is started to
        ensure synchronous playback.
        """
        if self.master_AWG() is None:
            for AWG in self.used_AWGs():
                self._start_AWG(AWG)
        else:
            for AWG in self.used_AWGs():
                if AWG != self.master_AWG():
                    self._start_AWG(AWG)
            tstart = time.time()
            for AWG in self.used_AWGs():
                if AWG != self.master_AWG():
                    good = False
                    while time.time() - tstart < 10:
                        if self._is_AWG_running(AWG):
                            good = True
                            break
                        else:
                            time.sleep(0.1)
                    if not good:
                        raise Exception('AWG {} did not start in 10s'
                                        .format(AWG))
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
            channels = self.channels

        # find the awgs that need to have the elements precompiled
        precompiled_channels = set()
        precompiled_awgs = set()
        normal_channels = set()
        normal_awgs = set()
        for c in channels:
            awg = self.get('{}_AWG'.format(c))
            if self.get('{}_precompile'.format(c)):
                if awg in normal_awgs:
                    raise Exception('AWG {} contains precompiled and not '
                                    'precompiled segments. Can not program '
                                    'AWGs'.format(awg))
                precompiled_awgs.add(awg)
                precompiled_channels.add(c)
            else:
                if awg in precompiled_awgs:
                    raise Exception('AWG {} contains precompiled and not '
                                    'precompiled channels. Can not program '
                                    'AWGs'.format(awg))
                normal_awgs.add(awg)
                normal_channels.add(c)

        # prequery all AWG clock values and AWG amplitudes
        self.AWGs_prequeried(True)

        # create the precompiled elements
        elements = {el.name: el for el in elements}
        if len(precompiled_awgs) > 0:
            precompiled_sequence = sequence.precompiled_sequence()
            #elements is a list of dictionaries with info about the element
            # a segment consists of elements that have been merged by precompiled_seq
            for segment in precompiled_sequence.elements:
                element_list = []
                # segment['wavename'] is a tuple having all the wfnames
                for wf in segment['wfname']:
                    if wf == 'codeword':
                        for wfname in precompiled_sequence.codewords.values():
                            wf_new = wfname
                            break
                    else:
                            wf_new = wf
                    # wvnames are used to reference objects of the class Elements
                    element_list.append(elements[wf_new])
                new_elt = element.combine_elements(element_list)

                segment['wfname'] = new_elt.name
                elements[new_elt.name] = new_elt
            self.precompiled_sequence = precompiled_sequence
        self.sequence = sequence
        self.elements = elements


        # dict(name of AWG ->
        #      dict(i, element name ->
        #           dict(channel id ->
        #                waveform data)))
        AWG_wfs = {}
        #elements is dictionary containing all elements. 
        for i, el in enumerate(elements.values()):
            #for precompiled elements: the name is a tuple created by combine_elements 
            #containing all the names of the elements, otherwise it is just a normal string
            if isinstance(el.name, tuple):
                _, waveforms = el.normalized_waveforms(precompiled_channels)
                #waveforms is dictionary with channels as keys and amplitudes as values
            else:
                _, waveforms = el.normalized_waveforms(normal_channels)
            for cname in waveforms:
                #goes through all channel names in waveforms
                if cname not in channels:
                    continue
                if not self.get('{}_active'.format(cname)):
                    continue
                cAWG = self.get('{}_AWG'.format(cname))
                cid = self.get('{}_id'.format(cname))
                if cAWG not in AWGs:
                    continue
                if cAWG not in AWG_wfs:
                    #creates a new dictionary for the AWG
                    AWG_wfs[cAWG] = {}
                if (i, el.name) not in AWG_wfs[cAWG]:
                    #for the tuple (i, el.name) as key we create a new dictionary
                    AWG_wfs[cAWG][i, el.name] = {}
                AWG_wfs[cAWG][i, el.name][cid] = waveforms[cname]
                #dictionary which keys are names of AWGs and dictionaries as values which
                #have tuples (i, el.name) as keys and dictionaries as values which have
                #channel id as keys and the corresponding waveforms as values

        self.AWG_wfs = AWG_wfs

        for AWG in AWG_wfs:
            obj = self.AWG_obj(AWG=AWG)
            #returns the instance of the class AWG that is requested
            if AWG in normal_awgs:
                #programs the AWG to play the elements in the order as in sequence
                self._program_awg(obj, sequence, AWG_wfs[AWG], loop=loop)
            else:
                self._program_awg(obj, precompiled_sequence, AWG_wfs[AWG],
                                  loop=loop)


        self.AWGs_prequeried(False)

    def _program_awg(self, obj, sequence, el_wfs, loop=True):
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
        """
        fail = None
        try:
            super()._program_awg(obj, sequence, el_wfs, loop)
        except AttributeError as e:
            fail = e
        if fail is not None:
            raise TypeError('Unsupported AWG instrument: {} of type {}. '
                            .format(obj.name, type(obj)) + str(fail))

    def _start_AWG(self, AWG):
        obj = self.AWG_obj(AWG=AWG)
        obj.start()

    def _stop_AWG(self, AWG):
        obj = self.AWG_obj(AWG=AWG)
        obj.stop()

    def _is_AWG_running(self, AWG):
        fail = None
        obj = self.AWG_obj(AWG=AWG)
        try:
            return super()._is_AWG_running(obj)
        except AttributeError as e:
            fail = e
        if fail is not None:
            raise TypeError('Unsupported AWG instrument: {} of type {}. '
                            .format(obj.name, type(obj)) + str(fail))

    def _set_default_AWG(self, AWG):
        if AWG is None:
            class Object(object):
                pass
            self.AWG = Object()
            self.AWG.name = None
        self.AWG = self.AWG_obj(AWG=AWG)

    def _get_default_AWG(self):
        return self.AWG.name

    def _set_inter_element_spacing(self, val):
        self._inter_element_spacing = val

    def _get_inter_element_spacing(self):
        if self._inter_element_spacing != 'auto':
            return self._inter_element_spacing
        else:
            max_spacing = 0
            for c in self.channels:
                max_spacing = max(max_spacing, self.get(
                    '{}_inter_element_deadtime'.format(c)))
            return max_spacing

    def AWGs_prequeried(self, status=None):
        if status is None:
            return self._AWGs_prequeried_state
        elif status:
            self._AWGs_prequeried_state = False
            self._clocks = {}
            for c in self.channels:
                self._clocks[c] = self.clock(c)
                self.get(c + '_amp') # prequery also the output amplitude values
            self._AWGs_prequeried_state = True
        else:
            self._AWGs_prequeried_state = False

    def _id_channel(self, cid, AWG):
        """
        Returns the channel name corresponding to the channel with id `cid` on
        the AWG `AWG`.

        Args:
            cid: An id of one of the channels.
            AWG: The name of the AWG.

        Returns: The corresponding channel name. If the channel is not found,
                 returns `None`.
        """
        for cname in self.channels:
            if self.get('{}_AWG'.format(cname)) == AWG and \
               self.get('{}_id'.format(cname)) == cid:
                return cname
        return None

# translate_from = ''.join(set(string.printable) - set(string.ascii_letters) -
#                          set(string.digits))
# translate_to = ''.join(['_'] * len(translate_from))
# translation_table = str.maketrans(translate_from, translate_to)
# def simplify_name(name):
#     if name is None:
#         return None
#     ret = name.translate(translation_table)
#     if len(ret) == 0 or ret[0] in string.digits:
#         return '_' + ret
#     else:
#         return ret

import base64
import hashlib
def simplify_name(s):
    x = base64.urlsafe_b64encode(hashlib.sha1(
        s.encode('ascii')).digest()).decode()[:20]
    xnew = []
    for c in x:
        xnew.append(chr((ord(c) - ord('A'))%26 + ord('A')))
    return ''.join(xnew)