from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
station = None
from pycqed.measurement.waveform_control import sequence
from copy import deepcopy

class awg_seq_swf(swf.Hard_Sweep):
    def __init__(self, awg_seq_func, awg_seq_func_kwargs,
                 parameter_name=None, unit='a.u.',
                 upload=True, return_seq=False):
        super().__init__()
        self.upload = upload
        self.awg_seq_func = awg_seq_func
        self.awg_seq_func_kwargs = awg_seq_func_kwargs
        self.unit = unit
        self.name = 'swf_'+ awg_seq_func.__name__

        if parameter_name != None:
            self.parameter_name = parameter_name
        else:
            self.parameter_name = 'points'

    def prepare(self, **kw):
        if self.parameter_name != 'points':
            self.awg_seq_func_kwargs[self.parameter_name] = self.sweep_points
        if self.upload:
            self.awg_seq_func(**self.awg_seq_func_kwargs)


def two_qubit_AllXY(pulse_dict, q0='q0', q1='q1', RO_target='all',
                    sequence_type ='simultaneous',
                    replace_q1_pulses_X180=False,
                    double_points=True,
                    verbose=False, upload=True,
                    return_seq=False):
    """
    Performs an AllXY sequence on two qubits.
    Has the option of replacing pulses on q1 with pi pulses

    Args:
        pulse_dict   (dict) : dictionary containing all pulse parameters
        q0, q1        (str) : target qubits for the sequence
        RO_target     (str) : target for the RO, can be a qubit name or 'all'
        sequence_type (str) : sequential| interleaved|simultaneous|sandwiched
            describes the order of the AllXY pulses
        replace_q1_pulses_X180 (bool) : if True replaces all pulses on q1 with
            X180 pulses.

        double_points (bool) : if True measures each point in the AllXY twice
        verbose       (bool) : verbose sequence generation
        upload        (bool) :
    """
    seq_name = 'two_qubit_AllXY_{}_{}'.format(q0, q1)
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []

    AllXY_pulse_combinations = [['I ', 'I '], ['X180 ', 'X180 '], ['Y180 ', 'Y180 '],
                          ['X180 ', 'Y180 '], ['Y180 ', 'X180 '],
                          ['X90 ', 'I '], ['Y90 ', 'I '], ['X90 ', 'Y90 '],
                          ['Y90 ', 'X90 '], ['X90 ', 'Y180 '], ['Y90 ', 'X180 '],
                          ['X180 ', 'Y90 '], ['Y180 ', 'X90 '], ['X90 ', 'X180 '],
                          ['X180 ', 'X90 '], ['Y90 ', 'Y180 '], ['Y180 ', 'Y90 '],
                          ['X180 ', 'I '], ['Y180 ', 'I '], ['X90 ', 'X90 '],
                          ['Y90 ', 'Y90 ']]
    if double_points:
        AllXY_pulse_combinations = [val for val in AllXY_pulse_combinations
                                    for _ in (0, 1)]

    if sequence_type == 'simultaneous':
        pulse_dict = deepcopy(pulse_dict) # prevents overwriting of dict
        for key in pulse_dict.keys():
            if q1 in key:
                pulse_dict[key]['refpoint'] = 'start'
                pulse_dict[key]['pulse_delay'] = 0

    pulse_list = []
    if not replace_q1_pulses_X180:
        for pulse_comb in AllXY_pulse_combinations:
            if sequence_type == 'interleaved' or sequence_type=='simultaneous':
                pulse_list+= [[pulse_comb[0] + q0] + [pulse_comb[0] + q1] +
                              [pulse_comb[1] + q0] + [pulse_comb[1] + q1] +
                              ['RO ' + RO_target]]
            elif sequence_type == 'sequential':
                pulse_list+= [[pulse_comb[0] + q0] + [pulse_comb[1] + q0] +
                              [pulse_comb[0] + q1] + [pulse_comb[1] + q1] +
                              ['RO ' + RO_target]]
            elif sequence_type == 'sandwiched':
                pulse_list+= [[pulse_comb[0] + q1] + [pulse_comb[0] + q0] +
                              [pulse_comb[1] + q0] + [pulse_comb[1] + q1] +
                              ['RO ' + RO_target]]
            else:
                raise ValueError("sequence_type {} must be in".format(sequence_type)+
                    " ['interleaved', simultaneous', 'sequential', 'sandwiched']")
    else:
        for pulse_comb in AllXY_pulse_combinations:
            if sequence_type == 'interleaved' or sequence_type=='simultaneous':
                pulse_list+= [[pulse_comb[0] + q0] + ['X180 ' + q1] +
                              [pulse_comb[1] + q0] + ['X180 ' + q1] +
                              ['RO ' + RO_target]]
            elif sequence_type == 'sequential':
                pulse_list+= [[pulse_comb[0] + q0] + [pulse_comb[1] + q0] +
                              ['X180 ' + q1] + ['X180 ' + q1] +
                              ['RO ' + RO_target]]
            elif sequence_type == 'sandwiched':
                pulse_list+= [['X180 ' + q1] + [pulse_comb[0] + q0] +
                              [pulse_comb[1] + q0] + ['X180 ' + q1] +
                              ['RO ' + RO_target]]
            else:
                raise ValueError("sequence_type {} must be in".format(sequence_type)+
                    " ['interleaved', simultaneous', 'sequential', 'sandwiched']")

    for i, pulse_comb in enumerate(pulse_list):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]
        el = multi_pulse_elt(i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


