import enum
import sys
import logging
import copy


def get_timetuples_since_event(timing_grid: list, target_labels: list,
                               start_label: str, end_label: str=None,
                               convert_clk_to_ns: bool=False) ->list:
    """
    Searches for target labels in a timing grid and returns the time between
    the the events and the timepoint corresponding to the start_label.

    N.B. timing is in clocks

    returns
        time_tuples (list) of tuples (time, label)
        end_time (int)
    """
    time_tuples = []
    for target_label in target_labels:
        time_points = get_timepoints_from_label(
            timing_grid=timing_grid, target_label=target_label,
            start_label=start_label, end_label=end_label,
            convert_clk_to_ns=convert_clk_to_ns)
        t0 = time_points['start_tp'].absolute_time
        for tp in time_points['target_tps']:
            time_tuples.append((tp.absolute_time-t0, target_label))
    end_time = time_points['end_tp'].absolute_time
    return time_tuples, end_time


def get_timepoints_from_label(
        timing_grid: list, target_label: str,
        start_label: str =None, end_label: str=None,
        convert_clk_to_ns: bool =False)->dict:
    """
    Extract timepoints from a timing grid based on their label.
        target_label : the label to search for in the timing grid
        timing_grid : a list of time_points to search in

    N.B. timing grid is in clocks

    returns
        timepoints (dict) with keys
            'start_tp'    starting time_point
            'target_tps'  list of time_points found with label target_label
            'end_tp'      end time_point
    """
    timing_grid = copy.deepcopy(timing_grid)

    start_idx = 0
    end_idx = None
    if start_label is not None:
        for i, tp in enumerate(timing_grid):
            if tp.label == start_label:
                start_idx = i
                break
        if start_idx == 0:
            logging.warning('Could not find {} in timing grid'.format(
                start_label))

    if end_label is not None:
        for j, tp in enumerate(timing_grid[start_idx:]):
            if tp.label == end_label:
                end_idx = start_idx + j
                break
        if end_idx is None:
            logging.warning('Could not find {} in timing grid'.format(
                end_label))

    target_indices = []

    if end_idx is not None:

        for k, tp in enumerate(timing_grid[start_idx:end_idx]):
            if tp.label == target_label:
                target_indices.append(start_idx + k)
        timepoints = {
            'start_tp': timing_grid[start_idx],
            'target_tps': [timing_grid[i] for i in target_indices],
            'end_tp': timing_grid[end_idx]}
    else:
        for k, tp in enumerate(timing_grid[start_idx:]):
            if tp.label == target_label:
                target_indices.append(start_idx + k)
        timepoints = {
            'start_tp': timing_grid[start_idx],
            'target_tps': [timing_grid[i] for i in target_indices],
            'end_tp': timing_grid[-1]}

    if convert_clk_to_ns:
        clock_cycle_time = 5  # 5ns per clock
        timepoints['start_tp'].absolute_time *= clock_cycle_time
        for target_tp in timepoints['target_tps']:
            target_tp.absolute_time *= clock_cycle_time
        timepoints['end_tp'].absolute_time *= clock_cycle_time
    return timepoints


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_positive_number(s):
    if is_number(s) is False:
        return False
    if float(s) > 0:
        return True
    else:
        return False


def is_natural(s):
    if is_int(s) is False:
        return False
    if int(s) >= 0:
        return True
    else:
        return False


def is_integer_array(arr):
    return all(isinstance(item, int) for item in arr)


def bitfield(n, width, little_endian=True):
    '''
    Return the 2's complement of the integer number $x$
    for a given bitwidth $n$.
    '''
    if (not is_natural(n)):
        raise ValueError('bitfield: n ({}) is not a positive'
                         ' integer number.'.format(n))

    if (not is_natural(width)):
        raise ValueError('bitfield: width ({}) is not a positive'
                         ' integer number.'.format(width))

    bin_str = '{0:{fill}{width}b}'.format(n, fill='0', width=width)
    if little_endian is True:
        return [int(digit) for digit in bin_str]
    else:
        return [int(digit) for digit in reversed(bin_str)]


def min_non_zero(a):
    '''
    If there is non zero element in a, return the minimum non-zero element.
    Otherwise, return 0.
    '''
    non_zero_a = [i for i in a if i != 0]
    if non_zero_a == []:
        return 0
    else:
        return min(non_zero_a)


def raw_print(s):
    sys.stdout.write(s)


def config_is_valid(config: dict)-> bool:
    """
    Tests if a qasm configuration specification is valid.
    returns True if it is valid and raises an exception otherwise
    """

    # the config can optionally contain a qubit map specification
    expected_keys = {'operation dictionary', 'hardware specification', 'luts'}

    if not expected_keys.issubset(set(config.keys())):
        raise ValueError('Config misses keys {}'.format(
            expected_keys - set(config.keys())))

    hw_spec = config['hardware specification']
    hw_spec_keys = {'qubit list', 'init time',
                    'cycle time', 'qubit_cfgs'}
    if not hw_spec_keys.issubset(set(hw_spec.keys())):
        raise ValueError('hardware specification misses keys {}'.format(
            hw_spec_keys - set(hw_spec.keys())))

    if not len(hw_spec['qubit list']) == len(hw_spec['qubit_cfgs']):
        raise ValueError('different number of qubits than qubit_cfgs')

    if not is_integer_array(hw_spec['qubit list']):
        raise ValueError('"qubit list" in the configuration file is not an'
                         ' integer array:\n\t{}'.format(
                             hw_spec['qubit list']))

    if not type(hw_spec['init time'] == int):
        raise ValueError('init_time {} must be int'.format(
            hw_spec['init time']))
    if not is_positive_number(hw_spec['init time']):
        raise ValueError('"init time" ({}) in the configuration file'
                         ' is not an positive number.'.format(
                             hw_spec['init time']))

    if not type(hw_spec['cycle time'] == int):
        raise ValueError('init_time {} must be int'.format(
            hw_spec['cycle time']))
    if not is_positive_number(hw_spec['cycle time']):
        raise ValueError('"cycle time" ({}) in the configuration file'
                         ' is not an positive number.'.format(
                             hw_spec['cycle time']))

    for i, ch in enumerate(hw_spec['qubit_cfgs']):
        if not set(ch.keys()).issubset({'rf', 'flux', 'measurement'}):
            raise ValueError('unexpected key found in qubit config')

    return True


class EventType(enum.Enum):
    '''
    Current type definition does not separate the technology-dependent part
    and technology-independent part.
    '''

    # N.B. integers below can be replaced by enum.auto() in python > 3.6
    NONE_EVENT = 1
    DECLARE = 2
    MAP = 3
    WAIT = 4
    RF = 5
    FLUX = 6
    MEASUREMENT = 7

    def __str__(self):
        return self.name.lower()


class time_point():
    def __init__(self, label='', absolute_time=-1, following_waiting_time=-1):
        self.label = label
        self.absolute_time = absolute_time
        self.following_waiting_time = following_waiting_time
        self.parallel_events = []

    def __repr__(self):
        base_str = ('time_point(label={}, absolute_time={}, '
                    'following_waiting_time={})')
        rep = base_str.format(self.label, self.absolute_time,
                              self.following_waiting_time)
        return rep


class qasm_event():

    def __init__(self):
        self.line_number = -1
        self.event_type = EventType.NONE_EVENT
        self.name = ''
        self.params = None
        self.duration = 0
        self.channel_latency = 0

    def __repr__(self):
        base_str = ('qasm_event({}, params={}, duration={})')
        rep = base_str.format(self.name, self.params, self.duration)
        return rep

    def __str__(self):
        base_str = ('qasm_event: "{:10s}", params={}, duration={}')
        rep = base_str.format(self.name, self.params, self.duration)
        return rep


class qumis_event():

    def __init__(self):
        self.qumis_name = ''
        self.codeword = -1
        self.awg_nr = -1    # used for pulse instruction
        self.duration = 0
        self.format = []    # used for trigger instruction
        self.trigger_bit = -1
        self.codeword_bit = []
        self.set_bits = []

    def __repr__(self):
        base_str = ('qumis_event({}, codeword={}, awg_nr={}, duration={},' +
                    'trigger_bit={}, codeword_bit={}, set_bits={})')
        rep = base_str.format(self.qumis_name, self.codeword, self.awg_nr,
                              self.duration, self.trigger_bit,
                              self.codeword_bit, self.set_bits)
        return rep

    def __str__(self):

        base_str = ('qumis_event: "{}", \n\tcodeword={}, \n\tawg_nr={},' +
                    ' \n\tduration={}, \n\ttrigger_bit={}, ' +
                    '\n\tcodeword_bit={}, \n\tset_bits={}\n')
        rep = base_str.format(self.qumis_name, self.codeword, self.awg_nr,
                              self.duration, self.trigger_bit,
                              self.codeword_bit, self.set_bits)
        return rep


class prog_line():

    def __init__(self, number=-1, content=''):
        self.number = number
        self.content = content

    def __repr__(self):
        return "prog_line(number={}, content={})".format(self.number, self.content)

    def __str__(self):
        return "{}: {}".format(self.number, self.content)


def lower_dict_key(origin_dict):
    return dict((k.lower(), v) for k, v in origin_dict.items())
