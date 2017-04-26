'''
This file implements a compiler that translates QASM program into QuMIS
instruction. More precisely, it defines an Intermediate Representation (IR)
and the process to translate the IR into QuMIS instructions.

It should contain the following information:
1. Timing information
    - Starting point
    - Duration
    - (optional) Ending point
2. The event
3. Loop
4. Register
5. Memory


The QASM file is case insensitive.

'''


'''
IR definition:
<time point, events>
time point: label, absolute time, pre_waiting time
event: type, duration
'''
import enum
import logging
import json
import string
import pycqed as pq
import os
import sys
import bisect
import copy


INIT_ALL_WAITING_TIME = 200000  # ns


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


def is_positive_integer(s):
    if is_int(s) is False:
        return False
    if int(s) > 0:
        return True
    else:
        return False


def is_integer_array(arr):
    return all(isinstance(item, int) for item in arr)


def raw_print(s):
    sys.stdout.write(s)


class EventType(enum.Enum):
    '''
    Current type definition does not separate the technology-dependent part
    and technology-independent part.
    '''
    NONE_EVENT = enum.auto()
    DECLARE = enum.auto()
    MAP = enum.auto()
    WAIT = enum.auto()
    RF = enum.auto()
    FLUX = enum.auto()
    MEASURE = enum.auto()

    def __str__(self):
        return self.name.lower()


class time_point():

    def __init__(self, label='', absolute_time=-1, following_waiting_time=-1):
        self.label = label
        self.absolute_time = absolute_time
        self.following_waiting_time = following_waiting_time
        self.parallel_events = []


class qasm_event():

    def __init__(self):
        self.line_number = -1
        self.event_type = EventType.NONE_EVENT
        self.name = ''
        self.params = None
        self.duration = 0
        self.channel_latency = 0
        # self.start_time_point = time_point()

    def __str__(self):
        return "({}, {})".format(self.name, self.event_type)


class qumis_event():
    # class pulse():
    #     def __init__(self):
    #         self.qumis_name = 'pulse'
    #         self.codeword = -1

    # class trigger():
    #     def __init__(self):
    #         self.qumis_name = 'trigger'
    #         self.codeword = -1
    #         self.format = []    # used for trigger instruction
    #         self.trigger_bit = -1
    #         self.codeword_bit = []

    # class measure():
    #     def __init__(self):
    #         self.qumis_name = 'measure'

    # class measure_trigger():
    #     def __init__(self):
    #         self.qumis_name = 'trigger'
    #         self.trigger_bit = -1
    def __init__(self):
        self.qumis_name = ''
        self.codeword = -1
        self.awg_nr = -1    # used for pulse instruction
        self.duration = 0
        self.format = []    # used for trigger instruction
        self.trigger_bit = -1
        self.codeword_bit = []


user_op_type = {
    "rf": EventType.RF,
    "flux": EventType.FLUX,
    "measurement": EventType.MEASURE
}

default_op_dict = {
    "I": {
        "parameters": 1,
        "type": EventType.WAIT
    },
    "qubit": {
        "parameters": -1,
        "type": EventType.DECLARE
    },

    "Init_all": {
        "parameters": 0,
        "type": EventType.WAIT
    },

    "Measure": {
        "parameters": 1,
        "duration": 300,
        "type": EventType.MEASURE
    },

    "Map": {
        "parameters": 2,
        "type": EventType.MAP
    }
}


class prog_line():

    def __init__(self, number=-1, content=''):
        self.number = number
        self.content = content

    def __str__(self):
        return "{}: {}".format(self.number, self.content)


def lower_dict_key(origin_dict):
    return dict((k.lower(), v) for k, v in origin_dict.items())


class QASM_QuMIS_Compiler():

    def __init__(self, filename=None, verbose_level=1):
        '''
        @param: filename, the QASM file to be compiled.
        @param: verbose_level, message level to be printed. integer, range
        from 0 to 6. The larger verbose_level is, the more information being
        printed. 0 prints everything. 6 prints nothing.
        '''
        self.filename = filename
        self.raw_lines = None
        self.prog_lines = None
        self.config_loaded = False
        self.qasm_op_dict = None
        self.declared_qubits = []
        self.qubit_map = {}
        self.verbose_level = verbose_level
        self.channel_latency_compensated = False

        self.config_filename = os.path.join(
            pq.__path__[0], 'instrument_drivers', 'physical_instruments',
            "_controlbox", "config.json")

    def compile(self):
        self.qumis_instructions = []
        self.read_file()
        self.line_to_event()
        self.build_dependency_graph()
        self.resolve_qubit_name()
        self.assign_timing_to_events()
        self.resolve_channel_latency()
        self.convert_to_hw_trigger()
        self.gen_full_time_grid()
        self.convert_to_qumis()
        return self.qumis_instructions

    def build_dependency_graph(self):
        pass

    def load_config(self):
        if (self.config_loaded is True):
            return

        self.qasm_op_dict = None

        with open(self.config_filename) as data_file:
            data = json.load(data_file)

        self.user_qasm_op_dict = data["operation dictionary"]
        for key in self.user_qasm_op_dict:
            op_spec = self.user_qasm_op_dict[key]
            op_type_str = op_spec["type"]
            if op_type_str not in user_op_type:
                se = SyntaxError("unsupported operation type ({})"
                                 " found.".format(op_spec["type"]))
                se.filename = self.config_filename
                raise se

            op_type_enum = user_op_type[op_type_str]
            op_spec["type"] = op_type_enum
            self.user_qasm_op_dict[key] = op_spec

        self.qasm_op_dict = {**default_op_dict, **self.user_qasm_op_dict}
        self.qasm_op_dict = lower_dict_key(self.qasm_op_dict)
        self.hardware_spec = data["hardware specification"]
        self.luts = data["luts"]
        self.print_op_dict()

        self.channels = self.hardware_spec["channels"]
        self.check_channel_format()

        self.physical_qubits = self.hardware_spec["qubit list"]
        if is_integer_array(self.physical_qubits) is False:
            raise ValueError('"qubit list" in the configuration file is not an'
                             ' integer array:\n\t{}'.format(
                                 self.physical_qubits))

        self.cycle_time = self.hardware_spec["cycle time"]
        if is_positive_number(self.cycle_time) is False:
            raise ValueError('"cycle time" ({}) in the configuration file'
                             ' is not an positive number.'.format(
                                 self.cycle_time))
        self.cycle_time = float(self.cycle_time)

        self.config_loaded = True

    def check_channel_format(self):
        pass

    def read_file(self):
        '''
        Read all lines in the file.
        '''
        try:
            prog_file = open(self.filename, 'r', encoding="utf-8")
            logging.info("open file", self.filename, "successfully.")
        except:
            raise OSError('\tError: Fail to open file ' + self.filename + ".")

        self.raw_lines = []
        self.prog_lines = []  # after removing comments.
        for line_number, line_content in enumerate(prog_file):
            self.raw_lines.append(
                prog_line(line_number + 1, line_content.strip(' \t\n\r')))

            line_content = self.remove_comment(line_content)
            line_content = line_content.lower()
            if (len(line_content) == 0):  # skip empty line and comment
                continue
            self.prog_lines.append(prog_line(line_number + 1, line_content))

        prog_file.close()
        self.print_lines()

    def print_lines(self):
        if self.verbose_level <= 1:
            print("Raw Lines:")
            for line in self.raw_lines:
                print(line)
            print("")

        if self.verbose_level <= 2:
            print("Program Lines:")
            for line in self.prog_lines:
                print(line)

            print("")

    def print_op_dict(self):
        if self.verbose_level < 3:
            print("QASM operation dictionary:")
            for key, value in self.qasm_op_dict.items():
                raw_print("{0: >10} : ".format(key))
                print(value)
            print("\n")

    def print_raw_events(self):
        if self.verbose_level < 4:
            print("Raw events:")
            for i, raw_events in enumerate(self.raw_event_list):
                raw_print("{0:5d} : ".format(i))
                for re in raw_events:
                    raw_print("({}, {}, {})".format(re.event_type,
                                                    re.name, re.params))
                raw_print("\n")
        print("")

    def print_timing_events(self):
        if self.verbose_level < 4:
            print("Timing events:")
            for i, timing_events in enumerate(self.timing_event_list):
                raw_print("{0:5d} : ".format(i))
                for re in timing_events:
                    raw_print("({}, {}, {})".format(re.event_type,
                                                    re.name, re.params))
                raw_print("\n")
        print("")

    def print_timing_grid(self):
        if self.verbose_level < 5:
            print("Timing gird:")
            if len(self.timing_grid) == 0:
                print("Empty timing grid.")
                return

            if self.timing_grid[0].absolute_time is not None:
                absolute_time_generated = True
            else:
                absolute_time_generated = False

            if absolute_time_generated is True:
                print("{0: >6s}  {1: <10s}"
                      "   Events".format("Idx", "absolute"))
            else:
                print("{0: >6s}  {1: <10s}   Events".format("Idx",
                                                            "next wait"))
            i = 0
            for tp in self.timing_grid:
                raw_print("{0:5d}:".format(i))
                if absolute_time_generated is True:
                    raw_print("   {0: <8d}    ".format(tp.absolute_time))
                else:
                    raw_print("   {0: <8d}    ".format(
                        tp.following_waiting_time))
                self.print_event_list(tp.parallel_events)
                raw_print('\n')
                i = i + 1

        print("")

    def print_hw_timing_grid(self):
        if self.verbose_level < 5:
            print("Hardware trigger timing gird:")
            if len(self.hw_timing_grid) == 0:
                print("Empty trigger timing grid.")
                return

            print("{0: >6s}  {1: <10s}"
                  "    Events".format("Idx", "absolute"))
            i = 0
            for tp in self.hw_timing_grid:
                raw_print("{0:5d}:".format(i))
                raw_print("   {0: <8d}    ".format(tp.absolute_time))
                self.print_event_list(tp.parallel_events)
                raw_print('\n')
                i = i + 1

        print("")

    def print_event_list(self, event_list):
        raw_print(" ")
        if event_list == []:
            raw_print('-')
            return

        if hasattr(event_list[0], 'event_type'):
            for e in event_list:
                raw_print(e.name)
                raw_print(" ")
                if len(e.params) == 2:
                    raw_print(str(e.params[0]))
                    raw_print(", ")
                    raw_print(str(e.params[1]))
                else:
                    raw_print(str(e.params[0]))
                if e.channel_latency != 0:
                    raw_print("({})".format(e.channel_latency))
                raw_print('; ')
        else:  # qumis event list
            for e in event_list:
                raw_print(e.qumis_name)
                if e.qumis_name.lower() == "measure":
                    pass
                elif e.qumis_name.lower() == "pulse":
                    raw_print(" ")
                    raw_print("awg")
                    raw_print(str(e.awg_nr))
                    raw_print("(%s)"%str(e.codeword))
                elif e.qumis_name.lower() == "trigger":
                    raw_print(" ")
                    raw_print("bits:")
                    if e.trigger_bit != -1:
                        raw_print("[{}]".format(str(e.trigger_bit)))
                    if e.codeword_bit != []:
                        raw_print(str(e.codeword_bit))
                    if e.format != []:
                        raw_print(" f:")
                        raw_print(str(e.format))
                    else:
                        raw_print(" d:")
                        raw_print(str(e.duration))
                    if e.codeword != -1:
                        raw_print(" cw: %s" %str(e.codeword))
                raw_print('; ')


    @classmethod
    def remove_comment(self, line):
        line = line.split('#', 1)[0]  # remove anything after '#' symbole
        line = line.strip(' \t\n\r')  # remove whitespace
        return line

    @classmethod
    def is_single_line_op(self, qasm_op_type):
        if (qasm_op_type == EventType.WAIT) or \
                (qasm_op_type == EventType.DECLARE) or \
                (qasm_op_type == EventType.MAP):
            return True
        else:
            return False

    def line_to_event(self):
        '''
        1. Convert each line in the QASM file into an event.
        2. Perform part of the syntax check.
        3. Translate parameter format if necessary.
        Timing information of events is not generated.
        raw_event_list contains all events before resolving timing information.
        '''
        self.load_config()
        self.raw_event_list = []
        for line in self.prog_lines:
            # print("line in prog_lines:", line)
            # print("line_content:", line.content)
            events = self.get_parallel_qasm_ops(line.content)
            # print("events:", events)
            raw_events = []
            for e in events:
                # print("e in events:", e)
                qasm_op_name = self.get_qasm_op_name(e)
                if qasm_op_name not in self.qasm_op_dict:
                    se = SyntaxError("unsuppported QASM operation {}.".format(
                        qasm_op_name))
                    se.filename = self.filename
                    se.lineno = line.number
                    raise se

                qasm_op_type = self.qasm_op_dict[qasm_op_name]["type"]

                if self.is_single_line_op(qasm_op_type) and (len(events) != 1):
                    se = SyntaxError("QASM instruction {} should exclusively "
                                     "occupy a line.".format(qasm_op_name))
                    se.filename = self.filename
                    se.lineno = line.number
                    raise se

                # check parameter
                qasm_op_params = self.get_qasm_op_params(e)
                if (qasm_op_name == "qubit") and (len(qasm_op_params) < 1):
                    se = SyntaxError("the QASM instruction qubit should "
                                     "contain at least one parameter as "
                                     "the declared qubit.".format(
                                         qasm_op_name))
                    se.filename = self.filename
                    se.lineno = line.number
                    raise se

                expected_num_of_params = self.qasm_op_dict[
                    qasm_op_name]["parameters"]
                if (qasm_op_name != "qubit") and \
                        (len(qasm_op_params) != expected_num_of_params):
                    se = SyntaxError("unexpected number of parameters for the"
                                     " QASM instruction {}. {} parameter(s)"
                                     " expected. Offered: {}.".format(
                                         qasm_op_name,
                                         expected_num_of_params,
                                         qasm_op_params))
                    se.filename = self.filename
                    se.lineno = line.number
                    raise se

                if (qasm_op_type == EventType.WAIT) and \
                        (expected_num_of_params == 1):
                    waiting_time, = qasm_op_params
                    if is_int(waiting_time) is False:
                        se = SyntaxError("parameter {} is not an "
                                         "integer.".format(waiting_time))
                        se.filename = self.filename
                        se.lineno = line.number
                        raise se

                    waiting_time = int(waiting_time)
                    if waiting_time < 0:
                        ve = ValueError("parameter {} is not "
                                        "positive.".format(waiting_time))
                        ve.filename = self.filename
                        ve.lineno = line.number
                        raise se

                    qasm_op_params = [waiting_time]

                raw_event = qasm_event()
                raw_event.line_number = line.number
                raw_event.name = qasm_op_name
                raw_event.params = qasm_op_params
                raw_event.event_type = qasm_op_type

                raw_event.duration = 0
                raw_events.append(raw_event)
            self.raw_event_list.append(raw_events)

    def is_wait_instr(self, qasm_op_name):
        description = self.qasm_op_dict[qasm_op_name]
        if (description["type"] == EventType.WAIT):
            return True
        else:
            return False

    @classmethod
    def is_wait_line(self, events):
        if events[0].event_type == EventType.WAIT:
            return True
        else:
            return False

    @classmethod
    def is_q_op_event(self, event):
        op_type = event.event_type
        if ((op_type == EventType.MEASURE) or
                (op_type == EventType.FLUX) or
                (op_type == EventType.RF)):
            return True
        else:
            return False

    @classmethod
    def get_parallel_qasm_ops(self, op_line):
        return [rawEle.replace(',', ' ').strip()
                for rawEle in op_line.split("|")]

    @classmethod
    def get_qasm_op_name(self, qasm_op):
        if qasm_op == '':
            raise ValueError("QASM operation cannot be empty")
        return qasm_op.split()[0]

    @classmethod
    def get_qasm_op_params(self, qasm_op):
        if qasm_op == '':
            raise ValueError("QASM operation cannot be empty")
        return qasm_op.split()[1:]

    def assign_timing_to_events(self):
        '''
        Resolve qubit name and map declared qubits into physical qubits.
        Arrange all operations into a timing gird.
        '''
        self.timing_grid = []

        for timing_events in self.timing_event_list:
            # each line starts a new time point
            # two thing to do for this time point:
            # 1. determine what happens at this moment
            # 2. determine the following waiting time
            tp = time_point()

            # determine the following waiting time
            if self.is_wait_line(timing_events):
                timing_event = timing_events[0]
                op_name = timing_event.name
                if self.qasm_op_dict[op_name]["parameters"] == 1:
                    following_waiting_time = timing_event.params[0]
                elif op_name == "init_all":
                    following_waiting_time = INIT_ALL_WAITING_TIME
                else:
                    se = SyntaxError("unsupported instruction ({})"
                                     " found.".format(timing_event.name))
                    se.filename = self.filename
                    se.lineno = timing_event.line_number
                    raise se
            else:
                following_waiting_time = self.get_max_duration(timing_events)

            tp.following_waiting_time = following_waiting_time

            # determine what happens at this moment
            if self.is_wait_line(timing_events) is False:
                tp.parallel_events.extend(timing_events)

            self.timing_grid.append(tp)

        self.get_absolute_timing()

        if self.verbose_level < 4:
            print("End of assigning timing:")
            self.print_timing_grid()

    def resolve_qubit_name(self):
        self.build_qubit_map()
        self.map_qubits()
        if self.verbose_level < 4:
            print("End of resolving qubit name:")
            self.print_timing_events()

    def build_qubit_map(self):
        i = 0
        while i < len(self.raw_event_list):
            raw_events = self.raw_event_list[i]
            for raw_event in raw_events:
                if (raw_event.event_type == EventType.DECLARE):
                    self.extend_dec_qubit_list(raw_event)
                    self.raw_event_list.pop(i)
                elif (raw_event.event_type == EventType.MAP):
                    self.add_qubit_map(raw_event)
                    self.raw_event_list.pop(i)
                else:
                    i = i + 1

        if (self.verbose_level < 4):
            print("End of building qubit map:")
            self.print_raw_events()

    def map_qubits(self):
        self.timing_event_list = []
        for raw_events in self.raw_event_list:
            timing_events = []
            for raw_event in raw_events:
                if (raw_event.event_type == EventType.WAIT):
                    timing_events.append(raw_event)
                if self.is_q_op_event(raw_event):
                    params = [self.qubit_map[dec_qubit]
                              for dec_qubit in raw_event.params]
                    raw_event.params = params
                    timing_events.append(raw_event)
            self.timing_event_list.append(timing_events)

    def extend_dec_qubit_list(self, raw_event):
        for q in raw_event.params:
            if self.declared_qubits.count(q):
                se = SyntaxError("Redefinition of {}".format(q))
                se.filename = self.filename
                se.lineno = raw_event.line_number
                raise se
            else:
                self.declared_qubits.append(q)

        if (len(self.declared_qubits) > len(self.physical_qubits)):
            se = SyntaxError("More qubits declared ({}) than available phys"
                             "ical qubits ({}).".format(
                                 len(self.declared_qubits),
                                 len(self.physical_qubits)))
            se.filename = self.filename
            raise se
        if self.verbose_level <= 2:
            print("Extend declared qubit list:")
            print("\tDeclared qubits: {}".format(self.declared_qubits))
            print("")

    def add_qubit_map(self, raw_event):
        dec_qubit, phys_qubit = raw_event.params

        if (dec_qubit not in self.declared_qubits):
            se = SyntaxError("undefined qubit ({}) found.".format(dec_qubit))
            se.filename = self.filename
            se.lineno = raw_event.line_number
            raise se

        if (dec_qubit in self.qubit_map):
            se = SyntaxError("remapping of the qubit {}.".format(dec_qubit))
            se.filename = self.filename
            se.lineno = raw_event.line_number
            raise se

        if (is_int(phys_qubit) is False):
            se = SyntaxError("the target qubit ({}) is not"
                             " an integer.".format(phys_qubit))
            se.filename = self.filename
            se.lineno = raw_event.line_number
            raise se

        phys_qubit = int(phys_qubit)

        if (phys_qubit not in self.physical_qubits):
            se = SyntaxError("physical qubit ({}) is not available.".format(
                self.filename,
                raw_event.line_number,
                phys_qubit))
            se.filename = self.filename
            se.lineno = raw_event.line_number
            raise se

        self.qubit_map[dec_qubit] = int(phys_qubit)

    def resolve_channel_latency(self):
        self.decompose()
        self.load_channel_latency()
        self.compensate_channel_latency()

    def decompose(self):
        for ti, tp in enumerate(self.timing_grid):
            parallel_events = []
            for event in tp.parallel_events:
                if len(event.params) > 1:
                    parallel_events.extend(self.decompose_op(event))
                else:
                    parallel_events.append(event)
            self.timing_grid[ti].parallel_events = parallel_events

        if self.verbose_level <= 3:
            print("End of decompose operations:")
            self.print_timing_grid()

    def decompose_op(self, event):
        '''
        This function currently only splits the operation for each parameter.
        '''
        new_event_list = []
        for param in event.params:
            new_event = qasm_event()
            new_event.line_number = event.line_number
            new_event.event_type = event.event_type
            new_event.name = event.name
            new_event.params = [param]
            new_event.duration = event.duration
            new_event.channel_latency = event.channel_latency
            new_event_list.append(new_event)

            print('new event list:  ')
            for e in new_event_list:
                raw_print(str(e))
                raw_print(str(e.params))
                raw_print('\n')

        return new_event_list

    def load_channel_latency(self):
        for ti, tp in enumerate(self.timing_grid):
            for idx, event in enumerate(tp.parallel_events):
                target_qubit = event.params[0]
                channel = self.channels[target_qubit]
                event.channel_latency = \
                    channel[str(event.event_type)]["latency"]

                tp.parallel_events[idx] = event
            self.timing_grid[ti] = tp

        if self.verbose_level < 3:
            print("get channel latency:")
            self.print_timing_grid()

    def compensate_channel_latency(self):
        min_latency = self.get_min_channel_latency()
        if self.verbose_level < 4:
            print("Min latency is:", min_latency)
        ti = 0
        while ti < len(self.timing_grid):
            tp = self.timing_grid[ti]
            idx = 0
            while idx < len(tp.parallel_events):
                event = tp.parallel_events[idx]
                compensate_time = event.channel_latency - min_latency
                if compensate_time != 0:
                    tp.parallel_events.pop(idx)

                    new_absolute_time = tp.absolute_time - compensate_time
                    tp_index, match = self.search_time_point(self.timing_grid,
                        new_absolute_time)
                    if match is False:
                        new_tp = time_point(absolute_time=new_absolute_time)
                        self.timing_grid.insert(tp_index, new_tp)
                        ti = ti + 1
                        self.timing_grid[tp_index].\
                            parallel_events.append(event)
                    else:
                        self.timing_grid[tp_index-1].\
                            parallel_events.append(event)
                else:
                    idx = idx + 1
            ti = ti + 1

        if self.verbose_level < 4:
            print("End of compensting channel latency:")
            self.print_timing_grid()

    def search_time_point(self, timing_grid, target_absolute_time):
        absolute_time_list = [tp.absolute_time for tp in timing_grid]
        idx = bisect.bisect(absolute_time_list, target_absolute_time)
        if len(absolute_time_list) == 0 or (idx == 0):
            return (idx, False)
        elif absolute_time_list[idx-1] == target_absolute_time:
            return (idx, True)
        else:
            return (idx, False)

    def get_min_channel_latency(self):
        return min([min([int(channel[sub_channel]["latency"])
                         for sub_channel in channel])
                    for channel in self.channels])

    def get_absolute_timing(self):
        current_time = 0
        for tp in self.timing_grid:
            tp.absolute_time = current_time
            current_time = tp.following_waiting_time + current_time

    def convert_to_hw_trigger(self):
        self.hw_timing_grid = []

        for tp in self.timing_grid:
            hw_events = []
            for event in tp.parallel_events:
                hw_event = qumis_event()
                target, = event.params
                channel = self.channels[target][str(event.event_type)]

                hw_event.qumis_name = channel["qumis"]

                if hw_event.qumis_name == "pulse":
                    hw_event.awg_nr = channel["awg_nr"]
                    lut_index = channel["lut"]
                    lut = self.luts[lut_index]
                    hw_event.codeword = lut[event.name]
                elif hw_event.qumis_name == "trigger":
                    hw_event.trigger_bit = channel["trigger bit"]
                    hw_event.format = channel["format"]
                    if event.event_type != EventType.MEASURE:
                        hw_event.codeword_bit = channel["codeword bit"]
                        lut_index = channel["lut"]
                        lut = self.luts[lut_index]
                        hw_event.codeword = lut[event.name]
                else:
                    pass

                hw_events.append(hw_event)
            tp.parallel_events = hw_events
            self.hw_timing_grid.append(tp)

        if self.verbose_level < 4:
            self.print_hw_timing_grid()

    def gen_full_time_grid(self):
        self.split_trigger_instruction()

    def split_trigger_instruction(self):
        new_tp_list = []
        for tp in self.hw_timing_grid:
            absolute_time = tp.absolute_time
            for hw_event in tp.parallel_events:
                if (hw_event.qumis_name == "pulse" or
                        hw_event.qumis_name == "measure"):
                    new_tp_list = self.add_new_tp_event(
                        new_tp_list, absolute_time, hw_event)
                    continue

                if hw_event.qumis_name == "trigger":
                    if hw_event.format == []:
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time, hw_event)
                        continue
                    if len(hw_event.format) == 1:
                        hw_event.duration, = hw_event.format
                        hw_event.format = []
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time, hw_event)
                        continue
                    if len(hw_event.format) == 2:
                        d1, d2 = hw_event.format
                        hw_event.format = []

                        new_absolute_time = tp.absolute_time - d1
                        new_hw_event = copy.copy(hw_event)
                        new_hw_event.duration = d1
                        new_hw_event.trigger_bit = -1
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, new_absolute_time, new_hw_event)

                        hw_event.duration = d2
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time, hw_event)

                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time + d2, None)

        self.hw_timing_grid = new_tp_list
        self.print_hw_timing_grid()

    def add_new_tp_event(self, timing_grid, absolute_time, event):
        tp_index, match = self.search_time_point(timing_grid, absolute_time)
        if match is False:
            new_tp = time_point(absolute_time=absolute_time)
            if event is not None:
                new_tp.parallel_events.append(event)
            timing_grid.insert(tp_index, new_tp)
        else:
            if event is not None:
                timing_grid[tp_index-1].parallel_events.append(event)

        return timing_grid

    def convert_to_qumis(self):
        pass

    def get_max_duration(self, timing_events):
        max_duration = 0
        for timing_event in timing_events:
            if self.is_q_op_event(timing_event) is False:
                raise ValueError("events should be gates or measurements.")

        max_duration = max(
            [self.qasm_op_dict[timing_event.name]["duration"]
             for timing_event in timing_events])
        return max_duration
