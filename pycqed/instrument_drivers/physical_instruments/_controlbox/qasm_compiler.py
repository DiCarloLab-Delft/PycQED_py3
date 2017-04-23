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


def raw_print(s):
    sys.stdout.write(s)


class time_point():

    def __init__(self):
        self.label = None
        self.absolute_time = None
        self.pre_waiting_time = None
        self.parallel_events = []


class EventType(enum.Enum):
    NONE_EVENT = enum.auto()
    WAIT = enum.auto()
    QOP = enum.auto()
    MSMT = enum.auto()
    DECLARE = enum.auto()
    MAP = enum.auto()

    # I need to think how to separate the technology dependent part and
    # technology independent part.

    # TRIGGER = auto()
    # AWG = auto()
    # MEASURE = auto()


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
        "type": EventType.MSMT
    },

    "Map": {
        "parameters": 2,
        "type": EventType.MAP
    }
}


class qasm_event():

    def __init__(self):
        self.line_number = -1
        self.event_type = EventType.NONE_EVENT
        self.name = ''
        self.params = None
        self.duration = 0
        # self.start_time_point = time_point()

    def __str__(self):
        return "({}, {})".format(self.name, self.event_type)


class prog_line():

    def __init__(self, number=-1, content=''):
        self.number = number
        self.content = content

    def __str__(self):
        return "{}: {}".format(self.number, self.content)


def lower_dict_key(origin_dict):
    return dict((k.lower(), v) for k, v in origin_dict.items())


class QASM_QuMIS_Compiler():

    def __init__(self, filename=None):
        self.filename = filename
        self.raw_lines = None
        self.prog_lines = None
        self.config_loaded = False
        self.qasm_op_dict = None
        self.declared_qubits = []
        self.qubit_map = {}

        self.config_filename = os.path.join(
            pq.__path__[0], 'instrument_drivers', 'physical_instruments',
            "_controlbox", "config.json")

    def load_config(self):
        if (self.config_loaded is True):
            return

        self.qasm_op_dict = None

        with open(self.config_filename) as data_file:
            data = json.load(data_file)

        self.user_qasm_op_dict = data["operation dictionary"]
        self.qasm_op_dict = {**default_op_dict, **self.user_qasm_op_dict}
        self.qasm_op_dict = lower_dict_key(self.qasm_op_dict)
        self.hardware_spec = data["hardware specification"]
        self.lut = data["lut"]
        # print(self.qasm_op_dict)

        self.physical_qubits = self.hardware_spec["qubit list"]
        self.cycle_time = self.hardware_spec["cycle time"]

        self.config_loaded = True

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
                prog_line(line_number, line_content.strip(' \t\n\r')))

            line_content = self.remove_comment(line_content)
            line_content = line_content.lower()
            if (len(line_content) == 0):  # skip empty line and comment
                continue
            self.prog_lines.append(prog_line(line_number, line_content))

        prog_file.close()

    def print_lines(self):
        # print("Raw Lines:")
        # for line in self.raw_lines:
        #     print(line)

        print("\nProgram Lines:")
        for line in self.prog_lines:
            print(line)

    def print_op_dict(self):
        print("QASM operation dictionary:")
        for key, value in self.qasm_op_dict.items():
            print(key, value)

    def print_raw_events(self):
        print("Raw events:")
        for i, raw_events in enumerate(self.raw_event_list):
            raw_print("{0:5d} : ".format(i))
            for re in raw_events:
                raw_print("({}, {}, {})".format(re.event_type,
                                                re.name, re.params))
            raw_print("\n")

    @classmethod
    def remove_comment(self, line):
        line = line.split('#', 1)[0]  # remove anything after '#' symbole
        line = line.strip(' \t\n\r')  # remove whitespace
        return line

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
            events = self.get_parallel_qasm_ops(line.content)
            raw_events = []
            for e in events:
                qasm_op_name = self.get_qasm_op_name(e)
                if qasm_op_name not in self.qasm_op_dict:
                    raise SyntaxError(
                        "{}, line {}: Unsuppported QASM operation {}.".format(
                            self.filename, line.number, qasm_op_name))

                qasm_op_type = self.qasm_op_dict[qasm_op_name]["type"]

                if (qasm_op_type != EventType.QOP) and (len(events) != 1):
                    raise SyntaxError(
                        "{}, line {}: QASM instruction {} should exclusively "
                        "occupy a line.".format(self.filename,
                                                line.number, qasm_op_name))

                # check parameter
                qasm_op_params = self.get_qasm_op_params(e)
                if (qasm_op_name == "qubit") and (len(qasm_op_params) < 1):
                    raise SyntaxError(
                        "{}, line {}: the QASM instruction qubit should"
                        " contain at least one parameter as the declared"
                        " qubit.".format(
                            self.filename,
                            line.number,
                            qasm_op_name))

                expected_num_of_params =  \
                    self.qasm_op_dict[qasm_op_name]["parameters"]
                if len(qasm_op_params) != expected_num_of_params:
                    raise SyntaxError(
                        "{}, line {}: unexpected number of parameters for the"
                        " QASM instruction {}. {} parameters are"
                        " expected.".format(
                            self.filename,
                            line.number,
                            qasm_op_name,
                            expected_num_of_params))

                if qasm_op_type == EventType.WAIT:
                    pre_waiting_time, = qasm_op_params
                    if is_int(pre_waiting_time) is False:
                        raise SyntaxError("{}, line {}: parameter {} is not an"
                                          " integer.".format(
                                              self.filename,
                                              line.number,
                                              pre_waiting_time))
                    pre_waiting_time = int(pre_waiting_time)
                    if pre_waiting_time < 0:
                        raise ValueError("{}, line {}: parameter {} is not"
                                          " positive.".format(
                                              self.filename,
                                              line.number,
                                              pre_waiting_time))
                    qasm_op_params = [pre_waiting_time]

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
    def get_parallel_qasm_ops(self, op_line):
        return [rawEle.strip(string.punctuation.translate(
                {ord('-'): None})) for rawEle in op_line.split("|")]

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

    def compile(self):
        self.qumis_instructions = []
        self.readfile()
        self.line_to_event()
        # self.build_dependency_graph()
        self.resolve_qubit_name()
        self.assign_timing_to_events()
        self.compensate_channel_latency()
        self.gen_full_time_grid()
        self.convert_to_qumis()
        return self.qumis_instructions

    def assign_timing_to_events(self):
        '''
        Resolve qubit name and map declared qubits into physical qubits.
        Arrange all operations into a timing gird.
        '''
        self.timing_grid = []

        tp = time_point()
        for raw_events in self.raw_event_list:
            for raw_event in raw_events:
                if (raw_event.event_type == EventType.NONE_EVENT):
                    continue
                elif (raw_event.event_type == EventType.DECLARE):
                    self.extend_dec_qubit_list(raw_event)
                elif (raw_event.event_type == EventType.MAP):
                    self.build_qubit_map(raw_event)
                elif (raw_event.event_type == EventType.WAIT):
                    tp = time_point()
                    tp.pre_waiting_time = self.get_pre_waiting_time(raw_event)

    def get_pre_waiting_time(self, raw_event):
        pre_waiting_time, = raw_event.params

        pre_waiting_time = int(pre_waiting_time)
        return int(pre_waiting_time)

    def resolve_qubit_name(self):
        self.map_qubits()

    def map_qubits(self, raw_event):
        dec_qubit, phys_qubit = raw_event.params

        if (dec_qubit not in self.declared_qubits):
            raise SyntaxError("{}, line {}: undefined qubit{} found.".format(
                self.filename, raw_event.line_number, dec_qubit))

    def extend_dec_qubit_list(self, raw_event):
        for q in raw_event.params:
            if self.declared_qubits.count(q):
                raise SyntaxError("{}, line {}: Redefinition of {}".format(
                    self.filename, raw_event.line_number, q))
            else:
                self.declared_qubits.append(q)

        if (len(self.declared_qubits) > len(self.physical_qubits)):
            raise Error("More qubits declared ({}) than available physical"
                        " qubits ({}).".format(len(self.declared_qubits),
                                               len(self.physical_qubits)))

    def compensate_channel_latency(self):
        pass

    def gen_time_grid(self):
        pass

    def convert_to_qumis(self):
        pass
