'''
This file implements a compiler that translates QASM program into QuMIS
instruction. More precisely, it defines an Intermediate Representation (IR)
and the process to translate the IR into QuMIS instructions.

It should contain the following information:
1. Timing information (done)
    - Starting point
    - Duration
    - (optional) Ending point
2. Events (done)
3. Loop (not done yet)
4. Register (not done yet)
5. Memory (not done yet)

The QASM file is case insensitive.

Read the class doc string for the usage.
'''

import logging
import json
import os
import bisect
import copy
from pycqed.utilities.general import int_to_bin
from pycqed.measurement.waveform_control_CC.qasm_compiler_helpers import (
    is_number, is_int, is_positive_number, is_natural, is_integer_array,
    bitfield, min_non_zero, raw_print, config_is_valid,
    EventType, time_point, qasm_event, qumis_event, prog_line, lower_dict_key)


MAX_TRIG_BITS = 7
DEFAULT_MEASURE_TIME = 300  # ns

user_op_type = {
    "rf": EventType.RF,
    "flux": EventType.FLUX,
    "measurement": EventType.MEASUREMENT
}

default_op_dict = {
    "Idx": {
        "parameters": 1,  # Parameter is wait in ns
        "type": EventType.WAIT
    },
    "qubit": {
        "parameters": -1,  # Accept indefinite number of parameters
        "type": EventType.DECLARE
    },

    "init_all": {
        "parameters": 0,
        "type": EventType.WAIT
    },

    "Map": {
        "parameters": 2,
        "type": EventType.MAP
    }
}


class QASM_QuMIS_Compiler():

    def __repr__(self):
        base_str = ('QASM_QuMIS_Compiler(config_filename={}, '
                    'verbosity_level={})')
        rep = base_str.format(self.config_filename, self.verbosity_level)
        return rep

    def __str__(self):
        base_str = ('QASM_QuMIS_Compiler: '
                    '\n\tbase_fp = {}'
                    '\n\tconfig_fn = {} '
                    '\n\tqasm_fn = {}'
                    '\n\tqumis_fn = {}'
                    '\n\tcompilation_completed = {}')
        s = [self.config_filename, self.filename, self.qumis_fn]
        common_s = os.path.commonprefix(s)
        path_strings = [sub.replace(common_s, '') for sub in s]

        rep = base_str.format(common_s,
                              *path_strings, self.compilation_completed)
        return rep

    def __init__(self, config_filename: str='', verbosity_level: int=1):
        '''
        @param: config_filename, file specifies the user-defined operation
        dictionary, hardware specification and the LUTs.
        @param: verbosity_level, message level to be printed. integer, range
        from 0 to 6. The larger verbosity_level is, the more information is
        printed. 0 prints nothing. 6 prints everything.

        Usage:
        1. Instantiate the class, e.g., qqc, with a configuration file path.
        2. qqc.compile(qasm_file, qumis_file_path) will compile the qasm_file
        and write the resultant qumis program into qumis file.
        3. qqc.dump_config(config_file_path) can dump current configuration
        into the file config_file_path in JSON format.
        '''
        self.raw_lines = None
        self.prog_lines = None
        self.qasm_op_dict = None
        self.declared_qubits = []
        self.qubit_map = {}
        self.verbosity_level = verbosity_level
        self.channel_latency_compensated = False
        self.qubit_map_from_config = False
        self.infinit_loop_qumis = True
        self.compilation_completed = False
        self.qumis_fn = ''
        self.filename = ''
        self.config_filename = config_filename

    def compile(self, filename: str, qumis_fn: str=None,
                config_fn: str ='', config: dict=None)-> bool:
        """
        Compiles the
        """
        self.filename = filename
        self.qumis_fn = qumis_fn
        self.qumis_instructions = []   # final result that should be uploaded
        self.hw_timing_grid = []       # operations on hardware
        self.timing_grid = []          # quantum operations
        self.load_config(config_filename=config_fn, config=config)
        self.read_file()               # fills up self.prog_lines
        self.line_to_event()           # fills up self.raw_event_list
        self.build_dependency_graph()  # empty function for now
        self.resolve_qubit_name()      # extract map from qasm and map to cfg
        self.assign_timing_to_events()
        self.resolve_channel_latency()
        self.convert_to_hw_trigger()
        self.gen_full_time_grid()
        self.convert_to_qumis()
        if self.verbosity_level >= 1:
            print("QuMIS generated successfully and written into {}".format(
                self.qumis_fn))
        self.compilation_completed = True
        return True

    def build_dependency_graph(self):
        pass

    def load_config(self, config_filename: str='', config: dict =None):
        self.config = None

        if config_filename is not '':
            self.config_filename = config_filename

        if self.config_filename != '':
            with open(self.config_filename) as data_file:
                self.config = json.load(data_file)

        if config is not None:
            self.config = copy.deepcopy(config)

        if self.config is None:
            raise ValueError('No config specified')

        self.qasm_op_dict = None
        if not config_is_valid(self.config):
            # this error should be unreachable but is here for readability
            raise ValueError('Configuration is not valid')

        self.hardware_spec = self.config["hardware specification"]
        if "qubit_map" in self.config.keys():
            self.qubit_map = self.config["qubit_map"]
            self.qubit_map_from_config = True

        self.luts = self.config["luts"]

        self.physical_qubits = self.hardware_spec["qubit list"]

        self.cycle_time = self.hardware_spec["cycle time"]
        self.init_time = self.hardware_spec["init time"]

        self.cycle_time = float(self.cycle_time)
        self.init_time = int(self.init_time / self.cycle_time)
        self.measureMENT_time = int(DEFAULT_MEASURE_TIME / self.cycle_time)

        self.qubit_cfgs = self.hardware_spec["qubit_cfgs"]

        for i in range(len(self.qubit_cfgs)):
            for key in self.qubit_cfgs[i]:
                # Dangerously overwriting latency in ns with latency in clocks
                self.qubit_cfgs[i][key]["latency"] = int(
                    self.qubit_cfgs[i][key]["latency"] / self.cycle_time)
                if "format" in self.qubit_cfgs[i][key]:
                    trig_format = []
                    for trig_dur in self.qubit_cfgs[i][key]["format"]:
                        trig_format.append(int(trig_dur / self.cycle_time))
                    self.qubit_cfgs[i][key]["format"] = trig_format

        self.user_qasm_op_dict = self.config["operation dictionary"]
        for key in self.user_qasm_op_dict:
            op_spec = self.user_qasm_op_dict[key]
            op_type_str = op_spec["type"]
            if op_type_str not in user_op_type.keys():
                raise SyntaxError(
                    "unsupported operation type ({}) for operation {}."
                    ", supported types: {}".format(op_spec["type"],
                                                   key, user_op_type))
                # se.filename = self.config_filename
                # raise se

            op_type_enum = user_op_type[op_type_str]
            op_spec["type"] = op_type_enum
            op_spec["duration"] = int(op_spec["duration"] / self.cycle_time)
            if op_type_enum == EventType.MEASUREMENT:
                self.measureMENT_time = int(op_spec["duration"] /
                                            self.cycle_time)
            self.user_qasm_op_dict[key] = op_spec

        for key in default_op_dict:
            if key == "init_all":
                default_op_dict[key]["duration"] = self.init_time
            if key == "measure":
                default_op_dict[key]["duration"] = self.measureMENT_time

        self.qasm_op_dict = {**default_op_dict, **self.user_qasm_op_dict}
        self.qasm_op_dict = lower_dict_key(self.qasm_op_dict)
        if self.verbosity_level > 3:
            self.print_op_dict()

    def dump_config(self, config_fn):
        op_dict = {}
        for op in self.user_qasm_op_dict:
            op_dict[op] = self.user_qasm_op_dict[op]
            if "duration" in self.user_qasm_op_dict[op]:
                op_dict[op]["duration"] =\
                    op_dict[op]["duration"] * self.cycle_time
            if "type" in self.user_qasm_op_dict[op]:
                if op_dict[op]["type"] == EventType.RF:
                    op_dict[op]["type"] = "rf"
                elif op_dict[op]["type"] == EventType.FLUX:
                    op_dict[op]["type"] = "flux"
                elif op_dict[op]["type"] == EventType.MEASUREMENT:
                    op_dict[op]["type"] = "measure"
                else:
                    raise ValueError("unexpected event type: {}".format(
                        op_dict[op]["type"]))

        hardware_spec = {}
        hardware_spec["qubit list"] = self.physical_qubits
        hardware_spec["cycle time"] = self.cycle_time
        hardware_spec["init time"] = self.init_time * self.cycle_time
        channels = []

        for c in self.qubit_cfgs:
            channel = {}
            for key in c:
                channel[key] = c[key]

                if "latency" in c[key]:
                    channel[key]["latency"] =\
                        c[key]["latency"] * self.cycle_time

                if "format" in c[key]:
                    trig_format = c[key]["format"]
                    for i in range(len(trig_format)):
                        trig_format[i] = trig_format[i] * self.cycle_time
                    channel[key]["format"] = trig_format

            channels.append(channel)

        hardware_spec["qubit_cfgs"] = channels

        data = {}
        data["operation dictionary"] = op_dict
        data["hardware specification"] = hardware_spec
        data["luts"] = self.luts
        with open(config_fn, 'w') as outfile:
            json.dump(self.config, outfile, indent=2)

    def read_file(self):
        '''
        Read all lines in the file.
        '''
        try:
            prog_file = open(self.filename, 'r', encoding="utf-8")
            logging.info("open file", str(self.filename), "successfully.")
        except:
            raise OSError('\tError: Failed to open file ' +
                          self.filename + ".")

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
        if self.verbosity_level >= 2:
            self.print_raw_lines()
        if self.verbosity_level >= 3:
            self.print_program_lines()

    def print_raw_lines(self):
        print("Raw Lines:")
        for line in self.raw_lines:
            print(line)
        print("")

    def print_program_lines(self):
        print("Program Lines:")
        for line in self.prog_lines:
            print(line)
        print("")

    def print_op_dict(self):
        print("QASM operation dictionary:")
        for key, value in self.qasm_op_dict.items():
            raw_print("{0: >10} : ".format(key))
            print(value)
        print("\n")

    def print_raw_events(self):
        print("Raw events:")
        for i, raw_events in enumerate(self.raw_event_list):
            raw_print("{0:5d} : ".format(i))
            for re in raw_events:
                raw_print("({}, {}, {})".format(re.event_type,
                                                re.name, re.params))
            raw_print("\n")
        print("")

    def print_timing_events(self):
        print("Timing events:")
        for i, timing_events in enumerate(self.timing_event_list):
            raw_print("{0:5d} : ".format(i))
            for re in timing_events:
                raw_print("({}, {}, {})".format(re.event_type,
                                                re.name, re.params))
            raw_print("\n")
        print("")

    def print_timing_grid(self):

        print("Timing grid:")
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
            self._print_event_list(tp.parallel_events)
            raw_print('\n')
            i = i + 1
        print("")

    def print_hw_timing_grid(self):
        print("Hardware trigger timing grid:")
        if len(self.hw_timing_grid) == 0:
            print("Empty trigger timing grid.")
            return

        print("{0: >6s}  {1: <10s}"
              "    Events".format("Idx", "absolute"))
        i = 0
        for tp in self.hw_timing_grid:
            raw_print("{0:5d}:".format(i))
            raw_print("   {0: <8d}    ".format(tp.absolute_time))
            self._print_event_list(tp.parallel_events)
            raw_print('\n')
            i = i + 1

        print("")

    def _print_event_list(self, event_list):
        """
        private method used for printing of the timing grid and hardware
        timing grid
        """
        raw_print(" ")
        if event_list == []:
            raw_print('-')
            return

        if hasattr(event_list[0], 'event_type'):  # qasm event list
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
                if e.codeword == -2:
                    if e.awg_nr != -1:
                        raw_print(" ")
                        raw_print("awg")
                        raw_print(str(e.awg_nr))
                        raw_print("(%s)" % str(e.codeword))
                    else:
                        raw_print(" ")
                        raw_print("bits:")
                        raw_print("[{}]".format(str(e.trigger_bit)))
                        raw_print(str(e.codeword_bit))

                if e.qumis_name.lower() == "measure":
                    pass
                elif e.qumis_name.lower() == "pulse":
                    raw_print(" ")
                    raw_print("awg")
                    raw_print(str(e.awg_nr))
                    raw_print("(%s)" % str(e.codeword))
                elif e.qumis_name.lower() == "trigger":
                    raw_print(" ")
                    raw_print("bits:")
                    if e.set_bits != []:
                        raw_print(str(e.set_bits))
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
                        raw_print(" cw: %s" % str(e.codeword))
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
        self.raw_event_list = []
        for line in self.prog_lines:
            events = self.get_parallel_qasm_ops(line.content)
            raw_events = []
            for e in events:
                qasm_op_name = self.get_qasm_op_name(e)
                if qasm_op_name not in self.qasm_op_dict:
                    se = SyntaxError("unsuppported QASM operation {}.".format(
                        qasm_op_name))
                    se.filename = self.filename
                    se.lineno = line.number
                    raise se

                qasm_op_type = self.qasm_op_dict[qasm_op_name]["type"]

                if self.is_single_line_op(qasm_op_type) and (len(events) != 1):
                    se = SyntaxError("QASM instruction {} should e "
                                     "occupy a line.".format(qasm_op_name))
                    se.filename = self.filename
                    se.lineno = line.number
                    raise se

                # check parameter
                qasm_op_params = self.get_qasm_op_params(e)
                if (qasm_op_name == "qubit") and (len(qasm_op_params) < 1):
                    se = SyntaxError("the QASM instruction qubit should "
                                     "contain at least one parameter as "
                                     "the declared qubit.")
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
                    waiting_time_ns, = qasm_op_params

                    waiting_time = int(waiting_time_ns)//self.cycle_time
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
        if ((op_type == EventType.MEASUREMENT) or
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
        Arrange all operations into a timing grid.

        The timing grid is defined in terms of clocks.
        '''
        self.timing_grid = []

        for timing_events in self.timing_event_list:
            # each line starts a new time point
            # two thing to do for this time point:
            # 1. determine what happens at this moment
            # 2. determine the following waiting time
            op_name = timing_events[0].name
            tp = time_point(label=op_name)

            # determine the following waiting time
            if self.is_wait_line(timing_events):
                timing_event = timing_events[0]
                op_name = timing_event.name
                if self.qasm_op_dict[op_name]["parameters"] == 1:
                    following_waiting_time = timing_event.params[0]
                elif op_name == "init_all":
                    following_waiting_time = self.init_time
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

        if self.verbosity_level > 4:
            print("End of assigning timing:")
            self.print_timing_grid()

    def resolve_qubit_name(self):
        self.build_qubit_map()
        self.map_qubits()
        if self.verbosity_level > 4:
            print("End of resolving qubit name:")
            self.print_timing_events()

    def build_qubit_map(self):
        self.declared_qubits = []
        i = 0
        while i < len(self.raw_event_list):
            raw_events = self.raw_event_list[i]
            for raw_event in raw_events:
                if (raw_event.event_type == EventType.DECLARE):
                    self.extend_dec_qubit_list(raw_event)
                    self.raw_event_list.pop(i)
                elif (raw_event.event_type == EventType.MAP):
                    if not self.qubit_map_from_config:
                        self.add_qubit_map(raw_event)
                    self.raw_event_list.pop(i)
                else:
                    i = i + 1
        if len(self.qubit_map) == 0:
            se = SyntaxError("Qubit map not found in the QASM file.")
            se.filename = self.filename
            raise se

        if (self.verbosity_level > 4):
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
        if self.verbosity_level >= 3:
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

        if self.verbosity_level >= 3:
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

        return new_event_list

    def load_channel_latency(self):
        for ti, tp in enumerate(self.timing_grid):
            for idx, event in enumerate(tp.parallel_events):
                target_qubit = event.params[0]
                channel = self.qubit_cfgs[target_qubit]
                event.channel_latency = \
                    channel[str(event.event_type)]["latency"]

                tp.parallel_events[idx] = event
            self.timing_grid[ti] = tp

        if self.verbosity_level > 3:
            print("get channel latency:")
            self.print_timing_grid()

    def compensate_channel_latency(self):
        min_latency = self.get_min_channel_latency()
        if self.verbosity_level > 4:
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

        if self.verbosity_level > 4:
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
                    for channel in self.qubit_cfgs])

    def get_absolute_timing(self):
        """
        Updates the self.timing_grid to contain a notion of their absoute
        time.
        Adds an extra time point add the end of the timing grid to denote
        the end of the program.
        """
        current_time = 0
        for tp in self.timing_grid:
            tp.absolute_time = current_time
            current_time = tp.following_waiting_time + current_time

        # Added the timing point corresponding to the end of the entire
        #   program into the timing list.
        tp = time_point()
        tp.absolute_time = current_time
        tp.parallel_events = []
        self.timing_grid.append(tp)


    def convert_to_hw_trigger(self):
        """
        This function takes events from the timing_grid and converts them to
        hw_triggers using the configuration
        """
        self.hw_timing_grid = []

        for tp in self.timing_grid:
            hw_events = []
            for event in tp.parallel_events:
                hw_event = qumis_event()
                targ_q_idx, = event.params
                channel_cfg = self.qubit_cfgs[
                    targ_q_idx][str(event.event_type)]

                tmp_qumis_name = channel_cfg["qumis"]

                if tmp_qumis_name == "pulse":
                    hw_event.awg_nr = channel_cfg["awg_nr"]

                    lut_index = channel_cfg["lut"]
                    lut = self.luts[lut_index]
                    hw_event.codeword = lut[event.name]

                    if hw_event.codeword == -2:
                        hw_event.qumis_name = event.name
                    else:
                        hw_event.qumis_name = tmp_qumis_name

                elif tmp_qumis_name == "trigger":
                    hw_event.trigger_bit = channel_cfg["trigger bit"]
                    hw_event.format = channel_cfg["format"]
                    hw_event.qumis_name = tmp_qumis_name

                    if 'codeword bit' in channel_cfg.keys():
                        hw_event.codeword_bit = channel_cfg["codeword bit"]
                        lut_index = channel_cfg["lut"]
                        lut = self.luts[lut_index]
                        hw_event.codeword = lut[event.name]

                        if hw_event.codeword == -2:
                            hw_event.qumis_name = event.name
                else:
                    pass
                hw_events.append(hw_event)

            tp.parallel_events = hw_events
            self.hw_timing_grid.append(tp)

        if self.verbosity_level > 4:
            print("After converting event timing grid to hardware timing:")
            self.print_hw_timing_grid()

    def gen_full_time_grid(self):
        self.split_trigger_codeword()
        self.apply_codeword()
        self.vertical_divide_trigger()
        self.compensate_minus_time()

    def apply_codeword(self):
        """
        This function converts codewords (int) to the correct combination
        of trigger bits for the trigger QuMIS instruction.

        NOTE: trigger bits starts counting from 1, instead of 0.
        """
        new_tp_list = []
        for tp in self.hw_timing_grid:
            absolute_time = tp.absolute_time
            for hw_event in tp.parallel_events:

                if hw_event.qumis_name == 'trigger':
                    hw_event.set_bits = []
                    if hw_event.trigger_bit != -1:
                        assert(hw_event.trigger_bit != 0)
                        hw_event.set_bits.append(hw_event.trigger_bit)
                        hw_event.trigger_bit = -1

                    if (hw_event.codeword != -1):
                        bitwidth = len(hw_event.codeword_bit)
                        codeword_array = bitfield(hw_event.codeword,
                                                  bitwidth, little_endian=False)

                        for i in range(bitwidth):
                            if codeword_array[i] == 1:
                                hw_event.set_bits.append(
                                    hw_event.codeword_bit[i])
                        hw_event.codeword = -1
                        hw_event.codeword_bit = []

                    if hw_event.set_bits != []:
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time, hw_event)

                else:  # for pulse, measure, and other dummy instructions.
                    new_tp_list = self.add_new_tp_event(
                        new_tp_list, absolute_time, hw_event)

        # at each processing step, add the last timing point back.
        # this is required as it is not added in the lines above as it is empty.
        new_tp_list.append(self.hw_timing_grid[-1])

        self.hw_timing_grid = new_tp_list
        if self.verbosity_level > 4:
            print("After applying codewords")
            self.print_hw_timing_grid()

    def compensate_minus_time(self):
        if len(self.hw_timing_grid) == 0:
            return
        if self.hw_timing_grid[0].absolute_time < 0:
            added_time = abs(self.hw_timing_grid[0].absolute_time)
            for i in range(len(self.hw_timing_grid)):
                self.hw_timing_grid[i].absolute_time += added_time

        if self.verbosity_level > 4:
            print("After compensating for negative time")
            self.print_hw_timing_grid()

    def vertical_divide_trigger(self):
        '''
        Multiple parallel trigger instructions would lead to the following
        signal shape:
        Timeslot:    1   2     3    4   5     6
        1.          ---|---|------|   |-----|----
        2.             |   |------|---|-----|----
        3.             |---|------|---|     |
        4.             |---|------|---|-----|
        In this case, we would require 6 trigger instructions to describe
        the digital signal at each time slot.
        This function resolves original parallel trigger instructions and
        generates equivalent sequential trigger instructions.
        '''
        trigger_bit_duration = [0]*8
        new_tp_list = []
        for ti, tp in enumerate(self.hw_timing_grid):
            absolute_time = tp.absolute_time
            for hw_event in tp.parallel_events:

                if hw_event.qumis_name == 'trigger':
                    for tb in hw_event.set_bits:
                        # If multiple trigger instructions occur at the same
                        # point in time at the same bit, use the longest one.
                        trigger_bit_duration[tb] = max(trigger_bit_duration[tb],
                                                        hw_event.duration)

                else:  # for pulse, measure and other dummy instructions.
                    new_tp_list = self.add_new_tp_event(
                        new_tp_list, absolute_time, hw_event)

            if max(trigger_bit_duration) == 0:
                continue

            min_duration = min_non_zero(trigger_bit_duration)
            next_trig_ending_time = min_duration + absolute_time
            next_trig_starting_time = self.get_next_trigger_instr_time(ti)

            # find the next stop time, insert current trigger instruction.
            # stage 1: next_trig_starting_time < next_trig_ending_time, so the
            #          next stop time is next_trig_starting_time.
            # stage 2： the same as stage 1.
            # stage 3: next_trig_ending_time < next_trig_starting_time, so the
            #           next stop time is next_trig_ending_time
            # stage 4, 5 6: next_trig_starting_time = -1, the next stop time is
            #            the next_trig_ending_time.

            current_time = absolute_time
            while True:
                reach_next_starting_time = False
                if (next_trig_starting_time == -1):
                    duration = min_duration
                elif (next_trig_starting_time <= next_trig_ending_time):
                    duration = next_trig_starting_time - absolute_time
                    reach_next_starting_time = True
                else:
                    duration = min_duration

                new_hw_event = qumis_event()
                new_hw_event.qumis_name = "trigger"
                new_hw_event.duration = duration
                new_hw_event.set_bits = [i for i, d in enumerate(
                    trigger_bit_duration) if d > 0]
                for i in new_hw_event.set_bits:
                    trigger_bit_duration[i] -= duration

                new_tp_list = self.add_new_tp_event(
                    new_tp_list, current_time, new_hw_event)

                current_time = absolute_time + duration

                min_duration = min_non_zero(trigger_bit_duration)

                if min_duration == 0 or reach_next_starting_time is True:
                    break

        # at each processing step, add the last timing point back.
        # this is required as it is not added in the lines above as it is empty.
        new_tp_list.append(self.hw_timing_grid[-1])
        self.hw_timing_grid = new_tp_list
        # if self.verbosity_level > 4:
        if self.verbosity_level > 1:
            print("after vertical_divide_trigger:")
            self.print_hw_timing_grid()

    def get_next_trigger_instr_time(self, l):
        i = l + 1
        while i < len(self.hw_timing_grid):
            # print("i:", i)
            tp = self.hw_timing_grid[i]
            for hw_event in tp.parallel_events:
                # hw_event.print_self()
                if hw_event.qumis_name == "trigger":
                    return tp.absolute_time
            i += 1
        return -1

    def split_trigger_codeword(self):
        '''
        Each trigger instruction contains two stages:
        1. preparing codeword;
        2. send trigger
        This function remove the original trigger instruction and for each
        stage generates a new trigger instruction.
        '''
        new_tp_list = []
        for tp in self.hw_timing_grid:
            absolute_time = tp.absolute_time
            for hw_event in tp.parallel_events:
                if hw_event.qumis_name == "trigger":
                    if hw_event.format == []:
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time, hw_event)
                        continue
                    if len(hw_event.format) == 1:
                        # If len format ==1 then it is a simple trigger
                        hw_event.duration, = hw_event.format
                        hw_event.format = []
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time, hw_event)
                        continue
                    if len(hw_event.format) == 2:
                        # This splits a trigger that needs to perpare a CW
                        # before raising a trigger bit.
                        d1, d2 = hw_event.format
                        hw_event.format = []

                        # This is the starting timepoint for preparing the CW
                        new_absolute_time = tp.absolute_time - d1
                        new_hw_event = copy.copy(hw_event)
                        new_hw_event.duration = d1
                        new_hw_event.trigger_bit = -1
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, new_absolute_time, event=new_hw_event)
                        # This is the starting timepoint for trigger + CW
                        hw_event.duration = d2
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time, event=hw_event)

                        # This is an empty timepoint for the end of this instr
                        new_tp_list = self.add_new_tp_event(
                            new_tp_list, absolute_time + d2, event=None)

                else:  # for pulse, measure and other dummy instructions.
                    new_tp_list = self.add_new_tp_event(
                        new_tp_list, absolute_time, hw_event)


        # at each processing step, add the last timing point back.
        # this is required as it is not added in the lines above as it is empty.
        new_tp_list.append(self.hw_timing_grid[-1])
        self.hw_timing_grid = new_tp_list
        if self.verbosity_level > 4:
            print("after split_trigger_codeword:")
            self.print_hw_timing_grid()

    def add_new_tp_event(self, timing_grid, absolute_time, event):
        tp_index, match = self.search_time_point(timing_grid, absolute_time)
        if match is False:
            new_tp = time_point(absolute_time=absolute_time)
            if event is not None:
                new_tp.parallel_events.append(event)
                if 0 in event.set_bits:
                    raise ValueError('Bits start counting at 1 instead of 0')
            timing_grid.insert(tp_index, new_tp)
        else:
            if event is not None:
                # because there is a matched timepoint, no new tp is inserted
                # the event is added to the existing timepoint
                timing_grid[tp_index-1].parallel_events.append(event)
                if 0 in event.set_bits:
                    raise ValueError('Bits start counting at 1 instead of 0')
        return timing_grid

    def get_max_duration(self, timing_events):
        max_duration = 0
        for timing_event in timing_events:
            if self.is_q_op_event(timing_event) is False:
                raise ValueError("Event {} should ".format(timing_event) +
                                 "be gates or measurements.")

        max_duration = max(
            [self.qasm_op_dict[timing_event.name]["duration"]
             for timing_event in timing_events])
        return max_duration

    def convert_to_qumis(self):
        '''
        The final step of compilation. Generate the QuMIS instructions.
        The instruction format:
           pulse xxxx, xxxx, xxxx
           trigger mask, duration
           measure
        '''
        self.qumis_instructions = []
        if self.infinit_loop_qumis:
            reps = ('mov r14, 0 \t# r14 stores number ' +
                    'of repetitions, 0 is infinite')
            self.qumis_instructions.append(reps)
            self.qumis_instructions.append('Exp_Start: ')
        previous_time = 0
        for tp in self.hw_timing_grid:
            pre_waiting_time = tp.absolute_time - previous_time
            if pre_waiting_time == 0:
                # Commenting out this makes it compile when it should not
                pass
                # if tp.absolute_time != 0:
                #     raise ValueError("Strange thing happened. Two time points "
                #                      "have the same absolute time")
            else:
                self.qumis_instructions.append("wait {:d}".format(
                    pre_waiting_time))

            previous_time = tp.absolute_time

            pulse_list = [None, None, None]
            # Loop overall hw_events that occur at the same time to turn
            # them into instructions
            for hw_event in tp.parallel_events:
                if hw_event.qumis_name == "pulse":
                    # Add the pulse to the pulse list for the
                    # respective channel
                    pulse_list[hw_event.awg_nr] = hw_event.codeword

                if hw_event.qumis_name == "trigger":
                    mask = 0
                    for bit in hw_event.set_bits:
                        mask += (1 << (MAX_TRIG_BITS-bit))
                    trigger_instruction = "trigger {0:07b}, {1:d}".format(
                        mask, hw_event.duration)
                    self.qumis_instructions.append(trigger_instruction)

                if hw_event.qumis_name == "measure":
                    self.qumis_instructions.append("measure")

            # If any pulses where added, create the right instruction
            if not all(cw is None for cw in pulse_list):
                pulse_cws = ['']*3
                for i, cw in enumerate(pulse_list):
                    if cw is None:  # don't trigger if no codeword specified
                        pulse_cws[i] = '0000'
                    else:  # Create the appropriate codeword
                        pulse_cws[i] = '1'+int_to_bin(cw, w=3, lsb_last=True)

                pulse_instruction = \
                    "pulse {}, {}, {}".format(pulse_cws[0],
                                              pulse_cws[1], pulse_cws[2])
                self.qumis_instructions.append(pulse_instruction)

        if self.qumis_instructions[0][:4] != "wait":
            if self.verbosity_level >= 4:
                print("Instruction 'Wait 1' added at the beginning.")
            self.qumis_instructions.insert(0, "wait 1")
        if self.infinit_loop_qumis:
            jump_to_start = ("beq r14, r14, "
                             "Exp_Start \t# Jump to start ad nauseam")
            self.qumis_instructions.append(jump_to_start)

        qumis_file = open(self.qumis_fn, "w")
        for qi in self.qumis_instructions:
            qumis_file.write("{}\n".format(qi))
        qumis_file.close()
        if self.verbosity_level >= 2:
            self.print_qumis()

    def print_qumis(self):
        print("qumis:")
        for qi in self.qumis_instructions:
            print('\t'+qi)
        print("")
