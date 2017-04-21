'''
This file implements a compiler that translates QASM program into QuMIS instruction.
More precisely, it defines an Intermediate Representation (IR) and the process
to translate the IR into QuMIS instructions.

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
    DECLARE = enum.auto()
    
    # I need to think how to separate the technology dependent part and 
    # technology independent part.

    # TRIGGER = auto()
    # AWG = auto()
    # MEASURE = auto()


class qasm_event():

    def __init__(self):
        self.line_number = -1
        self.event_type = EventType.NONE_EVENT
        self.name = ''
        self.params = None
        self.duration = 0
        # self.start_time_point = time_point()


class prog_line():

    def __init__(self, number=-1, content=''):
        self.number = number
        self.content = content

    def __str__(self):
        return "{}: {}".format(self.number, self.content)


class QASM_QuMIS_Compiler():

    def __init__(self, filename=None):
        self.filename = filename
        self.raw_lines = None
        self.prog_lines = None

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
            self.raw_lines.append(prog_line(line_number, line_content.strip(' \t\n\r')))

            line_content = self.remove_comment(line_content)
            if (len(line_content) == 0):  # skip empty line and comment
                continue
            self.prog_lines.append(prog_line(line_number, line_content))

        prog_file.close()

    def load_qasm_op_dict(self, filename):
        self.qasm_op_dict = None
        pass

    def print_lines(self):
        print("Raw Lines:")
        for line in self.raw_lines:
            print(line)

        print("\nProgram Lines:")
        for line in self.prog_lines:
            print(line)

    @classmethod
    def remove_comment(self, line):
        line = line.split('#', 1)[0]  # remove anything after '#' symbole
        line = line.strip(' \t\n\r')  # remove whitespace
        return line

    def line_to_event(self):
        '''
        Convert each line in the QASM file into an event.
        An event could be:
         - A wait or idling instruction
         - A single qubit gate
         - A two qubit gate

        At this moment, the timing information of events is not generated yet.

        raw_event_list contains all events before resolving timing information.
        '''
        self.load_qasm_op_dict()
        self.raw_event_list = []
        for line in self.prog_lines:
            events = self.get_parallel_qasm_ops(line.content)
            raw_events = []
            for e in events:
                raw_event = qasm_event()
                raw_event.line_number = line.number
                                
                qasm_op_name = self.get_qasm_op_name(e)
                qasm_op_params = self.get_qasm_op_params(e)
                
                raw_event = qasm_op_name
                raw_event.params = qasm_op_params

                if qasm_op_name not in self.qasm_op_dict:
                    raise SyntaxError(
                        "Line {}: Unsuppported operation found:".format(
                        line.number, qasm_op_name))

                if is_wait_instr(qasm_op_name):
                    if len(events) != 1:
                        raise SyntaxError(
                        "Line {}: the timing instruction is mixed with other "
                        "instructions:".format(line.number, qasm_op_name))
                    raw_event.event_type = EventType.WAIT
                else:
                    raw_event.event_type = EventType.QOP
                
                raw_event.duration = 0
                raw_events.append(raw_event)

    def is_wait_instr(self, qasm_op_name):
        if (qasm_op_name.lower() == "I"):
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
        self.assign_timing_to_events()
        self.compensate_channel_latency()
        self.gen_full_time_grid()
        self.convert_to_qumis()
        return self.qumis_instructions

    def assign_timing_to_events(self):
        pass
        
    def compensate_channel_latency(self):
        pass

    def gen_time_grid(self):
        pass

    def convert_to_qumis(self):
        pass
