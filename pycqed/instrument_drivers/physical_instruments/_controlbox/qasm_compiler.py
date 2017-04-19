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
label | absolute time | waiting time | event | duration
'''
from enum import Enum
import logging

class time_point():
    def __init__(self):
        self.label = None
        self.absolute_time = None
        self.pre_waiting_time = None
        self.parallel_events = []


class EventType(Enum):
    NONE_EVENT = auto()
    WAIT = auto()
    TRIGGER = auto()
    AWG = auto()
    MEASURE = auto()


class event():
    def __init__(self):
        self.line_number = -1
        self.event_type = EventType.NONE_EVENT
        self.params = None
        self.duration = 0
        self.start_time_point = time_point()

class prog_line():
    def __init__(self):
        self.number = -1
        self.content = ''

class qasm_qumis_compiler():
    def __init__(self, filename=None):
        self.filename = filename
        self.raw_lines = None
        self.prog_lines = None

    def readfile(self):
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
        for line_number, line_content in prog_file:
            self.raw_lines.append(prog_line(line_number, line_content))

            line_content = self.remove_comment(line_content)
            if (len(line) == 0):  # skip empty line and comment
                continue
            self.prog_lines.append(prog_line(line_number, line_content))

        prog_file.close()

    @classmethod
    def remove_comment(self, line):
        line = line.split('#', 1)[0]  # remove anything after '#' symbole
        line = line.strip(' \t\n\r')  # remove whitespace
        return line

    def line_to_event(self):
        pass

