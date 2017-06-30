import re


class qasm_loader:

    """
    QASM File Loder
    """

    # Description of lines that should be replace one or multiple lines.
    _replacements = {
        "init_all": {
            "start": ".c{counter}",
            "qloop": "prepz q{qubit}"
        },
        "RO": {
            "qloop": "measure q{qubit}"
        },
        "(x|X)180 q[0-9]+": 'rx180 q{qubit}',
        "(x|X)90 q[0-9]+": 'rx90 q{qubit}',
        "(y|Y)180 q[0-9]+": 'ry180 q{qubit}',
        "(y|Y)90 q[0-9]+": 'ry90 q{qubit}',
        "mY90 q[0-9]+": 'RY q{qubit}, -1.57079',
        "mX90 q[0-9]+": 'RX q{qubit}, -1.57079',
        'I q[0-9]+': ''
    }
    _circuit_counter = 0

    def __init__(self, file_name, qubits=2):
        # print("[+] pyqx : qasm_loader : loading file '%s' ..." % file_name)
        self.qubits = qubits
        with open(file_name) as f:
            file_lines = f.readlines()
        # lines pre-processing
        self.lines = []

        for i in range(0, len(file_lines)):
            file_lines[i] = file_lines[i].strip()
            # remove empty lines
            if len(file_lines[i]) == 0:
                continue
            seperated_file_lines = file_lines[i].split("|")
            for line in seperated_file_lines:
                line = line.strip()
                self.lines.append(line)
                c = line.find("#")              # remove comments
                if c != -1:
                    self.lines[-1] = line[:c]

                # replace line if line matches _replacement
                replacement_lines = self.replaceLine(line)
                if replacement_lines:
                    self.lines[-1:] = replacement_lines

    def load_circuits(self):
        n = len(self.lines)
        # print("[+] lines : ", n)
        i = 0
        self.circuits = []

        # load circuits
        while i < n:
            l = self.lines[i]
            i = i + 1
            if len(l) == 0:
                continue
            if l[0] != '.':
                continue
            # process a new subcircuit
            l = l.replace(" ", "")
            # print("[+] new circuit : ",l);
            cn = l[1:]  # circuit name
            # cr = ""     # circuit
            cr = []     # circuit
            # p = [cn,""]
            p = [cn, []]
            l = self.lines[i]
            end = False
            ng = 0  # number of gates
            while (i < n) and (end == False):
                l = self.lines[i]
                if l == []:
                    i = i+1
                    continue
                if l == "":       # skip empty lines
                    i = i+1
                    continue
                if (l[0] == '.'):
                    end = True
                else:
                    if (len(l) > 1):
                        # cr = cr + l + "; "
                        cr.append(l)
                        ng = ng + 1
                    i = i + 1
            p[1] = cr
            # print("[-] number of gates : ",ng)
            self.circuits.append(p)

    def get_circuits(self):
        return self.circuits

    def replaceLine(self, line):
        if line == "init_all" or line[0] == '.':
            self._circuit_counter += 1
        match = False
        for k in self._replacements:
            if re.match(k, line):
                match = k
                break

        if match:
            replace = self._replacements[match]
            # find qubit
            reg = re.search("q[0-9]+", line)
            if(reg):
                qubit = line[reg.start()+1:reg.end()]
            if isinstance(replace, dict):
                line = []
                if "start" in replace:
                    line += [replace["start"].format(
                        counter=self._circuit_counter, line=line)]
                if "qloop" in replace:
                    for i in range(self.qubits):
                        line += [replace['qloop'].format(
                            counter=self._circuit_counter,
                            qubit=i, line=line)]
                if "end" in replace:
                    line += [replace["end"].format(
                        counter=self._circuit_counter, line=line)]

            elif isinstance(replace, str):
                line = [replace.format(
                    counter=self._circuit_counter, qubit=qubit)]
            return line
        else:
            return False

# loading qasm test
'''
ql = qasm_loader('/Users/nader/Develop/demo/rb_0.qasm')
ql.load_circuits()
circuits = ql.get_circuits()

for i in range(0,len(circuits)):
    print(circuits[i][0])
    print(circuits[i][1])
    print("")
'''
