"""
  QX client is a python interface to connect to the QX simulator server and execute remote operations on it.
  Operations includes:
    - Creating qubits
    - Creating Circuits
    - Executing Circuits
    - Simulating Noise
    - Getting Measurement Outcomes

  The QX server can run on the local host computer or on any server on the network. Servers with large memory allow
  the simulation of more qubits and to speedup the simulation.
"""

import socket

class qx_client:

    """
    qx client
    """
    encoding = "utf-8"
    ack = "OK"
    timeout = 10.0
    buffer_size = 8192
    IllegalOperationException = Exception("Illegal Operation !")

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__qubits = 0
        self.__circuits = []
        # default main circuit is always there
        self.__circuits.append("default")
        self.__debug = 1

    def connect(self, host="localhost", port=5555):
        print("[+] connecting to QX server...")
        self.sock.connect((host, port))
        print("[+] connection to QX server established.")

    def __trace(self, operation):
        """ debug utility """
        if self.__debug != 0:
            print("[~] ", operation)

    # private raw tcp send method
    def __send(self, cmd):
        self.__trace("qx_client::__send : cmd : %s" % cmd)
        self.sock.sendall(cmd.encode(self.encoding))

    def __receive(self):
        return self.sock.recv(self.buffer_size).decode(self.encoding)

    def __receive_ack(self):
        self.__trace("qx_client::__receive_ack : receiving acknowlegement...")
        reply = ""
        try:
            self.sock.settimeout(self.timeout)
            while reply.find(self.ack) == -1:
                reply += self.__receive()
        except socket.timeout:
            print("[x] error : timeout while waiting for server reply")
            return ""
        self.__trace("qx_client::__receive_ack : received :\n%s" % reply)
        return reply

    def send_cmd(self, cmd):
        """
          send command and receive acknowlegement : command can be single or multiple gates
        """
        self.__send(cmd)
        return self.__receive_ack()

    def create_qubits(self, n):
        """
          create a quantum register of n qubits
        """
        # type safety check
        assert isinstance(n, int)
        # 2 or more qubits are quired in this version (measurement outcome has to
        # be parsed correctly for single qubit)
        # assert (n > 1)

        # qubit number check
        if self.__qubits != 0:
            # print("[!] warning : qx_client::create_qubits : qubit number redefined, all qubits and circuits will be reset before creation of new qubits !")
            self.send_cmd("reset")
            # raise Exception(n)
        if n <= 0:
            raise Exception(n)

        self.__trace("qx_client::create_qubits : %i qubits" % n)
        self.__qubits = n
        self.send_cmd("qubits %i" % n)

    def remove_qubits(self):
        """
          remove all the qubits and all the circuits on the server
        """
        self.__trace("qx_client::remove_qubits")
        # send the reset cmd to the server
        self.send_cmd("reset")
        # update our local information
        self.__qubits = 0
        self.__circuits = []
        # default main circuit is always there
        self.__circuits.append("default")

    # def create_circuit(self,name,batch_cmd):
    def create_circuit(self, name, gates):
        """
          create a circuit named 'name' using a batch command (batch command should be in the format 'cmd1; cmd2; cmd3;...'
          note: circuit should be created after creating the qubits, else an error code will be returned from the server
          batch command is problematic: it can be truncated causing errors on the server side...
        """
        # safety check
        if self.__qubits == 0:   # error : qubits should be created first
            raise self.IllegalOperationException
        '''
        if (len(batch_cmd) < self.buffer_size):
            self.send_cmd(".%s" % name)   # create the circuit named 'name'
            self.send_cmd(batch_cmd)    # build  the circuit using the batch command
        else:
            gates = batch_cmd.split("; ")
            batch = ""
            seq   = 0
            for g in gates:
                batch = batch + g + "; "
                seq = seq + 1
                if seq == 100:
                    self.send_cmd(batch)
                    seq = 0
                    batch = ""
            if seq != 0:
               self.send_cmd(batch)
        '''
        res = self.send_cmd(".%s" % name)   # create the circuit named 'name'
        # for g in gates:
        #    self.send_cmd(g)

        threshold = 100
        chunk_size = int(100)
        if (len(gates) > threshold):
            chunks = int(len(gates)/chunk_size)
            for c in range(0, chunks):
                batch = ''
                for i in range(0, chunk_size):
                    batch = batch + gates[c*chunk_size+i] + ' ; '
                # print("batch: ", batch,
                #       "gates:", gates[c*chunk_size:(c+1)*chunk_size])
                self.send_cmd(batch)
            for i in range(chunks*chunk_size, len(gates)):
                self.send_cmd(gates[i])
        else:
            for g in gates:
                self.send_cmd(g)

        # add    the circuit to our local circuit list
        self.__circuits.append(name)

    def run_circuit(self, name):
        '''
          execute the circuit named 'name'
        '''
        if name in self.__circuits:
            self.send_cmd("run %s" % name)
        else:
            raise self.IllegalOperationException

    def run_noisy_circuit(self, name, error_probability,
                          error_model="depolarizing_channel", iterations=1):
        '''
          noisy execution of the circuit named 'name' using the specified error model and error probability
        '''
        if name in self.__circuits:
            self.send_cmd("run_noisy %s %s %f %i" %
                          (name, error_model, error_probability, iterations))
        else:
            print("[~] qx_client : trying to execute ", name)
            print(self.__circuits)
            raise self.IllegalOperationException   # circuit does not exist

    def get_measurement(self, qubit):
        """
           display the measurement of qubit 'qubit' (quantum circuit programmer is responsible of measuring the qubit before)
        """
        measurement_register = self.send_cmd(
            "get_measurements")   # create the circuit named 'name'
        # print("[+] measurments register: ",measurement_register)
        start = measurement_register.find("|")
        end = measurement_register.rfind("|")
        measurement_register = measurement_register[start+2:end-1]
        # print("[+] sub-measurments register : ",measurement_register)
        bits = measurement_register.split(" | ")
        # print("[+] measurments bits: ",bits)
        # measurement bits should correspond to the qubit number
        assert(len(bits) == self.__qubits)
        return bits[len(bits)-qubit-1]

    def get_measurement_average(self, qubit):
        """
           display the measurement of qubit 'qubit' (quantum circuit programmer is responsible of measuring the qubit before)
        """
        m = self.send_cmd("measurement_average %i" %
                          qubit)   # create the circuit named 'name'
        # print("[+] measurement_average: ",m)
        return float(m.split('\n')[0])

    def list_circuits(self):
        circuits = self.send_cmd("circuits")
        # print("[+] created circuits: ", circuits)

    def disconnect(self):
        self.__trace("qx_client::disconnect : stopping qx server...")
        print("[+] stopping the QX server...")
        self.send_cmd("stop")
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        print("[+] disconnected.")

##########################################################################

"""
  Test code :
    create 3 circuits named 'init', 'epr' and 'measurement' then execute
  them and display the measurent outcomes.
"""
"""
qc = qx_client()
qc.connect()

# create 2 qubits
print("[+] creating 2 qubits...")
qc.create_qubits(2)

# create the circuits
print("[+] creating circuits...")
qc.create_circuit("init","prepz q0; prepz q1")
qc.create_circuit("epr","h q0; cnot q0,q1")
qc.create_circuit("measurement","measure q0; measure q1")

# execute the circuits
print("[+] executing circuits...")
qc.run_circuit("init")
qc.run_circuit("epr")
qc.run_circuit("measurement")

# get the measurement outcomes
print("[+] getting measurement outcome...")
m0 = qc.get_measurement(0) # get measurement outcome of qubit 0
m1 = qc.get_measurement(1) # get measurement outcome of qubit 1
print("[>] measurement outcome of qubit 0 : %i" % int(m0))
print("[>] measurement outcome of qubit 1 : %i" % int(m1))

# reinitialize the qubits to |00>
print("[+] reinitializing circuit to |00> ...")
qc.run_circuit("init")

# noisy execution of the circuits
print("[+] noisy execution of the circuits...")
for i in range(0,10):
    print("[+] noisy execution ", i)
    qc.run_noisy_circuit("epr",0.01)
    qc.run_noisy_circuit("measurement",0.01)
    # get the measurement outcomes
    print("[+] getting measurement outcome...")
    m0 = qc.get_measurement(0) # get measurement outcome of qubit 0
    m1 = qc.get_measurement(1) # get measurement outcome of qubit 1
    print("[>] measurement outcome of qubit 0 : %i" % int(m0))
    print("[>] measurement outcome of qubit 1 : %i" % int(m1))

print("[+] topping server and disconnecting ...")
qc.disconnect()
"""
