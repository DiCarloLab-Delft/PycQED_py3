#!/usr/bin/python

### setup logging before all imports (before any logging is done as to prevent a default root logger)
import CC_logging

import logging
#from qiskit.qasm.qasm import Qasm
from qiskit.circuit.quantumcircuit import QuantumCircuit
#import matplotlib.pyplot as plt


# configure our logger
log = logging.getLogger('demo_qasm')
log.setLevel(logging.DEBUG)
log.debug('starting')




src = """
OPENQASM 2.0;



// gate definitions
opaque i q;
opaque rx180 q;
opaque ry180 q;
opaque rx90 q;
opaque ry90 q;
opaque rxm90 q;
opaque rym90 q;


opaque cz c,t;

opaque prepz q;
opaque square q;
opaque spec q;

// gate aliases
opaque x q;
opaque y q;
opaque x180 q;
opaque y180 q;
opaque x90 q;
opaque y90 q;
opaque mx90 q;
opaque my q;


// test program (NB: identifiers cannot match gate names)
qreg d[9];      // data qubits
qreg ax[4];     // X ancillas
qreg az[4];     // Z ancillas
creg mx[4];
creg mz[4];

if(mx==0) x d;

// not possible with QuantumCircuit.from_qasm_str() if the gates are composite gate definitions:
// gate rx180 q { x() q; }
// gate rx180 q { u3(pi,0,pi) q; }
// x180 d;
// rx180 d;
// rx90 d;
//
// QuantumCircuit.from_qasm_str() gives:
// qiskit.exceptions.QiskitError: 'Custom non-opaque gates are not supported by as_to_dag module'
// https://github.com/Qiskit/qiskit-terra/issues/1566
// we need to change ast_to_dag::standard_extension


rx180 d;

measure ax -> mx;
barrier ax;
measure az -> mz;
barrier az;
 
"""

if 0:
    qasm = Qasm(data=src)
    ast = qasm.parse()

    # from ast_to_dag.py
    if ast.type == "program":
        for node in ast.children:
            if node.type == "format":
                # self.version = node.version()
                print('type={}: version={}'.format(node.type, node.version()))

            elif node.type == "gate":
                #self._process_gate(node)
                print('type={}: name={}'.format(node.type, node.name))

            elif node.type == "if":
                # self._process_if(node)
                """Process an if node."""
                creg_name = node.children[0].name
                cval = node.children[1].value
#                self.condition = (creg, cval)
#                self._process_node(node.children[2])
#                self.condition = None
                print('type={}: {}={}'.format(node.type, creg_name, cval))

            elif node.type == "qreg":
                print('{}: index={}, name={}'.format(node.type, node.index, node.name))

            elif node.type == "creg":
                print('{}: index={}, name={}'.format(node.type, node.index, node.name))

            elif node.type == "id":
                raise RuntimeError("internal error: _process_node on id")

            elif node.type == "int":
                raise RuntimeError("internal error: _process_node on int")

            elif node.type == "real":
                raise RuntimeError("internal error: _process_node on real")

            elif node.type == "indexed_id":
                raise RuntimeError("internal error: _process_node on indexed_id")

            else:
                print('type={}'.format(node.type))

            '''
            elif node.type == "id_list":
                # We process id_list nodes when they are leaves of barriers.

            elif node.type == "primary_list":
                # We should only be called for a barrier.

            elif node.type == "custom_unitary":
                self._process_custom_unitary(node)

            elif node.type == "universal_unitary":
                args = self._process_node(node.children[0])
                qid = self._process_bit_id(node.children[1])
                for element in qid:
                    self.dag.apply_operation_back(UBase(*args, element), self.condition)

            elif node.type == "cnot":
                self._process_cnot(node)

            elif node.type == "expression_list":
                return node.children

            elif node.type == "binop":
                raise RuntimeError("internal error: _process_node on binop")

            elif node.type == "prefix":
                raise RuntimeError("internal error: _process_node on prefix")

            elif node.type == "measure":
                self._process_measure(node)

            elif node.type == "barrier":
                ids = self._process_node(node.children[0])
                qubits = []
                for qubit in ids:
                    for j, _ in enumerate(qubit):
                        qubits.append(qubit[j])
                self.dag.apply_operation_back(Barrier(len(qubits)), qubits, [])

            elif node.type == "reset":
                id0 = self._process_bit_id(node.children[0])
                for i, _ in enumerate(id0):
                    self.dag.apply_operation_back(Reset(), [id0[i]], [], self.condition)

            elif node.type == "opaque":
                self._process_gate(node, opaque=True)

            elif node.type == "external":
                raise RuntimeError("internal error: _process_node on external")

            else:                
                raise RuntimeError("internal error: undefined node type",
                                          node.type, "line=%s" % node.line,
                                          "file=%s" % node.file)

            '''

    else:
        raise RuntimeError("Unexpected node type '{}'".format(ast.type))

else:
    circuit = QuantumCircuit.from_qasm_str(src)

    if 0:
        # text output
        t = circuit.draw(output='text')
        print(t)
    else:
        # matplotlib output
    #    fig = circuit.draw(output='mpl', interactive=True)
        fig = circuit.draw(output='mpl')
        fig.show()  # FIXME: configure Mac properly
        fig.savefig('circuitplot.png')


    # from assemble_circuits.py:
    for qreg in circuit.qregs:
        print('qreg: name={}, size={}'.format(qreg.name, qreg.size))
    for creg in circuit.cregs:
        print('creg: name={}, size={}'.format(qreg.name, qreg.size))
    for op_context in circuit.data:
        instruction = op_context[0].assemble()
        qargs = op_context[1]
        cargs = op_context[2]
        print('{}: qargs={}, cargs={}'.format(instruction.name, qargs, cargs))
        if hasattr(instruction, '_control'):
            ctrl_reg, ctrl_val = instruction._control
            print('if {} == {}'.format(ctrl_reg, ctrl_val))



# old
'''
// -- from QE
// --- QE Hardware primitives ---
// 3-parameter 2-pulse single qubit gate
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; } // 2-parameter 1-pulse single qubit gate
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
// 1-parameter 0-pulse single qubit gate
gate u1(lambda) q { U(0,0,lambda) q; } 

// Pauli gate: bit-flip
gate x a { u3(pi,0,pi) a; }
// Pauli gate: bit and phase flip
gate y a { u3(pi,pi/2,pi/2) a; }
// Pauli gate: phase flip
gate z a { u1(pi) a; }
'''