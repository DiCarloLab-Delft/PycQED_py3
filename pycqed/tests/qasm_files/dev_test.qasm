# another comment
# This is comment, which should be removed.
qubit q0, q1    # define qubits.

# map declared qubits into physical qubits
map q0 0
map q1 1

# comment in the mid
Init_all    # initialize all qubits
dummy q0 q1
X180 q0 | Y90 q1
Idx 100  # ns
CZ q0, q1
Idx 5
I q0
dummy q0 q1
mY90 q1
y180 q1
Measure q0 | Measure q1


# comment in the mid
Init_all    # initialize all qubits
dummy q0 q1
X180 q0 | Y90 q1
Idx 100  # ns
CZ q0, q1
Idx 5
I q0
mY90 q1
y180 q1
dummy q0 q1
Measure q0 | Measure q1



