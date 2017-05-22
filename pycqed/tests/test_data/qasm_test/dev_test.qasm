# another comment
# This is comment, which should be removed.
qubit q0, q1    # define qubits.

# map declared qubits into physical qubits
map q0 0
map q1 1

# comment in the mid
Init_all    # initialize all qubits
X180 q0 | Y90 q1
I 100  # ns
CZ q0, q1
mY90 q1
Measure q0 | Measure q1
my180 q1




