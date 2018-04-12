'''
5 primitives decomposition of the single qubit clifford group as per
Asaad et al. arXiv:1508.06676

In this decomposition the Clifford group is represented by 5 primitive gates
that are consecutively applied. (Note that X90 occurs twice in this list).
-X90-Y90-X90-mX180-mY180-

Note: now that I think some more about it this way of representing the 5
primitives decomposition may not be the most useful one.
'''

from copy import deepcopy


Five_primitives_decomposition = [[]]*(24)
# explictly reversing order because order of operators is order in time
Five_primitives_decomposition[0] = ['I']
Five_primitives_decomposition[1] = ['Y90', 'X90']
Five_primitives_decomposition[2] = ['X90', 'Y90', 'mX180']
Five_primitives_decomposition[3] = ['mX180']
Five_primitives_decomposition[4] = ['Y90', 'X90', 'mY180']
Five_primitives_decomposition[5] = ['X90', 'Y90', 'mY180']
Five_primitives_decomposition[6] = ['mY180']
Five_primitives_decomposition[7] = ['Y90', 'X90', 'mX180', 'mY180']
Five_primitives_decomposition[8] = ['X90', 'Y90']
Five_primitives_decomposition[9] = ['mX180', 'mY180']
Five_primitives_decomposition[10] = ['Y90', 'X90', 'mX180']
Five_primitives_decomposition[11] = ['X90', 'Y90', 'mX180', 'mY180']

Five_primitives_decomposition[12] = ['Y90',  'mX180']
Five_primitives_decomposition[13] = ['X90', 'mX180']
Five_primitives_decomposition[14] = ['X90', 'Y90', 'X90', 'mY180']
Five_primitives_decomposition[15] = ['Y90', 'mY180']
Five_primitives_decomposition[16] = ['X90']
Five_primitives_decomposition[17] = ['X90', 'Y90', 'X90']
Five_primitives_decomposition[18] = ['Y90', 'mX180', 'mY180']
Five_primitives_decomposition[19] = ['X90',  'mY180']
Five_primitives_decomposition[20] = ['X90', 'Y90', 'X90', 'mX180', 'mY180']
Five_primitives_decomposition[21] = ['Y90']
Five_primitives_decomposition[22] = ['X90', 'mX180', 'mY180']
Five_primitives_decomposition[23] = ['X90', 'Y90', 'X90', 'mX180']

'''
Gate decomposition decomposition of the clifford group as per
Eptstein et al. Phys. Rev. A 89, 062321 (2014)
'''
epstein_efficient_decomposition = [[]]*(24)
# explictly reversing order because order of operators is order in time
epstein_efficient_decomposition[0] = ['I']
epstein_efficient_decomposition[1] = ['Y90', 'X90']
epstein_efficient_decomposition[2] = ['mX90', 'mY90']
epstein_efficient_decomposition[3] = ['X180']
epstein_efficient_decomposition[4] = ['mY90', 'mX90']
epstein_efficient_decomposition[5] = ['X90', 'mY90']
epstein_efficient_decomposition[6] = ['Y180']
epstein_efficient_decomposition[7] = ['mY90', 'X90']
epstein_efficient_decomposition[8] = ['X90', 'Y90']
epstein_efficient_decomposition[9] = ['X180', 'Y180']
epstein_efficient_decomposition[10] = ['Y90', 'mX90']
epstein_efficient_decomposition[11] = ['mX90', 'Y90']

epstein_efficient_decomposition[12] = ['Y90', 'X180']
epstein_efficient_decomposition[13] = ['mX90']
epstein_efficient_decomposition[14] = ['X90', 'mY90', 'mX90']
epstein_efficient_decomposition[15] = ['mY90']
epstein_efficient_decomposition[16] = ['X90']
epstein_efficient_decomposition[17] = ['X90', 'Y90', 'X90']
epstein_efficient_decomposition[18] = ['mY90', 'X180']
epstein_efficient_decomposition[19] = ['X90', 'Y180']
epstein_efficient_decomposition[20] = ['X90', 'mY90', 'X90']
epstein_efficient_decomposition[21] = ['Y90']
epstein_efficient_decomposition[22] = ['mX90', 'Y180']
epstein_efficient_decomposition[23] = ['X90', 'Y90', 'mX90']

# assigning to this variable for legacy reasons
gate_decomposition = epstein_efficient_decomposition

# The fixed length decomposition
epstein_fixed_length_decomposition = deepcopy(epstein_efficient_decomposition)
for el in epstein_fixed_length_decomposition:
    for i in range(3-len(el)):
        el.append('I')


