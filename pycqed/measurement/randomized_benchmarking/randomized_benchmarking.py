import numpy as np
from pycqed.measurement.randomized_benchmarking.clifford_group import(
    clifford_lookuptable)

from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import (HZ_gate_decomposition, XY_gate_decomposition)


def calculate_net_clifford(cliffords):
    '''
    Calculates the net-clifford corresponding to a list of cliffords using the
    clifford lookuptable. The order of the input list "cliffords" is order in
    which they are applied in time.

    Note: the order corresponds to the order in a pulse sequence but is
        the reverse of what it would be in a chained dot product.

    '''
    net_cl = 0  # assumes element 0 is the Identity
    for i in range(len(cliffords)):
        # int is added to avoid deprecation warning, input is assumed to
        # be int in the first place
        net_cl = clifford_lookuptable[net_cl, int(cliffords[i])]
    return net_cl


def calculate_recovery_clifford(cl_in, desired_cl=0):
    '''
    Extracts the clifford that has to be applied to cl_in to make the net
    operation correspond to desired_cl from the clifford lookuptable.

    This operation should perform the inverse of calculate_net_clifford
    '''
    row = list(clifford_lookuptable[cl_in])
    return row.index(desired_cl)


def decompose_clifford_seq(clifford_sequence,
                           gate_decomp='HZ'):

    if gate_decomp is 'HZ':
        gate_decomposition = HZ_gate_decomposition
    elif gate_decomp is 'XY':
        gate_decomposition = XY_gate_decomposition
    else:
        raise ValueError('Specify a valid gate decomposition, "HZ" or "XY".')

    decomposed_seq = []

    for cl in clifford_sequence:
        decomposed_seq.extend(gate_decomposition[cl])
    return decomposed_seq


def convert_clifford_sequence_to_tape(clifford_sequence, lutmapping,
                                      gate_decomp='HZ'):
    '''
    Converts a list of qubit operations to the relevant pulse elements

    This method will be overwritten depending on the hardware implementation.
    '''
    # This is intended to replace the block below but not done because
    # I cannot test it at this moment (MAR)
    # decomposed_seq = decompose_clifford_seq(clifford_sequence,
    #                                         gate_decomposition)

    if gate_decomp is 'HZ':
        gate_decomposition = HZ_gate_decomposition
    elif gate_decomp is 'XY':
        gate_decomposition = XY_gate_decomposition
    else:
        raise ValueError('Specify a valid gate decomposition, "HZ" or "XY".')

    decomposed_seq = []
    for cl in clifford_sequence:
        decomposed_seq.extend(gate_decomposition[cl])
    tape = []
    for g in decomposed_seq:
        tape.append(lutmapping.index(g))
    return tape


def randomized_benchmarking_sequence(n_cl, desired_net_cl=0,
                                     seed=None):
    '''
    Generates a sequence of length "n_cl" random cliffords and appends a
    recovery clifford to make the net result correspond to applying the
    "desired_net_cl".
    The default behaviour is that the net clifford corresponds to an
    identity ("0"). If you want e.g. an inverting sequence you should set
    the desired_net_cl to "3" (corresponds to Pauli X).
    '''
    if seed is None:
        rb_cliffords = np.random.randint(0, 24, int(n_cl))
    else:
        rng_seed = np.random.RandomState(seed)
        rb_cliffords = rng_seed.randint(0, 24, int(n_cl))

    net_clifford = calculate_net_clifford(rb_cliffords)
    recovery_clifford = calculate_recovery_clifford(
        net_clifford, desired_net_cl)

    print(compare(np.dot(HZ_group[net_clifford],
                 HZ_group[recovery_clifford]), np.eye(2)))
    print(compare(np.dot(HZ_group[recovery_clifford],
                 HZ_group[net_clifford]), np.eye(2)))

    rb_cliffords = np.append(rb_cliffords, recovery_clifford)
    return rb_cliffords



class Gate():

    def __init__(self, gate_name='I', theta=0, dim='3D'):
        self.gate_name = gate_name
        self.theta = theta

        if dim == '3D':
            self.gates = {'I':np.eye(4),
                          'X':np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, np.cos(self.theta), -np.sin(self.theta)],
                                        [0, 0, np.sin(self.theta), np.cos(self.theta)]]),
                          'Y':np.array([[1, 0, 0, 0],
                                        [0, np.cos(self.theta), 0, np.sin(self.theta)],
                                        [0, 0, 1, 0],
                                        [0, -np.sin(self.theta), 0, np.cos(self.theta)]]),
                          'Z':np.array([[1, 0, 0, 0],
                                        [0, np.cos(self.theta), -np.sin(self.theta), 0],
                                        [0, np.sin(self.theta), np.cos(self.theta), 0],
                                        [0, 0, 0, 1]]),
                          'H':np.array([[1, 0, 0, 0],
                                        [0, 0, 0, 1],
                                        [0, 0, -1, 0],
                                        [0, 1, 0, 0]], dtype=int)}
        elif dim == '2D':
            self.gates = {'I':np.eye(2),
                          'X':np.array([[np.cos(self.theta/2),-1j*np.sin(self.theta/2)],
                                        [-1j*np.sin(self.theta/2), np.cos(self.theta/2)]]),
                          'Y':np.array([[np.cos(self.theta/2),-np.sin(self.theta/2)],
                                        [np.sin(self.theta/2), np.cos(self.theta/2)]]),
                          'Z':np.array([[np.exp(-1j*self.theta/2),0],
                                        [0, np.exp(1j*self.theta/2)]]),
                          'S':np.array([[(1-1j)/np.sqrt(2),0],
                                        [0,(1+1j)/np.sqrt(2)]]),
                          'S2':np.array([[-1j,0],
                                         [0,1j]]),
                          'H':np.array([[1/np.sqrt(2),1/np.sqrt(2)],
                                        [1/np.sqrt(2),-1/np.sqrt(2)]])}
        else:
            raise ValueError('Provide valid dimension.')

        self.gate = self.gates[self.gate_name]
        self.round_elts()
    def round_elts(self):
        if self.gate.dtype == 'complex128':
            for G in [np.real(self.gate), np.imag(self.gate)]:
                for i in range(G.shape[0]):
                    for j in range(G.shape[1]):
                        if np.abs(G[i,j])<1e-3:
                            G[i,j]=0
        else:
            for i in range(self.gate.shape[0]):
                for j in range(self.gate.shape[1]):
                    if np.abs(self.gate[i,j])<1e-3:
                        self.gate[i,j]=0

    def get(self):
        return self.gate

dim = '2D'
I = Gate('I',dim=dim).get()

X = Gate('X',theta=np.pi,dim=dim).get()
X_half = Gate('X', theta=np.pi/2,dim=dim).get()
X_mhalf = Gate('X', theta=-np.pi/2,dim=dim).get()

Y = Gate('Y',theta=np.pi,dim=dim).get()
Y_half = Gate('Y', theta=np.pi/2,dim=dim).get()
Y_mhalf = Gate('Y', theta=-np.pi/2,dim=dim).get()

Z = Gate('Z',theta=np.pi,dim=dim).get()
Z_half = Gate('Z', theta=np.pi/2,dim=dim).get()
Z_mhalf = Gate('Z', theta=-np.pi/2,dim=dim).get()

HZ_group = [np.empty([2, 2])]*(24)
HZ_group[0] = I
HZ_group[1] = np.linalg.multi_dot([X_half, Z_half][::-1])
HZ_group[2] = np.linalg.multi_dot([Z_mhalf, X_mhalf][::-1])
HZ_group[3] = X
HZ_group[4] = np.linalg.multi_dot([Z_mhalf, X_mhalf, Z_half, X_mhalf][::-1])
HZ_group[5] = np.linalg.multi_dot([Z_mhalf, X_mhalf, Z][::-1])
HZ_group[6] = np.linalg.multi_dot([Z_mhalf, X, Z_half][::-1])
HZ_group[7] = np.linalg.multi_dot([Z_mhalf, X_mhalf, Z_half, X_half][::-1])
HZ_group[8] = np.linalg.multi_dot([X_half, Z_mhalf, X_half, Z_half][::-1])
HZ_group[9] = Z
HZ_group[10] = np.linalg.multi_dot([Z, X_half, Z_half][::-1])
HZ_group[11] = np.linalg.multi_dot([Z_half, X_mhalf][::-1])

HZ_group[12] = np.linalg.multi_dot([Z_half, X_half, Z_half][::-1])
HZ_group[13] = X_mhalf
HZ_group[14] = Z_half
HZ_group[15] = np.linalg.multi_dot([Z_mhalf, X_mhalf, Z_half][::-1])
HZ_group[16] = X_half
HZ_group[17] = np.linalg.multi_dot([X, Z_half][::-1])
HZ_group[18] = np.linalg.multi_dot([Z_mhalf, X_mhalf, Z_half, X][::-1])
HZ_group[19] = np.linalg.multi_dot([X_half, Z_mhalf, X, Z_half][::-1])
HZ_group[20] = np.linalg.multi_dot([X_mhalf, Z, X_half, Z_half][::-1])
HZ_group[21] = np.linalg.multi_dot([Z_mhalf, X_half, Z_half][::-1])
HZ_group[22] = np.linalg.multi_dot([X_mhalf, Z_mhalf, X, Z_half][::-1])
HZ_group[23] = Z_mhalf

def compare(A,B):
    if ((np.round(np.real(A), decimals=4)==np.round(np.real(B), decimals=4)).all() or
            (np.round(np.real(A), decimals=4)==-np.round(np.real(B), decimals=4)).all() or
            (np.round(np.real(A), decimals=4)==1j*np.round(np.real(B), decimals=4)).all() or
            (np.round(np.real(A), decimals=4)==-1j*np.round(np.real(B), decimals=4)).all()):
        return True
    else:
        return False