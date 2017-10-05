import numpy as np
from pycqed.measurement.randomized_benchmarking.clifford_group import(
    clifford_lookuptable)

from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import(HZ_gate_decomposition, gate_decomposition)


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
                           gate_decomposition='HZ'):

    if gate_decomposition is 'HZ':
        gate_decomposition = HZ_gate_decomposition
    elif gate_decomposition is 'XY':
        gate_decomposition = gate_decomposition
    else:
        raise ValueError('Specify a valid gate decomposition, "HZ" or "XY".')

    decomposed_seq = []
    for cl in clifford_sequence:
        decomposed_seq.extend(gate_decomposition[cl])
    return decomposed_seq


def convert_clifford_sequence_to_tape(clifford_sequence, lutmapping,
                                      gate_decomposition=HZ_gate_decomposition):
    '''
    Converts a list of qubit operations to the relevant pulse elements

    This method will be overwritten depending on the hardware implementation.
    '''
    # This is intended to replace the block below but not done because
    # I cannot test it at this moment (MAR)
    # decomposed_seq = decompose_clifford_seq(clifford_sequence,
    #                                         gate_decomposition)
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

    rb_cliffords = np.append(rb_cliffords, recovery_clifford)

    return rb_cliffords

