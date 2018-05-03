import logging
import numpy as np
from pycqed.measurement.randomized_benchmarking.clifford_group import(
    clifford_lookuptable)
import pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group as tqc

from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import(gate_decomposition)


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
                           gate_decomposition=gate_decomposition):
    decomposed_seq = []
    for cl in clifford_sequence:
        decomposed_seq.extend(gate_decomposition[cl])
    return decomposed_seq


def convert_clifford_sequence_to_tape(clifford_sequence, lutmapping,
                                      gate_decomposition=gate_decomposition):
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


def randomized_benchmarking_sequence_old(n_cl:int,
                                     desired_net_cl:int =0,
                                     seed:int=None):
    '''
    Generates a sequence of "n_cl" random single qubit Cliffords followed
    by a a recovery Clifford to make the net result correspond
    to the "desired_net_cl".

    Args:
        n_cl           (int) : number of Cliffords
        desired_net_cl (int) : idx of the desired net clifford
        seed           (int) : seed used to initialize the random number
            generator.

    The default behaviour is that the net clifford corresponds to an
    identity ("0"). If you want e.g. an inverting sequence you should set
    the desired_net_cl to "3" (corresponds to Pauli X).
    '''
    logging.warning("deprecation warning, only exists for testing "
                    "equivalence to new function.")

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

##############################################################################
# New style RB sequences (using the hash-table method) compatible
# with Clifford object.
# More advanced sequences are avaliable using this method.
##############################################################################

def randomized_benchmarking_sequence(
        n_cl: int,
        desired_net_cl:int = 0,
        number_of_qubits:int = 1,
        max_clifford_idx: int = 11520,
        interleaving_cl: int = None,
        seed: int=None):
    """
    Generates a randomized benchmarking sequence for the one or two qubit
    clifford group.

    Args:
        n_cl           (int) : number of Cliffords
        desired_net_cl (int) : idx of the desired net clifford
        number_of_qubits(int): used to determine if Cliffords are drawn
            from the single qubit or two qubit clifford group.
        max_clifford_idx (int): used to set the index of the highest random
            clifford generated. Useful to generate e.g., simultaneous two
            qubit RB sequences.
        interleaving_cl (int): interleaves the sequence with a specific
            clifford if desired
        seed           (int) : seed used to initialize the random number
            generator.
    Returns:
        list of clifford indices (ints)

    N.B. in the case of the 1 qubit clifford group this function does the
    same as "randomized_benchmarking_sequence_old" but
    does not use the 24 by 24 lookuptable method to calculate the
    net clifford. It instead uses the "Clifford" objects used in
    constructing the two qubit Clifford classes.
    The old method exists to establish the equivalence between the two methods.

    """
    if number_of_qubits == 1:
        Cl = tqc.SingleQubitClifford
        group_size = np.min([24, max_clifford_idx])
    elif number_of_qubits ==2:
        Cl = tqc.TwoQubitClifford
        group_size = np.min([11520, max_clifford_idx])
    else:
        raise NotImplementedError()

    # Generate a random sequence of Cliffords
    if seed is None:
        rb_clifford_indices = np.random.randint(0, group_size, int(n_cl))
    if seed is not None:
        rng_seed = np.random.RandomState(seed)
        rb_clifford_indices = rng_seed.randint(0, group_size, int(n_cl))

    # Add interleaving cliffords if applicable
    if interleaving_cl is not None:
        rb_clif_ind_intl = np.empty(rb_clifford_indices.size*2, dtype=int)
        rb_clif_ind_intl[0::2] = rb_clifford_indices
        rb_clif_ind_intl[1::2] = interleaving_cl
        rb_clifford_indices = rb_clif_ind_intl

    # Calculate the net clifford
    net_clifford = Cl(0)
    for idx in rb_clifford_indices:
        cliff = Cl(idx)
        # order of operators applied in is right to left, therefore
        # the new operator is applied on the left side.
        net_clifford = cliff*net_clifford

    # determine the inverse of the sequence
    recovery_to_idx_clifford = net_clifford.get_inverse()
    recovery_clifford = Cl(desired_net_cl)*recovery_to_idx_clifford
    rb_clifford_indices = np.append(rb_clifford_indices,
                                    recovery_clifford.idx)
    return rb_clifford_indices


