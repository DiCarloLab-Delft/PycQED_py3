import logging
import numpy as np
from pycqed.measurement.randomized_benchmarking.clifford_group import(
    clifford_lookuptable)
import pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group as tqc

from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import (HZ_gate_decomposition, XY_gate_decomposition,
            Five_primitives_decomposition)


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

def decompose_clifford_seq_n_qubits(clifford_sequence_list, gate_decomp='HZ'):

    """
    Returns list of physical pulses for each Clifford element for each qubit in
    the following format: [ [C_pl0_qb0], [C_pl0_qb1],.., [C_pl0_qbN],...,
     [C_plN_qb0], ..., [C_plN_qbN] ], where C_pli_qbj is the pulse decomposition
     of the ith Clifford element for qubit j.
    :param clifford_sequence_list: list of lists of random Cliffords for each qubit
    :param gate_decomp: the physical decomposition for the Cliffords
    :return: decomposed_seq
    """
    if gate_decomp is 'HZ':
        gate_decomposition = HZ_gate_decomposition
    elif gate_decomp is 'XY':
        gate_decomposition = XY_gate_decomposition
    else:
        raise ValueError('Specify a valid gate decomposition, "HZ" or "XY".')

    # convert clifford_sequence_list to an array
    clifford_sequence_array = np.zeros(
        shape=(len(clifford_sequence_list), len(clifford_sequence_list[0])),
        dtype=type(clifford_sequence_list[0][0]))

    for i, cl_lst in enumerate(clifford_sequence_list):
        clifford_sequence_array[i] = cl_lst

    decomposed_seq = []

    # iterate over columns; decompose each element in the column into physical
    # pulses and ensure that the same number of finite duration pulses, and the
    # same total number of pulses occur for each Clifford element, i.e. pad with
    # 'I' and 'Z0' pulses
    # example: decomposed_seq_temp = [Cl0, Cl1, Cl2] = [['X90'], ['mZ90'],
    # ['Z180', 'X90', 'Z90']] after padding we have [['X90', 'Z0', 'Z0'],
    # ['mZ90', 'I', 'Z0'], ['Z180', 'X90', 'Z90']]. We do this because each of
    # these sublists of pulses are applied in parallel to different qubits.
    for idx1 in range(clifford_sequence_array.shape[1]):
        decomposed_seq_temp = []
        decomposed_seq_temp.extend(
            [gate_decomposition[clifford_sequence_array[idx0][idx1]] for
             idx0 in range(clifford_sequence_array.shape[0])])

        # # add extra I pulses to make each Clifford decomposition
        # # have the same total duration
        # nr_finite_duration_pulses = [len([y for y in x if 'Z' not in y]) for
        #                              x in decomposed_seq_temp]
        # for i, pulse_list in enumerate(decomposed_seq_temp):
        #     if nr_finite_duration_pulses[i] < max(nr_finite_duration_pulses):
        #         diff_finite_duration_pulses = max(nr_finite_duration_pulses) - \
        #                                       nr_finite_duration_pulses[i]
        #         pulse_list = pulse_list + ['I']*diff_finite_duration_pulses
        #     decomposed_seq_temp[i] = pulse_list
        #
        # # if all qubits receive same number of finite duration pulses in their
        # # respective Cl_i, patch with Z0 pulses until all qubits receive the
        # # same total nr of pulses
        # pulse_list_lengths = [len(j) for j in decomposed_seq_temp]
        # for i, pulse_list in enumerate(decomposed_seq_temp):
        #     if pulse_list_lengths[i]<max(pulse_list_lengths):
        #         nr_Z0_to_add = max(pulse_list_lengths)-pulse_list_lengths[i]
        #         pulse_list = pulse_list + ['Z0']*nr_Z0_to_add
        #     decomposed_seq_temp[i] = pulse_list

        decomposed_seq.extend(decomposed_seq_temp)

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
                                     seed=None, interleaved_gate=None):
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

    If IRB, pass in interleaved_gate as string. Example: 'X180'.
    '''
    # logging.warning("deprecation warning, only exists for testing "
    #                 "equivalence to new function.")

    if seed is None:
        rb_cliffords = np.random.randint(0, 24, int(n_cl))
    else:
        rng_seed = np.random.RandomState(seed)
        rb_cliffords = rng_seed.randint(0, 24, int(n_cl))

    if interleaved_gate is not None:
        rb_cliffords = np.repeat(rb_cliffords, 2)
        try:
            gate_idx = HZ_gate_decomposition.index([interleaved_gate])
        except ValueError:
            gate_idx = XY_gate_decomposition.index([interleaved_gate])
        rb_cliffords[1::2] = [gate_idx]*(len(rb_cliffords)//2)

    net_clifford = calculate_net_clifford(rb_cliffords)
    recovery_clifford = calculate_recovery_clifford(
        net_clifford, desired_net_cl)

    rb_cliffords = np.append(rb_cliffords, recovery_clifford)
    return rb_cliffords


def get_clifford_decomposition(decomposition_name: str):

    if decomposition_name is 'HZ':
        return HZ_gate_decomposition
    elif decomposition_name is 'XY':
        return XY_gate_decomposition
    elif decomposition_name is '5Primitives':
        return Five_primitives_decomposition
    else:
        raise ValueError('Specify a valid gate decomposition, "HZ", "XY",'
                         'or "5Primitives".')

##############################################################################
# New style RB sequences (using the hash-table method) compatible
# with Clifford object.
# More advanced sequences are avaliable using this method.
##############################################################################

def randomized_benchmarking_sequence_new(
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

    # Define Clifford group
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


