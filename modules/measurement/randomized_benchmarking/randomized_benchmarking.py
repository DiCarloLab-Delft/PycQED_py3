import numpy as np
from modules.measurement.randomized_benchmarking.clifford_group import(
    clifford_lookuptable)


def calculate_net_clifford(cliffords):
    '''
    Calculates the net-clifford corresponding to a list of cliffords using the
    clifford lookuptable. Order is order in which they are applied in time.

    Note: the order corresponds to the order in a pulse sequence but is
        the reverse of what it would be in a chained dot product.

    '''
    net_cl = 0  # assumes element 0 is the Identity
    for i in range(len(cliffords)):
        net_cl = clifford_lookuptable[net_cl, cliffords[i]]
    return net_cl

def calculate_recovery_clifford(cl_in, desired_cl=0):
    '''
    Extracts the clifford that has to be applied to cl_in to make the net
    operation correspond to desired_cl from the clifford lookuptable.
    '''
    raise NotImplementedError()
    return recovery_cl


def decompose_clifford_to_gates():
    '''
    Decomposes an element of the Clifford group into primitive gates.
    The set of primitive gates is I, x90, y90, x90, -x180, -Y180 as
    explored in the MSc. thesis of S.Asaad. This decomposition is
    arbitrary.

    Currently only available for single-qubit cliffords.
    '''
    raise NotImplementedError()


def convert_pulse_sequence_to_tape():
    '''
    Converts a list of qubit operations to the relevant pulse elements

    This method will be overwritten depending on the hardware implementation.
    '''
    raise NotImplementedError()


def randomized_benchmarking_sequence(n_cl, desired_net_cl=0,
                                     seed=None):
    '''
    Generates a sequence of "n_cl" random cliffords and appends a
    recovery clifford to make the net result correspond to applying the
    "desired_net_cl". The default behaviour is that the net clifford corresponds
    to an identity ("0"), if you want e.g. an inverting sequence you should set
    the desired_net_cl to "1".
    '''
    if seed is None:
        rb_cliffords = np.random.randint(0, 24, n_cl)
    else:
        rng_seed = np.random.RandomState(seed)
        rb_cliffords = rng_seed.randint(0, 24, n_cl)

    net_clifford = calculate_net_clifford(rb_cliffords)
    recovery_clifford = calculate_recovery_clifford(
        net_clifford, desired_net_cl)

    rb_cliffords.append(recovery_clifford)

    return rb_cliffords