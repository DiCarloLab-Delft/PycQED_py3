from pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group \
    import SingleQubitClifford, TwoQubitClifford
from os.path import join, dirname, abspath
from os import mkdir
import numpy as np
from zlib import crc32

output_dir = join(abspath(dirname(__file__)), 'clifford_hash_tables')
try:
    mkdir(output_dir)
except FileExistsError:
    pass

def construct_clifford_lookuptable(generator, indices):
    """
    """
    lookuptable = []
    for idx in indices:
        clifford = generator(idx=idx)
        # important to use crc32 hashing as this is a non-random hash
        hash_val = crc32(clifford.pauli_transfer_matrix.round().astype(int))
        lookuptable.append(hash_val)
    return lookuptable

def generate_hash_tables():
    print("Generating Clifford hash tables.")
    single_qubit_hash_lut = construct_clifford_lookuptable(
        SingleQubitClifford, np.arange(24))
    with open(join(output_dir, 'single_qubit_hash_lut.txt'), 'w') as f:
        for h in single_qubit_hash_lut:
            f.write(str(h)+'\n')
    two_qubit_hash_lut = construct_clifford_lookuptable(
        TwoQubitClifford, np.arange(11520))
    with open(join(output_dir, 'two_qubit_hash_lut.txt'), 'w') as f:
        for h in two_qubit_hash_lut:
            f.write(str(h)+'\n')
    print("Successfully generated Clifford hash tables.")

if __name__ == '__main__':
    generate_hash_tables()
