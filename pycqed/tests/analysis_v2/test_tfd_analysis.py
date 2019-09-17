from pycqed.analysis_v2 import tfd_analysis as tfda


def test_calc_tfd_hamiltonian():

    pauli_terms = {
        'XIII': 0,
        'IXII': 0,
        'IIXI': 0,
        'IIIX': 0,
        'XIXI': 0,
        'IXIX': 0,

        'ZZII': 0,
        'IIZZ': 0,
        'ZIZI': 0,
        'IZIZ': 0}

    energy_terms = tfda.calc_tfd_hamiltonian(
        pauli_terms=pauli_terms,
        g=1, T=0)
    assert 'H' in energy_terms.keys()
    assert 'H_A' in energy_terms.keys()
    assert 'H_B' in energy_terms.keys()
    assert 'H_AB' in energy_terms.keys()

    energy_terms['H'] == 0


def test_calc_tfd_hamiltonian_terms():
    pauli_terms = {
        'XIII': 0,
        'IXII': 0,
        'IIXI': 0,
        'IIIX': 0,
        'XIXI': 0,
        'IXIX': 0,

        'ZZII': 1,
        'IIZZ': 1,
        'ZIZI': 1,
        'IZIZ': 0}

    # Test scaling the temperature term T

    energy_terms = tfda.calc_tfd_hamiltonian(
        pauli_terms=pauli_terms,
        g=1, T=0)
    assert energy_terms['H'] == 2
    assert energy_terms['H_A'] == 1
    assert energy_terms['H_B'] == 1
    assert energy_terms['H_AB'] == 1

    energy_terms = tfda.calc_tfd_hamiltonian(
        pauli_terms=pauli_terms,
        g=1, T=2)
    assert energy_terms['H'] == 0
    assert energy_terms['H_A'] == 1
    assert energy_terms['H_B'] == 1
    assert energy_terms['H_AB'] == 1

    # Test scaling the coupling term g
    pauli_terms['XIII'] = 1
    energy_terms = tfda.calc_tfd_hamiltonian(
        pauli_terms=pauli_terms,
        g=0, T=2)
    assert energy_terms['H'] == 0
    assert energy_terms['H_A'] == 1
    assert energy_terms['H_B'] == 1
    assert energy_terms['H_AB'] == 1

    energy_terms = tfda.calc_tfd_hamiltonian(
        pauli_terms=pauli_terms,
        g=1, T=2)
    assert energy_terms['H'] == 1
    assert energy_terms['H_A'] == 2
    assert energy_terms['H_B'] == 1
    assert energy_terms['H_AB'] == 1