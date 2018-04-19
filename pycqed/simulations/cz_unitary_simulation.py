"""
April 2018
Simulates the trajectory implementing a CZ gate.
"""
import numpy as np
import qutip as qtp


U_target = qtp.Qobj([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, -1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, -1, 0],
                                [0, 0, 0, 0, 0, 1]],
                               type='oper',
                               dims=[[2, 3], [3, 2]])
U_target._type = 'oper'


psi_00 = qtp.tensor(qtp.basis(2, 0), qtp.basis(3, 0))
psi_01 = qtp.tensor(qtp.basis(2, 0), qtp.basis(3, 1))
psi_02 = qtp.tensor(qtp.basis(2, 0), qtp.basis(3, 2))
psi_10 = qtp.tensor(qtp.basis(2, 1), qtp.basis(3, 0))
psi_11 = qtp.tensor(qtp.basis(2, 1), qtp.basis(3, 1))
psi_12 = qtp.tensor(qtp.basis(2, 1), qtp.basis(3, 2))

basis_vecs = [psi_00, psi_01, psi_02, psi_10, psi_11, psi_12]


bases_states = ['00', '01', '02', '10', '11', '12']
e_ops_labels = ['psi_00', 'psi_01', 'psi_02', 'psi_10', 'psi_11', 'psi_12']
e_ops = [psi_00*psi_00.dag(), psi_01*psi_01.dag(), psi_02*psi_02.dag(),
         psi_10*psi_10.dag(), psi_11*psi_11.dag(), psi_12*psi_12.dag()]


def unitary_from_Ham(H, times):
    output = []
    for psi0 in basis_vecs:
        output.append(qtp.mesolve(H, psi0, times).states[-1])
    comb_output = np.hstack((output[0].full(),
                             output[1].full(),
                             output[2].full(),
                             output[3].full(),
                             output[4].full(),
                             output[5].full()))
    U_outp = qtp.Qobj(comb_output,
                      dims=[[2, 3], [2, 3]])
    return U_outp


def pro_fid_unitary_compsubspace(U, U_target):
    """
    Process fidelity in the computational subspace for a qubit and qutrit
    (dims U

    """
    inner = U.dag()*U_target
    part_idx = [0, 1, 3, 4]  # only computational subspace
    ptrace = 0
    for i in part_idx:
        ptrace += inner[i, i]
    dim = 4  # 2 qubits comp subspace

    return ptrace/dim


# Plts the output unitary
# qtp.hinton(U_outp)
