
import numpy as np
import qutip as qtp
import scipy

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)

np.set_printoptions(threshold=np.inf)

# Hardcoded number of levels for the two transmons.
# Currently only 3,3 or 4,3 are supported. The bottleneck is the function
# that changes to the dressed basis at sweet spot (matrix_change_of_variables)
n_levels_q0 = 3
n_levels_q1 = 3

# operators
b = qtp.tensor(qtp.destroy(n_levels_q1), qtp.qeye(n_levels_q0))  # spectator qubit
a = qtp.tensor(qtp.qeye(n_levels_q1), qtp.destroy(n_levels_q0))  # fluxing qubit
n_q0 = a.dag() * a
n_q1 = b.dag() * b


def basis_state(i, j, to_vector=True):
    # Returns ket |ij> as ket or vector in Liouville representation
    ket = qtp.tensor(
        qtp.ket([i],
        dim=[n_levels_q1]),
        qtp.ket([j],
        dim=[n_levels_q0]))  # notice it's a ket
    if to_vector:
        rho = qtp.operator_to_vector(qtp.ket2dm(ket))
    else:
        rho = ket
    return rho

# target in the case with no noise
# note that the Hilbert space is H_q1 /otimes H_q0
# E.g. for two qutrits the ordering of basis states is 00,01,02,10,11,12,20,21,22


def target_cond_phase(cond_phase=180):
    # Ideal conditional gate performed with avoided crossing 11-02
    # note that both states receive the same phase
    target_U = qtp.tensor(qtp.qeye(n_levels_q1), qtp.qeye(n_levels_q0))
    state11 = basis_state(1, 1, to_vector=False)
    state02 = basis_state(0, 2, to_vector=False)

    phase_factor = np.exp(1j * np.deg2rad(cond_phase % 360))

    target_U = target_U + (-1 + phase_factor) * state11 * state11.dag()
    target_U = target_U + (-1 + phase_factor) * state02 * state02.dag()

    return target_U


U_target = target_cond_phase()

U_target_diffdims = target_cond_phase()

# As of qutip 4.5.0 this is not needed anymore
# otherwise average_gate_fidelity doesn't work
# U_target_diffdims.dims = [
#     [n_levels_q0 * n_levels_q1],
#     [n_levels_q0 * n_levels_q1]
# ]


'''
remember that qutip uses the Liouville (matrix) representation for superoperators,
with column stacking.
E.g. for qutrits this means that
rho_{xy,x'y'}=rho[3*x+y,3*x'+y']
rho_{xy,x'y'}=operator_to_vector(rho)[3*x+y+27*x'+9*y']
where xy is the row and x'y' is the column
'''


def index_in_ket(indeces):
    # returns vector index of ket |xy>
    # Input(list): [x,y]
    x = indeces[0]
    y = indeces[1]
    return n_levels_q0*x+y


def index_in_vector_of_dm_matrix_element(ket_indeces,bra_indeces):
    # returns vector index of density matrix |xy><x_prime y_prime|
    # Input(list): [x,y], [x_prime,y_prime]
    x = ket_indeces[0]
    y = ket_indeces[1]
    x_prime = bra_indeces[0]
    y_prime = bra_indeces[1]
    return x*n_levels_q0 + y + x_prime*n_levels_q0**2*n_levels_q1 + y_prime*n_levels_q0*n_levels_q1


def list_of_vector_indeces(subspace):
    # returns list of 2-element lists depending on 'subspace'
    # See below for valid keywords
    full_list = []
    for i in range(0,n_levels_q1):
        for j in range(0,n_levels_q0):
            full_list.append([i,j])
    compsub_list = []
    for i in range(0,2):
        for j in range(0,2):
            compsub_list.append([i,j])

    if subspace == 'leaksub':
        return_list=full_list
        for elem in compsub_list:
            return_list.remove(elem)

    if subspace == 'compsub':
        return_list=compsub_list

    if subspace == 'all':
        return_list=full_list

    return return_list


def qubit_to_2qutrit_unitary(U_2q,right_or_left):
    # Transforms a unitary for a qubit into a unitary for a two-qudit system (possibly different qudits).
    # right_or_left ('right' or 'left') specifies whether U_2q acts on q0 or q1
    if right_or_left == 'right':
        d = n_levels_q0
    elif right_or_left == 'left':
        d = n_levels_q1
    U_temp = qtp.qeye(d).full()
    U_2q_matrix=U_2q.full()
    for i in range(np.size(U_2q_matrix,axis=0)):
        for j in range(np.size(U_2q_matrix,axis=1)):
            U_temp[i,j]=U_2q_matrix[i,j]
    U_temp=qtp.Qobj(U_temp,type='oper',dims=[[d],[d]])

    if right_or_left == 'right':
        return qtp.tensor(qtp.qeye(n_levels_q1),U_temp)
    elif right_or_left == 'left':
        return qtp.tensor(U_temp,qtp.qeye(n_levels_q0))


def bloch_sphere_rotation(angle, unit_vector):
    ### Inputs:
    # angle (float): rotation angle (in rad) around the axis specified by unit_vector
    # unit_vector: real vector with unit norm specifying the axis of rotation. If it doesn't have unit norm,
    #              then we normalize it

    unit_vector=unit_vector/np.linalg.norm(unit_vector)

    sigma_scalar_unit_vector = qtp.sigmax()*unit_vector[0]+qtp.sigmay()*unit_vector[1]+qtp.sigmaz()*unit_vector[2]

    return np.cos(angle/2)*qtp.qeye(2) - 1j * np.sin(angle/2) * sigma_scalar_unit_vector


########################################################################
# Functions to construct hamiltonian, collapse operators,
# and compute single quantities of interest
########################################################################


def coupled_transmons_hamiltonian_new(w_q0, w_q1, alpha_q0, alpha_q1, J):
    """
    Hamiltonian of two coupled anharmonic transmons.
    Because the intention is to tune one qubit into resonance with the other,
    the number of levels is limited.
        q1 -> static qubit, 3-levels
        q0 -> fluxing qubit, 3-levels

    intended avoided crossing:
        11 <-> 02     (q1 is the first qubit and q0 the second one)

    N.B. the frequency of q0 is expected to be larger than that of q1
        w_q0 > w_q1
        and the anharmonicities alpha negative
    """
    # this is because of a bug in numpy 1.12 with np.real (solved in >1.14)
    w_q0 = float(w_q0)
    w_q1 = float(w_q1)
    alpha_q0 = float(alpha_q0)
    alpha_q1 = float(alpha_q1)
    J = float(J)

    adag = a.dag()
    bdag = b.dag()

    H = w_q0 * n_q0 + w_q1 * n_q1 +  \
        1 / 2 * alpha_q0 * (adag * adag * a * a) + 1 / 2 * alpha_q1 * (bdag * bdag * b * b) +\
        J * (-1) * (adag * b + a * bdag)  # \
    # + J * (basis_state(0,1,to_vector=False)*basis_state(1,0,to_vector=False).dag() + \
    # basis_state(1,0,to_vector=False)*basis_state(0,1,to_vector=False).dag())
    # (a.dag() - a) * (-b + b.dag())              # we use the RWA so that the energy of |00> is 0 and avoid ambiguities
    H = H * (2 * np.pi)
    return H


def calc_hamiltonian(amp, fluxlutman, fluxlutman_static, which_gate: str = 'NE'):
    # all inputs should be given in terms of frequencies, i.e. without the 2*np.pi factor
    # instead, the output H includes already that factor
    w_q0 = fluxlutman.calc_amp_to_freq(amp, '01', which_gate=which_gate)
    w_q1 = fluxlutman.calc_amp_to_freq(amp, '10', which_gate=which_gate)
    alpha_q0 = fluxlutman.calc_amp_to_freq(amp, '02', which_gate=which_gate) - 2 * w_q0
    alpha_q1 = fluxlutman_static.q_polycoeffs_anharm()[-1]
    w_q0_intpoint = w_q1 - alpha_q0

    q_J2 = fluxlutman.get('q_J2_{}'.format(which_gate))
    J = q_J2 / np.sqrt(2)
    bus_freq = fluxlutman.get('bus_freq_{}'.format(which_gate))

    delta_q1 = w_q1 - bus_freq
    delta_q0_intpoint = (w_q0_intpoint) - bus_freq
    delta_q0 = (w_q0) - bus_freq
    J_temp = J / ((delta_q1 + delta_q0_intpoint) / (delta_q1 * delta_q0_intpoint)) * ((delta_q1 + delta_q0) / (delta_q1 * delta_q0))

    H = coupled_transmons_hamiltonian_new(w_q0=w_q0, w_q1=w_q1, alpha_q0=alpha_q0, alpha_q1=alpha_q1, J=J_temp)
    return H


def rotating_frame_transformation_propagator_new(U, t: float, H):
    """
    Transforms the frame of the unitary according to
        U' = U_{RF}*U
        NOTE: remember that this is how the time evolution operator changes from one picture to the other

    Args:
        U (QObj): Unitary or superoperator to be transformed
        t (float): time at which to transform
        H (QObj): hamiltonian to be rotated away

    """

    U_RF = (1j*H*t).expm()
    if U.type=='super':
        U_RF=qtp.to_super(U_RF)

    U_prime = U_RF * U
    """ U_RF only on one side because that's the operator that
    satisfies the Schroedinger equation in the interaction picture.
    """

    return U_prime


def rotating_frame_transformation_operators(operator, t: float, H):
    """
    Transforms the frame of an operator (hamiltonian, or jump operator) according to
        O' = U_{RF}*O*U_{RF}^dag

    Args:
        operator (QObj): operator to be transformed
        t (float): time at which to transform
        H (QObj): hamiltonian to be rotated away

    """

    U_RF = (1j*H*t).expm()

    return U_RF * operator * U_RF.dag()


def c_ops_amplitudedependent(T1_q0,T1_q1,Tphi01_q0_vec,Tphi01_q1):
    # case where the incoherent noise for qubit q0 is time dependent, or better pulse-amplitude dependent

    c_ops=[]

    if T1_q0 != 0:
        c_ops.append(np.sqrt(1/T1_q0)*a)

    if T1_q1 != 0:
        c_ops.append(np.sqrt(1/T1_q1)*b)


    rescaling_of_Tphi_02 = 2
    if rescaling_of_Tphi_02==2:
        rate_01_scaling = 4/9
        rate_02_scaling = 16/9
        rate_12_scaling = 4/9
    elif rescaling_of_Tphi_02==4:
        rate_01_scaling = 0
        rate_02_scaling = 4
        rate_12_scaling = 0
    else:
        log.warning('Unsupported rescaling of Tphi_02.')


    if Tphi01_q1 != 0:
        sigmaZinqutrit = qtp.Qobj([[1,0,0],
                                    [0,-1,0],
                                    [0,0,0]])
        collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(n_levels_q0))
        c_ops.append(collapse*np.sqrt(rate_01_scaling/(2*Tphi01_q1)))

        Tphi12_q1=Tphi01_q1
        sigmaZinqutrit = qtp.Qobj([[0,0,0],
                                    [0,1,0],
                                    [0,0,-1]])
        collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(n_levels_q0))
        c_ops.append(collapse*np.sqrt(rate_12_scaling/(2*Tphi12_q1)))

        Tphi02_q1=Tphi01_q1
        sigmaZinqutrit = qtp.Qobj([[1,0,0],
                                    [0,0,0],
                                    [0,0,-1]])
        collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(n_levels_q0))
        c_ops.append(collapse*np.sqrt(rate_02_scaling/(2*Tphi02_q1)))


    if n_levels_q0 == 3:

        if Tphi01_q0_vec != []:
            sigmaZinqutrit = qtp.Qobj([[1,0,0],
                                        [0,-1,0],
                                        [0,0,0]])
            collapse=qtp.tensor(qtp.qeye(n_levels_q1),sigmaZinqutrit)
            c_ops.append([collapse,np.sqrt(rate_01_scaling/(2*Tphi01_q0_vec))])

            Tphi12_q0_vec=Tphi01_q0_vec
            sigmaZinqutrit = qtp.Qobj([[0,0,0],
                                        [0,1,0],
                                        [0,0,-1]])
            collapse=qtp.tensor(qtp.qeye(n_levels_q1),sigmaZinqutrit)
            c_ops.append([collapse,np.sqrt(rate_12_scaling/(2*Tphi12_q0_vec))])

            Tphi02_q0_vec=Tphi01_q0_vec
            sigmaZinqutrit = qtp.Qobj([[1,0,0],
                                        [0,0,0],
                                        [0,0,-1]])
            collapse=qtp.tensor(qtp.qeye(n_levels_q1),sigmaZinqutrit)
            c_ops.append([collapse,np.sqrt(rate_02_scaling/(2*Tphi02_q0_vec))])

    elif n_levels_q0 >= 4:    # currently, when q0 is a 4-dit, we use simple model where decay is quadratic with sensitivity

        if Tphi01_q0_vec != []:

            dephasing_op_q0 = a.dag()*a
            c_ops.append([dephasing_op_q0,np.sqrt(2/Tphi01_q0_vec)])

    return c_ops


def phases_from_superoperator(U):
    """
    Returns the phases from the unitary or superoperator U
    """
    if U.type=='oper':
        index_00=index_in_ket([0,0])
        phi_00 = np.rad2deg(np.angle(U[index_00, index_00]))  # expected to equal 0 because of our
                                                # choice for the energy, not because of rotating frame. But not guaranteed including the coupling

        index_01=index_in_ket([0,1])
        phi_01 = np.rad2deg(np.angle(U[index_01, index_01]))

        index_10=index_in_ket([1,0])
        phi_10 = np.rad2deg(np.angle(U[index_10, index_10]))

        index_11=index_in_ket([1,1])
        phi_11 = np.rad2deg(np.angle(U[index_11, index_11]))

        index_02=index_in_ket([0,2])
        phi_02 = np.rad2deg(np.angle(U[index_02, index_02]))  # used only for avgatefid_superoperator_phasecorrected

        index_20=index_in_ket([2,0])
        phi_20 = np.rad2deg(np.angle(U[index_20, index_20]))  # used only for avgatefid_superoperator_phasecorrected

        index_12=index_in_ket([1,2])
        phi_12 = np.rad2deg(np.angle(U[index_12, index_12]))

        index_21=index_in_ket([2,1])
        phi_21 = np.rad2deg(np.angle(U[index_21, index_21]))

        if n_levels_q0 >= 4:
            index_03=index_in_ket([0,3])
            phi_03 = np.rad2deg(np.angle(U[index_03, index_03]))
        else:
            phi_03 = 0

    elif U.type=='super':
        phi_00 = 0   # we set it to 0 arbitrarily but it is indeed not knowable
        index_01=index_in_vector_of_dm_matrix_element([0,1],[0,0])
        phi_01 = np.rad2deg(np.angle(U[index_01, index_01]))   # actually phi_01-phi_00 etc

        index_10=index_in_vector_of_dm_matrix_element([1,0],[0,0])
        phi_10 = np.rad2deg(np.angle(U[index_10, index_10]))

        index_11=index_in_vector_of_dm_matrix_element([1,1],[0,0])
        phi_11 = np.rad2deg(np.angle(U[index_11, index_11]))

        index_02=index_in_vector_of_dm_matrix_element([0,2],[0,0])
        phi_02 = np.rad2deg(np.angle(U[index_02, index_02]))

        index_20=index_in_vector_of_dm_matrix_element([2,0],[0,0])
        phi_20 = np.rad2deg(np.angle(U[index_20, index_20]))

        index_12=index_in_vector_of_dm_matrix_element([1,2],[0,0])
        phi_12 = np.rad2deg(np.angle(U[index_12, index_12]))

        index_21=index_in_vector_of_dm_matrix_element([2,1],[0,0])
        phi_21 = np.rad2deg(np.angle(U[index_21, index_21]))

        if n_levels_q0 >= 4:
            index_03=index_in_vector_of_dm_matrix_element([0,3],[0,0])
            phi_03 = np.rad2deg(np.angle(U[index_03, index_03]))
        else:
            phi_03 = 0


    phi_cond = (phi_11 - phi_01 - phi_10 + phi_00) % 360  # still the right formula independently from phi_00

    return phi_00, phi_01, phi_10, phi_11, phi_02, phi_20, phi_12, phi_21, phi_03, phi_cond


def leakage_from_superoperator(U):
    if U.type=='oper':
        """
        Calculates leakage by summing over all in and output states in the
        computational subspace.
            L1 = 1- 1/2^{number computational qubits} sum_i sum_j abs(|<phi_i|U|phi_j>|)**2
        The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
        """
        sump = 0
        for i in range(4):
            for j in range(4):
                bra_i = qtp.tensor(qtp.ket([i//2], dim=[n_levels_q1]),
                                   qtp.ket([i % 2], dim=[n_levels_q0])).dag()
                ket_j = qtp.tensor(qtp.ket([j//2], dim=[n_levels_q1]),
                                   qtp.ket([j % 2], dim=[n_levels_q0]))
                p = np.abs((bra_i*U*ket_j).data[0, 0])**2
                sump += p
        sump /= 4  # divide by dimension of comp subspace
        L1 = 1-sump
        return np.real(L1)
    elif U.type=='super':
        """
        Calculates leakage by summing over all in and output states in the
        computational subspace.
            L1 = 1- 1/2^{number computational qubits} sum_i sum_j Tr(rho_{x'y'}C_U(rho_{xy}))
            where C_U is U in the channel representation
        The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
        """
        sump = 0
        for i in range(4):
            for j in range(4):
                ket_i = qtp.tensor(qtp.ket([i//2], dim=[n_levels_q1]),
                                   qtp.ket([i % 2], dim=[n_levels_q0])) #notice it's a ket
                rho_i=qtp.operator_to_vector(qtp.ket2dm(ket_i))
                ket_j = qtp.tensor(qtp.ket([j//2], dim=[n_levels_q1]),
                                   qtp.ket([j % 2], dim=[n_levels_q0]))
                rho_j=qtp.operator_to_vector(qtp.ket2dm(ket_j))
                p = (rho_i.dag()*U*rho_j).data[0, 0]
                sump += p
        sump /= 4  # divide by dimension of comp subspace
        sump=np.real(sump)
        L1 = 1-sump
        return L1


def seepage_from_superoperator(U):
    """
    Calculates seepage by summing over all in and output states outside the
    computational subspace.
        L1 = 1- 1/2^{number non-computational states} sum_i sum_j abs(|<phi_i|U|phi_j>|)**2
        The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
    """
    if U.type=='oper':
        sump = 0
        for i_list in list_of_vector_indeces('leaksub'):
            for j_list in list_of_vector_indeces('leaksub'):
                bra_i = qtp.tensor(qtp.ket([i_list[0]], dim=[n_levels_q1]),
                                   qtp.ket([i_list[1]], dim=[n_levels_q0])).dag()
                ket_j = qtp.tensor(qtp.ket([j_list[0]], dim=[n_levels_q1]),
                                   qtp.ket([j_list[1]], dim=[n_levels_q0]))
                p = np.abs((bra_i*U*ket_j).data[0, 0])**2
                sump += p
        sump /= n_levels_q1*n_levels_q0-4  # divide by number of non-computational states
        L1 = 1-sump
        return np.real(L1)
    elif U.type=='super':
        sump = 0
        for i_list in list_of_vector_indeces('leaksub'):
            for j_list in list_of_vector_indeces('leaksub'):
                ket_i = qtp.tensor(qtp.ket([i_list[0]], dim=[n_levels_q1]),
                                   qtp.ket([i_list[1]], dim=[n_levels_q0]))
                rho_i=qtp.operator_to_vector(qtp.ket2dm(ket_i))
                ket_j = qtp.tensor(qtp.ket([j_list[0]], dim=[n_levels_q1]),
                                   qtp.ket([j_list[1]], dim=[n_levels_q0]))
                rho_j=qtp.operator_to_vector(qtp.ket2dm(ket_j))
                p = (rho_i.dag()*U*rho_j).data[0, 0]
                sump += p
        sump /= n_levels_q1*n_levels_q0-4  # divide by number of non-computational states
        sump=np.real(sump)
        L1 = 1-sump
        return L1


def calc_population_02_state(U):
    """
    Calculates the population that escapes from |11> to |02>.
    Formula for unitary propagator:  population = |<02|U|11>|^2
    and similarly for the superoperator case.
        The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
    """
    if U.type=='oper':
        sump = 0
        for i_list in [[0,2]]:
            for j_list in [[1,1]]:
                bra_i = qtp.tensor(qtp.ket([i_list[0]], dim=[n_levels_q1]),
                                   qtp.ket([i_list[1]], dim=[n_levels_q0])).dag()
                ket_j = qtp.tensor(qtp.ket([j_list[0]], dim=[n_levels_q1]),
                                   qtp.ket([j_list[1]], dim=[n_levels_q0]))
                p = np.abs((bra_i*U*ket_j).data[0, 0])**2
                sump += p
        return np.real(sump)
    elif U.type=='super':
        sump = 0
        for i_list in [[0,2]]:
            for j_list in [[1,1]]:
                ket_i = qtp.tensor(qtp.ket([i_list[0]], dim=[n_levels_q1]),
                                   qtp.ket([i_list[1]], dim=[n_levels_q0]))
                rho_i=qtp.operator_to_vector(qtp.ket2dm(ket_i))
                ket_j = qtp.tensor(qtp.ket([j_list[0]], dim=[n_levels_q1]),
                                   qtp.ket([j_list[1]], dim=[n_levels_q0]))
                rho_j=qtp.operator_to_vector(qtp.ket2dm(ket_j))
                p = (rho_i.dag()*U*rho_j).data[0, 0]
                sump += p
        return np.real(sump)


def pro_avfid_superoperator_compsubspace(U,L1):
    """
    Average process (gate) fidelity in the qubit computational subspace for two qutrits.
    Leakage has to be taken into account, see Woods & Gambetta.
    The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
    """

    if U.type=='oper':
        inner = U.dag()*U_target
        part_idx = [index_in_ket([0,0]), index_in_ket([0,1]), index_in_ket([1,0]), index_in_ket([1,1])]  # only computational subspace
        ptrace = 0
        for i in part_idx:
            ptrace += inner[i, i]
        dim = 4  # 2 qubits comp subspace

        return np.real(((np.abs(ptrace))**2+dim*(1-L1))/(dim*(dim+1)))

    elif U.type=='super':
        kraus_form = qtp.to_kraus(U)
        dim=4 # 2 qubits in the computational subspace
        part_idx = [index_in_ket([0,0]), index_in_ket([0,1]), index_in_ket([1,0]), index_in_ket([1,1])]  # only computational subspace
        psum=0
        for A_k in kraus_form:
            ptrace = 0
            inner = U_target_diffdims.dag()*A_k # otherwise dimension mismatch
            for i in part_idx:
                ptrace += inner[i, i]
            psum += (np.abs(ptrace))**2

        return np.real((dim*(1-L1) + psum) / (dim*(dim + 1)))


def pro_avfid_superoperator_compsubspace_phasecorrected(U,L1,phases):
    """
    Average process (gate) fidelity in the qubit computational subspace for two qutrits
    Leakage has to be taken into account, see Woods & Gambetta
    The phase is corrected with Z rotations considering both transmons as qubits. The correction is done perfectly.
    The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
    """

    # Ucorrection = qtp.Qobj([[np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, np.exp(-1j*np.deg2rad(phases[1])), 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[3]-phases[-1])), 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[0])), 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[1])), 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[0]))]],
    #                 type='oper',
    #                 dims=[[3, 3], [3, 3]])

    Ucorrection = phase_correction_U(U,states_to_fix=[[0,1],[1,0]])

    if U.type=='oper':
        U=Ucorrection*U
        inner = U.dag()*U_target
        part_idx = [index_in_ket([0,0]), index_in_ket([0,1]), index_in_ket([1,0]), index_in_ket([1,1])]  # only computational subspace
        ptrace = 0
        for i in part_idx:
            ptrace += inner[i, i]
        dim = 4  # 2 qubits comp subspace

        return np.real(((np.abs(ptrace))**2+dim*(1-L1))/(dim*(dim+1)))

    elif U.type=='super':
        U=qtp.to_super(Ucorrection)*U
        kraus_form = qtp.to_kraus(U)
        dim=4 # 2 qubits in the computational subspace
        part_idx = [index_in_ket([0,0]), index_in_ket([0,1]), index_in_ket([1,0]), index_in_ket([1,1])]  # only computational subspace
        psum=0
        for A_k in kraus_form:
            ptrace = 0
            inner = U_target_diffdims.dag()*A_k # otherwise dimension mismatch
            for i in part_idx:
                ptrace += inner[i, i]
            psum += (np.abs(ptrace))**2

        ## To plot the Pauli error rates of the twirled channel:
        #calc_chi_matrix(qtp.to_super(U_target).dag()*U)

        return np.real((dim*(1-L1) + psum) / (dim*(dim + 1)))


def pro_avfid_superoperator_compsubspace_phasecorrected_onlystaticqubit(U,L1,phases):
    """
    Average process (gate) fidelity in the qubit computational subspace for two qutrits
    Leakage has to be taken into account, see Woods & Gambetta
    The phase is corrected with Z rotations considering both transmons as qubits. The correction is done perfectly.
    The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
    """

    # Ucorrection = qtp.Qobj([[np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[0])), 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[0])), 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[0]))]],
    #                 type='oper',
    #                 dims=[[3, 3], [3, 3]])

    Ucorrection = phase_correction_U(U,states_to_fix=[[1,0]])

    if U.type=='oper':
        U=Ucorrection*U
        inner = U.dag()*U_target
        part_idx = [index_in_ket([0,0]), index_in_ket([0,1]), index_in_ket([1,0]), index_in_ket([1,1])]  # only computational subspace
        ptrace = 0
        for i in part_idx:
            ptrace += inner[i, i]
        dim = 4  # 2 qubits comp subspace

        return np.real(((np.abs(ptrace))**2+dim*(1-L1))/(dim*(dim+1)))

    elif U.type=='super':
        U=qtp.to_super(Ucorrection)*U
        kraus_form = qtp.to_kraus(U)
        dim=4 # 2 qubits in the computational subspace
        part_idx = [index_in_ket([0,0]), index_in_ket([0,1]), index_in_ket([1,0]), index_in_ket([1,1])]  # only computational subspace
        psum=0
        for A_k in kraus_form:
            ptrace = 0
            inner = U_target_diffdims.dag()*A_k # otherwise dimension mismatch
            for i in part_idx:
                ptrace += inner[i, i]
            psum += (np.abs(ptrace))**2

        return np.real((dim*(1-L1) + psum) / (dim*(dim + 1)))


def pro_avfid_superoperator(U):
    """
    Average process (gate) fidelity in the whole space for two qutrits
    The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
    """
    if U.type=='oper':
        ptrace = np.abs((U.dag()*U_target).tr())**2
        dim = n_levels_q1*n_levels_q0  # dimension of the whole space
        return np.real((ptrace+dim)/(dim*(dim+1)))

    elif U.type=='super':
        return np.real(qtp.average_gate_fidelity(U,target=U_target_diffdims))


def pro_avfid_superoperator_phasecorrected(U,phases):
    """
    Average process (gate) fidelity in the whole space for two qutrits
    Qubit Z rotation and qutrit "Z" rotations are applied, taking into account the anharmonicity as well.
    The function assumes that the computational subspace (:= the 4 energy levels chosen as the two qubits) is given by
            the standard basis |0> /otimes |0>, |0> /otimes |1>, |1> /otimes |0>, |1> /otimes |1>.
            If this is not the case, one need to change the basis to that one, before calling this function.
    This function is quite useless because we are always interested in the computational subspace only.
    """
    # Ucorrection = qtp.Qobj([[np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, np.exp(-1j*np.deg2rad(phases[1])), 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, np.exp(-1j*np.deg2rad(phases[4]-phases[-1])), 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[3]-phases[-1])), 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[4]-phases[-1]+phases[2]-phases[0])), 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[5])), 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[5]+phases[1]-phases[0])), 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[4]-phases[-1]+phases[5]-phases[0]))]],
    #                 type='oper',
    #                 dims=[[3, 3], [3, 3]])

    Ucorrection = phase_correction_U(U,states_to_fix=[[0,1],[1,0],[2,0],[0,2]])

    if U.type=='oper':
        U=Ucorrection*U
        ptrace = np.abs((U.dag()*U_target).tr())**2
        dim = n_levels_q1*n_levels_q0  # dimension of the whole space
        return np.real((ptrace+dim)/(dim*(dim+1)))

    elif U.type=='super':
        U=qtp.to_super(Ucorrection)*U
        return np.real(qtp.average_gate_fidelity(U,target=U_target_diffdims))


def offset_difference_and_missing_fraction(U):

    X90 = bloch_sphere_rotation(np.pi/2, [1,0,0])
    X90_q0 = qubit_to_2qutrit_unitary(X90,'right')
    X180 = bloch_sphere_rotation(np.pi, [1,0,0])
    X180_q1 = qubit_to_2qutrit_unitary(X180, 'left')

    population_in_0_vec=[[],[],[],[]]  #[[q0 NOT pi pulsed],[q1 NOT pi pulsed],[q0 pi pulsed],[q1 pi pulsed]]
    for pi_pulse in [False,True]:

        n_samples=100
        for phi in np.linspace(0,2*np.pi,n_samples,endpoint=False):
            Uphi90 = bloch_sphere_rotation(np.pi/2, [np.cos(phi),np.sin(phi),0])
            Uphi90_q0 = qubit_to_2qutrit_unitary(Uphi90, 'right')

            pre_operations = X90_q0
            post_operations = Uphi90_q0
            if pi_pulse:
                pre_operations=pre_operations*X180_q1
                post_operations=post_operations*X180_q1
            pre_operations=qtp.to_super(pre_operations)
            post_operations=qtp.to_super(post_operations)
            if U.type=='oper':
                U=qtp.to_super(U)

            U_tot = post_operations * U * pre_operations

            population_in_0_q0 = U_tot[index_in_vector_of_dm_matrix_element([0,0],[0,0]),index_in_vector_of_dm_matrix_element([0,0],[0,0])]+ \
                                 U_tot[index_in_vector_of_dm_matrix_element([1,0],[1,0]),index_in_vector_of_dm_matrix_element([0,0],[0,0])]+\
                                 U_tot[index_in_vector_of_dm_matrix_element([2,0],[2,0]),index_in_vector_of_dm_matrix_element([0,0],[0,0])]
            population_in_0_q0=np.real(population_in_0_q0)

            population_in_0_q1 = U_tot[index_in_vector_of_dm_matrix_element([0,0],[0,0]),index_in_vector_of_dm_matrix_element([0,0],[0,0])]+ \
                                 U_tot[index_in_vector_of_dm_matrix_element([0,1],[0,1]),index_in_vector_of_dm_matrix_element([0,0],[0,0])]+\
                                 U_tot[index_in_vector_of_dm_matrix_element([0,2],[0,2]),index_in_vector_of_dm_matrix_element([0,0],[0,0])]
            if n_levels_q0 >= 4:
                population_in_0_q1 += U_tot[index_in_vector_of_dm_matrix_element([0,3],[0,3]),index_in_vector_of_dm_matrix_element([0,0],[0,0])]
            population_in_0_q1=np.real(population_in_0_q1)

            if not pi_pulse:
                population_in_0_vec[0].append(population_in_0_q0)
                population_in_0_vec[1].append(population_in_0_q1)
            else:
                population_in_0_vec[2].append(population_in_0_q0)
                population_in_0_vec[3].append(population_in_0_q1)

    average_q0_NOTpipulsed = np.average(population_in_0_vec[0])
    average_q0_pipulsed = np.average(population_in_0_vec[2])
    offset_difference = average_q0_NOTpipulsed-average_q0_pipulsed

    missing_fraction = 1-np.average(population_in_0_vec[3])

    # plot(x_plot_vec=[np.linspace(0,2*np.pi,n_samples,endpoint=False)],
    #                       y_plot_vec=[population_in_0_vec[0],population_in_0_vec[2]],
    #                       title='Offset difference',
    #                       xlabel='Phi (rad)', ylabel='Population in 0, q0', legend_labels=['q1 in 0','q1 in 1'])
    # plot(x_plot_vec=[np.linspace(0,2*np.pi,n_samples,endpoint=False)],
    #                       y_plot_vec=[population_in_0_vec[1],population_in_0_vec[3]],
    #                       title='Missing fraction',
    #                       xlabel='Phi (rad)', ylabel='Population in 0, q1', legend_labels=['q1 in 0','q1 in 1'])

    return offset_difference, missing_fraction

########################################################################
# Functions called by the main program
########################################################################


def distort_amplitude(fitted_stepresponse_ty,amp,tlist_new,sim_step_new):

    fitted_stepresponse_ty_temp=np.concatenate([np.zeros(1),fitted_stepresponse_ty[1]])    # to make gradient work properly
    impulse_response_temp=np.gradient(fitted_stepresponse_ty_temp)
    impulse_response= np.delete(impulse_response_temp,-1)     # to have t and y of the same length for interpolation

    # use interpolation to be sure that amp and impulse_response have the same delta_t separating two values
    amp_interp = interp1d(tlist_new,amp)
    impulse_response_interp = interp1d(fitted_stepresponse_ty[0],impulse_response)

    tlist_convol1 = tlist_new
    tlist_convol2 = np.arange(0, fitted_stepresponse_ty[0][-1],
                       sim_step_new)
    amp_convol = amp_interp(tlist_convol1)
    impulse_response_convol = impulse_response_interp(tlist_convol2)

    # Compute convolution
    convolved_amp=scipy.signal.convolve(amp_convol,impulse_response_convol)/sum(impulse_response_convol)
    amp_final=convolved_amp[0:np.size(tlist_convol1)]    # consider only amp during the gate time

    return amp_final


def shift_due_to_fluxbias_q0(fluxlutman,amp_final,fluxbias_q0,sim_control_CZ, which_gate: str = 'NE'):

    if not fluxlutman.get('czd_double_sided_{}'.format(which_gate)):
        omega_0 = compute_sweetspot_frequency([1,0,0],sim_control_CZ.w_q0_sweetspot())

        f_pulse = fluxlutman.calc_amp_to_freq(amp_final,'01', which_gate=which_gate)
        f_pulse = np.clip(f_pulse,a_min=None,a_max=omega_0)                    # necessary otherwise the sqrt below gives nan

        # Correction up to second order of the frequency due to flux noise, computed from w_q0(phi) = w_q0^sweetspot * sqrt(cos(pi * phi/phi_0))
        f_pulse_final = shift_due_to_fluxbias_q0_singlefrequency(f_pulse=f_pulse,omega_0=omega_0,fluxbias=fluxbias_q0,positive_branch=True)
        f_pulse_final = np.clip(f_pulse_final,a_min=None,a_max=omega_0)

        amp_final = fluxlutman.calc_freq_to_amp(f_pulse_final,state='01', which_gate=which_gate)

    else:
        half_length = int(np.size(amp_final)/2)
        amp_A = amp_final[0:half_length]                # positive and negative parts
        amp_B = amp_final[half_length:]


        omega_0 = compute_sweetspot_frequency([1,0,0],sim_control_CZ.w_q0_sweetspot())

        f_pulse_A = fluxlutman.calc_amp_to_freq(amp_A,'01', which_gate=which_gate)
        f_pulse_A = np.clip(f_pulse_A,a_min=None,a_max=omega_0)


        f_pulse_A = shift_due_to_fluxbias_q0_singlefrequency(f_pulse=f_pulse_A,omega_0=omega_0,fluxbias=fluxbias_q0,positive_branch=True)
        f_pulse_A = np.clip(f_pulse_A,a_min=None,a_max=omega_0)
        amp_A = fluxlutman.calc_freq_to_amp(f_pulse_A,state='01', which_gate=which_gate)


        f_pulse_B = fluxlutman.calc_amp_to_freq(amp_B,'01', which_gate=which_gate)
        f_pulse_B = np.clip(f_pulse_B,a_min=None,a_max=omega_0)

        f_pulse_B = shift_due_to_fluxbias_q0_singlefrequency(f_pulse=f_pulse_B,omega_0=omega_0,fluxbias=fluxbias_q0,positive_branch=False)
        f_pulse_B = np.clip(f_pulse_B,a_min=None,a_max=omega_0)
        amp_B = fluxlutman.calc_freq_to_amp(f_pulse_B,state='01',positive_branch=False, which_gate=which_gate)


        amp_final = np.concatenate([amp_A, amp_B])

    return amp_final


def return_jump_operators(sim_control_CZ, amp_final, fluxlutman, which_gate: str = 'NE'):

    T1_q0 = sim_control_CZ.T1_q0()
    T1_q1 = sim_control_CZ.T1_q1()
    T2_q0_amplitude_dependent = sim_control_CZ.T2_q0_amplitude_dependent()
    T2_q1 = sim_control_CZ.T2_q1()

    # time-independent jump operators on q1
    if T2_q1 != 0:                                        # we use 0 to mean that it is infinite
        if T1_q1 != 0:                                    # if it's 0 it means that we want to simulate only T_phi instead of T_2
            Tphi01_q1 = Tphi_from_T1andT2(T1_q1,T2_q1)
        else:
            Tphi01_q1 = T2_q1
    else:
        Tphi01_q1 = 0

    # time-dependent jump operators on q0
    if T2_q0_amplitude_dependent[0] != -1:

        omega_0 = compute_sweetspot_frequency([1,0,0],sim_control_CZ.w_q0_sweetspot())
        f_pulse = fluxlutman.calc_amp_to_freq(amp_final,'01', which_gate=which_gate)
        f_pulse_final = np.clip(f_pulse,a_min=None,a_max=omega_0)

        sensitivity = calc_sensitivity(f_pulse_final,compute_sweetspot_frequency([1,0,0],omega_0))
        for i in range(len(sensitivity)):
            if sensitivity[i] < 0.1:
                sensitivity[i] = 0.1
        inverse_sensitivity = 1/sensitivity
        T2_q0_vec=linear_with_offset(inverse_sensitivity,T2_q0_amplitude_dependent[0],T2_q0_amplitude_dependent[1])
        #for i in range(len(sensitivity)):    # manual fix for the TLS coupled at the sweetspot for Niels' Purcell device
        #    if sensitivity[i] <= 0.2:
        #        T2_q0_vec[i]=linear_with_offset(inverse_sensitivity[i],0,2e-6)

        # plot(x_plot_vec=[f_pulse_final/1e9],
        #                   y_plot_vec=[T2_q0_vec*1e6],
        #                   title='T2 vs frequency from fit',
        #                   xlabel='Frequency_q0 (GHz)', ylabel='T2 (mu s)')

        if T1_q0 != 0:
            Tphi01_q0_vec = Tphi_from_T1andT2(T1_q0,T2_q0_vec)
        else:
            Tphi01_q0_vec = T2_q0_vec
    else:
        Tphi01_q0_vec = []

    c_ops = c_ops_amplitudedependent(T1_q0 * sim_control_CZ.T2_scaling(), T1_q1 * sim_control_CZ.T2_scaling(),
                                    Tphi01_q0_vec * sim_control_CZ.T2_scaling(), Tphi01_q1 * sim_control_CZ.T2_scaling())
    return c_ops


def time_evolution_new(c_ops, sim_control_CZ, fluxlutman, fluxlutman_static, fluxbias_q1, amp, sim_step=None, intervals_list=None, which_gate: str = 'NE'):
    """
    Calculates the propagator (either unitary or superoperator)

    Args:
        sim_step(float): time between one point and another of amp
        c_ops (list of Qobj): time (in)dependent jump operators
        amp(array): amplitude in voltage describes the y-component of the trajectory to simulate. Should be equisampled in time
        fluxlutman,sim_control_CZ: instruments containing various parameters
        fluxbias_q1(float): random fluxbias on the spectator qubit

    Returns
        U_final(Qobj): propagator

    """

    q_freq_10 = fluxlutman.get('q_freq_10_{}'.format(which_gate))

    # We change the basis from the standard basis to the basis of eigenvectors of H_0
    # The columns of S are the eigenvectors of H_0, appropriately ordered
    if intervals_list is None:
        intervals_list = np.zeros(np.size(amp)) + sim_step

    H_0 = calc_hamiltonian(0, fluxlutman, fluxlutman_static, which_gate=which_gate)
    if sim_control_CZ.dressed_compsub():
        S = qtp.Qobj(matrix_change_of_variables(H_0), dims=[[n_levels_q1, n_levels_q0], [n_levels_q1, n_levels_q0]])
    else:
        S = qtp.tensor(qtp.qeye(n_levels_q1),qtp.qeye(n_levels_q0))       # line here to quickly switch off the use of S

    w_q1 = q_freq_10    # we 'save' the input value of w_q1
    if sim_control_CZ.sigma_q1() != 0:
        w_q1_sweetspot = sim_control_CZ.w_q1_sweetspot()
        if w_q1 > w_q1_sweetspot:
            log.warning('Operating frequency of q1 should be lower than its sweet spot frequency.')
            w_q1 = w_q1_sweetspot

        w_q1_biased = shift_due_to_fluxbias_q0_singlefrequency(
            f_pulse=w_q1,
            omega_0=w_q1_sweetspot,
            fluxbias=fluxbias_q1,
            positive_branch=True)
    else:
        w_q1_biased = w_q1

    log.debug('Changing fluxlutman q_freq_10_{} value to {}'.format(which_gate, w_q1_biased))
    fluxlutman.set('q_freq_10_{}'.format(which_gate), w_q1_biased)     # we insert the change to w_q1 in this way because then J1 is also tuned appropriately

    exp_L_total = 1
    # tt = 0
    for i in range(len(amp)):
        H = calc_hamiltonian(amp[i], fluxlutman, fluxlutman_static, which_gate=which_gate)
        H = S.dag() * H * S
        # qtp.Qobj(matrix_change_of_variables(H),dims=[[3, 3], [3, 3]])
        # Alternative for collapse operators that follow the basis of H
        # We do not believe that this would be the correct model.
        S_H = qtp.tensor(qtp.qeye(n_levels_q1), qtp.qeye(n_levels_q0))

        if c_ops != []:
            c_ops_temp = []
            for c in range(len(c_ops)):
                S_Hdag = S_H.dag()
                if isinstance(c_ops[c], list):
                    c_ops_temp.append(S_H * c_ops[c][0]*c_ops[c][1][i] * S_Hdag)    # c_ops are already in the H_0 basis
                else:
                    c_ops_temp.append(S_H * c_ops[c] * S_Hdag)

            # t1 = time.time()
            liouville_exp_t = (qtp.liouvillian(H, c_ops_temp) * intervals_list[i]).expm()
            # tt += time.time() - t1
        else:
            liouville_exp_t = (-1j * H * intervals_list[i]).expm()
        exp_L_total = liouville_exp_t * exp_L_total

    # log.warning('\n expm: {}\n'.format(tt))

    log.debug('Changing fluxlutman q_freq_10_{} value back to {}'.format(which_gate, w_q1))
    fluxlutman.set('q_freq_10_{}'.format(which_gate), w_q1)

    U_final = exp_L_total
    return U_final


def simulate_quantities_of_interest_superoperator_new(U, t_final, fluxlutman, fluxlutman_static, which_gate: str = 'NE'):
    """
    Calculates the quantities of interest from the propagator (either unitary or superoperator)

    t_final, w_q0, w_q1 used to move to the rotating frame

    """
    q_freq_10 = fluxlutman.get('q_freq_10_{}'.format(which_gate))

    U_final = U

    phases = phases_from_superoperator(U_final)         # order is phi_00, phi_01, phi_10, phi_11, phi_02, phi_20, phi_cond
    phi_cond = phases[-1]
    L1 = leakage_from_superoperator(U_final)
    population_02_state = calc_population_02_state(U_final)
    L2 = seepage_from_superoperator(U_final)
    avgatefid = pro_avfid_superoperator_phasecorrected(U_final, phases)
    avgatefid_compsubspace = pro_avfid_superoperator_compsubspace_phasecorrected(U_final, L1, phases)     # leakage has to be taken into account, see Woods & Gambetta
    coherent_leakage11 = np.abs(U_final[index_in_vector_of_dm_matrix_element([1, 1], [0, 2]), index_in_vector_of_dm_matrix_element([1, 1], [1, 1])])
    # print('avgatefid_compsubspace',avgatefid_compsubspace)
    offset_difference, missing_fraction = offset_difference_and_missing_fraction(U_final)

    population_transfer_12_21 = average_population_transfer_subspace_to_subspace(U_final, states_in=[[1, 2]], states_out=[[2,1]])
    if n_levels_q0 >= 4:
        population_transfer_12_03 = average_population_transfer_subspace_to_subspace(U_final, states_in=[[1, 2]], states_out=[[0,3]])
    else:
        population_transfer_12_03 = 0

    H_rotatingframe = coupled_transmons_hamiltonian_new(w_q0=fluxlutman.q_freq_01(), w_q1=q_freq_10,
                                                        alpha_q0=fluxlutman.q_polycoeffs_anharm()[-1], alpha_q1=fluxlutman_static.q_polycoeffs_anharm()[-1], J=0)  # old wrong way
    U_final_new = rotating_frame_transformation_propagator_new(U_final, t_final, H_rotatingframe)

    avgatefid_compsubspace_notphasecorrected = pro_avfid_superoperator_compsubspace(U_final_new, L1)
    # NOTE: a single qubit phase off by 30 degrees costs 5.5% fidelity
    # We now correct only for the phase of qubit left (q1), in the rotating frame
    avgatefid_compsubspace_pc_onlystaticqubit = pro_avfid_superoperator_compsubspace_phasecorrected_onlystaticqubit(U_final_new, L1, phases)

    phases = phases_from_superoperator(U_final_new)         # order is phi_00, phi_01, phi_10, phi_11, phi_02, phi_20, phi_cond
    phase_q0 = (phases[1] - phases[0]) % 360
    phase_q1 = (phases[2] - phases[0]) % 360
    cond_phase02 = (phases[4] - 2 * phase_q0 + phases[0]) % 360
    cond_phase12 = (phases[6] - 2 * phase_q0 - phase_q1 + phases[0]) % 360
    cond_phase21 = (phases[7] - phase_q0 - 2 * phase_q1 + phases[0]) % 360
    if n_levels_q0 >= 4:
        cond_phase03 = (phases[8] - 3 * phase_q0 + phases[0]) % 360
    else:
        cond_phase03 = 0
    cond_phase20 = (phases[5] - 2 * phase_q1 + phases[0]) % 360
    # print(cond_phase20+cond_phase02+phases[-1])

    phase_diff_12_02 = (phases[6] - phases[4] - phase_q1) % 360
    phase_diff_21_20 = (phases[7] - phases[5] - phase_q0) % 360

    return {'phi_cond': phi_cond, 'L1': L1, 'L2': L2, 'avgatefid_pc': avgatefid,
            'avgatefid_compsubspace_pc': avgatefid_compsubspace, 'phase_q0': phase_q0, 'phase_q1': phase_q1,
            'avgatefid_compsubspace': avgatefid_compsubspace_notphasecorrected,
            'avgatefid_compsubspace_pc_onlystaticqubit': avgatefid_compsubspace_pc_onlystaticqubit, 'population_02_state': population_02_state,
            'cond_phase02': cond_phase02, 'coherent_leakage11': coherent_leakage11,
            'offset_difference': offset_difference, 'missing_fraction': missing_fraction,
            'phase_diff_12_02': phase_diff_12_02, 'phase_diff_21_20': phase_diff_21_20,
            'cond_phase12': cond_phase12, 'cond_phase21': cond_phase21, 'cond_phase03': cond_phase03, 'cond_phase20': cond_phase20,
            'population_transfer_12_21': population_transfer_12_21, 'population_transfer_12_03': population_transfer_12_03}

########################################################################
# Support functions
########################################################################


def plot(x_plot_vec,y_plot_vec,title='No title',xlabel='No xlabel',ylabel='No ylabel',legend_labels=list(),yscale='linear'):
    # tool for plotting
    # x_plot_vec and y_plot_vec should be passed as either lists or np.array

    if isinstance(y_plot_vec,list):
        y_length=len(y_plot_vec)
    else:
        y_length=np.size(y_plot_vec)

    if legend_labels==[]:
        legend_labels=np.arange(y_length)

    for i in range(y_length):

        if isinstance(y_plot_vec[i],list):
            y_plot_vec[i]=np.array(y_plot_vec[i])
        if isinstance(legend_labels[i],int):
            legend_labels[i]=str(legend_labels[i])

        if len(x_plot_vec)==1:
            if isinstance(x_plot_vec[0],list):
                x_plot_vec[0]=np.array(x_plot_vec[0])
            plt.plot(x_plot_vec[0], y_plot_vec[i], label=legend_labels[i])
        else:
            if isinstance(x_plot_vec[i],list):
                x_plot_vec[i]=np.array(x_plot_vec[i])
            plt.plot(x_plot_vec[i], y_plot_vec[i], label=legend_labels[i])

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.show()


def gaussian(x,mean,sigma):    # normalized Gaussian
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mean)**2/(2*sigma**2))


def concatenate_CZpulse_and_Zrotations(Z_rotations_length,sim_step,tlist):
    if Z_rotations_length != 0:
        tlist_singlequbitrotations = np.arange(0,Z_rotations_length, sim_step)
        tlist = np.concatenate([tlist,tlist_singlequbitrotations+sim_step+tlist[-1]])
    return tlist


def dressed_frequencies(fluxlutman, fluxlutman_static, sim_control_CZ, which_gate: str = 'NE'):
    H_0 = calc_hamiltonian(0, fluxlutman, fluxlutman_static, which_gate=which_gate)   # computed at 0 amplitude

    # We change the basis from the standard basis to the basis of eigenvectors of H_0
    # The columns of S are the eigenvectors of H_0, appropriately ordered
    if sim_control_CZ.dressed_compsub():
        S = qtp.Qobj(matrix_change_of_variables(H_0),dims=[[n_levels_q1, n_levels_q0], [n_levels_q1, n_levels_q0]])
    else:
        S = qtp.tensor(qtp.qeye(n_levels_q1),qtp.qeye(n_levels_q0))       # line here to quickly switch off the use of S
    H_0_diag = S.dag()*H_0*S

    w_q0 = (H_0_diag[index_in_ket([0,1]),index_in_ket([0,1])]-H_0_diag[index_in_ket([0,0]),index_in_ket([0,0])]) / (2*np.pi)
    w_q1 = (H_0_diag[index_in_ket([1,0]),index_in_ket([1,0])]-H_0_diag[index_in_ket([0,0]),index_in_ket([0,0])]) / (2*np.pi)

    alpha_q0 = (H_0_diag[index_in_ket([0,2]),index_in_ket([0,2])]-H_0_diag[index_in_ket([0,0]),index_in_ket([0,0])]) / (2*np.pi) - 2*w_q0
    alpha_q1 = (H_0_diag[index_in_ket([2,0]),index_in_ket([2,0])]-H_0_diag[index_in_ket([0,0]),index_in_ket([0,0])]) / (2*np.pi) - 2*w_q1

    return np.real(w_q0), np.real(w_q1), np.real(alpha_q0), np.real(alpha_q1)


def shift_due_to_fluxbias_q0_singlefrequency(f_pulse,omega_0,fluxbias,positive_branch):

    if positive_branch:
        sign = 1
    else:
        sign = -1

    # Correction up to second order of the frequency due to flux noise, computed from w_q0(phi) = w_q0^sweetspot * sqrt(cos(pi * phi/phi_0))
    f_pulse_final = f_pulse - np.pi/2 * (omega_0**2/f_pulse) * np.sqrt(1 - (f_pulse**4/omega_0**4)) * sign * fluxbias  \
                          -np.pi**2/2 * omega_0 * (1+(f_pulse**4/omega_0**4)) / (f_pulse/omega_0)**3 * fluxbias**2
                          # with sigma up to circa 1e-3 \mu\Phi_0 the second order is irrelevant

    return f_pulse_final


def calc_sensitivity(freq,freq_sweetspot):
    # returns sensitivity in Phi_0 units and w_q0_sweetspot, times usual units
    return freq_sweetspot*np.pi/2 * np.sqrt(1-(freq/freq_sweetspot)**4)/freq


def Tphi_from_T1andT2(T1,T2):
    return 1/(-1/(2*T1)+1/T2)


def linear_with_offset(x, a, b):
    '''
    A linear signal with a fixed offset.
    '''
    return a * x + b


def matrix_change_of_variables(H_0):
    # matrix diagonalizing H_0 as
    #       S.dag()*H_0*S = diagonal

    # order enforced manually based on what we know of the two coupled-transmon system
    eigs,eigvectors=H_0.eigenstates()

    eigvectors_ordered_according2basis = []

    if n_levels_q0==3 and n_levels_q1==3:
        eigvectors_ordered_according2basis.append(eigvectors[0].full())   # 00 state
        eigvectors_ordered_according2basis.append(eigvectors[2].full())   # 01 state
        eigvectors_ordered_according2basis.append(eigvectors[5].full())   # 02 state
        eigvectors_ordered_according2basis.append(eigvectors[1].full())   # 10 state
        eigvectors_ordered_according2basis.append(eigvectors[4].full())   # 11 state
        eigvectors_ordered_according2basis.append(eigvectors[7].full())   # 12 state
        eigvectors_ordered_according2basis.append(eigvectors[3].full())   # 20 state
        eigvectors_ordered_according2basis.append(eigvectors[6].full())   # 21 state
        eigvectors_ordered_according2basis.append(eigvectors[8].full())   # 22 state

    if n_levels_q0==4 and n_levels_q1==3:
        eigvectors_ordered_according2basis.append(eigvectors[0].full())   # 00 state
        eigvectors_ordered_according2basis.append(eigvectors[2].full())   # 01 state
        eigvectors_ordered_according2basis.append(eigvectors[5].full())   # 02 state
        eigvectors_ordered_according2basis.append(eigvectors[8].full())   # 03 state
        eigvectors_ordered_according2basis.append(eigvectors[1].full())   # 10 state
        eigvectors_ordered_according2basis.append(eigvectors[4].full())   # 11 state
        eigvectors_ordered_according2basis.append(eigvectors[7].full())   # 12 state
        eigvectors_ordered_according2basis.append(eigvectors[10].full())  # 13 state
        eigvectors_ordered_according2basis.append(eigvectors[3].full())   # 20 state
        eigvectors_ordered_according2basis.append(eigvectors[6].full())   # 21 state
        eigvectors_ordered_according2basis.append(eigvectors[9].full())   # 22 state
        eigvectors_ordered_according2basis.append(eigvectors[11].full())  # 23 state

    S=np.hstack(eigvectors_ordered_according2basis)
    return S


def verify_phicond(U):          # benchmark to check that cond phase is computed correctly. Benchmark succeeded
    # superoperator case
    if U.type == 'oper':
        U=qtp.to_super(U)
    def calc_phi(U,list):
        # lists of 4 matrix elements 0 or 1
        number=index_in_vector_of_dm_matrix_element([list[0],list[1]],[list[2],list[3]])
        phase=np.rad2deg(np.angle(U[number,number]))
        return phase

    phi_01=calc_phi(U,[0,1,0,0])
    phi_10=calc_phi(U,[1,0,0,0])
    phi_11=calc_phi(U,[1,1,0,0])
    phi_cond = (phi_11-phi_01-phi_10) % 360
    print(phi_cond)

    phi_01=-calc_phi(U,[0,0,0,1])
    phi_10=calc_phi(U,[1,0,0,1])
    phi_11=calc_phi(U,[1,1,0,1])
    phi_cond = (phi_11-phi_01-phi_10) % 360
    print(phi_cond)

    phi_01=-calc_phi(U,[0,0,1,0])
    phi_10=calc_phi(U,[0,1,1,0])
    phi_11=calc_phi(U,[1,1,1,0])
    phi_cond = (phi_11-phi_01-phi_10) % 360
    print(phi_cond)

    phi_01=-calc_phi(U,[0,0,1,1])
    phi_10=calc_phi(U,[0,1,1,1])
    phi_11=-calc_phi(U,[1,0,1,1])
    phi_cond = (phi_11-phi_01-phi_10) % 360
    print(phi_cond)
    return phi_cond


# phases need to be averaged carefully, e.g. average of 45 and 315 degrees is 0, not 180
def average_phases(phases,weights):
    # phases has to be passed in degrees
    sines=np.sin(np.deg2rad(phases))
    cosines=np.cos(np.deg2rad(phases))
    # we separately average sine and cosine
    av_sines=np.average(sines,weights=weights)
    av_cosines=np.average(cosines,weights=weights)
    # need to normalize
    av_sines=av_sines/(av_sines**2+av_cosines**2)
    av_cosines=av_cosines/(av_sines**2+av_cosines**2)
    angle_temp_sin = np.arcsin(av_sines)
    angle_temp_cos = np.arccos(av_cosines)
    # then we combine them to give the unique angle with such sine and cosine
    # To avoid problems with the discontinuities of arcsin and arccos, we choose to use the average which is not very close to such discontinuities
    if np.abs(angle_temp_sin)<np.pi/3:
        if av_cosines >= 0:
            angle = angle_temp_sin
        else:
            angle = np.pi-angle_temp_sin
    elif np.abs(angle_temp_cos-np.pi/2)<np.pi/3:
        if av_sines >= 0:
            angle = angle_temp_cos
        else:
            angle = 2*np.pi-angle_temp_cos
    else:
        log.warning('Something wrong with averaging the phases.')
    return np.rad2deg(angle) % 360


def verify_CPTP(U):
    # args: U(Qobj): superoperator or unitary
    # returns: trace dist of the partial trace that should be the identity, i.e. trace dist should be zero for TP maps
    choi = qtp.to_choi(U)
    candidate_identity = choi.ptrace([0,1])    # 3 since we have a qutrit
    ptrace = qtp.tracedist(candidate_identity,qtp.tensor(qtp.qeye(n_levels_q1),qtp.qeye(n_levels_q0)))
    return ptrace


def return_instrument_args(fluxlutman, sim_control_CZ, fluxlutman_static, which_gate: str = 'NE'):

    fluxlutman_args = {'sampling_rate': fluxlutman.sampling_rate(),
                           'cz_length_' + which_gate: fluxlutman.get('cz_length_{}'.format(which_gate)),
                           'q_J2_' + which_gate: fluxlutman.get('q_J2_{}'.format(which_gate)),
                           'czd_double_sided_' + which_gate: fluxlutman.get('czd_double_sided_{}'.format(which_gate)),
                           'cz_lambda_2_' + which_gate: fluxlutman.get('cz_lambda_2_{}'.format(which_gate)),
                           'cz_lambda_3_' + which_gate: fluxlutman.get('cz_lambda_3_{}'.format(which_gate)),
                           'cz_theta_f_' + which_gate: fluxlutman.get('cz_theta_f_{}'.format(which_gate)),
                           'czd_length_ratio_' + which_gate: fluxlutman.get('czd_length_ratio_{}'.format(which_gate)),
                           'q_polycoeffs_freq_01_det': fluxlutman.q_polycoeffs_freq_01_det(),
                           'q_polycoeffs_anharm': fluxlutman.q_polycoeffs_anharm(),
                           'q_freq_01': fluxlutman.q_freq_01(),
                           'bus_freq_' + which_gate: fluxlutman.get('bus_freq_{}'.format(which_gate)),
                           'q_freq_10_' + which_gate: fluxlutman.get('q_freq_10_{}'.format(which_gate))}

    sim_control_CZ_args = {'Z_rotations_length': sim_control_CZ.Z_rotations_length(),
                                'voltage_scaling_factor': sim_control_CZ.voltage_scaling_factor(),
                                'distortions': sim_control_CZ.distortions(),
                                'T1_q0': sim_control_CZ.T1_q0(),
                                'T1_q1': sim_control_CZ.T1_q1(),
                                'T2_q0_amplitude_dependent': sim_control_CZ.T2_q0_amplitude_dependent(),
                                'T2_q1': sim_control_CZ.T2_q1(),
                                'w_q1_sweetspot': sim_control_CZ.w_q1_sweetspot(),
                                'dressed_compsub': sim_control_CZ.dressed_compsub(),
                                'sigma_q0': sim_control_CZ.sigma_q0(),
                                'sigma_q1': sim_control_CZ.sigma_q1(),
                                'T2_scaling': sim_control_CZ.T2_scaling(),
                                'look_for_minimum': sim_control_CZ.look_for_minimum(),
                                'n_sampling_gaussian_vec': sim_control_CZ.n_sampling_gaussian_vec(),
                                'cluster': sim_control_CZ.cluster(),
                                'detuning': sim_control_CZ.detuning(),
                                'initial_state': sim_control_CZ.initial_state(),
                                'total_idle_time': sim_control_CZ.total_idle_time(),
                                'waiting_at_sweetspot': sim_control_CZ.waiting_at_sweetspot(),
                                'w_q0_sweetspot': sim_control_CZ.w_q0_sweetspot(),
                                'repetitions': sim_control_CZ.repetitions(),
                                'time_series': sim_control_CZ.time_series(),
                                'overrotation_sims': sim_control_CZ.overrotation_sims(),
                                'axis_overrotation': sim_control_CZ.axis_overrotation()}

    fluxlutman_static_args = {'q_polycoeffs_anharm': fluxlutman_static.q_polycoeffs_anharm()}

    return fluxlutman_args, sim_control_CZ_args, fluxlutman_static_args


def return_instrument_from_arglist(fluxlutman, fluxlutman_args, sim_control_CZ, sim_control_CZ_args, fluxlutman_static, fluxlutman_static_args, which_gate: str = 'NE'):

    fluxlutman.sampling_rate(fluxlutman_args['sampling_rate'])
    fluxlutman.set('cz_length_{}'.format(which_gate), fluxlutman_args['cz_length_' + which_gate])
    fluxlutman.set('q_J2_{}'.format(which_gate), fluxlutman_args['q_J2_' + which_gate])
    fluxlutman.set('czd_double_sided_{}'.format(which_gate), fluxlutman_args['czd_double_sided_' + which_gate])
    fluxlutman.set('cz_lambda_2_{}'.format(which_gate), fluxlutman_args['cz_lambda_2_' + which_gate])
    fluxlutman.set('cz_lambda_3_{}'.format(which_gate), fluxlutman_args['cz_lambda_3_' + which_gate])
    fluxlutman.set('cz_theta_f_{}'.format(which_gate), fluxlutman_args['cz_theta_f_' + which_gate])
    fluxlutman.set('czd_length_ratio_{}'.format(which_gate), fluxlutman_args['czd_length_ratio_' + which_gate])
    fluxlutman.set('bus_freq_{}'.format(which_gate), fluxlutman_args['bus_freq_' + which_gate])
    fluxlutman.set('q_freq_10_{}'.format(which_gate), fluxlutman_args['q_freq_10_' + which_gate])
    fluxlutman.q_polycoeffs_freq_01_det(fluxlutman_args['q_polycoeffs_freq_01_det'])
    fluxlutman.q_polycoeffs_anharm(fluxlutman_args['q_polycoeffs_anharm'])
    fluxlutman.q_freq_01(fluxlutman_args['q_freq_01'])

    sim_control_CZ.Z_rotations_length(sim_control_CZ_args['Z_rotations_length'])
    sim_control_CZ.voltage_scaling_factor(sim_control_CZ_args['voltage_scaling_factor'])
    sim_control_CZ.distortions(sim_control_CZ_args['distortions'])
    sim_control_CZ.T1_q0(sim_control_CZ_args['T1_q0'])
    sim_control_CZ.T1_q1(sim_control_CZ_args['T1_q1'])
    sim_control_CZ.T2_q0_amplitude_dependent(sim_control_CZ_args['T2_q0_amplitude_dependent'])
    sim_control_CZ.T2_q1(sim_control_CZ_args['T2_q1'])
    sim_control_CZ.w_q1_sweetspot(sim_control_CZ_args['w_q1_sweetspot'])
    sim_control_CZ.dressed_compsub(sim_control_CZ_args['dressed_compsub'])
    sim_control_CZ.sigma_q0(sim_control_CZ_args['sigma_q0'])
    sim_control_CZ.sigma_q1(sim_control_CZ_args['sigma_q1'])
    sim_control_CZ.T2_scaling(sim_control_CZ_args['T2_scaling'])
    sim_control_CZ.look_for_minimum(sim_control_CZ_args['look_for_minimum'])
    sim_control_CZ.n_sampling_gaussian_vec(sim_control_CZ_args['n_sampling_gaussian_vec'])
    sim_control_CZ.cluster(sim_control_CZ_args['cluster'])
    sim_control_CZ.detuning(sim_control_CZ_args['detuning'])
    sim_control_CZ.initial_state(sim_control_CZ_args['initial_state'])
    sim_control_CZ.total_idle_time(sim_control_CZ_args['total_idle_time'])
    sim_control_CZ.waiting_at_sweetspot(sim_control_CZ_args['waiting_at_sweetspot'])
    sim_control_CZ.w_q0_sweetspot(sim_control_CZ_args['w_q0_sweetspot'])
    sim_control_CZ.repetitions(sim_control_CZ_args['repetitions'])
    sim_control_CZ.time_series(sim_control_CZ_args['time_series'])
    sim_control_CZ.overrotation_sims(sim_control_CZ_args['overrotation_sims'])
    sim_control_CZ.axis_overrotation(sim_control_CZ_args['axis_overrotation'])

    fluxlutman_static.q_polycoeffs_anharm(fluxlutman_static_args['q_polycoeffs_anharm'])

    return fluxlutman, sim_control_CZ, fluxlutman_static


def plot_spectrum(fluxlutman, fluxlutman_static, which_gate: str = 'NE'):
    eig_vec=[]
    amp_vec=np.arange(0,1.5,.01)
    for amp in amp_vec:
        H = calc_hamiltonian(amp, fluxlutman, fluxlutman_static, which_gate=which_gate)
        eigs=H.eigenenergies()
        eig_vec.append(eigs)
    eig_vec=np.array(eig_vec)/1e9/(2*np.pi)
    eig_plot=[]
    for j in range(len(eig_vec[0])):
        eig_plot.append(eig_vec[:,j])
    plot(x_plot_vec=[fluxlutman.calc_amp_to_freq(amp_vec,'01', which_gate=which_gate)/1e9],
                          y_plot_vec=eig_plot,
                          title='Spectrum',
                          xlabel=r'$\omega_{q0}$ (GHz)',ylabel='Frequency (GHz)')


def conditional_frequency(amp, fluxlutman, fluxlutman_static, which_gate: str = 'NE'):
    # returns the energy difference (in Hz) between the actual 11 state and the bare 11 state
    #              (whose energy is equal to the sum of the energies of the 01 and 10 states)
    # amp=0 returns the residual coupling
    H = calc_hamiltonian(amp, fluxlutman, fluxlutman_static, which_gate=which_gate)
    eigs=H.eigenenergies()
    cond_frequency = eigs[4]-eigs[1]-eigs[2]+eigs[0]
    cond_frequency = cond_frequency/(2*np.pi)
    return cond_frequency


def steady_state_populations(gamma12,gamma21,gamma23=0,gamma32=0.00001):
    normalization = gamma12*gamma32+gamma21*gamma32+gamma12*gamma23
    p_1_star = (gamma21*gamma32) / normalization
    p_2_star = (gamma12*gamma32) / normalization
    p_3_star = (gamma12*gamma23) / normalization
    return p_1_star, p_2_star, p_3_star

def calc_rates(L_1,L_12to03,t_cycle,T_1):
    gamma12 = L_1
    gamma21 = 2*L_1 + (1-np.exp(-t_cycle/(T_1/2)))
    gamma23 = L_12to03/2
    gamma21 = L_12to03/2 + (1-np.exp(-t_cycle/(T_1/3)))
    return gamma12,gamma21,gamma23,gamma32


def sensitivity_to_fluxoffsets(U_final_vec, input_to_parallelize, t_final, fluxlutman, fluxlutman_static, which_gate: str = 'NE'):
    '''
    Function used to study the effect of constant flux offsets on the quantities of interest.
    The input should be a series of propagators computed for different flux offsets,
    for a CZ gate that without offsets would be pretty good
    '''

    leakage_vec=[]
    infid_vec=[]
    phase_q0_vec=[]
    phase_q1_vec=[]
    fluxbias_q0_vec=[]
    cond_phase_vec=[]

    mid_index=int(len(U_final_vec)/2)    # mid_index corresponds to the quantities of interest without quasi-static flux noise
    #print(mid_index)

    for i in range(len(U_final_vec)):
        if U_final_vec[i].type == 'oper':
            U_final_vec[i] = qtp.to_super(U_final_vec[i])
        qoi_temp = simulate_quantities_of_interest_superoperator_new(U=U_final_vec[i], t_final=t_final, fluxlutman=fluxlutman, fluxlutman_static=fluxlutman_static, which_gate=which_gate)
        if i==mid_index:
            print('qoi_temp =',qoi_temp)
        leakage_vec.append(qoi_temp['L1'])
        infid_vec.append(1-qoi_temp['avgatefid_compsubspace_pc'])
        phase_q0_vec.append(qoi_temp['phase_q0'])
        phase_q1_vec.append(qoi_temp['phase_q1'])
        cond_phase_vec.append(qoi_temp['phi_cond'])

        fluxbias_q0=input_to_parallelize[i]['fluxbias_q0']
        fluxbias_q0_vec.append(fluxbias_q0)

    fluxbias_q0_vec=np.array(fluxbias_q0_vec)
    leakage_vec=np.array(leakage_vec)    # absolute value for the leakage
    cond_phase_vec=np.array(cond_phase_vec)-cond_phase_vec[mid_index]       # for phases, relative value to the case with no quasi-static noise
    phase_q0_vec=np.array(phase_q0_vec)-phase_q0_vec[mid_index]
    phase_q1_vec=np.array(phase_q1_vec)-phase_q1_vec[mid_index]
    infid_vec=np.array(infid_vec)        # absolute value for the infidelity

    plot(x_plot_vec=[np.array(fluxbias_q0_vec)*1e3],
                  y_plot_vec=[np.array(leakage_vec)*100],
                  title='Sensitivity to quasi_static flux offsets',
                  xlabel='Flux offset (m$\Phi_0$)',ylabel='Leakage $L_1$ (%)',
                  legend_labels=['leakage'])
    plot(x_plot_vec=[np.array(fluxbias_q0_vec)*1e3],
                  y_plot_vec=[cond_phase_vec,phase_q0_vec,phase_q1_vec],
                  title='Sensitivity to quasi_static flux offsets',
                  xlabel='Flux offset (m$\Phi_0$)',ylabel='Phase (deg)',
                  legend_labels=['conditional phase err','phase QR err','phase QL err'])
    plot(x_plot_vec=[np.array(fluxbias_q0_vec)*1e3],
                  y_plot_vec=[np.array(infid_vec)*100],
                  title='Sensitivity to quasi_static flux offsets',
                  xlabel='Flux offset (m$\Phi_0$)',ylabel='Infidelity $L_1$ (%)',
                  legend_labels=['infidelity'])

    print('fluxbias_q0_vec =',fluxbias_q0_vec.tolist())
    print('leakage_vec =',leakage_vec.tolist())
    print('cond_phase_vec =',cond_phase_vec.tolist())
    print('phase_q0_vec =',phase_q0_vec.tolist())
    print('phase_q1_vec =',phase_q1_vec.tolist())
    print('infid_vec =',infid_vec.tolist())


def repeated_CZs_decay_curves(U_superop_average,t_final,fluxlutman,fluxlutman_static, which_gate: str = 'NE'):
    '''
    Function used to study how the leakage accumulation differs from the case in which we use directly the gate that comes out of the simulations
    and the case in which we artificially dephase the leakage subspace wrt the computational subspace.
    The input should be the propagator of a CZ gate that is pretty good.
    '''

    leakage_vec=[]
    infid_vec=[]
    leakage_dephased_vec=[]
    infid_dephased_vec=[]

    popul_in_20=[]
    popul_in_02=[]
    popul_in_21from12=[]
    popul_test=[]
    popul_in_10from01=[]
    popul_in_12from12=[]

    popul_in_20_dephased=[]
    popul_in_02_dephased=[]
    popul_in_21from12_dephased=[]
    popul_test_dephased=[]
    popul_in_10from01_dephased=[]
    popul_in_12from12_dephased=[]

    if n_levels_q0 >= 4:
        popul_in_03from12=[]
        popul_in_03from12_dephased=[]


    dimensions = U_superop_average.dims

    U_temp = U_superop_average.full()

    U_temp = nullify_coherence(U_temp,[1,1],[0,2])
    U_temp = nullify_coherence(U_temp,[1,2],[2,1])

    if n_levels_q0 >= 4:
        U_temp = nullify_coherence(U_temp,[1,2],[0,3])
        U_temp = nullify_coherence(U_temp,[2,1],[0,3])


    U_superop_dephased = qtp.Qobj(U_temp,type='super',dims=dimensions)

    number_CZ_repetitions=60
    step_repetitions=1
    for n in range(1,number_CZ_repetitions,step_repetitions):        # we consider only odd n so that in theory it should be always a CZ
        U_superop_n=U_superop_average**n
        U_superop_dephased_n = U_superop_dephased**n
        qoi=simulate_quantities_of_interest_superoperator_new(U=U_superop_n,t_final=t_final*n,fluxlutman=fluxlutman,fluxlutman_static=fluxlutman_static, which_gate=which_gate)
        qoi_dephased=simulate_quantities_of_interest_superoperator_new(U=U_superop_dephased_n,t_final=t_final*n,fluxlutman=fluxlutman,fluxlutman_static=fluxlutman_static, which_gate=which_gate)
        leakage_vec.append(qoi['L1'])
        infid_vec.append(1-qoi['avgatefid_compsubspace_pc'])
        leakage_dephased_vec.append(qoi_dephased['L1'])
        infid_dephased_vec.append(1-qoi_dephased['avgatefid_compsubspace_pc'])

        popul_in_20.append(average_population_transfer_subspace_to_subspace(U_superop_n,states_in=[[1,1]],states_out=[[2,0]]))
        popul_in_02.append(average_population_transfer_subspace_to_subspace(U_superop_n,states_in=[[1,1]],states_out=[[0,2]]))
        popul_in_21from12.append(average_population_transfer_subspace_to_subspace(U_superop_n,states_in=[[1,2]],states_out=[[2,1]]))
        popul_test.append(average_population_transfer_subspace_to_subspace(U_superop_n,states_in=[[1,2]],states_out='all'))
        popul_in_10from01.append(average_population_transfer_subspace_to_subspace(U_superop_n,states_in=[[0,1]],states_out=[[1,0]]))
        popul_in_12from12.append(average_population_transfer_subspace_to_subspace(U_superop_n,states_in=[[1,2]],states_out=[[1,2]]))

        popul_in_20_dephased.append(average_population_transfer_subspace_to_subspace(U_superop_dephased_n,states_in=[[1,1]],states_out=[[2,0]]))
        popul_in_02_dephased.append(average_population_transfer_subspace_to_subspace(U_superop_dephased_n,states_in=[[1,1]],states_out=[[0,2]]))
        popul_in_21from12_dephased.append(average_population_transfer_subspace_to_subspace(U_superop_dephased_n,states_in=[[1,2]],states_out=[[2,1]]))
        popul_test_dephased.append(average_population_transfer_subspace_to_subspace(U_superop_dephased_n,states_in=[[1,2]],states_out='all'))
        popul_in_10from01_dephased.append(average_population_transfer_subspace_to_subspace(U_superop_dephased_n,states_in=[[0,1]],states_out=[[1,0]]))
        popul_in_12from12_dephased.append(average_population_transfer_subspace_to_subspace(U_superop_dephased_n,states_in=[[1,2]],states_out=[[1,2]]))

        if n_levels_q0 >= 4:
            popul_in_03from12.append(average_population_transfer_subspace_to_subspace(U_superop_n,states_in=[[1,2]],states_out=[[0,3]]))
            popul_in_03from12_dephased.append(average_population_transfer_subspace_to_subspace(U_superop_dephased_n,states_in=[[1,2]],states_out=[[0,3]]))


    plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(leakage_vec)*100,np.array(leakage_dephased_vec)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Leakage (%)',
                  legend_labels=['Using directly the $CZ$ from the simulations','Depolarizing the leakage subspace'])

    plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(popul_in_02)*100,np.array(popul_in_02_dephased)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Av. Population out (%)',
                  legend_labels=['11 to 02','11 to 02, dephased case'])

    plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(popul_in_20)*100,np.array(popul_in_20_dephased)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Av. Population out (%)',
                  legend_labels=['11 to 20','11 to 20, dephased case'])

    plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(popul_in_21from12)*100,np.array(popul_in_21from12_dephased)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Av. Population out (%)',
                  legend_labels=['12 to 21','12 to 21, dephased case'])

    if n_levels_q0 >= 4:
        plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(popul_in_03from12)*100,np.array(popul_in_03from12_dephased)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Av. Population out (%)',
                  legend_labels=['12 to 03','12 to 03, dephased case'])

    # test should give back always 1
    plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(popul_test)*100,np.array(popul_test_dephased)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Av. Population out (%)',
                  legend_labels=['test','test, dephased case'])

    plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(popul_in_10from01)*100,np.array(popul_in_10from01_dephased)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Av. Population out (%)',
                  legend_labels=['01 to 10','01 to 10, dephased case'])

    plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(popul_in_12from12)*100,np.array(popul_in_21from12)*100,np.array(popul_in_03from12)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Population (%)',
                  legend_labels=['12','21', '03'])

    plot(x_plot_vec=[np.arange(1,number_CZ_repetitions,step_repetitions)],
                  y_plot_vec=[np.array(popul_in_12from12_dephased)*100,np.array(popul_in_21from12_dephased)*100,np.array(popul_in_03from12_dephased)*100],
                  title='Repeated $CZ$ gates',
                  xlabel='Number of CZ gates',ylabel='Population (%)',
                  legend_labels=['12','21', '03'])


    print('leakage_vec',leakage_vec)
    print('leakage_dephased_vec',leakage_dephased_vec)
    print('popul_in_20_from_11',popul_in_20)
    print('popul_in_02_from_11',popul_in_02)
    print('popul_in_21_from_12',popul_in_21from12)
    print('popul_test',popul_test)
    print('popul_in_10_from_01',popul_in_10from01)
    if n_levels_q0 >= 4:
        print('popul_in_03_from_12',popul_in_03from12)


def add_waiting_at_sweetspot(tlist,amp,waiting_at_sweetspot):

    half_length = int(np.size(amp)/2)
    amp_A = amp[0:half_length]                # positive and negative parts
    amp_B = amp[half_length:]
    tlist_A = tlist[0:half_length]                # positive and negative parts
    tlist_B = tlist[half_length:]

    sim_step = tlist[1]-tlist[0]

    tlist_update = concatenate_CZpulse_and_Zrotations(waiting_at_sweetspot-sim_step/2,sim_step,tlist_A)
    tlist_update = concatenate_CZpulse_and_Zrotations(tlist_A[-1]+sim_step/2,sim_step,tlist_update)
    amp_mid = np.zeros(np.size(tlist_update)-np.size(tlist))
    amp = np.concatenate([amp_A,amp_mid,amp_B])

    return tlist_update, amp


def correct_phases(U):

    Ucorrection = phase_correction_U(U,states_to_fix=[[0,1],[1,0]])

    if U.type=='oper':
        U=Ucorrection*U

    elif U.type=='super':
        U=qtp.to_super(Ucorrection)*U

    return U


def phase_correction_U(U,states_to_fix):
    # function that apply phase corrections for adequate computation of fidelity to a CZ.

    phases=phases_from_superoperator(U)

    Ucorrection = qtp.tensor(qtp.qeye(n_levels_q1), qtp.qeye(n_levels_q0))

    for state in states_to_fix:

        if state == [0,1]:
            corr = qtp.tensor(qtp.qeye(n_levels_q1), qtp.qeye(n_levels_q0)) + \
                    (np.exp(-1j*np.deg2rad(phases[1]))-1) * qtp.tensor(qtp.qeye(n_levels_q1), qtp.ket2dm(qtp.ket([1],[n_levels_q0])))

        if state == [1,0]:
            corr = qtp.tensor(qtp.qeye(n_levels_q1), qtp.qeye(n_levels_q0)) + \
                    (np.exp(-1j*np.deg2rad(phases[2]))-1) * qtp.tensor(qtp.ket2dm(qtp.ket([1],[n_levels_q1])), qtp.qeye(n_levels_q0))

        if state == [0,2]:                      # !!!! correction for this state is currently wrong because it kills the conditional phase of 02
            corr = qtp.tensor(qtp.qeye(n_levels_q1), qtp.qeye(n_levels_q0)) + \
                    (np.exp(-1j*np.deg2rad(phases[4]))-1) * qtp.ket2dm(basis_state(0,2,to_vector=False))

        if state == [2,0]:
            corr = qtp.tensor(qtp.qeye(n_levels_q1), qtp.qeye(n_levels_q0)) + \
                    (np.exp(-1j*np.deg2rad(phases[5]))-1) * qtp.ket2dm(basis_state(2,0,to_vector=False))

        Ucorrection = corr*Ucorrection

    return Ucorrection



def compute_sweetspot_frequency(polycoeff,freq_at_0_amp):
    return polycoeff[1]**2/2/polycoeff[0]-polycoeff[2]+freq_at_0_amp



########################################################################
# functions for Ramsey/Rabi simulations
########################################################################


def calc_populations(U):
    # calculate populations for Ram/Echo-Z experiment

    hadamard_q0 = bloch_sphere_rotation(-np.pi,[1/np.sqrt(2),0,1/np.sqrt(2)])
    hadamard_q0 = qubit_to_2qutrit_unitary(hadamard_q0,'right')

    if U.type == 'oper':
        U_pi2_pulsed = hadamard_q0 * U * hadamard_q0
        populations = {'population_lower_state': np.abs(U_pi2_pulsed[index_in_ket([0,0]),index_in_ket([0,0])])**2,
                       'population_higher_state': np.abs(U_pi2_pulsed[index_in_ket([0,0]),index_in_ket([0,1])])**2}
    elif U.type == 'super':
        U_pi2_pulsed = qtp.to_super(hadamard_q0) * U * qtp.to_super(hadamard_q0)
        populations = {'population_lower_state': np.real(U_pi2_pulsed[index_in_vector_of_dm_matrix_element([0,0],[0,0]),index_in_vector_of_dm_matrix_element([0,0],[0,0])]),
                       'population_higher_state': np.real(U_pi2_pulsed[index_in_vector_of_dm_matrix_element([0,0],[0,1]),index_in_vector_of_dm_matrix_element([0,0],[0,0])])}

    return populations


def calc_populations_new(rho_out,population_states):
    # calculate populations for given states. Used to study dephasing in 11-02 subspace and to simulate Chevrons.

    populations = {'population_higher_state': np.abs((rho_out.dag()*qtp.operator_to_vector(qtp.ket2dm(population_states[0]))).data[0,0]),
                   'population_lower_state': np.abs((rho_out.dag()*qtp.operator_to_vector(qtp.ket2dm(population_states[1]))).data[0,0])}

    return populations


def quantities_of_interest_ramsey(U, initial_state, fluxlutman, fluxlutman_static, sim_control_CZ, which_gate: str = 'NE'):

    if initial_state == '11_dressed':
        freq = sim_control_CZ.w_q0_sweetspot() + sim_control_CZ.detuning()
        amp = fluxlutman.calc_freq_to_amp(freq, which_gate=which_gate)
        H = calc_hamiltonian(amp, fluxlutman, fluxlutman_static, which_gate=which_gate)
        eigs,eigvectors = H.eigenstates()
        psi_in = eigvectors[4]

        population_states = [eigvectors[5],eigvectors[4]]     # [higher state = 02, lower state = 11]

        rho_in = qtp.operator_to_vector(qtp.ket2dm(psi_in))
        rho_out = U * rho_in
        populations = calc_populations_new(rho_out,population_states)

    elif initial_state == '11_bare':
        amp = 0
        H = calc_hamiltonian(amp, fluxlutman, fluxlutman_static, which_gate=which_gate)
        eigs,eigvectors = H.eigenstates()
        psi_in = eigvectors[4]

        population_states = [eigvectors[5],eigvectors[4]]     # [higher state = 02, lower state = 11]

        rho_in = qtp.operator_to_vector(qtp.ket2dm(psi_in))
        rho_out = U * rho_in
        populations = calc_populations_new(rho_out,population_states)

    elif initial_state == 'ramsey_q0':
        populations = calc_populations(U)

    else:
        print('invalid keyword for initial state')

    return populations

########################################################################
# effective Pauli error rates
########################################################################

def calc_chi_matrix(U):

    '''
    Input: superoperator U for two qutrits, in the Liouville representation.
    Returns: chi matrix of the two-qubit subspace.
             Note that it will not be necessarily positive definite and normalized due to leakage.
    '''

    Pauli_gr_size = 16
    U_2qubits = np.zeros([Pauli_gr_size,Pauli_gr_size])

    indexlist=[]
    for x_prime in [0,1]:
        for y_prime in [0,1]:
            for x in [0,1]:
                for y in [0,1]:
                    indexlist.append(index_in_vector_of_dm_matrix_element([x,y],[x_prime,y_prime]))

    for i in range(Pauli_gr_size):              # projecting over the two qubit subspace
        for j in range(Pauli_gr_size):
            U_2qubits[i,j]=U[indexlist[i],indexlist[j]]

    U_2qubits=qtp.Qobj(U_2qubits,type='super',dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])
    chi_matrix = qtp.to_chi(U_2qubits)/Pauli_gr_size    # normalize so that the trace is 1
    #print(chi_matrix)

    paulis_label=['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ']
    paulis_label_mod=['1-II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ','leak']

    #qtp.hinton(chi_matrix,xlabels=paulis_label,ylabels=paulis_label,title='Chi matrix')

    diag = chi_matrix.diag()
    leak=1-sum(diag)           # we quantify leakage as the missing trace of the chi matrix (different notion from Wood&Gambetta)

    diag=np.concatenate((diag,np.array([leak])))
    diag[0]=1-diag[0]

    plot(x_plot_vec=[paulis_label_mod],
                          y_plot_vec=[diag],
                          title='Pauli error rates from the chi matrix',
                          xlabel='Pauli index',ylabel='Error',yscale='log')
    print(diag)


    return chi_matrix



def calc_diag_pauli_transfer_matrix(U,U_target):
    # not useful function because it is not immediate to infer the pauli error rates from it. Use calc_chi_matrix

    pauli_list_q1 = [qubit_to_2qutrit_unitary(qtp.qeye(2),'left'),qubit_to_2qutrit_unitary(qtp.sigmax(),'left'),
                     qubit_to_2qutrit_unitary(qtp.sigmay(),'left'),qubit_to_2qutrit_unitary(qtp.sigmaz(),'left')]
    pauli_list_q0 = [qubit_to_2qutrit_unitary(qtp.qeye(2),'right'),qubit_to_2qutrit_unitary(qtp.sigmax(),'right'),
                     qubit_to_2qutrit_unitary(qtp.sigmay(),'right'),qubit_to_2qutrit_unitary(qtp.sigmaz(),'right')]

    diag=[]
    paulis_label=['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ']

    for pauli_1 in pauli_list:
        for pauli_2 in pauli_list:
            pauli=pauli_1 * pauli_2
            pauli_vec=qtp.operator_to_vector(pauli)
            diag_elem=1/(n_levels_q0*n_levels_q1) * (pauli_vec.dag()*qtp.to_super(U_target)*U*pauli_vec).data[0,0]
            diag.append(np.real(diag_elem))
    print(diag)
    czf.plot(x_plot_vec=[paulis_label],
                  y_plot_vec=[diag],
                  title='Diagonal of the Pauli transfer matrix',
                  xlabel='Pauli index',ylabel='Value')


########################################################################
# Study of leakage
########################################################################


def population_transfer(U_superop,state_in,state_out):
    return np.abs((state_out.dag()*U_superop*state_in).data[0,0])


def test_population_transfer(pop1,pop2):
    # should give back always 1
    tot=pop1+pop2
    print(tot)


def average_population_transfer_subspace_to_subspace(U_superop,states_in,states_out):
    # computes population that goes from input to output state,
    # or average population going from one subspace to another.
    # Input: superoperator U_superop
    #        list of 2-element lists as states_in and states_out, or keywords specified below

    if states_in == 'compsub':
        states_in = list_of_vector_indeces('compsub')
    if states_out == 'leaksub':
        states_out = list_of_vector_indeces('leaksub')
    if states_out == 'all':
        states_out = list_of_vector_indeces('all')

    sump=0
    for indeces_list_in in states_in:
        state_in = basis_state(indeces_list_in[0],indeces_list_in[1])
        for indeces_list_out in states_out:
            state_out = basis_state(indeces_list_out[0],indeces_list_out[1])

            sump += population_transfer(U_superop,state_in,state_out)

    sump/=len(states_in)

    return sump


def nullify_coherence(U_temp,state_A,state_B):
    # U_temp: superop
    # state_X: list e.g. [0,1]
    # This function sets to 0 the coherence between state_A and state_B in the output density matrix (whatever the input)
    # by setting to 0 the appropriate matrix element in the superoperator.

    for x_prime in range(0,n_levels_q1):
        for y_prime in range(0,n_levels_q0):
            for x in range(0,n_levels_q1):
                for y in range(0,n_levels_q0):

                    U_temp[index_in_vector_of_dm_matrix_element(state_A,state_B),index_in_vector_of_dm_matrix_element([x,y],[x_prime,y_prime])]=0
                    U_temp[index_in_vector_of_dm_matrix_element(state_B,state_A),index_in_vector_of_dm_matrix_element([x,y],[x_prime,y_prime])]=0

    return U_temp















