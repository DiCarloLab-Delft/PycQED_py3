"""
April 2018
Simulates the trajectory implementing a CZ gate.

June 2018
Included noise in the simulation.

July 2018
Added distortions to simulation.
"""
import time
import numpy as np
import qutip as qtp
from pycqed.measurement import detector_functions as det
from scipy.interpolate import interp1d
from pycqed.measurement.waveform_control_CC import waveform as wf
import scipy
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.inf)



# operators
b = qtp.tensor(qtp.destroy(3), qtp.qeye(3))  # LSB is static qubit
a = qtp.tensor(qtp.qeye(3), qtp.destroy(3))
n_q0 = a.dag() * a
n_q1 = b.dag() * b

H_coupling = (a.dag() + a) * (b + b.dag())
H_c = n_q0


scalefactor=1e6


'''
alpha_q0 = -285e6 * 2*np.pi
alpha_q1 = -310e6 * 2*np.pi
w_q0 = 5.11e9 * 2*np.pi  # Higher frequency qubit (fluxing) qubit
w_q1 = 4.10e9 * 2*np.pi  # Lower frequency

J = 2.9e6 * 2 * np.pi  # coupling strength

# caracteristic timescales for jump operators
T1_q0=34e-6
T1_q1=42e-6
Tphi_q0_ket0toket0=0    # here useless parameters
Tphi_q0_ket1toket1=0
Tphi_q0_ket2toket2=0
Tphi_q1_ket0toket0=0
Tphi_q1_ket1toket1=0
T2_q0=23e-6   # these two are the coherence times for q0 and q1 as qubits
T2_q1=23e-6
Tphi_q0_sigmaZ_01=1/(-1/(2*T1_q0)+1/T2_q0)     # extracting Tphi which is not the Tphi above
Tphi_q0_sigmaZ_12=Tphi_q0_sigmaZ_01           # we will assume for the moment that the pure decoherence
                                              # is caused by wiggles in the frequency, which cause
                                              # a fluctuation half as large for 02 wrt 01 and 12
                                              # (ignoring the anharmonicity)
Tphi_q0_sigmaZ_02=Tphi_q0_sigmaZ_01/2
Tphi_q1_sigmaZ_01=1/(-1/(2*T1_q1)+1/T2_q1)
'''



# Hamiltonian
def coupled_transmons_hamiltonian(w_q0, w_q1, alpha_q0, alpha_q1, J, w_bus):
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
    eps=0
    delta_q1=w_q1-w_bus
    delta_q0_interactionpoint=(w_q1-alpha_q0)-w_bus
    delta_q0=(w_q0+eps)-w_bus

    J_new = J / ((delta_q1+delta_q0_interactionpoint)/(delta_q1*delta_q0_interactionpoint)) * (delta_q1+delta_q0)/(delta_q1*delta_q0)

    H_0 = w_q0 * n_q0 + w_q1 * n_q1 +  \
        1/2*alpha_q0*(a.dag()*a.dag()*a*a) + 1/2*alpha_q1*(b.dag()*b.dag()*b*b) +\
        J_new * (a.dag() + a) * (b + b.dag())
    return H_0


def hamiltonian_timedependent(H_0,eps,w_bus):
	w_q0=np.real(H_0[1,1])
	w_q1=np.real(H_0[3,3])
	alpha_q0=np.real(H_0[2,2])-2*w_q0
	J=np.real(H_0[1,3])

	delta_q1=w_q1-w_bus
	delta_q0_sweetspot=(w_q0)-w_bus
	delta_q0=(w_q0+eps)-w_bus

	J_new = J / ((delta_q1+delta_q0_sweetspot)/(delta_q1*delta_q0_sweetspot)) * (delta_q1+delta_q0)/(delta_q1*delta_q0)

	return H_0+eps*H_c+(J_new-J)*H_coupling


#H_0 = coupled_transmons_hamiltonian(w_q0=w_q0, w_q1=w_q1, alpha_q0=alpha_q0,alpha_q1=alpha_q1,J=J)


# target in the case with no noise
# note that the Hilbert space is H_q1 /otimes H_q0
# so the ordering of basis states below is 00,01,02,10,11,12,20,21,22
U_target = qtp.Qobj([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, -1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1]],
                    type='oper',
                    dims=[[3, 3], [3, 3]])
#U_target._type = 'oper'
U_target_diffdims = qtp.Qobj([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, -1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1]],
                    type='oper',
                    dims=[[9], [9]])     # otherwise average_gate_fidelity doesn't work

# if there is noise the target is the corresponding superoperator
U_super_target = qtp.to_super(U_target)

'''
remember that qutip uses the Liouville (matrix) representation for superoperators,
with column stacking.
This means that 
rho_{xy,x'y'}=rho[3*x+y,3*x'+y']
rho_{xy,x'y'}=operator_to_vector(rho)[3*x+y+27*x'+9*y']				VERIFY
where xy is the row and x'y' is the column
'''


def plot(x_plot_vec,y_plot_vec,title='No title',xlabel='No xlabel',ylabel='No ylabel',legend_labels=list(),yscale='linear'):

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



def jump_operators(T1_q0,T1_q1,Tphi_q0_ket0toket0,Tphi_q0_ket1toket1,Tphi_q0_ket2toket2,Tphi_q1_ket0toket0,Tphi_q1_ket1toket1,
					Tphi_q0_sigmaZ_01,Tphi_q0_sigmaZ_12,Tphi_q0_sigmaZ_02,Tphi_q1_sigmaZ_01,Tphi_q1_sigmaZ_12,Tphi_q1_sigmaZ_02):
    # time independent case
	c_ops=[]
	if T1_q0 != 0:
		c_ops.append(np.sqrt(1/T1_q0)*a)
	if T1_q1 != 0:
		c_ops.append(np.sqrt(1/T1_q1)*b)
	if Tphi_q0_ket0toket0 != 0:
		collapse=qtp.tensor(qtp.qeye(3),qtp.ket2dm(qtp.basis(3,0)))
		c_ops.append(np.sqrt(1/Tphi_q0_ket0toket0)*collapse)
	if Tphi_q0_ket1toket1 != 0:
		collapse=qtp.tensor(qtp.qeye(3),qtp.ket2dm(qtp.basis(3,1)))
		c_ops.append(np.sqrt(1/Tphi_q0_ket1toket1)*collapse)
	if Tphi_q0_ket2toket2 != 0:
		collapse=qtp.tensor(qtp.qeye(3),qtp.ket2dm(qtp.basis(3,2)))
		c_ops.append(np.sqrt(1/Tphi_q0_ket2toket2)*collapse)
	if Tphi_q1_ket0toket0 != 0:
		collapse=qtp.tensor(qtp.ket2dm(qtp.basis(3,0)),qtp.qeye(3))
		c_ops.append(np.sqrt(1/Tphi_q1_ket0toket0)*collapse)
	if Tphi_q1_ket1toket1 != 0:
		collapse=qtp.tensor(qtp.ket2dm(qtp.basis(3,1)),qtp.qeye(3))
		c_ops.append(np.sqrt(1/Tphi_q1_ket1toket1)*collapse)
	if Tphi_q0_sigmaZ_01 != 0:
		sigmaZinqutrit = qtp.Qobj([[1,0,0],
									[0,-1,0],
									[0,0,0]])
		collapse=qtp.tensor(qtp.qeye(3),sigmaZinqutrit)
		c_ops.append(np.sqrt(1/(2*Tphi_q0_sigmaZ_01))*collapse)
	if Tphi_q0_sigmaZ_12 != 0:
		sigmaZinqutrit = qtp.Qobj([[0,0,0],
									[0,1,0],
									[0,0,-1]])
		collapse=qtp.tensor(qtp.qeye(3),sigmaZinqutrit)
		c_ops.append(np.sqrt(1/(2*Tphi_q0_sigmaZ_12))*collapse)
	if Tphi_q0_sigmaZ_02 != 0:
		sigmaZinqutrit = qtp.Qobj([[1,0,0],
									[0,0,0],
									[0,0,-1]])
		collapse=qtp.tensor(qtp.qeye(3),sigmaZinqutrit)
		c_ops.append(np.sqrt(1/(2*Tphi_q0_sigmaZ_02))*collapse)
	if Tphi_q1_sigmaZ_01 != 0:
		sigmaZinqutrit = qtp.Qobj([[1,0,0],
									[0,-1,0],
									[0,0,0]])
		collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(3))
		c_ops.append(np.sqrt(1/(2*Tphi_q1_sigmaZ_01))*collapse)
	if Tphi_q1_sigmaZ_12 != 0:
		sigmaZinqutrit = qtp.Qobj([[0,0,0],
									[0,1,0],
									[0,0,-1]])
		collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(3))
		c_ops.append(np.sqrt(1/(2*Tphi_q1_sigmaZ_12))*collapse)
	if Tphi_q1_sigmaZ_02 != 0:
		sigmaZinqutrit = qtp.Qobj([[1,0,0],
									[0,0,0],
									[0,0,-1]])
		collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(3))
		c_ops.append(np.sqrt(1/(2*Tphi_q1_sigmaZ_02))*collapse)
	return c_ops


#c_ops=jump_operators(T1_q0,T1_q1,Tphi_q0_ket0toket0,Tphi_q0_ket1toket1,Tphi_q0_ket2toket2,Tphi_q1_ket0toket0,Tphi_q1_ket1toket1,
#					 Tphi_q0_sigmaZ_01,Tphi_q0_sigmaZ_12,Tphi_q0_sigmaZ_02,Tphi_q1_sigmaZ_01)


def c_ops_interpolating(T1_q0,T1_q1,Tphi01_q0_vec,Tphi01_q1):
    # case where the pure decoherence for qubit q0 is time dependent, or better pulse-amplitude dependent
    c_ops=[]

    if T1_q0 != 0:
        c_ops.append(np.sqrt(1/T1_q0)*a)

    if T1_q1 != 0:
        c_ops.append(np.sqrt(1/T1_q1)*b)

    if Tphi01_q1 != 0:                                 # we automatically put also the decoherence for 12 and 02
        sigmaZinqutrit = qtp.Qobj([[1,0,0],
                                    [0,-1,0],
                                    [0,0,0]])
        collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(3))
        c_ops.append(collapse*np.sqrt(1/(2*Tphi01_q1)))

        Tphi12_q1=Tphi01_q1
        sigmaZinqutrit = qtp.Qobj([[0,0,0],
                                    [0,1,0],
                                    [0,0,-1]])
        collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(3))
        c_ops.append(collapse*np.sqrt(1/(2*Tphi12_q1)))

        Tphi02_q1=Tphi01_q1/2
        sigmaZinqutrit = qtp.Qobj([[1,0,0],
                                    [0,0,0],
                                    [0,0,-1]])
        collapse=qtp.tensor(sigmaZinqutrit,qtp.qeye(3))
        c_ops.append(collapse*np.sqrt(1/(2*Tphi02_q1)))

    if Tphi01_q0_vec != []:                                 # we automatically put also the decoherence for 12 and 02
        sigmaZinqutrit = qtp.Qobj([[1,0,0],
                                    [0,-1,0],
                                    [0,0,0]])
        collapse=qtp.tensor(qtp.qeye(3),sigmaZinqutrit)
        c_ops.append([collapse,np.sqrt(1/(2*Tphi01_q0_vec))])

        Tphi12_q0_vec=Tphi01_q0_vec
        sigmaZinqutrit = qtp.Qobj([[0,0,0],
                                    [0,1,0],
                                    [0,0,-1]])
        collapse=qtp.tensor(qtp.qeye(3),sigmaZinqutrit)
        c_ops.append([collapse,np.sqrt(1/(2*Tphi12_q0_vec))])

        Tphi02_q0_vec=Tphi01_q0_vec/2
        sigmaZinqutrit = qtp.Qobj([[1,0,0],
                                    [0,0,0],
                                    [0,0,-1]])
        collapse=qtp.tensor(qtp.qeye(3),sigmaZinqutrit)
        c_ops.append([collapse,np.sqrt(1/(2*Tphi02_q0_vec))])

    return c_ops




def rotating_frame_transformation(U, t: float,
                                  w_q0: float=0, w_q1: float =0):
    """
    Transforms the frame of the unitary according to
        U' = U_{RF}*U*U_{RF}^dag
    with
        U_{RF} = e^{-i w_q0 a^dag a t } otimes e^{-i w_q1 b^dag b t }

    Args:
        U (QObj): Unitary to be transformed
        t (float): time at which to transform
        w_q0 (float): freq of frame for q0
        w_q1 (float): freq of frame for q1

    """
    U_RF = (1j*w_q0*n_q0*t).expm() * (1j*w_q1*n_q1*t).expm()

    U_prime = U_RF * U  
    """ U_RF only on one side because that's the operator that
    satisfies the Schroedinger equation in the interaction picture.
    Anyway we won't use this function.
    In case we would need to rotate in the new picture the jump operators as well !
    """
    return U_prime


def phases_from_superoperator(U):
    """
    Returns the phases from the unitary or superoperator U
    """
    if U.type=='oper':
        phi_00 = np.rad2deg(np.angle(U[0, 0]))  # expected to equal 0 because of our
        # choice for the energy, not because of rotating frame
        phi_01 = np.rad2deg(np.angle(U[1, 1]))
        phi_10 = np.rad2deg(np.angle(U[3, 3]))
        phi_11 = np.rad2deg(np.angle(U[4, 4]))
        phi_02 = np.rad2deg(np.angle(U[2, 2]))  # used only for avgatefid_superoperator_phasecorrected
        phi_20 = np.rad2deg(np.angle(U[6, 6]))

        phi_cond = (phi_11 - phi_01 - phi_10 + phi_00) % 360
        # notice the + even if it is irrelevant

        return phi_00, phi_01, phi_10, phi_11, phi_02, phi_20, phi_cond
    elif U.type=='super':
        phi_00 = 0   # we set it to 0 arbitrarily but it is actually not knowable
        phi_01 = np.rad2deg(np.angle(U[1, 1]))   # actually phi_01-phi_00
        phi_10 = np.rad2deg(np.angle(U[3, 3]))
        phi_11 = np.rad2deg(np.angle(U[4, 4]))
        phi_02 = np.rad2deg(np.angle(U[2, 2]))
        phi_20 = np.rad2deg(np.angle(U[6, 6]))

        phi_cond = (phi_11 - phi_01 - phi_10 + phi_00) % 360  # still the right formula
        # independently from phi_00

        return phi_00, phi_01, phi_10, phi_11, phi_02, phi_20, phi_cond
    # !!! check that this is a good formula for superoperators: there is a lot of redundancy
    #     there if the evolution is unitary, but not necessarily if it's noisy!


def pro_avfid_superoperator_compsubspace(U,L1):
    """
    Average process (gate) fidelity in the qubit computational subspace for two qutrits
    Leakage has to be taken into account, see Woods & Gambetta
    """

    if U.type=='oper':
        inner = U.dag()*U_target
        part_idx = [0, 1, 3, 4]  # only computational subspace
        ptrace = 0
        for i in part_idx:
            ptrace += inner[i, i]
        dim = 4  # 2 qubits comp subspace       

        return np.real(((np.abs(ptrace))**2+dim*(1-L1))/(dim*(dim+1)))

    elif U.type=='super':
        kraus_form = qtp.to_kraus(U)
        dim=4 # 2 qubits in the computational subspace
        part_idx = [0, 1, 3, 4]  # only computational subspace
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
    The phase is corrected with Z rotations considering both transmons as qubits
    """

    Ucorrection = qtp.Qobj([[np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, np.exp(-1j*np.deg2rad(phases[1])), 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[3]-phases[-1])), 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[0])), 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[1])), 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[0]))]],
                    type='oper',
                    dims=[[3, 3], [3, 3]])

    if U.type=='oper':
        U=Ucorrection*U
        inner = U.dag()*U_target
        part_idx = [0, 1, 3, 4]  # only computational subspace
        ptrace = 0
        for i in part_idx:
            ptrace += inner[i, i]
        dim = 4  # 2 qubits comp subspace       

        return np.real(((np.abs(ptrace))**2+dim*(1-L1))/(dim*(dim+1)))

    elif U.type=='super':
        U=qtp.to_super(Ucorrection)*U
        kraus_form = qtp.to_kraus(U)
        dim=4 # 2 qubits in the computational subspace
        part_idx = [0, 1, 3, 4]  # only computational subspace
        psum=0
        for A_k in kraus_form:
        	ptrace = 0
        	inner = U_target_diffdims.dag()*A_k # otherwise dimension mismatch
        	for i in part_idx:
        		ptrace += inner[i, i]
        	psum += (np.abs(ptrace))**2

        return np.real((dim*(1-L1) + psum) / (dim*(dim + 1)))


def leakage_from_superoperator(U):
    if U.type=='oper':
        """
        Calculates leakage by summing over all in and output states in the
        computational subspace.
            L1 = 1- 1/2^{number computational qubits} sum_i sum_j abs(|<phi_i|U|phi_j>|)**2
        """
        sump = 0
        for i in range(4):
            for j in range(4):
                bra_i = qtp.tensor(qtp.ket([i//2], dim=[3]),
                                   qtp.ket([i % 2], dim=[3])).dag()
                ket_j = qtp.tensor(qtp.ket([j//2], dim=[3]),
                                   qtp.ket([j % 2], dim=[3]))
                p = np.abs((bra_i*U*ket_j).data[0, 0])**2
                sump += p
        sump /= 4  # divide by dimension of comp subspace
        L1 = 1-sump
        return L1
    elif U.type=='super':
        """
        Calculates leakage by summing over all in and output states in the
        computational subspace.
            L1 = 1- 1/2^{number computational qubits} sum_i sum_j Tr(rho_{x'y'}C_U(rho_{xy}))
            where C is U in the channel representation
        """
        sump = 0
        for i in range(4):
            for j in range(4):
                ket_i = qtp.tensor(qtp.ket([i//2], dim=[3]),
                                   qtp.ket([i % 2], dim=[3])) #notice it's a ket
                rho_i=qtp.operator_to_vector(qtp.ket2dm(ket_i))
                ket_j = qtp.tensor(qtp.ket([j//2], dim=[3]),
                                   qtp.ket([j % 2], dim=[3]))
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
    """
    if U.type=='oper':
        sump = 0
        for i_list in [[0,2],[1,2],[2,0],[2,1],[2,2]]:
            for j_list in [[0,2],[1,2],[2,0],[2,1],[2,2]]:
                bra_i = qtp.tensor(qtp.ket([i_list[0]], dim=[3]),
                                   qtp.ket([i_list[1]], dim=[3])).dag()
                ket_j = qtp.tensor(qtp.ket([j_list[0]], dim=[3]),
                                   qtp.ket([j_list[1]], dim=[3]))
                p = np.abs((bra_i*U*ket_j).data[0, 0])**2  # could be sped up
                sump += p
        sump /= 5  # divide by number of non-computational states
        L1 = 1-sump
        return L1
    elif U.type=='super':
        sump = 0
        for i_list in [[0,2],[1,2],[2,0],[2,1],[2,2]]:
            for j_list in [[0,2],[1,2],[2,0],[2,1],[2,2]]:
                ket_i = qtp.tensor(qtp.ket([i_list[0]], dim=[3]),
                                   qtp.ket([i_list[1]], dim=[3]))
                rho_i=qtp.operator_to_vector(qtp.ket2dm(ket_i))
                ket_j = qtp.tensor(qtp.ket([j_list[0]], dim=[3]),
                                   qtp.ket([j_list[1]], dim=[3]))
                rho_j=qtp.operator_to_vector(qtp.ket2dm(ket_j))
                p = (rho_i.dag()*U*rho_j).data[0, 0]
                sump += p
        sump /= 5  # divide by number of non-computational states
        sump=np.real(sump)
        L1 = 1-sump
        return L1



def pro_avfid_superoperator(U):
    """
    Average process (gate) fidelity in the whole space for two qutrits
    """
    if U.type=='oper':
        ptrace = np.abs((U.dag()*U_target).tr())**2
        dim = 9  # dimension of the whole space
        return np.real((ptrace+dim)/(dim*(dim+1)))

    elif U.type=='super':
        return np.real(qtp.average_gate_fidelity(U,target=U_target_diffdims))


def pro_avfid_superoperator_phasecorrected(U,phases):
    """
    Average process (gate) fidelity in the whole space for a qubit and qutrit
    Qubit Z rotation and qutrit "Z" rotations are applied, taking into account the anharmonicity as well
    """
    Ucorrection = qtp.Qobj([[np.exp(-1j*np.deg2rad(phases[0])), 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, np.exp(-1j*np.deg2rad(phases[1])), 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, np.exp(-1j*np.deg2rad(phases[4]-phases[-1])), 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, np.exp(-1j*np.deg2rad(phases[2])), 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[3]-phases[-1])), 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[4]-phases[-1]+phases[2]-phases[0])), 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[5])), 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[5]+phases[1]-phases[0])), 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, np.exp(-1j*np.deg2rad(phases[4]-phases[-1]+phases[5]-phases[0]))]],
                    type='oper',
                    dims=[[3, 3], [3, 3]])

    if U.type=='oper':
        U=Ucorrection*U
        ptrace = np.abs((U.dag()*U_target).tr())**2
        dim = 9  # dimension of the whole space
        return np.real((ptrace+dim)/(dim*(dim+1)))

    elif U.type=='super':
        U=qtp.to_super(Ucorrection)*U
        return np.real(qtp.average_gate_fidelity(U,target=U_target_diffdims))



#tlist = np.arange(0, 240e-9, 1/2.4e9)


def matrix_change_of_variables(H_0):
    eigs,eigvectors=H_0.eigenstates()

    eigvectors_ordered_according2basis = []
    eigvectors_ordered_according2basis.append(eigvectors[0].full())   # 00 state
    eigvectors_ordered_according2basis.append(eigvectors[2].full())   # 01 state
    eigvectors_ordered_according2basis.append(eigvectors[5].full())   # 02 state
    eigvectors_ordered_according2basis.append(eigvectors[1].full())   # 10 state
    eigvectors_ordered_according2basis.append(eigvectors[4].full())   # 11 state
    eigvectors_ordered_according2basis.append(eigvectors[7].full())   # 12 state
    eigvectors_ordered_according2basis.append(eigvectors[3].full())   # 20 state
    eigvectors_ordered_according2basis.append(eigvectors[6].full())   # 21 state
    eigvectors_ordered_according2basis.append(eigvectors[8].full())   # 22 state

    S=np.hstack(eigvectors_ordered_according2basis)
    return S


def simulate_quantities_of_interest_superoperator(H_0, tlist, c_ops, w_bus, eps_vec,
                                    sim_step,
                                    verbose: bool=True):
    """
    Calculates the quantities of interest from the propagator U

    Args:
        H_0 (Qobj): static hamiltonian, see "coupled_transmons_hamiltonian"
            for the expected form of the Hamiltonian.
        tlist (array): times in s, describes the x component of the
            trajectory to simulate
        c-ops (list of Qobj): list of jump operators, time-independent at the momennt
        eps_vec(array): detuning describes the y-component of the trajectory
            to simulate.

    Returns
        phi_cond (float):   conditional phase (deg)
        L1      (float):    leakage
        L2      (float):    seepage
        avgatefid (float):  average gate fidelity in full space
        avgatefid_compsubspace (float):  average gate fidelity only in the computational subspace

    """

    
    # time is multiplied by scalefactor and frequency is divided by it
    tlist=tlist*scalefactor
    eps_vec=eps_vec/scalefactor
    sim_step=sim_step*scalefactor
    H_0=H_0/scalefactor
    w_bus=w_bus/scalefactor
    if c_ops!=[]:       # c_ops is a list of either operators or lists where the first element is
                                    # an operator and the second one is a list of the (time-dependent) coefficients
        for c in range(len(c_ops)):
            if isinstance(c_ops[c],list):
                c_ops[c][1]=c_ops[c][1]/np.sqrt(scalefactor)
            else:
                c_ops[c]=c_ops[c]/np.sqrt(scalefactor)


    '''								# step of 1/sampling_rate=1/2.4e9=0.4 ns seems good by itself
    sim_step_new=sim_step*2
    
    eps_interp = interp1d(tlist, eps_vec, fill_value='extrapolate')
    tlist_new = (np.linspace(0, np.max(tlist), 576/2)) 
    eps_vec_new=eps_interp(tlist_new)
    
    c_ops_new=[]
    for c in range(len(c_ops)):
        if isinstance(c_ops[c],list):
            c_ops_interp=interp1d(tlist,c_ops[c][1], fill_value='extrapolate')
            c_ops_new.append([c_ops[c][0],c_ops_interp(tlist_new)])
        else:
            c_ops_new.append(c_ops[c])

    # function only exists to wrap
    #def eps_t(t, args=None):
    #    return eps_interp(t)
    print(len(eps_vec),len(eps_vec_new))


  			
    t0 = time.time()

    exp_L_total_new=1
    for i in range(len(tlist_new)):
        H=H_0+eps_vec_new[i]*H_c
        c_ops_temp=[]
        for c in range(len(c_ops_new)):
            if isinstance(c_ops_new[c],list):
                c_ops_temp.append(c_ops_new[c][0]*c_ops_new[c][1][i])
            else:
                c_ops_temp.append(c_ops_new[c])
        liouville_exp_t=(qtp.liouvillian(H,c_ops_temp)*sim_step_new).expm()
        exp_L_total_new=liouville_exp_t*exp_L_total_new

    #exp_L_oneway=(qtp.liouvillian(H_0,c_ops)*240e-3).expm()

    t1 = time.time()
    print('\n alternative propagator_new',t1-t0)
    '''


    t0 = time.time()

    exp_L_total=1
    for i in range(len(tlist)):
        H=hamiltonian_timedependent(H_0,eps_vec[i],w_bus)
        if c_ops != []:
            c_ops_temp=[]
            for c in range(len(c_ops)):
                if isinstance(c_ops[c],list):
                    c_ops_temp.append(c_ops[c][0]*c_ops[c][1][i])
                else:
                    c_ops_temp.append(c_ops[c])
            liouville_exp_t=(qtp.liouvillian(H,c_ops_temp)*sim_step).expm()
        else:
        	liouville_exp_t=(-1j*H*sim_step).expm()
        exp_L_total=liouville_exp_t*exp_L_total

    #exp_L_oneway=(qtp.liouvillian(H_0,c_ops)*240e-3).expm()

    t1 = time.time()
    print('\n alternative propagator',t1-t0)
    
    '''						# qutip propagator not used anymore because it takes too much time
    t0 = time.time()
    if c_ops==[]:
    	nstepsmax=1000
    else:
    	nstepsmax=100000
    H_t = [H_0, [H_c, eps_vec]]
    U_t = qtp.propagator(H_t, tlist, c_ops, parallel=True, options=qtp.Options(nsteps=nstepsmax))   # returns unitary 'oper' if c_ops=[], otherwise 'super'
    t1 = time.time()
    print('/n propagator',t1-t0)
    if verbose:
        print('simulation took {:.2f}s'.format(t1-t0))
    '''

    U_final = exp_L_total

    # We change the basis of U_final to the basis of eigenvectors of H_0
    # The columns of S are the eigenvectors of H_0, appropriately ordered
    S = qtp.Qobj(matrix_change_of_variables(H_0),dims=[[3, 3], [3, 3]])
    U_final = S*U_final*S.dag()

    phases = phases_from_superoperator(U_final)
    phi_cond = phases[-1]
    L1 = leakage_from_superoperator(U_final)
    L2 = seepage_from_superoperator(U_final)
    avgatefid = pro_avfid_superoperator_phasecorrected(U_final,phases)
    avgatefid_compsubspace = pro_avfid_superoperator_compsubspace_phasecorrected(U_final,L1,phases)     # leakage has to be taken into account, see Woods & Gambetta
    print('avgatefid_compsubspace',avgatefid_compsubspace)
    
    '''
    U_final = exp_L_total_new
    phases2 = phases_from_superoperator(U_final)
    phi_cond2 = phases2[-1]
    L12 = leakage_from_superoperator(U_final)
    L22 = seepage_from_superoperator(U_final)
    avgatefid2 = pro_avfid_superoperator_phasecorrected(U_final,phases2)
    avgatefid_compsubspace2 = pro_avfid_superoperator_compsubspace_phasecorrected(U_final,L12,phases2)


    print(phi_cond-phi_cond2,phi_cond)
    print(L1-L12,L1)
    print(L2-L22,L2)
    print(avgatefid-avgatefid2,avgatefid)
    print(avgatefid_compsubspace-avgatefid_compsubspace2,avgatefid_compsubspace)
    '''  
    

    return {'phi_cond': phi_cond, 'L1': L1, 'L2': L2, 'avgatefid_pc': avgatefid, 'avgatefid_compsubspace_pc': avgatefid_compsubspace}


def spectrum(H_0,eps_vec):
    eigenvalues=[[],[],[],[],[],[],[],[],[]]
    for Omega in eps_vec:
        H=H_0+Omega*H_c
        eigs=H.eigenenergies()
        for i in range(len(eigs)):
            eigenvalues[i].append(eigs[i])
    return eigenvalues

def fix_theta_f(lambda_3,theta_i):
    lambda_1target=1
    return (theta_i+2*(lambda_1target+lambda_3))*360/(2*np.pi)


class CZ_trajectory_superoperator(det.Soft_Detector):
    def __init__(self, H_0, fluxlutman, noise_parameters_CZ, fitted_stepresponse_ty):
        """
        Detector for simulating a CZ trajectory.
        Args:
            fluxlutman (instr): an instrument that contains the parameters
                required to generate the waveform for the trajectory.
            noise_parameters_CZ: instrument that contains the noise parameters
            fitted_stepresponse_ty: list of two elements, corresponding to the time t 
                                    and the step response in volts along the y axis
        """
        super().__init__()
        self.value_names = ['Cost func', 'Cond phase', 'L1', 'L2', 'avgatefid_pc', 'avgatefid_compsubspace_pc']
        self.value_units = ['a.u.', 'deg', '%', '%', '%', '%']
        self.fluxlutman = fluxlutman
        self.H_0 = H_0
        self.noise_parameters_CZ = noise_parameters_CZ
        self.fitted_stepresponse_ty=fitted_stepresponse_ty      # list of 2 elements: stepresponse (=y)
                                                                # as a function of time (=t)


    def acquire_data_point(self, **kw):

        '''# BENCHMARK FOR HOW THE COUPLING IMPACTS THE HAMILTONIAN PARAMETERS
        eigs,eigvectors = self.H_0.eigenstates()
        eigs=eigs/(2*np.pi)
        print('omegaA =',eigs[1])
        print('omegaB =',eigs[2])
        print(eigs[4]-eigs[1]-eigs[2])
        print('etaA =',eigs[3]-2*eigs[1])
        print('etaB =',eigs[5]-2*eigs[2])
        print(eigvectors[4],'\n fidelity with 1 /otimes 1=',np.abs(eigvectors[4].dag().overlap(qtp.basis(9,4)))**2)
        print(eigvectors[5],'\n fidelity with 0 /otimes 2=',np.abs(eigvectors[5].dag().overlap(qtp.basis(9,2)))**2)
        '''

        sim_step=1/self.fluxlutman.sampling_rate()
        subdivisions_of_simstep=4
        sim_step_new=sim_step/subdivisions_of_simstep      # waveform is generated according to sampling rate of AWG,
                                                 # but we can use a different step for simulating the time evolution
        tlist = (np.arange(0, self.fluxlutman.cz_length(),
                           sim_step))
        tlist_new = (np.arange(0, self.fluxlutman.cz_length(),
                           sim_step_new))
        
        theta_i = np.arctan(2*self.fluxlutman.cz_J2() / (self.fluxlutman.cz_freq_01_max() - self.fluxlutman.cz_freq_interaction()))
        theta_f=fix_theta_f(self.fluxlutman.cz_lambda_3(),theta_i)
        #theta_i=theta_i*360/(2*np.pi)

        if not self.fluxlutman.czd_double_sided():
            f_pulse = wf.martinis_flux_pulse(
                length=self.fluxlutman.cz_length(),
                lambda_2=self.fluxlutman.cz_lambda_2(),
                lambda_3=self.fluxlutman.cz_lambda_3(),
                theta_f=theta_f,#self.fluxlutman.cz_theta_f(),
                f_01_max=self.fluxlutman.cz_freq_01_max(),
                J2=self.fluxlutman.cz_J2(),
                f_interaction=self.fluxlutman.cz_freq_interaction(),
                sampling_rate=self.fluxlutman.sampling_rate(),
                return_unit='f01')    # return in terms of omega
            amp = self.fluxlutman.detuning_to_amp((self.fluxlutman.cz_freq_01_max() - f_pulse)/(2*np.pi))
                     # transform detuning frequency to (positive) amplitude
        else:
            f_pulse,amp = self.get_f_pulse_double_sided()

        # For better accuracy in simulations, redefine f_pulse and amp in trems of sim_step_new
        tlist_temp=np.concatenate((tlist,np.array([self.fluxlutman.cz_length()])))
        f_pulse_temp=np.concatenate((f_pulse,np.array([f_pulse[-1]])))
        amp_temp=np.concatenate((amp,np.array([amp[-1]])))
        f_pulse_interp=interp1d(tlist_temp,f_pulse_temp)
        amp_interp=interp1d(tlist_temp,amp_temp)
        f_pulse=f_pulse_interp(tlist_new)
        amp=amp_interp(tlist_new)

        # plot(x_plot_vec=[tlist_new*1e9],
        #           y_plot_vec=[f_pulse/(2*np.pi)/1e9],
        #           title='Freq. of fluxing qubit during pulse',
        #           xlabel='Time (ns)',ylabel='Freq. (GHz)',legend_labels=['omega_B(t)'])


        amp=amp*self.noise_parameters_CZ.voltage_scaling_factor()


        # extract base frequency from the Hamiltonian
        w_q0 = np.real(self.H_0[1,1])
        #w_q1=np.real(self.H_0[3,3])
        #alpha_q0=np.real(self.H_0[2,2])-2*w_q0

        eps_vec = f_pulse - w_q0
        #detuning = -eps_vec/(2*np.pi)     # we express detuning in terms of frequency

        '''#BENCHMARK TO CHECK HOW THE COUPLING VARIES AS A FUNCTION OF DETUNING
        J_new=list()
        for eps in eps_vec:
        	H=hamiltonian_timedependent(self.H_0,eps)
        	J_new.append(np.real(H[1,3]))
        plot(x_plot_vec=[tlist_new*1e9],
            	  y_plot_vec=[np.array(J_new)/(2*np.pi)/1e6],
            	  title='Coupling during pulse',
                  xlabel='Time (ns)',ylabel='J (MHz)',legend_labels=['J(t)'])'''


        '''   USELESS   ####### functions that were used to convert from detuning to voltage but now we use 
                       functions from fluxlutman which are the same as those used in the experiment
        def invert_parabola(polynomial_coefficients,y):    # useless
        	a=polynomial_coefficients[0]
        	b=polynomial_coefficients[1]
        	c=polynomial_coefficients[2]
        	return (-b+np.sqrt(b**2-4*a*(c-y)))/(2*a)

        voltage_frompoly = invert_parabola(self.fluxlutman.polycoeffs_freq_conv(),detuning)

        voltage_frompoly_interp = interp1d(tlist,voltage_frompoly)

        voltage_frompoly_convol = voltage_frompoly_interp(tlist_convol1)

        convolved_voltage=scipy.signal.convolve(voltage_frompoly_convol,impulse_response_convol)/sum(impulse_response_convol)
        
        convolved_detuning=give_parabola(self.fluxlutman.polycoeffs_freq_conv(),convolved_voltage)

        eps_vec_convolved=-convolved_detuning*(2*np.pi)
        eps_vec_convolved=eps_vec_convolved[0:np.size(tlist_convol1)]
        f_pulse_convolved=eps_vec_convolved+w_q0
        
        '''

        def give_parabola(polynomial_coefficients,x):
            a=polynomial_coefficients[0]
            b=polynomial_coefficients[1]
            c=polynomial_coefficients[2]
            return a*x**2+b*x+c


        if self.noise_parameters_CZ.distortions():
            impulse_response=np.gradient(self.fitted_stepresponse_ty[1])

            # plot(x_plot_vec=[self.fitted_stepresponse_ty[0]],y_plot_vec=[self.fitted_stepresponse_ty[1]],
            # 	  title='Step response',
            #       xlabel='Time (ns)')
            # plot(x_plot_vec=[self.fitted_stepresponse_ty[0]],y_plot_vec=[impulse_response],
            # 	  title='Impulse response',
            #       xlabel='Time (ns)')

            # use interpolation to be sure that amp and impulse_response have the same delta_t separating two values
            #amp_interp = interp1d(tlist,amp)      # amp is now managed already above
            impulse_response_interp = interp1d(self.fitted_stepresponse_ty[0],impulse_response)

            tlist_convol1 = tlist_new
            tlist_convol2 = np.arange(0, self.fitted_stepresponse_ty[0][-1],
                               sim_step_new)
            #amp_convol = amp_interp(tlist_convol1)
            impulse_response_convol = impulse_response_interp(tlist_convol2)

            # plot(x_plot_vec=[tlist_convol1*1e9],y_plot_vec=[amp_convol],
            # 	  title='Pulse in voltage, length=240ns',
            #       xlabel='Time (ns)',ylabel='Amplitude (V)')
            # plot(x_plot_vec=[tlist_convol*1e9],y_plot_vec=[impulse_response_convol],
            # 	  title='Impulse response',
            #       xlabel='Time (ns)')

            convolved_amp=scipy.signal.convolve(amp_convol,impulse_response_convol)/sum(impulse_response_convol)

            # plot(x_plot_vec=[tlist_convol1*1e9,np.arange(np.size(convolved_amp))*sim_step*1e9],
            # 	  y_plot_vec=[amp_convol, convolved_amp],
            # 	  title='Net-zero, Pulse_length=240ns',
            #       xlabel='Time (ns)',ylabel='Amplitude (V)',legend_labels=['Ideal','Distorted'])

            convolved_detuning_new=give_parabola(self.fluxlutman.polycoeffs_freq_conv(),convolved_amp)

            # plot(x_plot_vec=[tlist*1e9,np.arange(np.size(convolved_amp))*sim_step*1e9],
            # 	  y_plot_vec=[detuning/1e9, convolved_detuning_new/1e9],
            # 	  title='Net-zero, Pulse_length=240ns',
            #       xlabel='Time (ns)',ylabel='Detuning (GHz)',legend_labels=['Ideal','Distorted'])


            eps_vec_convolved_new=-convolved_detuning_new*(2*np.pi)
            eps_vec_convolved_new=eps_vec_convolved_new[0:np.size(tlist_convol1)]
            f_pulse_convolved_new=eps_vec_convolved_new+w_q0
        else:
            if self.noise_parameters_CZ.voltage_scaling_factor() == 1.0:
        	    eps_vec_convolved_new=eps_vec
        	    f_pulse_convolved_new=f_pulse
            else:
                detuning_new=give_parabola(self.fluxlutman.polycoeffs_freq_conv(),amp)
                eps_vec_convolved_new=-detuning_new*(2*np.pi)
                f_pulse_convolved_new=eps_vec_convolved_new+w_q0




        T1_q0 = self.noise_parameters_CZ.T1_q0()
        T1_q1 = self.noise_parameters_CZ.T1_q1()
        T2_q0_sweetspot = self.noise_parameters_CZ.T2_q0_sweetspot()
        T2_q0_interaction_point = self.noise_parameters_CZ.T2_q0_interaction_point()
        T2_q0_amplitude_dependent = self.noise_parameters_CZ.T2_q0_amplitude_dependent()
        T2_q1 = self.noise_parameters_CZ.T2_q1()

        def Tphi_from_T1andT2(T1,T2):
            return 1/(-1/(2*T1)+1/T2)

        if T2_q0_sweetspot != 0:
            Tphi01_q0_sweetspot=Tphi_from_T1andT2(T1_q0,T2_q0_sweetspot)
        else:
            Tphi01_q0_sweetspot=0
        if T2_q0_interaction_point != 0:
            Tphi01_q0_interaction_point=Tphi_from_T1andT2(T1_q0,T2_q0_interaction_point)
        else:
            Tphi01_q0_interaction_point=0
        # Tphi01=Tphi12=2*Tphi02
        if T2_q1 != 0:
            Tphi01_q1 = Tphi_from_T1andT2(T1_q1,T2_q1)
        else:
            Tphi01_q1=0



        if T2_q0_amplitude_dependent != np.array([-1]):     # CHANGE ITTTTT
            # something like: 
            # - map f_pulse to T2 via polycoefficients OR interpolate dataset of T2 vs frequency/detuning
            # - compute Tphi01 from T2

            c_ops = c_ops_interpolating(T1_q0,T1_q1,Tphi01_q0_vec,Tphi01_q1)

        else:
            def omega_prime(omega):                                   # derivative of f_pulse
                '''
                frequency is w = w_0 * cos(phi_e/2)    where phi_e is the external flux through the SQUID.
                So the derivative wrt phi_e is
                     w_prime = - w_0/2 sin(phi_e/2) = - w_0/2 * sqrt(1-cos(phi_e/2)**2) = - w_0/2 * sqrt(1-(w/w_0)**2)
                Note: no need to know what phi_e is.
                '''
                return np.abs((w_q0/2)*np.sqrt(1-(omega/w_q0)**2))    # we actually return the absolute value because it's the only one who matters later

            if Tphi01_q0_interaction_point != 0:       # mode where the pure dephazing is amplitude-dependent
                w_min = np.nanmin(f_pulse_convolved_new)        
                omega_prime_min = omega_prime(w_min)

                f_pulse_convolved_new=np.clip(f_pulse_convolved_new,0,w_q0)
                f_pulse_convolved_new_prime = omega_prime(f_pulse_convolved_new)
                Tphi01_q0_vec = Tphi01_q0_sweetspot - f_pulse_convolved_new_prime/omega_prime_min*(Tphi01_q0_sweetspot-Tphi01_q0_interaction_point)
                         # we interpolate Tphi from the sweetspot to the interaction point (=worst point in terms of Tphi)
                         # by weighting depending on the derivative of f_pulse compared to the derivative at the interaction point
                c_ops = c_ops_interpolating(T1_q0,T1_q1,Tphi01_q0_vec,Tphi01_q1)
            else:                                       # mode where the collapse operators are time-independent, and possibly are 0
                c_ops=jump_operators(T1_q0,T1_q1,0,0,0,0,0,
                        Tphi01_q0_sweetspot,Tphi01_q0_sweetspot,Tphi01_q0_sweetspot/2,Tphi01_q1,Tphi01_q1,Tphi01_q1/2)



        qoi = simulate_quantities_of_interest_superoperator(
            H_0=self.H_0,
            tlist=tlist_new, c_ops=c_ops, w_bus=self.noise_parameters_CZ.w_bus(), eps_vec=eps_vec_convolved_new,
            sim_step=sim_step_new, verbose=False)

        cost_func_val = -np.log10(1-qoi['avgatefid_compsubspace_pc'])   # new cost function: infidelity
        #np.abs(qoi['phi_cond']-180) + qoi['L1']*100 * 5
        return cost_func_val, qoi['phi_cond'], qoi['L1']*100, qoi['L2']*100, qoi['avgatefid_pc']*100, qoi['avgatefid_compsubspace_pc']*100

    def get_f_pulse_double_sided(self):

        half_CZ_A = wf.martinis_flux_pulse(
            length=self.fluxlutman.cz_length()*self.fluxlutman.czd_length_ratio(),
            lambda_2=self.fluxlutman.cz_lambda_2(),
            lambda_3=self.fluxlutman.cz_lambda_3(),
            theta_f=self.fluxlutman.cz_theta_f(),
            f_01_max=self.fluxlutman.cz_freq_01_max(),
            J2=self.fluxlutman.cz_J2(),
            # E_c=self.fluxlutman.cz_E_c(),
            f_interaction=self.fluxlutman.cz_freq_interaction(),
            sampling_rate=self.fluxlutman.sampling_rate(),
            return_unit='f01')
        half_amp_A = self.fluxlutman.detuning_to_amp(
        	                      (self.fluxlutman.cz_freq_01_max() - half_CZ_A)/(2*np.pi))
                                  # first half is mapped to positive voltage
                                  # NOTE: negative part of the flux arc is ignored

        # Generate the second CZ pulse. If the params are np.nan, default
        # to the main parameter
        if not np.isnan(self.fluxlutman.czd_theta_f()):
            d_theta_f = self.fluxlutman.czd_theta_f()
        else:
            d_theta_f = self.fluxlutman.cz_theta_f()

        if not np.isnan(self.fluxlutman.czd_lambda_2()):
            d_lambda_2 = self.fluxlutman.czd_lambda_2()
        else:
            d_lambda_2 = self.fluxlutman.cz_lambda_2()
        if not np.isnan(self.fluxlutman.czd_lambda_3()):
            d_lambda_3 = self.fluxlutman.czd_lambda_3()
        else:
            d_lambda_3 = self.fluxlutman.cz_lambda_3()

        half_CZ_B = wf.martinis_flux_pulse(
            length=self.fluxlutman.cz_length()*(1-self.fluxlutman.czd_length_ratio()),
            lambda_2=d_lambda_2,
            lambda_3=d_lambda_3,
            theta_f=d_theta_f,
            f_01_max=self.fluxlutman.cz_freq_01_max(),
            J2=self.fluxlutman.cz_J2(),
            f_interaction=self.fluxlutman.cz_freq_interaction(),
            sampling_rate=self.fluxlutman.sampling_rate(),
            return_unit='f01')

        half_amp_B = self.fluxlutman.detuning_to_amp(
                (self.fluxlutman.cz_freq_01_max() - half_CZ_B)/(2*np.pi), positive_branch=False)
                                  # second half is mapped to negative voltage
                                  # NOTE: negative part of the flux arc is ignored


        # N.B. No amp scaling and offset present
        f_pulse = np.concatenate([half_CZ_A, half_CZ_B])
        amp = np.concatenate([half_amp_A, half_amp_B])
        return f_pulse,amp
