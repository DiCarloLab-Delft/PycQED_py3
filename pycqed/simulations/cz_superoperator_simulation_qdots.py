"""
17 July 2018
Started working with qdots

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



# eV_to_Hz = 1/4.1357e-15

# w_q1 = 18.4e9 * 2*np.pi
# w_q2 = 19.7e9 * 2*np.pi
# U_q1 = 3.5e-3 * eV_to_Hz * 2*np.pi      # meV converted to Hz
# U_q2 = 3.5e-3 * eV_to_Hz * 2*np.pi      # meV converted to Hz
# t_hopping = 210e6 * 2*np.pi


# # caracteristic timescales for jump operators
# T1_q1=50e-3
# T1_q2=3.7e-3
# T2_q1_hahn_sweetspot = 19e-6
# T2_q2_hahn_sweetspot = 7e-6



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
	#plt.ylim(-25,75)
	plt.show()



# Hamiltonian
def coupled_qdots_hamiltonian(w_q1, w_q2, U_q1, U_q2, t_hopping):
    """
    Hamiltonian of two coupled quantum dots.
    The 4 qubit levels correspond to the (1,1) charge states.
    Then there are the single states (0,2) and (2,0) involved.
    We artificially use here 3 extra states to keep the structure of the code that was created for two qutrits
           (dimension of the Hilbert space = 9).
    The extra levels do not couple to the other ones.

    """
    beta = (w_q1+w_q2)/2
    Delta_v = (w_q1-w_q2)/2

    crazy=beta*10000

    # the ordering of basis states below is 00,01,02,10,11,12,20,21,22
    H_0 = qtp.Qobj([[-beta, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, -Delta_v, t_hopping, 0, 0, 0, t_hopping, 0, 0],
                     [0, t_hopping, U_q2, -t_hopping, 0, 0, 0, 0, 0],
                     [0, 0, -t_hopping, Delta_v, 0, 0, -t_hopping, 0, 0],
                     [0, 0, 0, 0, beta, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, crazy, 0, 0, 0],
                     [0, t_hopping, 0, -t_hopping, 0, 0, U_q1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, crazy, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, crazy]],
                    type='oper',
                    dims=[[3, 3], [3, 3]])

    return H_0


# H_0 = coupled_qdots_hamiltonian(w_q1=w_q1, w_q2=w_q2, U_q1=U_q1, U_q2=U_q2, t_hopping=t_hopping)


H_c =      qtp.Qobj([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, +1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    type='oper',
                    dims=[[3, 3], [3, 3]])


# BENCHMARK OF THE SPECTRUM OF H_0

def spectrum_of_interest(H_0,controlparameter,desiredlevels):
	eigenvalues=list()
	for i in range(len(desiredlevels)):
		eigenvalues.append(list())
	for eps in controlparameter:
		H=H_0+eps*H_c
		eigs=H.eigenenergies()
		for i in desiredlevels:
			eigenvalues[i].append(eigs[i]/(2*np.pi)/1e9)

	return eigenvalues

# controlparameter=np.linspace(-4e-3*eV_to_Hz*2*np.pi,4e-3*eV_to_Hz*2*np.pi,1001)
# desiredlevels=np.arange(6)
# eigenvalues=spectrum_of_interest(H_0,controlparameter,desiredlevels)

# plot(x_plot_vec=[controlparameter/(2*np.pi)/eV_to_Hz*1e3],y_plot_vec=eigenvalues,title='First 6 energy levels for two qdots',
#   	xlabel='Detuning epsilon (meV)',ylabel='Frequency (GHz)',legend_labels=desiredlevels)




# BENCHMARK FORMULAS TO COMPUTE THE TIME TO DO A CZ

def gap_and_eigenvectors(H_0,Omega):
    H=H_0+Omega*H_c
    eigs,eigvectors=H.eigenstates()
    E_00=eigs[0]
    E_10=eigs[1]
    E_01=eigs[2]
    E_11=eigs[3]
    #E_02=eigs[4]
    gap=np.abs(E_11-E_01-E_10+E_00)
    return gap

def compute_cz_time(H_0,epsilon):
	gap=gap_and_eigenvectors(H_0,epsilon)
	return np.pi/gap

# cz_time=list()
# epsilon_range=np.linspace(3.0e-3*eV_to_Hz*2*np.pi,3.4e-3*eV_to_Hz*2*np.pi)
# for epsilon in epsilon_range:
# 	cz_time.append(compute_cz_time(H_0,epsilon)*1e9)
# #print(cz_time[0],cz_time[-1])

# plot(x_plot_vec=[epsilon_range/(2*np.pi)/eV_to_Hz*1e3],y_plot_vec=[cz_time],title='Time to do a CZ',
#    	xlabel='Detuning epsilon (meV)',ylabel='Time (ns)')




# target in the case with no noise
# note that the Hilbert space is H_q1 /otimes H_q2 modulo the fact that we don't really have qutrits
# the ordering of basis states below is 00,01,02,10,11,12,20,21,22
U_target = qtp.Qobj([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0, 0, 0, 0],      # - probaby not correct but see below
                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, -1, 0, 0, 0, 0],      # only this minus matters for av_gatefid_compsubspace_pc
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
rho_{xy,x'y'}=operator_to_vector(rho)[3*x+y+27*x'+9*y']
where xy is the row and x'y' is the column
'''


def jump_operators(T1_q0,T1_q1,Tphi_q0_ket0toket0,Tphi_q0_ket1toket1,Tphi_q0_ket2toket2,Tphi_q1_ket0toket0,Tphi_q1_ket1toket1,
					Tphi_q0_sigmaZ_01,Tphi_q0_sigmaZ_12,Tphi_q0_sigmaZ_02,Tphi_q1_sigmaZ_01,Tphi_q1_sigmaZ_12,Tphi_q1_sigmaZ_02):
	b = qtp.tensor(qtp.destroy(3), qtp.qeye(3))  # LSB is static qubit
	a = qtp.tensor(qtp.qeye(3), qtp.destroy(3))
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



tlist = np.arange(0, 240e-9, 1/2.4e9)


def time_evolution_squarepulse(H,c_ops,time,initial_propagator=1):
    '''scalefactor=1e6
    tlist_sim=tlist_sim*scalefactor
    eps_vec=eps_vec/scalefactor
    H_0=H_0/scalefactor
    if c_ops!=[]:       # c_ops is a list of either operators or lists where the first element is
                                    # an operator and the second one is a list of the (time-dependent) coefficients
        for c in range(len(c_ops)):
            if isinstance(c_ops[c],list):
                c_ops[c][1]=c_ops[c][1]/np.sqrt(scalefactor)
            else:
                c_ops[c]=c_ops[c]/np.sqrt(scalefactor)'''

    exp_L_total=initial_propagator
    '''
    if isinstance(tlist_sim,list):
        length_tlist=len(tlist_sim)
    else:
    	length_tlist=np.size(tlist_sim)'''
    c_ops_temp=list()
    if c_ops != []:
        for c in range(len(c_ops)):
            if isinstance(c_ops[c],list):
               c_ops_temp.append(c_ops[c][0]*c_ops[c][1][i])
            else:
               c_ops_temp.append(c_ops[c])
        liouville_exp_t=(qtp.liouvillian(H,c_ops_temp)*time).expm()
    else:
    	liouville_exp_t=(-1j*H*time).expm()
    exp_L_total=liouville_exp_t*exp_L_total

    return exp_L_total


def matrix_change_of_variables(H_0):
    eigs,eigvectors=H_0.eigenstates()

    eigvectors_ordered_according2basis = []
    eigvectors_ordered_according2basis.append(eigvectors[0].full())   # 00 state
    eigvectors_ordered_according2basis.append(eigvectors[2].full())   # 01 state
    eigvectors_ordered_according2basis.append(eigvectors[4].full())   # 02 state
    eigvectors_ordered_according2basis.append(eigvectors[1].full())   # 10 state
    eigvectors_ordered_according2basis.append(eigvectors[3].full())   # 11 state
    eigvectors_ordered_according2basis.append(eigvectors[7].full())   # 12 state
    eigvectors_ordered_according2basis.append(eigvectors[5].full())   # 20 state
    eigvectors_ordered_according2basis.append(eigvectors[6].full())   # 21 state
    eigvectors_ordered_according2basis.append(eigvectors[8].full())   # 22 state

    S=np.hstack(eigvectors_ordered_according2basis)
    return S


def simulate_quantities_of_interest_superoperator(H_0, tlist, c_ops, eps_vec,
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

    scalefactor=1e6  # otherwise qtp.propagator in parallel mode doesn't work
    # time is multiplied by scalefactor and frequency is divided by it
    tlist=tlist*scalefactor
    eps_vec=eps_vec/scalefactor
    sim_step=sim_step*scalefactor
    H_0=H_0/scalefactor
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
        H=H_0+eps_vec[i]*H_c
        c_ops_temp=[]
        for c in range(len(c_ops)):
            if isinstance(c_ops[c],list):
                c_ops_temp.append(c_ops[c][0]*c_ops[c][1][i])
            else:
                c_ops_temp.append(c_ops[c])
        liouville_exp_t=(qtp.liouvillian(H,c_ops_temp)*sim_step).expm()
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


def simulate_quantities_of_interest_superoperator2(U_final):
    """
    Calculates the quantities of interest from the propagator U_final

    Args:
        U_final = propagator (either unitary or superoperator)

    Returns
        phi_cond (float):   conditional phase (deg)
        L1      (float):    leakage
        L2      (float):    seepage
        avgatefid (float):  average gate fidelity in full space
        avgatefid_compsubspace (float):  average gate fidelity only in the computational subspace

    """

    phases = phases_from_superoperator(U_final)
    phi_cond = phases[-1]
    L1 = leakage_from_superoperator(U_final)
    L2 = seepage_from_superoperator(U_final)
    avgatefid = pro_avfid_superoperator_phasecorrected(U_final,phases)
    avgatefid_compsubspace = pro_avfid_superoperator_compsubspace_phasecorrected(U_final,L1,phases)     # leakage has to be taken into account, see Woods & Gambetta

    return {'phi_cond': phi_cond, 'L1': L1, 'L2': L2, 'avgatefid_pc': avgatefid, 'avgatefid_compsubspace_pc': avgatefid_compsubspace}





class CZ_trajectory_superoperator(det.Soft_Detector):
    def __init__(self, H_0, fluxlutman, noise_parameters_CZ):
        """
        Detector for simulating a CZ trajectory.
        Args:
            fluxlutman (instr): an instrument that contains the parameters
                required to generate the waveform for the trajectory.
        """
        super().__init__()
        self.value_names = ['Cost func', 'Cond phase', 'L1', 'L2', 'avgatefid_pc', 'avgatefid_compsubspace_pc']
        self.value_units = ['a.u.', 'deg', '%', '%', '%', '%']
        self.fluxlutman = fluxlutman
        self.H_0 = H_0
        self.noise_parameters_CZ = noise_parameters_CZ

    def acquire_data_point(self, **kw):
        '''
        tlist = (np.arange(0, self.fluxlutman.cz_length(),
                           1/self.fluxlutman.sampling_rate()))
        if not self.fluxlutman.czd_double_sided():
            f_pulse = wf.martinis_flux_pulse(
                length=self.fluxlutman.cz_length(),
                lambda_2=self.fluxlutman.cz_lambda_2(),
                lambda_3=self.fluxlutman.cz_lambda_3(),
                theta_f=self.fluxlutman.cz_theta_f(),
                f_01_max=self.fluxlutman.cz_freq_01_max(),
                J2=self.fluxlutman.cz_J2(),
                f_interaction=self.fluxlutman.cz_freq_interaction(),
                sampling_rate=self.fluxlutman.sampling_rate(),
                return_unit='f01')
        else:
            f_pulse = self.get_f_pulse_double_sided()

        # extract base frequency from the Hamiltonian
        w_q0 = np.real(self.H_0[1,1])
        eps_vec = f_pulse - w_q0
        '''



        T1_q0 = self.noise_parameters_CZ.T1_q0()
        T1_q1 = self.noise_parameters_CZ.T1_q1()
        T2_q0_sweetspot = self.noise_parameters_CZ.T2_q0_sweetspot()
        #T2_q0_interaction_point = self.noise_parameters_CZ.T2_q0_interaction_point()
        T2_q1 = self.noise_parameters_CZ.T2_q1()

        def Tphi_from_T1andT2(T1,T2):
            return 1/(-1/(2*T1)+1/T2)

        if T2_q0_sweetspot != 0:
            Tphi01_q0_sweetspot=Tphi_from_T1andT2(T1_q0,T2_q0_sweetspot)
        else:
            Tphi01_q0_sweetspot=0
        # if T2_q0_interaction_point != 0:
        #     Tphi01_q0_interaction_point=Tphi_from_T1andT2(T1_q0,T2_q0_interaction_point)
        # else:
        #     Tphi01_q0_interaction_point=0
        # Tphi01=Tphi12=2*Tphi02
        if T2_q1 != 0:
            Tphi01_q1 = Tphi_from_T1andT2(T1_q1,T2_q1)
        else:
            Tphi01_q1=0

        #collapse operators are given in the basis of H_0
        c_ops=jump_operators(T1_q0,T1_q1,0,0,0,0,0,
                    Tphi01_q0_sweetspot,0,Tphi01_q0_sweetspot/2,Tphi01_q1,0,Tphi01_q1/2)




        epsilon=self.noise_parameters_CZ.detuning() * 1e-3*eV_to_Hz*2*np.pi
        H=self.H_0+epsilon*H_c

        #We transform H in the basis of H_0 (collapse operators already expressed in that bais),
        #so that the time evolution is already expressed in the correct basis
        S = qtp.Qobj(matrix_change_of_variables(self.H_0),dims=[[3, 3], [3, 3]])
        H=S*H*S.dag()

        cz_time_ideal=compute_cz_time(self.H_0,epsilon)                           #time to acquire a 180 degrees conditional phase
        time=cz_time_ideal+self.noise_parameters_CZ.cz_time_offset() * 1e-9
        
        U_final=time_evolution_squarepulse(H,c_ops,time,initial_propagator=1)
        qoi = simulate_quantities_of_interest_superoperator2(U_final)


        cost_func_val = -np.log10(1-qoi['avgatefid_compsubspace_pc'])   # new cost function: infidelity
        #np.abs(qoi['phi_cond']-180) + qoi['L1']*100 * 5
        return cost_func_val, qoi['phi_cond'], qoi['L1']*100, qoi['L2']*100, qoi['avgatefid_pc']*100, qoi['avgatefid_compsubspace_pc']*100

    '''def get_f_pulse_double_sided(self):
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

        # N.B. No amp scaling and offset present
        f_pulse = np.concatenate([half_CZ_A, half_CZ_B])
        return f_pulse'''
