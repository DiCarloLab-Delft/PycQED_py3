import sys  
#custom made toolboxes
sys.path.append('D:\\Repository\\PycQED_py3\\pycqed')


from analysis import tomography_toolbox as tomography
import os
import time
from imp import reload
from matplotlib import pyplot as plt
import numpy as np
from analysis import measurement_analysis as MA 
from analysis import ramiro_analysis as RA
from analysis import thresholding_toolbox as thresholding
from analysis import fitting_models as fit_mods
import lmfit
import scipy as scipy
try:
    import qutip as qt
except ImportError as e:
    logging.warning('Could not import qutip, tomo code will not work')
import itertools
from pycqed.analysis_v2 import pytomo as csdp_tomo

comp_projectors = [qt.ket2dm(qt.tensor(qt.basis(2,0), qt.basis(2,0))),
                  qt.ket2dm(qt.tensor(qt.basis(2,0), qt.basis(2,1))),
                  qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(2,0))),
                  qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(2,1)))]

class TomoAnalysis():

    """Performs state tomography based on an overcomplete set of measurements
     and calibration measurements. Uses qutip to calculate resulting basis states
     from applied rotations

    Uses binary counting as general guideline in ordering states. Calculates
     rotations by using the qutip library
    BEFORE YOU USE THIS SET THE CORRECT ORDER BY CHANGING
        'rotation_matrixes'
        'measurement_basis' + 'measurement_basis_labels'
        to values corresponding to your experiment
        and maybe 'readout_basis'
    """

    # The set of single qubit rotation matrixes used in the tomography
    # measurement (will be assumed to be used on all qubits)
    rotation_matrixes = [qt.identity(2), qt.sigmax(),
                         qt.rotation(
                             qt.sigmay(), np.pi / 2), qt.rotation(qt.sigmay(), -1*np.pi / 2),
                         qt.rotation(qt.sigmax(), np.pi / 2), qt.rotation(qt.sigmax(), -np.pi / 2)]
    measurement_operator_labels = ['I', 'X', 'y', '-y', 'x','-x']
    #MAKE SURE THE LABELS CORRESPOND TO THE ROTATION MATRIXES DEFINED ABOVE

    # The set of single qubit basis operators and labels (normalized)
    measurement_basis = [
        qt.identity(2) , qt.sigmaz() , qt.sigmax() , qt.sigmay() ]
    measurement_basis_labels = ['I', 'Z', 'X', 'Y']
    # The operators used in the readout basis on each qubit
    readout_basis = [qt.identity(2) , qt.sigmaz() ]

    def __init__(self, n_qubits=2, check_labels=False):
        """
        keyword arguments:
        measurements_cal --- Should be an array of length 2 ** n_qubits
        measurements_tomo --- Should be an array of length length(rotation_matrixes) ** n_qubits
        n_qubits --- default(2) the amount of qubits present in the expirement
        n_quadratures --- default(1(either I or Q)) The amount of complete measurement data sets. For example a combined IQ measurement has 2 measurement sets.
        tomo_vars  : since this tomo does not have access to the original data, the vars should be given by
                        tomo_var_i = 1 / N_i * np.var(M_i) where i stands for the data corresponding to rotation i. 

        """
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits

        # Generate the vectors of matrixes that correspond to all measurements,
        # readout bases and rotations
        self.basis_vector = self._calculate_matrix_set(
            self.measurement_basis, n_qubits)
        self.readout_vector = self._calculate_matrix_set(
            self.readout_basis, n_qubits)

        self.rotation_vector = self._calculate_matrix_set(
            self.rotation_matrixes, n_qubits)

        # generate the basis change matrix from pauli to comp and back
        A = np.zeros((self.n_states**2, self.n_states**2), dtype=complex)
        for i in range(self.n_states**2):
            # do an orthonormal transformation. 
            A[:,i] = np.ravel(self.basis_vector[i].full()) 
        self.basis_pauli_to_comp_trafo_matrix= A / self.n_states
        self.basis_comp_to_pauli_trafo_matrix= np.linalg.inv(A) 

        #get dims of qutip objects
        self.qt_dims = [[2 for i in range(self.n_qubits)], [2 for i in range(self.n_qubits)]]

        if check_labels is True:
            print(self.get_meas_operator_labels(n_qubits))
            print(self.get_basis_labels(n_qubits))




    def execute_pseudo_inverse_tomo(self, meas_operators, meas_tomo, use_pauli_basis=False,
        verbose=False):
        """
        Performs a linear tomography by simple inversion of the system of equations due to calibration points
        TE_correction_matrix: a matrix multiplying the calibration points to correct for estimated mixture due to Thermal excitation. 
        """
        #some argument parsing to we allow more general input
        meas_operators = [meas_operators] if type(meas_operators) == qt.Qobj else meas_operators
        

        #for each independent set of measurements(with their own measurement operator, calculate the coeff matrix)
        coeff_matrixes = []
        for measurement_operator in meas_operators:
            coeff_matrixes.append(self.calculate_LI_coefficient_matrix(measurement_operator, do_in_pauli=use_pauli_basis))
        coefficient_matrix = np.vstack(coeff_matrixes)

        basis_decomposition = np.zeros(4 ** self.n_qubits, dtype=complex)
        
        if use_pauli_basis:
            # first skip beta0
            #basis_decomposition[1:] = np.dot(np.linalg.pinv(coefficient_matrix[:, 1:]), meas_tomo)
            basis_decomposition[:] = np.dot(np.linalg.pinv(coefficient_matrix[:, :]), meas_tomo)
            # re-add beta0
            #basis_decomposition[0] = betas[0]
            rho = self.trans_pauli_to_comp(basis_decomposition)
        else:
            basis_decomposition = np.conj(np.linalg.pinv(coefficient_matrix)).dot(meas_tomo)
            rho = qt.Qobj(np.reshape(basis_decomposition, [self.n_states, self.n_states]),
                     dims=self.qt_dims)
        return (basis_decomposition, rho)

    def execute_mle_T_matrix_tomo(self, measurement_operators, meas_tomo, weights_tomo =False,
                                show_time=True, ftol=0.01, xtol=0.001, full_output=0, max_iter=100,
                                            TE_correction_matrix = None):
        """
        Performs a least squares optimization using fmin_powell in order to get the closest physically realisable state.
    
        This is done by constructing a lower triangular matrix T consisting of 4 ** n qubits params
        Keyword arguments:
        measurement_operators: list of meas operators
        meas_tomo: list of n_rot measurements beloning to the measurmeent operator
        use_weights : default(False) Weighs the quadrature data by the std in the estimator of the mean
                    : since this tomo does not have access to the original data, the vars should be given by
                        tomo_var_i = 1 / N_i * np.var(M_i) where i stands for the data corresponding to rotation i. 
        --- arguments for scipy fmin_powel method below, see the powell documentation
        """
        # first we calculate the measurement matrixes
        tstart = time.time()

        measurement_operators = [measurement_operators] if type(measurement_operators) == qt.Qobj else measurement_operators
        
        measurement_vectors = []
        for measurement_operator in measurement_operators:
            measurement_vectors.append([m.full() for m in  self.get_measurement_vector(measurement_operator)])
        measurement_vector = np.vstack(measurement_vectors)
        # initiate with equal weights
        self.weights = weights_tomo if weights_tomo else np.ones(len(measurement_vector)) 
        # save it in the object for use in optimization
        self.measurements_tomo = meas_tomo
        self.measurement_vector_numpy = measurement_vector
        tlinear = time.time()
        # find out the starting rho by the linear tomo
        discard, rho0 = self.execute_pseudo_inverse_tomo(measurement_operators, meas_tomo)

        # now fetch the starting t_params from the cholesky decomp of rho
        tcholesky = time.time()
        T0 = np.linalg.cholesky(scipy.linalg.sqrtm((rho0.dag() * rho0).full()))
        t0 = np.zeros(4 ** self.n_qubits, dtype='complex' )
        di = np.diag_indices(2 ** self.n_qubits)
        tri = np.tril_indices(2 ** self.n_qubits, -1)
        t0[0:2 ** self.n_qubits] = T0[di]
        t0[2**self.n_qubits::2] = T0[tri].real
        t0[2**self.n_qubits+1::2] = T0[tri].imag
        topt = time.time()
        # minimize the likelihood function using scipy
        t_optimal = scipy.optimize.fmin_powell(
            self._max_likelihood_optimization_function, t0, maxiter=max_iter, full_output=full_output, ftol=ftol, xtol=xtol)
        if show_time is True:
            print(" Time to calc rotation matrixes %.2f " % (tlinear-tstart))
            print(" Time to do linear tomo %.2f " % (tcholesky-tlinear))
            print(" Time to build T %.2f " % (topt-tcholesky))
            print(" Time to optimize %.2f" % (time.time()-topt))
        return qt.Qobj(self.build_rho_from_triangular_params(t_optimal), dims=self.qt_dims)


    def execute_SDPA_2qubit_tomo(self, measurement_operators, counts_tomo, N_total=1, used_bins=[0,3],
                                 correct_measurement_operators=True, calc_chi_squared =False,
                                 correct_zero_count_bins=True, TE_correction_matrix = None):
        """
        Estimates a density matrix given single shot counts of 4 thresholded
        bins using a custom C semidefinite solver from Nathan Langford
        Each bin should correspond to a projection operator:
        0: 00, 1: 01, 2: 10, 3: 11
        The calibration counts are used in calculating corrections to the (ideal) measurement operators
        The tomo counts are used for the actual reconstruction.
        N_total is used for normalization. if counts_tomo is normalized use N_total=1

        """

        # If the tomography bins have zero counts, they not satisfy gaussian noise. If N>>1 then turning them into 1 fixes 
        # convergence problems without screwing the total statistics/estimate.
        if(np.sum(np.where(np.array(counts_tomo) == 0)) > 0):
                print("WARNING: Some bins contain zero counts, this violates gaussian assumptions. \n \
                        If correct_zero_count_bins=True these will be set to 1 to minimize errors")
        if correct_zero_count_bins:
            counts_tomo = np.array([[int(b) if b > 0 else  1 for b in bin_counts] for bin_counts in counts_tomo])


        #Select the correct data based on the bins used
        #(and therefore based on the projection operators used)
        data = counts_tomo[:,used_bins].T.flatten()
        #get the total number of counts per tomo
        N = np.array([np.sum(counts_tomo, axis=1) for k in used_bins]).flatten()

        # add weights based on the total number of data points kept each run
        # N_total is a bit arbitrary but should be the average number of total counts of all runs, since in nathans code this
        # average is estimated as a parameter. 
        weights = N/float(N_total)
        #get the observables from the rotation operators and the bins kept(and their corresponding projection operators)
        measurement_vectors = []
        for k in used_bins:
            measurement_vectors.append([m.full() for m in  self.get_measurement_vector(measurement_operators[k])])
        measurement_vector = np.vstack(measurement_vectors)
        # print('length of the measurement vector'len(measurement_vector))
        #calculate the density matrix using the csdp solver
        a = time.time()
        rho_nathan = csdp_tomo.tomo_state(data, measurement_vector, weights)
        b=time.time()-a
        print("time solving csdp: ", b)
        print(rho_nathan)
        n_estimate = rho_nathan.trace()
        print(n_estimate)
        rho = qt.Qobj(rho_nathan / n_estimate,dims=self.qt_dims)
        
        if((np.abs(N_total - n_estimate) / N_total > 0.03)):
            print('WARNING estimated N(%d) is not close to provided N(%d) '% (n_estimate,N_total))

        if calc_chi_squared:
            chi_squared = self._state_tomo_goodness_of_fit(rho, data, N, measurement_vector)
            return rho, chi_squared
        else:
            return rho

    def execute_SDPA_MC_2qubit_tomo(self,
                                    measurement_operators,
                                    counts_tomo,
                                    N_total,
                                    used_bins = [0,2],
                                    n_runs = 100,
                                    array_like = False,
                                    correct_measurement_operators = True,
                                    TE_correction_matrix=None):
        """
        Executes the SDPDA tomo n_runs times with data distributed via a Multinomial distribution
        in order to get a list of rhos from which one can calculate errorbars on various derived quantities
        returns a list of Qobjects (the rhos).
        If array_like is set to true it will just return a 3D array of rhos
        """

        rhos= []
        for i in range(n_runs):
            #generate a data set based on multinomial distribution with means according to the measured data
            mc = [np.random.multinomial(sum(counts),(np.array(counts)+0.0) / sum(counts)) for counts in counts_tomo]
            rhos.append(self.execute_SDPA_2qubit_tomo(measurement_operators,
                                                      mc,
                                                      N_total,
                                                      used_bins,
                                                      correct_measurement_operators,
                                                      TE_correction_matrix=TE_correction_matrix))

        if array_like:
            return np.array([rho.full() for rho in rhos])
        else:
            return rhos



    def calculate_LI_coefficient_matrix(self, measurement_operator, do_in_pauli=False):
        """
        Calculates the coefficient matrix used when inversing the linear system of equations needed to find rho
        Requires a calibrated measurement operator
        if do_in_pauli is true this will presume measurement operator is given in the  basis.
        """
        coefficient_matrix = np.zeros((len(self.rotation_matrixes) ** self.n_qubits, 4 ** self.n_qubits), dtype=complex)
        Ms = self.get_measurement_vector(measurement_operator, do_in_pauli=do_in_pauli)
        for i in range(len(Ms)):
            coefficient_matrix[i,:] =  np.ravel(Ms[i].full())
        return coefficient_matrix

    def get_measurement_vector(self, measurement_operator, do_in_pauli=False):
        """
        Returns a list of rotated measurement operator based on an initial measurement operator which should be obtained from the calibration
        """
        n_rotations = len(self.rotation_matrixes) ** self.n_qubits
        measurement_vector = []
        for i in range(n_rotations):
            R = self.rotation_vector[i]
            if do_in_pauli:
                M = self.trans_comp_to_pauli(R.dag() * measurement_operator * R)
            else:
                M = R.dag() * measurement_operator * R
            measurement_vector.append(M)
        return measurement_vector

    def calibrate_measurement_operator(self, cal_meas, calibration_points=None, TE_correction_matrix=None, transform_to_pauli=False):
        """
        Calibrates the measurement operator in any basis. Assumes cal_meas are the eigenvalues of the calibration_points
        defaults the calibration_points to be the computational basis projectors. 
        If TE_correction_matrix is set, this alters the computational basis states
        """
        if calibration_points is None:
            calibration_points = comp_projectors
        M = sum([cal_meas[i] * calibration_points[i] for i in range(len(cal_meas))])
        return M if not transform_to_pauli else self.trans_comp_to_pauli(M)


    def calibrate_bin_operators(self, calibration_counts, calibration_points=None, normalize=False):
        M_bins = []
        #calculate bin probabilities normalized horizontally
        if normalize:
            cal_probs = (calibration_counts / np.sum(calibration_counts, axis = 1, dtype=float)[:,np.newaxis])
        else:
            cal_probs = np.array(calibration_counts)#(calibration_counts / np.sum(calibration_counts, axis = 1, dtype=float)[:,np.newaxis])
        for probs in cal_probs.T:
            #calibrate the measurement operator for each bin in the same way as done with average tomo. 
            M_bins.append(self.calibrate_measurement_operator(probs, calibration_points))
        return M_bins

    



#########################################################################
#
# HELPERS
#


    def trans_pauli_to_comp(self, rho_pauli):
        """
        Converts a rho in the pauli basis, or pauli basis vector or Qobj of rho in pauli basis to comp basis. 
        """
        if(rho_pauli.shape[0] == self.n_states):
            basis_decomposition = np.ravel(rho_pauli.full()) if (type(rho_pauli) ==  qt.Qobj) else np.ravel(rho_pauli)
        else:
            basis_decomposition = rho_pauli

        return qt.Qobj(np.reshape(self.basis_pauli_to_comp_trafo_matrix.dot(basis_decomposition), [self.n_states, self.n_states]),
                     dims=self.qt_dims)

    def trans_comp_to_pauli(self, rho_comp):
        """
        Converts a rho in the computational basis, or comp basis vector or Qobj of rho in computational basis to Pauli basis. 
        """
        if(rho_comp.shape[0] == self.n_states):
            basis_decomposition = np.ravel(rho_comp.full()) if (type(rho_comp) ==  qt.Qobj) else np.ravel(rho_comp)
        else:
            basis_decomposition = rho_comp

        return qt.Qobj(np.reshape(self.basis_comp_to_pauli_trafo_matrix.dot(basis_decomposition), [self.n_states, self.n_states]),
                     dims=self.qt_dims)


    def _calculate_matrix_set(self, starting_set, n_qubits):
        """recursive function that returns len(starting_set) ** n_qubits
        measurement_basis states tensored with eachother based
        on the amount of qubits

        So for 2 qubits assuming your basis set is {I, X, Y, Z} you get II IX IY IZ XI XX XY XZ ...
        """
        if(n_qubits > 1):
            return [qt.tensor(x, y) for x in self._calculate_matrix_set(starting_set, n_qubits - 1)
                    for y in starting_set]
        else:
            return starting_set

    def get_basis_labels(self, n_qubits):
        """
        Returns the basis labels in the same order as the basis vector is parsed.
        Requires self.measurement_basis_labels to be set with the correct order corresponding to the matrixes in self.measurement_basis
        """
        if(n_qubits > 1):
            return [x + y for x in self.get_basis_labels(n_qubits - 1)
                    for y in self.measurement_basis_labels]
        else:
            return self.measurement_basis_labels

    def get_meas_operator_labels(self, n_qubits):
        """
        Returns a vector of the rotations in order based on self.measurement_operator_labels
        """
        if(n_qubits > 1):
            return [x + y for x in self.get_meas_operator_labels(n_qubits - 1)
                    for y in self.measurement_operator_labels]
        else:
            return self.measurement_operator_labels

    
    ###############################3
    # MLE T Matrix functions 
    #
    def build_rho_from_triangular_params(self, t_params):
        # build the lower triangular matrix T
        T_mat = np.zeros(
            (2 ** self.n_qubits, 2 ** self.n_qubits), dtype="complex")
        di = np.diag_indices(2 ** self.n_qubits)
        T_mat[di] = t_params[0:2**self.n_qubits]
        tri = np.tril_indices(2 ** self.n_qubits, -1)
        T_mat[tri] = t_params[2**self.n_qubits::2]
        T_mat[tri] += 1j * t_params[2**self.n_qubits+1::2]
        rho = np.dot(np.conj(T_mat.T),  T_mat) / \
            np.trace(np.dot(np.conj(T_mat.T),  T_mat))
        return rho

    def _max_likelihood_optimization_function(self, t_params):
        """
        Optimization function that is evaluated many times in the maximum likelihood method.
        Calculates the difference between expected measurement values and the actual measurement values based on a guessed rho

        Keyword arguments:
        t_params : cholesky decomp parameters used to construct the initial rho
        Requires:
        self.weights :  weights per measurement vector used in calculating the loss
        """
        rho = self.build_rho_from_triangular_params(t_params)
        L = 0 + 0j
        for i in range(len(self.measurement_vector_numpy)):
            expectation = np.trace(
                np.dot(self.measurement_vector_numpy[i], rho))
            L += ((expectation -
                   self.measurements_tomo[i]) ** 2) * self.weights[i]
        return L

#############################################################################################
    # CDSP tomo functions for likelihood.
    def _state_tomo_likelihood_function(self, rho, data, normalisations, observables, fixedweight=False):
        data_predicted = []
        for ii in range(len(data)):
            data_predicted.append((rho.full().dot(observables[ii])).trace()*normalisations[ii])

        data_predicted = np.array(data_predicted)

        if fixedweight:
            likely_function = np.sum( (data-data_predicted)**2/data )
        else:
            likely_function = np.sum( (data-data_predicted)**2/data_predicted )

        return likely_function

    """
        Calculates the goodness of fit. It has a normalisation which is just
        the sum of the counts in the superconducting case since there are no
        missing counts like in the photon case.
    """
    def _state_tomo_goodness_of_fit(self, rho, data, normalisations, observables,
                                   fixedweight=False, eig_cutoff=1e-6):

        likely_function = self._state_tomo_likelihood_function(rho, data, normalisations, observables,
                                                         fixedweight=fixedweight)
        num_data = len(data)
        num_eigs = np.sum(rho.eigenenergies()>eig_cutoff)
        rho_dim = rho.shape[0]
        num_dofs = num_eigs*(2*rho_dim-num_eigs)
        out = {}
        out['pure'] = likely_function / (num_data-(2*rho_dim-1))
        out['mixed'] = likely_function / (num_data-rho_dim**2)
        out['dofs'] = likely_function / (num_data-num_dofs)

        return out

#################################################################
#
# Data Generation (currently for 2 qubits only)
#
################################################################## 

def generate_tomo_data(rho, M, R, N, M_bins = None):
    """
    Generates data for tomography. Both returns expectation values(used for average tomo) 
    or bin counts( if you use thresholded tomo). Generates a single multinomial set of counts to get both data types. 
    """

    #decompose the measurement operator in its spectrum
    eigenvals, eigenstates = M.eigenstates()
    if M_bins is None:
        M_bins =  comp_projectors
    # now decompose the 
    probs = []
    for state in eigenstates:
        #calculate probability of ending up in this state
        probs.append(((R.dag() * (qt.ket2dm(state) * R)) * rho).tr().real)
    #run a multinomial distribution to determine the "experimental" measurement outcomes
    counts = np.random.multinomial(N, probs)
    # use the simulated percentages of states found to calc voltages
    expectations =  sum((counts / float(N)) * eigenvals)
    #calcultate bin counts via the projections of original eigenstates onto bin measurement operator. 
    bin_counts = [sum([counts[j] * (M_bins[i] * qt.ket2dm(eigenstates[j])).tr().real
                  for j in range(len(eigenstates))])
                  for i in range(len(M_bins))]

    return bin_counts, expectations

def get_TE_calibration_points(e_01, e_10, get_coefficient_matrix=False):
    """
    Mixes the standard computational basis projectors to account for a certain thermal excitation fraction in qubit 1 and 2
    get_coefficient_matrix : return a matrix so one can correct the normal measurement operators used in TOMO
    If it is set to false, just return the mixed calibration points.
    """
    P = comp_projectors
    R = [   qt.tensor(qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmax()),
            qt.tensor(qt.sigmax(), qt.qeye(2)),
            qt.tensor(qt.sigmax(), qt.sigmax())]
    #calculate the effect TE on the 00 state using probabilities to be excited(or not)
    c_00 = (1-e_01) * (1-e_10) * P[0] + e_01 * (1-e_10) * P[1] + e_10 * (1-e_01) * P[2] + e_01 * e_10 * P[3]
    #find the other points via bit flip rotations
    c_01 = R[1] * c_00 * R[1].dag()
    c_10 = R[2] * c_00 * R[2].dag()
    c_11 = R[3] * c_00 * R[3].dag()

    if get_coefficient_matrix:
        return np.array([np.diag(c_00.full()), np.diag(c_01.full()), np.diag(c_10.full()), np.diag(c_11.full())]).T
    else:
        return [c_00, c_01, c_10, c_11]












##############################################################33
#
# THRESHOLDED TOMO
#

class TomoAnalysisThresholded(TomoAnalysis):

    def do_thresholded_tomo():
        pass


################################################################
#
# Averaged TOMO
#

class TomoAnalysisAveraged(TomoAnalysis):

    def do_averaged_tomo():
             # calculate measurement operator from calibration points, can be seen as the betas!
        measurement_operator = self.calibrate_measurement_operator(meas_cal, calibration_points, transform_to_pauli=False)
        pass
