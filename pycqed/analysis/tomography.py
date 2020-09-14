import logging
try:
    import qutip as qtp
except ImportError as e:
    logging.warning('Could not import qutip, tomo code will not work')
import numpy as np
import time
import scipy
import os
import lmfit

import matplotlib.pyplot as plt
from pycqed.analysis import measurement_analysis as ma
import pycqed.analysis.tools.data_manipulation as dm_tools
from functools import reduce


class TomoAnalysis_JointRO():

    """
    Performs state tomography based on an overcomplete set of measurements
    and calibration measurements. Uses qutip to calculate resulting basis
    states from applied rotations.

    Works for Joint RO (like REM experiment).

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
    rotation_matrixes = [qtp.identity(2), qtp.sigmax(),
                         qtp.rotation(qtp.sigmax(), np.pi / 2),
                         qtp.rotation(qtp.sigmay(), np.pi / 2),
                         qtp.rotation(qtp.sigmax(), -np.pi / 2),
                         qtp.rotation(qtp.sigmay(), -np.pi / 2)]
    measurement_operator_labels = ['I', 'X', 'x', 'y', '-x', '-y']
    # MAKE SURE THE LABELS CORRESPOND TO THE ROTATION MATRIXES DEFINED ABOVE

    # The set of single qubit basis operators and labels
    measurement_basis = [
        qtp.identity(2), qtp.sigmaz(), qtp.sigmax(), qtp.sigmay()]
    measurement_basis_labels = ['I', 'Z', 'X', 'Y']
    # The operators used in the readout basis on each qubit
    readout_basis = [qtp.identity(2), qtp.sigmaz()]

    def __init__(self, measurements_cal, measurements_tomo,
                 n_qubits=2, n_quadratures=1, check_labels=True):
        """
        keyword arguments:
        measurements_cal --- Should be an array of length 2 ** n_qubits
        measurements_tomo --- Should be an array of length
            length(rotation_matrixes) ** n_qubits
        n_qubits --- default(2) the amount of qubits present in the experement
        n_quadratures --- default(1(either I or Q)) The amount of complete
            measurement data sets. For example a combined IQ measurement has
            2 measurement sets.

        """
        self.measurements_cal = measurements_cal
        self.measurements_tomo = measurements_tomo
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.n_quadratures = n_quadratures

        # Generate the vectors of matrixes that correspond to all measurements,
        # readout bases and rotations

        self.basis_vector = self._calculate_matrix_set(
            self.measurement_basis, n_qubits)
        self.readout_vector = self._calculate_matrix_set(
            self.readout_basis, n_qubits)
        self.rotation_vector = self._calculate_matrix_set(
            self.rotation_matrixes, n_qubits)

        if check_labels is True:
            print('Measurement op. labels: {}'.format(
                self.get_meas_operator_labels(n_qubits)))
            print('Basis labels: {}'.format(self.get_basis_labels(n_qubits)))

    def execute_pseudo_inverse_tomo(self):
        """
        Performs a linear tomography by simple inversion of the system of
        equations due to calibration points
        """

        # calculate beta positions in coefficient matrix
        coefficient_matrix = self._calculate_coefficient_matrix()
        basis_decomposition = np.zeros(4 ** self.n_qubits)
        # first skip beta0
        basis_decomposition[1:] = np.dot(
            np.linalg.pinv(coefficient_matrix[:, 1:]), self.measurements_tomo)
        # re-add beta0
        basis_decomposition[0] = 1
        # now recreate the rho
        rho = sum([basis_decomposition[i] * self.basis_vector[i] /
                   (2 ** self.n_qubits)
                   for i in range(len(basis_decomposition))])
        return (basis_decomposition, rho)

    def execute_least_squares_physical_tomo(self, use_weights=True, show_time=False,
                                            ftol=0.01, xtol=0.001, full_output=0,
                                            max_iter=1000):
        """
        Performs a max likelihood optimization using fmin_powell in order to
        get the closest physically realisable state.

        This is done by constructing a lower triangular matrix T consisting of
        4 ** n qubits params

        Keyword arguments:
        use_weights : default(true) Weighs the quadrature data by the std in
                      betas obtained
        --- arguments for scipy fmin_powel method below, see
            the powel documentation
        """
        # first we calculate the measurement matrices
        tstart = time.time()
        measurement_vector = []
        n_rot = len(self.rotation_matrixes) ** self.n_qubits
        # initiate with equal weights
        self.weights = np.ones(self.n_quadratures * n_rot)
        for quadrature in range(self.n_quadratures):
            betas = self._calibrate_betas(
                self.measurements_cal[quadrature * self.n_states:
                                      (1 + quadrature) * self.n_states])
            # determine the weights based on betas absolote difference and
            # accuracy
            if (use_weights):
                self.weights[
                    quadrature * n_rot:(1+quadrature) * n_rot] = (
                    max(betas) - min(betas)) / np.var(betas)
            for rotation_index, rotation in enumerate(self.rotation_vector):
                measurement_vector.append(
                    betas[0] * rotation.dag()
                    * self.readout_vector[0] * rotation)
                for i in range(1, len(betas)):
                    measurement_vector[n_rot * quadrature + rotation_index] += betas[
                        i] * rotation.dag() * self.readout_vector[i] * rotation
        # save it in the object for use in optimization
        self.measurement_vector = measurement_vector
        self.measurement_vector_numpy = [
            vec.full() for vec in measurement_vector]
        tlinear = time.time()
        # find out the starting rho by the linear tomo
        discard, rho0 = self.execute_pseudo_inverse_tomo()
        # now fetch the starting t_params from the cholesky decomp of rho
        tcholesky = time.time()
        T0 = np.linalg.cholesky(scipy.linalg.sqrtm((rho0.dag() * rho0).full()))
        t0 = np.zeros(4 ** self.n_qubits, dtype='complex')
        di = np.diag_indices(2 ** self.n_qubits)
        tri = np.tril_indices(2 ** self.n_qubits, -1)
        t0[0:2 ** self.n_qubits] = T0[di]
        t0[2**self.n_qubits::2] = T0[tri].real
        t0[2**self.n_qubits+1::2] = T0[tri].imag
        topt = time.time()
        # minimize the likelihood function using scipy
        t_optimal = scipy.optimize.fmin_powell(
            self._max_likelihood_optimization_function, t0, maxiter=max_iter,
            full_output=full_output, ftol=ftol, xtol=xtol)
        if show_time is True:
            print(" Time to calc rotation matrixes %.2f " % (tlinear-tstart))
            print(" Time to do linear tomo %.2f " % (tcholesky-tlinear))
            print(" Time to build T %.2f " % (topt-tcholesky))
            print(" Time to optimize %.2f" % (time.time()-topt))
        return qtp.Qobj(self.build_rho_from_triangular_params(t_optimal),
                        dims=[[2 for i in range(self.n_qubits)],
                              [2 for i in range(self.n_qubits)]])

    def execute_SDPA_MC_2qubit_tomo(self,
                                    counts_tomo,
                                    counts_cal,
                                    N_total,
                                    used_bins=[0, 2],
                                    n_runs=100,
                                    array_like=False,
                                    correct_measurement_operators=True):
        """
        Executes the SDPDA tomo n_runs times with data distributed via a Multinomial distribution
        in order to get a list of rhos from which one can calculate errorbars on various derived quantities
        returns a list of Qobjects (the rhos).
        If array_like is set to true it will just return a 3D array of rhos
        """

        rhos = []
        for i in range(n_runs):
            # generate a data set based on multinomial distribution with means according to the measured data
            mc = [np.random.multinomial(sum(counts), (np.array(
                counts)+0.0) / sum(counts)) for counts in counts_tomo]
            rhos.append(self.execute_SDPA_2qubit_tomo(mc,
                                                      counts_cal,
                                                      N_total,
                                                      used_bins,
                                                      correct_measurement_operators))

        if array_like:
            return np.array([rho.full() for rho in rhos])
        else:
            return rhos

    def execute_SDPA_2qubit_tomo(self, counts_tomo, counts_cal, N_total, used_bins=[0, 2],
                                 correct_measurement_operators=True):
        """
        Estimates a density matrix given single shot counts of 4 thresholded
        bins using a custom C semidefinite solver from Nathan Langford
        Each bin should correspond to a projection operator:
        0: 00, 1: 01, 2: 10, 3: 11
        The calibration counts are used in calculating corrections to the (ideal) measurement operators
        The tomo counts are used for the actual reconstruction.
        """
        if isinstance(used_bins, int):
            # allow for a single projection operator
            used_bins = [used_bins]

        Pm_corrected = self.get_meas_operators_from_cal(counts_cal,
                                                        correct_measurement_operators)

        # Select the correct data based on the bins used
        # (and therefore based on the projection operators used)
        data = np.array([float(count[k])
                         for count in counts_tomo for k in used_bins]).transpose()

        # get the total number of counts per tomo
        N = np.array([np.sum(counts_tomo, axis=1)
                      for k in used_bins]).flatten()

        # add weights based on the total number of data points kept each run
        #weights = np.sqrt(N)
        weights = N/float(N_total)

        # calculate the density matrix using the sdpa solver
        rho_nathan, n_estimate = self._tomoc_fw(
            [Pm_corrected[k] for k in used_bins], data, weights=weights)

        if((np.abs(N_total - n_estimate) / N_total > 0.03)):
            print('WARNING estimated N(%d) is not close to provided N(%d) ' %
                  (n_estimate, N_total))

        return rho_nathan

    def get_meas_operators_from_cal(self, counts_cal, correct_measurement_operators=True):
        """
        Used in the thresholded tomography. Returns the set of corrected measurement operators
        """
        # setup the projection operators
        Pm_0 = qtp.projection(2, 0, 0)
        Pm_1 = qtp.projection(2, 1, 1)
        Pm_00 = qtp.tensor(Pm_0, Pm_0)
        Pm_11 = qtp.tensor(Pm_1, Pm_1)
        Pm_01 = qtp.tensor(Pm_0, Pm_1)
        Pm_10 = qtp.tensor(Pm_1, Pm_0)
        Pm = [Pm_00, Pm_01, Pm_10, Pm_11]
        # calculate bin probabilities normalized horizontally
        probs = counts_cal / \
            np.sum(counts_cal, axis=1, dtype=float)[:, np.newaxis]
        # print(probs)
        # correct the measurement operators based on calibration point counts
        if correct_measurement_operators is True:
            # just calc P_m_corrected = probs.T * P_m (matrix product)
            d = range(len(Pm))
            l = range(np.shape(counts_cal)[1])
            Pm_corrected = [sum(probs.T[i][j] * Pm[j] for j in d) for i in l]
        else:
            Pm_corrected = Pm
        # print 'Printing operators'
        # print Pm_corrected
        # print 'End of operators'
        return Pm_corrected

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


##############################################################
#
#                    Private functions
#
##############################################################


    def _max_likelihood_optimization_function(self, t_params):
        """
        Optimization function that is evaluated many times in the maximum
        likelihood method.

        Calculates the difference between expected measurement values and the
        actual measurement values based on a guessed rho

        Keyword arguments:
        t_params : cholesky decomp parameters used to construct the initial rho
        Requires:
        self.weights :  weights per measurement vector used in calculating the
        loss
        """
        rho = self.build_rho_from_triangular_params(t_params)
        L = 0 + 0j
        for i in range(len(self.measurement_vector)):
            expectation = np.trace(
                np.dot(self.measurement_vector_numpy[i], rho))
            L += ((expectation -
                   self.measurements_tomo[i]) ** 2) * self.weights[i]
        return L

    def _calibrate_betas(self, measurements_cal):
        """
        calculates betas from calibration points for the initial measurement
        operator

        Betas are ordered by B0 -> II B1 -> IZ etc(binary counting)
        <0|Z|0> = 1, <1|Z|1> = -1

        Keyword arguments:
        measurements_cal --- array(2 ** n_qubits) should be ordered
            correctly (00, 01, 10, 11) for 2 qubits
        """
        cal_matrix = np.zeros((self.n_states, self.n_states))
        # get the coefficient matrix for the betas
        for i in range(self.n_states):
            for j in range(self.n_states):
                # perform bitwise AND and count the resulting 1s
                cal_matrix[i, j] = (-1)**(bin((i & j)).count("1"))
        # invert solve the simple system of equations
        betas = np.dot(np.linalg.inv(cal_matrix), measurements_cal)
        return betas

    def _calculate_coefficient_matrix(self):
        """
        Calculates the coefficient matrix used when inversing the linear
        system of equations needed to find rho
        If there are multiple measurements present this will return a matrix
        of (n_quadratures * n_rotation_matrixes ** n_qubits) x n_basis_vectors
        """
        coefficient_matrix = np.zeros(
            (self.n_quadratures * len(self.rotation_matrixes) ** self.n_qubits,
             4 ** self.n_qubits))
        n_rotations = len(self.rotation_matrixes) ** self.n_qubits
        # Now fill in 2 ** self.n_qubits betas into the coefficient matrix on
        # each row
        for quadrature in range(self.n_quadratures):
            # calibrate betas for this quadrature
            self.betas = self._calibrate_betas(
                self.measurements_cal[quadrature * self.n_states:
                                      (1 + quadrature) * self.n_states])
            for rotation_index in range(n_rotations):
                for beta_index in range(2 ** self.n_qubits):
                    (place, sign) = self._get_basis_index_from_rotation(
                        beta_index, rotation_index)
                    coefficient_matrix[
                        n_rotations * quadrature + rotation_index, place] = sign * self.betas[beta_index]
        return coefficient_matrix

    def _get_basis_index_from_rotation(self, beta_index, rotation_index):
        """
        Returns the position and sign of one of the betas in the coefficient
        matrix by checking to which basis matrix the readout matrix is mapped
        after rotation
        This is used in _calculate_coefficient_matrix
        """
        m = self.rotation_vector[rotation_index].dag(
        ) * self.readout_vector[beta_index] * self.rotation_vector[rotation_index]
        for basis_index, basis in enumerate(self.basis_vector):
            if(m == basis):
                return (basis_index, 1)
            elif(m == -basis):
                return (basis_index, -1)
        # if no basis is found raise an error
        raise Exception(
            'No basis vector found corresponding to the measurement rotation. Check that you have used Clifford Gates!')

    def _calculate_matrix_set(self, starting_set, n_qubits):
        return _calculate_matrix_set(starting_set, n_qubits)

    def _tomoc_fw(self, measurement_operators, data, weights=False, filename=False, reload_toolbox=False):
        """
        Wrapper function to parse the data for the SDPA tomo.
        TODO cut this code and build a toolbox based on a python version of a c-based semi-definite optimization wrapper, because this code is horrendous.
        Uses a  python parser to rewrite the MLE tomo into a semi-definite optimization problem which is then solved by a C-Library.
        requires a list of measurement operators and a set of data of dims: (len(measurement_operators) * len(self.rotation_vector,1)
        """
        # get directory of toolbox
        directory = os.path.dirname(__file__) + '/tools/tomoc_fw'
        if reload_toolbox is True:
            reload(pytomoc_fw)
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)
        if len(weights.shape) == 1:
            weights = np.expand_dims(weights, axis=1)
        if type(weights) is bool:
            if not weights:
                weights = np.ones(data.shape)
        observables = [rot.dag() * measurement_operator *
                       rot for rot in self.rotation_vector for measurement_operator in measurement_operators]
        observablearray = np.array(
            [np.ravel(obs.full(), order='C') for obs in observables])

        out = np.concatenate((data, weights, observablearray), axis=1)
        if not filename:
            filename = 'temp' + str(uuid.uuid4())
        with open(directory+'/'+filename+'.tomo', 'w') as f:
            np.savetxt(f, out.view(float), fmt='%.11g', delimiter=',')
        pytomoc_fw.execute_pytomoc_fw({}, f.name)
        filename_rho = directory+'\\'+filename

        rho = np.loadtxt(filename_rho+'.rhor', delimiter=',') + \
            1j*np.loadtxt(filename_rho+'.rhoi', delimiter=',')
        N_est = rho.trace()
        rho = rho/rho.trace()
        rho = qtp.Qobj(rho, dims=[[2, 2], [2, 2]])

        # delete temp files
        files = glob.glob(filename_rho+'*')
        for file in files:
            os.remove(file)

        return rho, N_est


def _calculate_matrix_set(starting_set, n_qubits):
    """
    recursive function that returns len(starting_set) ** n_qubits
    measurement_basis states tensored with each other based on the amount
    of qubits

    So for 2 qubits assuming your basis set is {I, X, Y, Z}
    you get II IX IY IZ XI XX XY XZ ...
    """
    if(n_qubits > 1):
        return [qtp.tensor(x, y) for x in
                _calculate_matrix_set(starting_set, n_qubits - 1)
                for y in starting_set]
    else:
        return starting_set


#########################
# Tomo helper functions #
#########################


def get_operators_label():
    labels = []
    for i in range(2**4):
        vector = get_pauli_op_vector(i)
        label = ''
        for j in range(2):
            if vector[j] == 0:
                label = 'I'+label
            if vector[j] == 1:
                label = 'Z'+label
            if vector[j] == 2:
                label = 'X'+label
            if vector[j] == 3:
                label = 'Y'+label
        labels.append(label)

    labels = ['IX', 'IY', 'IZ', 'XI', 'YI', 'ZI', 'XX',
              'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    return labels


def order_pauli_output2(pauli_op_dis):
    '''
    Converts Pauli counting fromatted as  IZXY q0 | IZXY q1  to
    pauli q0 | paulis q1 | pauli correlators
    '''
    pauli_1 = np.array([pauli_op_dis[2], pauli_op_dis[3], pauli_op_dis[1]])
    pauli_2 = np.array([pauli_op_dis[8], pauli_op_dis[12], pauli_op_dis[4]])
    pauli_corr = np.array([pauli_op_dis[10], pauli_op_dis[11], pauli_op_dis[9],
                           pauli_op_dis[14], pauli_op_dis[15],
                           pauli_op_dis[13], pauli_op_dis[6],
                           pauli_op_dis[7], pauli_op_dis[5]])
    return pauli_1, pauli_2, pauli_corr


def pauli_ops_from_density_matrix(rho):
    """
    Takes in a density matrix and returns a vector containing the expectation
    values for the Pauli operators. Works for two qubits.

    Args:
        rho (Qobj) : density matrix, qutip Qobj
    Returns:
        numpy array containing expectation values
    """
    operators = np.zeros(16, dtype=np.complex128)
    pauli = [qtp.identity(2), qtp.sigmaz(), qtp.sigmax(), qtp.sigmay()]
    for k in range(16):
        i = int(k % 4)
        j = int(((k - i)/4) % 4)
        operators[k] = (rho*qtp.tensor(pauli[j], pauli[i])).tr()
    operators = np.real(operators)
    return operators


def plot_target_pauli_set(pauli_set, ax):
    width = 0.6
    ind = np.arange(15)
    ax.bar(ind, pauli_set[1:], width, color='lightgray', align='center')


def plot_operators(results, ax, labels=None):
    # NB: reorders the pauli expectation values to correct for convention
    pauli_1, pauli_2, pauli_cor = order_pauli_output2(results)
    width = 0.35
    ind1 = np.arange(3)
    ind2 = np.arange(3, 6)
    ind3 = np.arange(6, 15)
    ax.bar(ind1, pauli_1, width, color='r', align='center')
    ax.bar(ind2, pauli_2, width, color='b', align='center')
    ax.bar(ind3, pauli_cor, width, color='purple', align='center')

    if labels is None:
        labels = get_operators_label()

    ax.set_xticks(np.arange(0, 2**4))
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.05, 1.05)


def get_pauli_op_vector(pauli_number):
    N = 2
    pauli_vector = np.zeros(N)
    rest = pauli_number
    for i in range(0, N, 1):
        value = rest % 4
        pauli_vector[i] = value
        rest = (rest-value)/4
    return pauli_vector


def calc_fidelity1(dens_mat1, dens_mat2):
    sqrt_2 = qtp.Qobj(dens_mat2).sqrtm()
    fid = ((sqrt_2 * qtp.Qobj(dens_mat1) * sqrt_2).sqrtm()).tr()
    return np.real(fid)


def calc_fid1_bell(densmat, bell):
    up = qtp.basis(2, 0)
    dn = qtp.basis(2, 1)
    rhos_bell = [qtp.ket2dm((qtp.tensor([up, up]) + qtp.tensor([dn, dn])).unit()),
                 qtp.ket2dm(
                     (qtp.tensor([up, dn]) + qtp.tensor([dn, up])).unit()),
                 qtp.ket2dm(
                     (qtp.tensor([up, up]) - qtp.tensor([dn, dn])).unit()),
                 qtp.ket2dm((qtp.tensor([up, dn]) - qtp.tensor([dn, up])).unit())]
    return calc_fidelity1(rhos_bell[bell], densmat)


def get_cardianal_pauli_exp(cardinal_idx):
    '''
    Returns a expectation values for the puali operators for the cardinal
    states. Input is the index of the cardinal state.
    Ordering of the cardinals is binary counting over [Z, -Z, X, -X, -Y, Y]
    Returns expectation values of:
        II|XI YI ZI|IX IY IZ|XX YX ZX XY YY ZY XZ YZ ZZ

    N.B. The cardinal counting is defined by the preparation pulses
    in
    '''
    X = np.array([1, 0, 0])
    Y = np.array([0, 1, 0])
    Z = np.array([0, 0, 1])
    pauli_basis_states = [Z, -Z, X, -X, -Y, Y]
    pauli_1 = pauli_basis_states[cardinal_idx % 6]
    pauli_2 = pauli_basis_states[cardinal_idx//6]

    pauli_corr = np.zeros(9)
    for i in range(3):
        for j in range(3):
            pauli_corr[3*i+j] = pauli_1[j]*pauli_2[i]
    pauli_vec = np.concatenate(([1], pauli_1, pauli_2, pauli_corr))
    return pauli_vec


def get_bell_pauli_exp(bell_idx, theta_q0=0, theta_q1=0):
    """
    Get's the pauli operators for the bell states.
    Args:
        bell_idx (int) : integer referring to a specific bell state.
            1: |Psi_m> = |00> - |11>   (<XX>,<YY>,<ZZ>) = (-1,+1,+1)
            2: |Psi_p> = |00> + |11>   (<XX>,<YY>,<ZZ>) = (+1,-1,+1)
            3: |Psi_m> = |01> - |10>   (<XX>,<YY>,<ZZ>) = (-1,-1,-1)
            4: |Psi_m> = |01> + |10>   (<XX>,<YY>,<ZZ>) = (+1,+1,-1)

        theta_q0  (float): angle to correct for single qubit phase errors
        theta_q1  (float):

    Phase error on the MSQ/q1:
        keeps <XX> unchanged
        exchanges <YY> with <ZY)
        exchanges <ZZ> with <YZ>

        if <YY> and <ZZ> have same sign, the residual <ZY> and <YZ> have
        opposite sign, and viceversa.

    Phase error on the LSQ/q0:
        exchanges <XX> with <XY>
        exchanges <YY> with <YX>
        keeps <ZZ> unchanged

        if <XX> and <YY> have same sign, the residual <XY> and <YX> have
        opposite sign, and viceversa.
    """

    # This snippet is for the WIP two qubit phases
    # single_q_paulis = [1] + [0]*3 + [0]*3

    # base_bell_paulis

    # paulic = [-np.cos(theta_q0)*np.cos(theta_q1),  # XX
    #           np.sin(theta_q0),
    #           0,
    #           np.sin(theta_q0),
    #           np.cos(theta_q0)*np.cos(theta_q1),  # YY
    #           -np.sin(theta_q1),         !!!!!!!!!!!!!!!! HERE MINUS SIGN ADDED
    #           0,
    #           np.sin(theta_q1),
    #           np.cos(theta_q0)*np.cos(theta_q1)]  # ZZ

    if bell_idx == 0:
        sets_bell = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             -np.cos(theta_q0), -np.sin(theta_q0),
             0, 0, np.sin(theta_q0), np.cos(theta_q0)])
    elif bell_idx == 1:
        sets_bell = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, np.cos(
            theta_q0), np.sin(theta_q0), 0, 0, np.sin(theta_q0), -np.cos(theta_q0)])
    elif bell_idx == 2:
        sets_bell = np.array(
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0,
             -np.cos(theta_q0), -np.sin(theta_q0),
             0, 0, np.sin(theta_q0), -np.cos(theta_q0)])
    elif bell_idx == 3:
        sets_bell = np.array(
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0,
             np.cos(theta_q0), -np.sin(theta_q0),
             0, 0, np.sin(theta_q0), np.cos(theta_q0)])
    else:
        raise ValueError('bell_idx must be 0, 1, 2 or 3')
    pauli1, pauli2, paulic = order_pauli_output2(sets_bell)

    return np.concatenate(([1], pauli1, pauli2, paulic))
    # return sets_bell


def calc_fid2_cardinal(pauli_op_dis, cardinal_state):
    """
    Calculates fidelity using the pauli set representation of the state

    F = (1+P*P_t)/4 (1 is for identity component)

    """

    pauli_expectations = np.concatenate(order_pauli_output2(pauli_op_dis))
    #  II XI YI ZI|IX IY IZ|XX YX ZX XY YY ZY XZ YZ ZZ
    target_expectations = get_cardianal_pauli_exp(cardinal_state)
    # 1 is for identity
    return 0.25*(1 + np.dot(pauli_expectations, target_expectations[1:]))


def calc_fid2_bell(pauli_op_dis, target_bell_idx, theta=0):
    """
    Calculates fidelity to one of the 4 bell states. Allows varying the angle
    """
    sets_bell = get_bell_pauli_exp(target_bell_idx, theta)
    pauli_expectations = np.concatenate(order_pauli_output2(pauli_op_dis))
    return 0.25*(1 + np.dot(pauli_expectations, sets_bell[1:]))


def rotated_bell_state(dummy_x, angle_MSQ, angle_LSQ,
                       contrast, target_bell=0):
    # only works for target_bell=0 for now.
    # to expand, need to figure out the signs in the elements.
    # order is set by looping I,Z,X,Y
    # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
    # II IZ IX IY ZI ZZ ZX ZY XI XZ XX XY YI YZ YX YY
    state = np.zeros(16)
    state[0] = 1.
    if target_bell == 0:
        state[5] = np.cos(angle_MSQ)
        state[6] = np.sin(angle_LSQ)*np.sin(angle_MSQ)
        state[7] = np.cos(angle_LSQ)*np.sin(angle_MSQ)
        state[10] = -np.cos(angle_LSQ)
        state[11] = np.sin(angle_LSQ)
        state[13] = -np.sin(angle_MSQ)
        state[14] = np.cos(angle_MSQ)*np.sin(angle_LSQ)
        state[15] = np.cos(angle_MSQ)*np.cos(angle_LSQ)
    elif target_bell == 1:
        state[5] = np.cos(angle_MSQ)
        state[6] = -np.sin(angle_LSQ)*np.sin(angle_MSQ)
        state[7] = -np.cos(angle_LSQ)*np.sin(angle_MSQ)
        state[10] = np.cos(angle_LSQ)
        state[11] = -np.sin(angle_LSQ)
        state[13] = -np.sin(angle_MSQ)
        state[14] = -np.cos(angle_MSQ)*np.sin(angle_LSQ)
        state[15] = -np.cos(angle_MSQ)*np.cos(angle_LSQ)
    elif target_bell == 2:
        state[5] = -np.cos(angle_MSQ)
        state[6] = -np.sin(angle_LSQ)*np.sin(angle_MSQ)
        state[7] = -np.cos(angle_LSQ)*np.sin(angle_MSQ)
        state[10] = -np.cos(angle_LSQ)
        state[11] = np.sin(angle_LSQ)
        state[13] = np.sin(angle_MSQ)
        state[14] = -np.cos(angle_MSQ)*np.sin(angle_LSQ)
        state[15] = -np.cos(angle_MSQ)*np.cos(angle_LSQ)
    elif target_bell == 3:
        state[5] = -np.cos(angle_MSQ)
        state[6] = np.sin(angle_LSQ)*np.sin(angle_MSQ)
        state[7] = np.cos(angle_LSQ)*np.sin(angle_MSQ)
        state[10] = np.cos(angle_LSQ)
        state[11] = -np.sin(angle_LSQ)
        state[13] = np.sin(angle_MSQ)
        state[14] = np.cos(angle_MSQ)*np.sin(angle_LSQ)
        state[15] = np.cos(angle_MSQ)*np.cos(angle_LSQ)
    return state


def rotated_bell_model(operators,target_bell):
    fit_func_wrapper = lambda dummy_x, angle_MSQ,\
        angle_LSQ, contrast: rotated_bell_state(dummy_x,
                                                angle_MSQ, angle_LSQ,
                                                contrast, target_bell)
    angles_model = lmfit.Model(fit_func_wrapper)

    angles_model.set_param_hint(
        'angle_MSQ', value=0., min=-np.pi, max=np.pi, vary=True)
    angles_model.set_param_hint(
        'angle_LSQ', value=0., min=-np.pi, max=np.pi, vary=True)
    angles_model.set_param_hint(
        'contrast', value=1., min=0., max=1., vary=False)
    params = angles_model.make_params()

    fit_res = angles_model.fit(data=operators,
                                    dummy_x=np.arange(
                                        len(operators)),
                                    params=params)
    return fit_res

class Tomo_Multiplexed(ma.MeasurementAnalysis):

    def __init__(self, auto=True, label='', timestamp=None,
                 MLE=False, target_cardinal=None, target_bell=None,
                 start_shot=0, end_shot=-1,
                 verbose=0,
                 single_shots=True,
                 fig_format='png',
                 q0_label='q0',
                 q1_label='q1',
                 q_post_select_label='qA',
                 q_post_select=False,  # parity ancilla
                 q_post_select_threshold=0,  # ancilla threshold
                 q_post_selection_states=[0],  # selecting on excited state
                 q_post_selection_indices=[0],
                 weight_channels=[0, 1],
                 close_fig=True,
                 tomo_after_ancilla_mmt=False,
                 q_post_select_initialisation=False,
                 nr_parity_check_rounds=1,
                 PF_tracking='first',
                 PF_parity_pattern=['ZZ'],
                 blossom_record=None,
                 **kw):
        self.label = label
        self.timestamp = timestamp
        self.target_cardinal = target_cardinal
        self.target_bell = target_bell
        self.start_shot = start_shot
        self.end_shot = end_shot
        self.MLE = MLE
        self.verbose = verbose
        self.fig_format = fig_format
        self.q0_label = q0_label
        self.q1_label = q1_label
        self.close_fig = close_fig
        self.single_shots = single_shots
        # parameters for parity measurements
        self.q_post_select_label = q_post_select_label
        self.q_post_select = q_post_select
        self.q_post_selection_states = q_post_selection_states
        self.q_post_selection_indices = q_post_selection_indices
        self.q_post_select_threshold = q_post_select_threshold
        self.weight_channels = weight_channels
        self.tomo_after_ancilla_mmt = tomo_after_ancilla_mmt
        self.q_post_select_initialisation = q_post_select_initialisation
        self.nr_parity_check_rounds = nr_parity_check_rounds
        self.PF_tracking = PF_tracking
        self.PF_parity_pattern = PF_parity_pattern
        self.blossom_record = blossom_record
        kw['h5mode'] = 'r+'
        super(Tomo_Multiplexed, self).__init__(auto=auto, timestamp=timestamp,
                                               label=label, **kw)
        # if auto is True:
        #     self.run_default_analysis()

    def run_default_analysis(self, **kw):
        self.get_naming_and_values()
        # hard coded number of segments for a 2 qubit state tomography
        # constraint imposed by UHFLI
        self.nr_segments = 64
        self.nr_measurement_segments = 36  # the rest is calibration points
        self.exp_name = os.path.split(self.folder)[-1][7:]
        nr_consecutive_mmts = self.nr_parity_check_rounds + \
            self.tomo_after_ancilla_mmt+self.q_post_select_initialisation
        if self.single_shots:
            # data qubit tomo results:
            # only selecting the last data qubit outcome
            measured_values1_preshaping = self.measured_values[self.weight_channels[0]
                                                               ][nr_consecutive_mmts-1::nr_consecutive_mmts]
            # only selecting the last data qubit outcome
            measured_values2_preshaping = self.measured_values[self.weight_channels[1]
                                                               ][nr_consecutive_mmts-1::nr_consecutive_mmts]
            nr_runs = int(len(measured_values1_preshaping)/self.nr_segments)

            # reshaping results according to segments and runs
            measured_values1 = np.zeros(
                (self.nr_segments, nr_runs))
            measured_values2 = np.zeros(
                (self.nr_segments, nr_runs))
            if self.PF_tracking in ['PF_first_all_phi_plus','PF_tracking', 'PF_reset', 'PF_reset_all_phi_plus', 'PF_reset_all_blossom', 'PF_no_error'] or 'selected_blossom' in self.PF_tracking:
                # preparing a Pauli record with dimensions according to the parity pattern
                self.PF_record = np.zeros(
                    (self.nr_measurement_segments, nr_runs, len(self.PF_parity_pattern)))
            for i in range(self.nr_segments):
                measured_values1[i, :] = measured_values1_preshaping[i::self.nr_segments]
                measured_values2[i, :] = measured_values2_preshaping[i::self.nr_segments]

            # ancilla qubit data preparation
            if self.q_post_select:
                measured_values_ancilla = np.zeros(
                    (self.nr_segments, nr_consecutive_mmts,  nr_runs))
                measured_values_ancilla_preshaping = self.measured_values[self.weight_channels[2]]
                for j in range(nr_consecutive_mmts):
                    measured_values_ancilla_select = measured_values_ancilla_preshaping[
                        j::nr_consecutive_mmts]
                    for i in range(self.nr_segments):
                        measured_values_ancilla[i, j, :] = measured_values_ancilla_select[i::self.nr_segments]

            if self.q_post_select_initialisation:  # only initizilizing the ancilla
                measured_values_ancilla_preshaping = self.measured_values[
                    self.weight_channels[2]][0::nr_consecutive_mmts]
                measured_values_ancilla_init = np.zeros(
                    (self.nr_segments, nr_runs))
                for i in range(self.nr_segments):
                    measured_values_ancilla_init[i,:] = measured_values_ancilla_preshaping[i::self.nr_segments]

            # digitizing and serivative of the ancilla outcomes and postselection of the data qubit outcomes

            # no selection on the calibration points, only the 36 runs
            for i in range(self.nr_measurement_segments):
                # post selecting on initialization measurement
                if self.q_post_select_initialisation:
                    shots_qA_init = measured_values_ancilla_init[i]
                    post_select_indices_0 = dm_tools.get_post_select_indices(
                        thresholds=[self.q_post_select_threshold],
                        init_measurements=[shots_qA_init],
                        positive_case=True)
                    measured_values1[i, :][post_select_indices_0] = np.nan
                    measured_values2[i, :][post_select_indices_0] = np.nan

                if self.q_post_select:
                    # first digitize the 2D array
                    # post selecting on parity measurement outcomes
                    # 1. selecting a trace to analyze
                    if self.tomo_after_ancilla_mmt:  # throw away last point
                        ancilla_outcomes = measured_values_ancilla[i, :-1, :]
                    else:
                        ancilla_outcomes = measured_values_ancilla[i, :, :]
                    # 2. digitize the trace to 0 (ground) and 1 (excited)
                    ancilla_outcomes = dm_tools.digitize(data=ancilla_outcomes,
                                                         threshold=self.q_post_select_threshold,
                                                         zero_state=0)
                    # 3. take the binary derivative for the 2D arrays
                    if not self.q_post_select_initialisation:
                        # the derivative will remove one colum of data, add zeros to compensate if no initialization mmt is there
                        zeros = np.zeros((1,np.shape(ancilla_outcomes)[1]))
                        ancilla_outcomes=np.append(ancilla_outcomes, zeros, axis=0)
                    ancilla_outcomes_derivative = dm_tools.binary_derivative_2D(
                        ancilla_outcomes, axis=1)
                    # getting the postselection indices based on the relevant postselection index
                    for index, state in zip(self.q_post_selection_indices, self.q_post_selection_states):
                        post_select_indices_0 = np.array(
                            np.where(ancilla_outcomes_derivative[index, :] != state)).flatten()
                        measured_values1[i, :][post_select_indices_0] = np.nan
                        measured_values2[i, :][post_select_indices_0] = np.nan
                    if self.PF_tracking=='PF_no_error':
                        if len(self.PF_parity_pattern)==2:
                            parity_valuess = [[0, 0], [0, 1], [1, 0], [1, 1]]
                            nr_rounds = int(self.nr_parity_check_rounds/2)
                        elif len(self.PF_parity_pattern)==1:
                            parity_valuess = [[0], [1]]
                            nr_rounds = self.nr_parity_check_rounds
                        post_select_indices_total=[]
                        for parity_values in parity_valuess:
                            parity_values=np.tile(parity_values,nr_rounds)
                            indices = np.arange(nr_rounds*len(self.PF_parity_pattern))
                            post_select_indices_sub=[]
                            for indexnow, (index, state) in enumerate(zip(indices, parity_values)):
                                post_indices = np.array(np.where(ancilla_outcomes_derivative[index, :] != state)).flatten()
                                 # = post_indices
                                post_select_indices_sub=np.unique(np.concatenate((post_indices, post_select_indices_sub)))
                            post_select_indices_total.append(post_select_indices_sub)
                        # self.post_select_indices=post_select_indices_total
                        b=reduce(np.intersect1d, (post_select_indices_total))
                        post_select_indices_0=[int(i) for i in b]
                        measured_values1[i, :][post_select_indices_0] = np.nan
                        measured_values2[i, :][post_select_indices_0] = np.nan

                    if 'selected_blossom' in self.PF_tracking:
                        #post-selecting outcomes where blossom has not detected a data-qubit error
                        if self.PF_tracking=='PF_selected_blossom':
                            post_select_indices_0 = np.array(np.where(np.array(self.blossom_record)[i, :, 1] !=0)).flatten()
                        else:
                            post_select_indices_0 = np.array(np.where(np.array(self.blossom_record)[i, :, 0] !=0)).flatten()
                        measured_values1[i, :][post_select_indices_0] = np.nan
                        measured_values2[i, :][post_select_indices_0] = np.nan

                    # 4. make a Pauli record in case the pauli frame of tomography is to be updated
                    if self.PF_tracking in ['PF_first_all_phi_plus','PF_tracking', 'PF_reset', 'PF_reset_all_phi_plus', 'PF_reset_all_blossom', 'PF_no_error'] or 'selected_blossom' in self.PF_tracking:
                        for j in range(len(self.PF_parity_pattern)):
                            parity_outcomes = ancilla_outcomes_derivative[j::len(
                                self.PF_parity_pattern)]
                            # adding bell-state specific number
                            if self.PF_tracking=='PF_tracking':
                                if self.q_post_selection_states[j]==1:
                                    added_nr = len(parity_outcomes)+1
                                else:
                                    added_nr = 0
                                self.PF_record[i, :, j] = np.sum(
                                    parity_outcomes, axis=0)+added_nr% 2
                            elif self.PF_tracking=='PF_first_all_phi_plus':
                                if len(parity_outcomes)==1:
                                    self.PF_record[i, :, j] = parity_outcomes
                                else:
                                    self.PF_record[i, :, j] = parity_outcomes[0]

                            elif self.PF_tracking=='PF_reset':
                                if self.q_post_selection_states[j]==1:
                                    self.PF_record[i, :, j] = parity_outcomes[-1]+1%2
                                else:
                                    self.PF_record[i, :, j] = parity_outcomes[-1]
                            elif self.PF_tracking in ['PF_reset_all_phi_plus', 'PF_no_error']:
                                if len(parity_outcomes)==1:
                                    self.PF_record[i, :, j] = parity_outcomes
                                else:
                                    self.PF_record[i, :, j] = parity_outcomes[-1]
                            elif 'blossom' in self.PF_tracking:
                                if len(self.PF_parity_pattern)==1:
                                    self.PF_record[i, :, j] = np.array(self.blossom_record)[i, :, 2+j]
                                elif len(self.PF_parity_pattern)==2:
                                    self.PF_record[i, :, j] = np.array(self.blossom_record)[i, :, 1+j]

            #calculate post-selection fraction
            if self.q_post_select:
                relevant_data=measured_values1[:36,:].flatten()
                self.fraction=1-len(np.where(np.isnan(relevant_data))[0])/np.size(relevant_data)

            # Get correlations between shots
            if self.start_shot != 0 or self.end_shot != -1:
                self.shots_q0 = self.shots_q0[:, self.start_shot:self.end_shot]
                self.shots_q1 = self.shots_q1[:, self.start_shot:self.end_shot]
            ########################################
            # Making  the first figure, tomo shots #
            ########################################
            avg_h1 = np.nanmean(measured_values1, axis=1)
            avg_h2 = np.nanmean(measured_values2, axis=1)
        else:
            avg_h1 = self.measured_values[0]
            avg_h2 = self.measured_values[1]

        # Binning all the points required for the tomo
        h1_00 = np.nanmean(avg_h1[36:36+7])
        h1_01 = np.nanmean(avg_h1[43:43+7])
        h1_10 = np.nanmean(avg_h1[50:50+7])
        h1_11 = np.nanmean(avg_h1[57:])

        h2_00 = np.nanmean(avg_h2[36:36+7])
        h2_01 = np.nanmean(avg_h2[43:43+7])
        h2_10 = np.nanmean(avg_h2[50:50+7])
        h2_11 = np.nanmean(avg_h2[57:])
        # Substract avg of all traces
        mean_h1 = (h1_00+h1_10+h1_01+h1_11)/4
        mean_h2 = (h2_00+h2_01+h2_10+h2_11)/4

        avg_h1 -= mean_h1
        avg_h2 -= mean_h2

        scale_h1 = (h1_00+h1_10-h1_01-h1_11)/4
        scale_h2 = (h2_00+h2_01-h2_10-h2_11)/4

        # normalizing all shots according to the calibration points
        avg_h1 = (avg_h1)/scale_h1
        avg_h2 = (avg_h2)/scale_h2

        # creating the correlation data
        if self.single_shots:
            # Get correlations between shots
            self.shots_q0q1 = np.multiply(measured_values2-mean_h2, measured_values1-mean_h1)
            if self.start_shot != 0 or self.end_shot != -1:
                self.shots_q0q1 = self.shots_q0q1[:, self.start_shot:self.end_shot]
            avg_h12 = np.nanmean(self.shots_q0q1, axis=1)
        else:
            avg_h12 = self.measured_values[2]

        #getting averages from the calibration points
        h12_00 = np.nanmean(avg_h12[36:36+7])
        h12_01 = np.nanmean(avg_h12[43:43+7])
        h12_10 = np.nanmean(avg_h12[50:50+7])
        h12_11 = np.nanmean(avg_h12[57:])
        mean_h12 = (h12_00+h12_11+h12_01+h12_10)/4
        # print('check mean', mean_h12)
        avg_h12 -= mean_h12
        scale_h12 = (h12_00+h12_11-h12_01-h12_10)/4
        avg_h12 = (avg_h12)/scale_h12
        # applying pauli frame update here
        # first suptracting offsets and rescaling all individual shots
        if self.PF_tracking in ['PF_first_all_phi_plus','PF_tracking', 'PF_reset', 'PF_reset_all_phi_plus','PF_reset_all_blossom', 'PF_no_error'] or 'selected_blossom' in self.PF_tracking:
            # first subtracting offsets
            measured_values1 -= mean_h1
            measured_values2 -= mean_h2
            self.shots_q0q1 -= mean_h12
            # and rescaling all individual shots
            measured_values1 = measured_values1/scale_h1
            measured_values2 = measured_values2/scale_h2
            self.shots_q0q1 = self.shots_q0q1/scale_h12
            for j, axis in enumerate(self.PF_parity_pattern):
                record = self.PF_record[:, :, j]
                if axis=='ZZ':
                    for i in np.concatenate([np.arange(0,12), np.arange(24,36)]):
                        for k in range(len(record[i,:])):
                            if record[i,k]==1:
                                measured_values1[i,k]=-1*measured_values1[i,k]
                                self.shots_q0q1[i,k]=-1*self.shots_q0q1[i,k]
                elif axis=='XX':
                    for i in np.concatenate([np.arange(12,36)]):
                        for k in range(len(record[i,:])):
                            if record[i,k]==1:
                                measured_values1[i,k]=-1*measured_values1[i,k]
                                self.shots_q0q1[i,k]=-1*self.shots_q0q1[i,k]

            #redefining the average values
            avg_h1 = np.nanmean(measured_values1, axis=1)
            avg_h2 = np.nanmean(measured_values2, axis=1)
            avg_h12 = np.nanmean(self.shots_q0q1, axis=1)



        # key for next step
        h1_00 = np.nanmean(avg_h1[36:36+7])
        h1_01 = np.nanmean(avg_h1[43:43+7])
        h1_10 = np.nanmean(avg_h1[50:50+7])
        h1_11 = np.nanmean(avg_h1[57:])

        h2_00 = np.nanmean(avg_h2[36:36+7])
        h2_01 = np.nanmean(avg_h2[43:43+7])
        h2_10 = np.nanmean(avg_h2[50:50+7])
        h2_11 = np.nanmean(avg_h2[57:])

        h12_00 = np.nanmean(avg_h12[36:36+7])
        h12_01 = np.nanmean(avg_h12[43:43+7])
        h12_10 = np.nanmean(avg_h12[50:50+7])
        h12_11 = np.nanmean(avg_h12[57:])

        std_h1_00 = np.nanstd(avg_h1[36:36+7])
        std_h1_01 = np.nanstd(avg_h1[43:43+7])
        std_h1_10 = np.nanstd(avg_h1[50:50+7])
        std_h1_11 = np.nanstd(avg_h1[57:])

        std_h2_00 = np.nanstd(avg_h2[36:36+7])
        std_h2_01 = np.nanstd(avg_h2[43:43+7])
        std_h2_10 = np.nanstd(avg_h2[50:50+7])
        std_h2_11 = np.nanstd(avg_h2[57:])

        std_h12_00 = np.nanstd(avg_h12[36:36+7])
        std_h12_01 = np.nanstd(avg_h12[43:43+7])
        std_h12_10 = np.nanstd(avg_h12[50:50+7])
        std_h12_11 = np.nanstd(avg_h12[57:])

        std_h1 = np.nanmean([std_h1_00, std_h1_01, std_h1_10, std_h1_11])
        std_h2 = np.nanmean([std_h2_00, std_h2_01, std_h2_10, std_h2_11])
        std_h12 = np.nanmean([std_h12_00, std_h12_01, std_h12_10, std_h12_11])
        std_arr = np.array([std_h1_00, std_h1_01, std_h1_10, std_h1_11, std_h2_00, std_h2_01,
                            std_h2_10, std_h2_11, std_h12_00, std_h12_01, std_h12_10, std_h12_11])

        # plt.plot([std_h1, std_h2, std_h12])
        # plt.plot(std_arr)
        # plt.show()

        fac = np.nanmean([std_h1, std_h2, std_h12])
        avg_h1 *= fac/std_h1
        avg_h2 *= fac/std_h2
        avg_h12 *= fac/std_h12

        h1_00 = np.nanmean(avg_h1[36:36+7])
        h1_01 = np.nanmean(avg_h1[43:43+7])
        h1_10 = np.nanmean(avg_h1[50:50+7])
        h1_11 = np.nanmean(avg_h1[57:])

        h2_00 = np.nanmean(avg_h2[36:36+7])
        h2_01 = np.nanmean(avg_h2[43:43+7])
        h2_10 = np.nanmean(avg_h2[50:50+7])
        h2_11 = np.nanmean(avg_h2[57:])

        h12_00 = np.nanmean(avg_h12[36:36+7])
        h12_01 = np.nanmean(avg_h12[43:43+7])
        h12_10 = np.nanmean(avg_h12[50:50+7])
        h12_11 = np.nanmean(avg_h12[57:])

        self.plot_TV_mode(avg_h1, avg_h2, avg_h12)
        #############################
        # Linear inversion tomo #
        #############################

        measurements_tomo = (
            np.array([avg_h1[0:36], avg_h2[0:36],
                      avg_h12[0:36]])).flatten()  # 108 x 1
        # get the calibration points by averaging over the five measurements
        # taken knowing the initial state we put in
        measurements_cal = np.array(
            [h1_00, h1_01, h1_10, h1_11,
             h2_00, h2_01, h2_10, h2_11,
             h12_00, h12_01, h12_10, h12_11])

        # before we calculate the tomo we need to set the correct order of the
        # rotation matrixes
        TomoAnalysis_JointRO.rotation_matrixes = [
            qtp.identity(2),
            qtp.sigmax(),
            qtp.rotation(qtp.sigmay(), np.pi / 2),
            qtp.rotation(qtp.sigmay(), -np.pi / 2),
            qtp.rotation(qtp.sigmax(), np.pi / 2),
            qtp.rotation(qtp.sigmax(), -np.pi / 2)]
        TomoAnalysis_JointRO.measurement_operator_labels = ['I', 'X', 'y',
                                                            '-y', 'x', '-x']
        TomoAnalysis_JointRO.measurement_basis = [qtp.identity(2),
                                                  qtp.sigmaz(), qtp.sigmax(),
                                                  qtp.sigmay()]
        TomoAnalysis_JointRO.measurement_basis_labels = ['I', 'Z', 'X', 'Y']
        # TomoAnalysis_JointRO.measurement_basis_labels = ['I', 'A', 'B', 'C']
        TomoAnalysis_JointRO.readout_basis = [qtp.identity(2), qtp.sigmaz()]

        # calculate the tomo
        tomo = TomoAnalysis_JointRO(
            measurements_cal, measurements_tomo, n_qubits=2, n_quadratures=3,
            check_labels=(self.verbose > 0))
        self.tomo = tomo
        self.meas_op_labels = np.concatenate(
            order_pauli_output2(tomo.get_basis_labels(2)))
        # operators are expectation values of Pauli operators, rho is density
        # mat
        (ops, self.rho) = tomo.execute_pseudo_inverse_tomo()

        # ops are in the wrong order. The following function gets them from
        # the density matrix in the correct order:
        self.operators = pauli_ops_from_density_matrix(self.rho)

        self.best_fidelity = -1
        if self.MLE:
            # mle reconstruction of density matrix
            self.rho_2 = tomo.execute_least_squares_physical_tomo(
                ftol=0.000001, xtol=0.0001)
            # reconstructing the pauli vector
            if self.verbose > 1:
                print(self.rho_2)
            if self.verbose > 0:
                print('Purity %.3f' % (self.rho_2*self.rho_2).tr())
            # calculates the Pauli operator expectation values based on the
            # matrix
            self.operators_mle = pauli_ops_from_density_matrix(
                self.rho_2)
            if self.verbose > 0:
                print(self.operators_mle)
        ########################
        # FIT PHASE CORRECTIONS
        ########################
        if self.MLE:
            self.operators_fit = self.operators_mle
        else:
            self.operators_fit = self.operators
        """
        bell_idx (int) : integer referring to a specific bell state.
            0: |Phi_m> = |00> - |11>   (<XX>,<YY>,<ZZ>) = (-1,+1,+1)
            1: |Phi_p> = |00> + |11>   (<XX>,<YY>,<ZZ>) = (+1,-1,+1)
            2: |Psi_m> = |01> - |10>   (<XX>,<YY>,<ZZ>) = (-1,-1,-1)
            3: |Psi_p> = |01> + |10>   (<XX>,<YY>,<ZZ>) = (+1,+1,-1)
        """

        fit_func_wrapper = lambda dummy_x, angle_MSQ,\
            angle_LSQ, contrast: rotated_bell_state(dummy_x,
                                                    angle_MSQ, angle_LSQ,
                                                    contrast, self.target_bell)
        angles_model = lmfit.Model(fit_func_wrapper)

        angles_model.set_param_hint(
            'angle_MSQ', value=0., min=-np.pi, max=np.pi, vary=True)
        angles_model.set_param_hint(
            'angle_LSQ', value=0., min=-np.pi, max=np.pi, vary=True)
        angles_model.set_param_hint(
            'contrast', value=1., min=0., max=1., vary=False)
        params = angles_model.make_params()

        self.fit_res = angles_model.fit(data=self.operators_fit,
                                        dummy_x=np.arange(
                                            len(self.operators_fit)),
                                        params=params)
        if self.target_bell is not None:
            self.plot_phase_corr()
        self.plot_LI()
        if self.MLE:
            self.plot_MLE()
        try:
            self.add_analysis_datagroup_to_file()
            self.save_fitted_parameters(fit_res=self.fit_res,
                                        var_name='MLE')
        except Exception as e:
            logging.warning(e)
        try:
            pars_dict = {'fidelity': self.fidelity,
                         'fidelity_mle': self.fidelity_mle,
                         'best_fidelity': self.best_fidelity,
                         'operators_mle_ZZ': self.operators_mle[5],
                         'operators_mle_XX': self.operators_mle[10],
                         'operators_mle_YY': self.operators_mle[15],
                         'operators_ZZ': self.operators[5],
                         'operators_XX': self.operators[10],
                         'operators_YY': self.operators[15],
                         'angle_LSQ': np.rad2deg(self.fit_res.best_values['angle_LSQ']),
                         'angle_MSQ': np.rad2deg(self.fit_res.best_values['angle_MSQ']),
                         'LSQ_name': self.q0_label,
                         'MSQ_name': self.q1_label,
                         'data_fraction': self.fraction}
            if 'no_error' in self.PF_tracking:
                name='tomography_results_decoding_{}_states_{}_indices_{}'.format(self.PF_tracking,
                                                                self.q_post_selection_states[:len(self.PF_parity_pattern)],
                                                                self.q_post_selection_indices[:len(self.PF_parity_pattern)])
            else:
                name='tomography_results_decoding_{}_states_{}_indices_{}'.format(self.PF_tracking,
                                                                self.q_post_selection_states,
                                                                self.q_post_selection_indices)
            self.save_dict_to_analysis_group(pars_dict, name)
        # only works if MLE and target bell were specified
        except Exception as e:
            print(e)

        self.data_file.close()

    def plot_TV_mode(self, avg_h0, avg_h1, avg_h01):

        figname = 'Tomography_shots_Exp_{}.{}'.format(self.exp_name,
                                                      self.fig_format)
        fig1, axs = plt.subplots(1, 3, figsize=(17, 4))
        fig1.suptitle(self.exp_name+' ' + self.timestamp_string, size=16)
        ax = axs[0]
        ax.plot(np.arange(self.nr_segments), avg_h0,
                'o-')
        ax.set_title('{}'.format(self.q0_label))
        ax = axs[1]
        ax.plot(np.arange(self.nr_segments), avg_h1,
                'o-')
        ax.set_title('{}'.format(self.q1_label))
        ax = axs[2]
        ax.plot(np.arange(self.nr_segments), avg_h01,
                'o-')
        ax.set_title('Correlations {}-{}'.format(self.q0_label, self.q1_label))
        savename = os.path.abspath(os.path.join(
            self.folder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig1.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig1)

    def plot_LI(self):
        # Making  the second figure, LI tomo
        fig2 = plt.figure(figsize=(15, 5))
        ax = fig2.add_subplot(121)
        if self.target_cardinal is not None:
            self.fidelity = calc_fid2_cardinal(self.operators,
                                               self.target_cardinal)
            target_expectations = get_cardianal_pauli_exp(
                self.target_cardinal)
            plot_target_pauli_set(target_expectations, ax)

        if self.target_bell is not None:
            self.fidelity = calc_fid2_bell(
                self.operators, self.target_bell)
            target_expectations = get_bell_pauli_exp(self.target_bell)
            plot_target_pauli_set(target_expectations, ax)
            txt_x_pos = 0
        else:
            txt_x_pos = 10

        plot_operators(self.operators, ax, labels=self.meas_op_labels)
        ax.set_title('Least squares tomography.')
        if self.verbose > 0:
            print(self.rho)
        qtp.matrix_histogram_complex(self.rho, xlabels=['00', '01', '10', '11'],
                                     ylabels=['00', '01', '10', '11'],
                                     fig=fig2, ax=fig2.add_subplot(
            122, projection='3d'))
        purity = (self.rho*self.rho).tr()
        msg = 'Purity: {:.3f}'.format(
            purity)
        if self.target_bell is not None or self.target_cardinal is not None:
            msg += '\nFidelity to target {:.3f}'.format(self.fidelity)
        if self.target_bell is not None:
            theta_vec = np.linspace(0., 2*np.pi, 1001)
            fid_vec = np.zeros(theta_vec.shape)
            for i, theta in enumerate(theta_vec):
                fid_vec[i] = calc_fid2_bell(self.operators,
                                            self.target_bell, theta)
            msg += '\nMAX Fidelity {:.3f} at {:.1f} deg'.format(
                np.max(fid_vec),
                theta_vec[np.argmax(fid_vec)]*180./np.pi)
        ax.text(txt_x_pos, .6, msg)

        figname = 'LI-Tomography_Exp_{}.{}'.format(self.exp_name,
                                                   self.fig_format)
        fig2.suptitle(self.exp_name+' ' + self.timestamp_string, size=16)
        savename = os.path.abspath(os.path.join(
            self.folder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig2.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig2)

    def plot_MLE(self):
        # Figure 3 MLE reconstruction
        fig3 = plt.figure(figsize=(15, 5))
        ax = fig3.add_subplot(121)

        if self.target_cardinal is not None:
            self.fidelity_mle = calc_fid2_cardinal(self.operators_mle,
                                                   self.target_cardinal)
            target_expectations = get_cardianal_pauli_exp(
                self.target_cardinal)
            plot_target_pauli_set(target_expectations, ax)
            txt_x_pos = 10
            if self.target_cardinal in [0,7]:
                txt_y_pos=-0.5
            else:
                txt_y_pos=0.5

        elif self.target_bell is not None:
            self.fidelity_mle = calc_fid2_bell(self.operators_mle,
                                               self.target_bell)
            target_expectations = get_bell_pauli_exp(self.target_bell)
            plot_target_pauli_set(target_expectations, ax)
            txt_x_pos = -1
            txt_y_pos = 0.5
        else:
            txt_x_pos = 10

        purity = (self.rho_2*self.rho_2).tr()
        if self.target_bell is not None or self.target_cardinal is not None:
            msg = 'Purity: {:.3f}\nFidelity to target {:.3f}'.format(
                purity, self.fidelity_mle)
            if self.target_bell is not None:
                theta_vec = np.linspace(0., 2*np.pi, 1001)
                fid_vec = np.zeros(theta_vec.shape)
                for i, theta in enumerate(theta_vec):
                    fid_vec[i] = calc_fid2_bell(self.operators_mle,
                                                self.target_bell, theta)
                msg += str('\nMAX Fidelity {:.3f} at \n  ' + self.q0_label
                           + '={:.1f} deg and\n  ' + self.q1_label
                           + '={:.1f} deg').format(self.best_fidelity,
                                                   self.fit_res.best_values['angle_LSQ'] *
                                                   180./np.pi,
                                                   self.fit_res.best_values['angle_MSQ']*180./np.pi)
            if self.q_post_select:
                msg +=str('\nselected frac. ={:.3f}'.format(self.fraction))

            ax.text(txt_x_pos, txt_y_pos, msg)

        plot_operators(self.operators_mle, ax, labels=self.meas_op_labels)
        ax.set_title('Max likelihood estimation tomography')
        qtp.matrix_histogram_complex(self.rho_2, xlabels=['00', '01', '10', '11'],
                                     ylabels=['00', '01', '10', '11'],
                                     fig=fig3,
                                     # limits=[0,0.5],
                                     ax=fig3.add_subplot(122, projection='3d'))
        if 'no_error' in self.PF_tracking:
            figname = 'MLE-Tomography_decoding_{}_states_{}_indices_{}_Exp_{}.{}'.format(self.PF_tracking,
                                                                               self.q_post_selection_states[:len(self.PF_parity_pattern)],
                                                                               self.q_post_selection_indices[:len(self.PF_parity_pattern)],
                                                                               self.exp_name,
                                                                               self.fig_format,
                                                                               )
        else:
            figname = 'MLE-Tomography_decoding_{}_states_{}_indices_{}_Exp_{}.{}'.format(self.PF_tracking,
                                                                               self.q_post_selection_states,
                                                                               self.q_post_selection_indices,
                                                                               self.exp_name,
                                                                               self.fig_format,
                                                                               )

        fig3.suptitle(self.exp_name+' ' + self.timestamp_string, size=16)
        savename = os.path.abspath(os.path.join(
            self.folder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig3.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig3)

    def plot_phase_corr(self):
        fig2 = plt.figure(figsize=(15, 5))
        ax = fig2.add_subplot(111)
        ordered_fit = np.concatenate(
            ([1], np.concatenate(order_pauli_output2(self.fit_res.best_fit))))
        plot_target_pauli_set(ordered_fit, ax)
        plot_operators(self.operators_fit, ax=ax, labels=self.meas_op_labels)
        fidelity = np.dot(self.fit_res.best_fit, self.operators_fit)*0.25
        self.best_fidelity = fidelity
        angle_LSQ_deg = self.fit_res.best_values['angle_LSQ']*180./np.pi
        angle_MSQ_deg = self.fit_res.best_values['angle_MSQ']*180./np.pi
        ax.set_title('Fit of single qubit phase errors')
        msg = ('MAX Fidelity at %.3f $\phi_{' + self.q0_label
               + '}=$%.1f deg and $\phi_{' + self.q1_label + '}=$%.1f deg') \
            % (fidelity, angle_LSQ_deg, angle_MSQ_deg)
        msg += "\n Chi sqr. %.3f" % self.fit_res.chisqr
        ax.text(0.5, .6, msg)
        figname = 'Fit_report_{}.{}'.format(self.exp_name,
                                            self.fig_format)
        fig2.suptitle(self.exp_name+' ' + self.timestamp_string, size=16)
        savename = os.path.abspath(os.path.join(
            self.folder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig2.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig2)
        angle_LSQ_deg = self.fit_res.best_values['angle_LSQ']*180./np.pi
        angle_MSQ_deg = self.fit_res.best_values['angle_MSQ']*180./np.pi
        ax.set_title('Fit of single qubit phase errors')
        msg = ('MAX Fidelity at %.3f $\phi_{' + self.q1_label
               + '}=$%.1f deg and $\phi_{' + self.q0_label + '}=$%.1f deg')\
            % (fidelity, angle_LSQ_deg, angle_MSQ_deg)
        msg += "\n Chi sqr. %.3f" % self.fit_res.chisqr
        ax.text(0.5, .6, msg)
        figname = 'Fit_report_{}.{}'.format(self.exp_name,
                                            self.fig_format)
        fig2.suptitle(self.exp_name+' ' + self.timestamp_string, size=16)
        savename = os.path.abspath(os.path.join(
            self.folder, figname))
        # value of 450dpi is arbitrary but higher than default
        fig2.savefig(savename, format=self.fig_format, dpi=450)
        if self.close_fig:
            plt.close(fig2)
