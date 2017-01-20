import qutip as qtp
import numpy as np
import time
import scipy
import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from pycqed.analysis import composite_analysis as ca


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
    # The set of single qubit basis operators and labels
    measurement_basis = [
        qtp.identity(2), qtp.sigmaz(), qtp.sigmax(), qtp.sigmay()]
    measurement_basis_labels = ['I', 'Z', 'X', 'Y']
    # The operators used in the readout basis on each qubit
    readout_basis = [qtp.identity(2), qtp.sigmaz()]

    def __init__(self, measurements_cal, measurements_tomo,
                 n_qubits=2, n_quadratures=1):
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

    def execute_linear_tomo(self):
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

    def execute_max_likelihood(self, use_weights=True, show_time=True,
                               ftol=0.01, xtol=0.001, full_output=0,
                               max_iter=100):
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
        discard, rho0 = self.execute_linear_tomo()
        # now fetch the starting t_params from the cholesky decomp of rho
        tcholesky = time.time()
        T0 = np.linalg.cholesky((rho0.dag() * rho0).full() / 2)
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
                        dims=[[2 for i in range(self.n_qubits)], [2 for i in
                                                                  range(self.n_qubits)]])

    def get_basis_labels(self, n_qubits):
        """
        Returns the basis labels in the same order as the basis vector is parsed.
        Requires self.measurement_basis_labels to be set with the correct order corresponding to the matrixes in self.measurement_basis
        """
        if(n_qubits > 1):
            return [y + x for x in self.get_basis_labels(n_qubits - 1)
                    for y in self.measurement_basis_labels]
        else:
            return self.measurement_basis_labels

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


def _calculate_matrix_set(starting_set, n_qubits):
    """
    recursive function that returns len(starting_set) ** n_qubits
    measurement_basis states tensored with each other based on the amount
    of qubits

    So for 2 qubits assuming your basis set is {I, X, Y, Z}
    you get II IX IY IZ XI XX XY XZ ...
    """
    if(n_qubits > 1):
        return [qtp.tensor(y, x) for x in
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
        operators[k] = (rho*qtp.tensor(pauli[i], pauli[j])).tr()
    operators = np.real(operators)
    return operators


def plot_target_pauli_set(pauli_set, ax):
    width = 0.35
    ind = np.arange(15)
    ax.bar(ind, pauli_set[1:], width, color='gray')

def plot_operators(results, ax):
    #     fig = plt.figure(figsize=(8,6))
    #     ax = fig.add_subplot(111)
    pauli_1, pauli_2, pauli_cor = order_pauli_output2(results)
    width = 0.35
    ind1 = np.arange(3)
    ind2 = np.arange(3, 6)
    ind3 = np.arange(6, 15)
    ind = np.arange(15)
    q1 = ax.bar(ind1, pauli_1, width, color='r')
    q1 = ax.bar(ind2, pauli_2, width, color='b')
    q2 = ax.bar(ind3, pauli_cor, width, color='purple')

    ax.set_title('2 Qubit State Tomography')
#         ax.set_ylim(-1,1)
    ax.set_xticks(np.arange(0, 2**4))
    ax.set_xticklabels(get_operators_label())


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
    Ordering of the cardinals is binary counting over [Z, -Z, X, -X, Y, -Y]

    Returns expectation values of:
        II|XI YI ZI|IX IY IZ|XX YX ZX XY YY ZY XZ YZ ZZ
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


def calc_fid2_bell(pauli_expectations, bell):
    sets_bell = [np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1]),
                 np.array([1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
                 np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]),
                 np.array([1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])]
    return 0.25*np.dot(pauli_expectations, sets_bell[bell])


def calc_fid2_bell_theta(pauli_set, theta, bell):
    if bell == 0:
        sets_bell = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, np.cos(
            theta), np.sin(theta), 0, 0, np.sin(theta), -np.cos(theta)])
    elif bell == 3:
        sets_bell = np.array(
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0,
             -np.cos(theta), -np.sin(theta),
             0, 0, np.sin(theta), -np.cos(theta)])
    elif bell == 2:
        sets_bell = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             -np.cos(theta), -np.sin(theta),
             0, 0, np.sin(theta), np.cos(theta)])
    elif bell == 1:
        sets_bell = np.array(
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0,
             np.cos(theta), -np.sin(theta),
             0, 0, np.sin(theta), np.cos(theta)])
    return 0.25*np.dot(pauli_set, sets_bell)


def analyse_tomo(t_start, t_stop=None, label='',
                 target_cardinal=None, target_bell=None,
                 save_figures=True, fig_format='png',
                 cut_shots=False,
                 MLE = True, close_fig=True,
                 start_shot=0, end_shot=-1, verbose=0):
    """
    TODO: add phase manhattan plot
    Performs two qubit linear inversion (LE) and maximum likelihood
    estimation (MLE) tomography analysis on different datasets.

    Args:
        t_start         (str) : starting timestamp to scan for files
        t_stop          (str) : stopping timestamp to scan for files
        label           (str) : label used for selecting files
        target_cardinal (int) : calculates fidelity to target state specified
        target_bell     (int) : calculates fidelity to target state specified
        save_figures    (bool): save figures next to the data
        fig_format      (str) : format for matplotlib savefig function
        cut_shots       (bool): ???
        start_shot      (int) : ??
        end_shot        (int) : ???
        verbose         (int) : verbosity level in print statements, higher is
                                more print statements

    Analysis creates 3 figures for each analyzed tomography.
        - Plot of I and Q values vs measurement
        - LI bar and Manhattan plot of state reconstruction
        - MLE bar and Manhattan plot of state reconstruction
    """
    if t_stop is None:
        t_stop = t_start
    opt_dict = {'scan_label': label}

    pdict = {'I': 'I',
             'Q': 'Q',
             'times': 'sweep_points'}
    # Important that the shots returned are labeled I and Q
    # default shot detector do this but something to watch out for
    nparams = ['I', 'Q', 'times']

    # used to extract data from multiple files
    tomo_scans = ca.quick_analysis(t_start=t_start, t_stop=t_stop,
                                   options_dict=opt_dict,
                                   params_dict_TD=pdict,
                                   numeric_params=nparams)
    # save_folder = a_tools.get_folder(tomo_scans.TD_timestamps[0])
    # print(save_folder)

    # hard coded number of segments for a 2 qubit state tomography
    nr_segments = 64
    # reshaping the data from a 1D array to a 64*nr_shots array
    shots_q0 = np.zeros(
        (nr_segments, int(len(tomo_scans.TD_dict['I'][0])/nr_segments)))
    shots_q1 = np.zeros(
        (nr_segments, int(len(tomo_scans.TD_dict['Q'][0])/nr_segments)))
    for i in range(nr_segments):
        shots_q0[i, :] = tomo_scans.TD_dict['I'][0][i::nr_segments]
        shots_q1[i, :] = tomo_scans.TD_dict['Q'][0][i::nr_segments]

    shots_q0q1 = np.multiply(shots_q1, shots_q0)

    # discarding shots FIXME: is this required or can we remove this feature?
    if cut_shots:
        shots_q0 = shots_q0[:, start_shot:end_shot]
        shots_q1 = shots_q1[:, start_shot:end_shot]
        shots_q0q1 = shots_q0q1[:, start_shot:end_shot]

    ##########################################
    # Making  the first figure, tomo shots
    ##########################################
    savefolder = ca.a_tools.get_folder(tomo_scans.TD_timestamps[0])
    exp_name = os.path.split(savefolder)[-1][7:]
    figname = 'Tomography_shots_Exp_{}.{}'.format(exp_name, fig_format)
    fig1, axs = plt.subplots(1, 3, figsize=(17, 4))
    fig1.suptitle(exp_name+' ' + tomo_scans.TD_timestamps[0], size=16)
    # fig1.set_title(figname)
    ax = axs[0]
    ax.plot(np.arange(nr_segments), np.mean(shots_q0, axis=1), 'o-')
    ax.set_title('Qubit 0'.format(exp_name))
    # ax.plot(np.arange(nr_segments),shots_q0[:,2])
    ax = axs[1]
    ax.plot(np.arange(nr_segments), np.mean(shots_q1, axis=1), 'o-')
    ax.set_title('Qubit 1'.format(exp_name))
    # ax.plot(np.arange(nr_segments),shots_q1[:,2])
    ax = axs[2]
    ax.plot(np.arange(nr_segments), np.mean(shots_q0q1, axis=1), 'o-')
    ax.set_title('Correlations'.format(exp_name))
    if save_figures:
        savename = os.path.abspath(os.path.join(
            savefolder, figname))
        fig1.savefig(savename, format=fig_format)
    if close_fig:
        plt.close(fig1)
    # Putting calibration points together for crosstalk correction (beta)
    avg_h1 = np.mean(shots_q0, axis=1)
    # FIXME unused matrices? Can be deleted but double check
    matrix = np.array(
        [[1, 1, 1, 1],
         [1, -1, 1, -1],
         [1, 1, -1, -1],
         [1, -1, -1, 1]])

    h1_00 = np.mean(avg_h1[36:36+7])
    h1_01 = np.mean(avg_h1[43:43+7])
    h1_10 = np.mean(avg_h1[50:50+7])
    h1_11 = np.mean(avg_h1[57:])

    h1_vec = np.array([h1_00, h1_01, h1_10, h1_11])

    avg_h2 = np.mean(shots_q1, axis=1)
    matrix = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])

    h2_00 = np.mean(avg_h2[36:36+7])
    h2_01 = np.mean(avg_h2[43:43+7])
    h2_10 = np.mean(avg_h2[50:50+7])
    h2_11 = np.mean(avg_h2[57:])

    h2_vec = np.array([h2_00, h2_01, h2_10, h2_11])

    avg_h12 = np.mean(shots_q0q1, axis=1)
    matrix = np.array(
        [[1, 1, 1, 1],
         [1, -1, 1, -1],
         [1, 1, -1, -1],
         [1, -1, -1, 1]])
    h12_00 = np.mean(avg_h12[36:36+7])
    h12_01 = np.mean(avg_h12[43:43+7])
    h12_10 = np.mean(avg_h12[50:50+7])
    h12_11 = np.mean(avg_h12[57:])

    h12_vec = np.array([h12_00, h12_01, h12_10, h12_11])

    np.concatenate((h1_vec, h2_vec, h12_vec), axis=0)

    np.shape(avg_h1)
    avg_h12 = np.mean(shots_q0q1, axis=1)

    #############################
    # Linear inversion tomo #
    #############################

    measurements_tomo = (
        np.array([avg_h1[0:36], avg_h2[0:36], avg_h12[0:36]])).flatten()  # 108 x 1
    # get the calibration points by averaging over the five measurements taken
    # knowing the initial state we put in
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

    # calculate the tomo
    tomo = TomoAnalysis_JointRO(
        measurements_cal, measurements_tomo, n_qubits=2, n_quadratures=3)
    # operators are expectation values of Pauli operators, rho is density mat
    (operators, rho) = tomo.execute_linear_tomo()

    # Making  the second figure, LI tomo
    fig2 = plt.figure(figsize=(15, 5))
    ax = fig2.add_subplot(121)
    # ax.set_xticks(np.arange(0,15))
    # ax.set_xticklabels(tomo.get_basis_labels(2)[1:])
    # ax.bar(range(len(operators)-1),operators[1:])
    if target_cardinal != None:
        fidelity = calc_fid2_cardinal(operators, target_cardinal)
        # cardinal_labels = ['ZZ', 'ZX', 'ZY']  # fixme
        target_state = target_cardinal
        target_expectations = get_cardianal_pauli_exp(target_cardinal)
        plot_target_pauli_set(target_expectations, ax)
    plot_operators(operators, ax)
    ax.set_title('Least squares tomography.')
    ax.set_ylim(-1, 1)
    if verbose > 0:
        print(rho)
    qtp.matrix_histogram_complex(rho, xlabels=['00', '01', '10', '11'],
                                 ylabels=['00', '01', '10', '11'],
                                 fig=fig2, ax=fig2.add_subplot(
        122, projection='3d'))
    purity = (rho*rho).tr()
    print(purity, target_state, fidelity)
    msg = 'Purity: {:.3f}\nTarget state {}\nFidelity to target {:.3f}'.format(
        purity, target_state, fidelity)
    ax.text(1, -.5, msg)
    # extra_text_info = AnchoredText(msg, loc='lower left')
    # ax.add_artist(extra_text_info)

    # fidelity to target cardinal
    # fidelity to target bell

    figname = 'LI-Tomography_Exp_{}.{}'.format(exp_name, fig_format)
    fig2.suptitle(exp_name+' ' + tomo_scans.TD_timestamps[0], size=16)
    if save_figures:
        savename = os.path.abspath(os.path.join(
            savefolder, figname))
        fig2.savefig(savename, format=fig_format)
    # FIXME: Here the Fidelity and purity should be added to the figure
    if close_fig:
        plt.close(fig2)
    #############################
    # MLE reconstruction
    #############################
    if MLE:
        bell_state_x = 2
        # mle reconstruction of density matrix
        rho_2 = tomo.execute_max_likelihood(ftol=0.000001, xtol=0.0001)
        # reconstructing the pauli vector
    #     print(rho_2.shape,pauli[0].shape)
        if verbose > 1:
            print(rho_2)
        if verbose > 0:
            print('Purity %.3f' % (rho_2*rho_2).tr())
        # calculates the Pauli operator expectation values based on the matrix
        # operators_mle = np.zeros(16, dtype=np.complex128)
        # pauli = [qtp.identity(2), qtp.sigmaz(), qtp.sigmax(), qtp.sigmay()]
        # for k in range(16):
        #     i = int(k % 4)
        #     j = int(((k - i)/4) % 4)
        #     operators_mle[k] = (rho_2*qtp.tensor(pauli[i], pauli[j])).tr()
        operators_mle = pauli_ops_from_density_matrix(rho_2)
    #     fid_sq = calc_fid1_bell(rho_2,bell_state_x)*100.
        if verbose > 0:
            print(operators_mle)
        print(rho_2)
        print("MLE", operators_mle)
        fid_p = calc_fid2_bell(np.real(operators_mle), bell_state_x)*100.
        print('Fidelity (product): %.1f %%' % fid_p)
    #     print('Fidelity (sqrt): %.1f %%'%fid_sq)
        # plot the data

        # Figure 3 MLE reconstruction
        fig3 = plt.figure(figsize=(15, 5))
        ax = fig3.add_subplot(121)

        purity = (rho_2*rho_2).tr()
        if target_cardinal != None:
            fidelity_mle = calc_fid2_cardinal(operators_mle, target_cardinal)
            target_state = target_cardinal
        print(purity, target_state, fidelity_mle)
        msg = 'Purity: {:.3f}\nTarget state {}\nFidelity to target {:.3f}'.format(
            purity, target_state, fidelity_mle)
        ax.text(1, -.5, msg)

        # ax.set_xticks(np.arange(0,15))
        # ax.set_xticklabels(tomo.get_basis_labels(2)[1:])
        # ax.bar(range(len(operators)-1),operators[1:])
        plot_operators(operators_mle, ax)
        ax.set_title('%s Max likelihood reconstructed tomography.' %
                     tomo_scans.TD_timestamps[0])
        ax.set_ylim(-1, 1)
        qtp.matrix_histogram_complex(rho_2, xlabels=['00', '01', '10', '11'],
                                     ylabels=['00', '01', '10', '11'],
                                     fig=fig3,
                                     ax=fig3.add_subplot(122, projection='3d'))

        figname = 'MLE-Tomography_Exp_{}.{}'.format(exp_name, fig_format)
        fig3.suptitle(exp_name+' ' + tomo_scans.TD_timestamps[0], size=16)
        if save_figures:
            savename = os.path.abspath(os.path.join(
                savefolder, figname))
            fig3.savefig(savename, format=fig_format)
        if close_fig:
            plt.close(fig3)

    # Fidelity angle sweep
    # theta_vec = np.linspace(0., 2*np.pi, 100)
    # fid_vec = np.zeros(theta_vec.shape)
    # for i, theta in enumerate(theta_vec):
    #     fid_vec[i] = calc_fid2_bell_theta(operators_mle, theta, target_bell)

    # fig = plt.figure(figsize=(8, 5))
    # ax = fig.add_subplot(111)
    # ax.plot(theta_vec, fid_vec)
    # label_str = 'MAX Fidelity %.3f at %.1f deg' % (
    #     np.max(fid_vec), theta_vec[np.argmax(fid_vec)]*180./np.pi)
    # ax.plot(theta_vec[np.argmax(fid_vec)], np.max(
    #     fid_vec), 'o', label=label_str)
    # ax.legend(loc='best')
    # ax.set_ylabel('Fidelity')
    # ax.set_xlabel('Phase (rad)')
    # ax.set_ylim(0, 1)
    # ax.set_xlim(0, 2*np.pi)
    # ax.set_title('%s Angle Sweep for best Bell-state fidelity' %
    #              tomo_scans.TD_timestamps[0])
    # save_name = os.path.abspath(
    #     save_folder+'\\%s_angle_sweep.PNG' % tomo_scans.TD_timestamps[0])
#     fig.savefig(save_name, format=save_format)
    # return rho_2, operators_mle, theta_vec[np.argmax(fid_vec)], np.max(fid_vec)
        return fidelity, fidelity_mle
    return fidelity
