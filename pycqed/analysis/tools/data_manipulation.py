"""
Part of the 'new' analysis toolbox.
This should contain all the functions that where previously/are now contained
in modules/analysis/analysis_toolbox.py in the Analysis tools section.
To give a feeling of what functions belong here, the old a_tools can be
roughly split into
- file-handling tools
- data manipulation tools
- plotting tools

"""
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def count_rounds_to_error(series):
    """
    returns the index of the first entry that is different
    from the initial value.

    input:
        series : array or list
    output:
        rounds_since_change: list

    Returns NAN if no error is found
    NOTE: superceded by count_rtf_and_term_cond()
    """
    last_s = series[0]
    for i, s in enumerate(series):
        if s == last_s:
            last_s = s
        else:
            return i
    print("Warning did not find any error")
    return np.NAN


def count_rtf_and_term_cond(
    series, only_count_min_1=False, return_termination_condition=True
):
    """
    returns the index of the first entry that is different
    from the initial value.

    input:
        series : array or list
    output:
        rounds to failure (int), termination condition (string)

    Returns the lenght of the timetrace +1  if no error is found
    """
    rtf = len(series) + 1
    termination_condition = None
    initial_s = series[0]

    for i, s in enumerate(series):
        if s != initial_s:
            rtf = i
            if i == len(series) - 1:
                # If termination occurs at last entry it is not possible
                # to determine the cause of termination (note this should be
                # a low probability event)
                termination_condition = "unknown"
            elif series[i + 1] == s:
                termination_condition = "double event"
            elif series[i + 1] != s:
                termination_condition = "single event"
            break
    if only_count_min_1:
        if initial_s == 1:
            rtf = 1
    if rtf == len(series) + 1:
        print("Warning did not find a termination event")
    if return_termination_condition:
        return rtf, termination_condition
    else:
        return rtf


def count_rounds_since_flip(series):
    """
    Used to extract number of consecutive elements that are identical

    input:
        series : array or list
    output:
        rounds_since_change: list
    """
    round_since_last_change = 0  # start at zero because
    last_s = series[0]
    rounds_since_change = []
    for i, s in enumerate(series):
        if s == last_s:
            round_since_last_change += 1
        else:
            rounds_since_change.append(round_since_last_change)
            round_since_last_change = 1
        last_s = s
    return rounds_since_change


def count_rounds_since_flip_split(series):
    """
    Used to extract rounds since flip in a binary sequence flipping between
    +1 and -1.

    input:
        series : array or list containing entries +1 and -1
    output:
        rounds_between_flips_m_to_p : list of consecutive entries in +1
        rounds_between_flips_p_to_m : list of consecutive entries in -1
    """
    nr_rounds_since_last_flip = 1
    last_s = +1
    rounds_between_flips_p_to_m = []
    rounds_between_flips_m_to_p = []
    for i, s in enumerate(series):

        if s == last_s:
            nr_rounds_since_last_flip += 1
        else:
            if s == +1:
                rounds_between_flips_m_to_p.append(nr_rounds_since_last_flip)
            elif s == -1:
                rounds_between_flips_p_to_m.append(nr_rounds_since_last_flip)
            else:
                raise ValueError(
                    "Unexpected value in series," + " expect only +1 and -1"
                )
            nr_rounds_since_last_flip = 1
        last_s = s
    return rounds_between_flips_m_to_p, rounds_between_flips_p_to_m


def binary_derivative(series):
    """
    Used to extract transitions between flipping and non-flipping
    part of data traces.

    When there is no change the value is 0.
    If there is a change the value is 1.
    """
    d_series = np.array(
        [0 if series[i + 1] == series[i] else 1 for i in range(len(series) - 1)]
    )
    return d_series


def binary_derivative_old(series):
    """
    Used to extract transitions between flipping and non-flipping
    part of data traces.
    """
    d_series = np.array(
        [1 if series[i + 1] == series[i] else -1 for i in range(len(series) - 1)]
    )
    return d_series


def binary_derivative_2D(data_array, axis=0):
    """
    Used to extract transitions between flipping and non-flipping
    part of data traces along a certain axis
    """
    if axis == 0:
        dd_array = np.array([binary_derivative(line) for line in data_array])
    elif axis == 1:
        dd_array = np.array([binary_derivative(line) for line in data_array.T]).T
    return dd_array


def butterfly_data_binning(Z, initial_state=0):
    """
    notation of coefficients
    Pjk_i = P(1st msmt outcome, 2nd msmt outcome, _ input state)
    epsj_i = eps(1st post msmst state, _input state)
    """
    if initial_state == 0:  # measurement induced excitation
        # first is declared second is input state
        eps0_0 = np.mean([1 if s == 1 else 0 for s in Z[:, 0]])
        eps1_0 = 1 - eps0_0

        P00_0 = np.mean([1 if (s_row[:2] == [1.0, 1.0]).all() else 0 for s_row in Z[:]])
        P01_0 = np.mean(
            [1 if (s_row[:2] == [1.0, -1.0]).all() else 0 for s_row in Z[:]]
        )
        P10_0 = np.mean(
            [1 if (s_row[:2] == [-1.0, 1.0]).all() else 0 for s_row in Z[:]]
        )
        P11_0 = np.mean(
            [1 if (s_row[:2] == [-1.0, -1.0]).all() else 0 for s_row in Z[:]]
        )
        return {
            "eps0_0": eps0_0,
            "eps1_0": eps1_0,
            "P00_0": P00_0,
            "P01_0": P01_0,
            "P10_0": P10_0,
            "P11_0": P11_0,
        }
    else:  # measurement induced relaxation
        # first is declared second is input state
        eps0_1 = np.mean([1 if s == 1 else 0 for s in Z[:, 0]])
        eps1_1 = 1 - eps0_1

        P00_1 = np.mean([1 if (s_row[:2] == [1.0, 1.0]).all() else 0 for s_row in Z[:]])
        P01_1 = np.mean(
            [1 if (s_row[:2] == [1.0, -1.0]).all() else 0 for s_row in Z[:]]
        )
        P10_1 = np.mean(
            [1 if (s_row[:2] == [-1.0, 1.0]).all() else 0 for s_row in Z[:]]
        )
        P11_1 = np.mean(
            [1 if (s_row[:2] == [-1.0, -1.0]).all() else 0 for s_row in Z[:]]
        )
        return {
            "eps0_1": eps0_1,
            "eps1_1": eps1_1,
            "P00_1": P00_1,
            "P01_1": P01_1,
            "P10_1": P10_1,
            "P11_1": P11_1,
        }


def butterfly_matrix_inversion(exc_coeffs, rel_coeffs):

    # combines all coeffs in a single dictionary
    rel_coeffs.update(exc_coeffs)
    coeffs = rel_coeffs
    matr = [[coeffs["eps0_0"], coeffs["eps0_1"]], [coeffs["eps1_0"], coeffs["eps1_1"]]]
    inv_matr = np.linalg.inv(matr)
    P_vec = [coeffs["P00_0"], coeffs["P01_0"]]
    eps_vec = np.dot(inv_matr, P_vec)
    [eps00_0, eps01_0] = eps_vec

    P_vec = [coeffs["P10_0"], coeffs["P11_0"]]
    eps_vec = np.dot(inv_matr, P_vec)
    [eps10_0, eps11_0] = eps_vec

    matr = [[coeffs["eps0_1"], coeffs["eps0_0"]], [coeffs["eps1_1"], coeffs["eps1_0"]]]
    inv_matr = np.linalg.inv(matr)
    P_vec = [coeffs["P00_1"], coeffs["P01_1"]]
    eps_vec = np.dot(inv_matr, P_vec)
    [eps01_1, eps00_1] = eps_vec

    P_vec = [coeffs["P10_1"], coeffs["P11_1"]]
    eps_vec = np.dot(inv_matr, P_vec)
    [eps11_1, eps10_1] = eps_vec

    return {
        "eps00_0": eps00_0,
        "eps01_0": eps01_0,
        "eps10_0": eps10_0,
        "eps11_0": eps11_0,
        "eps00_1": eps00_1,
        "eps01_1": eps01_1,
        "eps10_1": eps10_1,
        "eps11_1": eps11_1,
    }


def digitize(
    data, threshold: float, one_larger_than_threshold: bool = True, zero_state: int = -1
):
    """
    This funciton digitizes 2D arrays. When using postselection,
    first postselect if threshold for postslection is
    conservative than the threshold for digitization.

    Args:
        one_larger_than_threshold case :
            if True returns +1 for values above threshold and (zero_state) for
            values below threshold, the inverse if False.
        zero_state (int) : how to note the zero_state, this should be either
            -1 (eigenvalue) or 0 (ground state).

    """
    if one_larger_than_threshold:
        data_digitized = np.asarray(
            [
                [1 if d_element >= threshold else zero_state for d_element in d_row]
                for d_row in data
            ]
        )
    else:
        data_digitized = np.asarray(
            [
                [1 if d_element <= threshold else zero_state for d_element in d_row]
                for d_row in data
            ]
        )
    return data_digitized


def get_post_select_indices(thresholds, init_measurements, positive_case=True):
    post_select_indices = []
    for th, in_m in zip(thresholds, init_measurements):
        if positive_case:
            post_select_indices.append(np.where(in_m > th)[0])
        else:
            post_select_indices.append(np.where(in_m < th)[0])

    post_select_indices = np.unique(np.concatenate(post_select_indices))
    return post_select_indices


def postselect(data, threshold, positive_case=True):
    data_postselected = []
    if positive_case:
        for data_row in data:
            if data_row[0] <= threshold:
                data_postselected.append(data_row)
    else:
        for data_row in data:
            if data_row[0] >= threshold:
                data_postselected.append(data_row)
    return np.asarray(data_postselected)


def count_error_fractions(trace):
    """
    The counters produce the same results as the CBox counters in
    CBox.get_qubit_state_log_counters().
    Requires a boolean array or an array of ints as input.
    """
    no_err_counter = 0
    single_err_counter = 0
    double_err_counter = 0
    zero_counter = 0
    one_counter = 0

    for i in range(len(trace)):
        if i < (len(trace) - 1):
            if trace[i] == trace[i + 1]:
                # A single error is associated with a qubit error
                single_err_counter += 1
                if i < (len(trace) - 2):
                    if trace[i] == trace[i + 2]:
                        # If there are two errors in a row this is associated with
                        # a RO error, this counter must be substracted from the
                        # single counter
                        double_err_counter += 1
            else:
                no_err_counter += 1
        if trace[i] == 1:
            zero_counter += 1
        else:
            one_counter += 1

    return (
        no_err_counter,
        single_err_counter,
        double_err_counter,
        zero_counter,
        one_counter,
    )


def mark_errors_flipping(events):
    """
    Marks error fractions
    """
    single_errors = np.zeros(len(events) - 1)
    double_errors = np.zeros(len(events) - 2)

    for i in range(len(events) - 1):
        # A single error is associated with a qubit error
        if events[i] == events[i + 1]:
            single_errors[i] = 1
            if i < (len(events) - 2):
                # two identical outcomes equal to one
                if events[i] == events[i + 2]:
                    double_errors[i] = 1
    return single_errors, double_errors


def mark_errors_constant(events):
    """
    Marks error fractions
    """
    single_errors = np.zeros(len(events) - 1)
    double_errors = np.zeros(len(events) - 2)

    for i in range(len(events) - 1):
        # A single error is associated with a qubit error
        if events[i] != events[i + 1]:
            single_errors[i] = 1
            if i < (len(events) - 2):
                # two identical outcomes equal to one
                if events[i + 1] != events[i + 2]:
                    double_errors[i] = 1
    return single_errors, double_errors


def mark_errors_FB_to_ground(events):
    """
    Marks error fractions
    """
    single_errors = np.zeros(len(events) - 1)
    double_errors = np.zeros(len(events) - 2)

    for i in range(len(events) - 1):
        # A single error is associated with a qubit error
        if events[i] == 1:
            single_errors[i] = 1
            if i < (len(events) - 2):
                # two identical outcomes equal to one
                if events[i + 1] == 1:
                    double_errors[i] = 1
    return single_errors, double_errors


def flatten_2D_histogram(H, xedges, yedges):
    """
    Flattens a 2D histogram in preparation for fitting.
    Input is the output of the np.histogram2d() command.

    Inputs
        H: 2D array of counts of shape (yrows, xcols)
        xedges: 1D array of bin-edges along x
        yedges: ""

    Returns
        H_flat: flattened array of length (yrows*xcols)
        x_tiled_flat: 1D array of bin-x-centers of length (yrows*xcols)
        y_rep_flat: 1D array of bin-x-centers of length (yrows*xcols)
    """
    # Transpose because Histogram is H(yrows, xcols)
    H_flat = H.T.flatten()
    xstep = (xedges[1] - xedges[0]) / 2
    ystep = (yedges[1] - yedges[0]) / 2
    x = xedges[:-1] + xstep
    y = yedges[:-1] + ystep
    nr_rows = len(y)
    nr_cols = len(x)
    # tiling and rep is to make the indices match with the locations in the
    # 2D grid of H
    x_tiled_flat = np.tile(x, nr_cols)
    y_rep_flat = np.repeat(y, nr_rows)

    return H_flat, x_tiled_flat, y_rep_flat


def reject_outliers(data, m=6.0):
    """
    Reject outliers function from stack overflow
    http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return data[s < m]


def rotation_matrix(angle, as_array=False):
    """
    Returns a 2x2 rotation matrix based on an angle in degrees.

    rot_mat * vec shape(2,1) rotates the vector clockwise
    """
    rot_mat = np.matrix(
        [
            [np.cos(2 * np.pi * angle / 360), -np.sin(2 * np.pi * angle / 360)],
            [np.sin(2 * np.pi * angle / 360), np.cos(2 * np.pi * angle / 360)],
        ]
    )
    if as_array:
        rot_mat = np.array(rot_mat)
    return rot_mat


def rotate_complex(complex_number, angle, deg=True):
    """
    Rotates a complex number by an angle specified in degrees
    """
    if deg:
        angle = angle / 360 * 2 * np.pi
    rotated_number = complex_number * np.exp(1j * angle)
    return rotated_number


def get_outliers_fwd(x, threshold, plot_hist=False, ax=None):
    dif = np.zeros(x.shape)
    dif[1:] = x[1:] - x[:-1]
    if plot_hist:
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
        ax.hist(np.abs(dif[~np.isnan(dif)]))
        ax.axvline(threshold, color="k")
    return np.where(
        np.logical_or((np.abs(dif) > threshold), (np.isnan(x))), True, False
    )


def get_outliers_bwd(x, threshold, plot_hist=False, ax=None):
    x = x[::-1]
    dif = np.zeros(x.shape)
    dif[1:] = x[1:] - x[:-1]
    if plot_hist:
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
        ax.hist(np.abs(dif[~np.isnan(dif)]))
        ax.axvline(threshold, color="k")
    return np.where(
        np.logical_or((np.abs(dif) > threshold), (np.isnan(x))), True, False
    )


def get_outliers(x, threshold):
    return np.logical_and(
        get_outliers_fwd(x, threshold), get_outliers_bwd(x, threshold)[::-1]
    )


def get_generations_by_index(generation_indices, array):
    """
    Given a list of indices
    """
    generation_indices[0]
    generations = []

    current_gen_indices = deque([0], maxlen=2)
    for idx in generation_indices:
        current_gen_indices.append(int(idx))
        generations.append(array[current_gen_indices[0] : current_gen_indices[1]])

    return generations


def get_generation_means(generation_indices, array):
    generations = get_generations_by_index(generation_indices, array)
    means = [np.mean(gen) for gen in generations]
    return means


def filter_resonator_visibility(
    x,
    y,
    z,
    deg=True,
    cutoff_factor=0,
    sav_windowlen_factor=None,
    sav_polorder=4,
    hipass_left=0.05,
    hipass_right=0.1,
    **kw
):

    """
    Filters resonator-dac sweeps on phase data to show only the resonator dips
    and remove background
    Inputs
        Optional
        deg: True/False if data is in degree or rad. Expected rad and deg
        standard range is [-pi, pi] or [-180, 180], respectively.

        sav_windowlen_factor: Length of the window for the Savitsky-Golay
        filter, expressed in a factor of the total length of thedata
        -> value between 0 and 1. Default is 0.1

        sav_polorder: Polyorder of Savitsky-Golay filter. Default is 4

        hipass_left, hipass_right: Factor cutoff of all data of
        left and right side of high pass, as data is not shifted
    """
    # cutoff in frequency space optional if high freq data is noisy
    cutoff = round(len(x) * (1 - cutoff_factor))
    x_cut = x[:cutoff]
    restruct = []
    # Go line by line for filtering
    for i in range(len(z[0])):
        ppcut = z[:cutoff, i]
        # Pick type of data (deg or rad) to unwrap
        # Expected rad standard range is [-pi,pi]
        if deg:
            ppcut_rad = np.deg2rad(ppcut) + np.pi
            ppcut_unwrap = np.unwrap(ppcut_rad)
        else:
            ppcut_unwrap = np.unwrap(ppcut)
        # Remove linear offset of unwrap
        [a, b] = np.polyfit(x_cut, ppcut_unwrap, deg=1)
        fit = a * x_cut + b
        reduced = ppcut_unwrap - fit
        # Use Savitsky-Golay filter
        if sav_windowlen_factor is None:
            sav_windowlen_factor = round(0.1 * len(x) / 2) * 2 + 1
        red_filt = sig.savgol_filter(
            reduced, window_length=sav_windowlen_factor, polyorder=sav_polorder
        )

        # Flatten curve by removing the filtered signal
        flat = reduced - red_filt

        # Poor-mans high pass filter using
        # FFT -> Removing frequency components --> IFFT
        llcut_f = np.fft.fft(flat)
        left = round(hipass_left * len(x))
        right = round(hipass_right * len(x))

        # Cut and apply 'highpass filter'
        llcut_f[:left] = [0] * left
        llcut_f[-1 - right : -1] = [0] * right

        # Convert back to frequency domain
        llcut_if = np.fft.ifft(llcut_f)

        # Build new 2D dataset
        restruct.append(llcut_if)

    # Absolute value of complex IFFT result
    restruct = np.abs(np.array(restruct))

    return restruct


def populations_using_rate_equations(
    SI: np.array, SX: np.array, V0: float, V1: float, V2: float
):
    """
    Calculate populations using reference voltages.

    Parameters:
    -----------
    SI : array
        signal value for signal with I (Identity) added
    SX : array
        signal value for signal with X (π-pulse) added
    V0 : float
        Reference signal level for 0-state (calibration point).
    V1 : float
        Reference signal level for 1-state (calibration point).
    V2 : float
        Reference signal level for 2-state (calibration point).


    Returns:
    --------
    P0 : array
        population of the |0> state
    P1 : array
        population of the |1> state
    P2 : array
        population of the |2> state
    M_inv : 2D array
        Matrix inverse to find populations

    Based on equation (S1) from Asaad & Dickel et al. npj Quant. Info. (2016)

    To quantify leakage, we monitor the populations Pi of the three lowest
    energy states (i ∈ {0, 1, 2}) and calculate the average
    values <Pi>. To do this, we calibrate the average signal levels Vi for
    the transmons in level i, and perform each measurement twice, the second
    time with an added final π pulse on the 0–1 transition. This final π
    pulse swaps P0 and P1, leaving P2 unaffected. Under the assumption that
    higher levels are unpopulated (P0 +P1 +P2 = 1),

        [V0 −V2,   V1 −V2] [P0]  = [S −V2]
        [V1 −V2,   V0 −V2] [P1]  = [S' −V2]

    where S (S') is the measured signal level without (with) final π pulse.
    The populations are extracted by matrix inversion.
    """
    M = np.array([[V0 - V2, V1 - V2], [V1 - V2, V0 - V2]])
    M_inv = np.linalg.inv(M)

    # using lists instead of preallocated array allows this to work
    # with ufloats
    P0 = []
    P1 = []
    for i, (sI, sX) in enumerate(zip(SI, SX)):
        p0, p1 = np.dot(np.array([sI - V2, sX - V2]), M_inv)
        P0.append(p0)
        P1.append(p1)

    # [2020-07-09 Victor] added compatibility with inputing complex IQ
    # voltages in order to make rates equation work properly with "optimal IQ"
    # RO mode, regardless of the orientation of the blobs on the IQ-plane

    # There might be small imaginary part here in the cases where the measured
    # SI or SX are points outside the the triangle formed by the calibration
    # points
    P0 = np.real(P0)
    P1 = np.real(P1)

    P2 = 1 - P0 - P1

    return P0, P1, P2, M_inv
