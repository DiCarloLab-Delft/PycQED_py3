"""
File containing analyses for readout.
This includes
    - readout discrimination analysis
    - single shot readout analysis
    - multiplexed readout analysis (to be updated!)

Originally written by Adriaan, updated/rewritten by Rene May 2018
"""
import itertools
from copy import deepcopy
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lmfit
import numpy as np
from scipy.optimize import minimize
import scipy.constants as spconst

import pycqed.analysis.fitting_models as fit_mods
from pycqed.analysis.fitting_models import ro_gauss, ro_CDF, ro_CDF_discr, gaussian_2D, gauss_2D_guess, gaussianCDF, ro_double_gauss_guess
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis_v2.simple_analysis as sa
from pycqed.analysis.tools.plotting import SI_val_to_msg_str, \
    set_xlabel, set_ylabel, set_cbarlabel, flex_colormesh_plot_vs_xy
from pycqed.analysis_v2.tools.plotting import scatter_pnts_overlay
import pycqed.analysis.tools.data_manipulation as dm_tools
from pycqed.utilities.general import int2base
from pycqed.utilities.general import format_value_string
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
import pycqed.measurement.hdf5_data as h5d
import os
# import xarray as xr

# ADD from pagani detached. RDC 16-02-2023

def create_xr_data(proc_data_dict,qubit,timestamp):

    NUM_STATES = 3

    calibration_data = np.array([[proc_data_dict[f"{comp}{state}"] for comp in ("I", "Q")] for state in range(NUM_STATES)])

    arr_data = []
    for state_ind in range(NUM_STATES):
        state_data = []
        for meas_ind in range(1, NUM_STATES + 1):
            ind = NUM_STATES*state_ind + meas_ind
            meas_data = [proc_data_dict[f"{comp}M{ind}"] for comp in ("I", "Q")]
            state_data.append(meas_data)
        arr_data.append(state_data)

    butterfly_data = np.array(arr_data)

    NUM_STATES, NUM_MEAS_INDS, NUM_COMPS, NUM_SHOTS = butterfly_data.shape

    assert NUM_COMPS == 2

    STATES = list(range(0, NUM_STATES))
    MEAS_INDS = list(range(1, NUM_MEAS_INDS + 1))
    SHOTS = list(range(1, NUM_SHOTS + 1))

    exp_dataset = xr.Dataset(
        data_vars = dict(
            calibration = (["state", "comp", "shot"], calibration_data),
            characterization = (["state", "meas_ind", "comp", "shot"], butterfly_data),
        ),
        coords = dict(
            state = STATES,
            meas_ind = MEAS_INDS,
            comp = ["in-phase", "quadrature"],
            shot = SHOTS,
            qubit = qubit,
        ),
        attrs = dict(
            description = "Qutrit measurement butterfly data.",
            timestamp = timestamp,
        )
    )

    return exp_dataset

def QND_qutrit_anaylsis(NUM_STATES,
                        NUM_OUTCOMES,
                        STATES,
                        OUTCOMES,
                        char_data,
                        cal_data,
                        fid,
                        accuracy,
                        timestamp,
                        classifier):

    data = char_data.stack(stacked_dim = ("state", "meas_ind", "shot"))
    data = data.transpose("stacked_dim", "comp")

    predictions = classifier.predict(data)
    outcome_vec = xr.DataArray(
        data = predictions,
        dims = ["stacked_dim"],
        coords = dict(stacked_dim = data.stacked_dim)
    )

    digital_char_data = outcome_vec.unstack()
    matrix = np.zeros((NUM_STATES, NUM_OUTCOMES, NUM_OUTCOMES), dtype=float)

    for state in STATES:
        meas_arr = digital_char_data.sel(state=state)
        postsel_arr = meas_arr.where(meas_arr[0] == 0, drop=True)

        for first_out in OUTCOMES:
            first_cond = xr.where(postsel_arr[1] == first_out, 1, 0)

            for second_out in OUTCOMES:
                second_cond = xr.where(postsel_arr[2] == second_out, 1, 0)

                sel_shots = first_cond & second_cond
                joint_prob = np.mean(sel_shots)

                matrix[state, first_out, second_out] = joint_prob

    joint_probs = xr.DataArray(
        matrix,
        dims = ["state", "meas_1", "meas_2"],
        coords = dict(
            state = STATES,
            meas_1 = OUTCOMES,
            meas_2 = OUTCOMES,
        )
    )

    num_constraints = NUM_STATES
    num_vars = NUM_OUTCOMES * (NUM_STATES ** 2)

    def opt_func(variables, obs_probs, num_states: int) -> float:
        meas_probs = variables.reshape(num_states, num_states, num_states)
        probs = np.einsum("ijk, klm -> ijl", meas_probs, meas_probs)
        return np.linalg.norm(np.ravel(probs - obs_probs))

    cons_mat = np.zeros((num_constraints, num_vars), dtype=int)
    num_cons_vars = int(num_vars / num_constraints)
    for init_state in range(NUM_STATES):
        var_ind = init_state * num_cons_vars
        cons_mat[init_state, var_ind : var_ind + num_cons_vars] = 1

    constraints = {"type": "eq", "fun": lambda variables: cons_mat @ variables - 1}
    bounds = opt.Bounds(0, 1)

    ideal_probs = np.zeros((NUM_STATES, NUM_OUTCOMES, NUM_OUTCOMES), dtype=float)
    for state in range(NUM_STATES):
        ideal_probs[state, state, state] = 1
    init_vec = np.ravel(ideal_probs)

    result = opt.basinhopping(
        opt_func,
        init_vec,
        niter=500,
        minimizer_kwargs=dict(
            args=(joint_probs.data, NUM_STATES),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            tol=1e-12,
            options=dict(
                maxiter=10000,
            )
        )
    )
    # if not result.success:
    #     raise ValueError("Unsuccessful optimization, please check parameters and tolerance.")
    res_data = result.x.reshape((NUM_STATES, NUM_OUTCOMES, NUM_STATES))

    meas_probs = xr.DataArray(
        res_data,
        dims = ["input_state", "outcome", "output_state"],
        coords = dict(
            input_state = STATES,
            outcome = OUTCOMES,
            output_state = STATES,
        )
    )

    pred_joint_probs = np.einsum("ijk, klm -> ijl", meas_probs, meas_probs)

    true_vals = np.ravel(joint_probs)
    pred_vals = np.ravel(pred_joint_probs)

    ms_error = mean_squared_error(true_vals, pred_vals)
    rms_error = np.sqrt(ms_error)
    ma_error = mean_absolute_error(true_vals, pred_vals)

    print(f"RMS error of the optimised solution: {rms_error}")
    print(f"MA error of the optimised solution: {ma_error}")

    num_vars = 3 * (NUM_STATES ** 2)
    num_constraints = 3 * NUM_STATES

    def opt_func(
        variables,
        obs_probs,
        num_states: int,
    ) -> float:
        pre_mat, ro_mat, post_mat = variables.reshape(3, num_states, num_states)
        probs = np.einsum("ih, hm, ho -> imo", pre_mat, ro_mat, post_mat)
        return np.linalg.norm(probs - obs_probs)

    cons_mat = np.zeros((num_constraints, num_vars), dtype=int)
    for op_ind in range(3):
        for init_state in range(NUM_STATES):
            cons_ind = op_ind*NUM_STATES + init_state
            var_ind = (op_ind*NUM_STATES + init_state)*NUM_STATES
            cons_mat[cons_ind, var_ind : var_ind + NUM_STATES] = 1

    ideal_probs = np.tile(np.eye(NUM_STATES), (3, 1))
    init_vec = np.ravel(ideal_probs)

    constraints = {"type": "eq", "fun": lambda variables: cons_mat @ variables - 1}
    bounds = opt.Bounds(0, 1, keep_feasible=True)

    result = opt.basinhopping(
        opt_func,
        init_vec,
        minimizer_kwargs = dict(
            args = (meas_probs.data, NUM_STATES),
            bounds = bounds,
            constraints = constraints,
            method = "SLSQP",
            tol = 1e-12,
            options = dict(
                maxiter = 10000,
            )
        ),
        niter=500
    )


    # if not result.success:
    #     raise ValueError("Unsuccessful optimization, please check parameters and tolerance.")

    pre_trans, ass_errors, post_trans = result.x.reshape((3, NUM_STATES, NUM_STATES))

    pred_meas_probs = np.einsum("ih, hm, ho -> imo", pre_trans, ass_errors, post_trans)

    true_vals = np.ravel(meas_probs)
    pred_vals = np.ravel(pred_meas_probs)

    ms_error = mean_squared_error(true_vals, pred_vals)
    rms_error = np.sqrt(ms_error)
    ma_error = mean_absolute_error(true_vals, pred_vals)

    print(f"RMS error of the optimised solution: {rms_error}")
    print(f"MA error of the optimised solution: {ma_error}")

    QND_state = {}
    for state in STATES:
        state_qnd = np.sum(meas_probs.data[state,:, state])
        QND_state[f'{state}'] = state_qnd

    meas_qnd = np.mean(np.diag(meas_probs.sum(axis=1)))
    meas_qnd

    fit_res = {}
    fit_res['butter_prob'] = pred_meas_probs
    fit_res['mean_QND'] = meas_qnd
    fit_res['state_qnd'] = QND_state
    fit_res['ass_errors'] = ass_errors
    fit_res['qutrit_fidelity'] = accuracy*100
    fit_res['fidelity'] = fid
    fit_res['timestamp'] = timestamp

    # Meas leak rate
    L1 = 100*np.sum(fit_res['butter_prob'][:2,:,2])/2

    # Meas seepage rate
    s = 100*np.sum(fit_res['butter_prob'][2,:,:2])

    fit_res['L1'] = L1
    fit_res['seepage'] = s

    return fit_res

class measurement_butterfly_analysis(ba.BaseDataAnalysis):
    """
    This analysis extracts measurement butter fly
    """
    def __init__(self,
                 qubit:str,
                 t_start: str = None,
                 t_stop: str = None,
                 label: str = '',
                 f_state: bool = False,
                 cycle : int = 6,
                 options_dict: dict = None,
                 extract_only: bool = False,
                 auto=True
                 ):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only)

        self.qubit = qubit
        self.f_state = f_state

        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]
        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {'data': ('Experimental Data/Data', 'dset'),
                      'value_names': ('Experimental Data', 'attr:value_names')}
        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)
        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):

        if self.f_state:
            _cycle = 12
            I0, Q0 = self.raw_data_dict['data'][:,1][9::_cycle], self.raw_data_dict['data'][:,2][9::_cycle]
            I1, Q1 = self.raw_data_dict['data'][:,1][10::_cycle], self.raw_data_dict['data'][:,2][10::_cycle]
            I2, Q2 = self.raw_data_dict['data'][:,1][11::_cycle], self.raw_data_dict['data'][:,2][11::_cycle]
        else:
            _cycle = 8
            I0, Q0 = self.raw_data_dict['data'][:,1][6::_cycle], self.raw_data_dict['data'][:,2][6::_cycle]
            I1, Q1 = self.raw_data_dict['data'][:,1][7::_cycle], self.raw_data_dict['data'][:,2][7::_cycle]
        # Measurement
        IM1, QM1 = self.raw_data_dict['data'][0::_cycle,1], self.raw_data_dict['data'][0::_cycle,2]
        IM2, QM2 = self.raw_data_dict['data'][1::_cycle,1], self.raw_data_dict['data'][1::_cycle,2]
        IM3, QM3 = self.raw_data_dict['data'][2::_cycle,1], self.raw_data_dict['data'][2::_cycle,2]
        IM4, QM4 = self.raw_data_dict['data'][3::_cycle,1], self.raw_data_dict['data'][3::_cycle,2]
        IM5, QM5 = self.raw_data_dict['data'][4::_cycle,1], self.raw_data_dict['data'][4::_cycle,2]
        IM6, QM6 = self.raw_data_dict['data'][5::_cycle,1], self.raw_data_dict['data'][5::_cycle,2]
        # Rotate data
        center_0 = np.array([np.mean(I0), np.mean(Q0)])
        center_1 = np.array([np.mean(I1), np.mean(Q1)])
        if self.f_state:
            IM7, QM7 = self.raw_data_dict['data'][6::_cycle,1], self.raw_data_dict['data'][6::_cycle,2]
            IM8, QM8 = self.raw_data_dict['data'][7::_cycle,1], self.raw_data_dict['data'][7::_cycle,2]
            IM9, QM9 = self.raw_data_dict['data'][8::_cycle,1], self.raw_data_dict['data'][8::_cycle,2]
            center_2 = np.array([np.mean(I2), np.mean(Q2)])
        def rotate_and_center_data(I, Q, vec0, vec1):
            vector = vec1-vec0
            angle = np.arctan(vector[1]/vector[0])
            rot_matrix = np.array([[ np.cos(-angle),-np.sin(-angle)],
                                   [ np.sin(-angle), np.cos(-angle)]])
            # Subtract mean
            proc = np.array((I-(vec0+vec1)[0]/2, Q-(vec0+vec1)[1]/2))
            # Rotate theta
            proc = np.dot(rot_matrix, proc)
            return proc
        # proc cal points
        I0_proc, Q0_proc = rotate_and_center_data(I0, Q0, center_0, center_1)
        I1_proc, Q1_proc = rotate_and_center_data(I1, Q1, center_0, center_1)
        # proc M
        IM1_proc, QM1_proc = rotate_and_center_data(IM1, QM1, center_0, center_1)
        IM2_proc, QM2_proc = rotate_and_center_data(IM2, QM2, center_0, center_1)
        IM3_proc, QM3_proc = rotate_and_center_data(IM3, QM3, center_0, center_1)
        IM4_proc, QM4_proc = rotate_and_center_data(IM4, QM4, center_0, center_1)
        IM5_proc, QM5_proc = rotate_and_center_data(IM5, QM5, center_0, center_1)
        IM6_proc, QM6_proc = rotate_and_center_data(IM6, QM6, center_0, center_1)
        if np.mean(I0_proc) > np.mean(I1_proc):
            I0_proc *= -1
            I1_proc *= -1
            IM1_proc *= -1
            IM2_proc *= -1
            IM3_proc *= -1
            IM4_proc *= -1
            IM5_proc *= -1
            IM6_proc *= -1
        # Calculate optimal threshold
        ubins_A_0, ucounts_A_0 = np.unique(I0_proc, return_counts=True)
        ubins_A_1, ucounts_A_1 = np.unique(I1_proc, return_counts=True)
        ucumsum_A_0 = np.cumsum(ucounts_A_0)
        ucumsum_A_1 = np.cumsum(ucounts_A_1)
        # merge |0> and |1> shot bins
        all_bins_A = np.unique(np.sort(np.concatenate((ubins_A_0, ubins_A_1))))
        # interpolate cumsum for all bins
        int_cumsum_A_0 = np.interp(x=all_bins_A, xp=ubins_A_0, fp=ucumsum_A_0, left=0)
        int_cumsum_A_1 = np.interp(x=all_bins_A, xp=ubins_A_1, fp=ucumsum_A_1, left=0)
        norm_cumsum_A_0 = int_cumsum_A_0/np.max(int_cumsum_A_0)
        norm_cumsum_A_1 = int_cumsum_A_1/np.max(int_cumsum_A_1)
        # Calculating threshold
        F_vs_th = (1-(1-abs(norm_cumsum_A_0-norm_cumsum_A_1))/2)
        opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
        opt_idx = int(round(np.average(opt_idxs)))
        threshold = all_bins_A[opt_idx]
        # fidlity calculation from cal point
        P0_dig = np.array([ 0 if s<threshold else 1 for s in I0_proc ])
        P1_dig = np.array([ 0 if s<threshold else 1 for s in I1_proc ])
        M1_dig = np.array([ 0 if s<threshold else 1 for s in IM1_proc ])
        M2_dig = np.array([ 0 if s<threshold else 1 for s in IM2_proc ])
        M3_dig = np.array([ 0 if s<threshold else 1 for s in IM3_proc ])
        M4_dig = np.array([ 0 if s<threshold else 1 for s in IM4_proc ])
        M5_dig = np.array([ 0 if s<threshold else 1 for s in IM5_proc ])
        M6_dig = np.array([ 0 if s<threshold else 1 for s in IM6_proc ])
        Fidelity = (np.mean(1-P0_dig) + np.mean(P1_dig))/2
        # postselected init for zero
        I_mask = np.ones(len(IM1_proc))
        Q_mask = np.ones(len(QM1_proc))
        IM1_init =   np.array([ +1 if shot < threshold else np.nan for shot in IM1_proc ], dtype=float)
        QM1_init =   np.array([ +1 if shot < threshold else np.nan for shot in QM1_proc ], dtype=float)
        IM2_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in IM2_proc ], dtype=float)
        QM2_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in QM2_proc ], dtype=float)
        IM3_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in IM3_proc ], dtype=float)
        QM3_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in QM3_proc ], dtype=float)
        I_mask *= IM1_init
        Q_mask *= QM1_init
        IM2_dig_ps *= I_mask
        QM2_dig_ps *= Q_mask
        IM3_dig_ps *= I_mask
        QM3_dig_ps *= Q_mask
        fraction_discarded_I = np.sum(np.isnan(I_mask))/len(I_mask)
        fraction_discarded_Q = np.sum(np.isnan(Q_mask))/len(Q_mask)
        # # postselectted init for 1
        I1_mask = np.ones(len(IM4_proc))
        Q1_mask = np.ones(len(QM4_proc))
        IM4_init =  np.array([ +1 if shot < threshold else np.nan for shot in IM4_proc ], dtype=float)
        QM4_init =  np.array([ +1 if shot < threshold else np.nan for shot in QM4_proc ], dtype=float)
        IM5_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in IM5_proc ], dtype=float)
        QM5_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in QM5_proc ], dtype=float)
        IM6_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in IM6_proc ], dtype=float)
        QM6_dig_ps = np.array([ +1 if shot < threshold else -1 for shot in QM6_proc ], dtype=float)
        I1_mask *= IM4_init
        Q1_mask *= QM4_init
        IM5_dig_ps *= I1_mask
        QM5_dig_ps *= Q1_mask
        IM6_dig_ps *= I1_mask
        QM6_dig_ps *= Q1_mask
        fraction_discarded_I1 = np.sum(np.isnan(I1_mask))/len(I1_mask)
        fraction_discarded_Q1 = np.sum(np.isnan(Q1_mask))/len(Q1_mask)
        # digitize data
        M1_dig_ps = (1-IM1_init)/2 # turn into binary
        M2_dig_ps = (1-IM2_dig_ps)/2 # turn into binary
        M3_dig_ps = (1-IM3_dig_ps)/2 # turn into binary
        M4_dig_ps = (1-IM4_init)/2 # turn into binary
        M5_dig_ps = (1-IM5_dig_ps)/2 # turn into binary
        M6_dig_ps = (1-IM6_dig_ps)/2 # turn into binary
        #############
        ## p for prep 0
        #############
        p0_init_prep0 = 1-np.nanmean(M1_dig_ps)
        p1_init_prep0 = np.nanmean(M1_dig_ps)
        p0_M1_prep0 = 1-np.nanmean(M2_dig_ps)
        p1_M1_prep0 = np.nanmean(M2_dig_ps)
        #############
        ## p for prep 1
        #############
        p0_init_prep1 = 1-np.nanmean(M4_dig_ps)
        p1_init_prep1 = np.nanmean(M4_dig_ps)
        p0_M1_prep1 = 1-np.nanmean(M5_dig_ps)
        p1_M1_prep1 = np.nanmean(M5_dig_ps)
        #############
        ## pij calculation
        #############
        p00_prep0 = np.nanmean(1-np.logical_or(M2_dig_ps, M3_dig_ps))
        p00_prep1 = np.nanmean(1-np.logical_or(M5_dig_ps, M6_dig_ps))
        p01_prep0 = np.nanmean(1-np.logical_or(np.logical_not(M3_dig), M2_dig))
        p01_prep1 = np.nanmean(1-np.logical_or(np.logical_not(M6_dig), M5_dig))
        p10_prep0 = np.nanmean(1-np.logical_or(np.logical_not(M2_dig), M3_dig))
        p10_prep1 = np.nanmean(1-np.logical_or(np.logical_not(M5_dig), M6_dig))
        p11_prep0 = np.nanmean(np.logical_and(M2_dig, M3_dig))
        p11_prep1 = np.nanmean(np.logical_and(M5_dig, M6_dig))

        #############
        ## eps calculations
        #############
        ## calculation e +1 when prep 0
        t1 = ((p1_M1_prep1*p00_prep0)/(p0_M1_prep1))-p01_prep0
        t2 = ((p0_M1_prep0*p1_M1_prep1)/(p0_M1_prep1)-p1_M1_prep0)
        ep1_0_0 =  t1/t2
        ep1_1_0 = p0_M1_prep0 - ep1_0_0
        ## calculation e -1 when prep 0
        t1 = ((p1_M1_prep1*p10_prep0)/(p0_M1_prep1))-p11_prep0
        t2 = ((p0_M1_prep0*p1_M1_prep1)/(p0_M1_prep1)-p1_M1_prep0)
        en1_0_0 =  t1/t2
        en1_1_0 = p1_M1_prep0 - en1_0_0
        ## calculation e +1 when prep 1
        t1 = ((p1_M1_prep1*p00_prep1)/(p0_M1_prep1))-p01_prep1
        t2 = ((p0_M1_prep0*p1_M1_prep1)/(p0_M1_prep1)-p1_M1_prep0)
        ep1_0_1 =  t1/t2
        ep1_1_1 = p0_M1_prep1 - ep1_0_1
        ## calculation e -1 when prep 1
        t1 = ((p1_M1_prep1*p10_prep1)/(p0_M1_prep1))-p11_prep1
        t2 = ((p0_M1_prep0*p1_M1_prep1)/(p0_M1_prep1)-p1_M1_prep0)
        en1_0_1 =  t1/t2
        en1_1_1 = p1_M1_prep1 - en1_0_1
        # save proc dict
        # cal points
        self.proc_data_dict["I0"], self.proc_data_dict["Q0"] = I0, Q0
        self.proc_data_dict["I1"], self.proc_data_dict["Q1"] = I1, Q1
        if self.f_state:
            self.proc_data_dict["I2"], self.proc_data_dict["Q2"] = I2, Q2
            self.proc_data_dict["center_2"] = center_2
        # shots
        self.proc_data_dict["IM1"], self.proc_data_dict["QM1"] = (IM1,QM1,)  # used for postselection for 0 state
        self.proc_data_dict["IM2"], self.proc_data_dict["QM2"] = IM2, QM2
        self.proc_data_dict["IM3"], self.proc_data_dict["QM3"] = IM3, QM3
        self.proc_data_dict["IM4"], self.proc_data_dict["QM4"] = (IM4,QM4,)  # used for postselection for 1 state
        self.proc_data_dict["IM5"], self.proc_data_dict["QM5"] = IM5, QM5
        self.proc_data_dict["IM6"], self.proc_data_dict["QM6"] = IM6, QM6
        if self.f_state:
            self.proc_data_dict["IM7"], self.proc_data_dict["QM7"] = (IM7,QM7,)  # used for postselection for 2 state
            self.proc_data_dict["IM8"], self.proc_data_dict["QM8"] = IM8, QM8
            self.proc_data_dict["IM9"], self.proc_data_dict["QM9"] = IM9, QM9
        # center of the prepared states
        self.proc_data_dict["center_0"] = center_0
        self.proc_data_dict["center_1"] = center_1
        if self.f_state:
            self.proc_data_dict["center_2"] = center_2
        self.proc_data_dict['I0_proc'], self.proc_data_dict['Q0_proc'] = I0_proc, Q0_proc
        self.proc_data_dict['I1_proc'], self.proc_data_dict['Q1_proc'] = I1_proc, Q1_proc
        self.proc_data_dict['threshold'] = threshold
        self.qoi = {}
        self.qoi['p00_0'] = p00_prep0
        self.qoi['p00_1'] = p00_prep1
        self.qoi['p11_0'] = p11_prep0
        self.qoi['p11_1'] = p11_prep1
        self.qoi['p01_0'] = p01_prep0
        self.qoi['p01_1'] = p01_prep1
        self.qoi['p10_0'] = p10_prep0
        self.qoi['p10_1'] = p10_prep1
        self.qoi['Fidelity'] = Fidelity
        self.qoi['ps_fraction_0'] = fraction_discarded_I
        self.qoi['ps_fraction_1'] = fraction_discarded_I1
        self.qoi['ep1_0_0'] = ep1_0_0
        self.qoi['ep1_1_0'] = ep1_1_0
        self.qoi['en1_0_0'] = en1_0_0
        self.qoi['en1_1_0'] = en1_1_0
        self.qoi['ep1_0_1'] = ep1_0_1
        self.qoi['ep1_1_1'] = ep1_1_1
        self.qoi['en1_0_1'] = en1_0_1
        self.qoi['en1_1_1'] = en1_1_1
        if self.f_state:
            ## QND for qutrit RO
            dataset = create_xr_data(self.proc_data_dict,self.qubit,self.timestamp)
            NUM_STATES = NUM_OUTCOMES = 3
            STATES = np.arange(NUM_STATES)
            OUTCOMES = np.arange(NUM_OUTCOMES)
            data_subset = dataset.sel(state=STATES)
            char_data = data_subset.characterization.copy(deep=True)
            cal_data = data_subset.calibration.copy(deep=True)
            classifier = LinearDiscriminantAnalysis(
                solver="svd",
                shrinkage=None,
                tol=1e-4)
            train_data = cal_data.stack(stacked_dim = ("state", "shot"))
            train_data = train_data.transpose("stacked_dim", "comp")
            classifier.fit(train_data, train_data.state)
            accuracy = classifier.score(train_data, train_data.state)
            print(f"Total classifier accuracy on the calibration dataset: {accuracy*100:.3f} %")
            fid = {}
            for state in STATES:
                data_subset = train_data.sel(state = state)
                subset_labels = xr.full_like(data_subset.shot, state)
                state_accuracy = classifier.score(data_subset, subset_labels)
                fid[f'{state}'] = state_accuracy*100
                print(f"State {state} accuracy on the calibration dataset: {state_accuracy*100:.3f} %")
            fit_res = QND_qutrit_anaylsis(NUM_STATES,
                                          NUM_OUTCOMES,
                                          STATES,
                                          OUTCOMES,
                                          char_data,
                                          cal_data,
                                          fid,
                                          accuracy,
                                          self.timestamp,
                                          classifier)
            dec_bounds = _decision_boundary_points(classifier.coef_, classifier.intercept_)
            self.proc_data_dict['classifier'] = classifier
            self.proc_data_dict['dec_bounds'] = dec_bounds
            self.qoi['fit_res'] = fit_res

    def prepare_plots(self):

        self.axs_dict = {}
        fig, axs = plt.subplots(figsize=(4,2), dpi=200)
        # fig.patch.set_alpha(0)
        self.axs_dict['msmt_butterfly'] = axs
        self.figs['msmt_butterfly'] = fig
        self.plot_dicts['msmt_butterfly'] = {
            'plotfn': plot_msmt_butterfly,
            'ax_id': 'msmt_butterfly',
            'I0_proc': self.proc_data_dict['I0_proc'],
            'I1_proc': self.proc_data_dict['I1_proc'],
            'threshold': self.proc_data_dict['threshold'],
            'Fidelity': self.qoi['Fidelity'],
            'qubit': self.qubit,
            'qoi' : self.qoi,
            'timestamp': self.timestamp
        }
        if self.f_state:
            _shots_0 = np.hstack((np.array([self.proc_data_dict['I0']]).T,
                                  np.array([self.proc_data_dict['Q0']]).T))
            _shots_1 = np.hstack((np.array([self.proc_data_dict['I1']]).T,
                                  np.array([self.proc_data_dict['Q1']]).T))
            _shots_2 = np.hstack((np.array([self.proc_data_dict['I2']]).T,
                                  np.array([self.proc_data_dict['Q2']]).T))
            fig = plt.figure(figsize=(8,4), dpi=100)
            axs = [fig.add_subplot(121)]
            # fig.patch.set_alpha(0)
            self.axs_dict[f'IQ_readout_histogram_{self.qubit}'] = axs[0]
            self.figs[f'IQ_readout_histogram_{self.qubit}'] = fig
            self.plot_dicts[f'IQ_readout_histogram_{self.qubit}'] = {
                'plotfn': ssro_IQ_projection_plotfn2,
                'ax_id': f'IQ_readout_histogram_{self.qubit}',
                'shots_0': _shots_0,
                'shots_1': _shots_1,
                'shots_2': _shots_2,
                'classifier': self.proc_data_dict['classifier'],
                'dec_bounds': self.proc_data_dict['dec_bounds'],
                'Fid_dict': self.qoi['fit_res']['fidelity'],
                'qubit': self.qubit,
                'timestamp': self.timestamp
            }

    def run_post_extract(self):
        self.prepare_plots()  # specify default plots
        self.plot(key_list='auto', axs_dict=self.axs_dict)  # make the plots
        if self.options_dict.get('save_figs', False):
            self.save_figures(
                close_figs=self.options_dict.get('close_figs', True),
                tag_tstamp=self.options_dict.get('tag_tstamp', True))

def plot_msmt_butterfly(I0_proc, I1_proc,
                        threshold,Fidelity,
                        timestamp,
                        qubit,
                        qoi,
                        ax, **kw
                        ):
    # plot histogram of rotated shots
    fig = ax.get_figure()
    rang = np.max(list(np.abs(I0_proc))+list(np.abs(I1_proc)))
    ax.hist(I0_proc, range=[-rang, rang], bins=100, color='C0', alpha=.75, label='ground')
    ax.hist(I1_proc, range=[-rang, rang], bins=100, color='C3', alpha=.75, label='excited')
    ax.axvline(threshold, ls='--', lw=.5, color='k', label='threshold')
    ax.legend(loc='upper right', fontsize=3, frameon=False)
    ax.set_yticks([])
    ax.set_title('Rotated data', fontsize=9)
    ax.set_xlabel('Integrated voltage (mV)', size=8)

    if 0:
        # Write results
        text = '\n'.join(('Assignment Fid:',
                          rf'$\mathrm{"{F_{g}}"}:\:\:\:\:\:\:{fit_res["fidelity"]["0"]:.2f}$%',
                          rf'$\mathrm{"{F_{e}}"}:\:\:\:\:\:\:{fit_res["fidelity"]["1"]:.2f}$%',
                          rf'$\mathrm{"{F_{f}}"}:\:\:\:\:\:\:{fit_res["fidelity"]["2"]:.2f}$%',
                          rf'$\mathrm{"{F_{avg}}"}:\:\:\:{fit_res["qutrit_fidelity"]:.2f}$%',
                          '',
                          'QNDness:',
                          rf'$\mathrm{"{QND_{g}}"}:\:\:\:\:\:\:{fit_res["state_qnd"]["0"]*100:.2f}$%',
                          rf'$\mathrm{"{QND_{e}}"}:\:\:\:\:\:\:{fit_res["state_qnd"]["1"]*100:.2f}$%',
                          rf'$\mathrm{"{QND_{f}}"}:\:\:\:\:\:\:{fit_res["state_qnd"]["2"]*100:.2f}$%',
                          rf'$\mathrm{"{QND_{avg}}"}:\:\:\:{fit_res["mean_QND"]*100:.2f}$%',
                          '',
                          'L1 & seepage',
                          rf'$\mathrm{"{L_{1}}"}:\:\:\:{fit_res["L1"]:.2f}$%',
                          rf'$\mathrm{"{L_{2}}"}:\:\:\:{fit_res["seepage"]:.2f}$%',
                          ))
    else:
        text = '\n'.join(('Assignment Fid:',
                  rf'$\mathrm{"{F_{a}}"}:\:\:\:{qoi["Fidelity"]*100:.2f}$%',
                  '',
                  'QNDness:',
                  rf'$\mathrm{"{QND_{g}}"}:\:\:\:\:\:\:{qoi["p00_0"]*100:.2f}$%',
                  rf'$\mathrm{"{QND_{e}}"}:\:\:\:\:\:\:{qoi["p11_1"]*100:.2f}$%',
                  rf'$\mathrm{"{QND_{avg}}"}:\:\:\:{(qoi["p00_0"] + qoi["p11_1"]) * 0.5 *100:.2f}$%',
                  ))
    props = dict(boxstyle='round', facecolor='gray', alpha=0.15)
    ax.text(1.05, 0.99, text, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)
    fig.suptitle(f'Qubit {qubit}\n{timestamp}', y=1.1, size=9)

def ssro_IQ_projection_plotfn2(
    shots_0, 
    shots_1,
    shots_2,
    classifier,
    dec_bounds,
    Fid_dict,
    timestamp,
    qubit, 
    ax, **kw):
    fig = ax.get_figure()
    axs = fig.get_axes()
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:10000,0], shots_0[:10000,1], '.', color='C0', alpha=0.025)
    axs[0].plot(shots_1[:10000,0], shots_1[:10000,1], '.', color='C3', alpha=0.025)
    axs[0].plot(shots_2[:10000,0], shots_2[:10000,1], '.', color='C2', alpha=0.025)
    axs[0].plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
    axs[0].plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    axs[0].plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    axs[0].plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
    axs[0].plot(popt_0[1], popt_0[2], 'x', color='white')
    axs[0].plot(popt_1[1], popt_1[2], 'x', color='white')
    axs[0].plot(popt_2[1], popt_2[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_1)
    circle_2 = Ellipse((popt_2[1], popt_2[2]),
                      width=4*popt_2[3], height=4*popt_2[4],
                      angle=-popt_2[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_2)
    # Plot classifier zones
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    Lim_points = {}
    for bound in ['01', '12', '02']:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot 0 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['02']]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['02'], Lim_points['12']]
    _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    for bound in ['01', '12', '02']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
    axs[0].legend(frameon=False, loc=1)
    axs[0].set_xlabel('Integrated voltage I')
    axs[0].set_ylabel('Integrated voltage Q')
    axs[0].set_title(f'{timestamp}\nIQ plot qubit {qubit}')
    # Write fidelity textbox
    _f_avg = np.mean((Fid_dict["0"], Fid_dict["1"], Fid_dict["2"]))
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]:.1f}%',
                      f'$F_f$ : {Fid_dict["2"]:.1f}%',
                      f'$F_\mathrm{"{avg}"}$ : {_f_avg:.1f}%'))
    props = dict(boxstyle='round', facecolor='gray', alpha=.2)
    axs[0].text(1.05, 1, text, transform=axs[0].transAxes,
                verticalalignment='top', bbox=props)


############# END #################################

class Singleshot_Readout_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', do_fitting: bool = True,
                 data_file_path: str=None,
                 options_dict: dict=None, auto=True,
                 **kw):
        '''
        options dict options:
            'fixed_p10'   fixes p(e|g)  res_exc (do not vary in fit)
            'fixed_p01' : fixes p(g|pi) mmt_rel (do not vary in fit)
            'auto_rotation_angle' : (bool) automatically find the I/Q mixing angle
            'rotation_angle' : manually define the I/Q mixing angle (ignored if auto_rotation_angle is set to True)
            'nr_bins' : number of bins to use for the histograms
            'post_select' : (bool) sets on or off the post_selection based on an initialization measurement (needs to be in agreement with nr_samples)
            'post_select_threshold' : (float) threshold used for post-selection (only activated by above parameter)
            'nr_samples' : amount of different samples (e.g. ground and excited = 2 and with post-selection = 4)
            'sample_0' : index of first sample (ground-state)
            'sample_1' : index of second sample (first excited-state)
            'max_datapoints' : maximum amount of datapoints for culumative fit
            'log_hist' : use log scale for the y-axis of the 1D histograms
            'verbose' : see BaseDataAnalysis
            'presentation_mode' : see BaseDataAnalysis
            see BaseDataAnalysis for more.
        '''
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label, do_fitting=do_fitting,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         **kw)
        self.single_timestamp = True
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values',
            'value_names': 'value_names',
            'value_units': 'value_units'}

        self.numeric_params = []

        # Determine the default for auto_rotation_angle
        man_angle = self.options_dict.get('rotation_angle', False) is False
        self.options_dict['auto_rotation_angle'] = self.options_dict.get(
            'auto_rotation_angle', man_angle)

        self.predict_qubit_temp = 'predict_qubit_temp' in self.options_dict
        if self.predict_qubit_temp:
            self.qubit_freq = self.options_dict['qubit_freq']

        if auto:
            self.run_analysis()

    def process_data(self):
        """
        Responsible for creating the histograms based on the raw data
        """
        post_select = self.options_dict.get('post_select', False)
        post_select_threshold = self.options_dict.get(
            'post_select_threshold', 0)
        nr_samples = self.options_dict.get('nr_samples', 2)
        sample_0 = self.options_dict.get('sample_0', 0)
        sample_1 = self.options_dict.get('sample_1', 1)
        nr_bins = int(self.options_dict.get('nr_bins', 100))

        ######################################################
        #  Separating data into shots for 0 and shots for 1  #
        ######################################################
        meas_val = self.raw_data_dict['measured_values']
        unit = self.raw_data_dict['value_units'][0]
        # loop through channels
        shots = np.zeros((2, len(meas_val),), dtype=np.ndarray)
        for j, dat in enumerate(meas_val):
            assert unit == self.raw_data_dict['value_units'][
                j], 'The channels have been measured using different units. This is not supported yet.'
            sh_0, sh_1 = get_shots_zero_one(
                dat, post_select=post_select, nr_samples=nr_samples,
                post_select_threshold=post_select_threshold,
                sample_0=sample_0, sample_1=sample_1)
            shots[0, j] = sh_0
            shots[1, j] = sh_1
        #shots = np.array(shots, dtype=float)

        # Do we have two quadratures?
        if len(meas_val) == 2:
            ########################################################
            # Bin the data in 2D, to calculate the opt. angle
            ########################################################
            data_range_x = (np.min([np.min(b) for b in shots[:, 0]]),
                            np.max([np.max(b) for b in shots[:, 0]]))
            data_range_y = (np.min([np.min(b) for b in shots[:, 1]]),
                            np.max([np.max(b) for b in shots[:, 1]]))
            data_range_xy = (data_range_x, data_range_y)
            nr_bins_2D = int(self.options_dict.get(
                'nr_bins_2D', 6*np.sqrt(nr_bins)))
            H0, xedges, yedges = np.histogram2d(x=shots[0, 0],
                                                y=shots[0, 1],
                                                bins=nr_bins_2D,
                                                range=data_range_xy)
            H1, xedges, yedges = np.histogram2d(x=shots[1, 0],
                                                y=shots[1, 1],
                                                bins=nr_bins_2D,
                                                range=data_range_xy)
            binsize_x = xedges[1] - xedges[0]
            binsize_y = yedges[1] - yedges[0]
            bin_centers_x = xedges[:-1] + binsize_x
            bin_centers_y = yedges[:-1] + binsize_y
            self.proc_data_dict['2D_histogram_x'] = bin_centers_x
            self.proc_data_dict['2D_histogram_y'] = bin_centers_y
            self.proc_data_dict['2D_histogram_z'] = [H0, H1]

            # Find and apply the effective/rotated integrated voltage
            angle = self.options_dict.get('rotation_angle', 0)
            auto_angle = self.options_dict.get('auto_rotation_angle', True)
            if auto_angle:
                ##########################################
                #  Determining the rotation of the data  #
                ##########################################
                gauss2D_model_0 = lmfit.Model(gaussian_2D,
                                              independent_vars=['x', 'y'])
                gauss2D_model_1 = lmfit.Model(gaussian_2D,
                                              independent_vars=['x', 'y'])
                guess0 = gauss_2D_guess(model=gauss2D_model_0, data=H0.transpose(),
                                        x=bin_centers_x, y=bin_centers_y)
                guess1 = gauss_2D_guess(model=gauss2D_model_1, data=H1.transpose(),
                                        x=bin_centers_x, y=bin_centers_y)

                x2d = np.array([bin_centers_x]*len(bin_centers_y))
                y2d = np.array([bin_centers_y]*len(bin_centers_x)).transpose()
                fitres0 = gauss2D_model_0.fit(data=H0.transpose(), x=x2d, y=y2d,
                                              **guess0)
                fitres1 = gauss2D_model_1.fit(data=H1.transpose(),  x=x2d, y=y2d,
                                              **guess1)

                fr0 = fitres0.best_values
                fr1 = fitres1.best_values
                x0 = fr0['center_x']
                x1 = fr1['center_x']
                y0 = fr0['center_y']
                y1 = fr1['center_y']

                self.proc_data_dict['IQ_pos'] = [[x0, x1], [y0, y1]]
                dx = x1 - x0
                dy = y1 - y0
                mid = [x0 + dx/2, y0 + dy/2]
                angle = np.arctan2(dy, dx)
            else:
                mid = [0, 0]

            if self.verbose:
                ang_deg = (angle*180/np.pi)
                print('Mixing I/Q channels with %.3f degrees ' % ang_deg +
                      #'around point (%.2f, %.2f)%s'%(mid[0], mid[1], unit) +
                      ' to obtain effective voltage.')

            self.proc_data_dict['raw_offset'] = [*mid, angle]
            # create matrix
            rot_mat = [[+np.cos(-angle), -np.sin(-angle)],
                       [+np.sin(-angle), +np.cos(-angle)]]
            # rotate data accordingly
            eff_sh = np.zeros(len(shots[0]), dtype=np.ndarray)
            eff_sh[0] = np.dot(rot_mat[0], shots[0])  # - mid
            eff_sh[1] = np.dot(rot_mat[0], shots[1])  # - mid
        else:
            # If we have only one quadrature, use that (doh!)
            eff_sh = shots[:, 0]

        self.proc_data_dict['all_channel_int_voltages'] = shots
        # self.raw_data_dict['value_names'][0]
        self.proc_data_dict['shots_xlabel'] = 'Effective integrated Voltage'
        self.proc_data_dict['shots_xunit'] = unit
        self.proc_data_dict['eff_int_voltages'] = eff_sh
        self.proc_data_dict['nr_shots'] = [len(eff_sh[0]), len(eff_sh[1])]
        sh_min = min(np.min(eff_sh[0]), np.min(eff_sh[1]))
        sh_max = max(np.max(eff_sh[0]), np.max(eff_sh[1]))
        data_range = (sh_min, sh_max)
        eff_sh_sort = [np.sort(eff_sh[0]), np.sort(eff_sh[1])]
        x0, n0 = np.unique(eff_sh_sort[0], return_counts=True)
        cumsum0 = np.cumsum(n0)
        x1, n1 = np.unique(eff_sh_sort[1], return_counts=True)
        cumsum1 = np.cumsum(n1)

        self.proc_data_dict['cumsum_x'] = [x0, x1]
        self.proc_data_dict['cumsum_y'] = [cumsum0, cumsum1]

        all_x = np.unique(np.sort(np.concatenate((x0, x1))))
        md = self.options_dict.get('max_datapoints', 1000)
        if len(all_x) > md:
            all_x = np.linspace(*data_range, md)
        ecumsum0 = np.interp(x=all_x, xp=x0, fp=cumsum0, left=0)
        necumsum0 = ecumsum0/np.max(ecumsum0)
        ecumsum1 = np.interp(x=all_x, xp=x1, fp=cumsum1, left=0)
        necumsum1 = ecumsum1/np.max(ecumsum1)

        self.proc_data_dict['cumsum_x_ds'] = all_x
        self.proc_data_dict['cumsum_y_ds'] = [ecumsum0, ecumsum1]
        self.proc_data_dict['cumsum_y_ds_n'] = [necumsum0, necumsum1]

        ##################################
        #  Binning data into histograms  #
        ##################################
        h0, bin_edges = np.histogram(eff_sh[0], bins=nr_bins,
                                     range=data_range)
        h1, bin_edges = np.histogram(eff_sh[1], bins=nr_bins,
                                     range=data_range)
        self.proc_data_dict['hist'] = [h0, h1]
        binsize = (bin_edges[1] - bin_edges[0])
        self.proc_data_dict['bin_edges'] = bin_edges
        self.proc_data_dict['bin_centers'] = bin_edges[:-1]+binsize
        self.proc_data_dict['binsize'] = binsize

        #######################################################
        #  Threshold and fidelity based on culmulative counts #
        #######################################################
        # Average assignment fidelity: F_ass = (P01 - P10 )/2
        # where Pxy equals probability to measure x when starting in y
        F_vs_th = (1-(1-abs(necumsum0 - necumsum1))/2)
        opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
        opt_idx = int(round(np.average(opt_idxs)))
        self.proc_data_dict['F_assignment_raw'] = F_vs_th[opt_idx]
        self.proc_data_dict['threshold_raw'] = all_x[opt_idx]

    def prepare_fitting(self):
        ###################################
        #  First fit the histograms (PDF) #
        ###################################
        self.fit_dicts = OrderedDict()

        bin_x = self.proc_data_dict['bin_centers']
        bin_xs = [bin_x, bin_x]
        bin_ys = self.proc_data_dict['hist']
        m = lmfit.model.Model(ro_gauss)
        m.guess = ro_double_gauss_guess.__get__(m, m.__class__)
        params = m.guess(x=bin_xs, data=bin_ys,
                         fixed_p01=self.options_dict.get('fixed_p01', False),
                         fixed_p10=self.options_dict.get('fixed_p10', False))
        res = m.fit(x=bin_xs, data=bin_ys, params=params)

        self.fit_dicts['shots_all_hist'] = {
            'model': m,
            'fit_xvals': {'x': bin_xs},
            'fit_yvals': {'data': bin_ys},
            'guessfn_pars': {'fixed_p01': self.options_dict.get('fixed_p01', False),
                             'fixed_p10': self.options_dict.get('fixed_p10', False)},
        }

        ###################################
        #  Fit the CDF                    #
        ###################################
        m_cul = lmfit.model.Model(ro_CDF)
        cdf_xs = self.proc_data_dict['cumsum_x_ds']
        cdf_xs = [np.array(cdf_xs), np.array(cdf_xs)]
        cdf_ys = self.proc_data_dict['cumsum_y_ds']
        cdf_ys = [np.array(cdf_ys[0]), np.array(cdf_ys[1])]
        #cul_res = m_cul.fit(x=cdf_xs, data=cdf_ys, params=res.params)
        cum_params = res.params
        cum_params['A_amplitude'].value = np.max(cdf_ys[0])
        cum_params['A_amplitude'].vary = False
        cum_params['B_amplitude'].value = np.max(cdf_ys[1])
        cum_params['A_amplitude'].vary = False # FIXME: check if correct
        self.fit_dicts['shots_all'] = {
            'model': m_cul,
            'fit_xvals': {'x': cdf_xs},
            'fit_yvals': {'data': cdf_ys},
            'guess_pars': cum_params,
        }

    def analyze_fit_results(self):
        # Create a CDF based on the fit functions of both fits.
        fr = self.fit_res['shots_all']
        bv = fr.best_values

        # best values new
        bvn = deepcopy(bv)
        bvn['A_amplitude'] = 1
        bvn['B_amplitude'] = 1

        def CDF(x):
            return ro_CDF(x=x, **bvn)

        def CDF_0(x):
            return CDF(x=[x, x])[0]

        def CDF_1(x):
            return CDF(x=[x, x])[1]

        def infid_vs_th(x):
            cdf = ro_CDF(x=[x, x], **bvn)
            return (1-np.abs(cdf[0] - cdf[1]))/2

        self._CDF_0 = CDF_0
        self._CDF_1 = CDF_1
        self._infid_vs_th = infid_vs_th

        thr_guess = (3*bv['B_center'] - bv['A_center'])/2
        opt_fid = minimize(infid_vs_th, thr_guess)

        # for some reason the fit sometimes returns a list of values
        if isinstance(opt_fid['fun'], float):
            self.proc_data_dict['F_assignment_fit'] = (1-opt_fid['fun'])
        else:
            self.proc_data_dict['F_assignment_fit'] = (1-opt_fid['fun'])[0]

        self.proc_data_dict['threshold_fit'] = opt_fid['x'][0]

        # Calculate the fidelity of both

        ###########################################
        #  Extracting the discrimination fidelity #
        ###########################################

        def CDF_0_discr(x):
            return gaussianCDF(x, amplitude=1,
                               mu=bv['A_center'], sigma=bv['A_sigma'])

        def CDF_1_discr(x):
            return gaussianCDF(x, amplitude=1,
                               mu=bv['B_center'], sigma=bv['B_sigma'])

        def disc_infid_vs_th(x):
            cdf0 = gaussianCDF(x, amplitude=1, mu=bv['A_center'],
                               sigma=bv['A_sigma'])
            cdf1 = gaussianCDF(x, amplitude=1, mu=bv['B_center'],
                               sigma=bv['B_sigma'])
            return (1-np.abs(cdf0 - cdf1))/2

        self._CDF_0_discr = CDF_0_discr
        self._CDF_1_discr = CDF_1_discr
        self._disc_infid_vs_th = disc_infid_vs_th

        opt_fid_discr = minimize(disc_infid_vs_th, thr_guess)

        # for some reason the fit sometimes returns a list of values
        if isinstance(opt_fid_discr['fun'], float):
            self.proc_data_dict['F_discr'] = (1-opt_fid_discr['fun'])
        else:
            self.proc_data_dict['F_discr'] = (1-opt_fid_discr['fun'])[0]

        self.proc_data_dict['threshold_discr'] = opt_fid_discr['x'][0]

        fr = self.fit_res['shots_all']
        bv = fr.params
        self.proc_data_dict['residual_excitation'] = bv['A_spurious'].value
        self.proc_data_dict['relaxation_events'] = bv['B_spurious'].value

        ###################################
        #  Save quantities of interest.   #
        ###################################
        self.proc_data_dict['quantities_of_interest'] = {
            'SNR': self.fit_res['shots_all'].params['SNR'].value,
            'F_d': self.proc_data_dict['F_discr'],
            'F_a': self.proc_data_dict['F_assignment_raw'],
            'residual_excitation': self.proc_data_dict['residual_excitation'],
            'relaxation_events':
                self.proc_data_dict['relaxation_events']
        }
        self.qoi = self.proc_data_dict['quantities_of_interest']

    def prepare_plots(self):
        # Did we load two voltage components (shall we do 2D plots?)
        two_dim_data = len(
            self.proc_data_dict['all_channel_int_voltages'][0]) == 2

        eff_voltage_label = self.proc_data_dict['shots_xlabel']
        eff_voltage_unit = self.proc_data_dict['shots_xunit']
        x_volt_label = self.raw_data_dict['value_names'][0]
        x_volt_unit = self.raw_data_dict['value_units'][0]
        if two_dim_data:
            y_volt_label = self.raw_data_dict['value_names'][1]
            y_volt_unit = self.raw_data_dict['value_units'][1]
        z_hist_label = 'Counts'
        labels = self.options_dict.get(
            'preparation_labels', ['|g> prep.', '|e> prep.'])
        label_0 = labels[0]
        label_1 = labels[1]
        title = ('\n' + self.timestamps[0] + ' - "' +
                 self.raw_data_dict['measurementstring'] + '"')

        # 1D histograms (PDF)
        log_hist = self.options_dict.get('log_hist', False)
        bin_x = self.proc_data_dict['bin_edges']
        bin_y = self.proc_data_dict['hist']
        self.plot_dicts['hist_0'] = {
            'title': 'Binned Shot Counts' + title,
            'ax_id': '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': bin_x,
            'yvals': bin_y[0],
            'xwidth': self.proc_data_dict['binsize'],
            'bar_kws': {'log': log_hist, 'alpha': .4, 'facecolor': 'C0',
                        'edgecolor': 'C0'},
            'setlabel': label_0,
            'xlabel': eff_voltage_label,
            'xunit': eff_voltage_unit,
            'ylabel': z_hist_label,
        }

        self.plot_dicts['hist_1'] = {
            'ax_id': '1D_histogram',
            'plotfn': self.plot_bar,
            'xvals': bin_x,
            'yvals': bin_y[1],
            'xwidth': self.proc_data_dict['binsize'],
            'bar_kws': {'log': log_hist, 'alpha': .3, 'facecolor': 'C3',
                        'edgecolor': 'C3'},
            'setlabel': label_1,
            'do_legend': True,
            'xlabel': eff_voltage_label,
            'xunit': eff_voltage_unit,
            'ylabel': z_hist_label,
        }
        if log_hist:
            self.plot_dicts['hist_0']['yrange'] = (0.5, 1.5*np.max(bin_y[0]))
            self.plot_dicts['hist_1']['yrange'] = (0.5, 1.5*np.max(bin_y[1]))

        # CDF
        cdf_xs = self.proc_data_dict['cumsum_x']
        cdf_ys = self.proc_data_dict['cumsum_y']
        cdf_ys[0] = cdf_ys[0]/np.max(cdf_ys[0])
        cdf_ys[1] = cdf_ys[1]/np.max(cdf_ys[1])
        xra = (bin_x[0], bin_x[-1])

        self.plot_dicts['cdf_shots_0'] = {
            'title': 'Culmulative Shot Counts (no binning)' + title,
            'ax_id': 'cdf',
            'plotfn': self.plot_line,
            'xvals': cdf_xs[0],
            'yvals': cdf_ys[0],
            'setlabel': label_0,
            'xrange': xra,
            'line_kws': {'color': 'C0', 'alpha': 0.3},
            'marker': '',
            'xlabel': eff_voltage_label,
            'xunit': eff_voltage_unit,
            'ylabel': 'Culmulative Counts',
            'yunit': 'norm.',
            'do_legend': True,
        }
        self.plot_dicts['cdf_shots_1'] = {
            'ax_id': 'cdf',
            'plotfn': self.plot_line,
            'xvals': cdf_xs[1],
            'yvals': cdf_ys[1],
            'setlabel': label_1,
            'line_kws': {'color': 'C3', 'alpha': 0.3},
            'marker': '',
            'xlabel': eff_voltage_label,
            'xunit': eff_voltage_unit,
            'ylabel': 'Culmulative Counts',
            'yunit': 'norm.',
            'do_legend': True,
        }

        # Vlines for thresholds
        th_raw = self.proc_data_dict['threshold_raw']
        threshs = [th_raw, ]
        if self.do_fitting:
            threshs.append(self.proc_data_dict['threshold_fit'])
            threshs.append(self.proc_data_dict['threshold_discr'])

        for ax in ['1D_histogram', 'cdf']:
            self.plot_dicts[ax+'_vlines_thresh'] = {
                'ax_id': ax,
                'plotfn': self.plot_vlines_auto,
                'xdata': threshs,
                'linestyles': ['--', '-.', ':'],
                'labels': ['$th_{raw}$', '$th_{fit}$', '$th_{d}$'],
                'colors': ['0.3', '0.5', '0.2'],
                'do_legend': True,
            }

        # 2D Histograms
        if two_dim_data:
            iq_centers = None
            if 'IQ_pos' in self.proc_data_dict and self.proc_data_dict['IQ_pos'] is not None:
                iq_centers = self.proc_data_dict['IQ_pos']
                peak_marker_2D = {
                    'plotfn': self.plot_line,
                    'xvals': iq_centers[0],
                    'yvals': iq_centers[1],
                    'xlabel': x_volt_label,
                    'xunit': x_volt_unit,
                    'ylabel': y_volt_label,
                    'yunit': y_volt_unit,
                    'marker': 'x',
                    'aspect': 'equal',
                    'linestyle': '',
                    'color': 'black',
                    #'line_kws': {'markersize': 1, 'color': 'black', 'alpha': 1},
                    'setlabel': 'Peaks',
                    'do_legend': True,
                }
                peak_marker_2D_rot = deepcopy(peak_marker_2D)
                peak_marker_2D_rot['xvals'] = iq_centers[0]
                peak_marker_2D_rot['yvals'] = iq_centers[1]

            self.plot_dicts['2D_histogram_0'] = {
                'title': 'Raw '+label_0+' Binned Shot Counts' + title,
                'ax_id': '2D_histogram_0',
                # 'plotfn': self.plot_colorxy,
                'plotfn': plot_2D_ssro_histogram,
                'xvals': self.proc_data_dict['2D_histogram_x'],
                'yvals': self.proc_data_dict['2D_histogram_y'],
                'zvals': self.proc_data_dict['2D_histogram_z'][0].T,
                'xlabel': x_volt_label,
                'xunit': x_volt_unit,
                'ylabel': y_volt_label,
                'yunit': y_volt_unit,
                'zlabel': z_hist_label,
                'zunit': '-',
                'cmap': 'Blues',
            }
            if iq_centers is not None:
                dp = deepcopy(peak_marker_2D)
                dp['ax_id'] = '2D_histogram_0'
                self.plot_dicts['2D_histogram_0_marker'] = dp

            self.plot_dicts['2D_histogram_1'] = {
                'title': 'Raw '+label_1+' Binned Shot Counts' + title,
                'ax_id': '2D_histogram_1',
                # 'plotfn': self.plot_colorxy,
                'plotfn': plot_2D_ssro_histogram,
                'xvals': self.proc_data_dict['2D_histogram_x'],
                'yvals': self.proc_data_dict['2D_histogram_y'],
                'zvals': self.proc_data_dict['2D_histogram_z'][1].T,
                'xlabel': x_volt_label,
                'xunit': x_volt_unit,
                'ylabel': y_volt_label,
                'yunit': y_volt_unit,
                'zlabel': z_hist_label,
                'zunit': '-',
                'cmap': 'Reds',
            }
            if iq_centers is not None:
                dp = deepcopy(peak_marker_2D)
                dp['ax_id'] = '2D_histogram_1'
                self.plot_dicts['2D_histogram_1_marker'] = dp

            # Scatter Shots
            volts = self.proc_data_dict['all_channel_int_voltages']

            v_flat = np.concatenate(np.concatenate(volts))
            plot_range = (np.min(v_flat), np.max(v_flat))

            vxr = plot_range
            vyr = plot_range
            self.plot_dicts['2D_shots_0'] = {
                'title': 'Raw Shots' + title,
                'ax_id': '2D_shots',
                'aspect': 'equal',
                'plotfn': self.plot_line,
                'xvals': volts[0][0],
                'yvals': volts[0][1],
                'range': [vxr, vyr],
                'xrange': vxr,
                'yrange': vyr,
                'xlabel': x_volt_label,
                'xunit': x_volt_unit,
                'ylabel': y_volt_label,
                'yunit': y_volt_unit,
                'zlabel': z_hist_label,
                'marker': 'o',
                'linestyle': '',
                'color': 'C0',
                'line_kws': {'markersize': 0.25, 'color': 'C0', 'alpha': 0.5},
                'setlabel': label_0,
                'do_legend': True,
            }
            self.plot_dicts['2D_shots_1'] = {
                'ax_id': '2D_shots',
                'plotfn': self.plot_line,
                'xvals': volts[1][0],
                'yvals': volts[1][1],
                'aspect': 'equal',
                'range': [vxr, vyr],
                'xrange': vxr,
                'yrange': vyr,
                'xlabel': x_volt_label,
                'xunit': x_volt_unit,
                'ylabel': y_volt_label,
                'yunit': y_volt_unit,
                'zlabel': z_hist_label,
                'marker': 'o',
                'linestyle': '',
                'color': 'C3',
                'line_kws': {'markersize': 0.25, 'color': 'C3', 'alpha': 0.5},
                'setlabel': label_1,
                'do_legend': True,
            }
            if iq_centers is not None:
                dp = deepcopy(peak_marker_2D)
                dp['ax_id'] = '2D_shots'
                self.plot_dicts['2D_shots_marker'] = dp 
                self.plot_dicts['2D_shots_marker_line_0']={
                    'plotfn': self.plot_line,
                    'ax_id': '2D_shots',
                    'xvals': [0, iq_centers[0][0]],
                    'yvals': [0, iq_centers[1][0]],
                    'xlabel': x_volt_label,
                    'xunit': x_volt_unit,
                    'ylabel': y_volt_label,
                    'yunit': y_volt_unit,
                    'marker': '',
                    'aspect': 'equal',
                    'linestyle': '--',
                    'color': 'black'
                }
                self.plot_dicts['2D_shots_marker_line_1']={
                    'plotfn': self.plot_line,
                    'ax_id': '2D_shots',
                    'xvals': [0, iq_centers[0][1]],
                    'yvals': [0, iq_centers[1][1]],
                    'xlabel': x_volt_label,
                    'xunit': x_volt_unit,
                    'ylabel': y_volt_label,
                    'yunit': y_volt_unit,
                    'marker': '',
                    'aspect': 'equal',
                    'linestyle': '--',
                    'color': 'black'
                }

        # The cumulative histograms
        #####################################
        # Adding the fits to the figures    #
        #####################################
        if self.do_fitting:
            # todo: add seperate fits for residual and main gaussians
            x = np.linspace(bin_x[0], bin_x[-1], 150)
            para_hist_tmp = self.fit_res['shots_all_hist'].best_values
            para_cdf = self.fit_res['shots_all'].best_values
            para_hist = para_cdf
            para_hist['A_amplitude'] = para_hist_tmp['A_amplitude']
            para_hist['B_amplitude'] = para_hist_tmp['B_amplitude']

            ro_g = ro_gauss(x=[x, x], **para_hist)
            self.plot_dicts['new_fit_shots_0'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': ro_g[0],
                'setlabel': 'Fit '+label_0,
                'line_kws': {'color': 'C0'},
                'marker': '',
                'do_legend': True,
            }
            self.plot_dicts['new_fit_shots_1'] = {
                'ax_id': '1D_histogram',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': ro_g[1],
                'marker': '',
                'setlabel': 'Fit '+label_1,
                'line_kws': {'color': 'C3'},
                'do_legend': True,
            }

            self.plot_dicts['cdf_fit_shots_0'] = {
                'ax_id': 'cdf',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': self._CDF_0(x),
                'setlabel': 'Fit '+label_0,
                'line_kws': {'color': 'C0', 'alpha': 0.8},
                'linestyle': ':',
                'marker': '',
                'do_legend': True,
            }
            self.plot_dicts['cdf_fit_shots_1'] = {
                'ax_id': 'cdf',
                'plotfn': self.plot_line,
                'xvals': x,
                'yvals': self._CDF_1(x),
                'marker': '',
                'linestyle': ':',
                'setlabel': 'Fit '+label_1,
                'line_kws': {'color': 'C3', 'alpha': 0.8},
                'do_legend': True,
            }

        ##########################################
        # Add textbox (eg.g Thresholds, fidelity #
        # information, number of shots etc)      #
        ##########################################
        if not self.presentation_mode:
            fit_text = 'Thresholds:'
            fit_text += '\nName | Level | Fidelity'
            thr, th_unit = SI_val_to_msg_str(
                self.proc_data_dict['threshold_raw'],
                eff_voltage_unit, return_type=float)
            raw_th_msg = (
                '\n>raw   | ' +
                '{:.2f} {} | '.format(thr, th_unit) +
                '{:.1f}%'.format(
                    self.proc_data_dict['F_assignment_raw']*100))

            fit_text += raw_th_msg

            if self.do_fitting:
                thr, th_unit = SI_val_to_msg_str(
                    self.proc_data_dict['threshold_fit'],
                    eff_voltage_unit, return_type=float)
                fit_th_msg = (
                    '\n>fit     | ' +
                    '{:.2f} {} | '.format(thr, th_unit) +
                    '{:.1f}%'.format(
                        self.proc_data_dict['F_assignment_fit']*100))
                fit_text += fit_th_msg

                thr, th_unit = SI_val_to_msg_str(
                    self.proc_data_dict['threshold_discr'],
                    eff_voltage_unit, return_type=float)
                fit_th_msg = (
                    '\n>dis    | ' +
                    '{:.2f} {} | '.format(thr, th_unit) +
                    '{:.1f}%'.format(
                        self.proc_data_dict['F_discr']*100))
                fit_text += fit_th_msg
                snr = self.fit_res['shots_all'].params['SNR']

                fit_text += format_value_string('\nSNR (fit)', lmfit_par=snr)

                fr = self.fit_res['shots_all']
                bv = fr.params
                a_sp = bv['A_spurious']
                fit_text += '\n\nSpurious Excitations:'

                fit_text += format_value_string('\n$p(e|0)$', lmfit_par=a_sp)

                b_sp = bv['B_spurious']
                fit_text += format_value_string('\n$p(g|\\pi)$',
                                                lmfit_par=b_sp)

            if two_dim_data:
                offs = self.proc_data_dict['raw_offset']

                fit_text += '\n\nRotated by ${:.1f}^\\circ$'.format(
                    (offs[2]*180/np.pi) % 180)
                auto_rot = self.options_dict.get('auto_rotation_angle', True)
                fit_text += '(auto)' if auto_rot else '(man.)'
            else:
                fit_text += '\n\n(Single quadrature data)'

            fit_text += '\n\nTotal shots: %d+%d' % (*self.proc_data_dict['nr_shots'],)
            
            if self.predict_qubit_temp:
                h = spconst.value('Planck constant')
                kb = spconst.value('Boltzmann constant')
                res_exc = a_sp.value
                T_eff = h*self.qubit_freq/(kb*np.log((1-res_exc)/res_exc))
                fit_text += '\n\nQubit $T_{eff}$' \
                            + ' = {:.2f} mK\n   @ {:.3f} GHz' \
                            .format(T_eff*1e3, self.qubit_freq*1e-9)

            for ax in ['cdf', '1D_histogram']:
                self.plot_dicts['text_msg_' + ax] = {
                    'ax_id': ax,
                    'xpos': 1.05,
                    'horizontalalignment': 'left',
                    'plotfn': self.plot_text,
                    'box_props': 'fancy',
                    'text_string': fit_text,
                }


class Dispersive_shift_Analysis(ba.BaseDataAnalysis):
    '''
    Analisys for dispersive shift.
    Designed to be used with <CCL_Transmon>.measure-dispersive_shift_pulsed
    '''
    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', do_fitting: bool = True,
                 data_file_path: str=None,
                 options_dict: dict=None, auto=True,
                 **kw):
        '''
        Extract ground and excited state timestamps
        '''
        if (t_start is None) and (t_stop is None):
            ground_ts = a_tools.return_last_n_timestamps(1, contains='Resonator_scan_off')
            excited_ts= a_tools.return_last_n_timestamps(1, contains='Resonator_scan_on')
        elif (t_start is None) ^ (t_stop is None):
            raise ValueError('Must provide either none or both timestamps.')
        else:
            ground_ts = t_start # t_start is assigned to ground state
            excited_ts= t_stop  # t_stop is assigned to excited state

        super().__init__(t_start=ground_ts, t_stop=excited_ts,
                         label='Resonator_scan', do_fitting=do_fitting,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         **kw)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'
                            }
        self.numeric_params = []
        #self.proc_data_dict = OrderedDict()
        if auto:
            self.run_analysis()

    def process_data(self):
        '''
        Processing data
        '''
        # Frequencu sweep range in the ground/excited state
        self.proc_data_dict['data_freqs_ground'] = \
            self.raw_data_dict['sweep_points'][0]
        self.proc_data_dict['data_freqs_excited'] = \
            self.raw_data_dict['sweep_points'][1]

        # S21 mag (transmission) in the ground/excited state
        self.proc_data_dict['data_S21_ground'] = \
            self.raw_data_dict['measured_values'][0][0]
        self.proc_data_dict['data_S21_excited'] = \
            self.raw_data_dict['measured_values'][1][0]

        #self.proc_data_dict['f0_ground'] = self.raw_data_dict['f0'][0]

        #############################
        # Find resonator dips
        #############################
        pk_rep_ground = a_tools.peak_finder( \
                            self.proc_data_dict['data_freqs_ground'],
                            self.proc_data_dict['data_S21_ground'],
                            window_len=5)
        pk_rep_excited= a_tools.peak_finder( \
                            self.proc_data_dict['data_freqs_excited'],
                            self.proc_data_dict['data_S21_excited'],
                            window_len=5)

        min_idx_ground = np.argmin(pk_rep_ground['dip_values'])
        min_idx_excited= np.argmin(pk_rep_excited['dip_values'])

        min_freq_ground = pk_rep_ground['dips'][min_idx_ground]
        min_freq_excited= pk_rep_excited['dips'][min_idx_excited]

        min_S21_ground = pk_rep_ground['dip_values'][min_idx_ground]
        min_S21_excited= pk_rep_excited['dip_values'][min_idx_excited]

        dispersive_shift = min_freq_excited-min_freq_ground

        self.proc_data_dict['Res_freq_ground'] = min_freq_ground
        self.proc_data_dict['Res_freq_excited']= min_freq_excited
        self.proc_data_dict['Res_S21_ground'] = min_S21_ground
        self.proc_data_dict['Res_S21_excited']= min_S21_excited
        self.proc_data_dict['quantities_of_interest'] = \
            {'dispersive_shift': dispersive_shift}

        self.qoi = self.proc_data_dict['quantities_of_interest']

    def prepare_plots(self):


        x_range = [min(self.proc_data_dict['data_freqs_ground'][0],
                       self.proc_data_dict['data_freqs_excited'][0]) ,
                   max(self.proc_data_dict['data_freqs_ground'][-1],
                       self.proc_data_dict['data_freqs_excited'][-1])]

        y_range = [0, max(max(self.proc_data_dict['data_S21_ground']),
                          max(self.proc_data_dict['data_S21_excited']))]

        x_label = self.raw_data_dict['xlabel'][0]
        y_label = self.raw_data_dict['value_names'][0][0]

        x_unit = self.raw_data_dict['xunit'][0][0]
        y_unit = self.raw_data_dict['value_units'][0][0]

        title = 'Transmission in the ground and excited state'

        self.plot_dicts['S21_ground'] = {
            'title': title,
            'ax_id': 'Transmission_axis',
            'xvals': self.proc_data_dict['data_freqs_ground'],
            'yvals': self.proc_data_dict['data_S21_ground'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': x_label,
            'xunit': x_unit,
            'ylabel': y_label,
            'yunit': y_unit,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'C0', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['S21_excited'] = {
            'title': title,
            'ax_id': 'Transmission_axis',
            'xvals': self.proc_data_dict['data_freqs_excited'],
            'yvals': self.proc_data_dict['data_S21_excited'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': x_label,
            'xunit': x_unit,
            'ylabel': y_label,
            'yunit': y_unit,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'C1', 'alpha': 1},
            'marker': ''
            }

        ####################################
        # Plot arrow
        ####################################
        min_freq_ground = self.proc_data_dict['Res_freq_ground']
        min_freq_excited= self.proc_data_dict['Res_freq_excited']
        yval = y_range[1]/2
        dispersive_shift = int((min_freq_excited-min_freq_ground)*1e-4)*1e-2
        txt_str = r'$2_\chi/2\pi=$' + str(dispersive_shift) + ' MHz'

        self.plot_dicts['Dispersive_shift_line'] = {
            'ax_id': 'Transmission_axis',
            'xvals': [min_freq_ground , min_freq_excited] ,
            'yvals': [yval, yval] ,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['Dispersive_shift_vline'] = {
            'ax_id': 'Transmission_axis',
            'ymin': y_range[0],
            'ymax': y_range[1],
            'x': [min_freq_ground, min_freq_excited],
            'xrange': x_range,
            'yrange': y_range,
            'plotfn': self.plot_vlines,
            'line_kws': {'color': 'black', 'alpha': 0.5}
            }

        self.plot_dicts['Dispersive_shift_rmarker'] = {
            'ax_id': 'Transmission_axis',
            'xvals': [min_freq_ground] ,
            'yvals': [yval] ,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': 5
            }
        self.plot_dicts['Dispersive_shift_lmarker'] = {
            'ax_id': 'Transmission_axis',
            'xvals': [min_freq_excited] ,
            'yvals': [yval] ,
            'plotfn': self.plot_line,
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': 4
            }

        self.plot_dicts['Dispersive_shift_text'] = {
            'ax_id': 'Transmission_axis',
            'plotfn': self.plot_text,
            'xpos': .5,
            'ypos': .5,
            'horizontalalignment': 'center',
            'verticalalignment': 'bottom',
            'text_string': txt_str,
            'box_props': dict(boxstyle='round', pad=.4,
                              facecolor='white', alpha=0.)
            }


class RO_acquisition_delayAnalysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', do_fitting: bool = True,
                 data_file_path: str=None,
                 qubit_name = '',
                 options_dict: dict=None, auto=True,
                 **kw):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label, do_fitting=do_fitting,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         **kw)

        self.single_timestamp = True
        self.qubit_name = qubit_name
        self.params_dict = {'ro_pulse_length': '{}.ro_pulse_length'.format(self.qubit_name),
                            'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'
                            }
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        Processing data
        """
        self.Times = self.raw_data_dict['sweep_points']
        self.I_data_UHF = self.raw_data_dict['measured_values'][0]
        self.Q_data_UHF = self.raw_data_dict['measured_values'][1]
        self.pulse_length = float(self.raw_data_dict['ro_pulse_length'.format(self.qubit_name)])

        #######################################
        # Determine the start of the pusle
        #######################################
        def get_pulse_start(x, y, tolerance=2):
            '''
            The start of the pulse is estimated in three steps:
                1. Evaluate signal standard deviation in a certain interval as
                   function of time: f(t).
                2. Calculate the derivative of the aforementioned data: f'(t).
                3. Evaluate when the derivative exceeds a threshold. This
                   threshold is defined as max(f'(t))/5.
            This approach is more tolerant to noisy signals.
            '''
            pulse_baseline = np.mean(y) # get pulse baseline
            pulse_std      = np.std(y)  # get pulse standard deviation

            nr_points_interval = 200        # number of points in the interval
            aux = int(nr_points_interval/2)

            iteration_idx = np.arange(-aux, len(y)+aux)     # mask for circular array
            aux_list = [ y[i%len(y)] for i in iteration_idx] # circular array

            # Calculate standard deviation for each interval
            y_std = []
            for i in range(len(y)):
                interval = aux_list[i : i+nr_points_interval]
                y_std.append( np.std(interval) )

            y_std_derivative = np.gradient(y_std[:-aux])# calculate derivative
            threshold = max(y_std_derivative)/5        # define threshold
            start_index = np.where( y_std_derivative > threshold )[0][0] + aux

            return start_index-tolerance

        #######################################
        # Determine the end of depletion
        #######################################
        def get_pulse_length(x, y):
            '''
            Similarly to get_pulse_start, the end of depletion is
            set when the signal goes below 5% of its standard dev.
            '''
            pulse_baseline = np.mean(y)
            threshold      = 0.05*np.std(y)
            pulse_std = threshold+1
            i = 0
            while pulse_std > threshold:
                pulse_std = np.std(y[i:]-pulse_baseline)
                i += 1
            end_index = i-1
            return end_index

        Amplitude_I = max(abs(self.I_data_UHF))
        baseline_I = np.mean(self.I_data_UHF)
        start_index_I = get_pulse_start(self.Times, self.I_data_UHF)
        end_index_I = get_pulse_length(self.Times, self.I_data_UHF)

        Amplitude_Q = max(abs(self.Q_data_UHF))
        baseline_Q = np.mean(self.Q_data_UHF)
        start_index_Q = get_pulse_start(self.Times, self.Q_data_UHF)
        end_index_Q = get_pulse_length(self.Times, self.Q_data_UHF)

        self.proc_data_dict['I_Amplitude'] = Amplitude_I
        self.proc_data_dict['I_baseline'] = baseline_I
        self.proc_data_dict['I_pulse_start_index'] = start_index_I
        self.proc_data_dict['I_pulse_end_index'] = end_index_I
        self.proc_data_dict['I_pulse_start'] = self.Times[start_index_I]
        self.proc_data_dict['I_pulse_end'] = self.Times[end_index_I]

        self.proc_data_dict['Q_Amplitude'] = Amplitude_Q
        self.proc_data_dict['Q_baseline'] = baseline_Q
        self.proc_data_dict['Q_pulse_start_index'] = start_index_Q
        self.proc_data_dict['Q_pulse_end_index'] = end_index_Q
        self.proc_data_dict['Q_pulse_start'] = self.Times[start_index_Q]
        self.proc_data_dict['Q_pulse_end'] = self.Times[end_index_Q]

    def prepare_plots(self):

        I_start_line_x = [self.proc_data_dict['I_pulse_start'],
                          self.proc_data_dict['I_pulse_start']]
        I_pulse_line_x = [self.proc_data_dict['I_pulse_start']+self.pulse_length,
                          self.proc_data_dict['I_pulse_start']+self.pulse_length]
        I_end_line_x = [self.proc_data_dict['I_pulse_end'],
                        self.proc_data_dict['I_pulse_end']]

        Q_start_line_x = [self.proc_data_dict['Q_pulse_start'],
                          self.proc_data_dict['Q_pulse_start']]
        Q_pulse_line_x = [self.proc_data_dict['Q_pulse_start']+self.pulse_length,
                           self.proc_data_dict['Q_pulse_start']+self.pulse_length]
        Q_end_line_x = [self.proc_data_dict['Q_pulse_end'],
                        self.proc_data_dict['Q_pulse_end']]

        Amplitude = max(self.proc_data_dict['I_Amplitude'],
                        self.proc_data_dict['Q_Amplitude'])
        vline_y = np.array([1.1*Amplitude, -1.1*Amplitude])

        x_range= [self.Times[0], self.Times[-1]]
        y_range= [vline_y[1], vline_y[0]]

        I_title = str(self.qubit_name)+' Measured transients $I_{quadrature}$'
        Q_title = str(self.qubit_name)+' Measured transients $Q_{quadrature}$'

        ##########################
        # Transients
        ##########################
        self.plot_dicts['I_transients'] = {
            'title': I_title,
            'ax_id': 'I_axis',
            'xvals': self.Times,
            'yvals': self.I_data_UHF,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'line_kws': {'color': 'C0', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['Q_transients'] = {
            'title': Q_title,
            'ax_id': 'Q_axis',
            'xvals': self.Times,
            'yvals': self.Q_data_UHF,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'line_kws': {'color': 'C0', 'alpha': 1},
            'marker': ''
            }

        ##########################
        # Vertical lines
        ##########################
        # I quadrature
        self.plot_dicts['I_pulse_start'] = {
            'ax_id': 'I_axis',
            'xvals': I_start_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['I_pulse_end'] = {
            'ax_id': 'I_axis',
            'xvals': I_pulse_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['I_depletion_end'] = {
            'ax_id': 'I_axis',
            'xvals': I_end_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        # Q quadrature
        self.plot_dicts['Q_pulse_start'] = {
            'ax_id': 'Q_axis',
            'xvals': Q_start_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['Q_pulse_end'] = {
            'ax_id': 'Q_axis',
            'xvals': Q_pulse_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        self.plot_dicts['Q_depletion_end'] = {
            'ax_id': 'Q_axis',
            'xvals': Q_end_line_x,
            'yvals': vline_y,
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_line,
            'linestyle': '--',
            'line_kws': {'color': 'black', 'alpha': 1},
            'marker': ''
            }

        ########################
        # Plot pulse windows
        ########################

        I_pulse_bin = np.array([self.proc_data_dict['I_pulse_start'],
                    self.proc_data_dict['I_pulse_start']+self.pulse_length])
        I_depletion_bin = np.array([self.proc_data_dict['I_pulse_start']
                        +self.pulse_length, self.proc_data_dict['I_pulse_end']])

        Q_pulse_bin = np.array([self.proc_data_dict['Q_pulse_start'],
                    self.proc_data_dict['Q_pulse_start']+self.pulse_length])
        Q_depletion_bin = np.array([self.proc_data_dict['Q_pulse_start']
                        +self.pulse_length, self.proc_data_dict['Q_pulse_end']])

        self.plot_dicts['I_pulse_length'] = {
            'ax_id': 'I_axis',
            'xvals': I_pulse_bin,
            'yvals': vline_y,
            'xwidth': self.pulse_length,
            'ywidth': self.proc_data_dict['I_Amplitude'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_bar,
            'bar_kws': { 'alpha': .25, 'facecolor': 'C0'}
            }

        self.plot_dicts['I_pulse_depletion'] = {
            'ax_id': 'I_axis',
            'xvals': I_depletion_bin,
            'yvals': vline_y,
            'xwidth': self.pulse_length,
            'ywidth': self.proc_data_dict['I_Amplitude'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'I Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_bar,
            'bar_kws': { 'alpha': .25, 'facecolor': 'C1'}
            }

        self.plot_dicts['Q_pulse_length'] = {
            'ax_id': 'Q_axis',
            'xvals': Q_pulse_bin,
            'yvals': vline_y,
            'xwidth': self.pulse_length,
            'ywidth': self.proc_data_dict['Q_Amplitude'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_bar,
            'bar_kws': { 'alpha': .25, 'facecolor': 'C0'}
            }

        self.plot_dicts['Q_pulse_depletion'] = {
            'ax_id': 'Q_axis',
            'grid': True,
            'grid_kws': {'alpha': .25, 'linestyle': '--'},
            'xvals': Q_depletion_bin,
            'yvals': vline_y,
            'xwidth': self.pulse_length,
            'ywidth': self.proc_data_dict['Q_Amplitude'],
            'xrange': x_range,
            'yrange': y_range,
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': 's',
            'ylabel': 'Q Amplitude',
            'yunit': 'V',
            'plotfn': self.plot_bar,
            'bar_kws': { 'alpha': .25, 'facecolor': 'C1'}
            }


class Readout_landspace_Analysis(sa.Basic2DInterpolatedAnalysis):
    '''
    Analysis for Readout landscapes using adaptive sampling.
    Stores maximum fidelity parameters in quantities of interest dict as:
        - <analysis_object>.qoi['Optimal_parameter_X']
        - <analysis_object>.qoi['Optimal_parameter_Y']
    '''
    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 interp_method: str = 'linear',
                 options_dict: dict=None, auto=True,
                 **kw):

        super().__init__(t_start = t_start, t_stop = t_stop,
                         label = label,
                         data_file_path = data_file_path,
                         options_dict = options_dict,
                         auto = auto,
                         interp_method=interp_method,
                         **kw)
        if auto:
            self.run_analysis()

    def process_data(self):
        super().process_data()

        # Extract maximum interpolated fidelity
        idx = [i for i, s in enumerate(self.proc_data_dict['value_names']) \
                if 'F_a' in s][0]
        X = self.proc_data_dict['x_int']
        Y = self.proc_data_dict['y_int']
        Z = self.proc_data_dict['interpolated_values'][idx]

        max_idx = np.unravel_index(np.argmax(Z), (len(X),len(Y)) )
        self.proc_data_dict['Max_F_a_idx'] = max_idx
        self.proc_data_dict['Max_F_a'] = Z[max_idx[1],max_idx[0]]

        self.proc_data_dict['quantities_of_interest'] = {
            'Optimal_parameter_X': X[max_idx[1]],
            'Optimal_parameter_Y': Y[max_idx[0]]
            }

    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        for i, val_name in enumerate(self.proc_data_dict['value_names']):

            zlabel = '{} ({})'.format(val_name,
                                      self.proc_data_dict['value_units'][i])
            # Plot interpolated landscape
            self.plot_dicts[val_name] = {
                'ax_id': val_name,
                'plotfn': a_tools.color_plot,
                'x': self.proc_data_dict['x_int'],
                'y': self.proc_data_dict['y_int'],
                'z': self.proc_data_dict['interpolated_values'][i],
                'xlabel': self.proc_data_dict['xlabel'],
                'x_unit': self.proc_data_dict['xunit'],
                'ylabel': self.proc_data_dict['ylabel'],
                'y_unit': self.proc_data_dict['yunit'],
                'zlabel': zlabel,
                'title': '{}\n{}'.format(
                    self.timestamp, self.proc_data_dict['measurementstring'])
                }
            # Plot sampled values
            self.plot_dicts[val_name+str('_sampled_values')] = {
                'ax_id': val_name,
                'plotfn': scatter_pnts_overlay,
                'x': self.proc_data_dict['x'],
                'y': self.proc_data_dict['y'],
                'xlabel': self.proc_data_dict['xlabel'],
                'x_unit': self.proc_data_dict['xunit'],
                'ylabel': self.proc_data_dict['ylabel'],
                'y_unit': self.proc_data_dict['yunit'],
                'alpha': .75,
                'setlabel': 'Sampled points',
                'do_legend': True
                }
            # Plot maximum fidelity point
            self.plot_dicts[val_name+str('_max_fidelity')] = {
                'ax_id': val_name,
                'plotfn': self.plot_line,
                'xvals': [self.proc_data_dict['x_int']\
                            [self.proc_data_dict['Max_F_a_idx'][1]]],
                'yvals': [self.proc_data_dict['y_int']\
                            [self.proc_data_dict['Max_F_a_idx'][0]]],
                'xlabel': self.proc_data_dict['xlabel'],
                'xunit': self.proc_data_dict['xunit'],
                'ylabel': self.proc_data_dict['ylabel'],
                'yunit': self.proc_data_dict['yunit'],
                'marker': 'x',
                'linestyle': '',
                'color': 'red',
                'setlabel': 'Max fidelity',
                'do_legend': True,
                'legend_pos': 'upper right'
                }


class Multiplexed_Readout_Analysis_deprecated(ba.BaseDataAnalysis):
    """
    For two qubits, to make an n-qubit mux readout experiment.
    we should vectorize this analysis

    TODO: This needs to be rewritten/debugged!
    Suggestion:
        Use N*(N-1)/2 instances of Singleshot_Readout_Analysis,
          run them without saving the plots and then merge together the
          plot_dicts as in the cross_dephasing_analysis.
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='',
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 nr_of_qubits: int = 2,
                 qubit_names: list=None,
                 do_fitting: bool=True, auto=True):
        """
        Inherits from BaseDataAnalysis.
        Extra arguments of interest
            qubit_names (list) : used to label the experiments, names of the
                qubits. LSQ is last name in the list. If not specified will
                set qubit_names to [qN, ..., q1, q0]


        """
        self.nr_of_qubits = nr_of_qubits
        if qubit_names is None:
            self.qubit_names = list(reversed(['q{}'.format(i)
                                              for i in range(nr_of_qubits)]))
        else:
            self.qubit_names = qubit_names

        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values',
            'value_names': 'value_names',
            'value_units': 'value_units'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        Responsible for creating the histograms based on the raw data
        """
        # Determine the shape of the data to extract wheter to rotate or not
        nr_bins = int(self.options_dict.get('nr_bins', 100))

        # self.proc_data_dict['shots_0'] = [''] * nr_expts
        # self.proc_data_dict['shots_1'] = [''] * nr_expts

        #################################################################
        #  Separating data into shots for the different prepared states #
        #################################################################
        self.proc_data_dict['nr_of_qubits'] = self.nr_of_qubits
        self.proc_data_dict['qubit_names'] = self.qubit_names

        self.proc_data_dict['ch_names'] = self.raw_data_dict['value_names'][0]

        for ch_name, shots in self.raw_data_dict['measured_values_ord_dict'].items():
            self.proc_data_dict[ch_name] = shots[0]  # only 1 dataset
            self.proc_data_dict[ch_name +
                                ' all'] = self.proc_data_dict[ch_name]
            min_sh = np.min(self.proc_data_dict[ch_name])
            max_sh = np.max(self.proc_data_dict[ch_name])
            self.proc_data_dict['nr_shots'] = len(self.proc_data_dict[ch_name])

            base = 2
            number_of_experiments = base ** self.nr_of_qubits

            combinations = [int2base(
                i, base=base, fixed_length=self.nr_of_qubits) for i in
                range(number_of_experiments)]
            self.proc_data_dict['combinations'] = combinations

            for i, comb in enumerate(combinations):
                # No post selection implemented yet
                self.proc_data_dict['{} {}'.format(ch_name, comb)] = \
                    self.proc_data_dict[ch_name][i::number_of_experiments]
                #####################################
                #  Binning data into 1D histograms  #
                #####################################
                hist_name = 'hist {} {}'.format(
                    ch_name, comb)
                self.proc_data_dict[hist_name] = np.histogram(
                    self.proc_data_dict['{} {}'.format(
                        ch_name, comb)],
                    bins=nr_bins, range=(min_sh, max_sh))
                #  Cumulative histograms #
                chist_name = 'c'+hist_name
                # the cumulative histograms are normalized to ensure the right
                # fidelities can be calculated
                self.proc_data_dict[chist_name] = np.cumsum(
                    self.proc_data_dict[hist_name][0])/(
                    np.sum(self.proc_data_dict[hist_name][0]))

            self.proc_data_dict['bin_centers {}'.format(ch_name)] = (
                self.proc_data_dict[hist_name][1][:-1] +
                self.proc_data_dict[hist_name][1][1:]) / 2

            self.proc_data_dict['binsize {}'.format(ch_name)] = (
                self.proc_data_dict[hist_name][1][1] -
                self.proc_data_dict[hist_name][1][0])

        #####################################################################
        # Combining histograms of all different combinations and calc Fid.
        ######################################################################
        for ch_idx, ch_name in enumerate(self.proc_data_dict['ch_names']):
            # Create labels for the specific combinations
            comb_str_0, comb_str_1, comb_str_2 = get_arb_comb_xx_label(
                self.proc_data_dict['nr_of_qubits'], qubit_idx=ch_idx)

            # Initialize the arrays
            self.proc_data_dict['hist {} {}'.format(ch_name, comb_str_0)] = \
                [np.zeros(nr_bins), np.zeros(nr_bins+1)]
            self.proc_data_dict['hist {} {}'.format(ch_name, comb_str_1)] = \
                [np.zeros(nr_bins), np.zeros(nr_bins+1)]
            zero_hist = self.proc_data_dict['hist {} {}'.format(
                ch_name, comb_str_0)]
            one_hist = self.proc_data_dict['hist {} {}'.format(
                ch_name, comb_str_1)]

            # Fill them with data from the relevant combinations
            for i, comb in enumerate(self.proc_data_dict['combinations']):
                if comb[-(ch_idx+1)] == '0':
                    zero_hist[0] += self.proc_data_dict[
                        'hist {} {}'.format(ch_name, comb)][0]
                    zero_hist[1] = self.proc_data_dict[
                        'hist {} {}'.format(ch_name, comb)][1]
                elif comb[-(ch_idx+1)] == '1':
                    one_hist[0] += self.proc_data_dict[
                        'hist {} {}'.format(ch_name, comb)][0]
                    one_hist[1] = self.proc_data_dict[
                        'hist {} {}'.format(ch_name, comb)][1]
                elif comb[-(ch_idx+1)] == '2':
                    # Fixme add two state binning
                    raise NotImplementedError()

            chist_0 = np.cumsum(zero_hist[0])/(np.sum(zero_hist[0]))
            chist_1 = np.cumsum(one_hist[0])/(np.sum(one_hist[0]))

            self.proc_data_dict['chist {} {}'.format(ch_name, comb_str_0)] \
                = chist_0
            self.proc_data_dict['chist {} {}'.format(ch_name, comb_str_1)] \
                = chist_1
            ###########################################################
            #  Threshold and fidelity based on cumulative histograms  #

            qubit_name = self.proc_data_dict['qubit_names'][-(ch_idx+1)]
            centers = self.proc_data_dict['bin_centers {}'.format(ch_name)]
            fid, th = get_assignement_fid_from_cumhist(chist_0, chist_1,
                                                       centers)
            self.proc_data_dict['F_ass_raw {}'.format(qubit_name)] = fid
            self.proc_data_dict['threshold_raw {}'.format(qubit_name)] = th

    def prepare_plots(self):
        # N.B. If the log option is used we should manually set the
        # yscale to go from .5 to the current max as otherwise the fits
        # mess up the log plots.
        # log_hist = self.options_dict.get('log_hist', False)

        for ch_idx, ch_name in enumerate(self.proc_data_dict['ch_names']):
            q_name = self.proc_data_dict['qubit_names'][-(ch_idx+1)]
            th_raw = self.proc_data_dict['threshold_raw {}'.format(q_name)]
            F_raw = self.proc_data_dict['F_ass_raw {}'.format(q_name)]

            self.plot_dicts['histogram_{}'.format(ch_name)] = {
                'plotfn': make_mux_ssro_histogram,
                'data_dict': self.proc_data_dict,
                'ch_name': ch_name,
                'title': (self.timestamps[0] + ' \n' +
                          'SSRO histograms {}'.format(ch_name))}

            thresholds = [th_raw]
            threshold_labels = ['thresh. raw']

            self.plot_dicts['comb_histogram_{}'.format(q_name)] = {
                'plotfn': make_mux_ssro_histogram_combined,
                'data_dict': self.proc_data_dict,
                'ch_name': ch_name,
                'thresholds': thresholds,
                'threshold_labels': threshold_labels,
                'qubit_idx': ch_idx,
                'title': (self.timestamps[0] + ' \n' +
                          'Combined SSRO histograms {}'.format(q_name))}

            fid_threshold_msg = 'Summary {}\n'.format(q_name)
            fid_threshold_msg += r'$F_{A}$-raw: ' + '{:.3f} \n'.format(F_raw)
            fid_threshold_msg += r'thresh. raw: ' + '{:.3f} \n'.format(th_raw)

            self.plot_dicts['fid_threshold_msg_{}'.format(q_name)] = {
                'plotfn': self.plot_text,
                'xpos': 1.05,
                'ypos': .9,
                'horizontalalignment': 'left',
                'text_string': fid_threshold_msg,
                'ax_id': 'comb_histogram_{}'.format(q_name)}


def get_shots_zero_one(data, post_select: bool=False,
                       nr_samples: int=2, sample_0: int=0, sample_1: int=1,
                       post_select_threshold: float = None):
    if not post_select:
        shots_0, shots_1 = a_tools.zigzag(
            data, sample_0, sample_1, nr_samples)
    else:
        presel_0, presel_1 = a_tools.zigzag(
            data, sample_0, sample_1, nr_samples)

        shots_0, shots_1 = a_tools.zigzag(
            data, sample_0+1, sample_1+1, nr_samples)

    if post_select:
        post_select_shots_0 = data[0::nr_samples]
        shots_0 = data[1::nr_samples]

        post_select_shots_1 = data[nr_samples//2::nr_samples]
        shots_1 = data[nr_samples//2+1::nr_samples]

        # Determine shots to remove
        post_select_indices_0 = dm_tools.get_post_select_indices(
            thresholds=[post_select_threshold],
            init_measurements=[post_select_shots_0])

        post_select_indices_1 = dm_tools.get_post_select_indices(
            thresholds=[post_select_threshold],
            init_measurements=[post_select_shots_1])

        shots_0[post_select_indices_0] = np.nan
        shots_0 = shots_0[~np.isnan(shots_0)]

        shots_1[post_select_indices_1] = np.nan
        shots_1 = shots_1[~np.isnan(shots_1)]

    return shots_0, shots_1


def get_arb_comb_xx_label(nr_of_qubits, qubit_idx: int):
    """
    Returns labels of the form "xx0xxx", "xx1xxx", "xx2xxx"
    Length of the label is equal to the number of qubits
    """
    comb_str_0 = list('x'*nr_of_qubits)
    comb_str_0[-(qubit_idx+1)] = '0'
    comb_str_0 = "".join(comb_str_0)

    comb_str_1 = list('x'*nr_of_qubits)
    comb_str_1[-(qubit_idx+1)] = '1'
    comb_str_1 = "".join(comb_str_1)

    comb_str_2 = list('x'*nr_of_qubits)
    comb_str_2[-(qubit_idx+1)] = '2'
    comb_str_2 = "".join(comb_str_2)

    return comb_str_0, comb_str_1, comb_str_2


def get_assignement_fid_from_cumhist(chist_0, chist_1, bin_centers=None):
    """
    Returns the average assignment fidelity and threshold
        F_assignment_raw = (P01 - P10 )/2
            where Pxy equals probability to measure x when starting in y
    """
    F_vs_th = (1-(1-abs(chist_1 - chist_0))/2)
    opt_idx = np.argmax(F_vs_th)
    F_assignment_raw = F_vs_th[opt_idx]

    if bin_centers is None:
        bin_centers = np.arange(len(chist_0))
    threshold = bin_centers[opt_idx]

    return F_assignment_raw, threshold


def make_mux_ssro_histogram_combined(data_dict, ch_name, qubit_idx,
                                     thresholds=None, threshold_labels=None,
                                     title=None, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()
    markers = itertools.cycle(('v', '^', 'd'))

    comb_str_0, comb_str_1, comb_str_2 = get_arb_comb_xx_label(
        data_dict['nr_of_qubits'], qubit_idx=qubit_idx)

    ax.plot(data_dict['bin_centers {}'.format(ch_name)],
            data_dict['hist {} {}'.format(ch_name, comb_str_0)][0],
            linestyle='',
            marker=next(markers), alpha=.7, label=comb_str_0)
    ax.plot(data_dict['bin_centers {}'.format(ch_name)],
            data_dict['hist {} {}'.format(ch_name, comb_str_1)][0],
            linestyle='',
            marker=next(markers), alpha=.7, label=comb_str_1)

    if thresholds is not None:
        # this is to support multiple threshold types such as raw, fitted etc.
        th_styles = itertools.cycle(('--', '-.', '..'))
        for threshold, label in zip(thresholds, threshold_labels):
            ax.axvline(threshold, linestyle=next(th_styles), color='grey',
                       label=label)

    legend_title = "Prep. state [%s]" % ', '.join(data_dict['qubit_names'])
    ax.legend(title=legend_title, loc=1)  # top right corner
    ax.set_ylabel('Counts')
    # arbitrary units as we use optimal weights
    set_xlabel(ax, ch_name, 'a.u.')

    if title is not None:
        ax.set_title(title)


def make_mux_ssro_histogram(data_dict, ch_name, title=None, ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()
    nr_of_qubits = data_dict['nr_of_qubits']
    markers = itertools.cycle(('v', '<', '>', '^', 'd', 'o', 's', '*'))
    for i in range(2**nr_of_qubits):
        format_str = '{'+'0:0{}b'.format(nr_of_qubits) + '}'
        binning_string = format_str.format(i)
        ax.plot(data_dict['bin_centers {}'.format(ch_name)],
                data_dict['hist {} {}'.format(ch_name, binning_string)][0],
                linestyle='',
                marker=next(markers), alpha=.7, label=binning_string)

    legend_title = "Prep. state \n[%s]" % ', '.join(data_dict['qubit_names'])
    ax.legend(title=legend_title, loc=1)
    ax.set_ylabel('Counts')
    # arbitrary units as we use optimal weights
    set_xlabel(ax, ch_name, 'a.u.')

    if title is not None:
        ax.set_title(title)


def plot_2D_ssro_histogram(xvals, yvals, zvals, xlabel, xunit, ylabel, yunit, zlabel, zunit,
                           xlim=None, ylim=None,
                           title='',
                           cmap='viridis',
                           cbarwidth='10%',
                           cbarpad='5%',
                           no_label=False,
                           ax=None, cax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()
    if not no_label:
        ax.set_title(title)

    # Plotting the "heatmap"
    out = flex_colormesh_plot_vs_xy(xvals, yvals, zvals, ax=ax,
                                    plot_cbar=True, cmap=cmap)
    # Adding the colorbar
    if cax is None:
        ax.ax_divider = make_axes_locatable(ax)
        ax.cax = ax.ax_divider.append_axes(
            'right', size=cbarwidth, pad=cbarpad)
    else:
        ax.cax = cax
    ax.cbar = plt.colorbar(out['cmap'], cax=ax.cax)

    # Setting axis limits aspect ratios and labels
    ax.set_aspect(1)
    set_xlabel(ax, xlabel, xunit)
    set_ylabel(ax, ylabel, yunit)
    set_cbarlabel(ax.cbar, zlabel, zunit)
    if xlim is None:
        xlim = np.min([xvals, yvals]), np.max([xvals, yvals])
    ax.set_xlim(xlim)
    if ylim is None:
        ylim = np.min([xvals, yvals]), np.max([xvals, yvals])
    ax.set_ylim(ylim)
