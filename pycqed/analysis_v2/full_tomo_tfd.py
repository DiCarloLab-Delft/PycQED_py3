"""
Analysis for Thermal Field Double state VQE experiment
"""

import os
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from pycqed.analysis.analysis_toolbox import get_datafilepath_from_timestamp
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel, \
    cmap_to_alpha, cmap_first_to_alpha
import pycqed.measurement.hdf5_data as h5d
import pycqed.analysis_v2.multiplexed_readout_analysis as mux_an
import pycqed.analysis_v2.tfd_analysis as tfd_an
import pycqed.analysis_v2.tomo_functions as tomo_func
from functools import reduce

def flatten_list(l): return reduce(lambda x, y: x+y, l)


class TFD_3CZ_Analysis_Pauli_Tomo(tfd_an.TFD_3CZ_Analysis_Pauli_Strings):
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 g: float = 1, T: float = 1,
                 num_qubits: int = 4, complexity_of_readout_model: int = 1,
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):
        """
        Analysis for 3CZ version of the Thermal Field Double VQE circuit.

        Args:
            g (float):
                coupling strength (in theorist units)
            T (float):
                temperature (in theorist units)
        """

        self.num_qubits = num_qubits
        # complexity values
        # 0 = basic RO with main betas
        # 1 = considers w2 terms (w/ D2) on the single-qubit channel of X
        # 2 = considers w3 terms (w/ D2) on X-D4 channel
        self.complexity_of_readout_model = complexity_of_readout_model
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         g=g, T=T,
                         extract_only=extract_only)

    # def extract_data(self): # inherited from parent
    def process_data(self):
        self.proc_data_dict = {}
        combinations = self.raw_data_dict['combinations']
        self.num_states = 2**self.num_qubits
        centers_vec = np.zeros((self.num_states, self.num_qubits))
        self.num_segments = len(combinations)
        cal_point_seg_start = self.num_segments - self.num_states # 18 for 34 segments
        self.cal_point_seg_start = cal_point_seg_start

        data_shots = self.raw_data_dict['data'][:, :]
        self.proc_data_dict['raw_shots'] = data_shots[:, 1:]
        value_names = self.raw_data_dict['value_names']

        # 1. calculate centers of states
        for id_state in range(self.num_states):
            centers_this_state = np.mean(data_shots[cal_point_seg_start+id_state::self.num_segments, :],
                                         axis=0)[1:]
            centers_vec[id_state, :] = centers_this_state

        # 2. compute matrix for betas
        matrix_B = tomo_func.compute_beta_matrix(self.num_qubits)
        # 3. Computing threshold
        mn_voltages = tomo_func.define_thresholds_avg(data_shots=data_shots,
                                                      value_names=value_names,
                                                      combinations=combinations,
                                                      num_states=self.num_states)

        # 4. Bining weight-1 data
        shots_discr, qubit_state_avg = tomo_func.threshold_weight1_data(data_shots=data_shots,
                                                                        mn_voltages=mn_voltages,
                                                                        value_names=value_names,
                                                                        num_qubits=self.num_qubits,
                                                                        num_segments=self.num_segments)

        # 5. Compute betas weight-1
        betas_w1, op_idx_w1 = tomo_func.compute_betas_weight1(qubit_state_avg=qubit_state_avg,
                                                              matrix_B=matrix_B,
                                                              num_qubits=self.num_qubits,
                                                              cal_point_seg_start=cal_point_seg_start)
        # compute expected measurement from betas.
        # 6. Bining weight-2 data
        correlations = [['Z1', 'D1'], ['D1', 'X'], ['D3', 'X'], ['D3', 'Z1']]
        idx_qubit_ro = ['D1', 'Z1', 'X', 'D3']
        correl_discr, correl_avg = tomo_func.correlating_weight2_data(shots_discr=shots_discr,
                                                                      idx_qubit_ro=idx_qubit_ro,
                                                                      correlations=correlations,
                                                                      num_segments=self.num_segments)
        # 7. Compute betas weight-2
        betas_w2, op_idx_w2 = tomo_func.compute_betas_weight2(matrix_B=matrix_B,
                                                              correl_avg=correl_avg,
                                                              correlations=correlations,
                                                              cal_point_seg_start=cal_point_seg_start,
                                                              idx_qubit_ro=idx_qubit_ro,
                                                              num_qubits=self.num_qubits)
        self.raw_data_dict['ro_sq_raw_signal'] = qubit_state_avg
        self.raw_data_dict['ro_tq_raw_signal'] = correl_avg
        self.raw_data_dict['ro_sq_ch_names'] = idx_qubit_ro
        self.raw_data_dict['ro_tq_ch_names'] = correlations
        self.proc_data_dict['betas_w1'] = betas_w1
        self.proc_data_dict['betas_w2'] = betas_w2
        # 9. Computing inversion matrix for tomo
        """
        For channel K:
        Construct matrix procedure
            grab m_i, corresponds to pre-rot #bin(i) (with 0s for Is and 1s for Xs)
            grab betas_channel, (beta_i corresponding to op_i corresponding to slot i of operators vector)
            for each beta/op pair
                beta_i=betas_w1[ch,op] corresponds to i=op_idx_w1[ch,op] op_bin=format(i, '#0{}b'.format(self.num_qubits+2))[2:]
                rot_bin=format(i_rot, '#0{}b'.format(self.num_qubits+2))[2:]
                for each Z in op_i, if there is an X in pre-rot, flip sign of beta.
                solved by writing
                op_i    IZZI
                rot_i   XIXI
                op_bin  0110
                rot_bin 1010
                ------------
                output. 0010
                product 1 (=> flip sign)

        stack all matrices vertically
        stack all measurement vectors vertically
        """
        pre_rot_list = np.sort(np.unique(np.concatenate(
            (op_idx_w1.flatten(), op_idx_w2.flatten()))))
        # there is an abuse of naming here. pre_rot_list is not a list of pre-rotations
        # but rather a list of operators that appear in our mmt operator.
        # it serves to keep a track of what operators we have information (beta!=0) about, in order to put it in M_matrix.
        # ASSUMPTION: pre-rotations are proper, therefore, {ops in mmt} is the same set as {flips in pre-rot}
        num_ops = self.num_states
        num_chs_w1 = betas_w1.shape[0]
        num_chs_w2 = betas_w2.shape[0]
        num_prerots = pre_rot_list.shape[0]

        M_matrix = np.zeros((num_prerots*8, num_ops))
        # first w1 channels
        for ch_w1_id in range(num_chs_w1):
            this_M_matrix = np.zeros(
                (num_prerots, num_ops))  # prepare M_matrix
            for ir, id_rot in enumerate(pre_rot_list):
                rot_bin = format(id_rot, '#0{}b'.format(self.num_qubits+2))[2:]
                # grabbing betas and operators
                this_betas = betas_w1[ch_w1_id, :]
                this_op = op_idx_w1[ch_w1_id, :]
                for i_b, bt in enumerate(this_betas):
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#0{}b'.format(self.num_qubits+2))[2:]
                    # decide the sign
                    sign = np.product([1-2*int(rot_bin[k])*int(op_bin[k])
                                       for k in range(len(op_bin))])
                    # print(ir,id_op)
                    this_M_matrix[ir, id_op] = sign*bt
            M_matrix[ch_w1_id *
                     num_prerots:(ch_w1_id+1)*num_prerots, :] = this_M_matrix
        # now w2 channels
        for ch_w2_id in range(num_chs_w2):
            this_M_matrix = np.zeros(
                (num_prerots, num_ops))  # prepare M_matrix
            for ir, id_rot in enumerate(pre_rot_list):
                rot_bin = format(id_rot, '#0{}b'.format(self.num_qubits+2))[2:]
                # grabbing betas and operators
                this_betas = betas_w2[ch_w2_id, :]
                this_op = op_idx_w2[ch_w2_id, :]
                for i_b, bt in enumerate(this_betas):
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#0{}b'.format(self.num_qubits+2))[2:]
                    # decide the sign
                    sign = np.product([1-2*int(rot_bin[k])*int(op_bin[k])
                                       for k in range(len(op_bin))])
                    # print(ir,id_op)
                    this_M_matrix[ir, id_op] = sign*bt
            M_matrix[36+ch_w2_id*num_prerots:36 +
                     (ch_w2_id+1)*num_prerots, :] = this_M_matrix
        # if enabled, enhance mmt model for X
        self.proc_data_dict['tomography_matrix'] = M_matrix

        # 10. performing tomographic inversion
        # here pre_rot_list shifts to indeed become the actual list of performed pre_rotations (basis and flips).
        # they have to match the ordering from before. thats why the index() call.
        """
        M_matrix is the measurement matrix. all in Z basis.
        We re-interpret this with the knowledge of pre-rotations basis.

        Define a new whole_M_matrix (whole w.r.t. bases)
        for each pre-rotation (row):
            grab bases (from pre-rotation). ie. bN..b1b0 = ZZXY (no signs here)
            for each term in mmt_op:
                transform term to new bases. ie. ZIZZ -> ZIXY (for example above)
                locate on the whole_M_matrix (row=pre-rot
                                              col=locate operator in the inverted vector)
        invert whole_M_matrix and obtain operator_vec

        Necessary functions/conventions
        > Grab bases from pre-rot. bN..b1b0
        > Transform operator. ZIZZ into ZIXY
        > locate operator in vector. ZIXY in [IIII, IIIX, IIIY, IIIZ, IIXI, IIXX, IIXY...]
        """
        whole_M_matrix = np.zeros((0,0))
        for i_prerot,prerot in enumerate(prerot_vector):
            this_prerot_bases = tomo_func.grab_bases_from_prerot(prerot)
            this_flip_bin = # I=0;X=1
            for ch_w1_id in range(num_chs_w1):
                for i_op, op in enumerate(op_idx_w1[ch_w1_id,:]):
                    this_beta = betas_w1[ch_w1_id,i_op]
                    this_op_bin = format(op, '#0{}b'.format(self.num_qubits+2))[2:] # I=0;Z=1
                    rotated_op_idx, rotated_op = tomo_func.rotate_operator(op)
                    this_sign = np.product([1-2*int(this_flip_bin[k])*int(this_op_bin[k])
                                            for k in range(len(this_op_bin))]) # function of flips and this operator.
                    whole_M_matrix[i_prerot+ch_w1_id*num_prerots,
                                   rotated_op_idx] = this_sign*this_beta
            for ch_w2_id in range(num_chs_w2):
                for i_op, op in enumerate(op_idx_w2[ch_w1_id,:]):
                    this_beta = betas_w2[ch_w2_id,i_op]
                    this_op_bin = format(op, '#0{}b'.format(self.num_qubits+2))[2:] # I=0;Z=1
                    rotated_op_idx, rotated_op = tomo_func.rotate_operator(op)
                    this_sign = np.product([1-2*int(this_flip_bin[k])*int(this_op_bin[k])
                                            for k in range(len(this_op_bin))]) # function of flips and this operator.
                    whole_M_matrix[i_prerot+(num_w1_ch+ch_w2_id)*num_prerots,
                                   rotated_op_idx] = this_sign*this_beta

        whole_M_nobeta0 = whole_M_matrix[:, 1:]
        beta0_vec = whole_M_matrix[:, 0]
        inv_whole_M_nobeta0 = np.linalg.pinv(whole_M_nobeta0)

        pauli_terms = inv_whole_M_nobeta0 @ (prerot_mmt_vec-beta0_vec)
        # The next bit requires setting the convention on operator listing
        #     op_labels = [format(p, '#0{}b'.format(self.num_qubits+2))[2:].replace(
        #         '0', 'I').replace('1', basis) for p in range(16)]
        #     for i_op, op in enumerate(op_labels):
        #         if i_op > 0:
        #             tomo_dict[op] = pauli_terms[i_op-1]

        # 11. Keeping only relevant terms from the tomo
        desired_operators = ['ZZII', 'XIII', 'IXII', 'IIZZ',
                             'IIXI', 'IIIX', 'ZIZI', 'IZIZ', 'XIXI', 'IXIX']
        op_values = [tomo_dict[op] for op in desired_operators]

        in_dict = {}
        for i_op, op in enumerate(desired_operators):
            in_dict[op] = op_values[i_op]

        self.proc_data_dict['pauli_terms'] = in_dict
        self.proc_data_dict['energy_terms'] = tfd_an.calc_tfd_hamiltonian(
            pauli_terms=self.proc_data_dict['pauli_terms'],
            g=self.g, T=self.T)
        self.proc_data_dict['quantities_of_interest'] = {
            'g': self.g, 'T': self.T,
            'full_tomo_dict': tomo_dict,
            **self.proc_data_dict['pauli_terms'],
            **self.proc_data_dict['energy_terms']}

    def prepare_plots(self):
        self.plot_dicts['pauli_operators_Tomo'] = {
            'plotfn': tfd_an.plot_pauli_ops,
            'pauli_terms': self.proc_data_dict['pauli_terms'],
            'energy_terms': self.proc_data_dict['energy_terms']
        }
        self.plot_dicts['pauli_operators_Tomo_full'] = {
            'plotfn': tfd_an.plot_all_pauli_ops,
            'full_dict': self.proc_data_dict['quantities_of_interest']['full_tomo_dict']
            # 'pauli_terms': self.proc_data_dict['pauli_terms']
        }
        for ch_id,ch in enumerate(self.raw_data_dict['ro_sq_ch_names']):
            self.plot_dicts['TV_{}'.format(ch)] = {
                'plotfn': plot_tv_mode_with_ticks,
                'xticks': self.raw_data_dict['combinations'],
                'yvals': self.raw_data_dict['ro_sq_raw_signal'][ch_id,:],
                'ylabel': ch,
                'shade_from': self.cal_point_seg_start,
                # 'yunit': self.raw_data_dict['value_units'][0][i],
                'title': (self.raw_data_dict['timestamps'][0]+' - ' + ' TV: {}'.format(ch))}
        for ch_id,ch in enumerate(self.raw_data_dict['ro_tq_ch_names']):
            self.plot_dicts['TV_{}'.format(ch)] = {
                'plotfn': plot_tv_mode_with_ticks,
                'xticks': self.raw_data_dict['combinations'],
                'yvals': self.raw_data_dict['ro_tq_raw_signal'][:,ch_id],
                'ylabel': ch,
                'shade_from': self.cal_point_seg_start,
                # 'yunit': self.raw_data_dict['value_units'][0][i],
                'title': (self.raw_data_dict['timestamps'][0]+' - ' + ' TV: {}'.format(ch))}



def plot_tv_mode_with_ticks(xticks, yvals, ylabel, shade_from=0, xticks_rotation=90, yunit='', title='', ax=None, **kw):
    if ax is None:
        f, ax = plt.subplots()

    xvals = np.arange(len(yvals))
    ax.fill_betweenx(x1=[shade_from],x2=[xvals.max()],y=[-10,10], alpha=0.5, color='grey')
    ax.set_ylim(-1.05,1.05)
    ax.plot(xvals,yvals,'-o')
    ax.set_xticks(xvals)
    ax.set_xticklabels(xticks, rotation=xticks_rotation)

    # ax.set_ylabel(ylabel+ ' ({})'.format(yunit))
    ax.set_title(title)