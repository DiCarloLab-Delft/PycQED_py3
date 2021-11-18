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


class TFD_Analysis_Pauli_Tomo(tfd_an.TFD_Analysis_Pauli_Strings):
    def __init__(self, t_start: str = None, t_stop: str = None,
                 label: str = '',
                 g: float = 1, T=1,
                 num_qubits: int = 4, complexity_of_readout_model: int = 1,
                 options_dict: dict = None, extract_only: bool = False,
                 auto=True):
        """
        Analysis for the Thermal Field Double state QAOA experiment.

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
        # 8. Complicating betas on qubit X
        # M_X = II + I_X Z_D2 + Z_X I_D2 + Z_X Z_D2
        # DOES NOT REQUIRES EXTRA PRE-ROT TO SOLVE AS WE ALREADY TOGGLE X-D2 CORRELS
        beta_X_imp = np.zeros(4)
        op_idx_betaX = np.zeros(4, dtype=int)

        # FIXME: How to look for X without hardcoding the weightfunction number???
        ch_X_id = [i for i in range(len(value_names)) if b'X' in value_names[i]][0]
        z0 = 2**(3-idx_qubit_ro.index('X'))
        z1 = 2**(3-idx_qubit_ro.index('D1'))
        z0z1 = z1+z0
        op_list_bin = ['0000', format(z0, '#0{}b'.format(self.num_qubits+2))[2:],
                       format(z1, '#0{}b'.format(self.num_qubits+2))[2:],
                       format(z0z1, '#0{}b'.format(self.num_qubits+2))[2:]]
        #     op_id_list = [int(op,2) for op in op_list_bin]
        op_idx_betaX = [0, z0, z1, z0z1]
        #     print(op_id_list,op_list_bin)

        submatrix_B = matrix_B[op_idx_betaX, :]
        inv_subB = np.linalg.pinv(submatrix_B).transpose()
        beta_X_imp = inv_subB @ qubit_state_avg[ch_X_id, cal_point_seg_start:]

        # 8.bis Complicating betas on correl channel X-D4
        # M_XD4 = III + I_D4 I_X Z_D2 + I_D4 Z_X I_D2 + Z_D4 I_X I_D2
        #       + I_D4 Z_X Z_D2 + Z_D4 I_X Z_D2 + Z_D4 Z_X I_D2 +
        #       + Z_D4 Z_X Z_D2
        # REQUIRES EXTRA PRE-ROT TO SOLVE AS WE DO NOT TOGGLE D4-D2 CORRELS, AS WELL AS W3 CORRELS
        # EXTRA ROTS ARE (D4,X,Z2,D2 notation) XIIX & XXIX
        beta_XD4_imp = np.zeros(8)
        op_idx_betaXD4 = np.zeros(8, dtype=int)

        ch_XD4_id = 2 # from correlations variable above
        z0 = 2**(3-idx_qubit_ro.index('X'))
        z1 = 2**(3-idx_qubit_ro.index('D1'))
        z2 = 2**(3-idx_qubit_ro.index('D3'))
        z0z1 = z1+z0
        z0z2 = z0+z2
        z1z2 = z1+z2
        z0z1z2 = z0+z1+z2
        op_list_bin = ['0000',
                       format(z0, '#0{}b'.format(self.num_qubits+2))[2:],
                       format(z1, '#0{}b'.format(self.num_qubits+2))[2:],
                       format(z2, '#0{}b'.format(self.num_qubits+2))[2:],
                       format(z0z1, '#0{}b'.format(self.num_qubits+2))[2:],
                       format(z0z2, '#0{}b'.format(self.num_qubits+2))[2:],
                       format(z1z2, '#0{}b'.format(self.num_qubits+2))[2:],
                       format(z0z1z2, '#0{}b'.format(self.num_qubits+2))[2:]]
        #     op_id_list = [int(op,2) for op in op_list_bin]
        op_idx_betaXD4 = [0, z0, z1, z2, z0z1, z0z2, z1z2, z0z1z2]
        #     print(op_id_list,op_list_bin)

        submatrix_B = matrix_B[op_idx_betaXD4, :]
        inv_subB = np.linalg.pinv(submatrix_B).transpose()
        beta_XD4_imp = inv_subB @ correl_avg[cal_point_seg_start:, ch_XD4_id]

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
        if self.complexity_of_readout_model > 0:
            ch_w1_id = ch_X_id
            this_M_matrix = np.zeros(
                (num_prerots, num_ops))  # prepare M_matrix
            for ir, id_rot in enumerate(pre_rot_list):
                rot_bin = format(id_rot, '#0{}b'.format(self.num_qubits+2))[2:]
                # grabbing betas and operators
                this_betas = beta_X_imp
                this_op = op_idx_betaX
                for i_b, bt in enumerate(this_betas):
                    # print(i_b,bt,this_op)
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#0{}b'.format(self.num_qubits+2))[2:]
                    # decide the sign
                    sign = np.product([1-2*int(rot_bin[k])*int(op_bin[k])
                                       for k in range(len(op_bin))])
            #             print(ir,id_op)
                    this_M_matrix[ir, id_op] = sign*bt
            M_matrix[ch_w1_id *
                     num_prerots:(ch_w1_id+1)*num_prerots, :] = this_M_matrix

        if self.complexity_of_readout_model > 1:
            ch_w2_id = ch_XD4_id
            this_M_matrix = np.zeros(
                (num_prerots, num_ops))  # prepare M_matrix
            for ir, id_rot in enumerate(pre_rot_list):
                rot_bin = format(id_rot, '#0{}b'.format(self.num_qubits+2))[2:]
                # grabbing betas and operators
                this_betas = beta_XD4_imp
                this_op = op_idx_betaXD4
                for i_b, bt in enumerate(this_betas):
                    # print(i_b,bt,this_op)
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#0{}b'.format(self.num_qubits+2))[2:]
                    # decide the sign
                    sign = np.product([1-2*int(rot_bin[k])*int(op_bin[k])
                                       for k in range(len(op_bin))])
            #             print(ir,id_op)
                    this_M_matrix[ir, id_op] = sign*bt
            M_matrix[36+ch_w2_id*num_prerots:36 +
                     (ch_w2_id+1)*num_prerots, :] = this_M_matrix

        M_nobeta0 = M_matrix[:, 1:]
        beta0_vec = M_matrix[:, 0]
        inv_M_nobeta0 = np.linalg.pinv(M_nobeta0)

        # 10. performing tomographic inversion
        tomo_dict = {'IIII': 1}
        for basis in ['Z', 'X']:
            prerot_mmt_vec = np.zeros((inv_M_nobeta0.shape[1]))
            pre_rot_name_list = [
                basis+'-'+format(p, '#0{}b'.format(self.num_qubits+2))[2:].replace('0', 'I').replace('1', 'X') for p in pre_rot_list]
            pre_rot_idx_list = [combinations.index(
                p) for p in pre_rot_name_list]

            for ch_w1_id in range(num_chs_w1):
                prerot_mmt_vec[ch_w1_id*num_prerots:(
                    ch_w1_id+1)*num_prerots] = qubit_state_avg[ch_w1_id, pre_rot_idx_list]
            for ch_w2_id in range(num_chs_w2):
                prerot_mmt_vec[36+ch_w2_id*num_prerots:36 +
                               (ch_w2_id+1)*num_prerots] = correl_avg[pre_rot_idx_list, ch_w2_id]
            pauli_terms = inv_M_nobeta0 @ (prerot_mmt_vec-beta0_vec)
            op_labels = [format(p, '#0{}b'.format(self.num_qubits+2))[2:].replace(
                '0', 'I').replace('1', basis) for p in range(16)]
            for i_op, op in enumerate(op_labels):
                if i_op > 0:
                    tomo_dict[op] = pauli_terms[i_op-1]

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
        self.plot_dicts['pauli_operators_tomo'] = {
            'plotfn': tfd_an.plot_pauli_ops,
            'pauli_terms': self.proc_data_dict['pauli_terms'],
            'energy_terms': self.proc_data_dict['energy_terms']
        }
        self.plot_dicts['pauli_operators_tomo_full'] = {
            'plotfn': tfd_an.plot_all_pauli_ops,
            'full_dict': self.proc_data_dict['quantities_of_interest']['full_tomo_dict']
        }
        self.plot_dicts['expectation_values'] = {
            'plotfn': tfd_an.plot_expectation_values_TFD,
            'full_dict': self.proc_data_dict['quantities_of_interest']['full_tomo_dict'],
            'T': self.T

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


class TFD_fullTomo_2Q(tfd_an.TFD_Analysis_Pauli_Strings):
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

    def extract_data(self):
        """
        This is a new style (sept 2019) data extraction.
        This could at some point move to a higher level class.
        """
        self.get_timestamps()
        self.timestamp = self.timestamps[0]

        data_fp = get_datafilepath_from_timestamp(self.timestamp)
        param_spec = {
            'data': ('Experimental Data/Data', 'dset'),
            'combinations':  ('Experimental Data/Experimental Metadata/combinations', 'dset'),
            'gibbs_qubits':  ('Experimental Data/Experimental Metadata/gibbs_qubits', 'dset'),
            'value_names': ('Experimental Data', 'attr:value_names')}

        self.raw_data_dict = h5d.extract_pars_from_datafile(
            data_fp, param_spec)

        # For some reason the list is stored a list of length 1 arrays...
        self.raw_data_dict['combinations'] = [
            c[0] for c in self.raw_data_dict['combinations']]
        self.raw_data_dict['gibbs_qubits'] = [
            g[0] for g in self.raw_data_dict['gibbs_qubits']]

        # Parts added to be compatible with base analysis data requirements
        self.raw_data_dict['timestamps'] = self.timestamps
        self.raw_data_dict['folder'] = os.path.split(data_fp)[0]

    def process_data(self):
        self.proc_data_dict = {}
        combinations = self.raw_data_dict['combinations']
        self.num_states = 2**self.num_qubits
        centers_vec = np.zeros((self.num_states, self.num_qubits))
        self.num_segments = len(combinations)
        cal_point_seg_start = self.num_segments - self.num_states # 18 for 34 segments
        self.cal_point_seg_start = cal_point_seg_start
        correlations = [['D1', 'Z1'], ['D1', 'X'], ['X', 'D3'], ['Z1', 'D3']]
        idx_qubit_ro = ['D1', 'Z1', 'X', 'D3']

        partial_qubits = self.raw_data_dict['gibbs_qubits']
        partial_qubits_idx = [idx_qubit_ro.index(q) for i_q, q in enumerate(partial_qubits)]
        partial_correls_idx = [correlations.index(partial_qubits)]

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

        # 8. Computing inversion matrix for tomo
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

        list_ch_w1 = partial_qubits_idx
        list_ch_w2 = partial_correls_idx
        num_1q_ch = len(list_ch_w1)
        num_2q_ch = len(list_ch_w2)
        self.num_partial_qubits = 2
        prerot_vector = combinations[:cal_point_seg_start]
        num_prerot = len(prerot_vector)
        whole_M_matrix = np.zeros((num_prerot*(num_1q_ch+num_2q_ch), 4**self.num_partial_qubits))

        for i_prerot, prerot in enumerate(prerot_vector):
            this_prerot_bases = tomo_func.grab_bases_from_prerot(prerot, partial_qubits_idx)
            this_flip_bin = tomo_func.grab_flips_from_prerot(prerot).replace('I', '0').replace('F', '1')  # I=0;F=1
            for i_ch,ch_w1_id in enumerate(list_ch_w1):
                for i_op, op in enumerate(op_idx_w1[ch_w1_id, :]):
                    this_beta = betas_w1[ch_w1_id, i_op]
                    this_op_bin = format(op, '#0{}b'.format(self.num_qubits+2))[2:]  # I=0;Z=1
                    this_partial_op_bin = [this_op_bin[q_id] for q_id in partial_qubits_idx]
                    this_partial_op_bin = this_partial_op_bin[0]+this_partial_op_bin[1]
                    op_str = this_partial_op_bin.replace('0', 'I').replace('1', 'Z')
                    rotated_op_idx, rotated_op = tomo_func.rotate_operator(op_str, this_prerot_bases)
                    this_sign = np.product([1-2*int(this_flip_bin[k])*int(this_partial_op_bin[k])
                                            for k in range(len(this_partial_op_bin))]) # function of flips and this operator.
                    whole_M_matrix[i_prerot+i_ch*num_prerot,
                                   rotated_op_idx] = this_sign*this_beta
            for i_ch,ch_w2_id in enumerate(list_ch_w2):
                for i_op, op in enumerate(op_idx_w2[ch_w2_id,:]):
                    this_beta = betas_w2[ch_w2_id,i_op]
                    this_op_bin = format(op, '#0{}b'.format(self.num_qubits+2))[2:] # I=0;Z=1
                    this_partial_op_bin = [this_op_bin[c_id] for c_id in partial_qubits_idx]
                    this_partial_op_bin = this_partial_op_bin[0]+this_partial_op_bin[1]
                    op_str = this_partial_op_bin.replace('0', 'I').replace('1', 'Z')
        #             print(op,op_str,this_op_bin,this_prerot_bases,this_partial_op_bin)
                    rotated_op_idx, rotated_op = tomo_func.rotate_operator(op_str,this_prerot_bases)
                    this_sign = np.product([1-2*int(this_flip_bin[k])*int(this_partial_op_bin[k])
                                            for k in range(len(this_partial_op_bin))]) # function of flips and this operator.
                    whole_M_matrix[i_prerot+(num_1q_ch+i_ch)*num_prerot,
                                   rotated_op_idx] = this_sign*this_beta
        # 9. Inversion
        prerot_mmt_vec = np.concatenate((qubit_state_avg[partial_qubits_idx[0],:cal_point_seg_start],
                                         qubit_state_avg[partial_qubits_idx[1],:cal_point_seg_start],
                                         correl_avg[:cal_point_seg_start,partial_correls_idx[0]]))
        whole_M_nobeta0 = whole_M_matrix[:, 1:]
        beta0_vec = whole_M_matrix[:, 0]
        inv_whole_M_nobeta0 = np.linalg.pinv(whole_M_nobeta0)
        pauli_terms = inv_whole_M_nobeta0 @ (prerot_mmt_vec-beta0_vec)
        # 10. Keeping only relevant terms from the tomo
        self.operators_labels = ['II', 'IX', 'IY', 'IZ',
                                 'XI', 'XX', 'XY', 'XZ',
                                 'YI', 'YX', 'YY', 'YZ',
                                 'ZI', 'ZX', 'ZY', 'ZZ',
                                 ]
        self.op_values = {}
        self.op_values['II'] = 1
        self.op_values.update({self.operators_labels[i+1]: p for i, p in enumerate(pauli_terms)})
        self.proc_data_dict['quantities_of_interest'] = {
            'g': self.g, 'T': self.T,
            'full_tomo_dict': self.op_values}

    def prepare_plots(self):
        # plotting of bars disabled
        # self.plot_dicts['pauli_operators_Tomo'] = {
        #     'plotfn': tfd_an.plot_pauli_op,
        #     'pauli_terms': self.operators_labels[1:],
        #     'energy_terms': pauli_terms
        # }
        for ch_id, ch in enumerate(self.raw_data_dict['ro_sq_ch_names']):
            self.plot_dicts['TV_{}'.format(ch)] = {
                'plotfn': plot_tv_mode_with_ticks,
                'xticks': self.raw_data_dict['combinations'],
                'yvals': self.raw_data_dict['ro_sq_raw_signal'][ch_id,:],
                'ylabel': ch,
                'shade_from': self.cal_point_seg_start,
                # 'yunit': self.raw_data_dict['value_units'][0][i],
                'title': (self.raw_data_dict['timestamps'][0]+' - ' + ' TV: {}'.format(ch))}
        for ch_id, ch in enumerate(self.raw_data_dict['ro_tq_ch_names']):
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
    ax.fill_betweenx(x1=[shade_from],x2=[xvals.max()],y=[yvals.min(),yvals.max()], alpha=0.5, color='grey')
    ax.plot(xvals,yvals,'-o')
    ax.set_xticks(xvals)
    ax.set_xticklabels(xticks, rotation=xticks_rotation)

    # ax.set_ylabel(ylabel+ ' ({})'.format(yunit))
    ax.set_title(title)