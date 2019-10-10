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

        data_shots = self.raw_data_dict['data'][:, :]
        self.proc_data_dict['raw_shots'] = data_shots[:, 1:]
        value_names = self.raw_data_dict['value_names']

        # 1. calculate centers of states
        for id_state in range(self.num_states):
            centers_this_state = np.mean(data_shots[cal_point_seg_start+id_state::self.num_segments, :],
                                         axis=0)[1:]
            centers_vec[id_state, :] = centers_this_state

        # 2. compute matrix for betas
        matrix_B = np.zeros((self.num_states, self.num_states))

        for i in range(self.num_states):
            for j in range(self.num_states):
                # RO operator with I & Z from binary decomposition of i (0=I, 1=Z)
                # format is #0(n+2)b, [2:] erases the bin str indication
                operator_i = format(i, '#06b')[2:]
                # computational state j (binary decompose j
                # format is #0(n+2)b, [2:] erases the bin str indication
                state_j = format(j, '#06b')[2:]
                """
                trace is the product of 1 (if I) or (+/-1 if Z) for each qubit.
                For two binary words operator_word and state_word we need
                operator_word b_3 b_2 b_1 b_0
                state_word    s_3 s_2 s_1 s_0
                -----------------------------
                output_word   o_3 o_2 o_1 o_0

                where o_i follows
                if b_i==0:
                    o_i = 1
                else:
                    o_i = 1 - 2*s_i

                Solutions are o_i = 1 - 2*s_i*b_i
                Final solution is Prod{o_i}
                """
                trace_op_rho = np.product(
                    [1-2*int(state_j[k])*int(operator_i[k]) for k in range(len(state_j))])
                matrix_B[i, j] = trace_op_rho

        # 3. Computing threshold
        mn_voltages = {}
        for i, ch_name in enumerate(value_names):
            ch_id = list(value_names).index(ch_name)
            ch_data = data_shots[:, ch_id+1]  # select per channel
            mn_voltages[ch_name] = {'0': [], '1': []}
            for i_c, c in enumerate(combinations):
                if c[i] == '0':
                    mn_voltages[ch_name]['0'].append(
                        list(ch_data[i_c::self.num_states]))
                elif c[i] == '1':
                    mn_voltages[ch_name]['1'].append(
                        list(ch_data[i_c::self.num_states]))
            mn_voltages[ch_name]['0'] = np.mean(
                flatten_list(mn_voltages[ch_name]['0']))
            mn_voltages[ch_name]['1'] = np.mean(
                flatten_list(mn_voltages[ch_name]['1']))
            mn_voltages[ch_name]['threshold'] = np.mean(
                [mn_voltages[ch_name]['0'],
                 mn_voltages[ch_name]['1']])

        # 4. Bining weight-1 data
        shots_discr = np.zeros((data_shots.shape[0], 4))
        qubit_state_avg = np.zeros((self.num_qubits, self.num_segments))

        for k in mn_voltages.keys():
            id_channel = np.sum(np.where(value_names == k, np.arange(1, 5), 0))
            this_q_data = data_shots[:, id_channel]
            this_th = mn_voltages[k]['threshold']
            shots_discr[:, id_channel -
                        1] = np.where(this_q_data > this_th, -1, 1)
            qubit_state_avg[id_channel-1, :] = [np.mean(shots_discr[i_seg::self.num_segments,
                                                                    id_channel-1]) for i_seg in range(self.num_segments)]
        # 5. Compute betas weight-1
        betas_w1 = np.zeros((4, 2))
        op_idx_w1 = np.zeros((4, 2), dtype=int)
        for i in range(self.num_qubits):
            op_list_bin = ['0000', format(2**(3-i), '#06b')[2:]]
            op_id_list = [int(op, 2) for op in op_list_bin]
            op_idx_w1[i, :] = op_id_list
        #     print(op_id_list)

            submatrix_B = matrix_B[op_id_list, :]
            inv_subB = np.linalg.pinv(submatrix_B).transpose()
            betas_w1[i, :] = inv_subB @ qubit_state_avg[i, cal_point_seg_start:]

        # 6. Bining weight-2 data
        idx_qubit_ro = ['D4', 'X', 'Z2', 'D2']
        correlations = [['Z2', 'D2'], ['D2', 'X'], ['D4', 'X'], ['D4', 'Z2']]
        correlations_idx = [
            [idx_qubit_ro.index(c[0]), idx_qubit_ro.index(c[1])] for c in correlations]

        correl_discr = np.zeros((shots_discr.shape[0], len(correlations_idx)))
        correl_avg = np.zeros((self.num_segments, len(correlations_idx)))
        for i, c in enumerate(correlations_idx):
            correl_discr[:, i] = shots_discr[:, c[0]]*shots_discr[:, c[1]]
            correl_avg[:, i] = [
                np.mean(correl_discr[i_seg::self.num_segments, i]) for i_seg in range(self.num_segments)]

        # 7. Compute betas weight-2
        betas_w2 = np.zeros((4, 4))
        op_idx_w2 = np.zeros((4, 4), dtype=int)
        for i_c, c in enumerate(correlations):
            z0 = 2**(3-idx_qubit_ro.index(c[0]))
            z1 = 2**(3-idx_qubit_ro.index(c[1]))
            z0z1 = z1+z0
            op_list_bin = ['0000', format(z0, '#06b')[2:],
                           format(z1, '#06b')[2:],
                           format(z0z1, '#06b')[2:]]
        #     op_id_list = [int(op,2) for op in op_list_bin]
            op_id_list = [0, z0, z1, z0z1]
            op_idx_w2[i_c, :] = op_id_list
        #     print(op_id_list,op_list_bin)

            submatrix_B = matrix_B[op_id_list, :]
            inv_subB = np.linalg.pinv(submatrix_B).transpose()
            betas_w2[i_c, :] = inv_subB @ correl_avg[cal_point_seg_start:, i_c]
        # 8. Complicating betas on qubit X
        # M_X = II + I_X Z_D2 + Z_X I_D2 + Z_X Z_D2
        # DOES NOT REQUIRES EXTRA PRE-ROT TO SOLVE AS WE ALREADY TOGGLE X-D2 CORRELS
        beta_X_imp = np.zeros(4)
        op_idx_betaX = np.zeros(4, dtype=int)

        # FIXME: How to look for X without hardcoding the weightfunction number???
        ch_X_id = [i for i in range(len(value_names)) if b'X' in value_names[i]][0]
        z0 = 2**(3-idx_qubit_ro.index('X'))
        z1 = 2**(3-idx_qubit_ro.index('D2'))
        z0z1 = z1+z0
        op_list_bin = ['0000', format(z0, '#06b')[2:],
                       format(z1, '#06b')[2:],
                       format(z0z1, '#06b')[2:]]
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
        z1 = 2**(3-idx_qubit_ro.index('D2'))
        z2 = 2**(3-idx_qubit_ro.index('D4'))
        z0z1 = z1+z0
        z0z2 = z0+z2
        z1z2 = z1+z2
        z0z1z2 = z0+z1+z2
        op_list_bin = ['0000',
                       format(z0, '#06b')[2:],
                       format(z1, '#06b')[2:],
                       format(z2, '#06b')[2:],
                       format(z0z1, '#06b')[2:],
                       format(z0z2, '#06b')[2:],
                       format(z1z2, '#06b')[2:],
                       format(z0z1z2, '#06b')[2:]]
        #     op_id_list = [int(op,2) for op in op_list_bin]
        op_idx_betaXD4 = [0, z0, z1, z2, z0z1, z0z2, z1z2, z0z1z2]
        #     print(op_id_list,op_list_bin)

        submatrix_B = matrix_B[op_idx_betaXD4, :]
        inv_subB = np.linalg.pinv(submatrix_B).transpose()
        beta_XD4_imp = inv_subB @ correl_avg[cal_point_seg_start:, ch_XD4_id]

        """
        # 9. Computing inversion matrix for tomo
        For channel K:
        Construct matrix procedure
            grab m_i, corresponds to pre-rot #bin(i) (with 0s for Is and 1s for Xs)
            grab betas_channel, (beta_i corresponding to op_i corresponding to slot i of operators vector)
            for each beta/op pair
                beta_i=betas_w1[ch,op] corresponds to i=op_idx_w1[ch,op] op_bin=format(i, '#06b')[2:]
                rot_bin=format(i_rot, '#06b')[2:]
                for each Z in op_i, if there is an X in pre-rot, flip sign of beta.
                solved by writting
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
                rot_bin = format(id_rot, '#06b')[2:]
                # grabbing betas and operators
                this_betas = betas_w1[ch_w1_id, :]
                this_op = op_idx_w1[ch_w1_id, :]
                for i_b, bt in enumerate(this_betas):
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#06b')[2:]
                    # decide the sign
                    sign = np.product([1-2*int(rot_bin[k])*int(op_bin[k])
                                       for k in range(len(op_bin))])
        #             print(ir,id_op)
                    this_M_matrix[ir, id_op] = sign*bt
            M_matrix[ch_w1_id *
                     num_prerots:(ch_w1_id+1)*num_prerots, :] = this_M_matrix
        # now w2 channels
        for ch_w2_id in range(num_chs_w2):
            this_M_matrix = np.zeros(
                (num_prerots, num_ops))  # prepare M_matrix
            for ir, id_rot in enumerate(pre_rot_list):
                rot_bin = format(id_rot, '#06b')[2:]
                # grabbing betas and operators
                this_betas = betas_w2[ch_w2_id, :]
                this_op = op_idx_w2[ch_w2_id, :]
                for i_b, bt in enumerate(this_betas):
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#06b')[2:]
                    # decide the sign
                    sign = np.product([1-2*int(rot_bin[k])*int(op_bin[k])
                                       for k in range(len(op_bin))])
        #             print(ir,id_op)
                    this_M_matrix[ir, id_op] = sign*bt
            M_matrix[36+ch_w2_id*num_prerots:36 +
                     (ch_w2_id+1)*num_prerots, :] = this_M_matrix
        # if enabled, enhance mmt model for X
        if self.complexity_of_readout_model > 0:
            ch_w1_id = ch_X_id
            this_M_matrix = np.zeros(
                (num_prerots, num_ops))  # prepare M_matrix
            for ir, id_rot in enumerate(pre_rot_list):
                rot_bin = format(id_rot, '#06b')[2:]
                # grabbing betas and operators
                this_betas = beta_X_imp
                this_op = op_idx_betaX
                for i_b, bt in enumerate(this_betas):
                    # print(i_b,bt,this_op)
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#06b')[2:]
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
                rot_bin = format(id_rot, '#06b')[2:]
                # grabbing betas and operators
                this_betas = beta_XD4_imp
                this_op = op_idx_betaXD4
                for i_b, bt in enumerate(this_betas):
                    # print(i_b,bt,this_op)
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#06b')[2:]
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
                basis+'-'+format(p, '#06b')[2:].replace('0', 'I').replace('1', 'X') for p in pre_rot_list]
            pre_rot_idx_list = [combinations.index(
                p) for p in pre_rot_name_list]

            for ch_w1_id in range(num_chs_w1):
                prerot_mmt_vec[ch_w1_id*num_prerots:(
                    ch_w1_id+1)*num_prerots] = qubit_state_avg[ch_w1_id, pre_rot_idx_list]
            for ch_w2_id in range(num_chs_w2):
                prerot_mmt_vec[36+ch_w2_id*num_prerots:36 +
                               (ch_w2_id+1)*num_prerots] = correl_avg[pre_rot_idx_list, ch_w2_id]
            pauli_terms = inv_M_nobeta0 @ (prerot_mmt_vec-beta0_vec)
            op_labels = [format(p, '#06b')[2:].replace(
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
        self.plot_dicts['pauli_operators_Tomo'] = {
            'plotfn': tfd_an.plot_pauli_ops,
            'pauli_terms': self.proc_data_dict['pauli_terms'],
            'energy_terms': self.proc_data_dict['energy_terms']
        }


class TFD_3CZ_Analysis_Pauli_FullTomo(tfd_an.TFD_3CZ_Analysis_Pauli_Strings):
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

        data_shots = self.raw_data_dict['data'][:, :]
        self.proc_data_dict['raw_shots'] = data_shots[:, 1:]
        value_names = self.raw_data_dict['value_names']

        # 1. calculate centers of states
        for id_state in range(self.num_states):
            centers_this_state = np.mean(data_shots[cal_point_seg_start+id_state::self.num_segments, :],
                                         axis=0)[1:]
            centers_vec[id_state, :] = centers_this_state

        # 2. compute matrix for betas
        matrix_B = np.zeros((self.num_states, self.num_states))

        for i in range(self.num_states):
            for j in range(self.num_states):
                # RO operator with I & Z from binary decomposition of i (0=I, 1=Z)
                # format is #0(n+2)b, [2:] erases the bin str indication
                operator_i = format(i, '#06b')[2:]
                # computational state j (binary decompose j
                # format is #0(n+2)b, [2:] erases the bin str indication
                state_j = format(j, '#06b')[2:]
                """
                trace is the product of 1 (if I) or (+/-1 if Z) for each qubit.
                For two binary words operator_word and state_word we need
                operator_word b_3 b_2 b_1 b_0
                state_word    s_3 s_2 s_1 s_0
                -----------------------------
                output_word   o_3 o_2 o_1 o_0

                where o_i follows
                if b_i==0:
                    o_i = 1
                else:
                    o_i = 1 - 2*s_i

                Solutions are o_i = 1 - 2*s_i*b_i
                Final solution is Prod{o_i}
                """
                trace_op_rho = np.product(
                    [1-2*int(state_j[k])*int(operator_i[k]) for k in range(len(state_j))])
                matrix_B[i, j] = trace_op_rho

        # 3. Computing threshold
        mn_voltages = {}
        for i, ch_name in enumerate(value_names):
            ch_id = list(value_names).index(ch_name)
            ch_data = data_shots[:, ch_id+1]  # select per channel
            mn_voltages[ch_name] = {'0': [], '1': []}
            for i_c, c in enumerate(combinations):
                if c[i] == '0':
                    mn_voltages[ch_name]['0'].append(
                        list(ch_data[i_c::self.num_states]))
                elif c[i] == '1':
                    mn_voltages[ch_name]['1'].append(
                        list(ch_data[i_c::self.num_states]))
            mn_voltages[ch_name]['0'] = np.mean(
                flatten_list(mn_voltages[ch_name]['0']))
            mn_voltages[ch_name]['1'] = np.mean(
                flatten_list(mn_voltages[ch_name]['1']))
            mn_voltages[ch_name]['threshold'] = np.mean(
                [mn_voltages[ch_name]['0'],
                 mn_voltages[ch_name]['1']])

        # 4. Bining weight-1 data
        shots_discr = np.zeros((data_shots.shape[0], 4))
        qubit_state_avg = np.zeros((self.num_qubits, self.num_segments))

        for k in mn_voltages.keys():
            id_channel = np.sum(np.where(value_names == k, np.arange(1, 5), 0))
            this_q_data = data_shots[:, id_channel]
            this_th = mn_voltages[k]['threshold']
            shots_discr[:, id_channel -
                        1] = np.where(this_q_data > this_th, -1, 1)
            qubit_state_avg[id_channel-1, :] = [np.mean(shots_discr[i_seg::self.num_segments,
                                                                    id_channel-1]) for i_seg in range(self.num_segments)]
        # 5. Compute full betas weight-1
        betas_w1 = np.zeros((4, self.num_states))
        op_idx_w1 = np.zeros((4, self.num_states), dtype=int)
        for i in range(self.num_qubits):
            op_list_bin = [format(i, '#06b')[2:] for i in range(self.num_states)]
            op_id_list = [int(op, 2) for op in op_list_bin]
            op_idx_w1[i, :] = op_id_list
        #     print(op_id_list)

            submatrix_B = matrix_B[op_id_list, :]
            inv_subB = np.linalg.pinv(submatrix_B).transpose()
            betas_w1[i, :] = inv_subB @ qubit_state_avg[i, cal_point_seg_start:]

        # 6. Bining weight-2 data
        idx_qubit_ro = ['D4', 'X', 'Z2', 'D2']
        correlations = [['Z2', 'D2'], ['D2', 'X'], ['D4', 'X'], ['D4', 'Z2']]
        correlations_idx = [
            [idx_qubit_ro.index(c[0]), idx_qubit_ro.index(c[1])] for c in correlations]

        correl_discr = np.zeros((shots_discr.shape[0], len(correlations_idx)))
        correl_avg = np.zeros((self.num_segments, len(correlations_idx)))
        for i, c in enumerate(correlations_idx):
            correl_discr[:, i] = shots_discr[:, c[0]]*shots_discr[:, c[1]]
            correl_avg[:, i] = [
                np.mean(correl_discr[i_seg::self.num_segments, i]) for i_seg in range(self.num_segments)]

        # 7. Compute full betas weight-2
        betas_w2 = np.zeros((4, self.num_states))
        op_idx_w2 = np.zeros((4, self.num_states), dtype=int)
        for i_c, c in enumerate(correlations):
            op_list_bin = [format(i, '#06b')[2:] for i in range(self.num_states)]
        #     op_id_list = [int(op,2) for op in op_list_bin]
            op_id_list = [int(op, 2) for op in op_list_bin]
            op_idx_w2[i_c, :] = op_id_list
        #     print(op_id_list,op_list_bin)

            submatrix_B = matrix_B[op_id_list, :]
            inv_subB = np.linalg.pinv(submatrix_B).transpose()
            betas_w2[i_c, :] = inv_subB @ correl_avg[cal_point_seg_start:, i_c]


        """
        # 9. Computing inversion matrix for tomo
        For channel K:
        Construct matrix procedure
            grab m_i, corresponds to pre-rot #bin(i) (with 0s for Is and 1s for Xs)
            grab betas_channel, (beta_i corresponding to op_i corresponding to slot i of operators vector)
            for each beta/op pair
                beta_i=betas_w1[ch,op] corresponds to i=op_idx_w1[ch,op] op_bin=format(i, '#06b')[2:]
                rot_bin=format(i_rot, '#06b')[2:]
                for each Z in op_i, if there is an X in pre-rot, flip sign of beta.
                solved by writting
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
                rot_bin = format(id_rot, '#06b')[2:]
                # grabbing betas and operators
                this_betas = betas_w1[ch_w1_id, :]
                this_op = op_idx_w1[ch_w1_id, :]
                for i_b, bt in enumerate(this_betas):
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#06b')[2:]
                    # decide the sign
                    sign = np.product([1-2*int(rot_bin[k])*int(op_bin[k])
                                       for k in range(len(op_bin))])
        #             print(ir,id_op)
                    this_M_matrix[ir, id_op] = sign*bt
            M_matrix[ch_w1_id *
                     num_prerots:(ch_w1_id+1)*num_prerots, :] = this_M_matrix
        # now w2 channels
        for ch_w2_id in range(num_chs_w2):
            this_M_matrix = np.zeros(
                (num_prerots, num_ops))  # prepare M_matrix
            for ir, id_rot in enumerate(pre_rot_list):
                rot_bin = format(id_rot, '#06b')[2:]
                # grabbing betas and operators
                this_betas = betas_w2[ch_w2_id, :]
                this_op = op_idx_w2[ch_w2_id, :]
                for i_b, bt in enumerate(this_betas):
                    id_op = this_op[i_b]
                    op_bin = format(id_op, '#06b')[2:]
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
                basis+'-'+format(p, '#06b')[2:].replace('0', 'I').replace('1', 'X') for p in pre_rot_list]
            pre_rot_idx_list = [combinations.index(
                p) for p in pre_rot_name_list]

            for ch_w1_id in range(num_chs_w1):
                prerot_mmt_vec[ch_w1_id*num_prerots:(
                    ch_w1_id+1)*num_prerots] = qubit_state_avg[ch_w1_id, pre_rot_idx_list]
            for ch_w2_id in range(num_chs_w2):
                prerot_mmt_vec[36+ch_w2_id*num_prerots:36 +
                               (ch_w2_id+1)*num_prerots] = correl_avg[pre_rot_idx_list, ch_w2_id]
            pauli_terms = inv_M_nobeta0 @ (prerot_mmt_vec-beta0_vec)
            op_labels = [format(p, '#06b')[2:].replace(
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
        self.plot_dicts['pauli_operators_Tomo'] = {
            'plotfn': tfd_an.plot_pauli_ops,
            'pauli_terms': self.proc_data_dict['pauli_terms'],
            'energy_terms': self.proc_data_dict['energy_terms']
        }
