import numpy as np
from functools import reduce
def flatten_list(l): return reduce(lambda x, y: x+y, l)

def compute_beta_matrix(num_qubits):
    """
    Computes the matrix necesary to invert the beta coefficients.
    """
    num_states = 2**num_qubits
    matrix_B = np.zeros((num_states, num_states))

    for i in range(num_states):
        for j in range(num_states):
            # RO operator with I & Z from binary decomposition of i (0=I, 1=Z)
            # format is #0(n+2)b, [2:] erases the bin str indication
            operator_i = format(i, '#0{}b'.format(num_qubits+2))[2:]
            # computational state j (binary decompose j
            # format is #0(n+2)b, [2:] erases the bin str indication
            state_j = format(j, '#0{}b'.format(num_qubits+2))[2:]
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
    return matrix_B

def define_thresholds_avg(data_shots, value_names, combinations, num_states):
    """
    Defines the thresholds to be used in tomography
    """
    mn_voltages = {}
    for i, ch_name in enumerate(value_names):
        ch_id = list(value_names).index(ch_name)
        ch_data = data_shots[:, ch_id+1]  # select per channel
        mn_voltages[ch_name] = {'0': [], '1': []}
        for i_c, c in enumerate(combinations):
            if c[i] == '0':
                mn_voltages[ch_name]['0'].append(
                    list(ch_data[i_c::num_states]))
            elif c[i] == '1':
                mn_voltages[ch_name]['1'].append(
                    list(ch_data[i_c::num_states]))
        mn_voltages[ch_name]['0'] = np.mean(
            flatten_list(mn_voltages[ch_name]['0']))
        mn_voltages[ch_name]['1'] = np.mean(
            flatten_list(mn_voltages[ch_name]['1']))
        mn_voltages[ch_name]['threshold'] = np.mean(
            [mn_voltages[ch_name]['0'],
             mn_voltages[ch_name]['1']])
    return mn_voltages

def threshold_weight1_data(data_shots, mn_voltages, num_qubits, num_segments, value_names):
    """
    Classifies tomo data based on thresholds given
    """
    shots_discr = np.zeros((data_shots.shape[0], num_qubits))
    qubit_state_avg = np.zeros((num_qubits, num_segments))

    for k in mn_voltages.keys():
        id_channel = np.sum(np.where(value_names == k, np.arange(num_qubits)+1, 0))
        this_q_data = data_shots[:, id_channel]
        this_th = mn_voltages[k]['threshold']
        shots_discr[:, id_channel -
                    1] = np.where(this_q_data > this_th, -1, 1)
        qubit_state_avg[id_channel-1, :] = [np.mean(shots_discr[i_seg::num_segments,
                                                                id_channel-1]) for i_seg in range(num_segments)]
    return shots_discr,qubit_state_avg

def correlating_weight2_data(shots_discr, idx_qubit_ro, correlations, num_segments):
    """
    """
    correlations_idx = [
        [idx_qubit_ro.index(c[0]), idx_qubit_ro.index(c[1])] for c in correlations]

    correl_discr = np.zeros((shots_discr.shape[0], len(correlations_idx)))
    correl_avg = np.zeros((num_segments, len(correlations_idx)))
    for i, c in enumerate(correlations_idx):
        correl_discr[:, i] = shots_discr[:, c[0]]*shots_discr[:, c[1]]
        correl_avg[:, i] = [
            np.mean(correl_discr[i_seg::num_segments, i]) for i_seg in range(num_segments)]
    return correl_discr, correl_avg

def compute_betas_weight1(qubit_state_avg, matrix_B, num_qubits, cal_point_seg_start):
    """
    Computes weight-one betas
    """
    betas_w1 = np.zeros((num_qubits, 2))
    op_idx_w1 = np.zeros((num_qubits, 2), dtype=int)
    for i in range(num_qubits):
        op_list_bin = [format(0, '#0{}b'.format(num_qubits+2))[2:],
                       format(2**(num_qubits-1-i), '#0{}b'.format(num_qubits+2))[2:]]
        op_id_list = [int(op, 2) for op in op_list_bin]
        op_idx_w1[i, :] = op_id_list

        # print(op_id_list,op_idx_w1)
        submatrix_B = matrix_B[op_id_list, :]
        inv_subB = np.linalg.pinv(submatrix_B).transpose()
        betas_w1[i, :] = inv_subB @ qubit_state_avg[i, cal_point_seg_start:]
    return betas_w1, op_idx_w1

def compute_betas_weight2(matrix_B, correl_avg, correlations, idx_qubit_ro, num_qubits, cal_point_seg_start):
    """
    """
    betas_w2 = np.zeros((len(correlations), 4))
    op_idx_w2 = np.zeros((len(correlations), 4), # 4 comes out of 4 combinations in weight2 measurement operator
                         dtype=int)
    for i_c, c in enumerate(correlations):
        z0 = 2**(num_qubits-1-idx_qubit_ro.index(c[0]))
        z1 = 2**(num_qubits-1-idx_qubit_ro.index(c[1]))
        z0z1 = z1+z0
        op_list_bin = [format(0, '#0{}b'.format(num_qubits+2))[2:],
                       format(z0, '#0{}b'.format(num_qubits+2))[2:],
                       format(z1, '#0{}b'.format(num_qubits+2))[2:],
                       format(z0z1, '#0{}b'.format(num_qubits+2))[2:]]
        # op_id_list = [int(op,2) for op in op_list_bin]
        op_id_list = [0, z0, z1, z0z1]
        op_idx_w2[i_c, :] = op_id_list
        # print(op_id_list,op_list_bin)

        submatrix_B = matrix_B[op_id_list, :]
        inv_subB = np.linalg.pinv(submatrix_B).transpose()
        betas_w2[i_c, :] = inv_subB @ correl_avg[cal_point_seg_start:, i_c]
    return betas_w2, op_idx_w2


def grab_bases_from_prerot(prerotation_string, partial_qubits):
    return prerotation_string.split('-')[0]


def grab_flips_from_prerot(prerotation_string):
    return prerotation_string.split('-')[1]


def rotate_operator(op, bases):
    # needs convention of operators listing
    rotated_op_str = ''
    # print('[DEBUG] Tomo::operator_rotation')
    # print('[DEBUG] op={}'.format(op))
    # print('[DEBUG] bases={}'.format(bases))
    for i_ol, op_letter in enumerate(op):
        if op_letter == 'Z':
            rotated_op_str += bases[i_ol]
        elif op_letter == 'I':
            rotated_op_str += 'I'
        else:
            raise ValueError("Tomo::operator_rotation Measurement operator is not undestood {} in {}".format(op_letter,op))
    operator_str_base4 = rotated_op_str.replace('I', '0').replace('X', '1').replace('Y', '2').replace('Z', '3')
    rotated_op_idx = int(operator_str_base4,4) # transforms this into the integer in base 10
    
    return rotated_op_idx, rotated_op_str
