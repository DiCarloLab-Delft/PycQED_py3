import time
import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
# import dataprep for tomography module
# import tomography module
# using the data prep module of analysis V2
# from pycqed.analysis_v2 import tomography_dataprep as dataprep
from pycqed.analysis import measurement_analysis as ma
try:
    import qutip as qt
except ImportError as e:
    pass
    # logging.warning('Could not import qutip, tomo code will not work')

def reshape_block(shots_data, segments_per_block=16, block_size=4092, mode='truncate'):
    """
    inputs: shots_data 1D array of dimension N
    organizes data in blocks of dimension block_size.
    num of blocks is N/block_size
    """
    N = len(shots_data)
    # Data dimension needs to be an integer multiple of block_size
    assert(N%block_size==0)
    num_blocks = N//block_size
    full_segments = block_size//segments_per_block
    orfan_segments = block_size % segments_per_block
    missing_segments = segments_per_block - orfan_segments
#     print(N,num_blocks,full_segments,orfan_segments,missing_segments)
    reshaped_data = shots_data.reshape((num_blocks,block_size))
    if mode.lower()=='truncate':
        truncate_idx = full_segments*segments_per_block
        return reshaped_data[:,:truncate_idx]
    elif mode.lower()=='padd':
        padd_dim = (full_segments+1)*segments_per_block
        return_block = np.nan*np.ones((num_blocks,padd_dim))
        return_block[:,:block_size] = reshaped_data
        return return_block
    else:
        raise ValueError('Mode not understood. Needs to be truncate or padd')

def all_repetitions(shots_data,segments_per_block=16):
    flat_dim = shots_data.shape[0]*shots_data.shape[1]
    # Data dimension needs to divide the segments_per_block
    assert(flat_dim%segments_per_block==0)
    num_blocks = flat_dim // segments_per_block
    block_data = shots_data.reshape((num_blocks,segments_per_block))
    return block_data

def get_segments_average(shots_data, segments_per_block=16, block_size=4092, mode='truncate', average=True):
    reshaped_data = reshape_block(shots_data=shots_data,
                                      segments_per_block=segments_per_block,
                                      block_size=block_size,
                                      mode=mode)
    all_reps = all_repetitions(shots_data=reshaped_data,
                                       segments_per_block=segments_per_block)
    if average:
        return np.mean(all_reps,axis=0)
    else:
        return all_reps


class ExpectationValueCalculation_mmt_computer:

    def __init__(self, auto=True, label='', timestamp=None,
                 fig_format='png',
                 q0_label='q0',
                 q1_label='q1', close_fig=True, **kw):
        self.label = label
        self.timestamp = timestamp
        self.fig_format = fig_format
        # q0 == D2
        self.q0_label = q0_label
        # q1 == A
        self.q1_label = q1_label
        self.n_states = 2 ** 2
        self.ma_obj = ma.MeasurementAnalysis(auto=False, label=label,
                                             timestamp=timestamp)
        self.ma_obj.get_naming_and_values()
        # self.get_naming_and_values()
        # hard coded number of segments for a 2 qubit state tomography
        # constraint imposed by UHFLI
        self.nr_segments = 16
        # self.exp_name = os.path.split(self.folder)[-1][7:]

        avg_h1 = self.ma_obj.measured_values[0]
        avg_h2 = self.ma_obj.measured_values[1]
        avg_h12 = self.ma_obj.measured_values[2]
        h1_00 = np.mean(avg_h1[36:36+7])
        h1_01 = np.mean(avg_h1[43:43+7])
        h1_10 = np.mean(avg_h1[50:50+7])
        h1_11 = np.mean(avg_h1[57:])

        h2_00 = np.mean(avg_h2[36:36+7])
        h2_01 = np.mean(avg_h2[43:43+7])
        h2_10 = np.mean(avg_h2[50:50+7])
        h2_11 = np.mean(avg_h2[57:])

        h12_00 = np.mean(avg_h12[36:36+7])
        h12_01 = np.mean(avg_h12[43:43+7])
        h12_10 = np.mean(avg_h12[50:50+7])
        h12_11 = np.mean(avg_h12[57:])

        measurement_channel_1 = np.array([avg_h1[0], avg_h1[1], avg_h1[7],
                                          avg_h1[8], avg_h1[14], avg_h1[21],
                                          avg_h1[28], avg_h1[35]])
        measurement_channel_2 = np.array([avg_h2[0], avg_h2[1], avg_h2[7], avg_h2[8],
                                          avg_h2[14], avg_h2[21], avg_h2[28], avg_h2[35]])
        measurement_channel_3 = np.array([avg_h12[0], avg_h12[1], avg_h12[7],
                                          avg_h12[8], avg_h12[14], avg_h12[21],
                                          avg_h12[28],avg_h12[35]])
        self.measurements_tomo = np.array([measurement_channel_1,
                                           measurement_channel_2,
                                           measurement_channel_3]).flatten()
        # print(self.measurements_tomo)
        # print(len(self.measurements_tomo))

        # 108 x 1
        # get the calibration points by averaging over the five measurements
        # taken knowing the initial state we put in
        self.measurements_cal = np.array(
            [h1_00, h1_01, h1_10, h1_11,
             h2_00, h2_01, h2_10, h2_11,
             h12_00, h12_01, h12_10, h12_11])

    def _calibrate_betas(self):
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
        # print(cal_matrix)
        # print(np.linalg.inv(cal_matrix))
        betas = np.zeros(12)
        # print(self.measurements_cal[0:4])
        betas[0:4] = np.dot(np.linalg.inv(cal_matrix),
                            self.measurements_cal[0:4])
        self.betas_up = betas[0:4]
        betas[4:8] = np.dot(np.linalg.inv(cal_matrix),
                            self.measurements_cal[4:8])
        self.betas_p = betas[4:8]
        betas[8:] = np.dot(np.linalg.inv(cal_matrix),
                           self.measurements_cal[8:12])
        self.betas_pp = betas[8:]
        return betas

    def assemble_M_matrix_single_block(self, beta_array):
        M_matrix_single_block_row_1 = np.array([beta_array[0], beta_array[1],
                                                beta_array[2], beta_array[3],
                                                0, 0, 0, 0, 0, 0])
        M_matrix_single_block_row_2 = np.array([beta_array[0],
                                                -1*beta_array[1],
                                                beta_array[2],
                                                -1*beta_array[3],
                                                0, 0, 0, 0, 0, 0])
        M_matrix_single_block_row_3 = np.array([beta_array[0],
                                                beta_array[1],
                                                -1*beta_array[2],
                                                -1*beta_array[3],
                                                0, 0, 0, 0, 0, 0])
        M_matrix_single_block_row_4 = np.array([beta_array[0],
                                                -1*beta_array[1],
                                                -1*beta_array[2],
                                                beta_array[3],
                                                0, 0, 0, 0, 0, 0])
        M_matrix_single_block_row_5 = np.array([beta_array[0],
                                                0, 0, 0, -beta_array[1],
                                                -beta_array[2],
                                                beta_array[3], 0, 0, 0])
        M_matrix_single_block_row_6 = np.array([beta_array[0], 0, 0, 0,
                                                beta_array[1],
                                                beta_array[2],
                                                beta_array[3],
                                                0, 0, 0])
        M_matrix_single_block_row_7 = np.array([beta_array[0], 0, 0,
                                                0, 0, 0, 0, beta_array[1],
                                                beta_array[2],
                                                beta_array[3]])
        M_matrix_single_block_row_8 = np.array([beta_array[0], 0, 0, 0, 0,
                                                0, 0, -beta_array[1],
                                                -beta_array[2],
                                                beta_array[3]])
        M_matrix_single_block = np.vstack((M_matrix_single_block_row_1,
                                           M_matrix_single_block_row_2,
                                           M_matrix_single_block_row_3,
                                           M_matrix_single_block_row_4,
                                           M_matrix_single_block_row_5,
                                           M_matrix_single_block_row_6,
                                           M_matrix_single_block_row_7,
                                           M_matrix_single_block_row_8))
        M_matrix_single_block = M_matrix_single_block.reshape(8, 10)
        return M_matrix_single_block

    def assemble_M_matrix(self):
        Block1 = self.assemble_M_matrix_single_block(self.betas_up)
        Block2 = self.assemble_M_matrix_single_block(self.betas_p)
        Block3 = self.assemble_M_matrix_single_block(self.betas_pp)
        self.M_matrix = np.vstack((Block1, Block2, Block3)).reshape(24, 10)
        return self.M_matrix

    def invert_M_matrix(self):
        self.inverse_matrix = np.linalg.pinv(self.M_matrix)
        return self.inverse_matrix

    def execute_error_signalling(self, ev):
        II = (ev[0] - ev[3])/(1 - ev[3])
        IZ = (ev[1] - ev[2])/(1 - ev[3])
        ZI = (ev[1] - ev[2])/(1 - ev[3])
        ZZ = (ev[3] - ev[0])/(1 - ev[3])
        XX = (ev[4] + ev[5])/(1 - ev[3])
        YY = (ev[4] + ev[5])/(1 - ev[3])
        ev_error_signalling = np.array([II, IZ, ZI, ZZ, XX, YY])
        return ev_error_signalling

    def execute_expectation_value_calculation(self):
        # assemble matrix that connects RO with terms
        self._calibrate_betas()
        self.assemble_M_matrix()
        self.invert_M_matrix()
        # use it to get terms back from RO
        rescaled_measurements_tomo = self.measurements_tomo
        self.expect_values = np.dot(self.inverse_matrix,
                                    rescaled_measurements_tomo)
        expect_values_VQE = np.array([self.expect_values[0],
                                      self.expect_values[1],
                                      self.expect_values[2],
                                      self.expect_values[3],
                                      self.expect_values[6],
                                      self.expect_values[9]])
        return expect_values_VQE

    def execute_expectation_value_calculation_traceone(self):
        # assemble matrix that connects RO with terms
        self._calibrate_betas()
        self.assemble_M_matrix()
        self.inverse_matrix = np.linalg.pinv(self.M_matrix[:, 1:])
        # use it to get terms back from RO
        rescaled_measurements_tomo = self.measurements_tomo
        self.expect_values = np.dot(self.inverse_matrix,
                                    rescaled_measurements_tomo)
        expect_values_VQE = np.array([1,
                                      self.expect_values[0],
                                      self.expect_values[1],
                                      self.expect_values[2],
                                      self.expect_values[5],
                                      self.expect_values[8]])
        return expect_values_VQE

    def execute_expectation_value_calculation_T1signaling(self):
        # assemble matrix that connects RO with terms
        self._calibrate_betas()
        self.assemble_M_matrix()
        self.inverse_matrix = np.linalg.pinv(self.M_matrix[:, 1:])
        # use it to get terms back from RO
        rescaled_measurements_tomo = self.measurements_tomo
        self.expect_values = np.dot(self.inverse_matrix,
                                    rescaled_measurements_tomo)
        expect_values_VQE = np.array([1,
                                      self.expect_values[0],
                                      self.expect_values[1],
                                      self.expect_values[2],
                                      self.expect_values[5],
                                      self.expect_values[8]])
        expect_values_VQE = self.execute_error_signalling(expect_values_VQE)
        return expect_values_VQE


class ExpectationValueCalculation:

    def __init__(self, auto=True, label='', timestamp=None,
                 fig_format='png',
                 q0_label='q0',
                 q1_label='q1', close_fig=True, **kw):
        self.label = label
        self.timestamp = timestamp
        self.fig_format = fig_format
        # q0 == D2
        self.q0_label = q0_label
        # q1 == A
        self.q1_label = q1_label
        self.n_states = 2 ** 2
        self.ma_obj = ma.MeasurementAnalysis(auto=False, label=label,
                                             timestamp=timestamp)
        self.ma_obj.get_naming_and_values()
        # self.get_naming_and_values()
        # hard coded number of segments for a 2 qubit state tomography
        # constraint imposed by UHFLI
        # self.nr_segments = 16
        # self.exp_name = os.path.split(self.folder)[-1][7:]

        avg_h1 = self.ma_obj.measured_values[0]
        avg_h2 = self.ma_obj.measured_values[1]
        avg_h12 = self.ma_obj.measured_values[2]

        #this should be implemented with a flag
        #but 
        h1_00 = np.mean(avg_h1[36:36+7])
        h1_01 = np.mean(avg_h1[43:43+7])
        h1_10 = np.mean(avg_h1[50:50+7])
        h1_11 = np.mean(avg_h1[57:])

        h2_00 = np.mean(avg_h2[36:36+7])
        h2_01 = np.mean(avg_h2[43:43+7])
        h2_10 = np.mean(avg_h2[50:50+7])
        h2_11 = np.mean(avg_h2[57:])

        h12_00 = np.mean(avg_h12[36:36+7])
        h12_01 = np.mean(avg_h12[43:43+7])
        h12_10 = np.mean(avg_h12[50:50+7])
        h12_11 = np.mean(avg_h12[57:])

        
        mean_h1 = (h1_00+h1_10+h1_01+h1_11)/4
        mean_h2 = (h2_00+h2_01+h2_10+h2_11)/4
        mean_h12 = (h12_00+h12_11+h12_01+h12_10)/4

        #subtract beta 0 from all measurements
        #rescale them
        avg_h1 -= mean_h1
        avg_h2 -= mean_h2
        avg_h12 -= mean_h12

        scale_h1 = (h1_00+h1_10-h1_01-h1_11)/4
        scale_h2 = (h2_00+h2_01-h2_10-h2_11)/4
        scale_h12 = (h12_00+h12_11-h12_01-h12_10)/4

        avg_h1 = (avg_h1)/scale_h1
        avg_h2 = (avg_h2)/scale_h2
        avg_h12 = (avg_h12)/scale_h12
        #The averages have been redefined so redefine the cal terms
        h1_00 = np.mean(avg_h1[36:36+7])
        h1_01 = np.mean(avg_h1[43:43+7])
        h1_10 = np.mean(avg_h1[50:50+7])
        h1_11 = np.mean(avg_h1[57:])

        h2_00 = np.mean(avg_h2[36:36+7])
        h2_01 = np.mean(avg_h2[43:43+7])
        h2_10 = np.mean(avg_h2[50:50+7])
        h2_11 = np.mean(avg_h2[57:])

        h12_00 = np.mean(avg_h12[36:36+7])
        h12_01 = np.mean(avg_h12[43:43+7])
        h12_10 = np.mean(avg_h12[50:50+7])
        h12_11 = np.mean(avg_h12[57:])

        measurement_channel_1 = np.array([avg_h1[0],avg_h1[1],avg_h1[7],avg_h1[8],avg_h1[14],avg_h1[21],avg_h1[28],avg_h1[35]])
        measurement_channel_2 = np.array([avg_h2[0],avg_h2[1],avg_h2[7],avg_h2[8],avg_h2[14],avg_h2[21],avg_h2[28],avg_h2[35]])
        measurement_channel_3 = np.array([avg_h12[0],avg_h12[1],avg_h12[7],avg_h12[8],avg_h12[14],avg_h12[21],avg_h12[28],avg_h12[35]])
        self.measurements_tomo = np.array([measurement_channel_1,measurement_channel_2,measurement_channel_3]).flatten()

        # print(self.measurements_tomo)
        # print(len(self.measurements_tomo))

        # 108 x 1
        # get the calibration points by averaging over the five measurements
        # taken knowing the initial state we put in
        self.measurements_cal=np.array([h1_00, h1_01, h1_10, h1_11, h2_00, h2_01, h2_10, h2_11, h12_00, h12_01, h12_10, h12_11])

    def _calibrate_betas(self):
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
        # print(cal_matrix)
        # print(np.linalg.inv(cal_matrix))
        self.betas = np.zeros(12)
        # print(self.measurements_cal[0:4])
        self.betas[0:4] = np.dot(np.linalg.inv(cal_matrix),
                            self.measurements_cal[0:4])
        self.betas_up = self.betas[0:4]
        self.betas[4:8] = np.dot(np.linalg.inv(cal_matrix),
                            self.measurements_cal[4:8])
        self.betas_p = self.betas[4:8]
        self.betas[8:] = np.dot(np.linalg.inv(cal_matrix),
                           self.measurements_cal[8:12])
        self.betas_pp = self.betas[8:]
        return self.betas

    def assemble_M_matrix_single_block(self, beta_array):
        M_matrix_single_block_row_1 = np.array([beta_array[1],
                                                beta_array[2], beta_array[3],
                                                0, 0, 0, 0, 0, 0])
        M_matrix_single_block_row_2 = np.array([-1*beta_array[1],
                                                beta_array[2],
                                                -1*beta_array[3],
                                                0, 0, 0, 0, 0, 0])
        M_matrix_single_block_row_3 = np.array([beta_array[1],
                                                -1*beta_array[2],
                                                -1*beta_array[3],
                                                0, 0, 0, 0, 0, 0])
        M_matrix_single_block_row_4 = np.array([-1*beta_array[1],
                                                -1*beta_array[2],
                                                beta_array[3],
                                                0, 0, 0, 0, 0, 0])
        M_matrix_single_block_row_5 = np.array([0, 0, 0, -beta_array[1],
                                                -beta_array[2],
                                                beta_array[3], 0, 0, 0])
        M_matrix_single_block_row_6 = np.array([0, 0, 0,
                                                beta_array[1],
                                                beta_array[2],
                                                beta_array[3],
                                                0, 0, 0])
        M_matrix_single_block_row_7 = np.array([0, 0,
                                                0, 0, 0, 0, beta_array[1],
                                                beta_array[2],
                                                beta_array[3]])
        M_matrix_single_block_row_8 = np.array([0, 0, 0, 0,
                                                0, 0, -beta_array[1],
                                                -beta_array[2],
                                                beta_array[3]])
        M_matrix_single_block = np.vstack((M_matrix_single_block_row_1,
                                           M_matrix_single_block_row_2,
                                           M_matrix_single_block_row_3,
                                           M_matrix_single_block_row_4,
                                           M_matrix_single_block_row_5,
                                           M_matrix_single_block_row_6,
                                           M_matrix_single_block_row_7,
                                           M_matrix_single_block_row_8))
        M_matrix_single_block = M_matrix_single_block.reshape(8, 9)
        return M_matrix_single_block

    def assemble_M_matrix(self):
        Block1 = self.assemble_M_matrix_single_block(self.betas_up)
        Block2 = self.assemble_M_matrix_single_block(self.betas_p)
        Block3 = self.assemble_M_matrix_single_block(self.betas_pp)
        self.M_matrix = np.vstack((Block1, Block2, Block3)).reshape(24, 9)
        
        return self.M_matrix

    def invert_M_matrix(self):
        self.inverse_matrix = np.linalg.pinv(self.M_matrix)
        return self.inverse_matrix

    def execute_error_signalling(self, ev):
        II = (ev[0] - ev[3])/(1 - ev[3])
        IZ = (ev[1] - ev[2])/(1 - ev[3])
        ZI = (ev[1] - ev[2])/(1 - ev[3])
        ZZ = (ev[3] - ev[0])/(1 - ev[3])
        XX = (ev[4] + ev[5])/(1 - ev[3])
        YY = (ev[4] + ev[5])/(1 - ev[3])
        ev_error_signalling = np.array([II, IZ, ZI, ZZ, XX, YY])
        return ev_error_signalling

    def execute_expectation_value_calculation_traceone(self):
        # assemble matrix that connects RO with terms
        self._calibrate_betas()
        self.assemble_M_matrix()
        self.inverse_matrix = np.linalg.pinv(self.M_matrix)
        
        # use it to get terms back from RO
        rescaled_measurements_tomo = self.measurements_tomo
        self.expect_values = np.dot(self.inverse_matrix,
                                    rescaled_measurements_tomo)
        expect_values_VQE = np.array([1,
                                      self.expect_values[0],
                                      self.expect_values[1],
                                      self.expect_values[2],
                                      self.expect_values[5],
                                      self.expect_values[8]])
        return expect_values_VQE

    def execute_expectation_value_calculation_T1signaling(self):
        # assemble matrix that connects RO with terms
        self._calibrate_betas()
        self.assemble_M_matrix()
        self.inverse_matrix = np.linalg.pinv(self.M_matrix[:, 1:])
        # use it to get terms back from RO
        rescaled_measurements_tomo = self.measurements_tomo
        self.expect_values = np.dot(self.inverse_matrix,
                                    rescaled_measurements_tomo)
        expect_values_VQE = np.array([1,
                                      self.expect_values[0],
                                      self.expect_values[1],
                                      self.expect_values[2],
                                      self.expect_values[5],
                                      self.expect_values[8]])
        expect_values_VQE = self.execute_error_signalling(expect_values_VQE)
        return expect_values_VQE