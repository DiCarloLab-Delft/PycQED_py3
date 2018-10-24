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
        self.nr_segments = 16
        # self.exp_name = os.path.split(self.folder)[-1][7:]

        avg_h1 = self.ma_obj.measured_values[0]
        avg_h2 = self.ma_obj.measured_values[1]
        avg_h12 = self.ma_obj.measured_values[2]

        # Binning all the points required for the tomo
        h1_00 = np.mean(avg_h1[8:10])
        h1_01 = np.mean(avg_h1[10:12])
        h1_10 = np.mean(avg_h1[12:14])
        h1_11 = np.mean(avg_h1[14:])

        h2_00 = np.mean(avg_h2[8:10])
        h2_01 = np.mean(avg_h2[10:12])
        h2_10 = np.mean(avg_h2[12:14])
        h2_11 = np.mean(avg_h2[14:])

        h12_00 = np.mean(avg_h12[8:10])
        h12_01 = np.mean(avg_h12[10:12])
        h12_10 = np.mean(avg_h12[12:14])
        h12_11 = np.mean(avg_h12[14:])

        self.measurements_tomo = (
            np.array([avg_h1[0:8], avg_h2[0:8],
                      avg_h12[0:8]])).flatten()
        # print(self.measurements_tomo)
        # print(len(self.measurements_tomo))

        # 108 x 1
        # get the calibration points by averaging over the five measurements
        # taken knowing the initial state we put in
        self.measurements_cal = np.array(
            [h1_00, h1_01, h1_10, h1_11,
             h2_00, h2_01, h2_10, h2_11,
             h12_00, h12_01, h12_10, h12_11])
        # print(len(self.measurements_cal))


        # print(self.measurements_cal)

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
        betas[0:4] = np.dot(np.linalg.inv(cal_matrix), self.measurements_cal[0:4])
        # print(cal_matrix)
        # print(np.linalg.inv(cal_matrix))
        # print(self.measurements_cal[0:4])
        # print(betas[0:4])
        betas[4:8] = np.dot(np.linalg.inv(cal_matrix), self.measurements_cal[4:8])
        # print(betas[4:8])
        betas[8:] = np.dot(np.linalg.inv(cal_matrix), self.measurements_cal[8:12])
        # print(betas[8:])

        return betas

    def expectation_value_calculation_IdenZ(self):

        betas = self._calibrate_betas()
        #inverting the unprimed beta matrix
        #up is unprimed
        self.betas = betas
        # print(self.betas[0:4], self.betas[4:8], self.betas[8:])
        beta_0_up =self.betas[0]

        beta_1_up =self.betas[1]
        beta_2_up =self.betas[2]
        beta_3_up =self.betas[3]


        beta_matrix_up = np.array([[beta_0_up,beta_1_up,beta_2_up,beta_3_up],
                                        [beta_0_up,-1*beta_1_up,beta_2_up,-1*beta_3_up],
                                        [beta_0_up,beta_1_up,-1*beta_2_up,-1*beta_3_up],
                                        [beta_0_up,-1*beta_1_up,-1*beta_2_up,beta_3_up]])

        #assuming 0:4 are
        # expect_value_IdenZ_up = np.dot(np.linalg.inv(beta_matrix_up), self.measurements_tomo[1:4])

        expect_value_IdenZ_up = np.dot(np.linalg.inv(beta_matrix_up), self.measurements_tomo[0:4])

        #inverting the primed beta matrix
        #p is primed
        beta_0_p =self.betas[4]
        beta_1_p =self.betas[5]
        beta_2_p =self.betas[6]
        beta_3_p =self.betas[7]

        beta_matrix_p = np.array([[beta_0_p,beta_1_p,beta_2_p,beta_3_p],
                                        [beta_0_p,-1*beta_1_p,beta_2_p,-1*beta_3_p],
                                        [beta_0_p,beta_1_p,-1*beta_2_p,-1*beta_3_p],
                                        [beta_0_p,-1*beta_1_p,-1*beta_2_p,beta_3_p]])
        # beta_matrix_p = np.array([[-1*beta_1_p,beta_2_p,-1*beta_3_p],
        #                           [beta_1_p,-1*beta_2_p,-1*beta_3_p],
        #                           [-1*beta_1_p,-1*beta_2_p,beta_3_p]])
        #assuming 0:4 are
        expect_value_IdenZ_p = np.dot(np.linalg.inv(beta_matrix_p), self.measurements_tomo[8:12])
        # expect_value_IdenZ_p = np.dot(np.linalg.inv(beta_matrix_p), self.measurements_tomo[1:4])

        #inverting the unprimed beta matrix
        #up is unprimed
        beta_0_pp =self.betas[8]
        beta_1_pp =self.betas[9]
        beta_2_pp =self.betas[10]
        beta_3_pp =self.betas[11]

        beta_matrix_pp = np.array([[beta_0_pp,beta_1_pp,beta_2_pp,beta_3_pp],
                                        [beta_0_pp,-1*beta_1_pp,beta_2_pp,-1*beta_3_pp],
                                        [beta_0_pp,beta_1_pp,-1*beta_2_pp,-1*beta_3_pp],
                                        [beta_0_pp,-1*beta_1_pp,-1*beta_2_pp,beta_3_pp]])
        # beta_matrix_pp = np.array([[-1*beta_1_pp,beta_2_pp,-1*beta_3_pp],
        #                            [beta_1_pp,-1*beta_2_pp,-1*beta_3_pp],
        #                            [-1*beta_1_pp,-1*beta_2_pp,beta_3_pp]])
        #assuming 0:4 are
        expect_value_IdenZ_pp = np.dot(np.linalg.inv(beta_matrix_pp), self.measurements_tomo[16:20])
        # expect_value_IdenZ_pp = np.dot(np.linalg.inv(beta_matrix_p), self.measurements_tomo[1:4])

        #take the mean of calculated expectation values of II, IZ, ZI, ZZ
        #for three different beta vectors

        expect_value_IdenZ = np.mean( np.array([expect_value_IdenZ_up,
                                                expect_value_IdenZ_p,
                                                expect_value_IdenZ_pp]),
                                                axis=0 )

        print(expect_value_IdenZ_up)

        print(expect_value_IdenZ_p)

        print(expect_value_IdenZ_pp)
        return expect_value_IdenZ

    def expectation_value_calculation_XX(self):


        expect_value_XX_up = ((self.measurements_tomo[4] + self.measurements_tomo[5]) -2*self.betas[0])/2*self.betas[3]
        expect_value_XX_p = ((self.measurements_tomo[12] + self.measurements_tomo[13])-2*self.betas[4])/2*self.betas[7]
        expect_value_XX_pp = ((self.measurements_tomo[20] + self.measurements_tomo[21]) - 2*self.betas[8])/2*self.betas[11]
        expectation_value_XX = (expect_value_XX_up + expect_value_XX_p + expect_value_XX_pp)/3
        # print(expect_value_XX_up, expect_value_XX_p, expect_value_XX_pp)
        return expectation_value_XX

    def expectation_value_calculation_YY(self):


        expect_value_YY_up = ((self.measurements_tomo[6] + self.measurements_tomo[7]) -2*self.betas[0])/2*self.betas[3]
        expect_value_YY_p = ((self.measurements_tomo[14] + self.measurements_tomo[15])-2*self.betas[4])/2*self.betas[7]
        expect_value_YY_pp = ((self.measurements_tomo[22] + self.measurements_tomo[23]) - 2*self.betas[8])/2*self.betas[11]
        # print(expect_value_YY_up, expect_value_YY_p, expect_value_YY_pp)
        expectation_value_YY = (expect_value_YY_up + expect_value_YY_p + expect_value_YY_pp)/3

        return expectation_value_YY

    def execute_expectation_value_calculation(self):

        expect_values = np.zeros(6)
        expect_values[0:4] = self.expectation_value_calculation_IdenZ()
        # print(self.expectation_value_calculation_IdenZ())
        expect_values[4] = self.expectation_value_calculation_XX()
        # print(self.expectation_value_calculation_XX())
        expect_values[5] = self.expectation_value_calculation_YY()
        # print(self.expectation_value_calculation_YY())
        return expect_values, self.betas


class ExpectationValueCalculation2:

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
        h1_00 = np.mean(avg_h1[8:10])
        h1_01 = np.mean(avg_h1[10:12])
        h1_10 = np.mean(avg_h1[12:14])
        h1_11 = np.mean(avg_h1[14:])

        h2_00 = np.mean(avg_h2[8:10])
        h2_01 = np.mean(avg_h2[10:12])
        h2_10 = np.mean(avg_h2[12:14])
        h2_11 = np.mean(avg_h2[14:])

        h12_00 = np.mean(avg_h12[8:10])
        h12_01 = np.mean(avg_h12[10:12])
        h12_10 = np.mean(avg_h12[12:14])
        h12_11 = np.mean(avg_h12[14:])
        self.measurements_tomo = (
            np.array([avg_h1[0:8], avg_h2[0:8],
                      avg_h12[0:8]])).flatten()
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


class ExpectationValueCalculation3_shots:

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
        shots_I_q0 = get_segments_average(self.ma_obj.measured_values[0],
                                          segments_per_block=16,
                                          block_size=4094,
                                          average=False)
        shots_I_q1 = get_segments_average(self.ma_obj.measured_values[1],
                                          segments_per_block=16,
                                          block_size=4094,
                                          average=False)

        shots_I_q0q1 = np.multiply(shots_I_q0/(np.max(shots_I_q0)-np.min(shots_I_q0)),shots_I_q1/(np.max(shots_I_q1)-np.min(shots_I_q1)))

        avg_h1 = np.mean(shots_I_q0,axis=0)
        avg_h2 = np.mean(shots_I_q1,axis=0)
        avg_h12 = np.mean(shots_I_q0q1,axis=0)

        h1_00 = np.mean(avg_h1[8:10])
        h1_01 = np.mean(avg_h1[10:12])
        h1_10 = np.mean(avg_h1[12:14])
        h1_11 = np.mean(avg_h1[14:])

        h2_00 = np.mean(avg_h2[8:10])
        h2_01 = np.mean(avg_h2[10:12])
        h2_10 = np.mean(avg_h2[12:14])
        h2_11 = np.mean(avg_h2[14:])

        h12_00 = np.mean(avg_h12[8:10])
        h12_01 = np.mean(avg_h12[10:12])
        h12_10 = np.mean(avg_h12[12:14])
        h12_11 = np.mean(avg_h12[14:])
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
        h1_00 = np.mean(avg_h1[8:10])
        h1_01 = np.mean(avg_h1[10:12])
        h1_10 = np.mean(avg_h1[12:14])
        h1_11 = np.mean(avg_h1[14:])

        h2_00 = np.mean(avg_h2[8:10])
        h2_01 = np.mean(avg_h2[10:12])
        h2_10 = np.mean(avg_h2[12:14])
        h2_11 = np.mean(avg_h2[14:])

        h12_00 = np.mean(avg_h12[8:10])
        h12_01 = np.mean(avg_h12[10:12])
        h12_10 = np.mean(avg_h12[12:14])
        h12_11 = np.mean(avg_h12[14:])

        self.measurements_tomo = (
            np.array([avg_h1[0:8], avg_h2[0:8],
                      avg_h12[0:8]])).flatten()
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
        # II IZ ZI ZZ IX XI XX IY YI YY
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
        M_matrix_single_block_row_5 = np.array([beta_array[0],            # 36
                                                0, 0, 0, -1*beta_array[1],
                                                -1*beta_array[2],
                                                beta_array[3], 0, 0, 0])
        M_matrix_single_block_row_6 = np.array([beta_array[0], 0, 0, 0,   # 29
                                                beta_array[1],
                                                beta_array[2],
                                                beta_array[3],
                                                0, 0, 0])
        M_matrix_single_block_row_7 = np.array([beta_array[0], 0, 0,
                                                0, 0, 0, 0, beta_array[1],
                                                beta_array[2],
                                                beta_array[3]])
        M_matrix_single_block_row_8 = np.array([beta_array[0], 0, 0, 0, 0,
                                                0, 0, -1*beta_array[1],
                                                -1*beta_array[2],
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
        ZI = (ev[2] - ev[1])/(1 - ev[3])
        ZZ = (ev[3] - ev[0])/(1 - ev[3])
        XX = (ev[4] + ev[5])/(1 - ev[3])
        YY = (ev[5] + ev[4])/(1 - ev[3])
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
        print(self.expect_values)
        expect_values_VQE = np.array([1,
                                      self.expect_values[1],
                                      self.expect_values[2],
                                      self.expect_values[3],
                                      self.expect_values[6],
                                      self.expect_values[9]])
        self.expect_values = expect_values_VQE
        return expect_values_VQE

    def execute_expectation_value_calculation_traceone(self):
        # assemble matrix that connects RO with terms
        self._calibrate_betas()
        self.assemble_M_matrix()
        self.inverse_matrix = np.linalg.pinv(self.M_matrix[:, 1:])
        # use it to get terms back from RO
        beta_0_vec = np.repeat([self.betas_up[0],
                                self.betas_p[0],
                                self.betas_pp[0]], 8)
        rescaled_measurements_tomo = self.measurements_tomo - beta_0_vec
        self.expect_values = np.dot(self.inverse_matrix,
                                    rescaled_measurements_tomo)
        expect_values_VQE = np.array([1,
                                      self.expect_values[0],
                                      self.expect_values[1],
                                      self.expect_values[2],
                                      self.expect_values[5],
                                      self.expect_values[8]])
        self.expect_values = expect_values_VQE
        print(self.expect_values)
        return expect_values_VQE

    def execute_expectation_value_calculation_T1signaling(self):
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
        expect_values_VQE = self.execute_error_signalling(expect_values_VQE)
        self.expect_values = expect_values_VQE
        return expect_values_VQE




class ExpectationValueCalculation2_shots:

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
        shots_I_q0 = get_segments_average(self.ma_obj.measured_values[0],
                                          segments_per_block=16,
                                          block_size=4094,
                                          average=False)
        shots_I_q1 = get_segments_average(self.ma_obj.measured_values[1],
                                          segments_per_block=16,
                                          block_size=4094,
                                          average=False)

        shots_I_q0q1 = np.multiply(shots_I_q0/(np.max(shots_I_q0)-np.min(shots_I_q0)),shots_I_q1/(np.max(shots_I_q1)-np.min(shots_I_q1)))

        avg_h1 = np.mean(shots_I_q0,axis=0)
        avg_h2 = np.mean(shots_I_q1,axis=0)
        avg_h12 = np.mean(shots_I_q0q1,axis=0)
        h1_00 = np.mean(avg_h1[8:10])
        h1_01 = np.mean(avg_h1[10:12])
        h1_10 = np.mean(avg_h1[12:14])
        h1_11 = np.mean(avg_h1[14:])

        h2_00 = np.mean(avg_h2[8:10])
        h2_01 = np.mean(avg_h2[10:12])
        h2_10 = np.mean(avg_h2[12:14])
        h2_11 = np.mean(avg_h2[14:])

        h12_00 = np.mean(avg_h12[8:10])
        h12_01 = np.mean(avg_h12[10:12])
        h12_10 = np.mean(avg_h12[12:14])
        h12_11 = np.mean(avg_h12[14:])
        self.measurements_tomo = (
            np.array([avg_h1[0:8], avg_h2[0:8],
                      avg_h12[0:8]])).flatten()
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
