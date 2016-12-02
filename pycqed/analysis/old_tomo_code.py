

class Tomo_Analysis(MeasurementAnalysis):

    def __init__(self, num_qubits=2, quad='IQ', over_complete_set=False,
                 plot_oper=True, folder=None, auto=False, **kw):
        self.num_qubits = num_qubits
        self.num_states = 2**num_qubits
        self.over_complete_set = over_complete_set
        if over_complete_set:
            self.num_measurements = 6**num_qubits
        else:
            self.num_measurements = 4**num_qubits
        self.quad = quad
        self.plot_oper = plot_oper
        super(self.__class__, self).__init__(**kw)

    def run_default_analysis(self, **kw):
            self.get_naming_and_values()
            data_I = self.get_values(key='I')
            data_Q = self.get_values(key='Q')
            measurements_tomo = (np.array([data_I[0:36], data_Q[0:36]])).flatten()
            measurements_cal = np.array([np.average(data_I[36:39]),
                                        np.average(data_I[39:42]),
                                        np.average(data_I[42:45]),
                                        np.average(data_I[45:48]),
                                        np.average(data_Q[36:39]),
                                        np.average(data_Q[39:42]),
                                        np.average(data_Q[42:45]),
                                        np.average(data_Q[45:48])])

            if self.quad == 'IQ':
                self.use_both_quad = True
            else:
                self.use_both_quad = False
                if self.quad == 'Q':
                    measurements_tomo[0:self.num_measurements]=measurements_tomo[self.num_measurements:]
                    measurements_cal[0:self.num_states]=measurements_cal[self.num_states:]
                elif self.quad != 'I':
                    raise Error('Quadrature to use is not clear.')

            beta_I = self.calibrate_beta(measurements_cal=measurements_cal[0:self.num_states])
            beta_Q = np.zeros(self.num_states)
            if self.use_both_quad==True:
                beta_Q = self.calibrate_beta(measurements_cal=measurements_cal[self.num_states:])

            if self.use_both_quad==True:
                max_idx = 2*self.num_measurements
            else:
                max_idx = self.num_measurements

            results = self.calc_operators(measurements_tomo[:max_idx], beta_I, beta_Q)
            self.results = results
            self.dens_mat = self.calc_density_matrix(results)
            if self.plot_oper == True:
                self.plot_operators(**kw)

    def calibrate_beta(self, measurements_cal):
        #calibrates betas for the measurement operator
        cal_matrix = np.zeros((self.num_states,self.num_states))
        for i in range(self.num_states):
            for j in range(self.num_states):
                cal_matrix[i,j] = (-1)**(self.get_bit_sum(i & j))
        beta = np.dot(np.linalg.inv(cal_matrix),measurements_cal)
        print beta
        return beta

    def calc_operators(self,measurements_tomo,beta_I,beta_Q):
        M = self.num_measurements
        K = 4**self.num_qubits - 1
        if self.use_both_quad == False:
            measurement_matrix = np.zeros((M, K))
            measurements_tomo[:M] = measurements_tomo[:M] - beta_I[0]
            measurements_tomo[M:] = measurements_tomo[M:] - beta_Q[0]
        else:
            measurement_matrix = np.zeros((2*M,K))

        for i in range(M):
            for j in range(1,self.num_states):
                place, sign = self.rotate_M0_elem(i,j)
                measurement_matrix[i,place] = sign*beta_I[j]
        if self.use_both_quad == True:
            for i in range(M):
                for j in range(1,self.num_states):
                    place, sign = self.rotate_M0_elem(i,j)
                    measurement_matrix[i+M,place] = sign*beta_Q[j]

        inverse = np.linalg.pinv(measurement_matrix)
        pauli_operators = np.dot(inverse,measurements_tomo)

        p_op = np.zeros(4**self.num_qubits)
        p_op[0] = 1
        p_op[1:] = pauli_operators
        return np.real(p_op)

    def rotate_M0_elem(self,rotation,elem):
        # moves first the first one
        rot_vector = self.get_rotation_vector(rotation)
        # moves first the last one
        elem_op_vector = self.get_m0_elem_vector(elem)

        res_vector = np.zeros(len(rot_vector))
        sign = 1
        for i in range(len(rot_vector)):
            value = elem_op_vector[i]
            res_vector[i] = 0
            if value == 1:
                if rot_vector[i] == 0:
                    res_vector[i] = value
                    sign = sign
                if rot_vector[i] == 1:
                    res_vector[i] = value
                    sign = -1*sign
                if rot_vector[i] == 2:
                    res_vector[i] = 3
                    sign = sign
                if rot_vector[i] == 3:
                    res_vector[i] = 2
                    sign = -1*sign
                if rot_vector[i] == 4:
                    res_vector[i] = 3
                    sign = -1*sign
                if rot_vector[i] == 5:
                    res_vector[i] = 2
                    sign = sign
            else:
                res_vector[i] = value
                sign = sign

        res_number = self.get_pauli_op_number(res_vector) - 1
        # the minus 1 is to not consider the <II> in the pauli vector
        return np.array([res_number,sign])

    def calc_density_matrix(self, pauli_operators):
        Id2 = np.identity(2)
        Z_op = [[1+0.j,0+0.j],[0+0.j,-1+0.j]]
        X_op = [[0+0.j,1+0.j],[1+0.j,0+0.j]]
        Y_op = [[0+0.j,-1.j],[1.j,0+0.j]]
        rho = np.zeros((self.num_states,self.num_states))
        #np.kron works in the same way as bits (the most signifcant at left)
        for i in range(0,2**self.num_states):
            vector = self.get_pauli_op_vector(i)
            acum = 1
            for j in range(self.num_qubits-1,-1,-1):
                if vector[j] == 0:
                    temp = np.kron(Id2,acum)
                if vector[j] == 1:
                    temp = np.kron(Z_op,acum)
                if vector[j] == 2:
                    temp = np.kron(X_op,acum)
                if vector[j] == 3:
                    temp = np.kron(Y_op,acum)
                del acum
                acum = temp
                del temp
            rho = rho + acum*pauli_operators[i]
        return rho/self.num_states

    def get_pauli_op_number(self,pauli_vector):
        pauli_number = 0
        N = len(pauli_vector)
        for i in range(0,N,1):
            pauli_number += pauli_vector[N-i-1] * (4**i)
        return pauli_number

    def get_pauli_op_vector(self,pauli_number):
        N = self.num_qubits
        pauli_vector = np.zeros(N)
        rest = pauli_number
        for i in range(0,N,1):
            value = rest % 4
            pauli_vector[i] = value
            rest = (rest-value)/4
        return pauli_vector

    def get_m0_elem_vector(self,elem_number):
        elem_vector = np.zeros(self.num_qubits)
        rest = elem_number
        for i in range(self.num_qubits-1,-1,-1):
            value = rest % 2
            elem_vector[i] = value
            rest = (rest-value)/2
        return elem_vector

    def get_rotation_vector(self,rot_number):
        N = self.num_qubits
        rot_vector = np.zeros(N)
        rest = rot_number
        if self.over_complete_set:
            total = 6
        else:
            total = 4
        for i in range(N-1,-1,-1):
            value = rest % total
            rot_vector[i] = value
            rest = (rest-value)/total
        return rot_vector

    def get_bit_sum(self,number):
        N = self.num_qubits
        summ = 0
        rest = number
        for i in range(N-1,-1,-1):
            value = rest % 2
            summ += value
            rest = (rest-value)/2
        return summ

    def get_operators_label(self):
        labels=[]
        for i in range(2**self.num_states):
            vector = self.get_pauli_op_vector(i)
            label=''
            for j in range(self.num_qubits):
                    if vector[j] == 0:
                        label='I'+label
                    if vector[j] == 1:
                        label='Z'+label
                    if vector[j] == 2:
                        label='X'+label
                    if vector[j] == 3:
                        label='Y'+label
            labels.append(label)

        labels = ['IX','IY','IZ','XI','YI','ZI','XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']
        return labels

    def plot_operators(self, **kw):
        import qutip as qtip
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(121)
        pauli_1,pauli_2,pauli_cor = self.order_pauli_output2(self.results)
        width=0.35
        ind1 = np.arange(3)
        ind2 = np.arange(3,6)
        ind3 = np.arange(6,15)
        ind = np.arange(15)
        q1 = ax.bar(ind1, pauli_1, width, color='r')
        q1 = ax.bar(ind2, pauli_2, width, color='b')
        q2 = ax.bar(ind3, pauli_cor, width, color='purple')

        ax.set_title('%d Qubit State Tomography' % self.num_qubits)
        # ax.set_ylim(-1,1)
        ax.set_xticks(np.arange(0,2**self.num_states))
        ax.set_xticklabels(self.get_operators_label())
        ax2 = fig.add_subplot(122,projection='3d')
        qtip.matrix_histogram_complex(qtip.Qobj(self.dens_mat),
            xlabels=['00','01','10','11'],ylabels=['00','01','10','11'],
            fig=fig,ax=ax2)
        print 'working so far'
        self.save_fig(fig, figname=self.measurementstring, **kw)
        # print 'Concurrence = %f' % qt.concurrence(qt.Qobj(self.dens_mat,dims=[[2, 2], [2, 2]]))
        return

    def order_pauli_output2(self, pauli_op_dis):
        pauli_1 = np.array([pauli_op_dis[2],pauli_op_dis[3],pauli_op_dis[1]])
        pauli_2 = np.array([pauli_op_dis[8],pauli_op_dis[12],pauli_op_dis[4]])
        pauli_corr = np.array([pauli_op_dis[10],pauli_op_dis[11],pauli_op_dis[9],
                               pauli_op_dis[14],pauli_op_dis[15],pauli_op_dis[13],
                               pauli_op_dis[6],pauli_op_dis[7],pauli_op_dis[5]])
        return pauli_1,pauli_2,pauli_corr

