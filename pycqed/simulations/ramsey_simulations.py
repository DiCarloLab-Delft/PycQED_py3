"""
October 2018
Simulations of a Ramsey experiment in the presence of flux 1/f noise
"""
import time
import numpy as np
import qutip as qtp
from pycqed.measurement import detector_functions as det
from scipy.interpolate import interp1d
import scipy
import matplotlib.pyplot as plt
import logging

from pycqed.simulations import cz_superoperator_simulation_withdistortions_newdevice_singlequbitphases_newcode_fluxnoise2 as czu
from pycqed.tests import test_ramsey_simulations as tests


def time_evolution(H_vec, c_ops, sim_step):

    '''
    Arguments:
        H:        list of Hamiltonians at different times, each on for a time = sim_step
        c_ops:  list of collapse operators. if an element of the list is a single operator, then it is a time-independent one,
                otherwise, if it's a 2-list, then the first el. is the operator and the second one is a list of time-dependent coefficients.
                Note that in the first case the coefficient is included in the operator
        sim_step:  time for which each H[t] is on.

    '''

    exp_L_total=1
    for i in range(len(H_vec)):
        H=H_vec[i]
        if c_ops != []:
            c_ops_temp=[]
            for c in range(len(c_ops)):
                if isinstance(c_ops[c],list):
                    c_ops_temp.append(c_ops[c][0]*c_ops[c][1][i])    # c_ops are already in the H_0 basis
                else:
                    c_ops_temp.append(c_ops[c])
            liouville_exp_t=(qtp.liouvillian(H,c_ops_temp)*sim_step).expm()
        else:
            liouville_exp_t=(-1j*H*sim_step).expm()
        exp_L_total=liouville_exp_t*exp_L_total

    return exp_L_total


def freq_shift_from_fluxbias(frequency,frequency_target,fluxbias_q0,positive_arc):

    '''
    frequency_tanoise_parameters.CZrget = max frequency of the qubit
    positive_arc (bool) for single and double-sided
    '''

    if frequency > frequency_target:
        logging.warning('Detuning can only be negative. Freq = {}, Freq_max = {}'.format(frequency,frequency_target))
        frequency = frequency_target

    if positive_arc:
        sign = 1
    else:
        sign = -1

    frequency_biased = frequency - np.pi/2 * (frequency_target**2/frequency) * np.sqrt(1 - (frequency**4/frequency_target**4)) * fluxbias_q0 * sign - \
                                              - np.pi**2/2 * frequency_target * (1+(frequency**4/frequency_target**4)) / (frequency/frequency_target)**3 * fluxbias_q0**2
                                              # with sigma up to circa 1e-3 \mu\Phi_0 the second order is irrelevant

    return frequency_biased


def calc_populations(U):

    hadamard_singleq = qtp.Qobj([[1,1,0],
                                    [1,-1,0],
                                    [0,0,0]])/np.sqrt(2)
    hadamard_q0 = qtp.tensor(qtp.qeye(3),hadamard_singleq)

    U_pi2_pulsed = hadamard_q0 * U * hadamard_q0

    return {'population_in_0': np.abs(U_pi2_pulsed[0,0])**2, 'population_in_1': np.abs(U_pi2_pulsed[0,1])**2}






class ramsey_experiment(det.Soft_Detector):
    def __init__(self, fluxlutman, noise_parameters_CZ, control_parameters_ramsey):
        """
        Detector for simulating a Ramsey experiment.
        Args:
            fluxlutman (instr): an instrument that contains the parameters
                required to generate the waveform for the trajectory, and the hamiltonian as well.
            noise_parameters_CZ: instrument that contains the noise parameters, plus some more
        """
        super().__init__()
        self.value_names = ['population_in_0','population_in_1']
        self.value_units = ['%', '%']
        self.fluxlutman = fluxlutman
        self.noise_parameters_CZ = noise_parameters_CZ
        self.control_parameters_ramsey = control_parameters_ramsey

        # instr_list=[self.fluxlutman,self.noise_parameters_CZ,self.control_parameters_ramsey]
        # for instr in instr_list:
        #     instr.print_readable_snapshot()



    def acquire_data_point(self, **kw):

        ramsey = self.control_parameters_ramsey.ramsey()
        sigma = self.control_parameters_ramsey.sigma()
        detuning = self.control_parameters_ramsey.detuning_ramsey()

        t = self.control_parameters_ramsey.pulse_length()


        qoi_plot = list()    # used to verify convergence properties. If len(n_sampling_gaussian_vec)==1, it is useless
        n_sampling_gaussian_vec = [11]  # 11 guarantees excellent convergence. We choose it odd so that the central point of the Gaussian is included.
                                        # ALWAYS choose it odd
        for n_sampling_gaussian in n_sampling_gaussian_vec:
            # If sigma=0 there's no need for sampling
            weights=[]
            if sigma != 0:
                samplingpoints_gaussian = np.linspace(-5*sigma,5*sigma,n_sampling_gaussian)    # after 5 sigmas we cut the integral
                delta_x = samplingpoints_gaussian[1]-samplingpoints_gaussian[0]
                values_gaussian = czu.gaussian(samplingpoints_gaussian,mean=0,sigma=sigma)
            else:
                samplingpoints_gaussian = np.array([0])
                delta_x = 1
                values_gaussian = np.array([1])


            qoi_vec = list()

            for j_q0 in range(len(samplingpoints_gaussian)):
                fluxbias_q0 = samplingpoints_gaussian[j_q0]
                if sigma != 0:
                    weights.append(values_gaussian[j_q0]*delta_x)
                else:
                    weights=[1]


                f_q0_sweetspot = self.fluxlutman.q_freq_01()
                f_q0_detuned = f_q0_sweetspot + detuning


                H=[]
                if ramsey:

                    f_q0_biased = freq_shift_from_fluxbias(f_q0_detuned,f_q0_sweetspot,fluxbias_q0,positive_arc=True)
                    freq_rotating_frame = f_q0_biased-f_q0_sweetspot
                    H.append(czu.coupled_transmons_hamiltonian_new(w_q0=freq_rotating_frame, w_q1=0, alpha_q0=-2*freq_rotating_frame, alpha_q1=0, J=0))
                    sim_step = t

                else:

                    for positive in [True, False]:
                        f_q0_biased = freq_shift_from_fluxbias(f_q0_detuned,f_q0_sweetspot,fluxbias_q0,positive_arc=positive)
                        freq_rotating_frame = f_q0_biased-f_q0_sweetspot
                        H.append(czu.coupled_transmons_hamiltonian_new(w_q0=freq_rotating_frame, w_q1=0, alpha_q0=-2*freq_rotating_frame, alpha_q1=0, J=0))
                    sim_step = t/2

                c_ops=[]

                U_final = time_evolution(H, c_ops, sim_step)

                qoi = calc_populations(U_final)
                quantities_of_interest = [qoi['population_in_0']*100, qoi['population_in_1']*100]
                qoi_vec.append(np.array(quantities_of_interest))

            weights = np.array(weights)

            qoi_average = np.zeros(len(quantities_of_interest))
            for index in [0,1]:    # 4 is not trustable here, and we don't care anyway
                qoi_average[index] = np.average(np.array(qoi_vec)[:,index], weights=weights)
            qoi_plot.append(qoi_average)

        qoi_plot = np.array(qoi_plot)

        ### Plot to study the convergence properties of averaging over a Gaussian
        # for i in range(len(qoi_plot[0])):
        #     czu.plot(x_plot_vec=[n_sampling_gaussian_vec],
        #                   y_plot_vec=[qoi_plot[:,i]],
        #                   title='Study of convergence of average',
        #                   xlabel='n_sampling_gaussian points',ylabel=self.value_names[i])


        return qoi_plot[0,0], qoi_plot[0,1]



























