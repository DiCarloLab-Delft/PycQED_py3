
"""
October 2018
Francesco Battistel

Rearranging code for CZ simulations. Files without "_new" in pycqed/simulations are old and not supported anymore.
Needs to be copied into notebook to work

"""

from pycqed.simulations import cz_superoperator_simulation_new_functions as czf
import numpy as np
from pycqed.measurement import detector_functions as det
import matplotlib.pyplot as plt
from pycqed.measurement.waveform_control_CC import waveforms_flux as wfl
from scipy.interpolate import interp1d
import qutip as qtp
#np.set_printoptions(threshold=np.inf)



def compute_propagator_parallelizable(arglist):
    # arglist = [samplepoint_q0,samplepoint_q1,fluxlutman_args,noise_parameters_CZ_args,fitted_stepresponse_ty,instrument_number,cluster]

    fluxbias_q0 = arglist['fluxbias_q0']
    fluxbias_q1 = arglist['fluxbias_q1']
    fitted_stepresponse_ty = arglist['fitted_stepresponse_ty']
        
    if arglist['cluster']:
    
        fluxlutman_args = arglist['fluxlutman_args']       # [sampling_rate, cz_length, q_J2, czd_double_sided, cz_lambda_2, cz_lambda_3,
                                                           #        cz_theta_f, czd_length_ratio]
        noise_parameters_CZ_args = arglist['noise_parameters_CZ_args']       # [Z_rotations_length, voltage_scaling_factor, distortions, T1_q0, T1_q1, T2_q0_sweetspot, T2_q0_interaction_point,
                                                                             #      T2_q0_amplitude_dependent, T2_q1]
        number = arglist['number']


        fluxlutman = flm.AWG8_Flux_LutMan('fluxlutman_'+'{}'.format(number))
        noise_parameters_CZ = npCZ.NoiseParametersCZ('noise_parameters_CZ_'+'{}'.format(number))

        fluxlutman, noise_parameters_CZ = czf.return_instrument_from_arglist(fluxlutman,fluxlutman_args,noise_parameters_CZ,noise_parameters_CZ_args)
    else:
        fluxlutman = arglist['fluxlutman']
        noise_parameters_CZ = arglist['noise_parameters_CZ']



    sim_step=1/fluxlutman.sampling_rate()
    subdivisions_of_simstep=4                          # 4 is a good one, corresponding to a time step of 0.1 ns
    sim_step_new=sim_step/subdivisions_of_simstep      # waveform is generated according to sampling rate of AWG,
                                                       # but we can use a different step for simulating the time evolution
    tlist = np.arange(0, fluxlutman.cz_length(),
                       sim_step)
    
    eps_i = fluxlutman.calc_amp_to_eps(0, state_A='11', state_B='02')
    theta_i = wfl.eps_to_theta(eps_i, g=fluxlutman.q_J2())           # Beware theta in radian!


    if not fluxlutman.czd_double_sided():
        thetawave = wfl.martinis_flux_pulse(
            length=fluxlutman.cz_length(),
            lambda_2=fluxlutman.cz_lambda_2(),
            lambda_3=fluxlutman.cz_lambda_3(),
            theta_i=theta_i,
            theta_f=np.deg2rad(fluxlutman.cz_theta_f()),
            sampling_rate=fluxlutman.sampling_rate())    # return in terms of theta
        epsilon = wfl.theta_to_eps(thetawave, fluxlutman.q_J2())
        amp = fluxlutman.calc_eps_to_amp(epsilon, state_A='11', state_B='02')
                 # transform detuning frequency to (positive) amplitude
    else:
        amp = get_f_pulse_double_sided(fluxlutman,theta_i)


    # For better accuracy in simulations, redefine amp in terms of sim_step_new.
    # We split here below in two cases to keep into account that certain times net-zero is one AWG time-step longer
    # than the conventional pulse with the same pulse length.
    if len(tlist) == len(amp):
        tlist_temp=np.concatenate((tlist,np.array([fluxlutman.cz_length()])))
        tlist_new = np.arange(0, fluxlutman.cz_length(),
                       sim_step_new)
    else:
        tlist_temp=np.concatenate((tlist,np.array([fluxlutman.cz_length(),fluxlutman.cz_length()+sim_step])))
        tlist_new = np.arange(0, fluxlutman.cz_length()+sim_step,
                       sim_step_new)
    amp_temp=np.concatenate((amp,np.array([amp[0]])))    # amp should come back to the initial value, i.e. at the sweet spot
    amp_interp=interp1d(tlist_temp,amp_temp)
    amp=amp_interp(tlist_new)

    # We add the single qubit rotations at the end of the pulse
    if noise_parameters_CZ.Z_rotations_length() != 0:
        tlist_singlequbitrotations = np.arange(0,noise_parameters_CZ.Z_rotations_length(),sim_step_new)
        amp = np.concatenate([amp,np.zeros(len(tlist_singlequbitrotations))+amp[0]])
        tlist_new = czf.concatenate_CZpulse_and_Zrotations(noise_parameters_CZ.Z_rotations_length(),sim_step_new,tlist_new)

    t_final = tlist_new[-1]+sim_step_new

    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp],
    #                          title='Pulse with (possibly) single qubit rotations',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')


    amp = amp * noise_parameters_CZ.voltage_scaling_factor()       # recommended to change discretely the scaling factor


    ### Apply distortions
    if noise_parameters_CZ.distortions():
        amp_final = czf.distort_amplitude(fitted_stepresponse_ty=fitted_stepresponse_ty,amp=amp,tlist_new=tlist_new,sim_step_new=sim_step_new)
    else:
        amp_final = amp

    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp_final],
    #                          title='Pulse with distortions, absolute',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')
    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp_final-amp],
    #                          title='Pulse with distortions, difference',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')


    ### the fluxbias_q0 affects the pulse shape after the distortions have been taken into account
    amp_final, f_pulse_final = czf.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,amp_final=amp_final,fluxbias_q0=fluxbias_q0)

    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp_final-amp_final_new],
    #                          title='Pulse with distortions and shift due to fluxbias_q0, difference',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')
    # amp_final = amp_final_new
    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[f_pulse_final/1e9],
    #                          title='Pulse with distortions and shift due to fluxbias_q0',
    #                            xlabel='Time (ns)',ylabel='Frequency (GHz)')


    ### Obtain jump operators, possibly time-dependent (incoherent part of the noise)
    c_ops = czf.return_jump_operators(noise_parameters_CZ=noise_parameters_CZ, f_pulse_final=f_pulse_final, fluxlutman=fluxlutman)


    ### Compute propagator
    U_final = czf.time_evolution_new(c_ops=c_ops, noise_parameters_CZ=noise_parameters_CZ, 
                                 fluxlutman=fluxlutman, fluxbias_q1=fluxbias_q1, amp=amp_final, sim_step=sim_step_new)
    #print(czf.verify_CPTP(U_superop_average))


    if arglist['cluster']:
        fluxlutman.close()
        noise_parameters_CZ.close()

    return [U_final, t_final]




def get_f_pulse_double_sided(fluxlutman,theta_i):

    thetawave_A = wfl.martinis_flux_pulse(
        length=fluxlutman.cz_length()*fluxlutman.czd_length_ratio(),
        lambda_2=fluxlutman.cz_lambda_2(),
        lambda_3=fluxlutman.cz_lambda_3(),
        theta_i=theta_i,
        theta_f=np.deg2rad(fluxlutman.cz_theta_f()),
        sampling_rate=fluxlutman.sampling_rate())    # return in terms of theta
    epsilon_A = wfl.theta_to_eps(thetawave_A, fluxlutman.q_J2())
    amp_A = fluxlutman.calc_eps_to_amp(epsilon_A, state_A='11', state_B='02')
                 # transform detuning frequency to positive amplitude
    
    # Generate the second CZ pulse
    thetawave_B = wfl.martinis_flux_pulse(
        length=fluxlutman.cz_length()*(1-fluxlutman.czd_length_ratio()),
        lambda_2=fluxlutman.cz_lambda_2(),
        lambda_3=fluxlutman.cz_lambda_3(),
        theta_i=theta_i,
        theta_f=np.deg2rad(fluxlutman.cz_theta_f()),
        sampling_rate=fluxlutman.sampling_rate())    # return in terms of theta
    epsilon_B = wfl.theta_to_eps(thetawave_B, fluxlutman.q_J2())
    amp_B = fluxlutman.calc_eps_to_amp(epsilon_B, state_A='11', state_B='02', positive_branch=False)
                 # transform detuning frequency to negative amplitude

    # N.B. No amp scaling and offset present
    amp = np.concatenate([amp_A, amp_B])
    return amp







class CZ_trajectory_superoperator(det.Soft_Detector):
    def __init__(self, fluxlutman, noise_parameters_CZ, fitted_stepresponse_ty):
        """
        Detector for simulating a CZ trajectory.
        Args:
            fluxlutman (instr): an instrument that contains the parameters
                                required to generate the waveform for the trajectory, and the hamiltonian as well.
            noise_parameters_CZ: instrument that contains the noise parameters, plus some more
            fitted_stepresponse_ty: list of two elements, corresponding to the time t 
                                    and the step response in volts along the y axis
        Structure: compute input parameters necessary to compute time evolution (propagator), then compute quantities of interest
        Returns: quantites of interest
        """
        super().__init__()
        self.value_names = ['Cost func', 'Cond phase', 'L1', 'L2', 'avgatefid_pc', 'avgatefid_compsubspace_pc',
                            'phase_q0', 'phase_q1', 'avgatefid_compsubspace', 'avgatefid_compsubspace_pc_onlystaticqubit', 'population_02_state',
                            'cond_phase02']
        self.value_units = ['a.u.', 'deg', '%', '%', '%', '%', 'deg', 'deg', '%', '%', '%', 'deg']
        self.fluxlutman = fluxlutman
        self.noise_parameters_CZ = noise_parameters_CZ
        self.fitted_stepresponse_ty=fitted_stepresponse_ty      # list of 2 elements: stepresponse (=y)
                                                                # as a function of time (=t)

    def acquire_data_point(self, **kw):

        ### Extract relevant parameters to recreate the instrument locally (necessary for parallelization since intruments cannot be pickled)
        
        fluxlutman_args, noise_parameters_CZ_args = return_instrument_args(self.fluxlutman,self.noise_parameters_CZ)


        ### Discretize average (integral) over a Gaussian distribution
        mean = 0
        sigma_q0 = self.noise_parameters_CZ.sigma_q0()
        sigma_q1 = self.noise_parameters_CZ.sigma_q1()          # one for each qubit, in units of Phi_0
                 # 4e-6 is the same value as in the surface-17 paper of tom&brian. We see that 25 reproduces the T_phi^quasi-static for a Ramsey exp.

        qoi_plot = []    # used to verify convergence properties. If len(n_sampling_gaussian_vec)==1, it is useless
        n_sampling_gaussian_vec = self.noise_parameters_CZ.n_sampling_gaussian_vec()  # 11 guarantees excellent convergence.
                                                                                          # We choose it odd so that the central point of the Gaussian is included.
                                                                                          # ALWAYS choose it odd
        for n_sampling_gaussian in n_sampling_gaussian_vec:
            # If sigma=0 there's no need for sampling
            if sigma_q0 != 0:
                samplingpoints_gaussian_q0 = np.linspace(-5*sigma_q0,5*sigma_q0,n_sampling_gaussian)    # after 5 sigmas we cut the integral
                delta_x_q0 = samplingpoints_gaussian_q0[1]-samplingpoints_gaussian_q0[0]
                values_gaussian_q0 = czf.gaussian(samplingpoints_gaussian_q0,mean,sigma_q0)
            else:
                samplingpoints_gaussian_q0 = np.array([0])
                delta_x_q0 = 1
                values_gaussian_q0 = np.array([1])
            if sigma_q1 != 0:
                samplingpoints_gaussian_q1 = np.linspace(-5*sigma_q1,5*sigma_q1,n_sampling_gaussian)    # after 5 sigmas we cut the integral
                delta_x_q1 = samplingpoints_gaussian_q1[1]-samplingpoints_gaussian_q1[0]
                values_gaussian_q1 = czf.gaussian(samplingpoints_gaussian_q1,mean,sigma_q1)
            else:
                samplingpoints_gaussian_q1 = np.array([0])
                delta_x_q1 = 1
                values_gaussian_q1 = np.array([1])



            input_to_parallelize = []
            weights=[]
            number=-1           # used to number instruments that are created in the parallelization, to avoid conflicts

            
            for j_q0 in range(len(samplingpoints_gaussian_q0)):
                fluxbias_q0 = samplingpoints_gaussian_q0[j_q0]                     # q0 fluxing qubit
                for j_q1 in range(len(samplingpoints_gaussian_q1)): 
                    fluxbias_q1 = samplingpoints_gaussian_q1[j_q1]                 # q1 spectator qubit

                    number=number+1

                    if self.noise_parameters_CZ.cluster():
                        input_point = {'fluxbias_q0': fluxbias_q0,                  # need to pass it like this to the cluster
                                       'fluxbias_q1': fluxbias_q1,
                                       'fluxlutman_args': fluxlutman_args,
                                       'noise_parameters_CZ_args': noise_parameters_CZ_args,
                                       'fitted_stepresponse_ty': self.fitted_stepresponse_ty,
                                       'number': number,
                                       'cluster': self.noise_parameters_CZ.cluster()}
                    else:
                        input_point = {'fluxbias_q0': fluxbias_q0,                  # need to pass it like this to the cluster
                                       'fluxbias_q1': fluxbias_q1,
                                       'fluxlutman': self.fluxlutman,
                                       'noise_parameters_CZ': self.noise_parameters_CZ,
                                       'fitted_stepresponse_ty': self.fitted_stepresponse_ty,
                                       'number': number,
                                       'cluster': self.noise_parameters_CZ.cluster()}

                    weight = values_gaussian_q0[j_q0]*delta_x_q0 * values_gaussian_q1[j_q1]*delta_x_q1
                    weights.append(weight)

                    input_to_parallelize.append(input_point)


            if self.noise_parameters_CZ.cluster():
                y_list_of_lists = map_jobqueue_repeat(compute_propagator_parallelizable, 35, input_to_parallelize)          # function defined in notebook cluster

                y_list_of_lists = np.array(y_list_of_lists)
                U_final_vec = y_list_of_lists[:,0]
                t_final_vec = y_list_of_lists[:,1]
            else:
                U_final_vec = []
                t_final_vec = []
                for input_arglist in input_to_parallelize:
                    result_list = compute_propagator_parallelizable(input_arglist)
                    U_final_vec.append(result_list[0])
                    t_final_vec.append(result_list[1])


            for i in range(len(U_final_vec)):
                if U_final_vec[i].type == 'oper':
                    U_final_vec[i] = qtp.to_super(U_final_vec[i])           # weighted averaging needs to be done for superoperators
                U_final_vec[i] = U_final_vec[i] * weights[i]
            U_superop_average = np.sum(np.array(U_final_vec))               # computing resulting average propagator
            #print(czf.verify_CPTP(U_superop_average))


            t_final = t_final_vec[0]                                        # equal for all entries, we need it to compute phases in the rotating frame
            w_q0, w_q1, alpha_q0 = czf.dressed_frequencies(self.fluxlutman, self.noise_parameters_CZ)     # needed to compute phases in the rotating frame

            qoi = czf.simulate_quantities_of_interest_superoperator_new(U=U_superop_average,t_final=t_final,w_q0=w_q0,w_q1=w_q1,alpha_q0=alpha_q0)
            if self.noise_parameters_CZ.look_for_minimum():                             # if we look only for the minimum avgatefid_pc in the heat maps,
                                                                                        # then we optimize the search via higher-order cost function
                cost_func_val = (-np.log10(1-qoi['avgatefid_compsubspace_pc']))**4
            else:
                cost_func_val = (-np.log10(1-qoi['avgatefid_compsubspace_pc']))

            quantities_of_interest = [cost_func_val, qoi['phi_cond'], qoi['L1']*100, qoi['L2']*100, qoi['avgatefid_pc']*100, 
                             qoi['avgatefid_compsubspace_pc']*100, qoi['phase_q0'], qoi['phase_q1'], 
                             qoi['avgatefid_compsubspace']*100, qoi['avgatefid_compsubspace_pc_onlystaticqubit']*100, qoi['population_02_state']*100,
                             qoi['cond_phase02']]
            qoi_vec=np.array(quantities_of_interest)
            qoi_plot.append(qoi_vec)


        qoi_plot = np.array(qoi_plot)

        ## Plot to study the convergence properties of averaging over a Gaussian
        # for i in range(len(qoi_plot[0])):
        #     czf.plot(x_plot_vec=[n_sampling_gaussian_vec],
        #                   y_plot_vec=[qoi_plot[:,i]],
        #                   title='Study of convergence of average',
        #                   xlabel='n_sampling_gaussian points',ylabel=self.value_names[i])


        return qoi_plot[0,0], qoi_plot[0,1], qoi_plot[0,2], qoi_plot[0,3], qoi_plot[0,4], qoi_plot[0,5], qoi_plot[0,6], \
               qoi_plot[0,7], qoi_plot[0,8], qoi_plot[0,9], qoi_plot[0,10], qoi_plot[0,11]

           