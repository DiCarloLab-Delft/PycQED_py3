from pycqed.measurement import measurement_control as mc

import adaptive
from pycqed.instrument_drivers.meta_instrument.LutMans import flux_lutman as flm
from pycqed.instrument_drivers.virtual_instruments import noise_parameters_CZ_new as npCZ

from pycqed.simulations import cz_superoperator_simulation_new_functions as czf
import numpy as np
from pycqed.measurement import detector_functions as det
import matplotlib.pyplot as plt
from qcodes import Instrument
from pycqed.measurement.waveform_control_CC import waveforms_flux as wfl
from scipy.interpolate import interp1d
import qutip as qtp
#np.set_printoptions(threshold=np.inf)





def f_to_parallelize_new(arglist):

    fitted_stepresponse_ty = arglist['fitted_stepresponse_ty']
    fluxlutman_args = arglist['fluxlutman_args']       # [sampling_rate, cz_length, q_J2, czd_double_sided, cz_lambda_2, cz_lambda_3,
                                                           #        cz_theta_f, czd_length_ratio]
    noise_parameters_CZ_args = arglist['noise_parameters_CZ_args']       # [Z_rotations_length, voltage_scaling_factor, distortions, T1_q0, T1_q1, T2_q0_sweetspot, T2_q0_interaction_point,
                                                                             #      T2_q0_amplitude_dependent, T2_q1]
    number = arglist['number']
    adaptive_pars = arglist['adaptive_pars']

    try: 
        MC = Instrument.find_instrument('MC'+'{}'.format(number))
    except KeyError:
        MC = mc.MeasurementControl('MC'+'{}'.format(number), live_plot_enabled=False)
    from qcodes import station
    station = station.Station()
    station.add_component(MC)
    MC.station =station

    
    fluxlutman = flm.AWG8_Flux_LutMan('fluxlutman'+'{}'.format(number))
    station.add_component(fluxlutman)
    noise_parameters_CZ = npCZ.NoiseParametersCZ('noise_parameters_CZ'+'{}'.format(number))
    station.add_component(noise_parameters_CZ)

    fluxlutman, noise_parameters_CZ = czf.return_instrument_from_arglist(fluxlutman,fluxlutman_args,noise_parameters_CZ,noise_parameters_CZ_args)


    d=CZ_trajectory_superoperator(fluxlutman=fluxlutman, noise_parameters_CZ=noise_parameters_CZ,
                                  fitted_stepresponse_ty=fitted_stepresponse_ty, 
                                  qois=adaptive_pars.get('qois', 'all'))
    MC.set_detector_function(d)


    exp_metadata = {'double sided':fluxlutman.czd_double_sided(), 
                     'length': fluxlutman.cz_length(),
                     'distortions': noise_parameters_CZ.distortions(), 
                     'T2_scaling': noise_parameters_CZ.T2_scaling(), 
                     'sigma_q1': noise_parameters_CZ.sigma_q1(), 
                     'sigma_q0': noise_parameters_CZ.sigma_q0()}

    if adaptive_pars['mode']=='adaptive': 
        MC.set_sweep_functions([fluxlutman.cz_theta_f, fluxlutman.cz_lambda_2])
        if adaptive_pars['uniform']: 
            loss_per_triangle= adaptive.learner.learner2D.uniform_loss
        else: 
            loss_per_triangle=None
        MC.set_adaptive_function_parameters(
            {'adaptive_function': adaptive.Learner2D, 
            'loss_per_triangle': loss_per_triangle, 
             'goal':lambda l: l.npoints>adaptive_pars['n_points'], 
             'bounds':[(adaptive_pars['theta_f_min'], adaptive_pars['theta_f_max']), 
             (adaptive_pars['lambda2_min'], adaptive_pars['lambda2_max'])]})

        if noise_parameters_CZ.cluster():
            dat = MC.run('2D simulation_new_cluster2 double sided {} - length {:.0f} - distortions {} - T2_scaling {:.1f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.distortions(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='adaptive',exp_metadata=exp_metadata)

        else:
            if adaptive_pars['long_name']:
                dat = MC.run('2D simulation_new_2 double sided {} - length {:.0f} - distortions {} - T2_scaling {:.1f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.distortions(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='adaptive',exp_metadata=exp_metadata)
            else:
                dat = MC.run('2D simulation_new_2', 
                exp_metadata=exp_metadata, 
                mode='adaptive')

    elif adaptive_pars['mode']=='1D':
        MC.set_sweep_functions([fluxlutman.cz_theta_f])
        MC.set_sweep_points(np.linspace(adaptive_pars['theta_f_min'], 
            adaptive_pars['theta_f_max'],adaptive_pars['n_points']))
        if noise_parameters_CZ.cluster():
            dat = MC.run('1D simulation_new_cluster2 double sided {} - length {:.0f} - distortions {} - T2_scaling {:.1f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.distortions(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='1D',exp_metadata=exp_metadata)

        else:
            if adaptive_pars['long_name']:
                dat = MC.run('1D simulation_new_2 double sided {} - length {:.0f} - distortions {} - T2_scaling {:.1f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.distortions(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='1D',exp_metadata=exp_metadata)
            else:
                dat = MC.run('1D simulation_new_2', 
                exp_metadata=exp_metadata, 
                mode='1D')



    fluxlutman.close()
    noise_parameters_CZ.close()
    MC.close()





def compute_propagator(arglist):
    # arglist = [samplepoint_q0,samplepoint_q1,fluxlutman_args,noise_parameters_CZ_args,fitted_stepresponse_ty]

    fluxbias_q0 = arglist['fluxbias_q0']
    fluxbias_q1 = arglist['fluxbias_q1']
    fitted_stepresponse_ty = arglist['fitted_stepresponse_ty']
    fluxlutman = arglist['fluxlutman']
    noise_parameters_CZ = arglist['noise_parameters_CZ']


    sim_step=1/fluxlutman.sampling_rate()
    subdivisions_of_simstep=1                          # 4 is a good one, corresponding to a time step of 0.1 ns
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


    # Apply voltage scaling
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


    # We add the single qubit rotations at the end of the pulse
    if noise_parameters_CZ.Z_rotations_length() != 0:
        tlist_singlequbitrotations = np.arange(0,noise_parameters_CZ.Z_rotations_length(),sim_step_new)
        amp_Z_rotation = np.zeros(len(tlist_singlequbitrotations))+amp_final[0]
        amp_Z_rotation, f_pulse_Z_rotation = czf.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,amp_final=amp_Z_rotation,fluxbias_q0=fluxbias_q0)
        tlist_new = czf.concatenate_CZpulse_and_Zrotations(noise_parameters_CZ.Z_rotations_length(),sim_step_new,tlist_new)

    # We add the idle time at the end of the pulse (even if it's not at the end. It doesn't matter)
    if noise_parameters_CZ.total_idle_time() != 0:
        tlist_idle_time = np.arange(0,noise_parameters_CZ.total_idle_time(),sim_step_new)
        amp_idle_time = np.zeros(len(tlist_idle_time))+amp_final[0]
        double_sided = fluxlutman.czd_double_sided()
        fluxlutman.czd_double_sided(False)
        amp_idle_time, f_pulse_idle_time = czf.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,amp_final=amp_idle_time,fluxbias_q0=fluxbias_q0)
        fluxlutman.czd_double_sided(double_sided)
        tlist_new = czf.concatenate_CZpulse_and_Zrotations(noise_parameters_CZ.total_idle_time(),sim_step_new,tlist_new)


    ### the fluxbias_q0 affects the pulse shape after the distortions have been taken into account
    amp_final, f_pulse_final = czf.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,amp_final=amp_final,fluxbias_q0=fluxbias_q0)


    if noise_parameters_CZ.Z_rotations_length() != 0:
        amp_final=np.concatenate((amp_final,amp_Z_rotation))
        f_pulse_final=np.concatenate((f_pulse_final,f_pulse_Z_rotation))
    if noise_parameters_CZ.total_idle_time() != 0:
        amp_final=np.concatenate((amp_final,amp_idle_time))
        f_pulse_final=np.concatenate((f_pulse_final,f_pulse_idle_time))

    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp_final],
    #                          title='Pulse with (possibly) single qubit rotations',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')


    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp_final-amp_final_new],
    #                          title='Pulse with distortions and shift due to fluxbias_q0, difference',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')
    # amp_final = amp_final_new
    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[f_pulse_final/1e9],
    #                          title='Pulse with distortions and shift due to fluxbias_q0',
    #                            xlabel='Time (ns)',ylabel='Frequency (GHz)')


    t_final = tlist_new[-1]+sim_step_new


    ### Obtain jump operators, possibly time-dependent (incoherent part of the noise)
    c_ops = czf.return_jump_operators(noise_parameters_CZ=noise_parameters_CZ, f_pulse_final=f_pulse_final, fluxlutman=fluxlutman)


    ### Compute propagator
    U_final = czf.time_evolution_new(c_ops=c_ops, noise_parameters_CZ=noise_parameters_CZ, 
                                 fluxlutman=fluxlutman, fluxbias_q1=fluxbias_q1, amp=amp_final, sim_step=sim_step_new)
    #print(czf.verify_CPTP(U_superop_average))

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
    def __init__(self, fluxlutman, noise_parameters_CZ, fitted_stepresponse_ty, 
                qois='all'):
        """
        Detector for simulating a CZ trajectory.
        Args:
            fluxlutman (instr): an instrument that contains the parameters
                                required to generate the waveform for the trajectory, and the hamiltonian as well.
            noise_parameters_CZ: instrument that contains the noise parameters, plus some more
            fitted_stepresponse_ty: list of two elements, corresponding to the time t 
                                    and the step response in volts along the y axis
            qois: list
                list of quantities of interest, this can be used to return 
                only a select set of values. The list should contain 
                entries of "value_names". if qois=='all', all quantities are returned. 
        Structure: compute input parameters necessary to compute time evolution (propagator), then compute quantities of interest
        Returns: quantites of interest
        """
        super().__init__()
        self.value_names = ['Cost func', 'Cond phase', 'L1', 'L2', 'avgatefid_pc', 'avgatefid_compsubspace_pc',
                            'phase_q0', 'phase_q1', 'avgatefid_compsubspace', 'avgatefid_compsubspace_pc_onlystaticqubit', 'population_02_state',
                            'cond_phase02', 'coherent_leakage11', 'offset_difference', 'missing_fraction']
        self.value_units = ['a.u.', 'deg', '%', '%', '%', '%', 'deg', 'deg', '%', '%', '%', 'deg', '%', '%', '%']

        self.qois = qois
        if self.qois != 'all': 
            self.qoi_mask = [self.value_names.index(q) for q in qois]
            self.value_names = list(np.array(self.value_names)[self.qoi_mask])
            self.value_units = list(np.array(self.value_units)[self.qoi_mask])


        self.fluxlutman = fluxlutman
        self.noise_parameters_CZ = noise_parameters_CZ
        self.fitted_stepresponse_ty=fitted_stepresponse_ty      # list of 2 elements: stepresponse (=y)
                                                                # as a function of time (=t)

    def acquire_data_point(self, **kw):

        #czf.plot_spectrum(fluxlutman=self.fluxlutman,noise_parameters_CZ=self.noise_parameters_CZ)
        
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

                    input_point = {'fluxbias_q0': fluxbias_q0,                  # need to pass it like this to the cluster
                                   'fluxbias_q1': fluxbias_q1,
                                   'fluxlutman': self.fluxlutman,
                                   'noise_parameters_CZ': self.noise_parameters_CZ,
                                   'fitted_stepresponse_ty': self.fitted_stepresponse_ty}

                    weight = values_gaussian_q0[j_q0]*delta_x_q0 * values_gaussian_q1[j_q1]*delta_x_q1
                    weights.append(weight)

                    input_to_parallelize.append(input_point)


            U_final_vec = []
            t_final_vec = []
            for input_arglist in input_to_parallelize:
                result_list = compute_propagator(input_arglist)
                U_final_vec.append(result_list[0])
                t_final_vec.append(result_list[1])


            t_final = t_final_vec[0]                                        # equal for all entries, we need it to compute phases in the rotating frame
            w_q0, w_q1, alpha_q0 = czf.dressed_frequencies(self.fluxlutman, self.noise_parameters_CZ)     # needed to compute phases in the rotating frame


            '''
            #### Reproducing Leo's plots of cond_phase and leakage vs. flux offset (I order vs II order)
            leakage_vec=[]
            infid_vec=[]
            phase_q0_vec=[]
            phase_q1_vec=[]
            fluxbias_q0_vec=[]
            cond_phase_vec=[]

            mid_index=int(len(U_final_vec)/2)
            print(mid_index)

            for i in range(len(U_final_vec)):
                if U_final_vec[i].type == 'oper':
                    U_final_vec[i] = qtp.to_super(U_final_vec[i])
                qoi_temp = czf.simulate_quantities_of_interest_superoperator_new(U=U_final_vec[i],t_final=t_final,w_q0=w_q0,w_q1=w_q1,alpha_q0=alpha_q0)
                if i==mid_index:
                    print(qoi_temp)
                leakage_vec.append(qoi_temp['L1'])
                infid_vec.append(1-qoi_temp['avgatefid_compsubspace_pc'])
                phase_q0_vec.append(qoi_temp['phase_q0'])
                phase_q1_vec.append(qoi_temp['phase_q1'])
                cond_phase_vec.append(qoi_temp['phi_cond'])

                fluxbias_q0=input_to_parallelize[i]['fluxbias_q0']
                fluxbias_q0_vec.append(fluxbias_q0)

            #leakage_vec=np.array(leakage_vec)-leakage_vec[mid_index]
            cond_phase_vec=np.array(cond_phase_vec)-cond_phase_vec[mid_index]
            phase_q0_vec=np.array(phase_q0_vec)-phase_q0_vec[mid_index]
            phase_q1_vec=np.array(phase_q1_vec)-phase_q1_vec[mid_index]
            infid_vec=np.array(infid_vec)-infid_vec[mid_index]

            czf.plot(x_plot_vec=[np.array(fluxbias_q0_vec)*1e3],
                          y_plot_vec=[cond_phase_vec,phase_q0_vec,phase_q1_vec],
                          title='Sensitivity to quasi_static flux offsets',
                          xlabel='Flux offset (m$\Phi_0$)',ylabel='Phase (deg)',
                          legend_labels=['conditional phase err','phase QR err','phase QL err'])
            czf.plot(x_plot_vec=[np.array(fluxbias_q0_vec)*1e3],
                          y_plot_vec=[np.array(leakage_vec)*100],
                          title='Sensitivity to quasi_static flux offsets',
                          xlabel='Flux offset (m$\Phi_0$)',ylabel='Leakage $L_1$ (%)',
                          legend_labels=['leakage'])
            '''


            for i in range(len(U_final_vec)):
                if U_final_vec[i].type == 'oper':
                    U_final_vec[i] = qtp.to_super(U_final_vec[i])           # weighted averaging needs to be done for superoperators
                U_final_vec[i] = U_final_vec[i] * weights[i]
            U_superop_average = np.sum(np.array(U_final_vec))               # computing resulting average propagator
            #print(czf.verify_CPTP(U_superop_average))


            qoi = czf.simulate_quantities_of_interest_superoperator_new(U=U_superop_average,t_final=t_final,w_q0=w_q0,w_q1=w_q1,alpha_q0=alpha_q0)
            if self.noise_parameters_CZ.look_for_minimum():                             # if we look only for the minimum avgatefid_pc in the heat maps,
                                                                                        # then we optimize the search via higher-order cost function
                cost_func_val = (-np.log10(1-qoi['avgatefid_compsubspace_pc']))**4
            else:
                cost_func_val = (-np.log10(1-qoi['avgatefid_compsubspace_pc']))

            quantities_of_interest = [cost_func_val, qoi['phi_cond'], qoi['L1']*100, qoi['L2']*100, qoi['avgatefid_pc']*100, 
                             qoi['avgatefid_compsubspace_pc']*100, qoi['phase_q0'], qoi['phase_q1'], 
                             qoi['avgatefid_compsubspace']*100, qoi['avgatefid_compsubspace_pc_onlystaticqubit']*100, qoi['population_02_state']*100,
                             qoi['cond_phase02'], qoi['coherent_leakage11']*100, qoi['offset_difference']*100, qoi['missing_fraction']*100]
            qoi_vec=np.array(quantities_of_interest)
            qoi_plot.append(qoi_vec)


            '''leakage_vec=[]
            infid_vec=[]
            leakage_dephased_vec=[]
            infid_dephased_vec=[]
            dimensions = U_superop_average.dims
            U_temp = U_superop_average.full()
            U_temp[22,40]=0
            U_temp[38,40]=0
            U_temp[22,20]=0
            U_temp[38,20]=0
            U_superop_dephased = qtp.Qobj(U_temp,type='super',dims=dimensions)
            for n in range(1,201,2):
                U_superop_n=U_superop_average**n
                U_superop_dephased_n = U_superop_dephased**n
                qoi=czf.simulate_quantities_of_interest_superoperator_new(U=U_superop_n,t_final=t_final*n,w_q0=w_q0,w_q1=w_q1,alpha_q0=alpha_q0)
                qoi_dephased=czf.simulate_quantities_of_interest_superoperator_new(U=U_superop_dephased_n,t_final=t_final*n,w_q0=w_q0,w_q1=w_q1,alpha_q0=alpha_q0)
                leakage_vec.append(qoi['L1'])
                infid_vec.append(1-qoi['avgatefid_compsubspace_pc'])
                leakage_dephased_vec.append(qoi_dephased['L1'])
                infid_dephased_vec.append(1-qoi_dephased['avgatefid_compsubspace_pc'])

            czf.plot(x_plot_vec=[np.arange(1,200,2)],
                          #y_plot_vec=[leakage_vec,infid_vec,leakage_dephased_vec,infid_dephased_vec],
                          y_plot_vec=[np.array(leakage_vec)*100,np.array(leakage_dephased_vec)*100],
                          title='Repeated $CZ$ gates',
                          xlabel='Number of CZ gates',ylabel='Leakage (%)',
                          legend_labels=['Using directly the $CZ$ from the simulations','Depolarizing the leakage subspace'])

            print(leakage_vec)
            print(leakage_dephased_vec)'''


        qoi_plot = np.array(qoi_plot)

        ## Plot to study the convergence properties of averaging over a Gaussian
        # for i in range(len(qoi_plot[0])):
        #     czf.plot(x_plot_vec=[n_sampling_gaussian_vec],
        #                   y_plot_vec=[qoi_plot[:,i]],
        #                   title='Study of convergence of average',
        #                   xlabel='n_sampling_gaussian points',ylabel=self.value_names[i])

        return_values = [qoi_plot[0,0], qoi_plot[0,1], qoi_plot[0,2], qoi_plot[0,3], \
            qoi_plot[0,4], qoi_plot[0,5], qoi_plot[0,6], \
            qoi_plot[0,7], qoi_plot[0,8], qoi_plot[0,9], qoi_plot[0,10], \
            qoi_plot[0,11], qoi_plot[0,12], qoi_plot[0,13], qoi_plot[0,14]]
        if self.qois != 'all': 
            return np.array(return_values)[self.qoi_mask]
            
        else: 
            return return_values 












