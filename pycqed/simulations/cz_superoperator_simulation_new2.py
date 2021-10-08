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
import cma
#np.set_printoptions(threshold=np.inf)





def f_to_parallelize_new(arglist):
    # cluster wants a list as an argument.
    # Below the various list items are assigned to their own variable

    fitted_stepresponse_ty = arglist['fitted_stepresponse_ty']
    fluxlutman_args = arglist['fluxlutman_args']       # see function return_instrument_args in czf
    noise_parameters_CZ_args = arglist['noise_parameters_CZ_args']       # see function return_instrument_args in czf
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
            dat = MC.run('2D simulation_new_cluster2 double sided {} - length {:.1f} - distortions {} - waiting {:.2f} - T2_scaling {:.2f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.distortions(), noise_parameters_CZ.waiting_at_sweetspot(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='adaptive',exp_metadata=exp_metadata)

        else:
            if adaptive_pars['long_name']:
                dat = MC.run('2D simulation_new_2 double sided {} - length {:.1f} - distortions {} - T2_scaling {:.1f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
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
            dat = MC.run('1D simulation_new_cluster2 double sided {} - length {:.1f} - distortions {} - T2_scaling {:.1f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.distortions(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='1D',exp_metadata=exp_metadata)

        else:
            if adaptive_pars['long_name']:
                dat = MC.run('1D simulation_new_2 double sided {} - length {:.1f} - distortions {} - T2_scaling {:.1f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.distortions(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='1D',exp_metadata=exp_metadata)
            else:
                dat = MC.run('1D simulation_new_2', 
                exp_metadata=exp_metadata, 
                mode='1D')

    if adaptive_pars['mode']=='cma_optimizer': 
        MC.set_sweep_functions([fluxlutman.cz_theta_f, fluxlutman.cz_lambda_2])
        if adaptive_pars['uniform']: 
            loss_per_triangle= adaptive.learner.learner2D.uniform_loss
        else: 
            loss_per_triangle=None
        MC.set_adaptive_function_parameters(
            {'adaptive_function': cma.fmin,
             'x0': adaptive_pars['x0'], 'sigma0': adaptive_pars['sigma0'],
             # options for the CMA algorithm can be found using
             # "cma.CMAOptions()"
             'options': {'maxfevals': adaptive_pars['n_points'],    # maximum function cals
                         # Scaling for individual sigma's
                         'cma_stds': [5, 6, 3],
                         'ftarget': 0.005},     # Target function value
             })

        if noise_parameters_CZ.cluster():
            dat = MC.run('2D simulation_new_cluster2 double sided {} - length {:.1f} - waiting {:.2f} - T2_scaling {:.2f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.waiting_at_sweetspot(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='adaptive',exp_metadata=exp_metadata)

        else:
            if adaptive_pars['long_name']:
                dat = MC.run('2D simulation_new_2 double sided {} - length {:.1f} - distortions {} - T2_scaling {:.1f} - sigma_q1 {:.0f}, sigma_q0 {:.0f}'.format(fluxlutman.czd_double_sided(),
                fluxlutman.cz_length()*1e9, noise_parameters_CZ.distortions(), noise_parameters_CZ.T2_scaling(), noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6), 
                mode='adaptive',exp_metadata=exp_metadata)
            else:
                dat = MC.run('2D simulation_new_2', 
                exp_metadata=exp_metadata, 
                mode='adaptive')



    fluxlutman.close()
    noise_parameters_CZ.close()
    MC.close()





def compute_propagator(arglist):
    # I was parallelizing this function in the cluster, then I changed but the list as an argument remains.
    # Below each list item is assigned to its own variable

    fluxbias_q0 = arglist['fluxbias_q0']
    fluxbias_q1 = arglist['fluxbias_q1']
    fitted_stepresponse_ty = arglist['fitted_stepresponse_ty']
    fluxlutman = arglist['fluxlutman']
    noise_parameters_CZ = arglist['noise_parameters_CZ']


    sim_step=1/fluxlutman.sampling_rate()
    subdivisions_of_simstep=4                          # 4 is a good one, corresponding to a time step of 0.1 ns
    sim_step_new=sim_step/subdivisions_of_simstep      # waveform is generated according to sampling rate of AWG,
                                                       # but we can use a different step for simulating the time evolution
    tlist = np.arange(0, fluxlutman.cz_length(), sim_step)
    
    # residual_coupling=czf.conditional_frequency(0,fluxlutman,noise_parameters_CZ)      # To check residual coupling at the operating point.
    # print(residual_coupling)                                                       # Change amp to get the residual coupling at different points


    
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
    amp_temp=np.concatenate((amp,np.array([0])))    # amp should come back to the initial value, i.e. at the sweet spot
    amp_interp=interp1d(tlist_temp,amp_temp)
    amp=amp_interp(tlist_new)


    if fluxlutman.czd_double_sided() and noise_parameters_CZ.waiting_at_sweetspot()!=0:
        tlist_new, amp = czf.add_waiting_at_sweetspot(tlist_new,amp, noise_parameters_CZ.waiting_at_sweetspot())



    # Apply voltage scaling
    amp = amp * noise_parameters_CZ.voltage_scaling_factor() 


    ### Apply distortions
    if noise_parameters_CZ.distortions():
        amp_final = czf.distort_amplitude(fitted_stepresponse_ty=fitted_stepresponse_ty,amp=amp,tlist_new=tlist_new,sim_step_new=sim_step_new)
    else:
        amp_final = amp

    # Uncomment to get plots of the distorted pulse.
    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp_final],
    #                          title='Pulse with distortions, absolute',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')
    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp_final-amp],
    #                          title='Pulse with distortions, difference',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')

    ### the fluxbias_q0 affects the pulse shape after the distortions have been taken into account
    amp_final, f_pulse_final = czf.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,amp_final=amp_final,fluxbias_q0=fluxbias_q0,noise_parameters_CZ=noise_parameters_CZ)



    intervals_list = np.zeros(np.size(tlist_new)) + sim_step_new


    # We add the single qubit rotations at the end of the pulse
    if noise_parameters_CZ.Z_rotations_length() != 0:
        actual_Z_rotations_length = np.arange(0,noise_parameters_CZ.Z_rotations_length(),sim_step_new)[-1]+sim_step_new
        intervals_list = np.append(intervals_list,[actual_Z_rotations_length/2,actual_Z_rotations_length/2])
        amp_Z_rotation=[0,0]
        amp_Z_rotation, f_pulse_Z_rotation = czf.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,amp_final=amp_Z_rotation,fluxbias_q0=fluxbias_q0,noise_parameters_CZ=noise_parameters_CZ)

    # We add the idle time at the end of the pulse (even if it's not at the end. It doesn't matter)
    if noise_parameters_CZ.total_idle_time() != 0:
        actual_total_idle_time = np.arange(0,noise_parameters_CZ.total_idle_time(),sim_step_new)[-1]+sim_step_new
        intervals_list = np.append(intervals_list,[actual_total_idle_time/2,actual_total_idle_time/2])
        amp_idle_time=[0,0]
        double_sided = fluxlutman.czd_double_sided()                # idle time is single-sided so we save the fluxlutman.czd_double_sided() value, set it to False
                                                                    # and later restore it to the original value
        fluxlutman.czd_double_sided(False)
        amp_idle_time, f_pulse_idle_time = czf.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,amp_final=amp_idle_time,fluxbias_q0=fluxbias_q0,noise_parameters_CZ=noise_parameters_CZ)
        fluxlutman.czd_double_sided(double_sided)



    # We concatenate amp and f_pulse with the values they take during the Zrotations and idle_time.
    # It comes after the previous line because of details of the function czf.shift_due_to_fluxbias_q0
    if noise_parameters_CZ.Z_rotations_length() != 0:
        amp_final=np.concatenate((amp_final,amp_Z_rotation))
        f_pulse_final=np.concatenate((f_pulse_final,f_pulse_Z_rotation))
    if noise_parameters_CZ.total_idle_time() != 0:
        amp_final=np.concatenate((amp_final,amp_idle_time))
        f_pulse_final=np.concatenate((f_pulse_final,f_pulse_idle_time))

    # czf.plot(x_plot_vec=[np.arange(0,np.size(intervals_list))],y_plot_vec=[amp_final],
    #                          title='Pulse with (possibly) single qubit rotations and idle time',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')

    # czf.plot(x_plot_vec=[np.array(tlist_new)*1e9],y_plot_vec=[amp_final-amp_final_new],
    #                          title='Pulse with distortions and shift due to fluxbias_q0, difference',
    #                            xlabel='Time (ns)',ylabel='Amplitude (volts)')
    # amp_final = amp_final_new
    # czf.plot(x_plot_vec=[np.arange(0,np.size(intervals_list))],y_plot_vec=[f_pulse_final/1e9],
    #                          title='Pulse with distortions and shift due to fluxbias_q0',
    #                            xlabel='Time (ns)',ylabel='Frequency (GHz)')


    t_final = np.sum(intervals_list)        # actual overall gate length


    ### Obtain jump operators for Lindblad equation
    c_ops = czf.return_jump_operators(noise_parameters_CZ=noise_parameters_CZ, f_pulse_final=f_pulse_final, fluxlutman=fluxlutman)


    ### Compute propagator
    U_final = czf.time_evolution_new(c_ops=c_ops, noise_parameters_CZ=noise_parameters_CZ, 
                                 fluxlutman=fluxlutman, fluxbias_q1=fluxbias_q1, amp=amp_final, sim_step=sim_step_new, intervals_list=intervals_list)
    #print(czf.verify_CPTP(U_superop_average))    # simple check of CPTP property

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
        Returns: quantities of interest
        """
        super().__init__()
        self.value_names = ['Cost func', 'Cond phase', 'L1', 'L2', 'avgatefid_pc', 'avgatefid_compsubspace_pc',
                            'phase_q0', 'phase_q1', 'avgatefid_compsubspace', 'avgatefid_compsubspace_pc_onlystaticqubit', 'population_02_state',
                            'cond_phase02', 'coherent_leakage11', 'offset_difference', 'missing_fraction', '12_21_population_transfer', '12_03_population_transfer',
                            'phase_diff_12_02', 'phase_diff_21_20', 'cond_phase12', 'cond_phase21', 'cond_phase03', 'cond_phase20']
        self.value_units = ['a.u.', 'deg', '%', '%', '%', '%', 'deg', 'deg', '%', '%', '%', 'deg', '%', '%', '%', '%', '%', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg']

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
        
        ### Discretize average (integral) over a Gaussian distribution
        mean = 0
        sigma_q0 = self.noise_parameters_CZ.sigma_q0()
        sigma_q1 = self.noise_parameters_CZ.sigma_q1()          # one for each qubit, in units of Phi_0

        qoi_plot = []    # used to verify convergence properties. If len(n_sampling_gaussian_vec)==1, it is useless
        n_sampling_gaussian_vec = self.noise_parameters_CZ.n_sampling_gaussian_vec()      # 11 guarantees excellent convergence.
                                                                                          # We choose it odd so that the central point of the Gaussian is included.
                                                                                          # Always choose it odd
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



            input_to_parallelize = []               # This is actually the input that was parallelized in an old version.
                                                    # Currently it just creates a list that is provided sequentially to compute_propagator
            weights=[]
            number=-1           # used to number instruments that are created in the parallelization, to avoid conflicts in the cluster

            
            for j_q0 in range(len(samplingpoints_gaussian_q0)):
                fluxbias_q0 = samplingpoints_gaussian_q0[j_q0]                     # q0 fluxing qubit
                for j_q1 in range(len(samplingpoints_gaussian_q1)): 
                    fluxbias_q1 = samplingpoints_gaussian_q1[j_q1]                 # q1 spectator qubit

                    input_point = {'fluxbias_q0': fluxbias_q0,                  
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
            #w_q0, w_q1, alpha_q0, alpha_q1 = czf.dressed_frequencies(self.fluxlutman, self.noise_parameters_CZ)     # needed to compute phases in the rotating frame
            																										 # not used anymore


            ## Reproducing Leo's plots of cond_phase and leakage vs. flux offset (I order vs II order)
            #czf.sensitivity_to_fluxoffsets(U_final_vec,input_to_parallelize,t_final,self.fluxlutman,self.noise_parameters_CZ)


            for i in range(len(U_final_vec)):
                if U_final_vec[i].type == 'oper':
                    U_final_vec[i] = qtp.to_super(U_final_vec[i])           # weighted averaging needs to be done for superoperators
                U_final_vec[i] = U_final_vec[i] * weights[i]
            U_superop_average = sum(U_final_vec)              # computing resulting average propagator
            #print(czf.verify_CPTP(U_superop_average))


            qoi = czf.simulate_quantities_of_interest_superoperator_new(U=U_superop_average,t_final=t_final,fluxlutman=self.fluxlutman, noise_parameters_CZ=self.noise_parameters_CZ)

            if self.noise_parameters_CZ.look_for_minimum():                             # if we look only for the minimum avgatefid_pc in the heat maps,
                                                                                        # then we optimize the search via higher-order cost function
                cost_func_val = (-np.log10(1-qoi['avgatefid_compsubspace_pc']))**4
            else:
                cost_func_val = (-np.log10(1-qoi['avgatefid_compsubspace_pc']))

            quantities_of_interest = [cost_func_val, qoi['phi_cond'], qoi['L1']*100, qoi['L2']*100, qoi['avgatefid_pc']*100, 
                             qoi['avgatefid_compsubspace_pc']*100, qoi['phase_q0'], qoi['phase_q1'], 
                             qoi['avgatefid_compsubspace']*100, qoi['avgatefid_compsubspace_pc_onlystaticqubit']*100, qoi['population_02_state']*100,
                             qoi['cond_phase02'], qoi['coherent_leakage11']*100, qoi['offset_difference']*100, qoi['missing_fraction']*100, 
                             qoi['population_transfer_12_21']*100,qoi['population_transfer_12_03']*100,
                             qoi['phase_diff_12_02'], qoi['phase_diff_21_20'], qoi['cond_phase12'], qoi['cond_phase21'], qoi['cond_phase03'], qoi['cond_phase20']]
            qoi_vec=np.array(quantities_of_interest)
            qoi_plot.append(qoi_vec)


            ## To study the effect of the coherence of leakage on repeated CZs (simpler than simulating a full RB experiment):
            #czf.repeated_CZs_decay_curves(U_superop_average,t_final,self.fluxlutman,self.noise_parameters_CZ)


            #czf.plot_spectrum(self.fluxlutman,self.noise_parameters_CZ)


        qoi_plot = np.array(qoi_plot)

        ## Uncomment to study the convergence properties of averaging over a Gaussian
        # for i in range(len(qoi_plot[0])):
        #     czf.plot(x_plot_vec=[n_sampling_gaussian_vec],
        #                   y_plot_vec=[qoi_plot[:,i]],
        #                   title='Study of convergence of average',
        #                   xlabel='n_sampling_gaussian points',ylabel=self.value_names[i])

        return_values = [qoi_plot[0,0], qoi_plot[0,1], qoi_plot[0,2], qoi_plot[0,3], \
            qoi_plot[0,4], qoi_plot[0,5], qoi_plot[0,6], \
            qoi_plot[0,7], qoi_plot[0,8], qoi_plot[0,9], qoi_plot[0,10], \
            qoi_plot[0,11], qoi_plot[0,12], qoi_plot[0,13], qoi_plot[0,14], qoi_plot[0,15], qoi_plot[0,16], qoi_plot[0,17], qoi_plot[0,18],
            qoi_plot[0,19], qoi_plot[0,20], qoi_plot[0,21], qoi_plot[0,22]]
        if self.qois != 'all': 
            return np.array(return_values)[self.qoi_mask]
            
        else: 
            return return_values