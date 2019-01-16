from pycqed.measurement import measurement_control as mc

import adaptive
from pycqed.instrument_drivers.meta_instrument.LutMans import flux_lutman as flm
from pycqed.instrument_drivers.virtual_instruments import noise_parameters_CZ_new as npCZ

from pycqed.simulations import cz_superoperator_simulation_new_functions as czf
import numpy as np
from pycqed.measurement import detector_functions as det
import matplotlib.pyplot as plt
from pycqed.measurement.waveform_control_CC import waveforms_flux as wfl
from scipy.interpolate import interp1d
import qutip as qtp
from qcodes import Instrument
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

    d=ramsey_experiment(fluxlutman=fluxlutman, noise_parameters_CZ=noise_parameters_CZ,
                                                         fitted_stepresponse_ty=fitted_stepresponse_ty)
    MC.set_sweep_functions([fluxlutman.cz_length])
    MC.set_detector_function(d)
    MC.set_sweep_points(np.arange(0, adaptive_pars['max_time'], adaptive_pars['time_step']))

    exp_metadata = {'detuning': noise_parameters_CZ.detuning(), 
                     'sigma_q1': noise_parameters_CZ.sigma_q1(), 
                     'sigma_q0': noise_parameters_CZ.sigma_q0()}

    if noise_parameters_CZ.cluster():
        dat = MC.run('1D ramsey_new_cluster sigma_q1 {:.0f}, sigma_q0 {:.0f}, detuning {:.0f}'.format(noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6,
                                                                            noise_parameters_CZ.detuning()/1e6), 
            mode='1D',exp_metadata=exp_metadata)
    else:
        if adaptive_pars['long_name']:
            dat = MC.run('1D ramsey_new sigma_q1 {:.0f}, sigma_q0 {:.0f}, detuning {:.0f}'.format(noise_parameters_CZ.sigma_q1()*1e6, noise_parameters_CZ.sigma_q0()*1e6,
                                                                    noise_parameters_CZ.detuning()/1e6), 
                                                                    mode='1D',exp_metadata=exp_metadata)
        else:
            dat = MC.run('1D ramsey_new', mode='1D',exp_metadata=exp_metadata)


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


    sim_step=fluxlutman.cz_length()
    subdivisions_of_simstep=1                          # irrelevant for these simulations
    sim_step_new=sim_step/subdivisions_of_simstep      # waveform is generated according to sampling rate of AWG,
                                                       # but we can use a different step for simulating the time evolution
    tlist = [0]
    tlist_new = tlist


    freq = fluxlutman.q_freq_01() + noise_parameters_CZ.detuning()
    amp = [fluxlutman.calc_freq_to_amp(freq)]


    t_final = tlist_new[-1]+sim_step_new


    amp = amp * noise_parameters_CZ.voltage_scaling_factor() 
    amp_final=amp


    ### the fluxbias_q0 affects the pulse shape after the distortions have been taken into account
    #   Since we assume the hamiltonian to be constant on each side of the pulse, we just need two time steps
    if fluxlutman.czd_double_sided():
        amp_final=[amp_final[0],-amp_final[0]]    # Echo-Z
    else:
        amp_final=[amp_final[0],amp_final[0]]     # Ram-Z
    sim_step_new=sim_step_new/2
    amp_final, f_pulse_final = czf.shift_due_to_fluxbias_q0(fluxlutman=fluxlutman,amp_final=amp_final,fluxbias_q0=fluxbias_q0)


    ### Obtain jump operators, possibly time-dependent (incoherent part of the noise)
    c_ops = czf.return_jump_operators(noise_parameters_CZ=noise_parameters_CZ, f_pulse_final=f_pulse_final, fluxlutman=fluxlutman)


    ### Compute propagator
    U_final = czf.time_evolution_new(c_ops=c_ops, noise_parameters_CZ=noise_parameters_CZ, 
                                 fluxlutman=fluxlutman, fluxbias_q1=fluxbias_q1, amp=amp_final, sim_step=sim_step_new)
    #print(czf.verify_CPTP(U_superop_average))
    U_final = czf.rotating_frame_transformation_propagator_new(U=U_final, t=t_final, H=czf.calc_hamiltonian(amp[0],fluxlutman,noise_parameters_CZ))
                                    # important to use amp and NOT amp_final here because the fluxbias is random and unknown to us.

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







class ramsey_experiment(det.Soft_Detector):
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
        self.value_names = ['population_higher_state','population_lower_state']
        self.value_units = ['%', '%']
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


            for i in range(len(U_final_vec)):
                if U_final_vec[i].type == 'oper':
                    U_final_vec[i] = qtp.to_super(U_final_vec[i])           # weighted averaging needs to be done for superoperators
                U_final_vec[i] = U_final_vec[i] * weights[i]
            U_superop_average = np.sum(np.array(U_final_vec))               # computing resulting average propagator
            #print(czf.verify_CPTP(U_superop_average))


            t_final = t_final_vec[0]                                        # equal for all entries, we need it to compute phases in the rotating frame
            w_q0, w_q1, alpha_q0 = czf.dressed_frequencies(self.fluxlutman, self.noise_parameters_CZ)     # needed to compute phases in the rotating frame

            qoi = czf.quantities_of_interest_ramsey(U=U_superop_average,initial_state=self.noise_parameters_CZ.initial_state(),fluxlutman=self.fluxlutman,noise_parameters_CZ=self.noise_parameters_CZ)

            quantities_of_interest = [qoi['population_higher_state'], qoi['population_lower_state']]
            qoi_vec=np.array(quantities_of_interest)
            qoi_plot.append(qoi_vec)


        qoi_plot = np.array(qoi_plot)

        ## Uncomment to study the convergence properties of averaging over a Gaussian
        # for i in range(len(qoi_plot[0])):
        #     czf.plot(x_plot_vec=[n_sampling_gaussian_vec],
        #                   y_plot_vec=[qoi_plot[:,i]],
        #                   title='Study of convergence of average',
        #                   xlabel='n_sampling_gaussian points',ylabel=self.value_names[i])


        return qoi_plot[0,0], qoi_plot[0,1]














