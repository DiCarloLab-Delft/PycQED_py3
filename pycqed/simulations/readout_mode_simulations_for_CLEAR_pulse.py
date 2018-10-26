import numpy as np
import scipy.optimize as sp_opt
import scipy.integrate as sp_int
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as uhfqc

"""
This file contains the simulation and optimization functions
which calculate the parameters for the CLEAR pulse
"""


def get_CLEAR_parameters(omega_pf, omega_ro_mid, omega_drive, kappa_pf, J, chi,
             amp_base, length_total,  pf_0=0, ro_0=0, integration_time=None,
             sampling_rate=1.2e9, delta_amp_segments=None,
             length_segments=None,  sigma = 10e-9,
             conversion_factor=1e6, niter=50, T=0.2, nr_sigma=2):
    '''
    If no optional parameters are given, decent starting
    parametersre chosen and all parameters are rescaled
    to MHz and ms if no other rescaling parameter is given.
    '''
    if length_segments == None: length_segments = length_total/10

    if delta_amp_segments == None: delta_amp_segments = [
                    amp_base*0.1, amp_base*-0.5, amp_base*-1.5, amp_base*-0.5]

    if integration_time == None: integration_time = int(length_total*sampling_rate)
        
    if nr_sigma*sigma <= 0.1*length_total: 
        nr_sigma = int(.1* length_total/sigma)
        print('nr_sigma has been changed to: ''',nr_sigma)

    length_segments = [conversion_factor*length_segments]*4
    length_total = length_total*conversion_factor
    sigma = sigma*conversion_factor
    integration_time = integration_time*conversion_factor

    omega_pf, omega_ro_mid, omega_drive, kappa_pf, J, chi, sampling_rate = list(
        map(lambda x: x/conversion_factor,[omega_pf, omega_ro_mid, omega_drive,
                                           kappa_pf, J, chi, sampling_rate]))

    x0 = [ *delta_amp_segments]
    parameters = [amp_base, sigma,  omega_pf, omega_ro_mid, omega_drive,
                  kappa_pf, J, chi, length_total, length_segments, sampling_rate,
                  integration_time, pf_0, ro_0, nr_sigma ]

    result = sp_opt.basinhopping(
                    integration_value_function, x0,  minimizer_kwargs={
                    'args':parameters, 'options':{'eps': 1e-2}, 'bounds':(
                    (0,20),(0,20),(-20,0),(-20,0),)},
                    niter = niter, T = T, niter_success = 40, stepsize = 0.5)
    
    plt.show()
    
    return  list(result.x)



def solve_read_out_differential_equations(omega_pf, omega_ro_mid, omega_drive, kappa_pf,
                  J, chi, amp_base, length_total, delta_amp_segments,
                  length_segments,  sigma, pf_0=0, ro_0=0,
                  integration_time=4e-7, sampling_rate=1.2e9, nr_sigma=1):
    '''
    detuning_ropf is the detuning between of the pf from the ro-frequency in Hz
    delt_drive is the detuning of the drive frequency from the ro frequency in Hz
    '''

    duration = length_total+integration_time

    t = np.arange(0., duration, 1/sampling_rate)
    
    print(delta_amp_segments)

    solution = sp_int.odeint(helper_funct,[
                     np.real(ro_0), np.imag(ro_0), np.real(pf_0),
                     np.imag(pf_0),], t, args = ([omega_pf, omega_ro_mid,
                     omega_drive, kappa_pf, J, chi, amp_base, length_total,
                     delta_amp_segments, length_segments, sigma,
                     sampling_rate, nr_sigma],))

    return np.absolute( np.square(solution[:, 2])+ np.square(solution[:, 3]))





def integration_value_function( x, params):

    segment_delta =  list(x)
    amp_base, sigma,  omega_pf, omega_ro_mid, omega_drive,\
    kappa_pf, J, chi, length_total, length_segments, sampling_rate,\
    integration_time, pf_0, ro_0, nr_sigma = params

    ro_mode_1 = solve_read_out_differential_equations(omega_pf,
                omega_ro_mid, omega_drive, kappa_pf, J, chi, amp_base,
                length_total, segment_delta, length_segments,
                sigma, pf_0 = pf_0, ro_0 = ro_0,
                integration_time = integration_time,
                sampling_rate = sampling_rate, nr_sigma=nr_sigma)

    ro_mode_2 = solve_read_out_differential_equations(omega_pf,
                omega_ro_mid, omega_drive, kappa_pf, J, -1*chi, amp_base,
                length_total, segment_delta, length_segments,
                sigma, pf_0 = pf_0, ro_0 = ro_0,
                integration_time = integration_time,
                sampling_rate = sampling_rate, nr_sigma=nr_sigma)


    max_ro_photon_number = max(max(ro_mode_1), max(ro_mode_2))

    int1 = (np.sum(ro_mode_1[-int(integration_time/2 * sampling_rate):]) /
            max_ro_photon_number/(integration_time/2 * sampling_rate))
    int2 = (np.sum(ro_mode_2[-int(integration_time/2 * sampling_rate):]) /
           max_ro_photon_number/(integration_time/2 * sampling_rate))
    
    if int1>=int2:
        plt.semilogy(ro_mode_1/max_ro_photon_number )
    else:
        plt.semilogy(ro_mode_2/max_ro_photon_number )
        
    print(max(int1, int2))

    return max(int1, int2)


def gauss_CLEAR_pulse_value_at_t(t, amp_base, length_total, delta_amp_segments,
                               length_segments, sigma, nr_sigma, sampling_rate=1.2e9):
    '''
    returns the amplitude of the gauss filtered CLEAR envelope at a given time
    '''

    amp_pulse = uhfqc.CLEAR_shape(amp_base, length_total, delta_amp_segments,
                                  length_segments, sampling_rate = sampling_rate)

    amp_filtered = uhfqc.gaussian_filter( amp_pulse, sigma, nr_sigma,
                                          sampling_rate = sampling_rate)
    
    return amp_filtered[int(min(t*sampling_rate,len(amp_filtered)-1))]


def helper_funct(modes, t, params):
    '''
    This returns a list of the right  side of the
    differential equations describing the modes in ro and pf
    '''
    pf_mode_re, pf_mode_im, ro_mode_re, ro_mode_im = modes
    omega_pf, omega_ro_mid, omega_drive, kappa_pf, J, chi, amp_base, length_total,\
        delta_amp_segments, length_segments, sigma,sampling_rate, nr_sigma = params

    pulse = gauss_CLEAR_pulse_value_at_t(t, amp_base, length_total, delta_amp_segments,
                               length_segments, sigma, nr_sigma, sampling_rate=sampling_rate)

    pf_mode = (pf_mode_re+1j*pf_mode_im)
    ro_mode = (ro_mode_re+1j*ro_mode_im)

    derivs = [np.real(-1j *(omega_pf-omega_drive)*pf_mode -
                      kappa_pf/2*pf_mode-1j * J * ro_mode +
                      np.sqrt(kappa_pf) * pulse),
              np.imag(-1j *(omega_pf-omega_drive)*pf_mode -
                      kappa_pf/2*pf_mode-1j *  J * ro_mode +
                      np.sqrt(kappa_pf) * pulse),
              np.real(-1j*(omega_ro_mid-omega_drive+chi)*ro_mode-1j*J*pf_mode),
              np.imag(-1j*(omega_ro_mid-omega_drive+chi)*ro_mode-1j*J*pf_mode)]

    return derivs


