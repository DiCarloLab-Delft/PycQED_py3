import numpy as np
import scipy.optimize as sp_opt
import scipy.integrate as sp_int
import scipy.special as sp_spc
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as uhfqc

"""
This file contains the simulation and optimization functions
which calculate the parameters for the CLEAR pulse
"""


def get_CLEAR_amplitudes(omega_pf, omega_ro_mid, omega_drive, kappa_pf, J, chi,
                 amp_base, length_total,  pf_0=0, ro_0=0, integration_time=None,
                 sampling_rate=1.8e9, delta_amp_segments=None,
                 length_segments=None,  sigma = 10e-9,
                 conversion_factor=1e6, niter=20, T=0.2, max_amp_diff = 3):
    '''
    If no optional parameters are given, decent starting
    parametersre chosen and all parameters are rescaled
    to MHz and ms if no other rescaling parameter is given.
    '''
    if length_segments == None: length_segments = length_total/10
    if length_segments == None: length_segments = length_total/10

    if integration_time == None: integration_time = int(2*length_total*sampling_rate)

    length_segments = [conversion_factor*length_segments]*4
    length_total = length_total*conversion_factor
    sigma = sigma*conversion_factor
    integration_time = integration_time*conversion_factor

    omega_pf, omega_ro_mid, omega_drive, kappa_pf, J, chi, sampling_rate = list(
        map(lambda x: x/conversion_factor,[omega_pf, omega_ro_mid, omega_drive,
                                           kappa_pf, J, chi, sampling_rate]))


    if delta_amp_segments == None:
        delta_amp_segments = [0]*4

    norm = analytical_CLEAR(length_total/2, amp_base, length_total,
                            delta_amp_segments, length_segments, sigma, norm=1)
    max_val = max(solve_read_out_differential_equations(omega_pf,
                                                        omega_ro_mid, omega_drive, kappa_pf, J, chi,
                                                        pf_0 = pf_0, ro_0 = ro_0,
                                                        integration_time = integration_time,
                                                        sampling_rate = sampling_rate,  pulse_par=[
            amp_base, length_total, delta_amp_segments, length_segments, sigma, norm]))


    parameters = [amp_base, sigma,  omega_pf, omega_ro_mid, omega_drive,
                  kappa_pf, J, chi, length_total, length_segments, sampling_rate,
                  integration_time, pf_0, ro_0, max_val]


    x0 = delta_amp_segments

    result_ring_up = sp_opt.basinhopping(lambda x: integration_value_function(
        [x[0], x[1], x0[2], x0[3]], parameters), x0[:2],
         minimizer_kwargs={'options':{'eps': 1e-3}, 'method': 'L-BFGS-B', 'bounds':
             ((-amp_base*(max_amp_diff+1),amp_base*max_amp_diff),
              (-amp_base*(max_amp_diff+1),amp_base*max_amp_diff))},
         niter = niter, T = T, niter_success = 10, stepsize = 0.5)
    result_ring_down = sp_opt.basinhopping(lambda x: integration_value_function(
        [result_ring_up.x[0],result_ring_up.x[1], x[0], x[1]], parameters), x0[2:],
         minimizer_kwargs={'options':{'eps': 1e-3}, 'method': 'L-BFGS-B', 'bounds':
             ((-amp_base*(max_amp_diff+1),amp_base*max_amp_diff),
              (-amp_base*(max_amp_diff+1),amp_base*max_amp_diff))},
         niter = niter, T = T, niter_success = 10, stepsize = 0.5)
    return  list(result_ring_up.x)+list(result_ring_down.x)

def integration_value_function( x, params):

    segment_delta =  list(x)
    amp_base, sigma,  omega_pf, omega_ro_mid, omega_drive, \
    kappa_pf, J, chi, length_total, length_segments, sampling_rate, \
    integration_time, pf_0, ro_0, max_val= params

    norm = analytical_CLEAR(length_total/2, amp_base, length_total,
                            segment_delta, length_segments, sigma, norm=1)
    pulse_par = [amp_base, length_total, segment_delta,
                 length_segments, sigma, norm]
    ro_mode_1 = solve_read_out_differential_equations(omega_pf,
                      omega_ro_mid, omega_drive, kappa_pf, J, chi,
                      pf_0 = pf_0, ro_0 = ro_0,
                      integration_time = integration_time,
                      sampling_rate = sampling_rate,  pulse_par=pulse_par)

    ro_mode_2 = solve_read_out_differential_equations(omega_pf,
                      omega_ro_mid, omega_drive, kappa_pf, J, -1*chi,
                      pf_0 = pf_0, ro_0 = ro_0,
                      integration_time = integration_time,
                      sampling_rate = sampling_rate, pulse_par=pulse_par)

    max_ro_photon_number = max(max(ro_mode_1), max(ro_mode_2))

    if (0.7*max_val) < max_ro_photon_number < (1.3*max_val):

        int1 = np.abs(np.sum(ro_mode_1[-int(integration_time * sampling_rate):])/
                      (np.sum(ro_mode_1[:-int(integration_time * sampling_rate)])))
        int2 = np.abs(np.sum(ro_mode_2[-int(integration_time * sampling_rate):])/
                      (np.sum(ro_mode_2[:-int(integration_time * sampling_rate)])))
        inte = max(int1, int2)
    else:
        inte = 1e6

    return inte

def solve_read_out_differential_equations(omega_pf, omega_ro_mid,
                                          omega_drive, kappa_pf,
                                          J, chi, pf_0=0, ro_0=0,
                                          integration_time=4e-7,
                                          sampling_rate=1.8e9, pulse_par=[]):
    '''
    detuning_ropf is the detuning between of the pf from the ro-frequency in Hz
    delt_drive is the detuning of the drive frequency from the ro frequency in Hz
    '''

    t = np.arange(0., pulse_par[1]+integration_time, 1/sampling_rate)

    solution = sp_int.odeint(define_odes,[
        np.real(ro_0), np.imag(ro_0), np.real(pf_0),
        np.imag(pf_0),], t, args = ([omega_pf, omega_ro_mid,
                                     omega_drive, kappa_pf, J, chi,
                                     sampling_rate, pulse_par[-1], pulse_par[:-1]],),
                             rtol = 1e-3)

    return np.absolute( np.square(solution[:, 2])+ np.square(solution[:, 3]))

def define_odes(modes, t, params):
    '''
    This returns a list of the right  side of the
    differential equations describing the modes in ro and pf
    '''
    pf_mode_re, pf_mode_im, ro_mode_re, ro_mode_im = modes
    omega_pf, omega_ro_mid, omega_drive, kappa_pf, J, chi, \
    sampling_rate, norm, pulse_par = params

    pf_mode = (pf_mode_re+1j*pf_mode_im)
    ro_mode = (ro_mode_re+1j*ro_mode_im)

    derivs = [np.real(-1j *(omega_pf-omega_drive)*pf_mode -
                      kappa_pf/2*pf_mode-1j * J * ro_mode +
                      np.sqrt(kappa_pf) *analytical_CLEAR(t,*pulse_par,norm=norm)),
              np.imag(-1j *(omega_pf-omega_drive)*pf_mode -
                      kappa_pf/2*pf_mode-1j *  J * ro_mode +
                      np.sqrt(kappa_pf) *analytical_CLEAR(t,*pulse_par,norm=norm)),
              np.real(-1j*(omega_ro_mid-omega_drive+chi)*ro_mode-1j*J*pf_mode),
              np.imag(-1j*(omega_ro_mid-omega_drive+chi)*ro_mode-1j*J*pf_mode)]

    return derivs

def analytical_CLEAR(t, amp_base, length_total, delta_amp_segments,
                     length_segments, sigma,norm=1):

    amp = amp_base * np.sqrt(np.pi/2)* sigma *(
            (amp_base+delta_amp_segments[3])*
            (sp_spc.erf((length_segments[2]-length_total+t)/np.sqrt(2)/sigma)-
             sp_spc.erf((-length_total+t)/np.sqrt(2)/sigma))+
            (amp_base+delta_amp_segments[2])*
            (sp_spc.erf((length_segments[2]+length_segments[3]-
                         length_total+t)/np.sqrt(2)/sigma)-
             sp_spc.erf((length_segments[2]-length_total+t)/np.sqrt(2)/sigma))+
            (amp_base)*
            (sp_spc.erf((-length_segments[0]-length_segments[1]+t)/np.sqrt(2)/sigma)-
             sp_spc.erf((length_segments[2]+length_segments[3]-
                         length_total+t)/np.sqrt(2)/sigma))+
            (amp_base+delta_amp_segments[1])*
            (sp_spc.erf((t-length_segments[0])/np.sqrt(2)/sigma)-
             sp_spc.erf((t-length_segments[0]-length_segments[1])/np.sqrt(2)/sigma))+
            (amp_base+delta_amp_segments[0])*
            (sp_spc.erf((t)/np.sqrt(2)/sigma)-sp_spc.erf(
                (t-length_segments[0])/np.sqrt(2)/sigma)))

    return amp/norm

# This is not used at the moment
def gauss_CLEAR_pulse( amp_base, length_total, delta_amp_segments,
                       length_segments, sigma, nr_sigma, sampling_rate=1.2e9):


    amp_pulse = uhfqc.CLEAR_shape(amp_base, length_total, delta_amp_segments,
                                  length_segments, sampling_rate = sampling_rate)

    amp_filtered = uhfqc.gaussian_filter( amp_pulse, sigma, nr_sigma,
                                          sampling_rate = sampling_rate)

    return amp_filtered

