import logging
import numpy as np
from scipy.linalg import toeplitz
from scipy import special
from os.path import join
kernel_dir = None


def heaviside(t):
    hs = t >= 0
    # conversion required for type errors in numpy>=1.13
    return np.array(hs, dtype=int)


def square(t, width=1, t0=0):
    return heaviside(t-t0)-heaviside(t-t0-width)


def bounce(t, amp, time, sampling_rate=1):
    main_comp = (1-amp) * heaviside(t)
    bounce_comp = amp*np.double((t+(1/sampling_rate)/2) > time)*heaviside(t)
    return main_comp + bounce_comp

# generic function calculating the htilde (discrete impulse response) from
# a known step function
htilde = lambda fun, t, params, width=1: fun(t, params)-fun(t-width, params)


def kernel_generic(fun, t, *args, **kw):
    """
    Constructs a toeplitz matrix for a given function and performs
    the required inversion.
    """
    c = fun(t, *args, **kw)
    r = np.zeros(len(c))
    M = toeplitz(c=c, r=r)
    return np.linalg.inv(M)[:, 0]


def filter_matrix_generic2(fun, t, *params):
    logging.warning('use scipy.linalg.toeplitz instead')
    A = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        for j in range(len(t)):
            if t[i] < t[j]:
                A[i, j] = 0.
            else:
                A[i, j] = heaviside(i-j)*fun(t[i]-t[j], *params)
    return A


def save_kernel(kernel, save_file=None):
    '''Save kernel to specified kernels directory.'''
    # TODO: make this into a smarter safe function using the datadir of the
    # kernel object
    logging.warning('this does not save where you think it does')
    if save_file is None:
        save_file = kernel_name
    fn = join(kernel_dir, save_file)+'.txt'
    with open(fn, 'w') as f:
        for ii in np.arange(len(kernel)-1):
            f.write('%.12e\n' % (kernel[ii]))
        f.write('%.12e' % (kernel[-1]))
    print('File saved at {}'.format(fn))


# function calculating the filter matrix from an impulse response vector
def filter_matrix_from_htilde(htilde, t=None):
    if t is None:
        t = len(htilde)
    func = lambda ii: np.diag(htilde[ii]*np.ones(t-ii), k=-ii)
    return np.sum(np.array([func(tt) for tt in range(t)]), 0)


# Inverts "A" matrix describing a filter response to give the inverse
# precompensating filter.
kernel_from_filter_matrix = lambda A: np.linalg.inv(A)[:, 0]


def kernel_from_kernel_step(fun, kernel_length, params_dict, resolution=1):
    t_kernel = np.arange(0, kernel_length, resolution)
    return fun(t_kernel, params)-fun(t_kernel-width, params)


def kernel_from_kernel_stepvec(kernel_stepvec, width=1):
    kernel_out = np.zeros(kernel_stepvec.shape)
    kernel_out[:width] = kernel_stepvec[:width]
    kernel_out[width:] = kernel_stepvec[width:]-kernel_stepvec[:-width]
    # kernel_out = kernel_out / np.sum(kernel_out)
    return kernel_out


# response functions for a simple single-pole low-pass filter
# model parameter: tau (time constant)
step_lowpass = lambda t, tau: heaviside(t+1)*(1-np.exp(-(t+1)/tau))
htilde_lowpass = lambda t, tau: htilde(step_lowpass, t, tau)


# response functions for the low-pass filter arising from the skin effect of a coax
# model parameter: alpha (attenuation at 1 GHz)
def step_skineffect(t, alpha):
    # Surpresses the numpy divide by zero warning temporarily
    with np.errstate(divide='ignore'):
        return heaviside(t+1) * (1-special.erf(alpha/21./np.sqrt(t+1)))


htilde_skineffect = lambda t, alpha: step_skineffect(
    t, alpha) - step_skineffect(t-1, alpha)

# response functions for the high-pass filter arising from a bias-tee, modelled as a simple single-pole high-pass filter
# model parameter: tau (time constant)
step_biastee = lambda t, tau: heaviside(t)*(np.exp(-t/tau))
htilde_biastee = lambda t, tau: htilde(step_biastee, t, tau)


def kernel_biastee(t_kernel, tau):
    '''
    Calculate precompensation kernel for a bias tee.
    '''
    output_kernel = 1/tau*np.ones(t_kernel.shape)
    output_kernel[0] = 1.
    return output_kernel

# response functions for distortions arising from inline bounces in the flux-bias line which produce echoes as the qubit
# model parameters: pairs describing amplitude and arrival time (at the qubit) of an echo
# step_bounce = lambda t, pairs: heaviside(t) + np.sum(np.array(map(lambda pair: pair[0]*heaviside(t-pair[1]), pairs)),0)
step_bounce = lambda t, pairs: heaviside(
    t) + np.sum(np.array([pair[0]*heaviside(t - pair[1]) for pair in pairs]), 0)
htilde_bounce = lambda t, pairs: htilde(step_bounce, t, pairs)

# response functions for distortions arising from on-chip currents
# - additional decay effects arise from inductive time constants of complex current paths
# model parameters: pairs describing amplitude and time constant of each
# decay term
step_onchip = lambda t, pairs: heaviside(
    t)*(1 + np.sum(np.array(map(lambda pair: pair[0]*np.exp(-t/pair[1]), pairs)), 0))
htilde_onchip = lambda t, pairs: htilde(step_onchip, t, pairs)


def step_scope(t=None, params=None):
    # open file
    f = open(kernel_dir+params['file_name'], 'r')
    step = np.double(f.readlines())
    f.close()
    max_len_output = len(step)-2*params['points_per_ns']
    print('max trace time = %.3f ns' %
          (max_len_output/params['points_per_ns']))
    if t is None:
        t_index = np.array(range(max_len_output))
    elif np.double(t).shape == ():
        t_index_end = np.int(np.min(
            [params['step_first_point']+t*np.float(params['points_per_ns']), max_len_output]))
        t_index = np.array(range(0, t_index_end, params['points_per_ns']))
    else:
        t_scope = list((np.arange(
            max_len_output)-params['step_first_point'])/np.float(params['points_per_ns']))
#         t = t*(t>=-1)
        t_index = np.array([t_scope.index(tt)
                            for tt in t if (t_scope.index(tt) < max_len_output)])
    step = step-np.mean(step[:params['zero_average_over']])
#     step = step[params['step_first_point']:]
    step = step/np.mean(step[-params['top_average_over']:])
#     step = step[::params['points_per_ns']]/params['htilde_normalisation']
#     step = step/params['htilde_normalisation']
#     return step[t_index+params['points_per_ns']]
    return step[t_index]

htilde_scope = lambda t=None, params=None: htilde(step_scope, t, params)


def step_raw(file_name, process_step=True, step_params=None, norm_type='max'):

    f = open(kernel_dir+file_name, 'r')
    step = np.double(f.readlines())
    f.close()

    if process_step:
        if step_params is None:
            step_max = np.max(step)
            step_min = np.min(step)
            # Setting the threshold at 3/2 of the total range so it is robust
            # against compensation pulses
            step_mid = 2/3*abs(step_max-step_min) + step_min

            step_shift = step-step_mid
            step_upedge = np.where(step_shift > 0)[0]
            if step_upedge.size == 0:
                print('finding step edge failed')
            step_upedge = step_upedge[0]
            step_downedge = np.where(
                step_shift[step_upedge:] < 0)[0] + step_upedge
            if step_downedge.size == 0:
                print(
                    'did not find end of step - using trace end point instead')
                step_downedge = len(step)
            else:
                step_downedge = step_downedge[0]
            # print(step_max, step_min, step_mid, step_upedge, step_downedge)

            baseline_end = step_upedge - \
                np.where(
                    step[step_upedge-1::-1]-step[step_upedge:0:-1] > 0)[0][0]
            baseline_end_with_buffer = baseline_end-(step_upedge-baseline_end)
        #     print(baseline_end, baseline_end_with_buffer)
            baseline_mean = np.mean(step[:baseline_end_with_buffer])
            baseline_std = np.std(step[:baseline_end_with_buffer])
            start_sigmas = 2
            step_start = np.where(step > baseline_mean+6*baseline_std)[0][0]
        #     print(step_start)

            step_end = step_downedge - \
                np.where(
                    step[step_downedge-1::-1]-step[step_downedge:0:-1] < 0)[0][0]
            step_end_with_buffer = step_end-(step_downedge-step_end)
            step_length = step_end-baseline_end
            if norm_type == 'end':
                step_norm_window = step_length/10
                step_top = np.mean(
                    step[step_end_with_buffer:step_end_with_buffer-step_norm_window:-1])
                step_top_std = np.std(
                    step[step_end_with_buffer:step_end_with_buffer-step_norm_window:-1])
            elif norm_type == 'max':
                step_top = np.max(step)
            elif norm_type == 'index':
                # this is a stupid default value for the moment, because I
                # don't intend to use this yet
                step_index = step_end_with_buffer
                step_top = step[step_index]

            baseline = baseline_mean
            step_params = {'step_start': step_start, 'step_end':
                           step_end, 'baseline': baseline, 'step_top': step_top}

        else:
            if 'baseline' not in step_params:
                if 'baseline_end' not in step_params:
                    print('must specify either "baseline" or "baseline_end"')
                else:
                    step_params['baseline'] = np.mean(
                        step[:step_params['baseline_end']])
            if 'step_top' not in step_params:
                if norm_type == 'max':
                    step_params['step_top'] = np.max(step)
                else:
                    print(
                        'must specify "step_top" if "norm_type" is not "max"')

        step_norm = (
            step-step_params['baseline'])/(step_params['step_top'] -
                                           step_params['baseline'])

        return step_norm, step_params

    else:
        return step


def step_zeros(file_name, process_step=True,
               step_params=None, norm_type='max'):

    my_step_raw, my_step_params = step_raw(
        file_name, step_params=step_params, norm_type='max')
    my_step_zeros = np.zeros(my_step_raw.shape)
    my_step_zeros[my_step_params['step_start']:] = my_step_raw[
        my_step_params['step_start']:]

    return my_step_zeros, my_step_params


def step_sampled(file_name, step_width_ns, points_per_ns,
                 step_params=None, norm_type='max'):

    my_step_width_points = step_width_ns * points_per_ns
    my_step, my_step_params = step_raw(
        file_name, step_params=step_params, norm_type=norm_type)
#     idx_sample = np.arange(my_step_params['step_start'],my_step_params['step_end']-my_step_width_points,my_step_width_points)-my_step_width_points
#     t_sample = (idx_sample-(my_step_params['step_start']-my_step_width_points))/points_per_ns
    idx_sample = np.arange(my_step_params['step_start'], my_step_params[
                           'step_end']-my_step_width_points, my_step_width_points)
    t_sample = (idx_sample-my_step_params['step_start'])/points_per_ns

    return my_step[idx_sample], t_sample

# generic function calculating the htilde (discrete impulse response) from
# a known step function


def htilde_raw(step_vec, t=None, width=1):
    if t is None:
        t = np.arange(len(step_vec)-width)
    return step_vec[t+width]-step_vec[t]


def htilde_sampled(file_name, step_width_ns, points_per_ns,
                   step_params=None, norm_type='max'):

    my_step_width_points = step_width_ns * points_per_ns
    my_step, my_step_params = step_zeros(
        file_name, step_params=step_params, norm_type=norm_type)
    my_htilde = htilde_raw(my_step, width=my_step_width_points)
    idx_sample = np.arange(my_step_params['step_start'], my_step_params[
                           'step_end']-my_step_width_points, my_step_width_points)-my_step_width_points
    t_sample = (
        idx_sample-(my_step_params['step_start']-my_step_width_points))/points_per_ns

    return my_htilde[idx_sample.astype(int)+int(my_step_width_points/2)], t_sample+step_width_ns/2


def kernel_sampled(file_name, step_width_ns, points_per_ns, step_params=None,
                   max_points=600, return_step=True, norm_type='max'):

    my_htilde_sampled, my_htilde_t_sampled = htilde_sampled(
        file_name, step_width_ns, points_per_ns, step_params=step_params,
        norm_type=norm_type)
    if max_points is None:
        max_points = len(my_htilde_sampled)
    my_filter_matrix = filter_matrix_from_htilde(
        my_htilde_sampled[:max_points])
    print(my_filter_matrix.shape)
    my_kernel = kernel_from_filter_matrix(my_filter_matrix)

    if return_step:
        my_kernel_step = np.dot(
            np.linalg.inv(my_filter_matrix),
            np.ones((len(my_filter_matrix), 1))).flatten()
        return my_kernel, my_kernel_step
    else:
        return my_kernel


def get_all_sampled(file_name, step_width_ns, points_per_ns, max_points=600,
                    step_params=None, norm_type='max'):
    """
    step_width_ns (ints):  not all points are selected for inversion,
                    this selects the stepsize for the inversion
    points_per_ns (int) : sampling rate in GS/s


    Return an output dict containing
        t_direct     : time values as measured
        step_direct  : amplitude values as measured
        t_raw        : time shifted with autodetected start time
        step_raw     : normalized step
         FIXME raw should be renamed rescaled (or normalized or shifted)
    """

    f = open(kernel_dir+file_name, 'r')
    step_direct = np.double(f.readlines())
    f.close()


def get_all_sampled_vector(vector, step_width_ns, points_per_ns, max_points=600,
                           step_params=None, norm_type='max'):
    """
    step_width_ns (ints):  not all points are selected for inversion,
                    this selects the stepsize for the inversion
    points_per_ns (int) : sampling rate in GS/s


    Return an output dict containing
        t_direct     : time values as measured
        step_direct  : amplitude values as measured
        t_raw        : time shifted with autodetected start time
        step_raw     : normalized step
         FIXME raw should be renamed rescaled (or normalized or shifted)
    """

    step_direct = vector

#     print step_params

    my_step_raw, my_step_params = step_raw(
        file_name, step_params=step_params, norm_type='max')
    t_my_step_raw = (np.arange(len(my_step_raw)) -
                     my_step_params['step_start'])/np.float(points_per_ns)
    my_step_width_points = step_width_ns * points_per_ns

    my_step_zeros = np.zeros(my_step_raw.shape)
    my_step_zeros[my_step_params['step_start']:] = my_step_raw[
        my_step_params['step_start']:]

    my_htilde_raw = htilde_raw(my_step_zeros, width=my_step_width_points)
    t_my_htilde_raw = (np.arange(my_step_width_points, len(
        my_step_zeros))-my_step_params['step_start'])/np.float(points_per_ns)
    my_step_sampled, t_my_step_sampled = step_sampled(
        file_name, step_width_ns, points_per_ns, step_params=step_params,
        norm_type=norm_type)
    my_htilde_sampled, t_my_htilde_sampled = htilde_sampled(
        file_name, step_width_ns, points_per_ns, step_params=step_params,
        norm_type=norm_type)

    my_kernel_simple, my_kernel_step = kernel_sampled(
        file_name, step_width_ns, points_per_ns, step_params=step_params,
        max_points=max_points, return_step=True, norm_type=norm_type)
    t_my_kernel = np.arange(len(my_kernel_step))*step_width_ns

    output_dict = {}
    output_dict['step_direct'] = step_direct
    output_dict['step_raw'] = my_step_raw
    output_dict['step_zeros'] = my_step_zeros
    output_dict['step_params'] = my_step_params
    output_dict['t_step_raw'] = t_my_step_raw
    output_dict['htilde_raw'] = my_htilde_raw
    output_dict['t_htilde_raw'] = t_my_htilde_raw
    output_dict['step_sampled'] = my_step_sampled
    output_dict['t_step_sampled'] = t_my_step_sampled
    output_dict['htilde_sampled'] = my_htilde_sampled
    output_dict['t_htilde_sampled'] = t_my_htilde_sampled
    output_dict['kernel'] = my_kernel_simple
    output_dict['kernel_step'] = my_kernel_step
    output_dict['t_kernel'] = t_my_kernel

    return output_dict
kernel_name = 'my_current_kernel'
matrix_kernel_time = 400.


def bounce_kernel(amp: float=0.02, time: float=4,
                  length: float=601, sampling_rate: float=1):
    """
    Generates a bounce kernel, with the specified parameters.
    amp is the fraction of the signal that is reflected.

    kernel_step_function:
        (1-amp) heaviside(t) + amp*heaviside(t-time)

    Args:
        amp             (float) : amplitude of the decay
        time            (float) : time constant of the bounce
        length          (float) : total length of the kernel
        sampling_rate   (float) : sampling rate for which to generate the
            kernel. Default value is 1 to support deprecated behaviour
            in which everything is expressed in units of ns.
    returns:
        bounce_kernel (np.array) : the predistortion kernel to correct for
            an exponential decay.

    """
    nr_samples = int(length*sampling_rate)
    t_kernel = np.arange(nr_samples)/sampling_rate

    def htilde_bounce(t, amp, time, sampling_rate):
        first_comp = bounce(t, amp, time, sampling_rate=sampling_rate)
        sec_comp = bounce(t-1/sampling_rate, amp, time,
                          sampling_rate=sampling_rate)
        return first_comp - sec_comp

    if abs(amp) > 0.:
        kernel_bounce = kernel_generic(
            fun=htilde_bounce, t=t_kernel,
            amp=amp, time=time,
            sampling_rate=sampling_rate)
    else:
        kernel_bounce = np.zeros(int(nr_samples))
        kernel_bounce[0] = 1.
    kernel_bounce = kernel_bounce / np.sum(kernel_bounce)

    return kernel_bounce


def decay_kernel(amp: float=-1., tau: float =11000,
                 length: float=2000, offset: float=1,
                 sampling_rate: float=1):
    """
    Generates a decay kernel, with the specified parameters

    This kernel corrects for the following step function
        amp * exp(-t_kernel/tau) + offset

    amp and tau are the parameters estimated from the step function

    Args:
        amp             (float) : amplitude of the decay (should be negative)
        tau             (float) : time constant of the decay
        length          (float) : total length of the kernel
        offset          (float) : the offset, N.B. if the offset is close to 0
                one get's weird behaviour due to floating point errors.
        sampling_rate   (float) : sampling rate for which to generate the
            kernel. Default value is 1 to support deprecated behaviour
            in which everything is expressed in units of ns.
    returns:
        decay_kernel (np.array) : the predistortion kernel to correct for
            an exponential decay.
    """
    a = amp/offset
    amp_k = a / (a + 1)
    tau_k = (amp+1)*tau

    nr_samples = int(length*sampling_rate)
    t_kernel = np.arange(nr_samples)/sampling_rate

    # if abs(amp) > 0.:
    kernel_decay_step = (1 - amp_k*np.exp(-t_kernel/tau_k))/offset
    kernel_decay = kernel_from_kernel_stepvec(kernel_decay_step)
    # kernel_decay = np.zeros(kernel_decay_step.shape)
    # kernel_decay[0] = kernel_decay_step[0]
    # kernel_decay[1:] = kernel_decay_step[1:]-kernel_decay_step[:-1]
    # else:
    #     kernel_decay = np.zeros(int(nr_samples))
    #     kernel_decay[0] = 1.
    return kernel_decay


def skin_kernel(alpha=0., length=601):
    """
    Generates a skin effect kernel, with the specified parameters
    kernel_step_function
        heaviside(t+1)*(1-errf(alpha/21./np.sqrt(t+1)))
    """
    t_kernel = np.arange(int(length))
    if abs(alpha) > 0.:
        kernel_skineffect = kernel_generic(htilde_skineffect,
                                           t_kernel, alpha)
    else:
        kernel_skineffect = np.zeros(int(length))
        kernel_skineffect[0] = 1.
    return kernel_skineffect


def poly_kernel(coeffs: list, length: float=601, sampling_rate: float=1):
    """
    Generates the kernel that corresponds to a step response that is described
    by a polynomial.

    Note that the kernel corresponds to the impulse response and not to the
    step response.
    """

    nr_samples = int(length*sampling_rate)
    t_kernel = np.arange(nr_samples)/sampling_rate
    polyvec = np.polyval(coeffs, t_kernel)
    poly_kernel = kernel_from_kernel_stepvec(polyvec)

    return poly_kernel
