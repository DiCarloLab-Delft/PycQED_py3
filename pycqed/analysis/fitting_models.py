import numpy as np
import matplotlib.pyplot as plt
import lmfit
import logging
#################################
#   Fitting Functions Library   #
#################################


def RandomizedBenchmarkingDecay(numCliff, Amplitude, p, offset):
    val = Amplitude * (p**numCliff) + offset
    return val


def DoubleExpDampOscFunc(t, tau_1, tau_2,
                         freq_1, freq_2,
                         phase_1, phase_2,
                         amp_1, amp_2, osc_offset):
    cos_1 = amp_1 * (np.cos(2*np.pi*freq_1*t+phase_1)) * np.exp(-(t/tau_1))
    cos_2 = amp_2 * (np.cos(2*np.pi*freq_2*t+phase_2)) * np.exp(-(t/tau_2))
    return cos_1 + cos_2 + osc_offset


def double_RandomizedBenchmarkingDecay(numCliff, p, offset,
                                       invert=1):
    """
    A variety of the RB-curve that allows fitting both the inverting and
    non-inverting exponential.
    The amplitude of the decay curve is constrained to start at 0 or 1.
    The offset is the common point both curves converge to.

    pick invert to be 1 or 0
    """
    # Inverting clifford curve
    val_inv = (1-offset) * (p**numCliff) + offset
    # flipping clifford curve
    val_flip = -offset * (p**numCliff) + offset
    # Using invert as a boolean but not using if statement to allow for
    # arrays to be input in the function
    val = (1-invert) * val_flip + invert*val_inv
    return val


def LorentzFunc(f, amplitude, center, sigma):
    val = amplitude/np.pi * (sigma / ((f-center)**2 + sigma**2))
    return val


def Lorentzian(f, A, offset, f0, kappa):
    val = offset + A/np.pi * (kappa / ((f-f0)**2 + kappa**2))
    return val


def TwinLorentzFunc(f, amplitude_a, amplitude_b, center_a, center_b,
                    sigma_a, sigma_b, background=0):
    'twin lorentz with background'
    val = (amplitude_a/np.pi * (sigma_a / ((f-center_a)**2 + sigma_a**2)) +
           amplitude_b/np.pi * (sigma_b / ((f-center_b)**2 + sigma_b**2)) +
           background)
    return val


def Qubit_dac_to_freq(dac_voltage, f_max, E_c,
                      dac_sweet_spot, dac_flux_coefficient, asymmetry=0):
    '''
    The cosine Arc model for uncalibrated flux for asymmetric qubit.

    dac_voltage (V)
    f_max (Hz)
    E_c (Hz)
    dac_sweet_spot (V)
    dac_flux_coefficient (1/V)
    asym (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2)),
    '''
    qubit_freq = (f_max + E_c)*(
        asymmetry**2 + (1-asymmetry**2) *
        np.sqrt(abs(np.cos(dac_flux_coefficient*(dac_voltage-dac_sweet_spot)))))-E_c
    return qubit_freq


def Qubit_freq_to_dac(frequency, f_max, E_c,
                      dac_sweet_spot, dac_flux_coefficient, asymmetry=0,
                      branch='positive'):
    '''
    The cosine Arc model for uncalibrated flux for asymmetric qubit.
    This function implements the inverse of "Qubit_dac_to_freq"

    frequency (Hz)
    f_max (Hz)
    E_c (Hz)
    dac_sweet_spot (V)
    dac_flux_coefficient (1/V)
    asym (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2)),
    branch (enum: 'positive' 'negative')
    '''

    asymm_term = (asymmetry**2 + (1-asymmetry**2))
    dac_term = np.arccos(((frequency+E_c)/((f_max+E_c) * asymm_term))**2)
    if branch == 'positive':
        dac_voltage = (dac_term)/dac_flux_coefficient+dac_sweet_spot
    elif branch == 'negative':
        dac_voltage = -(dac_term)/dac_flux_coefficient+dac_sweet_spot
    else:
        raise ValueError('branch {} not recognized'.format(branch))

    return dac_voltage


def QubitFreqDac(dac_voltage, f_max, E_c,
                 dac_sweet_spot, dac_flux_coefficient, asymmetry=0):
    logging.warning('deprecated, replace QubitFreqDac with Qubit_dac_to_freq')
    return Qubit_dac_to_freq(dac_voltage, f_max, E_c,
                             dac_sweet_spot, dac_flux_coefficient, asymmetry)


def QubitFreqFlux(flux, f_max, E_c,
                  flux_zero, dac_offset=0):
    'The cosine Arc model for calibrated flux.'
    calculated_frequency = (f_max + E_c)*np.sqrt(np.abs(
        np.cos(np.pi*(flux-dac_offset)/flux_zero))) - E_c
    return calculated_frequency


def CosFunc(t, amplitude, frequency, phase, offset):
    '''
    parameters:
        t, time in s
        amplitude a.u.
        frequency in Hz (f, not omega!)
        phase in rad
        offset a.u.
    '''
    return amplitude*np.cos(2*np.pi*frequency*t + phase)+offset


def ExpDecayFunc(t, tau, amplitude, offset, n):
    return amplitude*np.exp(-(t/tau)**n)+offset


def ExpDampOscFunc(t, tau, n, frequency, phase, amplitude,
                   oscillation_offset, exponential_offset):
    return amplitude*np.exp(-(t/tau)**n)*(np.cos(
        2*np.pi*frequency*t+phase)+oscillation_offset) + exponential_offset


def GaussExpDampOscFunc(t, tau, tau_2, frequency, phase, amplitude,
                        oscillation_offset, exponential_offset):
    return amplitude*np.exp(-(t/tau_2)**2 - (t/tau))*(np.cos(
        2*np.pi*frequency*t+phase)+oscillation_offset) + exponential_offset


def ExpDampDblOscFunc(t, tau, n, freq_1, freq_2, phase_1, phase_2, amp_1, amp_2,
                      osc_offset_1, osc_offset_2, exponential_offset):
    '''
    Exponential decay with double cosine modulation
    '''
    exp_decay = np.exp(-(t/tau)**n)
    cos_1 = (np.cos(
        2*np.pi*freq_1*t+phase_1)+osc_offset_1)
    cos_2 = (np.cos(
        2*np.pi*freq_2*t+phase_2)+osc_offset_2)

    return amp_1*exp_decay*cos_1 + amp_2*exp_decay*cos_2 + exponential_offset


def HangerFuncAmplitude(f, f0, Q, Qe, A, theta):
    '''
    This is the function for a hanger  which does not take into account
    a possible slope.
    This function may be preferred over SlopedHangerFunc if the area around
    the hanger is small.
    In this case it may misjudge the slope
    Theta is the asymmetry parameter

    Note! units are inconsistent
    f is in Hz
    f0 is in GHz
    '''
    return abs(A*(1.-Q/Qe*np.exp(1.j*theta)/(1.+2.j*Q*(f/1.e9-f0)/f0)))


def HangerFuncComplex(f, pars):
    '''
    This is the complex function for a hanger which DOES NOT
    take into account a possible slope
    Input:
        f = frequency
        pars = parameters dictionary
               f0, Q, Qe, A, theta, phi_v, phi_0

    Author: Stefano Poletto
    '''
    f0 = pars['f0']
    Q = pars['Q']
    Qe = pars['Qe']
    A = pars['A']
    theta = pars['theta']
    phi_v = pars['phi_v']
    phi_0 = pars['phi_0']

    S21 = A*(1-Q/Qe*np.exp(1j*theta)/(1+2.j*Q*(f/1.e9-f0)/f0)) * \
        np.exp(1j*(phi_v*f+phi_0))

    return S21


def PolyBgHangerFuncAmplitude(f, f0, Q, Qe, A, theta, poly_coeffs):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible polynomial background
    # NOT DEBUGGED
    return np.abs((1.+np.polyval(poly_coeffs, (f/1.e9-f0)/f0)) *
                  HangerFuncAmplitude(f, f0, Q, Qe, A, theta))


def SlopedHangerFuncAmplitude(f, f0, Q, Qe, A, theta, slope):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible slope df
    return np.abs((1.+slope*(f/1.e9-f0)/f0) *
                  HangerFuncAmplitude(f, f0, Q, Qe, A, theta))


def SlopedHangerFuncComplex(f, f0, Q, Qe, A, theta, phi_v, phi_0, slope):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible slope df
    return (1.+slope*(f/1.e9-f0)/f0)*np.exp(1.j*(phi_v*f+phi_0-phi_v*f[0])) * \
        HangerFuncComplex(f, f0, Q, Qe, A, theta)


def linear_with_offset(x, a, b):
    '''
    A linear signal with a fixed offset.
    '''
    return a*x + b


def linear_with_background(x, a, b):
    '''
    A linear signal with a fixed background.
    '''
    return np.sqrt((a*x)**2 + b**2)


def linear_with_background_and_offset(x, a, b, c):
    '''
    A linear signal with a fixed background.
    '''
    return np.sqrt((a*x)**2 + b**2)+c


def gaussian_2D(x, y, amplitude=1,
                center_x=0, center_y=0,
                sigma_x=1, sigma_y=1):
    '''
    A 2D gaussian function. if you want to use this for fitting you need to
    flatten your data first.
    '''
    gaus = lmfit.lineshapes.gaussian
    val = (gaus(x, amplitude, center_x, sigma_x) *
           gaus(y, amplitude, center_y, sigma_y))
    return val


def TripleExpDecayFunc(t, tau1, tau2, tau3, amp1, amp2, amp3, offset, n):
    return (offset +
            amp1*np.exp(-(t/tau1)**n) +
            amp2*np.exp(-(t/tau2)**n) +
            amp3*np.exp(-(t/tau3)**n))


def avoided_crossing_mediated_coupling(flux, f_bus, f_center1, f_center2,
                                       c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_b,  g,  g ]
        [g,   f_1, 0 ]
        [g,   0,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2
    f_b = constant

    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return

    """
    if type(flux_state) == int:
        flux_state = [flux_state]*len(flux)

    frequencies = np.zeros([len(flux), 2])
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        matrix = [[f_bus, g, g],
                  [g, f_1, 0.],
                  [g, 0., f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return result


def avoided_crossing_direct_coupling(flux, f_center1, f_center2,
                                     c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_1, g ]
        [g,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2

    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return
    """

    if type(flux_state) == int:
        flux_state = [flux_state]*len(flux)

    frequencies = np.zeros([len(flux), 2])
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return result


######################
# Residual functions #
######################
def residual_complex_fcn(pars, cmp_fcn, x, y):
    '''
    Residual of a complex function with complex results 'y' and
    real input values 'x'
    For resonators 'x' is the the frequency, 'y' the complex transmission
    Input:
        pars = parameters dictionary (check the corresponding function 'cmp_fcn' for the parameters to pass)
        cmp_fcn = complex function
        x = input real values to 'cmp_fcn'
        y = output complex values from 'cmp_fcn'

    Author = Stefano Poletto
    '''
    cmp_values = cmp_fcn(x, pars)

    res = cmp_values-y
    res = np.append(res.real, res.imag)

    return res


####################
# Guess functions  #
####################
def exp_dec_guess(model, data, t):
    '''
    Assumes exponential decay in estimating the parameters
    '''
    offs_guess = data[np.argmax(t)]
    amp_guess = data[np.argmin(t)] - offs_guess
    # guess tau by looking for value closest to 1/e
    tau_guess = t[np.argmin(abs((amp_guess*(1/np.e) + offs_guess)-data))]
    params = model.make_params(amplitude=amp_guess,
                               tau=tau_guess,
                               n=1,
                               offset=offs_guess)
    return params


def Cos_guess(model, data, t):
    '''
    Guess for a cosine fit using FFT, only works for evenly spaced points
    '''
    amp_guess = abs(max(data)-min(data))/2  # amp is positive by convention
    offs_guess = np.mean(data)

    # Freq guess ! only valid with uniform sampling
    w = np.fft.fft(data)
    f = np.fft.fftfreq(len(data), t[1]-t[0])
    w[0] = 0  # Removes DC component from fourier transform

    # Use absolute value of complex valued spectrum
    abs_w = np.abs(w)
    freq_guess = abs(f[abs_w == max(abs_w)][0])
    ph_guess = (-2*np.pi*t[data == max(data)]*freq_guess)[0]
    # the condition data == max(data) can have several solutions
    #               (for example when discretization is visible)
    # to prevent errors we pick the first solution

    model.set_param_hint('period', expr='1/frequency')
    params = model.make_params(amplitude=amp_guess,
                               frequency=freq_guess,
                               phase=ph_guess,
                               offset=offs_guess)
    params['amplitude'].min = 0  # Ensures positive amp
    params['frequency'].min = 0

    return params


def gauss_2D_guess(model, data, x, y):
    '''
    takes the mean of every row/column and then uses the regular gauss guess
    function to get a guess for the model parameters.

    Assumptions on input data
        * input is a flattened version of a 2D grid.
        * total surface under the gaussians sums up to 1.

    Potential improvements:
        Make the input also accept a 2D grid of data to prevent reshaping.
        Find a way to extract amplitude guess from data itself, note that
        taking the sum of the data (which should correspond to all data under
                                    the curve) does not do the trick.

    Note: possibly not compatible if the model uses prefixes.
    '''
    data_grid = data.reshape(-1, len(np.unique(x)))
    x_proj_data = np.mean(data_grid, axis=0)
    y_proj_data = np.mean(data_grid, axis=1)

    x_guess = lmfit.models.GaussianModel().guess(x_proj_data, np.unique(x))
    y_guess = lmfit.models.GaussianModel().guess(y_proj_data, np.unique(y))

    params = model.make_params(amplitude=1,
                               center_x=x_guess['center'].value,
                               center_y=y_guess['center'].value,
                               sigma_x=x_guess['sigma'].value,
                               sigma_y=y_guess['sigma'].value)
    return params


def double_gauss_2D_guess(model, data, x, y):
    '''
    takes the mean of every row/column and then uses the guess
    function of the double gauss.

    Assumptions on input data
        * input is a flattened version of a 2D grid.
        * total surface under the gaussians sums up to 1.
    Note: possibly not compatible if the model uses prefixes.
    Note 2: see also gauss_2D_guess() for some notes on how to improve this
            function.
    '''
    data_grid = data.reshape(-1, len(np.unique(x)))
    x_proj_data = np.mean(data_grid, axis=0)
    y_proj_data = np.mean(data_grid, axis=1)

    # The syntax here is slighly different than when calling a regular guess
    # function because I do not overwrite the class attribute properly.
    x_guess = double_gauss_guess(model=None, data=x_proj_data, x=np.unique(x))
    y_guess = double_gauss_guess(model=None, data=y_proj_data, x=np.unique(y))

    if model is not None:
        pars = model.make_params(A_sigma_x=x_guess['A_sigma'],
                                 A_sigma_y=y_guess['A_sigma'],
                                 A_center_x=x_guess['A_center'],
                                 A_center_y=y_guess['A_center'],
                                 A_amplitude=1,
                                 B_sigma_x=x_guess['B_sigma'],
                                 B_sigma_y=y_guess['B_sigma'],
                                 B_center_y=y_guess['B_center'],
                                 B_center_x=x_guess['B_center'],
                                 B_amplitude=1)
        return pars
    else:
        return x_guess, y_guess


def double_gauss_guess(model, data, x=None, **kwargs):
    '''
    Finds a guess for the intial parametes of the double gauss model.
    Guess is based on taking the cumulative sum of the data and
    finding the points corresponding to 25% and 75%
    it finds sigma by using the property that ~33% of the data is contained
    in the range mu-sigma to mu+sigma.
    '''
    if x is None:
        x = np.arange(len(data))
    cdf = np.cumsum(data)
    norm_cdf = cdf/cdf[-1]
    par_dict = {'A_center': x[(np.abs(norm_cdf - 0.25)).argmin()],
                'B_center': x[(np.abs(norm_cdf - 0.75)).argmin()],
                'A_sigma': (x[(np.abs(norm_cdf - 0.25 - .33/2)).argmin()] -
                            x[(np.abs(norm_cdf - 0.25 + .33/2)).argmin()]),
                'B_sigma': (x[(np.abs(norm_cdf - 0.75 - .33/2)).argmin()] -
                            x[(np.abs(norm_cdf - 0.75 + .33/2)).argmin()])}

    amp = max(data)*(par_dict['A_sigma'] + par_dict['B_sigma'])/2.
    if model is not None:
        # Specify explicitly because not all pars are set to those from the par
        # dict
        pars = model.make_params(A_center=par_dict['A_center'],
                                 B_center=par_dict['B_center'],
                                 A_sigma=par_dict['A_sigma'],
                                 B_sigma=par_dict['B_sigma'],
                                 A_amplitude=amp, B_amplitude=amp)
        return pars
    # The else clause is added explicitly to reuse this function for the
    # 2D double gauss model
    else:
        return par_dict

#################################
#     User defined Models       #
#################################
# NOTE: it is actually better to instantiate the model within your analysis
# file, this prevents the model params having a memory.
# A valid reason to define it here would be if you want to add a guess function
CosModel = lmfit.Model(CosFunc)
CosModel.guess = Cos_guess

ExpDecayModel = lmfit.Model(ExpDecayFunc)
TripleExpDecayModel = lmfit.Model(TripleExpDecayFunc)
ExpDecayModel.guess = exp_dec_guess
ExpDampOscModel = lmfit.Model(ExpDampOscFunc)
GaussExpDampOscModel = lmfit.Model(GaussExpDampOscFunc)
ExpDampDblOscModel = lmfit.Model(ExpDampDblOscFunc)
DoubleExpDampOscModel = lmfit.Model(DoubleExpDampOscFunc)
HangerAmplitudeModel = lmfit.Model(HangerFuncAmplitude)
SlopedHangerAmplitudeModel = lmfit.Model(SlopedHangerFuncAmplitude)
PolyBgHangerAmplitudeModel = lmfit.Model(PolyBgHangerFuncAmplitude)
HangerComplexModel = lmfit.Model(HangerFuncComplex)
SlopedHangerComplexModel = lmfit.Model(SlopedHangerFuncComplex)
QubitFreqDacModel = lmfit.Model(QubitFreqDac)
QubitFreqFluxModel = lmfit.Model(QubitFreqFlux)
TwinLorentzModel = lmfit.Model(TwinLorentzFunc)
LorentzianModel = lmfit.Model(Lorentzian)
RBModel = lmfit.Model(RandomizedBenchmarkingDecay)
LinOModel = lmfit.Model(linear_with_offset)
LinBGModel = lmfit.Model(linear_with_background)
LinBGOModel = lmfit.Model(linear_with_background_and_offset)

# 2D models
Gaus2D_model = lmfit.Model(gaussian_2D, independent_vars=['x', 'y'])
Gaus2D_model.guess = gauss_2D_guess  # Note: not proper way to add guess func
DoubleGauss2D_model = (lmfit.Model(gaussian_2D, independent_vars=['x', 'y'],
                                   prefix='A_') +
                       lmfit.Model(gaussian_2D, independent_vars=['x', 'y'],
                                   prefix='B_'))
DoubleGauss2D_model.guess = double_gauss_2D_guess
###################################
# Models based on lmfit functions #
###################################

LorentzModel = lmfit.Model(lmfit.models.lorentzian)
Lorentz_w_background_Model = lmfit.models.LorentzianModel() + \
    lmfit.models.LinearModel()
PolyBgHangerAmplitudeModel = (HangerAmplitudeModel *
                              lmfit.models.PolynomialModel(degree=7))

DoubleGaussModel = (lmfit.models.GaussianModel(prefix='A_') +
                    lmfit.models.GaussianModel(prefix='B_'))
DoubleGaussModel.guess = double_gauss_guess  # defines a guess function


def plot_fitres2D_heatmap(fit_res, x, y, axs=None, cmap='viridis'):
    '''
    Convenience function for plotting results of flattened 2D fits.

    It could be argued this does not belong in fitting models (it is not a
    model) but I put it here as it is closely related to all the stuff we do
    with lmfit. If anyone has a better location in mind, let me know (MAR).
    '''
    # fixing the data rotation with [::-1]
    nr_cols = len(np.unique(x))
    data_2D = fit_res.data.reshape(-1, nr_cols, order='C')[::-1]
    fit_2D = fit_res.best_fit.reshape(-1, nr_cols, order='C')[::-1]
    guess_2D = fit_res.init_fit.reshape(-1, nr_cols, order='C')[::-1]
    if axs is None:
        f, axs = plt.subplots(1, 3, figsize=(14, 6))
    axs[0].imshow(data_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[1].imshow(fit_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[2].imshow(guess_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[0].set_title('data')
    axs[1].set_title('fit-result')
    axs[2].set_title('initial guess')
    return axs


# Before defining a new model, take a look at the built in models in lmfit.

# From http://lmfit.github.io/lmfit-py/builtin_models.html

# Built-in Fitting Models in the models module
# Peak-like models
    # GaussianModel
    # LorentzianModel
    # VoigtModel
    # PseudoVoigtModel
    # Pearson7Model
    # StudentsTModel
    # BreitWignerModel
    # LognormalModel
    # DampedOcsillatorModel
    # ExponentialGaussianModel
    # SkewedGaussianModel
    # DonaichModel
# Linear and Polynomial Models
    # ConstantModel
    # LinearModel
    # QuadraticModel
    # ParabolicModel
    # PolynomialModel
# Step-like models
    # StepModel
    # RectangleModel
# Exponential and Power law models
    # ExponentialModel
    # PowerLawModel
