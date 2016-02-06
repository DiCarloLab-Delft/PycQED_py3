import numpy as np
import matplotlib.pyplot as plt
import lmfit
#################################
#   Fitting Functions Library   #
#################################


def RandomizedBenchmarkingDecay(numCliff, Amplitude, p, offset):
    val = Amplitude * (p**numCliff) + offset
    return val


def LorentzFunc(f, amplitude, center, sigma):
    val = amplitude/np.pi * (sigma / ((f-center)**2 + sigma**2))
    return val


def Lorentzian(f, A, offset, f0, kappa):
    val = offset + A/np.pi * (kappa / ((f/1.e9-f0)**2 + kappa**2))
    return val


def TwinLorentzFunc(f, amplitude_a, amplitude_b, center_a, center_b,
                    sigma_a, sigma_b, background=0):
    'twin lorentz with background'
    val = (amplitude_a/np.pi * (sigma_a / ((f-center_a)**2 + sigma_a**2)) +
           amplitude_b/np.pi * (sigma_b / ((f-center_b)**2 + sigma_b**2)) +
           background)
    return val


def QubitFreqDac(dac_voltage, f_max, E_c,
                 dac_sweet_spot, dac_flux_coefficient, asymmetry=0):
    '''
    The cosine Arc model for uncalibrated flux for asymmetric qubit.
    dac_voltage (mV), f_max (GHz), E_c (MHz),
    asym (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2)),
    dac_sweet_spot (mV),
    dac_flux_coefficient (1/mV)
    '''
    calculated_frequency = (f_max + E_c)*(asymmetry**2 + (1-asymmetry**2) *
        np.cos(dac_flux_coefficient*(dac_voltage-dac_sweet_spot))**2)**0.25-E_c
    return calculated_frequency


def QubitFreqFlux(flux, f_max, E_c,
                  flux_zero, dac_offset=0):
    'The cosine Arc model for calibrated flux.'
    calculated_frequency = (f_max + E_c)*np.sqrt(np.abs(
        np.cos(np.pi*(flux-dac_offset)/flux_zero))) - E_c
    return calculated_frequency


def CosFunc(t, amplitude, frequency, phase, offset):
    # Naming convention, frequency should be Hz
    # omega is in radial freq
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
    '''
    return abs(A*(1.-Q/Qe*np.exp(1.j*theta)/(1.+2.j*Q*(f/1.e9-f0)/f0)))


def HangerFuncComplex(f, f0, Q, Qe, A, theta):
    # This is the function for a hanger which does not
    # take into account a possible slope
    # This function may be preferred over SlopedHangerFunc if the area around
    # the hanger is small.
    # In this case it may misjudge the slope
    # Theta is the asymmetry parameter
    return A*(1.-Q/Qe*np.exp(1.j*theta)/(1.+2.j*Q*(f/1.e9-f0)/f0))


def PolyBgHangerFuncAmplitude(f, f0, Q, Qe, A, theta, poly_coeffs):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible polynomial background
    # NOT DEBUGGED
    return np.abs((1.+np.polyval(poly_coeffs,(f/1.e9-f0)/f0)) *
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


def linear_with_background(x, a, b):
    '''
    A linear signal with a fixed background.
    '''
    return np.sqrt((a*x)**2 + b**2)


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

####################
# Guess functions  #
####################


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
    A_center = x[(np.abs(norm_cdf - 0.25)).argmin()]
    B_center = x[(np.abs(norm_cdf - 0.75)).argmin()]
    A_sigma = (x[(np.abs(norm_cdf - 0.25 - .33/2)).argmin()]
               - x[(np.abs(norm_cdf - 0.25 + .33/2)).argmin()])
    B_sigma = (x[(np.abs(norm_cdf - 0.75 - .33/2)).argmin()]
               - x[(np.abs(norm_cdf - 0.75 + .33/2)).argmin()])

    amp = max(data)*(A_sigma + B_sigma)/2.
    pars = model.make_params(A_center=A_center, B_center=B_center,
                             A_sigma=A_sigma, B_sigma=B_sigma,
                             A_amplitude=amp, B_amplitude=amp)
    return pars

#################################
#     User defined Models       #
#################################
CosModel = lmfit.Model(CosFunc)
ExpDecayModel = lmfit.Model(ExpDecayFunc)
ExpDampOscModel = lmfit.Model(ExpDampOscFunc)
GaussExpDampOscModel = lmfit.Model(GaussExpDampOscFunc)
ExpDampDblOscModel = lmfit.Model(ExpDampDblOscFunc)
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
LinBGModel = lmfit.Model(linear_with_background)

# 2D models
Gaussian_2D_model = lmfit.Model(gaussian_2D, independent_vars=['x', 'y'])
Gaussian_2D_model.guess = gauss_2D_guess


###################################
# Models based on lmfit functions #
###################################

LorentzModel = lmfit.Model(lmfit.models.lorentzian)
Lorentz_w_background_Model = lmfit.models.LorentzianModel() + \
    lmfit.models.LinearModel()
PolyBgHangerAmplitudeModel = HangerAmplitudeModel * lmfit.models.PolynomialModel(degree=7)

DoubleGaussModel = (lmfit.models.GaussianModel(prefix='A_') +
                    lmfit.models.GaussianModel(prefix='B_'))
DoubleGaussModel.guess = double_gauss_guess  # defines a guess function


def plot_fitres2D_heatmap(fit_res, x, y, axs=None, cmap='CMRmap'):
    '''
    Convenience function for plotting results of flattened 2D fits.

    It could be argued this does not belong in fitting models (it is not a
    model) but I put it here as it is closely related to all the stuff we do
    with lmfit. If anyone has a better location in mind, let me know (MAR).
    '''
    nr_cols = len(np.unique(x))
    data_2D = fit_res.data.reshape(-1, nr_cols)
    fit_2D = fit_res.best_fit.reshape(-1, nr_cols)
    guess_2D = fit_res.init_fit.reshape(-1, nr_cols)

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