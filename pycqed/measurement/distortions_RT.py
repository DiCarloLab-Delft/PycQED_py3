class RT_distortion:
    """
    Class that handles and automatizes the production of a distortion from
    a step response measured in a scope.

    Input:
        filepath   : Path to the file that contains the measured values.
        fit_opt    : {'step','kernel_step','numeric'} defines what is
                      fitted/inverted to produce the kernel.
        fit_model  : Sets the model for the fit (see distortion_fit_models)
        fit_guesses: Provides initial guesses for the fit.
        length     : Sets the length of the kernel to be generated
        outputfile : Sets the name for the output file.
        opt_dict   : Dictionary allowing access to more advanced options.
    Output:
        > Plots for step function and kernel step function
        > File with kernel

    """
    def __init__(self, filepath, fit_opt, fit_model,
                 fit_guesses, length, outputfile, opt_dict, auto=True):
        self.filepath = filepath
        self.fit_opt = fit_opt
        self.outputfile = outputfile
        self.fit_model = fit_model
        if auto:
            self.calculate()

    def calculate(self):
        self.plot_step()
        self.invert()
        self.fit()
