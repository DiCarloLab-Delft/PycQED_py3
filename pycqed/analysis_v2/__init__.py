"""
Analysis Framework Presentation by Nathan
Motiviation
    - Interact with data
    - Easy access to many files
    - Structure s.t. no code has to be rewritten, i.e. well-designed generic
        base classes
    - Standardized plots
Main features we discussed:
- Plotting methods of the analysis objects can take axes as arguments and draw
        their plots on
  them and also give axes objects as output
    -> can further manipulate this plots and easily plot several things on top
        of each other
- Options dictionary instead of **kw: options are saved, and for readability
        and consistency the
  options should be extracted at one single place
    -> easier to see available options
- Standard plotting functions in base class. To make a new plot, define plot
        dictionaries which
  contain the relevant parameters/arguments that are passed to the plotting
        funciton.
- Axes objects are stored in self.axs:dict.
- run_analysis() has standard steps (e.g. extract_data, fitting, plotting),
        which need to be
  implemented in subclasses.
- Fitting is a bit hacky at the moment, but the framework for doing this in a
        modular way (i.e.
  using models defined in a module) is basically there
    -> Maybe also implement this in a way that makes use of dictionaries for
        passing arguments?
Idea for making this a nice suite: split in to the following files
    1st file: Base class
    2nd file: Set of very generic anlyses (spetroscopy, ...)
    3rd file: Everyone's own specific stuff, including dirty hacks for work on
        day-to-day basis.

"""
