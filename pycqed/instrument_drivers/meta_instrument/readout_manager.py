from qcodes import Instrument

class ReadoutManager:
    """
    A class that can generate a detector function and the readout pulses
    and configure the readout instruments for arbitrary combination of readout
    channels.
    """
    def __init__(self, instruments=None):
        if instruments is None:
            instruments = {}

        # a dictionary mapping readout channel names to ReadoutChannel instances
        self.readout_channels = {}

        # a dictionary mapping names of the instruments used for readout to
        # instances of these instruments
        self.instruments = {}
        for i in instruments:
            self.instruments[i.name] = i

        # An iterable of the channels that are currently activated
        self.active_channels = set()

    def register_instrument(self, instrument):
        self.instruments[instrument.name] = instrument

    def register_channel(self, name, configuration):
        if name in self.readout_channels:
            raise KeyError('Readout channel with name {} already exists'
                           .format(name))
        self.readout_channels[name] = configuration

    def select_channels(self, channels):
        """
        Set the list of channels you want to read out. Configures the readout
        instrument parameters. Raises an exception if the selected channels
        have incompatible configuration values.

        Args:
            channels: a list of readout channel names for which the readout
                      pulses will be generated
        """
        par_vals = {i: dict() for i in self.instruments}
        self.active_channels = channels

        for cname in channels:
            c = self.readout_channels[cname]
            for iname in c.par_vals:
                if iname not in self.instruments:
                    m = "Readout manager does not have a reference to " \
                        "instrument '{}' specified in readout channel '{}'." \
                        .format(iname, cname)
                    raise ConfigurationError(m)
                for pname in c.par_vals[iname]:
                    if pname in par_vals[iname] and \
                            c.par_vals[iname][pname] != par_vals[iname][pname]:
                        m = "Value of the parameter '{}' in instrument '{}' " \
                            "set by readout channel '{}' conflicts with a " \
                            "previously set value.".format(pname, iname, cname)
                        raise ConfigurationError(m)
                    par_vals[iname][pname] = c.par_vals[iname][pname]

        for iname in par_vals:
            i = self.instruments[iname]
            for pname in par_vals[iname]:
                i.set(pname, par_vals[iname][pname])

    def readout_pulses(self):
        """
        Returns: A list of pulse dictionaries necessary for the readout of the
                 configured channels
        """
        unique_pulses = {}
        for cname in self.active_channels:
            pulses = self.readout_channels[cname]



class ReadoutChannel(Instrument):

    def __init__(self, name):
        super().__init__(name)
        self.par_vals = {}

    def channel_pulses(self):
        raise NotImplementedError()


class ConfigurationError(Exception):
    """
    Raise when there is a problem with the readout instrument configuration
    """