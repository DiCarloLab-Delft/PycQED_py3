import numpy as np
import matplotlib.pyplot as plt
from qcodes.plots.pyqtgraph import QtPlot
import logging
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import InstrumentRefParameter
from qcodes.utils import validators as vals
from pycqed.analysis.fit_toolbox.functions import PSD
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel

class Base_LutMan(Instrument):
    """
    The base LutMan is an abstract base class for the individual LutMans.
    The idea of the Lookuptable Manager (LutMan) is to provide a convenient
    interface to manage the waveforms loaded on specific lookuptables in
    an AWG.

    The LutMan provides
        - A set of basic waveforms that are generated based on
            parameters specified in the LutMan
        - A LutMap, relating waveform names to specific lookuptable indices
        - Methods to upload and regenerate these waveforms.
        - Methods to render waves.

    The Base LutMan does not provide a set of


    """

    def __init__(self, name, **kw):
        logging.info(__name__ + ' : Initializing instrument')
        super().__init__(name, **kw)
        self.add_parameter(
            'AWG', parameter_class=InstrumentRefParameter, docstring=(
                "Name of the AWG instrument used, note that this can also be "
                "a UHFQC or a CBox as these also contain AWG's"),
            vals=vals.Strings())

        self._add_waveform_parameters()
        self.add_parameter(
            'LutMap', docstring=(
                'Dictionary containing the mapping between waveform'
                ' names and parameter names (codewords).'),
            initial_value={}, vals=vals.Dict(),
            parameter_class=ManualParameter)
        self.add_parameter('sampling_rate', unit='Hz',
                           vals=vals.Numbers(1, 1e10),
                           initial_value=1e9,
                           parameter_class=ManualParameter)

        # Used to determine bounds in plotting.
        # overwrite in child classes if used.
        self._voltage_min = None
        self._voltage_max = None

        # initialize the _wave_dict to an empty dictionary
        self._wave_dict = {}
        self.set_default_lutmap()

    def set_default_lutmap(self):
        """
        Sets the "LutMap" parameter to

        """
        raise NotImplementedError()

    def _add_waveform_parameters(self):
        """
        Adds the parameters required to generate the standard waveforms
        """
        raise NotImplementedError()

    def generate_standard_waveforms(self):
        """
        Generates all the standard waveforms and populates self._wave_dict

        """
        raise NotImplementedError()

    def load_waveform_onto_AWG_lookuptable(self, waveform_name: str,
                                           regenerate_waveforms: bool=False):
        """
        Loads a specific waveform to the AWG
        """
        raise NotImplementedError()

    def load_waveforms_onto_AWG_lookuptable(
            self, regenerate_waveforms: bool=True, stop_start: bool = True):
        """
        Loads all waveforms specified in the LutMap to an AWG.

        Args:
            regenerate_waveforms (bool): if True calls
                generate_standard_waveforms before uploading.
            stop_start           (bool): if True stops and starts the AWG.
        """
        AWG = self.AWG.get_instr()

        if stop_start:
            AWG.stop()
        if regenerate_waveforms:
            self.generate_standard_waveforms()

        for waveform_name, lookuptable in self.LutMap().items():
            self.load_waveform_onto_AWG_lookuptable(waveform_name)

        if stop_start:
            AWG.start()

    def render_wave(self, wave_name, show=True, time_units='lut_index',
                    reload_pulses=True):
        """
        Renders a waveform
        """
        if reload_pulses:
            self.generate_standard_waveforms()
        fig, ax = plt.subplots(1, 1)
        if time_units == 'lut_index':
            x = np.arange(len(self._wave_dict[wave_name][0]))
            ax.set_xlabel('Lookuptable index (i)')
            if self._voltage_min is not None:
                ax.vlines(
                    2048, self._voltage_min, self._voltage_max, linestyle='--')
        elif time_units == 's':
            x = (np.arange(len(self._wave_dict[wave_name][0]))
                 / self.sampling_rate.get())

            if self._voltage_min is not None:
                ax.vlines(2048 / self.sampling_rate.get(),
                          self._voltage_min, self._voltage_max, linestyle='--')

        ax.set_title(wave_name)
        ax.plot(x, self._wave_dict[wave_name][0],
                marker='o', label='chI')
        ax.plot(x, self._wave_dict[wave_name][1],
                marker='o', label='chQ')
        ax.set_ylabel('Amplitude (V)')

        ax.legend()
        if self._voltage_min is not None:
            ax.set_axis_bgcolor('gray')
            ax.axhspan(self._voltage_min, self._voltage_max, facecolor='w',
                       linewidth=0)
            ax.set_ylim(self._voltage_min*1.1, self._voltage_max*1.1)

        ax.set_xlim(0, x[-1])
        if time_units == 's':
            set_xlabel(ax, 'time', 's')
        if show:
            plt.show()
        return fig, ax

    def render_wave_PSD(self, wave_name, show=True, reload_pulses=True,
                        f_bounds=None, y_bounds=None):
        if reload_pulses:
            self.generate_standard_waveforms()
        fig, ax = plt.subplots(1, 1)
        f_axis, PSD_I = PSD(
            self._wave_dict[wave_name][0], 1/self.sampling_rate())
        f_axis, PSD_Q = PSD(
            self._wave_dict[wave_name][1], 1/self.sampling_rate())

        ax.set_title(wave_name)
        ax.plot(f_axis, PSD_I,
                marker='o', label='chI')
        ax.plot(f_axis, PSD_Q,
                marker='o', label='chQ')
        ax.set_ylabel('Spectral density (V^2/Hz)')
        ax.legend()

        ax.set_yscale("log", nonposy='clip')
        if y_bounds is not None:
            ax.set_ylim(y_bounds[0], y_bounds[1])
        if f_bounds is not None:
            ax.set_xlim(f_bounds[0], f_bounds[1])
        set_xlabel(ax, 'Frequency', 'Hz')
        if show:
            plt.show()
        return fig, ax
